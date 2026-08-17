"""Declaring a per-instance parameterization once, as a table.

``manifest.py`` interprets the per-element vocabulary (see its ``ROLE_*``
constants) and ``Parameter.build_pymc`` consumes it.  What a component actually
has in hand at stage 2 is different and always the same shape: a per-instance
CHOICE read from its own config (``ld_law`` per band, ``mass_parameterization``
per planet, ``mist``/``parsec`` per star, ``fitvcve`` per orbit), plus a
statement of which parameters each choice uses and how.  Turning that into
masks is mechanical, and writing it out per component is how four
implementations of the same idea would drift apart -- so it is written once,
here.

Two helpers, and the difference between them is the whole reason there are two:

* :func:`mode_manifest` -- the parameterization table.  A parameter that a
  given mode does not name is INACTIVE for those elements (manifest role 4):
  it is not a parameter of that instance at all, so it is pinned for
  bookkeeping, given no potential, and reported nowhere.  That pin is
  STRUCTURAL: it goes in the ``mask`` channel, which a params file cannot
  override, because freeing it would add a dimension no likelihood term reads.

* :func:`pin_unselected` -- the OPT-IN pin, for a parameter that exists for
  every instance but is only wanted on some (a GP hyperparameter on the files
  that asked for a GP, a limb-darkening coefficient on the bands something
  reads, the BEER terms on the bands that fit them).  It pins through the
  ``"overrides"`` channel, which layers UNDER the params file, so a user who
  wants one back can still have it.  Three components had this loop written out
  line for line (``Instrument._register_gp``, ``Instrument._register_robust``,
  ``Band._pinned_manifest_entry``); it is one function now.

Nothing here knows any component's vocabulary: modes are opaque strings the
caller chose, and the table says what they mean.
"""

import numpy as np

from ..manifest import interpret_manifest_entry


def pin_unselected(n_elements, selected):
    """A manifest entry pinning every element OUTSIDE ``selected``.

    ``selected`` is any iterable of element indices (or a boolean mask).  The
    pin is ``sigma: 0`` through the manifest ``"overrides"`` channel, which
    layers under the user's params file: the unselected elements cost the
    sampler nothing, and a user who explicitly wants one back still wins.
    Returns ``{}`` when everything is selected, which is a free parameter
    carrying no options -- the same thing the three hand-written copies
    returned.

    This is the OPT-IN pin.  For a parameter that is not part of an instance's
    parameterization at all, use :func:`mode_manifest`'s inactive role, whose
    pin is structural and unreported.
    """
    mask = _selected_mask(selected, n_elements)
    if bool(np.all(mask)):
        return {}
    pin = np.full(int(n_elements), np.nan)
    pin[~mask] = 0.0
    return {"overrides": {"sigma": pin.tolist()}}


def merge_overrides(entry, overrides):
    """``entry`` with ``overrides`` merged in, whatever shape it arrived in.

    A manifest entry may be ``None``, a bare string naming an expressions:
    block, or a dict of options -- so a caller that ADDS an option to an
    existing entry has to read the vocabulary as a writer, and the obvious
    ``dict(entry) if isinstance(entry, dict) else {}`` silently DROPS a
    bare-string ``expr_key``, turning a derived parameter into a sampled one
    (review 4.5.3).  Reading through ``interpret_manifest_entry`` -- the one
    reader -- makes that unrepresentable.
    """
    parsed = interpret_manifest_entry(entry)
    out = dict(parsed.options)
    if parsed.expr_selectors:
        out["expr_key"] = dict(parsed.expr_selectors)
    elif parsed.expr_key is not None:
        out["expr_key"] = parsed.expr_key
    if parsed.output_expr_selectors:
        out["output_expr_key"] = dict(parsed.output_expr_selectors)
    elif parsed.output_expr_key is not None:
        out["output_expr_key"] = parsed.output_expr_key
    merged = dict(out.get("overrides", {}))
    merged.update(overrides or {})
    if merged:
        out["overrides"] = merged
    return out


def mode_manifest(modes, table, n_elements=None, options=None, where=""):
    """Manifest entries for a per-instance parameterization choice.

    ``modes`` is the mode of each ELEMENT (a list as long as the parameter
    vectors, e.g. ``["quadratic", "linear", "quadratic"]``).  ``table`` maps
    each mode to the parameters it uses::

        table = {
            "quadratic": {"q1": None, "q2": None,
                          "u1": "kipping", "u2": "kipping"},
            "linear":    {"u1": None},
        }

    A spec of ``None`` means the mode SAMPLES that parameter; a string names the
    ``expressions:`` block that derives it; a dict may carry ``expr_key`` plus
    any other manifest options.  A parameter a mode does not mention is
    INACTIVE on that mode's elements -- above, ``linear`` leaves ``q1``, ``q2``
    and ``u2`` out, because a linear law has no Kipping coordinates and no
    second coefficient.

    ``options`` adds per-parameter manifest options that are not mode-specific
    (``{"u2": {"inactive_value": 0.0}}`` -- a linear-law band's ``u2`` is
    exactly 0, so the bookkeeping pin should say so rather than inherit an
    unrelated default).

    Returns ``{param: entry}``, ready to merge into ``self.manifest``.  An
    entry is a plain free parameter (``{}`` plus its options) when every
    element samples it, so a single-mode system produces exactly what a
    hand-written manifest did.
    """
    modes = list(modes)
    n = int(n_elements if n_elements is not None else len(modes))
    if len(modes) != n:
        raise ValueError(
            f"[{where or 'mode_manifest'}] {len(modes)} mode(s) for {n} "
            f"element(s): the mode list is per element, so it must be exactly "
            f"as long as the parameter vectors it describes."
        )
    unknown = sorted({m for m in modes if m not in table})
    if unknown:
        raise ValueError(
            f"[{where or 'mode_manifest'}] unknown parameterization mode(s) "
            f"{unknown}; the table defines {sorted(table)}. A mode with no "
            f"table entry has no parameters, which is never what a caller "
            f"means."
        )

    params = []
    for mode in table:
        for param in table[mode]:
            if param not in params:
                params.append(param)

    out = {}
    for param in params:
        active = np.zeros(n, dtype=bool)
        selectors = {}
        extra = {}
        for i, mode in enumerate(modes):
            spec = table[mode].get(param, _ABSENT)
            if spec is _ABSENT:
                continue  # inactive on this element
            active[i] = True
            expr_key, opts = _split_spec(spec, param, mode, where)
            if expr_key is not None:
                selectors.setdefault(expr_key, np.zeros(n, dtype=bool))[i] = (
                    True
                )
            for key, value in opts.items():
                if key in extra and extra[key] != value:
                    raise ValueError(
                        f"[{where or 'mode_manifest'}] parameter '{param}' is "
                        f"given conflicting '{key}' options by different modes "
                        f"({extra[key]!r} vs {value!r}). A manifest option is "
                        f"per parameter, not per element; move the difference "
                        f"into the expression, or into 'overrides' if it is a "
                        f"per-element value."
                    )
                extra[key] = value

        if not bool(np.any(active)):
            # No instance uses this parameter, so it is not a parameter of this
            # system at all -- omit it rather than declaring a wholly inactive
            # vector.  That is what a component hand-writing the manifest did
            # (an all-linear band set has no u2 entry, and its consumers read
            # `"u2" in band.manifest` and substitute zeros), so keeping the
            # entry would change the graph of a system that made ONE choice.
            continue

        entry = dict(extra)
        entry.update((options or {}).get(param, {}))
        if selectors:
            # One block covering every ACTIVE element is the ordinary derived
            # parameter, and spelling it as a bare string keeps build_pymc on
            # its whole-vector path (identical graph) whenever nothing is
            # inactive either.
            if len(selectors) == 1:
                key, mask = next(iter(selectors.items()))
                # The bare string means EVERY element, so it is only correct
                # when the block really covers the whole vector -- not merely
                # every ACTIVE element.  A parameter that some mode leaves
                # inactive (a linear-law band's u2) must keep the selector, or
                # the inactive elements would be claimed by the expression and
                # the two roles would contradict.
                entry["expr_key"] = (
                    key if bool(np.all(mask)) else {key: mask.copy()}
                )
            else:
                entry["expr_key"] = {k: m.copy() for k, m in selectors.items()}
        if not bool(np.all(active)):
            entry["mask"] = active
        else:
            # `inactive_value` says what a masked-out element is held at, so it
            # cannot do anything when nothing is masked out.  Dropping it keeps
            # a single-mode expansion identical to the manifest a component
            # would have hand-written, rather than carrying an inert key.
            entry.pop("inactive_value", None)
        # Spell the two plain cases the way a hand-written manifest spells them:
        # `None` for a free parameter with no options, and the bare block name
        # for a wholly derived one.  Identical to the interpreter either way;
        # this keeps the expansion diff-free against the manifests it replaces,
        # which is how a reviewer can see that nothing moved.
        if not entry:
            out[param] = None
        elif list(entry) == ["expr_key"] and isinstance(
            entry["expr_key"], str
        ):
            out[param] = entry["expr_key"]
        else:
            out[param] = entry

    return out


class _Absent:
    """Sentinel: this mode does not name that parameter (-> inactive)."""

    def __repr__(self):  # pragma: no cover - debugging aid
        return "<absent>"


_ABSENT = _Absent()


def _split_spec(spec, param, mode, where):
    """``(expr_key, options)`` for one table cell.

    Accepts the same three shapes a manifest entry does, and for the same
    reason: a component author writing a mode table should not have to learn a
    second vocabulary.  ``interpret_manifest_entry`` does the parsing, so the
    two cannot drift.
    """
    if spec is None or isinstance(spec, str):
        entry = interpret_manifest_entry(spec)
    elif isinstance(spec, dict):
        entry = interpret_manifest_entry(dict(spec))
    else:
        raise TypeError(
            f"[{where or 'mode_manifest'}] mode '{mode}' gives parameter "
            f"'{param}' a spec of type {type(spec).__name__}; it must be None "
            f"(the mode samples it), a string naming an expressions: block, or "
            f"a dict of manifest options."
        )
    if entry.expr_selectors or entry.has_output_expression:
        raise ValueError(
            f"[{where or 'mode_manifest'}] mode '{mode}' gives parameter "
            f"'{param}' a per-element expr_key. The mode table IS the "
            f"per-element statement -- name one expressions: block per mode "
            f"and let the modes select the elements."
        )
    return entry.expr_key, dict(entry.options)


def _selected_mask(selected, n_elements):
    """Boolean mask from indices or a mask, sized to ``n_elements``."""
    from ..manifest import normalize_selector

    return normalize_selector(
        list(selected) if not isinstance(selected, np.ndarray) else selected,
        n_elements,
    )
