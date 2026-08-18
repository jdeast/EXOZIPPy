"""The one interpreter for the manifest vocabulary.

A component declares ``self.manifest`` at stage 3: a dict mapping parameter
name -> entry.  An entry may be

* ``None`` (or any falsy value) -- a free parameter with no options,
* a string -- the key of an ``expressions:`` block in the component's
  defaults.yaml (almost always ``"default"``),
* a dict -- options, optionally including ``"expr_key"``.

The vocabulary has three consumers, at three different lifecycle stages:

* ``graph.determine_pymc_build_order`` (the build order) -- needs the expression key
  and its dependency list to place the parameter in the build order;
* ``Component.add_parameter`` (stage 6) -- needs the expression key, its
  dependency list, and every remaining option;
* ``System.derived_params`` -- needs only "does this entry name an
  expression at all".

Each of them used to answer those questions itself, and they DISAGREED: a
dict without ``expr_key`` (``{"overrides": ...}``, ``{"lower": ...}``) is a
free parameter to ``add_parameter``, but ``graph.py`` fell back to the
``"default"`` expression for *any* dict.  That was inert only while no
pinned free parameter had an unused ``expressions:`` block in its
defaults.yaml, and became a hard "Dependency Error" the moment one did --
Band's linear-law ``u1``, whose Kipping expression the manifest deliberately
ignores, and planet's ``beam``, whose "off" entry made every orbit-less
config demand an RV semi-amplitude.  Both were fixed in graph.py by hand;
this module exists so the vocabulary cannot re-diverge.

An ``expr_key`` naming a block the resolved config does not define is an
ERROR (:class:`MissingExpressionError`), not a free parameter.  It used to
build the parameter free, silently: a typo, a renamed block, or a deleted
``expressions:`` section turned a derived parameter into a sampled one with
no message anywhere.  Breaking ``mulensinstrument.f_source``'s expr_key on
``examples/ob08092`` that way put a ``f_source_raw`` in ``model.free_RVs``
and moved the start logp from +6187.7 to -6.46e9 -- a fit that still runs
and still reports.  A free parameter has two spellings that say so
explicitly (``None``, or a dict of options with no ``expr_key``), so
nothing legitimate needs the silent fallback.

The vocabulary is also where the PER-ELEMENT roles are interpreted, for the
same reason: a modeling choice that differs between two instances of one
component (limb-darkening law per band, mass coordinate per planet,
evolutionary track per star, eccentricity parameterization per orbit) shows up
as some elements of a vector being sampled while others are derived or absent
entirely.  ``Parameter.build_pymc`` was uniform on that axis --
``is_derived = np.full(n_elements, expr_raw is not None)`` -- which is why
those four features each shipped a hard error or a silent downgrade instead.
See ``ROLE_*`` below for the four roles and ``ManifestEntry`` for the three
options that select them.

Nothing here imports from the rest of the package, so every consumer can
import it without a cycle.  It contains no component-specific knowledge --
it is a parser for a small data format, nothing more.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# The four roles an ELEMENT of a parameter vector can have.  Roles are per
# element because the modeling choices that select them are per instance: one
# band's limb darkening is Kipping-quadratic while another's is linear, one
# orbit uses sqrt(e)cos/sin(omega) while another uses V_c/V_e, one star has an
# evolutionary track and another does not.
#
#   SAMPLED   a raw coordinate of its own (the default, and every element of
#             every parameter before this vocabulary existed).
#   DERIVED   its value is an expression, and the model CONSUMES it.
#   REPORTED  its value is an expression that nothing consumes -- the reverse
#             direction of a parameterization flip (report sqrt(e)cos(omega)
#             for an orbit that sampled V_c/V_e).  Declared here; the deferred
#             build pass it needs lands with the first real consumer (vcve).
#   INACTIVE  not a parameter of this element's parameterization at all (a
#             non-MIST star's EEP).  Held at a bookkeeping value, given no
#             potential, and reported nowhere.
ROLE_SAMPLED = "sampled"
ROLE_DERIVED = "derived"
ROLE_REPORTED = "reported"
ROLE_INACTIVE = "inactive"


class MissingExpressionError(ValueError):
    """A manifest entry names an ``expressions:`` block that does not exist.

    Deliberately not a ``KeyError``: its ``str()`` would be the repr of the
    message, escaping the quotes and newlines this one needs.
    """


class ElementSelectorError(ValueError):
    """A per-element selector does not describe the parameter's elements."""


def normalize_selector(selector, n_elements, where=None):
    """A per-element selector -> a boolean mask of length ``n_elements``.

    Accepts ``None`` (every element), a scalar bool, a boolean mask, or any
    iterable of integer indices -- the three spellings components already
    write by hand (``star`` passes a list of bools, ``orbit`` a numpy bool
    array, ``PriorContribution`` an index list).

    The length is checked against ``n_elements`` and a mismatch RAISES rather
    than broadcasting or truncating.  That is the sizing hazard review 1.1.1
    reports for broadcast config keys: a mask sized from the config list is
    silently short for any parameter whose vector is longer than it (lens has
    one config entry and per-source vectors), and a short mask would quietly
    mark real elements inactive.
    """
    label = f"[{where}] " if where else ""
    if selector is None:
        return np.ones(int(n_elements), dtype=bool)
    if isinstance(selector, (bool, np.bool_)):
        return np.full(int(n_elements), bool(selector), dtype=bool)

    arr = np.atleast_1d(np.asarray(selector))
    if arr.dtype == bool:
        if arr.size != int(n_elements):
            raise ElementSelectorError(
                f"{label}element selector is a boolean mask of length "
                f"{arr.size}, but the parameter has {n_elements} element(s). "
                f"Size the mask from the parameter's own element count (its "
                f"manifest 'shape', when it carries one), not from the "
                f"component's config list."
            )
        return arr.astype(bool).copy()

    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        raise ElementSelectorError(
            f"{label}element selector must be None, a bool, a boolean mask, "
            f"or an iterable of integer indices; got dtype {arr.dtype}."
        )
    mask = np.zeros(int(n_elements), dtype=bool)
    for i in arr.ravel().tolist():
        if not (0 <= int(i) < int(n_elements)):
            raise ElementSelectorError(
                f"{label}element selector names index {i}, outside the "
                f"parameter's {n_elements} element(s)."
            )
        mask[int(i)] = True
    return mask


@dataclass(frozen=True)
class ExpressionSelection:
    """One ``expressions:`` block, and the elements it supplies.

    ``mask`` is ``None`` when the block supplies EVERY element (the historical
    case, and the one that keeps its byte-for-byte build path in
    ``Parameter.build_pymc``); otherwise it is a boolean mask.  ``output_only``
    marks a REPORTED selection (role 3): the elements are derived but consumed
    by nothing, so they carry no potential and contribute no build-order edge.
    """

    key: str
    config: Dict[str, Any]
    mask: Optional[np.ndarray] = None
    output_only: bool = False


@dataclass(frozen=True)
class ManifestEntry:
    """A manifest entry, split into its expression key(s) and its options.

    ``expr_key`` is the name of the ``expressions:`` block the entry selects,
    or ``None`` for a free parameter.  ``options`` holds everything else the
    entry carried (``shape``, ``names``, ``overrides``, ``deps``,
    ``table_note``, ``force_node``, direct field overrides such as ``lower``
    ...) -- never ``expr_key`` itself.

    Three options are per-element and are read here rather than by the
    consumers, so the four element roles have one interpreter:

    * ``expr_key`` may be a DICT ``{block name: element selector}`` instead of
      a string, selecting a different expression per element -- ``ecc`` from
      sqrt(e)cos/sin(omega) on one orbit and from V_c/V_e on the next.  The
      string form means "every element", and keeps its own build path.
    * ``output_expr_key`` takes the same two shapes for REPORTED elements
      (role 3): derived, but consumed by nothing.
    * ``mask`` is the ACTIVITY selector.  Elements outside it are INACTIVE
      (role 4): held at ``inactive_value`` (or their resolved ``initval``),
      never sampled, never given a potential, and never reported.
    """

    expr_key: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    expr_selectors: Optional[Dict[str, Any]] = None
    output_expr_key: Optional[str] = None
    output_expr_selectors: Optional[Dict[str, Any]] = None

    @property
    def names_expression(self) -> bool:
        """Does this entry name an expression (i.e. is the parameter derived)?

        This is the structural question, answered from the manifest alone:
        it does not check that the named block exists in the component's
        defaults.yaml.  It agrees with :meth:`expression_config` by
        construction, because the one case in which they could differ -- a
        named block the config does not define -- now raises there instead
        of answering "free".  Prefer ``expression_config`` wherever a
        resolved config is in scope, so a broken manifest is reported rather
        than merely labelled.

        True for a per-element entry as soon as ONE element is derived: the
        question it answers ("does an expression supply this parameter") has
        no per-element answer, and every consumer that needs one asks
        :meth:`expression_configs`.  REPORTED selections count too -- their
        elements are derived, they are merely consumed by nothing.
        """
        return bool(
            self.expr_key
            or self.expr_selectors
            or self.output_expr_key
            or self.output_expr_selectors
        )

    @property
    def has_output_expression(self) -> bool:
        """Does this entry declare REPORTED (output-only) elements?"""
        return bool(self.output_expr_key or self.output_expr_selectors)

    @property
    def is_per_element(self) -> bool:
        """Does this entry select expressions or activity per element?"""
        return bool(
            self.expr_selectors
            or self.output_expr_selectors
            or self.output_expr_key
            or self.options.get("mask") is not None
        )

    @property
    def deps(self) -> Optional[List[str]]:
        """The entry's own dependency override, or ``None`` if it has none."""
        return self.options.get("deps")

    @property
    def overrides(self) -> Optional[Dict[str, Any]]:
        """Component-computed per-element defaults, layered UNDER user params."""
        return self.options.get("overrides")

    @property
    def shape(self) -> Optional[tuple]:
        """A shape override for parameters that are not one-per-element."""
        return self.options.get("shape")

    @property
    def names(self) -> Optional[List[str]]:
        """A per-element name override for resolution and display labels."""
        return self.options.get("names")

    def expression_config(
        self,
        expressions: Optional[Dict[str, Any]],
        where: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """The ``expressions:`` block this entry selects, or ``None``.

        ``None`` means the entry names no expression: the parameter is free.
        A named block the resolved config does not define raises
        :class:`MissingExpressionError` -- see the module docstring for why
        that is not a free parameter either.  ``where`` is the
        ``component.parameter`` label the message is raised against.

        This is the question ``add_parameter``, ``graph.py`` and
        ``System.derived_params`` all ask, and they must get the same answer
        -- an expression graph.py wires an edge for but add_parameter never
        builds is a spurious edge (or a cycle); one add_parameter builds but
        graph.py missed is a node built out of order; and one
        ``derived_params`` reports but nothing builds is a parameter the
        reporting layer excuses from every check a sampled parameter gets.

        Per-element entries have no single block, so this raises for them;
        every consumer that can see an element count should ask
        :meth:`expression_configs` instead.
        """
        if self.expr_selectors:
            raise TypeError(
                f"[{where or 'manifest'}] this entry selects a different "
                f"expression per element "
                f"({', '.join(sorted(self.expr_selectors))}); there is no "
                f"single expressions: block. Call expression_configs(...) "
                f"with the parameter's element count."
            )
        if self.expr_key is None:
            return None
        return self._block(self.expr_key, expressions, where)

    def _block(self, key, expressions, where):
        blocks = expressions or {}
        if key not in blocks:
            available = ", ".join(sorted(blocks)) or "(none)"
            label = f"[{where}] " if where else ""
            raise MissingExpressionError(
                f"{label}the manifest selects the expressions: block "
                f"'{key}', but the resolved config defines no such "
                f"block (available: {available}). Either add it under "
                f"expressions: in the component's defaults.yaml, or -- if the "
                f"parameter is meant to be free -- say so explicitly: a "
                f"manifest value of None for a free parameter with no "
                f"options, or a dict of options carrying no 'expr_key' "
                f"(e.g. {{'shape': ...}}, {{'overrides': ...}}) for one with "
                f"options."
            )
        return blocks[key]

    def expression_configs(
        self,
        expressions: Optional[Dict[str, Any]],
        n_elements: Optional[int] = None,
        where: Optional[str] = None,
    ) -> List[ExpressionSelection]:
        """Every ``expressions:`` block this entry selects, with its elements.

        The per-element generalization of :meth:`expression_config`, and the
        form every consumer should prefer: ``[]`` for a free parameter, one
        :class:`ExpressionSelection` with ``mask=None`` for the uniform case,
        and one per block for a per-element entry.  REPORTED selections
        (``output_expr_key``) come last, flagged ``output_only``.

        ``n_elements`` is required only when a selector has to be normalized,
        which is exactly when the entry is per element; the uniform case (and
        so every consumer that predates this) needs nothing new.  Overlapping
        selections raise: two expressions supplying one element is ambiguous,
        and picking one silently is how the three hand-written manifest
        readers used to disagree.
        """
        out: List[ExpressionSelection] = []
        for selectors, key, output_only in (
            (self.expr_selectors, self.expr_key, False),
            (self.output_expr_selectors, self.output_expr_key, True),
        ):
            if selectors:
                if n_elements is None:
                    raise ElementSelectorError(
                        f"[{where or 'manifest'}] this entry selects an "
                        f"expression per element, so the parameter's element "
                        f"count is required to read it."
                    )
                for k in sorted(selectors):
                    out.append(
                        ExpressionSelection(
                            key=k,
                            config=self._block(k, expressions, where),
                            mask=normalize_selector(
                                selectors[k], n_elements, where
                            ),
                            output_only=output_only,
                        )
                    )
            elif key is not None:
                mask = None
                if output_only and n_elements is not None:
                    mask = np.ones(int(n_elements), dtype=bool)
                out.append(
                    ExpressionSelection(
                        key=key,
                        config=self._block(key, expressions, where),
                        mask=mask,
                        output_only=output_only,
                    )
                )

        claimed = None
        for sel in out:
            if sel.mask is None:
                continue
            if claimed is None:
                claimed = np.zeros(sel.mask.size, dtype=bool)
            clash = np.nonzero(claimed & sel.mask)[0]
            if clash.size:
                raise ElementSelectorError(
                    f"[{where or 'manifest'}] element(s) "
                    f"{clash.tolist()} are claimed by more than one "
                    f"expression (last: '{sel.key}'). Each element takes its "
                    f"value from exactly one expression, or from a sampled "
                    f"coordinate of its own."
                )
            claimed |= sel.mask
        return out

    def activity_mask(
        self, n_elements: int, where: Optional[str] = None
    ) -> np.ndarray:
        """Boolean mask of the ACTIVE elements (role 4 is the complement).

        All-True when the entry carries no ``mask`` option, which is every
        entry that predates this vocabulary.
        """
        return normalize_selector(self.options.get("mask"), n_elements, where)

    @property
    def inactive_value(self):
        """The bookkeeping value inactive elements are held at, or ``None``.

        ``None`` means "whatever the element resolved to" -- right for a
        parameter that is merely irrelevant to this instance (a non-MIST
        star's EEP).  A component whose masked-out element has a DEFINED value
        in the other parameterization states it (a linear-law band's ``u2`` is
        exactly 0), so the pin cannot drift with an unrelated default.
        """
        return self.options.get("inactive_value")

    def dep_names(
        self, expr_cfg: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """The dependency names for this entry's expression.

        A ``deps`` list on the manifest entry wins outright (components use
        it for per-instance dependencies -- an orbit's body groups); with no
        such list, the expression block's own ``deps`` apply.
        """
        manifest_deps = self.deps
        if manifest_deps is not None:
            return list(manifest_deps)
        return list((expr_cfg or {}).get("deps", []))


def interpret_manifest_entry(raw: Any) -> ManifestEntry:
    """Parse one raw manifest value into a :class:`ManifestEntry`.

    An empty-string ``expr_key`` is normalized to ``None``: it names no
    block, and treating it as a name is the kind of near-miss the three
    hand-written readers used to disagree about.  ``expr_key`` and
    ``output_expr_key`` each accept a string (every element) or a dict
    ``{block: element selector}`` (per element); an empty dict is normalized
    away exactly like an empty string.
    """
    if not raw:
        # None, {}, "", 0, False: a free parameter carrying no options.
        return ManifestEntry()
    if isinstance(raw, str):
        return ManifestEntry(expr_key=raw)
    if isinstance(raw, dict):
        options = dict(raw)  # never hand back a view of the live manifest
        expr_key, expr_selectors = _split_expr_key(
            options.pop("expr_key", None), "expr_key"
        )
        out_key, out_selectors = _split_expr_key(
            options.pop("output_expr_key", None), "output_expr_key"
        )
        return ManifestEntry(
            expr_key=expr_key,
            options=options,
            expr_selectors=expr_selectors,
            output_expr_key=out_key,
            output_expr_selectors=out_selectors,
        )
    raise TypeError(
        f"Manifest entries must be None, a string naming an expressions: "
        f"block, or a dict of options; got {type(raw).__name__} ({raw!r})."
    )


def _split_expr_key(value, field_name):
    """``(single key, per-element selectors)`` from one expr_key option."""
    if not value:
        return None, None
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict):
        selectors = {k: v for k, v in value.items() if k}
        return None, (selectors or None)
    raise TypeError(
        f"A manifest '{field_name}' must be a string naming one expressions: "
        f"block, or a dict mapping block names to per-element selectors; got "
        f"{type(value).__name__} ({value!r})."
    )
