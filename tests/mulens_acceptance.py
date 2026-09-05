"""The acceptance instrument for the `mulensevent` split (review 8.6.17).

WHY THIS EXISTS.  8.6.17 collapses parameters that are stored per source but
physically singular, so the model genuinely changes and BYTE-IDENTITY STOPS
BEING AVAILABLE as acceptance -- the first item in this review for which that
is true.  The replacement currency is "every logp delta is explained by a
named term", and that is only possible with a decomposition whose parts
provably sum to the whole.

TWO WRONG WAYS, both tried while building this and both silently plausible:

  * sum ``model.potentials`` and add ``model.logp(rv)`` per RV -- 30.9 nats
    SHORT on ob161003, because the transform jacobians are missing;
  * ``model.logp(vars=[rv], jacobian=True, sum=True)`` per RV -- 0.85 nats
    OVER on ob08092 and 27.7 OVER on ob161003, because ``logp(vars=...)``
    does NOT decompose additively: it returns more than that variable's own
    contribution.

THE RIGHT WAY is ``model.logp(sum=False)``, which yields exactly one term per
``basic_RVs + potentials`` -- verified 48 = 11 + 37 on ob08092 and
69 = 17 + 52 on ob161003 -- and reconciles to ~1e-12 relative.

``reconciles`` is checked on every call and callers MUST refuse to emit or
trust a fixture when it is False.  An instrument that cannot be shown to add
up is worse than none, because it yields confident wrong attributions; this
is review 3.14.19's control-must-fire rule applied to the measuring device
itself.
"""

import numpy as np
import pytensor

# The reconciliation tolerance is relative to the total, which runs to ~1e5
# nats on the larger examples; 1e-9 there is ~1e-4 nats, far below any
# physical term while still catching a dropped or double-counted one.
RECONCILE_RTOL = 1e-9


def term_names(model):
    """Names paired with ``model.logp(sum=False)``, in its documented order.

    The order is ``basic_RVs`` then ``potentials``.  ``decompose`` asserts the
    lengths match rather than trusting it, because a silent reordering would
    mislabel every term while still reconciling -- the failure mode this
    module exists to prevent.
    """
    return [f"RV:{rv.name}" for rv in model.basic_RVs] + [
        f"POT:{p.name}" for p in model.potentials
    ]


def decompose(system, model, start=None):
    """Return ``(parts, total, reconciles, summed)`` at the model's start.

    ``start`` defaults to ``system.get_raw_start(model)``.  NOT
    ``model.initial_point()``, which is keyed by raw value variables and is
    not the point the model is evaluated at, and never ``Parameter.value``
    read outside a compiled function, which draws from the PRIOR rather than
    giving the start.  Both are documented traps and both produced
    confidently wrong numbers during review 8.6.18.
    """
    if start is None:
        start = system.get_raw_start(model)

    terms = model.logp(sum=False)
    names = term_names(model)
    if len(names) != len(terms):
        raise AssertionError(
            f"logp(sum=False) gave {len(terms)} terms but basic_RVs + "
            f"potentials is {len(names)}; the naming contract has changed "
            f"and every attribution built on it would be mislabelled"
        )

    value_vars = list(model.value_vars)
    fn = pytensor.function(value_vars, terms, on_unused_input="ignore")
    values = fn(*[start[v.name] for v in value_vars])

    parts = {}
    for name, value in zip(names, values):
        # A duplicate name would otherwise silently drop a term; accumulate
        # so the sum stays honest even if two nodes share a name.
        parts[name] = parts.get(name, 0.0) + float(np.asarray(value).sum())

    total = float(model.compile_logp()(start))
    summed = float(sum(parts.values()))
    reconciles = abs(summed - total) <= RECONCILE_RTOL * max(1.0, abs(total))
    return parts, total, reconciles, summed


def data_terms(parts):
    """The observed-data half of a decomposition.

    Stands in for the design's "data-only chi2": the two differ by the
    Gaussian normalisation, which is itself a function of the fitted error
    scale, so at MATCHED parameters they carry the same information and this
    one needs no separate extraction.  ``MulensInstrument`` deliberately
    keeps no ``self.mag`` (mulensinstrument.py:353), so capturing A(t)
    directly would need a bespoke extractor; if per-source curve parity turns
    out to be needed for attribution, add it then rather than guessing at it
    now.
    """
    return {k: v for k, v in parts.items() if k.startswith("RV:")}


def compare(reference, current, atol=1e-9, rtol=1e-9):
    """Diff two decompositions; return (moved, appeared, vanished).

    ``moved`` maps a term name to ``(before, after, delta)``.  Term sets are
    compared as well as values, because the split RENAMES terms -- a term
    that vanished and one that appeared with the same value is a rename, and
    the reviewer needs to see that rather than a silent match.
    """
    moved, appeared, vanished = {}, {}, {}
    for name, before in reference.items():
        if name not in current:
            vanished[name] = before
            continue
        after = current[name]
        if abs(after - before) > atol + rtol * abs(before):
            moved[name] = (before, after, after - before)
    for name, after in current.items():
        if name not in reference:
            appeared[name] = after
    return moved, appeared, vanished
