"""The acceptance instrument for the `mulensevent` split (review 8.6.17).

WHY THIS EXISTS.  8.6.17 collapses parameters that are stored per source but
physically singular, so the model genuinely changes and BYTE-IDENTITY STOPS
BEING AVAILABLE as acceptance -- the first item in this review for which that
is true.  The replacement currency is "every logp delta is explained by a
named term", and that is only possible with a decomposition whose parts
provably sum to the whole.

TWO WRONG DECOMPOSITIONS, both tried while building this and both silently
plausible:

  * sum ``model.potentials`` and add ``model.logp(rv)`` per RV -- 30.9 nats
    SHORT on ob161003, because the transform jacobians are missing;
  * ``model.logp(vars=[rv], jacobian=True, sum=True)`` per RV -- 0.85 nats
    OVER on ob08092 and 27.7 OVER on ob161003, because ``logp(vars=...)``
    does NOT decompose additively: it returns more than that variable's own
    contribution.

THE RIGHT WAY is ``model.logp(sum=False)``, which yields exactly one term per
``basic_RVs + potentials`` -- verified 48 = 11 + 37 on ob08092 and
69 = 17 + 52 on ob161003 -- and reconciles to ~1e-12 relative.

TWO QUESTIONS THAT MUST NOT BE CONFLATED, which is the lesson of the first
version of this file.  It evaluated at ``system.get_raw_start(model)`` -- the
relaxation engine's NUMERICALLY SOLVED start (``sp.nsolve``,
config.py:3839) -- and compared the result across machines.  CI on macOS then
differed by 3.2e-05 nats (7.1e-10 relative) where Linux differed by 7.3e-12
(1.6e-16, i.e. machine epsilon).  Widening the tolerance would have buried
the interesting part, because those two numbers can mean very different
things:

  1. THE START POINT MOVED.  A solved start converging slightly differently
     on another platform is a solver-tolerance difference.  Harmless: a chain
     forgets its start.
  2. THE LIKELIHOOD FUNCTION CHANGED.  That would mean the posterior depends
     on the machine it runs on, which is a serious defect.

Evaluating at the engine's own start cannot tell them apart.  So a fixture
now STORES its start point, and the cross-machine comparison is made AT THE
STORED POINT -- identical parameter values on both machines, so any residual
is question 2 alone and the tolerance can be tight.  Question 1 is checked
separately, and strictly, on one machine (``make_mulens_fixtures.py
--check``), where bit-identity is the right bar.

``reconciles`` is checked on every call and callers MUST refuse to emit or
trust a fixture when it is False.  An instrument that cannot be shown to add
up is worse than none, because it yields confident wrong attributions; this
is review 3.14.19's control-must-fire rule applied to the measuring device
itself.
"""

import os

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


def raw_start(system, model):
    """The model's start point as plain lists, ready to store or replay.

    ``system.get_raw_start(model)`` -- NOT ``model.initial_point()``, which is
    keyed by raw value variables and is not the point the model is evaluated
    at, and never ``Parameter.value`` read outside a compiled function, which
    draws from the PRIOR rather than giving the start.  Both are documented
    traps and both produced confidently wrong numbers during review 8.6.18.
    """
    start = system.get_raw_start(model)
    return {k: np.asarray(v, dtype=float).tolist() for k, v in start.items()}


def decompose(system, model, start=None):
    """Return ``(parts, total, reconciles, summed)``.

    ``start`` is a point dict.  Pass the fixture's STORED start to ask "is the
    likelihood function the same"; pass ``None`` to use the engine's own
    solved start and ask "is everything the same, start included".  Those are
    different questions -- see the module docstring.
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
    missing = [v.name for v in value_vars if v.name not in start]
    if missing:
        raise AssertionError(
            f"the start point is missing {missing}; a fixture recorded "
            f"against a different parameter set cannot be replayed, and "
            f"filling the gaps with defaults would silently compare two "
            f"different points"
        )
    fn = pytensor.function(value_vars, terms, on_unused_input="ignore")
    values = fn(*[np.asarray(start[v.name]) for v in value_vars])

    parts = {}
    for name, value in zip(names, values):
        # A duplicate name would otherwise silently drop a term; accumulate
        # so the sum stays honest even if two nodes share a name.
        parts[name] = parts.get(name, 0.0) + float(np.asarray(value).sum())

    total = float(model.compile_logp()(start))
    summed = float(sum(parts.values()))
    reconciles = abs(summed - total) <= RECONCILE_RTOL * max(1.0, abs(total))
    return parts, total, reconciles, summed


def compare(reference, current, atol=0.0, rtol=0.0):
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


def record_deltas(case, rows):
    """Append per-term deltas for the CI dump (review 3.14.20).

    `rows` is an iterable of (name, before, after).  Recorded
    unconditionally, including when every term is inside tolerance: on macOS
    the acceptance tests PASS, so a dump that only fired on failure would
    never answer the question it exists for.

    One file per process because the suite runs under xdist; the controller
    aggregates them in `pytest_terminal_summary`.  Best-effort by design -- a
    diagnostic must never be able to fail a test run, so errors are
    swallowed and a missing EXOZIPPY_DELTA_DIR simply disables it.
    """
    import json

    directory = os.environ.get("EXOZIPPY_DELTA_DIR")
    if not directory:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "deltas-%d.jsonl" % os.getpid())
        with open(path, "a") as fh:
            for name, before, after in rows:
                fh.write(
                    json.dumps(
                        {
                            "case": case,
                            "term": name,
                            "before": before,
                            "after": after,
                        }
                    )
                    + "\n"
                )
    except Exception:
        pass


def compare_points(reference, current, atol=0.0, rtol=0.0):
    """Diff two start points; return {name: (before, after, max_abs_delta)}.

    Separate from ``compare`` because a start difference and a likelihood
    difference are different findings: the first is the relaxation engine
    converging differently, the second means the posterior depends on the
    machine.
    """
    out = {}
    for name, before in reference.items():
        if name not in current:
            out[name] = (before, None, float("inf"))
            continue
        b = np.atleast_1d(np.asarray(before, dtype=float))
        c = np.atleast_1d(np.asarray(current[name], dtype=float))
        if b.shape != c.shape:
            out[name] = (before, current[name], float("inf"))
            continue
        d = np.abs(c - b)
        if np.any(d > atol + rtol * np.abs(b)):
            out[name] = (before, current[name], float(d.max()))
    for name in current:
        if name not in reference:
            out[name] = (None, current[name], float("inf"))
    return out
