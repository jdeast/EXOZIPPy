"""Every sampler branch must begin at the polished start (reviews 1.3.6, 4.3.1).

HISTORY, because this file's assertions changed meaning once and must not be
read as if they had not.

`System.apply_polished_starts` writes the seed polish's result into each
`Parameter.raw_initval` as a NONZERO raw offset, and `System.get_raw_start`
existed to OVERRIDE `Model.initial_point()` -- its own docstring said so.  The
plain-NUTS branch read none of it: `pm.sample` was called with an explicit
`step=pm.NUTS(...)` and no `initvals=`, and pymc's own docstring says of
`init`, verbatim, "This argument is ignored when manually passing the NUTS
step method" -- so the DEFAULT sampler for every non-microlensing fit began at
`Model.initial_point()`, the creation-time initvals frozen at RV creation.
That was **1.3.6**, and its fix was to pass `initvals=`.  Four other branches
already passed the start by four different spellings (jax `initvals=`, nutpie
`init_mean`, ptde/ptde_async/demc `raw_starts=`), and nothing structurally
stopped the next one from forgetting -- which is how this one was missed.  So
1.3.6 also landed an EVERY-CALL-SITE guard over the whole dispatch, and this
file is it.

**4.3.1 removed the thing that guard was watching, and replaced it with a
stronger property.**  `System.recenter_whitening_anchor` now makes
`Model.initial_point()` the polished start BY CONSTRUCTION, by two mechanisms
because `raw` means two different things on the two element paths:

* LOGIT elements -- the polished displacement is folded into the ANCHOR and
  `raw_initval` is zeroed.  Free: section C's correction potential cancels the
  raw N(0,1) symbolically, so the anchor is pure parameterization and moving
  it changes no density.  `raw = 0` then IS the polished start.
* GAUSSIAN-PATH elements -- the center is NOT touched, because there
  `raw ~ N(0,1)` IS the prior and `gaussian_mus` is the prior MEAN whenever a
  `mu` was given; folding a start displacement in would move the prior, i.e.
  change the model.  `Model.set_initval` carries those instead.

So the three pymc-side branches now pass NO start at all, and the bug class
dies: no future branch can forget a start because there is nothing to pass.

**HOW THE GUARD IS PRESERVED IN SPIRIT.**  Deleting it would have removed the
only structural protection against this bug class, which is the entire
justification for 4.3.1.  It is instead split by mechanism:

1. The in-house samplers still take an explicit `raw_starts=` -- a multi-seed
   start SET is more than one point, so there is nothing structural to
   replace it with -- and `_START_KWARGS` guards those exactly as before.
2. The pymc-side branches are guarded by the MECHANISM instead of by a
   kwarg: `test_run_py_invokes_the_recentering_before_dispatch` pins that
   run.py calls `recenter_whitening_anchor`, and the dynamic tests below pin
   that after it `Model.initial_point()` equals the resolved start element
   for element.  That is strictly stronger than "a kwarg was passed": the old
   guard could be satisfied by passing a WRONG dict, and this cannot.
3. `nested_sample`'s exemption is still asserted to FIRE (docs/testing.md
   rule 3), so broadening it cannot silently empty the guard.

The DYNAMIC tests keep the shape 1.3.6 gave them, including its
non-vacuity precondition: a polish that demonstrably moved the start.  The
comparison point moved from "the start resolved at dispatch" to "the model's
own initial point BEFORE the re-centering ran", which is the same number
1.3.6 was measuring -- the creation-time anchor -- captured one step earlier
because the fix now erases the difference by the time the dispatch is
reached.
"""

import ast
import os
import re
from pathlib import Path

import numpy as np
import pymc as pm
import pytest

from exozippy import run as run_module

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "src" / "exozippy" / "run.py"


# ----------------------------------------------------------------------
# Static every-call-site guard
# ----------------------------------------------------------------------

# The sampler entry points that still take an EXPLICIT start, and the regex
# their call source must match.  All three are in-house and all three take a
# multi-seed start SET (`raw_starts`), which `Model.initial_point()` cannot
# express -- it holds one point.  So these keep the 1.3.6 guard unchanged.
_START_KWARGS = {
    "ptde_sample": r"\braw_starts\s*=\s*raw_starts\b",
    "ptde_async_sample": r"\braw_starts\s*=\s*raw_starts\b",
    "de_metropolis_sample": r"\braw_starts\s*=\s*raw_starts\b",
}

# The branches that take NO start, because `Model.initial_point()` IS the
# polished start after `recenter_whitening_anchor` (review 4.3.1).  They are
# guarded by the mechanism (below) rather than by a kwarg.  `sample_jax_nuts`
# is here because `set_initval` was measured to place its first draw bit for
# bit where `initvals=` did; the two `pm.sample` calls are here because
# pymc reads the model's initial point for both.
_MODEL_START_BRANCHES = {"sample_jax_nuts"}

# `nested_sample` is the one branch that takes no start for a DIFFERENT
# reason, and it is a design statement rather than an oversight: nested
# sampling draws its live points from the PRIOR, so there is no start to hand
# it and no model initial point it would consult either.  run.py says so out
# loud at the call site.  The exemption is asserted to FIRE below.
_NO_START_EXPECTED = {"nested_sample"}


def _call_sources(src, func_name):
    """Every `func_name(...)` call's source text in ``src``."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name == func_name:
            out.append(ast.get_source_segment(src, node))
    return out


@pytest.mark.parametrize("func,pattern", sorted(_START_KWARGS.items()))
def test_every_multiseed_call_site_consumes_the_start(func, pattern):
    """
    Given each in-house sampler run.py can dispatch to,
    When its call site is read out of run.py's source,
    Then it passes the multi-seed start set the polish produced.

    Regression (1.3.6): the plain-NUTS branch passed nothing, and it is the
    default.  These three were always correct, and pinning them is what stops
    a FUTURE branch (or a refactor of an existing one) from dropping the
    start unnoticed.  They cannot be converted to the structural mechanism
    4.3.1 gave the pymc branches, because a start SET is not a point:
    `Model.initial_point()` holds seed 0 and nothing else.
    """
    src = RUN_PY.read_text(encoding="utf-8")

    # ACT
    calls = _call_sources(src, func)

    # ASSERT
    assert calls, f"no {func}(...) call found in run.py"
    for call in calls:
        assert re.search(pattern, call), (
            f"{func}(...) no longer consumes the polished start "
            f"(expected {pattern!r}):\n{call}"
        )


def test_run_py_invokes_the_recentering_before_dispatch():
    """
    Given run.py's sampling path,
    When its source is read,
    Then `recenter_whitening_anchor` is called, and BEFORE the first sampler
      dispatch.

    This is what replaces the per-kwarg guard for the pymc-side branches
    (review 4.3.1).  With it, `Model.initial_point()` is the polished start
    and no branch needs handing one; without it, all three pymc branches
    silently begin at the creation-time anchor -- 1.3.6's defect, restored on
    three branches instead of one.  Ordering matters as much as presence: the
    whitening probe must measure its contours around the NEW anchor, which is
    why run.py places the call between the polish and the probe.
    """
    src = RUN_PY.read_text(encoding="utf-8")

    # ACT
    calls = _call_sources(src, "recenter_whitening_anchor")

    # ASSERT
    assert len(calls) == 1, (
        f"expected exactly one recenter_whitening_anchor(...) call in "
        f"run.py, found {len(calls)}: {calls}"
    )
    at = src.index("recenter_whitening_anchor(")
    probe_at = src.index("prepare_whitening(")
    dispatch_at = min(
        src.index(f"{name}(")
        for name in sorted(_START_KWARGS) + sorted(_MODEL_START_BRANCHES)
        if f"{name}(" in src
    )
    assert at < probe_at, (
        "the re-centering runs AFTER the whitening probe, so the probe "
        "measured its contours around the old anchor"
    )
    assert at < dispatch_at, (
        "the re-centering runs after a sampler dispatch, so that sampler "
        "starts from the creation-time anchor"
    )


@pytest.mark.parametrize("func", sorted(_NO_START_EXPECTED))
def test_the_no_start_exemption_fires_and_says_why(func):
    """
    Given nested sampling, the one branch whose search is prior-driven,
    When its call site is read,
    Then it really passes no start AND run.py explains why -- so the
      exemption is a statement, not a hole.

    docs/testing.md rule 3: an exemption must be asserted to FIRE, or a
    later change that broadens it silently empties the guard above.  The
    three sets are disjoint by construction and every branch is in exactly
    one.
    """
    assert not (_START_KWARGS.keys() & _NO_START_EXPECTED)
    assert not (_MODEL_START_BRANCHES & _NO_START_EXPECTED)
    assert not (_START_KWARGS.keys() & _MODEL_START_BRANCHES)

    src = RUN_PY.read_text(encoding="utf-8")

    calls = _call_sources(src, func)

    assert calls, f"{func}(...) call vanished from run.py"
    exemption_fired = any(
        not re.search(r"\braw_starts\s*=", call) for call in calls
    )
    assert exemption_fired, (
        "nested_sample now takes a start: move it out of _NO_START_EXPECTED "
        "and into _START_KWARGS"
    )
    assert "prior-driven" in src, (
        "the reason nested takes no start is no longer recorded at the call "
        "site; without it this exemption reads as an omission"
    )


def test_the_nuts_branch_keeps_its_explicit_step():
    """
    Given the plain-NUTS branch,
    When its call site is read,
    Then it still passes an explicit step= -- which is what makes
      `Model.initial_point()` authoritative.

    Dropping the step to make `init` live is the remedy neither 1.3.6 nor
    4.3.1 took: `initvals`' pymc docstring entry says the NUTS
    initialization methods "can overwrite the default", so a live adapt_diag
    could jitter the chain off the polished point.
    """
    src = RUN_PY.read_text(encoding="utf-8")

    plain = [
        c
        for c in _call_sources(src, "sample")
        if c.startswith("pm.sample(") and "nutpie" not in c
    ]

    assert len(plain) == 1
    assert re.search(r"\bstep\s*=\s*step\b", plain[0]), (
        "the plain-NUTS branch dropped its explicit step; a live `init` can "
        "overwrite the model's initial point and jitter the chain off the "
        "polished start"
    )


def test_pymc_still_ignores_init_when_a_step_is_passed():
    """
    Given the installed pymc,
    When pm.sample's docstring is read,
    Then it still states that `init` is ignored for a manually passed step.

    This is the upstream fact the whole item rests on, pinned rather than
    quoted: if a future pymc starts honoring `init` alongside an explicit
    step, `init=init` stops being inert and could overwrite the model's
    initial point (see review 5.3.3).
    """
    doc = pm.sample.__doc__ or ""

    assert (
        "This argument is ignored when manually passing the NUTS step method"
        in doc
    )
    assert (
        "Initialization methods for NUTS (see ``init`` keyword) can overwrite"
        in doc
    )


# ----------------------------------------------------------------------
# Dynamic test: a polish that MOVED the start must reach the sampler
# ----------------------------------------------------------------------

# Every sampled element's raw start is displaced by this much, in raw
# (whitened) units where the raw prior is N(0, 1) -- so it is a displacement
# in sigma, and it is far outside any rounding tolerance.
_POLISH_OFFSET = 0.75


class _StopAtDispatch(Exception):
    """Raised from the monkeypatched pm.sample to end the run at the
    dispatch, before any draw is taken."""

    def __init__(self, kwargs, initial_point):
        super().__init__("dispatch reached")
        self.kwargs = kwargs
        self.initial_point = initial_point


@pytest.fixture(scope="module")
def nuts_dispatch(tmp_path_factory):
    """Drive _run_fit to the plain-NUTS dispatch with a polish that has moved
    every sampled element's raw start off the anchor, and capture (a) the
    model's initial point just BEFORE the re-centering, (b) the same at the
    dispatch, and (c) what pm.sample was called with.

    The polish is STUBBED rather than run: a real L-BFGS polish on this
    prior-only model would land back on the anchor (for a logit element the
    correction potential leaves a raw-space density peaked at raw = 0), so a
    real polish here would make the test vacuous.  Everything downstream of
    the stub is the production path -- apply_polished_starts writes
    raw_initval, recenter_whitening_anchor folds it into the anchor and calls
    Model.set_initval, the whitening probe rescales, get_raw_start /
    get_mcmc_init re-read, run.py dispatches.

    Module-scoped (and so patching through ``pytest.MonkeyPatch.context()``
    rather than the function-scoped ``monkeypatch`` fixture) because it
    builds a System and compiles PyTensor graphs, which on a cold cache is
    the whole cost of this file.  ``--dist loadfile`` pins the module to one
    worker, so every consumer shares the one build.
    """
    from exozippy.system import System

    tmp_path = tmp_path_factory.mktemp("nuts_start")

    # A minimal, cheap-to-build System: one free orbit, no instruments and no
    # likelihood.  We need real Parameters, real free_RVs and the real run.py
    # dispatch, not a physically meaningful fit.
    config = {
        "name": "nuts_start_test",
        "orbit": [{"name": "test_orbit"}],
        "prefix": str(tmp_path / "nutsstart"),
        "sampler": {
            "method": "nuts",
            "tune": 1,
            "draws": 1,
            "chains": 1,
            "cores": 1,
            "recompute_trace": True,
        },
    }
    user_params = {
        "orbit.test_orbit.logP": {"initval": float(np.log10(10.0))},
        "orbit.test_orbit.tc": {"initval": 0.0},
        "orbit.test_orbit.secosw": {"initval": 0.0},
        "orbit.test_orbit.sesinw": {"initval": 0.0},
    }

    captured = {}

    def stub_polish(model, raw_starts, **kwargs):
        """Return the same starts displaced by _POLISH_OFFSET, in the shape
        polish_raw_starts promises: (polished_raws, dlps, method_name)."""
        polished = [
            {
                k: np.asarray(v, dtype=float) + _POLISH_OFFSET
                for k, v in s.items()
            }
            for s in raw_starts
        ]
        return polished, [0.0] * len(polished), "stub"

    def spy_get_mcmc_init(self, model):
        captured["system"] = self
        return _real_get_mcmc_init(self, model)

    def spy_recenter(self, model):
        """Record the state 1.3.6 was measuring -- the model's own initial
        point while it is still the creation-time anchor -- plus the start
        the polish resolved, the logp at each, the PHYSICAL values at each,
        and sum(raw**2), which sets the float64 cancellation bound below."""
        logp = model.compile_logp()
        pre = {
            k: np.asarray(v, dtype=float).copy()
            for k, v in model.initial_point().items()
        }
        resolved = self.get_raw_start(model)
        captured["anchor_before"] = pre
        captured["resolved_before"] = {
            k: np.asarray(v, dtype=float).copy() for k, v in resolved.items()
        }
        captured["lp_before"] = float(logp(resolved))
        captured["phys_before"] = self.get_internal_point(model, resolved)
        captured["raw_sq_before"] = sum(
            float(np.sum(np.asarray(v, dtype=float) ** 2))
            for v in resolved.values()
        )
        out = _real_recenter(self, model)
        captured["moved"] = out
        after = self.get_raw_start(model)
        captured["lp_after"] = float(logp(after))
        captured["phys_after"] = self.get_internal_point(model, after)
        return out

    def stub_sample(*args, **kwargs):
        # Inside run.py's `with model:` block, so the context IS the model
        # under test -- which is also the only place to get hold of it, since
        # System does not keep a reference.
        model = pm.modelcontext(None)
        captured["model"] = model
        raise _StopAtDispatch(kwargs, model.initial_point())

    _real_get_mcmc_init = System.get_mcmc_init
    _real_recenter = System.recenter_whitening_anchor

    orig_cwd = os.getcwd()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_module, "polish_raw_starts", stub_polish)
        mp.setattr(System, "get_mcmc_init", spy_get_mcmc_init)
        mp.setattr(System, "recenter_whitening_anchor", spy_recenter)
        mp.setattr(pm, "sample", stub_sample)
        os.chdir(tmp_path)
        try:
            with pytest.raises(_StopAtDispatch) as excinfo:
                run_module.run_fit(config, user_params=user_params)
        finally:
            os.chdir(orig_cwd)

    captured["kwargs"] = excinfo.value.kwargs
    captured["initial_point"] = excinfo.value.initial_point
    return captured


def test_the_polish_moved_the_start_off_the_creation_time_anchor(
    nuts_dispatch,
):
    """
    Given a polish that displaced every sampled element,
    When the model's creation-time initial point is compared to the start
      the polish resolved, at the moment before the re-centering runs,
    Then EVERY value variable differs, by an amount no tolerance can absorb.

    This is the test's own precondition, asserted rather than assumed, and it
    is 1.3.6's precondition measured one step earlier.  With an UNPOLISHED
    start every logit element's raw_initval is 0 and the two points agree
    exactly, so a test run against the anchor would pass with or without
    either fix.  That is the vacuity failure docs/testing.md warns about, and
    this assertion is what rules it out.

    The threshold is loose on purpose: the injected offset is _POLISH_OFFSET
    raw units, but what must hold is that the two points are separated by far
    more than float noise -- pinning the number would only make the test
    brittle against an honest re-measurement.
    """
    anchor = nuts_dispatch["anchor_before"]
    resolved = nuts_dispatch["resolved_before"]

    worst = 0.0
    unmoved = []
    for key, value in resolved.items():
        a = np.asarray(anchor[key], dtype=float).ravel()
        start = np.asarray(value, dtype=float).ravel()
        d = float(np.max(np.abs(start - a)))
        worst = max(worst, d)
        if d <= 1e-8:
            unmoved.append(key)

    assert not unmoved, (
        f"the stub polish left these value variables on the anchor: {unmoved}"
    )
    assert worst > 0.01, (
        f"the stub polish did not move the start (max |displacement| = "
        f"{worst:.3g} in raw units, i.e. in sigma); every test below would "
        f"be vacuous"
    )
    assert nuts_dispatch["moved"], (
        "recenter_whitening_anchor reported no element moved, so the "
        "mechanism under test never fired"
    )


def test_the_model_start_is_the_polished_start_by_construction(nuts_dispatch):
    """
    Given the plain-NUTS dispatch reached with a polished start,
    When `Model.initial_point()` at the dispatch is compared to
      `get_raw_start(model)` and to `get_mcmc_init(model)`,
    Then all three agree exactly, element for element.

    This is 4.3.1's whole claim and what replaces 1.3.6's "initvals was
    passed" assertion.  It is strictly stronger: the old guard could be
    satisfied by passing a wrong dict, and this cannot -- the model's OWN
    start is the polished one, so there is nothing left to pass and nothing
    left for a future branch to forget.  The sibling test proves the two
    points were a real distance apart before the re-centering ran.

    Recomputed from the live System rather than compared to a spied return
    value, so the assertion cannot be satisfied by whatever the last call
    happened to produce.
    """
    system = nuts_dispatch["system"]
    model = nuts_dispatch["model"]
    initial_point = nuts_dispatch["initial_point"]

    # ACT
    raw_start = system.get_raw_start(model)
    mcmc_init = system.get_mcmc_init(model)

    # ASSERT
    assert set(initial_point) == set(raw_start) == set(mcmc_init)
    for key in raw_start:
        for name, other in (
            ("get_raw_start", raw_start),
            ("get_mcmc_init", mcmc_init),
        ):
            np.testing.assert_allclose(
                np.asarray(initial_point[key], dtype=float),
                np.asarray(other[key], dtype=float),
                rtol=0,
                atol=0,
                err_msg=(
                    f"Model.initial_point()['{key}'] is not {name}'s value, "
                    f"so the chains do not start where the whitening probe "
                    f"measured and the startup table reports (4.3.1)"
                ),
            )


def test_the_dispatch_passes_no_start_of_its_own(nuts_dispatch):
    """
    Given the plain-NUTS dispatch,
    When pm.sample's kwargs are read,
    Then no `initvals` is passed -- the model's initial point is the start.

    The deletion is the point, not a tidy-up: five branches passing a start
    by five spellings is what made 1.3.6 possible, and a re-added `initvals=`
    here would be a start that can drift out of agreement with
    `Model.initial_point()` again.  The sibling test is what makes this safe
    to assert; on its own this one would be satisfied by a broken start too.
    """
    kwargs = nuts_dispatch["kwargs"]

    assert "initvals" not in kwargs, (
        "pm.sample is being handed initvals= again; after 4.3.1 the model's "
        "own initial point IS the polished start, and a second channel for "
        "it is exactly what let one branch disagree with the others"
    )
    # ...and the call is otherwise the production one, so this is not a
    # vacuous pass against a dispatch that never happened.
    assert kwargs.get("step") is not None
    assert "random_seed" in kwargs


def test_recentering_leaves_every_physical_value_bit_identical(nuts_dispatch):
    """
    Given a real System whose anchor was re-centered on a polished start,
    When every free RV and Deterministic is compared before and after,
    Then every PHYSICAL value is bit-identical; only the raw coordinates
      move, and they move to 0.

    This is the claim that makes the re-centering a change of COORDINATES.
    The new anchor is the old `lq` AT the start, so the value `raw = 0`
    decodes to is the value the displaced coordinate decoded to -- to the
    last bit, not to a tolerance.  If a physical value moves, the polished
    start has been silently relocated, which is the one thing this mechanism
    promises not to do.
    """
    before = nuts_dispatch["phys_before"]
    after = nuts_dispatch["phys_after"]

    raw_keys = set(nuts_dispatch["resolved_before"])
    physical = [k for k in before if k not in raw_keys]
    assert physical, "no physical (non-raw) node to compare"

    for key in physical:
        np.testing.assert_array_equal(
            np.asarray(after[key], dtype=float),
            np.asarray(before[key], dtype=float),
            err_msg=(
                f"'{key}' moved across the re-centering, so the polished "
                f"physical start was relocated rather than re-expressed"
            ),
        )

    # ...and the raw coordinates DID move, or the above is vacuous.
    assert any(
        not np.array_equal(
            np.asarray(after[k], dtype=float),
            np.asarray(before[k], dtype=float),
        )
        for k in raw_keys
    )


def test_the_only_logp_change_is_the_cancellation_it_removes(nuts_dispatch):
    """
    Given the same re-centering,
    When the TOTAL start logp is compared before and after,
    Then it agrees to within the float64 cancellation residual of the
      coordinate being abandoned, `0.5 * sum(raw**2) * 2**-52`.

    THE START LOGP IS NOT EXACTLY EQUAL, AND THAT IS NOT A MODEL CHANGE --
    recorded here because the obvious assertion is exact equality and it is
    wrong for a reason worth keeping.  Section C's correction potential
    cancels the raw N(0,1) SYMBOLICALLY, so each `(RV:X_raw,
    POT:logit_uniform_prior.X)` pair moves by exactly equal and opposite
    amounts (verified term by term on `examples/kelt4`: +2.347e5 against
    -2.347e5 on `rvinstrument.jitter_variance`, +226.2 against -226.2 on
    `orbit.cosi`, and so on for all 13 pairs).  What does not cancel is
    float64 round-off: summing two numbers of magnitude `0.5*raw**2` to get
    a residual of order 1 loses `0.5*raw**2 * 2**-52` nats, and that is the
    same cancellation `parameter.py`'s `_RAW_CANCELLATION_CLIP` and the
    runaway-lp bug are both about.

    Re-centering drives it to ZERO, because the start becomes `raw = 0`.  So
    the direction is the good one: same density, better arithmetic.  Measured
    on `examples/kelt4` with a real L-BFGS polish, whose worst element sat
    684.9 raw units out: total start logp 81.44003885515859 ->
    81.44003885517124, a move of 1.3e-11 nats (1.5e-13 relative) against a
    predicted bound of 5.2e-11 -- and every physical value bit-identical
    (the sibling test).

    The bound is DERIVED from the measured displacement rather than tuned,
    so this test stays meaningful whatever the polish does.  The factor of 4
    is slack for a bound that assumes the per-pair round-off adds in phase
    when in practice it partially cancels across pairs (kelt4 came out at
    0.24x).
    """
    before = nuts_dispatch["lp_before"]
    after = nuts_dispatch["lp_after"]
    raw_sq = nuts_dispatch["raw_sq_before"]

    # The residual the abandoned coordinate carried, plus a floor for the
    # ordinary last-bit noise of summing the terms at all.
    bound = 4.0 * (0.5 * raw_sq * 2.0**-52) + 8.0 * abs(before) * 2.0**-52

    assert abs(after - before) <= bound, (
        f"re-centering moved the start logp by {after - before!r} nats "
        f"({before!r} -> {after!r}), more than the float64 cancellation "
        f"residual {bound:.3g} that the abandoned coordinate "
        f"(sum(raw**2) = {raw_sq:.4g}) can account for. The anchor has been "
        f"treated as a posterior term somewhere -- which is exactly what the "
        f"Gaussian path forbids, and why that path goes through "
        f"Model.set_initval instead of the anchor"
    )
