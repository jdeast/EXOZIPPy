"""Every sampler branch must be handed the polished start (review 1.3.6).

`System.get_raw_start` exists to OVERRIDE `Model.initial_point()` -- its own
docstring says so -- and `System.apply_polished_starts` writes the seed
polish's result into each `Parameter.raw_initval` as a NONZERO raw offset,
"which get_raw_start, get_mcmc_init, and the sampler initvals all read".

The plain-NUTS branch read none of it.  `pm.sample` was called with an
explicit `step=pm.NUTS(...)` and no `initvals=`, and pymc 6.3.2's own
docstring says of `init`, verbatim, "This argument is ignored when manually
passing the NUTS step method" -- so the DEFAULT sampler for every
non-microlensing fit began at `Model.initial_point()`, the creation-time
initvals frozen at RV creation.  It looked correct by construction for as
long as the anchor WAS the start: before a polish every logit element's
`raw_initval` is 0, which is exactly what `initial_point()` holds.  A polish
is what separates them.

What that cost is NOT the whitening -- the whitening is a reparameterization
in the graph and the sampler samples `raw`, so a `pm.NUTS(target_accept=)`
identity metric on raw stays the right metric wherever the chain starts.
What is lost is the CO-LOCATION of the start with the metric: the probe
measures the scales around the POLISHED point (which is why the polish runs
before it) and the chain began at the anchor, so NUTS's first adaptation
windows saw curvature the metric was never measured for.  That is the
ob140939 mechanism, on the default sampler alone.

Two shapes of test, because the defect had two halves:

* the DYNAMIC test drives `_run_fit` to the dispatch with a polish that has
  demonstrably moved `raw_initval` off the anchor, and pins that the
  `initvals` kwarg equals `get_mcmc_init(model)` -- asserting only "initvals
  was passed" would pass against a pre-polish anchor and prove nothing;
* the STATIC every-call-site guard (the tests/test_random_seed.py shape --
  exactly the shape that would have caught this) pins that EVERY branch
  run.py dispatches to still consumes a start, including the ones that need
  jax, nutpie or a cluster and so are never exercised in the suite.

The explicit `step` is deliberately KEPT rather than dropped in favor of a
live `init`: `initvals`' own docstring entry reads "Initialization methods
for NUTS (see ``init`` keyword) can overwrite the default", so with `init`
live an adapt_diag jitter could move the chain back off the polished point --
the pathology seed_polish exists to prevent.  Keeping the step is what makes
`initvals=` authoritative, and `test_the_nuts_branch_keeps_its_explicit_step`
pins that.
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

# Every sampler entry point run.py dispatches to, and the regex its call
# source must match for the start to reach it.  The spellings are not ours to
# choose for pymc's two; the in-house samplers all take `raw_starts`.
_START_KWARGS = {
    "sample_jax_nuts": r"\binitvals\s*=\s*internal_start\b",
    "ptde_sample": r"\braw_starts\s*=\s*raw_starts\b",
    "ptde_async_sample": r"\braw_starts\s*=\s*raw_starts\b",
    "de_metropolis_sample": r"\braw_starts\s*=\s*raw_starts\b",
}

# `nested_sample` is the one exemption and it is a design statement, not an
# oversight: nested sampling draws its live points from the PRIOR, so there is
# no start to hand it.  run.py says so out loud at the call site.  The
# exemption is asserted to FIRE below, so broadening this set cannot silently
# empty the guard.
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
def test_every_sampler_call_site_consumes_the_start(func, pattern):
    """
    Given each sampler run.py can dispatch to,
    When its call site is read out of run.py's source,
    Then it passes the start the polish and the whitening rescale produced.

    Regression: the plain-NUTS branch passed nothing, and it is the default.
    This is the static shape that would have caught it -- these four branches
    were always correct, and pinning them is what stops a FUTURE branch (or a
    refactor of an existing one) from dropping the start unnoticed.
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


@pytest.mark.parametrize("func", sorted(_NO_START_EXPECTED))
def test_the_no_start_exemption_fires_and_says_why(func):
    """
    Given nested sampling, the one branch that takes no start,
    When its call site is read,
    Then it really passes no start AND run.py explains that the search is
      prior-driven -- so the exemption is a statement, not a hole.

    docs/testing.md rule 3: an exemption must be asserted to FIRE, or a
    later change that broadens it silently empties the guard above.  Both
    sets are disjoint by construction and every branch is in exactly one.
    """
    assert not (_START_KWARGS.keys() & _NO_START_EXPECTED)

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


def test_both_pm_sample_branches_consume_the_start():
    """
    Given run.py's two pm.sample branches (plain NUTS and nutpie),
    When their call sites are read,
    Then the nutpie one passes init_mean and the plain one passes initvals.

    Handled apart from the parametrized cases because `pm.sample` is an
    attribute call with more than one call site, and the two branches spell
    the start differently: nutpie ignores `initvals` entirely and takes a flat
    `init_mean` array instead.
    """
    src = RUN_PY.read_text(encoding="utf-8")

    # ACT
    pm_calls = [
        c for c in _call_sources(src, "sample") if c.startswith("pm.sample(")
    ]

    # ASSERT
    assert len(pm_calls) >= 2, f"expected 2+ pm.sample calls, got {pm_calls}"
    nutpie = [c for c in pm_calls if "nutpie" in c]
    plain = [c for c in pm_calls if "nutpie" not in c]
    assert len(nutpie) == 1, f"expected one nutpie branch, got {len(nutpie)}"
    assert len(plain) == 1, f"expected one plain-NUTS branch, got {len(plain)}"

    assert re.search(r"\binit_mean\b", nutpie[0]), (
        f"the nutpie branch no longer passes init_mean:\n{nutpie[0]}"
    )
    assert re.search(r"\binitvals\s*=\s*transformed_inits\b", plain[0]), (
        "the plain-NUTS branch does not pass initvals=transformed_inits, so "
        "the chains start from Model.initial_point() and the seed polish is "
        f"a no-op on the DEFAULT sampler (review 1.3.6):\n{plain[0]}"
    )


def test_the_nuts_branch_keeps_its_explicit_step():
    """
    Given the plain-NUTS branch,
    When its call site is read,
    Then it still passes an explicit step= -- which is what makes initvals=
      authoritative.

    Dropping the step to make `init` live is the remedy this fix deliberately
    did NOT take: `initvals`' pymc docstring entry says the NUTS
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
        "overwrite initvals and jitter the chain off the polished start"
    )


def test_pymc_still_ignores_init_when_a_step_is_passed():
    """
    Given the installed pymc,
    When pm.sample's docstring is read,
    Then it still states that `init` is ignored for a manually passed step.

    This is the upstream fact the whole item rests on, pinned rather than
    quoted: if a future pymc starts honoring `init` alongside an explicit
    step, `init=init` stops being inert and the interaction with initvals
    must be re-thought (see review 5.3.3).
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
# Dynamic test: a polish that MOVED the start must reach pm.sample
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
    every sampled element's raw start off the anchor, and capture what
    pm.sample was called with.

    The polish is STUBBED rather than run: a real L-BFGS polish on this
    prior-only model would land back on the anchor (for a logit element the
    correction potential leaves a raw-space density peaked at raw = 0), so a
    real polish here would make the test vacuous.  Everything downstream of
    the stub is the production path -- apply_polished_starts writes
    raw_initval, the whitening probe rescales it, get_raw_start /
    get_mcmc_init re-read it, run.py dispatches.

    Module-scoped (and so patching through ``pytest.MonkeyPatch.context()``
    rather than the function-scoped ``monkeypatch`` fixture) because it
    builds a System and compiles PyTensor graphs, which on a cold cache is
    the whole cost of this file.  ``--dist loadfile`` pins the module to one
    worker, so the two consumers share the one build.
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

    def stub_sample(*args, **kwargs):
        # Inside run.py's `with model:` block, so the context IS the model
        # under test -- which is also the only place to get hold of it, since
        # System does not keep a reference.
        model = pm.modelcontext(None)
        captured["model"] = model
        raise _StopAtDispatch(kwargs, model.initial_point())

    _real_get_mcmc_init = System.get_mcmc_init

    orig_cwd = os.getcwd()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_module, "polish_raw_starts", stub_polish)
        mp.setattr(System, "get_mcmc_init", spy_get_mcmc_init)
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


def test_the_polish_moved_the_start_off_the_anchor(nuts_dispatch):
    """
    Given a polish that displaced every sampled element,
    When the model's creation-time initial_point is compared to the start
      run.py resolved,
    Then EVERY value variable differs, by an amount no tolerance can absorb.

    This is the test's own precondition, asserted rather than assumed: with
    an UNPOLISHED start every logit element's raw_initval is 0 and the two
    points agree exactly, so a test run against the anchor would pass with or
    without the fix.  That is the vacuity failure docs/testing.md warns
    about, and this assertion is what rules it out.

    The threshold is loose on purpose.  The injected offset is
    _POLISH_OFFSET raw units, but the whitening probe runs AFTER the polish
    and ``set_whitening`` re-expresses a nonzero ``raw_initval`` in the new
    coordinates (dividing by the measured multiplier), so the surviving
    displacement is a measured quantity, not the injected one -- it was
    0.15 sigma when this was written.  What the test needs is that the two
    points are separated by far more than float noise, not a specific
    number, so pinning the number would only make the test brittle against
    an honest re-measurement.
    """
    system = nuts_dispatch["system"]
    model = nuts_dispatch["model"]
    initial_point = nuts_dispatch["initial_point"]

    # The start run.py resolved, read from the System rather than from the
    # captured kwargs: this precondition must hold whether or not the kwargs
    # carry it.
    resolved = system.get_mcmc_init(model)

    worst = 0.0
    unmoved = []
    for key, value in resolved.items():
        anchor = np.asarray(initial_point[key], dtype=float).ravel()
        start = np.asarray(value, dtype=float).ravel()
        d = float(np.max(np.abs(start - anchor)))
        worst = max(worst, d)
        if d <= 1e-8:
            unmoved.append(key)

    assert not unmoved, (
        f"the stub polish left these value variables on the anchor: {unmoved}"
    )
    assert worst > 0.01, (
        f"the stub polish did not move the start (max |displacement| = "
        f"{worst:.3g} in raw units, i.e. in sigma); the test below would be "
        f"vacuous"
    )


def test_the_nuts_branch_gets_the_polished_start(nuts_dispatch):
    """
    Given the plain-NUTS dispatch reached with a polished start,
    When pm.sample's initvals kwarg is compared to get_mcmc_init(model),
    Then they are equal element for element.

    Regression (review 1.3.6): there was no initvals kwarg at all, so the
    chains began at Model.initial_point() -- see the sibling test, which
    proves that point is a real distance away in raw (= sigma) units here.
    get_mcmc_init is recomputed from the live System rather than compared to
    a spied return value, so the assertion cannot be satisfied by whatever
    the last call happened to produce.
    """
    system = nuts_dispatch["system"]
    model = nuts_dispatch["model"]
    kwargs = nuts_dispatch["kwargs"]

    assert "initvals" in kwargs, (
        "pm.sample got no initvals=: with an explicit step, pymc ignores "
        "init=, so the chains start from Model.initial_point() and the seed "
        "polish is a no-op on the DEFAULT sampler (review 1.3.6)"
    )

    # ACT: recompute the authoritative start from the live System.
    expected = system.get_mcmc_init(model)

    # ASSERT
    got = kwargs["initvals"]
    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(
            np.asarray(got[key], dtype=float),
            np.asarray(expected[key], dtype=float),
            rtol=0,
            atol=0,
            err_msg=f"initvals['{key}'] is not get_mcmc_init's value",
        )
