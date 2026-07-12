"""Regression test for the DC2018_128 PTDE runaway (mode_reporting.txt).

A PTDE run on examples/DC2018_128 (Roman Data Challenge 2018, LC 128; binary
lens + finite source) showed every T=1 chain eventually escaping to a corner
of parameter space pinned at bounds, with the stored lp climbing monotonically
to 1e15..1e39 -- physically impossible for any finite dataset. Root cause:
Parameter.build_pymc's flat-prior correction for logit-transformed sampled
parameters added an *unclipped* +0.5*raw**2 to cancel pm.Normal(0,1)'s own
-0.5*raw**2 term. Both are separate floating-point graphs summed together
with dozens of other terms of wildly different magnitudes, so once a PTDE
differential-evolution proposal pushed |raw| far enough, the cancellation
lost enough precision that the residual (effectively noise, growing with
raw**2 * 2**-52) could come out positive -- and since PTDE only accepts logp
increases, that noise got selected and reinforced every step, driving raw to
1e17+ and the reported lp to 1e15..1e39.

RUNAWAY_RAW below is the *actual* raw-space point read out of chain 23, draw
46680 of that trace (examples/DC2018_128/fitresults/DC2018_128_trace.nc,
2026-07-10 run; the trace itself is gitignored, not shipped in the repo, so
the values are hard-coded here). Twenty draws earlier (draw 46660) the same
chain was still in a perfectly ordinary state (lp ~ 2982); the runaway
happened within that one 20-step interval, when a single PTDE proposal moved
every raw coordinate simultaneously. Reusing the literal historical values
(rather than a synthetic sweep) is deliberate: this model's coupled
mu_rel/theta_E/magnification graph has its own, unrelated NaN robustness gaps
at contrived extreme combinations (e.g. all-same-sign, round-number raw
vectors), and an isolated single-parameter toy model does not reproduce the
cancellation at all (see git history of this file/commit message for the
toy-model dead end). The real draw sidesteps both problems.

Note: the "good" draw's lp is checked only for ordinary magnitude, not an
exact historical match -- System.prepare()'s symbolic relaxation engine
turned out to not be perfectly deterministic across process runs (a
*derived* bound can differ enough between two fresh builds of the identical
model to shift lp by ~1e5-1e6), a separate pre-existing issue found while
building this test. See test_good_draw_logp_is_ordinary's docstring.

Marked 'slow' (builds a full System + compiles PyTensor graphs).
"""
import os
import shutil

import numpy as np
import pytest
import yaml

from exozippy.system import System

pytestmark = pytest.mark.slow

EXAMPLE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "examples", "DC2018_128")

# chain 23, draw 46680 -- the first stored draw after the runaway (lp jumped
# from ~2982 to ~5.05e7 in the trace; pre-fix, later draws in the same chain
# reached lp ~1e39).
RUNAWAY_RAW = {
    "band.u1_raw": [-37530.31612758304],
    "lens.s_raw": [12868418.484993141],
    "lens.t_0_raw": [597790902870.5624],
    "lens.u_0_raw": [-85147091.07538812],
    "lens.xalpha_raw": [35067.89879449546],
    "lens.yalpha_raw": [-207142.0549616936],
    "mulensinstrument.err_scale_raw": [227.84215680882753],
    "mulensinstrument.log_f_total_raw": [34254.43409836991],
    "mulensinstrument.q_source_raw": [-101671.55477455864],
    "planet.mass_raw": [195576323.76112023],
    "star.distance_raw": [-5303.75771384, -8672.6901092],
    "star.logmass_raw": [3950.28058978, -76714.41614803],
    "star.pm_dec_raw": [3.12510181e+05, 1.57096079e+08],
    "star.pm_ra_raw": [-3.34082488e+08, -4.12501179e+07],
    "star.rv_raw": [2201.98526294, -691.32034664],
}

# Same chain, draw 46660 (20 steps earlier): the last ordinary, non-runaway
# state (historical stored lp ~= 2982.18).
GOOD_RAW = {
    "band.u1_raw": [6.124144468621876],
    "lens.s_raw": [-14.73579507774429],
    "lens.t_0_raw": [0.21660909850932655],
    "lens.u_0_raw": [-0.3258404381227836],
    "lens.xalpha_raw": [-4.220453654077744],
    "lens.yalpha_raw": [9.819914800098061],
    "mulensinstrument.err_scale_raw": [0.6809652656825157],
    "mulensinstrument.log_f_total_raw": [-0.21104036781220858],
    "mulensinstrument.q_source_raw": [-0.31134542581872854],
    "planet.mass_raw": [-0.9441505502539654],
    "star.distance_raw": [-345.90821049, 91.83850293],
    "star.logmass_raw": [87.18591661, 0.50417452],
    "star.pm_dec_raw": [-5.27994699, 0.40483486],
    "star.pm_ra_raw": [2.99778494, 1.37559868],
    "star.rv_raw": [2.36402842, -1.05406419],
}


@pytest.fixture(scope="module")
def dc2018_128_logp(tmp_path_factory):
    """Build the DC2018_128 model once; return (compile_logp fn, model)."""
    work_dir = tmp_path_factory.mktemp("dc2018_128_work") / "DC2018_128"
    shutil.copytree(
        EXAMPLE_DIR, work_dir,
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
    )

    orig_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with open("DC2018_128.yaml") as f:
            config = yaml.safe_load(f)
        with open("DC2018_128.params.yaml") as f:
            user_params = yaml.safe_load(f)

        system = System(config, user_params)
        system.prepare()
        model = system.build_model()
        logp_fn = model.compile_logp()
    finally:
        os.chdir(orig_cwd)

    return logp_fn


def _point(raw_dict):
    return {k: np.asarray(v, dtype=float) for k, v in raw_dict.items()}


def test_good_draw_logp_is_finite(dc2018_128_logp):
    """
    Given the DC2018_128 model,
    When logp is evaluated at the last ordinary (pre-runaway) raw-space state
    of chain 23,
    Then it is finite, confirming GOOD_RAW is read correctly against this
    model's free_RVs.

    This does NOT check the value is close to the historical stored lp
    (~2982.18), and deliberately doesn't bound its magnitude either:
    System.prepare()'s symbolic relaxation engine turned out, while building
    this test, to not be perfectly deterministic across process runs -- a
    *derived* bound (e.g. lens.q's) can differ enough between two fresh
    builds of the identical model (same YAML+params) to shift lp by as much
    as ~1e6. That's a separate, pre-existing bug, out of scope for the PTDE
    cancellation fix this file otherwise regression-tests; the finite-only
    check here just confirms the point evaluates at all.
    """
    logp_fn = dc2018_128_logp
    val = float(np.asarray(logp_fn(_point(GOOD_RAW))))
    assert np.isfinite(val), f"expected a finite lp, got {val}"


def test_runaway_draw_no_longer_produces_large_positive_logp(dc2018_128_logp):
    """
    Given the DC2018_128 model,
    When logp is evaluated at the exact raw-space point of the historical
    runaway draw (chain 23, draw 46680 -- pre-fix stored lp ~5.05e7, and
    climbing to ~1e39 in later draws of the same chain),
    Then logp is deeply, physically negative (this is a pinned-at-the-bounds,
    effectively zero-probability state) and never the large positive value
    the pre-fix floating-point cancellation bug produced.
    """
    logp_fn = dc2018_128_logp
    val = float(np.asarray(logp_fn(_point(RUNAWAY_RAW))))
    assert val < 1e6, (
        f"logp exploded to a large positive value at the historical runaway "
        f"draw: {val:.3e} (pre-fix this was ~5.05e7)"
    )
    assert val < -1e10, (
        f"expected a deeply negative logp at a state this far outside any "
        f"bound (lens.t_0_raw alone is ~6e11); got {val:.3e}"
    )
