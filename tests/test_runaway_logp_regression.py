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

Note: the "good" draw's lp is pinned to the CURRENT model's value, not the
historical stored lp (~2982.18) -- the model has legitimately changed since
that trace (log_s sampling, pinned source mass, planet log_q default).  The
check used to be finite-only because System.prepare() was not deterministic
across process runs; that turned out to be PYTHONHASHSEED-ordered iteration
of eq.free_symbols in the relaxation engine's (since-deleted) sympy scale
passes, scattering init_scale by orders of magnitude and with it the
raw->physical map.  The whitening refactor removed the broken passes and
config.py now sorts every free_symbols walk, so the build is bit-for-bit
reproducible and the value check is back.  If a deliberate model change
moves the value, re-measure it at this exact raw point and update
GOOD_EXPECTED_LP.

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
    os.path.dirname(__file__), "..", "examples", "DC2018_128"
)

# chain 23, draw 46680 -- the first stored draw after the runaway (lp jumped
# from ~2982 to ~5.05e7 in the trace; pre-fix, later draws in the same chain
# reached lp ~1e39).
# The separation coordinate is now sampled as log_s (P2, multimode plan): the
# historical value was stored under lens.s_raw. Renaming the key preserves the
# point -- this extreme raw value pins the logit at its upper bound, and old s
# in [0.1, 10] and new log_s in [-1, 1] share the same physical endpoint
# (10**1 = 10), so the physical separation is unchanged.
RUNAWAY_RAW = {
    "band.u1_raw": [-37530.31612758304],
    "lens.log_s_raw": [12868418.484993141],
    "lens.t_0_raw": [597790902870.5624],
    "lens.u_0_raw": [-85147091.07538812],
    "lens.xalpha_raw": [35067.89879449546],
    "lens.yalpha_raw": [-207142.0549616936],
    "mulensinstrument.err_scale_raw": [227.84215680882753],
    "mulensinstrument.log_f_total_raw": [34254.43409836991],
    "mulensinstrument.q_source_raw": [-101671.55477455864],
    "planet.mass_raw": [195576323.76112023],
    "star.distance_raw": [-5303.75771384, -8672.6901092],
    # star.1 (Source) is a pure microlensing source: star.py now pins its
    # mass (nothing in mulensing physics consumes a source's mass), so only
    # star.0 (Lens)'s historical raw value remains sampled.
    "star.logmass_raw": [3950.28058978],
    "star.pm_dec_raw": [3.12510181e05, 1.57096079e08],
    "star.pm_ra_raw": [-3.34082488e08, -4.12501179e07],
    "star.rv_raw": [2201.98526294, -691.32034664],
}

# Same chain, draw 46660 (20 steps earlier): the last ordinary, non-runaway
# state (historical stored lp ~= 2982.18).
GOOD_RAW = {
    "band.u1_raw": [6.124144468621876],
    # log_s coordinate (was lens.s_raw pre-P2); extreme-negative raw pins the
    # logit at the shared lower endpoint (10**-1 = 0.1).
    "lens.log_s_raw": [-14.73579507774429],
    "lens.t_0_raw": [0.21660909850932655],
    "lens.u_0_raw": [-0.3258404381227836],
    "lens.xalpha_raw": [-4.220453654077744],
    "lens.yalpha_raw": [9.819914800098061],
    "mulensinstrument.err_scale_raw": [0.6809652656825157],
    "mulensinstrument.log_f_total_raw": [-0.21104036781220858],
    "mulensinstrument.q_source_raw": [-0.31134542581872854],
    "planet.mass_raw": [-0.9441505502539654],
    "star.distance_raw": [-345.90821049, 91.83850293],
    # star.1 (Source)'s mass is now pinned (see RUNAWAY_RAW's comment above).
    "star.logmass_raw": [87.18591661],
    "star.pm_dec_raw": [-5.27994699, 0.40483486],
    "star.pm_ra_raw": [2.99778494, 1.37559868],
    "star.rv_raw": [2.36402842, -1.05406419],
}


@pytest.fixture(scope="module")
def dc2018_128_logp(tmp_path_factory):
    """Build the DC2018_128 model once; return (compile_logp fn, model)."""
    work_dir = tmp_path_factory.mktemp("dc2018_128_work") / "DC2018_128"
    shutil.copytree(
        EXAMPLE_DIR,
        work_dir,
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
    )

    orig_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with open("DC2018_128.yaml") as f:
            config = yaml.safe_load(f)
        with open("DC2018_128.params.yaml") as f:
            user_params = yaml.safe_load(f)

        # The historical draws below were recorded when the planet mass was
        # sampled linearly; a lens body now defaults to log_q.  Unlike the
        # s -> log_s rename, the raw values cannot be carried across: the
        # runaway value pins the logit at an upper bound that means 260000
        # Mjup in one coordinate and q = 10 in the other, and the good draw
        # sits mid-range, where the two coordinates share nothing.  Pin the
        # coordinate the draws came from -- what this file regression-tests
        # (the unclipped raw**2 in the logit-uniform prior correction) lives
        # in Parameter.build_pymc and is the same in either parameterization.
        for entry in config.get("planet", []):
            entry.setdefault("mass_parameterization", "linear")

        system = System(config, user_params)
        system.prepare()
        model = system.build_model()
        logp_fn = model.compile_logp()
    finally:
        os.chdir(orig_cwd)

    return logp_fn


def _point(raw_dict):
    return {k: np.asarray(v, dtype=float) for k, v in raw_dict.items()}


# Measured at this exact raw point on a deterministic build (2026-08-07,
# post free_symbols-sort hardening).  Differs from the historical stored lp
# (~2982.18) because the model itself has changed since that trace; see the
# module docstring.
# Re-measured 2026-08-08 four times: the Op-path annual parallax fix
# (review 1.1) moved the mulens likelihood (-934.604); the galacticmodel
# prior normalization (reviews 1.3/1.4: pm->velocity Jacobian + mixture
# branch normalization) shifted it to -952.076; the genulens-fidelity
# upgrade (thick disk branch, bar cutoff, disk plateau, number-density
# anchors, R0 = 8.16 frame) nudged it to -953.817; the mu_rel helio->geo
# frame fix (t_E/pi_E now derive from mu_rel_geo at t0_par) moved the
# mulens likelihood at this raw point to -944.858.
# Re-measured 2026-08-10: normalizing the galacticmodel Chabrier IMF prior
# over the sampled logmass support (so it is comparable with the new
# IMF: Salpeter option) subtracts the truncated-lognormal constant
# log(sigma*sqrt(2pi)*(Phi(u_hi) - Phi(u_lo))) = +0.3568196 nats PER STAR.
# This config has two (star.Lens, star.Source), so the shift is exactly
# -0.7136392 and nothing else moved.
GOOD_EXPECTED_LP = -945.5716


def test_good_draw_logp_matches_deterministic_build(dc2018_128_logp):
    """
    Given the DC2018_128 model,
    When logp is evaluated at the last ordinary (pre-runaway) raw-space state
    of chain 23,
    Then it reproduces the pinned value of a deterministic build -- both
    confirming GOOD_RAW is read correctly against this model's free_RVs and
    regression-guarding System.prepare()'s run-to-run reproducibility
    (this check was finite-only while the relaxation engine's deleted sympy
    scale passes made init_scale, and with it the raw->physical map,
    PYTHONHASHSEED-dependent).
    """
    logp_fn = dc2018_128_logp
    val = float(np.asarray(logp_fn(_point(GOOD_RAW))))
    assert abs(val - GOOD_EXPECTED_LP) < 5.0, (
        f"lp at the good draw moved: got {val:.4f}, expected "
        f"{GOOD_EXPECTED_LP} +/- 5. Either build reproducibility broke "
        f"(PYTHONHASHSEED sensitivity) or the model changed deliberately -- "
        f"if the latter, re-measure and update GOOD_EXPECTED_LP."
    )


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
