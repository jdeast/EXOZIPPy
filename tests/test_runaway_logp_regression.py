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

GOOD_RAW is a fresh ordinary draw (chain 35, draw 1689 of a short run under the
current code), not the historical one, and the fixture restores the whitening it
was sampled under.  Both details are load-bearing, and the reason is the same:
a raw vector is not a portable description of a physical state.  It is a
coordinate in the WHITENED space and an offset from the initvals, so it moves if
either changes.  Evaluated without this run's whitening, this draw gives logp
-1.6e6 and chi2/N 3671 instead of 2979.9175 and 1.002.  The previous historical
draw stopped being "good" the same way, when #99 changed the proper-motion start
values under it.

So there are two distinct failure modes, and the fix differs:
  - the MODEL changed on purpose -> re-measure GOOD_EXPECTED_LP at this same raw
    point (the git log of this file lists the earlier such re-measurements);
  - the START VALUES changed -> the raw point now decodes elsewhere, and the
    honest fix is a fresh draw from a fresh run, not a new constant pinned to a
    stale point.  test_good_draw_is_actually_a_good_fit below is what tells the
    two apart: it fails only in the second case.

The value check used to be finite-only because System.prepare() was not
deterministic across process runs; that turned out to be PYTHONHASHSEED-ordered
iteration of eq.free_symbols in the relaxation engine's (since-deleted) sympy
scale passes.  The whitening refactor removed those and config.py now sorts
every free_symbols walk (and, since #92, every component-discovery walk), so the
build is bit-for-bit reproducible and the value check is back.

Marked 'slow' (builds a full System + compiles PyTensor graphs).
"""

import os
import shutil

import numpy as np
import pytensor
import pytest
import yaml

from exozippy import whitening
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
# band.u1_raw is likewise gone: this example is `finite_source: False`, so
# nothing reads the Z087 band's limb darkening, and Band now pins it (the same
# "overrides" mechanism as star.py's source-star pins, one topology further
# along).  Its historical value was -37530.31612758304; the parameter no longer
# exists, so the coordinate cannot be supplied.  The whitening fixture lost its
# band.u1 entry for the same reason -- every other entry is untouched.
RUNAWAY_RAW = {
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

# An ORDINARY, well-fitting draw from a fresh run under the current code:
# chain 35, draw 1689 of a short (4-temp, 500-tune, 2000-draw) fit of this same
# example, whose stored lp was 2979.9175 and whose mulens chi2/N is 1.002.
#
# Why a fresh draw rather than the historical one (chain 23, draw 46660): raw
# values are coordinates in the WHITENED space, so they only mean anything under
# the whitening they were sampled with, and they are offsets from the initvals,
# so they also move when a start value changes.  The galactic-model proper-motion
# seeding (#99) changed the initvals, which left the old draw decoding to a state
# with chi2/N ~ 261 -- still fine as a determinism probe, but no longer the
# "good draw" its name promised.  Evaluated WITHOUT the run's whitening this
# draw gives logp -1.6e6 and chi2/N 3671, which is what makes the fixture below
# restore it; with it, logp reproduces the trace's stored lp exactly.
# (band.u1_raw was 0.2176204790834511 here; see the note on RUNAWAY_RAW.)
GOOD_RAW = {
    "lens.log_s_raw": [-16.184426615189963],
    "lens.t_0_raw": [4.88478412864818],
    "lens.u_0_raw": [-42.52723267883055],
    "lens.xalpha_raw": [-109.86398001224084],
    "lens.yalpha_raw": [123.27869830685752],
    "mulensinstrument.err_scale_raw": [2.786029660640395],
    "mulensinstrument.log_f_total_raw": [0.6041700945394],
    "mulensinstrument.q_source_raw": [-38.74145614466925],
    "planet.mass_raw": [20.294111688917543],
    "star.distance_raw": [530.3867500365418, 353.39006307342015],
    "star.logmass_raw": [267.0717585861006],
    "star.pm_dec_raw": [71.60334495785527, 4.5867354001178064],
    "star.pm_ra_raw": [5.123900807437569, 23.36953273281145],
    "star.rv_raw": [-0.8192451905245356, -0.7059124589883945],
}

# The whitening the draws above were sampled under, as run.py persists it.
# Restoring it is what makes a raw-space point mean the same thing here as it
# did in the fit; without it the same numbers decode to a different physical
# state entirely.
WHITENING_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "DC2018_128_whitening.json"
)


@pytest.fixture(scope="module")
def dc2018_128_system(tmp_path_factory):
    """Build the DC2018_128 model once; return (system, model).

    The whitening the pinned raw points were sampled under is restored here, so
    a raw vector means the same thing as it did in the run it came from.
    """
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

        # Both pinned draws were sampled in the LINEAR planet-mass coordinate:
        # the historical runaway predates log_q, and the fresh good draw was
        # taken from a run configured this way to match it.  A lens body now
        # defaults to log_q, and the raw values cannot be carried across -- the
        # runaway value pins the logit at an upper bound meaning 260000 Mjup in
        # one coordinate and q = 10 in the other, and the good draw sits
        # mid-range, where the two share nothing.  What this file regression-
        # tests (the unclipped raw**2 in the logit-uniform prior correction)
        # lives in Parameter.build_pymc and is the same either way.
        for entry in config.get("planet", []):
            entry.setdefault("mass_parameterization", "linear")

        system = System(config, user_params)
        system.prepare()
        model = system.build_model()
        # build_model leaves the PRELIMINARY scales in place; run.py measures
        # the real ones at startup.  The pinned raw points come from a run, so
        # restore that run's whitening or they decode to the wrong state.
        assert whitening.load_whitening(system, WHITENING_FIXTURE), (
            f"could not restore the whitening fixture {WHITENING_FIXTURE}; "
            f"the pinned raw points are meaningless without it"
        )
    finally:
        os.chdir(orig_cwd)

    return system, model


@pytest.fixture(scope="module")
def dc2018_128_logp(dc2018_128_system):
    """The compiled logp of the model above."""
    _system, model = dc2018_128_system
    return model.compile_logp()


def _point(raw_dict):
    return {k: np.asarray(v, dtype=float) for k, v in raw_dict.items()}


# The lp this draw had in the run it came from, reproduced exactly here once the
# run's whitening is restored (see WHITENING_FIXTURE).  It is a genuinely good
# state, not merely a pinned one: mulens chi2/N at this point is 1.002.
#
# This value tracks the MODEL, so a deliberate physics change moves it -- the
# earlier history of re-measurements (annual parallax on the Op path, the
# galacticmodel prior normalizations, the genulens fidelity upgrade, the
# mu_rel helio->geo frame fix, the Chabrier IMF normalization) is in the git log
# of this file.  If a change moves it, re-measure at this exact raw point.  If
# a change moves the START VALUES instead, that is different: the raw point then
# decodes somewhere else, and the honest fix is a fresh draw from a fresh run
# rather than a new constant on a stale point.
#
# 2979.9175 -> 3398.8805 when the microlensing likelihood moved from magnitudes
# to flux.  That is a CHANGE OF MEASURE, not a change of fit: the density is now
# over F rather than over m = -2.5*log10(F), and the two differ by the constant
# Jacobian sum_i log|dm/dF|_i = N*log(2.5/ln10) - sum_i log(F_i) = +419.378 nats
# over this file's 870 epochs.  The measured move is +418.963, so all but -0.415
# nats of it is that constant; the -0.415 is the genuine second-order difference
# between a Gaussian in flux and a Gaussian in magnitudes -- 0.05% of a chi2 of
# ~870, i.e. the O(sigma_m) agreement the conversion promises.  The physical
# state at GOOD_RAW is untouched (the flux bootstrap that sets the start values
# always worked in flux internally and is bit-identical), which is why the
# chi2/N check below still passes at the same point.
#
# 3398.8805 -> 3401.5937 when Band started pinning limb darkening no consumer
# reads.  This example is `finite_source: False`, so band.u1 was a free RV that
# no likelihood term touched; dropping it removes its prior contribution at
# GOOD_RAW (+2.713 nats).  Another CHANGE OF MEASURE -- the model has one fewer
# dimension -- and not a change of fit: nothing in the light-curve sector moved,
# which is why the chi2/N check below is unchanged at the same point.
GOOD_EXPECTED_LP = 3401.5937


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


def test_good_draw_is_actually_a_good_fit(dc2018_128_system):
    """
    Given the good draw,
    When the microlensing light curve is evaluated there,
    Then it fits to about one chi2 per point.

    Guards the word "good".  A pinned lp alone cannot do that: it stays
    self-consistent even when the start values move under it and the point
    quietly stops describing a good state, which is exactly what happened when
    #99 reseeded the proper motions (chi2/N went to ~261 while the lp check was
    simply re-pinned).  chi2 is measured against the data, so it cannot drift
    along with the model.
    """
    # Arrange
    system, model = dc2018_128_system

    # Act
    obs = [v for v in model.observed_RVs if "mulens" in v.name][0]
    ins = obs.owner.inputs
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([ins[-2], ins[-1]]),
        on_unused_input="ignore",
    )
    point = _point(GOOD_RAW)
    mu, sigma = [
        np.asarray(a, dtype=float).ravel()
        for a in fn(*[point[v.name] for v in model.value_vars])
    ]
    data = np.asarray(obs.tag.observations.eval(), dtype=float).ravel()
    reduced = float(np.sum(((data - mu) / sigma) ** 2)) / data.size

    # Assert
    assert reduced < 1.5, (
        f"chi2/N at the 'good' draw is {reduced:.3f}, not ~1.  The raw point no "
        f"longer describes a good state -- most likely the start values moved "
        f"under it, in which case take a fresh draw from a fresh run rather "
        f"than re-pinning GOOD_EXPECTED_LP."
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
