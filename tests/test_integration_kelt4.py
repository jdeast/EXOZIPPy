"""End-to-end integration test: run_fit on the kelt4rvonly example.

Exercises the full pipeline:
  config parsing → system build → model construction → NUTS sampling
  -> unit conversion -> trace save -> mkparam output

All file I/O is redirected to a temp directory so nothing is written into the
source tree. run_fit is called once (module scope) and all tests share the result.

Marked 'slow'; excluded from fast CI with ``pytest -m "not slow"``.
"""

import os
import re
import shutil
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import yaml

from exozippy.run import run_fit

pytestmark = pytest.mark.slow

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "kelt4"


@pytest.fixture(scope="module")
def kelt4_result(tmp_path_factory):
    """
    Copy the kelt4 example to a temp directory, run run_fit once (on the
    RV-only config, kelt4_rvonly.yaml) with minimal sampler settings, and
    return (out_dir, work_dir) for all tests to share.

    work_dir — copy of the example directory (data files, params yaml)
    out_dir  -- where trace, plots, and mkparam output are written
    """
    work_dir = tmp_path_factory.mktemp("kelt4_work") / "kelt4"
    out_dir = tmp_path_factory.mktemp("kelt4_out")

    shutil.copytree(
        EXAMPLE_DIR,
        work_dir,
        # ".#*"/"#*#" are emacs lock/autosave droppings; the lock is a
        # dangling symlink that would abort the copy.
        ignore=shutil.ignore_patterns("fitresults*", ".#*", "#*#"),
    )

    orig_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with open("kelt4_rvonly.yaml") as f:
            config = yaml.safe_load(f)

        config["prefix"] = str(out_dir / "KELT-4A")
        config["sampler"] = {
            "method": "nuts",
            "tune": 2,
            "draws": 1,
            "chains": 1,
            "cores": 1,
            "measure_scales": False,
            "recompute_trace": True,
        }

        run_fit(config)
    finally:
        os.chdir(orig_cwd)

    return out_dir, work_dir


# ---------------------------------------------------------------------------
# Tests — all read from the shared kelt4_result fixture
# ---------------------------------------------------------------------------


def test_run_fit_kelt4_trace_file_written(kelt4_result):
    """
    Given the kelt4rvonly example with minimal sampler settings,
    When run_fit completes,
    Then a NetCDF trace file is written to the configured prefix path.
    """
    out_dir, _ = kelt4_result
    assert (out_dir / "KELT-4A_trace.nc").exists()


def test_run_fit_kelt4_trace_has_expected_variables(kelt4_result):
    """
    Given the kelt4rvonly example,
    When run_fit completes,
    Then the posterior contains the key orbital, planetary, and stellar parameters.
    """
    out_dir, _ = kelt4_result
    idata = az.from_netcdf(str(out_dir / "KELT-4A_trace.nc"))
    posterior_vars = set(idata.posterior.data_vars)

    expected = {
        "orbit.logP",
        "orbit.tc",
        "orbit.secosw",
        "orbit.sesinw",
        "planet.mass",
        "star.logmass",
        "rvinstrument.gamma",
    }
    missing = expected - posterior_vars
    assert not missing, f"Missing expected posterior variables: {missing}"


# ---------------------------------------------------------------------------
# The START the run actually began from (review 7.13.6)
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT A POSTERIOR TEST.  The fixture drives tune: 2, draws: 1,
# chains: 1, so NUTS's dual averaging has adapted nothing by the time the
# single draw is taken: the step size is still the step_scale / size**0.25
# heuristic and that draw is essentially ONE RANDOM JUMP of a fixed size, in
# raw space, away from the start.  Its physical size is set by the START'S
# CONDITIONING rather than by the model, so asserting hard physical bounds on
# it is a lottery: the same assertion on planet.mass has been observed at
# 0.81, 2.870, 5.7693 (local, on plain master), 6.724 and 8.718 (CI) while
# nothing about the fit was wrong.  It cost three separate triages.  The
# start, by contrast, is reproducible: the polish has no RNG, so repeated
# runs on one box under one arithmetic give BIT-IDENTICAL start values (what
# a CHANGE of arithmetic does to them is the next block down).  See review
# 7.13.6 and run.md's section on Model.initial_point() being the start.
#
# GOLDEN VALUES, ON PURPOSE.  These numbers are the whole point: an
# intentional change to where the sampler begins (a new polish, a re-centered
# whitening anchor, a changed hint or default) SHOULD require editing them,
# so the move is visible in the diff and has to be justified in the commit
# that makes it.  Review 1.3.6 was a wrong default start that survived months
# precisely because nothing in the suite asserted where the sampler begins.
# First recorded 2026-09-14 against 1fed94c1 (after batch 4F, PR #265,
# re-centered the whitening anchor on the polished start); re-recorded the
# same day when review 7.13.8 moved the polish's stop (below).
#
# WHY THE TOLERANCES ARE WHAT THEY ARE: THE VALUE IS POST-OPTIMIZER.
# Everything asserted here except the build logp is the output of an
# ITERATIVE OPTIMIZER (polish.py's L-BFGS-B, stopping on |grad|_inf <
# polish._LBFGS_GTOL nats/unit), and an optimizer's endpoint moves with ANY
# difference in the arithmetic that fed it, of any size.  Not load, not
# thread count, not the pytensor cache, not PYTHONHASHSEED -- fifteen
# same-machine runs varying all of those (load to 28.9) were bit-identical
# (review 7.13.8).  What moved it across CI runners was a different
# OpenBLAS kernel inside scipy's OWN L-BFGS-B bookkeeping (a different
# runner CPU picks a different DYNAMIC_ARCH kernel; the compiled objective
# has no BLAS op at all): that changes one iterate by ONE ULP at evaluation
# 4, and the stop then lands somewhere else on the same basin.  Cross-
# platform libm/compiler/SIMD differences are >= 1 ulp by construction, so
# this cannot be removed; it can only be kept from being AMPLIFIED.
#
# THE AMPLIFIER WAS THE STOP, AND IT IS FIXED (review 7.13.8, JDE
# 2026-09-14).  At the old gtol = 0.01 the polish stopped on the FIRST
# evaluation to dip under the threshold -- one of 177, the one before it
# read 0.017 -- 0.507 nats below the basin optimum (81.440 against 81.947),
# on a tc/logP ridge with Hessian condition number 5.4e6, at iteration 148
# of a 150 cap.  On that shoulder 1 ulp of arithmetic moved cosi by 8.5%.
# At the shipped gtol = 1e-4 (cap 400) the polish reaches the basin optimum
# and the same perturbation moves cosi by 8.6e-4; that is the ~20x
# tightening of every tolerance below against the first version of this
# file, and it is why the golden values themselves changed (cosi 0.50545 ->
# 0.50012, lp 81.4 -> 81.9).
#
# HOW THE TOLERANCES ARE CALIBRATED, AND THE RULE.  Never calibrate a
# post-optimizer golden value from repeated runs of one environment: seven
# bit-identical solo runs opened the first version of this test and proved
# nothing, and six environments (dev box solo and under -n6, four CI
# combinations) were a six-point SAMPLE from a distribution nobody had
# measured -- one whose full width turned out to be 5x the six-point
# spread.  The calibration is the ULP-PERTURBATION HARNESS instead: run
# polish._lbfgs_polish_one from this fixture's raw start with the compiled
# (lp, grad) multiplied by (1 + s * 2**-52), s in {-1, 0, +1} a hash of (x,
# component, seed) -- i.e. "the same function, computed by a different but
# equally correct arithmetic" -- for 16 seeds, and read the full width.
# tests/test_polish.py::test_ulp_perturbation_does_not_move_the_polished_start
# runs a 4-seed version of it on every CI run.  Measured 2026-09-14 under
# the shipped constants (seed 0 unperturbed = the golden values):
#
#   quantity         golden (seed 0)  16-seed full width   tolerance  headroom
#   star.A.logmass   0.08052772       2.4e-6 dex           2.0e-5 dex   8x
#   planet.b.mass    0.96091637       3.0e-4 rel           2.0e-3 rel   7x
#   orbit.b.logP     0.47554391       1.1e-7 dex           2.0e-5 dex   180x
#   orbit.b.cosi     0.50012485       8.6e-4 rel           5.0e-3 rel   6x
#   m sin i          0.83210871       2.6e-5 rel           2.0e-4 rel   8x
#   polished lp      81.947279        9.1e-7 nats          0.2 nats     (print)
#   iterations       268              240-294, 0 of 16 capped
#
# For comparison the SAME harness at the old gtol = 0.01 / cap 150 gives
# 2.7e-4 dex, 2.1e-2, 3.1e-6 dex, 8.5e-2, 9.5e-3 and 0.24 nats, with 3 of
# 16 seeds hitting the cap -- which is why the old 5e-2 cosi tolerance was
# not loose but MARGINAL, and the old 0.2-nat logp tolerance was already
# narrower than the distribution it sat in.  The cosi tolerance also has a
# first-principles ceiling: with cosi's Schur-complement curvature 0.0044
# nats/raw^2 the |grad| < 1e-4 stopping set spans +/-0.023 raw = +/-5e-4
# in cosi, about 2e-3 relative, so 5e-3 is above the WORST case the stop
# admits, not only 6x the sample.
#
# THE SCATTER IS NOT UNIFORM ACROSS PARAMETERS, AND THAT IS PHYSICS RATHER
# THAN NOISE.  Ranked by the harness width: the start logp (stationary at
# the optimum, second order), orbit.b.logP (pinned by the data),
# star.A.logmass (pinned by its Gaussian prior), m sin i (what the RV data
# actually constrains), planet.b.mass (m sin i / sin i, so it inherits cosi)
# and orbit.b.cosi (THE FLAT DIRECTION: RVs say nothing).  A single
# tolerance across that range is either vacuous at the top or red at the
# bottom, so `orbit.b.cosi` gets its own, and the hierarchy itself is the
# useful thing: if cosi ever stops being the loosest row, something has
# started constraining the inclination.
#
# And the tolerance is applied in each quantity's OWN domain -- absolute in
# dex for a dex/log quantity, relative for a linear one.  Applying a single
# rtol to everything is what broke the first version of this test:
# star.A.logmass is only 0.08, so dex scatter reads as ~40x larger RELATIVE
# than the same scatter in the physical mass.  A relative tolerance on a
# quantity whose zero is arbitrary measures the offset, not the error.
#
# A real start regression is far larger than any of these: review 1.3.6
# moved planet.mass by 8%, and the 7.13.8 stop change itself moved cosi by
# 1.1% -- both would have been red under the tolerances below, and the
# second WAS the visible, justified edit this file exists to force.
KELT4_DEX_ATOL = 2.0e-5  # dex, log/dex quantities (8x the harness width)
KELT4_LINEAR_RTOL = 2.0e-3  # relative, linear quantities (7x)
KELT4_FLAT_RTOL = 5.0e-3  # relative, the prior-dominated flat direction
#                           (cosi; 6x the harness width, above the
#                           stopping-set ceiling)

# The keys are the STARTUP TABLE's per-element display labels, which are not
# the trace's variable names (planet.b.mass here, planet.mass there).
# kind: "dex" -> KELT4_DEX_ATOL; "linear" -> KELT4_LINEAR_RTOL;
# "flat" -> KELT4_FLAT_RTOL (a direction the data does not constrain).
KELT4_START = {
    # label            value        units           kind
    "star.A.logmass": (0.08052772, "dex(solMass)", "dex"),
    "planet.b.mass": (0.96091637, "jupiterMass", "linear"),
    "orbit.b.logP": (0.47554391, "dex(d)", "dex"),
    # cosi is here because the mass story below turns on it: it is the one
    # parameter the RV data says nothing about, which is also why it is the
    # only "flat" row and needs its own, looser tolerance.
    "orbit.b.cosi": (0.50012485, "", "flat"),
}

# THE PLANET MASS AND m sin i ARE TWO DIFFERENT CLAIMS, AND EACH EARNS A
# DIFFERENT KIND OF ASSERTION (JDE's ruling, recorded in review 8.8.17).
#
#   m sin i is what the RV data actually CONSTRAINS, so it is the quantity
#   comparable to a published value, and it carries the LITERATURE check.
#
#   planet.b.mass is prior-dominated here and so is NOT comparable to a
#   transit-constrained published mass -- but it is perfectly DETERMINISTIC
#   given the same code and the same priors, so it carries a GOLDEN-VALUE
#   REGRESSION check against our own recorded number.  That is a regression
#   claim, not a physics one.
#
# Two natural mistakes this comment exists to prevent: do not "fix" the mass
# assertion by comparing it to the literature, and do not delete it as
# meaningless because it is prior-dependent.
#
# AND DO NOT READ THIS AS "m sin i IS THE TRUSTWORTHY ONE".  It is the other
# way round.  A mass marginalized over the inclination prior IS a posterior
# for the mass, and its width says how much of it is prior; m sin i is a
# LOWER BOUND that the field routinely quotes as though it were a
# measurement.  KELT-4Ab happens to agree well because it TRANSITS -- i is
# about 83 deg, so the planet sits near its minimum mass -- which is a
# property of this system, not evidence that m sin i is the better statistic.
# We report both because the field's standard is m sin i.
#
# WHY THE MASS STARTS ~9% ABOVE THE PUBLISHED 0.878 Mjup.  kelt4_rvonly.yaml
# is RV-only, so there is no inclination information.  The params file's
# `orbit.0.cosi: 0.11996` (the published, transit-derived i = 83.1 deg) is
# only a START, the polish is free to walk cosi, and it lands at 0.50012,
# i = 60.0 deg -- the basin optimum of the prior-dominated flat direction.
# `mass = m sin i / sin i` inherits that entirely.
#
# WHAT WAS MEASURED.  At this start sin i = 0.865953 and m sin i = 0.832109
# Mjup.
#
# WHICH PUBLISHED NUMBER TO COMPARE AGAINST -- and there are FOUR, which is
# what an earlier version of this comment got wrong.  Eastman+2016 (AJ 151,
# 45) reports an ECCENTRIC and a CIRCULAR solution for KELT-4Ab, each with a
# mass and an m sin i:
#
#     mass    eccentric  0.878 +0.070/-0.067    circular  0.902 +0.060/-0.059
#     m sin i eccentric  0.871 +0.069/-0.066    circular  0.896 +0.060/-0.058
#
# kelt4_rvonly.yaml FITS ECCENTRICITY (e = 0.110 at this start), so the
# ECCENTRIC column is the comparable one.  Against it:
#
#     m sin i  0.832109 vs 0.871   ->   -4.5%,  -0.59 sigma
#     mass     0.960916 vs 0.878   ->   +9.4%,  +1.18 sigma
#
# So m sin i sits INSIDE the published uncertainty and the mass is 1.18
# sigma high -- the prior-domination signature this file documents, and a
# factor 2 better in sigma for the quantity the RVs actually constrain.
# The earlier reading here ("-7.5%, about as far the other way") differenced
# against 0.902, the CIRCULAR MASS: wrong solution AND wrong quantity.  0.90
# and 0.878 were never two sources disagreeing -- they are two solutions in
# one paper, and a KELT-4Ab mass quoted without saying WHICH is unusable.
#
# The residual -4.5% needs no defect to explain it: this is a START (one
# L-BFGS local optimum, not a posterior), and the config uses two of the
# four published RV datasets (EXPERT and FIES are commented out) with no
# long-term trend for the BC companions.  Eccentricity is not a factor
# either way: sqrt(1-e^2) = 0.9939, a 0.6% term.
#
# `orbit.sini` is a manifest parameter but appears in neither this table nor
# the trace, so sin i is derived from cosi.  Implementing m sin i as a
# reported parameter is review item 8.8.17 and is not this test's business.
KELT4_START_MSINI = 0.832109  # Mjup; golden regression
# AND IT IS THE TIGHTEST PARAMETER ASSERTION IN THE FILE, which is the point
# and is measured rather than hoped for.  Over the 16 harness arithmetics
# in the table above, `orbit.b.cosi` moved by 8.6e-4 relative and
# `planet.b.mass` by 3.0e-4, while the PRODUCT moved by 2.6e-5 -- because
# mass and sin i are anti-correlated and it is m sin i that the RV data
# pins.  2e-4 is 8x that, tighter than every other parameter tolerance here,
# and it is the assertion that would notice the mass and the inclination
# drifting apart.
KELT4_MSINI_RTOL = 2.0e-4
# Eastman+2016 (AJ 151, 45), the ECCENTRIC solution's m sin i -- the
# quantity this config fits and the quantity the RVs constrain.  Asserted
# against the PUBLISHED UNCERTAINTY rather than an invented percentage, so
# the band means something: our start is 0.59 sigma low, and 1 sigma leaves
# ~1.7x headroom while still catching a factor-of-two or a unit slip.
KELT4_PUBLISHED_MSINI = 0.871  # Mjup
KELT4_PUBLISHED_MSINI_SIGMA = 0.066  # the lower error; we sit below

# THE GOLDEN START LOGP, and why it is the more robust of the two assertions
# (JDE, 2026-09-14).  logp is STATIONARY at an optimum, so optimizer scatter
# perturbs it only at SECOND order -- of order delta_theta**2 times the
# curvature -- PROVIDED THE POLISH ACTUALLY REACHES THE OPTIMUM.  Under the
# shipped gtol = 1e-4 it does: the 16 harness arithmetics span 9e-7 nats,
# six orders below the 0.1-nat resolution the polish line prints.  Any real
# change, by contrast, moves it by O(1) nats or more: a changed prior, a
# unit conversion slip, a likelihood term added or lost.  That is exactly
# the discrimination wanted, and it is why a loose tolerance here still
# bites.
#
# THE CAVEAT IS THE PROVISO, AND IT BIT ONCE.  Under the old gtol = 0.01 the
# stop was a first dip on a ridge 0.5 nats below the optimum, where logp is
# NOT stationary: the same harness spanned 0.24 nats there, wider than this
# 0.2-nat tolerance, and the six environments only agreed on "81.4" because
# they happened to sample the same half of that band.  A golden logp is the
# sharp detector this paragraph claims only when the optimizer is stopping
# at an optimum, which the perturbation test in tests/test_polish.py is
# what guarantees.
#
# Both ends of the polish are pinned, because they fail for different
# reasons.  The PRE-polish value is the BUILD start -- a plain evaluation with
# no optimizer in it at all, so it carries none of the scatter above and is
# the sharper detector of a prior/unit/likelihood change.  The POST-polish
# value is the point the sampler actually begins from.  The 7.13.8 stop
# change left the build value at -601.1 and moved the polished one from
# 81.4 to 81.9 -- the 0.507 nats the first-dip stop had been leaving on the
# table -- which is exactly the pair of signatures expected of "same model,
# better optimizer".
KELT4_BUILD_LOGP = -601.1  # lp at the build start, before the polish
KELT4_START_LOGP = 81.9  # lp at the polished start the sampler uses
KELT4_LOGP_ATOL = 0.2  # nats; 2x the 0.1-nat print resolution

# One row of run.inspect_start's startup table, as the file log handler (always
# DEBUG, so the table is there whatever logger_level the config asks for)
# writes it:
#   ... exozippy.run:   planet.b.mass |      0.96736983 |  0.100 |  jupiterMass | ...
_START_ROW = re.compile(
    r"exozippy\.run:\s+(?P<label>[A-Za-z_][\w.]*)\s+\|"
    r"\s+(?P<value>\S+)\s+\|"
    r"\s+(?P<scale>\S+)\s+\|"
    r"\s+(?P<units>[^|]*?)\s*\|"
)

# polish.py's own summary line, the only place the run reports a TOTAL logp:
#   ... exozippy.polish: Seed polish (L-BFGS): seed 0 lp -601.1 -> 81.9 (...)
_POLISH_LP = re.compile(
    r"Seed polish \(L-BFGS\): seed 0 lp\s+"
    r"(?P<before>-?[\d.]+)\s+->\s+(?P<after>-?[\d.]+)"
)


def read_start_table(log_path):
    """Parse `<prefix>.log` for run.inspect_start's startup table.

    Returns {display_label: (value, units)} in USER units, for every row whose
    Value column is a number (N/A rows and the header are skipped).

    This reads the start THE RUN ITSELF REPORTED, which is what keeps this an
    integration test: rebuilding the System from the same config would miss the
    pre-whitening polish and the anchor re-centering, both of which happen
    inside run_fit and both of which MOVE the start.  run_fit returns nothing
    and writes no machine-readable start file, so the log is the only existing
    channel; adding one purely for a test would be a design change.
    """
    values = {}
    for line in Path(log_path).read_text().splitlines():
        m = _START_ROW.search(line)
        if m is None:
            continue
        try:
            values[m.group("label")] = (
                float(m.group("value")),
                m.group("units").strip(),
            )
        except ValueError:
            continue  # the header row, or an N/A value
    return values


def read_polish_logp(log_path):
    """Total logp before and after the seed polish, from `<prefix>.log`.

    Returns (build_lp, polished_lp), or (None, None) if no polish line is
    present.  `inspect_start` prints a per-parameter Log-Prob column but no
    total, and the rows it suppresses (the logit-uniform log-volume terms)
    mean the printed column cannot be summed into one; the polish summary is
    the run's only whole-model logp.
    """
    for line in Path(log_path).read_text().splitlines():
        m = _POLISH_LP.search(line)
        if m is not None:
            return float(m.group("before")), float(m.group("after"))
    return None, None


def test_run_fit_kelt4_start_is_physical(kelt4_result):
    """
    Given the kelt4rvonly example and the polish + whitening-anchor pipeline,
    When run_fit reports the point the sampler starts from,
    Then that start -- its logp, and three of its parameter values -- is the
    recorded one, within a per-quantity tolerance, in user units.

    This deliberately does NOT look at the trace.  With tune: 2 / draws: 1 the
    single draw is one un-adapted jump whose physical size is set by the
    start's conditioning, not by the model, so no physical bound on it can be
    both tight and reliable (review 7.13.6).  The start is reproducible and is
    the thing worth pinning.

    The logp assertion is the discriminating one and the parameter values are
    the readable one; the reasoning behind both tolerances, and why neither
    can be tightened to the ~1e-9 of review 3.14.20, is in the comments above
    KELT4_DEX_ATOL and KELT4_START_LOGP.  The physical-range checks are kept
    so a wildly wrong start reports as "outside plausible range" rather than
    as a tolerance mismatch.
    """
    out_dir, _ = kelt4_result
    log_path = out_dir / "KELT-4A.log"
    start = read_start_table(log_path)
    build_lp, polished_lp = read_polish_logp(log_path)

    # HOW TO RE-CALIBRATE, when a future change moves the start: run the
    # ulp-perturbation harness described above KELT4_DEX_ATOL (16 seeds of
    # polish._lbfgs_polish_one from this fixture's raw start, objective x
    # (1 + s*2**-52)) and set each tolerance to ~5x the width it reports;
    # the golden values are its unperturbed seed.  Do NOT calibrate from a
    # `warnings.warn` probe reporting each CI platform's value -- that was
    # the first method, it sampled six points of a distribution 5x wider
    # than their spread, and it went red on the seventh (review 7.13.8).

    missing = set(KELT4_START) - set(start)
    assert not missing, (
        f"startup table has no row for {sorted(missing)}; "
        f"rows present: {sorted(start)}"
    )

    # Physical plausibility first: a readable failure for a badly wrong start.
    # logP = log10(3 d) = 0.476 and star logmass = 0.08 (1.2 Msun) for
    # KELT-4A; the planet is ~0.9 Mjup.
    logP, _u = start["orbit.b.logP"]
    assert 0.2 < logP < 0.7, (
        f"start logP={logP:.4f} outside plausible range [0.2, 0.7]"
    )
    planet_mass, _u = start["planet.b.mass"]
    assert 0.3 < planet_mass < 2.5, (
        f"start planet mass={planet_mass:.3f} Mjup outside [0.3, 2.5]"
    )
    star_logmass, _u = start["star.A.logmass"]
    assert -0.3 < star_logmass < 0.5, (
        f"start star logmass={star_logmass:.4f} outside [-0.3, 0.5]"
    )

    # The golden logp, both ends of the polish.
    assert polished_lp is not None, (
        "no 'Seed polish (L-BFGS): seed 0 lp ... -> ...' line in "
        f"{log_path.name}; the polish is what defines this start"
    )
    for what, got, expected in (
        ("build", build_lp, KELT4_BUILD_LOGP),
        ("polished", polished_lp, KELT4_START_LOGP),
    ):
        assert abs(got - expected) <= KELT4_LOGP_ATOL, (
            f"{what} start logp is {got!r} nats, recorded {expected!r} "
            f"(tolerance {KELT4_LOGP_ATOL} nats). logp is stationary at the "
            f"start, so optimizer scatter cannot move it this far -- a "
            f"prior, a unit conversion or a likelihood term changed. If this "
            f"move is intended, update KELT4_BUILD_LOGP/KELT4_START_LOGP and "
            f"say in the commit why the model now scores differently there."
        )

    # Then the golden parameter values, each in its own domain.
    for label, (expected, units, kind) in KELT4_START.items():
        got, got_units = start[label]
        assert got_units == units, (
            f"{label} start is reported in {got_units!r}, expected {units!r}"
        )
        if kind == "dex":
            ok = abs(got - expected) <= KELT4_DEX_ATOL
            bound = f"atol {KELT4_DEX_ATOL} dex"
        else:
            rtol = KELT4_FLAT_RTOL if kind == "flat" else KELT4_LINEAR_RTOL
            ok = np.isclose(got, expected, rtol=rtol, atol=0.0)
            bound = f"rtol {rtol}"
        assert ok, (
            f"{label} starts at {got!r} {units}, recorded {expected!r} "
            f"({bound}). If this move is intended, update KELT4_START and "
            f"say in the commit why the sampler now begins somewhere else."
        )

    # m sin i, the quantity the RV data constrains -- two assertions of two
    # different kinds, per the ruling above.  sin i comes from cosi because
    # orbit.sini is in neither the startup table nor the trace.
    cosi, _u = start["orbit.b.cosi"]
    sini = np.sqrt(max(1.0 - cosi**2, 0.0))
    msini = planet_mass * sini

    # (a) REGRESSION: our own recorded product, at the linear tolerance.
    assert np.isclose(
        msini, KELT4_START_MSINI, rtol=KELT4_MSINI_RTOL, atol=0.0
    ), (
        f"m sin i starts at {msini!r} Mjup (mass={planet_mass!r} x "
        f"sin i={sini!r} from cosi={cosi!r}), recorded "
        f"{KELT4_START_MSINI!r} (rtol {KELT4_MSINI_RTOL}). If this move is "
        f"intended, update KELT4_START_MSINI and say why in the commit."
    )

    # (b) PHYSICS: m sin i is the RV-constrained quantity, so unlike the
    #     mass it IS comparable to the published value -- and it must be
    #     compared against the ECCENTRIC solution, because that is what this
    #     config fits.  Asserted against the published UNCERTAINTY rather
    #     than an invented percentage: the start sits 0.59 sigma low, so one
    #     sigma is a real constraint with about 1.7x headroom, and it still
    #     catches a factor of two or a unit slip.
    offset = abs(msini - KELT4_PUBLISHED_MSINI)
    assert offset < KELT4_PUBLISHED_MSINI_SIGMA, (
        f"m sin i at the start is {msini:.4f} Mjup, "
        f"{offset / KELT4_PUBLISHED_MSINI_SIGMA:.2f} sigma from Eastman+2016's "
        f"eccentric-solution {KELT4_PUBLISHED_MSINI} +/- "
        f"{KELT4_PUBLISHED_MSINI_SIGMA} Mjup. m sin i is what the RVs "
        f"constrain, so unlike planet.b.mass this one IS comparable to the "
        f"literature -- but only to the ECCENTRIC column, since this config "
        f"fits eccentricity (the circular solution's 0.896 is a different "
        f"number for a different model)."
    )


def test_run_fit_kelt4_derived_parameters_are_self_consistent(kelt4_result):
    """
    Given the one draw this fixture's sampler budget produces,
    When a DERIVED parameter and its parents are read back out of the trace,
    Then the derived value equals the function of those parents evaluated at
    THAT SAME DRAW, in user units, to float tolerance.

    ONE DRAW IS SUFFICIENT AND MORE WOULD BUY NOTHING -- do not "improve"
    this by raising the budget.  This is an IDENTITY, not a distribution: a
    derived quantity is a deterministic function of its parents, so it either
    holds at every point or it is broken, and a converged posterior would
    demonstrate exactly the same thing at 100x the cost.  That is precisely
    what makes it the right use of a tune: 2 / draws: 1 trace, and the
    contrast with the assertion this file used to carry, which asked the same
    single draw for a statistical claim it could not support (review 7.13.6).

    WHAT IT GUARDS.  The old `..._posterior_in_sane_range` incidentally
    exercised the path where a derived quantity is computed during sampling,
    written to the trace, and converted to user units on the way out; the
    sibling tests already cover that variable's PRESENCE and its units, and
    nothing covered the identity.  The failure classes are: a derived value
    written in internal units but labelled user units; the two RECIPROCAL
    conversion factors being confused (CLAUDE.md's invariant -- the factor**2
    bug that shipped in the seed ledger); and a parent/child wiring slip that
    leaves both values individually plausible.  Neither recomputation below
    hand-writes a conversion factor, which is the point: `10**` between
    dex(d) and d, and a dimensionless ratio, are the two shapes where a
    mislabelled unit cannot hide.

    TWO PARAMETERS THAT LOOK LIKE OBVIOUS SUBJECTS AND ARE NOT, recorded so
    the next reader does not add them and find them missing:

    - `planet.mass` is SAMPLED in this configuration, not derived -- the
      trace carries a `planet.mass_raw` for it, and the relation in
      planet/symbolic_physics.py runs the other way (K is derived FROM the
      mass, eccentricity, sin i, period and stellar mass).  There is no
      identity to check for it here.
    - `star.mass` IS derived (`10**star.logmass`), and is printed in the
      startup table, but it does not appear in `idata.posterior` at all: a
      pure-expression parameter never does.  So the trace cannot be asked
      about it, and this is the documented gotcha rather than an omission.
    """
    out_dir, _ = kelt4_result
    idata = az.from_netcdf(str(out_dir / "KELT-4A_trace.nc"))
    post = idata.posterior

    def draw(name):
        assert name in post.data_vars, (
            f"{name} is not in idata.posterior; a pure-expression parameter "
            f"never is, so this identity cannot be checked from the trace. "
            f"Present: {sorted(post.data_vars)}"
        )
        return float(np.asarray(post[name].values).ravel()[0])

    # 1. orbit.period [d] == 10 ** orbit.logP [dex(d)].
    #    The dex/linear pair, and the one place a user-vs-internal unit
    #    mislabelling on either side shows up as a gross mismatch rather than
    #    as a plausible number.
    logP, period = draw("orbit.logP"), draw("orbit.period")
    assert period == pytest.approx(10.0**logP, rel=1e-12), (
        f"orbit.period={period!r} d but orbit.logP={logP!r} dex(d) implies "
        f"{10.0**logP!r} d, at the same draw. Either one of the two is not "
        f"in the units its label claims, or they are not the same quantity."
    )

    # 2. orbit.vcve == sqrt(1 - e^2) / (1 + e sin omega), with e and omega
    #    from the sqrt(e)cos/sin(omega) pair the sampler actually moves
    #    (orbit/physics.py: calc_vcve).  A multi-parent physics expression,
    #    all of it dimensionless, so what it tests is the wiring and the
    #    arithmetic rather than a unit.
    secosw, sesinw = draw("orbit.secosw"), draw("orbit.sesinw")
    ecc = secosw**2 + sesinw**2
    omega = np.arctan2(sesinw, secosw)
    expected_vcve = np.sqrt(max(1.0 - ecc**2, 0.0)) / max(
        1.0 + ecc * np.sin(omega), 1e-12
    )
    assert draw("orbit.vcve") == pytest.approx(expected_vcve, rel=1e-12), (
        f"orbit.vcve={draw('orbit.vcve')!r} but (secosw={secosw!r}, "
        f"sesinw={sesinw!r}) -> e={ecc!r}, omega={omega!r} gives "
        f"{expected_vcve!r} at the same draw."
    )


def test_run_fit_kelt4_posterior_in_user_units(kelt4_result):
    """
    Given the kelt4rvonly example,
    When run_fit completes,
    Then the posterior values are in user-facing units (not internal math units).

    logP should be in log10(days). If unit conversion regressed the value would
    be far outside the expected range for a ~3-day planet.
    """
    out_dir, _ = kelt4_result
    idata = az.from_netcdf(str(out_dir / "KELT-4A_trace.nc"))

    # log10(3 days) ≈ 0.476. Tight bounds; internal units would land far outside.
    logP = float(idata.posterior["orbit.logP"].values.mean())
    assert 0.3 < logP < 0.65, (
        f"logP={logP:.4f} outside tight window [0.3, 0.65] for a ~3 d orbit in "
        f"user units (log10 days). Possible unit-conversion regression."
    )


def test_run_fit_kelt4_mkparam_written(kelt4_result):
    """
    Given the kelt4rvonly example with parameter_file: kelt4.params.yaml,
    When run_fit completes,
    Then mkparam writes the next versioned params file in the work directory.
    """
    _, work_dir = kelt4_result
    # kelt4.params.yaml → next version is kelt4.params.2.yaml
    expected = work_dir / "kelt4.params.2.yaml"
    assert expected.exists(), (
        f"mkparam did not write {expected.name}; "
        f"yaml files present: {[f.name for f in work_dir.glob('*.yaml')]}"
    )
