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
import warnings
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
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
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
# nothing about the fit was wrong.  It cost three separate triages.  The start,
# by contrast, is reproducible: the polish has no RNG, so seven consecutive
# runs on one box gave BIT-IDENTICAL start values.  See review 7.13.6 and
# run.md's section on Model.initial_point() being the start.
#
# GOLDEN VALUES, ON PURPOSE.  These numbers are the whole point: an
# intentional change to where the sampler begins (a new polish, a re-centered
# whitening anchor, a changed hint or default) SHOULD require editing them,
# so the move is visible in the diff and has to be justified in the commit
# that makes it.  Review 1.3.6 was a wrong default start that survived months
# precisely because nothing in the suite asserted where the sampler begins.
# Recorded 2026-09-14 against 1fed94c1, i.e. after batch 4F (PR #265)
# re-centered the whitening anchor on the polished start.
#
# WHY THE TOLERANCES ARE THIS LOOSE, AND WHY TIGHTENING THEM WILL GO RED.
# Bit-identical on ONE box is not portable.  MEASURED on all four shipped CI
# combinations plus the dev box (2026-09-14):
#
#   platform             star.A.logmass   planet.b.mass   orbit.b.logP
#   dev box (linux)        0.08057130      0.96736983      0.47562107
#   CI ubuntu 3.12         0.08047306      0.96681714      0.47562081
#   CI ubuntu 3.13         0.08047306      0.96681714      0.47562081
#   CI ubuntu 3.14         0.08047306      0.96681714      0.47562081
#   CI macOS 3.12          0.08073805      0.96338280      0.47562067
#
#   spread                 2.65e-4 dex     4.1e-3 rel      4.0e-7 dex
#
# The three ubuntu Pythons agreeing to the LAST DIGIT is the control: this is
# the platform, not the interpreter.  And it is NOT the ~1e-9 build
# difference of review 3.14.20 -- it is five orders of magnitude bigger --
# because the value being asserted is POST-POLISH and the polish is an
# ITERATIVE optimizer that terminates on |grad| < 0.01 nats/unit.  A BLAS or
# LAPACK difference moves the point at which that test first passes, so a
# different BLAS build lands somewhere else on the same basin floor.  Three
# clusters, one per platform family, is exactly that signature.
#
# So the tolerance is applied in each quantity's OWN domain -- absolute in
# dex for a dex/log quantity, relative for a linear one.  Applying a single
# rtol to everything is what broke the first version of this test:
# star.A.logmass is only 0.08, so 2.65e-4 of dex scatter reads as 3.3e-3
# RELATIVE and blew an rtol of 1e-3, while the same scatter in the physical
# mass is 6.1e-4 and would have passed.  A relative tolerance on a quantity
# whose zero is arbitrary measures the offset, not the error.
#
# Headroom over the measured spread is ~11x in dex and ~3.7x in the linear
# rtol -- the linear one is tighter because planet.b.mass is the widest
# scatter in the table and CI runner images change.  A real start regression
# is far larger: review 1.3.6 moved planet.mass by 8%.
KELT4_DEX_ATOL = 3.0e-3  # dex, for log/dex quantities (11x observed)
KELT4_LINEAR_RTOL = 1.5e-2  # relative, for linear quantities (3.7x observed)

# The keys are the STARTUP TABLE's per-element display labels, which are not
# the trace's variable names (planet.b.mass here, planet.mass there).
# kind: "dex" -> compare with KELT4_DEX_ATOL; "linear" -> KELT4_LINEAR_RTOL.
KELT4_START = {
    # label            value        units           kind
    "star.A.logmass": (0.08057130, "dex(solMass)", "dex"),
    "planet.b.mass": (0.96736983, "jupiterMass", "linear"),
    "orbit.b.logP": (0.47562107, "dex(d)", "dex"),
    # cosi is here because the mass story below turns on it: it is the one
    # parameter the RV data says nothing about.
    "orbit.b.cosi": (0.50545129, "", "linear"),
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
# WHY THE MASS STARTS ~7% ABOVE THE PUBLISHED 0.90 Mjup.  kelt4_rvonly.yaml
# is RV-only, so there is no inclination information.  The params file's
# `orbit.0.cosi: 0.11996` (the published, transit-derived i = 83.1 deg) is
# only a START, the polish is free to walk cosi, and it lands at 0.50545,
# i = 59.6 deg.  `mass = m sin i / sin i` inherits that entirely.
#
# WHAT WAS MEASURED, AND WHY THE LITERATURE BAND IS WIDE.  At this start
# sin i = 0.862855 and m sin i = 0.834700 Mjup.  Against a published 0.902
# that is -7.5% (-6.8% against the published m sin i of 0.8956); against the
# 0.878 that run.md quotes it is -4.9%.  So the agreement is real but not
# tight, and three things explain the gap without any of them being a defect:
# this is a START (one L-BFGS local optimum), it uses two of the four
# published RV datasets (EXPERT and FIES are commented out of the config) and
# no long-term trend for the BC companions, and the repo records no published
# mass of its own -- the review doc quotes 0.90 and run.md quotes 0.878.  The
# eccentricity difference is NOT one of them: e = 0.0789 here gives
# sqrt(1-e^2) = 0.9969, a 0.3% effect.  The band below is therefore
# calibrated from that -7.5%, not from a claim of agreement, and it is still
# tight enough to catch a factor-of-two or a unit slip.
#
# `orbit.sini` is a manifest parameter but appears in neither this table nor
# the trace, so sin i is derived from cosi.  Implementing m sin i as a
# reported parameter is review item 8.8.17 and is not this test's business.
KELT4_START_MSINI = 0.834700  # Mjup; golden regression
# m sin i's own cross-platform spread has NOT been measured -- `orbit.b.cosi`
# was added to the table in the same commit that added this, so the CI sweep
# that calibrated the three rows above predates it.  3e-2 until the
# calibration warning below reports it, then tighten toward KELT4_LINEAR_RTOL.
# It is still 2.5x tighter than the mass's own offset from published, so it
# is not vacuous meanwhile.
KELT4_MSINI_RTOL = 3.0e-2
KELT4_PUBLISHED_MASS = 0.90  # Mjup, Beatty+2016; see the band above
KELT4_PUBLISHED_RTOL = 0.15

# THE GOLDEN START LOGP, and why it is the more robust of the two assertions
# (JDE, 2026-09-14).  logp is STATIONARY at an optimum, so the 6e-4 of
# optimizer scatter above perturbs it only at SECOND order -- of order
# delta_theta**2 times the curvature, which for this start is ~1e-4 nats,
# below the 0.1-nat resolution the polish line prints.  Any real change, by
# contrast, moves it by O(1) nats or more: a changed prior, a unit conversion
# slip, a likelihood term added or lost.  That is exactly the discrimination
# wanted, and it is why a loose tolerance here still bites.
#
# Both ends of the polish are pinned, because they fail for different
# reasons.  The PRE-polish value is the BUILD start -- a plain evaluation with
# no optimizer in it at all, so it carries none of the scatter above and is
# the sharper detector of a prior/unit/likelihood change.  The POST-polish
# value is the point the sampler actually begins from.
#
# AND THE PREDICTION HELD, MEASURED.  Both numbers came back IDENTICAL on
# the dev box, CI ubuntu 3.12, 3.13 and 3.14, and CI macOS 3.12 -- -601.1 and
# 81.4 on all five -- while the parameter values under them scattered by up
# to 4.1e-3 relative on the same runs.  That is the stationarity argument
# confirmed rather than assumed, and it is why the tolerance here is 0.2 nats
# (twice the printed resolution) against the 1.5e-2 the linear values need.
# So this is the assertion that will catch a changed prior first, and by a
# wide margin.
KELT4_BUILD_LOGP = -601.1  # lp at the build start, before the polish
KELT4_START_LOGP = 81.4  # lp at the polished start the sampler uses
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
#   ... exozippy.polish: Seed polish (L-BFGS): seed 0 lp -601.1 -> 81.4 (...)
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

    # CALIBRATION PROBE, TEMPORARY -- DELETE IT AFTER THE NEXT CI ROUND.
    # pytest -q prints the warnings summary, which is how each CI platform
    # reports its own numbers for a golden value whose cross-platform spread
    # has not been measured yet.  It has already done its job once: the two
    # logp values and the three parameter rows above are calibrated from what
    # it reported on ubuntu 3.12/3.13/3.14 and macOS 3.12.  What is still
    # UNMEASURED is `orbit.b.cosi` and the m sin i product, which were added
    # after that sweep -- hence KELT4_MSINI_RTOL's provisional 3e-2.  Once
    # this round reports them, tighten that and remove this block
    # (review 7.13.6).
    warnings.warn(
        "kelt4 start calibration: "
        f"build_lp={build_lp!r} polished_lp={polished_lp!r} "
        + " ".join(f"{k}={start.get(k, (None,))[0]!r}" for k in KELT4_START),
        stacklevel=1,
    )

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
            ok = np.isclose(got, expected, rtol=KELT4_LINEAR_RTOL, atol=0.0)
            bound = f"rtol {KELT4_LINEAR_RTOL}"
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
    #     mass it IS comparable to the published value.  The band is wide
    #     for the reasons recorded above (a start, not a posterior; two of
    #     four RV datasets; no published value in the repo) and is there to
    #     catch a gross regression -- a factor of two, a unit slip -- not to
    #     claim tight agreement.
    assert msini == pytest.approx(
        KELT4_PUBLISHED_MASS, rel=KELT4_PUBLISHED_RTOL
    ), (
        f"m sin i at the start is {msini:.4f} Mjup, more than "
        f"{KELT4_PUBLISHED_RTOL:.0%} from the published "
        f"{KELT4_PUBLISHED_MASS} Mjup. m sin i is what the RVs constrain, so "
        f"unlike planet.b.mass this one IS comparable to the literature."
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
