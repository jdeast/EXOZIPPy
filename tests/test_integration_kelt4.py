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
# nothing about the fit was wrong.  It cost three separate triages.  The start
# itself, by contrast, is deterministic -- the pre-whitening polish has no RNG
# -- so that is what this test measures.  See review 7.13.6 and run.md's
# section on Model.initial_point() being the start.
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
# The keys are the STARTUP TABLE's per-element display labels, which are not
# the trace's variable names (planet.b.mass here, planet.mass there).
KELT4_START = {
    # label            value        units
    "star.A.logmass": (0.08057130, "dex(solMass)"),
    "planet.b.mass": (0.96736983, "jupiterMass"),
    "orbit.b.logP": (0.47562107, "dex(d)"),
}

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


def test_run_fit_kelt4_start_is_physical(kelt4_result):
    """
    Given the kelt4rvonly example and the polish + whitening-anchor pipeline,
    When run_fit reports the point the sampler starts from,
    Then that start is the recorded one, to 1e-3 relative, in user units.

    This deliberately does NOT look at the trace.  With tune: 2 / draws: 1 the
    single draw is one un-adapted jump whose physical size is set by the
    start's conditioning, not by the model, so no physical bound on it can be
    both tight and reliable (review 7.13.6).  The start is deterministic and
    is the thing worth pinning.

    rtol=1e-3 rather than exact equality: this start is the output of an
    L-BFGS polish, so its last digits are a BLAS/scipy detail and not
    portable, and model construction generally is platform-dependent at the
    ~1e-9 level (review 3.14.20 -- whose own NNLS mechanism does not apply to
    this RV-only config, but the class of difference does).  1e-3 is far
    tighter than any real start regression and far looser than that noise.
    The physical-range checks are kept alongside so a wildly wrong start
    reports as "outside plausible range" rather than as a tolerance mismatch.
    """
    out_dir, _ = kelt4_result
    start = read_start_table(out_dir / "KELT-4A.log")

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

    # Then the golden values themselves.
    for label, (expected, units) in KELT4_START.items():
        got, got_units = start[label]
        assert got_units == units, (
            f"{label} start is reported in {got_units!r}, expected {units!r}"
        )
        assert np.isclose(got, expected, rtol=1e-3, atol=0.0), (
            f"{label} starts at {got!r} {units}, recorded {expected!r}. "
            f"If this move is intended, update KELT4_START and say in the "
            f"commit why the sampler now begins somewhere else."
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
