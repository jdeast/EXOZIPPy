"""End-to-end integration test: run_fit on the kelt4rvonly example.

Exercises the full pipeline:
  config parsing → system build → model construction → NUTS sampling
  → unit conversion → trace save → mkprior output

All file I/O is redirected to a temp directory so nothing is written into the
source tree. run_fit is called once (module scope) and all tests share the result.

Marked 'slow'; excluded from fast CI with ``pytest -m "not slow"``.
"""
import os
import shutil
import yaml
import numpy as np
import pytest
import arviz as az
from pathlib import Path

from exozippy.run import run_fit

pytestmark = pytest.mark.slow

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "kelt4rvonly"


@pytest.fixture(scope="module")
def kelt4_result(tmp_path_factory):
    """
    Copy the kelt4rvonly example to a temp directory, run run_fit once with minimal
    sampler settings, and return (out_dir, work_dir) for all tests to share.

    work_dir — copy of the example directory (data files, params yaml)
    out_dir  — where trace, plots, and mkprior output are written
    """
    work_dir = tmp_path_factory.mktemp("kelt4_work") / "kelt4rvonly"
    out_dir = tmp_path_factory.mktemp("kelt4_out")

    shutil.copytree(
        EXAMPLE_DIR, work_dir,
        ignore=shutil.ignore_patterns("fitresults"),
    )

    orig_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with open("kelt4.yaml") as f:
            config = yaml.safe_load(f)

        config["prefix"] = str(out_dir / "KELT-4A")
        config["sampler"] = {
            "method": "nuts",
            "tune": 2,
            "draws": 1,
            "chains": 1,
            "cores": 1,
            "check_curvatures": False,
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


def test_run_fit_kelt4_posterior_in_sane_range(kelt4_result):
    """
    Given the kelt4rvonly example seeded at the MAP from kelt4.params.2.yaml,
    When run_fit completes with 1 draw starting near MAP,
    Then key parameters are within physically plausible ranges (user units).
    """
    out_dir, _ = kelt4_result
    idata = az.from_netcdf(str(out_dir / "KELT-4A_trace.nc"))
    post = idata.posterior

    # logP ≈ log10(3 d) ≈ 0.476 for KELT-4Ab
    logP = float(post["orbit.logP"].values.mean())
    assert 0.2 < logP < 0.7, f"logP={logP:.4f} outside plausible range [0.2, 0.7]"

    # Planet mass ≈ 0.9 Mjup; allow a broad range given only 1 draw
    planet_mass = float(post["planet.mass"].values.mean())
    assert 0.3 < planet_mass < 2.5, f"planet mass={planet_mass:.3f} Mjup outside [0.3, 2.5]"

    # Star logmass ≈ 0.08 (≈1.2 Msun)
    star_logmass = float(post["star.logmass"].values.mean())
    assert -0.3 < star_logmass < 0.5, f"star logmass={star_logmass:.4f} outside [-0.3, 0.5]"


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


def test_run_fit_kelt4_mkprior_written(kelt4_result):
    """
    Given the kelt4rvonly example with parameter_file: kelt4.params.2.yaml,
    When run_fit completes,
    Then mkprior writes the next versioned params file in the work directory.
    """
    _, work_dir = kelt4_result
    # kelt4.params.2.yaml → next version is kelt4.params.3.yaml
    expected = work_dir / "kelt4.params.3.yaml"
    assert expected.exists(), (
        f"mkprior did not write {expected.name}; "
        f"yaml files present: {[f.name for f in work_dir.glob('*.yaml')]}"
    )
