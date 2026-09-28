"""End-to-end integration test: run_fit on the ob09020 example.

The microlensing twin of tests/test_integration_kelt4.py.  It exists because
nothing else in the suite ran a shipped microlensing config THROUGH run_fit:
tests/test_examples_prepare.py stops at prepare() by design, the acceptance
replays build the model and score the recorded logp terms, and the keplerian
lens-motion tests build the model and check the geometry -- so the start-point
plot, the whitening probe, PTDE and the wrap-up were never exercised on a
binary-lens config.  examples/ob09020 shipped for nine days with a TypeError
in its first start-point plot before a collaborator ran it (2026-09-25).

ob09020 is the right example for this because it is the one that crosses the
most component seams at once: a binary lens (VBMicrolensing, no gradient ->
the PTDE path), finite source, keplerian lens orbital motion driven by an
RV-constrained orbit (so the lens geometry is DERIVED, the parameterization
that broke), the galactic model and two RV instruments.

Budget: seed_polish off, a two-rung PTDE ladder, three draws.  Per
docs/testing.md ("A sampler budget too small to adapt cannot test a
posterior") nothing here asserts a posterior quantity -- files, variable
names, and the START are what this budget determines.  With the polish off
the start is the relaxation engine's, i.e. the user's own seeds wherever the
engine reproduces them exactly, so the start assertions are the user-start
contract (PR #252), not golden values.

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

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "ob09020"
_NAME = "OGLE-2009-BLG-020"


# Every Nth row of the large photometry files is kept.  One logp of the full
# model costs ~3 s (VBMicrolensing finite-source binary lens over 2837 points,
# most of them near the caustic), and both scale probes -- the whitening pass
# and PTDE's own fallback when that is off -- are serial loops of tens of
# evaluations per raw element: an hour per pass at full size.  The CONFIG is
# untouched; only the copied data are thinned, and row 0 is always kept so
# CAO's `mask: [0]` (a row INDEX, on-disk order) still names the failed
# measurement it was written for.  Files under _THIN_MIN_ROWS rows (CT13_I,
# CT13_V, FarmCove, Possum) are left whole.
_THIN_EVERY = 10
_THIN_MIN_ROWS = 100


def _thin_photometry(work_dir):
    for name in (
        "phot.dat",
        "Bron_OB09020U.pho",
        "CAO_OB09020U.pho",
        "Kumeu_OB09020U.pho",
        "VLO_OB09020U.pho",
    ):
        path = work_dir / name
        lines = path.read_text().splitlines(keepends=True)
        # The muFUN files open with a '#' header block; the reader skips
        # it, and the mask counts DATA rows, so thin only the data rows.
        header = [ln for ln in lines if ln.lstrip().startswith("#")]
        data = [ln for ln in lines if not ln.lstrip().startswith("#")]
        if len(data) < _THIN_MIN_ROWS:
            continue
        path.write_text("".join(header + data[::_THIN_EVERY]))


@pytest.fixture(scope="module")
def ob09020_result(tmp_path_factory):
    """Copy the example to a temp directory, run run_fit once with a minimal
    PTDE budget, and return (out_dir, work_dir) for every test to share."""
    work_dir = tmp_path_factory.mktemp("ob09020_work") / "ob09020"
    out_dir = tmp_path_factory.mktemp("ob09020_out")

    shutil.copytree(
        EXAMPLE_DIR,
        work_dir,
        ignore=shutil.ignore_patterns("fitresults*", ".#*", "#*#", "*~"),
    )
    _thin_photometry(work_dir)

    orig_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        with open("ob09020.yaml") as f:
            config = yaml.safe_load(f)

        config["prefix"] = str(out_dir / _NAME)
        config["sampler"] = {
            "method": "ptde_async",
            "n_temps": 2,
            "tune": 0,
            "draws": 3,
            # A pipeline budget, not a sampling one: the DE floor is 4 and
            # the default 2 x n_params = 116 per rung turned the start
            # population plus three draws into ~900 evaluations.  PTDE's
            # key is `n_chains`; `chains` is the HMC/DEMC spelling and is
            # silently method-only here.
            "n_chains": 12,
            "cores": 2,
            # The polish is what moves the start off the user's seeds; off,
            # the start table below is the user-start contract.
            "seed_polish": False,
            # Like the kelt4 fixture.  The whitening pass probes every raw
            # element twice (measure, then re-measure against the final
            # barriers); with it off PTDE runs its own single, cheaper probe
            # for the start dispersion.  Both have their own tests; here the
            # probe is the budget, and this is the smaller half of it.
            "measure_scales": False,
            "recompute_trace": True,
        }

        run_fit(config)
    finally:
        os.chdir(orig_cwd)

    return out_dir, work_dir


# One row of run.inspect_start's startup table, as the file log handler
# writes it (same parser as test_integration_kelt4.read_start_table):
#   ... exozippy.run:   star.L1.logmass |     -0.05158703 |  0.015 | dex(solMass) | ...
_START_ROW = re.compile(
    r"exozippy\.run:\s+(?P<label>[A-Za-z_][\w.]*)\s+\|"
    r"\s+(?P<value>\S+)\s+\|"
    r"\s+(?P<scale>\S+)\s+\|"
    r"\s+(?P<units>[^|]*?)\s*\|"
)


def _read_start_table(log_path):
    values = {}
    for line in Path(log_path).read_text().splitlines():
        m = _START_ROW.search(line)
        if m is None:
            continue
        try:
            values[m.group("label")] = float(m.group("value"))
        except ValueError:
            continue  # N/A rows and the header
    return values


def test_run_fit_ob09020_trace_file_written(ob09020_result):
    """
    Given the ob09020 example with a minimal PTDE budget,
    When run_fit completes,
    Then a NetCDF trace is written and the run log carries no traceback.
    """
    out_dir, _ = ob09020_result
    assert (out_dir / f"{_NAME}_trace.nc").exists()
    log_text = (out_dir / f"{_NAME}.log").read_text()
    assert "Traceback" not in log_text, log_text[-3000:]


def test_run_fit_ob09020_start_plots_written(ob09020_result):
    """
    Given the ob09020 example,
    When run_fit reaches the start-point plots (before sampling),
    Then every data component's plot exists -- the microlensing light curve,
      its zoom, and the RV plots.  This is the step that died with
      "The type's shape ((2,)) is not compatible with the data's ((1,))" when
      the orbit-derived lens geometry was listed as a compiled-plotter input.
    """
    out_dir, _ = ob09020_result
    for suffix in (
        "_start_mulens.pdf",
        "_start_mulens_zoom.pdf",
        "_start_RV_unphased.pdf",
        "_start_RV_phased_L.pdf",
    ):
        assert (out_dir / f"{_NAME}{suffix}").exists(), suffix


def test_run_fit_ob09020_trace_has_expected_variables(ob09020_result):
    """
    Given the ob09020 example,
    When run_fit completes,
    Then the posterior carries the sampled microlensing, orbit, stellar and
      instrument coordinates and the derived event parameters.
    """
    out_dir, _ = ob09020_result
    idata = az.from_netcdf(str(out_dir / f"{_NAME}_trace.nc"))
    posterior_vars = set(idata.posterior.data_vars)

    expected = {
        "source.t_0",
        "source.u_0",
        "orbit.logP",
        "orbit.xbigomega",
        "star.logmass",
        "star.distance",
        "mulensinstrument.log_f_total",
        "rvinstrument.gamma",
    }
    missing = expected - posterior_vars
    assert not missing, f"Missing expected posterior variables: {missing}"
    # No sampled geometry coordinates in keplerian mode (C24, 5b).  The
    # event's t_E / theta_E / pi_E are pure-expression parameters and so
    # are never in idata.posterior either (they are reached through the
    # component's Parameter.posterior; see outputs.md) -- not asserted
    # present here for that reason.
    for name in ("lens.log_s", "lens.xalpha", "lens.yalpha"):
        assert name not in posterior_vars, name


def test_run_fit_ob09020_starts_at_the_user_seeds(ob09020_result):
    """
    Given the ob09020 example run with the polish off,
    When the startup table is read from the run's own log,
    Then the sampled parameters the relaxation engine reproduces exactly
      start at the values ob09020.params.yaml sets (the user-start contract),
      and the derived companion mass ratio is the ratio of the two lens
      masses it is built from -- an identity that survives any budget.
    """
    out_dir, _ = ob09020_result
    table = _read_start_table(out_dir / f"{_NAME}.log")
    assert table, "no startup table in the run log"

    # The table prints eight significant figures (2454917.25 for this
    # value), so the tolerance is the print resolution, not the seed's.
    assert table["source.Source.t_0"] == pytest.approx(2454917.252, abs=0.01)
    assert table["source.Source.u_0"] == pytest.approx(0.0613, abs=1e-9)
    assert table["orbit.L.period"] == pytest.approx(276.555, rel=1e-8)
    assert table["star.L1.mass"] == pytest.approx(0.888, rel=1e-8)
    assert table["star.L1.distance"] == pytest.approx(747.0, rel=1e-8)

    q = 10 ** (table["star.L2.logmass"] - table["star.L1.logmass"])
    assert table["lens.L2.q"] == pytest.approx(q, rel=1e-6)
    assert table["lens.L2.q"] == pytest.approx(0.273, rel=1e-6)
