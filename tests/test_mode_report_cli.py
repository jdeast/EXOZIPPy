"""
Tests for the exozippy-modes CLI (cli_modes.py) and the shared reporting
pipeline it uses (outputs/report_pipeline.py).

exozippy-modes reprocesses a previously saved trace (<prefix>_trace.nc)
through outputs.modes.identify_modes and System.distribute_posterior without
re-sampling, rewriting <prefix>_modes.txt/_definitions.tex/_template.tex/
_results.csv and persisting the mode labels back into the trace file. It
must use the exact same identify_modes -> distribute_posterior -> LaTeX/CSV
pipeline that run.run_fit() uses on a live fit (outputs.report_pipeline.
build_mode_reports), so the two call sites cannot drift apart.
"""

import yaml
import numpy as np
import arviz as az
import pytest
from click.testing import CliRunner

from exozippy.system import System
from exozippy import run as run_module
from exozippy import cli_modes

pytestmark = pytest.mark.slow

N_CHAIN, N_DRAW = 4, 300
N = N_CHAIN * N_DRAW


def _orbit_config_and_params():
    """A minimal, cheap-to-build System configuration (single free orbit,
    no instruments/likelihood -- we only need real Parameter labels and
    free_RV ('*_raw') names, not a physically meaningful fit)."""
    config = {"name": "modes_cli_test", "orbit": [{"name": "test_orbit"}]}
    user_params = {
        "orbit.test_orbit.logP": {"initval": float(np.log10(10.0))},
        "orbit.test_orbit.tc": {"initval": 0.0},
        "orbit.test_orbit.secosw": {"initval": 0.0},
        "orbit.test_orbit.sesinw": {"initval": 0.0},
    }
    return config, user_params


def _free_rv_names():
    """Build the same System in-process (bypassing YAML I/O) just to read
    off the real free_RV ('*_raw') names -- these must match what the CLI's
    own System.build_model() produces from the on-disk YAML for the
    synthetic trace below to be a valid input to identify_modes."""
    config, user_params = _orbit_config_and_params()
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    return [v.name for v in model.free_RVs]


def _write_config(tmp_path):
    """Write config.yaml + params.yaml to tmp_path; returns (config_path, prefix)."""
    config, user_params = _orbit_config_and_params()
    prefix = tmp_path / "testfit"
    params_path = tmp_path / "params.yaml"
    config_path = tmp_path / "config.yaml"

    config = dict(config)
    config["prefix"] = str(prefix)
    config["parameter_file"] = str(params_path)

    with open(params_path, "w") as f:
        yaml.safe_dump(user_params, f)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    return config_path, prefix


def _write_synthetic_trace(prefix, rng, w2=0.3, sep=10.0):
    """Two Gaussian modes (70/30), mixed within every chain, over the real
    free_RV dimensions, plus a matching sample_stats['lp'].  Returns the
    (chain, draw) truth labels for comparison."""
    names = _free_rv_names()
    truth = (rng.random(N) < w2).astype(int)

    posterior = {}
    for i, name in enumerate(names):
        # first raw dim carries the mode separation; the rest are noise
        shift = sep * truth if i == 0 else 0.0
        posterior[name] = (rng.normal(0, 1, N) + shift).reshape(N_CHAIN, N_DRAW)

    lp = (rng.normal(1000, 3, N) - 5 * truth).reshape(N_CHAIN, N_DRAW)

    idata = az.from_dict({
        "posterior": posterior,
        "sample_stats": {"lp": lp},
    })

    trace_path = str(prefix) + "_trace.nc"
    idata.to_netcdf(trace_path)
    return trace_path, truth.reshape(N_CHAIN, N_DRAW)


# ----------------------------------------------------------------------

def test_cli_reproduces_pipeline_outputs(tmp_path):
    """
    Given a saved trace with two well-separated modes (70/30) mixed within
      every chain,
    When `exozippy-modes config.yaml` runs,
    Then it exits cleanly, writes the modes/definitions/template/results
      files, and rewrites the trace with a 'mode' posterior variable whose
      labels recover the true mode assignment.
    """
    rng = np.random.default_rng(42)
    config_path, prefix = _write_config(tmp_path)
    trace_path, truth = _write_synthetic_trace(prefix, rng)

    runner = CliRunner()
    result = runner.invoke(cli_modes.main, [str(config_path)])

    assert result.exit_code == 0, result.output + "\n" + repr(result.exception)

    for suffix in ("_modes.txt", "_definitions.tex", "_template.tex", "_results.csv"):
        p = tmp_path / (prefix.name + suffix)
        assert p.exists(), f"{p} was not written"
        assert p.stat().st_size > 0

    reloaded = az.from_netcdf(trace_path)
    assert "mode" in reloaded.posterior
    da = reloaded.posterior["mode"]
    assert da.dims == ("chain", "draw")
    assert da.attrs["n_modes"] == 2

    found = da.values.ravel()
    truth_flat = truth.ravel()
    ok = found >= 0
    assert ((found[ok] == 1) == (truth_flat[ok] == 1)).mean() > 0.95


def test_cli_persisted_labels_idempotent(tmp_path):
    """
    Given a saved trace already reprocessed once by the CLI (mode labels
      persisted to the trace file),
    When the CLI runs again on the same trace,
    Then it completes and reproduces identical mode labels (identify_modes
      is deterministic under its default seed).
    """
    rng = np.random.default_rng(7)
    config_path, prefix = _write_config(tmp_path)
    trace_path, _ = _write_synthetic_trace(prefix, rng)

    runner = CliRunner()
    r1 = runner.invoke(cli_modes.main, [str(config_path)])
    assert r1.exit_code == 0, r1.output
    labels_1 = az.from_netcdf(trace_path).posterior["mode"].values.copy()

    r2 = runner.invoke(cli_modes.main, [str(config_path)])
    assert r2.exit_code == 0, r2.output
    labels_2 = az.from_netcdf(trace_path).posterior["mode"].values.copy()

    np.testing.assert_array_equal(labels_1, labels_2)


def test_cli_min_weight_flag_drops_minor_mode(tmp_path):
    """
    Given the same two-mode (70/30) trace,
    When the CLI runs with --min-weight above the minor mode's fraction,
    Then only the dominant mode survives (n_modes == 1 in the rewritten
      trace attrs).
    """
    rng = np.random.default_rng(99)
    config_path, prefix = _write_config(tmp_path)
    trace_path, _ = _write_synthetic_trace(prefix, rng)

    runner = CliRunner()
    result = runner.invoke(cli_modes.main, [str(config_path), "--min-weight", "0.5"])

    assert result.exit_code == 0, result.output
    reloaded = az.from_netcdf(trace_path)
    assert reloaded.posterior["mode"].attrs["n_modes"] == 1


def test_cli_missing_trace_reports_error(tmp_path):
    """
    Given a config whose trace file was never generated,
    When the CLI runs,
    Then it fails loudly (non-zero exit / FileNotFoundError) rather than
      silently producing empty or bogus reports.
    """
    config_path, prefix = _write_config(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli_modes.main, [str(config_path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


def test_run_and_cli_share_the_same_pipeline_function():
    """
    Given run.py's live-fit path and cli_modes.py's saved-trace path,
    When each module is imported,
    Then both reference the identical build_mode_reports function object
      from outputs.report_pipeline -- proving the refactor removed the
      duplicated identify_modes/distribute_posterior/build_latex_output/
      build_csv_output block rather than merely copying it.
    """
    from exozippy.outputs.report_pipeline import build_mode_reports

    assert run_module.build_mode_reports is build_mode_reports
    assert cli_modes.build_mode_reports is build_mode_reports
