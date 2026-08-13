"""
Tests for the exozippy-modes CLI (cli_modes.py) and the shared reporting
pipeline it uses (outputs/report_pipeline.py).

exozippy-modes reprocesses a previously saved trace (<prefix>_trace.nc)
through outputs.modes.identify_modes and System.distribute_posterior without
re-sampling, rewriting <prefix>_modes.txt/_definitions.tex/_table.tex/
_results.csv and persisting the mode labels back into the trace file. It
must use the exact same identify_modes -> distribute_posterior -> LaTeX/CSV
pipeline that run.run_fit() uses on a live fit (outputs.report_pipeline.
build_mode_reports), so the two call sites cannot drift apart.
"""

import csv
import shutil
import subprocess

import arviz as az
import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from exozippy import cli_modes
from exozippy import run as run_module
from exozippy.outputs.ledger import SeedRecord
from exozippy.outputs.report_pipeline import build_mode_reports
from exozippy.system import System

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
        posterior[name] = (rng.normal(0, 1, N) + shift).reshape(
            N_CHAIN, N_DRAW
        )

    lp = (rng.normal(1000, 3, N) - 5 * truth).reshape(N_CHAIN, N_DRAW)

    idata = az.from_dict(
        {
            "posterior": posterior,
            "sample_stats": {"lp": lp},
        }
    )

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

    for suffix in (
        "_modes.txt",
        "_definitions.tex",
        "_table.tex",
        "_results.csv",
    ):
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
    result = runner.invoke(
        cli_modes.main, [str(config_path), "--min-weight", "0.5"]
    )

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


# ----------------------------------------------------------------------
# Review 2.8.1 / 2.8.2: the pipeline's own output files
#
# The trigger for both is the ob140939 setup -- a multi-seed fit with
# rejected seeds whose surviving posterior is UNIMODAL -- run under a
# prefix that contains an underscore (DC2018_128 and
# KMT-2019-BLG-1806_nt8long are both real prefixes in this repo).
# ----------------------------------------------------------------------

# Spelled out rather than imported, so the test pins the contract instead
# of whatever the code happens to define.
MODE_COLUMNS = (
    "parname",
    "mode",
    "weight",
    "weight_err",
    "value",
    "up_err",
    "low_err",
)


def _prepared_system():
    config, user_params = _orbit_config_and_params()
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    return system, model


def _unimodal_idata(names, rng):
    posterior = {
        n: rng.normal(0, 1, N).reshape(N_CHAIN, N_DRAW) for n in names
    }
    lp = rng.normal(1000, 3, N).reshape(N_CHAIN, N_DRAW)
    return az.from_dict({"posterior": posterior, "sample_stats": {"lp": lp}})


def _seed_ledger(names):
    """Two Laplace records: one on the surviving mode, one far outside it
    (rejected).  Hand-built so the test does not need a real polish pass."""

    def rec(k, offset):
        return SeedRecord(
            seed_index=k,
            lp_max=1000.0 - 10.0 * k,
            delta_lp=10.0 * k,
            laplace_logw=1000.0 - 10.0 * k,
            raw_point={n: np.array([offset]) for n in names},
            raw_scales={n: np.array([1.0]) for n in names},
            phys={"orbit.logP": np.array([1.0 + offset])},
            phys_sigma={"orbit.logP": np.array([0.1])},
            sampled_idx={"orbit.logP": [0]},
        )

    return [rec(0, 0.0), rec(1, 500.0)]


def _pipeline_outputs(tmp_path, prefix_name):
    system, model = _prepared_system()
    rng = np.random.default_rng(3)
    names = [v.name for v in model.free_RVs]
    idata = _unimodal_idata(names, rng)
    prefix = tmp_path / prefix_name
    report = build_mode_reports(
        system,
        idata,
        str(prefix),
        raise_on_invalid=False,
        seed_ledger=_seed_ledger(names),
    )
    assert report.n_modes == 1  # the case the review is about
    return prefix


def test_unimodal_fit_with_rejected_seeds_writes_a_rectangular_csv(tmp_path):
    """
    Given a multi-seed fit whose surviving posterior is unimodal and whose
      seed ledger holds a rejected solution,
    When the reporting pipeline writes <prefix>_results.csv,
    Then the file is rectangular and its header comment describes its rows:
      csv.reader sees one row width, and DictReader keyed on the header
      reads the rejected-seed row back with its mode key.

    Regression for review 2.8.1: the header path took its column set from
    the mode report (4 columns, unimodal) while append_ledger_csv always
    wrote 7, so the file was unparseable.
    """
    # ARRANGE / ACT
    prefix = _pipeline_outputs(tmp_path, "OB140939_unimodal")
    csv_path = prefix.parent / (prefix.name + "_results.csv")

    # ASSERT
    lines = csv_path.read_text().splitlines()
    header = lines[0]
    assert [c.strip() for c in header.lstrip("# ").split(",")] == list(
        MODE_COLUMNS
    )

    with open(csv_path, newline="") as f:
        rows = [
            r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")
        ]
    assert {len(r) for r in rows} == {len(MODE_COLUMNS)}
    modes = {r[1] for r in rows}
    assert "all" in modes and "rejected-seed1" in modes


def test_underscored_prefix_produces_a_compilable_caption(tmp_path):
    """
    Given an output prefix containing an underscore,
    When the reporting pipeline writes <prefix>_table.tex,
    Then the caption escapes it -- no bare underscore survives in the
      caption text, and (where pdflatex is installed) that caption text
      compiles.

    Regression for review 2.8.2: the raw prefix.stem went into
    \\tablecaption{}, so the final table of a long fit would not compile.
    """
    # ARRANGE / ACT
    prefix = _pipeline_outputs(tmp_path, "KMT-2019-BLG-1806_nt8long")
    tmpl = (prefix.parent / (prefix.name + "_table.tex")).read_text()

    # ASSERT
    caption = next(
        ln for ln in tmpl.splitlines() if ln.startswith(r"\tablecaption")
    )
    assert r"KMT-2019-BLG-1806\_nt8long" in caption
    # everything before \label is typeset text: no bare underscore there
    typeset = caption.split(r"\label")[0]
    assert "_" not in typeset.replace(r"\_", "")

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not installed")
    doc = tmp_path / "caption.tex"
    body = caption[len(r"\tablecaption{") :].split(r"\label")[0]
    doc.write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        + body
        + "\n\\end{document}\n"
    )
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", doc.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]


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
