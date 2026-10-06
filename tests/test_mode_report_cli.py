"""
Tests for the exozippy-modes CLI (cli_modes.py) and the reporting pipeline
(outputs/report_pipeline.py).

exozippy-modes reprocesses a previously saved trace (<prefix>_trace.nc)
without re-sampling by running the fit itself with `sampler:
{recompute_trace: false}` (review 1.3.9): the whole live wrap-up, through
run._wrap_up, so it cannot drift from a live fit.  It used to call
build_mode_reports directly and skipped the degeneracy fold and the burn-in
trim.  The end-to-end equality with a live fit's reports is
tests/test_wrapup_resume.py; the CLI's own contract is pinned here.
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


def _invoke_capturing_run_fit(monkeypatch, args):
    """Run the CLI with run.run_fit replaced by a recorder."""
    seen = {}

    def _fake_run_fit(config, user_params=None):
        seen["config"] = config

    monkeypatch.setattr(run_module, "run_fit", _fake_run_fit)
    result = CliRunner().invoke(cli_modes.main, args)
    return result, seen


def test_cli_runs_the_live_fit_with_recompute_trace_false(
    tmp_path, monkeypatch
):
    """
    Given a config whose saved trace exists,
    When `exozippy-modes config.yaml` runs,
    Then it hands the config to run.run_fit -- the live path -- with
      `sampler: {recompute_trace: false}`, and changes nothing else (no
      modes.force unless asked).
    """
    rng = np.random.default_rng(42)
    config_path, prefix = _write_config(tmp_path)
    _write_synthetic_trace(prefix, rng)
    with open(config_path) as f:
        on_disk = yaml.safe_load(f)

    result, seen = _invoke_capturing_run_fit(monkeypatch, [str(config_path)])

    assert result.exit_code == 0, result.output + repr(result.exception)
    config = seen["config"]
    assert config["sampler"]["recompute_trace"] is False
    assert "force" not in (config.get("modes") or {})
    expected = dict(on_disk)
    expected["sampler"] = dict(on_disk.get("sampler") or {})
    expected["sampler"]["recompute_trace"] = False
    assert config == expected


def test_cli_force_sets_modes_force(tmp_path, monkeypatch):
    """
    Given --force,
    When the CLI runs,
    Then the run gets `modes: {force: true}` -- the live fit's own
      override of the invalid-draw gate, not a CLI-only behaviour.
    """
    rng = np.random.default_rng(7)
    config_path, prefix = _write_config(tmp_path)
    _write_synthetic_trace(prefix, rng)

    result, seen = _invoke_capturing_run_fit(
        monkeypatch, [str(config_path), "--force"]
    )

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert seen["config"]["modes"]["force"] is True


def test_cli_missing_trace_reports_error(tmp_path, monkeypatch):
    """
    Given a config whose trace file was never generated,
    When the CLI runs,
    Then it fails loudly (FileNotFoundError) and never reaches run_fit --
      `recompute_trace: false` with no trace on disk would SAMPLE.
    """
    config_path, prefix = _write_config(tmp_path)

    result, seen = _invoke_capturing_run_fit(monkeypatch, [str(config_path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)
    assert seen == {}


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


def test_run_uses_the_pipeline_and_the_cli_uses_run():
    """
    Given run.py's live wrap-up and cli_modes.py's reprocessing path,
    When each module is read,
    Then run.py calls outputs.report_pipeline.build_mode_reports and the
      CLI calls nothing but run.run_fit -- one wrap-up, two entry points
      (review 1.3.9).  The CLI used to call build_mode_reports itself and
      so skipped the fold and the burn-in trim a live fit runs first.
    """
    import ast
    import inspect

    from exozippy.outputs.report_pipeline import build_mode_reports

    assert run_module.build_mode_reports is build_mode_reports
    assert not hasattr(cli_modes, "build_mode_reports")
    tree = ast.parse(inspect.getsource(cli_modes))
    called = {
        getattr(n.func, "id", getattr(n.func, "attr", None))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert "run_fit" in called
    assert not called & {
        "build_mode_reports",
        "identify_modes",
        "distribute_posterior",
        "build_model",
    }
