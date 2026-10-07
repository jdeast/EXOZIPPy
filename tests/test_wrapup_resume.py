"""Resume after a wrap-up raise (review 2.14.12, JDE ruling 2026-09-25).

"Everything between sampling and saving the trace shouldn't raise, but
after we save the trace, we should be able to restart with
recompute_trace=False to resume after a raise."

End to end on the kelt4 RV-only example under DE-MC -- the sampler family
that writes no lp, so lp is computed from the model AFTER the save
(run._finish_saved_trace).  That computation is the failure injected: it is
the exact step whose swallowed failure used to ship a trace without lp
(review 2.3.21), and it is the earliest post-save stage, so the rerun has
to finish the trace itself as well as the reports.

  * A clean fit (A) gives the reference outputs.
  * The same fit (B) with the lp computation raising: run_fit raises the
    ORIGINAL error, the trace is on disk stamped internal-unfinished, the
    summary file says where wrap-up failed and how to resume, no restart
    file was written, and an ordinary reader (mkparam) refuses the
    unfinished trace.
  * B rerun with `recompute_trace: false` and the fault gone: completes,
    the trace is finished in place, and the tables, CSV and restart file
    match A's -- the resumed wrap-up IS the live one.

Marked 'slow' (three short real fits); excluded from the pre-push tier.
"""

import os
import shutil
from pathlib import Path

import arviz as az
import pytest
import yaml

from exozippy import run as run_module
from exozippy.trace_meta import (
    POSTERIOR_UNITS,
    POSTERIOR_UNITS_UNFINISHED,
    UNITS_ATTR,
    UnfinishedTraceError,
)

pytestmark = pytest.mark.slow

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "kelt4"
PREFIX = "fitresults/KELT-4A"


def _workdir(tmp_path_factory, name):
    work = tmp_path_factory.mktemp(name) / "kelt4"
    shutil.copytree(
        EXAMPLE_DIR,
        work,
        ignore=shutil.ignore_patterns("fitresults*", ".#*", "#*#"),
    )
    return work


def _config(work, recompute):
    with open(work / "kelt4_rvonly.yaml") as f:
        config = yaml.safe_load(f)
    config["prefix"] = PREFIX
    config["sampler"] = {
        "method": "demc",
        "tune": 10,
        "draws": 10,
        "chains": 8,
        "cores": 1,
        "seed": 20261006,
        # measure_scales stays ON (the default): the whitening state, and
        # with it the polished anchor the raw draws decode under, is then
        # persisted beside the trace and restored by the rerun.
        "recompute_trace": recompute,
    }
    config["modeling"] = {"compile": False}
    return config


def _fit(work, recompute):
    cwd = os.getcwd()
    os.chdir(work)
    try:
        run_module.run_fit(_config(work, recompute))
    finally:
        os.chdir(cwd)


def _restart_files(work):
    return sorted(p.name for p in work.glob("*.params.*.yaml"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Run A clean; run B with the post-save lp computation failing."""
    work_a = _workdir(tmp_path_factory, "resume_a")
    _fit(work_a, recompute=True)

    work_b = _workdir(tmp_path_factory, "resume_b")
    real = run_module._compute_lp_from_model

    def _broken(*args, **kwargs):
        raise RuntimeError("injected post-save failure")

    run_module._compute_lp_from_model = _broken
    try:
        with pytest.raises(RuntimeError) as err:
            _fit(work_b, recompute=True)
    finally:
        run_module._compute_lp_from_model = real

    failed = {
        "error": err.value,
        "trace_units": az.from_netcdf(
            str(work_b / (PREFIX + "_trace.nc"))
        ).attrs.get(UNITS_ATTR),
        "summary": (work_b / (PREFIX + "_summary.txt")).read_text(
            encoding="utf-8"
        ),
        "restart_files": _restart_files(work_b),
    }
    return work_a, work_b, failed


def test_a_post_save_failure_raises_and_leaves_the_trace(runs):
    """
    Given a fit whose first post-save stage raises,
    When run_fit returns,
    Then the ORIGINAL error propagated (carrying the wrap-up note), and the
      trace is on disk, stamped unfinished -- the draws survived.
    """
    _, _, failed = runs

    assert "injected post-save failure" in str(failed["error"])
    assert any(
        "WRAP-UP FAILED" in note
        for note in getattr(failed["error"], "__notes__", [])
    )
    assert failed["trace_units"] == POSTERIOR_UNITS_UNFINISHED


def test_the_failure_is_stated_and_no_restart_file_is_written(runs):
    """
    Given the failed fit,
    When its artifacts are read,
    Then the summary opens with where it failed and how to resume, and no
      restart parameter file was written.
    """
    _, _, failed = runs

    summary = failed["summary"]
    assert summary.lstrip("!\n").startswith("WRAP-UP FAILED at stage")
    assert "finishing the saved trace" in summary
    assert "recompute_trace: false" in summary
    assert failed["restart_files"] == []


def test_an_ordinary_reader_refuses_the_unfinished_trace(runs):
    """
    Given the unfinished trace,
    When mkparam is pointed at it directly,
    Then it raises UnfinishedTraceError instead of seeding the next fit
      from internal-unit draws.
    """
    from exozippy.mkparam import write_param_file

    work_a, work_b, _ = runs
    unfinished = work_b.parent / "unfinished_copy"
    shutil.copytree(work_b, unfinished)
    # the rerun below finishes work_b's trace in place; mkparam reads the
    # untouched copy, so this test does not depend on test ordering
    cwd = os.getcwd()
    os.chdir(unfinished)
    try:
        trace = az.from_netcdf(PREFIX + "_trace.nc")
        if trace.attrs.get(UNITS_ATTR) != POSTERIOR_UNITS_UNFINISHED:
            pytest.skip("the copy was taken after the resume finished it")
        with pytest.raises(UnfinishedTraceError, match="recompute_trace"):
            write_param_file(
                _config(unfinished, recompute=False),
                trace_path=PREFIX + "_trace.nc",
            )
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def resumed(runs):
    work_a, work_b, _ = runs
    _fit(work_b, recompute=False)
    return work_a, work_b


def test_a_recompute_false_rerun_finishes_the_trace(resumed):
    """
    Given the failed fit, the fault removed,
    When it is rerun with recompute_trace: false,
    Then it completes and the trace on disk is finished in place: user
      units, and lp present.
    """
    _, work_b = resumed

    trace = az.from_netcdf(str(work_b / (PREFIX + "_trace.nc")))
    assert trace.attrs[UNITS_ATTR] == POSTERIOR_UNITS
    assert "lp" in trace.sample_stats.data_vars
    assert "WRAP-UP FAILED" not in (
        work_b / (PREFIX + "_summary.txt")
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name",
    [
        "_results.csv",
        "_table.tex",
        "_modes.txt",
        pytest.param(
            "_definitions.tex",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "PRE-EXISTING, not the resume: a sigma prior written "
                    "without mu is displayed centered on Parameter.initval "
                    "(_own_prior_str), and a FRESH fit's seed polish moves "
                    "initval -- so the uninterrupted fit prints the polished "
                    "start (N(1.6117, 0.05)) and the rerun, which skips the "
                    "polish, the user's value (N(1.61, 0.05))"
                ),
            ),
        ),
    ],
)
def test_the_resumed_wrapup_reproduces_an_uninterrupted_fit(resumed, name):
    """
    Given a clean fit (A) and the failed-then-resumed fit (B) of the same
    config and seed,
    When their reports are compared,
    Then they are identical: the resume is the live wrap-up, not a second
      implementation of it.
    """
    work_a, work_b = resumed

    a = (work_a / (PREFIX + name)).read_text(encoding="utf-8")
    b = (work_b / (PREFIX + name)).read_text(encoding="utf-8")
    assert a == b


def test_the_resumed_restart_file_matches(resumed):
    """
    Given the same pair,
    When the restart parameter files are compared,
    Then they are identical too (one each, same name, same contents).
    """
    work_a, work_b = resumed

    assert _restart_files(work_a) == _restart_files(work_b) != []
    for name in _restart_files(work_a):
        assert (work_a / name).read_text(encoding="utf-8") == (
            work_b / name
        ).read_text(encoding="utf-8")


def test_exozippy_modes_reproduces_the_live_wrapup(resumed, tmp_path):
    """
    Given the clean fit (A) -- whose trace carries the DE-MC burn-in
      transient that the live wrap-up trims --
    When `exozippy-modes` reprocesses a copy of it,
    Then its tables, CSV and mode report are identical to the live fit's
      (review 1.3.9): the CLI runs the live wrap-up, fold and burn-in trim
      included, instead of a partial copy of it.
    """
    from click.testing import CliRunner

    from exozippy import cli_modes

    work_a, _ = resumed
    copy = tmp_path / "kelt4"
    shutil.copytree(work_a, copy)
    with open(copy / "cli.yaml", "w") as f:
        # sort_keys=False: block order is the table's row order
        yaml.safe_dump(_config(copy, recompute=True), f, sort_keys=False)

    cwd = os.getcwd()
    os.chdir(copy)
    try:
        result = CliRunner().invoke(cli_modes.main, ["cli.yaml"])
    finally:
        os.chdir(cwd)

    assert result.exit_code == 0, repr(result.exception)
    for name in ("_results.csv", "_table.tex", "_modes.txt"):
        assert (copy / (PREFIX + name)).read_text(encoding="utf-8") == (
            work_a / (PREFIX + name)
        ).read_text(encoding="utf-8"), name
