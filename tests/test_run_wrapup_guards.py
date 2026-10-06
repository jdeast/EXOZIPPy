"""run.py's "do not lose the fit" guards, under the three-phase ruling.

* ``sigterm_as_interrupt`` (review 3.3.1) around every direct ``pm.sample``,
  so a scheduler SIGTERM keeps a partial trace.
* The three-phase contract (review 2.14.12, JDE 2026-09-25): "everything
  between sampling and saving the trace shouldn't raise, but after we save
  the trace, we should be able to restart with recompute_trace=False to
  resume after a raise."  So the SAMPLING -> SAVE window
  (``_save_sampled_trace``) holds only work that cannot fail, and the
  wrap-up after the save (``_wrap_up``) swallows NOTHING: a failure is
  recorded in the artifacts (``_record_wrapup_failure``) and re-raised.
  This file used to pin the opposite -- ``nonfatal_wrapup`` around every
  wrap-up plot -- which the ruling retired.

The structural guards read the source rather than running a fit: reaching
the save or a wrap-up stage for real costs a full sample-plus-wrap-up.  The
end-to-end resume (a post-save raise, then a ``recompute_trace: false``
rerun reproducing an uninterrupted fit) is tests/test_wrapup_resume.py.
"""

import ast
import inspect
import logging
from pathlib import Path

import pytest

from exozippy import run as run_module


def _function_node(name):
    tree = ast.parse(Path(inspect.getfile(run_module)).read_text())
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_names(func):
    names = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            f = node.func
            names.add(getattr(f, "id", getattr(f, "attr", None)))
    return names


# What may run between the sampler returning and the trace being on disk:
# bookkeeping that cannot fail by construction, the write itself, and the
# last-record dump on a failed write (which re-raises).
SAVE_WINDOW_ALLOWED = {
    "sel",
    "slice",
    "int",
    "str",
    "_sanitize_netcdf_attrs",
    "apply_metadata",
    "_write_trace_atomic",
    "critical",
    "open",
    "dump",
}


def test_the_save_window_runs_only_what_cannot_fail():
    """
    Given the SAMPLING -> SAVE window (_save_sampled_trace),
    When every call in it is read from the source,
    Then each is on the allowlist -- in particular lp computation, the
      unit conversion and the stamp BUILD are not there (they can fail, and
      a raise here loses the trace; review 2.14.12 prerequisite 1).
    """
    called = _called_names(_function_node("_save_sampled_trace"))

    assert called - SAVE_WINDOW_ALLOWED == set(), (
        f"call(s) in the sampling->save window that can fail: "
        f"{sorted(called - SAVE_WINDOW_ALLOWED)}"
    )
    assert "_write_trace_atomic" in called


def test_run_fit_does_no_post_processing_before_the_save():
    """
    Given _run_fit,
    When its calls are read from the source,
    Then it computes no lp, converts no units and writes no netCDF itself:
      those live in _save_sampled_trace (the write) and _finish_saved_trace
      (lp + units, AFTER the save).  This is where they used to sit, between
      pm.sample and idata.to_netcdf, and _ensure_lp's swallowed failure
      shipped a trace without lp (review 2.3.21).
    """
    called = _called_names(_function_node("_run_fit"))

    for name in (
        "_ensure_lp",
        "_compute_lp_from_model",
        "_convert_posterior_to_user_units",
        "to_netcdf",
        "stamp_structural_metadata",
    ):
        assert name not in called, f"_run_fit calls {name} directly"
    assert "_save_sampled_trace" in called
    assert "structural_metadata" in called  # built BEFORE sampling


def test_nothing_after_the_save_swallows_an_exception():
    """
    Given the post-save code (_wrap_up and _finish_saved_trace),
    When it is read from the source,
    Then it holds no except clause at all, and nonfatal_wrapup is gone --
      a post-save failure is a code bug, recovered by the fix plus a
      `recompute_trace: false` rerun (review 2.14.12).
    """
    assert not hasattr(run_module, "nonfatal_wrapup")
    for name in ("_wrap_up", "_finish_saved_trace"):
        handlers = [
            node
            for node in ast.walk(_function_node(name))
            if isinstance(node, ast.ExceptHandler)
        ]
        assert not handlers, (
            f"{name} has {len(handlers)} except clause(s) at line(s) "
            f"{[h.lineno for h in handlers]}"
        )


def test_the_wrapup_runs_under_the_failure_recorder_and_reraises():
    """
    Given _run_fit's call to _wrap_up,
    When its enclosing try is read from the source,
    Then the handler records the failure and re-raises with a bare `raise`
      (the ORIGINAL exception, not a replacement).
    """
    func = _function_node("_run_fit")
    tries = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Try)
        and "_wrap_up" in _called_names(ast.Module(node.body, []))
    ]
    assert len(tries) == 1
    (handler,) = tries[0].handlers
    assert "_record_wrapup_failure" in _called_names(
        ast.Module(handler.body, [])
    )
    assert isinstance(handler.body[-1], ast.Raise)
    assert handler.body[-1].exc is None


def test_a_wrapup_failure_is_stated_where_the_user_looks(tmp_path, caplog):
    """
    Given a wrap-up stage that raised,
    When the failure is recorded,
    Then the summary file opens with the note (stage, saved trace, the
      recompute_trace: false remedy, and that NO restart file was written),
      any summary already there is kept below it, the log has it at ERROR,
      and the exception carries it as a note -- which is what reaches the
      traceback run_fit writes into the GUI's status.json.
    """
    prefix = tmp_path / "fit"
    summary = Path(str(prefix) + "_summary.txt")
    summary.write_text("old summary body\n", encoding="utf-8")
    exc = ValueError("degenerate KDE")

    with caplog.at_level(logging.ERROR, logger="exozippy.run"):
        run_module._record_wrapup_failure(
            exc, "detailed trace plots", prefix, str(prefix) + "_trace.nc"
        )

    text = summary.read_text(encoding="utf-8")
    assert text.index("WRAP-UP FAILED") < text.index("old summary body")
    for needle in (
        "detailed trace plots",
        "fit_trace.nc",
        "recompute_trace: false",
        "NO restart parameter",
        "degenerate KDE",
    ):
        assert needle in text, needle
    assert "WRAP-UP FAILED" in caplog.text
    assert any("WRAP-UP FAILED" in n for n in exc.__notes__)


def test_the_status_file_carries_the_wrapup_failure(tmp_path, monkeypatch):
    """
    Given a fit whose wrap-up fails with GUI status output on,
    When run_fit records the terminal state,
    Then status.json's error holds the wrap-up note, not only the bare
      traceback (review 2.14.12 prerequisite 4).
    """
    import json

    prefix = tmp_path / "fit"

    def _fake_run_fit(cfg, gui, user_params=None):
        gui.phase("writing")
        exc = RuntimeError("mode pass bug")
        run_module._record_wrapup_failure(
            exc, "mode identification", prefix, str(prefix) + "_trace.nc"
        )
        raise exc

    monkeypatch.setattr(run_module, "_run_fit", _fake_run_fit)

    with pytest.raises(RuntimeError, match="mode pass bug"):
        run_module.run_fit({"prefix": str(prefix), "gui": {"snapshot": True}})

    status = json.loads(
        Path(str(prefix) + "_gui_status.json").read_text(encoding="utf-8")
    )
    assert status["phase"] == "error"
    assert "WRAP-UP FAILED at stage 'mode identification'" in status["error"]


def test_every_pm_sample_call_is_sigterm_wrapped():
    """
    Given _run_fit's sampler dispatch (review 3.3.1),
    When each direct pm.sample call is located,
    Then all of them sit inside `with sigterm_as_interrupt()`.

    The docstring claimed coverage of "every branch that calls pm.sample
    directly" while nutpie's was bare, so a scheduler SIGTERM there was an
    immediate kill with no partial trace.  nutpie is the branch that most
    needed it: it reaches pm.sample through an EXTERNAL sampler, and
    nutpie.sample really does catch the interrupt and return the draws
    taken so far.
    """
    tree = ast.parse(Path(inspect.getfile(run_module)).read_text())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_fit"
    )

    found = []

    def walk(node, inside):
        if isinstance(node, ast.With):
            inside = inside or any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "id", None)
                == "sigterm_as_interrupt"
                for item in node.items
            )
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "sample"
                and getattr(f.value, "id", None) == "pm"
            ):
                found.append(inside)
        for child in ast.iter_child_nodes(node):
            walk(child, inside)

    walk(func, False)

    assert found, "no pm.sample call found in _run_fit"
    assert all(found), (
        f"{found.count(False)} of {len(found)} pm.sample calls in _run_fit "
        "are not wrapped in sigterm_as_interrupt -- a scheduler SIGTERM "
        "there kills the fit with no partial trace"
    )


# ---------------------------------------------------------------------------
# Wrap-up VISIBILITY: a long fit must not go silent, and the polish must not
# go serial (reviews 2.3.5 and 6.11.3)
# ---------------------------------------------------------------------------

# Every call that starts a polish has to hand over a core grant.  Since
# 6.11.3 an omitted grant no longer means SERIAL -- `cores=None` resolves to
# _common.default_cores() like every other pool in the package -- so this
# guard is about the OTHER half of that fix: a call site that names no grant
# silently ignores the user's `sampler: cores:`, taking the default even from
# someone who asked for 4 on a shared machine.  Pinned across the whole
# package rather than at the two known sites, so a THIRD caller cannot
# reintroduce it.
CORE_GRANTING_CALLS = (
    "polish_raw_starts",
    "polish_rounds",
    "run_hot_mode_discovery",
)


def _package_source_files():
    root = Path(inspect.getfile(run_module)).parent
    return sorted(root.rglob("*.py"))


def test_every_polish_call_site_passes_a_core_grant():
    """
    Given every call in the package that starts (or forwards to) a polish,
    When each call's keywords are read from the source,
    Then all of them pass `cores=` -- so the grant the user configured
      reaches the stage, instead of the stage quietly picking its own.
    """
    offenders = []
    seen = 0
    for path in _package_source_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name not in CORE_GRANTING_CALLS:
                continue
            seen += 1
            kwargs = {kw.arg for kw in node.keywords}
            if "cores" not in kwargs and None not in kwargs:
                offenders.append(f"{path.name}:{node.lineno} {name}")

    assert seen, "no polish call sites found -- the names must have changed"
    assert not offenders, (
        "polish call site(s) with no core grant: "
        + ", ".join(offenders)
        + " -- that stage then resolves its own grant and a user's "
        "`sampler: cores:` never reaches it (review 6.11.3)"
    )


def test_wrapup_stage_lines_carry_elapsed_time_and_the_stage(caplog):
    """
    Given the wrap-up progress announcer,
    When a stage starts,
    Then one INFO line names the stage and stamps elapsed time since
      wrap-up began, and the closing line reports the total.

    Wrap-up used to log NOTHING between the sampler finishing and the
    reports appearing -- on examples/ob09020 that was 38+ silent minutes,
    and telling "computing" from "hung" needed /proc/<pid>/stat (2.3.5a).
    """
    # ARRANGE
    progress = run_module.WrapupProgress()

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        progress.stage("hot-chain suppressed-mode search")
        progress.done()

    # ASSERT
    assert "Wrap-up (t+" in caplog.text
    assert "hot-chain suppressed-mode search" in caplog.text
    assert "Wrap-up complete in" in caplog.text


def _wrapup_stage_labels(func_name="_run_fit"):
    """Every literal label passed to ``wrapup.stage(...)`` in ``func``."""
    tree = ast.parse(Path(inspect.getfile(run_module)).read_text())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )
    labels = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "stage"
            and getattr(node.func.value, "id", None) == "wrapup"
            and node.args
        ):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                labels.append(arg.value)
            else:
                # an f-string / concatenation: keep its literal pieces
                labels.extend(
                    piece.value
                    for piece in ast.walk(arg)
                    if isinstance(piece, ast.Constant)
                    and isinstance(piece.value, str)
                )
    return labels


def test_the_convergence_summary_announces_itself():
    """
    Given _run_fit's wrap-up,
    When its wrapup.stage labels are read out of the source,
    Then the convergence summary has one.

    Every neighbouring stage got a stage line in PR #214 and this write did
    not, so the log jumped from "mode identification" to "corner plots"
    across a step that does its own az.summary (review 2.3.12).  Read from
    the source for the reason the whole file gives: reaching this line for
    real costs a full sample-plus-wrap-up.  The stage label is also what a
    failure is reported against (WrapupProgress.current).
    """
    labels = _wrapup_stage_labels("_wrap_up")

    assert labels, "no wrapup.stage calls found in _wrap_up"
    assert any("convergence summary" in label for label in labels), labels


def test_an_interrupt_during_wrapup_says_what_survived(caplog, monkeypatch):
    """
    Given a fit interrupted during WRAP-UP rather than during sampling,
    When run_fit handles the KeyboardInterrupt,
    Then it says the trace is already saved and names how to regenerate the
      remaining reports without re-sampling.

    Sampling documents its own interrupt behavior; wrap-up documented none,
    so an impatient Ctrl-C felt like it might cost the multi-day trace it
    cannot (review 2.3.5d).  The phase is read from the run's own reporter,
    which records it even when GUI status output is off -- the default.
    """
    # ARRANGE
    config = {"prefix": "fitresults/planet"}

    def _fake_run_fit(cfg, gui, user_params=None):
        gui.phase("writing")
        raise KeyboardInterrupt

    monkeypatch.setattr(run_module, "_run_fit", _fake_run_fit)

    # ACT
    with caplog.at_level(logging.WARNING, logger="exozippy.run"):
        with pytest.raises(KeyboardInterrupt):
            run_module.run_fit(config)

    # ASSERT
    assert "Interrupted during wrap-up" in caplog.text
    assert "fitresults/planet_trace.nc" in caplog.text
    assert "exozippy-modes" in caplog.text


def test_an_interrupt_during_sampling_makes_no_such_claim(caplog, monkeypatch):
    """
    Given a fit interrupted during SAMPLING,
    When run_fit handles the KeyboardInterrupt,
    Then the wrap-up notice does NOT fire -- at that point the trace is not
      on disk yet, and telling the user it is would be false assurance.
    """

    # ARRANGE
    def _fake_run_fit(cfg, gui, user_params=None):
        gui.phase("sampling")
        raise KeyboardInterrupt

    monkeypatch.setattr(run_module, "_run_fit", _fake_run_fit)

    # ACT
    with caplog.at_level(logging.WARNING, logger="exozippy.run"):
        with pytest.raises(KeyboardInterrupt):
            run_module.run_fit({"prefix": "fitresults/planet"})

    # ASSERT
    assert "Interrupted during wrap-up" not in caplog.text
