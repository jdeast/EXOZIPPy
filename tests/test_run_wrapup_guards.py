"""run.py's two "do not lose the fit" guards, and the calls each must cover.

* ``sigterm_as_interrupt`` (review 3.3.1) around every direct ``pm.sample``,
  so a scheduler SIGTERM keeps a partial trace.
* ``nonfatal_wrapup`` (review 2.3.1) around every wrap-up plot, below.

Both are pinned by locating the calls in the source rather than by running a
fit: reproducing either failure for real means a full sample-plus-wrap-up,
minutes per case, and what the items are about is purely which calls sit
inside the guard.

Wrap-up steps must not be able to kill a finished fit (review 2.3.1).

Everything after ``pm.sample`` returns is a REPORT on a fit that already
finished.  The plotting block between the tables and ``write_param_file`` was
the one stretch of bare calls in an otherwise wrapped wrap-up, so a crash
there -- a degenerate-KDE failure inside ``save_multipage_trace`` is the one
seen in practice, and any short or stopped run can provoke it -- skipped the
restart file and the final paper.tex regeneration of a multi-day fit.
"""

import ast
import inspect
import logging
from pathlib import Path

import pytest

from exozippy import run as run_module
from exozippy.run import nonfatal_wrapup

# The calls the review item names.  Each must sit inside a nonfatal_wrapup
# block within _run_fit's wrap-up; a bare one is the regression.
GUARDED_WRAPUP_CALLS = (
    "make_corner",
    "plot_corner",
    "save_multipage_trace",
    "get_draws",
)


def test_a_crashing_wrapup_step_warns_and_continues(caplog):
    """
    Given a wrap-up step that raises,
    When it runs inside nonfatal_wrapup,
    Then the exception is swallowed, and the log names the step and keeps the
      traceback (which is what reaches the GUI's status.json).
    """
    # ARRANGE / ACT
    with caplog.at_level(logging.WARNING):
        with nonfatal_wrapup("detailed trace plot"):
            raise ValueError("degenerate KDE")

    # ASSERT -- reaching here at all is the swallow
    assert "detailed trace plot" in caplog.text
    assert "non-fatal" in caplog.text
    assert "degenerate KDE" in caplog.text  # exc_info=True kept the traceback


def test_a_clean_wrapup_step_logs_nothing(caplog):
    """
    Given a wrap-up step that succeeds,
    When it runs inside nonfatal_wrapup,
    Then nothing is logged -- the guard is inert on the happy path.
    """
    with caplog.at_level(logging.WARNING):
        with nonfatal_wrapup("corner plot"):
            pass

    assert caplog.text == ""


def test_an_interrupt_is_not_swallowed():
    """
    Given a user interrupting wrap-up,
    When the interrupt is raised inside nonfatal_wrapup,
    Then it propagates.

    KeyboardInterrupt and SystemExit are not Exception subclasses: somebody
    pressing Ctrl+C during wrap-up wants it to stop, and a guard that ate the
    interrupt would make the remaining steps un-interruptible one by one.
    """
    with pytest.raises(KeyboardInterrupt):
        with nonfatal_wrapup("corner plot"):
            raise KeyboardInterrupt


def _wrapup_call_guard_states(func_name):
    """(call name, is-inside-a-nonfatal_wrapup) for every named call in ``func``.

    Read from the source rather than by running a fit: reproducing the failure
    for real means a full sample-plus-wrap-up, minutes per case, and what the
    item is about is purely which calls sit inside the guard.
    """
    tree = ast.parse(Path(inspect.getfile(run_module)).read_text())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )

    found = []

    def walk(node, inside):
        if isinstance(node, ast.With):
            inside = inside or any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "id", None)
                == "nonfatal_wrapup"
                for item in node.items
            )
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name in GUARDED_WRAPUP_CALLS:
                found.append((name, inside))
        for child in ast.iter_child_nodes(node):
            walk(child, inside)

    walk(func, False)
    return found


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


def test_every_named_wrapup_call_is_guarded():
    """
    Given _run_fit's wrap-up,
    When each of the review item's plotting calls is located in the source,
    Then every occurrence has a `with nonfatal_wrapup(...)` ancestor -- and
      all four are actually present, so a rename cannot vacuously pass.
    """
    states = _wrapup_call_guard_states("_run_fit")

    assert {name for name, _ in states} == set(GUARDED_WRAPUP_CALLS)
    unguarded = sorted({name for name, inside in states if not inside})
    assert not unguarded, (
        f"unguarded wrap-up call(s) in _run_fit: {unguarded} -- a crash there "
        "skips the restart file and the final paper.tex of a finished fit"
    )


# ---------------------------------------------------------------------------
# Wrap-up VISIBILITY: a long fit must not go silent, and the polish must not
# go serial (reviews 2.3.5 and 6.11.3)
# ---------------------------------------------------------------------------

# Every call that starts a polish has to hand over a core grant.  Without one
# _resolve_polish_cores returns 1, no pool is built, and the DE engine -- the
# branch every gradient-free (VBM-backed) microlensing fit takes -- runs on
# one core.  run.py's pre-sampling call had this fixed once already; the
# hot-mode call site in outputs/ledger.py was missed and cost 38+ minutes at
# 1/36 throughput on examples/ob09020 (6.11.3).  Pinned across the whole
# package rather than at the two known sites, so a THIRD caller cannot
# reintroduce it.
CORE_GRANTING_CALLS = ("polish_raw_starts", "run_hot_mode_discovery")


def _package_source_files():
    root = Path(inspect.getfile(run_module)).parent
    return sorted(root.rglob("*.py"))


def test_every_polish_call_site_passes_a_core_grant():
    """
    Given every call in the package that starts (or forwards to) a polish,
    When each call's keywords are read from the source,
    Then all of them pass `cores=` -- the omission 6.11.3 found is a silent
      1/N-throughput regression with no failing test of its own.
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
        + " -- the DE engine then runs serial on one core while the rest of "
        "the machine the fit just held sits idle (review 6.11.3)"
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
