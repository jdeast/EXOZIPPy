"""Wrap-up steps must not be able to kill a finished fit (review 2.3.1).

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
