"""`measure_scales: false` must not skip the whitening RESTORE (review 2.3.3).

The restore-vs-remeasure split lives inside ``prepare_whitening`` and is well
tested there (tests/test_whitening.py).  What was wrong was the gate in
run.py: the whole call sat inside ``if measure_scales:``, so a trace sampled
under MEASURED scales and reloaded with ``measure_scales: false`` decoded its
raw draws under PRELIMINARY scales -- silently, with no message -- defeating
the documented "on the reuse path the whitening is a property of the draws"
policy.

``measure_scales`` gates the MEASUREMENT, which is a statement about a run
that is about to sample.  The reuse path's honest answer when the trace
really was sampled without measurement is already built in: there is no
whitening file, so the restore warns and keeps the preliminary scales, which
for that trace ARE the sampled coordinates.
"""

import ast
import inspect
from pathlib import Path

from exozippy import run as run_module


def _if_guarding(call_name, func_name):
    """The innermost ``ast.If`` whose body contains a call to ``call_name``."""
    tree = ast.parse(Path(inspect.getfile(run_module)).read_text())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == func_name
    )

    found = []

    def walk(node, enclosing_if):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name == call_name:
                found.append(enclosing_if)
        for child in ast.iter_child_nodes(node):
            walk(child, node if isinstance(node, ast.If) else enclosing_if)

    walk(func, None)
    return found


def test_the_whitening_call_is_not_gated_on_measure_scales_alone():
    """
    Given _run_fit's whitening step,
    When its guarding condition is read,
    Then reusing_trace appears in it -- the restore is not skippable by
      turning the measurement off.
    """
    guards = _if_guarding("prepare_whitening", "_run_fit")

    assert guards, "prepare_whitening is no longer called from _run_fit"
    for guard in guards:
        assert guard is not None, "the whitening step lost its guard entirely"
        names = {
            node.id
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Name)
        }
        assert "reusing_trace" in names, (
            "prepare_whitening is gated without consulting reusing_trace: a "
            "trace sampled under measured scales would decode under "
            "preliminary ones whenever measure_scales is false"
        )


def test_measure_scales_still_gates_the_fresh_path():
    """
    Given the same condition,
    When measure_scales is read out of it,
    Then it is still there -- `measure_scales: false` must still skip the
      probe on a run that is about to sample, which is the whole key.
    """
    guards = _if_guarding("prepare_whitening", "_run_fit")

    for guard in guards:
        names = {
            node.id
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Name)
        }
        assert "measure_scales" in names


def test_the_guard_is_a_disjunction_not_a_conjunction():
    """
    Given `reusing_trace or measure_scales`,
    When each combination is evaluated,
    Then the reuse path runs whatever measure_scales says, and a fresh run
      with the key off does not.

    Evaluating the real source expression rather than restating it, so a
    later `and` cannot pass by naming both variables.
    """
    guards = _if_guarding("prepare_whitening", "_run_fit")
    expr = ast.Expression(body=guards[0].test)
    ast.fix_missing_locations(expr)
    code = compile(expr, "<guard>", "eval")

    def guard(reusing_trace, measure_scales):
        return bool(
            eval(  # noqa: S307 -- the source under test, not user input
                code,
                {},
                {
                    "reusing_trace": reusing_trace,
                    "measure_scales": measure_scales,
                },
            )
        )

    assert guard(reusing_trace=True, measure_scales=False) is True
    assert guard(reusing_trace=True, measure_scales=True) is True
    assert guard(reusing_trace=False, measure_scales=True) is True
    assert guard(reusing_trace=False, measure_scales=False) is False
