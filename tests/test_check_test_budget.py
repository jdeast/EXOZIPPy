"""Tests for the per-test time budget check (scripts/check_test_budget.py).

The script reads the `--durations=0` transcript CI already writes and WARNS;
it must never fail a job over a timing, and it must never report "within
budget" for a transcript it could not read -- that would be a silent pass
for a step that did not run.
"""

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# By path, like tests/test_pytest_shard.py: scripts/ is not a package.
_NAME = "_check_test_budget"
if _NAME in sys.modules:
    budget = sys.modules[_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(
        _NAME, _REPO_ROOT / "scripts" / "check_test_budget.py"
    )
    budget = importlib.util.module_from_spec(_SPEC)
    sys.modules[_NAME] = budget
    _SPEC.loader.exec_module(budget)

_TRANSCRIPT = """\
============================= slowest durations ==============================
1328.64s setup    tests/test_slow_fixture.py::test_a
90.00s call     tests/test_slow_call.py::test_b[case-1]
9.99s call     tests/test_slow_call.py::test_c
55.00s call     tests/test_wide.py::test_d
50.00s call     tests/test_wide.py::test_e
0.01s teardown tests/test_slow_fixture.py::test_a
3.00s call     tests/test_fast.py::test_f
=========================== 6 passed in 1900.00s ============================
"""


def test_phases_and_files_over_budget_are_listed_slowest_first(
    tmp_path, capsys
):
    """
    Given a transcript with one slow fixture, one slow call, a fast test in
    the slow call's file, and a file whose tests are each under the phase
    budget but sum past the file budget,
    When the check runs with a 100 s file budget and --annotate,
    Then exactly the slow fixture and the slow call are phase offenders, the
    fixture's file and the wide file are file offenders, each is annotated,
    and the exit status is 0.
    """
    path = tmp_path / "durations.txt"
    path.write_text(_TRANSCRIPT)

    rc = budget.main([str(path), "--annotate", "--file-budget", "100"])
    out = capsys.readouterr().out

    assert rc == 0
    warnings = [ln for ln in out.splitlines() if ln.startswith("::warning")]
    assert any("test_slow_fixture.py costs 1329" in w for w in warnings)
    assert any("test_wide.py costs 105" in w for w in warnings)
    assert any("test_a setup took 1329 s" in w for w in warnings)
    assert any("test_b[case-1] call took 90 s" in w for w in warnings)
    # Under budget, so neither annotated nor listed.
    assert "test_c" not in out
    assert "test_fast.py" not in out
    assert len(warnings) == 4
    # The summary lines name the offenders slowest first.
    listed = [
        ln
        for ln in out.splitlines()
        if ln.startswith(("FILE", "setup", "call"))
    ]
    assert listed[0].startswith("FILE 1328.7 s")


def test_without_annotate_nothing_is_raised_as_a_warning(tmp_path, capsys):
    path = tmp_path / "durations.txt"
    path.write_text(_TRANSCRIPT)
    assert budget.main([str(path)]) == 0
    assert "::warning" not in capsys.readouterr().out


def test_annotations_are_capped_with_a_pointer_to_the_rest(tmp_path, capsys):
    """GitHub renders at most 10 warnings per step, so past the cap the last
    annotation says how many more the job summary carries."""
    lines = [
        f"{100 + i}.00s call     tests/test_x.py::test_{i}" for i in range(15)
    ]
    path = tmp_path / "durations.txt"
    path.write_text("\n".join(lines) + "\n")

    budget.main([str(path), "--annotate", "--file-budget", "1e9"])
    warnings = [
        ln
        for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("::warning")
    ]
    assert len(warnings) == budget._MAX_ANNOTATIONS + 1
    assert "and 6 more" in warnings[-1]


def test_a_transcript_with_no_duration_lines_is_an_error(tmp_path):
    """An empty transcript means the suite step did not run (or ran without
    --durations), which must not read as 'every test within budget'."""
    path = tmp_path / "durations.txt"
    path.write_text("no tests ran\n")
    assert budget.main([str(path)]) == 2
    assert budget.main([str(tmp_path / "missing.txt")]) == 2


def test_the_job_summary_is_written_under_actions(tmp_path, monkeypatch):
    path = tmp_path / "durations.txt"
    path.write_text(_TRANSCRIPT)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    budget.main([str(path), "--title", "Budget, shard 3/4"])

    text = summary.read_text()
    assert text.startswith("### Budget, shard 3/4")
    assert "`tests/test_slow_fixture.py::test_a`" in text


def test_the_line_pattern_matches_gen_durations():
    """Both scripts read the same transcript; one regex drifting from the
    other would make the budget check and the shard weights disagree about
    what a duration line is."""
    spec = importlib.util.spec_from_file_location(
        "_gen_durations_for_budget",
        _REPO_ROOT / "scripts" / "gen_durations.py",
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert budget._LINE.pattern == gen._LINE.pattern
