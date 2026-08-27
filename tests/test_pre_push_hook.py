"""Tests for the pre-push hook's entry point (scripts/pre_push_suite.sh).

The hook's old entry, `poetry run pytest`, had two failure modes from a git
worktree, and only one of them was loud. The loud one (poetry builds a fresh
empty venv, pytest is absent) announces itself. The other does not: the main
venv's editable install is a path entry pointing at the SHARED checkout's src,
so the suite imports the OTHER tree's code and reports green.

So the property worth pinning is not that the script runs -- it is that the
src it puts on the import path belongs to the tree the push was launched from.
A test that only checked "the script exits 0" would have passed against the
bug it exists to prevent.

The script is exercised through EXOZIPPY_PREPUSH_DRYRUN, which resolves
everything and prints it without spending the suite's runtime on itself.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pre_push_suite.sh"


def _git(cwd, *args):
    """Run a git command, failing the test loudly on a nonzero exit."""
    subprocess.run(
        ("git",) + args, cwd=str(cwd), check=True, capture_output=True
    )


def _run_script(cwd):
    """Run the hook script in dry-run mode from cwd; return its stdout.

    EXOZIPPY_VENV_PYTHON short-circuits the poetry lookup: this test is about
    the source root the script chooses, and the running interpreter is a
    perfectly good stand-in for the project venv (it has pytest, which is how
    this test is executing).
    """
    env = dict(os.environ)
    env["EXOZIPPY_PREPUSH_DRYRUN"] = "1"
    env["EXOZIPPY_VENV_PYTHON"] = sys.executable
    # A deliberately hostile inherited value: the empty entry is what puts the
    # current directory on sys.path, and it is present in real shells here.
    env["PYTHONPATH"] = ":/nonexistent/inherited"
    done = subprocess.run(
        [str(SCRIPT)], cwd=str(cwd), env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _pythonpath_line(stdout):
    """Extract the single PYTHONPATH value the script exported."""
    for line in stdout.splitlines():
        if line.startswith("pre-push: PYTHONPATH"):
            return line.split(None, 2)[2].strip()
    pytest.fail(f"script printed no PYTHONPATH line:\n{stdout}")


@pytest.fixture
def repo_with_worktree(tmp_path):
    """A throwaway repo with a src/ dir and one linked worktree.

    Built from scratch rather than against the real checkout so the test never
    touches the project's own worktree list.
    """
    main = tmp_path / "main"
    (main / "src").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "master", str(main))
    (main / "src" / "marker.txt").write_text("main tree\n")
    _git(main, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(
        main,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-qm",
        "initial",
    )

    linked = tmp_path / "linked"
    _git(main, "worktree", "add", "-q", "-b", "topic", str(linked))
    (linked / "src").mkdir(exist_ok=True)
    return main, linked


def test_a_worktree_gets_its_own_src_on_the_import_path(repo_with_worktree):
    """Given a push launched from a linked git worktree,
    When the hook resolves the suite's import path,
    Then it is the WORKTREE's src, not the main checkout's.

    This is the silent half of the bug: the old entry left the main venv's
    editable path entry in charge, so the suite tested the shared checkout and
    reported green on code that was never run."""
    # Arrange
    main, linked = repo_with_worktree

    # Act
    pythonpath = _pythonpath_line(_run_script(linked))

    # Assert
    assert pythonpath == str(linked / "src")
    assert str(main / "src") not in pythonpath


def test_the_main_tree_still_resolves_to_itself(repo_with_worktree):
    """Given a push launched from the main checkout,
    When the hook resolves the suite's import path,
    Then it is that checkout's src.

    Worth its own case because git reports --git-common-dir as the RELATIVE
    ".git" here and as an absolute path from a worktree; a fix that handled
    only the absolute form would break the ordinary push."""
    # Arrange
    main, _linked = repo_with_worktree

    # Act
    pythonpath = _pythonpath_line(_run_script(main))

    # Assert
    assert pythonpath == str(main / "src")


def test_an_inherited_pythonpath_is_discarded_and_reported(repo_with_worktree):
    """Given an inherited PYTHONPATH in the environment,
    When the hook runs,
    Then it is replaced, and the replacement is announced.

    The suite must import exactly this tree's src plus the venv, so a third
    module source is dropped -- but dropping it silently would be the same
    class of defect as the one being fixed, so the script says what it did."""
    # Arrange
    _main, linked = repo_with_worktree

    # Act
    stdout = _run_script(linked)

    # Assert
    assert _pythonpath_line(stdout) == str(linked / "src")
    assert "/nonexistent/inherited" not in _pythonpath_line(stdout)
    assert "discarded inherited PYTHONPATH" in stdout


def test_a_missing_interpreter_fails_with_an_actionable_message(tmp_path):
    """Given an interpreter that does not exist,
    When the hook runs,
    Then it exits nonzero naming both the lookup it tried and the fix.

    The old entry's failure was a bare ModuleNotFoundError traceback from
    inside pytest, which names neither."""
    # Arrange
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(tmp_path, "init", "-q", "-b", "master", str(repo))
    env = dict(os.environ)
    env["EXOZIPPY_VENV_PYTHON"] = str(tmp_path / "no_such_python")

    # Act
    done = subprocess.run(
        [str(SCRIPT)], cwd=str(repo), env=env, capture_output=True, text=True
    )

    # Assert
    assert done.returncode != 0
    assert "EXOZIPPY_VENV_PYTHON" in done.stderr
    assert "poetry install" in done.stderr


def test_the_hook_entry_points_at_this_script():
    """Given .pre-commit-config.yaml,
    When the pre-push hook's entry is read,
    Then it is this script and not `poetry run pytest`.

    Pins the wiring, not the script: the reasoning above is worthless if the
    config drifts back to the spelling that cannot work in a worktree."""
    # Arrange
    config = (SCRIPT.parents[1] / ".pre-commit-config.yaml").read_text()

    # Act / Assert
    assert "entry: scripts/pre_push_suite.sh" in config
    assert "entry: poetry run pytest" not in config
