#!/usr/bin/env bash
#
# The pre-push hook's entry point (wired in .pre-commit-config.yaml).
#
# WHY THIS EXISTS, and why the obvious `poetry run pytest` is wrong here.
#
# The hook used to be `entry: poetry run pytest`, which cannot work from a git
# worktree -- and every batch of work since batch 8 is developed in one, so
# that was on the path of every push. Two separate failures, one of which is
# silent:
#
#   (a) LOUD.  Poetry names a project's virtualenv from a hash of the PROJECT
#       PATH.  A worktree is a different path, so `poetry run` there does not
#       find the populated venv -- it CREATES a fresh empty one and dies on
#       ModuleNotFoundError: pytest, leaving an orphan venv behind (these have
#       reached tens of GB).
#
#   (b) SILENT, and the one that matters.  Even once pytest runs, the main
#       venv's editable install is a plain path entry (`exozippy.pth`)
#       pointing at the SHARED checkout's src.  So the suite imports the OTHER
#       tree's code and passes -- a green hook that proves nothing about what
#       you are pushing.
#
# Setting PYTHONPATH fixes (b) because a PYTHONPATH entry precedes site
# processing in sys.path, so it wins over the .pth (verified: with a fake
# package on PYTHONPATH, `import exozippy` resolves to the fake). This holds
# because the editable install is a path entry and NOT an `__editable__`
# meta-path finder -- a finder would be consulted before sys.path and would
# beat PYTHONPATH. If a future `poetry install` starts emitting a finder
# instead, this script stops being sufficient and the hook must switch to
# installing into the tree it is testing.
#
# The design rule: resolve the interpreter and the source root OURSELVES, from
# git, so the hook tests the tree it was launched from BY CONSTRUCTION rather
# than by the caller remembering to export something. The two rejected
# alternatives are recorded in docs/testing.md.
#
# NOTE: the hook tests the WORKING TREE, not the commits being pushed. That is
# unchanged by this script and is documented in docs/testing.md.
#
# Usage: normally invoked by the hook with no arguments. Any arguments are
# forwarded to pytest. Set EXOZIPPY_PREPUSH_DRYRUN=1 to print what it resolved
# and exit without running the suite (this is what tests/test_pre_push_hook.py
# exercises). Set EXOZIPPY_VENV_PYTHON to name the interpreter explicitly and
# skip the poetry lookup.

set -euo pipefail

# The tree being pushed from -- and therefore the src/ the suite must import.
root="$(git rev-parse --show-toplevel)"

# The main checkout, which is where the populated poetry venv lives. In the
# main tree `--git-common-dir` is the relative ".git"; in a worktree it is an
# absolute path to the main tree's .git. Resolving relative-to-root handles
# both, and in the main tree it lands back on root itself.
common="$(git rev-parse --git-common-dir)"
case "$common" in
    /*) ;;
    *) common="$root/$common" ;;
esac
main="$(dirname "$(cd "$common" && pwd)")"

# Resolve the interpreter. EXOZIPPY_VENV_PYTHON is the escape hatch for an
# environment poetry cannot describe (conda, a hand-built venv, CI).
if [ -n "${EXOZIPPY_VENV_PYTHON:-}" ]; then
    py="$EXOZIPPY_VENV_PYTHON"
    py_source="EXOZIPPY_VENV_PYTHON"
else
    # Ask poetry from the MAIN tree, never from here: asked from a worktree it
    # answers with the empty venv it would create -- failure (a) above.
    py="$(cd "$main" && poetry env info --executable 2>/dev/null || true)"
    py_source="poetry env info --executable (from $main)"
fi

if [ -z "$py" ] || [ ! -x "$py" ]; then
    echo "pre-push: cannot find the project interpreter." >&2
    echo "  tried: $py_source" >&2
    echo "  Run 'poetry install' in $main, or set EXOZIPPY_VENV_PYTHON to an" >&2
    echo "  interpreter that has this project's dependencies." >&2
    exit 1
fi

# Fail here rather than inside pytest, so the message names the cause. This is
# the exact symptom failure (a) produced, and it deserves better than a
# ModuleNotFoundError traceback.
if ! "$py" -c "import pytest" >/dev/null 2>&1; then
    echo "pre-push: $py has no pytest installed." >&2
    echo "  resolved via: $py_source" >&2
    echo "  Run 'poetry install' in $main." >&2
    exit 1
fi

# REPLACE rather than prepend. The suite must import exactly two things: this
# tree's src, and the venv's dependencies. A third source of modules is a
# reproducibility hazard, and an inherited PYTHONPATH is usually junk -- the
# development box's real one carries an EMPTY entry (which puts the current
# directory on sys.path) and five copies of an unrelated conda lib dir. What
# was discarded is printed rather than dropped silently.
discarded="${PYTHONPATH:-}"
export PYTHONPATH="$root/src"

# Print the resolution. The bug this script fixes was a hook that passed while
# testing the wrong code, so what it tested must be visible rather than
# inferred.
echo "pre-push: testing tree   $root"
echo "pre-push: interpreter    $py"
echo "pre-push: PYTHONPATH     $PYTHONPATH"
if [ -n "$discarded" ]; then
    echo "pre-push: discarded inherited PYTHONPATH ($discarded)"
fi

if [ -n "${EXOZIPPY_PREPUSH_DRYRUN:-}" ]; then
    echo "pre-push: dry run, not running the suite"
    exit 0
fi

cd "$root"
exec "$py" -m pytest "$@"
