"""Tests for the CI shard split (scripts/pytest_shard.py).

The failure mode this guards is specific and quiet: a split that drops a file
still reports green on every shard, so the coverage is gone and nothing says
so. Hence the properties pinned here are exhaustiveness and disjointness
against the REAL tests/ directory, not just against a fixture.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# By path, for the same reason the other scripts/ tests do it: scripts/ is not
# a package, and putting it on sys.path would make getdata.py / mkparam.py
# importable as top-level modules.
_NAME = "_pytest_shard"
if _NAME in sys.modules:
    shard_mod = sys.modules[_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(
        _NAME, _REPO_ROOT / "scripts" / "pytest_shard.py"
    )
    shard_mod = importlib.util.module_from_spec(_SPEC)
    sys.modules[_NAME] = shard_mod
    _SPEC.loader.exec_module(shard_mod)


@pytest.mark.parametrize("total", [1, 2, 3, 4])
def test_the_real_test_tree_partitions_exactly(total):
    """Given this repository's own tests/ directory,
    When it is split into shards,
    Then every file lands in exactly one shard.

    Against the real directory rather than a fixture, because the thing that
    breaks this is a real file appearing somewhere the globs do not look --
    a subpackage, or a `*_test.py`. A fixture would keep passing."""
    # Arrange
    files = shard_mod.discover_test_files(_REPO_ROOT / "tests")
    assert files, "no test files discovered at all"

    # Act
    shards = [shard_mod.shard(files, i, total) for i in range(1, total + 1)]

    # Assert -- exhaustive and disjoint.
    union = [f for s in shards for f in s]
    assert sorted(union) == files
    assert len(union) == len(set(union)), "a file landed in two shards"


def test_this_very_file_is_in_some_shard():
    """Given the shard split,
    When it runs over tests/,
    Then this file is in it.

    A canary for the glob: if someone narrows the patterns or makes the walk
    non-recursive, the file asserting it is covered stops being covered, and
    the assertion is what notices."""
    # Arrange / Act
    files = shard_mod.discover_test_files(_REPO_ROOT / "tests")

    # Assert
    assert Path(__file__).resolve().as_posix() in files


def test_both_of_pytests_default_patterns_are_collected(tmp_path):
    """Given a tests dir holding test_*.py and *_test.py and a non-test file,
    When the files are discovered,
    Then both test patterns are found and the other file is not.

    `*_test.py` matters even though the repo has none today: pytest's default
    `python_files` collects it, so a `test_*.py`-only glob would hand pytest a
    file list that silently omits it."""
    # Arrange
    (tmp_path / "test_alpha.py").touch()
    (tmp_path / "beta_test.py").touch()
    (tmp_path / "conftest.py").touch()
    (tmp_path / "helper.py").touch()

    # Act
    found = {Path(f).name for f in shard_mod.discover_test_files(tmp_path)}

    # Assert
    assert found == {"test_alpha.py", "beta_test.py"}


def test_a_test_file_in_a_subdirectory_is_not_lost(tmp_path):
    """Given a test file nested in a subpackage,
    When the files are discovered,
    Then it is included.

    tests/ is flat today. If it stops being flat, a non-recursive walk would
    drop the whole subtree out of every shard without failing anything."""
    # Arrange
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "test_nested.py").touch()
    (tmp_path / "test_top.py").touch()

    # Act
    found = {Path(f).name for f in shard_mod.discover_test_files(tmp_path)}

    # Assert
    assert found == {"test_top.py", "test_nested.py"}


def test_the_split_is_balanced_on_a_long_tail(tmp_path):
    """Given many files of widely varying cost,
    When they are round-robined,
    Then the shards come out close to even.

    This is the property that justifies round-robin over a duration-aware
    pack. Measured on the real suite: 2 shards land within 1.04x of ideal.
    Here the same shape is reproduced synthetically -- one 5% file and a long
    tail -- so the reasoning is pinned rather than just asserted in a comment."""
    # Arrange -- costs shaped like the measured suite: no file dominant.
    costs = {f"test_{i:03d}.py": 1.0 / (i + 1) for i in range(200)}
    files = sorted(costs)

    # Act
    shards = [shard_mod.shard(files, i, 2) for i in range(1, 3)]
    loads = [sum(costs[Path(f).name] for f in s) for s in shards]

    # Assert
    ideal = sum(costs.values()) / 2
    assert max(loads) / ideal < 1.15


@pytest.mark.parametrize("index,total", [(0, 2), (3, 2), (-1, 4)])
def test_an_out_of_range_shard_index_is_rejected(index, total):
    """Given a shard index outside 1..total,
    When a shard is requested,
    Then it raises rather than silently returning something.

    Shard indices come from a CI matrix, where an off-by-one is easy and would
    otherwise mean a shard that runs nothing (green, and no coverage) or two
    shards running the same files."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        shard_mod.shard(["test_a.py", "test_b.py"], index, total)
