"""Partition the test files into shards, for splitting the suite across CI jobs.

Why shard at all
----------------
The suite's cost is ~6700 worker-seconds and it is a LONG TAIL, not a few hot
spots: the top 30 of 202 files are 67% of it and the worst single file is 5.0%.
Measured 2026-08-25, `call` (test bodies) is 84.7% of that and `setup` (shared
`System` builds in fixtures) only 15.3%, so there is no structural cut that
buys much -- and where several assertions CAN share one fit, a module-scoped
fixture already does it. What is left is parallelism, and one runner's core
count caps how much of it a single job can use.

Why a round-robin over the sorted file list
-------------------------------------------
Because it needs no state and it measures well. Against the real per-file
durations, a 2-way round-robin of the alphabetical list lands within **1.04x**
of a perfectly balanced split (3491 vs 3224 worker-seconds). A duration-aware
greedy pack does reach 1.00x, but it needs a checked-in durations file that
goes stale silently, and 4% is not worth that.

The balance is a property of the long tail, not luck: with 202 files and no
file over 5%, alternating them cannot concentrate much. It degrades at higher
shard counts -- measured 1.37x at N=3 and 1.33x at N=4 -- so if this ever goes
past 2 shards, re-measure before assuming it is still fine.

Why the file is the unit
------------------------
`--dist loadfile` already pins a file to one worker so module-scoped fixtures
are shared rather than rebuilt. Splitting at any finer grain would break that,
and splitting at a coarser one is what this is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# pytest's own default `python_files`, which this repo does not override.
# BOTH patterns, deliberately: today every test file is `test_*.py`, but a
# `*_test.py` would be collected by pytest and silently dropped from every
# shard by a `test_*.py`-only glob. A shard split that loses coverage still
# reports green, which is the worst way for this to fail.
_PATTERNS = ("test_*.py", "*_test.py")


def discover_test_files(tests_dir: Path) -> list[str]:
    """Every file pytest would collect under ``tests_dir``, sorted.

    Recursive, for the same reason both patterns are matched: `tests/` has no
    subpackages today, and a future one must not fall out of the split.
    """
    found: set[Path] = set()
    for pattern in _PATTERNS:
        found.update(tests_dir.rglob(pattern))
    # Sorted by POSIX string so the partition is identical on every platform;
    # a shard that disagrees with its peers about the ordering would double-run
    # some files and skip others.
    return sorted(p.as_posix() for p in found)


def shard(files: list[str], index: int, total: int) -> list[str]:
    """The ``index``-of-``total`` slice, 1-based. Round-robin; see the module docstring."""
    if not 1 <= index <= total:
        raise ValueError(f"shard {index} is not in 1..{total}")
    return files[index - 1 :: total]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print the test files belonging to one shard, space-separated, "
            "for `pytest $(scripts/pytest_shard.py --shard 1 --of 2)`."
        )
    )
    parser.add_argument(
        "--shard", type=int, required=True, help="1-based shard index"
    )
    parser.add_argument(
        "--of", type=int, required=True, help="total number of shards"
    )
    parser.add_argument(
        "--tests-dir",
        default=None,
        help="defaults to tests/ beside this script's repo root",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "check that the shards partition the file list "
            "exactly -- every file in one shard, none in two "
            "-- and report the totals on stderr"
        ),
    )
    args = parser.parse_args(argv)

    tests_dir = (
        Path(args.tests_dir)
        if args.tests_dir
        else Path(__file__).resolve().parents[1] / "tests"
    )
    files = discover_test_files(tests_dir)
    if not files:
        print(f"no test files found under {tests_dir}", file=sys.stderr)
        return 1

    if args.verify:
        union: list[str] = []
        for i in range(1, args.of + 1):
            union.extend(shard(files, i, args.of))
        if sorted(union) != files or len(union) != len(files):
            print(
                f"shards do not partition {len(files)} files: "
                f"{len(union)} assigned, {len(set(union))} distinct",
                file=sys.stderr,
            )
            return 1
        print(
            f"{len(files)} test files partition cleanly into {args.of} shards",
            file=sys.stderr,
        )

    mine = shard(files, args.shard, args.of)
    print(
        f"shard {args.shard}/{args.of}: {len(mine)} of {len(files)} test files",
        file=sys.stderr,
    )
    print(" ".join(mine))
    return 0


if __name__ == "__main__":
    sys.exit(main())
