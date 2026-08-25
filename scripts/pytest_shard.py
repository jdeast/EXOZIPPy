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

How the split is chosen
-----------------------
Two strategies, and which one runs depends only on whether
``tests/durations.json`` is present.

**Duration-aware (preferred).** Longest-processing-time-first packing over the
recorded per-file worker-seconds. Measured against the real durations it hits
**1.00x** of a perfectly balanced split at every shard count.

**Round-robin over the sorted file list (fallback).** No state at all, and it
measures well *at two shards*: **1.04x** of ideal. But it degrades as the long
tail stops averaging out -- **1.37x at N=3**, **1.33x at N=4**, **1.48x at
N=6** -- which is precisely why the durations file exists. At the 4 shards CI
now runs, round-robin would waste about a third of the gain.

The durations file is a **weighting hint, never a correctness input**: an
unknown file is charged the median cost, so a stale file rebalances badly and
never drops or duplicates anything. Staleness is reported on every run rather
than left to be discovered.

What sharding cannot fix
------------------------
``--dist loadfile`` pins a whole file to one worker, so a shard can never
finish faster than its slowest single file's SERIAL time. Measured on ubuntu,
that floor is ~262 s (``test_rm_ltt.py``) against a ~225 s fixed per-job cost,
so past **4 shards** the binding constraint stops being the spread and becomes
one file: 6 shards buys about a minute for 8 more jobs. Below ~8 minutes the
lever is splitting slow FILES, not adding machines.

Why the file is the unit
------------------------
`--dist loadfile` already pins a file to one worker so module-scoped fixtures
are shared rather than rebuilt. Splitting at any finer grain would break that,
and splitting at a coarser one is what this is.
"""

from __future__ import annotations

import argparse
import json
import statistics
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


# Beside the tests it weights, so `pytest --collect-only` never sees it and a
# checkout always carries it with the files it describes.
_DURATIONS_FILE = "durations.json"

# Charged to a file the durations record does not mention. The MEDIAN, not zero
# and not the mean: zero would make every new file free and pile them all into
# one shard, and the mean is dragged up by the long tail so it would overcharge
# the typical new file.
_UNKNOWN_COST = "median"


def load_durations(tests_dir: Path) -> dict[str, float]:
    """Per-file weights from tests/durations.json, or {} if absent/unusable.

    Returning {} rather than raising is deliberate: this file is a weighting
    hint. A missing, truncated or hand-mangled one must degrade the BALANCE of
    the split, never its correctness, so the caller falls back to round-robin.
    """
    path = tests_dir / _DURATIONS_FILE
    try:
        payload = json.loads(path.read_text())
        durations = payload["durations"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if not isinstance(durations, dict):
        return {}
    return {
        str(k): float(v)
        for k, v in durations.items()
        if isinstance(v, (int, float))
    }


def weigh(files: list[str], durations: dict[str, float]) -> dict[str, float]:
    """Cost per file, charging the median to anything unrecorded."""
    known = [
        durations[Path(f).name] for f in files if Path(f).name in durations
    ]
    fallback = statistics.median(known) if known else 1.0
    return {f: durations.get(Path(f).name, fallback) for f in files}


def pack(
    files: list[str], total: int, weights: dict[str, float]
) -> list[list[str]]:
    """Longest-processing-time-first bin packing into ``total`` shards.

    Heaviest file first into whichever shard is currently lightest. Ties break
    on the file name so the partition is identical on every machine -- two
    shards disagreeing about it would double-run some files and skip others.
    """
    groups: list[list[str]] = [[] for _ in range(total)]
    load = [0.0] * total
    for f in sorted(files, key=lambda f: (-weights[f], f)):
        i = min(range(total), key=lambda k: (load[k], k))
        groups[i].append(f)
        load[i] += weights[f]
    return [sorted(g) for g in groups]


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
        "--round-robin",
        action="store_true",
        help=(
            "ignore tests/durations.json and use the stateless round-robin "
            "split. For reproducing a split from before the durations file "
            "existed, or bisecting a balance problem to it."
        ),
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

    durations = {} if args.round_robin else load_durations(tests_dir)
    if durations:
        weights = weigh(files, durations)
        groups = pack(files, args.of, weights)
        unknown = [f for f in files if Path(f).name not in durations]
        loads = [sum(weights[f] for f in g) for g in groups]
        ideal = sum(loads) / args.of
        strategy = (
            f"duration-aware, worst shard {max(loads) / ideal:.2f}x of ideal"
        )
        if unknown:
            share = 100 * len(unknown) / len(files)
            note = (
                f"{len(unknown)} of {len(files)} test files ({share:.0f}%) are "
                f"absent from {_DURATIONS_FILE} and were charged the median "
                f"cost; regenerate with scripts/gen_durations.py"
            )
            print(f"WARNING: {note}", file=sys.stderr)
            # A GitHub Actions annotation, so growing staleness surfaces on the
            # pull request instead of only in a log nobody opens. Warn rather
            # than fail: a stale weighting costs balance, never coverage, and a
            # red suite would be the wrong severity for "please re-measure".
            if share >= 20:
                print(f"::warning title=Stale shard durations::{note}")
    else:
        groups = [files[i :: args.of] for i in range(args.of)]
        strategy = "round-robin (no usable durations file)"

    if args.verify:
        union = [f for g in groups for f in g]
        if sorted(union) != files or len(union) != len(files):
            print(
                f"shards do not partition {len(files)} files: "
                f"{len(union)} assigned, {len(set(union))} distinct",
                file=sys.stderr,
            )
            return 1
        print(
            f"{len(files)} test files partition cleanly into {args.of} "
            f"shards; {strategy}",
            file=sys.stderr,
        )

    if not 1 <= args.shard <= args.of:
        print(f"shard {args.shard} is not in 1..{args.of}", file=sys.stderr)
        return 1
    mine = groups[args.shard - 1]
    print(
        f"shard {args.shard}/{args.of}: {len(mine)} of {len(files)} test files",
        file=sys.stderr,
    )
    print(" ".join(mine))
    return 0


if __name__ == "__main__":
    sys.exit(main())
