"""Partition the test files into shards, for splitting the suite across CI jobs.

Why shard at all
----------------
The suite's cost is ~12000 worker-seconds on a CI runner (2026-09-14) and it
is a LONG TAIL, not a few hot spots: measured 2026-08-25 the top 30 of 202
files were 67% of it and the worst single file 5.0%, and `call` (test bodies)
was 84.7% of the total against `setup` (shared `System` builds in fixtures)
at 15.3%, so there is no structural cut that buys much -- and where several
assertions CAN share one fit, a module-scoped fixture already does it. What
is left is parallelism, and one runner's core count caps how much of it a
single job can use.

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
finish faster than its slowest single file's SERIAL time, and a job's wall
clock is well predicted (within 4% on four master runs, 2026-09-14) by

    wall = ~100 s fixed + max(shard worker-seconds / workers, slowest file)

Once the slowest file exceeds the per-worker share, adding shards moves
NOTHING for the shard that carries it: on 2026-09-14 one 873 s file
(test_mulens_acceptance.py) held the worst ubuntu job at 16.2 min at 4, 5, 6
and 8 shards alike. The lever there is splitting slow FILES (that one became
test_mulens_acceptance_a.py / _b.py), and adding jobs also has a hard
ceiling: the free plan runs 20 concurrent jobs and 5 macOS, and a push
already uses most of both. See the comment above ``shard:`` in
.github/workflows/tests.yml for the current count.

Why the file is the unit
------------------------
`--dist loadfile` already pins a file to one worker so module-scoped fixtures
are shared rather than rebuilt. Splitting at any finer grain would break that,
and splitting at a coarser one is what this is.

Keeping the weights current
---------------------------
``tests/durations.json`` is regenerated from CI artifacts by
scripts/gen_durations.py, and the loop is closed by
.github/workflows/refresh-durations.yml, which does that weekly and proposes
the result when the balance has moved. This script is the other half of the
loop: ``--verify`` reports how stale the weights are on every CI job (stderr,
the job's step summary, and a ``::warning::`` annotation when files are
absent), and ``--balance-json`` gives the refresh workflow the numbers it
decides on -- the predicted worst shard with the current packing against the
packing the stale file would have produced.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
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

# Past this many days without a re-measurement the weights are flagged as
# stale even when every file is present. The weekly refresh workflow only
# proposes a new file when the balance has MOVED, so a healthy loop can
# legitimately leave a static suite's file untouched for a while; this is
# the backstop for the loop itself having stopped, not a nag about the
# calendar. Generous on purpose.
_STALE_AFTER_DAYS = 120


def load_payload(tests_dir: Path, path: Path | None = None) -> dict:
    """The whole durations JSON, or {} if absent/unusable.

    ``path`` overrides the default location, which is how the refresh
    workflow prices a candidate file that is not yet in tests/.
    """
    target = path if path is not None else tests_dir / _DURATIONS_FILE
    try:
        payload = json.loads(target.read_text())
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def durations_from_payload(payload: dict) -> dict[str, float]:
    durations = payload.get("durations")
    if not isinstance(durations, dict):
        return {}
    return {
        str(k): float(v)
        for k, v in durations.items()
        if isinstance(v, (int, float))
    }


def load_durations(
    tests_dir: Path, path: Path | None = None
) -> dict[str, float]:
    """Per-file weights from tests/durations.json, or {} if absent/unusable.

    Returning {} rather than raising is deliberate: this file is a weighting
    hint. A missing, truncated or hand-mangled one must degrade the BALANCE of
    the split, never its correctness, so the caller falls back to round-robin.
    """
    return durations_from_payload(load_payload(tests_dir, path))


def measured_age_days(
    payload: dict, today: datetime.date | None = None
) -> int | None:
    """Days since the file's ``_measured_on`` stamp, or None if it has none.

    The stamp is the TRANSCRIPT's date (gen_durations.py takes it from the
    artifact's mtime), so this is the age of the measurement and not of the
    commit that recorded it -- the only age that says anything about
    staleness.
    """
    stamp = payload.get("_measured_on")
    if not isinstance(stamp, str):
        return None
    try:
        measured = datetime.date.fromisoformat(stamp)
    except ValueError:
        return None
    today = today or datetime.date.today()
    return (today - measured).days


def weigh(files: list[str], durations: dict[str, float]) -> dict[str, float]:
    """Cost per file, charging the MEDIAN to anything unrecorded.

    The median, not zero and not the mean. Zero would make every newly added
    file free, so the packer would pile all of them into one shard -- and new
    files are exactly the ones nobody has measured yet. The mean is dragged up
    by the long tail, so it would overcharge the typical new file.
    """
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


def balance(files: list[str], durations: dict[str, float], total: int) -> dict:
    """Pack ``files`` ``total`` ways by ``durations`` and describe the result.

    The dictionary is what ``--balance-json`` prints and what the refresh
    workflow compares. ``worst_shard_seconds`` is the heaviest shard's summed
    worker-seconds; with ``--dist loadfile`` the other number a job's wall
    clock can hinge on is ``slowest_file_seconds`` (see the module docstring),
    so both are reported.
    """
    weights = weigh(files, durations)
    groups = pack(files, total, weights)
    loads = [sum(weights[f] for f in g) for g in groups]
    unknown = sorted(
        Path(f).name for f in files if Path(f).name not in durations
    )
    # Ties on weight break to the first name, matching pack()'s ordering.
    slowest = min(files, key=lambda f: (-weights[f], f)) if files else ""
    return {
        "shards": total,
        "files": len(files),
        "unknown": len(unknown),
        "unknown_files": unknown,
        "shard_seconds": [round(x, 1) for x in loads],
        "worst_shard_seconds": round(max(loads), 1) if loads else 0.0,
        "ideal_shard_seconds": round(sum(loads) / total, 1) if loads else 0.0,
        "slowest_file": Path(slowest).name if slowest else "",
        "slowest_file_seconds": round(weights[slowest], 1) if slowest else 0.0,
        "groups": groups,
    }


def price_packing(
    groups: list[list[str]], durations: dict[str, float]
) -> float:
    """Heaviest shard of an EXISTING packing, priced by ``durations``.

    This is how a stale weighting is scored honestly: pack by the OLD weights
    (what CI actually ran), then charge each shard what the NEW measurement
    says those files cost. The difference against ``balance()`` under the new
    weights is the wall clock the stale file was wasting.
    """
    files = [f for g in groups for f in g]
    weights = weigh(files, durations)
    return round(max(sum(weights[f] for f in g) for g in groups), 1)


def staleness_lines(
    payload: dict,
    report: dict,
    age_days: int | None,
) -> list[str]:
    """Human-readable staleness summary, one fact per line, no markup."""
    lines = [
        f"{report['files']} test files; {report['unknown']} absent from "
        f"{_DURATIONS_FILE} (charged the median)",
        f"measured on {payload.get('_measured_on', 'unknown')}"
        + (f", {age_days} days ago" if age_days is not None else "")
        + f"; source: {payload.get('_generated_from', 'unknown')}",
        f"predicted worst shard {report['worst_shard_seconds']:.0f} "
        f"worker-s against an ideal {report['ideal_shard_seconds']:.0f}; "
        f"slowest file {report['slowest_file']} at "
        f"{report['slowest_file_seconds']:.0f} s serial",
    ]
    if report["unknown_files"]:
        shown = report["unknown_files"][:12]
        more = report["unknown"] - len(shown)
        lines.append(
            "absent: "
            + ", ".join(shown)
            + (f", and {more} more" if more > 0 else "")
        )
    return lines


def write_step_summary(title: str, lines: list[str]) -> bool:
    """Append a short section to the GitHub job summary, if there is one.

    ``$GITHUB_STEP_SUMMARY`` is set on every Actions step; writing to it puts
    the text on the job's summary page, which is what people read when a run
    looks slow. Off CI the variable is unset and this is a no-op. Returns
    whether anything was written.
    """
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return False
    try:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(f"### {title}\n\n")
            for line in lines:
                fh.write(f"- {line}\n")
            fh.write("\n")
    except OSError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print the test files belonging to one shard, space-separated, "
            "for `pytest $(scripts/pytest_shard.py --shard 1 --of 2)`."
        )
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="1-based shard index (required unless --balance-json)",
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
        "--durations-file",
        default=None,
        help=(
            "weights to pack by, instead of tests/durations.json. The "
            "refresh workflow uses it to price a regenerated file before "
            "it is committed."
        ),
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
            "-- and report the totals and the weights' staleness on stderr "
            "(and on the job summary when run under GitHub Actions)"
        ),
    )
    parser.add_argument(
        "--balance-json",
        action="store_true",
        help=(
            "instead of a file list, print one JSON object describing the "
            "packing: file counts, files absent from the weights, predicted "
            "worst and ideal shard worker-seconds, the slowest file, and the "
            "measurement's date and age"
        ),
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        metavar="OLD_DURATIONS",
        help=(
            "with --balance-json: also pack by this OLDER weights file and "
            "price that packing with the current weights, reported as "
            "stale_packing_worst_shard_seconds -- the shard CI would have "
            "run under the old file, costed honestly"
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

    durations_path = Path(args.durations_file) if args.durations_file else None
    payload = (
        {} if args.round_robin else load_payload(tests_dir, durations_path)
    )
    durations = durations_from_payload(payload)

    if args.balance_json:
        if not durations:
            print("no usable durations file to balance", file=sys.stderr)
            return 1
        report = balance(files, durations, args.of)
        report.pop("groups")
        report["measured_on"] = payload.get("_measured_on")
        report["generated_from"] = payload.get("_generated_from")
        report["age_days"] = measured_age_days(payload)
        if args.compare_to:
            old = load_durations(tests_dir, Path(args.compare_to))
            if not old:
                print(
                    f"{args.compare_to} is not a usable durations file",
                    file=sys.stderr,
                )
                return 1
            stale_groups = pack(files, args.of, weigh(files, old))
            report["stale_packing_worst_shard_seconds"] = price_packing(
                stale_groups, durations
            )
            report["stale_unknown"] = sum(
                1 for f in files if Path(f).name not in old
            )
        print(json.dumps(report, indent=2))
        return 0

    if args.shard is None:
        parser.error("--shard is required unless --balance-json is given")

    if durations:
        report = balance(files, durations, args.of)
        groups = report["groups"]
        ideal = report["ideal_shard_seconds"] or 1.0
        strategy = (
            f"duration-aware, worst shard "
            f"{report['worst_shard_seconds'] / ideal:.2f}x of ideal"
        )
        age_days = measured_age_days(payload)
        lines = staleness_lines(payload, report, age_days)
        unknown = report["unknown"]
        share = 100 * unknown / len(files)
        stale_age = age_days is not None and age_days > _STALE_AFTER_DAYS
        if unknown or stale_age:
            reasons = []
            if unknown:
                reasons.append(
                    f"{unknown} of {len(files)} test files ({share:.0f}%) are "
                    f"absent from {_DURATIONS_FILE} and were charged the "
                    f"median cost"
                )
            if stale_age:
                reasons.append(
                    f"the weights were measured {age_days} days ago"
                )
            note = (
                "; ".join(reasons)
                + ". The weekly refresh-durations workflow regenerates the "
                "file from CI artifacts; run it now with `gh workflow run "
                "refresh-durations.yml`, or see docs/testing.md"
            )
            print(f"WARNING: {note}", file=sys.stderr)
            # A GitHub Actions annotation, so staleness surfaces on the pull
            # request instead of only in a log nobody opens. Warn rather
            # than fail: a stale weighting costs balance, never coverage,
            # and a red suite would be the wrong severity for "please
            # re-measure". From shard 1 only, so a run carries one
            # annotation per os+python leg rather than one per job.
            # On STDERR, like every other diagnostic here: the workflow
            # captures this script's stdout wholesale as the pytest file
            # list (`files=$(... --verify)`), and the first run of this
            # step handed pytest the annotation text as test paths --
            # shard 1 of every leg collected nothing and exited 5. The
            # runner processes workflow commands on either stream.
            if args.shard == 1:
                print(
                    f"::warning title=Stale shard durations::{note}",
                    file=sys.stderr,
                )
        # The fuller report only under --verify, which is how CI calls
        # this; a developer asking for a file list gets the file list.
        if args.verify:
            for line in lines:
                print(f"durations: {line}", file=sys.stderr)
            write_step_summary(
                f"Shard split {args.shard}/{args.of}: {strategy}", lines
            )
    else:
        groups = [files[i :: args.of] for i in range(args.of)]
        strategy = "round-robin (no usable durations file)"
        if args.verify:
            write_step_summary(
                f"Shard split {args.shard}/{args.of}: {strategy}",
                [
                    f"{len(files)} test files; no usable {_DURATIONS_FILE}, "
                    f"so the split is round-robin (about a third worse "
                    f"balanced at 4 shards)"
                ],
            )

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
