"""Build tests/durations.json from a ``pytest --durations`` transcript.

The file this writes balances the CI shard split (see
scripts/pytest_shard.py and docs/testing-cache.md).

MEASURE ON CI, NOT ON A WORKSTATION. That is not a style preference, it is the
lesson from the first sharded runs. Weights taken on a 36-core box at ``-n 6``
balanced the RECORDED sums perfectly -- all four shards within 1301 equal
ubuntu-seconds -- and still produced a 1.6x spread in real wall clock, because
the heavy files cost relatively more on a runner than on the workstation.
Shard 1 came in at 14:19 against shard 4's 8:38 with identical predicted
spread. The packing was right; the weights were measured on the wrong machine.

So the source of truth is a CI run, whose jobs upload their ``--durations``
transcripts as artifacts. Each job runs only ONE shard, so a full set needs
every shard of one os+python combination -- and this accepts several
transcripts for exactly that reason:

    gh run download <run-id> -p 'durations-ubuntu-latest-3.12-*' -D /tmp/dur
    poetry run python scripts/gen_durations.py /tmp/dur/*/durations.txt

Use the SLOWEST combination (ubuntu), not macOS and not a mixture: mixed
weights are worse than either platform's own, and ubuntu is both the slowest
and three of the four matrix legs.

A local run still works, and is fine for a rough refresh:

    poetry run pytest -q -n6 --dist loadfile --durations=0 --durations-min=0 \\
        > /tmp/durations.txt
    poetry run python scripts/gen_durations.py /tmp/durations.txt

Per FILE, not per test, because ``--dist loadfile`` schedules whole files: a
per-test breakdown would be more data carrying no more usable signal.

Everything stays behind ``main()`` and the CLI answers ``--help``, per the
convention tests/test_scripts_smoke.py enforces on everything in scripts/.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import sys
from pathlib import Path

# The line shape pytest emits for --durations, e.g.
#   12.34s call     tests/test_alpha.py::test_one[case]
_LINE = re.compile(r"^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+?)::(\S+)\s*$")

# Test ids naming an example directory that is not in the repository. A
# developer's untracked examples/<name>/ is collected by
# test_examples_prepare locally and never on CI, so counting its cost would
# skew the shard balance for everyone else. Matched against the parametrize id.
_LOCAL_ONLY = ("ob09020",)


def parse(*transcripts: str) -> tuple[dict[str, float], dict[str, float]]:
    """(per-file worker-seconds, per-file seconds excluded as local-only).

    Accepts several transcripts and SUMS them, which is what makes merging a
    CI run's per-shard artifacts work: each job reports only the files in its
    own shard, so the union across shards is one whole suite and no file
    appears twice. Feeding the same transcript in twice would double its
    files, so pass each shard exactly once.
    """
    per_file: collections.Counter[str] = collections.Counter()
    skipped: collections.Counter[str] = collections.Counter()
    for transcript in transcripts:
        for line in transcript.splitlines():
            m = _LINE.match(line.strip())
            if not m:
                continue
            if any(tag in m[4] for tag in _LOCAL_ONLY):
                skipped[m[3]] += float(m[1])
                continue
            per_file[m[3]] += float(m[1])
    return dict(per_file), dict(skipped)


def build(
    per_file: dict[str, float],
    skipped: dict[str, float],
    measured_on: str = "unknown",
    source: str = "unknown",
) -> dict:
    # Basenames, because the shard split matches on them: an absolute path
    # from whoever generated the file would be useless everywhere else.
    durations = {
        Path(f).name: round(v, 2) for f, v in sorted(per_file.items())
    }
    return {
        "_comment": (
            "Per-file worker-seconds, for balancing the CI shard split. "
            "Regenerate with scripts/gen_durations.py -- see "
            "docs/testing-cache.md. These are RELATIVE weights: absolute "
            "values depend on the machine and only their ratios matter. A "
            "file absent from this map is charged the median cost, so adding "
            "a test does not require updating it."
        ),
        "_generated_from": source,
        # From the transcript's mtime rather than "now", so regenerating from
        # an old transcript records when it was MEASURED and not when it was
        # converted -- the whole point of the field is judging staleness.
        "_measured_on": measured_on,
        "_excluded_local_only": {
            Path(f).name: round(v, 2) for f, v in sorted(skipped.items())
        },
        "_total_worker_seconds": round(sum(durations.values()), 1),
        "_file_count": len(durations),
        "durations": durations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build tests/durations.json from a `pytest --durations` "
            "transcript, for balancing the CI shard split."
        )
    )
    parser.add_argument(
        "transcripts",
        nargs="+",
        help=(
            "one or more files holding `pytest --durations=0` output. Pass "
            "every shard of one CI os+python combination to get a whole "
            "suite; see the module docstring."
        ),
    )
    parser.add_argument(
        "--output",
        default="tests/durations.json",
        help="where to write the JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "free-text note recorded as _generated_from, e.g. "
            "'CI run 32895346307, ubuntu-latest 3.12, 4 shards at -n4'. "
            "Defaults to the transcript filenames."
        ),
    )
    args = parser.parse_args(argv)

    paths = [Path(t) for t in args.transcripts]
    per_file, skipped = parse(*(p.read_text() for p in paths))
    if not per_file:
        listed = ", ".join(str(p) for p in paths)
        print(
            f"no duration lines found in {listed} -- was pytest run with "
            f"--durations=0 --durations-min=0?",
            file=sys.stderr,
        )
        return 1

    # The NEWEST transcript's mtime: with a merged set they are all from one
    # run, and the point of the field is judging staleness.
    stamp = datetime.date.fromtimestamp(
        max(p.stat().st_mtime for p in paths)
    ).isoformat()
    source = args.source or " + ".join(p.name for p in paths)
    payload = build(per_file, skipped, measured_on=stamp, source=source)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output} from {len(paths)} transcript(s): "
        f"{payload['_file_count']} files, "
        f"{payload['_total_worker_seconds']:.0f} worker-seconds"
    )
    for name, value in sorted(
        payload["durations"].items(), key=lambda kv: -kv[1]
    )[:5]:
        print(f"   {value:7.1f}s  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
