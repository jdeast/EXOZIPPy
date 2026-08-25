"""Build tests/durations.json from a ``pytest --durations`` transcript.

The file this writes balances the CI shard split (see
scripts/pytest_shard.py and docs/testing-cache.md). Regenerate it after a
change that moves the suite's shape substantially:

    poetry run pytest -q -n6 --dist loadfile --durations=0 --durations-min=0 \\
        > /tmp/durations.txt
    poetry run python scripts/gen_durations.py /tmp/durations.txt \\
        tests/durations.json

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


def parse(transcript: str) -> tuple[dict[str, float], dict[str, float]]:
    """(per-file worker-seconds, per-file seconds excluded as local-only)."""
    per_file: collections.Counter[str] = collections.Counter()
    skipped: collections.Counter[str] = collections.Counter()
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
        "_generated_from": "pytest -q -n6 --dist loadfile --durations=0",
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
        "transcript",
        help="file holding the output of `pytest --durations=0`",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="tests/durations.json",
        help="where to write the JSON (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    per_file, skipped = parse(Path(args.transcript).read_text())
    if not per_file:
        print(
            f"no duration lines found in {args.transcript} -- was pytest run "
            f"with --durations=0 --durations-min=0?",
            file=sys.stderr,
        )
        return 1

    stamp = datetime.date.fromtimestamp(
        Path(args.transcript).stat().st_mtime
    ).isoformat()
    payload = build(per_file, skipped, measured_on=stamp)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output}: {payload['_file_count']} files, "
        f"{payload['_total_worker_seconds']:.0f} worker-seconds"
    )
    for name, value in sorted(
        payload["durations"].items(), key=lambda kv: -kv[1]
    )[:5]:
        print(f"   {value:7.1f}s  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
