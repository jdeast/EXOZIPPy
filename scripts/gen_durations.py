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

THIS IS NORMALLY RUN BY CI, not by hand: .github/workflows/refresh-durations.yml
does exactly the two commands above weekly (and on `gh workflow run
refresh-durations.yml`) against the latest green master run, and proposes the
result when the balance has moved. Run it by hand when a PR adds a heavy test
file and should not wait for Monday. The convention is in docs/testing.md,
"Keeping tests/durations.json current".

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
import os
import re
import subprocess
import sys
from pathlib import Path

# The line shape pytest emits for --durations, e.g.
#   12.34s call     tests/test_alpha.py::test_one[case]
_LINE = re.compile(r"^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+?)::(\S+)\s*$")


def local_only_examples(repo_root: Path) -> tuple[str, ...]:
    """Names of ``examples/<name>/`` directories on disk that git does not track.

    A developer's untracked example is collected by test_examples_prepare
    (and, if a fixture exists for it, by the mulens acceptance files) LOCALLY
    and never on CI, so counting its cost from a workstation transcript would
    skew the shard balance for everyone else.

    DERIVED, NOT WRITTEN DOWN. This used to be a hardcoded
    ``_LOCAL_ONLY = ("ob09020",)``, and the tuple outlived the fact it
    recorded: examples/ob09020 was committed, CI started paying for its five
    prepare cases and the single heaviest replay in the suite (316 s together,
    measured 2026-09-14), and the generator went on silently dropping them --
    so test_mulens_acceptance.py was weighed at 732 s where CI paid 873 and
    test_examples_prepare.py at 300 where CI paid 475. Asking git what is
    untracked cannot go stale that way. On CI (a clean checkout) the answer is
    always empty, which is exactly right: whatever CI ran, CI paid for.

    Returns () when there is no examples/ directory or git cannot be asked,
    so a transcript from a tarball still converts -- with nothing excluded.
    That is the conservative direction: an over-counted file shows up in the
    output where a reader can see it, a dropped one does not.
    """
    examples = repo_root / "examples"
    if not examples.is_dir():
        return ()
    # Ask about repo_root and nothing else: git exports GIT_DIR (and in a
    # worktree GIT_WORK_TREE / GIT_INDEX_FILE) into the environment of every
    # hook it runs, and an inherited GIT_DIR makes `git -C <other repo>`
    # answer for the hook's repository instead. The pre-push suite hit
    # exactly that: the test's throwaway repository reported both of its
    # examples untracked because the index consulted was this one's.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--", "examples"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env=env,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    tracked = {
        line.split("/", 2)[1]
        for line in out.splitlines()
        if line.count("/") >= 2
    }
    return tuple(
        sorted(
            d.name
            for d in examples.iterdir()
            if d.is_dir() and d.name not in tracked
        )
    )


def _parametrize_id(test_id: str) -> str:
    """The ``[...]`` part of ``test_name[param-id]``, or "" when there is none.

    The local-only tags are matched INSIDE the brackets only: an example
    directory's name is a parametrize id, never part of a test function's
    name, and matching the whole id would let a short directory name
    (``hat3``) hit an unrelated test that merely mentions it.
    """
    _, bracket, rest = test_id.partition("[")
    return rest if bracket else ""


def parse(
    *transcripts: str, local_only: tuple[str, ...] = ()
) -> tuple[dict[str, float], dict[str, float]]:
    """(per-file worker-seconds, per-file seconds excluded as local-only).

    Accepts several transcripts and SUMS them, which is what makes merging a
    CI run's per-shard artifacts work: each job reports only the files in its
    own shard, so the union across shards is one whole suite and no file
    appears twice. Feeding the same transcript in twice would double its
    files, so pass each shard exactly once.

    ``local_only`` names example directories whose parametrized cases are
    excluded (see ``local_only_examples``). Empty by default: nothing is
    dropped unless the caller has established that it does not run on CI.
    """
    per_file: collections.Counter[str] = collections.Counter()
    skipped: collections.Counter[str] = collections.Counter()
    for transcript in transcripts:
        for line in transcript.splitlines():
            m = _LINE.match(line.strip())
            if not m:
                continue
            params = _parametrize_id(m[4])
            if params and any(tag in params for tag in local_only):
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
    local_only = local_only_examples(Path(__file__).resolve().parents[1])
    if local_only:
        print(
            f"excluding untracked local examples: {', '.join(local_only)}",
            file=sys.stderr,
        )
    per_file, skipped = parse(
        *(p.read_text() for p in paths), local_only=local_only
    )
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
