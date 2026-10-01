"""Flag slow tests and slow files from a ``pytest --durations=0`` transcript.

CI already writes one transcript per job (``durations.txt``, uploaded as the
``durations-<os>-<python>-<shard>`` artifact; see .github/workflows/tests.yml).
This reads it and reports, as GitHub ``::warning::`` annotations and on the
job's summary page:

  * every test PHASE (setup / call / teardown) over ``--phase-budget``
    seconds (default 60), and
  * every test FILE whose summed worker-seconds exceed ``--file-budget``
    (default 300).

WHY A FILE BUDGET AT ALL. ``--dist loadfile`` pins a whole file to one
worker, so no shard, at any shard count, finishes faster than its slowest
file's SERIAL time -- the "loadfile floor" of docs/testing-cache.md. On
2026-10-01 one fixture (test_integration_ob09020.py, 1329 s) WAS that floor
for every CI job that drew it. A 300 s file is roughly half the per-worker
share of a 4-shard x 4-worker job, so a file over it is on its way to setting
the floor; a test phase over 60 s is usually a structure test that samples,
or a fit fixture with a sampling budget nobody re-checked.

A WARNING, NEVER A FAILURE. Runner speed varies by +/-40% job to job and a
cold compile cache can triple a build-only test (only shard 1's compiledir
is saved), so a hard gate here would go red for reasons that are not in the
diff. The point is that a slow test surfaces on the pull request that made
it slow, instead of three weeks later in a durations refresh. The rationale
and the current offenders are in docs/testing.md, "Per-test time budget".

Annotations are capped (GitHub shows at most 10 warnings per step); the job
summary carries the full list. Exit status is 0 whatever the timings, and 2
only when a transcript cannot be read or holds no duration lines at all --
an empty transcript means the step it reads did not run, which should not
read as "all tests within budget".

    python scripts/check_test_budget.py durations.txt --annotate

Everything stays behind ``main()`` and the CLI answers ``--help``, per the
convention tests/test_scripts_smoke.py enforces on everything in scripts/.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from pathlib import Path

# The line shape pytest emits for --durations, e.g.
#   12.34s call     tests/test_alpha.py::test_one[case]
# The same pattern as scripts/gen_durations.py, which consumes the same
# transcript.
_LINE = re.compile(r"^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+?)::(\S+)\s*$")

_DEFAULT_PHASE_BUDGET = 60.0
_DEFAULT_FILE_BUDGET = 300.0
# GitHub renders at most 10 warning annotations per step; keep one free for
# the "and N more" line.
_MAX_ANNOTATIONS = 9


def parse(transcripts: list[str]) -> tuple[list[tuple], dict[str, float]]:
    """(phases, per-file seconds) from one or more transcript texts.

    ``phases`` is a list of ``(seconds, phase, file, test)``. Several
    transcripts are summed, like gen_durations.py, so a whole leg's shards can
    be checked together; each file appears in exactly one shard.
    """
    phases = []
    per_file: collections.Counter[str] = collections.Counter()
    for text in transcripts:
        for line in text.splitlines():
            m = _LINE.match(line.strip())
            if not m:
                continue
            secs = float(m[1])
            phases.append((secs, m[2], m[3], m[4]))
            per_file[m[3]] += secs
    return phases, dict(per_file)


def over_budget(
    phases: list[tuple],
    per_file: dict[str, float],
    phase_budget: float,
    file_budget: float,
) -> tuple[list[tuple], list[tuple[str, float]]]:
    """The offending phases and files, each slowest first."""
    slow_phases = sorted(
        (p for p in phases if p[0] > phase_budget), reverse=True
    )
    slow_files = sorted(
        ((f, s) for f, s in per_file.items() if s > file_budget),
        key=lambda fs: -fs[1],
    )
    return slow_phases, slow_files


def report_lines(
    slow_phases, slow_files, phase_budget, file_budget, total, n_files
):
    lines = [
        f"{total:.0f} worker-seconds over {n_files} files in this transcript; "
        f"budget {phase_budget:.0f} s per test phase, "
        f"{file_budget:.0f} s per file (warnings, not failures -- see "
        "docs/testing.md, 'Per-test time budget')"
    ]
    if not slow_phases and not slow_files:
        lines.append("every test phase and every file is within budget")
        return lines
    for f, s in slow_files:
        lines.append(f"FILE {s:.1f} s  `{f}`")
    for secs, phase, f, test in slow_phases:
        lines.append(f"{phase} {secs:.1f} s  `{f}::{test}`")
    return lines


def annotations(slow_phases, slow_files, phase_budget, file_budget):
    """The ``::warning::`` lines, capped at what GitHub will render."""
    out = []
    for f, s in slow_files:
        out.append(
            f"::warning file={f},title=Test file over time budget::"
            f"{f} costs {s:.0f} worker-seconds (budget {file_budget:.0f} s). "
            "--dist loadfile runs a file on one worker, so this is a floor "
            "on the wall clock of any shard that draws it."
        )
    for secs, phase, f, test in slow_phases:
        out.append(
            f"::warning file={f},title=Test over time budget::"
            f"{test} {phase} took {secs:.0f} s (budget {phase_budget:.0f} s)."
        )
    if len(out) > _MAX_ANNOTATIONS:
        more = len(out) - _MAX_ANNOTATIONS
        out = out[:_MAX_ANNOTATIONS] + [
            f"::warning title=Tests over time budget::and {more} more; "
            "the job summary lists every one"
        ]
    return out


def write_step_summary(title: str, lines: list[str]) -> bool:
    """Append a section to ``$GITHUB_STEP_SUMMARY``; a no-op off CI."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return False
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(f"### {title}\n\n")
        for line in lines:
            fh.write(f"- {line}\n")
        fh.write("\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Warn about test phases and test files over a time budget, from "
            "`pytest --durations=0` transcripts."
        )
    )
    parser.add_argument(
        "transcripts", nargs="+", help="durations transcript(s) to read"
    )
    parser.add_argument(
        "--phase-budget",
        type=float,
        default=_DEFAULT_PHASE_BUDGET,
        help=f"seconds per test phase (default {_DEFAULT_PHASE_BUDGET:.0f})",
    )
    parser.add_argument(
        "--file-budget",
        type=float,
        default=_DEFAULT_FILE_BUDGET,
        help=(
            f"summed worker-seconds per test file "
            f"(default {_DEFAULT_FILE_BUDGET:.0f})"
        ),
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help=(
            "also print GitHub ::warning:: annotations (CI passes this on "
            "one leg only, so a run carries one set rather than one per leg)"
        ),
    )
    parser.add_argument(
        "--title",
        default="Test time budget",
        help="heading for the job summary section",
    )
    args = parser.parse_args(argv)

    texts = []
    for path in args.transcripts:
        try:
            texts.append(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"cannot read transcript {path}: {exc}", file=sys.stderr)
            return 2
    phases, per_file = parse(texts)
    if not phases:
        print(
            f"no --durations lines in {', '.join(args.transcripts)}; was the "
            "suite run with --durations=0 --durations-min=0?",
            file=sys.stderr,
        )
        return 2

    slow_phases, slow_files = over_budget(
        phases, per_file, args.phase_budget, args.file_budget
    )
    lines = report_lines(
        slow_phases,
        slow_files,
        args.phase_budget,
        args.file_budget,
        sum(per_file.values()),
        len(per_file),
    )
    for line in lines:
        print(line)
    write_step_summary(args.title, lines)
    if args.annotate:
        for line in annotations(
            slow_phases, slow_files, args.phase_budget, args.file_budget
        ):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
