"""Rank EXPENSIVE tests by how little unique coverage they contribute.

WHAT THIS PRODUCES IS A CANDIDATE LIST FOR A HUMAN TO JUDGE, NEVER A VERDICT.
Coverage overlap is not redundancy. Two tests can execute exactly the same
lines and assert entirely different properties of the result -- one that a
`logp` is finite, another that it equals a reference value -- and in this
codebase that is the norm rather than the exception. A test whose every line
is covered elsewhere may still be the only thing pinning a number.

Run it, read the assertions of whatever it flags, and expect to keep most of
them. When this was first run over the whole suite (2026-08-25) it flagged 12
of 62 expensive tests and NONE of them survived review -- see "What the first
run found" below. That is a useful result, not a failure: it says the suite is
not carrying redundant expensive tests.

Usage
-----
Two steps, because the measurement is expensive and worth reusing::

    # 1. whole suite with per-test coverage contexts (~+33% over a normal run)
    COVERAGE_FILE=/tmp/cov/.coverage \\
    poetry run pytest -q -n6 --dist loadfile \\
        --cov=exozippy --cov-context=test --cov-report= \\
        --durations=0 --durations-min=0 | tee /tmp/cov/durations.txt

    # 2. analyse
    poetry run python scripts/find_redundant_tests.py /tmp/cov/.coverage \\
        --transcript /tmp/cov/durations.txt

Two signals per test
--------------------
* **unique lines** -- lines no other test in the suite executes. Many means
  definitively load-bearing. Zero is necessary but nowhere near sufficient.
* **a dominating test** -- another single test whose covered lines are a
  superset. The strongest signal a machine can give, because it names the
  specific test whose assertions have to be compared.

Four filters, each of which exists because its absence produced a confidently
wrong answer on a real run
--------------------------------------------------------------------------
1. **Cost is PER TEST, not per file.** The first version weighted by file, so
   every test in test_vcve.py was charged that file's 208 s and the top
   candidate came out as
   ``test_both_roots_satisfy_the_quadratic[0.5-1.2]`` -- ten lines, no
   measurable runtime, a table-driven numeric check whose deletion would save
   nothing and lose an assertion.
2. **Only tests above --min-seconds are examined.** Removing a cheap test
   saves nothing and risks coverage for it.
3. **Parametrize siblings cannot dominate each other.** Cases of one test
   necessarily execute the same lines, so they dominate mutually and
   meaninglessly: the first run flagged
   ``..._gradient_finite[-5.0]`` as dominated by ``[0.0]`` AND ``[0.0]`` as
   dominated by ``[-5.0]``. What a human would drop is the family, or some of
   its cases, never one case picked by a coverage tie with its own twin.
4. **Tests below --min-lines are reported, not judged.** Coverage cannot see
   work done in a subprocess. ``test_run_lifecycle_status_snapshot_and_
   graceful_stop`` costs 128 s and covers 74 LINES because it spawns
   ``python -m exozippy.cli`` in a child that is not traced -- so it looked
   maximally redundant while being one of the most expensive tests in the
   suite. In-process tests here cover 2800-5900 lines, so the gap is
   unambiguous.

What the first run found, so the shape is known
-----------------------------------------------
12 candidates of 62 expensive tests, all explained without deleting anything:

* **11 of 12 were dominated by a test in the SAME FILE, and 5 mutually** (A by
  B and B by A). Mutual domination means IDENTICAL coverage, which is the
  signature of several tests building one model and asserting different things
  about it. Peers, not duplicates. test_per_element_soft_bound.py and
  test_orbit_crossing.py each form such a cycle.
* The most expensive,
  ``test_rm_two_instrument_logp_and_gradient_finite_on_both_backends``, is a
  known blind spot: both backends run the same graph-CONSTRUCTION lines while
  the JAX execution is compiled rather than traced, and a finite C-backend
  dlogp does not validate the JAX path.
* The single cross-file hit was coincidence -- dominated by a
  ``test_shipped_example_prepares`` case, which runs ``prepare()`` on every
  shipped config and so dominates many things while testing none of them.

Everything stays behind ``main()`` and the CLI answers ``--help``, per the
convention tests/test_scripts_smoke.py enforces on everything in scripts/.
"""

from __future__ import annotations

import argparse
import collections
import re
import sqlite3
import sys
from pathlib import Path

# pytest's --durations line shape, e.g.
#   12.34s call     tests/test_alpha.py::test_one[case]
_DURATION_LINE = re.compile(r"^([0-9.]+)s\s+(call|setup|teardown)\s+(\S+)\s*$")


def load_contexts(db_path: Path) -> tuple[dict[str, dict[int, bytes]], dict]:
    """``{test id: {file id: numbits}}`` plus ``{file id: path}``.

    pytest-cov labels a context ``"<test id>|<phase>"``; the phases are merged,
    because setup and call together are what a test costs and covers.
    """
    con = sqlite3.connect(db_path)
    files = dict(con.execute("select id, path from file"))
    contexts = dict(con.execute("select id, context from context"))
    per_test: dict[str, dict[int, bytes]] = collections.defaultdict(dict)
    for file_id, ctx_id, numbits in con.execute(
        "select file_id, context_id, numbits from line_bits"
    ):
        ctx = contexts.get(ctx_id, "")
        if not ctx:
            continue
        test = ctx.split("|", 1)[0]
        prev = per_test[test].get(file_id)
        if prev is None:
            per_test[test][file_id] = numbits
        else:
            per_test[test][file_id] = _union(prev, numbits)
    return per_test, files


def _union(a: bytes, b: bytes) -> bytes:
    """Bitwise OR of two numbits blobs, which may differ in length."""
    if len(b) > len(a):
        a, b = b, a
    merged = bytearray(a)
    for i, byte in enumerate(b):
        merged[i] |= byte
    return bytes(merged)


def load_costs(transcript: Path) -> collections.Counter:
    """Per-test seconds, summed over setup/call/teardown."""
    cost: collections.Counter = collections.Counter()
    for line in transcript.read_text().splitlines():
        m = _DURATION_LINE.match(line.strip())
        if m:
            cost[m[3]] += float(m[1])
    return cost


def family(test: str) -> str:
    """The test function a parametrize case belongs to."""
    return test.split("[", 1)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank expensive tests by how little unique line coverage they "
            "contribute. Produces CANDIDATES for human review, never "
            "verdicts -- see the module docstring."
        )
    )
    parser.add_argument(
        "database", help="a coverage database with per-test contexts"
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help=(
            "a `pytest --durations=0` transcript, for PER-TEST cost. Required "
            "because per-FILE weights charge a file's whole cost to each of "
            "its tests, which ranks 0-second parametrize cases as expensive."
        ),
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=20.0,
        help=(
            "only examine tests costing at least this much (default: "
            "%(default)s). Removing a cheap test saves nothing."
        ),
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=500,
        help=(
            "below this many covered lines, treat the test as doing its work "
            "where coverage cannot see it -- a subprocess -- and draw no "
            "conclusion (default: %(default)s)"
        ),
    )
    args = parser.parse_args(argv)

    # AFTER parse_args, deliberately. coverage is a dev-only dependency, and
    # tests/test_scripts_smoke.py drives --help in-process on every script in
    # scripts/ -- so importing it any earlier makes --help raise
    # ModuleNotFoundError wherever coverage is absent. Asking a script what its
    # flags are must not require its heaviest dependency.
    from coverage.numbits import numbits_intersection, numbits_to_nums

    per_test, files = load_contexts(Path(args.database))
    if not per_test:
        print(
            "no per-test contexts in this database -- was pytest run with "
            "--cov-context=test?",
            file=sys.stderr,
        )
        return 1
    cost = load_costs(Path(args.transcript))
    if not cost:
        print(
            f"no duration lines in {args.transcript} -- was pytest run with "
            f"--durations=0 --durations-min=0?",
            file=sys.stderr,
        )
        return 1

    def size(cov):
        return sum(len(numbits_to_nums(nb)) for nb in cov.values())

    def is_subset(small, big):
        for file_id, nb in small.items():
            other = big.get(file_id)
            if other is None or numbits_intersection(nb, other) != nb:
                return False
        return True

    print(f"{len(per_test)} test contexts over {len(files)} source files")

    # How many tests cover each line, so "unique to one test" is a lookup.
    owners: collections.Counter = collections.Counter()
    for cov in per_test.values():
        for file_id, nb in cov.items():
            for line in numbits_to_nums(nb):
                owners[(file_id, line)] += 1

    fam_cost: collections.Counter = collections.Counter()
    for test in per_test:
        fam_cost[family(test)] += cost.get(test, 0.0)

    candidates = sorted(
        (t for t in per_test if cost.get(t, 0.0) >= args.min_seconds),
        key=lambda t: -cost.get(t, 0.0),
    )
    print(
        f"{len(candidates)} of {len(per_test)} tests cost >= "
        f"{args.min_seconds:.0f}s and are worth examining\n"
    )

    rows, invisible, unmeasured = [], [], []
    for test in candidates:
        cov = per_test[test]
        total = size(cov)
        if total == 0:
            unmeasured.append(test)
            continue
        if total < args.min_lines:
            invisible.append((cost.get(test, 0.0), total, test))
            continue
        unique = sum(
            1
            for file_id, nb in cov.items()
            for line in numbits_to_nums(nb)
            if owners[(file_id, line)] == 1
        )
        dominator = None
        if unique == 0:
            for other, other_cov in per_test.items():
                if other == test or family(other) == family(test):
                    continue
                if size(other_cov) < total:
                    continue
                if is_subset(cov, other_cov):
                    dominator = other
                    break
        rows.append((cost.get(test, 0.0), total, unique, test, dominator))

    rows.sort(key=lambda r: (r[2], -r[0]))
    print(f"{'test_s':>8} {'fam_s':>8} {'lines':>6} {'uniq':>5}  test")
    for c, total, unique, test, dom in rows:
        flag = "  <-- CANDIDATE" if dom else ""
        print(
            f"{c:8.1f} {fam_cost[family(test)]:8.1f} {total:6d} "
            f"{unique:5d}  {test}{flag}"
        )
        if dom:
            print(f"{'':32}dominated by: {dom}")

    if invisible:
        print(
            f"\nWORK INVISIBLE TO COVERAGE -- these {len(invisible)} tests "
            f"cover fewer than {args.min_lines} lines, which here means they "
            f"spawn a subprocess. Nothing can be concluded about them:"
        )
        for c, n, test in sorted(invisible, key=lambda r: -r[0]):
            print(f"   {c:7.1f}s {n:5d} lines  {test}")

    if unmeasured:
        print(
            f"\nNOT MEASURED -- no coverage was recorded for these "
            f"{len(unmeasured)} tests, so nothing can be concluded either way:"
        )
        for test in unmeasured:
            print(f"   {test}")

    flagged = sum(1 for r in rows if r[4])
    print(
        f"\n{flagged} of {len(rows)} examined tests contribute NO unique line "
        f"AND are\nfully covered by one other test. Read their ASSERTIONS "
        f"before touching them:\ncoverage overlap is not redundancy. See this "
        f"script's docstring."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
