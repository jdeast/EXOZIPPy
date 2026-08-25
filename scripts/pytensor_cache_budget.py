"""Bound a PyTensor compiledir by ENTRY COUNT, evicting least-recently-used.

Why this exists, and why ``pytensor-cache cleanup`` is not it
-------------------------------------------------------------
PyTensor caches every compiled C module as one subdirectory of the
compiledir, holding a ``key.pkl`` plus the ``.so``. The first time any
process compiles, ``ModuleCache.__init__`` calls ``refresh()``, which walks
EVERY subdirectory and unpickles EVERY ``key.pkl`` it has not already
loaded. That walk is O(entries) file reads, it happens once per process,
and under ``pytest -n 6`` it happens six times -- serialized, because
``refresh()`` holds the compile lock while it runs.

On this repo that walk grew to 4035 entries / 4.1 GB and started costing
60-135 s, landing on whichever test happened to trigger the first compile
in its worker. That test then blew pytest-timeout's 300 s cap and went RED
for a reason that had nothing to do with it. The cost is dominated by cold
PAGE-CACHE reads of those 4035 ``key.pkl`` files: an immediate re-run, with
the same 4035 entries but the pages now hot, paid almost nothing. So the
quantity to bound is the ENTRY COUNT, which is what the walk is linear in.

``pytensor-cache cleanup`` does not bound it. That command is
``compiledir.cleanup()`` + ``ModuleCache.clear_old()``, and ``clear_old``
only deletes entries older than
``age_thresh_del = cmodule__age_thresh_use + 7 days``, i.e. 31 days. On a
repository whose suite runs daily, nothing is ever 31 days untouched --
every entry gets its atime refreshed by the very ``refresh()`` walk we are
trying to shorten. Measured on the 4.1 GB cache above: 4035 -> 4034
entries, 4.1 GB -> 4.1 GB, 44 s spent. Age is precisely the knob that does
not work here; count is the one that does.

Ordering
--------
Eviction is least-recently-used on the ``st_atime`` of ``key.pkl`` -- the
same stat field PyTensor's own ``last_access_time()`` reads for its age
policy, so this agrees with PyTensor about which entries are "recent".
Note that under the usual ``relatime`` mount option atime is only rewritten
when it is already older than mtime or than 24 hours, so this is a
day-granularity LRU rather than an exact one. That is fine for the purpose:
the goal is to bound the count, and any sane eviction order achieves it.
``max(atime, mtime)`` is used so a ``noatime`` mount (where atime is frozen
at creation) degrades to newest-first rather than to arbitrary order.

Concurrency
-----------
This prunes without taking PyTensor's compile lock, so it assumes no other
process is walking the same compiledir at the same time. The test suite
calls it from ``pytest_configure`` on the xdist CONTROLLER, before any
worker exists, which satisfies that. Two prunes racing each other are
harmless (both rmtree the same directory; the loser's ``FileNotFoundError``
is swallowed). A prune racing another process's ``refresh()`` is not
protected against -- ``refresh()`` does a bare ``os.listdir`` on each entry
-- so do not point a long-running interactive fit at a compiledir while a
suite is starting up against it. Keeping the suite's compiledir separate
from the interactive one, which is the other half of this change, is what
makes that assumption hold by construction.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# PyTensor names a compiled module "<something>.so" (or ".pyd" on Windows);
# see cmodule.module_name_from_dir, which this mirrors. An entry carrying a
# key.pkl but no module file is one refresh() would delete and warn about,
# so we treat it as broken and remove it up front.
_MODULE_SUFFIXES = (".so", ".pyd")

# refresh() skips this one by name; so must we, or we would delete the
# compile lock out from under a concurrent process.
_LOCK_DIR = "lock_dir"

# The root conftest gives every xdist worker its OWN base_compiledir, one
# level down, named after PYTEST_XDIST_WORKER ("gw0", "gw1", ...). So the
# tree under the suite's base_compiledir is
#
#     ~/.pytensor-pytest/compiledir_<platform>/     <- -n0 runs only
#     ~/.pytensor-pytest/gw0/compiledir_<platform>/ <- worker 0
#     ~/.pytensor-pytest/gw1/compiledir_<platform>/ <- worker 1
#     ...
#
# and the ONLY one pytensor.config resolves in the controller is the first.
# That is why the budget has to be walked explicitly over the rest: pruning
# what pytensor.config points at bounds the directory no worker ever reads
# and leaves the six that they do read completely unbounded. Measured on
# this box before the fix: controller 2280 entries (budget 3000, so never
# pruned), gw0-gw5 1455-1562 entries EACH, none of them ever considered.
# In CI the same omission made every saved cache artifact bigger than the
# last -- ubuntu 3.14 went 374 -> 494 -> 606 -> 661 -> 781 MB over five
# consecutive master merges -- until the repository-wide 10 GB cache budget
# went into eviction, which is exactly what the Zenodo-spectra and
# ephemeris caches cannot afford to lose.
_WORKER_DIR_GLOB = "gw*"


@dataclass
class PruneStats:
    """What one prune pass found and removed."""

    scanned: int = 0
    kept: int = 0
    removed_broken: int = 0
    removed_lru: int = 0
    removed_platform_dirs: list[str] = field(default_factory=list)
    # True when the cheap pre-check proved we were under budget and the
    # per-entry pass never ran. `scanned` is then a listdir name count, i.e.
    # an upper bound on the entry count rather than a measurement of it, and
    # the summary must not claim otherwise.
    skipped: bool = False

    @property
    def removed(self) -> int:
        return self.removed_broken + self.removed_lru

    def summary(self, compiledir: Path, max_entries: int) -> str:
        if self.skipped:
            return (
                f"pytensor compiledir {compiledir}: at most {self.scanned} "
                f"entries, within the budget of {max_entries}, not scanned"
            )
        parts = [
            f"pytensor compiledir {compiledir}: {self.scanned} entries "
            f"scanned, budget {max_entries}, kept {self.kept}, removed "
            f"{self.removed} ({self.removed_broken} broken, "
            f"{self.removed_lru} least-recently-used)"
        ]
        if self.removed_platform_dirs:
            parts.append(
                "also removed stale sibling compiledirs: "
                + ", ".join(sorted(self.removed_platform_dirs))
            )
        return "; ".join(parts)


def _module_present(files: list[str]) -> bool:
    return any(name.endswith(_MODULE_SUFFIXES) for name in files)


def _last_used(key_pkl: Path) -> float:
    """Best available "when was this entry last wanted" timestamp.

    See the module docstring on relatime and noatime for why this is
    max(atime, mtime) rather than atime alone.
    """
    st = key_pkl.stat()
    return max(st.st_atime, st.st_mtime)


def _rmtree(path: Path) -> None:
    # ignore_errors: a concurrent prune, or a directory the user cannot
    # write, must not abort the pass. A cache entry we failed to delete
    # costs one extra key.pkl read, which is the thing we are optimizing,
    # not a correctness problem.
    shutil.rmtree(path, ignore_errors=True)


def prune_compiledir(
    compiledir: Path,
    max_entries: int,
    dry_run: bool = False,
) -> PruneStats:
    """Evict entries from ``compiledir`` until at most ``max_entries`` remain.

    Broken entries (no ``key.pkl``, or a ``key.pkl`` with no compiled
    module beside it) are removed first and do not count against the
    budget; the remainder are evicted least-recently-used.
    """
    stats = PruneStats()
    if not compiledir.is_dir():
        return stats

    # CHEAP PRE-CHECK, and it is not an optimization detail -- without it
    # this function reproduces the very cost it exists to remove. The scan
    # below opens one directory and stats one key.pkl PER ENTRY, which on a
    # cold page cache measured 72 s over 4034 entries: paid on every pytest
    # invocation, including the `-n0 -x` single-test runs that are supposed
    # to be the fast path.
    #
    # One listdir of the parent bounds the entry count from ABOVE (the names
    # include lock_dir and any stray files, neither of which is an entry), so
    # a raw count within budget proves the real count is too, and we can skip
    # the pass entirely. Steady state is therefore one directory read. The
    # cost is only paid when it buys something: when we are actually over
    # budget and about to reclaim.
    #
    # What this skips when under budget is the broken-entry sweep. That is
    # deliberate -- refresh() removes those itself, and there were 8 of them
    # in 4034.
    try:
        names = os.listdir(compiledir)
    except OSError:
        return stats
    if len(names) <= max_entries:
        stats.scanned = stats.kept = len(names)
        stats.skipped = True
        return stats

    live: list[tuple[float, Path]] = []
    for name in sorted(names):
        if name == _LOCK_DIR:
            continue
        entry = compiledir / name
        try:
            files = os.listdir(entry)
        except NotADirectoryError:
            continue
        except OSError:
            continue
        stats.scanned += 1
        if "key.pkl" not in files or not _module_present(files):
            stats.removed_broken += 1
            if not dry_run:
                _rmtree(entry)
            continue
        try:
            live.append((_last_used(entry / "key.pkl"), entry))
        except OSError:
            # Vanished between listdir and stat. Nothing to do and nothing
            # to report -- it is already not costing us a read.
            continue

    # Newest first, so the tail of the list is what goes.
    live.sort(key=lambda pair: pair[0], reverse=True)
    keep, evict = live[:max_entries], live[max_entries:]
    stats.kept = len(keep)
    stats.removed_lru = len(evict)
    if not dry_run:
        for _, entry in evict:
            _rmtree(entry)
    return stats


def sweep_other_platform_dirs(
    base_compiledir: Path,
    keep: Path,
    dry_run: bool = False,
) -> list[str]:
    """Remove sibling ``compiledir_*`` trees that this platform will never read.

    PyTensor names the compiledir after the platform, the processor, the
    Python version and the bit width (configdefaults._default_compiledir),
    so a kernel upgrade, a Python patch release or a CI runner image bump
    silently strands the whole previous tree: nothing reads it, nothing
    deletes it, and it keeps counting against disk (and, in CI, against the
    size of the saved cache). This is only safe on a base_compiledir owned
    by one purpose -- the suite's own -- which is why it is a separate
    opt-in function rather than part of prune_compiledir.
    """
    removed = []
    if not base_compiledir.is_dir():
        return removed
    keep = keep.resolve()
    for entry in sorted(base_compiledir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("compiledir_"):
            continue
        if entry.resolve() == keep:
            continue
        removed.append(entry.name)
        if not dry_run:
            _rmtree(entry)
    return removed


def enforce_budget(
    base_compiledir: Path,
    compiledir: Path,
    max_entries: int,
    sweep_platforms: bool = False,
    dry_run: bool = False,
) -> PruneStats:
    """Full pass: drop stranded platform trees, then bound the live one."""
    stats = prune_compiledir(compiledir, max_entries, dry_run=dry_run)
    if sweep_platforms:
        stats.removed_platform_dirs = sweep_other_platform_dirs(
            base_compiledir, compiledir, dry_run=dry_run
        )
    return stats


def worker_compiledirs(
    base_compiledir: Path, compiledir: Path
) -> list[tuple[Path, Path]]:
    """``(base, compiledir)`` for every per-xdist-worker tree under base.

    ``compiledir`` supplies the platform directory NAME to look for, which
    is what makes this correct rather than a guess: PyTensor derives that
    name from the platform, the processor, the Python version and the bit
    width, and every worker on this machine resolves the same one, because
    the conftest changes only the base. So the worker's live tree is
    ``base/gwN/<same name>`` and anything else named ``compiledir_*`` beside
    it is stranded by a kernel or Python bump, exactly as at the top level.

    Sorted, and worker directories that hold no compiledir at all are still
    returned: prune_compiledir on a missing directory is a cheap no-op, and
    reporting the pair keeps the summary honest about what was considered.
    """
    if not base_compiledir.is_dir():
        return []
    pairs = []
    for entry in sorted(base_compiledir.glob(_WORKER_DIR_GLOB)):
        if not entry.is_dir():
            continue
        pairs.append((entry, entry / compiledir.name))
    return pairs


def enforce_budget_tree(
    base_compiledir: Path,
    compiledir: Path,
    max_entries: int,
    sweep_platforms: bool = False,
    dry_run: bool = False,
) -> list[tuple[Path, PruneStats]]:
    """Bound the controller's compiledir AND every per-worker one.

    ``max_entries`` is PER COMPILEDIR, not a total across the tree, because
    the cost it exists to bound is per compiledir: each worker process
    builds its own ModuleCache and walks only its own directory, so what
    determines that walk's length is one directory's entry count. The price
    of that denominator is disk -- a budget of N with W workers holds up to
    (W + 1) x N entries -- which is why the default is sized against ONE
    run's per-worker working set rather than against several.

    Returns one (compiledir, stats) pair per directory considered, the
    controller's first.
    """
    results = [
        (
            compiledir,
            enforce_budget(
                base_compiledir,
                compiledir,
                max_entries,
                sweep_platforms=sweep_platforms,
                dry_run=dry_run,
            ),
        )
    ]
    for worker_base, worker_compiledir in worker_compiledirs(
        base_compiledir, compiledir
    ):
        results.append(
            (
                worker_compiledir,
                enforce_budget(
                    worker_base,
                    worker_compiledir,
                    max_entries,
                    sweep_platforms=sweep_platforms,
                    dry_run=dry_run,
                ),
            )
        )
    return results


def summarize_tree(
    results: list[tuple[Path, PruneStats]], max_entries: int
) -> str:
    """One line for the pytest header, however many compiledirs there were.

    The per-directory detail is dropped once everything is within budget --
    seven identical "not scanned" lines in a test header is noise. What is
    worth a line every run is the total, because that is the number that
    grew unnoticed for weeks.

    "at most" when any directory was within budget, and that hedge is not
    padding: the cheap pre-check proves a directory is under budget from one
    listdir, whose name count includes lock_dir and any stray file, so it
    bounds the entry count from above rather than measuring it. PruneStats
    already refuses to claim otherwise per directory; the total must not
    launder those bounds into a figure that looks counted.
    """
    if not results:
        return ""
    touched = [(d, st) for d, st in results if st.removed]
    total = sum(st.kept for _, st in results)
    bound = "at most " if any(st.skipped for _, st in results) else ""
    head = (
        f"pytensor compiledirs: {len(results)} pruned to <= {max_entries} "
        f"entries each, {bound}{total} entries held in total"
    )
    if not touched:
        return head
    detail = "; ".join(
        f"{d.parent.name}/{d.name}: removed {st.removed}" for d, st in touched
    )
    platforms = sorted(
        name for _, st in results for name in st.removed_platform_dirs
    )
    if platforms:
        detail += "; stale sibling compiledirs removed: " + ", ".join(
            platforms
        )
    return head + " (" + detail + ")"


def _resolve_compiledir(args: argparse.Namespace) -> tuple[Path, Path]:
    """Ask PyTensor where its compiledir is, unless told explicitly.

    Importing pytensor is deliberately deferred to here: the test suite
    calls prune_compiledir() directly from the xdist controller, which has
    no reason to pay a pytensor import (or to load numpy into a process
    that is about to spawn six workers).
    """
    if args.compiledir:
        compiledir = Path(args.compiledir).expanduser()
        return compiledir.parent, compiledir
    import pytensor

    return Path(pytensor.config.base_compiledir), Path(
        pytensor.config.compiledir
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bound a PyTensor compiledir by entry count, evicting "
            "least-recently-used entries. Unlike `pytensor-cache cleanup`, "
            "this reclaims space on a cache that is used every day."
        )
    )
    parser.add_argument(
        "--compiledir",
        help=(
            "compiledir to prune. Defaults to whatever pytensor.config "
            "resolves, which honours PYTENSOR_FLAGS."
        ),
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=1500,
        help="maximum cache entries to keep (default: %(default)s)",
    )
    parser.add_argument(
        "--sweep-other-platforms",
        action="store_true",
        help=(
            "also delete sibling compiledir_* trees belonging to another "
            "platform/Python/kernel. Only safe on a base_compiledir owned "
            "by a single purpose."
        ),
    )
    parser.add_argument(
        "--include-worker-dirs",
        action="store_true",
        help=(
            "also prune the per-xdist-worker compiledirs (gw*/) that the "
            "test suite's conftest creates one level below the base. "
            "pytensor.config resolves only the controller's, which under "
            "-n is the one directory no worker ever reads -- so without "
            "this the budget bounds nothing that matters. --max-entries is "
            "per compiledir, not a total."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be removed, remove nothing",
    )
    args = parser.parse_args(argv)

    base, compiledir = _resolve_compiledir(args)
    prefix = "[dry run] " if args.dry_run else ""
    if args.include_worker_dirs:
        results = enforce_budget_tree(
            base,
            compiledir,
            args.max_entries,
            sweep_platforms=args.sweep_other_platforms,
            dry_run=args.dry_run,
        )
        print(prefix + summarize_tree(results, args.max_entries))
        for directory, stats in results:
            print(prefix + "  " + stats.summary(directory, args.max_entries))
        return 0
    stats = enforce_budget(
        base,
        compiledir,
        args.max_entries,
        sweep_platforms=args.sweep_other_platforms,
        dry_run=args.dry_run,
    )
    print(prefix + stats.summary(compiledir, args.max_entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
