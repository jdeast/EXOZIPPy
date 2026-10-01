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
For the test suite's own cache, see "Per-RUN compiledirs" below (review
2.13.5): every run compiles in a private tree, and the one shared tree is
touched only under ``base/.lock``, so the assumption in the next paragraph
holds there by construction.

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
import re
import secrets
import shutil
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    # No flock: the per-run layout below cannot lock the shared tree, so the
    # root conftest leaves the suite's compiledir unmanaged there (see
    # docs/testing-cache.md). Nothing else in this module needs it.
    fcntl = None

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


# ---------------------------------------------------------------------------
# Seeding a cold worker compiledir from a warm one
# ---------------------------------------------------------------------------
# The per-worker compiledirs are ~95% redundant copies of each other: measured
# on this repo, each of gw0-gw5 held 1455-1562 entries against 1564 distinct
# entries for a whole cold run, because most of what gets compiled is shared
# infrastructure that every file's model builds rather than anything specific
# to the files that worker drew.
#
# That redundancy bills twice:
#
#   1. Changing the worker count makes the NEW workers compile everything from
#      scratch. Measured on the run that took CI from -n2 to -n4: ubuntu 3.12
#      went 43:21 -> 52:24 and 3.13 went 36:39 -> 39:12, all green, purely
#      because gw2 and gw3 started empty.
#   2. It makes the saved CI cache scale with the worker count, so the cache
#      cannot absorb more parallelism -- sharding the matrix 2x at -n4 would
#      want ~8 entries x ~1.5 GB, back over GitHub's 10 GB repository budget.
#
# Both dissolve if only ONE tree is stored and the others are derived from it.
# Deriving is cheap because of what a cache entry is made of. Measured over 148
# entries: the .so is 85.3% of the bytes, the .cpp 13.4%, and key.pkl 1.3%.
# Only key.pkl is ever rewritten in place -- PyTensor appends to it when a
# second key maps to one compiled module -- so only key.pkl has to be a private
# copy. Everything else can be a hard link.
#
# Measured on 398 entries: 5.3 s and 7.1 MB of real disk, against 53.6 s and
# 218 MB for a full copy. Extrapolated to a 1800-entry tree, ~24 s and ~32 MB
# per extra worker, instead of a cold compile of every graph.
_MUTABLE_ENTRY_FILES = frozenset({"key.pkl"})


@dataclass
class SeedStats:
    """What one seeding pass did."""

    seeded_dirs: list[str] = field(default_factory=list)
    skipped_dirs: list[str] = field(default_factory=list)
    entries: int = 0
    linked: int = 0
    copied: int = 0
    # Files that FELL BACK to a copy because os.link refused them. Tracked and
    # reported because the fallback is silent and turns a 5-second metadata
    # operation into a multi-minute byte copy. The way it happens in practice
    # is a cross-device link (EXDEV): point EXOZIPPY_TEST_COMPILEDIR at a
    # different filesystem from the source tree and every link fails. That is
    # not hypothetical -- it is how the first measurement of this code was
    # taken by mistake, reporting 0 hardlinked / 1988 copied.
    link_fallbacks: int = 0

    def summary(self) -> str:
        if not self.seeded_dirs:
            return ""
        parts = [
            f"seeded {len(self.seeded_dirs)} worker compiledir(s) from a warm "
            f"one ({', '.join(self.seeded_dirs)}): {self.entries} entries, "
            f"{self.linked} hard-linked, {self.copied} copied"
        ]
        if self.link_fallbacks:
            parts.append(
                f"WARNING: {self.link_fallbacks} hard links fell back to "
                "copies (cross-device compiledir?), so this was far more "
                "expensive than it should be"
            )
        return "; ".join(parts)


def _entry_names(compiledir: Path) -> list[str]:
    """Real cache entries in ``compiledir``: subdirectories, nothing else.

    Directories only, unlike prune_compiledir's cheap pre-check, which counts
    raw listdir names deliberately because it needs an UPPER bound. Here an
    over-count is actively wrong: a compiledir always holds an ``__init__.py``,
    and counting that as an entry made a brand-new tree look warm enough to be
    a seed source -- which is how a first run reported "seeded 2 worker
    compiledir(s) ... 0 entries, 0 hard-linked, 0 copied".
    """
    try:
        return [
            name
            for name in os.listdir(compiledir)
            if name != _LOCK_DIR and (compiledir / name).is_dir()
        ]
    except OSError:
        return []


def choose_seed_source(base_compiledir: Path, compiledir: Path) -> Path | None:
    """The warmest compiledir in the tree, or None if nothing is warm.

    "Warmest" is simply the largest entry count. The controller's own tree is
    a candidate because that is what a -n0 run populates, and in CI it is the
    restored one when only a single canonical tree is saved.
    """
    candidates = [compiledir]
    candidates += [
        d for _, d in worker_compiledirs(base_compiledir, compiledir)
    ]
    best, best_count = None, 0
    for candidate in candidates:
        count = len(_entry_names(candidate))
        if count > best_count:
            best, best_count = candidate, count
    return best


def seed_compiledir(source: Path, target: Path) -> tuple[int, int, int, int]:
    """Populate ``target`` from ``source``. Returns (entries, linked, copied, fallbacks).

    Hard-links every file in every cache entry except key.pkl, which is copied
    -- see the module comment above for why that split is exactly right.

    Existing files in ``target`` are left alone, so this is safe to re-run and
    safe against a partially populated target.
    """
    entries = linked = copied = fallbacks = 0
    try:
        names = sorted(_entry_names(source))
    except OSError:  # pragma: no cover - defensive
        return (0, 0, 0, 0)

    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        src_entry = source / name
        if not src_entry.is_dir():
            continue
        entries += 1
        for root, _dirs, files in os.walk(src_entry):
            rel = Path(root).relative_to(source)
            (target / rel).mkdir(parents=True, exist_ok=True)
            for filename in files:
                src_file = Path(root) / filename
                dst_file = target / rel / filename
                if dst_file.exists():
                    continue
                if filename in _MUTABLE_ENTRY_FILES:
                    try:
                        shutil.copy2(src_file, dst_file)
                        copied += 1
                    except OSError:
                        pass
                    continue
                try:
                    os.link(src_file, dst_file)
                    linked += 1
                except OSError:
                    # Cross-device, or a filesystem without hard links.
                    try:
                        shutil.copy2(src_file, dst_file)
                        copied += 1
                        fallbacks += 1
                    except OSError:
                        pass
    return (entries, linked, copied, fallbacks)


def seed_worker_compiledirs(
    base_compiledir: Path,
    compiledir: Path,
    n_workers: int,
    cold_fraction: float = 0.25,
    dry_run: bool = False,
) -> SeedStats:
    """Give every worker this run will start a warm compiledir.

    A worker directory is seeded when it holds less than ``cold_fraction`` of
    the warmest tree's entry count. That threshold, rather than "is empty", is
    what makes the pass idempotent AND useful: steady state seeds nothing and
    costs one listdir per worker, while a worker that was interrupted halfway
    through populating itself still gets topped up.

    Called from the xdist CONTROLLER before any worker exists -- the same
    reason the prune lives there. A worker cannot do this for itself: by the
    time it runs it already holds the ModuleCache it would be seeding.
    """
    stats = SeedStats()
    if n_workers <= 0:
        return stats

    source = choose_seed_source(base_compiledir, compiledir)
    if source is None:
        # Nothing warm anywhere: a genuinely first-ever run. Every worker
        # compiles from scratch and there is nothing to copy from.
        return stats
    source_count = len(_entry_names(source))
    if source_count == 0:
        # Nothing warm anywhere. Returning here rather than looping is what
        # keeps a first-ever run from reporting that it seeded directories it
        # in fact left empty.
        return stats
    threshold = source_count * cold_fraction

    for index in range(n_workers):
        target = base_compiledir / f"gw{index}" / compiledir.name
        if target.resolve() == source.resolve():
            continue
        count = len(_entry_names(target))
        if count >= threshold:
            stats.skipped_dirs.append(f"gw{index}")
            continue
        if dry_run:
            stats.seeded_dirs.append(f"gw{index}")
            continue
        entries, linked, copied, fallbacks = seed_compiledir(source, target)
        if linked + copied == 0:
            # Claim nothing. The test is on FILES PLACED, not on entry
            # directories walked: a hollow entry (a directory with no files,
            # which is a broken cache entry the prune removes anyway) would
            # otherwise be reported as a successful seed of "1 entries, 0
            # hard-linked, 0 copied".
            stats.skipped_dirs.append(f"gw{index}")
            continue
        stats.seeded_dirs.append(f"gw{index}")
        stats.entries += entries
        stats.linked += linked
        stats.copied += copied
        stats.link_fallbacks += fallbacks
    return stats


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


# ---------------------------------------------------------------------------
# Per-RUN compiledirs: two concurrent suites never share a live tree
# ---------------------------------------------------------------------------
# Review 2.13.5.  The layout above gave every xdist worker ``base/gwN`` -- the
# worker id and nothing else -- so two suites started on one machine against
# one shared base (the whole point of sharing it: a fresh worktree starts
# warm) both put their gw0 on ``base/gw0``.  Each controller then pruned the
# tree at startup, and the prune treats an entry with no ``key.pkl`` as broken
# -- which is exactly what an entry the OTHER suite is compiling right now
# looks like, because PyTensor writes ``key.pkl`` last.  So each suite
# deleted the other's in-flight compiles, and the victim failed three ways,
# all naming a ``compiledir_*/tmp...`` path: the link step ("cannot open
# output file"), the import of the .so ("dlimport"), and the import of the
# tmp module itself (``ModuleNotFoundError: No module named 'tmp...'``).
# The sweep's own docstring stated the precondition the layout broke: "only
# safe on a base_compiledir owned by one purpose".
#
# The layout that replaces it:
#
#     base/shared/compiledir_<platform>/          <- the warm tree; NO process
#                                                    ever compiles in it
#     base/runs/<token>/compiledir_<platform>/    <- a -n0 run's private tree
#     base/runs/<token>/gwN/compiledir_<platform>/ <- worker N of that run
#     base/runs/<token>/alive.lock                <- flock'd for the run's life
#     base/.lock                                  <- guards base/shared
#
# A run seeds its private trees from the shared one at startup (hard links,
# key.pkl copied, exactly as above), compiles only into its own trees, and at
# exit MOVES the entries it created into the shared tree.  Every touch of the
# shared tree happens under ``base/.lock``: seeding under a SHARED flock,
# promotion, pruning and the platform sweep under an EXCLUSIVE one.
#
# Why that makes the prune safe even while other runs are live, which is a
# stronger property than "prune only when alone": no live PyTensor process
# ever opens the shared tree.  A run's .so files are hard links, so deleting
# the shared name leaves the run's own link -- the inode -- intact, and its
# key.pkl is a private copy.  The only readers of the shared tree are other
# runs' seeding passes, and those hold the shared lock.  Requiring "no other
# live run" on top would only starve the budget on a busy box.
#
# Why promotion is atomic per entry: it is one ``os.rename`` of the entry
# DIRECTORY, on one filesystem by construction (both live under base), so the
# shared tree holds either the whole entry or nothing.  It runs after every
# worker has exited, and it moves only entries whose ``key.pkl`` is present
# and ends in pickle's STOP opcode -- PyTensor writes key.pkl after the .so is
# built and imported, so a key.pkl is the commit marker, and the STOP check
# rejects one truncated by a worker killed mid-write (Ctrl-C).
#
# Liveness, for reaping the run directory a crashed run left behind: the
# owner holds an exclusive flock on ``alive.lock`` for its whole life, and the
# kernel drops it when the process dies however it dies.  So "can I take that
# lock?" is a liveness test that survives PID reuse.  The token records host
# and pid as well, and a directory whose host is not this one is NEVER reaped:
# on a base shared over NFS a remote run's liveness cannot be checked from
# here.
_SHARED_DIR = "shared"
_RUNS_DIR = "runs"
_BASE_LOCK = ".lock"
_ALIVE_LOCK = "alive.lock"
# A run directory is created under this prefix and renamed into place once
# its alive.lock is held, so no reaper ever sees a run directory whose owner
# has not yet locked it.
_CREATING_PREFIX = "."
# pickle's STOP opcode: the last byte of every complete pickle, any protocol.
_PICKLE_STOP = b"."


def shared_compiledir(base: Path, platform_name: str) -> Path:
    """The shared warm tree for this platform under the suite's base."""
    return base / _SHARED_DIR / platform_name


def new_run_token(host: str | None = None, pid: int | None = None) -> str:
    """``<random>-<pid>-<host>``: unique per run, and parseable for liveness.

    Host LAST, because a hostname may itself contain ``-``; the random part
    is hex and the pid is digits, so a two-way split from the left is exact.
    """
    host = socket.gethostname() if host is None else host
    pid = os.getpid() if pid is None else pid
    return f"{secrets.token_hex(4)}-{pid}-{host}"


def parse_run_token(name: str) -> tuple[int, str]:
    """``(pid, host)`` from a run directory name. Raises on anything else.

    Raising, not skipping: ``base/runs`` is written only by this module, so a
    name it cannot parse is something else's directory sitting where the
    suite reaps, and guessing whether it may be deleted is the wrong call.
    """
    parts = name.lstrip(_CREATING_PREFIX).split("-", 2)
    if len(parts) != 3 or not parts[1].isdigit() or not parts[2]:
        raise ValueError(
            f"{name!r} under the suite's runs/ directory is not a run "
            "directory this suite created (expected <hex>-<pid>-<host>); "
            "remove it by hand"
        )
    return int(parts[1]), parts[2]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_state(run_dir: Path, host: str | None = None) -> str:
    """``"live"``, ``"dead"`` or ``"remote"`` for one run directory."""
    host = socket.gethostname() if host is None else host
    pid, owner_host = parse_run_token(run_dir.name)
    if owner_host != host:
        return "remote"
    if run_dir.name.startswith(_CREATING_PREFIX):
        # Its owner has not locked alive.lock yet (or died before it could),
        # so the pid is the only evidence there is.
        return "live" if _pid_alive(pid) else "dead"
    try:
        fd = os.open(run_dir / _ALIVE_LOCK, os.O_RDWR)
    except FileNotFoundError as exc:
        # create_run_dir locks alive.lock BEFORE renaming the directory into
        # place, so a published run directory always has one.
        raise RuntimeError(
            f"run directory {run_dir} has no {_ALIVE_LOCK}; it was not made "
            "by create_run_dir -- remove it by hand"
        ) from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return "live"
    finally:
        os.close(fd)
    return "dead"


@dataclass
class RunDir:
    """One suite run's private compiledir root, alive while ``alive_fd`` is."""

    path: Path
    alive_fd: int

    def trees(self, platform_name: str) -> list[Path]:
        """Every private compiledir this run could have written."""
        trees = [self.path / platform_name]
        trees += sorted(self.path.glob(f"gw*/{platform_name}"))
        return trees

    def close(self) -> None:
        if self.alive_fd >= 0:
            os.close(self.alive_fd)  # drops the flock
            self.alive_fd = -1


def create_run_dir(base: Path, token: str | None = None) -> RunDir:
    """Create ``base/runs/<token>`` and hold its liveness lock.

    Built under a dot-prefixed name and renamed into place only once
    alive.lock is flock'd, so a concurrent reaper never sees a published run
    directory it could mistake for a dead one.
    """
    token = new_run_token() if token is None else token
    runs = base / _RUNS_DIR
    runs.mkdir(parents=True, exist_ok=True)
    creating = runs / (_CREATING_PREFIX + token)
    final = runs / token
    creating.mkdir()
    fd = os.open(creating / _ALIVE_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(fd, f"{socket.gethostname()} {os.getpid()}\n".encode())
    os.rename(creating, final)
    return RunDir(final, fd)


class BaseLock:
    """``flock`` on ``base/.lock``, polled so a wait has a deadline.

    ``acquire`` returns False rather than raising when the deadline passes:
    every caller's response to contention is to SKIP its step (prune,
    seed, promote), which is always the benign direction -- a run that does
    not seed starts colder, a prune that does not run leaves the budget
    unenforced once, a promotion that does not run loses one run's delta.
    None of them can delete anything a live run needs.
    """

    def __init__(self, base: Path):
        self.path = base / _BASE_LOCK
        self.fd = -1

    def acquire(
        self, exclusive: bool, timeout: float, poll: float = 0.25
    ) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(self.fd, mode | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.release()
                    return False
                time.sleep(poll)

    def release(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


@dataclass
class ReapStats:
    reaped: list[str] = field(default_factory=list)
    live: list[str] = field(default_factory=list)
    remote: list[str] = field(default_factory=list)


def reap_dead_runs(
    base: Path, keep: Path | None = None, host: str | None = None
) -> ReapStats:
    """Remove the run directories of runs that died without cleaning up.

    Call with the EXCLUSIVE base lock held, so two reapers never race on one
    directory.  ``keep`` is the caller's own run directory.
    """
    stats = ReapStats()
    runs = base / _RUNS_DIR
    if not runs.is_dir():
        return stats
    for run_dir in sorted(runs.iterdir()):
        if keep is not None and run_dir == keep:
            continue
        state = run_state(run_dir, host=host)
        if state == "dead":
            _rmtree(run_dir)
            stats.reaped.append(run_dir.name)
        elif state == "live":
            stats.live.append(run_dir.name)
        else:
            stats.remote.append(run_dir.name)
    return stats


def choose_run_seed_source(
    base: Path, platform_name: str, cold_fraction: float = 0.25
) -> Path | None:
    """The tree a new run seeds from: the shared one, unless it is still cold.

    The pre-2.13.5 trees (``base/compiledir_*`` and ``base/gwN/``) are read
    as seed sources only while the shared tree holds less than
    ``cold_fraction`` of the warmest of them -- i.e. on the first run after
    the layout changed, or the first CI run restoring an old-layout cache.
    That run promotes everything it seeded into the shared tree, and from
    then on the shared tree wins.  Picking the warmest by raw count every
    time would keep choosing a frozen legacy tree that sits a few entries
    above the budget the shared tree is pruned to, and re-promote the same
    evicted entries on every run.

    The legacy trees are never pruned or deleted here: a checkout that
    predates this layout may still be running its suite in them.
    """
    shared = shared_compiledir(base, platform_name)
    shared_count = len(_entry_names(shared))
    legacy = [base / platform_name]
    legacy += [d for _, d in worker_compiledirs(base, base / platform_name)]
    best, best_count = None, 0
    for candidate in legacy:
        count = len(_entry_names(candidate))
        if count > best_count:
            best, best_count = candidate, count
    if shared_count and shared_count >= cold_fraction * best_count:
        return shared
    return best


def seed_run(
    run: RunDir, source: Path, platform_name: str, n_workers: int
) -> SeedStats:
    """Seed every private tree this run will use, in parallel.

    One target per worker (``gw0``..``gw<n-1>``), or the run's own tree for a
    -n0 run.  Threads, because each pass is a few thousand link/copy
    syscalls and the GIL is released across them; measured on this box a
    2036-entry seed is ~1.6 s against a warm page cache and ~30 s against a
    cold one, and only the first pass pays the cold reads.
    """
    if n_workers > 0:
        labels = [f"gw{i}" for i in range(n_workers)]
        targets = [run.path / label / platform_name for label in labels]
    else:
        labels = ["run"]
        targets = [run.path / platform_name]
    stats = SeedStats()
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        results = list(pool.map(lambda t: seed_compiledir(source, t), targets))
    for label, (entries, linked, copied, fallbacks) in zip(
        labels, results, strict=True
    ):
        if linked + copied == 0:
            stats.skipped_dirs.append(label)
            continue
        stats.seeded_dirs.append(label)
        stats.entries += entries
        stats.linked += linked
        stats.copied += copied
        stats.link_fallbacks += fallbacks
    return stats


def _entry_complete(entry: Path) -> bool:
    """A cache entry PyTensor finished: a module, and a whole key.pkl."""
    try:
        files = os.listdir(entry)
    except OSError:
        return False
    if "key.pkl" not in files or not _module_present(files):
        return False
    try:
        with open(entry / "key.pkl", "rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return False
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) == _PICKLE_STOP
    except OSError:
        return False


@dataclass
class PromoteStats:
    promoted: int = 0
    already_shared: int = 0
    incomplete: int = 0

    def summary(self, shared: Path) -> str:
        return (
            f"promoted {self.promoted} new compiledir entries into {shared} "
            f"({self.already_shared} already there, {self.incomplete} "
            "incomplete and dropped)"
        )


def promote_run(run: RunDir, shared: Path, platform_name: str) -> PromoteStats:
    """Move every entry this run created into the shared tree.

    Call with the EXCLUSIVE base lock held and after every process of the
    run has stopped compiling.  An entry whose name the shared tree already
    has was seeded from it (or promoted by a sibling worker of this run) and
    is left behind to be deleted with the run directory.
    """
    stats = PromoteStats()
    shared.mkdir(parents=True, exist_ok=True)
    for tree in run.trees(platform_name):
        for name in sorted(_entry_names(tree)):
            target = shared / name
            if target.exists():
                stats.already_shared += 1
                continue
            entry = tree / name
            if not _entry_complete(entry):
                stats.incomplete += 1
                continue
            os.rename(entry, target)
            stats.promoted += 1
    return stats


def remove_run_dir(run: RunDir) -> None:
    """Delete a finished run's directory (its lock must already be closed)."""
    if run.alive_fd >= 0:
        raise RuntimeError(
            f"refusing to delete {run.path} while this process still holds "
            "its liveness lock"
        )
    _rmtree(run.path)


def enforce_shared_budget(
    base: Path, platform_name: str, max_entries: int, dry_run: bool = False
) -> PruneStats:
    """Bound the shared tree and sweep stranded platforms beside it.

    Call with the EXCLUSIVE base lock held.  The sweep is safe here, unlike
    on the legacy layout, because ``base/shared`` is owned by one purpose by
    construction: nothing but this module writes in it, and nothing compiles
    in it.
    """
    return enforce_budget(
        base / _SHARED_DIR,
        shared_compiledir(base, platform_name),
        max_entries,
        sweep_platforms=True,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Naming the race when it fires anyway (review 2.13.5)
# ---------------------------------------------------------------------------
# All three recorded spellings -- "cannot open output file" at link time,
# "dlimport" on the .so, ``ModuleNotFoundError: No module named 'tmp...'`` on
# the module -- carry the deleted DIRECTORY and share no other wording, so the
# match is on the path.
RACE_SIGNATURE = re.compile(r"compiledir_[^\s/'\"]*/tmp")
RACE_MESSAGE = (
    "{n} failure(s)/error(s) name a compiledir_*/tmp path: the signature of "
    "review 2.13.5, a PyTensor cache entry deleted under a live run (another "
    "suite sharing this compiledir, or a pre-2.13.5 checkout running beside "
    "this one). Triage by error class: compile/import errors on that path are "
    "environmental; an assertion on a number still has to be explained."
)


def compiledir_race_hits(reports) -> list[str]:
    """Node ids of the failed/errored reports that carry the race signature."""
    return [
        report.nodeid
        for report in reports
        if RACE_SIGNATURE.search(report.longreprtext)
    ]


def report_compiledir_race(terminalreporter) -> None:
    """``pytest_terminal_summary`` body: one line when the race fired.

    It only NAMES the failures -- no retry, no skip, no softening -- and it
    repeats the triage rule because the rule is the point: of three failures
    in one recorded run, two were this race and the third was a real
    regression.
    """
    reports = []
    for key in ("failed", "error"):
        reports += terminalreporter.stats.get(key, [])
    hits = compiledir_race_hits(reports)
    if hits:
        terminalreporter.write_line(
            RACE_MESSAGE.format(n=len(hits)), yellow=True, bold=True
        )


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


def _main_suite_base(base: Path, max_entries: int, dry_run: bool) -> int:
    import pytensor

    platform_name = Path(pytensor.config.compiledir).name
    prefix = "[dry run] " if dry_run else ""
    lock = BaseLock(base)
    if not lock.acquire(exclusive=True, timeout=120.0):
        print(f"{base}/{_BASE_LOCK} is held by a running suite; nothing done")
        return 1
    try:
        if dry_run:
            reaped = ReapStats()
        else:
            reaped = reap_dead_runs(base)
        stats = enforce_shared_budget(
            base, platform_name, max_entries, dry_run=dry_run
        )
    finally:
        lock.release()
    shared = shared_compiledir(base, platform_name)
    print(prefix + stats.summary(shared, max_entries))
    print(
        f"{prefix}runs: {len(reaped.reaped)} dead reaped, {len(reaped.live)} "
        f"live, {len(reaped.remote)} on other hosts (never touched)"
    )
    return 0


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
        "--suite-base",
        help=(
            "the test suite's compiledir BASE (EXOZIPPY_TEST_COMPILEDIR, "
            "default ~/.pytensor-pytest). Under its .lock, reap dead runs' "
            "directories, prune base/shared to --max-entries and sweep "
            "stranded platform trees inside it -- what every suite run does "
            "at startup (review 2.13.5). Ignores --compiledir, "
            "--include-worker-dirs and --sweep-other-platforms."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be removed, remove nothing",
    )
    args = parser.parse_args(argv)

    if args.suite_base:
        return _main_suite_base(
            Path(args.suite_base).expanduser(), args.max_entries, args.dry_run
        )

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
