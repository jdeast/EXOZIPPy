"""Repo-root conftest: process-level setup that must happen before imports.

This file is imported at pytest startup -- before any test module (and before
the first ``import numpy`` / ``import pytensor``), and freshly in every xdist
worker subprocess. Everything here depends on that timing: the native
libraries below read their environment once, when they first load, so setting
these variables here is early enough and setting them later would be a no-op.

Two unrelated concerns live here for that one reason. Thread pinning comes
first because it has to; the PyTensor compile cache follows.
"""

import importlib.util
import os
import shlex
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# BLAS / OpenMP thread pinning
# ---------------------------------------------------------------------------
# Why pin to 1: the suite runs ``-n 6`` (six worker processes). With the thread
# vars unset, each worker's BLAS grabs *all* cores, so on a 36-core box that is
# 6 x 36 = 216 threads fighting over 36 cores -- a context-switch storm that,
# stacked with six concurrent full-System builds, pushes a loaded machine into
# swap and can freeze it for a long time. One BLAS thread per worker keeps the
# core count matched to the worker count (6 busy cores, not 216 oversubscribed).
# The math here is not BLAS-bound anyway -- the cost is pytensor graph compiles
# and Python -- so single-threaded BLAS costs no measurable wall time.
#
# ``setdefault`` so an explicit override in the environment still wins (e.g. a
# developer profiling BLAS scaling can export their own values).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")


# ---------------------------------------------------------------------------
# The test suite's own PyTensor compile cache
# ---------------------------------------------------------------------------
# PyTensor caches each compiled C module as one subdirectory of its compiledir.
# The first compile in any process builds the ModuleCache, whose ``refresh()``
# walks EVERY subdirectory and unpickles EVERY ``key.pkl``. That walk is
# O(entries) file reads, it runs once per process, and it holds the compile
# lock while it does -- so under ``-n 6`` it runs six times, serialized.
#
# Left pointing at the shared interactive ``~/.pytensor``, that directory grew
# to 4035 entries / 4.1 GB, fed by months of interactive fits and by every
# agent worktree on this box. The walk then cost 60-135 s and was billed to
# whichever test happened to trigger the first compile in its worker -- which
# blew pytest-timeout's 300 s cap and turned an innocent test RED. It presents
# as flaky because the cost is dominated by cold PAGE-CACHE reads: re-running
# the same suite against the same 4035 entries, with the pages now hot, paid
# almost nothing.
#
# Two changes, and both are needed:
#
#   1. The suite gets its OWN base_compiledir, so hours-long interactive fits
#      and parallel agent worktrees no longer inflate the thing the suite has
#      to walk at startup, and vice versa. It is under $HOME rather than in
#      the checkout deliberately: every worktree of this repo then SHARES one
#      warm cache, which is the case that hurts most here -- a fresh worktree
#      would otherwise pay a full cold compile.
#
#   2. That directory is bounded by ENTRY COUNT, pruned least-recently-used on
#      the controller before any worker starts (see pytest_configure below).
#      Count, because the walk is linear in it. Not bytes, and above all not
#      AGE: ``pytensor-cache cleanup`` only deletes entries untouched for 31
#      days, and on a repo whose suite runs daily nothing ever is -- the
#      refresh walk itself keeps bumping their atimes. Measured, that command
#      took 44 s to go from 4035 entries to 4034.
#
# Set EXOZIPPY_TEST_COMPILEDIR to relocate it, or to the empty string to opt
# out entirely and use whatever PyTensor would have chosen (useful when
# bisecting something that smells cache-shaped).
_COMPILEDIR_ENV = "EXOZIPPY_TEST_COMPILEDIR"
_BUDGET_ENV = "EXOZIPPY_TEST_COMPILEDIR_MAX_ENTRIES"

# PER COMPILEDIR, not a total across the tree -- see the per-worker section
# below, and enforce_budget_tree's docstring for why that is the right
# denominator (each worker walks only its own directory, so one directory's
# entry count is what the startup walk is linear in).
#
# The number is sized against ONE run's per-worker working set. A full run
# creates 1564 distinct entries, and measurement shows a worker's own
# directory holds very nearly all of them -- 1455 to 1562 across gw0-gw5 --
# because most of what gets compiled is shared infrastructure that every
# file's model builds, not something specific to the files that worker drew.
# So the budget has to clear ~1600 or a run would evict entries it still
# needs, and there is no point going far above it: the headroom that 3000
# used to buy ("several worktrees") is not available at this denominator,
# because with W workers the tree now holds up to (W + 1) x this.
#
# 3000 was the old value and it was applied to the CONTROLLER's compiledir,
# which under -n is the one directory no worker ever reads. It therefore
# bounded nothing: the controller sat at 2280 entries and never hit 3000,
# while the six directories that do get read grew without any bound at all.
_DEFAULT_MAX_ENTRIES = 2000

_raw_compiledir = os.environ.get(_COMPILEDIR_ENV)
if _raw_compiledir is None:
    _BASE_COMPILEDIR = Path.home() / ".pytensor-pytest"
elif _raw_compiledir.strip() == "":
    _BASE_COMPILEDIR = None
else:
    _BASE_COMPILEDIR = Path(_raw_compiledir).expanduser()

if _BASE_COMPILEDIR is not None:
    if "pytensor" in sys.modules:
        # Not fatal, but the redirect below cannot work: base_compiledir is
        # declared ``mutable=False``, so PyTensor has already resolved and
        # frozen it. Say so rather than silently running against the shared
        # cache and leaving someone to wonder why the budget never applies.
        warnings.warn(
            "pytensor was imported before the root conftest ran, so the test "
            "suite's private compiledir could not be configured; the shared "
            f"cache will be used instead of {_BASE_COMPILEDIR}",
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        # Ours goes FIRST and any pre-existing flags are appended, because
        # parse_config_string() builds a dict left to right, so a duplicate
        # key later in the string wins. A developer who exported their own
        # base_compiledir therefore still gets it.
        _existing = os.environ.get("PYTENSOR_FLAGS", "")
        _ours = ",".join(
            [
                f"base_compiledir={shlex.quote(str(_BASE_COMPILEDIR))}",
                # PyTensor serializes ALL compilation behind one lock per
                # compiledir, so under -n 6 the six workers queue for it. Its
                # default acquire timeout is 120 s (compile__wait * 24), and
                # against a genuinely EMPTY compiledir that is not enough: a
                # measured cold run had four tests die on
                # `filelock._error.Timeout` while waiting their turn --
                # test_nsnl, test_rossiter, test_distance_volume_prior and
                # test_multiplanet, i.e. whichever ones happened to queue
                # behind a long compile, exactly the same lottery as the
                # refresh-walk timeout this file exists to fix.
                #
                # This is pre-existing and is not caused by the private
                # compiledir above -- but that change makes EVERY developer
                # pay one cold run when they first adopt it, which turns a
                # rare failure into a guaranteed one. 600 s covers five
                # workers queued behind a long compile.
                #
                # The cost of raising it: a lock left behind by a genuinely
                # dead process takes longer to break. Live holders refresh
                # the lock every half period, so this only delays recovery
                # from a hard kill, and only for the one process that hits
                # it.
                "compile__timeout=600",
            ]
        )
        os.environ["PYTENSOR_FLAGS"] = ",".join(
            p for p in (_ours, _existing) if p
        )

# ---------------------------------------------------------------------------
# Per-xdist-worker compiledir
# ---------------------------------------------------------------------------
# PyTensor serializes ALL compilation behind one lock per compiledir, and
# compile__timeout=600 above is exactly pytest-timeout's own 600 s ceiling --
# so on a cold cache a worker queued behind the others' compiles dies by
# pytest-timeout without ever failing the lock.  Measured (ezsuite 15363115,
# 15363286): the cluster suite jobs override base_compiledir to per-job local
# scratch (cold by construction, deliberately -- the home directory is NFS
# and the client drops advisory locks under -n 6), and whichever test owned
# the largest compile at the wrong moment died -- first the KMT provenance
# fixtures behind a 122-compile seeding storm (fixed at the source), then the
# kelt4 hierarchical logp, the suite's biggest single compile, which passes
# alone in ~143 s.  A worker suffix on the FINAL winning base_compiledir
# removes the shared lock entirely; the price is duplicated compiles of
# common ops across workers, paid in parallel instead of in a queue.
# Appended as the rightmost flag, so it wins whatever base won above
# (parse_config_string builds its dict left to right).
#
# The OTHER price, which went unpaid for weeks: these directories are not
# where pytensor.config points in the controller, so the budget below has to
# be walked over them explicitly. It was not, and they grew without bound --
# locally to six directories of ~1500 entries each, and in CI to a saved
# cache artifact that got bigger on every master merge until the
# repository-wide 10 GB budget started evicting the Zenodo and ephemeris
# caches that the suite cannot cheaply re-download. _prune_compiledir now
# covers the whole tree; do not "simplify" it back to pytensor.config's
# single directory.
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker and "pytensor" not in sys.modules:
    _flags = os.environ.get("PYTENSOR_FLAGS", "")
    _base = None
    for _part in _flags.split(","):
        if _part.strip().startswith("base_compiledir="):
            _base = _part.split("=", 1)[1].strip().strip("'\"")
    if _base:
        os.environ["PYTENSOR_FLAGS"] = ",".join(
            part
            for part in (
                _flags,
                "base_compiledir=" + shlex.quote(str(Path(_base) / _worker)),
            )
            if part
        )


def _load_budget_module():
    """Import scripts/pytensor_cache_budget.py by path.

    By path, and not by putting scripts/ on sys.path: that directory holds
    ``getdata.py``, ``mkparam.py`` and ``mkticsed.py``, whose names would then
    become importable top-level modules and shadow nothing today but are one
    rename away from shadowing something. Loading the single file we want
    keeps the blast radius at that file.
    """
    name = "_pytensor_cache_budget"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent / "scripts" / "pytensor_cache_budget.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec_module, per the importlib docs: @dataclass
    # resolves cls.__module__ through sys.modules while the class body is
    # being processed, and raises AttributeError on None if it is missing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Filled in by pytest_configure on the controller and read back by
# pytest_report_header. A module global rather than an attribute stapled onto
# ``config``: the controller is a single process, and pytest's Config is not
# ours to grow attributes on.
_compiledir_summary = None


def _xdist_worker(config):
    return hasattr(config, "workerinput")


def _xdist_active(config):
    return bool(config.getoption("numprocesses", 0) or 0)


def _prune_compiledir(config):
    """Bound the suite's compiledirs. Controller only, before workers exist.

    Doing it here rather than in a fixture is what makes the prune safe
    without taking PyTensor's compile lock: ``pytest_configure`` on the
    controller runs before xdist spawns a single worker, so nothing else is
    walking the directories yet. It is also the only moment at which the
    per-worker trees can be pruned at all -- a worker cannot prune its own,
    because by the time it runs it is already holding the ModuleCache it
    would be deleting under itself.

    EVERY compiledir in the tree, not just the one pytensor.config resolves.
    That distinction is the whole point: this process is the controller, so
    what it resolves is ``base/compiledir_<platform>``, and under -n that is
    the one directory none of the workers ever opens. See
    enforce_budget_tree.
    """
    budget = int(os.environ.get(_BUDGET_ENV, _DEFAULT_MAX_ENTRIES))
    module = _load_budget_module()
    if module is None:  # pragma: no cover - defensive
        return None
    import pytensor  # noqa: PLC0415 -- must follow the PYTENSOR_FLAGS write above

    results = module.enforce_budget_tree(
        Path(pytensor.config.base_compiledir),
        Path(pytensor.config.compiledir),
        budget,
        # Safe here and only here: this base_compiledir belongs to the test
        # suite alone, so a sibling compiledir_* tree is a stranded kernel or
        # Python version that nothing will ever read again.
        sweep_platforms=True,
    )
    return module.summarize_tree(results, budget)


def _seed_worker_compiledirs(config):
    """Give every worker this run will start a warm compiledir.

    Controller only, before any worker exists -- same placement and same
    reason as _prune_compiledir, and for seeding it is not merely convenient
    but necessary: a worker cannot seed its own compiledir, because by the
    time it runs it is already holding the ModuleCache it would be seeding.

    This exists because the per-worker compiledirs are ~95% redundant copies
    of one another (each holds ~1500 of the 1564 entries a whole cold run
    creates -- most of what compiles is shared infrastructure every file's
    model builds). Without it, raising -n makes the NEW workers compile every
    graph from scratch: measured when CI went from -n2 to -n4, ubuntu 3.12
    went 43:21 -> 52:24 purely because gw2 and gw3 started empty. It also
    lets CI store ONE canonical tree instead of one per worker, which is what
    keeps the saved cache under GitHub's 10 GB repository budget as the worker
    count and the shard count grow.

    See seed_worker_compiledirs for the hard-link/copy split and the measured
    costs.
    """
    module = _load_budget_module()
    if module is None:  # pragma: no cover - defensive
        return None
    n_workers = int(config.getoption("numprocesses", 0) or 0)
    if n_workers <= 0:
        # -n0: this process IS the worker and uses the controller compiledir,
        # which is the seed source rather than a seed target.
        return None
    import pytensor  # noqa: PLC0415 -- must follow the PYTENSOR_FLAGS write above

    stats = module.seed_worker_compiledirs(
        Path(pytensor.config.base_compiledir),
        Path(pytensor.config.compiledir),
        n_workers,
    )
    summary = stats.summary()
    if summary:
        # stderr rather than pytest_report_header, and CI is the reason: it
        # runs `pytest -q`, which suppresses the header entirely -- and
        # seeding is the step most worth seeing there, being where a restored
        # single-tree cache gets fanned back out, and where a cross-device
        # hard-link fallback would show up as a sudden multi-minute startup.
        # It prints only when a seed actually happened, so the steady state is
        # silent rather than one more line of noise.
        print(summary, file=sys.stderr, flush=True)


def _warm_module_cache():
    """Pay the ModuleCache walk here, where no test can be blamed for it.

    ``pytest_configure`` runs before pytest-timeout arms its per-test SIGALRM,
    so however long the walk takes it cannot fail a test. Before this existed
    the walk was paid lazily by whichever test compiled first in each worker,
    which is exactly how a 300 s timeout landed on an unrelated vcve test.
    """
    try:
        import pytensor  # noqa: PLC0415 -- see _prune_compiledir
        from pytensor.link.c.cmodule import get_module_cache  # noqa: PLC0415

        get_module_cache(pytensor.config.compiledir)
    except Exception as exc:  # pragma: no cover - pure optimization
        # Never fail a session over a warm-up. If PyTensor moves this API the
        # only consequence is that the walk goes back to being paid lazily.
        warnings.warn(
            f"could not pre-warm the PyTensor module cache: {exc!r}",
            RuntimeWarning,
            stacklevel=1,
        )


def pytest_configure(config):
    global _compiledir_summary

    if _BASE_COMPILEDIR is None:
        return

    if _xdist_worker(config):
        # Every worker builds its own ModuleCache, so every worker has to warm
        # its own. They contend on the compile lock doing it, which is why
        # bounding the entry count matters as much as moving the cost.
        _warm_module_cache()
        return

    # PRUNE FIRST, THEN SEED, and the order is load-bearing: pruning picks the
    # seed source down to the budget, so a freshly seeded worker starts inside
    # the budget instead of immediately over it. Seeding first would copy
    # entries that the very next prune would evict.
    _compiledir_summary = _prune_compiledir(config)
    _seed_worker_compiledirs(config)
    if not _xdist_active(config):
        # -n0: this process runs the tests itself, so it is also the one that
        # needs the cache warm.
        _warm_module_cache()


def pytest_report_header(config):
    return [_compiledir_summary] if _compiledir_summary else []
