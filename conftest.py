"""Repo-root conftest: process-level setup that must happen before imports.

This file is imported at pytest startup -- before any test module (and before
the first ``import numpy`` / ``import pytensor``), and freshly in every xdist
worker subprocess. Everything here depends on that timing: the native
libraries below read their environment once, when they first load, so setting
these variables here is early enough and setting them later would be a no-op.

Two unrelated concerns live here for that one reason. Thread pinning comes
first because it has to; the PyTensor compile cache follows, with the hook
that names the compiledir race (review 2.13.5) at the very end.
"""

import atexit
import importlib.util
import os
import shlex
import sys
import time
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
#   2. Its shared tree is bounded by ENTRY COUNT, pruned least-recently-used
#      at session start, under a lock (see _start_run below).
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

# Applied to the SHARED tree (base/shared/compiledir_*), the one tree every
# run seeds its workers from (see "One private compiledir root PER RUN"
# below). Each worker walks its own seeded copy, so the shared tree's entry
# count is what every worker's startup walk is linear in.
#
# The number is sized against ONE run's per-worker working set. A full run
# creates 1564 distinct entries, and measurement shows a worker's own
# directory holds very nearly all of them -- 1455 to 1562 across gw0-gw5 --
# because most of what gets compiled is shared infrastructure that every
# file's model builds, not something specific to the files that worker drew.
# So the budget has to clear ~1600 or a run would evict entries it still
# needs, and there is no point going far above it.
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
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec_module, per the importlib docs: @dataclass
    # resolves cls.__module__ through sys.modules while the class body is
    # being processed, and raises AttributeError on None if it is missing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# One private compiledir root PER RUN (review 2.13.5)
# ---------------------------------------------------------------------------
# The base above is SHARED -- across worktrees, deliberately, so a fresh
# worktree starts warm -- and until 2026-09-30 every run's xdist workers
# compiled straight into ``base/gwN``. Two suites started on one machine
# (an agent's targeted run beside a pre-push full suite, two pushes a minute
# apart) therefore shared gw0..gwN, and each controller's startup prune
# deleted the other's in-flight compiles: an entry PyTensor is still
# building has no key.pkl yet, which is exactly what the prune calls broken.
# The victims failed on a ``compiledir_*/tmp...`` path in one of three
# spellings -- "cannot open output file", "dlimport", or
# ``ModuleNotFoundError: No module named 'tmp...'`` -- and looked like a dozen
# real failures (26 failed + 11 errors on 2026-09-30).
#
# Now each run (the xdist CONTROLLER, or the single -n0 process) creates
# ``base/runs/<token>/`` at import, compiles only there -- ``<token>/gwN`` per
# worker, ``<token>`` itself for -n0 -- seeds it from ``base/shared`` by hard
# link in pytest_configure, and at exit moves what it compiled back into
# ``base/shared`` and deletes itself. Nothing ever compiles in the shared
# tree, so pruning it cannot reach a live run. The mechanics, the locking
# and the liveness rule are in pytensor_cache_budget.py ("Per-RUN
# compiledirs") and docs/testing-cache.md.
#
# The run directory reaches the workers through this variable, which the
# controller ALWAYS sets when it manages the cache -- to the run directory,
# or to "unmanaged" when it deliberately does not (an explicit
# base_compiledir in PYTENSOR_FLAGS, or no flock on this platform). A worker
# that finds it unset has a controller that did not run this file, which no
# correct setup produces, so that raises rather than guessing a layout.
_RUN_DIR_ENV = "_EXOZIPPY_TEST_RUN_DIR"
_UNMANAGED = "unmanaged"

_worker = os.environ.get("PYTEST_XDIST_WORKER")
_RUN = None  # the controller's RunDir handle
_RUN_DIR = None  # this run's private root, in the controller and the workers
_unmanaged_reason = None


def _user_sets_base_compiledir(flags):
    return any(
        part.strip().startswith("base_compiledir=")
        for part in flags.split(",")
    )


if _BASE_COMPILEDIR is not None:
    if _worker:
        _from_controller = os.environ.get(_RUN_DIR_ENV)
        if _from_controller is None:
            raise RuntimeError(
                f"xdist worker {_worker} found no {_RUN_DIR_ENV} in its "
                "environment: the controller did not run the root conftest, "
                "so this worker cannot know which per-run compiledir is its "
                "own (review 2.13.5)"
            )
        if _from_controller != _UNMANAGED:
            _RUN_DIR = Path(_from_controller)
    else:
        _budget = _load_budget_module()
        if _user_sets_base_compiledir(os.environ.get("PYTENSOR_FLAGS", "")):
            _unmanaged_reason = "PYTENSOR_FLAGS sets base_compiledir itself"
        elif _budget.fcntl is None:  # pragma: no cover - Windows
            _unmanaged_reason = "this platform has no fcntl.flock"
        else:
            _RUN = _budget.create_run_dir(_BASE_COMPILEDIR)
            _RUN_DIR = _RUN.path
        os.environ[_RUN_DIR_ENV] = str(_RUN_DIR) if _RUN_DIR else _UNMANAGED
        if _unmanaged_reason:
            warnings.warn(
                f"the suite's compiledir {_BASE_COMPILEDIR} is UNMANAGED "
                f"because {_unmanaged_reason}: no per-run directory, no "
                "seeding, no budget, and two concurrent suites on that base "
                "can corrupt each other (review 2.13.5)",
                RuntimeWarning,
                stacklevel=1,
            )

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
        # base_compiledir therefore still gets it (and the run is then
        # unmanaged, above).
        _existing = os.environ.get("PYTENSOR_FLAGS", "")
        _target = _BASE_COMPILEDIR if _RUN_DIR is None else _RUN_DIR
        _ours = ",".join(
            [
                f"base_compiledir={shlex.quote(str(_target))}",
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
        _flags = ",".join(p for p in (_ours, _existing) if p)
        if _worker:
            # Per-xdist-worker: see the next section. APPENDED, as the
            # rightmost base_compiledir, and that position is load-bearing:
            # a worker INHERITS the controller's PYTENSOR_FLAGS, which already
            # names the run directory, so anything earlier in the string
            # loses to it. Putting the worker's directory first instead sent
            # every worker of a run into ONE unseeded tree, ``<run>/
            # compiledir_*``: the run stayed correct and recompiled every
            # graph it needed on every run, which is how it was caught (a
            # warm subset 3x slower, 273 entries "promoted" per run that were
            # all duplicates by module hash).
            if _RUN_DIR is not None:
                _worker_base = _RUN_DIR / _worker
            else:
                # UNMANAGED run: the pre-2.13.5 suffix on whatever base won.
                _base = None
                for _part in _flags.split(","):
                    if _part.strip().startswith("base_compiledir="):
                        _base = _part.split("=", 1)[1].strip().strip("'\"")
                _worker_base = Path(_base) / _worker
            _flags = ",".join(
                [_flags, "base_compiledir=" + shlex.quote(str(_worker_base))]
            )
        os.environ["PYTENSOR_FLAGS"] = _flags

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
# alone in ~143 s.  A worker suffix (``<run>/gwN``) removes the shared lock
# entirely; the price is duplicated compiles of common ops across workers,
# paid in parallel instead of in a queue.
#
# The OTHER price: duplicated entries. Each worker tree is seeded from the
# shared one at startup and promotes only what it newly compiled, so the
# duplicates live only for one run and the shared tree holds one copy.


# Filled in by pytest_configure on the controller and read back by
# pytest_report_header and the exit hook. Module globals rather than
# attributes stapled onto ``config``: the controller is a single process, and
# pytest's Config is not ours to grow attributes on.
_compiledir_summary = []
_PLATFORM_NAME = None

# How long a run waits for base/.lock before SKIPPING the step instead.
# Exclusive holders (another run's prune or promotion) hold it for seconds;
# a shared holder is a seeding pass, ~2-30 s. Skipping is always benign --
# see BaseLock.
_PRUNE_LOCK_TIMEOUT = 120.0
_SEED_LOCK_TIMEOUT = 600.0
_PROMOTE_LOCK_TIMEOUT = 600.0


def _xdist_worker(config):
    return hasattr(config, "workerinput")


def _xdist_active(config):
    return bool(config.getoption("numprocesses", 0) or 0)


def _start_run(config):
    """Reap dead runs, bound the shared tree, seed this run. Controller only.

    Before any worker exists, because a worker cannot seed its own
    compiledir: by the time its pytest_configure runs it is about to build
    the ModuleCache it would be seeding.

    PRUNE FIRST, THEN SEED, and the order is load-bearing: pruning brings
    the seed source down to the budget, so a freshly seeded worker starts
    inside it instead of immediately over it.
    """
    global _PLATFORM_NAME
    module = _load_budget_module()
    import pytensor  # noqa: PLC0415 -- must follow the PYTENSOR_FLAGS write above

    # The NAME only. PyTensor derives it from the platform, the processor, the
    # Python version and the bit width; the base it sits under is ours.
    _PLATFORM_NAME = Path(pytensor.config.compiledir).name
    budget = int(os.environ.get(_BUDGET_ENV, _DEFAULT_MAX_ENTRIES))
    base = _BASE_COMPILEDIR
    shared = module.shared_compiledir(base, _PLATFORM_NAME)

    lock = module.BaseLock(base)
    if lock.acquire(exclusive=True, timeout=_PRUNE_LOCK_TIMEOUT):
        try:
            reaped = module.reap_dead_runs(base, keep=_RUN.path)
            pruned = module.enforce_shared_budget(base, _PLATFORM_NAME, budget)
        finally:
            lock.release()
        _compiledir_summary.append(
            module.summarize_tree([(shared, pruned)], budget)
        )
        _compiledir_summary.append(
            f"pytensor run dir {_RUN.path}: {len(reaped.live)} other live "
            f"run(s), {len(reaped.remote)} on other hosts (never touched), "
            f"{len(reaped.reaped)} dead run dir(s) reaped"
        )
    else:
        _compiledir_summary.append(
            f"pytensor compiledir budget NOT enforced this run: {base}/.lock "
            f"was held for more than {_PRUNE_LOCK_TIMEOUT:.0f} s (review "
            "2.13.5; skipping is the benign direction)"
        )

    n_workers = int(config.getoption("numprocesses", 0) or 0)
    started = time.monotonic()
    if not lock.acquire(exclusive=False, timeout=_SEED_LOCK_TIMEOUT):
        print(
            f"pytensor compiledir: {base}/.lock held for more than "
            f"{_SEED_LOCK_TIMEOUT:.0f} s; this run starts COLD rather than "
            "wait longer",
            file=sys.stderr,
            flush=True,
        )
        return
    try:
        source = module.choose_run_seed_source(base, _PLATFORM_NAME)
        if source is None:
            # A genuinely first-ever run: nothing warm to copy from.
            return
        stats = module.seed_run(_RUN, source, _PLATFORM_NAME, n_workers)
    finally:
        lock.release()
    summary = stats.summary()
    if summary:
        # stderr rather than pytest_report_header, and CI is the reason: it
        # runs `pytest -q`, which suppresses the header entirely -- and
        # seeding is the step most worth seeing there, being where a restored
        # cache gets fanned back out, and where a cross-device hard-link
        # fallback would show up as a sudden multi-minute startup.
        print(
            f"{summary}; from {source} in {time.monotonic() - started:.1f} s",
            file=sys.stderr,
            flush=True,
        )


def _finish_run():
    """Promote this run's new entries into the shared tree, then delete it.

    An ``atexit`` handler, registered below at import -- i.e. BEFORE pytensor
    is imported -- and that ordering is the reason it is not
    pytest_unconfigure: atexit runs handlers last-registered-first, so this
    runs AFTER PyTensor's own ModuleCache exit hook (clear_old /
    clear_unversioned on the compiledir), which would otherwise walk a
    directory this has just moved away. It also runs after xdist has torn
    its workers down, on a normal exit and after Ctrl-C alike. A run killed
    outright never gets here; the next run's reap removes its directory.
    """
    global _RUN
    if _RUN is None:
        return
    module = _load_budget_module()
    run = _RUN
    _RUN = None
    try:
        if _PLATFORM_NAME is not None:
            shared = module.shared_compiledir(_BASE_COMPILEDIR, _PLATFORM_NAME)
            lock = module.BaseLock(_BASE_COMPILEDIR)
            if lock.acquire(exclusive=True, timeout=_PROMOTE_LOCK_TIMEOUT):
                try:
                    stats = module.promote_run(run, shared, _PLATFORM_NAME)
                finally:
                    lock.release()
                if stats.promoted or stats.incomplete:
                    print(stats.summary(shared), file=sys.stderr, flush=True)
            else:
                print(
                    f"pytensor compiledir: {_BASE_COMPILEDIR}/.lock held for "
                    f"more than {_PROMOTE_LOCK_TIMEOUT:.0f} s; this run's new "
                    "entries are NOT promoted to the shared tree",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        run.close()
        module.remove_run_dir(run)


if _RUN is not None:
    atexit.register(_finish_run)


def _warm_module_cache():
    """Pay the ModuleCache walk here, where no test can be blamed for it.

    ``pytest_configure`` runs before pytest-timeout arms its per-test SIGALRM,
    so however long the walk takes it cannot fail a test. Before this existed
    the walk was paid lazily by whichever test compiled first in each worker,
    which is exactly how a 300 s timeout landed on an unrelated vcve test.
    """
    try:
        import pytensor  # noqa: PLC0415 -- see _start_run
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
    if _BASE_COMPILEDIR is None:
        return

    if _xdist_worker(config):
        # Every worker builds its own ModuleCache, so every worker has to warm
        # its own. Each walks only its own seeded tree.
        _warm_module_cache()
        return

    if _RUN is not None:
        _start_run(config)
    if not _xdist_active(config):
        # -n0: this process runs the tests itself, so it is also the one that
        # needs the cache warm.
        _warm_module_cache()


def pytest_report_header(config):
    return list(_compiledir_summary)


# ---------------------------------------------------------------------------
# Name the compiledir race when it fires anyway (review 2.13.5)
# ---------------------------------------------------------------------------
# The per-run layout above removes the race between two suites that BOTH run
# this conftest. It cannot protect a suite from a checkout that predates it,
# from an unmanaged base (see _unmanaged_reason), or from anything else that
# deletes a live compiledir. When that happens the cost is not the red run --
# it is that a phantom red is indistinguishable from a real one. So say what
# it is; report_compiledir_race has the signature and the triage rule.


def pytest_terminal_summary(terminalreporter):
    _load_budget_module().report_compiledir_race(terminalreporter)
