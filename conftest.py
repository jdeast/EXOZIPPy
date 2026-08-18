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

# One full cold run of this suite creates roughly 1000 entries, so the budget
# has to be a comfortable multiple of that or every run would evict what the
# next one needs and the cache would never be warm. 3000 holds about three
# runs' worth of divergent branches -- enough that switching between two or
# three worktrees stays warm -- while keeping the startup walk to about a
# quarter of the 4035-entry disaster this replaces.
_DEFAULT_MAX_ENTRIES = 3000

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
        _ours = f"base_compiledir={shlex.quote(str(_BASE_COMPILEDIR))}"
        os.environ["PYTENSOR_FLAGS"] = ",".join(
            p for p in (_ours, _existing) if p
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
    """Bound the suite's compiledir. Controller only, before workers exist.

    Doing it here rather than in a fixture is what makes the prune safe
    without taking PyTensor's compile lock: ``pytest_configure`` on the
    controller runs before xdist spawns a single worker, so nothing else is
    walking the directory yet.
    """
    budget = int(os.environ.get(_BUDGET_ENV, _DEFAULT_MAX_ENTRIES))
    module = _load_budget_module()
    if module is None:  # pragma: no cover - defensive
        return None
    import pytensor  # noqa: PLC0415 -- must follow the PYTENSOR_FLAGS write above

    stats = module.enforce_budget(
        Path(pytensor.config.base_compiledir),
        Path(pytensor.config.compiledir),
        budget,
        # Safe here and only here: this base_compiledir belongs to the test
        # suite alone, so a sibling compiledir_* tree is a stranded kernel or
        # Python version that nothing will ever read again.
        sweep_platforms=True,
    )
    return stats.summary(Path(pytensor.config.compiledir), budget)


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

    _compiledir_summary = _prune_compiledir(config)
    if not _xdist_active(config):
        # -n0: this process runs the tests itself, so it is also the one that
        # needs the cache warm.
        _warm_module_cache()


def pytest_report_header(config):
    return [_compiledir_summary] if _compiledir_summary else []
