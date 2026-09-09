"""Shared test helpers.

Plain classes (not fixtures) — imported explicitly by test files that need them.
Pytest adds the tests/ directory to sys.path, so ``from conftest import ...`` works.
"""

import glob
import json
import os
import platform
import shutil
import tempfile

import numpy as np
import pytensor.tensor as pt
import pytest

from exozippy.components.parameter import Parameter
from exozippy.config import ConfigManager

# The PTDE samplers build their worker pools with
# multiprocessing.get_context("fork"), which raises
# `ValueError: cannot find context for 'fork'` on Windows -- fork simply does
# not exist there, and the only alternative, "spawn", re-imports the module in
# each worker and requires all worker state to be picklable. Converting them is
# a real piece of work, not a portability tweak, so these tests are skipped
# rather than left permanently red.
#
# This is not only a Windows concern: Python 3.14 deprecates fork in
# multi-threaded processes, and the ubuntu CI logs already emit
# "DeprecationWarning: This process is multi-threaded, use of fork() may lead
# to deadlocks in the child". The eventual fix is fork -> spawn everywhere.
requires_fork = pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason=(
        "PTDE uses multiprocessing's fork start method, which does not exist "
        "on this platform (see conftest.requires_fork)"
    ),
)


@pytest.fixture(autouse=True)
def _no_shared_download_cache(monkeypatch):
    """Keep the machine-level Zenodo cache out of the test suite by default.

    utilities/zenodo.py caches large downloads under ~/.cache/exozippy and
    adopts an already-present destination into it. Both are exactly what we
    want in a fit and exactly what we do not want in a test: the suite would
    md5 (and possibly copy) the real 250 MB NextGen spectra, and tests using
    fake payloads would leave entries in the developer's own cache.

    Tests that exercise the cache opt back in by pointing EXOZIPPY_CACHE_DIR
    at a tmp_path of their own; with it switched off here, everything else
    behaves exactly as it did before the cache existed. The module-level
    latches are reset too, so one test's unwritable-cache warning cannot
    leak into the next.
    """
    from exozippy.utilities import zenodo

    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", "")
    # raising=False so this fixture keeps working against a zenodo.py that
    # predates the cache (bisects, and the pre-fix run that proves the cache
    # tests really do fail without it).
    monkeypatch.setattr(zenodo, "_cache_disabled_reason", None, raising=False)
    monkeypatch.setattr(zenodo, "_adoption_attempted", set(), raising=False)


class _DummyConfigManager:
    """Minimal ConfigManager stub for tests that only need a no-op hint interface."""

    user_params = {}

    def add_hint(self, *args, **kwargs):
        pass

    def add_scale_hint(self, *args, **kwargs):
        pass

    def seed_start_value(self, path, seed=0):
        # No seed hints in the stub (the real ConfigManager returns None for
        # a path no seed set carries).
        return None


class _DummyComponent:
    """Stub component whose only observable property is n_elements."""

    def __init__(self, n_elements):
        self.n_elements = n_elements


class _DummySystem:
    """Empty system namespace for tests that attach attributes manually."""

    pass


class _MockParam:
    """Minimal Parameter stand-in: initval, a PyTensor value, hard bounds.

    Shared rather than per-module because two suites (test_galactic_model and
    test_ffp_mass_function) exercise the same GalacticModel.build_likelihood
    and so need the same stand-in.  Keeping a copy in each was a live
    landmine: every change to the part of Parameter's surface that
    build_likelihood reads had to be mirrored by hand, and PR #117 shipped
    with only one copy updated, reddening CI on a suite it had not touched.
    """

    def __init__(self, initval, lower=None, upper=None, is_sampled=None):
        self.initval = np.atleast_1d(np.asarray(initval, dtype=np.float64))
        self.value = pt.as_tensor_variable(self.initval)
        self.lower = lower
        self.upper = upper
        # build_pymc's per-element sampled mask.  None = the model has not
        # been built, which Parameter.element_is_sampled reads as "not
        # sampled" -- the same conservative answer.
        self.is_sampled = is_sampled
        self.prior_contributions = []

    def element_start(self, index=0):
        arr = self.initval
        return float(arr[index] if arr.size > index else arr[0])

    def element_is_sampled(self, index=0):
        if self.is_sampled is None:
            return False
        mask = np.atleast_1d(self.is_sampled)
        return bool(mask[index] if mask.size > index else mask[0])

    def add_prior_contribution(self, *args, **kwargs):
        """Reporting-only hook (see parameter.PriorContribution).

        build_likelihood declares what its potentials ARE so the reported
        tables can describe them; the declaration changes no math.  Recorded
        rather than dropped so a test can assert on it.
        """
        self.prior_contributions.append((args, kwargs))


class MockSystem:
    """Minimal System mock for ConfigManager and ModelAuditor tests.

    Usage::
        system = MockSystem(user_params)
        system.star = Star([...], system.config_manager)
    """

    def __init__(self, user_params):
        self.user_params = user_params
        self.config_manager = ConfigManager(user_params)
        self.star = None

    def get_parameter_lookup(self):
        return {p.label: p for p in self.get_all_parameters()}

    def get_all_parameters(self):
        if self.star is None:
            return []
        return [
            v for v in self.star.__dict__.values() if isinstance(v, Parameter)
        ]


# ---------------------------------------------------------------------------
# Acceptance delta dump (review 3.14.20)
# ---------------------------------------------------------------------------
# macOS and Linux build slightly different models -- `scipy.optimize.nnls` at
# build time disagrees between LAPACK implementations, and it sets seeds,
# scales AND bounds.  The effect is deterministic per platform (repeat runs on
# one machine are byte-identical), but whether macOS sits CONSISTENTLY one
# side of Linux or scatters was never established, and it matters: a
# directional offset does not average out over draws, scatter largely does.
#
# The acceptance tolerance is loose enough that macOS passes silently, so
# these hooks print every term's delta and a SIGN TALLY whether or not
# anything failed.  Numbers only; the verdict depends on them.
_DELTA_DIR_ENV = "EXOZIPPY_DELTA_DIR"


def pytest_configure(config):
    """Point the acceptance dump at a fresh directory (controller only).

    Set in the environment rather than on `config` so xdist WORKERS inherit
    it: they are spawned after this hook runs, and the recording happens in
    the worker processes.
    """
    if hasattr(config, "workerinput"):
        return  # an xdist worker; the controller has already done this
    directory = os.environ.get(_DELTA_DIR_ENV)
    if not directory:
        directory = os.path.join(
            tempfile.gettempdir(), "exozippy_acceptance_deltas"
        )
        os.environ[_DELTA_DIR_ENV] = directory
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the per-term deltas and the sign tally that answers 3.14.20."""
    if hasattr(config, "workerinput"):
        return
    directory = os.environ.get(_DELTA_DIR_ENV)
    if not directory or not os.path.isdir(directory):
        return

    rows = []
    for entry in sorted(glob.glob(os.path.join(directory, "deltas-*.jsonl"))):
        with open(entry) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    if not rows:
        return

    write = terminalreporter.write_line
    write("")
    write("=" * 72)
    write("acceptance delta dump -- review 3.14.20 (platform comparison)")
    write("=" * 72)
    write(
        "The fixtures were recorded on the Linux reference machine, so ON "
        "THAT MACHINE"
    )
    write(
        "these deltas are ~0 by construction.  A log of zeros here means "
        "'this is the"
    )
    write(
        "reference platform', NOT 'no cross-platform difference exists' -- "
        "compare a"
    )
    write("macOS run against a Linux one.")
    write("")
    write("platform : %s" % platform.platform())
    write("python   : %s" % platform.python_version())
    try:
        import numpy

        write("numpy    : %s" % numpy.__version__)
    except Exception:
        pass
    try:
        import scipy

        write("scipy    : %s" % scipy.__version__)
    except Exception:
        pass
    try:
        # Which BLAS/LAPACK actually backs nnls is the mechanism at issue.
        import numpy as _np

        cfg = _np.__config__.show(mode="dicts")
        blas = (cfg or {}).get("Build Dependencies", {}).get("blas", {})
        lapack = (cfg or {}).get("Build Dependencies", {}).get("lapack", {})
        if blas or lapack:
            write(
                "blas     : %s %s"
                % (blas.get("name", "?"), blas.get("version", ""))
            )
            write(
                "lapack   : %s %s"
                % (lapack.get("name", "?"), lapack.get("version", ""))
            )
    except Exception:
        pass
    write("")

    def rel(row):
        before = row["before"]
        denom = abs(before) if abs(before) > 0 else 1.0
        return (row["after"] - before) / denom

    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    write("per fixture:")
    write(
        "  %-26s %6s  %12s  %s" % ("case", "terms", "max |rel|", "worst term")
    )
    for case in sorted(by_case):
        case_rows = by_case[case]
        worst = max(case_rows, key=lambda r: abs(rel(r)))
        write(
            "  %-26s %6d  %12.3e  %s"
            % (case, len(case_rows), abs(rel(worst)), worst["term"])
        )
    write("")

    above = sum(1 for r in rows if r["after"] > r["before"])
    below = sum(1 for r in rows if r["after"] < r["before"])
    exact = sum(1 for r in rows if r["after"] == r["before"])
    write("sign tally over %d terms:" % len(rows))
    write("  current > recorded : %d" % above)
    write("  current < recorded : %d" % below)
    write("  bit-identical      : %d" % exact)
    write(
        "  (one-sided counts => a DIRECTIONAL offset, which does not average "
        "out;"
    )
    write(
        "   mixed counts => scatter.  All-identical => this is the reference "
        "machine.)"
    )
    write("")

    moved = sorted(
        (r for r in rows if r["after"] != r["before"]),
        key=lambda r: abs(rel(r)),
        reverse=True,
    )
    if moved:
        write("largest %d deltas:" % min(10, len(moved)))
        for row in moved[:10]:
            write(
                "  %-22s %-30s %+.6e  rel %+.3e"
                % (
                    row["case"],
                    row["term"],
                    row["after"] - row["before"],
                    rel(row),
                )
            )
    write("=" * 72)
