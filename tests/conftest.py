"""Shared test helpers.

Plain classes (not fixtures) — imported explicitly by test files that need them.
Pytest adds the tests/ directory to sys.path, so ``from conftest import ...`` works.
"""

import os

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
