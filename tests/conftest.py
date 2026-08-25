"""Shared test helpers.

Plain classes (not fixtures) — imported explicitly by test files that need them.
Pytest adds the tests/ directory to sys.path, so ``from conftest import ...`` works.
"""

import os

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


def write_synthetic_mist_grid(
    root,
    masses=(0.5, 1.0, 2.0),
    initfehs=(-0.5, 0.0, 0.5),
    eeps=(1, 300, 454, 605, 807),
    model="MISTv2.5",
    alpha=0.0,
    vvcrit=0.0,
):
    """Write a tiny MIST track grid and return the ``model_root`` for it.

    The shipped grid is a ~130 MB parquet that is gitignored (see
    mist_grid.load_mist_grid's FileNotFoundError), so no test may depend on
    it.  This writes a handful of tracks in the same layout and column set,
    for tests that need the evolutionarymodel component to actually load and
    interpolate something.

    The tracks are crude but monotone and physically ordered: radius and age
    grow with EEP, teff falls, feh_mist sits just below initfeh, and
    dEEP_dage is smallest in the middle (the "main sequence"), which is what
    the EEP -> age Jacobian is supposed to reward.

    Returns the directory to pass as the component's ``model_root:``.
    """
    import pandas as pd

    root = os.fspath(root)
    eep_dir = os.path.join(root, "MIST", model, "EEPs")
    os.makedirs(eep_dir, exist_ok=True)

    n_eep = len(eeps)
    rows = []
    for mass in masses:
        for feh in initfehs:
            for k, eep in enumerate(eeps):
                frac = k / max(n_eep - 1, 1)
                # Slowest in the middle of the track, fast at both ends.
                speed = 1e-9 * (1.0 + 50.0 * abs(frac - 0.5))
                rows.append(
                    {
                        "mass": float(mass),
                        "EEP": int(eep),
                        "initfeh": float(feh),
                        "feh_mist": float(feh) - 0.02 * frac,
                        "radius_mist": float(mass) * (1.0 + 3.0 * frac),
                        "teff_mist": 6000.0 * mass**0.5 - 1500.0 * frac,
                        "age_mist": 1.0e9 * (0.1 + 10.0 * frac) / mass**2,
                        "dEEP_dage": speed,
                        "here_be_dragons": 0.0,
                    }
                )

    fname = f"afe_p{abs(alpha) * 10:.0f}_vvcrit{vvcrit:0.1f}.grid.parquet"
    pd.DataFrame(rows).to_parquet(
        os.path.join(eep_dir, fname), engine="pyarrow", index=False
    )
    return root
