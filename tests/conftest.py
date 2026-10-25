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
    # No raw config: no orbit types are declared, so every orbit is the
    # keplerian default (Planet.keplerian_geometry reads them from here).
    system_config = {}

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
    at_eep=None,
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

    ``at_eep`` is ``{column: {eep: value}}``: the value every track carries
    in that column at that EEP, replacing the smooth default -- for pinning
    the Jacobian's clip against a table defect, or the dragon penalty.

    Returns the directory to pass as the component's ``model_root:`` -- the
    ``MIST`` directory itself (``<model_root>/<model>/EEPs``), mirroring
    ``mist_grid.DEFAULT_MIST_MODEL_ROOT``.  Both file paths come from
    ``mist_grid`` (``grid_path``, ``grid_yaml_path``), the component's own
    spelling of them, never a copy (review 4.8.4).
    """
    import pandas as pd

    from exozippy.components.evolutionarymodel import mist_grid

    mist_root = os.path.join(os.fspath(root), "MIST")
    parquet = mist_grid.grid_path(
        model=model, alpha=alpha, vvcrit=vvcrit, model_root=mist_root
    )
    parquet.parent.mkdir(parents=True, exist_ok=True)
    at_eep = dict(at_eep or {})

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
                        "dEEP_dage": at_eep.get("dEEP_dage", {}).get(
                            eep, speed
                        ),
                        "here_be_dragons": at_eep.get(
                            "here_be_dragons", {}
                        ).get(eep, 0.0),
                    }
                )

    pd.DataFrame(rows).to_parquet(parquet, engine="pyarrow", index=False)
    # The component reads <model>.grid.yaml beside the parquet for the prose
    # citation (see models/MIST/MISTv2.5/EEPs/MISTv2.5.grid.yaml); only the
    # keys it consumes are written.
    yaml_path = mist_grid.grid_yaml_path(model=model, model_root=mist_root)
    with open(yaml_path, "w") as f:
        f.write(f'model: {model}\ncitation: "Dotter:2016, Choi:2016"\n')
    return mist_root


# Synthetic Claret coefficients: LINEAR in every axis, so the trilinear
# interpolation reproduces them exactly and a test can assert a value
# without a hand-computed constant of its own (docs/testing.md).
SYNTHETIC_CLARET = {
    "a": lambda t, g, z: (
        0.40 - 2.0e-5 * (t - 5000.0) + 0.01 * (g - 4.0) + 0.02 * z
    ),
    "b": lambda t, g, z: (
        0.25 + 1.0e-5 * (t - 5000.0) - 0.01 * (g - 4.0) - 0.01 * z
    ),
    "u": lambda t, g, z: (
        0.55 - 3.0e-5 * (t - 5000.0) + 0.01 * (g - 4.0) + 0.01 * z
    ),
}


def write_synthetic_claret_tables(root, bands=("TESS", "V", "I")):
    """Write Claret limb-darkening tables in the published layout and return
    the directory -- to be patched in as ``claret_tables.DEFAULT_TABLE_DIR``.

    The real tables are fetched from Zenodo on first use, so no unit test
    may depend on them.  These span exactly the box
    (claret_tables.BOX) on the published axes, with the SYNTHETIC_CLARET
    coefficients; a quadratic table for every band and a linear one for
    every band that has one (claret_tables.TABLE_BANDS).
    """
    import pandas as pd

    from exozippy.models.Claret import claret_tables as ct

    teff = np.arange(3500.0, 10000.0 + 1.0, 250.0)
    logg = np.arange(2.0, 5.0 + 0.01, 0.5)
    feh = np.array(
        [
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.5,
            1.0,
        ]
    )
    T, G, Z = np.meshgrid(teff, logg, feh, indexing="ij")
    os.makedirs(root, exist_ok=True)
    for band in bands:
        for law, columns in ct.LAW_COLUMNS.items():
            if band not in ct.TABLE_BANDS[law]:
                continue
            df = pd.DataFrame(
                {"teff": T.ravel(), "logg": G.ravel(), "feh": Z.ravel()}
            )
            for c in columns:
                df[c] = SYNTHETIC_CLARET[c](df["teff"], df["logg"], df["feh"])
            df.attrs["meta"] = {"band": band, "law": law, "synthetic": True}
            df.to_parquet(ct.table_path(band, law, root), index=False)
    return root


@pytest.fixture
def synthetic_claret_tables(tmp_path, monkeypatch):
    """Point every Claret table reader at freshly written synthetic tables."""
    from exozippy.models.Claret import claret_tables as ct

    root = write_synthetic_claret_tables(tmp_path / "claret")
    monkeypatch.setattr(ct, "DEFAULT_TABLE_DIR", root)
    # Pin the synthetic files as if they were the published ones, so the
    # readers go through the real ensure_tables md5 check, not around it.
    pins = {
        f.name: {"size": f.stat().st_size, "md5": ct._md5(f)}
        for f in sorted(root.glob("*.parquet"))
    }
    monkeypatch.setattr(ct, "_TABLE_FILES", pins)
    monkeypatch.setattr(ct, "_verified", set())
    return root


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


def _prewarm_zenodo_records(config):
    """Put every registered Zenodo record's files in place, once.

    None of them is tracked: the NextGen spectra and BC tables, the MIST EEP
    grid and the Claret limb-darkening tables are all fetched by their
    loaders on first use, and `_no_shared_download_cache` switches the
    machine cache off for every TEST -- so without this a fresh worktree
    would download them inside whichever test first reads one, under that
    test's pytest-timeout, and a Zenodo outage would fail the pre-push suite
    (review 2.13.4). On CI the 128 MB MIST grid did exactly that: a trickling
    transfer inside an example test ran into the 600 s timeout on two runners
    at once.

    Run here, before any test and with the real environment (the machine
    cache ON), every fetch is a hard link from the machine cache when it is
    warm and a stat when the tree already has the file. The set of records
    is zenodo_assets.RECORDS and each is fetched by its owner's own
    fetch_all() (zenodo.fetch_every_record), so a new record is covered
    without touching this file -- and a record with no owner RAISES here,
    naming it.

    Controller only, before xdist spawns its workers, so they all find the
    files in place. A network failure is REPORTED, not raised: the suite also
    holds hundreds of tests that never read these files, and each test that
    does re-raises the loader's own error (file, record, how to pre-fetch)
    when it gets there.
    """
    from exozippy.utilities import zenodo

    try:
        zenodo.fetch_every_record()
    except RuntimeError as e:
        import warnings

        warnings.warn(
            f"The Zenodo data files could not all be put in place before the "
            f"suite; every test that reads a missing one will fail with the "
            f"reason: {e}",
            stacklevel=1,
        )


def pytest_configure(config):
    """Point the acceptance dump at a fresh directory (controller only).

    Set in the environment rather than on `config` so xdist WORKERS inherit
    it: they are spawned after this hook runs, and the recording happens in
    the worker processes.
    """
    if hasattr(config, "workerinput"):
        return  # an xdist worker; the controller has already done this
    _prewarm_zenodo_records(config)
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
