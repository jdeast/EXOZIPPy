"""
Integration test suite for ParallaxGridSearch (no mocking).

All tests run against the real mmexo.fitters.SFitFitter and real OGLE
photometry data loaded from disk via MULENS_DATA_PATH.

Test classes
------------
TestCoarseGrid       — coarse grid evaluation only (refine=False)
TestExpansion        — edge-driven grid expansion followed by refinement
TestSkipOptimization — skip_optimization=True flag behavior
TestFindLocalMinima  — find_local_minima() preconditions
TestRefinement       — refinement around minima and save/load round-trip
TestBest             — best property in all four states
TestSaveLoad         — save/load error and warning paths
TestPlaceholders     — documented coverage gaps (all skipped)

See TestPlaceholders for behaviors not yet tested.
"""

import os
import numpy as np
import pytest
from exozippy import MULENS_DATA_PATH
from exozippy.mmexofast.gridsearches import ParallaxGridSearch, BaseRectGridSearch
from MulensModel import MulensData
import exozippy.mmexofast as mmexo


# ---------------------------------------------------------------------------
# Block 1 — Data File and Dataset
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(
    MULENS_DATA_PATH,
    'OB140939',
    'n20100310.I.OGLE.OB140939.txt',
)

# Loaded once at module import time; shared by all tests in this suite.
_MULENS_KWARGS = mmexo.observatories.get_kwargs(DATA_FILE)
_DATASET = MulensData(file_name=DATA_FILE, **_MULENS_KWARGS)
DATASETS = [_DATASET]


# ---------------------------------------------------------------------------
# Block 2 — Static Parameters
# ---------------------------------------------------------------------------

# Fixed model parameters used in every ParallaxGridSearch instantiation
# unless explicitly overridden by a test.
STATIC_PARAMS = {
    't_0': 2456836.1933785137,
    't_E': 22.471177939504067,
    'u_0': 0.9461578773096889,
}


# ---------------------------------------------------------------------------
# Block 3 — Known Reference Values
# ---------------------------------------------------------------------------

# Precomputed ground-truth values for OB140939.
# Used only in assertions where we have strong prior knowledge.
KNOWN_PI_E_N       = 3.8925249891046954
KNOWN_PI_E_E       = -2.720754113010126
KNOWN_CHI2_MINIMUM = 1247.1694346729575

# chi2 at (pi_E_E=0, pi_E_N=0) with skip_optimization=True:
KNOWN_CHI2_ORIGIN_SKIPOPT = 1269.502449


# ---------------------------------------------------------------------------
# Block 4 — Grid Specifications
# ---------------------------------------------------------------------------

# 5x5 grid, step=0.5, range [-1.0, 1.0] in both axes.
# Does NOT contain the true minimum (pi_E_N≈3.89, pi_E_E≈-2.72).
# With refine=False: minimum appears on the grid boundary.
# With refine=True: boundary minimum triggers expansion toward true minimum.
COARSE_GRID_PARAMS = {
    'pi_E_E': [-1.0, 1.0, 0.5],
    'pi_E_N': [-1.0, 1.0, 0.5],
}

# 3x3 grid, step=5. True minimum region is covered.
# Nearest grid point to true minimum: (pi_E_N=5, pi_E_E=-5).
# Step=5 exceeds 1σ uncertainties (σ_N≈3.7, σ_E≈3.1), so the coarse grid
# alone fails the convergence criterion; genuine refinement is required.
REFINEMENT_GRID_PARAMS = {
    'pi_E_N': [0.0, 10.0, 5.0],
    'pi_E_E': [-10.0, 0.0, 5.0],
}

# 3x3 tight grid around the true minimum.
# Center (pi_E_N=3.9, pi_E_E=-2.7) is the nearest grid point to the true
# minimum (3.8925, -2.7208). Used to test best property fallback when refine=False.
BEST_FALLBACK_GRID_PARAMS = {
    'pi_E_N': [2.9, 4.9, 1.0],
    'pi_E_E': [-3.7, -1.7, 1.0],
}

# 3x3 grid for skip_optimization tests.
# Axis indexing:
#   pi_E_E: [0.0, 0.5, 1.0]  → index 0 is pi_E_E=0.0
#   pi_E_N: [-1.0, -0.5, 0.0] → index 2 is pi_E_N=0.0
# Therefore the origin (pi_E_E=0, pi_E_N=0) lives at chi2_grid[0, 2].
SKIP_OPT_GRID_PARAMS = {
    'pi_E_E': [0.0, 1.0, 0.5],
    'pi_E_N': [-1.0, 0.0, 0.5],
}


# ---------------------------------------------------------------------------
# Module-level tests
# ---------------------------------------------------------------------------

def test_imports():
    """Smoke test: verify the full import chain is healthy before any test runs."""
    assert MULENS_DATA_PATH is not None
    assert ParallaxGridSearch is not None
    assert BaseRectGridSearch is not None
    assert MulensData is not None
    assert mmexo is not None


def test_data_file_exists():
    """
    Verify the OGLE photometry file is present at the expected path.

    Every test in this suite depends on this file. A missing file causes a
    collection error at module import (MulensData raises). This test provides
    a clearer diagnostic by checking the path explicitly before anything else.
    """
    assert os.path.isfile(DATA_FILE), (
        f"Required data file not found: {DATA_FILE}\n"
        f"MULENS_DATA_PATH resolves to: {MULENS_DATA_PATH}\n"
        "Check the data directory structure and MULENS_DATA_PATH definition."
    )
