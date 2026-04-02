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
# Block 1 — Data File, Coordinates, and Dataset
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(
    MULENS_DATA_PATH,
    'OB140939',
    'n20100310.I.OGLE.OB140939.txt',
)

# Sky coordinates for OB140939, required by ParallaxGridSearch via
# fitter_kwargs. Loaded from the coords.txt file that lives alongside
# the photometry data.
COORDS_FILE = os.path.join(
    MULENS_DATA_PATH,
    'OB140939',
    'coords.txt',
)
with open(COORDS_FILE) as _f:
    COORDS = _f.read().strip()

# fitter_kwargs passed to every ParallaxGridSearch instantiation in this
# suite. Only coords is required; all other SFitFitter keyword arguments
# (mag_methods, limb_darkening_coeffs_u, limb_darkening_coeffs_gamma,
# fix_source_flux, fix_blend_flux) are left as None (their defaults).
FITTER_KWARGS = {'coords': COORDS}

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
# Module-scoped shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def refinement_result(tmp_path_factory):
    """
    Run a ParallaxGridSearch with refine=True on REFINEMENT_GRID_PARAMS once
    per module. Save results to a temporary file and cache both.

    SHARED BY: TestRefinement, TestBest, TestSaveLoad.
    PLACEMENT: Module level — pytest cannot share class-scoped fixtures
    across class boundaries.

    READ-ONLY CONTRACT: All consuming tests MUST NOT call run() or mutate
    the returned instance. Mutation corrupts every subsequent test in this
    module that uses this fixture.

    Grid design rationale:
      - REFINEMENT_GRID_PARAMS: 3x3, step=5
      - True minimum: pi_E_N≈3.8925, pi_E_E≈-2.7208 (between grid points)
      - Nearest grid point: (pi_E_N=5, pi_E_E=-5)
      - Step 5 > 1σ uncertainties (σ_N≈3.7, σ_E≈3.1); coarse grid cannot
        satisfy convergence (n//2=1 point per side within Δχ²=1). At least
        one refinement sub-grid is computed.
      - After one refinement (step≈5/3≈1.67 < 1σ), convergence is met at
        level 1.

    Returns:
        dict with keys:
            'instance': ParallaxGridSearch after run(refine=True, ...)
            'filepath': pathlib.Path to saved JSON results file
    """
    tmp_dir = tmp_path_factory.mktemp("parallax_refinement")
    filepath = tmp_dir / "refinement_results.json"

    searcher = ParallaxGridSearch(
        static_params=STATIC_PARAMS,
        datasets=DATASETS,
        grid_params=REFINEMENT_GRID_PARAMS,
        fitter_kwargs=FITTER_KWARGS,
    )

    searcher.run(
        refine=True,
        point_density_in_minimum=3,
        max_refinements=2,
        max_expansions=4,
    )

    searcher.save_results(filepath)

    return {"instance": searcher, "filepath": filepath}


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


def test_refinement_fixture_runs(refinement_result):
    """
    Verify the module-scoped refinement fixture executes without error and
    produces a populated ParallaxGridSearch instance with a saved file.

    This test runs at module level so fixture failures surface before
    TestRefinement or TestBest are reached, giving clearer error attribution.
    """
    instance = refinement_result["instance"]
    assert instance is not None
    assert instance.results_history is not None
    assert len(instance.results_history) > 0
    filepath = refinement_result["filepath"]
    assert filepath.exists(), f"Expected saved file at {filepath}"


# ---------------------------------------------------------------------------
# TestCoarseGrid
# ---------------------------------------------------------------------------

class TestCoarseGrid:
    """
    Tests for run(refine=False) on COARSE_GRID_PARAMS.

    Grid: 5x5, step=0.5, range [-1.0, 1.0] in both pi_E_E and pi_E_N axes.
    The true minimum (pi_E_N≈3.8925, pi_E_E≈-2.7208) lies far outside this
    range. With refine=False the grid is evaluated exactly as specified and
    no expansion or refinement is performed. Because the chi2 surface slopes
    toward the true minimum, the lowest chi2 within the grid will appear on
    the boundary rather than in the interior.
    """

    @pytest.fixture(scope="class")
    def coarse_searcher(self):
        """
        Instantiate and run ParallaxGridSearch with refine=False on
        COARSE_GRID_PARAMS. Returned once per class and cached.

        READ-ONLY CONTRACT: test_runs_without_error and test_minimum_is_on_edge
        must not call run() or mutate the returned instance. Mutation will
        corrupt the shared state for the other test in this class.
        """
        searcher = ParallaxGridSearch(
            static_params=STATIC_PARAMS,
            datasets=DATASETS,
            grid_params=COARSE_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
        )
        searcher.run(refine=False)
        return searcher

    def test_runs_without_error(self, coarse_searcher):
        """
        run(refine=False) completes and populates results_history with a
        fully evaluated 5x5 chi2 grid.

        Grid dimensions:
          pi_E_E axis: 5 points [-1.0, -0.5, 0.0, 0.5, 1.0]
          pi_E_N axis: 5 points [-1.0, -0.5, 0.0, 0.5, 1.0]

        Every cell must be evaluated (no None entries in result_grid).
        """
        assert coarse_searcher.results_history is not None

        chi2_grid = coarse_searcher.results_history[0]['chi2_grid']
        assert chi2_grid.shape == (5, 5), (
            f"Expected chi2_grid shape (5, 5); got {chi2_grid.shape}"
        )

        result_grid = coarse_searcher.results_history[0]['result_grid']
        assert all(cell is not None for cell in result_grid.ravel()), (
            "Every grid cell must be evaluated; found None entries"
        )

    def test_minimum_is_on_edge(self, coarse_searcher):
        """
        The best coarse-grid minimum lies on the grid boundary (±1.0).

        The true minimum is far outside [-1, 1]; the chi2 surface slopes
        toward it, so within the grid the lowest chi2 is at a boundary point.

        Note: find_local_minima() returns a list of (chi2, params_dict, level)
        tuples sorted best-first. self.minima stays None because refine=False.
        """
        minima = coarse_searcher.find_local_minima()
        assert len(minima) >= 1, "Expected at least one minimum"

        _, best_params, _ = minima[0]
        pi_e_e = best_params['pi_E_E']
        pi_e_n = best_params['pi_E_N']

        assert (pi_e_e in (-1.0, 1.0) or pi_e_n in (-1.0, 1.0)), (
            f"Expected best grid point on boundary (±1.0); "
            f"got pi_E_N={pi_e_n}, pi_E_E={pi_e_e}"
        )