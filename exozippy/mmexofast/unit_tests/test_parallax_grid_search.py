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
TestFitterKwargs     — fitter_kwargs validation at construction time
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

COORDS_FILE = os.path.join(
    MULENS_DATA_PATH,
    'OB140939',
    'coords.txt',
)
with open(COORDS_FILE) as _f:
    COORDS = _f.read().strip()

# fitter_kwargs passed to every ParallaxGridSearch instantiation.
# Only coords is required; all other SFitFitter keyword arguments
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

# No-parallax solution values for OB140939.
# Used for grids centered on or near the origin (pi_E_E≈0, pi_E_N≈0).
STATIC_PARAMS = {
    't_0': 2456836.1933785137,
    't_E': 22.471177939504067,
    'u_0': 0.9461578773096889,
}

# Parallax (u0+) solution values for OB140939.
# Used for grids centered on or containing the true parallax minimum
# (pi_E_N≈3.89, pi_E_E≈-2.72). Dramatically reduces optimizer travel
# distance: the parallax solution has u_0≈3.12 and t_E≈9.26, a 230% and
# 59% displacement from STATIC_PARAMS through a flat likelihood landscape.
STATIC_PARAMS_PAR = {
    't_0': 2456836.241149,
    't_E': 9.263272,
    'u_0': 3.123346,
}


# ---------------------------------------------------------------------------
# Block 3 — Known Reference Values
# ---------------------------------------------------------------------------

# Precomputed ground-truth values for OB140939.
# Used only in assertions where we have strong prior knowledge.
KNOWN_PI_E_N       = 3.8925249891046954
KNOWN_PI_E_E       = -2.720754113010126
KNOWN_CHI2_MINIMUM = 1246.865

# The OB140939 parallax chi2 landscape is very shallow: total chi2
# variation across the minimum region is < 1.0, and solutions drift by
# ~0.3 chi2 units depending on optimizer starting point and trajectory.
# KNOWN_CHI2_MINIMUM is approximate; KNOWN_PI_E_N and KNOWN_PI_E_E are
# within Δχ²=1.0 of KNOWN_CHI2_MINIMUM.

# Minimum expected chi2 improvement over the no-parallax solution.
# True improvement ≈ 22.6 (1269.502449 - 1246.865); threshold of 22.0
# gives ~0.6 units of margin for optimizer drift in a shallow landscape.
# Used in place of tight assertions against KNOWN_CHI2_MINIMUM where the
# shallow landscape makes exact values unreliable.
CHI2_IMPROVEMENT_MIN = 22.0

# chi2 at (pi_E_E=0, pi_E_N=0) with skip_optimization=True:
KNOWN_CHI2_ORIGIN_SKIPOPT = 1269.502449


# ---------------------------------------------------------------------------
# Block 4 — Grid Specifications
# ---------------------------------------------------------------------------

# 4x4 grid, step=0.2, pi_E_N in [3.0, 3.6], pi_E_E in [-2.6, -2.0].
# Does NOT contain the true minimum: pi_E_N≈3.89 > 3.6, pi_E_E≈-2.72 < -2.6.
# With refine=False: minimum appears at nearest corner (pi_E_N=3.6, pi_E_E=-2.6).
# With refine=True: corner minimum triggers one expansion in each direction.
# Use STATIC_PARAMS_PAR: grid is in the parallax-minimum region.
COARSE_GRID_PARAMS = {
    'pi_E_N': [3.0, 3.6, 0.2],
    'pi_E_E': [-2.6, -2.0, 0.2],
}

# 3x3 grid, step=0.2, centered on the true minimum.
# pi_E_N axis: [3.7, 3.9, 4.1] — center 3.9 nearest to true value 3.8925
# pi_E_E axis: [-2.9, -2.7, -2.5] — center -2.7 nearest to true value -2.7208
# step=0.2 is expected to exceed the actual 1σ uncertainties; coarse grid
# fails convergence criterion and genuine refinement is required.
# NOTE: if test_refinement_was_triggered fails with level=0, step=0.2 is
# smaller than the actual 1σ; increase step size here.
# Use STATIC_PARAMS_PAR.
REFINEMENT_GRID_PARAMS = {
    'pi_E_N': [3.7, 4.1, 0.2],
    'pi_E_E': [-2.9, -2.5, 0.2],
}

# 3x3 tight grid centered on the true minimum. Same geometry as
# REFINEMENT_GRID_PARAMS. Center (pi_E_N=3.9, pi_E_E=-2.7) is the
# nearest grid point to the true minimum (3.8925, -2.7208).
# Used to test best property fallback when refine=False.
# Use STATIC_PARAMS_PAR.
BEST_FALLBACK_GRID_PARAMS = {
    'pi_E_N': [3.7, 4.1, 0.2],
    'pi_E_E': [-2.9, -2.5, 0.2],
}

# 3x3 grid for skip_optimization tests. Grid is near the origin where
# STATIC_PARAMS is correct and KNOWN_CHI2_ORIGIN_SKIPOPT was computed.
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
      - REFINEMENT_GRID_PARAMS: 3x3, step=0.2, centered on true minimum
      - True minimum: pi_E_N≈3.8925, pi_E_E≈-2.7208 (near center point)
      - Nearest coarse grid point: (pi_E_N=3.9, pi_E_E=-2.7)
      - step=0.2 expected > actual 1σ; coarse grid fails convergence.
      - After refinement, convergence met at level >= 1.
      - STATIC_PARAMS_PAR used: grid is in the parallax-minimum region.

    Returns:
        dict with keys:
            'instance': ParallaxGridSearch after run(refine=True, ...)
            'filepath': pathlib.Path to saved JSON results file
    """
    tmp_dir = tmp_path_factory.mktemp("parallax_refinement")
    filepath = tmp_dir / "refinement_results.json"

    searcher = ParallaxGridSearch(
        static_params=STATIC_PARAMS_PAR,
        datasets=DATASETS,
        grid_params=REFINEMENT_GRID_PARAMS,
        fitter_kwargs=FITTER_KWARGS,
    )

    searcher.run(
        refine=True,
        point_density_in_minimum=3,   # n; convergence needs n//2=1 point per side
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

    Grid: 4x4, step=0.2, pi_E_N in [3.0, 3.6], pi_E_E in [-2.6, -2.0].
    The true minimum (pi_E_N≈3.8925, pi_E_E≈-2.7208) lies just outside
    both the pi_E_N upper edge (3.6) and the pi_E_E lower edge (-2.6).
    With refine=False the grid is evaluated exactly as specified and the
    nearest corner (pi_E_N=3.6, pi_E_E=-2.6) should be the global minimum
    on the grid. Uses STATIC_PARAMS_PAR because the grid is in the
    parallax-minimum region.
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
            static_params=STATIC_PARAMS_PAR,
            datasets=DATASETS,
            grid_params=COARSE_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
        )
        searcher.run(refine=False)
        return searcher

    def test_runs_without_error(self, coarse_searcher):
        """
        run(refine=False) completes and populates results_history with a
        fully evaluated 4x4 chi2 grid.

        Grid dimensions:
          pi_E_N axis: 4 points [3.0, 3.2, 3.4, 3.6]
          pi_E_E axis: 4 points [-2.6, -2.4, -2.2, -2.0]

        Every cell must be evaluated (no None entries in result_grid).
        """
        assert coarse_searcher.results_history is not None

        chi2_grid = coarse_searcher.results_history[0]['chi2_grid']
        assert chi2_grid.shape == (4, 4), (
            f"Expected chi2_grid shape (4, 4); got {chi2_grid.shape}"
        )

        result_grid = coarse_searcher.results_history[0]['result_grid']
        assert all(cell is not None for cell in result_grid.ravel()), (
            "Every grid cell must be evaluated; found None entries"
        )

    def test_minimum_is_on_edge(self, coarse_searcher):
        """
        The best coarse-grid minimum lies on the grid boundary.

        The true minimum is just outside both the pi_E_N upper edge (3.6)
        and the pi_E_E lower edge (-2.6); the chi2 surface slopes toward the
        nearest corner (pi_E_N=3.6, pi_E_E=-2.6), which should be the global
        minimum on the grid.

        Note: find_local_minima() returns (chi2, params_dict, level) tuples
        sorted best-first. self.minima stays None because refine=False.
        """
        minima = coarse_searcher.find_local_minima()
        assert len(minima) >= 1, "Expected at least one minimum"

        _, best_params, _ = minima[0]
        pi_e_e = best_params['pi_E_E']
        pi_e_n = best_params['pi_E_N']

        assert (pi_e_e in (-2.6, -2.0) or pi_e_n in (3.0, 3.6)), (
            f"Expected best grid point on boundary; "
            f"got pi_E_N={pi_e_n}, pi_E_E={pi_e_e}"
        )
