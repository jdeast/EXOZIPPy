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


class TestExpansion:
    """
    Tests for run(refine=True) edge-driven expansion and refinement.

    Sequence
    --------
    1. The coarse grid (pi_E_N: 3.0–3.6, pi_E_E: -2.6–-2.0, step=0.2) places
       the true minimum (pi_E_N≈3.89, pi_E_E≈-2.72) outside its boundary —
       beyond the upper edge in pi_E_N and below the lower edge in pi_E_E.
    2. run(refine=True) detects that the best grid point lies on a corner and
       triggers edge-driven expansion.
    3. With point_density_in_minimum=3 and step=0.2 each expansion adds
       3×0.2=0.6 to the range.  One expansion in pi_E_N raises the upper edge
       3.6 → 4.2 (covers 3.89 ✓); one expansion in pi_E_E lowers the bottom
       -2.6 → -3.2 (covers -2.72 ✓).  max_expansions=4 provides ample margin.
    4. Once the minimum is interior the refinement loop runs; the convergence
       level satisfies level >= 1.

    Note on shallow chi² landscape
    --------------------------------
    The OB140939 parallax chi² surface has total variation < 1.0 near the
    minimum.  The exact level at which convergence is satisfied depends on
    where the 1σ boundary falls relative to the 0.2 step, so tests assert
    level >= 1 rather than level == 1.
    """

    @pytest.fixture(scope="class")
    def expansion_searcher(self):
        """
        Class-scoped fixture: instantiate and run ParallaxGridSearch with
        refine=True so that edge-driven expansion and subsequent refinement
        are exercised.

        READ-ONLY CONTRACT: tests must not mutate the returned searcher.
        The fixture is evaluated once per class; any mutation would silently
        corrupt results for later tests in the same class.
        """
        searcher = ParallaxGridSearch(
            static_params=STATIC_PARAMS_PAR,
            datasets=DATASETS,
            grid_params=COARSE_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
        )
        searcher.run(
            refine=True,
            point_density_in_minimum=3,
            max_refinements=2,
            max_expansions=4,
        )
        return searcher

    def test_grid_shape_changed(self, expansion_searcher):
        """
        After edge-driven expansion the recorded grid shape is larger than
        the original (4, 4) coarse grid.

        The coarse grid has 4 points in each dimension (pi_E_N: 3.0, 3.2, 3.4,
        3.6; pi_E_E: -2.6, -2.4, -2.2, -2.0), giving shape (4, 4).  At least
        one expansion must be reflected in results_history[0].
        """
        grid_shape = expansion_searcher.results_history[0]['metadata']['grid_shape']
        assert grid_shape != (4, 4), (
            f"Expected expanded grid shape larger than (4, 4), got {grid_shape}"
        )

    def test_minimum_off_edge(self, expansion_searcher):
        """
        After expansion, the best minimum lies beyond the original grid boundary,
        confirming that expansion worked.

        Original boundary: pi_E_N upper edge=3.6, pi_E_E lower edge=-2.6.
        The coarse grid corner (3.6, -2.6) was the best coarse point, which
        triggered expansion.  After expansion, new points beyond those edges
        are evaluated; even on a shallow chi² surface the gradient points away
        from the original corner, so at least one new point should beat it.

        The test only asserts the weak inequalities pi_E_N > 3.6 and
        pi_E_E < -2.6 — i.e. that the minimum moved beyond the original edges.
        """
        minima = expansion_searcher.find_local_minima()
        assert len(minima) >= 1, "Expected at least one local minimum to be found"
        _, best_params, _ = minima[0]
        assert best_params['pi_E_N'] > 3.6, (
            f"pi_E_N = {best_params['pi_E_N']:.4f} should be > 3.6 "
            f"(true value ≈3.89)"
        )
        assert best_params['pi_E_E'] < -2.6, (
            f"pi_E_E = {best_params['pi_E_E']:.4f} should be < -2.6 "
            f"(true value ≈-2.72)"
        )

    def test_refinement_was_triggered(self, expansion_searcher):
        """
        The best minimum required at least one refinement level (level >= 1).

        The coarse grid minimum was on an edge and cannot satisfy the
        convergence criterion (requires n//2=1 point per side within Δχ²=1);
        refinement must be triggered after expansion moves the minimum to an
        interior point.  The exact level is not asserted because the shallow
        chi² landscape (total variation < 1.0) makes the convergence level
        uncertain.
        """
        minima = expansion_searcher.find_local_minima()
        _, _, level = minima[0]
        assert level >= 1, (
            f"level = {level}; coarse grid minimum was on an edge and should "
            f"not satisfy convergence without refinement"
        )


class TestSkipOptimization:
    """
    Tests for the skip_optimization=True flag.

    When skip_optimization=True, chi² is computed at exactly the grid-point
    parameter values without running a numerical optimizer.  This means the
    returned chi² values are always >= the optimized values at every grid
    point, because the optimizer can only lower chi².

    Grid indexing for SKIP_OPT_GRID_PARAMS
    ----------------------------------------
    chi2_grid[i, j] where i indexes pi_E_E and j indexes pi_E_N.
    pi_E_E axis values: [0.0, 0.5, 1.0] → index 0 is pi_E_E=0.0
    pi_E_N axis values: [-1.0, -0.5, 0.0] → index 2 is pi_E_N=0.0
    Therefore chi2_grid[0, 2] is the cell for (pi_E_E=0.0, pi_E_N=0.0).

    Note: STATIC_PARAMS (not STATIC_PARAMS_PAR) is used for all
    instantiations in this class because SKIP_OPT_GRID_PARAMS is defined
    in the origin region, which corresponds to the STATIC_PARAMS solution.
    """

    def test_chi2_at_origin(self):
        """
        With skip_optimization=True the chi² at (pi_E_E=0.0, pi_E_N=0.0)
        matches the known value KNOWN_CHI2_ORIGIN_SKIPOPT to within 0.01.

        Because no optimizer is run the result is fully determined by the
        static parameters and the grid-point values, making it reproducible
        and comparable to the pre-computed reference.
        """
        searcher = ParallaxGridSearch(
            static_params=STATIC_PARAMS,
            datasets=DATASETS,
            grid_params=SKIP_OPT_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
            skip_optimization=True,
        )
        searcher.run(refine=False)
        chi2_grid = searcher.results_history[0]['chi2_grid']
        assert abs(chi2_grid[0, 2] - KNOWN_CHI2_ORIGIN_SKIPOPT) < 0.01, (
            f"chi² at origin = {chi2_grid[0, 2]:.6f}, "
            f"expected {KNOWN_CHI2_ORIGIN_SKIPOPT:.6f} (tolerance 0.01)"
        )

    def test_chi2_larger_than_optimized(self):
        """
        At every finite grid point the skip-optimization chi² is >= the
        optimized chi².

        The numerical optimizer can only lower chi² relative to the starting
        grid-point values, so chi2_skip >= chi2_opt must hold element-wise
        across all grid points where both values are finite.
        """
        searcher_skip = ParallaxGridSearch(
            static_params=STATIC_PARAMS,
            datasets=DATASETS,
            grid_params=SKIP_OPT_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
            skip_optimization=True,
        )
        searcher_skip.run(refine=False)

        searcher_opt = ParallaxGridSearch(
            static_params=STATIC_PARAMS,
            datasets=DATASETS,
            grid_params=SKIP_OPT_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
            skip_optimization=False,
        )
        searcher_opt.run(refine=False)

        chi2_skip = searcher_skip.results_history[0]['chi2_grid']
        chi2_opt = searcher_opt.results_history[0]['chi2_grid']

        finite_mask = np.isfinite(chi2_skip) & np.isfinite(chi2_opt)
        assert np.all(chi2_skip[finite_mask] >= chi2_opt[finite_mask]), (
            "skip_optimization chi² should be >= optimized chi² at every "
            "finite grid point; optimizer can only lower chi²"
        )


class TestFindLocalMinima:
    """
    Tests for the find_local_minima() method.

    Covered here
    ------------
    - Precondition error: calling find_local_minima() before run() raises
      ValueError.

    NOT covered here
    ----------------
    - Multi-minimum behavior: verifying that find_local_minima() correctly
      identifies and returns multiple isolated minima.  See the placeholder
      test_multi_minimum_behavior below for the rationale and the conditions
      needed before this can be implemented.
    """

    def test_raises_before_run(self):
        """
        find_local_minima() raises ValueError if called before run().

        The method requires an internal results structure that is only
        populated by run(); calling it on a fresh instance should raise
        ValueError to signal the violated precondition.
        """
        searcher = ParallaxGridSearch(
            static_params=STATIC_PARAMS_PAR,
            datasets=DATASETS,
            grid_params=COARSE_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
        )
        with pytest.raises(ValueError):
            searcher.find_local_minima()

    @pytest.mark.skip(
        reason=(
            "TODO: requires a dataset / grid configuration with a known "
            "multi-minimum chi2 landscape. OB140939 with the current grid "
            "settings does not reliably produce multiple isolated minima. "
            "Consider a different microlensing event or a synthetic dataset "
            "before implementing this test."
        )
    )
    def test_multi_minimum_behavior(self):
        pass


class TestRefinement:
    """
    Tests for run(refine=True) using REFINEMENT_GRID_PARAMS, which is
    centered on the known true minimum so that expansion is not needed.

    Refinement mechanics
    --------------------
    The 3×3 coarse grid has step=0.2.  With point_density_in_minimum=3 the
    convergence criterion requires n//2=1 point per side within Δχ²=1.  The
    step=0.2 is expected to exceed the actual 1σ width of the chi² surface,
    so the coarse grid fails convergence and refinement is triggered at
    level >= 1.

    Shallow chi² landscape
    -----------------------
    The OB140939 parallax chi² surface has total variation < 1.0 near the
    minimum.  KNOWN_PI_E_N and KNOWN_PI_E_E are within Δχ²=1.0 of the
    minimum, meaning many nearby points are statistically indistinguishable.
    Tests use level >= 1 and position tolerance 1.0 to reflect this genuine
    uncertainty.

    IMPORTANT — READ-ONLY CONTRACT
    --------------------------------
    All tests in this class consume the module-scoped refinement_result
    fixture, which is shared with TestBest and TestSaveLoad.  Tests MUST NOT
    call run() on the instance or mutate any attribute of the returned
    searcher.  Any mutation would silently corrupt results for other test
    classes that share the same fixture instance.
    """

    def test_refinement_was_triggered(self, refinement_result):
        """
        At least one refinement level was computed (level >= 1).

        step=0.2 is expected to exceed the actual 1σ width of the chi²
        surface, so the coarse grid cannot satisfy the convergence criterion
        and refinement must be triggered.  If this test fails with level=0
        it means step=0.2 is already smaller than the actual 1σ; the fix is
        to increase the step size in REFINEMENT_GRID_PARAMS.

        Note: level == 1 is NOT asserted because the shallow chi² landscape
        (total variation < 1.0) makes the exact convergence level uncertain.
        """
        searcher = refinement_result["instance"]
        minima = searcher.find_local_minima()
        assert len(minima) >= 1, "Expected at least one local minimum"
        _, _, level = minima[0]
        assert level >= 1, (
            "Expected at least one refinement level. If level=0, step=0.2 is "
            "smaller than the actual 1σ; increase REFINEMENT_GRID_PARAMS step size."
        )

    def test_chi2_at_minimum(self, refinement_result):
        """
        The best chi² after refinement is within 0.3 of KNOWN_CHI2_MINIMUM.

        This tests whether the refinement algorithm converged accurately, not
        just whether it found the right region.  KNOWN_CHI2_MINIMUM is
        approximate (~0.3 unit drift due to optimizer trajectory on the
        shallow chi² surface), so a tolerance of 0.3 is tight enough to
        verify convergence while accommodating genuine numerical variability.

        Note: parameter location is NOT asserted here — on a shallow chi²
        surface (total variation < 1.0) many nearby points are statistically
        indistinguishable, making location assertions fragile.  Chi² is the
        actual objective and is the right quantity to test.
        """
        searcher = refinement_result["instance"]
        minima = searcher.find_local_minima()
        assert len(minima) >= 1, "Expected at least one local minimum"
        best_chi2, _, _ = minima[0]
        assert abs(best_chi2 - KNOWN_CHI2_MINIMUM) < 0.3, (
            f"chi² = {best_chi2:.6f}, expected within 0.3 of "
            f"KNOWN_CHI2_MINIMUM={KNOWN_CHI2_MINIMUM:.6f}"
        )

    def test_save_load_round_trip(self, refinement_result):
        """
        A saved results file round-trips correctly through load_results().

        Checks that grid parameters, grid shape, chi² values, static
        parameters, skip_optimization flag, parameters_to_fit, minima, and
        dataset handling are all preserved or correctly handled:
        - datasets is None after loading (cannot be serialised)
        - minima is not None and has the same length as the original
        - the best-minimum chi² is numerically close after the round trip
        """
        filepath = refinement_result["filepath"]
        original = refinement_result["instance"]
        loaded = ParallaxGridSearch.load_results(filepath)

        assert loaded.grid_params == original.grid_params, (
            "grid_params mismatch after load"
        )
        assert (
            loaded.results_history[0]['metadata']['grid_shape']
            == original.results_history[0]['metadata']['grid_shape']
        ), "grid_shape mismatch after load"
        assert np.allclose(
            loaded.results_history[0]['chi2_grid'],
            original.results_history[0]['chi2_grid'],
            equal_nan=True,
        ), "chi2_grid values differ after load"
        assert loaded.static_params == original.static_params, (
            "static_params mismatch after load"
        )
        assert loaded.skip_optimization == original.skip_optimization, (
            "skip_optimization mismatch after load"
        )
        assert loaded.parameters_to_fit == original.parameters_to_fit, (
            "parameters_to_fit mismatch after load"
        )
        assert loaded.datasets is None, (
            "datasets should be None after loading (not serialisable)"
        )
        assert loaded.minima is not None, (
            "minima should not be None after loading"
        )
        assert len(loaded.minima) == len(original.minima), (
            f"minima length mismatch: loaded {len(loaded.minima)}, "
            f"original {len(original.minima)}"
        )
        assert np.isclose(
            loaded.minima[0]['chi2'], original.minima[0]['chi2']
        ), (
            f"best-minimum chi² mismatch after load: "
            f"loaded {loaded.minima[0]['chi2']}, "
            f"original {original.minima[0]['chi2']}"
        )


class TestBest:
    """
    Tests for the ``best`` property across all four behavioral modes.

    Behavioral modes
    ----------------
    +--------+----------------------------------+----------------------------------+
    | Mode   | Condition                        | Returns                          |
    +--------+----------------------------------+----------------------------------+
    | 1      | before run()                     | None                             |
    | 2      | after run(refine=False)          | best coarse grid point (dict)    |
    | 3      | after run(refine=True)           | best refined minimum (from       |
    |        |                                  | self.minima)                     |
    | 4      | chi² quality                     | meaningful improvement over      |
    |        |                                  | no-parallax solution             |
    +--------+----------------------------------+----------------------------------+

    Shallow chi² landscape
    -----------------------
    The OB140939 parallax chi² surface has ~0.3 unit optimizer drift near
    the minimum.  KNOWN_CHI2_MINIMUM is therefore approximate.  Tests use
    CHI2_IMPROVEMENT_MIN (minimum expected Δχ² over the no-parallax solution)
    rather than tight bounds against KNOWN_CHI2_MINIMUM, because
    KNOWN_CHI2_ORIGIN_SKIPOPT is stable (computed without an optimizer) and
    provides a reliable anchor.

    IMPORTANT — READ-ONLY CONTRACT
    --------------------------------
    Tests 4 and 5 consume the module-scoped ``refinement_result`` fixture,
    which is shared with TestRefinement and TestSaveLoad.  These tests MUST
    NOT call run() on the instance or mutate any attribute.  Any mutation
    would silently corrupt results for other test classes that share the
    same fixture instance.
    """

    @pytest.fixture(scope="class")
    def best_fallback_searcher(self):
        """
        Class-scoped fixture: run ParallaxGridSearch with refine=False on
        BEST_FALLBACK_GRID_PARAMS (3×3, centered on the true minimum).

        The center grid point (pi_E_N=3.9, pi_E_E=-2.7) is the nearest grid
        point to the true minimum (3.8925, -2.7208), so it should be returned
        as the best coarse grid point.

        READ-ONLY CONTRACT: tests must not mutate the returned searcher.
        The fixture is evaluated once per class; any mutation would silently
        corrupt results for later tests in the same class.
        """
        searcher = ParallaxGridSearch(
            static_params=STATIC_PARAMS_PAR,
            datasets=DATASETS,
            grid_params=BEST_FALLBACK_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
        )
        searcher.run(refine=False)
        return searcher

    def test_returns_none_before_run(self):
        """
        Before run() is called, best returns None (Mode 1).

        A fresh instance has no results; best must not attempt to access
        uninitialised state.  STATIC_PARAMS is used here because static
        params do not affect the pre-run state being tested.
        """
        # fresh instance isolates pre-run state
        searcher = ParallaxGridSearch(
            static_params=STATIC_PARAMS,
            datasets=DATASETS,
            grid_params=COARSE_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
        )
        assert searcher.best is None

    def test_fallback_to_coarse_grid(self, best_fallback_searcher):
        """
        After run(refine=False), best returns a non-None result with a chi²
        showing meaningful improvement over the no-parallax solution (Mode 2).

        Parameter location is NOT asserted: with skip_optimization=False the
        optimizer can drift pi_E_N and pi_E_E away from exact grid point
        values.  Chi² is the right quantity to test here — KNOWN_CHI2_ORIGIN_SKIPOPT
        is stable (computed without an optimizer) and CHI2_IMPROVEMENT_MIN
        gives margin for optimizer drift in the shallow landscape.
        """
        best = best_fallback_searcher.best
        assert best is not None, "best should not be None after run(refine=False)"
        threshold = KNOWN_CHI2_ORIGIN_SKIPOPT - CHI2_IMPROVEMENT_MIN
        assert best['chi2'] < threshold, (
            f"chi² = {best['chi2']:.6f} should be < {threshold:.6f} "
            f"(= {KNOWN_CHI2_ORIGIN_SKIPOPT:.6f} - {CHI2_IMPROVEMENT_MIN:.1f})"
        )

    def test_minima_none_after_refine_false(self, best_fallback_searcher):
        """
        After run(refine=False), self.minima is None.

        Refinement was not requested, so the minima attribute should never
        be populated; best falls back to the coarse grid result (Mode 2).
        """
        assert best_fallback_searcher.minima is None, (
            "minima should be None after run(refine=False)"
        )

    def test_minima_not_none_after_refine_true(self, refinement_result):
        """
        After run(refine=True), self.minima is populated (Mode 3 precondition).

        READ-ONLY: this test must not call run() or mutate the instance.
        """
        searcher = refinement_result["instance"]
        assert searcher.minima is not None, (
            "minima should not be None after run(refine=True)"
        )
        assert len(searcher.minima) > 0, (
            "minima should contain at least one entry after run(refine=True)"
        )

    def test_chi2_close_to_known_minimum(self, refinement_result):
        """
        After run(refine=True), best['chi2'] is below KNOWN_CHI2_MINIMUM + 2.0
        (Mode 3 quality check).

        KNOWN_CHI2_MINIMUM is approximate (~0.3 unit drift due to optimizer
        trajectory on the shallow chi² surface); a tolerance of +2.0 provides
        comfortable margin while verifying that the minimum region was found.

        READ-ONLY: this test must not call run() or mutate the instance.
        """
        searcher = refinement_result["instance"]
        best = searcher.best
        assert best is not None, "best should not be None after run(refine=True)"
        threshold = KNOWN_CHI2_MINIMUM + 2.0
        assert best['chi2'] < threshold, (
            f"chi² = {best['chi2']:.6f} should be < {threshold:.6f} "
            f"(= KNOWN_CHI2_MINIMUM {KNOWN_CHI2_MINIMUM:.6f} + 2.0)"
        )

class TestSaveLoad:
    """
    Tests for save/load error and warning paths.

    Round-trip correctness (grid_params, chi2_grid, static_params, minima,
    etc.) is already covered by test_save_load_round_trip in TestRefinement.
    This class covers the two remaining paths:

    - save_results() called before run() raises ValueError (no results to
      save yet).
    - Loading a ParallaxGridSearch results file via the base class
      BaseRectGridSearch.load_results() emits a UserWarning because the
      saved class name does not match the loading class.
    """

    def test_save_raises_before_run(self, tmp_path):
        """
        save_results() raises ValueError if called before run().

        There are no results to serialise on a fresh instance; the method
        must detect this and raise ValueError rather than writing an empty
        or malformed file.
        """
        searcher = ParallaxGridSearch(
            static_params=STATIC_PARAMS_PAR,
            datasets=DATASETS,
            grid_params=COARSE_GRID_PARAMS,
            fitter_kwargs=FITTER_KWARGS,
        )
        # Do NOT call run()
        filepath = tmp_path / "no_results.json"
        with pytest.raises(ValueError):
            searcher.save_results(filepath)

    @pytest.mark.skip(
        reason=(
                "TODO: verifying that load_results() emits a UserWarning on class name "
                "mismatch requires a concrete subclass of BaseRectGridSearch to satisfy "
                "the ABC. BaseRectGridSearch itself cannot be instantiated directly "
                "(_fit_grid_point is abstract). This path is unreachable through the "
                "public API without a dedicated concrete subclass. Implement when a "
                "suitable concrete subclass is available for use in tests."
        )
    )
    def test_class_name_mismatch_warning(self):
        pass


class TestPlaceholders:
    """
    Documented coverage gaps that are not yet implemented.

    Each placeholder records why the test cannot be written yet and what
    precondition must be satisfied before it can be promoted to a real test.

    To promote a placeholder:
      1. Remove the @pytest.mark.skip decorator.
      2. Replace ``pass`` with the test logic.
      3. Update the expected skip count in this docstring from 5 to 4.
    """

    @pytest.mark.skip(
        reason=(
            "TODO: find_local_minima() multi-minimum behavior requires a "
            "dataset or grid configuration that reliably produces multiple "
            "isolated local chi2 minima. OB140939 with the current grid "
            "parameters does not satisfy this requirement. Consider a "
            "different microlensing event or a synthetic chi2 landscape."
        )
    )
    def test_find_local_minima_multi_minimum(self):
        pass

    @pytest.mark.skip(
        reason=(
            "TODO: get_uncertainties() testing requires exact expected "
            "uncertainty ranges derived from a definitive reference for this "
            "dataset and grid configuration. Implementing with vague bounds "
            "(e.g., '> 0') would give false confidence. Derive exact expected "
            "ranges from a published or agreed-upon reference before "
            "implementing."
        )
    )
    def test_get_uncertainties_specific(self):
        pass

    @pytest.mark.skip(
        reason=(
            "TODO: ParallaxGridSearch does not support log-space parameters "
            "— 'pi_E_E' and 'pi_E_N' are always evaluated on a linear grid. "
            "Log-space grid tests belong to a dedicated BaseRectGridSearch "
            "subclass designed for log-space parameters. Implement when such "
            "a subclass is available and has known expected values for a test "
            "dataset."
        )
    )
    def test_log_space_parameters(self):
        pass

    @pytest.mark.skip(
        reason=(
            "TODO: verifying that load_results() emits a UserWarning on "
            "class name mismatch requires a concrete subclass of "
            "BaseRectGridSearch to satisfy the ABC. BaseRectGridSearch "
            "itself cannot be instantiated directly (_fit_grid_point is "
            "abstract). This path is unreachable through the public API "
            "without a dedicated concrete subclass. Implement when a "
            "suitable concrete subclass is available for use in tests."
        )
    )
    def test_class_name_mismatch_warning(self):
        pass

