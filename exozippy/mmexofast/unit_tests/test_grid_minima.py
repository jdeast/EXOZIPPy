"""
Unit tests for grid minima extraction.

To visualize the test grids and minima, run:
    pytest test_grid_minima.py --plot-grids
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from exozippy.mmexofast.gridsearches import BaseRectGridSearch


@pytest.fixture
def plot_grids(request):
    return request.config.getoption("--plot-grids")


class MockGridSearch(BaseRectGridSearch):
    """Mock grid search for testing minima extraction."""

    def __init__(self, datasets=None, verbose=False):
        super().__init__(datasets=datasets if datasets else [], verbose=verbose)
        self.grid_params = {
            'param1_min': 0.0,
            'param1_max': 1.0,
            'param1_step': 0.1,
            'param2_min': 0.0,
            'param2_max': 1.0,
            'param2_step': 0.1
        }

    def _setup_grid(self):
        pass  # Not needed for tests

    def _fit_grid_point(self, grid_params):
        pass  # Not needed for tests

    def set_mock_results(self, results):
        """Set mock results for testing."""
        self.results = results


def plot_grid(grid, title, ax=None):
    """
    Plot chi2 grid.

    Parameters
    ----------
    grid : MockGridSearch
        Grid search object with results
    title : str
        Plot title
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig, ax, im
        Figure, axes, and image objects
    """
    # Extract grid dimensions
    param1_vals = sorted(set(r['param1'] for r in grid.results))
    param2_vals = sorted(set(r['param2'] for r in grid.results))

    # Create chi2 grid
    chi2_grid = np.full((len(param2_vals), len(param1_vals)), np.nan)

    for result in grid.results:
        i = param1_vals.index(result['param1'])
        j = param2_vals.index(result['param2'])
        chi2_grid[j, i] = result['chi2']

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.imshow(chi2_grid, cmap='Set1', origin='lower',
                   extent=[param1_vals[0], param1_vals[-1],
                           param2_vals[0], param2_vals[-1]],
                   aspect='equal', vmin=100, vmax=190)

    ax.set_xlabel('param1')
    ax.set_ylabel('param2')
    ax.set_title(title)

    return fig, ax, im


def add_minima_to_plot(ax, local_minima, separated_minima):
    """
    Add minima markers to an existing plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add markers to
    local_minima : list of tuple
        All local minima as (chi2, params) tuples
    separated_minima : list of tuple
        Selected separated minima as (chi2, params) tuples
    """
    # Mark all local minima with black x
    if local_minima:
        local_x = [params['param1'] for _, params in local_minima]
        local_y = [params['param2'] for _, params in local_minima]
        ax.plot(local_x, local_y, 'kx', markersize=10, markeredgewidth=2,
                label='Local minima')

    # Mark selected separated minima with red o
    if separated_minima:
        sep_x = [params['param1'] for _, params in separated_minima]
        sep_y = [params['param2'] for _, params in separated_minima]
        ax.plot(sep_x, sep_y, 'ro', markersize=12, markerfacecolor='none',
                markeredgewidth=2, label='Separated minima')

    ax.legend()


def plot_grid_with_minima(grid, local_minima, separated_minima, title):
    """
    Plot chi2 grid with local and separated minima marked.

    Parameters
    ----------
    grid : MockGridSearch
        Grid search object with results
    local_minima : list of tuple
        All local minima as (chi2, params) tuples
    separated_minima : list of tuple
        Selected separated minima as (chi2, params) tuples
    title : str
        Plot title
    """
    fig, ax, im = plot_grid(grid, title)
    add_minima_to_plot(ax, local_minima, separated_minima)
    plt.colorbar(im, ax=ax, label='chi2')
    plt.tight_layout()
    plt.show()


def test_clear_4_minima(plot_grids):
    """Test case A: 4 distinct minima, well-separated."""
    grid = MockGridSearch()

    # Create mock results with 4 clear minima at different chi2 levels
    results = []
    minima_dict = {
        (2, 2): 115.0,
        (2, 8): 125.0,
        (8, 2): 135.0,
        (8, 8): 145.0
    }

    for i in range(11):
        for j in range(11):
            param1 = round(i * 0.1, 10)
            param2 = round(j * 0.1, 10)

            if (i, j) in minima_dict:
                chi2 = minima_dict[(i, j)]
            else:
                chi2 = 190.0  # Background

            results.append({
                'param1': param1,
                'param2': param2,
                'chi2': chi2,
                'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0},
                'success': True
            })

    grid.set_mock_results(results)

    # Find local minima
    local_minima = grid.find_local_minima()
    assert len(local_minima) == 4, f"Expected 4 minima, found {len(local_minima)}"

    # Select separated minima
    separated = grid.select_separated_minima(local_minima, min_separation=1.0)
    assert len(separated) == 4, f"Expected 4 separated minima, found {len(separated)}"

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, separated,
                              "Test A: Clear 4 Minima")


def test_ring_constant_chi2(plot_grids):
    """Test case B1: Ring with constant chi2 (no true local minima)."""
    grid = MockGridSearch()

    results = []
    center_i, center_j = 5, 5
    radius = 3

    for i in range(11):
        for j in range(11):
            param1 = round(i * 0.1, 10)
            param2 = round(j * 0.1, 10)

            # Create ring of constant chi2
            dist = np.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
            if abs(dist - radius) < 0.5:
                chi2 = 115.0  # Ring - all equal
            else:
                chi2 = 190.0  # Background

            results.append({
                'param1': param1,
                'param2': param2,
                'chi2': chi2,
                'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0},
                'success': True
            })

    grid.set_mock_results(results)

    # Find local minima (should find none or very few due to equal neighbors)
    local_minima = grid.find_local_minima()
    # With constant chi2, no true local minima exist
    assert len(local_minima) == 1, f"Expected 1 minima for constant ring, found {len(local_minima)}"

    if plot_grids:
        separated = grid.select_separated_minima(local_minima, min_separation=1.0) if local_minima else []
        plot_grid_with_minima(grid, local_minima, separated,
                              "Test B1: Ring - Constant Chi2")


def test_ring_with_local_minima(plot_grids):
    """Test case B2: Ring with 4 local minima at different depths."""
    grid = MockGridSearch()

    results = []
    center_i, center_j = 5, 5
    radius = 3

    # Exact grid points at 90° intervals on the ring
    minima_locs = [(8, 5), (5, 8), (2, 5), (5, 2)]  # E, N, W, S
    minima_chi2 = [115.0, 125.0, 135.0, 145.0]
    minima_dict = dict(zip(minima_locs, minima_chi2))

    for i in range(11):
        for j in range(11):
            param1 = round(i * 0.1, 10)
            param2 = round(j * 0.1, 10)

            dist_from_center = np.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)

            if abs(dist_from_center - radius) < 0.5:
                # On the ring - default to 155, but set minima explicitly
                if (i, j) in minima_dict:
                    chi2 = minima_dict[(i, j)]
                else:
                    chi2 = 155.0
            else:
                chi2 = 190.0  # Background

            results.append({
                'param1': param1,
                'param2': param2,
                'chi2': chi2,
                'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0},
                'success': True
            })

    grid.set_mock_results(results)

    # Find local minima
    local_minima = grid.find_local_minima()
    assert len(local_minima) == 4, f"Expected 4 minima, found {len(local_minima)}"

    # Select separated minima
    separated = grid.select_separated_minima(local_minima, min_separation=1.0)
    assert len(separated) == 4, f"Expected 4 separated minima"

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, separated,
                              "Test B2: Ring - With Local Minima")


def test_arc_constant_chi2(plot_grids):
    """Test case C1: Arc with constant chi2 (no true local minima)."""
    grid = MockGridSearch()

    results = []

    for i in range(11):
        for j in range(11):
            param1 = round(i * 0.1, 10)
            param2 = round(j * 0.1, 10)

            # Create diagonal arc with constant chi2
            if abs(i - j) <= 1:
                chi2 = 115.0  # Arc - all equal
            else:
                chi2 = 190.0  # Background

            results.append({
                'param1': param1,
                'param2': param2,
                'chi2': chi2,
                'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0},
                'success': True
            })

    grid.set_mock_results(results)

    # Find local minima (should find none or very few)
    local_minima = grid.find_local_minima()
    assert len(local_minima) == 1, f"Expected 1 minima for constant arc, found {len(local_minima)}"

    if plot_grids:
        separated = grid.select_separated_minima(local_minima, min_separation=1.0) if local_minima else []
        plot_grid_with_minima(grid, local_minima, separated,
                              "Test C1: Arc - Constant Chi2")


def test_arc_with_local_minima(plot_grids):
    """Test case C2: Arc with 2 local minima near endpoints."""
    grid = MockGridSearch()

    results = []

    # Define arc points along diagonal from (2,2) to (8,8)
    arc_points = [(i, i) for i in range(2, 9)]

    # Minima at the endpoints with different chi2
    minima_dict = {
        (2, 2): 115.0,  # Bottom-left endpoint
        (8, 8): 125.0  # Top-right endpoint
    }

    for i in range(11):
        for j in range(11):
            param1 = round(i * 0.1, 10)
            param2 = round(j * 0.1, 10)

            if (i, j) in arc_points:
                # On arc - check if it's a minimum
                if (i, j) in minima_dict:
                    chi2 = minima_dict[(i, j)]
                else:
                    chi2 = 155.0  # Default arc value
            else:
                chi2 = 190.0  # Off arc

            results.append({
                'param1': param1,
                'param2': param2,
                'chi2': chi2,
                'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0},
                'success': True
            })

    grid.set_mock_results(results)

    # Find local minima
    local_minima = grid.find_local_minima()
    assert len(local_minima) == 2, f"Expected 2 minima, found {len(local_minima)}"

    # Select separated minima
    separated = grid.select_separated_minima(local_minima, min_separation=1.0)
    assert len(separated) == 2, f"Expected 2 separated minima"

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, separated,
                              "Test C2: Arc - With Local Minima")


def test_single_minimum(plot_grids):
    """Test case D: Only one clear minimum in grid."""
    grid = MockGridSearch()

    results = []

    for i in range(11):
        for j in range(11):
            param1 = round(i * 0.1, 10)
            param2 = round(j * 0.1, 10)

            # Single minimum at center
            if (i, j) == (5, 5):
                chi2 = 115.0
            else:
                chi2 = 190.0

            results.append({
                'param1': param1,
                'param2': param2,
                'chi2': chi2,
                'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0},
                'success': True
            })

    grid.set_mock_results(results)

    # Find local minima
    local_minima = grid.find_local_minima()
    assert len(local_minima) == 1, f"Expected 1 minimum, found {len(local_minima)}"

    # Select separated minima
    separated = grid.select_separated_minima(local_minima, min_separation=1.0)
    assert len(separated) == 1

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, separated,
                              "Test D: Single Minimum")


def test_failed_fits_excluded(plot_grids):
    """Test case E: Failed fits scattered throughout grid."""
    grid = MockGridSearch()

    results = []

    for i in range(11):
        for j in range(11):
            param1 = round(i * 0.1, 10)
            param2 = round(j * 0.1, 10)

            # Two minima, but some fits failed
            if (i, j) in [(2, 2), (8, 8)]:
                chi2 = 115.0
                success = True
            elif (i + j) % 3 == 0:  # ~1/3 of points failed
                chi2 = np.nan
                success = False
            else:
                chi2 = 190.0
                success = True

            results.append({
                'param1': param1,
                'param2': param2,
                'chi2': chi2,
                'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0} if success else None,
                'success': success
            })

    grid.set_mock_results(results)

    # Find local minima (should ignore failed fits)
    local_minima = grid.find_local_minima()
    assert len(local_minima) == 2, f"Expected 2 minima, found {len(local_minima)}"
    assert all(np.isfinite(chi2) for chi2, _ in local_minima), "All minima should have finite chi2"

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, local_minima,
                              "Test E: Failed Fits Excluded")


def test_asymmetric_minima(plot_grids):
    """Test case F: Different numbers of minima (2 close, 1 far)."""
    grid = MockGridSearch()

    results = []

    for i in range(11):
        for j in range(11):
            param1 = round(i * 0.1, 10)
            param2 = round(j * 0.1, 10)

            # 3 minima: 2 close together on left (one slightly lower), 1 on right
            if (i, j) == (2, 5):  # Best minimum
                chi2 = 114.0
            elif (i, j) == (3, 5):  # Nearby, but higher
                chi2 = 115.0
            elif (i, j) == (8, 5):  # Far away
                chi2 = 125.0
            else:
                chi2 = 190.0

            results.append({
                'param1': param1,
                'param2': param2,
                'chi2': chi2,
                'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0},
                'success': True
            })

    grid.set_mock_results(results)

    # Find local minima
    local_minima = grid.find_local_minima()
    assert len(local_minima) == 2, f"Expected 2 minima, found {len(local_minima)}"

    # Select separated minima - should get both (they are 6 steps apart)
    separated = grid.select_separated_minima(local_minima, min_separation=1.0)
    assert len(separated) == 2, f"Expected 2 separated minima, found {len(separated)}"

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, separated,
                              "Test F: Asymmetric Minima")
