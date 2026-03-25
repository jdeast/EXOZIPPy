"""
Unit tests for grid minima extraction.

To visualize the test grids and minima, run:
    pytest test_grid_minima.py --plot-grids
"""
import numpy as np
import pytest
import matplotlib.pyplot as plt

from exozippy.mmexofast.gridsearches import BaseRectGridSearch


# ----------------------------------------------------------------
# Mock infrastructure
# ----------------------------------------------------------------

@pytest.fixture
def plot_grids(request):
    return request.config.getoption("--plot-grids")


def plot_grid(grid, title, ax=None):
    """
    Plot chi2 grid.

    Parameters
    ----------
    grid : MockGridSearch
        Grid search object with results_history populated
    title : str
        Plot title
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    im : matplotlib.image.AxesImage
    """
    level = grid.results_history[0]
    chi2_grid = level['chi2_grid']
    param_arrays = level['metadata']['param_arrays']
    param_names = level['metadata']['param_names']

    param1_vals = param_arrays[param_names[0]]
    param2_vals = param_arrays[param_names[1]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.imshow(chi2_grid.T, cmap='Set1', origin='lower',
                   extent=[param1_vals[0], param1_vals[-1],
                           param2_vals[0], param2_vals[-1]],
                   aspect='equal', vmin=100, vmax=190)

    ax.set_xlabel(param_names[0])
    ax.set_ylabel(param_names[1])
    ax.set_title(title)

    return fig, ax, im


def add_minima_to_plot(ax, local_minima):
    """
    Add minima markers to an existing plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add markers to
    local_minima : list of tuple
        All local minima as (chi2, params, level) tuples
    """
    if local_minima:
        local_x = [params['param1'] for _, params, _ in local_minima]
        local_y = [params['param2'] for _, params, _ in local_minima]
        ax.plot(local_x, local_y, 'kx', markersize=10, markeredgewidth=2,
                label='Local minima')
        ax.legend()


def plot_grid_with_minima(grid, local_minima, title):
    """
    Plot chi2 grid with local minima marked.

    Parameters
    ----------
    grid : MockGridSearch
        Grid search object with results_history populated
    local_minima : list of tuple
        All local minima as (chi2, params, level) tuples
    title : str
        Plot title
    """
    fig, ax, im = plot_grid(grid, title)
    add_minima_to_plot(ax, local_minima)
    plt.colorbar(im, ax=ax, label='chi2')
    plt.tight_layout()
    plt.show()


class MockGridSearch(BaseRectGridSearch):
    """Minimal mock for testing find_local_minima()."""

    def __init__(self):
        super().__init__(
            grid_params={
                'param1': [0.0, 1.0, 0.1],
                'param2': [0.0, 1.0, 0.1]
            }
        )

    def _fit_grid_point(self, grid_params):
        pass

    def set_mock_results(self, results):
        """Convert flat results list to structured grid arrays.

        Parameters
        ----------
        results : list of dict
            Each dict must contain 'param1', 'param2', 'chi2',
            'params', and 'success' keys.
        """
        param_names = list(self.grid_params.keys())

        param_arrays = {}
        for name in param_names:
            min_val, max_val, step = self.grid_params[name]
            n = int(np.round((max_val - min_val) / step)) + 1
            param_arrays[name] = np.linspace(min_val, max_val, n)

        grid_shape = tuple(len(param_arrays[name]) for name in param_names)

        chi2_grid = np.full(grid_shape, np.nan)
        result_grid = np.empty(grid_shape, dtype=object)

        for result in results:
            if not result.get('success', False):
                continue
            indices = tuple(
                np.argmin(np.abs(param_arrays[name] - result[name]))
                for name in param_names
            )
            chi2_grid[indices] = result['chi2']
            result_grid[indices] = result

        self.results_history = [{
            'chi2_grid': chi2_grid,
            'result_grid': result_grid,
            'metadata': {
                'param_names': param_names,
                'param_arrays': param_arrays,
                'grid_shape': grid_shape
            }
        }]


def _make_result(i, j, chi2, success=True):
    """Create a single mock result dict.

    Parameters
    ----------
    i : int
        Grid index for param1 (param1 = i * 0.1)
    j : int
        Grid index for param2 (param2 = j * 0.1)
    chi2 : float
    success : bool

    Returns
    -------
    dict
    """
    param1 = i * 0.1
    param2 = j * 0.1
    return {
        'param1': param1,
        'param2': param2,
        'chi2': chi2,
        'params': {'param1': param1, 'param2': param2, 't_0': 2459000.0} if success else None,
        'success': success
    }


def _make_full_grid(chi2_func, success_func=None):
    """Create 11x11 grid of mock results.

    Parameters
    ----------
    chi2_func : callable
        (i, j) -> float
    success_func : callable or None
        (i, j) -> bool. If None, all points succeed.

    Returns
    -------
    list of dict
    """
    results = []
    for i in range(11):
        for j in range(11):
            chi2 = chi2_func(i, j)
            success = success_func(i, j) if success_func is not None else True
            results.append(_make_result(i, j, chi2, success))
    return results


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

def test_4_clear_minima(plot_grids):
    """Four distinct point minima, well-separated, at different chi2 levels."""
    minima_locs = {(2, 2): 115.0, (2, 8): 125.0, (8, 2): 135.0, (8, 8): 145.0}

    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(
        chi2_func=lambda i, j: minima_locs.get((i, j), 190.0)
    ))

    local_minima = grid.find_local_minima()

    assert len(local_minima) == 4, f"Expected 4 minima, got {len(local_minima)}"

    chi2_values = [chi2 for chi2, _, _ in local_minima]
    assert chi2_values == sorted(chi2_values), "Should be sorted by chi2"
    assert chi2_values[0] == 115.0

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test A: Clear 4 Minima")


def test_ring_constant_chi2(plot_grids):
    """Ring of equal chi2 values forms one connected plateau -> one minimum."""
    center_i, center_j = 5, 5
    radius = 3

    def chi2_func(i, j):
        dist = np.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
        return 115.0 if abs(dist - radius) < 0.5 else 190.0

    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(chi2_func))

    local_minima = grid.find_local_minima()
    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test B1: Ring - Constant Chi2")

    assert len(local_minima) == 1, \
        f"Constant ring should give 1 minimum, got {len(local_minima)}"


def test_ring_with_local_minima(plot_grids):
    """Ring with 4 distinct local minima at different depths."""
    center_i, center_j = 5, 5
    radius = 3
    minima_locs = {(8, 5): 115.0, (5, 8): 125.0, (2, 5): 135.0, (5, 2): 145.0}

    def chi2_func(i, j):
        dist = np.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
        if abs(dist - radius) < 0.5:
            return minima_locs.get((i, j), 155.0)
        return 190.0

    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(chi2_func))

    local_minima = grid.find_local_minima()

    assert len(local_minima) == 4, f"Expected 4 minima, got {len(local_minima)}"

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test B2: Ring - With Local Minima")


def test_arc_constant_chi2(plot_grids):
    """Diagonal arc of equal chi2 values forms one connected plateau -> one minimum."""
    def chi2_func(i, j):
        return 115.0 if abs(i - j) <= 1 else 190.0

    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(chi2_func))

    local_minima = grid.find_local_minima()

    assert len(local_minima) == 1, \
        f"Constant arc should give 1 minimum, got {len(local_minima)}"

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test C1: Arc - Constant Chi2")


def test_arc_with_local_minima(plot_grids):
    """Arc with 2 distinct local minima at endpoints."""
    arc_points = {(i, i) for i in range(2, 9)}
    minima_locs = {(2, 2): 115.0, (8, 8): 125.0}

    def chi2_func(i, j):
        if (i, j) in arc_points:
            return minima_locs.get((i, j), 155.0)
        return 190.0

    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(chi2_func))

    local_minima = grid.find_local_minima()

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test C2: Arc - With Local Minima")

    assert len(local_minima) == 2, f"Expected 2 minima, got {len(local_minima)}"

    chi2_values = [chi2 for chi2, _, _ in local_minima]
    assert chi2_values[0] == 115.0
    assert chi2_values[1] == 125.0


def test_single_minimum(plot_grids):
    """Single point minimum at grid center."""
    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(
        chi2_func=lambda i, j: 115.0 if (i, j) == (5, 5) else 190.0
    ))

    local_minima = grid.find_local_minima()

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test D: Single Minimum")

    assert len(local_minima) == 1, f"Expected 1 minimum, got {len(local_minima)}"
    assert local_minima[0][0] == 115.0


def test_failed_fits_excluded(plot_grids):
    """Failed fits (NaN chi2) should not prevent finding true minima."""
    def chi2_func(i, j):
        return 115.0 if (i, j) in [(2, 2), (8, 8)] else 190.0

    def success_func(i, j):
        return (i, j) in [(2, 2), (8, 8)] or (i + j) % 3 != 0

    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(chi2_func, success_func))

    local_minima = grid.find_local_minima()

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test E: Failed Fits Excluded")

    assert len(local_minima) == 2, f"Expected 2 minima, got {len(local_minima)}"
    assert all(np.isfinite(chi2) for chi2, _, _ in local_minima)


def test_asymmetric_minima(plot_grids):
    """Two adjacent points (one lower) plus one distant minimum -> 2 minima total."""
    chi2_map = {(2, 5): 114.0, (3, 5): 115.0, (8, 5): 125.0}

    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(
        chi2_func=lambda i, j: chi2_map.get((i, j), 190.0)
    ))

    local_minima = grid.find_local_minima()

    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test F: Asymmetric Minima")

    assert len(local_minima) == 2, f"Expected 2 minima, got {len(local_minima)}"
    assert local_minima[0][0] == 114.0, "Best minimum should be at (2,5)"
    assert local_minima[1][0] == 125.0, "Second minimum should be at (8,5)"


def test_minimum_at_edge(plot_grids):
    """Minimum at grid edge should be detected."""
    # Minima at each edge: left, right, top, bottom
    edge_minima = {
        (0, 5): 115.0,  # left edge
        (10, 5): 125.0,  # right edge
        (5, 0): 135.0,  # bottom edge
        (5, 10): 145.0  # top edge
    }

    grid = MockGridSearch()
    grid.set_mock_results(_make_full_grid(
        chi2_func=lambda i, j: edge_minima.get((i, j), 190.0)
    ))

    local_minima = grid.find_local_minima()
    if plot_grids:
        plot_grid_with_minima(grid, local_minima, "Test G: Edge Minima")

    assert len(local_minima) == 4, \
        f"Expected 4 edge minima, got {len(local_minima)}"

    chi2_values = [chi2 for chi2, _, _ in local_minima]
    assert chi2_values == sorted(chi2_values)
    assert chi2_values[0] == 115.0
