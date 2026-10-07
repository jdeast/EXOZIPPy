"""
Tests for outputs/contour_plot.py's Contour: the highest-density contour
levels of a 2-D sample.

They follow AAA with Given/When/Then docstrings.
"""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from exozippy.outputs.contour_plot import Contour


def _fraction_inside(contour, x, y, level):
    density = RegularGridInterpolator(
        (contour.X[:, 0], contour.Y[0, :]), contour.Z
    )(np.column_stack([x, y]))
    return float(np.mean(density >= level))


def test_each_level_encloses_its_probability_of_the_draws():
    """
    Given 4000 draws from a correlated 2-D Gaussian (a Teff-logg cloud),
    When Contour is asked for the 1- and 2-sigma levels with Scott's
      bandwidth,
    Then the levels come out in increasing density, and the region above
      each holds that fraction of the draws (to the KDE's smoothing).
    """
    rng = np.random.default_rng(3)
    cov = [[100.0**2, 0.8 * 100.0 * 0.03], [0.8 * 100.0 * 0.03, 0.03**2]]
    x, y = rng.multivariate_normal([5500.0, 4.4], cov, size=4000).T

    contour = Contour(
        x,
        y,
        x_err=np.std(x),
        y_err=np.std(y),
        bw_method="scott",
        probs=(0.9545, 0.6827),
    )

    two, one = contour.levels
    assert two < one
    assert _fraction_inside(contour, x, y, one) == _approx(0.6827)
    assert _fraction_inside(contour, x, y, two) == _approx(0.9545)


def _approx(value):
    """A probability to the KDE's smoothing of the draws."""
    return pytest.approx(value, abs=0.04)


def test_the_defaults_are_the_evolutionary_models_three_levels():
    """
    Given no bandwidth or probabilities,
    When a Contour is built,
    Then it keeps the evolutionary model's own contour plot unchanged:
      bandwidth 0.7 and the 99.7%, 95% and 68% levels, widest first.
    """
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(2, 500))

    contour = Contour(x, y, x_err=1.0, y_err=1.0)

    assert contour.bw_method == 0.7
    assert contour.probs == (0.997, 0.95, 0.68)
    assert len(contour.levels) == 3
    assert list(contour.levels) == sorted(contour.levels)
