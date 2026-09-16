"""Phased panels compute the point-only work once (review 6.5.1).

`_phased_arrays` (RV) and `_phased_lc_arrays` (transit) are called once per
orbit and once per (planet, instrument) respectively, and each rebuilt the
same point-dependent arrays every time: the marshalled parameter values,
the model matrix at the OBSERVED times, and the per-observation GP and
detrend corrections.  The spaghetti re-runs the whole loop per posterior
draw, so the waste multiplies by the number of draws.

Both are hoisted to a `_phased_*_shared` dict built once per (component,
point).  The tests below pin BOTH halves: that the compiled evaluator is
called the hoisted number of times, and that the arrays are unchanged.
"""

import numpy as np
import pytest

from exozippy.system import System

_P1, _P2 = 4.0, 11.0
_TC = 2459200.0


def _star_params():
    return {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
    }


# --------------------------------------------------------------------------
# RV: two member orbits
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_orbit_rv(tmp_path_factory):
    """One RV instrument on a star with TWO planets (two member orbits)."""
    path = tmp_path_factory.mktemp("hoist_rv") / "two.rv"
    t = np.linspace(2459180.0, 2459260.0, 60)
    rv = 30.0 * np.sin(2 * np.pi * (t - _TC) / _P1) + 12.0 * np.sin(
        2 * np.pi * (t - _TC) / _P2
    )
    np.savetxt(path, np.column_stack([t, rv, np.full_like(t, 4.0)]))

    config = {
        "run": {"name": "hoist_rv"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}, {"name": "c"}],
        "orbit": [
            {"name": "b", "primary": ["A"], "companion": ["b"]},
            {"name": "c", "primary": ["A"], "companion": ["c"]},
        ],
        "rvinstrument": [{"name": "HIRES", "file": str(path)}],
    }
    params = {
        **_star_params(),
        "orbit.b.period": {"initval": _P1},
        "orbit.b.tc": {"initval": _TC},
        "orbit.c.period": {"initval": _P2},
        "orbit.c.tc": {"initval": _TC},
    }
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.compile_plotter_functions(model)
    return system, point


def test_rv_matrix_is_evaluated_once_per_orbit_plus_once_for_the_data(
    two_orbit_rv, monkeypatch
):
    """
    Given two member orbits,
    When plot_data builds the unphased panel and both phased panels,
    Then the compiled plot-grid model is evaluated three times -- once for
    the unphased span and one period window per orbit (their time grids
    differ, so those cannot be shared) -- and the likelihood's own model at
    the observed times exactly ONCE, hoisted out of the per-orbit loop.

    Before the hoist the data-times pass was repeated per orbit, and it is
    the expensive one on a real data set.  (It used to be a second plot
    graph evaluated at the data times; it is the likelihood's node now,
    compiled against the plot parameters -- reviews 1.5.5, 7.14.1.)
    """
    system, point = two_orbit_rv
    comp = system.rvinstrument
    assert len(comp._plot_orbit_map) == 2

    grid_calls = []
    real_grid = comp._rv_grid_fn
    monkeypatch.setattr(
        comp,
        "_rv_grid_fn",
        lambda t, *a: (grid_calls.append(len(t)), real_grid(t, *a))[1],
    )
    data_calls = []
    real_data = comp._rv_data_fn
    monkeypatch.setattr(
        comp,
        "_rv_data_fn",
        lambda *a: (data_calls.append(1), real_data(*a))[1],
    )

    comp.plot_data(system, point)

    assert len(grid_calls) == 3
    assert len(data_calls) == 1


def test_rv_phased_arrays_are_unchanged_by_the_hoist(two_orbit_rv):
    """
    Given the shared dict the loop now passes down,
    When _phased_arrays is called with it and without it,
    Then every array is identical -- the hoist is an optimization only.
    """
    system, point = two_orbit_rv
    comp = system.rvinstrument
    shared = comp._phased_shared(system, point)

    for col, o_idx in enumerate(comp._plot_orbit_map):
        a = comp._phased_arrays(system, point, col, o_idx)
        b = comp._phased_arrays(system, point, col, o_idx, shared=shared)
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


# --------------------------------------------------------------------------
# Transit: two planets on one light curve
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def two_planet_transit(tmp_path_factory):
    """One light curve of a star with TWO planets."""
    path = tmp_path_factory.mktemp("hoist_lc") / "two.TESS.dat"
    n = 240
    t = np.linspace(_TC - 0.25, _TC + 0.25, n)
    flux = 1.0 - 0.01 * (np.abs(t - _TC) < 0.05)
    np.savetxt(path, np.column_stack([t, flux, np.full_like(t, 3.0e-4)]))

    config = {
        "run": {"name": "hoist_lc"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}, {"name": "c"}],
        "orbit": [
            {"name": "b", "primary": ["A"], "companion": ["b"]},
            {"name": "c", "primary": ["A"], "companion": ["c"]},
        ],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "TESS", "file": str(path), "band": "TESS"}],
    }
    params = {
        **_star_params(),
        "orbit.b.period": {"initval": _P1},
        "orbit.b.tc": {"initval": _TC},
        "orbit.c.period": {"initval": _P2},
        "orbit.c.tc": {"initval": _TC},
    }
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.compile_plotter_functions(model)
    return system, point


def test_lc_matrix_at_the_data_is_evaluated_once_per_instrument(
    two_planet_transit, monkeypatch
):
    """
    Given two planets on one light curve,
    When plot_data builds the unphased panel and both phased panels,
    Then the compiled plot-grid model is evaluated three times -- once for
    the unphased span and one period window per planet (different time
    grids), each serving every instrument at once -- and the likelihood's
    own model at the observed times exactly ONCE, shared by every phased
    panel.

    Before the hoist the observed-times pass ran once per PLANET.  (It
    used to be a second plot graph smeared in NumPy at the data times; it
    is the likelihood's node now -- reviews 1.5.6, 7.14.1.)
    """
    system, point = two_planet_transit
    comp = system.transit
    assert system.planet.n_elements == 2

    grid_calls = []
    real_grid = comp._lc_grid_fn
    monkeypatch.setattr(
        comp,
        "_lc_grid_fn",
        lambda t, *a: (grid_calls.append(len(t)), real_grid(t, *a))[1],
    )
    data_calls = []
    real_data = comp._lc_data_fn
    monkeypatch.setattr(
        comp,
        "_lc_data_fn",
        lambda *a: (data_calls.append(1), real_data(*a))[1],
    )

    comp.plot_data(system, point)

    assert len(grid_calls) == 3
    assert len(data_calls) == 1


def test_lc_phased_arrays_are_unchanged_by_the_hoist(two_planet_transit):
    """
    Given the shared dict the loop now passes down,
    When _phased_lc_arrays is called with it and without it,
    Then every array is identical.
    """
    system, point = two_planet_transit
    comp = system.transit
    shared = comp._phased_lc_shared(system, point)

    for p_idx in range(system.planet.n_elements):
        a = comp._phased_lc_arrays(system, point, p_idx, 0)
        b = comp._phased_lc_arrays(system, point, p_idx, 0, shared=shared)
        assert a.keys() == b.keys()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
