"""Transit's phased panels fold on the planet's OWN orbit (review 1.5.7).

``orbit.period`` and ``orbit.tc`` are per-ORBIT vectors, but
``Transit._phased_lc_arrays`` is called per PLANET.  It used to read them at
the planet index, which agrees with the orbit index only when planet i
happens to sit on orbit i -- true in every shipped transit example, never
by contract (``planet.orbit_ndx`` is free, and the orbit list can carry
stellar orbits too).  Otherwise every phased panel folded on another
orbit's period and tc, silently.

The fixture lists the orbits in the OPPOSITE order to the planets, so
planet 0 sits on orbit 1 and planet 1 on orbit 0.  No graph is rebuilt for
the fix: the plot already reads period/tc from the evaluated point, and
only the element index changes.
"""

import numpy as np
import pytest

from exozippy.system import System

_P_B, _P_C = 4.0, 11.0
_TC_B, _TC_C = 2459200.0, 2459203.3


@pytest.fixture(scope="module")
def swapped_orbits(tmp_path_factory):
    """Two planets whose orbits are listed in reverse order."""
    path = tmp_path_factory.mktemp("swapped_lc") / "two.TESS.dat"
    t = np.linspace(_TC_B - 0.25, _TC_C + 0.25, 600)
    flux = (
        1.0
        - 0.01 * (np.abs(t - _TC_B) < 0.05)
        - 0.01 * (np.abs(t - _TC_C) < 0.05)
    )
    np.savetxt(path, np.column_stack([t, flux, np.full_like(t, 3.0e-4)]))

    config = {
        "run": {"name": "swapped_lc"},
        "star": [{"name": "A", "mist": False}],
        # planet b (index 0) -> orbit 1; planet c (index 1) -> orbit 0.
        "planet": [
            {"name": "b", "orbit_ndx": 1},
            {"name": "c", "orbit_ndx": 0},
        ],
        "orbit": [
            {"name": "c", "primary": ["A"], "companion": ["c"]},
            {"name": "b", "primary": ["A"], "companion": ["b"]},
        ],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "TESS", "file": str(path), "band": "TESS"}],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "orbit.b.period": {"initval": _P_B},
        "orbit.b.tc": {"initval": _TC_B},
        "orbit.c.period": {"initval": _P_C},
        "orbit.c.tc": {"initval": _TC_C},
    }
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.compile_plotter_functions(model)
    return system, point


def test_fixture_really_puts_planet_zero_on_orbit_one(swapped_orbits):
    """Guard: the config must actually exercise planet index != orbit index,
    or the test below passes vacuously."""
    system, _ = swapped_orbits
    assert list(system.planet.names) == ["b", "c"]
    assert list(system.orbit.names) == ["c", "b"]
    np.testing.assert_array_equal(system.planet.orbit_map, [1, 0])


def test_phased_arrays_fold_on_the_planets_own_orbit(swapped_orbits):
    """
    Given planet b on orbit 1 and planet c on orbit 0,
    When _phased_lc_arrays folds each planet,
    Then it uses that planet's own orbit period and tc, not the orbit at
    the planet's index.
    """
    system, point = swapped_orbits
    comp = system.transit
    b = comp._phased_lc_arrays(system, point, 0, 0)
    c = comp._phased_lc_arrays(system, point, 1, 0)
    assert np.isclose(b["P_ref"], _P_B) and np.isclose(b["tc_ref"], _TC_B)
    assert np.isclose(c["P_ref"], _P_C) and np.isclose(c["tc_ref"], _TC_C)


def test_plot_data_phased_meta_carries_each_planets_orbit(swapped_orbits):
    """The chart the PDF and GUI draw reports the same fold."""
    system, point = swapped_orbits
    specs = system.transit.plot_data(system, point)
    phased = {
        s.meta["planet"]: s.meta for s in specs if s.meta.get("phase_folded")
    }
    assert set(phased) == {"b", "c"}
    assert np.isclose(phased["b"]["period"], _P_B)
    assert np.isclose(phased["b"]["tc"], _TC_B)
    assert np.isclose(phased["c"]["period"], _P_C)
    assert np.isclose(phased["c"]["tc"], _TC_C)
