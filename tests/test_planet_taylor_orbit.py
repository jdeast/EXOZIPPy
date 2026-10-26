"""A PLANETARY lens companion on a linear (Taylor) orbit (review 8.8.20).

Linear lens orbital motion (C24) was only ever exercised with a stellar
companion (examples/ob09020), and a planet companion was refused outright:
Planet derived its Keplerian geometry (p, a, a/R*, b, K, teq, the transit
durations) whenever ANY orbit component existed, so a planet on a Taylor
orbit would have read that orbit's inactive Keplerian bookkeeping, and
Orbit._taylor_consumers refused every planet on a Taylor orbit.  The question
is now per planet (Planet.keplerian_geometry), and the refusal only fires
where a transit would model the planet.  Pinned here: the planetary lens
builds, its companion moves, and the planet carries no Keplerian geometry.
"""

import numpy as np
import pytensor
import pytest
from test_lens_orbital_motion import _T0_PAR, _write_binary_lc

pytestmark = pytest.mark.slow


def _planet_lens_system(tmp_path):
    from exozippy.system import System

    lc = _write_binary_lc(tmp_path / "lc.dat")
    config = {
        "star": [{"name": "Host"}, {"name": "Source"}],
        "planet": [{"name": "b"}],
        "mulensevent": [{"finite_source": True, "t0_par": _T0_PAR}],
        "lens": [{"body": "star.Host"}, {"body": "planet.b", "orbit": "L"}],
        "orbit": [
            {
                "name": "L",
                "type": "linear",
                "primary": ["Host"],
                "companion": ["b"],
            }
        ],
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    params = {
        "source.Source.t_0": {"initval": _T0_PAR},
        "source.Source.u_0": {"initval": 0.06},
        "mulensevent.t_E": {"initval": 60.0},
        "source.Source.rho": {"initval": 0.0015},
        "lens.b.s": {"initval": 1.1},
        "lens.b.alpha": {"initval": 120.0},
        "lens.b.q": {"initval": 3e-3},
        "mulensevent.pi_E_N": {"initval": -0.02},
        "mulensevent.pi_E_E": {"initval": 0.15},
        "orbit.L.ds_dt": {"initval": 0.1},
        "orbit.L.dalpha_dt": {"initval": 30.0},
        "star.Host.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("Host", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 268.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -29.0, "sigma": 0}
    system = System(config, user_params=params)
    system.prepare()
    return system


def test_a_planet_companion_moves_on_a_linear_lens_orbit(tmp_path):
    system = _planet_lens_system(tmp_path)
    assert system.lens.companion_motion == ["linear"]
    # no Keplerian geometry: nothing to derive it from
    assert not system.planet.keplerian_geometry(system)
    for name in ("p", "a", "ar", "K", "teq", "t14"):
        assert name not in system.planet.manifest, name
    model = system.build_model()
    times = np.linspace(_T0_PAR - 30.0, _T0_PAR + 30.0, 21)
    S, _ = system.mulensevent._companion_geometry_series(times, system)
    start = system.get_raw_start(model)
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([S]),
        on_unused_input="ignore",
    )
    (S,) = fn(*[start[v.name] for v in model.value_vars])
    assert np.ptp(S[0]) > 1e-4
    assert np.isfinite(model.compile_logp()(model.initial_point()))


# ---------------------------------------------------------------------------
# A static planet beside a moving stellar companion (OGLE-2013-BLG-0341)
# ---------------------------------------------------------------------------
def _triple_lens_config(lc, with_static_orbit=True):
    """Star + star + planet lens: the stellar companion L2 moves on a linear
    orbit, the planet b does not move and is paired with its host L1 by a
    `type: static` orbit (the OGLE-2013-BLG-0341 shape)."""
    orbits = [
        {
            "name": "L",
            "type": "linear",
            "primary": ["L1"],
            "companion": ["L2"],
        }
    ]
    if with_static_orbit:
        orbits.append(
            {
                "name": "bL1",
                "type": "static",
                "primary": ["L1"],
                "companion": ["b"],
            }
        )
    return {
        "star": [{"name": "L1"}, {"name": "L2"}, {"name": "Source"}],
        "planet": [{"name": "b"}],
        "mulensevent": [{"finite_source": True, "t0_par": _T0_PAR}],
        "lens": [
            {"body": "star.L1"},
            {"body": "star.L2", "orbit": "L"},
            {"body": "planet.b"},
        ],
        "orbit": orbits,
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }


def _triple_lens_params():
    params = {
        "source.Source.t_0": {"initval": _T0_PAR},
        "source.Source.u_0": {"initval": 0.06},
        "mulensevent.t_E": {"initval": 60.0},
        "source.Source.rho": {"initval": 0.0015},
        "lens.L2.s": {"initval": 1.1},
        "lens.L2.alpha": {"initval": 120.0},
        "lens.L2.q": {"initval": 0.3},
        "lens.b.s": {"initval": 0.9},
        "lens.b.alpha": {"initval": 40.0},
        "lens.b.q": {"initval": 3e-3},
        "mulensevent.pi_E_N": {"initval": -0.02},
        "mulensevent.pi_E_E": {"initval": 0.15},
        "orbit.L.ds_dt": {"initval": 0.1},
        "orbit.L.dalpha_dt": {"initval": 30.0},
        "star.L1.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("L1", "L2", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 268.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -29.0, "sigma": 0}
    return params


def test_a_static_planet_builds_beside_a_linear_stellar_companion(tmp_path):
    """
    Given a three-body lens (OGLE-2013-BLG-0341's shape): star L1, a stellar
      companion L2 on a linear lens orbit, and a planet b that does not move,
      paired with L1 by a `type: static` orbit,
    When the system is prepared and built,
    Then it builds: b's host is L1 and its orbit is the static one (never
      the linear orbit, which an orbit_ndx default of 0 used to pick and
      Orbit._taylor_consumers refused), b carries no Keplerian geometry, the
      static orbit declares nothing, L2 moves and b stays put.
    """
    from exozippy.system import System

    lc = _write_binary_lc(tmp_path / "lc.dat")
    system = System(_triple_lens_config(lc), user_params=_triple_lens_params())
    system.prepare()

    assert system.planet.orbit_ndx == [1]
    assert list(system.planet.star_map) == [0]
    assert list(system.orbit.is_static) == [False, True]
    assert system.lens.companion_motion == ["linear", None]
    assert not system.planet.keplerian_geometry(system)
    for name in ("p", "a", "ar", "K", "teq", "t14"):
        assert name not in system.planet.manifest, name

    model = system.build_model()
    # The orbits' only sampled elements are the linear orbit's rates: the
    # static orbit declares no parameter at all.
    orbit_rvs = sorted(
        v.name for v in model.free_RVs if v.name.startswith("orbit.")
    )
    assert orbit_rvs == ["orbit.dalpha_dt_raw", "orbit.ds_dt_raw"]

    times = np.linspace(_T0_PAR - 30.0, _T0_PAR + 30.0, 21)
    S, _ = system.mulensevent._companion_geometry_series(times, system)
    start = system.get_raw_start(model)
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([S]),
        on_unused_input="ignore",
    )
    (S,) = fn(*[start[v.name] for v in model.value_vars])
    assert np.ptp(S[0]) > 1e-4  # L2 moves
    assert np.ptp(S[1]) == 0.0  # b does not
    assert np.isfinite(model.compile_logp()(start))


def test_without_the_static_orbit_the_planet_host_is_ambiguous(tmp_path):
    """
    Given the same lens with NO orbit block naming the planet,
    When the system is constructed,
    Then it RAISES asking for the orbit block: the lens holds two stars, so
      the planet's host is not left implicit -- and nothing falls back to
      orbit 0 (the linear orbit) any more.
    """
    from exozippy.system import System

    lc = _write_binary_lc(tmp_path / "lc.dat")
    with pytest.raises(ValueError, match="type: static"):
        System(
            _triple_lens_config(lc, with_static_orbit=False),
            user_params=_triple_lens_params(),
        )
