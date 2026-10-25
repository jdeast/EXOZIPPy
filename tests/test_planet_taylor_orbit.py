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
