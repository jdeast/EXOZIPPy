"""The orbit-crossing / Hill-sphere barrier (review 8.8.9).

The port of EXOFASTv2's `exofast_chi2v2.pro`:445-468.  What was here before
was the bare apsidal test -- outer periastron outside inner apastron -- on
ADJACENT pairs only.  What this adds is the Hill-radius pad on each side, all
pairs, and EXOFASTv2's `alloworbitcrossing` off switch, spelled per planet.

Kept SOFT, deliberately: EXOFASTv2 returns +infinity chi2, and a wall with no
gradient is nothing a gradient sampler can follow out of.
"""

import numpy as np
import pytest

from exozippy.system import System


def _config(n_planets, **planet_extra):
    names = ["b", "c", "d"][:n_planets]
    planets = []
    for k, nm in enumerate(names):
        # orbit_ndx is what puts each planet on its OWN orbit; without it
        # every planet maps to orbit 0 and the pairs are skipped as
        # co-orbital.
        cfg = {"name": nm, "orbit_ndx": k}
        cfg.update(planet_extra.get(nm, {}))
        planets.append(cfg)
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": planets,
        "orbit": [
            {"name": nm, "primary": ["A"], "companion": [nm]} for nm in names
        ],
    }


_PERIODS = {"b": 3.0, "c": 8.0, "d": 30.0}


def _system(n_planets, periods=None, eccs=None, **planet_extra):
    periods = periods or _PERIODS
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
    }
    for nm in ["b", "c", "d"][:n_planets]:
        params[f"orbit.{nm}.period"] = {"initval": periods[nm]}
        params[f"orbit.{nm}.tc"] = {"initval": 2455010.0}
        params[f"planet.{nm}.radius"] = {"initval": 1.0}
        if eccs and nm in eccs:
            params[f"orbit.{nm}.ecc"] = {"initval": eccs[nm]}
            params[f"orbit.{nm}.omega"] = {"initval": 90.0}
    system = System(_config(n_planets, **planet_extra), params)
    system.prepare()
    return system, system.build_model()


def _bounds(model):
    return {
        p.name
        for p in model.potentials
        if p.name.startswith("planet.crossing_bound_")
    }


# ---------------------------------------------------------------------------
# 1. Coverage: all pairs, not just adjacent
# ---------------------------------------------------------------------------


def test_every_pair_is_constrained_not_only_adjacent_ones():
    """
    Given three planets,
    When the model is built,
    Then all three PAIRS carry a barrier, not the two adjacent ones.
      Adjacent-only was an approximation with nothing to recommend it: a
      widely eccentric outer planet can cross the innermost while missing
      its neighbour.
    """
    # Arrange / Act
    _, model = _system(3)

    # Assert
    assert _bounds(model) == {
        "planet.crossing_bound_b_c",
        "planet.crossing_bound_c_d",
        "planet.crossing_bound_b_d",
    }


def test_a_single_planet_gets_no_barrier():
    """
    Given one planet,
    When the model is built,
    Then there is no pair and no potential.
    """
    _, model = _system(1)
    assert _bounds(model) == set()


# ---------------------------------------------------------------------------
# 2. The off switch
# ---------------------------------------------------------------------------


def test_allow_orbit_crossing_exempts_every_pair_the_planet_is_in():
    """
    Given `allow_orbit_crossing: true` on the middle planet,
    When the model is built,
    Then only the pair that does not involve it survives.  Per planet and
      not per fit because the exception exists for a PAIR -- a resonant or
      co-orbital one -- and naming a member is how a user says which.
    """
    _, model = _system(3, c={"allow_orbit_crossing": True})
    assert _bounds(model) == {"planet.crossing_bound_b_d"}


def test_the_switch_is_a_declared_config_key():
    """
    Given the schema a config is validated against,
    When it is read,
    Then `allow_orbit_crossing` is in it -- an undeclared key is the
      silently-ignored failure the WIP guards exist to prevent.
    """
    from exozippy.components.planet.planet import Planet

    assert "allow_orbit_crossing" in {
        entry["key"] for entry in Planet.config_schema()
    }


# ---------------------------------------------------------------------------
# 3. The Hill pad is really there, and it is a soft barrier
# ---------------------------------------------------------------------------


def _logp(model):
    return float(model.compile_logp()(model.initial_point()))


def test_the_hill_pad_penalizes_a_geometry_the_bare_apsidal_test_allows():
    """
    Given two planets whose apsides just miss but whose Hill spheres
      overlap,
    When the barrier is evaluated,
    Then it charges something.  This is exactly what the pre-port version
      did not do: `h = (1-e) a (m_p/3 M_*)^(1/3)` is what turns "the orbits
      do not intersect" into "the orbits are not within a Hill radius".
    """
    # Arrange: a pair packed close enough that the pads overlap.
    system, model = _system(
        2,
        periods={"b": 10.0, "c": 11.0},
    )
    crossing = [
        p for p in model.potentials if p.name == "planet.crossing_bound_b_c"
    ]
    assert crossing, "the pair must carry a barrier"

    # Act
    point = model.initial_point()
    penalty = float(
        model.compile_fn(model.replace_rvs_by_values([crossing[0]]))(
            {
                k: v
                for k, v in point.items()
                if k
                in model.compile_fn(
                    model.replace_rvs_by_values([crossing[0]])
                ).f._finder
            }
        )[0]
    )

    # Assert: a real cost, not the ~0 an easily satisfied bound gives.
    assert penalty < -1.0


def test_a_widely_separated_pair_pays_almost_nothing():
    """
    Given two planets a factor of ten apart in period,
    When the barrier is evaluated,
    Then the start logp is finite and its gradient is too -- the barrier is
      a penalty, not a wall, so nothing is excluded and nothing is NaN.
    """
    _, model = _system(2, periods={"b": 3.0, "c": 300.0})
    point = model.initial_point()
    assert np.isfinite(_logp(model))
    grads = model.compile_dlogp()(point)
    assert np.isfinite(np.atleast_1d(grads)).all()


def test_the_barrier_is_soft_and_finite_even_when_badly_violated():
    """
    Given two planets on nearly the same orbit -- a geometry EXOFASTv2
      rejects with +infinity chi2,
    When the start logp and gradient are evaluated,
    Then both are finite.  A hard wall gives NUTS nothing to follow out of
      the forbidden region and NaNs the JAX backward pass, which is why
      this half of the port is deliberately NOT faithful.
    """
    _, model = _system(2, periods={"b": 10.0, "c": 10.05})
    point = model.initial_point()
    logp = _logp(model)
    grads = model.compile_dlogp()(point)
    assert np.isfinite(logp)
    assert np.isfinite(np.atleast_1d(grads)).all()


@pytest.mark.parametrize("mass", [-5.0, 0.0])
def test_a_non_positive_planet_mass_keeps_the_gradient_finite(mass):
    """
    Given a planet whose mass is zero or negative -- legal in `linear` mass
      mode, where crossing zero is what avoids the Lucy-Sweeney bias,
    When the Hill radius is computed,
    Then the logp and gradient stay finite.  `x**(1/3)` is NaN below zero,
      and flooring AFTER the root would multiply its infinite derivative at
      zero by pt.maximum's zero gradient and give NaN again -- so the floor
      goes on the argument, the calc_theta_E idiom.
    """
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
    }
    for nm, period in (("b", 3.0), ("c", 8.0)):
        params[f"orbit.{nm}.period"] = {"initval": period}
        params[f"orbit.{nm}.tc"] = {"initval": 2455010.0}
        params[f"planet.{nm}.radius"] = {"initval": 1.0}
    params["planet.b.mass"] = {"initval": mass}

    system = System(_config(2, b={"mass_parameterization": "linear"}), params)
    system.prepare()
    model = system.build_model()

    point = model.initial_point()
    assert np.isfinite(_logp(model))
    assert np.isfinite(np.atleast_1d(model.compile_dlogp()(point))).all()
