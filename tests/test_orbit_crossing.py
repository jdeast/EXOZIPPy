"""The orbit-crossing / Hill-sphere barrier (review 8.8.9).

Descended from EXOFASTv2's `exofast_chi2v2.pro`:445-468.  What was here
before was the bare apsidal test -- outer periastron outside inner apastron
-- on ADJACENT pairs only; then the faithful port, which padded each side by
that planet's OWN Hill radius.  What the threshold is stated in NOW is the
MUTUAL Hill radius,

    R_H = ((m1 + m2) / 3 M_*)^(1/3) (a1 + a2) / 2,

with a per-planet `min_hill_separation` defaulting to 1.0 -- EXOFASTv2's
one-Hill pad, so the DEFAULT strictness stays at the port's while the
formulation becomes the one the dynamics literature quotes separations in.
All pairs and EXOFASTv2's `alloworbitcrossing` off switch are unchanged.

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


def _penalty(model, name="planet.crossing_bound_b_c"):
    """The named crossing potential, evaluated at the start point."""
    crossing = [p for p in model.potentials if p.name == name]
    assert crossing, f"{name} must exist"
    fn = model.compile_fn(model.replace_rvs_by_values([crossing[0]]))
    point = model.initial_point()
    args = {k: v for k, v in point.items() if k in fn.f._finder}
    return float(fn(args)[0])


def test_the_hill_pad_penalizes_a_geometry_the_bare_apsidal_test_allows():
    """
    Given two planets whose apsides just miss but whose Hill spheres
      overlap,
    When the barrier is evaluated,
    Then it charges something.  This is exactly what the pre-port version
      did not do: the mutual Hill radius is what turns "the orbits do not
      intersect" into "the orbits are not within a Hill radius".
    """
    # Arrange: a pair packed close enough that the pads overlap.
    _, model = _system(2, periods={"b": 10.0, "c": 11.0})

    # Act / Assert: a real cost, not the ~0 an easily satisfied bound gives.
    assert _penalty(model) < -1.0


# ---------------------------------------------------------------------------
# 3b. The threshold is in MUTUAL Hill radii, and it is settable
# ---------------------------------------------------------------------------


def test_the_pad_is_the_mutual_hill_radius_not_the_sum_of_the_two_own_ones():
    """
    Given a packed pair,
    When the barrier's argument is reconstructed from the resolved start
      values both ways,
    Then it matches the MUTUAL form and not the per-planet-pad form that
      preceded it.  For an equal-mass pair at e = 0 the two differ by a
      constant factor -- summed own pads give (a1+a2)(m/3M)^(1/3) while
      the mutual radius gives 2^(1/3)/2 = 0.63 of that -- so the mutual
      threshold is the LOOSER of the two at the same k = 1, which is the
      only sense in which "the formulation changes, the default strictness
      does not" is inexact.

    Note the reference gap uses the resolved eccentricity rather than
    assuming zero: sqrt(e)cos/sin(omega) do not resolve to exactly 0, and
    an assumed-circular reference misses by ~2% of the penalty.
    """
    # Arrange
    system, model = _system(2, periods={"b": 10.0, "c": 11.0})
    planet = system.planet
    a = np.atleast_1d(planet.a.initval) * np.ones(2)
    m = np.atleast_1d(planet.mass.initval) * np.ones(2)
    m_star = np.atleast_1d(system.star.mass.initval)[0]
    ecc = np.atleast_1d(system.orbit.ecc.initval) * np.ones(2)

    gap = a[1] * (1.0 - ecc[1]) - a[0] * (1.0 + ecc[0])
    r_mutual = ((m[0] + m[1]) / (3.0 * m_star)) ** (1.0 / 3.0) * 0.5 * sum(a)
    own_pads = sum(
        (1.0 - ecc[k]) * a[k] * (m[k] / (3.0 * m_star)) ** (1.0 / 3.0)
        for k in (0, 1)
    )

    import pytensor.tensor as pt

    from exozippy.potentials import soft_lower_bound

    def barrier(pad):
        return float(
            soft_lower_bound(
                pt.as_tensor_variable(gap - pad), 0.0, scale=float(a[0])
            ).eval()
        )

    # Act
    measured = _penalty(model)

    # Assert
    assert measured == pytest.approx(barrier(r_mutual), rel=1e-6)
    assert measured != pytest.approx(barrier(own_pads), rel=1e-3)
    # ... and the mutual radius really is the smaller pad here.
    assert r_mutual < own_pads


def test_min_hill_separation_tightens_the_threshold():
    """
    Given a pair that is comfortably separated at the default 1 mutual
      Hill radius,
    When one planet asks for the ~8-12 mutual Hill radii the empirical
      long-term stability literature quotes,
    Then the same geometry is now penalized.  A pair takes the LARGER of
      its two members' values, since the number says how much room that
      planet needs.
    """
    # Arrange / Act
    _, relaxed = _system(2, periods={"b": 10.0, "c": 30.0})
    _, strict = _system(
        2,
        periods={"b": 10.0, "c": 30.0},
        c={"min_hill_separation": 10.0},
    )

    # Assert
    assert _penalty(relaxed) == pytest.approx(0.0, abs=1e-9)
    assert _penalty(strict) < -1.0


def test_min_hill_separation_is_a_declared_config_key():
    """
    Given the schema a config is validated against,
    When it is read,
    Then `min_hill_separation` is in it -- an undeclared key is silently
      ignored, which is exactly what a stability prior must not be.
    """
    from exozippy.components.planet.planet import Planet

    assert "min_hill_separation" in {
        entry["key"] for entry in Planet.config_schema()
    }


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
