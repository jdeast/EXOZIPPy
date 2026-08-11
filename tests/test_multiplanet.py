"""Multi-planet systems and the planet-planet non-crossing barrier.

Nothing in examples/ or tests/ built a system with more than one planet, so
``Planet.build_likelihood``'s ``n_elements >= 2`` branch was dark code and had
rotted into an AttributeError (it walked a ``self.planets`` list of per-planet
objects that has not existed since the vectorized refactor, and reached for
``inner.orbit.a.value`` / ``outer.orbit.a_val``, neither of which exists).
Layer 1 here is the coverage that keeps that path lit.

Layer 2 pins the barrier itself.  It used to be
``pt.switch(ok, 0.0, -np.inf)``: the right value and no gradient at all for
NUTS to follow out of the forbidden region, so the assertions below are on
the GRADIENT, not only on the value.

The companion regression for the m_pos_constraint plateau (review item
2.4.2) lives in tests/test_planet_mass_barrier.py -- it needs only one
planet, so keeping it out of this file lets each item fail on its own.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.replace import graph_replace

from exozippy.system import System

# ---------------------------------------------------------------------------
# Fixtures: one two-planet RV system, built once
# ---------------------------------------------------------------------------
_P_INNER = 20.0  # planet 'c'
_P_OUTER = 111.0  # planet 'b'


def _write_rv(path, seed, n=40):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, n))
    rv = (
        30.0 * np.sin(2 * np.pi * t / _P_OUTER)
        + 10.0 * np.sin(2 * np.pi * t / _P_INNER)
        + rng.normal(0, 3.0, n)
    )
    np.savetxt(path, np.column_stack([t, rv, np.full(n, 3.0)]))
    return path


def _build(planets, orbits, params=None, files=None):
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": planets,
        "orbit": orbits,
        "rvinstrument": [{"name": "HIRES", "file": files[0]}],
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
    }
    user_params.update(params or {})
    system = System(config, user_params)
    system.prepare()
    return system, system.build_model()


@pytest.fixture(scope="module")
def rv_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("multiplanet_rv")
    return str(_write_rv(d / "two_planets.rv", 7))


@pytest.fixture(scope="module")
def two_planet_system(rv_file):
    """A star with two planets on two orbits, measured by one RV data set.

    RV data put both planets on the mass-constrained side, so the mass is
    sampled in 'linear' mode -- the mode in which the m_pos_constraint
    plateau of item 2.4.2 is reachable.
    """
    return _build(
        planets=[{"name": "b"}, {"name": "c", "orbit_ndx": 1}],
        orbits=[
            {"name": "b", "primary": ["A"], "companion": ["b"]},
            {"name": "c", "primary": ["A"], "companion": ["c"]},
        ],
        params={
            "orbit.b.logP": {"initval": np.log10(_P_OUTER)},
            "orbit.b.tc": {"initval": 2455010.0},
            "orbit.c.logP": {"initval": np.log10(_P_INNER)},
            "orbit.c.tc": {"initval": 2455005.0},
        },
        files=[rv_file],
    )


@pytest.fixture(scope="module")
def one_planet_system(rv_file):
    return _build(
        planets=[{"name": "b"}],
        orbits=[{"name": "b", "primary": ["A"], "companion": ["b"]}],
        params={
            "orbit.b.logP": {"initval": np.log10(_P_OUTER)},
            "orbit.b.tc": {"initval": 2455010.0},
        },
        files=[rv_file],
    )


def _potential(model, needle):
    hits = [p for p in model.potentials if needle in p.name]
    assert len(hits) == 1, (
        f"{needle}: found {[p.name for p in model.potentials]}"
    )
    return hits[0]


# ---------------------------------------------------------------------------
# 1. A >= 2 planet system builds at all (the coverage gap)
# ---------------------------------------------------------------------------
def test_two_planet_system_builds(two_planet_system):
    """
    Given a star with two planets on two orbits,
    When the model is built,
    Then it builds -- build_likelihood's multi-planet branch used to raise
    AttributeError('Planet' object has no attribute 'planets') here.
    """
    system, model = two_planet_system

    assert system.planet.n_elements == 2
    assert system.planet.names == ["b", "c"]
    # Vector parameters really do carry both planets.
    assert len(np.atleast_1d(system.planet.mass.initval)) == 2
    assert "planet.mass_raw" in {v.name for v in model.free_RVs}


def test_two_planet_start_is_finite(two_planet_system):
    """
    Given the two-planet model,
    When logp and its gradient are evaluated at the start point,
    Then both are finite -- the old -inf crossing switch could kill the
    start outright, and a NaN gradient freezes every JAX chain.
    """
    _, model = two_planet_system

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert all(np.all(np.isfinite(g)) for g in model.compile_dlogp()(point))


def test_two_planet_mass_mode_is_linear(two_planet_system):
    """RV data measure both orbits, so the signed linear mass is sampled --
    which is what makes the m_pos_constraint region below reachable."""
    system, _ = two_planet_system
    assert system.planet.mass_parameterization == "linear"


# ---------------------------------------------------------------------------
# 2. The planet-planet non-crossing barrier
# ---------------------------------------------------------------------------
def test_crossing_bound_is_added_once_ordered_inner_to_outer(
    two_planet_system,
):
    """
    Given two planets whose starting semi-major axes differ,
    When the model is built,
    Then exactly one non-crossing barrier is added, named inner-then-outer
    (planet 'c' is the shorter period, so it is the inner one).
    """
    system, model = two_planet_system

    names = [p.name for p in model.potentials if "crossing_bound" in p.name]
    assert names == ["planet.crossing_bound_c_b"]

    a_init = system.planet._initial_semimajor_axes(system, system.orbit)
    assert a_init[1] < a_init[0]  # c inside b
    assert np.all(np.isfinite(a_init))


def test_single_planet_system_has_no_crossing_bound(one_planet_system):
    """One planet cannot cross anything; the barrier must not appear."""
    _, model = one_planet_system
    assert not [p for p in model.potentials if "crossing_bound" in p.name]


def _crossing_fn(system, model):
    """Compile the crossing barrier and its gradient as a function of the
    per-planet semi-major axes and the per-orbit eccentricities."""
    pot = _potential(model, "crossing_bound")
    a = pt.dvector("a")
    ecc = pt.dvector("ecc")
    node = graph_replace(
        pot,
        {system.planet.arsun.value: a, system.orbit.ecc.value: ecc},
    )
    return pytensor.function(
        [a, ecc], [node, pt.grad(node.sum(), a)], on_unused_input="ignore"
    )


@pytest.mark.parametrize("ratio", [1.5, 3.0])
def test_crossing_bound_is_inert_for_separated_orbits(
    two_planet_system, ratio
):
    """
    Given circular orbits with the outer one well outside the inner one,
    When the barrier is evaluated,
    Then it is negligible -- the constraint costs nothing where it should.
    """
    system, model = two_planet_system
    f = _crossing_fn(system, model)

    a_init = system.planet._initial_semimajor_axes(system, system.orbit)
    a = a_init.copy()
    a[0] = a[1] * ratio  # outer 'b' pushed outside inner 'c'

    value, _ = f(a, np.zeros(system.orbit.n_elements))
    assert value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("ratio", [0.95, 0.5, 0.1])
def test_crossing_bound_pushes_the_orbits_apart(two_planet_system, ratio):
    """
    Given an outer planet whose periastron has fallen inside the inner
    planet's apastron,
    When the barrier and its gradient are evaluated,
    Then the penalty is finite (not -inf) and its gradient is a restoring
    force: it pushes the outer semi-major axis out and the inner one in.

    This is the assertion the old pt.switch(..., 0.0, -np.inf) fails: -inf
    has the right sign and no gradient at all for NUTS to follow.
    """
    system, model = two_planet_system
    f = _crossing_fn(system, model)

    a_init = system.planet._initial_semimajor_axes(system, system.orbit)
    a = a_init.copy()
    a[0] = a[1] * ratio

    value, grad = f(a, np.zeros(system.orbit.n_elements))

    assert np.isfinite(value)
    assert value < -1.0
    assert np.all(np.isfinite(grad))
    assert grad[0] > 0.0  # push the outer planet ('b') outward
    assert grad[1] < 0.0  # pull the inner planet ('c') inward


def test_crossing_bound_deepens_monotonically(two_planet_system):
    """Deeper inside the forbidden region costs more, so the sampler always
    has a direction to walk -- a plateau (or a -inf) would not."""
    system, model = two_planet_system
    f = _crossing_fn(system, model)

    a_init = system.planet._initial_semimajor_axes(system, system.orbit)
    zeros = np.zeros(system.orbit.n_elements)

    values = []
    for ratio in (0.9, 0.7, 0.5):
        a = a_init.copy()
        a[0] = a[1] * ratio
        values.append(float(f(a, zeros)[0]))

    assert values[0] > values[1] > values[2]


def test_crossing_bound_responds_to_eccentricity(two_planet_system):
    """
    Given two nested circular orbits that do not cross,
    When the inner planet's eccentricity is raised until its apastron
    reaches the outer planet's periastron,
    Then the barrier turns on -- the constraint is on the apses, not on the
    semi-major axes alone.
    """
    system, model = two_planet_system
    f = _crossing_fn(system, model)

    a_init = system.planet._initial_semimajor_axes(system, system.orbit)
    a = a_init.copy()
    a[0] = a[1] * 1.5

    circular = float(f(a, np.zeros(system.orbit.n_elements))[0])
    # orbit index 1 is planet 'c' (the inner one): e = 0.8 puts its apastron
    # at 1.8 a_c, well outside the outer planet's 1.5 a_c periastron.
    eccentric = np.zeros(system.orbit.n_elements)
    eccentric[system.planet.orbit_map[1]] = 0.8
    assert float(f(a, eccentric)[0]) < circular - 1.0
