"""planet.m_pos_constraint: the barrier keeping the total mass positive.

The barrier used to be applied to ``planet.m_total``, whose physics function
``calc_m_total`` clips the sum at 1e-9 solar masses.  A clipped input makes
the soft bound see a *constant* on the whole ``star.mass + planet.mass < 0``
region -- ``log(sigmoid(~0)) = -0.693`` -- so the penalty had the right sign,
a plausible magnitude, and a gradient of exactly zero.  That is a flat
plateau with no restoring force: precisely the pathology the log_q notes in
CLAUDE.md warn about, and it is reachable, because a planet whose mass RV or
astrometric data constrain samples the *signed* linear mass, whose lower
bound is -1000 Mjup (about -0.95 Msun).

So these tests assert on the GRADIENT.  A value-only test passes on the
broken code, which is how this survived.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.replace import graph_replace

from exozippy.system import System

_PERIOD = 17.0


@pytest.fixture(scope="module")
def rv_system(tmp_path_factory):
    """One star, one planet, one RV data set.

    RV data put the planet on the mass-constrained side, so the mass is
    sampled in 'linear' mode -- the mode in which the region below zero is
    reachable at all.
    """
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, 40))
    rv = 30.0 * np.sin(2 * np.pi * t / _PERIOD) + rng.normal(0, 3.0, 40)
    path = tmp_path_factory.mktemp("mass_barrier") / "a.rv"
    np.savetxt(path, np.column_stack([t, rv, np.full(40, 3.0)]))

    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": [{"name": "HIRES", "file": str(path)}],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(_PERIOD)},
        "orbit.b.tc": {"initval": 2455010.0},
    }
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


@pytest.fixture(scope="module")
def barrier(rv_system):
    """Compile m_pos_constraint and d/d(planet mass), as a function of the
    planet and star mass vectors in internal units (solar masses).

    The potential is cut out of the built model, so this measures what
    Planet.build_likelihood actually put there -- including, on the broken
    code, calc_m_total's pt.maximum clip.
    """
    system, model = rv_system

    hits = [p for p in model.potentials if "m_pos_constraint" in p.name]
    assert len(hits) == 1, [p.name for p in model.potentials]

    m_p = pt.dvector("m_p")
    m_s = pt.dvector("m_s")
    node = graph_replace(
        hits[0],
        {system.planet.mass.value: m_p, system.star.mass.value: m_s},
    )
    return pytensor.function(
        [m_p, m_s], [node, pt.grad(node.sum(), m_p)], on_unused_input="ignore"
    )


def _at(barrier, m_total, m_star=1.0):
    value, grad = barrier(np.array([m_total - m_star]), np.array([m_star]))
    return float(np.atleast_1d(value)[0]), float(np.atleast_1d(grad)[0])


def test_mass_mode_is_linear(rv_system):
    """RV data measure the orbit, so the signed linear mass is sampled --
    which is what makes the region below zero reachable."""
    system, _ = rv_system
    assert system.planet.mass_parameterization == "linear"
    assert np.min(np.atleast_1d(system.planet.mass.lower)) < -0.9


# The transition width is softness * scale = 0.01 * 0.88 = 0.0088 solMass,
# so "safely positive" starts a few hundredths of a solar mass above zero.
@pytest.mark.parametrize("m_total", [1.0, 0.5, 0.05])
def test_barrier_is_inert_for_a_positive_total_mass(barrier, m_total):
    """
    Given a physical total mass,
    When the barrier is evaluated,
    Then it costs nothing -- the fix must not perturb any real fit.
    """
    value, _ = _at(barrier, m_total)
    assert value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("m_total", [-1e-3, -0.02, -0.5, -1.0])
def test_barrier_has_a_restoring_gradient_below_zero(barrier, m_total):
    """
    Given a total mass driven below zero by a negative planet mass,
    When the barrier and its gradient are evaluated,
    Then the gradient is strictly positive: it pushes the planet mass back
    up, out of the forbidden region.

    On the clipped input the gradient is exactly 0.0 here (pt.maximum sends
    no derivative down the unselected branch), so the sampler feels nothing
    at all no matter how far negative it wanders.
    """
    value, grad = _at(barrier, m_total)

    assert np.isfinite(value) and np.isfinite(grad)
    assert grad > 0.0


@pytest.mark.parametrize("m_total", [-0.02, -0.5, -1.0])
def test_barrier_gradient_is_the_designed_steepness(barrier, m_total):
    """Well inside the forbidden region the log-sigmoid is linear, with
    slope 4.4 / (scale * softness) = 4.4 / (0.88 * 0.01) = 500 nats per
    solar mass -- the historical steepness the docstring in
    Planet.build_likelihood quotes.  (Not 0, which is what the clipped
    input gave; and not scaled by anything, which is the other way this
    could go wrong.)"""
    _, grad = _at(barrier, m_total)
    assert grad == pytest.approx(500.0, rel=1e-3)


def test_barrier_deepens_monotonically(barrier):
    """
    Given a sequence of total masses going further below zero,
    When the barrier is evaluated at each,
    Then the penalty keeps growing.  On the clipped input every one of these
    returns the same -0.693: a plateau, which is what removes the restoring
    force.
    """
    values = [_at(barrier, mt)[0] for mt in (-0.01, -0.1, -0.5, -1.0)]

    assert all(np.isfinite(values))
    assert values[0] > values[1] > values[2] > values[3]
    # Not the log(sigmoid(0)) plateau the clipped input produced.
    assert not np.allclose(values, -np.log(2.0))


def test_barrier_is_not_the_clipped_plateau(barrier):
    """The direct statement of the bug: feeding the clipped m_total made
    every point below zero identical.  Two very different total masses must
    not give the same penalty."""
    shallow, _ = _at(barrier, -0.01)
    deep, _ = _at(barrier, -1.0)
    assert deep < shallow - 100.0
