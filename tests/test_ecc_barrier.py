"""orbit.e_collision_bound: the barrier keeping the eccentricity physical.

Two defects, one fix.

1. The barrier was applied to ``orbit.ecc``, whose physics function
   ``calc_ecc`` clips the sum ``secosw^2 + sesinw^2`` at ``MAX_ECC = 0.9999``.
   A clipped input freezes the soft bound at a *constant* over the whole
   ``e > 0.9999`` region: the penalty has the right sign and a plausible
   magnitude, and a gradient of exactly zero.  That is a flat plateau with no
   restoring force -- the identical mistake documented (and already fixed) for
   ``m_total`` in ``Planet.build_likelihood``.  It is not a corner case:
   ``secosw`` and ``sesinw`` are each uniform on [-1, 1], so the clipped part
   of the sampled square is ``4 - pi*0.9999`` out of 4, i.e. 21.5% of the
   prior volume.

2. The barrier lived in ``Planet.build_likelihood``, so an orbit with no
   planet -- a stellar binary -- carried no eccentricity bound at all.

So these tests assert on the GRADIENT, and on a planet-free system.  A
value-only test on a planet-bearing system passes on the broken code, which
is how this survived.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.compile.mode import Mode
from pytensor.graph.replace import graph_replace

from exozippy.components.orbit.physics import MAX_ECC
from exozippy.components.parameter import Parameter
from exozippy.system import System

# 4.4 / (scale * softness) = 4.4 / (0.88 * 0.01): the designed steepness of
# the barrier, in nats per unit eccentricity.
STEEPNESS = 500.0

LOGP = 1.23
TC = 2456000.0


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------
def _build(config, params):
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


@pytest.fixture(scope="module")
def planet_system():
    """One star, one planet, one orbit -- the topology that had a bound."""
    return _build(
        {
            "star": [{"name": "A", "mist": False}],
            "planet": [{"name": "b"}],
            "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        },
        {
            "star.A.mass": {"initval": 1.0},
            "star.A.radius": {"initval": 1.0},
            "orbit.b.logP": {"initval": LOGP},
            "orbit.b.tc": {"initval": TC},
        },
    )


@pytest.fixture(scope="module")
def binary_system():
    """Two stars, one orbit, NO planet component at all.

    The topology that carried no eccentricity bound whatsoever, because the
    bound lived on the planet component.
    """
    return _build(
        {
            "star": [
                {"name": "A", "mist": False},
                {"name": "B", "mist": False},
            ],
            "orbit": [{"name": "AB", "primary": ["A"], "companion": ["B"]}],
        },
        {
            "star.A.mass": {"initval": 1.0},
            "star.B.mass": {"initval": 0.8},
            "orbit.AB.logP": {"initval": LOGP},
            "orbit.AB.tc": {"initval": TC},
        },
    )


@pytest.fixture(scope="module")
def mixed_system():
    """A planet orbit (index 0) and a stellar binary orbit (index 1).

    Pins that the per-orbit threshold is a vector: the planet's collision
    limit on its own orbit, the hard ceiling on the starry one.
    """
    return _build(
        {
            "star": [
                {"name": "A", "mist": False},
                {"name": "B", "mist": False},
                {"name": "C", "mist": False},
            ],
            "planet": [{"name": "b"}],
            "orbit": [
                {"name": "b", "primary": ["A"], "companion": ["b"]},
                {"name": "BC", "primary": ["C"], "companion": ["B"]},
            ],
        },
        {
            "star.A.mass": {"initval": 1.0},
            "star.B.mass": {"initval": 0.8},
            "star.C.mass": {"initval": 0.7},
            "orbit.b.logP": {"initval": LOGP},
            "orbit.BC.logP": {"initval": 4.0},
            "orbit.b.tc": {"initval": TC},
            "orbit.BC.tc": {"initval": TC},
        },
    )


# ---------------------------------------------------------------------------
# Barrier extraction
# ---------------------------------------------------------------------------
def _bounds(model):
    return [
        p for p in model.potentials if p.name.endswith("e_collision_bound")
    ]


def _barrier(system, model, mode=None):
    """Compile the eccentricity barrier and its gradient as a function of
    (secosw, sesinw[, max_ecc]).

    The potential is cut out of the *built* model, so this measures what the
    component actually put there -- including, on the broken code,
    calc_ecc's clip.  ``max_ecc`` is replaced too: it descends from ``ar``,
    hence from the star's mass and radius, which are still free random
    variables in the graph and would otherwise redraw on every call.
    """
    hits = _bounds(model)
    assert len(hits) == 1, [p.name for p in model.potentials]

    sc = pt.dvector("secosw")
    ss = pt.dvector("sesinw")
    repl = {system.orbit.secosw.value: sc, system.orbit.sesinw.value: ss}
    inputs = [sc, ss]

    planet = getattr(system, "planet", None)
    if isinstance(getattr(planet, "max_ecc", None), Parameter):
        me = pt.dvector("max_ecc")
        repl[planet.max_ecc.value] = me
        inputs.append(me)

    node = graph_replace(hits[0], repl)
    total = node.sum()
    return pytensor.function(
        inputs,
        [node, pt.grad(total, sc), pt.grad(total, ss)],
        on_unused_input="ignore",
        mode=mode,
    )


def _at(fn, e_raw, n_orbits=1, index=0, max_ecc=None, quiet=0.05):
    """Evaluate at eccentricity ``e_raw`` on orbit ``index``.

    The pair is split evenly, secosw = sesinw = sqrt(e/2); every other orbit
    sits at a quiet, well-allowed eccentricity.
    """
    sc = np.full(n_orbits, quiet)
    ss = np.full(n_orbits, quiet)
    s = np.sqrt(e_raw / 2.0)
    sc[index] = s
    ss[index] = s
    args = [sc, ss]
    if max_ecc is not None:
        args.append(np.atleast_1d(np.asarray(max_ecc, dtype=float)))
    val, g_sc, g_ss = fn(*args)
    return (
        float(np.atleast_1d(val)[index]),
        float(np.atleast_1d(g_sc)[index]),
        float(np.atleast_1d(g_ss)[index]),
    )


@pytest.fixture(scope="module")
def planet_barrier(planet_system):
    return _barrier(*planet_system)


@pytest.fixture(scope="module")
def binary_barrier(binary_system):
    return _barrier(*binary_system)


# ---------------------------------------------------------------------------
# 1. The clipped region is a real part of the prior, not a corner case
# ---------------------------------------------------------------------------
def test_the_clipped_region_is_a_fifth_of_the_prior_volume(planet_system):
    """
    Given secosw and sesinw sampled uniformly on their own bounds,
    When the fraction of that square with secosw^2 + sesinw^2 > MAX_ECC is
    computed,
    Then it is 21.5% -- so the plateau below is reachable prior volume, not
    a measure-zero edge.
    """
    system, _ = planet_system
    for name in ("secosw", "sesinw"):
        par = getattr(system.orbit, name)
        assert np.all(np.atleast_1d(par.lower) == -1.0)
        assert np.all(np.atleast_1d(par.upper) == 1.0)

    # The square has area 4; the allowed disc has radius sqrt(MAX_ECC).
    analytic = (4.0 - np.pi * MAX_ECC) / 4.0
    rng = np.random.default_rng(20260812)
    draw = rng.uniform(-1.0, 1.0, (200_000, 2))
    monte_carlo = np.mean(np.sum(draw**2, axis=1) > MAX_ECC)

    assert analytic == pytest.approx(0.2147, abs=1e-4)
    assert monte_carlo == pytest.approx(analytic, abs=5e-3)


# ---------------------------------------------------------------------------
# 2. The gradient above the clip (the bug)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("e_raw", [1.0, 1.05, 1.3, 1.8])
def test_barrier_has_a_restoring_gradient_above_the_clip(
    binary_barrier, e_raw
):
    """
    Given an eccentricity driven past MAX_ECC, where calc_ecc clips,
    When the barrier and its gradient are evaluated,
    Then both are finite and the gradient points back toward the allowed
    region: with secosw = sesinw > 0, d/d(secosw) and d/d(sesinw) are both
    strictly negative, shrinking e.

    On the clipped input the gradient is exactly 0.0 here (pt.clip sends no
    derivative down the saturated branch), so the sampler feels nothing at
    all no matter how far past e = 1 it wanders.
    """
    value, g_sc, g_ss = _at(binary_barrier, e_raw)

    assert np.isfinite(value) and np.isfinite(g_sc) and np.isfinite(g_ss)
    assert g_sc < 0.0
    assert g_ss < 0.0


@pytest.mark.parametrize("e_raw", [1.05, 1.3, 1.8])
def test_barrier_gradient_is_the_designed_steepness(binary_barrier, e_raw):
    """Well inside the forbidden region the log-sigmoid is linear in e with
    slope -4.4 / (scale * softness) = -500 nats per unit eccentricity.  The
    chain rule through e = secosw^2 + sesinw^2 gives d/d(secosw) =
    -500 * 2 * secosw.  (Not 0, which is what the clipped input gave; and
    not rescaled by anything, which is the other way this could go wrong.)
    """
    _, g_sc, g_ss = _at(binary_barrier, e_raw)
    expected = -STEEPNESS * 2.0 * np.sqrt(e_raw / 2.0)

    assert g_sc == pytest.approx(expected, rel=1e-3)
    assert g_ss == pytest.approx(expected, rel=1e-3)


def test_barrier_deepens_monotonically_past_the_clip(binary_barrier):
    """
    Given a sequence of eccentricities going further past MAX_ECC,
    When the barrier is evaluated at each,
    Then the penalty keeps growing.  On the clipped input every one of these
    returns the same constant: a plateau, which is what removes the
    restoring force.
    """
    values = [_at(binary_barrier, e)[0] for e in (1.0, 1.2, 1.5, 1.9)]

    assert all(np.isfinite(values))
    assert values[0] > values[1] > values[2] > values[3]
    assert not np.allclose(values, values[0])


def test_barrier_is_not_the_clipped_plateau(binary_barrier):
    """The direct statement of the bug: feeding the clipped eccentricity made
    every point above MAX_ECC identical.  Two very different eccentricities
    must not give the same penalty."""
    shallow, _, _ = _at(binary_barrier, 1.0)
    deep, _, _ = _at(binary_barrier, 1.9)

    assert deep < shallow - 400.0


def test_a_planet_orbit_also_gets_a_gradient_past_the_clip(planet_barrier):
    """The plateau, on the topology that DID carry a bound.

    The tests above use the planet-free system, where the threshold is the
    clean hard ceiling -- but a planet-bearing orbit had a bound all along,
    and it was flat above MAX_ECC just the same, because the clip is inside
    calc_ecc and has nothing to do with the collision limit.
    """
    e_raw = 1.3
    value, g_sc, g_ss = _at(planet_barrier, e_raw, max_ecc=0.9)
    expected = -STEEPNESS * 2.0 * np.sqrt(e_raw / 2.0)

    assert np.isfinite(value)
    assert g_sc == pytest.approx(expected, rel=1e-3)
    assert g_ss == pytest.approx(expected, rel=1e-3)

    shallow, _, _ = _at(planet_barrier, 1.0, max_ecc=0.9)
    deep, _, _ = _at(planet_barrier, 1.9, max_ecc=0.9)
    assert deep < shallow - 400.0


def test_the_plateau_is_absent_without_pytensor_rewrites(binary_system):
    """The same restoring gradient under Mode(linker="py", optimizer="None").

    A canonicalization rewrite can mask (or manufacture) a saturated branch,
    so the defect and its fix are established in the unrewritten graph too,
    not only under FAST_RUN.
    """
    fn = _barrier(*binary_system, mode=Mode(linker="py", optimizer="None"))
    value, g_sc, g_ss = _at(fn, 1.3)

    assert np.isfinite(value)
    assert g_sc == pytest.approx(-STEEPNESS * 2.0 * np.sqrt(0.65), rel=1e-3)
    assert g_ss == pytest.approx(-STEEPNESS * 2.0 * np.sqrt(0.65), rel=1e-3)


def test_barrier_has_a_finite_nonzero_gradient_in_jax(binary_system):
    """
    Given the barrier compiled through PyTensor's JAX linker,
    When it is evaluated past the clip,
    Then value and gradient are finite and non-zero.

    The house rule is that soft bounds must never be built from a `where` or
    `switch` over branch values that can be invalid, because JAX
    differentiates through the unselected branch: log(sigmoid(.)) is clipped
    at 700 inside potentials.py for exactly that reason.  A NaN here means
    every numpyro chain freezes at its start.  (The JAX path was also
    verified by actually sampling; see the module this fix landed with.)
    """
    fn = _barrier(*binary_system, mode="JAX")
    value, g_sc, g_ss = _at(fn, 1.3)

    assert np.isfinite(value) and value < 0.0
    assert np.isfinite(g_sc) and g_sc < 0.0
    assert np.isfinite(g_ss) and g_ss < 0.0


# ---------------------------------------------------------------------------
# 3. Ownership: one bound per orbit, planet or no planet
# ---------------------------------------------------------------------------
def test_a_stellar_binary_orbit_carries_the_bound(binary_system):
    """
    Given a two-star system with no planet component,
    When the model is built,
    Then the orbit still carries exactly one eccentricity bound.  It used to
    carry none: the bound was added by Planet.build_likelihood, which never
    ran.
    """
    _, model = binary_system
    hits = _bounds(model)

    assert [p.name for p in hits] == ["orbit.e_collision_bound"]


def test_a_planet_bearing_orbit_carries_exactly_one_bound(planet_system):
    """
    Given a system with a planet,
    When the model is built,
    Then there is exactly ONE eccentricity bound, on the orbit component --
    moving the bound must not leave a second copy behind on the planet.
    """
    _, model = planet_system
    hits = _bounds(model)

    assert [p.name for p in hits] == ["orbit.e_collision_bound"]
    assert not [p for p in model.potentials if p.name.startswith("planet.e_")]


def test_planet_free_orbits_use_the_hard_ceiling(binary_barrier):
    """
    Given an orbit with no planet, hence no collision limit,
    When the barrier is evaluated on either side of MAX_ECC,
    Then it is inert below and biting above: the threshold is the hard
    ceiling calc_ecc clips at.
    """
    below, _, _ = _at(binary_barrier, MAX_ECC - 0.05)
    above, g_sc, _ = _at(binary_barrier, MAX_ECC + 0.05)

    assert below == pytest.approx(0.0, abs=1e-6)
    assert above == pytest.approx(-STEEPNESS * 0.05, rel=1e-2)
    assert g_sc < 0.0


# ---------------------------------------------------------------------------
# 4. The planet collision limit is still the threshold where there is one
# ---------------------------------------------------------------------------
def test_the_threshold_is_the_planet_collision_limit(planet_barrier):
    """
    Given a planet whose orbit has a collision limit well below MAX_ECC,
    When the barrier is evaluated on either side of that limit,
    Then it turns on at the limit, not at the hard ceiling -- the historical
    behaviour this fix must preserve.
    """
    limit = 0.5
    below, _, _ = _at(planet_barrier, limit - 0.05, max_ecc=limit)
    above, g_sc, _ = _at(planet_barrier, limit + 0.05, max_ecc=limit)

    assert below == pytest.approx(0.0, abs=1e-6)
    assert above == pytest.approx(-STEEPNESS * 0.05, rel=1e-2)
    assert g_sc < 0.0


@pytest.mark.parametrize("e_raw", [0.0, 0.01, 0.3, 0.8])
def test_barrier_is_inert_for_an_allowed_eccentricity(planet_barrier, e_raw):
    """
    Given a physically allowed eccentricity,
    When the barrier is evaluated,
    Then it costs nothing -- the fix must not perturb any real fit.  Below
    MAX_ECC the clipped and unclipped inputs are identical, so no pinned
    logp can move.
    """
    value, _, _ = _at(planet_barrier, e_raw, max_ecc=0.9)

    assert value == pytest.approx(0.0, abs=1e-6)


def test_threshold_is_per_orbit(mixed_system):
    """
    Given one planet orbit and one stellar binary orbit in the same system,
    When the barrier is evaluated at an eccentricity between the planet's
    collision limit and the hard ceiling,
    Then only the planet's orbit is penalized: the threshold is a vector,
    the planet's limit on its own orbit and MAX_ECC on the starry one.
    """
    system, model = mixed_system
    assert list(np.atleast_1d(system.planet.orbit_map)) == [0]

    fn = _barrier(system, model)
    e_raw = 0.7
    planet_orbit = _at(fn, e_raw, n_orbits=2, index=0, max_ecc=[0.5])
    star_orbit = _at(fn, e_raw, n_orbits=2, index=1, max_ecc=[0.5])

    assert planet_orbit[0] == pytest.approx(-STEEPNESS * 0.2, rel=1e-2)
    assert planet_orbit[1] < 0.0
    assert star_orbit[0] == pytest.approx(0.0, abs=1e-6)


def test_forward_model_eccentricity_is_still_clipped(planet_system):
    """
    Given the barrier now reads the unclipped sum,
    When calc_ecc is evaluated past the ceiling,
    Then it still clips: a Kepler solve, calc_K's sqrt(1 - e^2) and calc_tp's
    sqrt(1 - e) are all meaningless at e >= 1, so the forward model must
    never see the raw value.  Only the *bound* was moved off the clip.
    """
    from exozippy.components.orbit.physics import calc_ecc, ecc_from_sqrte

    sc = pt.as_tensor_variable(np.array([0.05, 0.9, 0.99]))
    ss = pt.as_tensor_variable(np.array([0.05, 0.9, 0.99]))

    clipped = calc_ecc(sc, ss).eval()
    raw = ecc_from_sqrte(sc, ss).eval()

    np.testing.assert_allclose(raw, [0.005, 1.62, 1.9602])
    np.testing.assert_allclose(clipped, [0.005, MAX_ECC, MAX_ECC])
    assert np.all(clipped <= MAX_ECC)
