"""The transit-chord parameterization (Eastman 2024, arXiv:2309.14410).

The geometric half of the pair whose eccentricity half is tests/test_vcve.py,
and it exists for the same reason: a transit measures a DURATION, which the
chord and V_c/V_e carry directly, while cos i reaches it only through
``b = kappa cos i`` with kappa itself fitted.

Three things here are not the vcve file over again, and they are what these
tests are mostly about:

* the inversion is a square root, not a quadratic -- one branch, no mixture --
  but the sign of cos i is NOT recoverable from a chord, so it travels as a
  context node carrying the orbit's own i180 convention;
* the chord is defined by a PLANET (its radius ratio) while cos i belongs to
  the orbit, so an orbit with no single planet has no chord at all, and one in
  a system with no transit data has one that means nothing;
* the seed bridge is a small solver rather than a relation -- not because a/R*
  is unavailable (it resolves now) but because that relation costs 7.6 s of
  sympy per System.

Cheap tests build no model.  The two that do share one built system per
parameterization, because compiling a transit likelihood is what costs.
"""

import numpy as np
import pytensor.tensor as pt
import pytest

from exozippy.components.orbit import physics
from exozippy.system import System


def _f(node):
    return np.atleast_1d(np.asarray(node.eval(), dtype=float))


def chord_forward(cosi, p, ar, ecc, esinw):
    """The chord, in numpy, as the reference the port must reproduce."""
    b = ar * abs(cosi) * (1.0 - ecc**2) / (1.0 + esinw)
    return np.sqrt((1.0 + p) ** 2 - b**2)


# ---------------------------------------------------------------------------
# 1. The algebra
# ---------------------------------------------------------------------------

_GEOM = [
    # (cosi, p, ar, ecc, esinw)
    (0.05, 0.1, 10.0, 0.0, 0.0),
    (0.089, 0.0959, 8.0, 0.0, 0.0),
    (0.02, 0.05, 20.0, 0.3, 0.15),
    (0.11, 0.15, 6.0, 0.4, -0.2),
]


@pytest.mark.parametrize("cosi,p,ar,ecc,esinw", _GEOM)
def test_the_inversion_round_trips_the_forward_relation(
    cosi, p, ar, ecc, esinw
):
    """
    Given a transiting geometry,
    When its chord is computed and then inverted,
    Then cos i comes back.

    The pair of functions is the whole parameterization, so this is the test
    that a factor or a squared term is not quietly wrong -- the same role
    test_vcve.py's round trip plays, and for the same reason: an inversion
    that is wrong by an algebraic slip still returns plausible numbers.
    """
    chord = chord_forward(cosi, p, ar, ecc, esinw)
    got = _f(
        physics.calc_cosi_from_chord(
            pt.as_tensor_variable(chord), p, ar, ecc, esinw, 1.0
        )
    )[0]

    assert got == pytest.approx(cosi, rel=1e-9)


@pytest.mark.parametrize("cosi,p,ar,ecc,esinw", _GEOM)
def test_the_forward_direction_matches_the_reference(cosi, p, ar, ecc, esinw):
    """
    Given the same geometry,
    When the chord is computed from cos i,
    Then it is sqrt((1 + p)^2 - b^2) with b from Winn 2010 eq 7.

    Independent of the round trip above: that one would pass if BOTH
    directions shared a wrong kappa, and this one checks the definition.
    """
    got = _f(
        physics.calc_chord_from_cosi(
            pt.as_tensor_variable(cosi), p, ar, ecc, esinw
        )
    )[0]

    assert got == pytest.approx(
        chord_forward(cosi, p, ar, ecc, esinw), rel=1e-9
    )


def test_the_chord_is_blind_to_the_sign_of_cos_i():
    """
    Given two orbits differing only in the sign of cos i,
    When their chords are computed,
    Then the chords are identical -- and the sign comes back only from the
      sign the caller supplies.

    This is why `chord_sign` is a context node rather than something the
    inversion recovers: a transit at i and at 180 - i is the same transit, so
    the parameterization genuinely cannot see the difference.  Sampling the
    sign would be the piecewise-constant coordinate the V_c/V_e half went out
    of its way to avoid.
    """
    kw = dict(p=0.1, ar=10.0, ecc=0.0, esinw=0.0)
    up = _f(physics.calc_chord_from_cosi(pt.as_tensor_variable(0.05), **kw))[0]
    down = _f(
        physics.calc_chord_from_cosi(pt.as_tensor_variable(-0.05), **kw)
    )[0]

    assert up == pytest.approx(down, rel=1e-12)

    chord = pt.as_tensor_variable(up)
    assert _f(physics.calc_cosi_from_chord(chord, 0.1, 10.0, 0.0, 0.0, 1.0))[
        0
    ] == pytest.approx(0.05, rel=1e-9)
    assert _f(physics.calc_cosi_from_chord(chord, 0.1, 10.0, 0.0, 0.0, -1.0))[
        0
    ] == pytest.approx(-0.05, rel=1e-9)


def test_the_jacobian_matches_a_finite_difference():
    """
    Given the analytic log|d(chord)/d(cos i)|,
    When it is compared to a numerical derivative of the forward relation,
    Then they agree.

    Magnitude only.  The DIRECTION it is applied in is a separate statement
    and a separate test below, because a finite-difference check passes under
    either sign -- the lesson the V_c/V_e half paid for.
    """
    for cosi, p, ar, ecc, esinw in _GEOM:
        h = 1e-8
        num = (
            chord_forward(cosi + h, p, ar, ecc, esinw)
            - chord_forward(cosi - h, p, ar, ecc, esinw)
        ) / (2 * h)
        chord = chord_forward(cosi, p, ar, ecc, esinw)
        ana = np.exp(
            _f(
                physics.chord_log_jacobian(
                    pt.as_tensor_variable(chord), p, ar, ecc, esinw
                )
            )[0]
        )
        assert ana == pytest.approx(abs(num), rel=1e-5)


def test_subtracting_the_jacobian_flattens_the_implied_prior_on_cos_i():
    """
    Given a chord drawn uniformly, as its logit-transformed bounds imply,
    When the derived cos i values are weighted by exp(-log|d(chord)/d(cos i)|),
    Then their density is flat in cos i -- and it is NOT flat unweighted, nor
      with the opposite sign.

    The sign is the term, and this is the test that pins it.  An isotropic
    prior is uniform in cos i, which is exactly what sampling cos i directly
    gives; the chord must not change that silently, and the correction that
    keeps it is the RECIPROCAL of the derivative.  Both wrong answers are
    asserted against so that "flat" cannot pass by accident.
    """
    p, ar, ecc, esinw = 0.1, 10.0, 0.0, 0.0
    rng = np.random.default_rng(0)
    chord = rng.uniform(0.0, 1.0 + p, 400_000)
    cosi = _f(
        physics.calc_cosi_from_chord(
            pt.as_tensor_variable(chord), p, ar, ecc, esinw, 1.0
        )
    )
    jac = np.exp(
        _f(
            physics.chord_log_jacobian(
                pt.as_tensor_variable(chord), p, ar, ecc, esinw
            )
        )
    )
    keep = (cosi > 0.0) & (cosi < (1.0 + p) / ar)
    cosi, jac = cosi[keep], jac[keep]

    bins = np.linspace(0.0, (1.0 + p) / ar, 10)

    def spread(weights):
        h, _ = np.histogram(cosi, bins=bins, weights=weights)
        h = h / h.sum()
        return h.max() / h.min()

    assert spread(1.0 / jac) < 1.05  # flat to 5%
    assert spread(np.ones_like(cosi)) > 1.5  # unweighted: piled up at grazing
    assert spread(jac) > 3.0  # wrong sign: worse than unweighted


# ---------------------------------------------------------------------------
# 2. The shields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chord,p",
    [
        (2.5, 0.1),  # longer than the star: no transiting geometry at all
        (1.1, 0.1),  # exactly the grazing limit
        (0.0, 0.1),  # a chord of zero
        (1.0, 0.0),  # a point planet
    ],
)
def test_the_inversion_and_the_jacobian_stay_finite(chord, p):
    """
    Given chords with no geometry behind them, at the grazing limit, and at
      zero,
    When cos i and the Jacobian are evaluated,
    Then every value is finite.

    "It must be impossible to draw a NaN likelihood": the radicand is floored
    INSIDE the sqrt so the hard shield cannot itself produce one, and every
    log argument is floored.
    """
    ct = pt.as_tensor_variable(chord)
    cosi = _f(physics.calc_cosi_from_chord(ct, p, 10.0, 0.0, 0.0, 1.0))[0]
    jac = _f(physics.chord_log_jacobian(ct, p, 10.0, 0.0, 0.0))[0]

    assert np.isfinite(cosi) and cosi >= 0.0
    assert np.isfinite(jac)


def test_the_radicand_is_reported_unfloored_for_the_soft_bound():
    """
    Given a chord past the grazing limit,
    When the radicand helper is asked,
    Then it returns the NEGATIVE value, not the floored one.

    Same split as `vcve_discriminant`: the floored quantity is flat across the
    whole forbidden region, so the soft bound needs the honest number to have
    a gradient pointing back to a geometry that transits.
    """
    value = _f(physics.chord_radicand(pt.as_tensor_variable(2.0), 0.1))[0]

    assert value < 0.0
    assert value == pytest.approx(1.1**2 - 4.0)


# ---------------------------------------------------------------------------
# 3. End to end
# ---------------------------------------------------------------------------


def _transit_config(lc, **orbit_keys):
    orbit = {"name": "b", "primary": ["A"], "companion": ["b"]}
    orbit.update(orbit_keys)
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [orbit],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "inst0", "file": lc, "band": "TESS"}],
    }


_PARAMS = {
    "star.0.radius": {"initval": 1.61, "sigma": 0.05},
    "star.0.mass": {"initval": 1.204, "sigma": 0.05},
    "star.0.teff": {"initval": 6207, "sigma": 100},
    "star.0.feh": {"initval": -0.116, "sigma": 0.08},
    "orbit.0.period": {"initval": 2.99},
    "orbit.0.tc": {"initval": 2459634.3},
    "orbit.0.cosi": {"initval": 0.05},
    "planet.0.radius": {"initval": 1.7},
}


@pytest.fixture(scope="module")
def transit_lc(tmp_path_factory):
    path = tmp_path_factory.mktemp("chord") / "lc.dat"
    rng = np.random.default_rng(11)
    t = np.linspace(2459634.1, 2459634.5, 150)
    np.savetxt(
        path,
        np.column_stack(
            [t, 1.0 + rng.normal(0.0, 1e-3, t.size), np.full(t.size, 1e-3)]
        ),
    )
    return str(path)


@pytest.fixture(scope="module")
def chord_fit(transit_lc):
    """One built `fitchord` transit model, shared by the tests that read it."""
    system = System(
        _transit_config(transit_lc, fitchord=True), user_params=dict(_PARAMS)
    )
    system.prepare()
    model = system.build_model()
    return system, model, system.get_raw_start(model)


def test_a_chord_orbit_samples_the_chord_and_reports_cos_i(chord_fit):
    """
    Given a transit fit with fitchord: true,
    When the model is built,
    Then the chord is sampled, cos i is DERIVED from it, both potentials are
      present, and the logp is finite at the start.

    cos i is derived rather than reported, unlike the sqrt(e) pair on a
    V_c/V_e orbit: the transit model consumes it (through planet.b), which is
    exactly what role 3 forbids.
    """
    system, model, start = chord_fit

    raw = {v.name for v in model.free_RVs}
    assert "orbit.chord_raw" in raw
    assert "orbit.cosi_raw" not in raw
    assert system.orbit.chord.is_sampled.tolist() == [True]
    assert system.orbit.cosi.is_derived.tolist() == [True]
    assert system.orbit.cosi.is_reported.tolist() == [False]

    names = {p.name for p in model.potentials}
    assert "orbit.chord_jacobian" in names
    assert "orbit.chord_geometry" in names

    assert np.isfinite(model.compile_logp()(start))


def test_the_geometry_identity_holds_at_the_start(chord_fit):
    """
    Given a built chord fit,
    When cos i, the chord, b and the radius ratio are evaluated together,
    Then chord^2 + b^2 = (1 + p)^2.

    The one identity the whole parameterization rests on, checked on the real
    graph rather than on the physics functions in isolation -- which is what
    catches a context node wired to the wrong planet.
    """
    system, model, start = chord_fit

    fn = model.compile_fn(
        model.replace_rvs_by_values(
            [
                system.orbit.chord.value,
                system.planet.b.value,
                system.planet.p.value,
            ]
        ),
        inputs=model.value_vars,
        point_fn=True,
        on_unused_input="ignore",
    )
    chord, b, p_ratio = (float(np.atleast_1d(v)[0]) for v in fn(start))

    assert chord**2 + b**2 == pytest.approx((1.0 + p_ratio) ** 2, rel=1e-9)


def test_a_seeded_cos_i_reaches_the_chord(transit_lc):
    """
    Given a params file that seeds cos i on a chord orbit,
    When the relaxation engine resolves the start,
    Then the chord starts at the value that geometry implies.

    This is what lets one params file drive either parameterization.  The
    bridge is a solver rather than the relation it could now be (a/R* resolves
    since planet/symbolic_physics.py gained the Kepler and a/R* relations), for
    a measured reason: as a relation it costs 7.6 s of sympy per System, being
    quadratic in the chord and transcendental in omega.  What the test pins is
    the RESULT, so it holds either way.
    """
    system = System(
        _transit_config(transit_lc, fitchord=True), user_params=dict(_PARAMS)
    )
    system.prepare()

    chord = system.config_manager.resolve("orbit", "chord", shape=(1,))
    cosi = system.config_manager.resolve("orbit", "cosi", shape=(1,))

    assert float(np.atleast_1d(cosi["initval"])[0]) == pytest.approx(0.05)
    # Not the defaults.yaml backstop, and inside the transiting range.
    seeded = float(np.atleast_1d(chord["initval"])[0])
    assert seeded != pytest.approx(1.0)
    assert 0.0 < seeded < 1.2


def test_an_orbit_with_no_single_planet_has_no_chord(transit_lc):
    """
    Given an orbit whose companion group holds two planets,
    When `fitchord: true` is asked for,
    Then it raises, naming the ambiguity.

    Two planets sharing an orbit have two different chords, so "the" chord is
    a question with no correct answer.  Raising beats picking one: the wrong
    radius ratio would give a wrong inclination that nothing else contradicts.
    """
    cfg = _transit_config(transit_lc, fitchord=True)
    cfg["planet"] = [{"name": "b"}, {"name": "c"}]
    cfg["orbit"][0]["companion"] = ["b", "c"]

    with pytest.raises(ValueError, match="exactly one planet"):
        System(cfg, user_params=dict(_PARAMS)).prepare()


def test_the_default_follows_the_data_topology(transit_lc):
    """
    Given the same transit system with and without radial velocities,
    When the orbit's parameterization is resolved,
    Then transits alone turn BOTH halves on, and adding RVs turns both off.

    The paper's method is the pair, so they flip together; and the condition
    is that a transit is the only thing measuring the orbit, since an RV
    amplitude constrains the conventional coordinates directly.
    """
    cfg = _transit_config(transit_lc)
    system = System(cfg, user_params=dict(_PARAMS))
    system.prepare()
    assert system.orbit.fitvcve == [True]
    assert system.orbit.fitchord == [True]
    assert system.orbit.inc_modes == ["chord"]
    assert system.orbit.ecc_modes == ["vcve"]

    with_rv = _transit_config(transit_lc)
    with_rv["rvinstrument"] = [{"name": "rv0", "file": _rv_file(transit_lc)}]
    system = System(with_rv, user_params=dict(_PARAMS))
    system.prepare()
    assert system.orbit.fitvcve == [False]
    assert system.orbit.fitchord == [False]
    assert system.orbit.inc_modes == ["cosi"]


def _rv_file(lc_path):
    """A small RV file next to the light curve."""
    import os

    path = os.path.join(os.path.dirname(lc_path), "rv.dat")
    if not os.path.exists(path):
        rng = np.random.default_rng(5)
        t = np.linspace(2459600.0, 2459660.0, 30)
        np.savetxt(
            path,
            np.column_stack(
                [t, rng.normal(0.0, 10.0, t.size), np.full(t.size, 10.0)]
            ),
        )
    return path


def test_an_explicit_false_beats_the_transit_only_default(transit_lc):
    """
    Given a transit-only orbit that asks for the conventional coordinates,
    When the parameterization is resolved,
    Then both halves are off -- `fitvcve: false` carries `fitchord` with it.

    The user's coupling rule: turning the eccentricity half off turns the
    geometry half off too, unless the geometry half was asked for by name.
    """
    system = System(
        _transit_config(transit_lc, fitvcve=False), user_params=dict(_PARAMS)
    )
    system.prepare()
    assert system.orbit.fitchord == [False]
    assert system.orbit.inc_modes == ["cosi"]

    system = System(
        _transit_config(transit_lc, fitvcve=False, fitchord=True),
        user_params=dict(_PARAMS),
    )
    system.prepare()
    assert system.orbit.ecc_modes == ["hk"]
    assert system.orbit.inc_modes == ["chord"]


# ---------------------------------------------------------------------------
# 4. What the chord needed from the engine, and what fixing it fixed
# ---------------------------------------------------------------------------
#
# Every assertion here reads a built Parameter, never `ConfigManager.resolve`.
# That is not incidental: resolve() hands back the engine's solved values in
# INTERNAL units and a defaults.yaml value in the USER unit, so a test that
# mixes two parameters from it silently compares solar masses with Jupiter
# masses (this is how the first draft of these tests failed).  A Parameter has
# been through `__post_init__`, so its fields are internal throughout.


@pytest.fixture(scope="module")
def cosi_fit(transit_lc):
    """A built conventional (cos i, sqrt(e) pair) transit fit."""
    system = System(
        _transit_config(transit_lc, fitvcve=False), user_params=dict(_PARAMS)
    )
    system.prepare()
    model = system.build_model()
    return system, model


def _first(param):
    return float(np.atleast_1d(param.initval)[0])


def test_the_engine_resolves_the_scaled_semi_major_axis(cosi_fit):
    """
    Given a system with a mass, a radius and a period,
    When the relaxation engine runs,
    Then planet.a, planet.ar and planet.m_total all resolve, and they satisfy
      Kepler's third law and the definition of a/R*.

    Nothing resolved a/R* before the chord work: `planet.ar` comes from
    `planet.a`, which comes from Kepler, and the engine had a relation for
    neither.  The gap had already been papered over once
    (Planet._initial_semimajor_axes recomputed the start by hand), so it was
    closed instead -- three relations saying what planet/physics.py already
    computes.  This pins that they resolve AND that they are right.
    """
    from exozippy.constants import KEPLER_CONST

    system, _model = cosi_fit
    planet, orbit, star = system.planet, system.orbit, system.star

    m_total = _first(planet.m_total)
    a = _first(planet.a)
    ar = _first(planet.ar)

    assert np.isfinite([m_total, a, ar]).all()
    assert m_total == pytest.approx(
        _first(star.mass) + _first(planet.mass), rel=1e-9
    )
    expected_a = (
        KEPLER_CONST * m_total ** (1 / 3) * _first(orbit.period) ** (2 / 3)
    )
    assert a == pytest.approx(expected_a, rel=1e-9)
    assert ar == pytest.approx(a / _first(star.radius), rel=1e-9)


def test_a_circular_orbit_resolves_omega_to_the_runtime_convention(transit_lc):
    """
    Given an orbit pinned circular (the sqrt(e) pair at exactly zero),
    When the engine runs,
    Then omega resolves to pi/2 -- what physics.calc_omega returns there -- and
      esinw and ecosw resolve with it.

    `atan2(0, 0)` is undefined, so the relation could not say, and the engine
    left omega unresolved FOREVER: every quantity downstream of it (esinw,
    ecosw, V_c/V_e's reported start, the transit chord) was stuck at defaults
    while the model used pi/2.  The engine and the model disagreed about omega
    for every circular fit; this is that disagreement, pinned closed.
    """
    params = dict(_PARAMS)
    params["orbit.0.secosw"] = {"initval": 0.0, "sigma": 0}
    params["orbit.0.sesinw"] = {"initval": 0.0, "sigma": 0}

    system = System(
        _transit_config(transit_lc, fitvcve=False), user_params=params
    )
    system.prepare()
    system.build_model()

    assert _first(system.orbit.omega) == pytest.approx(np.pi / 2)
    assert _first(system.orbit.esinw) == pytest.approx(0.0)
    assert _first(system.orbit.ecosw) == pytest.approx(0.0)


def test_an_eccentric_orbit_keeps_the_relations_answer(transit_lc):
    """
    Given an orbit that is NOT circular,
    When the engine runs,
    Then omega comes from atan2, not from the circular convention.

    The convention solver must cover the exactly-zero case and nothing else:
    anywhere else the relation is well posed, and a solver competing with it
    would be a second implementation of the same physics.
    """
    params = dict(_PARAMS)
    params["orbit.0.secosw"] = {"initval": 0.3}
    params["orbit.0.sesinw"] = {"initval": 0.2}

    system = System(
        _transit_config(transit_lc, fitvcve=False), user_params=params
    )
    system.prepare()
    system.build_model()

    assert _first(system.orbit.omega) == pytest.approx(
        np.arctan2(0.2, 0.3), rel=1e-9
    )


def test_the_crossing_barrier_reads_the_engines_semi_major_axis(transit_lc):
    """
    Given a two-planet system,
    When the crossing barrier asks for its per-planet starting semi-major axes,
    Then they are the engine's resolved planet.a, ordered inner to outer.

    `Planet._initial_semimajor_axes` used to recompute Kepler's third law by
    hand -- a second copy of `calc_arsun` living in planet.py -- because the
    engine did not resolve `planet.a`.  Now that it does, the helper is a read,
    and this pins the two together so the copy cannot come back.
    """
    cfg = _transit_config(transit_lc, fitvcve=False)
    cfg["planet"] = [{"name": "b"}, {"name": "c", "orbit_ndx": 1}]
    cfg["orbit"] = [
        {"name": "b", "primary": ["A"], "companion": ["b"], "fitvcve": False},
        {"name": "c", "primary": ["A"], "companion": ["c"], "fitvcve": False},
    ]
    params = dict(_PARAMS)
    params["orbit.1.period"] = {"initval": 9.0}
    params["orbit.1.tc"] = {"initval": 2459634.4}
    params["planet.1.radius"] = {"initval": 1.1}

    system = System(cfg, user_params=params)
    system.prepare()
    system.build_model()

    a_init = system.planet._initial_semimajor_axes()
    resolved = np.atleast_1d(system.planet.a.initval).astype(float)

    assert np.all(np.isfinite(a_init))
    assert a_init == pytest.approx(resolved, rel=1e-12)
    # ...and the longer period really is the outer planet, which is the only
    # thing the barrier uses the ordering for.
    assert a_init[1] > a_init[0]
