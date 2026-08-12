"""Chen & Kipping 2017 planet mass-radius relation (components/planet).

The relation is ported from EXOFASTv2's massradius_chen.pro and applied as
a per-planet Gaussian potential by the planet component.  Default: on when
exactly one side is data-constrained (transits measure the radius, RVs and
astrometry measure the mass); a per-planet `chen: true/false` key overrides.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.planet import physics
from exozippy.components.planet.planet import Planet
from exozippy.constants import MSUN_TO_MEARTH, RSUN_TO_REARTH


def _call(fn, mpearth):
    """Evaluate a chen physics function at a scalar mass in Mearth.

    Build the input as pt.dscalar, NOT pt.as_tensor_variable(<python
    float>): pytensor autocasts a bare float to the smallest dtype that
    represents it (float32), and the power ops then compute in float32,
    silently losing ~1e-7.  The model always feeds float64.
    """
    mp = pt.dscalar("mpearth")
    return float(fn(mp).eval({mp: float(mpearth)}))


# Double-precision reference values computed from the massradius_chen.pro
# formulas (branch thresholds 2.04 / 131.58079 / 26644.8321 Mearth;
# exponents 0.279 / 0.589 / -0.044 / 0.881; fractional scatters 0.0403 /
# 0.1460 / 0.0737 / 0.0443).
_IDL_REFERENCE = [
    # (mpearth, rpearth, rperr)
    (0.5, 0.8241620849532405, 0.03321373202361559),
    (1.0, 1.0, 0.0403),
    (2.04, 1.2200758102842784, 0.04916905515445642),
    (10.0, 3.1118212923756734, 0.4543259086868483),
    (131.58079, 14.197733646754438, 2.072869112426148),
    (317.828, 13.657368863291406, 1.0065480852245767),
    (1000.0, 12.985644042161486, 0.9570419659073015),
    (26644.8321, 11.239229062719382, 0.8283311819224184),
    (50000.0, 19.568793662362484, 0.866897559242658),
]


# ---------------------------------------------------------------------------
# 1. The relation itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mpearth,rpearth,rperr", _IDL_REFERENCE)
def test_relation_matches_exofastv2(mpearth, rpearth, rperr):
    """
    Given a mass in Earth masses,
    When the radius and scatter are predicted,
    Then they match massradius_chen.pro to double precision.
    """
    assert _call(physics.calc_chen_radius, mpearth) == pytest.approx(
        rpearth, rel=1e-12
    )
    assert _call(physics.calc_chen_radius_sigma, mpearth) == pytest.approx(
        rperr, rel=1e-12
    )


@pytest.mark.parametrize("brk", physics.CHEN_MASS_BREAKS)
def test_relation_is_continuous_at_the_breaks(brk):
    """
    Given a break mass between two power-law segments,
    When the radius is evaluated just below and just above it,
    Then the two segments meet (the chained normalizations enforce this).
    """
    below = _call(physics.calc_chen_radius, brk * (1.0 - 1e-12))
    above = _call(physics.calc_chen_radius, brk * (1.0 + 1e-12))
    assert above == pytest.approx(below, rel=1e-9)


@pytest.mark.parametrize(
    "mpearth", [1e-10, 0.5, 2.0399, 2.0401, 10.0, 131.6, 1000.0, 5e4]
)
def test_relation_is_differentiable(mpearth):
    """
    Given any positive mass (including the EXOFASTv2 clip floor 1e-10),
    When the gradients of the prediction and scatter are taken,
    Then they are finite (needed for NUTS/numpyro).
    """
    mp = pt.dscalar("mpearth")
    for fn in (physics.calc_chen_radius, physics.calc_chen_radius_sigma):
        g = pt.grad(fn(mp), mp)
        assert np.isfinite(float(g.eval({mp: float(mpearth)})))


def test_published_constants():
    """Pin the constants against massradius_chen.pro."""
    assert physics.CHEN_MASS_BREAKS == (2.04, 131.58079, 26644.8321)
    assert physics.CHEN_EXPONENTS == (0.279, 0.589, -0.044, 0.881)
    assert physics.CHEN_RP_FRAC == (0.0403, 0.1460, 0.0737, 0.0443)


def test_penalty_is_finite_even_for_negative_mass():
    """
    Given a negative planet mass (the mass lower bound is negative so RV
    fits can assess detection significance),
    When the clipped penalty and its gradient are evaluated (C backend and,
    when jax is installed, the JAX path -- unselected switch/maximum
    branches NaN there first),
    Then both are finite.
    """
    mp = pt.dscalar("mp")  # solMass
    rp = pt.dscalar("rp")  # solRad
    mpearth = pt.maximum(mp * MSUN_TO_MEARTH, 1e-10)
    pred = physics.calc_chen_radius(mpearth)
    sigma = physics.calc_chen_radius_sigma(mpearth)
    logp = -0.5 * pt.sqr((rp * RSUN_TO_REARTH - pred) / sigma) - pt.log(sigma)
    grads = pt.grad(logp, [mp, rp])

    point = {"mp": -0.001, "rp": 0.1}
    f = pytensor.function([mp, rp], [logp] + grads)
    assert all(np.isfinite(v) for v in f(**point))

    jax = pytest.importorskip("jax")  # noqa: F841
    f_jax = pytensor.function([mp, rp], [logp] + grads, mode="JAX")
    assert all(np.isfinite(np.asarray(v)) for v in f_jax(**point))


# ---------------------------------------------------------------------------
# 2. The per-planet default decision
# ---------------------------------------------------------------------------
class _FakeOrbit:
    def __init__(self, memberships, bodies):
        self._memberships = memberships  # {star_idx: [(orbit_idx, role)]}
        self._bodies = bodies  # {orbit_idx: [(comp, idx), ...]}

    def star_membership(self, star_idx):
        return self._memberships.get(int(star_idx), [])

    def bodies(self, i):
        return self._bodies.get(int(i), [])


class _FakeRV:
    def __init__(self, star_ndx):
        self.star_ndx = star_ndx


class _FakeAstrometry:
    def __init__(self, modes, rel_orbit, config):
        self.modes = modes
        self.rel_orbit = rel_orbit
        self.config = config


class _FakeSystem:
    def __init__(self, **comps):
        self.active_components = comps


def _resolve(planet_cfg, **comps):
    comp = Planet(planet_cfg, config_manager=None)
    comp._resolve_chen(_FakeSystem(**comps))
    return comp.chen


_ONE_PLANET_ORBIT = _FakeOrbit(
    memberships={0: [(0, "primary")]},
    bodies={0: [("star", 0), ("planet", 0)]},
)


def test_transit_only_defaults_on():
    """Transit data with no mass constraint -> chen constrains the mass."""
    assert _resolve(
        [{"name": "b"}], transit=object(), orbit=_ONE_PLANET_ORBIT
    ) == [True]


def test_rv_only_defaults_on():
    """RV data with no radius constraint -> chen constrains the radius."""
    assert _resolve(
        [{"name": "b"}],
        rvinstrument=_FakeRV([0]),
        orbit=_ONE_PLANET_ORBIT,
    ) == [True]


def test_transit_plus_rv_defaults_off():
    """Both sides measured by data -> the relation would double-count."""
    assert _resolve(
        [{"name": "b"}],
        transit=object(),
        rvinstrument=_FakeRV([0]),
        orbit=_ONE_PLANET_ORBIT,
    ) == [False]


def test_no_data_defaults_off():
    """Neither side measured -> nothing for the relation to propagate."""
    assert _resolve([{"name": "b"}], orbit=_ONE_PLANET_ORBIT) == [False]


def test_rv_only_applies_per_planet_via_orbit_membership():
    """
    Given two planets on different orbits and RVs of star 0 only,
    When the default is resolved,
    Then only the planet in an observed orbit gets the relation.
    """
    orbit = _FakeOrbit(
        memberships={0: [(0, "primary")], 1: [(1, "primary")]},
        bodies={
            0: [("star", 0), ("planet", 0)],
            1: [("star", 1), ("planet", 1)],
        },
    )
    chen = _resolve(
        [{"name": "b"}, {"name": "c", "orbit_ndx": 1, "star_ndx": 1}],
        rvinstrument=_FakeRV([0]),
        orbit=orbit,
    )
    assert chen == [True, False]


def test_astrometry_rel_mode_counts_as_a_mass_constraint():
    ast = _FakeAstrometry(modes=["rel"], rel_orbit=[0], config=[{}])
    assert _resolve(
        [{"name": "b"}],
        astrometryinstrument=ast,
        orbit=_ONE_PLANET_ORBIT,
    ) == [True]


def test_astrometry_gaia_mode_uses_primary_membership():
    """
    Given a gaia-mode target star that is the companion (not primary) of
    the planet's orbit,
    When the default is resolved,
    Then the orbit does not count as mass-constrained (the photocenter
    wobble model sums only orbits whose primary group holds the target).
    """
    orbit = _FakeOrbit(
        memberships={0: [(0, "companion")]},
        bodies={0: [("star", 0), ("planet", 0)]},
    )
    ast = _FakeAstrometry(
        modes=["gaia"], rel_orbit=[None], config=[{"star_ndx": 0}]
    )
    assert _resolve(
        [{"name": "b"}], astrometryinstrument=ast, orbit=orbit
    ) == [False]


def test_user_override_beats_the_default():
    """chen: true/false wins regardless of the data topology."""
    assert _resolve(
        [{"name": "b", "chen": True}],
        transit=object(),
        rvinstrument=_FakeRV([0]),
        orbit=_ONE_PLANET_ORBIT,
    ) == [True]
    assert _resolve(
        [{"name": "b", "chen": False}],
        transit=object(),
        orbit=_ONE_PLANET_ORBIT,
    ) == [False]


def test_non_bool_chen_is_actionable():
    with pytest.raises(ValueError, match="must be true or false"):
        _resolve([{"name": "b", "chen": "yes"}], transit=object())


def _derives(planet_cfg, **comps):
    comp = Planet(planet_cfg, config_manager=None)
    comp._resolve_chen(_FakeSystem(**comps))
    return comp.chen_derives


def test_chen_derives_tracks_the_unmeasured_side():
    """
    Given each data topology,
    When the relation is resolved,
    Then chen_derives names the quantity the relation (not data) provides:
    the mass in a transit-only fit, the radius in an RV-only fit, nothing
    when chen is off, and both when chen is forced on with no data at all.
    """
    assert _derives(
        [{"name": "b"}], transit=object(), orbit=_ONE_PLANET_ORBIT
    ) == [{"mass"}]
    assert _derives(
        [{"name": "b"}], rvinstrument=_FakeRV([0]), orbit=_ONE_PLANET_ORBIT
    ) == [{"radius"}]
    assert _derives(
        [{"name": "b"}],
        transit=object(),
        rvinstrument=_FakeRV([0]),
        orbit=_ONE_PLANET_ORBIT,
    ) == [set()]
    assert _derives(
        [{"name": "b", "chen": True}],
        transit=object(),
        rvinstrument=_FakeRV([0]),
        orbit=_ONE_PLANET_ORBIT,
    ) == [set()]
    assert _derives(
        [{"name": "b", "chen": True}], orbit=_ONE_PLANET_ORBIT
    ) == [{"mass", "radius"}]


# ---------------------------------------------------------------------------
# 3. End to end through a real System (RV-only -> default on)
# ---------------------------------------------------------------------------
def _write_rv(path, seed, n=40):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, n))
    rv = 30.0 * np.sin(2 * np.pi * t / 17.0) + rng.normal(0, 3.0, n)
    err = np.full(n, 3.0)
    np.savetxt(path, np.column_stack([t, rv, err]))
    return str(path)


@pytest.fixture(scope="module")
def rv_file(tmp_path_factory):
    return _write_rv(tmp_path_factory.mktemp("chen_rv") / "a.rv", 7)


def _rv_system(rv_file, planet_extra=None):
    from exozippy.system import System

    planet_cfg = {"name": "b"}
    planet_cfg.update(planet_extra or {})
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [planet_cfg],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": [{"name": "inst", "file": rv_file}],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(17.0)},
        "orbit.b.tc": {"initval": 2455010.0},
    }
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


def test_rv_only_system_gets_the_chen_potential(rv_file):
    """
    Given an RV-only fit with no chen key,
    When the model is built,
    Then chen defaults on: the potential and prediction exist and logp and
    dlogp are finite at the start.
    """
    system, model = _rv_system(rv_file)

    assert system.planet.chen == [True]
    assert "planet.chen_prior" in {p.name for p in model.potentials}
    assert "planet.chen_radius_pred" in {d.name for d in model.deterministics}

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    grads = model.compile_dlogp()(point)
    assert np.all(
        np.isfinite(
            np.concatenate([np.atleast_1d(g) for g in np.atleast_1d(grads)])
        )
    )


def test_chen_false_removes_the_potential(rv_file):
    """chen: false opts the planet out entirely -- no potential, no notes."""
    system, model = _rv_system(rv_file, planet_extra={"chen": False})

    assert system.planet.chen == [False]
    assert "planet.chen_prior" not in {p.name for p in model.potentials}
    assert "planet.chen_radius_pred" not in {
        d.name for d in model.deterministics
    }
    assert system.planet.radius.table_note is None
    assert system.planet.mass.table_note is None


def test_table_notes_mark_radius_dependents_in_an_rv_only_fit(rv_file):
    """
    Given an RV-only fit (chen derives the radius from the mass),
    When the model is built,
    Then every radius-dependent parameter carries the
    'Derived from \\citet{Chen:2017}' note -- and mass-side parameters,
    which the data measure, do not.
    """
    system, _ = _rv_system(rv_file)
    note = r"Derived from \citet{Chen:2017}"

    # radius itself and everything computed from it
    assert system.planet.radius.table_note == note
    assert system.planet.density.table_note == note  # mass, radius
    assert system.planet.logg.table_note == note  # mass, radius
    assert system.planet.p.table_note == note  # radius / star.radius

    # measured side and unrelated parameters stay clean
    assert system.planet.mass.table_note is None
    assert system.planet.K.table_note is None
    assert system.star.radius.table_note is None


def test_prediction_matches_the_direct_formula(rv_file):
    """
    Given the built model,
    When planet.chen_radius_pred is evaluated at the initial point,
    Then it equals the relation applied to the initial mass (in internal
    solRad units, comparable to planet.radius).
    """
    system, model = _rv_system(rv_file)
    point = model.initial_point()

    compiled = model.compile_fn(
        model.replace_rvs_by_values(
            [model["planet.chen_radius_pred"], model["planet.mass"]]
        ),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    pred, mass_arr = compiled(point)
    pred_rsun = float(np.atleast_1d(pred)[0])
    mass = float(np.atleast_1d(mass_arr)[0])
    mpearth = max(mass * MSUN_TO_MEARTH, 1e-10)
    expected = _call(physics.calc_chen_radius, mpearth) / RSUN_TO_REARTH
    assert pred_rsun == pytest.approx(expected, rel=1e-10)
