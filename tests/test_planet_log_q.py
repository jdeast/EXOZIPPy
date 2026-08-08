"""Tests for the planet log_q reparameterization.

A planet whose mass cannot be negative samples log_q = log10(m_p / m_host)
and derives the mass, instead of sampling the (signed) linear mass.  That is
every planet except the ones RV or astrometric data measure -- those keep the
signed linear coordinate so a marginal detection does not pick up the
Lucy-Sweeney positive-definite bias.  Microlensing lens bodies always use
log_q: the magnification model clips q to [1e-9, 100], so q <= 0 is
meaningless there no matter what else measures the orbit.
"""

import logging

import numpy as np
import pytensor
import pytest

from exozippy.components.mulensing import physics as mulens_physics
from exozippy.components.planet.planet import Planet
from exozippy.system import System


# ---------------------------------------------------------------------------
# 1. The per-planet default decision (no model build)
# ---------------------------------------------------------------------------
class _FakeOrbit:
    def __init__(self, memberships, bodies):
        self._memberships = memberships
        self._bodies = bodies

    def star_membership(self, star_idx):
        return self._memberships.get(int(star_idx), [])

    def bodies(self, i):
        return self._bodies.get(int(i), [])


class _FakeRV:
    def __init__(self, star_ndx):
        self.star_ndx = star_ndx


class _FakeLens:
    def __init__(self, bodies):
        self.lens_bodies = [bodies]


class _FakeSystem:
    def __init__(self, **comps):
        self.active_components = comps


class _FakeConfigManager:
    """Just enough for _reconcile_mass_user_params."""

    def __init__(self, user_params=None):
        self.user_params = user_params if user_params is not None else {}


_ONE_PLANET_ORBIT = _FakeOrbit(
    memberships={0: [(0, "primary")]},
    bodies={0: [("star", 0), ("planet", 0)]},
)


def _mode(planet_cfg, user_params=None, **comps):
    comp = Planet(planet_cfg, config_manager=_FakeConfigManager(user_params))
    comp._resolve_mass_parameterization(_FakeSystem(**comps))
    return comp.mass_parameterization


def test_rv_planet_defaults_to_linear():
    """
    Given RV data measuring the planet's orbit,
    When the mass coordinate is resolved,
    Then it stays linear so the amplitude can flip sign.
    """
    assert (
        _mode(
            [{"name": "b"}],
            rvinstrument=_FakeRV([0]),
            orbit=_ONE_PLANET_ORBIT,
        )
        == "linear"
    )


def test_transit_only_planet_defaults_to_log_q():
    """
    Given transit data only (Chen determines the mass),
    When the mass coordinate is resolved,
    Then it is log_q: the negative half-axis is a flat plateau under Chen's
    1e-10 Mearth clip, with no restoring force.
    """
    assert (
        _mode([{"name": "b"}], transit=object(), orbit=_ONE_PLANET_ORBIT)
        == "log_q"
    )


def test_lens_body_defaults_to_log_q_even_with_rv():
    """
    Given a planet that is both a microlensing lens body and RV-measured,
    When the mass coordinate is resolved,
    Then log_q wins: microlensing hard-forbids q <= 0.
    """
    assert (
        _mode(
            [{"name": "b"}],
            lens=_FakeLens([("star", 0), ("planet", 0)]),
            rvinstrument=_FakeRV([0]),
            orbit=_ONE_PLANET_ORBIT,
        )
        == "log_q"
    )


def test_per_planet_override_wins():
    """A mass_parameterization key beats the topology default either way."""
    assert (
        _mode(
            [{"name": "b", "mass_parameterization": "log_q"}],
            rvinstrument=_FakeRV([0]),
            orbit=_ONE_PLANET_ORBIT,
        )
        == "log_q"
    )
    assert (
        _mode(
            [{"name": "b", "mass_parameterization": "linear"}],
            lens=_FakeLens([("star", 0), ("planet", 0)]),
        )
        == "linear"
    )


def test_bad_override_raises():
    with pytest.raises(ValueError, match="must be 'linear' or 'log_q'"):
        _mode([{"name": "b", "mass_parameterization": "logq"}])


_TWO_PLANET_ORBITS = _FakeOrbit(
    memberships={0: [(0, "primary")]},
    bodies={0: [("star", 0), ("planet", 0)], 1: [("star", 0), ("planet", 1)]},
)


def test_mixed_topology_falls_back_to_linear(caplog):
    """
    Given one RV-measured planet and one that is not,
    When the mass coordinate is resolved,
    Then both fall back to linear (one mode per component) and the reason is
    logged, rather than a build failing later.
    """
    with caplog.at_level(logging.INFO):
        mode = _mode(
            [{"name": "b"}, {"name": "c", "orbit_ndx": 1}],
            rvinstrument=_FakeRV([0]),
            orbit=_TWO_PLANET_ORBITS,
        )

    assert mode == "linear"
    assert "not yet supported" in caplog.text


def test_explicit_mixed_override_raises():
    """An explicit per-planet disagreement is a user error, not a fallback."""
    with pytest.raises(ValueError, match="must share one"):
        _mode(
            [
                {"name": "b", "mass_parameterization": "linear"},
                {
                    "name": "c",
                    "orbit_ndx": 1,
                    "mass_parameterization": "log_q",
                },
            ],
            orbit=_TWO_PLANET_ORBITS,
        )


def test_fixed_mass_translates_to_log_q():
    """
    Given a user pinning planet.mass with sigma=0 in log_q mode,
    When the mass coordinate is resolved,
    Then the pin moves onto the sampled log_q -- sigma=0 on a derived
    parameter is a silent no-op.
    """
    up = {"planet.0.mass": {"initval": 1.0, "sigma": 0.0}}
    assert _mode([{"name": "b"}], user_params=up, transit=object()) == "log_q"

    assert "sigma" not in up["planet.0.mass"]
    assert up["planet.0.log_q"]["sigma"] == 0.0


def test_positive_sigma_on_mass_is_left_alone():
    """A real Gaussian prior on the mass works on the derived node; only the
    sigma=0 pin needs moving."""
    up = {"planet.0.mass": {"initval": 1.0, "sigma": 0.1}}
    _mode([{"name": "b"}], user_params=up, transit=object())

    assert up["planet.0.mass"]["sigma"] == 0.1
    assert "planet.0.log_q" not in up


def test_negative_mass_initval_in_log_q_mode_raises():
    """A negative mass is unrepresentable as log_q; say so rather than
    silently deriving a positive one from the defaults.yaml start."""
    up = {"planet.0.mass": {"initval": -1.0}}
    with pytest.raises(ValueError, match="cannot represent it"):
        _mode([{"name": "b"}], user_params=up, transit=object())


def test_stale_log_q_in_linear_mode_raises():
    """A mkparam restart file from a log_q fit must not be silently ignored
    when the topology has since resolved to linear."""
    up = {"planet.0.log_q": {"initval": -3.0}}
    with pytest.raises(ValueError, match="samples a linear mass"):
        _mode(
            [{"name": "b"}],
            user_params=up,
            rvinstrument=_FakeRV([0]),
            orbit=_ONE_PLANET_ORBIT,
        )


# ---------------------------------------------------------------------------
# 2. End to end: a microlensing binary lens builds in log_q
# ---------------------------------------------------------------------------
def _binary_config(planet_extra=None):
    """Minimal single-source binary lens (star + planet companion)."""
    planet_cfg = {"name": "b"}
    planet_cfg.update(planet_extra or {})
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [planet_cfg],
        "lens": [
            {
                "name": "Lens",
                "lenses": ["star.0", "planet.0"],
                "sources": ["star.1"],
            }
        ],
    }


def _binary_params(extra=None):
    p = {
        "lens.Lens.t_0": {"initval": 2460025.0, "init_scale": 0.1},
        "lens.Lens.u_0": {"initval": 0.1, "init_scale": 0.01},
        "lens.Lens.t_E": {"initval": 30.0, "init_scale": 1.0},
        "lens.Lens.s": {"initval": 0.98},
        "lens.Lens.alpha": {"initval": 60.0, "init_scale": 0.9},
        "lens.Lens.q": {"initval": 1e-3, "init_scale": 1e-4},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("Lens", "Source"):
        p[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        p[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    p.update(extra or {})
    return p


def _build(planet_extra=None, extra_params=None):
    system = System(
        _binary_config(planet_extra), user_params=_binary_params(extra_params)
    )
    system.prepare()
    return system, system.build_model()


@pytest.fixture(scope="module")
def lens_system():
    return _build()


def test_lens_planet_samples_log_q_and_derives_mass(lens_system):
    """
    Given a binary lens whose companion is a planet,
    When the model is built,
    Then log_q is the free RV, mass is not, and mass is a derived node.
    """
    system, model = lens_system

    free_names = [v.name for v in model.free_RVs]
    assert "planet.log_q_raw" in free_names
    assert "planet.mass_raw" not in free_names

    assert system.planet.mass_parameterization == "log_q"
    assert system.planet.manifest["log_q"] is None
    assert system.planet.manifest["mass"]["expr_key"] == "default"

    # force_node keeps the mass in the trace, as it is when sampled -- tables
    # and plots read it from there.
    assert "planet.mass" in {d.name for d in model.deterministics}


def test_derived_mass_matches_the_relation(lens_system):
    """mass == 10**log_q * star.mass[star_map] at the start point."""
    system, model = lens_system

    with model:
        f = pytensor.function(
            model.free_RVs,
            [
                system.planet.mass.value,
                system.planet.log_q.value,
                system.star.mass.value,
            ],
            on_unused_input="ignore",
        )
        zeros = [
            np.zeros_like(model.initial_point()[v.name]).astype(float)
            for v in model.free_RVs
        ]
        mass, log_q, star_mass = [np.atleast_1d(x) for x in f(*zeros)]

    host = star_mass[system.planet.star_map]
    np.testing.assert_allclose(mass, 10.0**log_q * host, rtol=1e-6)


def test_lens_q_seeds_the_log_q_start(lens_system):
    """
    Given a user lens.q initval,
    When the relaxation engine runs,
    Then it back-solves through the mass to a log_q start, and mass keeps
    the matching value.
    """
    system, _ = lens_system

    mass = np.atleast_1d(system.planet.mass.initval)
    log_q = np.atleast_1d(system.planet.log_q.initval)
    star_mass = np.atleast_1d(system.star.mass.initval)[system.planet.star_map]

    np.testing.assert_allclose(log_q, np.log10(mass / star_mass), atol=1e-6)
    # lens.q = 1e-3 was the user's input; the chain should land on it.
    np.testing.assert_allclose(10.0**log_q, [1e-3], rtol=1e-3)

    scale = np.atleast_1d(system.planet.log_q.init_scale)
    assert np.all(np.isfinite(scale)) and np.all(scale > 0)


def test_logp_is_finite_at_the_start(lens_system):
    """The reparameterized model still starts in a valid place."""
    _, model = lens_system

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert all(np.all(np.isfinite(g)) for g in model.compile_dlogp()(point))


def test_override_keeps_the_linear_mass_sampled():
    """mass_parameterization: linear on a lens body forces the old
    coordinate, negative lower bound and all."""
    system, model = _build({"mass_parameterization": "linear"})

    free_names = [v.name for v in model.free_RVs]
    assert "planet.mass_raw" in free_names
    assert "planet.log_q_raw" not in free_names
    assert system.planet.manifest["mass"] is None
    assert np.min(np.atleast_1d(system.planet.mass.lower)) < 0.0


def test_log_q_relation_is_inert_in_linear_mode():
    """
    Given a linear-mode planet,
    When the relaxation engine runs,
    Then log_q absorbs Eq(mass, 10**log_q * star_mass) (it carries the lowest
    rank), so the mass keeps the value the user asked for.
    """
    system, _ = _build(
        {"mass_parameterization": "linear"},
        extra_params={"planet.b.mass": {"initval": 2.5}},
    )

    # Parameter.initval is in internal units (solMass); the params file is in
    # the user unit (jupiterMass).
    mjup_to_msun = system.config_manager.get_conversion_factor(
        "planet", "mass"
    )
    np.testing.assert_allclose(
        np.atleast_1d(system.planet.mass.initval),
        [2.5 * mjup_to_msun],
        rtol=1e-6,
    )


def test_fixed_mass_really_pins_the_sampled_element():
    """
    Given planet.b.mass sigma=0 written the way a user writes it (by name),
    When the model is built in log_q mode,
    Then log_q is not sampled -- i.e. the translation survives the
    name-to-index standardization and reaches the built Parameter.
    """
    system, model = _build(
        extra_params={"planet.b.mass": {"initval": 1.0, "sigma": 0.0}}
    )

    assert not np.any(np.atleast_1d(system.planet.log_q.is_sampled))
    assert "planet.log_q_raw" not in [v.name for v in model.free_RVs]


def test_export_reports_the_real_derivation_not_the_yaml_block():
    """
    Given planet.mass now carries an expressions block used only in log_q
    mode,
    When the solution is exported,
    Then 'derived' follows the built manifest, not the static yaml -- else a
    linear-mode mass would be reported derived and skipped by the solve API's
    bounds check.
    """
    system, _ = _build({"mass_parameterization": "linear"})
    export = system.config_manager.export_solution(
        derived_params=system.derived_params()
    )
    assert export["parameters"]["planet.b.mass"]["derived"] is False

    system, _ = _build({"mass_parameterization": "log_q"})
    export = system.config_manager.export_solution(
        derived_params=system.derived_params()
    )
    assert export["parameters"]["planet.b.mass"]["derived"] is True
    # star.mass is derived from logmass in every topology -- a control.
    assert export["parameters"]["star.Lens.mass"]["derived"] is True


# ---------------------------------------------------------------------------
# 3. calc_theta_E's mass guard
# ---------------------------------------------------------------------------
def test_theta_E_is_finite_for_negative_lens_mass():
    """
    Given a negative total lens mass (a linear-mode planet mass can drag
    mlens_total below zero),
    When theta_E is evaluated,
    Then it is finite -- lens.build_likelihood takes log(theta_E), so a NaN
    would poison the logp and its gradient over the whole region instead of
    letting the theta_E_singularity soft bound penalize it.
    """
    import pytensor.tensor as pt

    m = pt.dscalar("m")
    pi_rel = pt.dscalar("pi_rel")
    theta_E = mulens_physics.calc_theta_E(m, pi_rel)

    val = float(theta_E.eval({m: -1.0, pi_rel: 0.1}))
    assert np.isfinite(val)
    assert np.isfinite(float(pt.log(theta_E).eval({m: -1.0, pi_rel: 0.1})))
