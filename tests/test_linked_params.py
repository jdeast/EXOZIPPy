"""
Tests for user-defined parameter links (linking.py).

A params-file entry may set any numeric field (initval, mu, sigma, lower,
upper, init_scale) to a string expression referencing other parameters:

  - hard link:   star.A.age: {initval: star.B.age, sigma: 0}
  - soft link:   star.A.age: {initval: star.B.age, sigma: 1}
  - bound link:  star.A.av:  {lower: star.B.av}
  - algebra:     orbit.b.omega: {initval: "orbit.c.omega + math.pi", sigma: 0}

Unit convention: referenced parameters contribute their values in their own
user units; the expression result is interpreted in the target's user unit.
"""

import numpy as np
import pymc as pm
import pytest

from exozippy.config import ConfigManager
from exozippy.components.star.star import Star

DEG2RAD = np.pi / 180.0


def _build_star_param(user_params, param_names, config=None):
    """Build star parameters through the full config -> model pipeline.

    Returns (config_manager, star, model).
    """
    config = config or {"star": [{"name": "A"}, {"name": "B"}]}
    cm = ConfigManager(user_params, system_config=config)
    cm.finalize_user_params()

    star = Star(config["star"], cm)
    star.manifest = {p: None for p in param_names}

    with pm.Model() as model:
        for p in param_names:
            star.add_parameter(model, p, system=None)

    return cm, star, model


def _eval(model, tensor, point):
    """Evaluate a graph tensor at a raw point (same pattern as
    System.get_internal_point: free RVs are fed directly as inputs)."""
    import pytensor
    fn = pytensor.function(model.free_RVs, tensor, on_unused_input="ignore")
    return np.asarray(fn(*[point[rv.name] for rv in model.free_RVs]))


# ----------------------------------------------------------------------
# Hard link: sigma = 0
# ----------------------------------------------------------------------

def test_hard_link_fixes_parameter_to_other_parameter():
    """
    Given star.A.age = {initval: star.B.age, sigma: 0},
    When the model is built and star.B.age moves,
    Then star.A.age is never sampled and always equals star.B.age exactly.
    """
    # ARRANGE / ACT
    user_params = {"star.A.age": {"initval": "star.B.age", "sigma": 0}}
    cm, star, model = _build_star_param(user_params, ["age"])

    # ASSERT: A (index 0) is not sampled; only B's raw element exists
    assert list(star.age.is_sampled) == [False, True]
    point = model.initial_point()
    assert point["star.age_raw"].shape == (1,)

    # ASSERT: A tracks B exactly for arbitrary raw values
    for raw in (-3.0, 0.0, 1.7):
        point["star.age_raw"] = np.array([raw])
        ages = _eval(model, star.age.value, point)
        assert np.isclose(ages[0], ages[1], rtol=1e-12)


def test_hard_link_relaxation_engine_solves_snapshot_initval():
    """
    Given star.A.logmass hard-linked to star.B.logmass = 0.5,
    When finalize_user_params runs,
    Then the engine seeds A's initval from B AND derives dependent
    parameters (mass) through the physics relations using the linked value.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {
        "star.B.logmass": {"initval": 0.5},
        "star.A.logmass": {"initval": "star.B.logmass", "sigma": 0},
    }
    cm = ConfigManager(user_params, system_config=config)

    # ACT
    cm.finalize_user_params()

    # ASSERT: A's initval snapshot equals B's value
    assert np.isclose(cm.user_params["star.0.logmass"]["initval"], 0.5, rtol=1e-3)
    # ASSERT: physics relations saw the linked value (mass = 10**logmass)
    mass_entry = (cm.user_params.get("star.A.mass")
                  or cm.user_params.get("star.0.mass"))
    assert mass_entry is not None
    assert np.isclose(mass_entry["initval"], 10 ** 0.5, rtol=1e-3)


# ----------------------------------------------------------------------
# Soft link: sigma > 0
# ----------------------------------------------------------------------

def test_soft_link_samples_both_and_penalizes_difference():
    """
    Given star.A.age = {initval: star.B.age, sigma: 1},
    When the model is built,
    Then both ages are sampled and a Gaussian potential penalizes
    (A - B) / sigma dynamically.
    """
    # ARRANGE / ACT
    user_params = {"star.A.age": {"initval": "star.B.age", "sigma": 1.0}}
    cm, star, model = _build_star_param(user_params, ["age"])

    # ASSERT: both elements sampled
    assert list(star.age.is_sampled) == [True, True]
    point = model.initial_point()
    assert point["star.age_raw"].shape == (2,)

    pot = next(p for p in model.potentials if p.name == "link_mu.star.age.0")

    # ASSERT: potential equals -0.5*((A - B)/sigma)^2 at arbitrary points
    for raw in ([0.0, 0.0], [1.0, -1.0], [-2.0, 0.5]):
        point["star.age_raw"] = np.array(raw)
        ages = _eval(model, star.age.value, point)
        pot_val = _eval(model, pot, point)
        expected = -0.5 * (ages[0] - ages[1]) ** 2
        assert np.isclose(pot_val, expected, rtol=1e-10)


def test_soft_link_does_not_double_count_static_gaussian_prior():
    """
    Given a soft link on element A only,
    When the model is built,
    Then no static gaussian_prior potential is applied to element A
    (the linked potential replaces it, not stacks with it).
    """
    # ARRANGE / ACT
    user_params = {"star.A.age": {"initval": "star.B.age", "sigma": 1.0}}
    cm, star, model = _build_star_param(user_params, ["age"])

    # ASSERT: the only Gaussian-type potential on star.age is the link
    static_names = [p.name for p in model.potentials
                    if p.name == "gaussian_prior.star.age"]
    assert static_names == []


# ----------------------------------------------------------------------
# Dynamic bound link
# ----------------------------------------------------------------------

def test_lower_bound_link_enforces_ordering_constraint():
    """
    Given star.A.av = {lower: star.B.av},
    When the model is built,
    Then A's extinction can never fall below B's, for any raw values,
    and the normalized-span potential is present.
    """
    # ARRANGE / ACT
    user_params = {
        "star.A.av": {"initval": 0.7, "lower": "star.B.av"},
        "star.B.av": {"initval": 0.5},
    }
    cm, star, model = _build_star_param(user_params, ["av"])

    # ASSERT: both sampled, span normalization potential registered
    assert list(star.av.is_sampled) == [True, True]
    assert any(p.name == "link_span.star.av" for p in model.potentials)

    # ASSERT: constraint holds across the raw space, including extremes
    point = model.initial_point()
    for raw_a in (-8.0, -2.0, 0.0, 2.0, 8.0):
        for raw_b in (-5.0, 0.0, 5.0):
            point["star.av_raw"] = np.array([raw_a, raw_b])
            av = _eval(model, star.av.value, point)
            assert av[0] >= av[1] - 1e-12
            assert av[0] <= 100.0 + 1e-9


# ----------------------------------------------------------------------
# Algebraic expressions
# ----------------------------------------------------------------------

def test_algebraic_link_orbit_omega_snapshot():
    """
    Given orbit.b.omega = {initval: "orbit.c.omega + math.pi", sigma: 0},
    When the relaxation engine runs,
    Then b's initval snapshot equals c's omega plus pi, in omega's USER
    units (degrees) -- referenced params contribute their user-unit values.
    """
    # ARRANGE
    config = {"orbit": [{"name": "b"}, {"name": "c"}]}
    user_params = {
        "orbit.c.omega": {"initval": 90.0},
        "orbit.b.omega": {"initval": "orbit.c.omega + math.pi", "sigma": 0},
    }
    cm = ConfigManager(user_params, system_config=config)

    # ACT
    cm.finalize_user_params()

    # ASSERT: snapshot in user units (deg): 90 + pi
    assert np.isclose(cm.user_params["orbit.0.omega"]["initval"],
                      90.0 + np.pi, rtol=1e-6)


def test_algebraic_hard_link_with_unit_conversion_in_graph():
    """
    Given star.A.ra = {initval: "star.B.ra + 10", sigma: 0} where ra has
    user unit deg and internal unit rad,
    When the model is built,
    Then A's internal (radian) value always equals B's plus 10 degrees.
    """
    # ARRANGE / ACT
    user_params = {
        "star.B.ra": {"initval": 200.0},
        "star.A.ra": {"initval": "star.B.ra + 10", "sigma": 0},
    }
    cm, star, model = _build_star_param(user_params, ["ra"])

    # ASSERT
    point = model.initial_point()
    for raw in (-1.0, 0.0, 2.5):
        point["star.ra_raw"] = np.array([raw])
        ra = _eval(model, star.ra.value, point)  # internal units (rad)
        assert np.isclose(ra[0], ra[1] + 10.0 * DEG2RAD, rtol=1e-12)


def test_cross_parameter_hard_link_within_component():
    """
    Given star.A.av hard-linked to "0.01 * star.A.age" (a different
    parameter of the same component),
    When av is built,
    Then age is materialized automatically and av tracks it dynamically.
    """
    # ARRANGE / ACT
    user_params = {"star.A.av": {"initval": "0.01 * star.A.age", "sigma": 0}}
    cm, star, model = _build_star_param(user_params, ["age", "av"])

    # ASSERT
    point = model.initial_point()
    for raw in (-1.0, 0.0, 1.0):
        point["star.age_raw"] = np.array([raw, 0.0])
        age = _eval(model, star.age.value, point)
        av = _eval(model, star.av.value, point)
        assert np.isclose(av[0], 0.01 * age[0], rtol=1e-12)


# ----------------------------------------------------------------------
# Initialization-only link (no sigma)
# ----------------------------------------------------------------------

def test_initval_link_without_sigma_seeds_start_only():
    """
    Given star.A.age = {initval: star.B.age} with star.B.age = 2.0 and no sigma,
    When the model is built,
    Then A starts at B's value but remains independently sampled with no
    runtime tie (no link potential).
    """
    # ARRANGE / ACT
    user_params = {
        "star.B.age": {"initval": 2.0},
        "star.A.age": {"initval": "star.B.age"},
    }
    cm, star, model = _build_star_param(user_params, ["age"])

    # ASSERT: snapshot applied, both sampled, no dynamic potentials
    assert np.isclose(cm.user_params["star.0.age"]["initval"], 2.0, rtol=1e-3)
    assert list(star.age.is_sampled) == [True, True]
    assert not any(p.name.startswith("link_") for p in model.potentials)
    # Starting point equals the linked value
    point = model.initial_point()
    ages = _eval(model, star.age.value, point)
    assert np.isclose(ages[0], 2.0, rtol=1e-6)


# ----------------------------------------------------------------------
# Static-field links: sigma / init_scale snapshots
# ----------------------------------------------------------------------

def test_sigma_link_snapshots_numerically():
    """
    Given star.A.age with sigma = "0.5 * star.B.age" and star.B.age = 4.0,
    When finalize_user_params runs,
    Then sigma resolves to the static numeric snapshot 2.0.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {
        "star.B.age": {"initval": 4.0},
        "star.A.age": {"sigma": "0.5 * star.B.age"},
    }
    cm = ConfigManager(user_params, system_config=config)

    # ACT
    cm.finalize_user_params()

    # ASSERT
    assert np.isclose(cm.user_params["star.0.age"]["sigma"], 2.0, rtol=1e-3)


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------

def test_unknown_instance_in_link_raises():
    """
    Given a link referencing a non-existent instance star.C.age,
    When the ConfigManager is constructed,
    Then a ValueError names the bad reference.
    """
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {"star.A.age": {"initval": "star.C.age", "sigma": 0}}
    with pytest.raises(ValueError, match="no instance named"):
        ConfigManager(user_params, system_config=config)


def test_non_numeric_non_link_string_raises():
    """
    Given a garbage string in a numeric field,
    When the ConfigManager is constructed,
    Then a ValueError is raised rather than a deep numpy crash.
    """
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {"star.A.age": {"initval": "not_a_number"}}
    with pytest.raises(ValueError, match="neither a number"):
        ConfigManager(user_params, system_config=config)


def test_self_referencing_link_raises():
    """
    Given a link whose expression references its own target,
    When the ConfigManager is constructed,
    Then a ValueError is raised.
    """
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {"star.A.age": {"initval": "2 * star.A.age", "sigma": 0}}
    with pytest.raises(ValueError, match="references its own"):
        ConfigManager(user_params, system_config=config)


def test_circular_hard_links_raise():
    """
    Given two elements hard-linked to each other (A := B, B := A),
    When the parameter is built,
    Then a circular-link ValueError is raised.
    """
    user_params = {
        "star.A.age": {"initval": "star.B.age", "sigma": 0},
        "star.B.age": {"initval": "star.A.age", "sigma": 0},
    }
    with pytest.raises(ValueError, match="[Cc]ircular"):
        _build_star_param(user_params, ["age"])
