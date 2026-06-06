import numpy as np
import pytest
import pymc as pm
from unittest.mock import patch
from exozippy.config import ConfigManager
from exozippy.components.rvinstrument.rvinstrument import RVInstrument
from exozippy.components.star.star import Star
from exozippy.components.parameter import Parameter
from exozippy.system import System


class MockSystem:
    """Mock system to isolate ConfigManager and Auditor testing from the full PyMC graph."""

    def __init__(self, user_params):
        self.user_params = user_params
        self.config_manager = ConfigManager(user_params)
        self.star = None

    def get_parameter_lookup(self):
        lookup = {}
        if self.star:
            for attr in self.star.__dict__.values():
                if isinstance(attr, Parameter):
                    lookup[attr.label] = attr
        return lookup

    def get_all_parameters(self):
        if self.star:
            return [v for v in self.star.__dict__.values() if isinstance(v, Parameter)]
        return []


def test_instrument_name_override_resolves_correctly():
    """Verify that System correctly identifies and registers instruments."""
    config = {
        "star": [{"name": "A"}],
        "rvinstrument": [{"name": "HIRES"}, {"name": "HARPS"}],
    }
    # Don't pass user_params, don't trigger load_data, don't register parameters.
    # Just build the system and check the registry.
    system = System(config, {})

    # Check that the registry was built correctly
    inst = system.active_components['rvinstrument']
    assert len(inst.config) == 2
    assert inst.config[0]["name"] == "HIRES"


def test_gaussian_prior_scale_override_applies_correctly():
    """
    Given a user configuration that provides a specific init_scale alongside a Gaussian sigma,
    When the parameter is built,
    Then it should respect the explicit init_scale rather than defaulting to the sigma width.
    """
    # ARRANGE
    label = "star.A.radius_test3"
    user_params = {label: {"initval": 1.0, "sigma": 0.05, "init_scale": 0.00085, "lower": 0.0, "upper": 10.0}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)
    star.name = "A"

    # ACT
    with pm.Model(name="model_test3") as model:
        star.manifest = {"radius_test3": {}}
        star.add_parameter(model=model, param_name="radius_test3", system=None)

    # ASSERT
    assert np.isclose(star.radius_test3.init_scale[0], 0.00085)


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_unrecognized_yaml_subkey_triggers_auditor_warning(mock_logp, caplog):
    """
    Given a YAML dictionary containing a misspelled sub-key (e.g., 'sigm' instead of 'sigma'),
    When the ModelAuditor inspects the starting state,
    Then it should log a warning flagging the unused key.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})

    label = "star.A.mass_test4"
    user_params = {label: {"initval": 1.0, "sigm": 0.05, "lower": 0.0, "upper": 10.0}}  # Misspelled 'sigm'
    system = MockSystem(user_params)
    star = Star([{"name": "A"}], system.config_manager)
    system.star = star

    with pm.Model(name="model_test4") as model:
        star.manifest = {"mass_test4": {}}
        star.add_parameter(model=model, param_name="mass_test4", system=None)

    # ACT
    from exozippy.run import inspect_start
    import logging
    with caplog.at_level(logging.WARNING):
        inspect_start(model, system, {}, {}, {}, calc_curvature=False)

    # ASSERT
    assert "sigm" in caplog.text


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_unrecognized_top_level_yaml_key_triggers_auditor_warning(mock_logp, caplog):
    """
    Given a YAML configuration containing a completely unrecognized top-level parameter,
    When the ModelAuditor inspects the starting state,
    Then it should log a warning explicitly naming the orphaned key.
    """
    # ARRANGE
    from exozippy.run import inspect_start
    mock_logp.return_value = ({}, {})

    user_params = {"star.A.radiuss": 1.0}  # Misspelled 'radiuss'
    system = MockSystem(user_params)
    star = Star([{"name": "A"}], system.config_manager)
    system.star = star

    with pm.Model(name="model_test5") as model:
        star.manifest = {"mass": {}}
        star.add_parameter(model=model, param_name="mass", system=None)

    # ACT
    import logging
    with caplog.at_level(logging.WARNING):
        inspect_start(model, system, {}, {}, {}, calc_curvature=False)

    # ASSERT
    assert "star.A.radiuss" in caplog.text


def test_user_boundary_overrides_tighten_but_never_expand_limits():
    """
    Given internal component boundary defaults,
    When a user attempts to override those bounds via YAML,
    Then the system should accept tightening bounds but reject expanding bounds.
    """
    scenarios = [
        {"user": -10.0, "internal": 0.0, "expected": 0.0, "type": "lower", "other_type": "upper", "other_val": 10.0},
        {"user": 0.5, "internal": 0.0, "expected": 0.5, "type": "lower", "other_type": "upper", "other_val": 10.0},
        {"user": 20.0, "internal": 10.0, "expected": 10.0, "type": "upper", "other_type": "lower", "other_val": 0.0},
    ]

    for s in scenarios:
        # ARRANGE
        label = f"star.A.mass_{s['type']}"
        user_params = {
            label: {
                s['type']: s['user'],
                s['other_type']: s['other_val']
            }
        }

        # Inject the internal defaults directly into the base dictionary
        cm = ConfigManager(user_params)
        cm.base_defaults[f"mass_{s['type']}"] = {
            s['type']: s['internal'],
            s['other_type']: s['other_val'],
            "unit": "", "internal_unit": ""
        }

        star = Star([{"name": "A"}], cm)
        star.name = "A"

        # ACT
        with pm.Model(name=f"model_{label.replace('.', '_')}") as model:
            star.manifest = {label.split('.')[-1]: {}}
            star.add_parameter(model=model, param_name=label.split('.')[-1], system=None)
            p = getattr(star, label.split('.')[-1])
            val = p.lower[0] if s['type'] == 'lower' else p.upper[0]

        # ASSERT
        assert np.isclose(val, s['expected'])


def test_defining_mu_and_sigma_creates_valid_gaussian_prior():
    """
    Given a user parameter override containing both mu and sigma,
    When the Parameter object is instantiated,
    Then it should properly store both values to inform the PyMC prior.
    """
    # ARRANGE
    label = "star.A.mass_sampled"
    user_params = {label: {"mu": 1.1, "sigma": 0.1, "lower": 0.0, "upper": 10.0}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)
    star.name = "A"

    # ACT
    with pm.Model(name="model_sampled") as model:
        star.manifest = {"mass_sampled": {}}
        star.add_parameter(model=model, param_name="mass_sampled", system=None)

    # ASSERT
    p = star.mass_sampled
    assert p.mu is not None and p.mu[0] == 1.1
    assert p.sigma is not None and p.sigma[0] == 0.1


def test_derived_parameter_correctly_registers_gaussian_potential():
    """
    Given a derived deterministic parameter that a user assigns a mu and sigma to,
    When the parameter is resolved,
    Then it should successfully hold the prior values to generate a PyMC Potential.
    """
    # ARRANGE
    user_params = {"orbit.b.period": {"mu": 10.0, "sigma": 0.01}}
    config_manager = ConfigManager(user_params)

    # ACT
    resolved = config_manager.resolve("orbit", "period", shape=(1,), names=["b"])
    resolved.pop("expressions", None)

    p = Parameter(
        label="orbit.period",
        is_derived=True,
        **resolved
    )

    # ASSERT
    assert p.mu is not None and p.mu[0] == 10.0


def test_explicit_init_scale_overrides_default_sigma_scaling():
    """
    Given a parameter with a defined Gaussian prior (mu, sigma) AND an explicit init_scale,
    When the parameter is built,
    Then the tuning scale (init_scale) should decouple from the physics prior (sigma).
    """
    # ARRANGE
    label = "star.A.radius_custom"
    user_params = {label: {"mu": 1.0, "sigma": 0.05, "init_scale": 0.001, "lower": 0.0, "upper": 10.0}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)

    # ACT
    with pm.Model(name="model_custom_scale") as model:
        star.manifest = {"radius_custom": {}}
        star.add_parameter(model=model, param_name="radius_custom", system=None)

    # ASSERT
    p = star.radius_custom
    assert p.mu is not None and np.isclose(p.mu[0], 1.0)
    assert np.isclose(p.init_scale[0], 0.001)


def test_vectorized_overrides_apply_to_correct_indices():
    """
    Given a configuration with multiple components of the same type (e.g., binary stars),
    When a user overrides a parameter for only one specific component by name,
    Then the override should apply to the targeted index while leaving the other index at default.
    """
    # ARRANGE
    user_params = {
        "star.B.mass": {"initval": 0.85, "sigma": 0.02},
        "star.A.mass": {"initval": 1.0}
    }
    config_manager = ConfigManager(user_params)
    star = Star([{"name": "A"}, {"name": "B"}], config_manager)
    star.name = "A"

    # ACT
    with pm.Model(name="model_vector") as model:
        star.manifest = {"mass": {}}
        star.add_parameter(model=model, param_name="mass", system=None)

    # ASSERT
    p = star.mass
    assert np.isclose(p.initval[0], 1.0)
    assert np.isnan(p.sigma[0])
    assert np.isclose(p.initval[1], 0.85)
    assert np.isclose(p.sigma[1], 0.02)


def test_config_manager_resolves_vectors_with_mixed_overrides():
    """
    Given a multi-element parameter where elements have different types of overrides (one dict, one float),
    When the ConfigManager resolves the payload,
    Then it should cleanly construct synchronized arrays for all internal keys.
    """
    # ARRANGE
    user_params = {
        "orbit.b.logP": {"initval": 1.2, "sigma": 0.05},
        "orbit.c.logP": {"initval": 3.4}
    }
    cm = ConfigManager(user_params)

    # ACT
    resolved = cm.resolve("orbit", "logP", shape=(2,), internal_overrides={"initval": [0.0, 0.0], "init_scale": 1.0},
                          names=["b", "c"])

    # ASSERT
    assert np.allclose(resolved['initval'], [1.2, 3.4])
    assert np.allclose(resolved['init_scale'], [0.05, 1.0])
    assert resolved['mu'] is None


def test_direct_component_resolution_prevents_nan_initialization():
    """
    Given an empty user parameter set,
    When the ConfigManager resolves a core physical parameter,
    Then it should fall back to standard defaults and avoid injecting NaNs.
    """
    # ARRANGE
    cm = ConfigManager({})

    # ACT
    logP_cfg = cm.resolve("orbit", "logP", shape=(2,), names=["b", "c"])
    logP_init = logP_cfg['initval']

    # ASSERT
    assert logP_init[0] is not None
    assert not np.isnan(logP_init[0])


def test_system_initialization_safely_resolves_global_defaults():
    """
    Given a sparse user configuration overriding only a few specific components,
    When ConfigManager probes the global namespace,
    Then it should cleanly retrieve defaults for unspecified attributes (like radius).
    """
    # ARRANGE
    user_params = {
        "star.0.mass": {"initval": 0.98},
        "orbit.0.logP": {"initval": 2.0}
    }
    cm = ConfigManager(user_params)

    # ACT & ASSERT
    resolved_star = cm.resolve("star", "mass", shape=(1,), names=["A"])
    assert resolved_star["initval"][0] == 0.98

    resolved_rad = cm.resolve("star", "radius", shape=(1,), names=["A"])
    assert resolved_rad["initval"][0] == 1.0


def test_config_overrides_apply_identically_via_name_or_index():
    """
    Given a configuration overriding a parameter by list index and another by custom name,
    When the ConfigManager resolves the parameters,
    Then both methods should inject the exact same properties.
    """
    # ARRANGE
    cfg_by_index = ConfigManager({"star.0.mass": {"initval": 1.23, "sigma": 0.05}})
    cfg_by_name = ConfigManager({"star.A.mass": {"initval": 1.23, "sigma": 0.05}})

    # ACT
    resolved_index = cfg_by_index.resolve("star", "mass", shape=(1,), names=["A"])
    resolved_name = cfg_by_name.resolve("star", "mass", shape=(1,), names=["A"])

    # ASSERT
    assert np.isclose(resolved_index["initval"][0], 1.23)
    assert np.isclose(resolved_name["initval"][0], 1.23)
    assert np.isclose(resolved_index["sigma"][0], 0.05)
    assert np.isclose(resolved_name["sigma"][0], 0.05)


from astropy import units as u


def test_user_unit_override_translates_to_internal_and_back():
    """
    Given a user configuration that changes the default unit of a parameter (e.g., Jupiter mass to Earth mass),
    When the ConfigManager resolves the parameter and initializes it,
    Then the internal value should be properly converted, and from_internal should return the exact Earth mass value.
    """
    # ARRANGE
    user_params = {
        "planet.b.mass": {"initval": 1.0, "unit": "earthMass"}
    }
    cm = ConfigManager(user_params)

    # ACT
    resolved = cm.resolve("planet", "mass", shape=(1,), names=["b"])
    resolved.pop("expressions", None)

    p = Parameter(label="planet.b.mass", **resolved)

    internal_val = p.initval[0]
    user_val = p.from_internal(internal_val)[0]

    # ASSERT
    assert p.unit[0] == u.earthMass
    assert np.isclose(internal_val, 3.00273e-6, rtol=1e-3)
    assert np.isclose(user_val, 1.0)


def test_unit_override_scales_default_values_to_new_units():
    """
    Given a parameter with a default of 1.0 JupiterMass,
    When a user overrides the unit to 'earthMass' without overriding the initval,
    Then the initval should be automatically scaled to ~317.8 EarthMasses.
    """
    # ARRANGE
    user_params = {"planet.b.mass": {"unit": "earthMass"}}
    cm = ConfigManager(user_params)

    # ACT
    resolved = cm.resolve("planet", "mass", shape=(1,), names=["b"])
    resolved.pop("expressions", None)

    p = Parameter(label="planet.b.mass", **resolved)

    # ASSERT
    assert p.unit[0] == u.earthMass
    user_val = p.from_internal(p.initval)[0]
    assert np.isclose(user_val, 317.8, rtol=1e-2)
    assert np.isclose(p.initval[0], 0.000954, rtol=1e-3)


def test_config_manager_scales_arbitrary_units_generically():
    """
    Given a parameter with a default unit of 'm' (meters),
    When a user overrides the unit to 'km' (kilometers) and an auto-estimate provides a value in 'm',
    Then the ConfigManager should scale the estimate value by 0.001.
    """
    # ARRANGE
    user_params = {"star.A.arm_length": {"unit": "km"}}
    cm = ConfigManager(user_params)
    cm.base_defaults["arm_length"] = {
        "unit": "m",
        "initval": 1000.0,
        "internal_unit": "m"
    }

    # ACT
    resolved = cm.resolve("star", "arm_length", shape=(1,),
                          internal_overrides={"initval": [5000.0]}, names=["A"])

    # ASSERT
    assert np.isclose(resolved["initval"][0], 5.0)
    assert resolved["unit"] == "km"