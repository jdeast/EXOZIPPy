import numpy as np
import pytest
import pymc as pm
from unittest.mock import patch
from exozippy.config import ConfigManager
from exozippy.components.rv_instrument.rv_instrument import RVInstrument
from exozippy.components.star.star import Star
from exozippy.components.parameter import Parameter
from exozippy.diagnostics import ModelAuditor


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
    """
    Given a user parameter dictionary that overrides a specific instrument's gamma by name,
    When the RVInstrument component parses the configuration,
    Then the specific initial value should perfectly match the user override in internal units.
    """
    # ARRANGE
    user_params = {"inst.HIRES.gamma": {"initval": 123.45}}
    config_manager = ConfigManager(user_params)
    inst = RVInstrument([{"name": "HIRES"}], config_manager)
    inst.name = "HIRES"
    inst.prefix = "inst"

    # ACT
    with pm.Model(name="test_inst"):
        inst.build_pars_from_dict({"gamma": {"initval": [0.0]}}, shape=(1,), prefix="inst")

    # ASSERT
    expected = 123.45 / inst.gamma._get_conversion_factors()[0]
    assert np.isclose(inst.gamma.initval[0], expected)


def test_gaussian_prior_scale_override_applies_correctly():
    """
    Given a user configuration that provides a specific init_scale alongside a Gaussian sigma,
    When the parameter is built,
    Then it should respect the explicit init_scale rather than defaulting to the sigma width.
    """
    # ARRANGE
    label = "star.A.radius_test3"
    user_params = {label: {"initval": 1.0, "sigma": 0.05, "init_scale": 0.00085}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)
    star.name = "A"
    star.prefix = "star"

    # ACT
    with pm.Model(name="model_test3"):
        star.add_parameter("radius_test3", config_manager, prefix="star")

    # ASSERT
    assert np.isclose(star.radius_test3.init_scale[0], 0.00085)


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_unrecognized_yaml_subkey_triggers_auditor_warning(mock_logp, capsys):
    """
    Given a YAML dictionary containing a misspelled sub-key (e.g., 'sigm' instead of 'sigma'),
    When the ModelAuditor inspects the starting state,
    Then it should print a warning to standard output flagging the unused key.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})

    label = "star.A.mass_test4"
    user_params = {label: {"initval": 1.0, "sigm": 0.05}}  # Misspelled 'sigm'
    system = MockSystem(user_params)
    star = Star([{"name": "A"}], system.config_manager)
    star.prefix = "star"
    system.star = star

    with pm.Model(name="model_test4") as model:
        star.add_parameter("mass_test4", system.config_manager, prefix="star")

    # ACT
    from exozippy.run import inspect_start
    inspect_start(model, system, {}, {}, {}, calc_curvature=False)

    # ASSERT
    out = capsys.readouterr().out
    assert "sigm" in out


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_unrecognized_top_level_yaml_key_triggers_auditor_warning(mock_logp, capsys):
    """
    Given a YAML configuration containing a completely unrecognized top-level parameter,
    When the ModelAuditor inspects the starting state,
    Then it should print a warning explicitly naming the orphaned key.
    """
    # ARRANGE
    from exozippy.run import inspect_start
    mock_logp.return_value = ({}, {})

    user_params = {"star.A.radiuss": 1.0}  # Misspelled 'radiuss'
    system = MockSystem(user_params)
    star = Star([{"name": "A"}], system.config_manager)
    system.star = star

    with pm.Model(name="model_test5") as model:
        star.add_parameter("mass", system.config_manager, prefix="star")

    # ACT
    inspect_start(model, system, {}, {}, {}, calc_curvature=False)

    # ASSERT
    captured = capsys.readouterr()
    assert "star.A.radiuss" in captured.out


def test_user_boundary_overrides_tighten_but_never_expand_limits():
    """
    Given internal component boundary defaults,
    When a user attempts to override those bounds via YAML,
    Then the system should accept tightening bounds but reject expanding bounds.
    """
    scenarios = [
        # Attempting to expand a 0.0 lower bound to -10.0 should be rejected (remains 0.0)
        {"user": -10.0, "internal": 0.0, "expected": 0.0, "type": "lower"},
        # Attempting to tighten a 0.0 lower bound to 0.5 should be accepted (becomes 0.5)
        {"user": 0.5, "internal": 0.0, "expected": 0.5, "type": "lower"},
    ]

    for s in scenarios:
        # ARRANGE
        label = f"star.A.mass_{s['type']}"
        user_params = {label: {s['type']: s['user']}}
        config_manager = ConfigManager(user_params)

        star = Star([{"name": "A"}], config_manager)
        star.name = "A"
        star.prefix = "star"

        # ACT
        with pm.Model(name=f"model_{label.replace('.', '_')}"):
            star.add_parameter(label.split('.')[-1], config_manager, prefix="star", **{s['type']: s['internal']})
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
    user_params = {label: {"mu": 1.1, "sigma": 0.1}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)
    star.name = "A"
    star.prefix = "star"

    # ACT
    with pm.Model(name="model_sampled"):
        star.add_parameter("mass_sampled", config_manager, prefix="star")

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
    user_params = {label: {"mu": 1.0, "sigma": 0.05, "init_scale": 0.001}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)
    star.prefix = "star"

    # ACT
    with pm.Model(name="model_custom_scale"):
        star.add_parameter("radius_custom", config_manager, prefix="star")

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
        "star.B.mass": {"initval": 0.85, "sigma": 0.02}
    }
    config_manager = ConfigManager(user_params)
    star = Star([{"name": "A"}, {"name": "B"}], config_manager)
    star.prefix = "star"

    # ACT
    with pm.Model(name="model_vector"):
        # Base initval of [1.0, 1.0] representing standard DNA defaults
        star.add_parameter("mass", config_manager, prefix="star", shape=(2,), initval=[1.0, 1.0])

    # ASSERT
    p = star.mass

    # Star A (Index 0) should retain defaults (except for heuristic scale balancing)
    assert np.isclose(p.initval[0], 1.0)
    assert np.isnan(p.sigma[0])
    assert np.isclose(p.init_scale[0], 0.02)  # Heuristic smoothing applies the tightest scale globally

    # Star B (Index 1) should reflect explicit overrides
    assert np.isclose(p.initval[1], 0.85)
    assert np.isclose(p.sigma[1], 0.02)
    assert np.isclose(p.init_scale[1], 0.02)


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
    # Resolve the dictionary directly to bypass the heavy system-graph dependencies
    resolved = cm.resolve("planet", "mass", shape=(1,), names=["b"])
    resolved.pop("expressions", None)

    # Initialize the parameter with the resolved dictionary
    p = Parameter(label="planet.b.mass", **resolved)

    internal_val = p.initval[0]
    user_val = p.from_internal(internal_val)[0]

    # ASSERT
    # 1. Did the ConfigManager actually pass the string override?
    assert p.unit[0] == u.earthMass, f"Unit override failed! Expected earthMass, got {p.unit[0]}"

    # 2. Did the gatekeeper correctly convert 1.0 Earth mass -> ~3.00273e-6 Solar masses?
    assert np.isclose(internal_val, 3.00273e-6, rtol=1e-3), "Conversion to internal unit failed!"

    # 3. Does the table/plotter correctly convert the internal Solar mass back to 1.0 Earth mass?
    assert np.isclose(user_val, 1.0), "Roundtrip conversion back to user unit failed!"


def test_unit_override_scales_default_values_to_new_units():
    """
    Given a parameter with a default of 1.0 JupiterMass,
    When a user overrides the unit to 'earthMass' without overriding the initval,
    Then the initval should be automatically scaled to ~317.8 EarthMasses.
    NOTE: data-driven initial values will be properly translated to the user's desired unit
          the underlying code must initialize the value in the default user units.
    """
    # ARRANGE
    # We only override the unit, leaving initval to come from defaults.yaml (which is 1.0)
    user_params = {"planet.b.mass": {"unit": "earthMass"}}
    cm = ConfigManager(user_params)

    # ACT
    resolved = cm.resolve("planet", "mass", shape=(1,), names=["b"])

    # Remove the physics registry dict so Parameter doesn't complain
    resolved.pop("expressions", None)

    p = Parameter(label="planet.b.mass", **resolved)

    # ASSERT
    # 1. Check unit was captured
    assert p.unit[0] == u.earthMass

    # 2. Check that the initval (now in Earth units) is ~317.8
    # (Since 1.0 Jupiter Mass = 317.8 Earth Masses)
    user_val = p.from_internal(p.initval)[0]
    assert np.isclose(user_val, 317.8, rtol=1e-2)

    # 3. Check that physics hasn't changed (Internal Solar Mass should be same as 1.0 Jupiter)
    assert np.isclose(p.initval[0], 0.000954, rtol=1e-3)


def test_config_manager_scales_arbitrary_units_generically():
    """
    Given a parameter with a default unit of 'm' (meters),
    When a user overrides the unit to 'km' (kilometers) and a heuristic provides a value in 'm',
    Then the ConfigManager should scale the heuristic value by 0.001.
    This test creates a fictitious parameter to make sure we're not cheating
    """
    # 1. ARRANGE
    # Simulate a default config for a fictitious length parameter
    user_params = {"star.A.arm_length": {"unit": "km"}}
    cm = ConfigManager(user_params)

    # Manually inject a default into the manager's base_defaults for testing
    cm.base_defaults["arm_length"] = {
        "unit": "m",
        "initval": 1000.0,
        "internal_unit": "m"
    }

    # 2. ACT
    # Heuristic provides 5000.0 meters
    resolved = cm.resolve("star", "arm_length", shape=(1,),
                          internal_overrides={"initval": [5000.0]}, names=["A"])

    # 3. ASSERT
    # The resolved value should be 5.0 (5000m scaled to km)
    assert np.isclose(resolved["initval"][0], 5.0)
    assert resolved["unit"] == "km"