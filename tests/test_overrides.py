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


def test_instrument_name_override():
    user_params = {"inst.HIRES.gamma": {"initval": 123.45}}
    config_manager = ConfigManager(user_params)
    inst = RVInstrument([{"name": "HIRES"}], config_manager)
    inst.name = "HIRES"
    inst.prefix = "inst"

    with pm.Model(name="test_inst"):
        inst.build_pars_from_dict({"gamma": {"initval": [0.0]}}, shape=(1,), prefix="inst")

    expected = 123.45 / inst.gamma._get_conversion_factors()[0]
    assert np.isclose(inst.gamma.initval[0], expected)


def test_gaussian_prior_scale_override():
    label = "star.A.radius_test3"
    user_params = {label: {"initval": 1.0, "sigma": 0.05, "init_scale": 0.00085}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)
    star.name = "A"
    star.prefix = "star"

    with pm.Model(name="model_test3"):
        star.add_parameter("radius_test3", config_manager, prefix="star")

    assert np.isclose(star.radius_test3.init_scale[0], 0.00085)


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_ignored_yaml_warning(mock_logp, capsys):
    mock_logp.return_value = ({}, {})

    label = "star.A.mass_test4"
    user_params = {label: {"initval": 1.0, "sigm": 0.05}}
    system = MockSystem(user_params)
    star = Star([{"name": "A"}], system.config_manager)
    star.prefix = "star"
    system.star = star

    with pm.Model(name="model_test4") as model:
        star.add_parameter("mass_test4", system.config_manager, prefix="star")

    from exozippy.run import inspect_start
    inspect_start(model, system, {}, {}, {}, calc_curvature=False)
    out = capsys.readouterr().out
    assert "sigm" in out


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_unrecognized_parameter_warning(mock_logp, capsys):
    from exozippy.run import inspect_start
    mock_logp.return_value = ({}, {})

    user_params = {"star.A.radiuss": 1.0}
    system = MockSystem(user_params)
    star = Star([{"name": "A"}], system.config_manager)
    system.star = star

    with pm.Model(name="model_test5") as model:
        star.add_parameter("mass", system.config_manager, prefix="star")

    inspect_start(model, system, {}, {}, {}, calc_curvature=False)
    captured = capsys.readouterr()
    assert "star.A.radiuss" in captured.out


def test_boundary_clipping_logic():
    scenarios = [
        {"user": -10.0, "internal": 0.0, "expected": 0.0, "type": "lower"},
        {"user": 0.5, "internal": 0.0, "expected": 0.5, "type": "lower"},
    ]
    for s in scenarios:
        label = f"star.A.mass_{s['type']}"
        user_params = {label: {s['type']: s['user']}}
        config_manager = ConfigManager(user_params)

        star = Star([{"name": "A"}], config_manager)
        star.name = "A"
        star.prefix = "star"

        with pm.Model(name=f"model_{label.replace('.', '_')}"):
            star.add_parameter(label.split('.')[-1], config_manager, prefix="star", **{s['type']: s['internal']})

            p = getattr(star, label.split('.')[-1])
            val = p.lower[0] if s['type'] == 'lower' else p.upper[0]
            assert np.isclose(val, s['expected'])


def test_sampled_prior_and_default_scale():
    label = "star.A.mass_sampled"
    user_params = {label: {"mu": 1.1, "sigma": 0.1}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)
    star.name = "A"
    star.prefix = "star"

    with pm.Model(name="model_sampled"):
        star.add_parameter("mass_sampled", config_manager, prefix="star")

    p = star.mass_sampled
    assert p.mu is not None and p.mu[0] == 1.1
    assert p.sigma is not None and p.sigma[0] == 0.1


def test_derived_parameter_potential():
    user_params = {"orbit.b.period": {"mu": 10.0, "sigma": 0.01}}
    config_manager = ConfigManager(user_params)

    resolved = config_manager.resolve("orbit", "period", shape=(1,), names=["b"])
    resolved.pop("expressions", None)

    p = Parameter(
        label="orbit.period",
        is_derived=True,
        **resolved
    )
    assert p.mu is not None and p.mu[0] == 10.0


def test_mu_sigma_with_explicit_init_scale():
    label = "star.A.radius_custom"
    user_params = {label: {"mu": 1.0, "sigma": 0.05, "init_scale": 0.001}}
    config_manager = ConfigManager(user_params)

    star = Star([{"name": "A"}], config_manager)
    star.prefix = "star"

    with pm.Model(name="model_custom_scale"):
        star.add_parameter("radius_custom", config_manager, prefix="star")

    p = star.radius_custom
    assert p.mu is not None and np.isclose(p.mu[0], 1.0)
    assert np.isclose(p.init_scale[0], 0.001)


def test_vectorized_overrides():
    user_params = {
        "star.B.mass": {"initval": 0.85, "sigma": 0.02}
    }
    config_manager = ConfigManager(user_params)
    star = Star([{"name": "A"}, {"name": "B"}], config_manager)
    star.prefix = "star"

    with pm.Model(name="model_vector"):
        star.add_parameter("mass", config_manager, prefix="star", shape=(2,), initval=[1.0, 1.0])

    p = star.mass

    assert np.isclose(p.initval[0], 1.0)
    assert np.isnan(p.sigma[0])

    assert np.isclose(p.initval[1], 0.85)
    assert np.isclose(p.sigma[1], 0.02)
    assert np.isclose(p.init_scale[1], 0.02)
    # FIX: Heuristics apply the available sigma scale (0.02) uniformly to keep the mass matrix stable.
    assert np.isclose(p.init_scale[0], 0.02)


def test_config_manager_vector_resolution():
    user_params = {
        "orbit.b.logP": {"initval": 1.2, "sigma": 0.05},
        "orbit.c.logP": {"initval": 3.4}
    }
    cm = ConfigManager(user_params)
    resolved = cm.resolve("orbit", "logP", shape=(2,), internal_overrides={"initval": [0.0, 0.0], "init_scale": 1.0}, names=["b", "c"])

    assert np.allclose(resolved['initval'], [1.2, 3.4])
    assert np.allclose(resolved['init_scale'], [0.05, 1.0])
    assert resolved['mu'] is None


def test_direct_component_resolution_math_safety():
    cm = ConfigManager({})
    logP_cfg = cm.resolve("orbit", "logP", shape=(2,), names=["b", "c"])
    logP_init = logP_cfg['initval']

    assert logP_init[0] is not None
    assert not np.isnan(logP_init[0])


def test_system_initialization_safety():
    user_params = {
        "star.0.mass": {"initval": 0.98},
        "orbit.0.logP": {"initval": 2.0}
    }
    cm = ConfigManager(user_params)

    resolved_star = cm.resolve("star", "mass", shape=(1,), names=["A"])
    assert resolved_star["initval"][0] == 0.98
    resolved_rad = cm.resolve("star", "radius", shape=(1,), names=["A"])
    assert resolved_rad["initval"][0] == 1.0