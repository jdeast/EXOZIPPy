import pytest
import numpy as np
import pymc as pm
import pytensor
from astropy import units as u
from exozippy.components.parameter import Parameter


def test_parameter_sampling_scenarios():
    with pm.Model() as model:
        p1 = Parameter(label="p1", initval=10.0, init_scale=2.0)
        p1.build_pymc()
        p2 = Parameter(label="p2", initval=10.0, init_scale=1.0)
        p2.build_pymc()
        p3 = Parameter(label="p3", initval=10.0, mu=10.0, sigma=0.5)
        p3.build_pymc()

        logp_fn = model.compile_logp()

        def get_raw_sigma(var_label):
            p_mu = model.initial_point()
            for k in p_mu: p_mu[k] = np.zeros_like(p_mu[k])
            p_step = p_mu.copy()
            p_step[f"{var_label}_raw"] = np.array([1.0])
            return np.sqrt(-0.5 / (logp_fn(p_step) - logp_fn(p_mu)))

        # Verify Scenario 1 & 2: Raw sigma is always 1000.0 (Unconstrained)
        assert np.isclose(get_raw_sigma("p1"), 1000.0)
        assert np.isclose(get_raw_sigma("p2"), 1000.0)
        # Verify Scenario 3: Raw sigma is 1.0 (Unit Normal Prior)
        assert np.isclose(get_raw_sigma("p3"), 1.0)


def test_unit_roundtrip():
    p = Parameter(label="m", unit=u.jupiterMass, internal_unit=u.solMass, initval=1.0)
    assert np.isclose(p.from_internal(p.initval), 1.0)