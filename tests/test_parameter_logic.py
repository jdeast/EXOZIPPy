import pytest
import numpy as np
import pymc as pm
from astropy import units as u
from exozippy.components.parameter import Parameter


def test_parameter_scaling_adapts_to_initialization_scenarios():
    """
    Given Parameters initialized with scaling only vs. explicit Gaussian priors,
    When the internal PyMC normal distributions are created,
    Then unconstrained parameters should use a flat (1000.0) raw sigma,
    while prior-constrained parameters use a unit (1.0) raw sigma.
    """
    # ARRANGE
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
            # ACT (internal helper): Back-calculate the implied sigma from the logp drop
            return np.sqrt(-0.5 / (logp_fn(p_step) - logp_fn(p_mu)))

        # ASSERT
        # Verify Scenario 1 & 2: Raw sigma is always 1000.0 (Unconstrained)
        assert np.isclose(get_raw_sigma("p1"), 1000.0)
        assert np.isclose(get_raw_sigma("p2"), 1000.0)
        # Verify Scenario 3: Raw sigma is 1.0 (Unit Normal Prior)
        assert np.isclose(get_raw_sigma("p3"), 1.0)


def test_parameter_unit_conversion_roundtrips_cleanly():
    """
    Given a Parameter defined with distinct user and internal units,
    When a value is passed into its `from_internal` method,
    Then it should perfectly reconstruct the original user-unit value.
    """
    # ARRANGE
    p = Parameter(label="m", unit=u.jupiterMass, internal_unit=u.solMass, initval=1.0)

    # ACT
    restored_val = p.from_internal(p.initval)

    # ASSERT
    assert np.isclose(restored_val, 1.0)


def test_out_of_bounds_parameter_applies_logp_penalty():
    """
    Given a PyMC model with a parameter bounded strictly below 10.0,
    When the model evaluates a parameter value slightly exceeding that bound,
    Then the soft boundary potential should heavily penalize the log-probability.
    """
    # ARRANGE
    with pm.Model() as model:
        p = Parameter(label="bounded_param", initval=5.0, init_scale=1.0, upper=10.0)
        p.build_pymc()

        logp_fn = model.compile_logp()

        point_inside = model.initial_point()
        point_inside["bounded_param_raw"] = np.array([0.0])  # Maps to 5.0

        point_outside = model.initial_point()
        point_outside["bounded_param_raw"] = np.array([5.1])  # Maps to 10.1

        # ACT
        logp_inside = logp_fn(point_inside)
        logp_outside = logp_fn(point_outside)

        # ASSERT
        assert logp_outside < logp_inside - 20.0, "Soft boundary did not apply a sufficient penalty!"