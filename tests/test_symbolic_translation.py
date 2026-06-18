import pytest
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from exozippy.components.parameter import Parameter


def get_val(node, inputs_dict=None):
    """
    Helper to quickly evaluate a PyTensor node, bypassing PyMC's
    unpredictable initial_point dictionaries.
    """
    if inputs_dict is None:
        inputs_dict = {}

    keys = list(inputs_dict.keys())
    values = list(inputs_dict.values())

    # Safely compile the isolated PyTensor path
    fn = pytensor.function(keys, node, on_unused_input='ignore')
    return fn(*values)


def test_derived_prior_numerical_evaluation():
    """
    Tests the mathematical accuracy of the applied prior using pure PyTensor evaluation.
    Proves that the penalty is calculated in the PHYSICAL space, not the log space.
    """
    with pm.Model() as model:
        logmass = Parameter(label="logmass", initval=0.0, init_scale=0.1, lower=-2.0, upper=2.0)
        logmass_node = logmass.build_pymc()

        mass = Parameter(
            label="mass",
            initval=1.2,  # Prior mean
            sigma=0.05,  # Prior width
            expression=lambda: pt.pow(10.0, logmass_node)
        )
        mass.build_pymc()

    mass_potential = model['gaussian_prior.mass']

    # Pass a pure scalar (0.0) instead of a 1D array ([0.0])
    val = get_val(mass_potential, {logmass_node: np.array(0.0)})

    # MATH CHECK:
    # Mass evaluates to 10^0.0 = 1.0
    # Prior is N(mu=1.2, sigma=0.05)
    # Penalty = -0.5 * ((1.0 - 1.2) / 0.05)^2 = -8.0
    np.testing.assert_allclose(val, -8.0, atol=1e-5, err_msg="Derived prior calculated incorrect penalty!")


def test_sampled_parameter_fallback_scaling():
    """
    Tests the distinction between `sigma` (scientific) and `init_scale` (sampling estimate).
    - Sampled params with `sigma` + bounds: logit transform (hard bounds) with
      the Gaussian prior applied on the physical value (truncated normal);
      sigma also sets the whitening scale.
    - Sampled params with only `init_scale` + bounds: logit transform with a
      flat prior; init_scale affects whitening only, never the posterior.
    """
    with pm.Model() as model:
        logP = Parameter(label="logP", initval=0.5, init_scale=0.01, lower=-1.0, upper=2.0)
        logP.build_pymc()

        ecc = Parameter(label="ecc", initval=0.05, mu=0.0, sigma=0.05, lower=0.0, upper=1.0)
        ecc.build_pymc()

    potential_names = [p.name for p in model.potentials]
    # sigma on a bounded sampled param is a Gaussian potential on the physical
    # value (the raw N(0,1) is cancelled by the flat-prior correction)
    assert "gaussian_prior.ecc" in potential_names, "bounded sampled sigma must add a Gaussian potential"
    assert "gaussian_prior.logP" not in potential_names, "init_scale should never create a Gaussian prior!"

    logp_fn = model.compile_logp()
    logp_center = logp_fn({"ecc_raw": np.array([0.0]), "logP_raw": np.array([0.0])})
    assert np.isfinite(logp_center), "logp should be finite at raw=0"

    # The Gaussian prior penalizes deviations in physical space:
    # raw=3 moves ecc from 0.05 to ~0.55, ~11 sigma from mu=0.
    logp_3sigma = logp_fn({"ecc_raw": np.array([3.0]), "logP_raw": np.array([0.0])})
    assert logp_3sigma < logp_center - 3.0, "large deviation should be penalized"