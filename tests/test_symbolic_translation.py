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


def test_derived_parameter_prior_registration():
    """
    Ensures that if a derived parameter (with an expression) is given a sigma,
    it successfully registers a pm.Potential in the model.
    """
    with pm.Model() as model:
        logmass = Parameter(label="logmass", initval=0.0, init_scale=0.1, lower=-2.0, upper=2.0)
        logmass_node = logmass.build_pymc()

        mass = Parameter(
            label="mass",
            initval=1.2,
            sigma=0.05,
            expression=lambda: pt.pow(10.0, logmass_node)
        )
        mass.build_pymc()

    potential_names = [p.name for p in model.potentials]
    assert "gaussian_prior.mass" in potential_names, "Derived parameter prior was ignored!"


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
def test_derived_parameter_bounds_translation():
    """
    Tests that setting physical upper/lower bounds on a derived parameter
    translates to soft boundaries in the PyMC graph.
    """
    with pm.Model() as model:
        logP = Parameter(label="logP", initval=1.0, init_scale=0.1, lower=0.0, upper=2.0)
        logP_node = logP.build_pymc()

        period = Parameter(
            label="period",
            lower=5.0,
            upper=15.0,
            expression=lambda: pt.pow(10.0, logP_node)
        )
        period.build_pymc()

    potential_names = [p.name for p in model.potentials]
    assert "low_bound.period" in potential_names, "Lower bound on derived parameter ignored"
    assert "up_bound.period" in potential_names, "Upper bound on derived parameter ignored"


def test_sampled_parameter_fallback_scaling():
    """
    Tests the distinction between `sigma` (scientific) and `init_scale` (sampling estimate).
    If a sampled parameter has NO sigma, it should just be bounded without applying a Gaussian penalty.
    """
    with pm.Model() as model:
        logP = Parameter(label="logP", initval=0.5, init_scale=0.01, lower=-1.0, upper=2.0)
        logP.build_pymc()

        ecc = Parameter(label="ecc", initval=0.0, sigma=0.05, lower=0.0, upper=1.0)
        ecc.build_pymc()

    potential_names = [p.name for p in model.potentials]
    assert "gaussian_prior.ecc" in potential_names, "sigma failed to create a Gaussian prior!"
    assert "gaussian_prior.logP" not in potential_names, "init_scale leaked and created a false scientific prior!"