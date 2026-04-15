import pytest
import numpy as np
import pymc as pm
from astropy import units as u

from exozippy.components.parameter import Parameter
from exozippy.config import ConfigManager
from exozippy.components.star.star import Star
from exozippy.diagnostics import ModelAuditor

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

def test_auditor_handles_partially_frozen_vector_parameters():
    """
    Given a vectorized parameter where one element is sampled and another is frozen,
    When the ModelAuditor extracts the sampler curvatures,
    Then it should correctly map the compressed PyMC array back to the original shape
    without throwing an IndexError, placing NaNs in the frozen slots.
    """

    # ARRANGE
    class MockSystem:
        """Minimal system mock to satisfy the Auditor's dependency injection."""

        def __init__(self, user_params):
            self.user_params = user_params
            self.config_manager = ConfigManager(user_params)

        def get_parameter_lookup(self):
            return {p.label: p for p in self.get_all_parameters()}

        def get_all_parameters(self):
            return [v for v in self.star.__dict__.values() if hasattr(v, 'build_pymc')]

    # Override the second star's mass to be completely frozen
    user_params = {
        "star.0.mass": {"initval": 1.0, "init_scale": 0.1},
        "star.1.mass": {"initval": 1.0, "init_scale": 0.0}  # Frozen!
    }

    system = MockSystem(user_params)
    star = Star([{"name": "0"}, {"name": "1"}], system.config_manager)
    system.star = star

    with pm.Model() as model:
        # Build the parameter. PyMC will detect the 0.0 scale and compress
        # 'star.mass_raw' to a shape of (1,) instead of (2,)
        star.add_parameter("mass", system.config_manager, shape=(2,))

    # Mimic the transformed initialization dict created by `system.get_mcmc_init()`
    transformed_inits = {"star.mass_raw": np.array([0.0])}

    auditor = ModelAuditor(model, system, transformed_inits)

    # ACT
    # This invokes PyTensor to calculate the gradient, and then expands the result
    curvatures = auditor.get_curvatures()

    # ASSERT
    assert "star.mass" in curvatures
    curv = curvatures["star.mass"]

    # 1. Did it successfully restore the original shape?
    assert len(curv) == 2, "Curvature array was not expanded to match the physical shape!"

    # 2. Does the sampled parameter have a real curvature?
    assert not np.isnan(curv[0]), "The sampled element's curvature was improperly wiped out!"

    # 3. Did the frozen parameter safely receive a NaN?
    assert np.isnan(curv[1]), "The frozen element did not receive a NaN padding!"


import pytensor.tensor as pt
from exozippy.config import ConfigManager
from exozippy.components.parameter import Parameter


def test_parameter_bypasses_float_conversion_for_tensor_expressions():
    """
    Given a derived parameter that is built dynamically with a PyTensor expression (like flux),
    When the Parameter class initializes,
    Then it should recognize the PyTensor object and bypass the NumPy float casting.
    """
    # ARRANGE
    mock_config = ConfigManager({})

    # Create a dummy PyTensor expression mimicking a derived formula
    teff = pt.dvector('teff')
    mock_expr = (teff / 5778.0) ** 4

    # ACT
    # If the bug is present, this will immediately raise:
    # TypeError: float() argument must be a string or a real number, not 'TensorVariable'
    # or ValueError: setting an array element with a sequence.
    param = Parameter(
        label="star.flux",
        shape=(1,),
        initval=mock_expr,
        expression=mock_expr,
        unit="",  # Keep units blank to avoid Astropy interference in this test
        internal_unit=""
    )

    # ASSERT
    # If we reach here, the code survived the __post_init__ crash point!
    # We also verify it successfully held onto the raw expression.
    assert param.initval == mock_expr, "The initval should be the identical PyTensor expression object!"
    assert hasattr(param.initval, 'owner'), "initval must retain its symbolic PyTensor properties"
    with pytest.raises(TypeError):
        float(param.initval)
    assert param.expression is mock_expr, "The PyTensor expression was mangled or lost!"


def test_parameter_builds_from_list_of_tensors():
    """
    Given a parameter whose expression is a list of independent PyTensor variables,
    When build_pymc is called,
    Then it should safely stack them into a single TensorVariable without an object dtype crash.
    """
    import pymc as pm
    mock_config = ConfigManager({})

    # Create two separate scalar tensor variables
    t1 = pt.dscalar('t1')
    t2 = pt.dscalar('t2')

    # The expression is a Python list containing the tensors
    expr_list = [t1, t2]

    param = Parameter(
        label="star.list_param",
        shape=(2,),
        expression=expr_list,
        unit="",
        internal_unit=""
    )

    with pm.Model() as model:
        # If the list-stacking bug is present, this will throw:
        # TypeError: Unsupported dtype for TensorType: object
        val = param.build_pymc()

    assert hasattr(val, 'type'), "Did not return a valid PyTensor variable!"
    assert val.type.ndim == 1, "The list should be stacked into a 1D vector!"

def test_parameter_strips_astropy_quantities_from_expressions():
    """
    Given a parameter whose derived expression returns an Astropy Quantity
    wrapping an array of PyTensor nodes,
    When build_pymc is called,
    Then it should strip the unit to prevent Astropy's .tolist() NotImplementedError.
    """
    import pymc as pm
    from exozippy.config import ConfigManager

    mock_config = ConfigManager({})
    t1 = pt.dscalar('t1_q')
    t2 = pt.dscalar('t2_q')

    # Mock an Astropy Quantity to bypass Astropy's strict object-array ban
    class MockQuantity:
        def __init__(self, value, unit):
            self.value = value
            self.unit = unit

    expr_quantity = MockQuantity(np.array([t1, t2], dtype=object), u.day)

    param = Parameter(
        label="star.quantity_param",
        shape=(2,),
        expression=expr_quantity,
        unit="",
        internal_unit=""
    )

    with pm.Model() as model:
        val = param.build_pymc()

    assert hasattr(val, 'type'), "Did not return a valid PyTensor variable!"
    assert not hasattr(val, 'unit'), "The Astropy unit was not successfully stripped!"


def test_generate_posterior_strips_quantities_before_walking():
    """
    Given a derived parameter whose expression is an Astropy Quantity,
    When generate_posterior is called,
    Then it should strip the unit so PyTensor's ancestors() can walk the graph.
    """
    mock_config = ConfigManager({})
    t_raw = pt.dvector('t_raw')

    # Simulate a Quantity-wrapped expression
    class MockQuantity:
        def __init__(self, value, unit):
            self.value = value
            self.unit = unit

    expr_q = MockQuantity(t_raw * 2.0, u.day)

    param = Parameter(
        label="star.test_post",
        expression=expr_q
    )

    # This bundle simulates what ArviZ provides after sampling
    bundle = {"t_raw": np.array([[1.0, 2.0, 3.0]])}

    # If the bug is present, this raises AttributeError: 'Quantity' has no 'owner'
    post = param.generate_posterior(bundle)

    assert post is not None
    assert np.allclose(post.ravel(), [2.0, 4.0, 6.0])


def test_derived_parameter_retains_numeric_initval():
    """
    Given a parameter (Period) derived from another (logP),
    When the parameter is built,
    Then it should have a valid numeric .initval (not None)
    so that downstream parameters (like Tc) can use it for heuristics.
    """
    from exozippy.components.parameter import Parameter
    import pytensor.tensor as pt
    import numpy as np

    # 1. Simulate the 'parent' (logP)
    logP_val = 0.477  # log10(3.0)
    logP_node = pt.dscalar('logP')

    # 2. Simulate the 'physics' function (10**logP)
    def calc_period(lp):
        return 10 ** lp

    # 3. Create the derived parameter
    # In the real app, add_parameter handles the math to generate this initval
    expected_init = calc_period(logP_val)

    period_param = Parameter(
        label="planet.period",
        expression=lambda: calc_period(logP_node),
        initval=expected_init,  # This must NOT be overwritten by None
        unit="d",
        internal_unit="d"
    )

    # 4. Verify the numeric initval survived
    assert period_param.initval is not None, "Derived parameter initval was incorrectly set to None!"
    assert np.isclose(period_param.initval, 2.9991625, atol=1e-5)

    # 5. Verify it can still build a PyMC node (the symbolic side works)
    import pymc as pm
    with pm.Model():
        node = period_param.build_pymc()
        assert hasattr(node, 'owner'), "PyMC node should be a symbolic expression"