import numpy as np
import pymc as pm
import pytensor.tensor as pt
from exozippy.components.orbit import Orbit


def test_pure_rv_curve_evaluation():
    """
    Independent test of the Orbit.get_radial_velocity PyTensor graph.
    If this passes, the physics engine is 100% correct.
    """
    # 1. Setup minimal config
    config = [{"name": "test_orbit"}]
    user_params = {
        "orbit.test_orbit.logP": {"initval": np.log10(10.0)},  # 10 day period
        "orbit.test_orbit.tc": {"initval": 0.0},
        "orbit.test_orbit.secosw": {"initval": 0.0},  # e=0, w=0
        "orbit.test_orbit.sesinw": {"initval": 0.0}
    }

    orbit = Orbit(config, user_params)

    with pm.Model() as model:
        orbit.build_parameters(model)

        # Setup inputs
        times = np.linspace(0, 10, 100)  # 1 full period
        t_tensor = pt.as_tensor_variable(times)

        # Force K = 1.0 (R_sun / day)
        k_tensor = pt.as_tensor_variable([1.0])
        orbit_map = pt.as_tensor_variable([0])

        # 2. Get the PyTensor RV node
        rv_node = orbit.get_radial_velocity(t_tensor, k_tensor, orbit_map)

        # 3. Evaluate the graph numerically
        rv_calculated = rv_node.eval().flatten()

    # 4. Pure Python Truth
    # For e=0, w=0, tc=0: RV = K * cos(2*pi*t/P)
    # Wait: True anomaly f = 2*pi*(t-tp)/P.
    # Since w=0, Tp = Tc.
    rv_truth = 1.0 * np.cos(2 * np.pi * times / 10.0)

    print("\n--- Model Output (First 5 points) ---")
    print(rv_calculated[:5])
    print("--- Truth Output (First 5 points) ---")
    print(rv_truth[:5])

    # Assert they are identical
    np.testing.assert_allclose(rv_calculated, rv_truth, atol=1e-7)


if __name__ == "__main__":
    test_pure_rv_curve_evaluation()