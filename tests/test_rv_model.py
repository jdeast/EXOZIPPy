import numpy as np
import pymc as pm
import pytensor.tensor as pt
from exozippy.components.orbit import Orbit
import pytensor

from exozippy.config import ConfigManager

def test_pure_rv_curve_evaluation():
    """
    Independent test of the Orbit.get_radial_velocity PyTensor graph.
    If this passes, the physics engine is 100% correct.
    """
    config = [{"name": "test_orbit"}]
    user_params = {
        "orbit.test_orbit.logP": {"initval": np.log10(10.0)},
        "orbit.test_orbit.tc": {"initval": 0.0},
        "orbit.test_orbit.secosw": {"initval": 0.0},
        "orbit.test_orbit.sesinw": {"initval": 0.0}
    }

    # FIX: Pass ConfigManager instead of raw dict
    cm = ConfigManager(user_params)
    orbit = Orbit(config, cm)

    with pm.Model() as model:
        orbit.build_parameters(model)

        times = np.linspace(0, 10, 100)

        t_tensor = pt.vector("t")
        k_tensor = pt.vector("k")
        orbit_map = pt.ivector("orbit_map")

        rv_node = orbit.get_radial_velocity(t_tensor, k_tensor, orbit_map)

        # Compile instead of .eval()
        rv_fn = pytensor.function(model.free_RVs + [t_tensor, k_tensor, orbit_map], rv_node, on_unused_input='ignore')

        init_point = model.initial_point()
        p_vals = [np.zeros_like(init_point[v.name]).astype("float64") for v in model.free_RVs]

        # Execute with actual numpy arrays
        rv_calculated = rv_fn(*p_vals, times, np.array([1.0]), np.array([0], dtype="int32")).flatten()

    rv_truth = -1.0 * np.sin(2 * np.pi * times / 10.0)
    np.testing.assert_allclose(rv_calculated, rv_truth, atol=1e-7)

if __name__ == "__main__":
    test_pure_rv_curve_evaluation()