import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor
from exozippy.components.orbit import Orbit
from exozippy.config import ConfigManager
from exozippy.system import System

def test_isolated_orbital_mechanics_match_pure_numpy_sinusoid():
    """
    Given a 10-day circular orbit configuration,
    When the PyTensor radial velocity graph is compiled and evaluated,
    Then the output should perfectly match a pure numpy sinusoidal truth array.
    """
    # ARRANGE
    config = {"orbit": [{"name": "test_orbit"}]}
    user_params = {
        "orbit.test_orbit.logP": {"initval": np.log10(10.0)},
        "orbit.test_orbit.tc": {"initval": 0.0},
        "orbit.test_orbit.secosw": {"initval": 0.0},
        "orbit.test_orbit.sesinw": {"initval": 0.0}
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    orbit = system.orbit

    with model:

        times = np.linspace(0, 10, 100)
        t_tensor = pt.vector("t")
        k_tensor = pt.vector("k")
        orbit_map = pt.ivector("orbit_map")

        rv_node = orbit.get_radial_velocity(t_tensor, k_tensor, orbit_map)
        rv_fn = pytensor.function(model.free_RVs + [t_tensor, k_tensor, orbit_map], rv_node, on_unused_input='ignore')

        init_point = model.initial_point()
        p_vals = [np.zeros_like(init_point[v.name]).astype("float64") for v in model.free_RVs]

        # ACT
        rv_calculated = rv_fn(*p_vals, times, np.array([1.0]), np.array([0], dtype="int32")).flatten()

    # ASSERT
    rv_truth = -1.0 * np.sin(2 * np.pi * times / 10.0)
    np.testing.assert_allclose(rv_calculated, rv_truth, atol=1e-7)

if __name__ == "__main__":
    test_isolated_orbital_mechanics_match_pure_numpy_sinusoid()