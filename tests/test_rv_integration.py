import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from astropy import units as u
from exozippy.components.orbit.orbit import Orbit
from exozippy.config import ConfigManager
from exozippy.components.parameter import Parameter
from exozippy.system import System

def test_radial_velocity_handles_descending_node_and_eccentricity():
    """
    Given an Orbit model with randomized eccentricities and omegas,
    When the radial velocity is evaluated at a specific descending node time,
    Then the RV should perfectly match the expected analytical semi-amplitude.
    """
    # ARRANGE
    np.random.seed(42)
    P_days, Tc, K_user = 10.0, 2450000.0, 100.0
    n = 2.0 * np.pi / P_days
    omegas = np.concatenate([np.linspace(0, 2 * np.pi, 9)[:-1], np.random.uniform(0, 2 * np.pi, 5)])

    for i, omega in enumerate(omegas):
        ecc = 0.2 if i < 8 else np.random.uniform(0.1, 0.8)
        user_params = {
            "orbit.0.logP": {"initval": np.log10(P_days)},
            "orbit.0.tc": {"initval": Tc},
            "orbit.0.secosw": {"initval": np.sqrt(ecc) * np.cos(omega)},
            "orbit.0.sesinw": {"initval": np.sqrt(ecc) * np.sin(omega)}
        }
        config = {"orbit": [{"name": "test"}]}
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
        orbit = system.orbit

        with model:
            tp_fn = pytensor.function(model.free_RVs, orbit.tp.value, on_unused_input='ignore')
            t_var = pt.vector("t")
            K_var = pt.vector("K_int")
            rv_node = orbit.get_radial_velocity(t_var, K_var, pt.as_tensor_variable([0], dtype="int32"))
            rv_fn = pytensor.function(model.free_RVs + [t_var, K_var], rv_node, on_unused_input='ignore')

            init_point = model.initial_point()
            p_vals = [np.zeros_like(init_point[v.name]) for v in model.free_RVs]

            # ACT
            Tp_val = tp_fn(*p_vals)
            f_D = -omega
            E_D = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(f_D / 2), np.sqrt(1.0 + ecc) * np.cos(f_D / 2))
            t_D = Tp_val + (E_D - ecc * np.sin(E_D)) / n

            k_param = Parameter(label="K", unit="m/s", internal_unit="solRad/d", initval=K_user)
            rv_internal = rv_fn(*p_vals, np.array([t_D[0]], dtype="float64"), k_param.initval.astype("float64"))
            rv_ms = k_param.from_internal(rv_internal).flatten()[0]

        # ASSERT
        expected_rv = K_user * (1.0 + ecc * np.cos(omega))
        np.testing.assert_allclose(rv_ms, expected_rv, rtol=1e-5)


def test_radial_velocity_matches_kelt4_periastron_benchmark():
    """
    Given a circular orbit matching KELT-4Ab,
    When the RV is evaluated exactly at periastron (Tc - P/4),
    Then the output should exactly match the analytical K semi-amplitude.
    """
    # ARRANGE
    user_params = {
        "orbit.0.logP": {"initval": 0.4756},
        "orbit.0.tc": {"initval": 2456190.302},
        "orbit.0.secosw": {"initval": 0.0},
        "orbit.0.sesinw": {"initval": 0.0}
    }
    config = {"orbit": [{"name": "b"}]}
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    orbit = system.orbit

    with model:
        t_var = pt.vector("t")
        K_var = pt.vector("K_int")
        rv_node = orbit.get_radial_velocity(t_var, K_var, pt.as_tensor_variable([0], dtype="int32"))
        rv_fn = pytensor.function(model.free_RVs + [t_var, K_var], rv_node, on_unused_input='ignore')

        k_user_ms = 108.6
        k_int = k_user_ms * orbit.config_manager.get_conversion_factor("planet", "K")

        init_point = model.initial_point()
        p_vals = [np.zeros_like(np.atleast_1d(init_point[v.name])).astype("float64") for v in model.free_RVs]

        P = 10 ** 0.4756
        tp_eval = 2456190.302 - P / 4.0

        # ACT
        rv_internal = rv_fn(*p_vals, np.array([tp_eval], dtype="float64"), np.array([k_int], dtype="float64"))
        k_param = Parameter(label="K", unit="m/s", internal_unit="solRad/d")
        rv_final_ms = k_param.from_internal(rv_internal).flatten()[0]

    # ASSERT
    np.testing.assert_allclose(rv_final_ms, k_user_ms, rtol=1e-5)


def test_radial_velocity_matches_earth_periastron_benchmark():
    """
    Given a circular orbit matching Earth's parameters,
    When the RV is evaluated exactly at periastron,
    Then the output should match Earth's known reflex velocity on the Sun.
    """
    # ARRANGE
    user_params = {
        "orbit.0.logP": {"initval": np.log10(365.25)},
        "orbit.0.tc": {"initval": 2450000.0},
        "orbit.0.secosw": {"initval": 0.0},
        "orbit.0.sesinw": {"initval": 0.0}
    }
    # ARRANGE
    config = {"orbit": [{"name": "Earth"}]}
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    orbit = system.orbit

    with model:
        k_ms = 0.089438027
        k_int = (k_ms * u.m / u.s).to(u.R_sun / u.d).value

        t_var = pt.vector("t")
        K_var = pt.vector("K_int")
        rv_node = orbit.get_radial_velocity(t_var, K_var, pt.as_tensor_variable([0], dtype="int32"))
        rv_fn = pytensor.function(model.free_RVs + [t_var, K_var], rv_node, on_unused_input='ignore')

        init_point = model.initial_point()
        p_vals = [np.zeros_like(np.atleast_1d(init_point[v.name])).astype("float64") for v in model.free_RVs]

        P = 365.25
        tp_eval = 2450000.0 - P / 4.0

        # ACT
        rv_internal = rv_fn(*p_vals, [tp_eval], [k_int])
        rv_val_ms = (rv_internal[0] * u.R_sun / u.d).to(u.m / u.s).value

    # ASSERT
    np.testing.assert_allclose(rv_val_ms, k_ms, rtol=1e-6)


def test_radial_velocity_computes_correctly_over_vectorized_time_array():
    """
    Given an initialized PyMC model with a single circular orbit,
    When the radial velocity function is evaluated across an array of time steps,
    Then it should correctly compute the corresponding RV curve in internal units.
    """
    # ARRANGE
    user_params = {
        "orbit.0.logP": {"initval": np.log10(10.0)},
        "orbit.0.tc": {"initval": 0.0},
        "orbit.0.secosw": {"initval": 0.0},
        "orbit.0.sesinw": {"initval": 0.0}
    }
    config = {"orbit": [{"name": "b"}]}
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    orbit = system.orbit

    with model:

        t = np.array([0.0, 2.5, 5.0, 7.5])
        t_pt = pt.vector("t")
        K_pt = pt.vector("K")
        ndx_pt = pt.ivector("ndx")

        rv_node = orbit.get_radial_velocity(t_pt, K_pt, ndx_pt)
        rv_fn = pytensor.function(model.free_RVs + [t_pt, K_pt, ndx_pt], rv_node, on_unused_input='ignore')

        init_point = model.initial_point()
        p_vals = [np.zeros_like(init_point[v.name]).astype("float64") for v in model.free_RVs]

        k_ms = 10.0
        m_to_int = (u.m / u.s).to(u.R_sun / u.d)
        int_to_m = (u.R_sun / u.d).to(u.m / u.s)

        # ACT
        rv_val = rv_fn(*p_vals, t, np.array([k_ms * m_to_int]), np.array([0], dtype="int32")) * int_to_m

    # ASSERT
    expected = np.array([0.0, -10.0, 0.0, 10.0])
    np.testing.assert_allclose(rv_val.flatten(), expected, atol=1e-7)