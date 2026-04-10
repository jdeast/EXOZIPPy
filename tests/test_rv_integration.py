import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from astropy import units as u
from exozippy.components.orbit.orbit import Orbit
from exozippy.config import ConfigManager
from exozippy.components.parameter import Parameter


def test_orbit_rv_descending_node_comprehensive():
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
        orbit = Orbit([{"name": "test"}], ConfigManager(user_params))

        with pm.Model() as model:
            orbit.build_parameters(model)

            tp_fn = pytensor.function(model.free_RVs, orbit.tp.value, on_unused_input='ignore')

            t_var = pt.vector("t")
            K_var = pt.vector("K_int")
            rv_node = orbit.get_radial_velocity(t_var, K_var, pt.as_tensor_variable([0], dtype="int32"))
            rv_fn = pytensor.function(model.free_RVs + [t_var, K_var], rv_node, on_unused_input='ignore')

            # Use initial_point to avoid sniffing tags
            init_point = model.initial_point()
            p_vals = [np.zeros_like(init_point[v.name]) for v in model.free_RVs]

            Tp_val = tp_fn(*p_vals)
            f_D = -omega
            E_D = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(f_D / 2), np.sqrt(1.0 + ecc) * np.cos(f_D / 2))
            t_D = Tp_val + (E_D - ecc * np.sin(E_D)) / n

            k_param = Parameter(label="K", unit="m/s", internal_unit="solRad/d", initval=K_user)
            rv_internal = rv_fn(*p_vals, [t_D[0]], [k_param.initval])
            rv_ms = k_param.from_internal(rv_internal).flatten()[0]

        expected_rv = K_user * (1.0 + ecc * np.cos(omega))
        np.testing.assert_allclose(rv_ms, expected_rv, rtol=1e-5)


def test_orbit_rv_kelt4_benchmark():
    """
    Test KELT-4Ab: Circular orbit.
    Physics: RV = K * cos(f + w).
    At Tc (Conjunction), f + w = pi/2, so RV = 0.
    At f + w = 0 (Periastron), RV = K.
    """
    user_params = {
        "orbit.0.logP": {"initval": 0.4756},
        "orbit.0.tc": {"initval": 2456190.302},
        "orbit.0.secosw": {"initval": 0.0},
        "orbit.0.sesinw": {"initval": 0.0}
    }
    orbit = Orbit([{"name": "b"}], ConfigManager(user_params))

    with pm.Model() as model:
        orbit.build_parameters(model)

        t_var = pt.vector("t")
        K_var = pt.vector("K_int")
        rv_node = orbit.get_radial_velocity(t_var, K_var, pt.as_tensor_variable([0], dtype="int32"))
        rv_fn = pytensor.function(model.free_RVs + [t_var, K_var], rv_node, on_unused_input='ignore')

        k_user_ms = 108.6
        k_int = k_user_ms * orbit.config_manager.get_conversion_factor("planet", "K")

        # Force deterministic evaluation at initvals
        init_point = model.initial_point()
        p_vals = [np.zeros_like(np.atleast_1d(init_point[v.name])).astype("float64") for v in model.free_RVs]

        P = 10 ** 0.4756
        # EVALUATION POINT:
        # For a circular orbit, Tc = Tp + P/4.
        # We want to evaluate at Periastron (Tp) where RV should hit the peak K.
        tp_eval = 2456190.302 - P / 4.0

        # Ensure 1D inputs to avoid "Wrong number of dimensions" error
        t_input = np.array([tp_eval], dtype="float64")
        k_input = np.array([k_int], dtype="float64")

        rv_internal = rv_fn(*p_vals, t_input, k_input)

        k_param = Parameter(label="K", unit="m/s", internal_unit="solRad/d")
        rv_final_ms = k_param.from_internal(rv_internal).flatten()[0]

    # Verification: At periastron, RV should equal the semi-amplitude K
    assert np.isclose(rv_final_ms, k_user_ms, rtol=1e-5)


def test_orbit_rv_earth_benchmark():
    # Similar fix for Earth benchmark
    user_params = {
        "orbit.0.logP": {"initval": np.log10(365.25)},
        "orbit.0.tc": {"initval": 2450000.0},
        "orbit.0.secosw": {"initval": 0.0},
        "orbit.0.sesinw": {"initval": 0.0}
    }
    cfg = ConfigManager(user_params)
    orbit = Orbit([{"name": "Earth"}], cfg)

    with pm.Model() as model:
        orbit.build_parameters(model)
        k_ms = 0.089438027
        k_int = (k_ms * u.m / u.s).to(u.R_sun / u.d).value

        # Use the compiled function approach to avoid non-deterministic .eval()
        t_var = pt.vector("t")
        K_var = pt.vector("K_int")
        rv_node = orbit.get_radial_velocity(t_var, K_var, pt.as_tensor_variable([0], dtype="int32"))
        rv_fn = pytensor.function(model.free_RVs + [t_var, K_var], rv_node, on_unused_input='ignore')

        init_point = model.initial_point()
        p_vals = [np.zeros_like(np.atleast_1d(init_point[v.name])).astype("float64") for v in model.free_RVs]

        # Evaluate at Tp (Tc - P/4) to get the max RV = K
        P = 365.25
        tp_eval = 2450000.0 - P / 4.0
        rv_internal = rv_fn(*p_vals, [tp_eval], [k_int])

        rv_val_ms = (rv_internal[0] * u.R_sun / u.d).to(u.m / u.s).value

    np.testing.assert_allclose(rv_val_ms, k_ms, rtol=1e-6)

def test_orbit_rv_vectorization():
    # Mock Config
    user_params = {
        "orbit.0.logP": {"initval": np.log10(10.0)},  # 10 day period
        "orbit.0.tc": {"initval": 0.0},
        "orbit.0.secosw": {"initval": 0.0},  # Circular
        "orbit.0.sesinw": {"initval": 0.0}
    }
    cfg = ConfigManager(user_params)
    orbit = Orbit([{"name": "b"}], cfg)

    with pm.Model() as model:
        orbit.build_parameters(model)

        t = np.array([0.0, 2.5, 5.0, 7.5])
        t_pt = pt.as_tensor_variable(t)

        # 1. Convert 10 m/s to Internal Units (R_sun/day)
        k_ms = 10.0
        # Multiplier to go from m/s -> R_sun/day
        m_to_int = (u.m / u.s).to(u.R_sun / u.d)
        K = pt.as_tensor_variable([k_ms * m_to_int])

        ndx = pt.as_tensor_variable([0], dtype="int32")

        # 2. Call the physics
        rv_node = orbit.get_radial_velocity(t_pt, K, ndx)

        # 3. Convert the result BACK to m/s for easy comparison
        int_to_m = (u.R_sun / u.d).to(u.m / u.s)
        rv_val = rv_node.eval() * int_to_m

    expected = np.array([10.0, 0.0, -10.0, 0.0])
    np.testing.assert_allclose(rv_val.flatten(), expected, atol=1e-7)
