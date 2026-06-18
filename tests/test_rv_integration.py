import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from astropy import units as u
from exozippy.components.orbit.orbit import Orbit
from exozippy.config import ConfigManager
from exozippy.components.parameter import Parameter
from exozippy.system import System

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Parametrized (omega, ecc) grid — generated once at import time
# ---------------------------------------------------------------------------

_P_DAYS = 10.0
_TC = 2450000.0
_K_USER_MS = 100.0
_N = 2.0 * np.pi / _P_DAYS

np.random.seed(42)
_omegas = np.concatenate([np.linspace(0, 2 * np.pi, 9)[:-1],
                          np.random.uniform(0, 2 * np.pi, 5)])
_eccs   = [0.2 if i < 8 else float(np.random.uniform(0.1, 0.8))
           for i in range(len(_omegas))]
_CASES  = list(zip(_omegas, _eccs))


def _case_id(params):
    omega, ecc = params
    return f"w={omega:.3f}_e={ecc:.3f}"


@pytest.fixture(scope="module")
def compiled_rv_functions():
    """Compile tp/rv pytensor functions once for all (omega, ecc) cases."""
    user_params = {
        "orbit.0.logP":   {"initval": np.log10(_P_DAYS)},
        "orbit.0.tc":     {"initval": _TC},
        "orbit.0.secosw": {"initval": 0.0},
        "orbit.0.sesinw": {"initval": 0.0},
    }
    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        cm = ConfigManager(user_params)
        orbit_comp = Orbit([{"name": "test"}], cm)

        with pm.Model():
            orbit_comp.register_parameters(system=None)
            for param_name in orbit_comp.manifest:
                orbit_comp.add_parameter(model=pm.modelcontext(None),
                                         param_name=param_name, system=None)

            t_var = pt.vector("t")
            K_var = pt.vector("K_int")
            rv_node = orbit_comp.get_radial_velocity(
                t_var, K_var, pt.as_tensor_variable([0], dtype="int32"))

            # Treat the physical parameter values as free symbolic inputs so
            # the same compiled function works for every (omega, ecc) pair.
            tp_fn = pytensor.function(
                inputs=[orbit_comp.logP.value, orbit_comp.tc.value,
                        orbit_comp.secosw.value, orbit_comp.sesinw.value],
                outputs=[orbit_comp.tp.value],
                on_unused_input="ignore",
            )
            rv_fn = pytensor.function(
                inputs=[orbit_comp.logP.value, orbit_comp.tc.value,
                        orbit_comp.secosw.value, orbit_comp.sesinw.value,
                        t_var, K_var],
                outputs=[rv_node],
                on_unused_input="ignore",
            )
    return tp_fn, rv_fn


@pytest.mark.parametrize("omega_ecc", _CASES, ids=[_case_id(c) for c in _CASES])
def test_radial_velocity_handles_descending_node_and_eccentricity(omega_ecc, compiled_rv_functions):
    """
    Given an Orbit model with specific eccentricity and argument of periastron,
    When the radial velocity is evaluated at the descending node time,
    Then the RV matches the analytical semi-amplitude K*(1 + e*cos(omega)).
    """
    omega, ecc = omega_ecc
    tp_fn, rv_fn = compiled_rv_functions

    logP_in   = np.array([np.log10(_P_DAYS)], dtype="float64")
    tc_in     = np.array([_TC],               dtype="float64")
    secosw_in = np.array([np.sqrt(ecc) * np.cos(omega)], dtype="float64")
    sesinw_in = np.array([np.sqrt(ecc) * np.sin(omega)], dtype="float64")

    k_param = Parameter(label="K", unit="m/s", internal_unit="solRad/d",
                        initval=_K_USER_MS)

    Tp_val = tp_fn(logP_in, tc_in, secosw_in, sesinw_in)
    f_D = -omega
    E_D = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(f_D / 2),
                            np.sqrt(1.0 + ecc) * np.cos(f_D / 2))
    t_D = float(np.ravel(Tp_val)[0]) + (E_D - ecc * np.sin(E_D)) / _N

    rv_internal = rv_fn(logP_in, tc_in, secosw_in, sesinw_in,
                        np.array([t_D], dtype="float64"),
                        k_param.initval.astype("float64"))
    rv_ms = k_param.from_internal(rv_internal).flatten()[0]

    expected_rv = _K_USER_MS * (1.0 + ecc * np.cos(omega))
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
