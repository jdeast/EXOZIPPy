import numpy as np
import pytest
import os
import pytensor
import pymc as pm
from exozippy.components.orbit import Orbit
from exozippy.config import ConfigManager

pytestmark = pytest.mark.slow


_DATA_PATH = os.path.join(os.path.dirname(__file__), 'tc2tp.txt')

_IDL_ROWS = (
    [tuple(row) for row in np.loadtxt(_DATA_PATH)]
    if os.path.exists(_DATA_PATH)
    else []
)


def _row_id(row):
    _tc, _period, e, omega, _tp = row
    return f"e={e:.2f}_w={omega:.4f}"


@pytest.fixture(scope="module")
def compiled_tp_function():
    """Build and compile the PyTensor Tp function once for the whole module."""
    if not os.path.exists(_DATA_PATH):
        pytest.skip(f"IDL validation file not found at {_DATA_PATH}")

    config = [{"name": "test_planet"}]
    user_params = {
        "orbit.logP": {"initval": 1.0},
        "orbit.0.logP": {"initval": 1.0},
    }

    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        cm = ConfigManager(user_params)
        orbit_comp = Orbit(config, cm)

        with pm.Model() as model:
            orbit_comp.register_parameters(system=None)
            for param_name in orbit_comp.manifest:
                orbit_comp.add_parameter(model=model, param_name=param_name, system=None)

            calc_tp = pytensor.function(
                inputs=[
                    orbit_comp.logP.value,
                    orbit_comp.tc.value,
                    orbit_comp.secosw.value,
                    orbit_comp.sesinw.value,
                ],
                outputs=[orbit_comp.tp.value],
                on_unused_input="ignore",
            )

    return calc_tp


@pytest.mark.parametrize("row", _IDL_ROWS, ids=[_row_id(r) for r in _IDL_ROWS])
def test_time_of_periastron_matches_idl_benchmark_grid(row, compiled_tp_function):
    """
    Given a (tc, period, e, omega, tp_idl) row from the IDL benchmark grid,
    When PyTensor evaluates the Tp conversion,
    Then Tp matches the IDL output modulo the orbital period.
    """
    tc_val, period_val, e_val, w_val, tp_idl = row

    logP_in    = np.array([np.log10(period_val)],           dtype="float64")
    tc_in      = np.array([tc_val],                         dtype="float64")
    secosw_in  = np.array([np.sqrt(e_val) * np.cos(w_val)], dtype="float64")
    sesinw_in  = np.array([np.sqrt(e_val) * np.sin(w_val)], dtype="float64")

    tp_python = compiled_tp_function(logP_in, tc_in, secosw_in, sesinw_in)[0]

    diff = tp_python - tp_idl
    remainder = diff % period_val
    tol = 1e-7
    assert (
        np.isclose(remainder, 0, atol=tol)
        or np.isclose(remainder, period_val, atol=tol)
    ), (
        f"e={e_val}, omega={w_val}, P={period_val}\n"
        f"IDL Tp={tp_idl}  Python Tp={tp_python[0]}\n"
        f"Δ={tp_python[0] - tp_idl}  Δ/P={(tp_python[0] - tp_idl) / period_val}"
    )
