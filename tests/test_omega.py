import numpy as np
import pytest
import os
import pytensor
import pymc as pm
from exozippy.components.orbit import Orbit

# FIX: Need ConfigManager for the new architecture
from exozippy.config import ConfigManager


def get_data_path():
    """Locates the IDL output file in the same directory as this test."""
    return os.path.join(os.path.dirname(__file__), 'tc2tp.txt')


def test_exozippy_tp_vs_idl():
    """Checks exozippy logic against IDL grid, reading period from each row."""
    data_path = get_data_path()
    if not os.path.exists(data_path):
        pytest.fail(f"IDL validation file not found at {data_path}")

    data = np.loadtxt(data_path)
    compiled_fns = {}

    for row in data:
        tc_val, period_val, e_val, w_val, tp_idl = row

        if period_val not in compiled_fns:
            config = [{"name": "test_planet"}]
            user_params = {
                "orbit.logP": {"initval": np.log10(period_val)},
                "orbit.0.logP": {"initval": np.log10(period_val)}
            }

            # FIX: Pass ConfigManager instead of raw dict
            cm = ConfigManager(user_params)
            orbit_comp = Orbit(config, cm)

            with pm.Model() as model:
                orbit_comp.build_parameters(model)

                compiled_fns[period_val] = pytensor.function(
                    inputs=[
                        orbit_comp.logP.value,
                        orbit_comp.tc.value,
                        orbit_comp.secosw.value,
                        orbit_comp.sesinw.value
                    ],
                    outputs=[orbit_comp.tp.value],
                    on_unused_input='ignore'
                )

        logP_in = np.array([np.log10(period_val)], dtype="float64")
        tc_in = np.array([tc_val], dtype="float64")
        secosw_in = np.array([np.sqrt(e_val) * np.cos(w_val)], dtype="float64")
        sesinw_in = np.array([np.sqrt(e_val) * np.sin(w_val)], dtype="float64")

        # FIX: Unpack a single output from the compiled function
        tp_python = compiled_fns[period_val](
            logP_in,
            tc_in,
            secosw_in,
            sesinw_in
        )[0]

        diff = tp_python - tp_idl
        remainder = diff % period_val

        tol = 1e-7
        success = np.isclose(remainder, 0, atol=tol) or \
                  np.isclose(remainder, period_val, atol=tol)

        assert success, (
            f"Math Mismatch!\n"
            f"Inputs: e={e_val}, omega={w_val}, P={period_val}\n"
            f"IDL Tp: {tp_idl}\n"
            f"Python Tp: {tp_python[0]}\n"
            f"Diff: {tp_python[0] - tp_idl}\n"
            f"Diff/Period: {(tp_python[0] - tp_idl) / period_val}"
        )