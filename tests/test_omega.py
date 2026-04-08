import numpy as np
import pytest
import os
import pytensor
import pymc as pm
from exozippy.components.orbit import Orbit


def get_data_path():
    """Locates the IDL output file in the same directory as this test."""
    return os.path.join(os.path.dirname(__file__), 'tc2tp.txt')


def test_exozippy_tp_vs_idl():
    """Checks exozippy logic against IDL grid, reading period from each row."""
    data_path = get_data_path()
    if not os.path.exists(data_path):
        pytest.fail(f"IDL validation file not found at {data_path}")

    # Load IDL data: tc, period, e, omega, tp_idl
    data = np.loadtxt(data_path)

    # We'll cache compiled functions by period to keep the test fast
    compiled_fns = {}

    for row in data:
        tc_val, period_val, e_val, w_val, tp_idl = row

        # 1. Get or create the compiled function for this specific period
        if period_val not in compiled_fns:
            config = [{"name": "test_planet"}]
            # Set the period in user_params so build_parameters can see it
            user_params = {
                "orbit.logP": {"initval": np.log10(period_val)},
                "orbit.0.logP": {"initval": np.log10(period_val)}
            }

            orbit_comp = Orbit(config, user_params)

            with pm.Model() as model:
                orbit_comp.build_parameters(model)

                # Compile the .tp.value Deterministic node
                compiled_fns[period_val] = pytensor.function(
                    inputs=[
                        orbit_comp.logP.value,
                        orbit_comp.tc.value,
                        orbit_comp.secosw.value,
                        orbit_comp.sesinw.value
                    ],
                    outputs=[orbit_comp.tp.value, orbit_comp.debug_E0, orbit_comp.debug_M0],                    on_unused_input='ignore'
                )

        # 2. Map IDL inputs to exozippy's sampling space
        # Based on your trace:
        # Args 0, 2, and 3 are Vectors (shape (1,))
        # Arg 1 is a Scalar (shape ())

        logP_in = np.array([np.log10(period_val)], dtype="float64")  # Shape (1,)
        tc_in = np.float64(tc_val)  # Shape () - NO array wrapping
        secosw_in = np.array([np.sqrt(e_val) * np.cos(w_val)], dtype="float64")
        sesinw_in = np.array([np.sqrt(e_val) * np.sin(w_val)], dtype="float64")

        # 3. Execute the logic
        tp_python, e0_py, m0_py = compiled_fns[period_val](
            logP_in,
            tc_in,
            secosw_in,
            sesinw_in
        )

        # 4. Verification with periodic wrap-around
        diff = tp_python - tp_idl
        remainder = diff % period_val

        tol = 1e-7
        success = np.isclose(remainder, 0, atol=tol) or \
                  np.isclose(remainder, period_val, atol=tol)

        assert success, (
            f"Math Mismatch!\n"
            f"Inputs: e={e_val}, omega={w_val}, P={period_val}\n"
            f"IDL Tp: {tp_idl}\n"
            f"Python Tp: {tp_python[0]}\n"  # [0] because tp_python is a 1-element array
            f"Diff: {tp_python[0] - tp_idl}\n"
            f"Diff/Period: {(tp_python[0] - tp_idl) / period_val}"
        )