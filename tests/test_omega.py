import numpy as np
import pytest
import os
import pytensor
import pymc as pm
from exozippy.components.orbit import Orbit
from exozippy.config import ConfigManager


def get_data_path():
    """Locates the IDL output file in the same directory as this test."""
    return os.path.join(os.path.dirname(__file__), 'tc2tp.txt')


def test_time_of_periastron_matches_idl_benchmark_grid():
    """
    Given a grid of period, eccentricity, and omega values from a legacy IDL model,
    When PyTensor compiles and evaluates the Time of Periastron (Tp) conversion,
    Then the calculated Tp should match the IDL outputs modulo the orbital period.
    """
    # ARRANGE: Load data
    data_path = get_data_path()
    if not os.path.exists(data_path):
        pytest.fail(f"IDL validation file not found at {data_path}")

    data = np.loadtxt(data_path)

    # Wrap the PyTensor build and execution in FAST_COMPILE mode
    with pytensor.config.change_flags(mode="FAST_COMPILE"):

        # ARRANGE: Build Graph & Compile EXACTLY ONCE
        config = [{"name": "test_planet"}]
        # Initvals here do not matter; the compiled function inputs will override them
        user_params = {
            "orbit.logP": {"initval": 1.0},
            "orbit.0.logP": {"initval": 1.0}
        }

        cm = ConfigManager(user_params)
        orbit_comp = Orbit(config, cm)

        with pm.Model() as model:
            orbit_comp.register_parameters(system=None)
            for param_name in orbit_comp.manifest:
                orbit_comp.add_parameter(model=model, param_name=param_name, system=None)

            # The heavy lifting happens here, just once
            calc_tp = pytensor.function(
                inputs=[
                    orbit_comp.logP.value,
                    orbit_comp.tc.value,
                    orbit_comp.secosw.value,
                    orbit_comp.sesinw.value
                ],
                outputs=[orbit_comp.tp.value],
                on_unused_input='ignore'
            )

        # ACT & ASSERT: Loop through data and execute the fast compiled function
        for row in data:
            tc_val, period_val, e_val, w_val, tp_idl = row

            logP_in = np.array([np.log10(period_val)], dtype="float64")
            tc_in = np.array([tc_val], dtype="float64")
            secosw_in = np.array([np.sqrt(e_val) * np.cos(w_val)], dtype="float64")
            sesinw_in = np.array([np.sqrt(e_val) * np.sin(w_val)], dtype="float64")

            # Execute the compiled function with the new inputs
            tp_python = calc_tp(
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