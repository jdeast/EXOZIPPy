import numpy as np
import pytest
from exozippy.components.planet.symbolic_physics import solve_companion_mass

def test_earth_mass():
    # Inputs
    K = 1.14256145e-5  # radSol/day
    e = 0.0
    sini = 1.0
    P = 365.25
    m1 = 1.0

    expected_mass = 3.0888219e-6  # M_sol

    result = solve_companion_mass(K, e, sini, P, m1)

    # Assert tolerance of 1%
    assert result == pytest.approx(expected_mass, rel=1e-2)
    print(f"Test Passed: Companion mass = {result:.4e} M_sol")