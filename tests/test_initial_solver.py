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

@pytest.mark.parametrize("omega_deg", [12.8, 95.0, 170.0, -120.0, -30.0])
def test_ecc_omega_initvals_seed_correct_secosw_branch(omega_deg):
    """
    Given: user initvals for ecc and omega (all four quadrants)
    When: the relaxation engine solves the orbit parameterization bridges
    Then: secosw/sesinw land on the branch matching the user's omega
          (regression: an unmapped 'omega' symbol used to leave the solver
          with only ecc = secosw^2 + sesinw^2 and a sign ambiguity,
          producing omega ~ 180 deg regardless of the input)
    """
    from exozippy.config import ConfigManager

    ecc = 0.451
    user_params = {
        "orbit.0.logP": {"initval": np.log10(185.6)},
        "orbit.0.tc": {"initval": 2457404.32},
        "orbit.0.ecc": {"initval": ecc},
        "orbit.0.omega": {"initval": omega_deg},
    }
    cm = ConfigManager(user_params, system_config={"orbit": [{"name": "b"}]})
    cm.finalize_user_params()

    sc = cm.resolve("orbit", "secosw", shape=(1,), names=["b"])["initval"][0]
    ss = cm.resolve("orbit", "sesinw", shape=(1,), names=["b"])["initval"][0]

    w = np.radians(omega_deg)
    assert sc == pytest.approx(np.sqrt(ecc) * np.cos(w), abs=1e-4)
    assert ss == pytest.approx(np.sqrt(ecc) * np.sin(w), abs=1e-4)
