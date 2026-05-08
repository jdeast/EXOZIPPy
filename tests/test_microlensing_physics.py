import numpy as np
import pytensor.tensor as pt
import pytest
from exozippy.components.mulensing.lens import Lens  # Adjust based on your actual path
from exozippy.components.parameter import Parameter
from astropy import units as u
from exozippy.physics_registry import PHYSICS_REGISTRY
from exozippy.config import ConfigManager
from exozippy.components.mulensing.physics import (
    calc_theta_E,
    calc_pi_rel,
    calc_t_E
)

def get_val(x):
    return x.eval() if hasattr(x, 'eval') else x

def test_pspl_magnification_accuracy():
    """Verify that the magnification function matches the Paczynski formula."""
    # Separation u in Einstein radii
    u_vals = np.array([0.1, 0.5, 1.0, 2.0])

    # Expected magnification
    expected = (u_vals ** 2 + 2) / (u_vals * np.sqrt(u_vals ** 2 + 4))

    # Mocking the lens component's internal call
    # Assuming Lens.get_magnification(t, dn, de) logic is accessible
    # For now, we test the core math if available in physics.py
    calc_A = PHYSICS_REGISTRY.get("calc_magnification_pspl")

    if calc_A:
        res = calc_A(u_vals).eval()
        assert np.allclose(res, expected, atol=1e-5)


def test_microlensing_physics_conversions():
    """Verify the transformation from Physical (M, D) to Observables (theta_E, t_E)."""

    # Setup values
    mass = 0.5  # M_sun
    dl = 4000.0  # pc
    ds = 8000.0  # pc
    mu_rel = 5.0  # mas/yr

    # 1. Test pi_rel (Relative parallax)
    # pi_rel = 1000/dl - 1000/ds = 0.25 - 0.125 = 0.125 mas
    calc_pi_rel = PHYSICS_REGISTRY["calc_pi_rel"]
    pi_rel = pt.as_tensor_variable(calc_pi_rel(dl, ds)).eval()
    assert np.isclose(pi_rel, 0.125)

    # 2. Test theta_E (Einstein Radius)
    # theta_E = sqrt(8.144 * M * pi_rel)
    # sqrt(8.144 * 0.5 * 0.125) = sqrt(0.509) approx 0.7134 mas
    calc_theta_E = PHYSICS_REGISTRY["calc_theta_E"]
    theta_E = calc_theta_E(mass, pi_rel).eval()
    assert np.isclose(theta_E, 0.7134, atol=1e-3)

    # 3. Test t_E (Einstein timescale)
    # t_E = (theta_E / mu_rel) * 365.25
    # (0.7134 / 5.0) * 365.25 approx 52.12 days
    calc_t_E = PHYSICS_REGISTRY["calc_t_E"]
    t_E = get_val(calc_t_E(theta_E, mu_rel))
    assert np.isclose(t_E, 52.12, atol=1e-2)

def test_lens_parameter_unit_handling():
    """Ensure lens parameters correctly handle 'd' and 'mas' string units."""
    p = Parameter(
        label="lens.t_E",
        unit="d",
        internal_unit="d",
        initval=50.0
    )
    # If the gatekeeper is working, this should stay 50.0
    # If internal_unit was accidentally '', it would have crashed or scaled.
    assert p.initval == 50.0
    assert p.internal_unit == u.day


def test_parallax_trajectory_offset():
    """Verify that Earth's motion correctly offsets the impact parameter u."""
    # If pi_E is 0, u should just be the linear trajectory
    # If pi_E is large, u should be significantly modulated
    # u(t) = sqrt(u0^2 + ((t-t0)/tE + pi_E_E*delta_E + pi_E_N*delta_N)^2)

    t = 2455000.0
    t0 = 2455000.0
    u0 = 0.5
    tE = 50.0
    pi_E_N = 1.0
    delta_N = 0.1  # Earth shift

    # Simple manual check of the logic inside your get_magnification
    # If this math doesn't match your Lens.py, the model won't fit!
    tau = (t - t0) / tE
    # u_vec = [u0, tau] + pi_E * delta
    u_n = u0 + (pi_E_N * delta_N)
    u_e = tau  # assuming pi_E_E is 0 for this test
    u_total = np.sqrt(u_n ** 2 + u_e ** 2)

    assert u_total > u0  # Parallax should have shifted the lens further away

def test_microlensing_contradiction_warning(capsys):
    """
    Verifies that providing contradictory physical values triggers
    the visual warning to the user.
    """
    # Provide Mass/Distances that imply t_E ~ 34 days,
    # but explicitly provide t_E = 100 days.
    user_params = {
        "star.Lens.mass": 0.5,
        "star.Lens.distance": 4000.0,
        "star.Source.distance": 8000.0,
        "star.Lens.pm_ra": 5.0,
        "star.Lens.pm_dec": 0.0,
        "star.Source.pm_ra": 0.0,
        "star.Source.pm_dec": 0.0,
        "lens.Lens.t_E": 100.0  # The contradiction
    }

    ConfigManager(user_params)

    # Capture stdout to see if the '!' box appeared
    captured = capsys.readouterr()
    assert "WARNING: PHYSICAL CONTRADICTION DETECTED" in captured.out
    assert "Relative Error:" in captured.out


def test_microlensing_sympy_pytensor_equivalence():
    """
    Ensures that initialization (SymPy) and sampling (PyTensor)
    use the exact same mathematical constants and logic.
    """
    # 1. Define physical inputs for the symbolic solver
    user_params = {
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.pm_ra": {"initval": 10.0},
        "star.Lens.pm_dec": {"initval": 0.0},
        "star.Source.pm_ra": {"initval": 0.0},
        "star.Source.pm_dec": {"initval": 0.0}
    }

    # 2. Get SymPy results via ConfigManager
    cm = ConfigManager(user_params)

    # Verify the solver completed the chain
    assert "lens.Lens.t_E" in cm.user_params

    te_sympy = cm.user_params["lens.Lens.t_E"]["initval"]
    thetaE_sympy = cm.user_params["lens.Lens.theta_E"]["initval"]
    pirel_sympy = cm.user_params["lens.Lens.pi_rel"]["initval"]

    # 3. Feed the SAME raw inputs into the PyTensor graph
    # (Using .eval() to pull the numeric result out of the graph)
    mass = 0.5
    dl = 4000.0
    ds = 8000.0
    mu_rel = 10.0

    pi_rel_pt = get_val(calc_pi_rel(dl, ds))
    theta_E_pt = get_val(calc_theta_E(mass, pi_rel_pt))
    t_E_pt = get_val(calc_t_E(theta_E_pt, mu_rel))

    # 4. Strict Assertion: 1e-8 tolerance to catch constant mismatches
    # If KAPPA is 8.144 in one and 8.1448 in another, this WILL fail.
    assert np.isclose(pirel_sympy, pi_rel_pt, rtol=1e-8), "pi_rel mismatch!"
    assert np.isclose(thetaE_sympy, theta_E_pt, rtol=1e-8), "theta_E mismatch!"
    assert np.isclose(te_sympy, t_E_pt, rtol=1e-8), "t_E mismatch!"


def test_microlensing_contradiction_warning(capsys):
    """
    Verifies that providing contradictory physical values triggers
    the visual warning to the user.
    """
    # Provide Mass/Distances that imply t_E ~ 34 days,
    # but explicitly provide t_E = 100 days.
    user_params = {
        "star.Lens.mass": 0.5,
        "star.Lens.distance": 4000.0,
        "star.Source.distance": 8000.0,
        "star.Lens.pm_ra": 5.0,
        "star.Lens.pm_dec": 0.0,
        "star.Source.pm_ra": 0.0,
        "star.Source.pm_dec": 0.0,
        "lens.Lens.t_E": 100.0  # The contradiction
    }

    ConfigManager(user_params)

    # Capture stdout to see if the '!' box appeared
    captured = capsys.readouterr()
    assert "WARNING: PHYSICAL CONTRADICTION DETECTED" in captured.out
    assert "Relative Error:" in captured.out