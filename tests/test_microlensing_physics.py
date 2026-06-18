import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from exozippy.components.mulensing.lens import Lens
from exozippy.components.parameter import Parameter
from astropy import units as u
from exozippy.physics_registry import PHYSICS_REGISTRY
from exozippy.config import ConfigManager
from exozippy.components.mulensing.physics import (
    calc_theta_E,
    calc_pi_rel,
    calc_t_E
)
from exozippy.system import System


def get_val(x):
    return x.eval() if hasattr(x, 'eval') else x


@pytest.mark.slow
def test_pspl_magnification_accuracy():
    """
    Given a PSPL model evaluated at t=t0 with zero observer positions (no
    parallax correction), when get_magnification is called, then the output
    must equal the analytical Paczynski formula A(u0) = (u0^2+2)/(u0*sqrt(u0^2+4)).

    At t=t0 with obs=0: tau=0 and u2=u0^2, so this reduces to a single-point
    check of the inline formula in lens.py:383.
    """
    u0_val = 0.3
    t0_val = 2460025.0

    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}],
    }
    user_params = {
        "lens.Lens.t_0":        {"initval": t0_val},
        "lens.Lens.u_0":        {"initval": u0_val},
        "lens.Lens.pi_E_N":     {"initval": 0.0, "sigma": 0.0},
        "lens.Lens.pi_E_E":     {"initval": 0.0, "sigma": 0.0},
        "star.Lens.distance":   {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.mass":       {"initval": 0.5},
        "star.Lens.pm_ra":      {"initval": 0.0},
        "star.Lens.pm_dec":     {"initval": 0.0},
        "star.Source.pm_ra":    {"initval": 0.0},
        "star.Source.pm_dec":   {"initval": 0.0},
        "star.Source.ra":       {"initval": 0.0},
        "star.Source.dec":      {"initval": 0.0},
        "star.Lens.ra":         {"initval": 0.0},
        "star.Lens.dec":        {"initval": 0.0},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()

    obs_zero = np.zeros((1, 3), dtype=np.float64)
    t_at_peak = np.array([t0_val])

    with model:
        A_node = system.lens.get_magnification(t_at_peak, obs_zero, system, index=0)
        f = pytensor.function(model.free_RVs, A_node, on_unused_input="ignore")
        ip = model.initial_point()
        zero_in = [np.zeros_like(ip[v.name]).astype("float64") for v in model.free_RVs]
        A_result = float(f(*zero_in)[0])

    expected = (u0_val**2 + 2) / (u0_val * np.sqrt(u0_val**2 + 4))
    np.testing.assert_allclose(A_result, expected, rtol=1e-6)


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



def test_microlensing_sympy_pytensor_equivalence():
    """
            Ensures that initialization (SymPy) and sampling (PyTensor)
            use the exact same mathematical constants and logic.
            """
    # 1. Define Topology
    system_config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}]
    }

    user_params = {
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.pm_ra": {"initval": 10.0},
        "star.Lens.pm_dec": {"initval": 0.0},
        "star.Source.pm_ra": {"initval": 0.0},
        "star.Source.pm_dec": {"initval": 0.0}
    }

    # 2. Pass topology and explicitly trigger the solver
    cm = ConfigManager(user_params, system_config=system_config)
    cm.finalize_user_params()

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


def test_microlensing_contradiction_warning(caplog):
    """
    Verifies that providing contradictory physical values triggers
    a logged warning to the user.
    """
    # 1. Define Topology
    system_config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}]
    }

    # Provide Mass/Distances that imply t_E ~ 34 days,
    # but explicitly provide t_E = 100 days.
    user_params = {
        "star.Lens.distance": 4000.0,
        "star.Source.distance": 8000.0,
        "lens.Lens.pi_rel": 0.999
    }

    # 2. Pass topology and explicitly trigger the solver
    import logging
    cm = ConfigManager(user_params, system_config=system_config)
    with caplog.at_level(logging.WARNING):
        cm.finalize_user_params()

    assert "contradiction detected" in caplog.text.lower()
    assert "sacrificing" in caplog.text.lower()