import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from astropy import units as u

from exozippy.components.mulensing.lens import Lens
from exozippy.components.mulensing.physics import (
    MU_REL_FLOOR,
    THETA_E_FLOOR,
    calc_mu_rel_mag,
    calc_pi_rel,
    calc_t_E,
    calc_theta_E,
)
from exozippy.components.parameter import Parameter
from exozippy.config import ConfigManager
from exozippy.constants import KAPPA, RSUN_TO_AU
from exozippy.physics_registry import PHYSICS_REGISTRY
from exozippy.system import System


def get_val(x):
    return x.eval() if hasattr(x, "eval") else x


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
        "lens.Lens.t_0": {"initval": t0_val},
        "lens.Lens.u_0": {"initval": u0_val},
        "lens.Lens.pi_E_N": {"initval": 0.0, "sigma": 0.0},
        "lens.Lens.pi_E_E": {"initval": 0.0, "sigma": 0.0},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.pm_ra": {"initval": 0.0},
        "star.Lens.pm_dec": {"initval": 0.0},
        "star.Source.pm_ra": {"initval": 0.0},
        "star.Source.pm_dec": {"initval": 0.0},
        "star.Source.ra": {"initval": 0.0},
        "star.Source.dec": {"initval": 0.0},
        "star.Lens.ra": {"initval": 0.0},
        "star.Lens.dec": {"initval": 0.0},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()

    obs_zero = np.zeros((1, 3), dtype=np.float64)
    t_at_peak = np.array([t0_val])

    with model:
        A_node = system.lens.get_magnification(
            t_at_peak, obs_zero, system, index=0
        )
        f = pytensor.function(model.free_RVs, A_node, on_unused_input="ignore")
        ip = model.initial_point()
        zero_in = [
            np.zeros_like(ip[v.name]).astype("float64") for v in model.free_RVs
        ]
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
    p = Parameter(label="lens.t_E", unit="d", internal_unit="d", initval=50.0)
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
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}],
    }

    user_params = {
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.pm_ra": {"initval": 10.0},
        "star.Lens.pm_dec": {"initval": 0.0},
        "star.Source.pm_ra": {"initval": 0.0},
        "star.Source.pm_dec": {"initval": 0.0},
    }

    # 2. Pass topology and explicitly trigger the solver
    cm = ConfigManager(user_params, system_config=system_config)
    cm.finalize_user_params()

    # Verify the solver completed the chain.  Derived values are injected
    # under the canonical INDEX form (lens.0.t_E) -- the only spelling
    # ConfigManager.resolve reads for every element.  See the inject-back
    # comment in finalize_user_params and tests/test_nsnl.py.
    assert "lens.0.t_E" in cm.user_params

    te_sympy = cm.user_params["lens.0.t_E"]["initval"]
    thetaE_sympy = cm.user_params["lens.0.theta_E"]["initval"]
    pirel_sympy = cm.user_params["lens.0.pi_rel"]["initval"]

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


def test_calc_theta_E_negative_pi_rel_returns_tiny_positive_not_nan():
    """
    Given a negative pi_rel (source in front of the lens -- unphysical),
    When calc_theta_E is called,
    Then it returns THETA_E_FLOOR: positive and tiny, not NaN and not
      exactly 0, so downstream parameters (rho, pi_E) stay finite, the Op
      receives no NaN, and log(theta_E) in the event-rate prior is finite.
      The floor is 6 decades below the 1e-6 turn-on of the
      theta_E_singularity soft bound, so that barrier is fully engaged
      wherever the floor bites.
    """
    # Arrange
    calc_theta_E = PHYSICS_REGISTRY["calc_theta_E"]
    mass = pt.as_tensor_variable(0.5)

    # Act: pi_rel < 0 means source is closer than lens
    theta_E_neg = calc_theta_E(mass, pt.as_tensor_variable(-0.1)).eval()
    theta_E_zero = calc_theta_E(mass, pt.as_tensor_variable(0.0)).eval()

    # Assert: finite and positive, no log(0) wall
    assert np.isfinite(theta_E_neg), (
        f"Expected finite value, got {theta_E_neg}"
    )
    assert theta_E_neg == THETA_E_FLOOR
    assert theta_E_zero == THETA_E_FLOOR
    assert THETA_E_FLOOR < 1e-6, "floor must sit inside the singularity bound"

    # Positive pi_rel still works correctly
    theta_E_pos = calc_theta_E(mass, pt.as_tensor_variable(0.125)).eval()
    assert theta_E_pos > 0.0


def test_calc_rho_uses_the_shared_theta_E_floor():
    """
    Given a theta_E inside the old private 1e-10 floor but above the shared
      THETA_E_FLOOR = 1e-12,
    When calc_rho is called,
    Then it divides by that theta_E itself, not by 1e-10: rho used to be
      computed against a DIFFERENT theta_E than t_E and pi_E anywhere in
      [1e-12, 1e-10), i.e. three numbers describing one lens while disagreeing
      about it (review 2.6.2).
    """
    # Arrange
    calc_rho = PHYSICS_REGISTRY["calc_rho"]
    radius, distance = 1.0, 1000.0
    theta_E = 1e-11
    theta_star_mas = (radius * RSUN_TO_AU / distance) * 1000.0

    # Act
    got = calc_rho(
        pt.as_tensor_variable(radius),
        pt.as_tensor_variable(distance),
        pt.as_tensor_variable(theta_E),
    ).eval()

    # Assert
    assert got == pytest.approx(theta_star_mas / theta_E, rel=1e-12)


def test_calc_rho_floors_at_theta_e_floor_and_does_not_scrub_nan():
    """
    Given theta_E = 0 and, separately, theta_E = NaN,
    When calc_rho is called,
    Then zero is floored at THETA_E_FLOOR (finite, no division by zero) while
      the NaN PROPAGATES.  A floor must never be paired with a NaN
      substitution (the PR #142 policy): substituting turns a failed
      computation into a healthy-looking likelihood with a zero gradient,
      which is the failure the floor exists to prevent.
    """
    # Arrange
    calc_rho = PHYSICS_REGISTRY["calc_rho"]
    radius, distance = 1.0, 1000.0
    theta_star_mas = (radius * RSUN_TO_AU / distance) * 1000.0

    def rho_at(theta_E):
        return calc_rho(
            pt.as_tensor_variable(radius),
            pt.as_tensor_variable(distance),
            pt.as_tensor_variable(theta_E),
        ).eval()

    # Act
    at_zero = rho_at(0.0)
    at_nan = rho_at(np.nan)

    # Assert
    assert np.isfinite(at_zero)
    assert at_zero == pytest.approx(theta_star_mas / THETA_E_FLOOR, rel=1e-12)
    assert np.isnan(at_nan)


def test_calc_theta_E_is_unchanged_in_the_physical_regime():
    """
    Given physical lens masses and positive pi_rel,
    When calc_theta_E is called,
    Then the value is bit-for-bit sqrt(KAPPA * M * pi_rel) -- the floor
      added for the unphysical branch must not perturb any reachable
      posterior region.
    """
    # Arrange
    calc_theta_E = PHYSICS_REGISTRY["calc_theta_E"]

    for mass_v, pi_rel_v in [(0.5, 0.125), (0.08, 0.02), (1.4, 1.0)]:
        # Act
        got = calc_theta_E(
            pt.as_tensor_variable(mass_v), pt.as_tensor_variable(pi_rel_v)
        ).eval()

        # Assert
        assert got == np.sqrt(KAPPA * mass_v * pi_rel_v)


def test_event_rate_prior_and_gradient_are_finite_for_negative_pi_rel():
    """
    Given a source in front of the lens (pi_rel < 0), a lens mass dragged
      negative by a linear-mass planet, or an exactly zero relative proper
      motion (the two stars share one pm default),
    When the event-rate prior log(mu_rel_geo) + log(theta_E) and its
      gradient w.r.t. the underlying sampled quantities are evaluated,
    Then both are finite.

    Before the fix theta_E was exactly 0 there, so the prior was a -inf
    wall -- and its gradient was NaN, because sqrt'(0) is infinite and
    pt.maximum's zero gradient on the clamped side turns that into
    0 * inf.  A NaN gradient poisons the whole logp for NUTS; the
    source_behind_lens soft bound is what is meant to push back.
    """
    # Arrange
    d_l, d_s, mass = pt.dscalar("d_l"), pt.dscalar("d_s"), pt.dscalar("m")
    mu_ra, mu_dec = pt.dscalar("mu_ra"), pt.dscalar("mu_dec")
    theta_E = calc_theta_E(mass, calc_pi_rel(d_l, d_s))
    mu_rel = calc_mu_rel_mag(mu_ra, mu_dec)
    prior = pt.log(pt.maximum(mu_rel, MU_REL_FLOOR)) + pt.log(
        pt.maximum(theta_E, THETA_E_FLOOR)
    )
    inputs = [d_l, d_s, mass, mu_ra, mu_dec]
    fn = pytensor.function(inputs, [prior] + pt.grad(prior, inputs))

    cases = {
        "physical": (1000.0, 8000.0, 0.5, 3.0, 4.0),
        "source in front of lens": (8000.0, 1000.0, 0.5, 3.0, 4.0),
        "source and lens co-located": (1000.0, 1000.0, 0.5, 3.0, 4.0),
        "negative lens mass": (1000.0, 8000.0, -0.3, 3.0, 4.0),
        "zero relative proper motion": (1000.0, 8000.0, 0.5, 0.0, 0.0),
    }

    for label, args in cases.items():
        # Act
        out = fn(*args)

        # Assert
        assert np.all(np.isfinite(out)), f"{label}: non-finite {out}"


def test_microlensing_contradiction_no_override(caplog):
    """
    When all variables in a violated equation are RANK_USER, the solver must
    leave every user value untouched and log a debug message — not silently
    sacrifice one value to satisfy the equation.
    """
    # Given: distances that imply pi_rel ~ 0.125, but user also sets pi_rel = 0.999
    system_config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}],
    }
    user_params = {
        "star.Lens.distance": 4000.0,
        "star.Source.distance": 8000.0,
        "lens.Lens.pi_rel": 0.999,
    }

    import logging

    cm = ConfigManager(user_params, system_config=system_config)
    with caplog.at_level(logging.DEBUG):
        cm.finalize_user_params()

    # When all variables are RANK_USER the solver skips and logs at debug level
    assert "over-constrained" in caplog.text.lower()
    assert "leaving all user values unchanged" in caplog.text.lower()
