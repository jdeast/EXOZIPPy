"""Tests for the heliocentric -> geocentric mu_rel frame conversion.

The star pm_ra/pm_dec are barycentric observables (constrained by the
galactic kinematic prior), but the light-curve trajectory is in the
Skowron+2011 GEOCENTRIC convention (Earth's position and velocity at t0_par
define the frame), so t_E and the pi_E direction must derive from
mu_rel_geo = mu_rel_helio - pi_rel * v_earth_perp(t0_par)/AU (Gould 2004).
The SIGN of that conversion is what these tests pin -- the flipped sign
changes t_E by tens of percent at pi_rel ~ 0.35 mas.
"""

import numpy as np
import pytensor.tensor as pt

from exozippy.components.mulensing.physics import (
    calc_mu_dec_rel_geo,
    calc_mu_ra_rel_geo,
)


def test_calc_mu_rel_geo_values():
    """
    Given a heliocentric relative proper motion, pi_rel, and an Earth
      velocity projection,
    When the geo conversion physics functions are evaluated,
    Then mu_geo = mu_helio - pi_rel * vperp, component by component.
    """
    mu_ra = pt.dscalar("mu_ra")
    mu_dec = pt.dscalar("mu_dec")
    pi_rel = pt.dscalar("pi_rel")
    v_e = pt.dscalar("v_e")
    v_n = pt.dscalar("v_n")

    ra_geo = calc_mu_ra_rel_geo(mu_ra, pi_rel, v_e).eval(
        {mu_ra: 3.1, pi_rel: 0.35, v_e: 4.2}
    )
    dec_geo = calc_mu_dec_rel_geo(mu_dec, pi_rel, v_n).eval(
        {mu_dec: -2.4, pi_rel: 0.35, v_n: -1.7}
    )
    assert np.isclose(float(ra_geo), 3.1 - 0.35 * 4.2, rtol=0, atol=1e-14)
    assert np.isclose(float(dec_geo), -2.4 + 0.35 * 1.7, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# First-principles sign pin
# ---------------------------------------------------------------------------

_EPS_ECL = np.radians(23.44)


def _s_of_t(t):
    """Synthetic circular 'Earth' orbit in the equatorial frame (AU)."""
    th = 2 * np.pi * np.asarray(t) / 365.25
    return np.stack(
        [
            np.cos(th),
            np.sin(th) * np.cos(_EPS_ECL),
            np.sin(th) * np.sin(_EPS_ECL),
        ],
        axis=-1,
    )


def _v_of_t(t, dt=0.5):
    """AU/day, same finite difference as MulensInstrument."""
    return (_s_of_t(np.asarray(t) + dt) - _s_of_t(np.asarray(t) - dt)) / (
        2 * dt
    )


def test_mu_geo_sign_matches_first_principles_trajectory():
    """
    Given a lens-source pair defined HELIOCENTRICALLY (mu_helio, pi_rel,
      theta_E) observed from a moving platform with full parallax,
    When the light-curve trajectory is computed the model's way -- Skowron
      deltas at t0_par, tau_p/u_p with t_E and pi_E derived from mu_rel_geo
      via the actual physics functions,
    Then the model magnification-trajectory u(t) reproduces the
      first-principles heliocentric u(t) to float precision -- and the
      OPPOSITE conversion sign (mu_helio + pi_rel*vperp) matches for no
      orientation convention at all.

    The trajectory formula (tau_p/u_p) is copied from Lens.get_magnification
    and is separately pinned against MulensModel; what this test adds is the
    frame-conversion sign feeding it.
    """
    # ARRANGE: sky position, anchor epoch, and the code's projections
    ra, dec = np.radians(268.0), np.radians(-28.5)
    e_hat = np.array([-np.sin(ra), np.cos(ra), 0.0])
    n_hat = np.array(
        [-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)]
    )

    def proj_en(xyz):
        return xyz @ e_hat, xyz @ n_hat

    t0p = 137.0
    t = np.linspace(t0p - 120, t0p + 120, 2001)

    # Skowron deltas exactly as MulensInstrument._abs_to_delta builds them
    dev = _s_of_t(t) - _s_of_t([t0p])[0] - np.outer(t - t0p, _v_of_t([t0p])[0])
    delta_e, delta_n = proj_en(dev)

    # Physical (heliocentric) parameters
    pi_rel = 0.35  # mas
    theta_E = 0.55  # mas
    mu_helio = np.array([3.1, -2.4])  # (E, N), mas/yr
    u0, t0 = 0.31, t0p

    vperp_en = np.array(proj_en(_v_of_t([t0p])[0] * 365.25))  # AU/yr
    s_full_e, s_full_n = proj_en(_s_of_t(t))
    s0_e, s0_n = proj_en(_s_of_t([t0p])[0])

    def best_match(mu_geo):
        """Min over orientation conventions of max|u_model - u_truth|."""
        mu_mag = float(np.hypot(*mu_geo))
        m_hat = mu_geo / mu_mag
        tE = theta_E / (mu_mag / 365.25)  # days, calc_t_E convention
        piE_E = pi_rel / theta_E * m_hat[0]
        piE_N = pi_rel / theta_E * m_hat[1]
        best = np.inf
        for p_sign in (+1, -1):
            # Truth: heliocentric linear motion + FULL parallax shift
            # (apparent relative position -= pi_rel * s_perp / AU), with
            # the constant offset chosen so the geocentric-linear part has
            # impact u0 at t0.
            p_hat = p_sign * np.array([-m_hat[1], m_hat[0]])
            th0 = theta_E * u0 * p_hat + pi_rel * np.array([s0_e, s0_n])
            dt_yr = (t - t0) / 365.25
            th_e = th0[0] + mu_helio[0] * dt_yr - pi_rel * s_full_e
            th_n = th0[1] + mu_helio[1] * dt_yr - pi_rel * s_full_n
            u_truth = np.hypot(th_e, th_n) / theta_E
            for u0_sign in (+1, -1):
                # Model: Lens.get_magnification's trajectory formula
                tau = (t - t0) / tE
                tau_p = tau - delta_n * piE_N - delta_e * piE_E
                u_p = u0_sign * u0 + delta_n * piE_E - delta_e * piE_N
                u_model = np.hypot(tau_p, u_p)
                best = min(best, float(np.max(np.abs(u_model - u_truth))))
        return best

    # ACT: convert through the ACTUAL physics functions
    mu_ra_s = pt.dscalar("mu_ra")
    mu_dec_s = pt.dscalar("mu_dec")
    pi_s = pt.dscalar("pi")
    ve_s = pt.dscalar("ve")
    vn_s = pt.dscalar("vn")
    mu_geo_code = np.array(
        [
            float(
                calc_mu_ra_rel_geo(mu_ra_s, pi_s, ve_s).eval(
                    {mu_ra_s: mu_helio[0], pi_s: pi_rel, ve_s: vperp_en[0]}
                )
            ),
            float(
                calc_mu_dec_rel_geo(mu_dec_s, pi_s, vn_s).eval(
                    {mu_dec_s: mu_helio[1], pi_s: pi_rel, vn_s: vperp_en[1]}
                )
            ),
        ]
    )
    mu_geo_flipped = mu_helio + pi_rel * vperp_en

    # ASSERT
    assert best_match(mu_geo_code) < 1e-9, (
        "physics-function mu_geo does not reproduce the heliocentric truth"
    )
    assert best_match(mu_geo_flipped) > 0.1, (
        "flipped conversion sign should NOT match -- the pin is meaningless"
    )


def test_earth_vperp_fallback_without_mulens_data():
    """
    Given a Lens in a system with no mulensinstrument (no t0_par anchor),
    When _earth_vperp_en is called,
    Then it returns (0, 0) -- mu_rel_geo degrades to the heliocentric value
      instead of crashing.
    """
    from types import SimpleNamespace

    from exozippy.components.mulensing.lens import Lens

    mock_self = SimpleNamespace(prefix="lens")
    mock_system = SimpleNamespace()  # no mulensinstrument attribute
    v_e, v_n = Lens._earth_vperp_en(mock_self, mock_system)
    assert v_e == 0.0 and v_n == 0.0
