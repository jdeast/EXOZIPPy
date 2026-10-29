"""Microlensing unit and frame conversions: known answers and round trips.

Review 7.16.2.  Each conversion below crosses a unit or frame boundary --
mas <-> rad on the sky, R_sun/pc -> mas, AU/day -> AU/yr -> mas/yr, mass and
parallax -> Einstein radius and back -- and each gets one check against an
answer that does not come from the code under test (astropy's own spherical
offsets and frame differentials, or the closed form written out from the
IAU constants) and one round trip through the code's own inverse to machine
precision.

The Earth-velocity projection (``MulensEvent._earth_vperp_en``) is the Gould
(2004) helio -> geo conversion constant (mulensing/conventions.md C5): it
turns AU/day into AU/yr, so ``pi_rel`` [mas] times it is mas/yr.  The sign
of its USE is pinned by tests/test_mu_rel_geo.py and the velocity itself by
tests/test_ephemeris_reception_instant.py; what is pinned here is the
projection and the unit, against astropy's own transformation of a
velocity into a proper motion.
"""

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytensor.tensor as pt
import pytest
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
    SphericalDifferential,
    SphericalRepresentation,
    get_body_barycentric_posvel,
)
from astropy.time import Time

from exozippy.components.mulensing.mulensevent import MulensEvent
from exozippy.components.mulensing.physics import (
    calc_pi_E_E,
    calc_pi_E_N,
    calc_pi_rel,
    calc_rho,
    calc_t_E,
    calc_theta_E,
)
from exozippy.components.star.physics import (
    calc_dec_from_lens_offset,
    calc_dl_from_pirel,
    calc_logmass_from_thetae,
    calc_logmass_from_thetae_binary,
    calc_ra_from_lens_offset,
)

# Written-out reference constants (never recomputed through exozippy).
R_SUN_M = 6.957e8  # IAU 2015 B3
PC_M = 3.0856775814913673e16  # IAU 2015 B2
AU_M = 149597870700.0
RAD_TO_MAS = 180.0 / np.pi * 3600.0e3
KAPPA_GOULD = 8.144  # mas / M_sun, Gould (2000)

# OGLE-2014-BLG-0939 (examples/ob140939) and two other sight lines,
# including a high-|dec| one where the 1/cos(dec) of an RA offset matters.
SKY = [
    (266.801041667, -21.3829722222),
    (17.5, 62.0),
    (271.0, -29.5),
]


def _ev(expr, **vals):
    syms = {k: pt.dscalar(k) for k in vals}
    out = expr(**syms) if callable(expr) else expr
    return float(out.eval({syms[k]: v for k, v in vals.items()}))


# --- the lens's sky position from the source's (mas offset -> rad) -------


@pytest.mark.parametrize("ra_deg,dec_deg", SKY)
def test_lens_position_offset_matches_astropy_spherical_offsets(
    ra_deg, dec_deg
):
    """
    Given a source position and a lens-minus-source offset of a few mas
      on the (East, North) axes,
    When calc_ra/dec_from_lens_offset place the lens,
    Then astropy's spherical_offsets_to from source to lens returns the
      same (East, North) offset to 1e-6 mas (known answer), and feeding
      astropy's offset back through the code recovers the lens position
      to machine precision (round trip).
    """
    ra_s, dec_s = np.radians(ra_deg), np.radians(dec_deg)
    off_e, off_n = 3.7, -1.9  # mas

    ra_l = _ev(
        lambda ra, dec, oe: calc_ra_from_lens_offset(ra, dec, oe),
        ra=ra_s,
        dec=dec_s,
        oe=off_e,
    )
    dec_l = _ev(
        lambda dec, on: calc_dec_from_lens_offset(dec, on),
        dec=dec_s,
        on=off_n,
    )

    src = SkyCoord(ra=ra_s * u.rad, dec=dec_s * u.rad)
    lens = SkyCoord(ra=ra_l * u.rad, dec=dec_l * u.rad)
    d_e, d_n = src.spherical_offsets_to(lens)
    assert d_e.to_value(u.mas) == pytest.approx(off_e, abs=1e-6)
    assert d_n.to_value(u.mas) == pytest.approx(off_n, abs=1e-6)

    # Round trip: astropy's measured offset back through the code.
    ra_back = _ev(
        lambda ra, dec, oe: calc_ra_from_lens_offset(ra, dec, oe),
        ra=ra_s,
        dec=dec_s,
        oe=d_e.to_value(u.mas),
    )
    dec_back = _ev(
        lambda dec, on: calc_dec_from_lens_offset(dec, on),
        dec=dec_s,
        on=d_n.to_value(u.mas),
    )
    assert ra_back == pytest.approx(ra_l, abs=1e-15)
    assert dec_back == pytest.approx(dec_l, abs=1e-15)


# --- rho: R_sun and pc -> mas ---------------------------------------------


def test_rho_angular_radius_is_the_closed_form_in_mas():
    """
    Given a 10 R_sun clump giant at 8 kpc and theta_E = 1 mas,
    When calc_rho converts R_sun/pc to an angle,
    Then theta_* = R/D in mas from the IAU constants (5.81 uas), and
      rho * theta_E returns theta_* (the round trip through theta_E).
    """
    radius, dist, theta_e = 10.0, 8000.0, 0.73
    expected_theta_star = radius * R_SUN_M / (dist * PC_M) * RAD_TO_MAS
    assert expected_theta_star == pytest.approx(5.81e-3, abs=1e-5)

    rho = _ev(lambda r, d, t: calc_rho(r, d, t), r=radius, d=dist, t=theta_e)
    assert rho * theta_e == pytest.approx(expected_theta_star, rel=1e-13)


# --- theta_E, pi_rel, mass: forward and inverse ---------------------------


@pytest.mark.parametrize(
    "mass,d_l,d_s", [(0.5, 4000.0, 8000.0), (0.08, 7200.0, 8300.0)]
)
def test_theta_e_from_mass_matches_gould_and_inverts(mass, d_l, d_s):
    """
    Given a lens mass and the two distances,
    When pi_rel and theta_E are computed, then inverted by the fitthetae
      and fitpirel inverses (calc_logmass_from_thetae, calc_dl_from_pirel),
    Then theta_E = sqrt(kappa M pi_rel) with Gould (2000)'s published
      kappa to its quoted 4 digits, and the inverses recover M and D_L to
      machine precision.
    """
    pi_rel = _ev(lambda a, b: calc_pi_rel(a, b), a=d_l, b=d_s)
    assert pi_rel == pytest.approx(1000.0 / d_l - 1000.0 / d_s, rel=1e-15)
    theta_e = _ev(lambda m, p: calc_theta_E(m, p), m=mass, p=pi_rel)
    assert theta_e == pytest.approx(
        np.sqrt(KAPPA_GOULD * mass * pi_rel), rel=1e-4
    )

    logm = _ev(
        lambda t, p: calc_logmass_from_thetae(t, p), t=theta_e, p=pi_rel
    )
    assert 10.0**logm == pytest.approx(mass, rel=1e-13)
    d_l_back = _ev(lambda s, p: calc_dl_from_pirel(s, p), s=d_s, p=pi_rel)
    assert d_l_back == pytest.approx(d_l, rel=1e-13)


def test_binary_host_mass_inverse_returns_the_host_not_the_total():
    """
    Given a host of 0.4 M_sun with a log_q = -1.3 companion, so theta_E
      references the TOTAL mass M_h (1 + q),
    When calc_logmass_from_thetae_binary inverts theta_E,
    Then it returns the host's 0.4 M_sun to machine precision.
    """
    m_host, log_q, pi_rel = 0.4, -1.3, 0.11
    m_tot = m_host * (1.0 + 10.0**log_q)
    theta_e = _ev(lambda m, p: calc_theta_E(m, p), m=m_tot, p=pi_rel)
    logm = _ev(
        lambda t, p, q: calc_logmass_from_thetae_binary(t, p, q),
        t=theta_e,
        p=pi_rel,
        q=log_q,
    )
    assert 10.0**logm == pytest.approx(m_host, rel=1e-13)


def test_t_e_and_pi_e_round_trip_to_the_relative_proper_motion():
    """
    Given theta_E, pi_rel and a relative proper motion (E, N) in mas/yr,
    When t_E (days) and pi_E (N, E) are computed,
    Then mu = theta_E / t_E * 365.25 d recovers |mu| (mas/day -> mas/yr),
      pi_E's direction recovers mu's direction (C8: phi_pi = atan2(E, N)),
      and |pi_E| = pi_rel / theta_E -- each to machine precision.
    """
    theta_e, pi_rel, mu_e, mu_n = 0.62, 0.21, -3.1, 4.4
    mu = np.hypot(mu_e, mu_n)

    t_e = _ev(lambda t, m: calc_t_E(t, m), t=theta_e, m=mu)
    # Known answer: mas/yr -> mas/day by the Julian year.
    assert t_e == pytest.approx(theta_e / mu * 365.25, rel=1e-15)
    assert theta_e / t_e * 365.25 == pytest.approx(mu, rel=1e-15)

    pn = _ev(
        lambda p, t, n, m: calc_pi_E_N(p, t, n, m),
        p=pi_rel,
        t=theta_e,
        n=mu_n,
        m=mu,
    )
    pe = _ev(
        lambda p, t, e, m: calc_pi_E_E(p, t, e, m),
        p=pi_rel,
        t=theta_e,
        e=mu_e,
        m=mu,
    )
    assert np.hypot(pn, pe) == pytest.approx(pi_rel / theta_e, rel=1e-15)
    assert np.arctan2(pe, pn) == pytest.approx(
        np.arctan2(mu_e, mu_n), abs=1e-15
    )


# --- Earth velocity on the sky: AU/day -> AU/yr, (E, N) ---------------------


@pytest.mark.parametrize("ra_deg,dec_deg", SKY)
@pytest.mark.parametrize("t0_par", [2456836.0, 2457000.0, 2460200.5])
def test_earth_vperp_matches_astropy_velocity_projection(
    ra_deg, dec_deg, t0_par
):
    """
    Given the Earth's barycentric velocity at t0_par in AU/day (the form
      MulensInstrument stores) and a line of sight,
    When _earth_vperp_en projects it to (East, North) in AU/yr,
    Then it equals astropy's own decomposition of that velocity into
      (d lon cos lat, d lat) at unit distance -- the proper motion it would
      impose on an object 1 AU away, i.e. the Gould (2004) factor that
      multiplies pi_rel [mas] into mas/yr -- to 1e-12 AU/yr; and the
      radial part plus the two tangential parts rebuild |v| (round trip).
    """
    _, vel = get_body_barycentric_posvel("earth", Time(t0_par, format="jd"))
    v_au_day = vel.xyz.to_value(u.au / u.day)
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    system = SimpleNamespace(
        mulensinstrument=SimpleNamespace(
            _earth_vel_ref=v_au_day, _source_radec_rad=(ra, dec)
        )
    )

    v_e, v_n = MulensEvent._earth_vperp_en(
        SimpleNamespace(prefix="mulensevent"), system
    )

    # astropy: a point 1 AU along the line of sight carrying the Earth's
    # velocity; its spherical differential is the angular rate.
    pos = SkyCoord(ra=ra * u.rad, dec=dec * u.rad).cartesian * u.au
    rep = CartesianRepresentation(pos.xyz).with_differentials(
        CartesianDifferential(vel.xyz.to(u.au / u.yr))
    )
    sph = rep.represent_as(
        SphericalRepresentation, differential_class=SphericalDifferential
    )
    dif = sph.differentials["s"]
    exp_e = (dif.d_lon * np.cos(sph.lat)).to_value(u.rad / u.yr)  # (AU/yr)/AU
    exp_n = dif.d_lat.to_value(u.rad / u.yr)
    exp_r = dif.d_distance.to_value(u.au / u.yr)

    assert v_e == pytest.approx(exp_e, abs=1e-12)
    assert v_n == pytest.approx(exp_n, abs=1e-12)
    # Round trip: the projection loses nothing -- radial + tangential = |v|.
    assert np.sqrt(v_e**2 + v_n**2 + exp_r**2) == pytest.approx(
        np.linalg.norm(vel.xyz.to_value(u.au / u.yr)), rel=1e-13
    )
    # Scale sanity against the textbook: |v_earth| ~ 29.8 km/s ~ 6.28 AU/yr.
    assert np.hypot(v_e, v_n) < 2.0 * np.pi * 1.02
