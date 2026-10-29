import numpy as np
from astropy import units as u

# import exozippy.components.celestial_body.physics
import exozippy.components.planet.physics
import exozippy.components.star.physics
from exozippy.constants import KEPLER_CONST
from exozippy.physics_registry import PHYSICS_REGISTRY


def test_rv_semi_amplitude_matches_jupiter_sun_benchmark():
    """
    Given Jupiter's mass and a 1-year orbit around a Solar-mass star,
    When the internal K calculation is evaluated and converted back to user units,
    Then the result should match the known ~28.4 m/s semi-amplitude.
    """
    # ARRANGE
    calc_K = PHYSICS_REGISTRY["calc_K"]

    m_star = 1.0
    m_planet = (1.0 * u.Mjup).to(u.M_sun).value
    m_total = m_star + m_planet
    period = 365.25
    ecc = 0.0
    sini = 1.0
    a = KEPLER_CONST * (m_total ** (1 / 3)) * (period ** (2 / 3))

    # ACT
    k_internal = calc_K(m_planet, m_total, ecc, a, sini, period).eval()
    k_ms = (k_internal * u.R_sun / u.day).to(u.m / u.s).value

    # ASSERT
    np.testing.assert_allclose(k_ms, 28.43, rtol=1e-2)


def test_rv_semi_amplitude_matches_the_astropy_closed_form_off_the_benchmark():
    """
    Given an eccentric, inclined orbit (e = 0.3, sin i = 0.8) of a massive
      companion (0.2 Msun around 0.9 Msun, so M_star and M_total differ by
      20%),
    When calc_K is evaluated with the semi-major axis computed from astropy's
      G (not the codebase's KEPLER_CONST),
    Then K matches the textbook semi-amplitude to 1e-10:
          K = (2 pi G / P)^(1/3) m sin i / (M_total^(2/3) sqrt(1 - e^2)).

    The Jupiter-Sun benchmark above sits at e = 0, sin i = 1 and m << M at
    rtol 1e-2, so a lost sqrt(1-e^2), a dropped sin i or M_star in place of
    M_total all pass it (review 7.16.1).
    """
    from astropy import constants as const

    # ARRANGE
    calc_K = PHYSICS_REGISTRY["calc_K"]
    m_star, m_comp = 0.9, 0.2
    m_total = m_star + m_comp
    period_d, ecc, sini = 17.3, 0.3, 0.8
    p_q = period_d * u.day
    a_q = (const.G * m_total * u.M_sun * p_q**2 / (4 * np.pi**2)) ** (1 / 3)
    a_rsun = a_q.to(u.R_sun).value

    # ACT
    k_internal = calc_K(m_comp, m_total, ecc, a_rsun, sini, period_d).eval()
    k_ms = (k_internal * u.R_sun / u.day).to(u.m / u.s).value

    # ASSERT
    expected = (
        (
            (2 * np.pi * const.G / p_q) ** (1 / 3)
            * (m_comp * u.M_sun)
            * sini
            / ((m_total * u.M_sun) ** (2 / 3) * np.sqrt(1 - ecc**2))
        )
        .to(u.m / u.s)
        .value
    )
    np.testing.assert_allclose(k_ms, expected, rtol=1e-10)


def test_stellar_density_matches_solar_benchmark():
    """
    Given a mass and radius of 1.0 in Solar units,
    When the density is calculated and converted to cgs units,
    Then the result should match the known solar density of ~1.41 g/cm^3.
    """
    # ARRANGE
    calc_rho = PHYSICS_REGISTRY["calc_density"]

    # ACT
    rho_internal = calc_rho(1.0, 1.0).eval()
    rho_cgs = (rho_internal * u.M_sun / u.R_sun**3).to(u.g / u.cm**3).value

    # ASSERT
    np.testing.assert_allclose(rho_cgs, 1.4097, rtol=1e-3)
