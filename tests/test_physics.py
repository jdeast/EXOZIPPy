import numpy as np
from astropy import units as u

from exozippy.physics_registry import PHYSICS_REGISTRY
from exozippy.constants import KEPLER_CONST
import exozippy.components.planet.physics
import exozippy.components.celestial_body.physics


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
    arsun = KEPLER_CONST * (m_total ** (1 / 3)) * (period ** (2 / 3))

    # ACT
    k_internal = calc_K(m_planet, m_total, ecc, arsun, sini, period).eval()
    k_ms = (k_internal * u.R_sun / u.day).to(u.m / u.s).value

    # ASSERT
    np.testing.assert_allclose(k_ms, 28.43, rtol=1e-2)


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
    rho_cgs = (rho_internal * u.M_sun / u.R_sun ** 3).to(u.g / u.cm ** 3).value

    # ASSERT
    np.testing.assert_allclose(rho_cgs, 1.4097, rtol=1e-3)