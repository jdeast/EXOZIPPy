import numpy as np
import pytest
import pytensor.tensor as pt
from astropy import units as u
from astropy import constants as const

# Internal Physics Registry
from exozippy.physics import PHYSICS_REGISTRY
from exozippy.constants import G, KEPLER_CONST


def test_calc_K_value():
    """
    Standard Verification:
    Jupiter around the Sun at 1 AU (365.25 days)
    should yield K ~ 28.4 m/s.
    """
    calc_K = PHYSICS_REGISTRY["calc_K"]

    # 1. Define Inputs in Internal Units (M_sun, R_sun, days)
    m_star = 1.0
    m_planet = (1.0 * u.Mjup).to(u.M_sun).value  # ~0.000954
    m_total = m_star + m_planet
    period = 365.25
    ecc = 0.0
    sini = 1.0

    # a in R_sun (Standard Kepler's 3rd)
    # a^3 = G * M * P^2 / (4pi^2)
    arsun = KEPLER_CONST * (m_total ** (1 / 3)) * (period ** (2 / 3))

    # 2. Execute the Registry Function
    # Returns K in Internal Units (R_sun / day)
    k_internal = calc_K(m_planet, m_total, ecc, arsun, sini, period).eval()

    # 3. Convert Result to m/s
    k_ms = (k_internal * u.R_sun / u.day).to(u.m / u.s).value

    print(f"\nInternal K: {k_internal} R_sun/d")
    print(f"Converted K: {k_ms} m/s")

    # Truth: 28.43 m/s
    np.testing.assert_allclose(k_ms, 28.43, rtol=1e-2)


def test_calc_density_sun():
    """Verify stellar density for the Sun is ~1.41 g/cm3."""
    calc_rho = PHYSICS_REGISTRY["calc_density"]

    # Sun mass=1, radius=1
    rho_internal = calc_rho(1.0, 1.0).eval()

    # Convert M_sun/R_sun^3 to g/cm3
    rho_cgs = (rho_internal * u.M_sun / u.R_sun ** 3).to(u.g / u.cm ** 3).value

    np.testing.assert_allclose(rho_cgs, 1.4097, rtol=1e-3)


def test_calc_arsun():
    """Verify semi-major axis for 1 year at 1 M_sun is ~215 R_sun (1 AU)."""
    calc_arsun = PHYSICS_REGISTRY["calc_arsun"]

    # 1 M_sun, 365.25 days
    a_rsun = calc_arsun(1.0, 365.25).eval()

    # 1 AU in R_sun is ~215.03
    np.testing.assert_allclose(a_rsun, 215.03, rtol=1e-3)