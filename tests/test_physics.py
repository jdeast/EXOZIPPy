import numpy as np
import pytest
from astropy import units as u

# FIX: Updated import path
from exozippy.physics_registry import PHYSICS_REGISTRY
from exozippy.constants import KEPLER_CONST

# FIX: Import the component physics files to trigger the @register_physics decorators
import exozippy.components.planet.physics
import exozippy.components.celestial_body.physics


def test_keplerian_k_velocity():
    """Verify K ~ 28.4 m/s for Jupiter around Sun."""
    calc_K = PHYSICS_REGISTRY["calc_K"]

    m_star = 1.0
    m_planet = (1.0 * u.Mjup).to(u.M_sun).value
    m_total = m_star + m_planet
    period = 365.25
    ecc = 0.0
    sini = 1.0
    arsun = KEPLER_CONST * (m_total ** (1 / 3)) * (period ** (2 / 3))

    # Calculate in internal units (R_sun/day)
    k_internal = calc_K(m_planet, m_total, ecc, arsun, sini, period).eval()

    # Convert to m/s
    k_ms = (k_internal * u.R_sun / u.day).to(u.m / u.s).value
    np.testing.assert_allclose(k_ms, 28.43, rtol=1e-2)


def test_stellar_density():
    """Verify Sun density is ~1.41 g/cm3."""
    calc_rho = PHYSICS_REGISTRY["calc_density"]
    rho_internal = calc_rho(1.0, 1.0).eval()
    rho_cgs = (rho_internal * u.M_sun / u.R_sun ** 3).to(u.g / u.cm ** 3).value
    np.testing.assert_allclose(rho_cgs, 1.4097, rtol=1e-3)