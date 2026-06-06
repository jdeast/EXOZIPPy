import pytensor.tensor as pt
import numpy as np
from ...constants import KEPLER_CONST, DENSITY_CONST, LOGG_CONST
from ...physics_registry import register_physics

@register_physics
def calc_density(mass, radius):
    """
    Calculates density of a sphere from mass and radius.
    mass: solar masses
    radius: solar radii
    returns: msol/rsol3 (internal)
    """
    return DENSITY_CONST * mass / (radius * pt.sqr(radius))

@register_physics
def calc_logg(mass, radius):
    """
    Calculates surface gravity (logg) from mass and radius.
    mass: planet mass, in solar masses
    radius: solar radii
    returns: cgs (log10)
    Note: this odd form of logg is designed to simplify the symbolic math and chain rule derivatives
    """
    return LOGG_CONST + pt.log10(mass) - 2.0 * pt.log10(radius)

@register_physics
def calc_m_total(planet_mass, star_mass):
    return pt.maximum(star_mass + planet_mass, 1e-9)

@register_physics
def calc_arsun(m_total, period):
    m13 = pt.power(m_total, 1.0 / 3.0)
    p2 = pt.sqr(period)
    p23 = pt.power(p2, 1.0 / 3.0)
    return KEPLER_CONST * m13 * p23

@register_physics
def calc_arstar(arsun, rstar):
    return arsun / rstar

@register_physics
def calc_p(radius, star_radius):
    return radius / star_radius

@register_physics
def calc_K(mass, m_total, ecc, arsun, sini, period):
    ecc_factor = 1.0 / pt.sqrt(1.0 - pt.sqr(ecc))
    return 2.0 * np.pi * (arsun * sini * (mass / m_total) * ecc_factor / period)

@register_physics
def calc_max_ecc(ar, p):
    return 1.0 - 1.0/ar - p/ar