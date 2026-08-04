import numpy as np
import pytensor.tensor as pt

from ...constants import (
    C_MPS,
    DENSITY_CONST,
    KEPLER_CONST,
    LOGG_CONST,
    SOLRAD_PER_DAY_TO_MPS,
)
from ...physics_registry import register_physics

# Sphere geometry is not planet-specific, and PHYSICS_REGISTRY is a flat
# namespace keyed by function name -- so "calc_density" must have exactly one
# owner. planet/defaults.yaml still resolves it by name through the registry;
# this import just keeps the name available here too.
from ..star.physics import calc_density  # noqa: F401


@register_physics
def calc_logg_from_mass(mass, radius):
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
    return (
        2.0 * np.pi * (arsun * sini * (mass / m_total) * ecc_factor / period)
    )


@register_physics
def calc_max_ecc(ar, p):
    return 1.0 - 1.0 / ar - p / ar


# Bolometric approximation of the Doppler beaming factor (Faigler & Mazeh
# 2011, eq. 1: A_beam = (4-alpha)*K/c, with the bandpass-dependent spectral
# index alpha set to 0). Confirmed against EXOFASTv2's step2pars.pro line
# 260, which uses beam = 4*K/c -- i.e. alpha_beam=1, not alpha=0 as the
# 2011 paper's bolometric case would give. alpha_beam's true value runs
# 0.8-1.2 depending on bandpass, but EXOFASTv2 fixes it at 1 (factor of 4),
# so we match that rather than the paper's exact bolometric limit.
BEAM_FACTOR = 4.0


@register_physics
def calc_beam_from_K(K):
    """Doppler beaming amplitude (ppm) from the RV semi-amplitude K.

    K arrives in its internal unit (solRad/d, see planet/defaults.yaml);
    converted to m/s before forming the dimensionless K/c ratio.
    """
    k_mps = K * SOLRAD_PER_DAY_TO_MPS
    return BEAM_FACTOR * (k_mps / C_MPS) * 1e6
