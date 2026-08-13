import numpy as np
import pytensor.tensor as pt

from ...constants import DENSITY_CONST, KEPLER_CONST, LOGG_CONST
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
def calc_mass_from_log_q(log_q, star_mass):
    """Planet mass from the sampled log10 mass ratio and the host mass.

    log_q: log10(m_planet / m_host), dimensionless
    star_mass: solar masses
    returns: solar masses (always positive -- see
    Planet._resolve_mass_parameterization for when this coordinate applies)
    """
    return pt.power(10.0, log_q) * star_mass


@register_physics
def calc_m_total(planet_mass, star_mass):
    return pt.maximum(star_mass + planet_mass, 1e-9)


@register_physics
# The parameter this feeds is named `a` (user unit AU); the FUNCTION keeps
# the arsun name because it computes the internal value, which is in solRad
# -- the one layer where the unit is fixed by convention rather than config.
def calc_arsun(m_total, period):
    m13 = pt.power(m_total, 1.0 / 3.0)
    p2 = pt.sqr(period)
    p23 = pt.power(p2, 1.0 / 3.0)
    return KEPLER_CONST * m13 * p23


@register_physics
def calc_arstar(a, rstar):
    return a / rstar


@register_physics
def calc_p(radius, star_radius):
    return radius / star_radius


@register_physics
def calc_K(mass, m_total, ecc, a, sini, period):
    ecc_factor = 1.0 / pt.sqrt(1.0 - pt.sqr(ecc))
    return 2.0 * np.pi * (a * sini * (mass / m_total) * ecc_factor / period)


@register_physics
def calc_max_ecc(ar, p):
    return 1.0 - 1.0 / ar - p / ar


# --- Chen & Kipping 2017 mass-radius relation -------------------------------
# Ported from EXOFASTv2's massradius_chen.pro (Chen & Kipping 2017, ApJ 834,
# 17, Table 2).
# https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract
# A continuous broken power law R(M) in Earth units: the segment
# normalizations chain so adjacent segments meet at the break masses.
# Segments: Terran worlds (<= 2.04 Mearth) / Neptunian worlds (<= 0.414
# Mjup) / Jovian worlds (<= 0.08 Msun) / Stars.

CHEN_MASS_BREAKS = (2.04, 131.58079, 26644.8321)  # Mearth
CHEN_EXPONENTS = (0.279, 0.589, -0.044, 0.881)
# Per-segment scatter, as a fraction of the predicted radius.
CHEN_RP_FRAC = (0.0403, 0.1460, 0.0737, 0.0443)

_CHEN_T1, _CHEN_T2, _CHEN_T3 = CHEN_MASS_BREAKS
_CHEN_S1, _CHEN_S2, _CHEN_S3, _CHEN_S4 = CHEN_EXPONENTS
_CHEN_N1 = 1.0
_CHEN_N2 = _CHEN_T1 ** (_CHEN_S1 - _CHEN_S2)
_CHEN_N3 = _CHEN_N2 * _CHEN_T2 ** (_CHEN_S2 - _CHEN_S3)
_CHEN_N4 = _CHEN_N3 * _CHEN_T3 ** (_CHEN_S3 - _CHEN_S4)


@register_physics
def calc_chen_radius(mpearth):
    """Chen & Kipping 2017 predicted radius (Rearth) from mass (Mearth).

    mpearth must be positive: every switch branch (and its gradient) is
    evaluated for every input, and a non-integer power of a negative mass
    is NaN.  Callers clip first (EXOFASTv2 uses mpearth > 1d-10).
    """
    return pt.switch(
        mpearth <= _CHEN_T1,
        _CHEN_N1 * mpearth**_CHEN_S1,
        pt.switch(
            mpearth <= _CHEN_T2,
            _CHEN_N2 * mpearth**_CHEN_S2,
            pt.switch(
                mpearth <= _CHEN_T3,
                _CHEN_N3 * mpearth**_CHEN_S3,
                _CHEN_N4 * mpearth**_CHEN_S4,
            ),
        ),
    )


@register_physics
def calc_chen_radius_sigma(mpearth):
    """Scatter (Rearth) of the Chen & Kipping prediction at mass mpearth.

    Fractional per segment, so it scales with the prediction.  Same
    positive-mass requirement as calc_chen_radius.
    """
    frac = pt.switch(
        mpearth <= _CHEN_T1,
        CHEN_RP_FRAC[0],
        pt.switch(
            mpearth <= _CHEN_T2,
            CHEN_RP_FRAC[1],
            pt.switch(
                mpearth <= _CHEN_T3,
                CHEN_RP_FRAC[2],
                CHEN_RP_FRAC[3],
            ),
        ),
    )
    return calc_chen_radius(mpearth) * frac
