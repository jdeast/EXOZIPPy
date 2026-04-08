import pytensor.tensor as pt
from .constants import DENSITY_CONST, LOGG_CONST, FBOL_CONST, LUM_CONST, KEPLER_CONST, TWOPI
import numpy as np

"""Calculates density of a sphere from mass and radius.
mass in solar masses
radius in solar radii
density in msol/rsol3
used in star and planet components
"""
def calc_density(mass, radius):
    return DENSITY_CONST * mass / (radius * pt.sqr(radius))

def calc_logg(mass, radius):
    """Calculates surface gravity (logg) from mass and radius."""
    return LOGG_CONST + pt.log10(mass) - 2.0 * pt.log10(radius)

def calc_luminosity(radius, teff):
    return LUM_CONST * pt.sqr(radius) * pt.sqr(pt.sqr(teff))

def calc_fbol(luminosity, distance):
    return FBOL_CONST * luminosity / pt.sqr(distance)

def calc_parallax(distance):
    return 1e3 / distance

def calc_absmag(appmag, distance):
    return appmag - 5.0 * pt.log10(distance) + 5.0

def calc_arsun(m_total, period):
    m13 = pt.power(m_total, 1.0 / 3.0)
    p2 = pt.sqr(period)
    p23 = pt.power(p2, 1.0 / 3.0)
    return KEPLER_CONST * m13 * p23

def calc_arstar(arsun, rstar):
    return arsun / rstar

def calc_b(ar,cosi,ecc,esinw):
    return ar * cosi * (1.0 - pt.sqr(ecc)) / (1.0 - esinw)

def calc_p(radius, star_radius):
    return radius / star_radius

def calc_K(mass, m_total, ecc, arsun, sini, period):
    const = 2.0 * np.pi
    mass_ratio = mass / m_total
    ecc_factor = 1.0 / pt.sqrt(1.0 - pt.sqr(ecc))
    return const * (arsun*sini*mass_ratio*ecc_factor/period)

def calc_period(logP):
    return 10**logP

def calc_n(period):
    return TWOPI/period

def calc_ecc(secosw, sesinw):
    e_raw = pt.sqr(sesinw) + pt.sqr(secosw)
    return pt.clip(e_raw, 0.0, 0.9999)

def calc_omega(secosw, sesinw):
    return pt.arctan2(sesinw, secosw)

def calc_sinw(omega):
    return pt.sin(omega)

def calc_cosw(omega):
    return pt.cos(omega)

def calc_esinw(ecc, sesinw):
    return pt.sqrt(ecc)*sesinw

def calc_ecosw(ecc, secosw):
    return pt.sqrt(ecc)*secosw

def calc_cosw(omega):
    return pt.cos(omega)

def calc_inc(cosi):
    return pt.arccos(cosi)

def calc_sini(inc):
    return pt.sin(inc)

# Stable Tc -> Tp logic
# arctan(x/y) -> arctan2(x,y)
# we multiply both x and y by sqrt(e) so we can use the step parameter directly and avoid the singularity at e=0
def calc_tp(ecc, sesinw, secosw, tc, n):
    E0 = 2.0 * pt.arctan2(
        pt.sqrt(1.0 - ecc) * (pt.sqrt(ecc) - sesinw),
        pt.sqrt(1.0 + ecc) * secosw
    )
    M0 = E0 - ecc * pt.sin(E0)
    return tc - M0/n

def calc_jitter(jitter_variance):
    # Safety switch to prevent NaNs if jitter_variance is momentarily negative
    return pt.switch(pt.lt(jitter_variance, 0.0), 0.0, pt.sqrt(jitter_variance))

PHYSICS_REGISTRY = {
    # star calculations
    "calc_density": calc_density,
    "calc_logg": calc_logg,
    "calc_luminosity": calc_luminosity,
    "calc_fbol": calc_fbol,
    "calc_parallax": calc_parallax,
    "calc_absmag": calc_absmag,
    # planet calculations

    # orbit calculations
    "calc_period": calc_period,
    "calc_n": calc_n,
    "calc_ecc": calc_ecc,
    "calc_omega": calc_omega,
    "calc_inc": calc_inc,
    "calc_sini": calc_sini,
    "calc_sinw": calc_sinw,
    "calc_cosw": calc_cosw,
    "calc_esinw": calc_esinw,
    "calc_ecosw": calc_ecosw,

    "calc_arsun": calc_arsun,
    "calc_arstar": calc_arstar,
    "calc_b": calc_b,
    "calc_p": calc_p,
    "calc_K": calc_K,
    "calc_tp": calc_tp,

    # rv instrument calculations
    "calc_jitter": calc_jitter
}