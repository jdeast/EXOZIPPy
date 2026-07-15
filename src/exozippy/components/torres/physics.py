"""Torres+2010 empirical mass/radius relations.

Ported from EXOFASTv2's massradius_torres.pro.  Both relations are
polynomials in X = log10(Teff) - 4.1, logg and [Fe/H], and both predict a
*base-10 logarithm* -- log10(M/Msun) and log10(R/Rsun) -- which is why the
scatter below is quoted in dex rather than as a fraction (contrast
components/mann, whose scatter is fractional).

Reference
---------
Torres, Andersen & Gimenez 2010, A&ARv 18, 67 (Table 4)
"""

import pytensor.tensor as pt

from ...physics_registry import register_physics

# Torres+2010 Table 4.  Both dot into [1, X, X^2, X^3, logg^2, logg^3, feh].
_A_COEFF = (1.5689, 1.3787, 0.4243, 1.139, -0.14250, 0.01969, 0.10100)   # log M
_B_COEFF = (2.4427, 0.6679, 0.1771, 0.705, -0.21415, 0.02306, 0.04173)   # log R

# Scatter of the relations, in dex.  EXOFASTv2 hardcodes these as
# ulogm/urstar in exofast_chi2v2.pro; they are exposed as optional
# per-instance overrides (logm_floor / logr_floor) by the component.
LOGM_FLOOR = 0.027
LOGR_FLOOR = 0.014

# Torres+2010 is calibrated on detached eclipsing binaries; EXOFASTv2 warns
# below this mass, where the Mann+ relations are the better choice.
MSTAR_MIN = 0.6


def _terms(teff, logg, feh):
    """The shared [1, X, X^2, X^3, logg^2, logg^3, feh] basis."""
    x = pt.log10(teff) - 4.1
    return (
        1.0,
        x,
        pt.sqr(x),
        x ** 3,
        pt.sqr(logg),
        logg ** 3,
        feh,
    )


def _dot(coeff, terms):
    return sum(c * t for c, t in zip(coeff, terms))


@register_physics
def calc_torres_logmass(teff, logg, feh):
    """log10(M/Msun) from Teff (K), logg (cgs dex) and [Fe/H] (dex)."""
    return _dot(_A_COEFF, _terms(teff, logg, feh))


@register_physics
def calc_torres_logradius(teff, logg, feh):
    """log10(R/Rsun) from Teff (K), logg (cgs dex) and [Fe/H] (dex)."""
    return _dot(_B_COEFF, _terms(teff, logg, feh))
