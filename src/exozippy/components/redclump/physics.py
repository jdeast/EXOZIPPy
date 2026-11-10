"""Registered physics of the red-clump reference (redclump.md)."""

import pytensor.tensor as pt

from ...constants import LOG_L0_CONST, LUM_CONST
from ...physics_registry import register_physics


@register_physics
def calc_rc_radius(mi, teff, bc_i0):
    """The clump's radius (Rsun) from its absolute I magnitude: M_bol =
    M_I + BC_I(A_V = 0), L/Lsun = 10^(-0.4 (M_bol - M_bol,sun)) with the
    IAU zero point the SED uses (calc_absbolmag's inverse), and R from L =
    LUM_CONST R^2 Teff^4 (star.physics.calc_luminosity's inverse)."""
    mbol = mi + bc_i0
    lum = 10.0 ** (-0.4 * (mbol - LOG_L0_CONST))
    return pt.sqrt(lum / (LUM_CONST * teff**4))


@register_physics
def calc_rc_vi0(bc_v0, bc_i0):
    """(V-I)_0 = M_V - M_I = BC_I - BC_V at A_V = 0."""
    return bc_i0 - bc_v0


@register_physics
def calc_rc_identity(value):
    """A derived element whose value is a context node the component built
    (the clump's band extinctions and color excesses off the SED's grid)."""
    return value


@register_physics
def calc_rc_av_ratio(av_star, av_clump):
    """A tied star's A_V relative to the clump's.  av_clump's lower bound is
    strictly positive (defaults.yaml), so the ratio and its gradient are
    finite on the whole support."""
    return av_star / av_clump
