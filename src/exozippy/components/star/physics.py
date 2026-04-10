import pytensor.tensor as pt
from ...constants import LUM_CONST, FBOL_CONST
from ...physics_registry import register_physics

@register_physics
def calc_luminosity(radius, teff):
    return LUM_CONST * pt.sqr(radius) * pt.sqr(pt.sqr(teff))

@register_physics
def calc_fbol(luminosity, distance):
    return FBOL_CONST * luminosity / pt.sqr(distance)

@register_physics
def calc_parallax(distance):
    return 1e3 / distance

@register_physics
def calc_absmag(appmag, distance):
    return appmag - 5.0 * pt.log10(distance) + 5.0