import pytensor.tensor as pt
from ...constants import DENSITY_CONST, LOGG_CONST
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
    mass: solar masses
    radius: solar radii
    returns: cgs (log10)
    Note: this odd form of logg is designed to simplify the symbolic math and chain rule derivatives
    """
    return LOGG_CONST + pt.log10(mass) - 2.0 * pt.log10(radius)