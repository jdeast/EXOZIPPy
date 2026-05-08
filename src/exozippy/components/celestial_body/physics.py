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
def calc_logg(logmass, radius):
    """
    Calculates surface gravity (logg) from mass and radius.
    logmass: log_10 of stellar mass, in solar masses
    radius: solar radii
    returns: cgs (log10)
    Note: this odd form of logg is designed to simplify the symbolic math and chain rule derivatives
    """
    return LOGG_CONST + logmass - 2.0 * pt.log10(radius)

@register_physics
def calc_mass(logmass):
    """
    Calculates surface gravity (logg) from mass and radius.
    mass: solar masses
    radius: solar radii
    returns: cgs (log10)
    Note: this odd form of logg is designed to simplify the symbolic math and chain rule derivatives
    """
    return 10**logmass