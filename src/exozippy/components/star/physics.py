import pytensor.tensor as pt

from ...constants import DENSITY_CONST, FBOL_CONST, LOGG_CONST, LUM_CONST
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
def calc_logg_from_logmass(logmass, radius):
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
    Calculates stellar mass from its base-10 logarithm.
    logmass: log_10 of stellar mass, in solar masses
    returns: solar masses
    Note: logmass is the SAMPLED coordinate and mass is derived from it, so
    logmass's bounds are the real hard support (see star/defaults.yaml);
    star.mass carries no lower/upper of its own.
    """
    return 10**logmass


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


@register_physics
def calc_pm_from_murel(pm_source, mu_rel_component):
    # fitmurel inverse (one component; ra and dec each call it): the LENS
    # star's pm derived from the sampled source pm and the sampled
    # heliocentric relative pm.  Linear, |J| = 1, so no correction
    # potential accompanies the swap.  pm_source arrives through a
    # same-parameter element dep (star.pm_*[murel_source_map]), i.e. the
    # pre-patch tensor of the very parameter being built -- see
    # OwnPrePatchRef in components/parameter.py.
    return pm_source + mu_rel_component
