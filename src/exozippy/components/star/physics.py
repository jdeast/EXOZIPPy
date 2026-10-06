import numpy as np
import pytensor.tensor as pt

from ...constants import (
    DENSITY_CONST,
    FBOL_CONST,
    KAPPA,
    LOGG_CONST,
    LUM_CONST,
)
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


# --- Proper motion relative to Sgr A* (mulensing/conventions.md C31) -----
#
# Sgr A*'s apparent (reflex) proper motion, Galactic components, mas/yr:
# mu_l* = mu_l cos(b) = -6.411 +/- 0.008 along the plane and
# mu_b = -0.219 +/- 0.007 toward the North Galactic Pole (Reid & Brunthaler
# 2020, ApJ 892, 39, abstract; references.bib key ReidBrunthaler:2020).
SGRA_PM_L_COSB = np.float64(-6.411)
SGRA_PM_B = np.float64(-0.219)

# ICRS position of the North Galactic Pole: astropy's Galactic frame,
# SkyCoord(l=0, b=90, frame="galactic").icrs.  Hard-coded so the conversion
# is a closed form the pytensor graph can carry; tests/test_star_pm_sgra.py
# checks the whole conversion against astropy at several sky positions.
NGP_RA_RAD = np.radians(192.85947789477606)
NGP_DEC_RAD = np.radians(27.128252414968028)

# Strictly positive floor on the cos^2(b) radicand (CLAUDE.md: a floor of 0
# rebuilds the 0*inf gradient).  It binds only within ~1e-10 rad of a
# Galactic pole, where the local Galactic axes are undefined anyway.
_COSB2_FLOOR = np.float64(1e-20)


def sgra_pm_equatorial(ra, dec, xp=pt):
    """Sgr A*'s proper motion resolved on the LOCAL equatorial axes at
    (ra, dec): returns ``(pm_ra_cosdec, pm_dec)`` in mas/yr.

    The (mu_l*, mu_b) vector above is rotated by the position angle between
    Galactic and equatorial north at the star's position (the standard
    rotation, e.g. Poleski 2013, arXiv:1306.2945):

        C1 = sin(dec_G) cos(dec) - cos(dec_G) sin(dec) cos(ra - ra_G)
        C2 = cos(dec_G) sin(ra - ra_G),   cos(b) = sqrt(C1^2 + C2^2)
        pm_ra*  = (C1 mu_l* - C2 mu_b) / cos(b)
        pm_dec  = (C2 mu_l* + C1 mu_b) / cos(b)

    ra, dec in RADIANS (star.ra/dec's internal unit).  ``xp`` is the array
    module: pytensor (the model graph) or numpy (start values, tests).
    """
    d_ra = ra - NGP_RA_RAD
    c1 = np.sin(NGP_DEC_RAD) * xp.cos(dec) - np.cos(NGP_DEC_RAD) * xp.sin(
        dec
    ) * xp.cos(d_ra)
    c2 = np.cos(NGP_DEC_RAD) * xp.sin(d_ra)
    cosb = xp.sqrt(xp.maximum(c1 * c1 + c2 * c2, _COSB2_FLOOR))
    pm_ra_cosdec = (c1 * SGRA_PM_L_COSB - c2 * SGRA_PM_B) / cosb
    pm_dec = (c2 * SGRA_PM_L_COSB + c1 * SGRA_PM_B) / cosb
    return pm_ra_cosdec, pm_dec


@register_physics
def calc_pm_ra_sgra(pm_ra, ra, dec):
    # star.pm_ra_sgra: the star's absolute (ICRS) pm_ra*cos(dec) minus Sgr
    # A*'s, both resolved on the local axes at the star's own position.  A
    # pure translation of pm_ra at fixed (ra, dec); the full map
    # (pm, ra, dec) -> (pm_sgra, ra, dec) is unit-triangular, so |J| = 1
    # even with ra/dec sampled and a mu/sigma on it needs no Jacobian (C31).
    return pm_ra - sgra_pm_equatorial(ra, dec)[0]


@register_physics
def calc_pm_dec_sgra(pm_dec, ra, dec):
    # pm_dec twin of calc_pm_ra_sgra.
    return pm_dec - sgra_pm_equatorial(ra, dec)[1]


@register_physics
def calc_dl_from_pirel(d_source, pi_rel):
    # fitpirel inverse (swap 2): the LENS star's distance derived from
    # the sampled log-relative-parallax and the sampled source distance.
    # pi_rel[mas] = 1000/D_l - 1000/D_s  =>  D_l = 1000/(pi_rel + 1000/D_s),
    # automatically 0 < D_l < D_s for pi_rel > 0 (the self-guarding map).
    # d_source arrives through a same-parameter element dep
    # (star.distance[murel_source_map]); see OwnPrePatchRef.
    # NONLINEAR: the |dD_l/dlog_pi_rel| Jacobian potential lives in
    # Lens.build_likelihood (unlike fitmurel's |J| = 1 swap).
    return 1000.0 / (pi_rel + 1000.0 / d_source)


@register_physics
def calc_logmass_from_thetae(theta_E, pi_rel):
    # fitthetae inverse (swap 3), single lens body: the HOST star's
    # logmass derived from the sampled Einstein radius and the relative
    # parallax, theta_E^2 = kappa * M * pi_rel.  Log-linear in the
    # sampled coordinate (|J| = 2, constant), so no Jacobian potential.
    return pt.log10(theta_E**2 / (KAPPA * pt.maximum(pi_rel, 1e-12)))


@register_physics
def calc_logmass_from_thetae_binary(theta_E, pi_rel, log_q):
    # fitthetae inverse, star + one log_q companion: theta_E references
    # the TOTAL lens mass, so M_host = M_tot / (1 + q).
    return pt.log10(
        theta_E**2
        / (KAPPA * pt.maximum(pi_rel, 1e-12))
        / (1.0 + pt.power(10.0, log_q))
    )
