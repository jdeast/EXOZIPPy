import pytensor.tensor as pt
import numpy as np
from ...physics_registry import register_physics
from ...constants import KAPPA, RSUN_TO_AU


@register_physics
def calc_pi_rel(dist_lens, dist_source):
    # Parallax = 1000 / distance (pc) -> mas
    # no matter what we do, we must not compute a NaN.
    # we make up values so we can compute some likelihood
    # then introduce penalties (see lens.build_likelihood) that will reject such non-physical solutions
    return (1000.0 / dist_lens) - (1000.0 / dist_source)

@register_physics
def calc_theta_E(mass_lens, pi_rel):
    # Angular Einstein Radius in mas. Guard against negative pi_rel (lens behind source)
    return pt.sqrt(KAPPA * mass_lens * pi_rel)

@register_physics
def calc_mu_ra_rel(pm_ra_lens, pm_ra_source):
    return pm_ra_lens - pm_ra_source

@register_physics
def calc_mu_dec_rel(pm_dec_lens, pm_dec_source):
    return pm_dec_lens - pm_dec_source

@register_physics
def calc_mu_rel_mag(mu_ra_rel, mu_dec_rel):
    return pt.sqrt(pt.sqr(mu_ra_rel) + pt.sqr(mu_dec_rel))

@register_physics
def calc_t_E(theta_E, mu_rel_mag):
    # Convert mu_rel_mag from mas/yr to mas/day, then divide theta_E
    return theta_E / (mu_rel_mag / 365.25)

@register_physics
def calc_pi_E_N(pi_rel, theta_E, mu_dec_rel, mu_rel_mag):
    # pi_E points in the direction of relative proper motion
    pi_E_mag = pi_rel / theta_E
    return pi_E_mag * (mu_dec_rel / mu_rel_mag)

@register_physics
def calc_pi_E_E(pi_rel, theta_E, mu_ra_rel, mu_rel_mag):
    pi_E_mag = pi_rel / theta_E
    return pi_E_mag * (mu_ra_rel / mu_rel_mag)

@register_physics
def calc_f_source_from_frac(log_f_total, q_frac):
    """Derives source flux from total flux and fraction."""
    return pt.exp(log_f_total) * q_frac

@register_physics
def calc_f_blend_from_frac(log_f_total, q_frac):
    """Derives blend flux from total flux and remainder."""
    return pt.exp(log_f_total) * (1.0 - q_frac)

@register_physics
def calc_rho(radius, distance, theta_E):
    theta_star_mas = (radius * RSUN_TO_AU / distance) * 1000.0
    return theta_star_mas/theta_E