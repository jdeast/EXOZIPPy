import pytensor.tensor as pt
import numpy as np
from ...physics_registry import register_physics
from ...constants import KAPPA


@register_physics
def calc_pi_rel(dist_lens, dist_source):
    # Parallax = 1000 / distance (pc) -> mas
    return (1000.0 / dist_lens) - (1000.0 / dist_source)

@register_physics
def calc_theta_E(mass_lens, pi_rel):
    # Angular Einstein Radius in mas. Guard against negative pi_rel (lens behind source)
    safe_pi = pt.maximum(pi_rel, 1e-9)
    return pt.sqrt(KAPPA * mass_lens * safe_pi)

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
    safe_mu = pt.maximum(mu_rel_mag, 1e-9)
    return theta_E / (safe_mu / 365.25)

@register_physics
def calc_pi_E_N(pi_rel, theta_E, mu_dec_rel, mu_rel_mag):
    # pi_E points in the direction of relative proper motion
    pi_E_mag = pi_rel / pt.maximum(theta_E, 1e-9)
    return pi_E_mag * (mu_dec_rel / pt.maximum(mu_rel_mag, 1e-9))

@register_physics
def calc_pi_E_E(pi_rel, theta_E, mu_ra_rel, mu_rel_mag):
    pi_E_mag = pi_rel / pt.maximum(theta_E, 1e-9)
    return pi_E_mag * (mu_ra_rel / pt.maximum(mu_rel_mag, 1e-9))