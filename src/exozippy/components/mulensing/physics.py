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
    # Angular Einstein Radius in mas.
    # Guard against negative pi_rel (source in front of lens): no lensing occurs,
    # but we must return a finite value so downstream parameters (rho, pi_E) don't
    # propagate NaN into the Op.  The lens.build_likelihood potentials penalise
    # this unphysical configuration so the sampler rejects it.
    return pt.sqrt(KAPPA * mass_lens * pt.maximum(pi_rel, 0.0))

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
def calc_q(*masses):
    # (companion_1, ..., companion_k, primary) -> per-companion mass ratios
    # q_j = M_companion_j / M_primary.  Each dep arrives as a length-1 slice
    # (scalar bracket maps in Lens.build_maps); k companions concatenate to a
    # shape-(k,) vector.
    companions, primary = masses[:-1], masses[-1]
    if len(companions) == 1:
        return companions[0] / primary
    return pt.concatenate([pt.atleast_1d(c) for c in companions]) / primary

@register_physics
def calc_mlens_total(*masses):
    # Total lens mass: sum over all lens bodies.  theta_E, t_E, rho, and pi_E
    # are referenced to the TOTAL mass for multi-body lenses (community
    # convention for binary-lens parameters).
    total = masses[0]
    for m in masses[1:]:
        total = total + m
    return total

@register_physics
def calc_f_source(log_f_total, q_source):
    return pt.power(10, log_f_total) * q_source

@register_physics
def calc_f_blend(log_f_total, q_source):
    return pt.power(10, log_f_total) * (1.0 - q_source)

@register_physics
def calc_rho(radius, distance, theta_E):
    theta_star_mas = (radius * RSUN_TO_AU / distance) * 1000.0
    theta_E_safe = pt.maximum(pt.nan_to_num(theta_E, nan=0.0), 1e-10)
    return theta_star_mas / theta_E_safe

@register_physics
def calc_alpha(xalpha, yalpha):
    return pt.arctan2(yalpha, xalpha)