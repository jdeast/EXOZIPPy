import pytensor.tensor as pt
from ...constants import TWOPI
from ...physics_registry import register_physics
import numpy as np


@register_physics
def calc_period(logP):
    return 10**logP

@register_physics
def calc_group_mass(*group_masses):
    """Total mass of a body group: sum of the per-component-type weighted
    sums injected by Orbit.add_parameter (one term per component type)."""
    total = group_masses[0]
    for m in group_masses[1:]:
        total = total + m
    return total

@register_physics
def calc_n(period):
    return TWOPI/period

@register_physics
def calc_ecc(secosw, sesinw):
    e_raw = pt.sqr(sesinw) + pt.sqr(secosw)
    return pt.clip(e_raw, 0.0, 0.9999)

@register_physics
def calc_omega(secosw, sesinw):
    e_raw = pt.sqr(sesinw) + pt.sqr(secosw)
    # If exactly 0, enforce the limit so the RV phase stays perfectly aligned
    return pt.switch(pt.eq(e_raw, 0.0), np.pi / 2.0, pt.arctan2(sesinw, secosw))

@register_physics
def calc_bigomega(xbigomega, ybigomega):
    # Longitude of the ascending node from its direction vector; the radius
    # is a free positive scale absorbed by the N(0,1) priors (same geometry
    # sampler trick as the microlensing trajectory angle alpha).
    return pt.arctan2(ybigomega, xbigomega)

@register_physics
def calc_sinw(omega):
    return pt.sin(omega)

@register_physics
def calc_cosw(omega):
    return pt.cos(omega)

@register_physics
def calc_esinw(ecc, sesinw):
    return pt.sqrt(ecc)*sesinw

@register_physics
def calc_ecosw(ecc, secosw):
    return pt.sqrt(ecc)*secosw

@register_physics
def calc_inc(cosi):
    return pt.arccos(cosi)

@register_physics
def calc_sini(inc):
    return pt.sin(inc)

@register_physics
def calc_b(ar, cosi, ecc, esinw):
    return ar * cosi * (1.0 - pt.sqr(ecc)) / (1.0 - esinw)

# Stable Tc -> Tp logic
# arctan(x/y) -> arctan2(x,y)
# we multiply both x and y by sqrt(e) so we can use the step parameter directly and avoid the singularity at e=0
@register_physics
def calc_tp(ecc, sesinw, secosw, tc, n):
    E0 = 2.0 * pt.arctan2(
        pt.sqrt(1.0 - ecc) * (pt.sqrt(ecc) - sesinw),
        pt.sqrt(1.0 + ecc) * secosw
    )
    M0 = E0 - ecc * pt.sin(E0)
    return tc - M0/n