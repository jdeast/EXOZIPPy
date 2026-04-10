import pytensor.tensor as pt
from ...physics_registry import register_physics

@register_physics
def calc_transit_jitter(jitter_variance):
    return pt.switch(pt.lt(jitter_variance, 0.0), 0.0, pt.sqrt(jitter_variance))

@register_physics
def calc_u1(q1, q2):
    return 2.0 * pt.sqrt(q1) * q2

@register_physics
def calc_u2(q1, q2):
    return pt.sqrt(q1) * (1.0 - 2.0 * q2)