import pytensor.tensor as pt
from exozippy.physics_registry import register_physics


@register_physics
def calc_u1_from_kipping(q1, q2):
    """Kipping (2013): u1 = 2*sqrt(q1)*q2"""
    return 2.0 * pt.sqrt(q1) * q2


@register_physics
def calc_u2_from_kipping(q1, q2):
    """Kipping (2013): u2 = sqrt(q1)*(1 - 2*q2)"""
    return pt.sqrt(q1) * (1.0 - 2.0 * q2)
