import pytensor.tensor as pt
from ...physics_registry import register_physics

# Limb-darkening physics lives on the Band component
# (calc_u1_from_kipping / calc_u2_from_kipping in band/physics.py).

@register_physics
def calc_transit_jitter(jitter_variance):
    return pt.switch(pt.lt(jitter_variance, 0.0), 0.0, pt.sqrt(jitter_variance))