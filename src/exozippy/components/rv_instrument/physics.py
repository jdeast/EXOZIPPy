import pytensor.tensor as pt
from ...physics_registry import register_physics

@register_physics
def calc_jitter(jitter_variance):
    # Safety switch to prevent NaNs if jitter_variance is momentarily negative
    return pt.switch(pt.lt(jitter_variance, 0.0), 0.0, pt.sqrt(jitter_variance))