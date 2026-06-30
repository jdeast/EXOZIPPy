"""Shared soft-bound (log-sigmoid barrier) potentials.

One formula for every soft constraint in the codebase: ~0 inside the bound,
asymptotically linear outside, with the turn-on width set by `softness`
(default 1%) of the constraint's natural `scale`. The penalty is smooth and
its gradient is bounded by the steepness, so NUTS feels a restoring force
instead of a cliff.

The 4.4 constant sets the penalty to ~-0.01 at one turn-on width inside the
bound (log(sigmoid(4.4)) ≈ -0.012), so the barrier is negligible in the
allowed region.

Implementation note: The argument to log(sigmoid(.)) is clipped at 700
before the call.  PyTensor's piecewise log-sigmoid expansion includes an
exp(arg) branch that is never *selected* when arg > 18, but JAX still
differentiates through it in the backward pass, giving exp(820)=inf and
then 0*inf=NaN.  Capping at 700 keeps exp(arg) finite everywhere (exp(700)
~ 1e304), so the unselected branch contributes 0, not NaN, to every VJP.
"""
import numpy as np
import pytensor.tensor as pt

_MAX_ARG = 700.0  # exp(700) ~ 1e304, finite in float64; exp(710+) = inf


def _steepness(scale, softness):
    return 4.4 / (np.maximum(scale, 1e-12) * softness)


def soft_lower_bound(val, threshold, scale, softness=0.01):
    """Log-sigmoid penalty for val < threshold; ~0 for val > threshold.

    `scale` is the natural unit of the constrained quantity (e.g. 440 pc for
    a distance, 1e-5 mas for theta_E).  `softness` (default 0.01) sets the
    transition width as a fraction of scale: the penalty rises from ~0 to
    ~-4.4 nats over a distance of softness * scale on either side of the
    threshold.  Outside that window it is either negligible (allowed side) or
    grows linearly with steepness = 4.4 / (scale * softness) nats per unit
    (forbidden side).

    To tune the boundary:
      - Move where the penalty starts: change threshold.
      - Change how steeply the penalty rises per unit: change scale or softness.
        A *larger* scale (or *larger* softness) gives a gentler slope; a
        *smaller* one gives a sharper cliff.
      - The transition width is softness * scale, independently of threshold.
    """
    arg = pt.minimum((val - threshold) * _steepness(scale, softness), _MAX_ARG)
    return pt.log(pt.sigmoid(arg))


def soft_upper_bound(val, threshold, scale, softness=0.01):
    """Log-sigmoid penalty for val > threshold; ~0 for val < threshold.

    Same `scale`/`softness` semantics as soft_lower_bound (see that docstring).
    """
    arg = pt.minimum((threshold - val) * _steepness(scale, softness), _MAX_ARG)
    return pt.log(pt.sigmoid(arg))
