"""Shared soft-bound (log-sigmoid barrier) potentials.

One formula for every soft constraint in the codebase: ~0 inside the bound,
asymptotically linear outside, with the turn-on width set by `softness`
(default 1%) of the constraint's natural `scale`. The penalty is smooth and
its gradient is bounded by the steepness, so NUTS feels a restoring force
instead of a cliff.

The 4.4 constant sets the penalty to ~-0.01 at one turn-on width inside the
bound (log(sigmoid(4.4)) ≈ -0.012), so the barrier is negligible in the
allowed region.
"""
import numpy as np
import pytensor.tensor as pt


def _steepness(scale, softness):
    return 4.4 / (np.maximum(scale, 1e-12) * softness)


def soft_lower_bound(val, threshold, scale, softness=0.01):
    """Log-sigmoid penalty for val < threshold; ~0 for val > threshold."""
    return pt.log(pt.sigmoid((val - threshold) * _steepness(scale, softness)))


def soft_upper_bound(val, threshold, scale, softness=0.01):
    """Log-sigmoid penalty for val > threshold; ~0 for val < threshold."""
    return pt.log(pt.sigmoid((threshold - val) * _steepness(scale, softness)))
