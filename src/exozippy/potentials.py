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
The FORBIDDEN side has the mirror-image trap once the argument passes -709
(review 1.8.14); `_log_sigmoid` handles it without capping the penalty.
"""

import numpy as np
import pytensor.tensor as pt

_MAX_ARG = 700.0  # exp(700) ~ 1e304, finite in float64; exp(710+) = inf


def _steepness(scale, softness):
    # pt.maximum (not np.maximum) so `scale` may be a pytensor.shared vector
    # (the barrier scales are measured at startup and set in place); it
    # accepts plain numpy input just the same.
    return 4.4 / (pt.maximum(scale, 1e-12) * softness)


def _log_sigmoid(arg):
    """``log(sigmoid(arg))`` with a finite gradient on BOTH sides, under JAX.

    PyTensor rewrites ``log(sigmoid(z))`` to ``-softplus(-z)``, and its JAX
    softplus is a ``jnp.where`` cascade whose unselected ``exp(-z)`` branch
    overflows once ``z < -709``: the value stays right (the linear branch is
    selected) but the VJP multiplies that ``inf`` by zero and the gradient is
    NaN.  ``_MAX_ARG`` bounds the allowed side only, so the FORBIDDEN side
    hit this as soon as a bound was violated by more than 700 nats -- for the
    V_c/V_e real-root shield (440 nats per unit discriminant) everywhere past
    ``d = -1.6``, most of vcve's support, where prior-only numpyro chains sat
    still with NaN gradients (review 1.8.14).  The C backend never showed it.

    The fix keeps the penalty linear and the slope exact without clipping it
    flat: the sigmoid is evaluated at ``max(arg, -_MAX_ARG)``, and the
    remainder ``min(arg + _MAX_ARG, 0)`` is added back, where
    ``log sigmoid(-700) == -700`` to float precision.  For
    ``arg >= -_MAX_ARG`` the added term is exactly 0.0, so every value on
    that side is bit-identical to the plain form.
    """
    return pt.log(pt.sigmoid(pt.maximum(arg, -_MAX_ARG))) + pt.minimum(
        arg + _MAX_ARG, 0.0
    )


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
    return _log_sigmoid(arg)


def soft_upper_bound(val, threshold, scale, softness=0.01):
    """Log-sigmoid penalty for val > threshold; ~0 for val < threshold.

    Same `scale`/`softness` semantics as soft_lower_bound (see that docstring).
    """
    arg = pt.minimum((threshold - val) * _steepness(scale, softness), _MAX_ARG)
    return _log_sigmoid(arg)
