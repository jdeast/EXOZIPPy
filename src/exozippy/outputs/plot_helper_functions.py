import numpy as np


@staticmethod
def _extend_window(window, values, margin):
    """``window`` widened so every finite value sits ``margin`` inside it.

    A value comfortably inside changes nothing; one within ``margin`` of
    an edge, or fully outside, pushes that edge out to ``value -/+
    margin``.  "Very near the bound" and "outside the bound" are the same
    comparison once the margin is folded in, which is why this is one
    expression rather than two cases.
    """
    lo, hi = float(window[0]), float(window[1])
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo = min(lo, float(vals.min()) - margin)
        hi = max(hi, float(vals.max()) + margin)
    return lo, hi


@staticmethod
def _padded_range(arrays, pad_frac):
    """``[min, max]`` over ``arrays``, with ``pad_frac`` breathing room.

    ``None`` when nothing finite is on the chart, so the caller omits the
    range key and lets the renderer autoscale -- the honest answer when
    there is nothing to scale to.  A single point (zero span) is given a
    pad from its own magnitude rather than a zero-width axis.
    """
    finite = (
        np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
        if arrays
        else np.empty(0)
    )
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    pad = pad_frac * (hi - lo)
    if pad <= 0.0:
        pad = pad_frac * max(abs(lo), 1.0)
    return lo - pad, hi + pad
