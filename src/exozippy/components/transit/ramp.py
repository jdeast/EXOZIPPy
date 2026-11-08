"""Start values for a transit file's ``fitramp:`` exponential ramp.

EXOFASTv2's FITRAMP (exofast_chi2v2.pro, "exponential ramp (e.g., Spitzer &
JWST)") multiplies a file's model by

    R(t) = 1 + A * exp((t0 - t) / tau) - A
         = 1 - A * (1 - exp(-(t - t0) / tau))

so R = 1 at the file's first exposure and levels off at 1 - A.  The MODEL is
built in ``Transit._lc_model``; this module only measures where its two
parameters start.  It is a numpy least-squares fit to the data, never part
of the likelihood, so it does not have to match the model's op sequence --
only its shape.

What is deliberately NOT ported: EXOFASTv2's own seed (mkss.pro's
``curvefit`` of ``ramp_func``).  ``ramp_func`` evaluates ``exp(a1 * x)`` on
``x = t0 - t`` and steers toward ``a1 < 0``, i.e. ``exp(+(t - t0) / tau)``,
a GROWING exponential, while the chi2 model decays; it also omits the
``-A`` and F0.  So its start values come from a different shape than the
one that is fit.  See ``components/instrument.md`` ("fitramp").
"""

import numpy as np

# The ramp time constant's upper bound, in units of the file's time span
# (EXOFASTv2's ``rampexp gt transit.timerange*10d0`` rejection).
TAU_SPAN_FACTOR = 10.0

# The ramp time constant's lower bound in days: defaults.yaml's
# ``ramp_tau.lower`` (tests/test_transit_fitramp.py pins the two equal).
TAU_FLOOR = 1.0e-5

# Number of trial time constants in the seed's log-spaced grid.
_N_TAU = 60

# Low-outlier clip for the seed (in robust sigma): drops the transit itself,
# which otherwise biases the fitted late-time level.
_CLIP_SIGMA = 3.0
_N_CLIP = 2


def seed_ramp(time, flux, err, tau_floor):
    """Start values ``(baseline, A, tau)`` for one file's ramp.

    For each trial ``tau`` on a log grid from ``max(2 * cadence, 2 *
    tau_floor)`` to half the upper bound, the model ``c0 + c1 * e(t)`` with
    ``e(t) = exp(-(t - t0) / tau)`` is LINEAR in ``(c0, c1)``, so it is a
    weighted least-squares solve; the ``tau`` with the lowest chi2 wins.
    The fitted curve equals ``B * (1 - A + A * e)``, so ``B = c0 + c1`` is
    the flux at ``t0`` (what ``baseline`` means under the ``-A`` form) and
    ``A = c1 / B``.  Low outliers (the transit) are clipped between passes.

    ``time`` need not be sorted; ``t0`` is its minimum, as in the model.
    The returned ``A`` is clipped to ``[-0.9, 0.9]`` and ``tau`` lies
    strictly inside ``(tau_floor, TAU_SPAN_FACTOR * span)``.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    # Sign: the ramp DECAYS from t0, exp(-(t - t0) / tau) with tau > 0 --
    # the chi2 model's sign.  EXOFASTv2's ramp_func seeds with the OPPOSITE
    # sign (a growing exponential, an EXOFASTv2 bug); do not "match
    # EXOFASTv2" by flipping it.
    dt = time - time.min()
    span = float(dt.max())
    steps = np.diff(np.sort(time))
    steps = steps[steps > 0]
    cadence = float(np.median(steps))
    tau_lo = max(2.0 * cadence, 2.0 * tau_floor)
    tau_hi = 0.5 * TAU_SPAN_FACTOR * span
    taus = np.geomspace(tau_lo, max(tau_hi, 1.01 * tau_lo), _N_TAU)

    keep = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    best = None
    for _ in range(_N_CLIP + 1):
        w = 1.0 / err[keep] ** 2
        best = None
        for tau in taus:
            design = np.column_stack(
                [np.ones(keep.sum()), np.exp(-dt[keep] / tau)]
            )
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(
                design * sw[:, None], flux[keep] * sw, rcond=None
            )
            chi2 = float(np.sum(w * (flux[keep] - design @ coef) ** 2))
            if best is None or chi2 < best[0]:
                best = (chi2, coef, tau)
        _, (c0, c1), tau = best
        resid = flux - (c0 + c1 * np.exp(-dt / tau))
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep])))
        sigma = 1.4826 * mad if mad > 0 else np.median(err[keep])
        keep = keep & (resid > -_CLIP_SIGMA * sigma)

    _, (c0, c1), tau = best
    baseline = float(c0 + c1)
    amp = float(np.clip(c1 / baseline, -0.9, 0.9))
    return baseline, amp, float(tau)
