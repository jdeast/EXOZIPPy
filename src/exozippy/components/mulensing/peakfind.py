"""Built-in PSPL peak finder: t_0, u_0 and t_E from the light curves alone.

WHY THIS EXISTS.  Until this module, ``push_seed_hints`` was the ONLY source
of t_0/u_0/t_E start values, and it reads MMEXOFAST's JSON.  So
``mmexofast: false`` left those three at their ``defaults.yaml`` values --
on DC2018 event 128 that is a start 1,445 days from the event's own peak,
which no sampler recovers from -- and MMEXOFAST was therefore a hard
dependency of every microlensing fit rather than an optional accelerator
(review 8.4.9).

WHY IT IS CHEAP, which is the part that makes it worth having at all.  PSPL
flux is LINEAR in the two flux parameters once the magnification is known:

    flux(t) = f_source * A(t; t_0, u_0, t_E) + f_blend

so for any trial geometry both drop out through a 2x2 weighted
least-squares solve, and the search is 3-DIMENSIONAL rather than 5.  A
coarse grid over data-driven t_0 candidates followed by Nelder-Mead
converges on ~40,000 epochs in a few seconds.  Measured against MMEXOFAST
on the DC2018 events, this reproduces its t_0/u_0/t_E to 4-5 significant
figures wherever MMEXOFAST produced a fit at all, and produces a seed on
events where MMEXOFAST returned none (one raised NoAnomalyFoundError, one
returned an empty solution list).

WHAT IT DELIBERATELY DOES NOT DO.  It does not search for the COMPANION.
The expensive half of MMEXOFAST is the binary-lens search, and a seed does
not need it: s, alpha and q keep their ``defaults.yaml`` starts and the
sampler finds the anomaly itself.  It does not fit parallax either -- pi_E
is held at zero, matching ``run_or_load``'s own ``no_parallax=True``
default, because a seed that cannot resolve the trajectory asymmetry should
not pretend to.
"""

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Trial values for the coarse grid.  u_0 spans the high-magnification regime
# through to the unmagnified wing; t_E spans the day-scale free-floating
# candidates through to the season-long events.  Deliberately coarse: its
# only job is to land Nelder-Mead in the right t_0 basin, and a finer grid
# costs time without changing which basin wins.
_U0_GRID = (0.005, 0.02, 0.05, 0.15, 0.4, 0.8, 1.2)
_TE_GRID = (3.0, 8.0, 20.0, 50.0, 120.0)
_MAX_T0_CANDIDATES = 12
# Two t_0 candidates closer together than this descend into the same
# minimum, so keeping both just spends the grid twice on one basin.
_T0_SEPARATION_DAYS = 1.0


def _solve_fluxes(amp, flux, ivar):
    """chi2 with f_source and f_blend PROFILED OUT analytically.

    Returns (chi2, f_source, f_blend).  A singular normal matrix means the
    trial magnification is effectively constant across the data -- t_E far
    outside the baseline, or u_0 so large there is no event -- which is a
    legitimate trial to reject, hence ``inf`` rather than an exception.
    """
    s11 = float(np.sum(ivar * amp * amp))
    s10 = float(np.sum(ivar * amp))
    s00 = float(np.sum(ivar))
    b1 = float(np.sum(ivar * amp * flux))
    b0 = float(np.sum(ivar * flux))
    det = s11 * s00 - s10 * s10
    if not np.isfinite(det) or abs(det) < 1e-30:
        return np.inf, 0.0, 0.0
    f_source = (b1 * s00 - b0 * s10) / det
    f_blend = (b0 * s11 - b1 * s10) / det
    resid = flux - (f_source * amp + f_blend)
    return float(np.sum(ivar * resid * resid)), f_source, f_blend


def _total_chi2(theta, curves, mag_fn):
    """Summed chi2 over every light curve.

    Each curve gets its OWN f_source and f_blend -- different filters see a
    different source colour and a different blend -- but they SHARE the
    geometry, which is what makes a two-band fit worth more than two
    one-band fits.
    """
    t_0, u_0, t_E = theta
    if not (u_0 > 0 and t_E > 0) or not np.isfinite(t_0):
        return np.inf
    total = 0.0
    for t, flux, ivar in curves:
        chi2, _, _ = _solve_fluxes(mag_fn(t, t_0, u_0, t_E), flux, ivar)
        if not np.isfinite(chi2):
            return np.inf
        total += chi2
    return total


def _t0_candidates(curves):
    """Epochs of greatest flux excess in the densest curve, thinned.

    The chi2 surface in t_0 is a forest of narrow minima -- roughly one per
    candidate peak -- so a local optimizer started at the wrong one stays
    there.  Candidates come from the data's own excess rather than a uniform
    scan so the grid tracks where a peak could actually be; the scatter is
    an MAD so a genuine peak does not inflate its own detection threshold.
    """
    t, flux, _ = max(curves, key=lambda c: len(c[0]))
    base = float(np.median(flux))
    scatter = 1.4826 * float(np.median(np.abs(flux - base))) or 1.0
    order = np.argsort((flux - base) / scatter)[::-1]
    picked = []
    for i in order:
        ti = float(t[i])
        if all(abs(ti - p) > _T0_SEPARATION_DAYS for p in picked):
            picked.append(ti)
        if len(picked) >= _MAX_T0_CANDIDATES:
            break
    return picked or [float(0.5 * (t.min() + t.max()))]


def find_pspl_seed(curves, mag_fn=None):
    """Fit a point-lens point-source model to ``curves``.

    ``curves`` is a list of ``(time, flux, inverse_variance)`` arrays, all
    in flux units.  ``mag_fn(t, t_0, u_0, t_E)`` supplies the magnification;
    the default is the bare Paczynski curve, and callers that want the
    component's own conventions (the shared u_0 floor, observer offsets)
    should pass a partial of ``MulensInstrument._pspl_magnification``.

    Returns a dict with t_0, u_0, t_E, chi2, n_points and converged, or None
    when there is nothing fittable.
    """
    curves = [c for c in curves if len(c[0]) >= 4]
    if not curves:
        return None

    if mag_fn is None:
        def mag_fn(t, t_0, u_0, t_E):
            u2 = ((t - t_0) / t_E) ** 2 + u_0 * u_0
            return (u2 + 2.0) / np.sqrt(u2 * (u2 + 4.0))

    best_chi2, best = np.inf, None
    for t_0 in _t0_candidates(curves):
        for u_0 in _U0_GRID:
            for t_E in _TE_GRID:
                chi2 = _total_chi2((t_0, u_0, t_E), curves, mag_fn)
                if chi2 < best_chi2:
                    best_chi2, best = chi2, (t_0, u_0, t_E)
    if best is None:
        return None

    # Refine in log(u_0) and log(t_E) so the optimizer cannot step either
    # one negative -- both are positive by definition and a Nelder-Mead
    # simplex has no way to know that in the raw coordinates.
    def wrapped(p):
        return _total_chi2((p[0], np.exp(p[1]), np.exp(p[2])), curves, mag_fn)

    res = minimize(
        wrapped,
        [best[0], np.log(best[1]), np.log(best[2])],
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-3},
    )
    if not np.isfinite(res.fun) or res.fun > best_chi2:
        # The refinement made it worse (or diverged): keep the grid point.
        # Returning the refined value anyway would hand the sampler a start
        # that is demonstrably worse than one we already had.
        t_0, u_0, t_E, chi2, ok = best[0], best[1], best[2], best_chi2, False
    else:
        t_0 = float(res.x[0])
        u_0 = float(np.exp(res.x[1]))
        t_E = float(np.exp(res.x[2]))
        chi2 = float(res.fun)
        ok = bool(res.success)
    return {
        "t_0": float(t_0),
        "u_0": float(u_0),
        "t_E": float(t_E),
        "chi2": float(chi2),
        "n_points": int(sum(len(c[0]) for c in curves)),
        "converged": ok,
    }


def push_peak_find_hints(seed, config_manager, source="peak finder"):
    """Seed t_0, u_0 and t_E from ``find_pspl_seed``'s result.

    Only those three.  The companion geometry (log_s, alpha, q) and rho keep
    their ``defaults.yaml`` starts, so this is a PARTIAL seed by design and
    not by accident -- see the module docstring.  ``add_seed_hints`` puts it
    at PRECEDENCE_DERIVED_DATA, the same tier MMEXOFAST's seeds occupy: this
    is a derivation FROM THE DATA, so every user entry outranks it.
    """
    if not seed:
        return 0
    config_manager.add_seed_hints([{
        "source.0.t_0": float(seed["t_0"]),
        "source.0.u_0": float(seed["u_0"]),
        "mulensevent.0.t_E": float(seed["t_E"]),
    }])
    logger.info(
        "Peak finder (%s): t_0 = %.4f, u_0 = %.4f, t_E = %.3f d "
        "from %d epochs (chi2 = %.1f%s).  s, alpha, q and rho keep their "
        "defaults; the sampler finds the anomaly.",
        source, seed["t_0"], seed["u_0"], seed["t_E"], seed["n_points"],
        seed["chi2"], "" if seed["converged"] else ", NOT converged",
    )
    return 1
