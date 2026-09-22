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

WHAT IT DOES DO SINCE 2026-09-21: IT LOOKS FOR THE PRIMARY FIRST.  A
point-lens fit locks onto the BIGGEST feature in the curve, and on a wide
binary that is the planetary-caustic anomaly, not the primary peak.  On
DC2018-226 (truth t_0 = 2459958.95, u_0 = 1.11, t_E = 17.8 d; s = 4.2,
q = 7e-4) the primary's peak fell in a season gap, its declining wing was a
4-sigma-per-day excess, and the anomaly 68 days later was a 12-sigma spike
over 179 points.  Eleven of the twelve t_0 candidates sat in the anomaly's
window, the fit seeded t_0 at the anomaly's epoch, and the sampler then
modelled the anomaly as its own one-day event with a q ~ 0.03 companion
hung on it -- an honest, in-band, wrong-topology posterior that never
visited the truth's basin in 50k draws (review 2.4.14).  MMEXOFAST's
EventFinder has the same blind spot on the same event (measured).  Seeded
with the primary's PSPL instead, the same sampler found s ~ 4.2, q ~ 7e-4
(experiment A), so the primary's (t_0, u_0, t_E) is ALL the seed has to
deliver.  ``find_pspl_seed`` therefore fits once, measures the width of
the feature it found from the data, MASKS that feature, fits the rest of
the curve again, and hands back whichever of the two fits is the BROADER
significant feature -- the primary event by definition, since an anomaly is
short compared to t_E.  The anomaly's window is reported alongside so the
caller can log it; no s, alpha or q is ever derived from it (JDE
2026-09-17: no estimator anywhere -- the sampler finds the companion).
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
# The PSPL refinement has a degenerate direction: u_0 -> 0 with t_E -> inf
# keeps the peak's SHAPE while the linear flux solve absorbs the amplitude,
# and on a low-amplitude event the chi2 surface slopes gently that way, so
# an unbounded Nelder-Mead walks off to a source parked on the lens for the
# whole light curve.  Two of the 44 DC2018 events (001 and 226) came back
# with u_0 = 5e-10 and t_E = 1e7-4e8 DAYS -- a "seed" whose start logp was
# -2.5e12 / -1.3e9, whose evaluation costs 15 s (a finite source on the
# point-lens singularity at every epoch), and whose polish then spent ten
# hours in a point-lens basin with the errors inflated 400x (review 2.4.14).
# A refinement that lands outside these is thrown away in favour of the
# grid point it started from, which is a real fit to the data.
_U0_MIN = 1e-4  # below this the source is on the lens: no PSPL peak shape
_U0_MAX = 10.0  # above this the magnification is unity: no event at all
_TE_SPAN_FACTOR = 2.0  # t_E longer than twice the data span is unmeasured
# Two t_0 candidates closer together than this descend into the same
# minimum, so keeping both just spends the grid twice on one basin.
_T0_SEPARATION_DAYS = 1.0

# The primary-first pass (module docstring).  A feature's width is measured
# from the data as the FWHM of its binned flux excess; the bins are a few
# cadences wide so a single noisy epoch cannot end the walk, with a floor so
# a dense curve does not bin at the noise scale.
_FEATURE_BIN_CADENCES = 10
_FEATURE_MIN_BIN_DAYS = 0.05
_FEATURE_MIN_POINTS_PER_BIN = 3
# The mask is this many half-widths either side of the feature, so the
# anomaly's wings (and the caustic-approach shoulders a planetary bump has)
# do not leak into the second fit; and at least this many bins even when
# the feature is unresolved.
_FEATURE_PAD = 3.0
_FEATURE_MIN_HALF_BINS = 2
# The second fit has to beat a FLAT baseline on the masked data by this much
# to count as a second feature at all.  A 3-parameter fit to pure noise
# over a dozen t_0 candidates gains a few tens; the DC2018-226 primary's
# wing alone gains ~2,500.
_SECOND_FEATURE_MIN_DCHI2 = 100.0
# ... and be this many times BROADER than the feature it was found behind
# to be called the primary.  On 226 the ratio is ~40 (a 1 d anomaly against
# a 41 d FWHM primary); a genuine PSPL whose masked refit picked up a noise
# bump elsewhere fails the significance cut long before this one.
_PRIMARY_WIDTH_RATIO = 3.0
# When the free refinement runs into the degeneracy, the fallback refits
# (t_0, t_E) at each grid u_0 instead of returning the raw grid point.  Fits
# within this chi2 of the best are indistinguishable -- a peak that fell in
# a gap constrains the WING, which fixes t_0 and t_E for any u_0 but not
# u_0 itself -- and among them the LARGEST u_0 is reported: the least
# magnified reading of an unobserved peak is the a priori likeliest, and
# the high-magnification alternative is the one that costs a sampler most
# to walk away from.
_WING_DEGENERACY_DCHI2 = 4.0


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


def refinement_is_degenerate(u_0, t_E, span, min_t_E=0.0):
    """True when a refined (u_0, t_E) is the degenerate non-peak (see _U0_MIN).

    ``min_t_E`` is the data's cadence when the caller knows it: an event
    shorter than the spacing of its own epochs has no measured shape, and a
    (t_0, t_E) refit at fixed u_0 can walk there as freely as the free one
    walks to t_E -> inf.
    """
    return bool(
        not np.isfinite(u_0)
        or not np.isfinite(t_E)
        or u_0 < _U0_MIN
        or u_0 > _U0_MAX
        or (span > 0 and t_E > _TE_SPAN_FACTOR * span)
        or (min_t_E > 0 and t_E < min_t_E)
    )


def _paczynski(t, t_0, u_0, t_E):
    u2 = ((t - t_0) / t_E) ** 2 + u_0 * u_0
    return (u2 + 2.0) / np.sqrt(u2 * (u2 + 4.0))


def _fit_pspl(curves, mag_fn):
    """One grid + Nelder-Mead PSPL fit to ``curves`` (see find_pspl_seed)."""
    curves = [c for c in curves if len(c[0]) >= 4]
    if not curves:
        return None

    best_chi2, best = np.inf, None
    per_u0 = {u_0: (np.inf, None) for u_0 in _U0_GRID}
    for t_0 in _t0_candidates(curves):
        for u_0 in _U0_GRID:
            for t_E in _TE_GRID:
                chi2 = _total_chi2((t_0, u_0, t_E), curves, mag_fn)
                if chi2 < best_chi2:
                    best_chi2, best = chi2, (t_0, u_0, t_E)
                if chi2 < per_u0[u_0][0]:
                    per_u0[u_0] = (chi2, (t_0, t_E))
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
    span = float(
        max(c[0].max() for c in curves) - min(c[0].min() for c in curves)
    )
    dense_t = np.sort(max(curves, key=lambda c: len(c[0]))[0])
    cadence = float(np.median(np.diff(dense_t))) if dense_t.size > 1 else 0.0
    ref_u0, ref_tE = float(np.exp(res.x[1])), float(np.exp(res.x[2]))
    degenerate = refinement_is_degenerate(ref_u0, ref_tE, span, cadence)
    if not np.isfinite(res.fun) or res.fun > best_chi2:
        # The refinement made it worse (or diverged): keep the grid point.
        # Returning the refined value anyway would hand the sampler a start
        # that is demonstrably worse than one we already had.
        t_0, u_0, t_E, chi2, ok = best[0], best[1], best[2], best_chi2, False
    elif degenerate:
        # A LOWER chi2 that is not a peak: see _U0_MIN above.  The free
        # refinement is discarded and (t_0, t_E) are refit at each grid
        # u_0 instead (_WING_DEGENERACY_DCHI2): on a peak that fell in a
        # gap the wing fixes t_0 to a day or two for ANY u_0 (DC2018-226:
        # t_0 within 2.5 d of the truth at every grid u_0, chi2 within 1.5
        # of each other, while the raw grid point sat 30 d off at the
        # season's first epoch), and the caller is told, because a seed
        # this wrong does not fail loudly downstream -- it fails ten hours
        # later.
        t_0, u_0, t_E, chi2, ok = _refit_at_grid_u0(
            per_u0, curves, mag_fn, span, cadence
        )
        logger.warning(
            "Peak finder: the PSPL refinement ran into the u_0 -> 0, "
            "t_E -> inf degeneracy (u_0 = %.3g, t_E = %.3g d against a "
            "%.1f d data span) and is discarded; refitting t_0 and t_E at "
            "each grid u_0 gives t_0 = %.4f, u_0 = %.3g, t_E = %.3g d "
            "(u_0 itself is NOT constrained by these data: the largest "
            "u_0 within %.0f chi2 of the best is reported).  This is what "
            "a low-amplitude, gap-peaked or anomaly-dominated event looks "
            "like to a point-lens fit: expect the sampler to need a "
            "companion search, not a longer polish.",
            ref_u0,
            ref_tE,
            span,
            t_0,
            u_0,
            t_E,
            _WING_DEGENERACY_DCHI2,
        )
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


def _refit_at_grid_u0(per_u0, curves, mag_fn, span, cadence):
    """(t_0, t_E) refit at each grid u_0; the fallback for a degenerate
    free refinement.  Returns (t_0, u_0, t_E, chi2, converged).  A refit
    that is itself degenerate, or worse than its grid point, contributes
    the grid point, so when nothing refines the result is the best grid
    point exactly as before."""
    fits = []
    for u_0, (grid_chi2, start) in per_u0.items():
        if start is None or not np.isfinite(grid_chi2):
            continue
        t0g, tEg = start

        def wrapped(p, u_0=u_0):
            return _total_chi2((p[0], u_0, np.exp(p[1])), curves, mag_fn)

        res = minimize(
            wrapped,
            [t0g, np.log(tEg)],
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-3},
        )
        t_E = float(np.exp(res.x[1]))
        if not np.isfinite(res.fun) or res.fun > grid_chi2:
            fits.append((grid_chi2, t0g, u_0, tEg, False))
        elif refinement_is_degenerate(u_0, t_E, span, cadence):
            fits.append((grid_chi2, t0g, u_0, tEg, False))
        else:
            fits.append((float(res.fun), float(res.x[0]), u_0, t_E, bool(res.success)))
    best = min(f[0] for f in fits)
    within = [f for f in fits if f[0] <= best + _WING_DEGENERACY_DCHI2]
    chi2, t_0, u_0, t_E, ok = max(within, key=lambda f: f[2])
    return t_0, u_0, t_E, chi2, ok


def _observed_width(seed, curves, mag_fn):
    """Span of the observed epochs over which ``seed``'s model excess is
    at least half its OBSERVED maximum, on the densest curve.

    The model's own FWHM would be the natural width, but on a peak that
    fell in a gap the wing leaves u_0 unconstrained (see
    _WING_DEGENERACY_DCHI2) and the model FWHM tracks that arbitrary u_0
    -- 0.5 d at u_0 = 0.005, 37 d at u_0 = 1.2, for fits of identical chi2
    on DC2018-226.  The width of the excess the DATA actually carry does
    not: ~11 d for every one of those fits.
    """
    t = max(curves, key=lambda c: len(c[0]))[0]
    amp = np.asarray(mag_fn(t, seed["t_0"], seed["u_0"], seed["t_E"])) - 1.0
    top = float(np.nanmax(amp)) if amp.size else 0.0
    if not np.isfinite(top) or top <= 0:
        return 0.0
    sel = amp >= 0.5 * top
    return float(t[sel].max() - t[sel].min())


def _flat_chi2(curves):
    """chi2 of a constant flux per curve: the no-event baseline."""
    total = 0.0
    for _t, flux, ivar in curves:
        w = float(np.sum(ivar))
        if w <= 0:
            continue
        mean = float(np.sum(ivar * flux)) / w
        total += float(np.sum(ivar * (flux - mean) ** 2))
    return total


def feature_window(curves, t_peak):
    """Data-driven extent of the flux feature at ``t_peak``.

    On the densest curve, the flux excess over the median baseline (in
    units of the MAD scatter) is binned a few cadences wide, and the walk
    outward from the peak's bin stops at the first bin whose mean excess
    falls below HALF the peak bin's: a full width at half maximum measured
    from the data, not from a model, because the model that found the
    feature may be the degenerate grid point (a 1 d anomaly "fit" with
    t_E = 50 d) and says nothing about the feature's real width.  A gap in
    the data ends the walk on that side.

    Returns ``(lo, hi, fwhm)``: the mask window, ``_FEATURE_PAD`` half-widths
    either side of the peak with a floor of ``_FEATURE_MIN_HALF_BINS`` bins,
    and the measured width itself.
    """
    t, flux, _ivar = max(curves, key=lambda c: len(c[0]))
    order = np.argsort(t)
    t, flux = t[order], flux[order]
    base = float(np.median(flux))
    scatter = 1.4826 * float(np.median(np.abs(flux - base))) or 1.0
    excess = (flux - base) / scatter
    cadence = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    dt = max(_FEATURE_MIN_BIN_DAYS, _FEATURE_BIN_CADENCES * cadence)

    # Bin k covers [t_peak + k dt, t_peak + (k+1) dt); the peak straddles
    # bins -1 and 0.
    k_all = np.floor((t - t_peak) / dt).astype(int)
    means = {}
    for k in np.unique(k_all):
        sel = k_all == k
        if sel.sum() >= _FEATURE_MIN_POINTS_PER_BIN:
            means[int(k)] = float(np.mean(excess[sel]))
    peak = max(means.get(-1, -np.inf), means.get(0, -np.inf))
    if not np.isfinite(peak) or peak <= 0:
        half_bins = _FEATURE_MIN_HALF_BINS
        return t_peak - half_bins * dt, t_peak + half_bins * dt, 2 * half_bins * dt
    half = 0.5 * peak

    def walk(step):
        k = 0 if step > 0 else -1
        n = 0
        while True:
            m = means.get(k)
            if m is None or m < half:
                break
            n += 1
            k += step
        return n

    right = max(walk(+1), 1) * dt
    left = max(walk(-1), 1) * dt
    floor = _FEATURE_MIN_HALF_BINS * dt
    lo = t_peak - max(_FEATURE_PAD * left, floor)
    hi = t_peak + max(_FEATURE_PAD * right, floor)
    return float(lo), float(hi), float(left + right)


def _mask_window(curves, lo, hi):
    out = []
    for t, flux, ivar in curves:
        keep = (t < lo) | (t > hi)
        if keep.sum() >= 4:
            out.append((t[keep], flux[keep], ivar[keep]))
    return out


def find_pspl_seed(curves, mag_fn=None, primary_first=True):
    """Fit a point-lens point-source model to ``curves``.

    ``curves`` is a list of ``(time, flux, inverse_variance)`` arrays, all
    in flux units.  ``mag_fn(t, t_0, u_0, t_E)`` supplies the magnification;
    the default is the bare Paczynski curve, and callers that want the
    component's own conventions (the shared u_0 floor, observer offsets)
    should pass a partial of ``MulensInstrument._pspl_magnification``.

    With ``primary_first`` (the default; see the module docstring) the
    feature the first fit found is masked and the rest of the curve is fit
    again.  When that second fit is a significant feature of its own AND at
    least ``_PRIMARY_WIDTH_RATIO`` times broader than the first, the first
    was an anomaly riding on the second, and the second is returned as the
    seed with the first's window under ``"anomaly"``.  Otherwise the first
    fit is returned unchanged and ``"anomaly"`` is None.

    Returns a dict with t_0, u_0, t_E, chi2, n_points, converged and
    anomaly, or None when there is nothing fittable.
    """
    if mag_fn is None:
        mag_fn = _paczynski
    first = _fit_pspl(curves, mag_fn)
    if first is None:
        return None
    first["anomaly"] = None
    if not primary_first:
        return first

    usable = [c for c in curves if len(c[0]) >= 4]
    lo, hi, fwhm_1 = feature_window(usable, first["t_0"])
    masked = _mask_window(usable, lo, hi)
    second = _fit_pspl(masked, mag_fn) if masked else None
    if second is None or lo <= second["t_0"] <= hi:
        # The rest of the curve fits the same feature from its wings: one
        # event, and the first fit is it.
        return first
    dchi2 = _flat_chi2(masked) - second["chi2"]
    if not np.isfinite(dchi2) or dchi2 < _SECOND_FEATURE_MIN_DCHI2:
        return first
    # The first feature's width is measured from the data by construction
    # (its t_0 candidates ARE epochs of peak excess); the second's peak may
    # lie in a gap -- 226's primary peaked between seasons -- so its width
    # is the span of observed epochs its model lights up (_observed_width).
    width_2 = _observed_width(second, masked, mag_fn)
    if not (width_2 > _PRIMARY_WIDTH_RATIO * fwhm_1):
        return first
    logger.warning(
        "Peak finder: the strongest feature (t_0 = %.4f, FWHM %.2f d) is "
        "an ANOMALY riding on a broader event; with it masked "
        "(%.4f..%.4f) the rest of the curve fits a PSPL at t_0 = %.4f, "
        "u_0 = %.3g, t_E = %.3g d (FWHM %.1f d, delta chi2 %.0f over a "
        "flat baseline), which is the primary and is the seed.  s, alpha "
        "and q stay generic: the sampler finds the companion.",
        first["t_0"],
        fwhm_1,
        lo,
        hi,
        second["t_0"],
        second["u_0"],
        second["t_E"],
        width_2,
        dchi2,
    )
    second["anomaly"] = {
        "t_0": first["t_0"],
        "window": (float(lo), float(hi)),
        "fwhm": float(fwhm_1),
        "u_0": first["u_0"],
        "t_E": first["t_E"],
        "chi2": first["chi2"],
    }
    return second


T_0_PATH = "source.0.t_0"


def t_0_is_already_available(config_manager):
    """True when t_0 is named outright or derivable from what is.

    WHY THIS AND NOT `user_hints_sufficient`, which is the obvious choice
    and was the first one used here.  That function asks whether EVERY
    observable this topology needs is available -- t_0, u_0, t_E, plus rho
    and s/alpha/q where they apply -- which is far too strong a trigger for
    a peak finder.  A config that names t_0 and u_0 and legitimately DERIVES
    t_E from the galactic model's kinematics fails it, because t_E comes
    from theta_E and mu_rel and mu_rel comes from the proper motions.  The
    peak finder then fired and supplied a PSPL t_E in place of the
    kinematic one -- which is exactly what tests/test_seed_quality.py's
    multi-source case measures, and it changed that measurement
    (chi2/N 6.94 with the galactic seed).

    t_0 is the right question because t_0 is the only one whose default is
    UNRECOVERABLE: on DC2018-128 `defaults.yaml` puts it 1,445 days from the
    event's own peak, where the likelihood is flat and no sampler returns.
    A wrong-but-finite u_0 or t_E start is a slow fit; a wrong t_0 is not a
    fit at all.  So the finder earns its keep precisely when t_0 is absent,
    and has no business overriding a model that is already answering.
    """
    entry = config_manager.user_params.get(T_0_PATH)
    if isinstance(entry, dict) and (
        entry.get("initval") is not None or entry.get("mu") is not None
    ):
        return True
    try:
        return T_0_PATH in config_manager.probe_derivable([T_0_PATH])
    except Exception:  # noqa: BLE001
        # A probe that cannot run is not evidence that t_0 is available, and
        # guessing "available" here would silently disable the finder.
        return False


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
    config_manager.add_seed_hints(
        [
            {
                "source.0.t_0": float(seed["t_0"]),
                "source.0.u_0": float(seed["u_0"]),
                "mulensevent.0.t_E": float(seed["t_E"]),
            }
        ]
    )
    logger.info(
        "Peak finder (%s): t_0 = %.4f, u_0 = %.4f, t_E = %.3f d "
        "from %d epochs (chi2 = %.1f%s).  s, alpha, q and rho keep their "
        "defaults; the sampler finds the anomaly.",
        source,
        seed["t_0"],
        seed["u_0"],
        seed["t_E"],
        seed["n_points"],
        seed["chi2"],
        "" if seed["converged"] else ", NOT converged",
    )
    anomaly = seed.get("anomaly")
    if anomaly:
        logger.info(
            "Peak finder (%s): a %.2f d anomaly at t_0 = %.4f was masked "
            "(%.4f..%.4f) to find that primary.",
            source,
            anomaly["fwhm"],
            anomaly["t_0"],
            anomaly["window"][0],
            anomaly["window"][1],
        )
    return 1
