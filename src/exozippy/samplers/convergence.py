"""Post-hoc burn-in detection and convergence diagnostics.

Burn-in is a post-processing step here, not an a-priori guess: the initial
transient is trimmed AFTER sampling and recomputed before every
convergence check, following the EXOFASTv2 getburnndx philosophy
(github.com/jdeast/EXOFASTv2/blob/master/getburnndx.pro). Two design
choices matter:

  * Detection is done in PARAMETER space (rank split-Rhat / bulk-ESS), not
    in lp/chi2 space. A likelihood-flat degenerate direction (e.g. the
    microlensing mass-distance-proper-motion ridge) reaches its stationary
    lp almost immediately while the parameters themselves drift for a large
    fraction of the run; an lp-threshold detector -- which is what
    getburnndx used -- then declares "converged" and under-trims. Because
    rank-normalized Rhat and bulk-ESS are invariant under the monotonic
    raw->physical transform, a caller may pass either representation and get
    identical answers (PTDE passes raw; run.py passes physical).

  * The burn-in is chosen to MAXIMIZE the worst parameter's bulk-ESS of the
    trimmed trace, not to first-cross an Rhat threshold. Trimming more
    lowers Rhat (removes transient) but also lowers total ESS (fewer draws);
    a slowly drifting transient has a huge autocorrelation time and so
    suppresses bulk-ESS, which then jumps up once the drift is trimmed away
    and falls again as further trimming just discards good samples. The
    worst-parameter-ESS peak is that knee. Thresholds are used only to
    DECIDE early stopping, never to discard a finished run: an almost-
    converged fit is reported with a loud warning and left to the user.

Stuck chains (getburnndx's other job) are dropped before any statistic:
a chain whose lp never reached the good-likelihood region never found the
mode and would inflate between-chain variance. As in getburnndx, if fewer
than a few chains survive we keep them all rather than trust a tiny
survivor set.
"""

import logging

import arviz as az
import numpy as np

logger = logging.getLogger(__name__)

# Coarse grid of candidate burn-in fractions. Fine enough to locate the ESS
# knee, coarse enough that the scan is ~10 statistic evaluations. Capped at
# 0.75 so at least a quarter of the trace always survives (bulk-ESS needs
# samples); the maximize-ESS rule rarely wants to trim that far anyway.
_BURNIN_FRACS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.75)

# Thin the retained draws to at most this many per chain before computing
# Rhat/ESS. The estimates are stable well below the raw draw count, and this
# bounds the cost of a check independently of how long the run is.
_STAT_DRAW_BUDGET = 2000

# Below this many retained draws per chain, Rhat/ESS are too noisy to trust.
_MIN_DRAWS_PER_CHAIN = 16

# getburnndx: do not drop chains if fewer than this many would remain good.
_MIN_GOOD_CHAINS = 3


def default_var_names(posterior, exclude=()):
    """Physical, non-index variables worth judging convergence on.

    ``exclude`` names variables to leave out: the REPORT-ONLY
    Deterministics (review 2.6.14, ``System.report_only_labels`` /
    ``trace_meta.report_only_vars``), which are in the trace so a dead
    wrap-up cannot lose them, not so they move the burn-in this scan picks.

    Drops the integer ``mode`` label (not a sampled parameter) and any
    ``*_raw`` unconstrained variable that ALSO has a physical partner in the
    same trace (they are rank-identical, so the raw copy is redundant). When
    only the raw copy exists -- as in PTDE's live raw-only store, where every
    key ends in ``_raw`` -- the raw variables are kept, otherwise nothing
    would be left to judge. ``posterior`` is a mapping name -> ndarray or an
    xarray Dataset.
    """
    names = list(getattr(posterior, "data_vars", posterior))
    present = set(names)
    out = []
    exclude = set(exclude)
    for v in names:
        if v == "mode" or v in exclude:
            continue
        if v.endswith("_raw") and v[: -len("_raw")] in present:
            continue
        out.append(v)
    return out


def good_chain_mask(lp):
    """Boolean mask of chains that reached the good-likelihood region.

    getburnndx criterion: the best chain is the one attaining the highest
    single lp; the health threshold is that chain's own median lp; a chain
    is "good" if its maximum lp ever crossed the threshold (it found the
    mode at least once). If fewer than ``_MIN_GOOD_CHAINS`` pass, every
    chain is kept rather than trusting a too-small survivor set.

    Parameters
    ----------
    lp : ndarray, shape (n_chains, n_draws)

    Returns
    -------
    (mask, reliable) : (ndarray[bool], bool)
        ``reliable`` is False when the <min-good fallback kept every chain.
    """
    lp = np.asarray(lp, dtype=float)
    n_chains = lp.shape[0]
    if lp.ndim != 2 or n_chains < 2:
        return np.ones(n_chains, dtype=bool), True
    chain_max = np.nanmax(lp, axis=1)
    best = int(np.nanargmax(chain_max))
    threshold = np.nanmedian(lp[best])
    mask = chain_max >= threshold  # the best chain always passes
    if int(mask.sum()) < min(_MIN_GOOD_CHAINS, n_chains):
        return np.ones(n_chains, dtype=bool), False
    return mask, True


def _rhat_ess(sub):
    """max rank split-Rhat and min bulk-ESS over the variables in ``sub``.

    ``sub`` maps var name -> ndarray (n_chains, n_draws, *shape). Returns
    (max_rhat, min_ess, worst_rhat_var, worst_ess_var) or None if the
    statistics could not be formed (too few draws/chains, all-NaN).
    """
    try:
        idata = az.from_dict({"posterior": sub})
        rhat_ds = az.rhat(idata)
        ess_ds = az.ess(idata)
    except Exception:
        return None

    max_rhat, worst_rhat = -np.inf, None
    for name, da in rhat_ds.data_vars.items():
        vals = da.values
        if vals.size and not np.all(np.isnan(vals)):
            m = float(np.nanmax(vals))
            if m > max_rhat:
                max_rhat, worst_rhat = m, name

    min_ess, worst_ess = np.inf, None
    for name, da in ess_ds.data_vars.items():
        vals = da.values
        if vals.size and not np.all(np.isnan(vals)):
            m = float(np.nanmin(vals))
            if m < min_ess:
                min_ess, worst_ess = m, name

    if not np.isfinite(max_rhat) or not np.isfinite(min_ess):
        return None
    return max_rhat, min_ess, worst_rhat, worst_ess


def _thin_stride(n_draws):
    return max(1, n_draws // _STAT_DRAW_BUDGET)


# --- Ensemble-mean drift (review 1.4.4) -----------------------------------
#
# Rhat compares chains with each other, so it is blind to a drift that every
# chain makes TOGETHER.  In a differential-evolution population that is the
# normal way to be unconverged: every proposal is built from the other
# chains, so the population translates along a curved ridge as one body and
# the between-chain variance stays ~0 the whole way.  Measured on ob140939
# (sync ptde, 24 rungs): the min_ess/max_rhat early stop fired at 3995 of
# 50000 draws with max Rhat 1.009 while the median lens distance was still
# climbing 2426 -> 2840 pc; full-length runs put it at 3.3-4.7 kpc.
#
# The test asks the question Rhat cannot: is the ENSEMBLE MEAN of each
# sampled element, and of lp, stationary across the tail?  See
# :func:`ensemble_drift` for the statistic and samplers.md, "The early stop
# also requires a stationary ensemble mean", for the calibration.

# Family-wise false-alarm rate of ONE drift check, Bonferroni-split over the
# elements tested.  A false alarm costs one more check interval (~11% more
# draws, _convergence_check_schedule) during a live run, and a spurious
# "NOT CONVERGED" line at wrap-up; 1% keeps both rare while the drifts this
# exists for sit far above the threshold (ob140939: |z| ~ 10-40).
DRIFT_FAMILY_ALPHA = 0.01

# Practical-significance floor: a drift smaller than this many posterior
# standard deviations across the whole tail is not reported as one.  Without
# it the test's power grows without bound with the run length, and a long
# run would be held for a statistically real but immaterial wander (a tenth
# of a sigma moves no reported median by more than its own rounding).
DRIFT_MIN_SD = 0.1

# The ensemble-mean series is block-averaged to at most this many points
# before the regression and the IACT.  Block means keep every draw's
# information (the series mean and the fitted slope are unchanged), and the
# IACT of the block-mean series is the right inflation factor for ITS mean
# (outputs/autocorr.py) -- this only bounds the cost of a check.
_DRIFT_SERIES_BUDGET = 2000

# Fewer block means than this cannot carry a slope and an IACT.
_DRIFT_MIN_POINTS = 16

# A series with fewer effectively independent points than this (n / tau of
# its detrended ensemble mean) is UNRESOLVED: its IACT, and so its z, cannot
# be trusted (per-test false alarms of 5-20% at a nominal 1% below it), so
# it does not count as stationary.  Above it the calibrated test is within
# 1-2.6% at a nominal 1% (samplers.md).
_DRIFT_MIN_NEFF = 20.0

# Small-sample inflation of z**2 on a short window: 1 + SCALE/(n_eff -
# OFFSET), fitted to stationary AR(1) at n/tau = 5..100 (see _drift_one).
_DRIFT_BIAS_SCALE = 12.0
_DRIFT_BIAS_OFFSET = 7.5


def _block_mean_series(m, budget=_DRIFT_SERIES_BUDGET):
    """Average a 1-D series into <= ``budget`` contiguous blocks.

    Any remainder is dropped from the START of the series, so the latest
    draws -- the ones a stop decision is about -- are always kept.
    """
    n = m.shape[0]
    b = max(1, -(-n // budget))  # ceil(n / budget)
    k = n // b
    return m[n - k * b :].reshape(k, b, *m.shape[1:]).mean(axis=1)


def _drift_one(series):
    """Slope test on one ensemble-mean series.

    Returns ``(z, drift, p, n_eff)``.  ``drift`` is the fitted linear change
    of the series across its whole length and ``z`` the slope over its
    Monte Carlo standard error: the OLS error with the residual variance
    inflated by the residuals' integrated autocorrelation time ``tau``
    (Geyer, ``outputs.autocorr.iact``) -- the long-run (spectral density at
    zero) variance of the noise, which is the large-sample variance of a
    regression slope on a smooth regressor under serially correlated noise.
    ``n_eff = n / tau`` is the number of effectively independent points.

    Two small-sample corrections, both calibrated on stationary AR(1) and on
    a two-timescale mixture (samplers.md has the table).  On a window only a
    few tens of ``tau`` long, Geyer's truncation underestimates ``tau`` (at
    a true n/tau = 5 it reports ~12) and detrending removes low-frequency
    power, so ``z**2`` averages ``1 + 12 / (n_eff - 7.5)`` rather than 1;
    ``z`` is divided by the square root of that.  And the reference
    distribution is a Student t with ``n_eff - 2`` degrees of freedom, not
    a normal, for the heavy tail of a ratio with a noisy denominator.
    Neither correction is trusted below ``_DRIFT_MIN_NEFF``: the caller
    treats such a series as UNRESOLVED (stationarity not shown), never as
    stationary.
    """
    from scipy.stats import t as student_t

    from exozippy.outputs.autocorr import iact

    n = series.shape[0]
    t = np.arange(n, dtype=float)
    t -= t.mean()
    sxx = float((t * t).sum())
    y = series - series.mean()
    beta = float((t * y).sum() / sxx)
    resid = y - beta * t
    var_r = float(resid.var())
    drift = beta * (n - 1)
    if var_r == 0.0:
        # An exact straight line: the slope is known without error.
        if drift == 0.0:
            return 0.0, drift, 1.0, float(n)
        return float(np.sign(drift) * np.inf), drift, 0.0, float(n)
    tau = iact(resid)
    n_eff = n / tau
    z = beta / np.sqrt(var_r * tau / sxx)
    if n_eff > _DRIFT_BIAS_OFFSET:
        z /= np.sqrt(1.0 + _DRIFT_BIAS_SCALE / (n_eff - _DRIFT_BIAS_OFFSET))
    p = 2.0 * student_t.sf(abs(z), max(n_eff - 2.0, 1.0))
    return float(z), drift, float(p), float(n_eff)


def ensemble_drift(
    posterior,
    lp=None,
    var_names=None,
    family_alpha=DRIFT_FAMILY_ALPHA,
    min_drift_sd=DRIFT_MIN_SD,
):
    """Is the ENSEMBLE MEAN of every element, and of lp, stationary?

    The input is the window to judge (already trimmed to good chains and to
    the tail): ``posterior`` maps var name -> ndarray (n_chains, n_draws,
    *shape), ``lp`` is (n_chains, n_draws) or None.

    WHY THE ENSEMBLE MEAN.  A population sampler is ONE Markov chain whose
    state is the whole population (outputs.md, "Mode-weight errors: the
    population, not the chains"); a collective drift is invisible to Rhat
    (no between-chain spread) and to per-chain ESS (the drift is shared), but
    it is the first thing the chain-average ``m_t = mean_c x_{c,t}`` shows.
    For independent chains (NUTS) the same series is merely a less noisy
    view of each chain's own trend, so the test is valid there too.

    THE STATISTIC.  For each scalar element: block-average ``m_t`` to at most
    ``_DRIFT_SERIES_BUDGET`` points, fit a straight line, and divide the
    slope by its Monte Carlo error -- the OLS error with the residual
    variance inflated by the residuals' IACT (:func:`_drift_one`).  Two
    simpler choices were rejected:

    * a difference of two window means against the windows' own spread (or
      their own IACT) is CIRCULAR: on a series that is still moving, the
      move inflates both the spread and the IACT, so a steady ramp scores
      only ~2-3 sigma however large it is.  Detrending first, as the
      regression does, measures the noise with the trend removed;
    * comparing against the spread of the DRAWS (not of the mean) ignores
      autocorrelation and false-alarms on every DE run, whose population
      mean has an IACT of hundreds of draws.

    Every element is tested, in whatever coordinates the trace holds (raw
    draws live, physical at wrap-up) -- ``z`` is dimensionless, so no
    whitening is needed to compare them, and naming "the leading physical
    parameters" would need component knowledge this module must not have.
    Elements whose draws are all identical cannot drift and are skipped.
    The tests are Bonferroni-split at ``family_alpha``; Bonferroni is
    conservative for correlated elements (a ridge moves several at once),
    which errs toward stopping, and the z threshold only grows like
    sqrt(2 ln K) with the number of elements K.

    An element is FLAGGED when its p-value is below ``family_alpha / K``
    AND its drift across the window exceeds ``min_drift_sd`` posterior
    standard deviations (the pooled sd of its draws in the window).  An
    element is UNRESOLVED when its detrended ensemble mean holds fewer than
    ``_DRIFT_MIN_NEFF`` effectively independent points: the window is too
    short, in units of the POPULATION's correlation time, to show anything.
    A curved drift lands here too (its curvature stays in the residuals and
    inflates their IACT), which is why unresolved never counts as
    stationary.

    Returns a dict:

        stationary   : bool, nothing flagged and nothing unresolved
        n_tests      : int, elements tested (lp included)
        alpha_each   : float, the per-element p threshold
        n_points     : int, block means per series
        flagged      : list of (name, z, drift_sd, p, n_eff), smallest p first
        unresolved   : the same tuples, for n_eff < _DRIFT_MIN_NEFF
        worst        : the smallest-p tuple tested
    """
    var_names = var_names or default_var_names(posterior)
    series = []  # (name, block-mean series, pooled sd)

    def _add(name, draws):
        # draws: (n_chains, n_draws)
        sd = float(np.std(draws))
        if not np.isfinite(sd):
            # A stored T=1 draw is a state the sampler accepted; a
            # non-finite one is an upstream bug, not a drift.
            raise ValueError(
                f"ensemble_drift: {name} holds non-finite draws in the "
                f"window being judged"
            )
        if sd == 0.0:
            return
        series.append((name, _block_mean_series(draws.mean(axis=0)), sd))

    if lp is not None:
        _add("lp", np.asarray(lp, dtype=float))
    for v in var_names:
        a = np.asarray(posterior[v], dtype=float)
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        for j in range(flat.shape[2]):
            label = v if flat.shape[2] == 1 else f"{v}[{j}]"
            _add(label, flat[:, :, j])

    n_tests = len(series)
    out = {
        "stationary": True,
        "n_tests": n_tests,
        "alpha_each": family_alpha / max(n_tests, 1),
        "n_points": int(series[0][1].shape[0]) if series else 0,
        "flagged": [],
        "unresolved": [],
        "worst": None,
    }
    if not series:
        # Every element (and lp) is constant over the window: nothing can
        # have drifted.
        return out
    if out["n_points"] < _DRIFT_MIN_POINTS:
        # Too short to carry a slope and an IACT: stationarity has not been
        # SHOWN, and the caller must not stop on its absence.
        out["stationary"] = False
        return out

    results = []
    for name, m, sd in series:
        z, drift, p, n_eff = _drift_one(m)
        results.append((name, z, float(drift / sd), p, n_eff))
    results.sort(key=lambda r: (r[3], -abs(r[1])))
    out["worst"] = results[0]
    out["flagged"] = [
        r
        for r in results
        if r[3] < out["alpha_each"] and abs(r[2]) > min_drift_sd
    ]
    flagged = {r[0] for r in out["flagged"]}
    out["unresolved"] = [
        r for r in results if r[4] < _DRIFT_MIN_NEFF and r[0] not in flagged
    ]
    out["stationary"] = not out["flagged"] and not out["unresolved"]
    return out


def format_drift(drift):
    """One-line description of an :func:`ensemble_drift` result for a log."""
    if drift is None:
        return "ensemble drift not judged"
    if drift["worst"] is None:
        if drift["n_tests"] == 0:
            return "ensemble drift: every element constant"
        return "ensemble drift not judged (too few draws)"
    k = drift["n_tests"]
    if drift["flagged"]:
        head = f"ensemble DRIFTING ({len(drift['flagged'])}/{k} elements)"
        name, z, dsd, p, n_eff = drift["flagged"][0]
    elif drift["unresolved"]:
        head = (
            f"ensemble drift UNRESOLVED ({len(drift['unresolved'])}/{k} "
            f"elements under {_DRIFT_MIN_NEFF:.0f} independent means)"
        )
        name, z, dsd, p, n_eff = min(drift["unresolved"], key=lambda r: r[4])
    else:
        head = f"ensemble stationary ({k} elements)"
        name, z, dsd, p, n_eff = drift["worst"]
    return (
        f"{head}: worst {name} z={z:+.1f} p={p:.2g} (p<{drift['alpha_each']:.2g}"
        f" flags), {dsd:+.2f} sd across the tail, n_eff={n_eff:.0f}"
    )


def converged_on_tail(
    posterior, lp, min_ess, max_rhat, tail_frac=0.5, var_names=None
):
    """Cheap early-stop test: is the trace converged on its last ``tail_frac``?

    Used for the live auto-stop decision so the full burn-in scan
    (:func:`find_burnin`) runs only once, at wrap-up. A fixed, generous trim
    is conservative: if the true transient is longer than the trim, the tail
    stays contaminated and we keep sampling. Stuck chains are dropped first
    (see :func:`good_chain_mask`).

    Rhat/ESS are necessary but not sufficient: a population that drifts as
    one body keeps Rhat ~1 the whole way (review 1.4.4), so the tail must
    ALSO show a stationary ensemble mean (:func:`ensemble_drift`, on the
    same good chains and the same tail, lp included).

    ``posterior`` maps var name -> ndarray (n_chains, n_draws, *shape);
    ``lp`` is (n_chains, n_draws) or None. Returns
    (converged, max_rhat, min_ess, drift); NaN stats and a None drift when
    it could not be judged.
    """
    var_names = var_names or default_var_names(posterior)
    if not var_names:
        return False, float("nan"), float("nan"), None
    any_arr = posterior[var_names[0]]
    n_chains, n_draws = any_arr.shape[0], any_arr.shape[1]

    good_idx = np.arange(n_chains)
    if lp is not None:
        mask, _ = good_chain_mask(lp)
        good_idx = np.nonzero(mask)[0]
    if len(good_idx) < 2 or n_draws < 2 * _MIN_DRAWS_PER_CHAIN:
        return False, float("nan"), float("nan"), None

    burnin = int((1.0 - tail_frac) * n_draws)
    thin = _thin_stride(n_draws - burnin)
    sub = {v: posterior[v][good_idx][:, burnin::thin] for v in var_names}
    stat = _rhat_ess(sub)
    if stat is None:
        return False, float("nan"), float("nan"), None
    rhat, ess, _, _ = stat
    # The drift test reads the UNTHINNED tail: it block-averages instead,
    # which keeps every draw's information in the ensemble mean.
    drift = ensemble_drift(
        {v: posterior[v][good_idx][:, burnin:] for v in var_names},
        None if lp is None else np.asarray(lp)[good_idx][:, burnin:],
        var_names=var_names,
    )
    converged = (
        (max_rhat is None or rhat <= max_rhat)
        and (min_ess is None or ess >= min_ess)
        and drift["stationary"]
    )
    return converged, rhat, ess, drift


def find_burnin(posterior, lp=None, var_names=None):
    """Locate the burn-in that maximizes the worst parameter's bulk-ESS.

    Scans ``_BURNIN_FRACS`` on the good-chain, once-thinned trace and picks
    the trim with the largest min bulk-ESS (the ESS knee -- see module
    docstring). Returns a diagnostics dict:

        burnin           : int, draws to discard (in ORIGINAL, un-thinned units)
        burnin_frac      : float, that burn-in as a fraction of the trace
        max_rhat         : float, worst Rhat AFTER trimming
        min_ess          : float, worst bulk-ESS AFTER trimming
        worst_rhat_var   : str, variable attaining max_rhat
        worst_ess_var    : str, variable attaining min_ess
        good_mask        : ndarray[bool], chains kept
        good_reliable    : bool, False if the <min-good fallback kept all
        n_chains_used    : int
        n_draws          : int, draws per chain in the input trace

    Degenerate inputs (single chain, <2 good chains, no usable statistic)
    return burnin=0 with NaN stats.
    """
    var_names = var_names or default_var_names(posterior)
    any_arr = posterior[var_names[0]]
    n_chains, n_draws = any_arr.shape[0], any_arr.shape[1]

    if lp is not None:
        good_mask, good_reliable = good_chain_mask(lp)
    else:
        good_mask, good_reliable = np.ones(n_chains, dtype=bool), True
    good_idx = np.nonzero(good_mask)[0]

    diag = {
        "burnin": 0,
        "burnin_frac": 0.0,
        "max_rhat": float("nan"),
        "min_ess": float("nan"),
        "worst_rhat_var": None,
        "worst_ess_var": None,
        "good_mask": good_mask,
        "good_reliable": good_reliable,
        "n_chains_used": int(len(good_idx)),
        "n_draws": int(n_draws),
    }
    if len(good_idx) < 2 or n_draws < 2 * _MIN_DRAWS_PER_CHAIN:
        return diag

    # Thin the full good-chain trace ONCE so ESS is comparable across
    # candidates: a larger burn-in then simply leaves fewer retained draws
    # (lower raw ESS), which is exactly the "fewer samples" side of the
    # tradeoff we want the maximize-ESS rule to see.
    thin = _thin_stride(n_draws)
    thinned = {v: posterior[v][good_idx][:, ::thin] for v in var_names}
    n_thin = thinned[var_names[0]].shape[1]

    best = None
    for frac in _BURNIN_FRACS:
        b = int(frac * n_thin)
        if n_thin - b < _MIN_DRAWS_PER_CHAIN:
            break
        stat = _rhat_ess({v: a[:, b:] for v, a in thinned.items()})
        if stat is None:
            continue
        rhat, ess, worst_rhat, worst_ess = stat
        if best is None or ess > best["min_ess"]:
            best = {
                "burnin": int(b * thin),
                "burnin_frac": float(frac),
                "max_rhat": rhat,
                "min_ess": ess,
                "worst_rhat_var": worst_rhat,
                "worst_ess_var": worst_ess,
            }
    if best is not None:
        diag.update(best)
    return diag


# Groups the burn-in / stuck-chain trim is allowed to slice.  ALLOW-LIST,
# not "every group that happens to have chain/draw dims": the trim's indices
# mean "T=1 chain k" and "T=1 draw n", so they are only meaningful for groups
# indexed by the SAME axes as `posterior`.
#
# `posterior_hot` (ptde_async store_hot_chains) is the counter-example that
# motivated this (review 2.9.2, #104): its chain axis is
# (n_temps-1) x n_chains hot-rung chains, and its draw axis is its own thinned
# one, so slicing it by T=1 good-chain indices and a T=1 burn-in silently
# returned a handful of arbitrary hot chains with their heads cut off.  It is
# a live input -- outputs.ledger.discover_hot_modes clusters it to record
# posterior-suppressed modes as "considered and rejected" -- and today that
# consumer only escapes because it runs BEFORE this call in run.py.  An
# allow-list makes the invariant hold by construction instead of by statement
# ordering.
#
# Why an allow-list and not a deny-list naming `posterior_hot`: a deny-list
# has to be extended for every future group or that group is silently
# mangled, which is exactly the failure being fixed.  The allow-list fails
# the other way -- a new group is left untrimmed, which is visible in the
# data rather than a silent corruption -- and _warn_untrimmed_groups makes
# even that loud for any group that does carry chain/draw dims.
_TRIMMED_GROUPS = (
    "posterior",
    "sample_stats",
    "log_likelihood",
    "posterior_predictive",
)

# Groups known NOT to share the T=1 draw axes; excluded silently.
_UNTRIMMED_GROUPS = ("posterior_hot",)


def _group_names(idata):
    """Group names present on an InferenceData/DataTree, root excluded."""
    try:
        names = [str(g).strip("/") for g in idata.groups]
    except TypeError:  # legacy arviz: groups() is a method
        names = [str(g).strip("/") for g in idata.groups()]
    return [n for n in names if n]


def _warn_untrimmed_groups(idata, names):
    """Loudly surface any unrecognized group carrying chain/draw dims."""
    for name in names:
        if name in _TRIMMED_GROUPS or name in _UNTRIMMED_GROUPS:
            continue
        dims = getattr(idata[name], "sizes", {})
        if "chain" in dims or "draw" in dims:
            logger.warning(
                "Convergence trim: group '%s' has chain/draw dims but is not "
                "in _TRIMMED_GROUPS; it is being carried across UNTRIMMED "
                "(still holds burn-in and stuck chains). Add it to "
                "samplers/convergence._TRIMMED_GROUPS if it shares the T=1 "
                "draw axes, or to _UNTRIMMED_GROUPS if it does not.",
                name,
            )


def trim_groups(idata, good_idx, burnin):
    """Drop stuck chains + burn-in draws from the T=1 groups only.

    Returns a new InferenceData in which every group named in
    ``_TRIMMED_GROUPS`` is sliced to ``good_idx`` chains and draws from
    ``burnin`` on, and EVERY other group -- ``posterior_hot`` above all -- is
    carried across byte-for-byte.
    """
    names = _group_names(idata)
    _warn_untrimmed_groups(idata, names)
    trimmed = idata.copy()
    idx = list(good_idx)
    for name in names:
        if name not in _TRIMMED_GROUPS:
            continue
        group = idata[name]
        ds = group.to_dataset() if hasattr(group, "to_dataset") else group
        trimmed[name] = ds.isel(
            chain=idx, draw=slice(burnin, None), missing_dims="ignore"
        )
    return trimmed


def analyze_idata(
    idata, min_ess=None, max_rhat=None, var_names=None, exclude=()
):
    """Find burn-in + stuck chains on an InferenceData and return a trimmed view.

    Convenience wrapper for the reporting path: reads the physical posterior
    (dropping ``*_raw`` and ``mode``) and ``sample_stats['lp']``, runs
    :func:`find_burnin`, tags the diagnostics with a ``converged`` verdict
    against the thresholds (informational only -- nothing is discarded), and
    returns ``(trimmed_idata, diag)`` where ``trimmed_idata`` has the stuck
    chains and the burn-in draws removed from the T=1 groups
    (``_TRIMMED_GROUPS``).  Every other group is carried across untouched --
    see that constant for why this is an allow-list.
    """
    posterior = idata.posterior
    var_names = var_names or default_var_names(posterior, exclude=exclude)
    arrays = {v: posterior[v].values for v in var_names}

    lp = None
    ss = getattr(idata, "sample_stats", None)
    if ss is not None and "lp" in ss.data_vars:
        lp = ss["lp"].values

    diag = find_burnin(arrays, lp=lp, var_names=var_names)
    finite = np.isfinite(diag["max_rhat"]) and np.isfinite(diag["min_ess"])
    diag["converged"] = bool(
        finite
        and (max_rhat is None or diag["max_rhat"] <= max_rhat)
        and (min_ess is None or diag["min_ess"] >= min_ess)
    )
    diag["max_rhat_threshold"] = max_rhat
    diag["min_ess_threshold"] = min_ess

    # 2.4.13: Rhat/ESS presuppose stationarity.  Record whether that holds
    # so the verdict can refuse to present them as convergence evidence.
    eq = check_equilibration(lp, diag.get("burnin", 0))
    diag["equilibration"] = eq
    if eq is not None and not eq["equilibrated"]:
        diag["converged"] = False

    good_idx = np.nonzero(diag["good_mask"])[0]

    # 1.4.4: the same ensemble-mean drift test the live early stop applies,
    # on the retained draws, so a run that ran to its draw budget (or was
    # sampled with the early stop off, or by another sampler) is told when
    # its population was still moving.  It reads the SAMPLED coordinates
    # (``*_raw``), as the live check does: they are finite on every draw the
    # sampler accepted, while a derived physical value may be NaN on a draw
    # the mode pass has yet to label invalid.
    diag["drift"] = None
    if len(good_idx) >= 2 and diag["n_draws"] - diag["burnin"] >= (
        2 * _MIN_DRAWS_PER_CHAIN
    ):
        b = diag["burnin"]
        drift_names = [
            v
            for v in posterior.data_vars
            if v.endswith("_raw") and v not in set(exclude)
        ] or var_names
        drift = ensemble_drift(
            {v: posterior[v].values[good_idx][:, b:] for v in drift_names},
            None if lp is None else np.asarray(lp)[good_idx][:, b:],
            var_names=drift_names,
        )
        diag["drift"] = drift
        if not drift["stationary"]:
            diag["converged"] = False

    trimmed = trim_groups(idata, good_idx, diag["burnin"])
    return trimmed, diag


def check_equilibration(lp, burnin=0):
    """Is the T=1 log-posterior still TRENDING at the end of the run?

    Rhat and ESS both assume the chains are sampling a stationary
    distribution.  On a run that is still climbing they are computed
    anyway and mean nothing -- review 2.4.13, measured on ob09020, which
    reported `burn-in = 0 draws` and a UNIFORM Rhat ~2.0 across all 27
    nuisance elements while its T=1 lp climbed monotonically from 44,223 to
    47,749 and was still rising at draw 442 of 1000.  Uniform Rhat is the
    signature of a common drift; a real mixing pathology localizes.  That
    run sent the investigation to the temperature ladder twice, and the
    ladder was healthy both times.

    The test compares the mean lp of the final quarter against the third
    quarter, in units of the post-burn-in lp scatter.  At equilibrium the
    mean is stationary, so that difference is ~0 within the standard error;
    a persistent climb makes it large and POSITIVE.  Returns a dict, or None
    when there is no usable lp (the caller must not infer "equilibrated"
    from absence of evidence).
    """
    if lp is None:
        return None
    a = np.asarray(lp, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] < 8:
        return None
    a = a[:, int(burnin) :] if 0 < int(burnin) < a.shape[1] - 8 else a
    n = a.shape[1]
    if n < 8:
        return None
    q3, q4 = a[:, n // 2 : 3 * n // 4], a[:, 3 * n // 4 :]
    if q3.size == 0 or q4.size == 0:
        return None
    m3, m4 = float(np.nanmean(q3)), float(np.nanmean(q4))
    drift = m4 - m3

    # NOISE, ESTIMATED TREND-IMMUNELY.  The obvious yardstick -- the sd of
    # the whole lp series -- is CIRCULAR: for a chain that is still climbing
    # that sd is dominated by the climb itself, so the drift never looks
    # large next to it.  A linear ramp of 3,526 nats with 5.4-nat noise
    # scored 0.87 "sigma" and passed on the first version of this check.
    # The lag-1 difference removes any smooth trend and leaves the noise:
    # sd(diff) = sqrt(2) * sigma for white noise.
    d = np.diff(a, axis=1)
    noise = float(np.nanstd(d)) / np.sqrt(2.0)
    if not np.isfinite(noise) or noise <= 0:
        return None

    # 3 sigma is deliberately conservative.  The standard error of a
    # difference of means is sigma/sqrt(n_eff), i.e. SMALLER than sigma, so
    # comparing the drift against sigma itself already errs toward silence;
    # 3x further protects a well-mixed but strongly autocorrelated chain
    # from being called non-stationary.  The case this must catch --
    # ob09020 -- sits at ~160 sigma, not 3.
    return {
        "lp_drift_nats": drift,
        "lp_scatter_nats": noise,
        "lp_drift_sigma": drift / noise,
        "equilibrated": bool(abs(drift) <= 3.0 * noise),
    }


def log_convergence(diag, log=logger):
    """Emit the convergence verdict: INFO when good, loud WARNING when not."""
    n_draws = diag.get("n_draws", 0)
    frac = diag.get("burnin_frac", 0.0)
    summary = (
        f"burn-in = {diag['burnin']} draws ({100 * frac:.0f}% of {n_draws}); "
        f"post-burn-in max Rhat={diag['max_rhat']:.3f} "
        f"({diag.get('worst_rhat_var')}), "
        f"min ESS={diag['min_ess']:.0f} ({diag.get('worst_ess_var')}); "
        f"chains kept {diag.get('n_chains_used')}"
    )
    if not diag.get("good_reliable", True):
        log.warning(
            "Convergence: fewer than %d chains reached the good-likelihood "
            "region; keeping ALL chains -- results may be contaminated by "
            "stuck chains.",
            _MIN_GOOD_CHAINS,
        )
    eq = diag.get("equilibration")
    if eq is not None and not eq["equilibrated"]:
        log.warning(
            "NOT EQUILIBRATED -- Rhat and ESS are NOT convergence evidence "
            "here: the T=1 log-posterior is still trending (%+.1f nats "
            "between the last two quarters, %.1f x its own scatter of %.1f). "
            "The chains have not reached a stationary distribution, so the "
            "reported Rhat/ESS describe a transient.  %s.  Remedy: a longer "
            "run, a `tune` phase (whose draws are discarded), or a better "
            "start -- not a different temperature ladder (review 2.4.13).",
            eq["lp_drift_nats"],
            eq["lp_drift_sigma"],
            eq["lp_scatter_nats"],
            summary,
        )
        return
    drift = diag.get("drift")
    if drift is not None and not drift["stationary"]:
        log.warning(
            "ENSEMBLE MEAN NOT STATIONARY -- %s.  The chains move together, "
            "so Rhat (which compares chains) cannot see it: the population "
            "is still drifting or has not been sampled for enough of its own "
            "correlation times.  %s.  Remedy: a longer run; the reported "
            "medians of the drifting elements describe where the population "
            "WAS (review 1.4.4).",
            format_drift(drift),
            summary,
        )
        return
    if drift is not None:
        summary += "; " + format_drift(drift)
    if diag.get("converged", False):
        log.info("Convergence OK: %s", summary)
    else:
        log.warning(
            "CONVERGENCE NOT REACHED: %s [thresholds: Rhat<=%s, ESS>=%s] -- "
            "reporting anyway; inspect the trace and consider a longer or "
            "reseeded run.",
            summary,
            diag.get("max_rhat_threshold"),
            diag.get("min_ess_threshold"),
        )
