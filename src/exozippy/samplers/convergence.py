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

import numpy as np
import arviz as az

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


def default_var_names(posterior):
    """Physical, non-index variables worth judging convergence on.

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
    for v in names:
        if v == "mode":
            continue
        if v.endswith("_raw") and v[:-len("_raw")] in present:
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
    mask = chain_max >= threshold          # the best chain always passes
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


def converged_on_tail(posterior, lp, min_ess, max_rhat, tail_frac=0.5,
                      var_names=None):
    """Cheap early-stop test: is the trace converged on its last ``tail_frac``?

    Used for the live auto-stop decision so the full burn-in scan
    (:func:`find_burnin`) runs only once, at wrap-up. A fixed, generous trim
    is conservative: if the true transient is longer than the trim, the tail
    stays contaminated and we keep sampling, so this never stops early on an
    unequilibrated run -- it only ever stops a little late. Stuck chains are
    dropped first (see :func:`good_chain_mask`).

    ``posterior`` maps var name -> ndarray (n_chains, n_draws, *shape);
    ``lp`` is (n_chains, n_draws) or None. Returns
    (converged, max_rhat, min_ess); NaN stats when it could not be judged.
    """
    var_names = var_names or default_var_names(posterior)
    if not var_names:
        return False, float("nan"), float("nan")
    any_arr = posterior[var_names[0]]
    n_chains, n_draws = any_arr.shape[0], any_arr.shape[1]

    good_idx = np.arange(n_chains)
    if lp is not None:
        mask, _ = good_chain_mask(lp)
        good_idx = np.nonzero(mask)[0]
    if len(good_idx) < 2 or n_draws < 2 * _MIN_DRAWS_PER_CHAIN:
        return False, float("nan"), float("nan")

    burnin = int((1.0 - tail_frac) * n_draws)
    thin = _thin_stride(n_draws - burnin)
    sub = {v: posterior[v][good_idx][:, burnin::thin] for v in var_names}
    stat = _rhat_ess(sub)
    if stat is None:
        return False, float("nan"), float("nan")
    rhat, ess, _, _ = stat
    converged = ((max_rhat is None or rhat <= max_rhat)
                 and (min_ess is None or ess >= min_ess))
    return converged, rhat, ess


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
        "burnin": 0, "burnin_frac": 0.0,
        "max_rhat": float("nan"), "min_ess": float("nan"),
        "worst_rhat_var": None, "worst_ess_var": None,
        "good_mask": good_mask, "good_reliable": good_reliable,
        "n_chains_used": int(len(good_idx)), "n_draws": int(n_draws),
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
                "burnin": int(b * thin), "burnin_frac": float(frac),
                "max_rhat": rhat, "min_ess": ess,
                "worst_rhat_var": worst_rhat, "worst_ess_var": worst_ess,
            }
    if best is not None:
        diag.update(best)
    return diag


def analyze_idata(idata, min_ess=None, max_rhat=None, var_names=None):
    """Find burn-in + stuck chains on an InferenceData and return a trimmed view.

    Convenience wrapper for the reporting path: reads the physical posterior
    (dropping ``*_raw`` and ``mode``) and ``sample_stats['lp']``, runs
    :func:`find_burnin`, tags the diagnostics with a ``converged`` verdict
    against the thresholds (informational only -- nothing is discarded), and
    returns ``(trimmed_idata, diag)`` where ``trimmed_idata`` has the stuck
    chains and the burn-in draws removed from every group that has those dims.
    """
    posterior = idata.posterior
    var_names = var_names or default_var_names(posterior)
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

    good_idx = np.nonzero(diag["good_mask"])[0]
    # isel across the whole InferenceData; missing_dims="ignore" leaves groups
    # without chain/draw (e.g. observed_data) untouched.
    trimmed = idata.isel(chain=good_idx.tolist(),
                         draw=slice(diag["burnin"], None),
                         missing_dims="ignore")
    return trimmed, diag


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
            "stuck chains.", _MIN_GOOD_CHAINS)
    if diag.get("converged", False):
        log.info("Convergence OK: %s", summary)
    else:
        log.warning(
            "CONVERGENCE NOT REACHED: %s [thresholds: Rhat<=%s, ESS>=%s] -- "
            "reporting anyway; inspect the trace and consider a longer or "
            "reseeded run.",
            summary, diag.get("max_rhat_threshold"),
            diag.get("min_ess_threshold"))
