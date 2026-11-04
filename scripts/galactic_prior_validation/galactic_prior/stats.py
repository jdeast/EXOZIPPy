"""Weighted-sample statistics for the galactic-prior validation (numpy only).

Every distribution compared here is a WEIGHTED sample: genulens prints one
event weight per line (``wtj``), and the EXOZIPPy prior is evaluated by
importance sampling, so its draws carry importance weights.  Quantiles, KS
distances, bootstrap errors and the importance-sampling diagnostics are
therefore all written for weighted samples.
"""

import numpy as np

# Quantiles stored and compared everywhere.
QUANTILES = (0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95)


def qkey(q):
    """0.05 -> 'q05'."""
    return f"q{int(round(100 * q)):02d}"


def _clean(x, w):
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    if x.shape != w.shape:
        raise ValueError(f"values {x.shape} and weights {w.shape} differ")
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[ok], w[ok]


def weighted_quantiles(x, w=None, qs=QUANTILES):
    """Quantiles of x under weights w (midpoint-CDF interpolation).

    Returns an array in the order of ``qs``.  With equal weights this is
    numpy's ``method="hazen"`` quantile.
    """
    x, w = _clean(x, w)
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(qs, float), cdf, x)


def kish_ess(w):
    """Kish effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w**2))


def bootstrap_quantile_se(columns, w, qs=QUANTILES, n_boot=100, seed=0):
    """Standard error of weighted quantiles by a weighted (Poisson) bootstrap.

    Each replicate multiplies every draw's weight by an independent
    Poisson(1) count -- the large-n limit of resampling draws with
    replacement, with each draw keeping its own weight -- and the SAME
    counts are used for every column, so the replicates are joint.  This is
    the error of an importance-sampling estimate (it includes the weight
    variance) and of genulens's weighted events alike.

    ``columns`` maps name -> 1-D array (all the same length as ``w``).
    Returns {name: array of se, ordered as ``qs``}.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w, float)
    qs = np.asarray(qs, float)
    sorted_cols = {}
    for name, x in columns.items():
        x = np.asarray(x, float)
        order = np.argsort(x, kind="stable")
        sorted_cols[name] = (x[order], order)
    reps = {name: np.empty((n_boot, qs.size)) for name in columns}
    for b in range(n_boot):
        wb = w * rng.poisson(1.0, size=w.size)
        for name, (xs, order) in sorted_cols.items():
            ws = wb[order]
            cdf = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
            reps[name][b] = np.interp(qs, cdf, xs)
    return {name: r.std(axis=0, ddof=1) for name, r in reps.items()}


def weighted_ecdf(x, w, grid):
    """F(grid) = P(X <= grid) for a weighted sample."""
    x, w = _clean(x, w)
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    cw = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
    return cw[np.searchsorted(x, grid, side="right")]


def weighted_ks(x1, w1, x2, w2):
    """Two-sample KS distance max |F1 - F2| between two weighted samples."""
    x1, w1 = _clean(x1, w1)
    x2, w2 = _clean(x2, w2)
    grid = np.unique(np.concatenate([x1, x2]))
    return float(
        np.max(
            np.abs(weighted_ecdf(x1, w1, grid) - weighted_ecdf(x2, w2, grid))
        )
    )


def gpd_fit(exceedances):
    """Shape k and scale sigma of a generalized Pareto fit (Zhang & Stephens 2009).

    The empirical-Bayes estimator PSIS uses (Vehtari et al. 2024, JMLR 25,
    72), including its weakly informative prior on k.  ``exceedances`` are
    the positive tail values minus the threshold.
    """
    x = np.sort(np.asarray(exceedances, float))
    n = x.size
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(np.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1, dtype=float) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b += 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    len_scale = n * (np.log(-(b / k)) - k - 1.0)
    with np.errstate(over="ignore"):
        # An overflow is a weight that underflows to 0: dropped below.
        weights = 1.0 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    keep = weights >= 10 * np.finfo(float).eps
    weights, b = weights[keep], b[keep]
    weights /= weights.sum()
    b_post = np.sum(b * weights)
    k_post = np.log1p(-b_post * x).mean()
    sigma = -k_post / b_post
    return float((n * k_post + prior_k * 0.5) / (n + prior_k)), float(sigma)


def pareto_khat(log_w):
    """Pareto k-hat of importance log-weights (PSIS diagnostic).

    The generalized Pareto shape fitted to the largest
    M = min(0.2 S, 3 sqrt(S)) weights.  k < 0.7 means the importance
    estimate's error behaves (Vehtari et al. 2024); k >= 0.7 means the
    weight distribution's tail is too heavy for the estimate to be trusted.
    """
    lw = np.asarray(log_w, float)
    lw = lw[np.isfinite(lw)]
    s = lw.size
    m = int(np.ceil(min(0.2 * s, 3.0 * np.sqrt(s))))
    lw = np.sort(lw - lw.max())
    tail = np.exp(lw[-m:])
    cutoff = np.exp(lw[-m - 1])
    return gpd_fit(tail - cutoff)[0]


def importance_diagnostics(log_w):
    """Kish ESS, its fraction, the largest normalized weight and Pareto k."""
    lw = np.asarray(log_w, float)
    finite = np.isfinite(lw)
    w = np.exp(lw[finite] - lw[finite].max())
    ess = kish_ess(w)
    return {
        "n_draws": int(lw.size),
        "n_zero_weight": int((~finite).sum()),
        "kish_ess": ess,
        "ess_fraction": ess / lw.size,
        "max_weight_fraction": float(w.max() / w.sum()),
        "pareto_khat": pareto_khat(lw[finite]),
    }


def fraction_with_se(mask, w, n_boot=100, seed=1):
    """Weighted fraction of draws in ``mask`` and its bootstrap se."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w, float)
    mask = np.asarray(mask, bool)
    f = float(w[mask].sum() / w.sum())
    reps = np.empty(n_boot)
    for b in range(n_boot):
        wb = w * rng.poisson(1.0, size=w.size)
        reps[b] = wb[mask].sum() / wb.sum()
    return f, float(reps.std(ddof=1))


def binned_quantiles(log_edges, counts, qs=QUANTILES):
    """Quantiles of t_E from a histogram in log10 t_E (linear within bins)."""
    c = np.asarray(counts, float)
    cdf = np.concatenate([[0.0], np.cumsum(c)]) / c.sum()
    return 10 ** np.interp(np.asarray(qs, float), cdf, log_edges)
