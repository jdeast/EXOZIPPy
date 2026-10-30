"""Uniformity statistics for simulation-based calibration ranks.

Ranks are integers in {0, ..., L} (the number of the L posterior draws that
fall below the truth), uniform under correct calibration.  No exozippy import:
the smoke test exercises these on synthetic ranks in milliseconds.

* ``chi2_uniformity`` -- Pearson chi2 over ``n_bins`` equal-width bins
  (Talts et al. 2018 use the histogram itself).
* ``ks_uniformity`` -- KS of the randomized fractional rank
  ``(r + U) / (L + 1)``, which is exactly U(0, 1) under the null (the
  randomization only spreads each discrete rank over its own cell).
* ``ecdf_band_gamma`` / ``ecdf_diff`` -- the ECDF of the ranks against its
  expectation with the SIMULTANEOUS band of Sailynoja, Buerkner & Vehtari
  (2022, Stat. Comput. 32, 32): the per-point binomial level ``gamma`` is
  adjusted by simulation so that a uniform sample stays inside the band at
  ALL L evaluation points with probability 1 - alpha.  A pointwise 95% band
  is crossed somewhere by most uniform samples, which is why the adjustment
  is the whole point.
"""

import numpy as np
from scipy import stats

_GAMMA_CACHE = {}


def chi2_uniformity(r, L, n_bins):
    r = np.asarray(r, dtype=int)
    if (L + 1) % n_bins:
        raise ValueError(
            f"L + 1 = {L + 1} ranks do not split into {n_bins} bins"
        )
    counts = np.bincount(r // ((L + 1) // n_bins), minlength=n_bins)
    return float(stats.chisquare(counts).pvalue)


def ks_uniformity(r, L, rng):
    r = np.asarray(r, dtype=float)
    u = (r + rng.uniform(size=r.size)) / (L + 1)
    return float(stats.kstest(u, "uniform").pvalue)


def _tail_min(F_counts, n, p):
    """Two-sided binomial tail probability, minimized over evaluation points.

    ``F_counts`` is (..., K) counts #(r <= k); ``p`` is (K,) expectations.
    """
    lo = stats.binom.cdf(F_counts, n, p)
    hi = stats.binom.sf(F_counts - 1, n, p)
    return np.min(2.0 * np.minimum(lo, hi), axis=-1)


def _counts(r, L):
    r = np.asarray(r, dtype=int)
    hist = np.bincount(r, minlength=L + 1)
    # Evaluate at k = 0..L-1: at k = L the ECDF is 1 by construction.
    return np.cumsum(hist)[:L]


def ecdf_band_gamma(n, L, alpha=0.05, n_sim=4000, rng=None):
    rng = rng or np.random.default_rng(12345)
    p = (np.arange(L) + 1.0) / (L + 1.0)
    sims = rng.integers(0, L + 1, size=(n_sim, n))
    counts = np.stack([_counts(s, L) for s in sims])
    minima = _tail_min(counts, n, p)
    gamma = float(np.quantile(minima, alpha))
    _GAMMA_CACHE[(n, L, alpha)] = (gamma, np.sort(minima))
    return gamma


def ecdf_band_gamma_cached(n, L, alpha=0.05):
    if (n, L, alpha) not in _GAMMA_CACHE:
        ecdf_band_gamma(n, L, alpha)
    return _GAMMA_CACHE[(n, L, alpha)][0]


def ecdf_diff(r, L, gamma, alpha=0.05):
    """(gamma, p-value, inside-band) for one rank sample.

    The p-value is the fraction of simulated uniform samples whose worst
    pointwise tail probability is at least as extreme as this sample's.
    """
    r = np.asarray(r, dtype=int)
    n = r.size
    p = (np.arange(L) + 1.0) / (L + 1.0)
    counts = _counts(r, L)
    obs = float(_tail_min(counts[None, :], n, p)[0])
    key = (n, L, alpha)
    if key not in _GAMMA_CACHE:
        ecdf_band_gamma(n, L, alpha)
    _, minima = _GAMMA_CACHE[key]
    pval = float(np.searchsorted(minima, obs, side="right") / minima.size)
    lo = stats.binom.ppf(gamma / 2.0, n, p)
    hi = stats.binom.ppf(1.0 - gamma / 2.0, n, p)
    inside = bool(np.all((counts >= lo) & (counts <= hi)))
    return gamma, pval, inside


def rank_hist_band(n, n_bins, level=0.99):
    a = (1.0 - level) / 2.0
    return (
        float(stats.binom.ppf(a, n, 1.0 / n_bins)),
        float(stats.binom.ppf(1.0 - a, n, 1.0 / n_bins)),
    )
