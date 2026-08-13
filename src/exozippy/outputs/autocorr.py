"""Integrated autocorrelation time of a Monte Carlo estimator's mean.

Shared by ``outputs/evidence.py`` -- which inflates the bridge-sampling error
bar by the IACT of its posterior-side bridge function -- and by
``outputs/modes.py``, which turns the IACT of a mode-indicator series into an
effective sample size for that mode's occupancy weight.

Both callers want the same number: the factor by which the variance of the
sample MEAN exceeds the i.i.d. value ``var/N``.  Two things inflate it, and
both are counted here:

  * **within-chain autocorrelation** -- the chain-averaged autocovariance,
    truncated by Geyer's initial positive/monotone rule on the pair sums
    ``rho_2t + rho_2t+1``;
  * **between-chain scatter of the chain means**, beyond what the
    within-chain autocorrelation predicts.  For chains stuck in different
    modes this term is the entire answer: every chain is internally constant,
    so a within-chain-only estimate would report ``tau = 1`` and hand back an
    absurdly tight error bar on a weight that is pure initialization.

Chains may be **ragged** (a mode does not occupy the same number of draws in
every chain, and invalid draws are dropped), so nothing here assumes a
rectangular array.

ORDER IS THE INPUT.  Reordering a series -- subsampling it with an unsorted
``rng.choice``, say -- destroys precisely the structure being measured, and
any autocorrelated series then reads ``tau ~ 1``.  If a series must be
shortened, thin it with a stride: the IACT measured on the strided series is
the correct inflation factor for the mean OF THAT SERIES, which is what the
caller actually averages, so no separate thinning correction is needed.  (A
useful invariant: ``tau_thinned / N_thinned`` tracks ``tau_full / N_full``
until the stride exceeds the correlation time, past which the thinned answer
is merely conservative.)
"""

import numpy as np


def _as_chains(chains):
    """Normalize the input to a list of 1-D float arrays, one per chain."""
    if isinstance(chains, np.ndarray):
        a = np.asarray(chains, dtype=float)
        if a.ndim == 0:
            return [a.reshape(1)]
        if a.ndim == 1:
            return [a]
        if a.ndim == 2:
            return [a[c] for c in range(a.shape[0])]
        raise ValueError(
            f"autocorr: expected a 1-D or (chain, draw) 2-D array, got "
            f"shape {a.shape}"
        )
    return [np.asarray(c, dtype=float).ravel() for c in chains]


def _autocov(x, n_lag):
    """Biased autocovariance of a centered 1-D series at lags 0..n_lag-1."""
    n = x.size
    y = x - x.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(y, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n] / n
    out = np.zeros(n_lag)
    out[: min(n_lag, n)] = acov[: min(n_lag, n)]
    return out


def iact(chains):
    """Variance-inflation factor of the mean over one or more chains (>= 1).

    Parameters
    ----------
    chains : 1-D array (one chain), 2-D array of shape (chain, draw), or a
        sequence of 1-D arrays of possibly different lengths.

    Returns
    -------
    float
        ``tau`` such that ``Var(mean) = tau * Var(x) / N``, where ``N`` is
        the total number of values across all chains.

    Notes
    -----
    This is the Vehtari et al. (2021) multi-chain construction, generalized
    to ragged chain lengths, with Geyer's initial positive/monotone pair
    truncation.  Its virtue over "take the larger of a within-chain and a
    between-chain estimate" is that the between-chain variance B enters only
    as ``B / n_bar`` inside ``var_plus``.  On well-mixed chains that term is
    O(tau/n) and changes nothing, so the chi-square noise of a (C-1)-degree-
    of-freedom quantity cannot inflate a perfectly good error bar; on chains
    stuck in different modes the within-chain variance W collapses, ``B/n``
    dominates, every ``rho_t`` stays pinned at 1, and tau grows until the
    effective sample size is of order the number of chains -- which is the
    honest answer there.
    """
    segs = [c for c in _as_chains(chains) if c.size > 0]
    segs = [c[np.isfinite(c)] for c in segs]
    segs = [c for c in segs if c.size > 0]
    n_chain = len(segs)
    n_total = sum(c.size for c in segs)
    if n_chain == 0 or n_total < 4:
        return 1.0

    lengths = np.array([c.size for c in segs], dtype=float)
    means = np.array([float(c.mean()) for c in segs])
    grand_mean = float((lengths * means).sum() / n_total)

    # W: pooled within-chain variance.  B: between-chain variance, written in
    # the ragged-safe form that reduces to n * var(chain means, ddof=1) when
    # every chain has the same length.
    dof = float((lengths - 1).sum())
    if dof <= 0:
        return 1.0
    W = sum(float(((c - c.mean()) ** 2).sum()) for c in segs) / dof
    if n_chain > 1:
        B = float((lengths * (means - grand_mean) ** 2).sum()) / (n_chain - 1)
    else:
        B = 0.0
    n_bar = n_total / n_chain
    var_plus = ((n_bar - 1.0) * W + B) / n_bar
    if not np.isfinite(var_plus) or var_plus <= 0:
        return 1.0

    # Chain-averaged autocovariance, weighted by chain length; at lag t only
    # the chains long enough to have that lag contribute.
    n_lag = max(4, int(n_bar))
    acov_sum = np.zeros(n_lag)
    weight = np.zeros(n_lag)
    for c in segs:
        m = min(n_lag, c.size)
        acov_sum[:m] += c.size * _autocov(c, n_lag)[:m]
        weight[:m] += c.size
    valid = weight > 0
    acov_bar = np.zeros(n_lag)
    acov_bar[valid] = acov_sum[valid] / weight[valid]
    n_lag = int(valid.sum()) if valid.all() else int(np.argmin(valid))
    if n_lag < 2:
        return 1.0

    rho = 1.0 - (W - acov_bar[:n_lag]) / var_plus

    # Geyer initial positive sequence on the pair sums, with the initial
    # monotone clamp: truncating on pairs is far less noisy than the per-lag
    # "rho_k <= 0" rule, which on a long correlated series stops early on a
    # single negative sample.
    tau = -1.0
    prev = np.inf
    t = 0
    while t + 1 < n_lag:
        p = rho[t] + rho[t + 1]
        if t > 0 and p <= 0:
            break
        p = min(p, prev)  # initial monotone sequence
        prev = p
        tau += 2.0 * p
        t += 2

    if not np.isfinite(tau):
        return 1.0
    return float(min(max(tau, 1.0), float(n_total)))


def ess(chains):
    """Effective sample size of the mean: total draws divided by ``iact``."""
    segs = [c for c in _as_chains(chains) if c.size > 0]
    n_total = sum(c.size for c in segs)
    if n_total == 0:
        return 0.0
    return n_total / iact(segs)
