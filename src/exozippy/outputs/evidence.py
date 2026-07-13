"""Per-mode local evidence estimation via warp bridge sampling.

This is the *fallback* multimodal-weighting path in EXOZIPPy.  The primary
paths (declared-degeneracy jump proposals / folded likelihoods) weight known
degeneracies exactly; this module exists for (a) modes nothing else explains
and (b) as an independent cross-check.  It is deliberately self-diagnosing:
when its own internal diagnostics cannot support a confident answer it REFUSES
for that mode and the report falls back to occupancy-with-provenance, rather
than emitting a shaky number.

Method (per mode, in the raw / unconstrained sampled space that
outputs.modes already clusters on):

  1. Membership.  Draws are assigned to modes by the integer ``mode`` label
     that identify_modes attaches to idata.posterior (or by ModeReport.labels).

  2. Proposal.  A normalized multivariate Gaussian (default) or Student-t is
     fit to the mode's raw draws over the *full* free-RV vector.  Raw space is
     the non-centered N(0, 1) space, so a Gaussian is a natural, bound-free
     reference with an analytic normalizing constant.

  3. Bridge estimator (Meng & Wong 1996, optimal bridge).  Using the mode's
     posterior draws and fresh draws from the proposal, the iterative optimal
     bridge estimator returns the mode's local log-evidence lnZ_k relative to
     the (normalized) proposal, plus an approximate relative-MSE diagnostic
     (Fruhwirth-Schnatter 2004) that yields the lnZ error bar.

  4. Refusal.  If the relative-MSE diagnostic is too large -- the signature of
     a proposal that poorly supports the target, e.g. draws piled against a
     parameter bound so the raw-space tail is fat and one-sided -- the mode is
     refused: no number is reported, the reason is logged, and the caller
     falls back to occupancy for the whole report (softmax evidence weights
     need every mode's lnZ, so a single refusal invalidates the set).

  5. Weights.  Accepted lnZ_k -> softmax weights w_k; lnZ error bars propagate
     to softmax weight uncertainties dw_k by linearization.

The logp evaluations reuse run.py's fork-parallel pattern (compiled PyTensor
logp inherited by forked workers via copy-on-write; only numpy point arrays
cross the IPC boundary).

Optional cross-checks (pm.sample_smc, dynesty nested sampling) are intentionally
NOT implemented here: they are lower priority than the bridge path and are
"logged but never trusted alone".  See estimate_mode_evidences docstring.
"""

from dataclasses import dataclass
import logging
import multiprocessing as mp

import numpy as np

logger = logging.getLogger(__name__)


# Refuse a mode whose bridge relative-MSE exceeds this.  re2 is the estimator's
# relative mean-squared error on Z; sqrt(re2) is the coefficient of variation
# of Z and (to first order) the lnZ error bar.  0.25 -> refuse once the lnZ
# error bar reaches ~0.5 nat, at which point softmax weights are too soft to
# trust as a posterior-mass statement.
DEFAULT_RE2_MAX = 0.25


@dataclass
class EvidenceResult:
    """Bridge-sampling result for one mode.

    lnZ / lnZ_err are the mode's local log-evidence relative to its proposal
    and its (relative-MSE-derived) error bar.  ``refused`` is True when the
    diagnostics could not support a confident answer; ``reason`` says why.
    """
    mode: int
    lnZ: float
    lnZ_err: float
    re2: float
    n_post: int
    n_prop: int
    refused: bool
    reason: str = ""


# ----------------------------------------------------------------------
# pure bridge-sampling core (no model/PyMC dependency -- unit testable)
# ----------------------------------------------------------------------

def _logmeanexp(a):
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if not np.isfinite(m):
        return m
    return m + np.log(np.mean(np.exp(a - m)))


def _iact(x):
    """Integrated autocorrelation time of a 1-D series (>= 1).

    Sums the normalized autocorrelation with Geyer's initial-positive-sequence
    truncation.  Used to inflate the bridge error bar for autocorrelated
    posterior draws; returns ~1 for i.i.d. input.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return 1.0
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 0:
        return 1.0
    # autocovariance via FFT
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n] / n
    rho = acov / acov[0]
    tau = 1.0
    for k in range(1, n):
        if rho[k] <= 0:
            break
        tau += 2.0 * rho[k]
    return max(tau, 1.0)


def bridge_lnZ(l1, l2, maxiter=1000, tol=1e-10):
    """Optimal iterative bridge-sampling estimate of a log normalizing constant.

    Parameters
    ----------
    l1 : array, log(unnormalized target) - log(proposal) at the *posterior*
        draws (samples from the target).
    l2 : array, the same log-ratio at fresh draws from the *proposal*.
    maxiter, tol : iteration controls for the fixed-point update.

    Returns
    -------
    (lnZ, lnZ_err, re2, converged)
        lnZ is the estimated log-evidence of the target relative to the
        (normalized) proposal; re2 is the approximate relative mean-squared
        error of the Z estimate (Fruhwirth-Schnatter 2004) and
        lnZ_err = sqrt(re2).
    """
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    l1 = l1[np.isfinite(l1)]
    l2 = l2[np.isfinite(l2)]
    N1, N2 = l1.size, l2.size
    if N1 < 2 or N2 < 2:
        return np.nan, np.inf, np.inf, False

    log_s1 = np.log(N1 / (N1 + N2))
    log_s2 = np.log(N2 / (N1 + N2))

    lr = 0.0
    converged = False
    for _ in range(maxiter):
        # log(num_j) over proposal draws l2; log(den_i) over posterior draws l1
        log_num = (l2 - np.logaddexp(log_s1 + l2, log_s2 + lr))
        log_den = -np.logaddexp(log_s1 + l1, log_s2 + lr)
        lr_new = _logmeanexp(log_num) - _logmeanexp(log_den)
        if not np.isfinite(lr_new):
            return np.nan, np.inf, np.inf, False
        if abs(lr_new - lr) < tol:
            lr = lr_new
            converged = True
            break
        lr = lr_new

    r = np.exp(lr)
    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)
    # f1 over proposal samples, f2 over posterior samples (converged r)
    f1 = np.exp(l2) / (s1 * np.exp(l2) + s2 * r)
    f2 = 1.0 / (s1 * np.exp(l1) + s2 * r)
    m1, m2 = f1.mean(), f2.mean()
    if not (np.isfinite(m1) and np.isfinite(m2)) or m1 <= 0 or m2 <= 0:
        return lr, np.inf, np.inf, converged
    tau = _iact(f2)
    term1 = (f1.var() / m1**2) / N2
    term2 = tau * (f2.var() / m2**2) / N1
    re2 = float(term1 + term2)
    lnZ_err = float(np.sqrt(re2)) if np.isfinite(re2) else np.inf
    return float(lr), lnZ_err, re2, converged


# ----------------------------------------------------------------------
# proposal fitting + multivariate-normal density
# ----------------------------------------------------------------------

def _fit_gaussian(X, shrink_floor=1e-6):
    """Regularized mean/covariance of raw draws X (n, d).

    Shrinks the sample covariance toward its diagonal when draws are scarce
    relative to the dimension, and floors eigenvalues so the density is always
    proper (constant/degenerate raw dimensions do not make it singular).
    """
    n, d = X.shape
    mu = X.mean(axis=0)
    if n > 1:
        S = np.cov(X, rowvar=False)
    else:
        S = np.eye(d)
    S = np.atleast_2d(S)
    if S.shape != (d, d):
        S = np.eye(d) * float(np.atleast_1d(S).ravel()[0])
    # Ledoit-Wolf-style shrinkage toward the diagonal when n is small vs d.
    alpha = min(1.0, d / max(n, 1) * 0.5)
    S = (1.0 - alpha) * S + alpha * np.diag(np.diag(S))
    # eigenvalue floor relative to the mean variance
    mean_var = max(float(np.mean(np.diag(S))), 1e-300)
    floor = shrink_floor * mean_var
    w, V = np.linalg.eigh(S)
    w = np.clip(w, floor, None)
    S = (V * w) @ V.T
    return mu, S


def _mvn_logpdf(X, mu, S):
    """log N(X; mu, S) for a stack of points X (n, d)."""
    X = np.atleast_2d(X)
    d = mu.size
    L = np.linalg.cholesky(S)
    diff = X - mu
    # L is lower-triangular; np.linalg.solve handles it fine (scipy-free).
    Linv_diff = np.linalg.solve(L, diff.T)
    quad = np.sum(Linv_diff**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def _sample_gaussian(mu, S, n, rng):
    L = np.linalg.cholesky(S)
    z = rng.standard_normal((n, mu.size))
    return mu + z @ L.T


# ----------------------------------------------------------------------
# fork-parallel raw-space logp evaluation (mirrors run.py pattern)
# ----------------------------------------------------------------------

# Module-level so forked children inherit the compiled logp via copy-on-write
# without pickling it (only numpy point arrays cross IPC).
_EV_LP_FN = None
_EV_LAYOUT = None   # list of (value_name, start, size, shape)


def _ev_eval_block(args):
    block, offset = args
    n = block.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        row = block[i]
        point = {}
        for vname, start, size, shape in _EV_LAYOUT:
            seg = row[start:start + size]
            point[vname] = seg.reshape(shape) if len(shape) else np.atleast_1d(seg)
        try:
            out[i] = float(_EV_LP_FN(point))
        except Exception:
            out[i] = np.nan
    return offset, out


def _build_layout(model, idata):
    """Column layout of the concatenated free-RV raw vector.

    Returns (layout, D) where layout is a list of
    (rv_name, value_name, start, size, shape) and D is the total dimension.
    rv_name indexes idata.posterior; value_name is the logp_fn input key.
    """
    layout = []
    start = 0
    post = idata.posterior
    for rv in model.free_RVs:
        vv = model.rvs_to_values.get(rv)
        if vv is None or rv.name not in post.data_vars:
            continue
        shape = tuple(post[rv.name].shape[2:])
        size = int(np.prod(shape)) if shape else 1
        layout.append((rv.name, vv.name, start, size, shape))
        start += size
    return layout, start


def _posterior_matrix(idata, layout, D):
    """Stack posterior draws into (N, D) in the layout's column order."""
    post = idata.posterior
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    N = n_chain * n_draw
    X = np.empty((N, D), dtype=float)
    for rv_name, _vname, start, size, _shape in layout:
        arr = np.asarray(post[rv_name].values, dtype=float).reshape(N, size)
        X[:, start:start + size] = arr
    return X


def _batch_logp(model, layout, points):
    """logp (jacobian=True) at each row of points (M, D) via fork pool.

    jacobian=True gives the density over the unconstrained raw space -- the
    space the sampler explores and the space the proposal lives in -- so the
    bridge ratio is over one consistent measure.
    """
    M = points.shape[0]
    if M == 0:
        return np.zeros(0)
    with model:
        logp_fn = model.compile_logp(jacobian=True)

    # layout for the evaluator uses (value_name, start, size, shape)
    eval_layout = [(vname, start, size, shape)
                   for _rv, vname, start, size, shape in layout]

    global _EV_LP_FN, _EV_LAYOUT
    _EV_LP_FN = logp_fn
    _EV_LAYOUT = eval_layout

    n_workers = max(1, min(mp.cpu_count(), M))
    if n_workers == 1 or M < 64:
        _off, vals = _ev_eval_block((points, 0))
        return vals

    blocks = np.array_split(points, n_workers)
    args = []
    off = 0
    for b in blocks:
        args.append((b, off))
        off += b.shape[0]
    ctx = mp.get_context("fork")
    out = np.full(M, np.nan)
    with ctx.Pool(n_workers) as pool:
        for offset, vals in pool.map(_ev_eval_block, args):
            out[offset:offset + vals.size] = vals
    return out


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------

def _get_labels(idata, mode_report):
    """Per-draw mode labels, preferring the attached posterior variable."""
    post = idata.posterior
    if "mode" in post.data_vars:
        n = post.sizes["chain"] * post.sizes["draw"]
        return np.asarray(post["mode"].values, dtype=int).reshape(n)
    return np.asarray(mode_report.labels, dtype=int).reshape(-1)


def estimate_mode_evidences(model, idata, mode_report,
                            n_proposal=None,
                            max_posterior_draws=4000,
                            re2_max=DEFAULT_RE2_MAX,
                            seed=20260712):
    """Estimate each mode's local evidence by warp bridge sampling.

    Parameters
    ----------
    model : the PyMC model whose free RVs are the raw sampled variables.
    idata : InferenceData with the posterior (and ideally the ``mode`` label
        variable from identify_modes).
    mode_report : the ModeReport whose modes are to be weighted.
    n_proposal : proposal draws per mode (default: match the mode's posterior
        draw count, capped at max_posterior_draws).
    max_posterior_draws : subsample each mode's posterior draws to at most this
        many for the (many) logp evaluations.
    re2_max : refuse a mode whose bridge relative-MSE exceeds this.
    seed : RNG seed for reproducible proposal draws.

    Returns
    -------
    list[EvidenceResult], one per mode in mode_report.modes.

    Notes
    -----
    Optional free-evidence cross-checks (pm.sample_smc, dynesty nested
    sampling) were left unimplemented: they are explicitly lower priority than
    this bridge path and "logged but never trusted alone", so wiring them in
    without a trustworthy combination rule would add risk without changing the
    reported weight.  The bridge estimate is the single source of truth here.
    """
    rng = np.random.default_rng(seed)
    results = []

    layout, D = _build_layout(model, idata)
    if D == 0:
        logger.warning("estimate_mode_evidences: no free RVs match the "
                       "posterior; refusing all modes")
        for k in range(mode_report.n_modes):
            results.append(EvidenceResult(k, np.nan, np.inf, np.inf, 0, 0,
                                          True, "no usable free RVs"))
        return results

    Xall = _posterior_matrix(idata, layout, D)
    labels = _get_labels(idata, mode_report)

    # Fit proposals and stage every point that needs a logp evaluation, so all
    # modes' posterior + proposal points share a single fork-parallel logp pass.
    stage = []            # (mode, kind, X1, Y, mu, S)
    all_points = []
    for k in range(mode_report.n_modes):
        sel = labels == k
        X1 = Xall[sel]
        if X1.shape[0] < 4:
            stage.append((k, None, X1, None, None, None))
            continue
        if X1.shape[0] > max_posterior_draws:
            idx = rng.choice(X1.shape[0], max_posterior_draws, replace=False)
            X1 = X1[idx]
        mu, S = _fit_gaussian(X1)
        n2 = n_proposal or X1.shape[0]
        n2 = int(min(n2, max_posterior_draws))
        Y = _sample_gaussian(mu, S, n2, rng)
        stage.append((k, "ok", X1, Y, mu, S))
        all_points.append(X1)
        all_points.append(Y)

    if all_points:
        stacked = np.vstack(all_points)
        logp_all = _batch_logp(model, layout, stacked)
    else:
        logp_all = np.zeros(0)

    cursor = 0
    for k, kind, X1, Y, mu, S in stage:
        if kind is None:
            results.append(EvidenceResult(
                k, np.nan, np.inf, np.inf, int(X1.shape[0]), 0, True,
                "too few draws to fit a proposal"))
            continue
        n1, n2 = X1.shape[0], Y.shape[0]
        logp1 = logp_all[cursor:cursor + n1]
        cursor += n1
        logp2 = logp_all[cursor:cursor + n2]
        cursor += n2

        logq1 = _mvn_logpdf(X1, mu, S)
        logq2 = _mvn_logpdf(Y, mu, S)
        l1 = logp1 - logq1
        l2 = logp2 - logq2

        lnZ, lnZ_err, re2, converged = bridge_lnZ(l1, l2)

        refused = False
        reason = ""
        if not converged or not np.isfinite(lnZ):
            refused, reason = True, "bridge iteration did not converge"
        elif not np.isfinite(re2) or re2 > re2_max:
            refused = True
            reason = (f"relative-MSE diagnostic re2={re2:.3g} exceeds "
                      f"{re2_max:g} (proposal poorly supports the target -- "
                      f"likely a bound pileup); lnZ error bar ~{lnZ_err:.2g} nat")
        results.append(EvidenceResult(
            k, lnZ, lnZ_err, re2, n1, n2, refused, reason))
        if refused:
            logger.warning("evidence: mode %d refused (%s)", k + 1, reason)
        else:
            logger.info("evidence: mode %d lnZ=%.3f +/- %.3f (re2=%.3g)",
                        k + 1, lnZ, lnZ_err, re2)
    return results


def softmax_weights(lnZ, lnZ_err):
    """Softmax weights and linearized uncertainties from lnZ +/- lnZ_err.

    w_k = softmax(lnZ)_k; dw_k propagated from independent lnZ errors via the
    softmax Jacobian dw_k/dlnZ_j = w_k (delta_kj - w_j).
    """
    lnZ = np.asarray(lnZ, dtype=float)
    err = np.asarray(lnZ_err, dtype=float)
    w = np.exp(lnZ - np.max(lnZ))
    w = w / w.sum()
    dw = np.empty_like(w)
    for k in range(w.size):
        jac = w[k] * ((np.arange(w.size) == k).astype(float) - w)
        dw[k] = np.sqrt(np.sum((jac * err) ** 2))
    return w, dw


def apply_evidence_weighting(mode_report, results, re2_max=DEFAULT_RE2_MAX):
    """Replace occupancy weights with softmax evidence weights, if confident.

    Softmax evidence weights need every mode's lnZ, so a single refused mode
    invalidates the whole set: in that case the occupancy weights and their
    provenance are kept unchanged (with a note), honoring the project
    invariant that a shaky weight is never emitted.  Returns True iff evidence
    weights were applied.
    """
    if not results or len(results) != mode_report.n_modes:
        return False

    refused = [r for r in results if r.refused]
    if refused:
        ids = ", ".join(str(r.mode + 1) for r in refused)
        note = (f"evidence weighting refused for mode(s) {ids}; kept occupancy "
                f"weights. First reason: {refused[0].reason}")
        mode_report.notes.append(note)
        mode_report.provenance = (
            "occupancy (evidence weighting refused: " + refused[0].reason +
            "; see notes)")
        logger.warning("evidence weighting refused for mode(s) %s; "
                       "falling back to occupancy", ids)
        return False

    lnZ = np.array([r.lnZ for r in results], dtype=float)
    err = np.array([r.lnZ_err for r in results], dtype=float)
    w, dw = softmax_weights(lnZ, err)
    for k, m in enumerate(mode_report.modes):
        m.weight = float(w[k])
        m.weight_err = float(dw[k])
        m.lnZ = float(lnZ[k])
        m.lnZ_err = float(err[k])
    max_re2 = max(r.re2 for r in results)
    n_post = min(r.n_post for r in results)
    n_prop = min(r.n_prop for r in results)
    mode_report.provenance = (
        f"evidence (bridge sampling: max relative MSE {max_re2:.2g}, "
        f"N_post>={n_post}, N_prop>={n_prop}; lnZ error bars propagated to "
        f"weight uncertainties)")
    mode_report.weights_reliable = True
    mode_report.notes.append(
        "mode weights are bridge-sampling evidence weights: "
        + "; ".join(f"mode {r.mode + 1} lnZ={r.lnZ:.2f}+/-{r.lnZ_err:.2f}"
                    for r in results))
    logger.info("evidence weighting applied: weights=%s",
                [f"{x:.3f}" for x in w])
    return True
