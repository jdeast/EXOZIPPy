"""Per-mode local evidence estimation by bridge sampling.

The estimator is the plain Meng & Wong (1996) optimal bridge against a
Gaussian proposal fitted to the mode's own raw draws.  It is NOT warp bridge
sampling: no warp transformation of the target is applied anywhere in this
module (see the note under step 4 for the one place a warp would earn its
keep).  The module was titled "warp bridge sampling" until 2026-08-11, which
described an implementation that never existed.

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

  2. Proposal.  A normalized multivariate Gaussian -- the only proposal family
     implemented -- is fit to the mode's raw draws over the *full* free-RV
     vector (``_fit_gaussian``, shrunk toward its diagonal and eigenvalue-
     floored so the density is always proper).  Raw space is the non-centered
     N(0, 1) space, so a Gaussian is a natural, bound-free reference with an
     analytic normalizing constant.

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

     A proposal draw at which the target logp is NOT FINITE is refusal
     evidence too, and is handled here rather than thrown away -- see the
     "Unsupported proposal draws" note below.

     That bound-pileup case is exactly what a warp would fix, and is the
     obvious next step if refusals prove common: Warp-III (Meng & Schilling
     2002) symmetrizes the target about its mode by mixing x with 2*mu - x,
     turning the one-sided tail the Gaussian proposal cannot cover into a
     symmetric one it can.  Not implemented -- recorded here so the option is
     not rediscovered from scratch.

  5. Weights.  Accepted lnZ_k -> softmax weights w_k; lnZ error bars propagate
     to softmax weight uncertainties dw_k by linearization.

Unsupported proposal draws
--------------------------
The fitted Gaussian is unbounded, so some of its draws can land where the
target density is zero (logp = -inf) or where the logp evaluation fails
(nan).  Those draws are NOT dropped.  Dropping them and renormalizing the
proposal-side average over the survivors is exactly the estimator for a
proposal truncated to the target's support, which is not the proposal that
was drawn from: it inflates the numerator by N2 / N2_kept and biases lnZ
UPWARD by log(N2 / N2_kept) per mode.  Worse, it defeats the very diagnostic
meant to catch it -- the discarded draws are the ones with the smallest
bridge function, so removing them LOWERS re2 and the proposal is reported as
healthier than it is.

The bridge identity already has the right value for such a draw: the optimal
bridge function is p~/(s1 p~ + s2 Z q), which is exactly 0 where p~ = 0.  So
an unsupported draw enters the proposal-side average as a zero, N2 stays the
number of draws actually taken, lnZ stays unbiased, and the zeros widen the
spread of the bridge function that re2 measures.

That leaves a residual the error bar cannot express: a nan is a FAILED
evaluation, not a demonstrated zero, and scoring it as zero density biases
lnZ downward by up to log(1 / (1 - f)) nats for an unsupported fraction f.
That systematic has no sign and no error bar, so past a threshold the mode is
refused outright rather than reported (see DEFAULT_UNSUPPORTED_MAX).

The logp evaluations reuse run.py's fork-parallel pattern (compiled PyTensor
logp inherited by forked workers via copy-on-write; only numpy point arrays
cross the IPC boundary).

Optional cross-checks (pm.sample_smc, dynesty nested sampling) are intentionally
NOT implemented here: they are lower priority than the bridge path and are
"logged but never trusted alone".  See estimate_mode_evidences docstring.
"""

import logging
import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np

from .autocorr import iact as _iact

logger = logging.getLogger(__name__)


# Refuse a mode whose bridge relative-MSE exceeds this.  re2 is the estimator's
# relative mean-squared error on Z; sqrt(re2) is the coefficient of variation
# of Z and (to first order) the lnZ error bar.  0.25 -> refuse once the lnZ
# error bar reaches ~0.5 nat, at which point softmax weights are too soft to
# trust as a posterior-mass statement.
DEFAULT_RE2_MAX = 0.25


# Refuse a mode when this fraction of its proposal draws land where the
# target logp is not finite.
#
# Those draws are kept (as zero-density draws, which is their correct bridge
# value -- see the module docstring), so lnZ itself stays unbiased and re2
# does grow.  But the growth is only ~f / ((1 - f) N2): a purely Monte-Carlo
# effect, invisible at N2 ~ 1e3 for any f a real fit produces.  re2 is
# therefore the WRONG gate for this, and the honest one is the systematic the
# error bar cannot carry: nan means the logp evaluation FAILED, not that the
# density is demonstrably zero, so scoring the non-finite draws as zeros can
# shift lnZ by up to log(1 / (1 - f)) nats in an undetermined direction.
#
# 0.10 caps that unquantified systematic at 0.105 nat -- five times tighter
# than the 0.5-nat lnZ error bar at which re2_max already refuses, so it can
# never be the loose gate of the two, and small enough that softmax weights
# move by at most ~2.5 percentage points because of it.
#
# It is also far above what ordinary modes do.  Measured on real saved traces
# (fit the proposal exactly as estimate_mode_evidences does, then count the
# non-finite target logp among its draws):
#
#   KELT-4 RV-only (15 raw dims, 4 modes):        0 / 1933, 0 / 275,
#                                                 0 / 164,  0 / 108
#   DC2018_128 Roman binary lens (27 raw dims):   0 / 429,  0 / 424
#
# i.e. exactly zero on every mode of either fit, so 10% is not a threshold
# ordinary modes come anywhere near.  A model with hard-edged derived
# quantities (e < 1, a variance that can go negative) can produce a few
# percent; 10% is not reachable without the fitted Gaussian genuinely
# straddling an edge of the support, which is the case being refused.
DEFAULT_UNSUPPORTED_MAX = 0.10


# Outcome vocabulary for one mode's estimate, mirroring the four-state
# hot-search reporting in outputs/ledger.py: "it worked", "it ran and the
# diagnostics said no", and "it could not be attempted" must be
# distinguishable in the report, never collapsed into a bare missing number.
EV_OK = "ok"
EV_REFUSED = "refused"  # generic: a result carrying no state of its own
EV_NO_FREE_RVS = "no-free-rvs"
EV_TOO_FEW_DRAWS = "too-few-draws"
EV_UNSUPPORTED = "unsupported-proposal-mass"
EV_INVALID_POSTERIOR = "invalid-posterior-draws"
EV_NOT_CONVERGED = "not-converged"
EV_RE2 = "re2-too-large"


@dataclass
class EvidenceResult:
    """Bridge-sampling result for one mode.

    lnZ / lnZ_err are the mode's local log-evidence relative to its proposal
    and its (relative-MSE-derived) error bar.  ``refused`` is True when the
    diagnostics could not support a confident answer; ``reason`` says why.

    ``n_post`` / ``n_prop`` are the draw counts the estimate was actually
    computed FROM, so the provenance line describes what was used and not
    what was staged.  ``status`` carries the machine-readable outcome: a
    ``state`` from the EV_* vocabulary plus the raw counts
    (``n_prop_unsupported``, ``frac_unsupported``, ``n_post_invalid``, ...),
    following the status-dict pattern outputs/ledger.py uses for the
    hot-chain search.
    """

    mode: int
    lnZ: float
    lnZ_err: float
    re2: float
    n_post: int
    n_prop: int
    refused: bool
    reason: str = ""
    status: dict = field(default_factory=dict)

    @property
    def state(self):
        """EV_* outcome label.

        Falls back to the generic labels when ``status`` was never filled in
        (a hand-built result), so a renderer never has to print an empty
        bracket.
        """
        return self.status.get("state") or (
            EV_REFUSED if self.refused else EV_OK
        )


# ----------------------------------------------------------------------
# pure bridge-sampling core (no model/PyMC dependency -- unit testable)
# ----------------------------------------------------------------------


def _logmeanexp(a):
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if not np.isfinite(m):
        return m
    return m + np.log(np.mean(np.exp(a - m)))


# Fixed-point convergence tolerance, RELATIVE to the magnitude of lr itself.
# lr is a log-evidence at raw model-logp scale (|lr| ~ 1e4..1e6 on real
# datasets), and it is only ever known to float64 relative precision: at
# |lr| = 1e6 one ULP is already 1.2e-10, so the ABSOLUTE 1e-10 test this
# replaced sat inside the round-off floor of its own arithmetic and reported
# spurious non-convergence on ~25% of otherwise perfect estimates (measured;
# see tests).  1e-10 relative is ~4.5e5 ULPs -- comfortably above the
# sqrt(N)-term summation noise of the two _logmeanexp calls -- while the
# absolute step it admits, 1e-10 * |lnZ|, is 1e-4 nat even at |lnZ| = 1e6:
# four orders of magnitude below the 0.5-nat error bar at which a mode is
# refused, so nothing statistically meaningful can hide under it.  A bridge
# iteration that has genuinely not converged is still moving by O(0.01-1)
# nat per step, which is many orders above this at any scale.
DEFAULT_BRIDGE_RTOL = 1e-10


def _segments(values, mask, lengths):
    """Split ``values`` (already reduced by ``mask``) back into chain segments.

    ``lengths`` describes the un-filtered array; ``mask`` says which of its
    entries survived.  Returns one 1-D array per chain, in sampling order.
    """
    if lengths is None:
        return [values]
    out = []
    pos = 0  # cursor in the original array
    fpos = 0  # cursor in the filtered array
    for length in lengths:
        kept = int(mask[pos : pos + length].sum())
        out.append(values[fpos : fpos + kept])
        pos += length
        fpos += kept
    return out


def bridge_lnZ(
    l1,
    l2,
    maxiter=1000,
    tol=DEFAULT_BRIDGE_RTOL,
    l1_chains=None,
    status=None,
):
    """Optimal iterative bridge-sampling estimate of a log normalizing constant.

    Parameters
    ----------
    l1 : array, log(unnormalized target) - log(proposal) at the *posterior*
        draws (samples from the target), **in sampling order**.
    l2 : array, the same log-ratio at fresh draws from the *proposal*.
    l1_chains : sequence of int, optional -- the per-chain lengths that
        partition ``l1`` (they must sum to ``len(l1)``).  Given, the IACT that
        inflates the posterior-side error term is measured chain by chain and
        combined; omitted, ``l1`` is treated as one long chain, which
        concatenates independent chains end to end and biases the IACT.
    maxiter : maximum fixed-point iterations.
    tol : RELATIVE convergence tolerance on lr (see DEFAULT_BRIDGE_RTOL);
        the effective absolute threshold is ``tol * max(1, |lr|)``.
    status : dict, optional -- filled in with the draw bookkeeping the caller
        needs to judge the result: ``n_prop`` / ``n_prop_unsupported`` /
        ``frac_unsupported`` (proposal draws taken, of which the target logp
        was not finite), and ``n_post`` / ``n_post_invalid`` /
        ``frac_post_invalid`` (posterior draws used, of which some had to be
        discarded).  Same in/out dict convention as
        outputs.ledger.discover_hot_modes.

    Returns
    -------
    (lnZ, lnZ_err, re2, converged)
        lnZ is the estimated log-evidence of the target relative to the
        (normalized) proposal; re2 is the approximate relative mean-squared
        error of the Z estimate (Fruhwirth-Schnatter 2004) and
        lnZ_err = sqrt(re2).

    Notes
    -----
    Non-finite entries of ``l2`` are proposal draws the target does not
    support.  They are RETAINED at their correct bridge value of zero rather
    than dropped -- dropping them renormalizes the proposal-side average over
    a proposal nobody sampled from and biases lnZ up by log(N2 / N2_kept)
    (see the module docstring).  Non-finite entries of ``l1`` have no such
    correct value -- a draw the sampler produced cannot have zero target
    density, so it is a failed evaluation -- and are discarded and counted.

    Everything downstream of the fixed point is computed **recentered by lr**.
    The textbook bridge diagnostic is written with

        f1 = exp(l2) / (s1 exp(l2) + s2 r),   f2 = 1 / (s1 exp(l1) + s2 r)

    at r = exp(lr), and both expressions overflow or underflow to nan/inf as
    soon as |lr| or |l| exceeds ~709 -- which is *always*, since l1 and l2
    carry the raw model logp (-1e4 .. -1e6 for any real dataset).  Dividing
    through by exp(l2) and by r respectively gives the algebraically identical

        f1 = 1 / (s1 + s2 exp(lr - l2)),      f2' = 1 / (s1 exp(l1 - lr) + s2)

    where every exponent is a *difference* of same-scale log quantities and so
    stays O(1).  f1 is unchanged; f2' = r * f2 is f2 rescaled by a constant,
    and only the scale-free ratios ``var/mean^2`` (and the IACT, which is
    computed from the normalized autocorrelation) enter re2 -- so re2 and
    lnZ_err are exactly the quantities the unrecentered form intends, and
    reproduce it to float precision wherever it was representable at all.
    """
    if status is None:
        status = {}
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    finite1 = np.isfinite(l1)
    finite2 = np.isfinite(l2)
    n_prop_taken = int(l2.size)
    n_unsupported = int(n_prop_taken - finite2.sum())
    n_post_invalid = int(l1.size - finite1.sum())
    # Posterior side: no valid substitute, so drop-and-count.
    l1 = l1[finite1]
    # Proposal side: the target puts no mass there, and the optimal bridge
    # function IS zero there, so keep the draw and let it count as a zero.
    # -inf is the exact log of that zero and propagates correctly through
    # both the fixed point (logaddexp) and the diagnostic (1/inf -> 0).
    l2 = np.where(finite2, l2, -np.inf)
    N1, N2 = l1.size, l2.size
    status.update(
        n_prop=N2,
        n_prop_unsupported=n_unsupported,
        frac_unsupported=(n_unsupported / N2 if N2 else 0.0),
        n_post=N1,
        n_post_invalid=n_post_invalid,
        frac_post_invalid=(
            n_post_invalid / (N1 + n_post_invalid)
            if (N1 + n_post_invalid)
            else 0.0
        ),
    )
    if N1 < 2 or N2 < 2:
        return np.nan, np.inf, np.inf, False

    log_s1 = np.log(N1 / (N1 + N2))
    log_s2 = np.log(N2 / (N1 + N2))

    lr = 0.0
    converged = False
    for _ in range(maxiter):
        # log(num_j) over proposal draws l2; log(den_i) over posterior draws l1
        log_num = l2 - np.logaddexp(log_s1 + l2, log_s2 + lr)
        log_den = -np.logaddexp(log_s1 + l1, log_s2 + lr)
        lr_new = _logmeanexp(log_num) - _logmeanexp(log_den)
        if not np.isfinite(lr_new):
            return np.nan, np.inf, np.inf, False
        if abs(lr_new - lr) < tol * max(1.0, abs(lr_new)):
            lr = lr_new
            converged = True
            break
        lr = lr_new

    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)
    # Recentered bridge functions -- see the Notes above.  f1 lives on the
    # proposal draws (i.i.d. by construction), f2 on the posterior draws.
    with np.errstate(over="ignore"):
        f1 = 1.0 / (s1 + s2 * np.exp(lr - l2))
        f2 = 1.0 / (s1 * np.exp(l1 - lr) + s2)
    m1, m2 = f1.mean(), f2.mean()
    if not (np.isfinite(m1) and np.isfinite(m2)) or m1 <= 0 or m2 <= 0:
        return float(lr), np.inf, np.inf, converged
    # f2 is a POSTERIOR-draw series: it inherits the sampler's
    # autocorrelation, so its contribution to the estimator variance is
    # inflated by the IACT.  The caller must hand the posterior draws over in
    # sampling order, one entry per chain (see _posterior_chains); measuring
    # the IACT of a reordered series silently returns ~1 and understates the
    # error bar by sqrt(tau).
    tau = _iact(_segments(f2, finite1, l1_chains))
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
_EV_LAYOUT = None  # list of (value_name, start, size, shape)


def _ev_eval_block(args):
    block, offset = args
    n = block.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        row = block[i]
        point = {}
        for vname, start, size, shape in _EV_LAYOUT:
            # ``shape`` is the VALUE VARIABLE's shape, not the trace's -- see
            # _build_layout.  pymc's compiled logp checks ndim exactly and the
            # except below swallows the mismatch as a nan, so getting this
            # wrong refuses every mode of a whole model type silently.
            point[vname] = row[start : start + size].reshape(shape)
        try:
            out[i] = float(_EV_LP_FN(point))
        except Exception:
            out[i] = np.nan
    return offset, out


def _build_layout(model, idata):
    """Column layout of the concatenated free-RV raw vector.

    Returns (layout, D, skipped) where layout is a list of
    (rv_name, value_name, start, size, shape), D is the total dimension, and
    ``skipped`` names the free RVs that could not be laid out at all.

    A skipped RV is a LAYOUT/TRACE MISMATCH, and it has to be reported as
    one.  The model's logp still needs a value for it, so every evaluation
    of every draw comes back non-finite, and the mode is then refused as
    EV_INVALID_POSTERIOR -- whose whole argument ("a draw the sampler
    produced cannot have zero target density, so these are failed
    evaluations") indicts the draws, when the draws are fine and the trace
    simply does not carry one of the model's variables.  The caller names
    them in the warning, in the refusal reason and in the status dict.
    rv_name indexes idata.posterior; value_name is the logp_fn input key.

    ``size`` comes from the trace (it is just an element count) but ``shape``
    is reconciled against the VALUE VARIABLE, because the two disagree in a
    case every real fit hits: an EXOZIPPy Parameter with one element is a
    length-1 vector RV, ``shape=(1,)``, and the trace stores it SQUEEZED to
    0-d.  pymc's compiled logp checks ndim exactly and _ev_eval_block's broad
    except turns the mismatch into a nan, so taking the shape from the trace
    would silently nan every draw of every one-element parameter -- and
    taking it from the trace unconditionally likewise breaks a genuinely 0-d
    RV in the other direction.  Neither failure is visible except as a mode
    refused for a reason that is not the real one.
    """
    layout = []
    skipped = []
    start = 0
    post = idata.posterior
    for rv in model.free_RVs:
        vv = model.rvs_to_values.get(rv)
        if vv is None or rv.name not in post.data_vars:
            skipped.append(rv.name)
            continue
        trace_shape = tuple(post[rv.name].shape[2:])
        size = int(np.prod(trace_shape)) if trace_shape else 1
        ndim = vv.type.ndim
        if ndim == len(trace_shape):
            shape = trace_shape
        elif ndim == 0:
            shape = ()  # trace kept a length-1 axis the model does not have
        elif ndim == 1:
            shape = (size,)  # trace squeezed the model's length-1 vector
        else:
            static = tuple(rv.type.shape)
            shape = (
                static
                if len(static) == ndim and None not in static
                else trace_shape
            )
        layout.append((rv.name, vv.name, start, size, shape))
        start += size
    return layout, start, skipped


def _posterior_matrix(idata, layout, D):
    """Stack posterior draws into (N, D) in the layout's column order."""
    post = idata.posterior
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    N = n_chain * n_draw
    X = np.empty((N, D), dtype=float)
    for rv_name, _vname, start, size, _shape in layout:
        arr = np.asarray(post[rv_name].values, dtype=float).reshape(N, size)
        X[:, start : start + size] = arr
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
    eval_layout = [
        (vname, start, size, shape)
        for _rv, vname, start, size, shape in layout
    ]

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
            out[offset : offset + vals.size] = vals
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


def _mode_draw_index(labels, mode, n_chain, n_draw, max_draws):
    """Row indices of ``mode``'s draws, in sampling order, thinned by stride.

    Returns ``(index, chain_lengths)``: ``index`` selects rows of the
    (chain, draw) row-major posterior matrix, grouped by chain and ascending
    in draw within each chain; ``chain_lengths`` partitions it back into
    chains for the IACT.

    The thinning is a **uniform stride applied within each chain**, never an
    unsorted ``rng.choice``: the retained series has to stay a time series or
    the IACT of the bridge function reads ~1 whatever the sampler did, and
    the posterior-side error term is then understated by sqrt(tau) (a factor
    of ~20 on a chain with tau ~ 400).  No separate thinning correction is
    needed downstream -- the mean the estimator forms is the mean of exactly
    this strided series, so its own IACT is the right inflation factor.

    A mode's draws within a chain need not be contiguous (the chain may have
    left and returned); the retained subsequence is treated as a stationary
    series, which is the same approximation the stride already makes.
    """
    sel_idx = np.flatnonzero(labels == mode)
    if sel_idx.size == 0:
        return sel_idx, []
    stride = max(1, int(np.ceil(sel_idx.size / max(max_draws, 1))))
    chain_of = sel_idx // n_draw
    kept = []
    chain_lengths = []
    for c in range(n_chain):
        idx_c = sel_idx[chain_of == c][::stride]
        if idx_c.size:
            kept.append(idx_c)
            chain_lengths.append(int(idx_c.size))
    if not kept:
        return np.zeros(0, dtype=int), []
    return np.concatenate(kept), chain_lengths


def estimate_mode_evidences(
    model,
    idata,
    mode_report,
    n_proposal=None,
    max_posterior_draws=4000,
    re2_max=DEFAULT_RE2_MAX,
    unsupported_max=DEFAULT_UNSUPPORTED_MAX,
    seed=20260712,
):
    """Estimate each mode's local evidence by bridge sampling.

    Meng & Wong (1996) optimal bridge against a Gaussian proposal fitted to
    each mode's own raw draws; no warp is applied (see the module docstring).

    Parameters
    ----------
    model : the PyMC model whose free RVs are the raw sampled variables.
    idata : InferenceData with the posterior (and ideally the ``mode`` label
        variable from identify_modes).
    mode_report : the ModeReport whose modes are to be weighted.
    n_proposal : proposal draws per mode (default: match the mode's posterior
        draw count, capped at max_posterior_draws).
    max_posterior_draws : thin each mode's posterior draws to at most this
        many for the (many) logp evaluations.  Thinning is a per-chain stride
        so the retained draws stay a time series (see _mode_draw_index).
    re2_max : refuse a mode whose bridge relative-MSE exceeds this.
    unsupported_max : refuse a mode when more than this fraction of its
        proposal draws land where the target logp is not finite (see
        DEFAULT_UNSUPPORTED_MAX).  Those draws are counted and kept as
        zero-density draws either way -- they are never dropped.
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

    layout, D, skipped_rvs = _build_layout(model, idata)
    # Named here, once, before anything is evaluated: a free RV the trace
    # does not carry makes EVERY logp evaluation non-finite, and the refusal
    # that follows blames the draws.  See _build_layout.
    if skipped_rvs:
        logger.warning(
            "evidence: %d of the model's free variables are absent from the "
            "posterior and were left out of the bridge layout (%s). The "
            "model logp still needs them, so every draw will re-evaluate to "
            "a non-finite logp -- this is a layout/trace mismatch, not bad "
            "draws.",
            len(skipped_rvs),
            ", ".join(skipped_rvs),
        )
    skipped_note = (
        ""
        if not skipped_rvs
        else (
            f"; NOTE {len(skipped_rvs)} free variable(s) absent from the "
            f"posterior were omitted from the bridge layout "
            f"({', '.join(skipped_rvs)}), which alone makes every "
            f"evaluation non-finite"
        )
    )
    if D == 0:
        logger.warning(
            "estimate_mode_evidences: no free RVs match the "
            "posterior; refusing all modes"
        )
        for k in range(mode_report.n_modes):
            results.append(
                EvidenceResult(
                    k,
                    np.nan,
                    np.inf,
                    np.inf,
                    0,
                    0,
                    True,
                    "no usable free RVs" + skipped_note,
                    {
                        "state": EV_NO_FREE_RVS,
                        "skipped_rvs": list(skipped_rvs),
                        "n_skipped_rvs": len(skipped_rvs),
                    },
                )
            )
        return results

    Xall = _posterior_matrix(idata, layout, D)
    labels = _get_labels(idata, mode_report)
    n_chain = int(idata.posterior.sizes["chain"])
    n_draw = int(idata.posterior.sizes["draw"])

    # Fit proposals and stage every point that needs a logp evaluation, so all
    # modes' posterior + proposal points share a single fork-parallel logp pass.
    stage = []  # (mode, kind, X1, Y, mu, S, chain_lengths)
    all_points = []
    for k in range(mode_report.n_modes):
        sel_idx, chain_lengths = _mode_draw_index(
            labels, k, n_chain, n_draw, max_posterior_draws
        )
        X1 = Xall[sel_idx]
        if X1.shape[0] < 4:
            stage.append((k, None, X1, None, None, None, None))
            continue
        mu, S = _fit_gaussian(X1)
        n2 = n_proposal or X1.shape[0]
        n2 = int(min(n2, max_posterior_draws))
        Y = _sample_gaussian(mu, S, n2, rng)
        stage.append((k, "ok", X1, Y, mu, S, chain_lengths))
        all_points.append(X1)
        all_points.append(Y)

    if all_points:
        stacked = np.vstack(all_points)
        logp_all = _batch_logp(model, layout, stacked)
    else:
        logp_all = np.zeros(0)

    cursor = 0
    for k, kind, X1, Y, mu, S, chain_lengths in stage:
        if kind is None:
            results.append(
                EvidenceResult(
                    k,
                    np.nan,
                    np.inf,
                    np.inf,
                    int(X1.shape[0]),
                    0,
                    True,
                    "too few draws to fit a proposal",
                    {"state": EV_TOO_FEW_DRAWS},
                )
            )
            continue
        n1, n2 = X1.shape[0], Y.shape[0]
        logp1 = logp_all[cursor : cursor + n1]
        cursor += n1
        logp2 = logp_all[cursor : cursor + n2]
        cursor += n2

        logq1 = _mvn_logpdf(X1, mu, S)
        logq2 = _mvn_logpdf(Y, mu, S)
        l1 = logp1 - logq1
        l2 = logp2 - logq2

        status = {
            "skipped_rvs": list(skipped_rvs),
            "n_skipped_rvs": len(skipped_rvs),
        }
        lnZ, lnZ_err, re2, converged = bridge_lnZ(
            l1, l2, l1_chains=chain_lengths, status=status
        )

        n_unsup = status.get("n_prop_unsupported", 0)
        frac_unsup = status.get("frac_unsupported", 0.0)
        n_bad_post = status.get("n_post_invalid", 0)
        frac_bad_post = status.get("frac_post_invalid", 0.0)
        # The counts above are already IN the estimate: the unsupported
        # proposal draws entered the bridge average as zeros (unbiased lnZ,
        # and their spread is inside re2).  What follows decides whether the
        # part of them that cannot be quantified -- a nan is a failed
        # evaluation, not a demonstrated zero -- is too large to report
        # around.  See DEFAULT_UNSUPPORTED_MAX.
        refused = False
        reason = ""
        state = EV_OK
        # Posterior side first: a draw the sampler produced cannot have zero
        # target density, so a non-finite logp there is an unambiguous
        # EVALUATION failure and diagnoses itself.  Reading such a run as a
        # proposal-support problem would name the wrong cause -- and when the
        # evaluation is broken, the proposal side is non-finite too.
        if frac_bad_post > unsupported_max:
            refused = True
            state = EV_INVALID_POSTERIOR
            reason = (
                f"{n_bad_post} of {n_bad_post + n1} posterior draws "
                f"({frac_bad_post:.1%}) re-evaluated to a non-finite logp, "
                f"above the {unsupported_max:.0%} limit; a draw the sampler "
                f"produced cannot have zero target density, so these are "
                f"failed evaluations and the mode's draws cannot be trusted"
                + skipped_note
            )
        elif frac_unsup > unsupported_max:
            refused = True
            state = EV_UNSUPPORTED
            # log1p(-1) is -inf, and an all-unsupported proposal is exactly
            # the case that reaches this; say inf rather than warn.
            syst = (
                np.inf if frac_unsup >= 1.0 else float(-np.log1p(-frac_unsup))
            )
            reason = (
                f"{n_unsup} of {n2} proposal draws ({frac_unsup:.1%}) land "
                f"where the target logp is not finite, over the "
                f"{unsupported_max:.0%} limit: the fitted Gaussian puts mass "
                f"outside the target's support, and lnZ then carries up to "
                f"{syst:.2f} nat of systematic that no error bar covers"
            )
        elif not converged or not np.isfinite(lnZ):
            refused, state = True, EV_NOT_CONVERGED
            reason = "bridge iteration did not converge"
        elif not np.isfinite(re2) or re2 > re2_max:
            refused = True
            state = EV_RE2
            reason = (
                f"relative-MSE diagnostic re2={re2:.3g} exceeds "
                f"{re2_max:g} (proposal poorly supports the target -- "
                f"likely a bound pileup); lnZ error bar ~{lnZ_err:.2g} nat"
            )
        status["state"] = state
        results.append(
            EvidenceResult(
                k,
                lnZ,
                lnZ_err,
                re2,
                status.get("n_post", n1),
                status.get("n_prop", n2),
                refused,
                reason,
                status,
            )
        )
        if refused:
            logger.warning(
                "evidence: mode %d refused [%s] (%s)", k + 1, state, reason
            )
        else:
            logger.info(
                "evidence: mode %d lnZ=%.3f +/- %.3f (re2=%.3g, "
                "N_post=%d, N_prop=%d of which %d unsupported)",
                k + 1,
                lnZ,
                lnZ_err,
                re2,
                status.get("n_post", n1),
                status.get("n_prop", n2),
                n_unsup,
            )
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


def evidence_status_to_text(results):
    """One human-readable line per mode's estimate outcome.

    Rendered into the mode report so a reader can tell an estimate that ran
    and was believed from one that ran and was refused, and WHY -- the same
    reason outputs/ledger.py renders its four hot-search states instead of
    letting "nothing happened" and "it found nothing" print identically.
    """
    lines = []
    for r in results:
        counts = (
            f"N_post={r.n_post}, N_prop={r.n_prop}"
            f" ({r.status.get('n_prop_unsupported', 0)} outside the target "
            f"support, kept as zero-density draws)"
        )
        if r.refused:
            lines.append(
                f"mode {r.mode + 1}: REFUSED [{r.state}] -- {r.reason} "
                f"[{counts}]"
            )
        else:
            lines.append(
                f"mode {r.mode + 1}: lnZ={r.lnZ:.2f}+/-{r.lnZ_err:.2f} "
                f"(re2={r.re2:.3g}; {counts})"
            )
    return lines


def apply_evidence_weighting(
    mode_report, results, re2_max=DEFAULT_RE2_MAX, idata=None
):
    """Replace occupancy weights with softmax evidence weights, if confident.

    Softmax evidence weights need every mode's lnZ, so a single refused mode
    invalidates the whole set: in that case the occupancy weights and their
    provenance are kept unchanged (with a note), honoring the project
    invariant that a shaky weight is never emitted.  Returns True iff evidence
    weights were applied.

    Each mode's occupancy weight and its mixing-derived error bar survive on
    ``ModeInfo.occ_weight`` / ``occ_weight_err`` either way, so the report can
    print the two side by side -- evidence weighting is a CROSS-CHECK on
    occupancy, and the comparison is the point.

    ``idata``, when given, has its ``posterior['mode'].attrs`` refreshed: that
    variable is built by ``ModeReport.attach`` at identify_modes time and
    would otherwise keep advertising the superseded occupancy weights and
    provenance to anything reading the trace back from disk.
    """
    if not results or len(results) != mode_report.n_modes:
        return False

    refused = [r for r in results if r.refused]
    if refused:
        ids = ", ".join(str(r.mode + 1) for r in refused)
        note = (
            f"evidence weighting refused for mode(s) {ids}; kept occupancy "
            f"weights. Reasons: "
            + "; ".join(
                f"mode {r.mode + 1} [{r.state}] {r.reason}" for r in refused
            )
        )
        mode_report.notes.append(note)
        # Every mode's outcome, not just the refused ones: an accepted mode
        # sitting next to a refused one is part of the picture a reader needs.
        mode_report.notes.append(
            "bridge-sampling evidence per mode -- "
            + "; ".join(evidence_status_to_text(results))
        )
        mode_report.provenance = (
            f"occupancy (evidence weighting refused for mode(s) {ids} "
            f"[{refused[0].state}]: " + refused[0].reason + "; see notes)"
        )
        logger.warning(
            "evidence weighting refused for mode(s) %s; "
            "falling back to occupancy",
            ids,
        )
        if idata is not None:
            mode_report.attach(idata)
        return False

    lnZ = np.array([r.lnZ for r in results], dtype=float)
    err = np.array([r.lnZ_err for r in results], dtype=float)
    w, dw = softmax_weights(lnZ, err)
    for k, m in enumerate(mode_report.modes):
        if not np.isfinite(m.occ_weight):
            # Reports built before this ran (or by hand) still get a record
            # of what occupancy said, so the cross-check stays possible.
            m.occ_weight = float(m.weight)
            m.occ_weight_err = float(m.weight_err)
        m.weight = float(w[k])
        m.weight_err = float(dw[k])
        m.lnZ = float(lnZ[k])
        m.lnZ_err = float(err[k])
    max_re2 = max(r.re2 for r in results)
    # Counts as USED, not as staged: these numbers exist so a reader can tell
    # what the estimate was computed from.  They used to be quoted before the
    # non-finite draws were removed, which described a sample size the
    # estimator never saw.  Nothing is silently removed any more -- the
    # unsupported proposal draws are kept as zero-density draws -- so the
    # count is quoted alongside instead of hidden.
    n_post = min(r.n_post for r in results)
    n_prop = min(r.n_prop for r in results)
    n_unsup = sum(r.status.get("n_prop_unsupported", 0) for r in results)
    n_bad_post = sum(r.status.get("n_post_invalid", 0) for r in results)
    used = (
        f"N_post>={n_post} used"
        + (f" ({n_bad_post} discarded as non-finite)" if n_bad_post else "")
        + f", N_prop>={n_prop} used"
        + (
            f" ({n_unsup} outside the target support, kept as zero-density "
            f"draws)"
            if n_unsup
            else ""
        )
    )
    mode_report.provenance = (
        f"evidence (bridge sampling: max relative MSE {max_re2:.2g}, "
        f"{used}; lnZ error bars propagated to weight uncertainties)"
    )
    mode_report.weights_reliable = True
    mode_report.notes.append(
        "mode weights are bridge-sampling evidence weights: "
        + "; ".join(evidence_status_to_text(results))
    )
    # Cross-check line: evidence weighting needs no mode transitions at all,
    # occupancy needs them and reports its own mixing-derived error bar, so
    # disagreement beyond the two error bars localizes the problem (ladder
    # tuning on the occupancy side, proposal fit on the evidence side).
    disagree = []
    for m in mode_report.modes:
        if not np.isfinite(m.occ_weight):
            continue
        spread = np.hypot(
            m.weight_err if np.isfinite(m.weight_err) else 0.0,
            m.occ_weight_err if np.isfinite(m.occ_weight_err) else 0.0,
        )
        gap = abs(m.weight - m.occ_weight)
        disagree.append(
            f"mode {m.index + 1}: evidence {m.weight:.3f}+/-"
            f"{m.weight_err:.3f} vs occupancy {m.occ_weight:.3f}+/-"
            f"{m.occ_weight_err:.3f}"
            + (
                "  <-- differ by > 3 sigma"
                if gap > 3 * max(spread, 1e-12)
                else ""
            )
        )
    if disagree:
        mode_report.notes.append(
            "evidence-vs-occupancy cross-check -- " + "; ".join(disagree)
        )
    if idata is not None:
        mode_report.attach(idata)
    logger.info(
        "evidence weighting applied: weights=%s", [f"{x:.3f}" for x in w]
    )
    return True
