"""Pre-whitening seed polish: promote each solution-estimate start to its
basin's optimum BEFORE anything downstream consumes the start.

Why before whitening: the startup probe (whitening.probe_scales) measures
logp contours around the start.  From a start far below its basin's optimum
the contours are gradient-dominated and the measured scales come out
arbitrarily tight -- on examples/ob140939 a start ~5900 nats low (raw error
bars underestimated; err_scale starting at 1.0) measured scales ~1000x too
small, NUTS's mass matrix re-widened the raw posterior to sd ~5e3 units, and
86% of draws diverged against parameter.py's _RAW_CANCELLATION_CLIP wall at
|raw| = 1e4.  Polishing first fixes that at the source; the probe then
measures curvature at a genuine optimum, the barrier steepness measurement
uses honest unit steps, PTDE scatters chains around a real basin center, and
NUTS starts inside the typical set.  This generalizes the PTDE-only seed
polish (samplers/ptde.py polish_seed_starts, PR #56), which ran inside
_make_starts -- after the whitening probe had already measured its scales
around the unpolished start, and never on the NUTS path at all.

Two engines, dispatched on gradient availability:

- L-BFGS-B on the compiled logp + gradient in raw space (raw space is
  unconstrained -- hard bounds live inside the logit transform -- and smooth:
  bounds are soft barriers, not -inf walls).  ~100-300 evaluations to the
  mode.  The tolerances are deliberately LOOSE (the goal is "within a few
  nats", not the exact MAP): for hierarchical/scale-like parameters the exact
  MAP is not in the typical set and can run toward degenerate corners, and a
  loose stop both avoids that and keeps the cost trivial.
- The PR #56 T=1 DE-MC polish (gradient-free, the sampler's own move) when
  the gradient graph cannot be built or is non-finite at the start -- e.g.
  the binary-lens magnification Op has no analytic gradient.

Stopping: `seed_polish: N` (default DEFAULT_POLISH_STEPS) is a step CAP, not
a target -- "at most N".  The two engines stop differently, because they see
different information and the differentiable one's criterion has no
counterpart on the other path:

- L-BFGS-B stops on the GRADIENT NORM (_LBFGS_GTOL, nats per raw unit) with
  maxiter as its safety net.  That is a real statement about the local
  surface and needs no history.  Measured on examples/ob140939 (4 literature
  seeds): 2 of 4 reach |grad| < 0.01 at 432 and 655 iterations; the other 2
  are still crawling along a flat direction at 5000, where 33x the default
  cap buys 9 nats out of a 547-nat climb (1.6%) and doubles the wall time.
  So the cap is not a stand-in for the tolerance -- it is the bound on
  hierarchical-MAP drift it was documented to be, and 150 is a reasonable
  place for it.  Do NOT layer a logp-improvement rule on top: the
  _LBFGS_FTOL note below records what per-iteration improvement tests do to
  a curved valley.
- The gradient-free DE polish (samplers.ptde.polish_seed_starts) has no
  gradient by construction -- it exists because the binary-lens
  magnification Op has none -- and its only observable, the best-lp history,
  is a STAIRCASE of exactly-flat plateaus.  A best-lp improvement window
  therefore cannot separate "converged" from "has not jumped yet": on
  examples/DC2018_128 it stops 38-137 nats short, by the SAME amount for
  tol = 0.05, 0.5 and 2.0 nats.  The measurement is tabulated on
  ptde.POLISH_TOL_NATS.  That engine's default stopping criterion is
  therefore the step cap; the tolerance is an opt-in
  (polish_raw_starts(tol=...)) for a surface known to be smooth.

The cap always remains, so nothing can polish forever.

Seed-provenance gate (resolve_polish_steps): 'auto' polishes SOLUTION
ESTIMATES -- the single canonical start (user/literature initvals, the
relaxation engine's solution) and MMEXOFAST seed sets -- but never a
multi-seed set WITHOUT seed hints, which is a posterior-draw restart
(mkprior stratified draws): those are already at equilibrium, and polishing
K draws per basin would collapse them onto K copies of the basin optimum,
destroying the restart's overdispersion.
"""

import logging
import multiprocessing as mp

import numpy as np

logger = logging.getLogger(__name__)

# Step CAP on the polish -- "at most this many" -- not a target.  On the
# L-BFGS path the gradient tolerance usually ends it first; on the
# gradient-free path the cap IS the criterion (see "Stopping" in the module
# docstring for the measurement behind both statements).
DEFAULT_POLISH_STEPS = 150

# L-BFGS stopping: terminate on the GRADIENT (plus the maxiter cap),
# never on per-iteration improvement. scipy's `ftol` fires on the FIRST
# iteration whose gain is small -- and it is RELATIVE to |f|, so the old
# 1e-3 quit whenever an iteration gained < 1e-3*|lp| (~2 nats at
# lp ~ 1900), stranding ob140939's seeds ~15 nats below their basin
# peaks; even an absolute per-iteration threshold dies on a slow first
# bend of a curved valley (measured: iteration 1 gains 0.004 nats, the
# remaining 2.0 arrive over the next 33). ftol is therefore disabled.
# gtol is nats per raw unit -- one preliminary whitening scale, so
# 0.01 nat/unit is deep inside the flat top of the basin. maxiter stays
# as the guard against hierarchical-MAP collapse (scale-like parameters
# can run toward degenerate corners if polished without bound).
_LBFGS_FTOL = 1e-12  # effectively off; gtol + maxiter terminate
_LBFGS_GTOL = 1e-2
# Non-finite logp guard: L-BFGS line searches handle inf poorly, so a
# non-finite evaluation returns this plateau plus a quadratic pull back
# toward the last finite iterate's neighborhood (gradient points home).
_NONFINITE_PENALTY = 1e15

# Sentinel: "caller said nothing", so the DE engine's own defaults apply.
# `None` cannot serve -- tol=None is the meaningful "disable the tolerance,
# run the full n_steps" request.
_UNSET = object()


def resolve_polish_steps(spec, n_seeds, has_seed_hints):
    """Map the sampler-config `seed_polish` value to a step CAP.

    'auto' (default): DEFAULT_POLISH_STEPS when the starts are solution
    estimates -- a single canonical start (n_seeds == 1) or component-pushed
    seed hints (MMEXOFAST) -- and 0 for a multi-seed set without hints
    (posterior-draw restarts; see module docstring).  True/'on' and
    False/None/'off' force it; an int gives the cap directly (`seed_polish: N`
    = "at most N steps", not "exactly N" -- both engines stop on their own
    tolerance first; see "Stopping" in the module docstring).

    The bool test comes FIRST and by isinstance.  `spec in (True, "on")`
    matched the integer 1 (1 == True in Python), so `seed_polish: 1` asked
    for one step and got 150 (notes/code_review_20260808.txt 2.9.1).  The
    symmetric `0 == False` match was harmless -- 0 steps IS off -- and stays
    harmless here: 0 now falls through to the int path and returns 0.
    """
    if isinstance(spec, bool):
        return DEFAULT_POLISH_STEPS if spec else 0
    if spec is None:
        return 0
    if isinstance(spec, str):
        key = spec.lower()
        if key == "auto":
            return (
                DEFAULT_POLISH_STEPS if (n_seeds == 1 or has_seed_hints) else 0
            )
        if key == "on":
            return DEFAULT_POLISH_STEPS
        if key == "off":
            return 0
    return max(0, int(spec))


def _compile_logp_grad(model):
    """Compiled point-function returning [logp, grad_1, ..., grad_n] over
    model.value_vars, or None when the gradient graph cannot be built (an
    Op without an analytic gradient, e.g. binary-lens magnification)."""
    import pytensor

    value_vars = list(model.value_vars)
    try:
        lp_node = model.logp()
        grads = pytensor.grad(lp_node, wrt=value_vars)
        return model.compile_fn(
            [lp_node] + list(grads),
            inputs=value_vars,
            on_unused_input="ignore",
        )
    except Exception as e:
        logger.info(
            f"Seed polish: gradient graph unavailable ({type(e).__name__}: "
            f"{e}); falling back to the gradient-free DE polish."
        )
        return None


def _lbfgs_polish_one(center, fn_lp_grad, keys, shapes, sizes, maxiter):
    """L-BFGS-B ascent of logp from one raw start dict.

    Stops on the gradient norm (_LBFGS_GTOL); `maxiter` is the safety cap.
    Returns (polished_dict, lp0, lp_best, n_evals, n_iter, hit_cap)."""
    from scipy.optimize import minimize

    def flatten(d):
        return np.concatenate(
            [np.asarray(d[k], dtype=float).reshape(-1) for k in keys]
        )

    def unflatten(x):
        out, ofs = {}, 0
        for k, shp, n in zip(keys, shapes, sizes):
            out[k] = x[ofs : ofs + n].reshape(shp)
            ofs += n
        return out

    x0 = flatten(center)
    n_evals = [0]

    def objective(x):
        n_evals[0] += 1
        vals = fn_lp_grad(unflatten(x))
        lp = float(vals[0])
        if not np.isfinite(lp):
            return (
                _NONFINITE_PENALTY + float(np.sum((x - x0) ** 2)),
                2.0 * (x - x0),
            )
        g = np.concatenate(
            [np.asarray(v, dtype=float).reshape(-1) for v in vals[1:]]
        )
        if not np.all(np.isfinite(g)):
            g = np.where(np.isfinite(g), g, 0.0)
        return -lp, -g

    lp0 = -objective(x0)[0]
    res = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(maxiter),
            "maxfun": int(max(4 * maxiter, 200)),
            "ftol": _LBFGS_FTOL,
            "gtol": _LBFGS_GTOL,
        },
    )
    # res.x is the best iterate L-BFGS-B saw; never worse than the start
    # except pathological line-search exits -- guard anyway.
    lp_best = -float(res.fun)
    n_iter = int(getattr(res, "nit", 0))
    hit_cap = n_iter >= int(maxiter)
    if np.isfinite(lp_best) and lp_best >= lp0:
        return unflatten(res.x), lp0, lp_best, n_evals[0], n_iter, hit_cap
    return (
        {k: np.array(v, dtype=float, copy=True) for k, v in center.items()},
        lp0,
        lp0,
        n_evals[0],
        n_iter,
        hit_cap,
    )


def polish_raw_starts(
    model,
    raw_starts,
    n_steps=DEFAULT_POLISH_STEPS,
    seed_indices=None,
    logp_fn=None,
    rng=None,
    tol=_UNSET,
    tol_window=_UNSET,
    cores=None,
    adapt_gamma=_UNSET,
):
    """Polish each raw start toward its own basin's optimum.

    Dispatch: L-BFGS-B on logp+grad when the model's gradient graph builds
    and is finite at seed 0; otherwise the PR #56 T=1 DE-MC polish
    (samplers/ptde.polish_seed_starts) with unit jitter scales (one raw unit
    = one preliminary whitening scale, DE's population self-adapts from
    there).

    ``n_steps`` is the safety CAP for either engine; each stops on its own
    tolerance first (see "Stopping" in the module docstring).  ``tol`` /
    ``tol_window`` override the DE engine's tolerance (``tol=None`` restores
    a fixed ``n_steps`` sweeps); they do not reach the L-BFGS path, which
    stops on _LBFGS_GTOL.

    Returns (polished_starts, dlps, method) with method in
    {"lbfgs", "de", "none"}.  A seed is never made worse: any engine result
    below the seed's own lp is discarded in favor of the seed.
    """
    if isinstance(raw_starts, dict):
        raw_starts = [raw_starts]
    if seed_indices is None:
        seed_indices = list(range(len(raw_starts)))
    keys = list(raw_starts[0].keys())
    shapes = [np.shape(raw_starts[0][k]) for k in keys]
    sizes = [int(np.asarray(raw_starts[0][k]).size) for k in keys]

    fn_lp_grad = _compile_logp_grad(model)
    if fn_lp_grad is not None:
        vals = fn_lp_grad(raw_starts[0])
        grad_finite = np.all(
            [np.all(np.isfinite(np.asarray(v))) for v in vals]
        )
        if not grad_finite:
            logger.info(
                "Seed polish: gradient non-finite at the start; falling "
                "back to the gradient-free DE polish."
            )
            fn_lp_grad = None

    if fn_lp_grad is not None:
        polished, dlps = [], []
        for s, center in enumerate(raw_starts):
            best, lp0, lp_best, n_evals, n_iter, hit_cap = _lbfgs_polish_one(
                center, fn_lp_grad, keys, shapes, sizes, maxiter=n_steps
            )
            polished.append(best)
            dlps.append(lp_best - lp0)
            reason = (
                f"hit the {int(n_steps)}-iteration cap"
                if hit_cap
                else f"converged: |grad| < {_LBFGS_GTOL} nats/unit"
            )
            logger.info(
                f"Seed polish (L-BFGS): seed {seed_indices[s]} lp "
                f"{lp0:.1f} -> {lp_best:.1f} (dlp=+{lp_best - lp0:.1f}, "
                f"{n_iter} iterations / {n_evals} evaluations, {reason})"
            )
        return polished, dlps, "lbfgs"

    # Gradient-free fallback: the PR #56 polish, jittering at one raw unit.
    from .samplers import _common
    from .samplers.ptde import polish_seed_starts

    if logp_fn is None:
        logp_fn = model.compile_logp()
    if rng is None:
        rng = np.random.default_rng(0)
    scales = {
        k: np.ones(np.shape(raw_starts[0][k]), dtype=float) for k in keys
    }
    de_kwargs = {}
    if tol is not _UNSET:
        de_kwargs["tol"] = tol
    if tol_window is not _UNSET:
        de_kwargs["tol_window"] = tol_window
    if adapt_gamma is not _UNSET:
        de_kwargs["adapt_gamma"] = adapt_gamma

    # The DE engine is a population method with nothing shared between
    # members within a sweep, so it parallelizes exactly as the sampler
    # does -- and until this was wired up it ran SERIAL while the job held
    # every core the sampler was about to use.  Same worker contract as
    # ptde_async: install the compiled logp in _common BEFORE forking so
    # children inherit it copy-on-write, then hand the pool
    # _common._eval_logp (module-level, so picklable by reference).
    n_proc = _resolve_polish_cores(cores, len(raw_starts))
    pool = None
    if n_proc > 1:
        _common.set_worker_globals(logp_fn)
        pool = mp.Pool(processes=n_proc)
        logger.info(
            f"Seed polish: DE engine on {n_proc} worker process(es), "
            f"proposals pooled across all {len(raw_starts)} seed(s)."
        )
    try:
        polished, dlps = polish_seed_starts(
            raw_starts,
            _common._eval_logp if pool is not None else logp_fn,
            rng,
            scales,
            n_steps=n_steps,
            pool=pool,
            **de_kwargs,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return polished, dlps, "de"


def _resolve_polish_cores(cores, n_seeds):
    """Worker count for the DE polish.

    Capped at the batch size (n_seeds * pop_size) upstream by there simply
    being nothing more to hand out; here we only guard against asking for
    more processes than the machine has, and against the degenerate 1.
    """
    if cores is None:
        return 1
    try:
        n = int(cores)
    except (TypeError, ValueError):
        return 1
    if n <= 1:
        return 1
    return max(1, min(n, mp.cpu_count()))
