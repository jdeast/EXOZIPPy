"""Pre-whitening seed polish: promote each solution-estimate start to its
basin's optimum BEFORE anything downstream consumes the start.

Why before whitening: the startup probe (whitening.probe_scales) measures
logp contours around the start.  From a start far below its basin's optimum
the contours are gradient-dominated and the measured scales come out
arbitrarily tight -- on examples/ob140939 a start ~5900 nats low (2026-08,
raw error bars underestimated; err_scale starting at 1.0) measured scales
~1000x too small, NUTS's mass matrix re-widened the raw posterior to sd ~5e3
units, and 86% of draws diverged against parameter.py's
_RAW_CANCELLATION_CLIP wall at |raw| = 1e4.  (The same example's four
literature seeds start 565-3493 nats below their polished values today,
2026-09-14.)  Polishing first fixes that at the source; the probe then
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

Stopping: `seed_polish: N` (default: each engine's own cap,
DEFAULT_POLISH_STEPS L-BFGS iterations or DEFAULT_DE_POLISH_SWEEPS DE sweeps
per round) is a step CAP, not a target -- "at most N".  The two engines stop
differently, because they see different information and the differentiable
one's criterion has no counterpart on the other path:

- L-BFGS-B stops on the GRADIENT NORM (_LBFGS_GTOL, nats per raw unit) with
  maxiter as its safety net.  That is a real statement about the local
  surface and needs no history -- but it is a statement about the CURRENT
  iterate only, so the tolerance has to be tight enough that the first
  iterate to satisfy it is at the optimum and not on a shoulder: at the
  old 1e-2 it was a first dip mid-climb on examples/kelt4's tc/logP ridge,
  and one ulp of arithmetic moved the polished cosi by 8.5% (the note on
  _LBFGS_GTOL below has the measurement; review 7.13.8).  At the shipped
  1e-4 kelt4 converges in 240-294 iterations across 16 perturbed
  arithmetics.  Measured on examples/ob140939 (4 literature seeds, 17 raw
  coordinates, 2026-09-14) under the shipped gtol = 1e-4 / cap 400: every
  seed climbs 565-3493 nats and every seed is a CAP-STOP at 400 (|grad|
  0.03-0.3 at the stop), sitting 6-8 nats below the best value a 9000-
  20000-iteration run reaches -- and 400 -> 5000 iterations buys 3-8 of
  those nats for 12.5x the wall time (0.25 s -> 3 s per seed), with seeds
  2 and 3 still crawling along a flat direction at 20000.  Under the old
  1e-2 / 150 the same four seeds were cap-stops too, 0.3-1.9 nats lower.
  So the cap is not a stand-in for the tolerance -- it is the bound on
  hierarchical-MAP drift it was documented to be -- and on ob140939 it is
  the stop that fires, by design: a flat direction that gives back <1.5%
  of the climb per decade of iterations is exactly what the cap exists to
  bound.  Do NOT layer a logp-improvement rule on top: the _LBFGS_FTOL
  note below records what per-iteration improvement tests do to a curved
  valley.
- The gradient-free DE polish (samplers.ptde.polish_seed_starts) has no
  gradient by construction -- it exists because the binary-lens
  magnification Op has none -- and its only observable, the best-lp history,
  is a STAIRCASE of exactly-flat plateaus, so no improvement threshold
  separates "converged" from "has not jumped yet" (ptde.POLISH_TOL_WINDOW
  has the measurements).  It is therefore run in ROUNDS (polish_rounds,
  review 2.4.14, JDE ruling 2026-09-14): each round runs at most
  DEFAULT_DE_POLISH_SWEEPS sweeps and a seed leaves it early when its best
  point gains less than polish_tol_nats(D) over the last
  ptde.POLISH_TOL_WINDOW = 400 sweeps ("stopped on rate"); between rounds
  the model is re-whitened at the polished point and the next round starts
  a FRESH population there.  The loop ends when a whole round gains less
  than polish_tol_nats(D), or at POLISH_MAX_ROUNDS.  A stop on a flat step
  is therefore cheap -- one probe and a restart -- and the last word belongs
  to a fresh population in measured coordinates, never to a stale one.

The caps always remain (per round and on the number of rounds), so nothing
can polish forever.

Seed-provenance gate (resolve_polish_steps): 'auto' polishes SOLUTION
ESTIMATES -- the single canonical start (user/literature initvals, the
relaxation engine's solution) and component seed sets -- but never a
multi-seed set WITHOUT seed hints, which is a posterior-draw restart
(mkparam stratified draws): those are already at equilibrium, and polishing
K draws per basin would collapse them onto K copies of the basin optimum,
destroying the restart's overdispersion.  Nor, since review 2.4.14 (d), a
params file that DECLARES its seeds to be posterior draws
(`overdisperse: false`, which mkparam writes for K > 1): the declaration is
the writer's statement, and seed hints pushed by a component on top of such
a file used to switch the polish back on and collapse the draws.
"""

import logging
import multiprocessing as mp

import numpy as np

logger = logging.getLogger(__name__)

# Step CAP on the L-BFGS polish -- "at most this many" iterations -- not a
# target: the gradient tolerance usually ends it first (see "Stopping" in the
# module docstring for the measurement).
DEFAULT_POLISH_STEPS = 400

# Per-ROUND sweep cap on the gradient-free DE polish (review 2.4.14, JDE
# ruling 2026-09-14: "raise the cap").  Order 10^3, not 10^4, because that is
# where a fresh population does its work: on examples/DC2018_128's two seeds
# run to 15000 sweeps, the first 1000 brought 80% / 74% of the total gain and
# the next 14000 the rest, at 14x the cost, from a population that had
# frozen (80-90% of its sweeps accepting nothing).  More sweeps than that go
# to a NEW round -- re-whitened, re-populated -- instead (polish_rounds).
DEFAULT_DE_POLISH_SWEEPS = 1000

# Cap on the number of DE polish rounds.  With the per-round cap this bounds
# the polish at 8000 sweeps; the accepted price (JDE 2026-09-14: CPU-days, on
# EXOFASTv2 precedent) is the wall clock, not the bound.
POLISH_MAX_ROUNDS = 8


class _EngineDefaultCap:
    """Sentinel step cap: "the dispatched engine's own default".

    The two engines have different caps (DEFAULT_POLISH_STEPS iterations for
    L-BFGS, DEFAULT_DE_POLISH_SWEEPS sweeps per round for DE), and which one
    runs is only known once the gradient graph has been tried -- so
    resolve_polish_steps cannot return a number for 'auto'/'on'.  Truthy, so
    `if polish_steps:` still reads "polish".
    """

    def __repr__(self):
        return "ENGINE_DEFAULT"


ENGINE_DEFAULT = _EngineDefaultCap()


def polish_tol_nats(n_params):
    """The DE polish's improvement threshold, in nats, for a D-dim model.

    max(1, D/2): a T=1 posterior's typical set lies ~D/2 nats below its
    maximum (the mean of chi2_D / 2), so a polish gain smaller than that
    moves the start by less than the depth the sampler will spread its chains
    over anyway -- it is invisible to everything downstream.  Used twice, for
    the same reason: as the in-round window threshold (a seed that gains less
    than this over ptde.POLISH_TOL_WINDOW sweeps stops) and as the round
    threshold (a round that gains less than this ends the loop).  The floor
    keeps a 1-parameter model from stopping on a half-nat jitter.  ABSOLUTE
    nats; see _LBFGS_FTOL for why never relative to |lp|.  Measured on the
    DC2018_128 histories the window's stop does not depend on the threshold
    (tol 1, 13.5 = D/2 and 50 nats stop on the same sweep): the staircase
    makes the window LENGTH the operative choice.
    """
    return max(1.0, 0.5 * float(n_params))


# L-BFGS stopping: terminate on the GRADIENT (plus the maxiter cap),
# never on per-iteration improvement. scipy's `ftol` fires on the FIRST
# iteration whose gain is small -- and it is RELATIVE to |f|, so the old
# 1e-3 quit whenever an iteration gained < 1e-3*|lp| (~2 nats at
# lp ~ 1900), stranding ob140939's seeds ~15 nats below their basin
# peaks when first measured (2026-08); re-measured 2026-09-14 on the
# current model (lp ~ 10600, so the threshold is ~10 nats/iteration) it
# quits after 6-11 iterations, 10-119 nats below where the gradient stop
# + cap land the same seeds.  Even an absolute per-iteration threshold
# dies on a slow first bend of a curved valley (measured: iteration 1
# gains 0.004 nats, the remaining 2.0 arrive over the next 33). ftol is
# therefore disabled.
#
# gtol is nats per raw unit -- one preliminary whitening scale.  It was
# 1e-2 ("deep inside the flat top of the basin"), and that reasoning is
# right about a well-conditioned basin and WRONG about a ridge, which is
# what a real start climbs: scipy's gtol test is `max|proj g| <= gtol`
# on the CURRENT iterate, so it fires on the FIRST evaluation that dips
# under the threshold, whether or not the neighbours do.  Measured on
# examples/kelt4 RV-only (review 7.13.8, 2026-09-14): of 177 evaluations
# exactly one had |grad|_inf < 0.01 -- the last; the one before it read
# 0.017 and four before that 0.19 -- and it fired 0.507 nats BELOW the
# basin optimum (81.440 against 81.947) on a ridge whose Hessian has
# condition number 5.4e6 (the tc/logP degeneracy; orbit.0.tc sits ~1200
# orbits from the RV data), at iteration 148 of a 150 cap.  A first-dip
# stop on such a ridge is an amplifier: a synthetic ONE-ULP perturbation
# of (lp, grad) -- the objective times (1 + s*2^-52), i.e. the same
# function computed by a different but equally correct arithmetic --
# moved the polished cosi across 0.487..0.530 (8.5% relative width over
# 16 seeds), the planet mass by 2.1%, the polished lp by 0.24 nats, and
# 3 of the 16 arithmetics hit the 150 cap.  That is the whole mechanism
# behind the cross-CI-runner scatter of tests/test_integration_kelt4.py:
# a different OpenBLAS kernel inside scipy's own L-BFGS-B bookkeeping
# (not the objective, which has no BLAS op) perturbs one iterate by 1
# ulp at evaluation 4 and the stop lands elsewhere on the shoulder.
#
# At 1e-4 the same 16 perturbed arithmetics span 8.6e-4 in cosi, 3.0e-4
# in the mass and 1e-6 nats in lp, all sitting at the basin optimum,
# for ~120 more iterations (240-294 against 134-150) and +0.05 s.  The
# cap is 400 so that band -- and the wider one on a model with several
# such ridges -- is not a cap-stop: 150 already capped 3/16 arithmetics
# at the OLD tolerance.  maxiter stays as the guard against
# hierarchical-MAP collapse (scale-like parameters can run toward
# degenerate corners if polished without bound); on ob140939 it is the
# stop for all four literature seeds, and that is its documented job
# (module docstring).  tests/test_polish.py pins the perturbation spread under
# the shipped constants so a first-dip stop cannot come back unnoticed.
_LBFGS_FTOL = 1e-12  # effectively off; gtol + maxiter terminate
_LBFGS_GTOL = 1e-4
# Non-finite logp guard: L-BFGS line searches handle inf poorly, so a
# non-finite evaluation returns this plateau plus |x - x0|^2, whose gradient
# 2*(x - x0) points back at THE POLISH START x0 -- not at the last finite
# iterate, which the objective never records.  x0 is the fixed seed the
# polish was handed (and logp/grad there are checked finite before the
# L-BFGS path is taken at all), so the pull is always toward a point the
# logp is known to be defined at; the quadratic is what makes the plateau
# navigable, since a flat 1e15 gives the line search no direction.
_NONFINITE_PENALTY = 1e15

# Sentinel: "caller said nothing", so the DE engine's own defaults apply.
# `None` cannot serve -- tol=None is the meaningful "disable the tolerance,
# run the full n_steps" request.
_UNSET = object()


def resolve_polish_steps(spec, n_seeds, has_seed_hints, *, overdisperse):
    """Map the sampler-config `seed_polish` value to a step CAP.

    'auto' (default): ENGINE_DEFAULT (the dispatched engine's own cap) when
    the starts are solution estimates -- a single canonical start
    (n_seeds == 1) or component-pushed seed hints (the peak finder) -- and 0
    for a multi-seed set without hints (posterior-draw restarts; see module
    docstring), and 0 whenever the params file declares
    `overdisperse: false` (``overdisperse``, System.overdisperse): the
    writer's statement that its seeds ARE posterior draws outranks seed
    hints a component pushed on top of them (review 2.4.14 (d)).
    True/'on' and False/None/'off' force it; an int gives the cap directly
    (`seed_polish: N` = "at most N steps" -- per ROUND on the DE engine --
    not "exactly N"; both engines stop on their own criterion first; see
    "Stopping" in the module docstring).  Forcing a polish onto a declared
    posterior-draw set is honored but warned about.

    ``overdisperse`` is REQUIRED: it was the missing input (2.4.14 (d)), and
    a default would let the next call site forget it the same way.

    The bool test comes FIRST and by isinstance.  `spec in (True, "on")`
    matched the integer 1 (1 == True in Python), so `seed_polish: 1` asked
    for one step and got 150 (review 2.9.1, #104).  The
    symmetric `0 == False` match was harmless -- 0 steps IS off -- and stays
    harmless here: 0 now falls through to the int path and returns 0.
    """
    if not isinstance(overdisperse, bool):
        raise TypeError(
            f"resolve_polish_steps: overdisperse must be the params file's "
            f"boolean declaration (System.overdisperse), got {overdisperse!r}"
        )
    if isinstance(spec, bool):
        cap = ENGINE_DEFAULT if spec else 0
    elif spec is None:
        cap = 0
    elif isinstance(spec, str) and spec.lower() == "auto":
        if not overdisperse:
            return 0
        return ENGINE_DEFAULT if (n_seeds == 1 or has_seed_hints) else 0
    elif isinstance(spec, str) and spec.lower() == "on":
        cap = ENGINE_DEFAULT
    elif isinstance(spec, str) and spec.lower() == "off":
        cap = 0
    else:
        cap = max(0, int(spec))
    if cap and not overdisperse and n_seeds > 1:
        logger.warning(
            f"seed_polish: {spec!r} forces the polish onto {n_seeds} seeds "
            f"the params file declares to be posterior draws "
            f"(`overdisperse: false`).  Each will be driven to its basin "
            f"optimum, which collapses the draws' spread; set "
            f"`seed_polish: auto` to keep them as written."
        )
    return cap


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
    n_steps=ENGINE_DEFAULT,
    seed_indices=None,
    logp_fn=None,
    rng=None,
    tol=_UNSET,
    tol_window=_UNSET,
    cores=None,
    adapt_gamma=_UNSET,
    eval_timeout=None,
    asynchronous=True,
    engine=None,
):
    """Polish each raw start toward its own basin's optimum.

    Dispatch: L-BFGS-B on logp+grad when the model's gradient graph builds
    and is finite at seed 0; otherwise the PR #56 T=1 DE-MC polish
    (samplers/ptde.polish_seed_starts) with unit jitter scales (one raw unit
    = one preliminary whitening scale, DE's population self-adapts from
    there).

    ``n_steps`` is the safety CAP for either engine (ENGINE_DEFAULT: the
    dispatched engine's own, DEFAULT_POLISH_STEPS iterations or
    DEFAULT_DE_POLISH_SWEEPS sweeps); each stops on its own criterion first
    (see "Stopping" in the module docstring).  On the DE engine that
    criterion is the improvement window, ON by default here:
    ``polish_tol_nats(D)`` nats over the last ptde.POLISH_TOL_WINDOW sweeps.
    ``tol`` / ``tol_window`` override it (``tol=None`` restores a fixed
    ``n_steps`` sweeps); they do not reach the L-BFGS path, which stops on
    _LBFGS_GTOL.  This is ONE round; polish_rounds is the pipeline's loop.

    ``engine`` (None = dispatch on the gradient, as above) may be ``"de"`` to
    skip the gradient attempt -- what polish_rounds passes for its later
    rounds, where round 1 has already found the graph has no usable
    gradient.

    ``cores`` is the DE engine's worker grant; None means AUTO (the same
    rule a sampler uses when nothing names one), and ``cores=1`` is how a
    caller asks for serial.  The L-BFGS path ignores it -- it is a few
    hundred evaluations and forks nothing.

    ``eval_timeout`` (seconds, default None = wait forever) is the DE
    engine's per-logp-call wall-clock budget, the same contract the PTDE
    samplers' ``sampler: eval_timeout:`` key carries: a call that exceeds it
    is abandoned and scored -inf, and the pool it wedged is recycled before
    the next batch.  It needs a pool (``cores > 1``), and the L-BFGS path
    ignores it -- scipy calls the gradient function in-process, where there
    is nothing to time out against.  **run.py does not currently pass one**;
    see run.md for why that is a config-vocabulary decision rather than an
    oversight.

    ``asynchronous`` (default True) selects the DE engine's ptde_async-style
    loop on a real pool -- one proposal in flight per population member,
    results consumed in arrival order, so one slow VBM evaluation costs one
    worker and not the batch (review 2.4.14).  ``False`` restores the
    synchronous sweep-batch engine, which is bit-reproducible for a given
    rng and the only one the serial path runs.  See
    ``ptde.polish_seed_starts``.

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

    if engine not in (None, "de"):
        raise ValueError(
            f"polish_raw_starts: engine={engine!r}; expected None (dispatch "
            f"on the gradient) or 'de'"
        )
    lbfgs_cap = (
        DEFAULT_POLISH_STEPS if n_steps is ENGINE_DEFAULT else int(n_steps)
    )
    de_cap = (
        DEFAULT_DE_POLISH_SWEEPS if n_steps is ENGINE_DEFAULT else int(n_steps)
    )

    fn_lp_grad = None if engine == "de" else _compile_logp_grad(model)
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
                center, fn_lp_grad, keys, shapes, sizes, maxiter=lbfgs_cap
            )
            polished.append(best)
            dlps.append(lp_best - lp0)
            reason = (
                f"hit the {lbfgs_cap}-iteration cap"
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
    # The improvement window is ON by default on this path (the module
    # docstring's "Stopping"; ptde.POLISH_TOL_WINDOW has the measurements).
    n_params = int(sum(sizes))
    de_kwargs = {"tol": polish_tol_nats(n_params)}
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
            f"proposals pooled across all {len(raw_starts)} seed(s), "
            f"at most {de_cap} sweeps."
        )
    else:
        # The serial case is ANNOUNCED, not silent.  "gradient graph
        # unavailable" above reads as a note about capability; what it
        # actually means for the user is the expensive branch, and on one
        # core it is the whole wall clock of this stage (review 6.11.3,
        # examples/ob09020: 1 core of 36 for 38 minutes).  Since cores=None
        # now means AUTO, reaching here at all means somebody asked for
        # serial -- or the machine has one core -- so the line reports the
        # request rather than accusing the caller of forgetting.
        logger.info(
            f"Seed polish: DE engine running SERIAL on one core "
            f"(cores={cores!r}), {len(raw_starts)} seed(s), at most "
            f"{de_cap} sweeps; this gradient-free branch is far more "
            f"expensive than L-BFGS."
        )
    _common.warn_serial_eval_timeout(
        eval_timeout, pool, n_proc, "Seed polish", logger
    )

    def _recycle(dead):
        """Swap in a fresh pool after a logp call wedged a worker.

        THIS FUNCTION IS WHY THE POOL STAYS OURS.  polish_seed_starts cannot
        own the teardown -- it is handed `pool` and does not know how many
        workers to fork -- so it calls back here and we rebind the name the
        `finally` below tears down.  Without the rebind that `finally` would
        close the corpse and leak the live pool's workers for the rest of
        the process.
        """
        nonlocal pool
        pool = _common.recycle_pool(dead, n_proc)
        return pool

    try:
        polished, dlps = polish_seed_starts(
            raw_starts,
            _common._eval_logp if pool is not None else logp_fn,
            rng,
            scales,
            n_steps=de_cap,
            pool=pool,
            eval_timeout=eval_timeout,
            pool_recycler=_recycle if pool is not None else None,
            asynchronous=asynchronous,
            **de_kwargs,
        )
    finally:
        if pool is not None:
            # terminate(), never close() + join(): a worker wedged in a
            # pathological logp never finishes its task, so close() leaves it
            # running and join() waits for it forever -- and a recycled pool
            # has SIGTERM-ignoring workers on top of that (review 2.4.1).
            _common._shutdown_pool(pool)
    return polished, dlps, "de"


def _seed_physical(lookup, raw):
    """{raw key: full physical element vector} for one raw start dict, under
    the CURRENT whitening (Parameter.phys_from_raw)."""
    from .system import _parameter_for_raw_key

    return {
        key: np.asarray(
            _parameter_for_raw_key(key, lookup).phys_from_raw(vec),
            dtype=float,
        )
        for key, vec in raw.items()
    }


def _seed_raw(lookup, phys, template):
    """Inverse of _seed_physical under the CURRENT whitening: the raw start
    dict (shaped like ``template``) of a physical point."""
    from .system import _parameter_for_raw_key

    return {
        key: np.asarray(
            _parameter_for_raw_key(key, lookup).raw_from_initval(phys[key]),
            dtype=float,
        ).reshape(np.shape(template[key]))
        for key in template
    }


def _raw_distance(a, b):
    return float(
        np.sqrt(
            sum(np.sum((np.asarray(a[k]) - np.asarray(b[k])) ** 2) for k in a)
        )
    )


def _warn_if_seeds_converged(system, model, lookup, origins_phys, r):
    """Across-round basin-coverage check for a multi-seed set.

    Each round's trust region (ptde.polish_seed_starts) is built from that
    round's own starts, so it guarantees per round that no seed crosses the
    midpoint toward a neighbour -- but across rounds the midpoints move,
    and the re-whitening rescales every axis by a different factor, so a
    region carried over from round 1 is not a region in round 2's
    coordinates at all.  Measured: carrying round 1's region forward
    (centred on the ORIGINAL seeds, half the ORIGINAL separation, re-expressed
    in the re-whitened coordinates) put DC2018_128's polished seeds OUTSIDE
    their own regions, and round 2 refused 6030 / 6795 Metropolis-accepted
    proposals -- all of them.  So the protection stays per round, and this
    check makes the cumulative drift VISIBLE instead: it compares the seeds'
    separation now with their original separation, both measured in the
    current coordinates, and warns past the same 1/4 the per-round check
    uses.
    """
    raws, idx = system.get_raw_starts(model)
    orig = [_seed_raw(lookup, ph, raws[0]) for ph in origins_phys]
    for a in range(len(raws)):
        for b in range(a + 1, len(raws)):
            d0 = _raw_distance(orig[a], orig[b])
            d1 = _raw_distance(raws[a], raws[b])
            if d0 > 0 and d1 < 0.25 * d0:
                logger.warning(
                    f"Seed polish: after round {r}, seeds {idx[a]} and "
                    f"{idx[b]} are {d1:.1f} scale units apart against "
                    f"{d0:.1f} originally (both measured in the current "
                    f"coordinates) -- the rounds have walked them toward "
                    f"one basin; the sampler may start these chains in what "
                    f"is effectively one basin."
                )


def polish_rounds(
    system,
    model,
    n_steps=ENGINE_DEFAULT,
    *,
    cores,
    rewhiten,
    max_rounds=POLISH_MAX_ROUNDS,
    tol=None,
    rng=None,
):
    """The pipeline's seed polish: polish, re-whiten, polish again, until a
    round stops paying (review 2.4.14, JDE ruling 2026-09-14 part 3).

    Round 1 is exactly one polish_raw_starts over the starts get_raw_starts
    builds, adopted with apply_polished_starts -- what run.py did before.
    On the L-BFGS engine that is also the LAST round: it stops on the
    gradient, which is a statement about the surface, and re-running it
    from its own optimum is out of this item's scope.  So every model with a
    gradient keeps a bit-identical start.

    On the DE engine the loop continues, because a DE round ends for reasons
    that are NOT statements about the surface: the population freezes (the
    fixed 2.38/sqrt(2D) step accepts ~0.3% at T=1, and 80-90% of sweeps
    accept nothing after ~1000 on DC2018_128), and -- in round 1 -- every
    proposal is drawn in PRELIMINARY whitening units, because the probe has
    not measured anything yet.  JDE: "The probe measures scales AT the start
    and DE proposals are drawn in those units, so a bad start hobbles the
    optimizer meant to escape it."  Each later round therefore:

      1. re-centers the whitening anchor on the polished start
         (recenter_whitening_anchor) and re-whitens there
         (whitening.measure_and_whiten -- the same probe run.py runs next,
         in the same order; skipped when ``rewhiten`` is False, i.e.
         `measure_scales: false`, which asks for no probe);
      2. re-derives every seed in the new raw coordinates (get_raw_starts);
      3. polishes EVERY seed with a FRESH population jittered at one (now
         measured) unit around it.  Every seed, not only the ones still
         climbing: the multi-seed trust region is built from the seeds in
         the call, and a lone seed would get an infinite one;
      4. adopts the result, as round 1 did.

    The anchor is NOT re-centered after the last round: run.py does that
    next, unconditionally, on every path, and owns it.

    Stopping: when no seed's round gained ``tol`` nats (default
    polish_tol_nats(D)), or after ``max_rounds`` rounds.  When re-whitening
    is on, round 2 always runs: round 1's gain in preliminary units is not
    evidence about the measured surface (on the 30-event static DC2018 sweep
    the 400-sweep polish bought EXACTLY 0.0 nats on seed 0 in 26/30 events).
    Gains are comparable across rounds because each round's is measured
    from that round's own start, AFTER the re-whitening: the whitening is a
    reparameterization (whitening.md), and the only thing the probe moves
    in logp is the soft-bound barrier steepness, which is therefore never
    counted as a polish gain.

    The final whitening is NOT measured here: run.py's prepare_whitening
    probes once more at the final point and persists it, exactly as on
    every other path.

    ``n_steps`` is the per-ROUND cap (resolve_polish_steps' value); ``cores``
    the DE engine's worker grant.  Returns a summary dict: ``method``,
    ``rounds`` (one {seed index: gain} dict per round), ``stop``
    ("lbfgs", "converged" or "round cap") and ``wall_s``.
    """
    import time

    from .whitening import measure_and_whiten

    t0 = time.monotonic()
    raw_starts, seed_indices = system.get_raw_starts(model)
    logp_fn = model.compile_logp()
    if rng is None:
        rng = np.random.default_rng(0)
    n_params = int(sum(np.asarray(v).size for v in raw_starts[0].values()))
    if tol is None:
        tol = polish_tol_nats(n_params)
    lookup = {p.label: p for p in system.get_all_parameters()}
    # The ORIGINAL seeds, physically, for the across-round coverage check.
    origins_phys = (
        [_seed_physical(lookup, r) for r in raw_starts]
        if len(raw_starts) > 1
        else None
    )

    polished, dlps, method = polish_raw_starts(
        model,
        raw_starts,
        n_steps=n_steps,
        seed_indices=seed_indices,
        logp_fn=logp_fn,
        rng=rng,
        cores=cores,
    )
    system.apply_polished_starts(polished, seed_indices)
    rounds = [dict(zip(seed_indices, (float(d) for d in dlps)))]
    summary = {"method": method, "rounds": rounds, "stop": "lbfgs"}
    if method != "de":
        summary["wall_s"] = time.monotonic() - t0
        return summary

    def _wants_another():
        if rewhiten and len(rounds) == 1:
            return True
        return any(d >= tol for d in rounds[-1].values())

    while _wants_another() and len(rounds) < int(max_rounds):
        r = len(rounds) + 1
        forced = rewhiten and r == 2
        climbing = [k for k, d in rounds[-1].items() if d >= tol]
        why = (
            "round 2 always runs when re-whitening: round 1 drew its "
            "proposals in preliminary units"
            if forced
            else f"seed(s) {climbing} gained >= {tol:.3g} nats last round"
        )
        how = (
            "re-whitening at the polished point"
            if rewhiten
            else "restarting at the polished point (measure_scales is off, "
            "so no re-whitening)"
        )
        logger.info(f"Seed polish round {r}/{int(max_rounds)}: {how}; {why}.")
        # Re-center first, exactly as run.py orders it before its own probe:
        # the probe measures around the anchor, which must be the start.
        system.recenter_whitening_anchor(model)
        if rewhiten:
            measure_and_whiten(
                system, model, system.get_raw_start(model), logp_fn
            )
        raw_all, idx_all = system.get_raw_starts(model)
        if list(idx_all) != list(seed_indices):
            raise RuntimeError(
                f"polish_rounds: round {r} re-derived seed indices "
                f"{list(idx_all)} where round 1 had {list(seed_indices)}; "
                f"apply_polished_starts writes every seed back, so the set "
                f"cannot change between rounds"
            )
        polished, dlps, _m = polish_raw_starts(
            model,
            raw_all,
            n_steps=n_steps,
            seed_indices=seed_indices,
            logp_fn=logp_fn,
            rng=rng,
            cores=cores,
            engine="de",
        )
        system.apply_polished_starts(polished, seed_indices)
        rounds.append(dict(zip(seed_indices, (float(d) for d in dlps))))
        if origins_phys is not None:
            _warn_if_seeds_converged(system, model, lookup, origins_phys, r)

    stop = "round cap" if _wants_another() else "converged"
    summary["stop"] = stop
    summary["wall_s"] = time.monotonic() - t0
    per_seed = "; ".join(
        f"seed {k}: +{sum(rd[k] for rd in rounds):.1f} ("
        + " + ".join(f"{rd[k]:.1f}" for rd in rounds)
        + ")"
        for k in seed_indices
    )
    logger.info(
        f"Seed polish: {len(rounds)} DE round(s) in "
        f"{summary['wall_s']:.0f} s; gain per seed (per round): {per_seed}.  "
        + (
            f"Stopped: the last round gained < {tol:.3g} nats on every seed."
            if stop == "converged"
            else f"Stopped at the {int(max_rounds)}-round cap with a seed "
            f"still gaining >= {tol:.3g} nats per round -- the start may "
            f"still be below its basin optimum."
        )
    )
    return summary


def _resolve_polish_cores(cores, n_seeds):
    """Worker count for the DE polish.

    ``cores=None`` means AUTO -- the same grant a sampler takes when nothing
    names one (_common.default_cores) -- and NOT serial.  It used to mean
    serial, which is how review 6.11.3 happened: outputs/ledger.py's
    hot-mode polish passed nothing, so the one branch every gradient-free
    microlensing fit takes ran on 1 core of 36 for 38 minutes while the rest
    of the machine the sampler had just filled sat idle.  Nothing about this
    stage wants fewer cores than the sampling either side of it, so the
    default now says so, and a caller that wants serial passes ``cores=1``.

    Capped at the batch size (n_seeds * pop_size) upstream by there simply
    being nothing more to hand out; here we only guard against asking for
    more processes than the machine has, and against the degenerate 1.
    """
    from .samplers._common import default_cores

    if cores is None:
        return default_cores()
    try:
        n = int(cores)
    except (TypeError, ValueError):
        # Unreadable value: say so and take the auto grant.  Silently
        # dropping to one core is what made the original bug invisible.
        #
        # The message says the same two things run.resolve_cores_setting's
        # refusal says -- an absent/None cores IS the automatic grant, and
        # cores=1 is how to ask for serial -- because the two used to
        # disagree about a value neither could use (review 5.3.3e).  Only
        # the OUTCOME differs, and positionally: run.py can still refuse the
        # run, while this can be reached from a wrap-up stage that must not
        # kill a finished fit.
        logger.warning(
            f"Seed polish: cores={cores!r} is not a number of cores; using "
            f"the automatic grant instead (which is also what an absent "
            f"cores takes; cores=1 is serial)."
        )
        return default_cores()
    # `cores <= 0` is the automatic grant, the same as None (review 2.4.8).
    # It used to be swept into the `n <= 1` serial arm, while the same 0 was
    # serial in create_pool and AUTO in nested.py.  run.py warns about it at
    # the parse boundary, where the user's own spelling is still in hand; by
    # the time it reaches this stage the only thing left to do is agree with
    # the other two resolvers.  ONE is still serial -- that is the statement
    # a caller makes when they mean it.
    if n <= 0:
        return default_cores()
    if n == 1:
        return 1
    return max(1, min(n, mp.cpu_count()))
