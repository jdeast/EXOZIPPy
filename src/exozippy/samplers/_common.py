"""
Shared, zero-statistical-risk scaffolding for the PTDE samplers.

exozippy.samplers.ptde (synchronous) and exozippy.samplers.ptde_async
(asynchronous) are two dispatch loops over the same non-sampling machinery:
worker-pool plumbing, compiled raw -> physical conversion, start-population
generation, signal handling, posterior assembly, and diagnostics. That
machinery lives here exactly once so the two samplers cannot drift apart
again (they did -- see notes/code_review_20260808.txt section 4: the sync
sampler grew the lp-plausibility guard while async had neither the parameter
nor the check, and log_interval silently meant different things in the two).

The sampling LOOPS stay separate by design: this module must never grow
accept/reject, temperature-swap, or adaptation logic. Anything statistical
belongs in the sampler that owns it.

Helpers that log take the calling sampler's ``log`` (a logging.Logger) so
messages appear under "exozippy.samplers.ptde" / "...ptde_async" -- the
logger names tests and users filter on.
"""

import gc
import logging
import multiprocessing as mp
import os
import signal
import threading
import time

import arviz as az
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import vectorize_graph

# Force single-threaded BLAS/OMP in every forked worker.  Without this,
# numpy (OpenBLAS/MKL) and C extensions (VBBinaryLensing) each spawn their
# own thread pool, producing n_workers x n_blas_threads threads on a fixed
# number of physical cores and causing catastrophic scheduler thrash -- and,
# separately, each with a memory arena sized to the (over-subscribed) thread
# count, which is what actually blows up h_vmem once forked N-ways.
# exozippy/__init__.py already sets these before numpy/pytensor/pymc/arviz/
# jax are imported at all -- the only point where it's effective, since a
# native thread pool can't be shrunk after the fact by setting os.environ
# once numpy et al. are already loaded (import exozippy always runs
# __init__.py first, before this module). This block is redundant there;
# kept as a guard for any environment that imports the samplers without
# ever importing the exozippy package proper.
for _tvar in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_tvar, "1")

logger = logging.getLogger(__name__)

# Shared with outputs.modes.identify_modes: |lp| above this is numerically
# broken, not a real posterior mode (no realistic dataset's logp reaches
# 1e12). Imported (not duplicated) so the two ceilings can't drift apart.
from exozippy.outputs.modes import DEFAULT_LP_ABS_MAX  # noqa: E402

# The chain-initialization probe lives in exozippy.whitening (it is also the
# engine behind the data-driven whitening rescale done at model setup); PTDE
# re-probes the already-whitened model, where every scale is ~1, so the
# default (narrow) dynamic range is the right one here.
from exozippy.whitening import probe_scales as _probe_scales  # noqa: E402

# ---------------------------------------------------------------------------
# Worker-side logp evaluation (fork-inherited globals)
# ---------------------------------------------------------------------------

# Module-level logp function: set in the parent process (set_worker_globals)
# before the Pool is forked.  Fork children inherit the compiled PyTensor
# function via copy-on-write without pickling.  Proposals (dicts of numpy
# arrays) are the only IPC payload.
_PTDE_LOGP_FN = None

# Diagnostic flag (see collect_rung_timing in ptde_sample, which carries
# the rationale): set in the parent before forking, like
# _PTDE_LOGP_FN, so workers inherit it via copy-on-write. When True,
# _eval_logp times its own call and returns (lp, elapsed_seconds) instead of
# a bare float, so the parent can attribute wall time to a rung.
_PTDE_COLLECT_TIMING = False


def set_worker_globals(logp_fn, collect_timing=False):
    """Install the compiled logp function (and the timing flag) in this
    module's namespace BEFORE the worker pool is forked, so children inherit
    both via copy-on-write. Both samplers must call this instead of writing
    the globals directly -- _eval_logp's __globals__ point here."""
    global _PTDE_LOGP_FN, _PTDE_COLLECT_TIMING
    _PTDE_LOGP_FN = logp_fn
    _PTDE_COLLECT_TIMING = collect_timing


class PositionalLogp:
    """A dict-in logp that calls the compiled function BY POSITION.

    ``model.compile_logp()`` returns a pymc ``PointFunc``, whose ``__call__``
    is ``self.f(**point)``: pytensor then looks every name up, and -- with
    ``trust_input`` off, its default -- runs each value through the input's
    ``filter`` (a dtype/shape check and possible copy).  That is a per-INPUT
    cost on a call whose useful work may be microseconds, and PTDE makes one
    such call per proposal, n_temps x n_chains of them per step.  Measured
    per call on a 27-element model, evaluated 20k times (review 6.4.3):

        raw vars   PointFunc(dict)   this wrapper    saved
             3          7.5 us          3.1 us       2.4x
            10         18.8 us          6.2 us       3.0x
            20         35.7 us         10.1 us       3.5x
            27         45.9 us         13.0 us       3.5x

    i.e. ~1.4 us per raw variable, against ~0.4 us here.  On the 20-variable,
    432-proposal step that a DC2018-class fit runs, that is ~11 ms of worker
    CPU per step.

    THE COERCION IS NOT OPTIONAL, and dropping it is how this turns into a
    crash or worse.  ``trust_input`` disables filtering entirely, and the
    values PTDE hands over are NOT always what the filter would have made
    them: ``de_proposal`` computes ``pop[i][k] + gamma*(...)``, and for a
    0-d parameter numpy returns a np.float64 SCALAR rather than a 0-d array.
    Passed straight through, that raises inside the numba backend
    (reproduced: "Vectorized inputs must be arrays") -- and an unfiltered
    wrong dtype is worse, since it is read as raw memory.  So each value goes
    through ``np.asarray(v, dtype)`` here, which is exactly the conversion
    the filter would have done and costs ~0.1 us.

    Falls back to the wrapped callable unchanged if the function cannot be
    introspected (a plain callable in a test, a future pymc that renames the
    attributes) -- this is an optimization, and losing it must never mean
    losing the fit.  The wrapped function is one the SAMPLER compiled, so
    setting trust_input on it cannot surprise another holder.
    """

    def __init__(self, logp_fn):
        self.logp_fn = logp_fn
        self.spec = None
        f = getattr(logp_fn, "f", None)
        try:
            inputs = [i for i in f.maker.inputs if not i.implicit]
            spec = [(i.variable.name, i.variable.type.dtype) for i in inputs]
            if any(name is None for name, _ in spec):
                raise AttributeError("unnamed compiled-logp input")
        except (AttributeError, TypeError):
            logger.debug(
                "PositionalLogp: cannot introspect %r; keeping the dict "
                "call path",
                type(logp_fn),
            )
            return
        f.trust_input = True
        self.f = f
        self.spec = spec

    def __call__(self, point):
        if self.spec is None:
            return self.logp_fn(point)
        return self.f(
            *[np.asarray(point[name], dtype=dt) for name, dt in self.spec]
        )


# Exception types _eval_logp has already reported, so a failing region does
# not emit one log line per proposal.  Per worker process, by construction.
_LOGP_EXC_SEEN = set()


def _report_logp_exception(exc, proposal):
    """Log the first occurrence of each logp exception type in this worker.

    The -inf below is kept -- a failed evaluation has to return SOMETHING and
    the samplers are built around it -- but -inf is not "absent": both
    Metropolis tests read it as ZERO POSTERIOR DENSITY, so a region that
    merely raises is excluded from the posterior exactly as if the model had
    ruled it out.  If it borders the mode, the posterior is truncated there
    and the failures are absorbed into the acceptance rate as ordinary
    rejections.  It used to be a bare `except Exception: return -np.inf`
    with no counter, no warning and no sample_stats entry -- and it also
    swallowed the exception before ptde_async's error_callback could fire,
    making that logger.error dead code.  Note the SAME logp called on the
    parent side during initialization is not wrapped at all, so an identical
    failure is fatal at startup and was invisible during sampling.
    """
    key = type(exc).__name__
    if key in _LOGP_EXC_SEEN:
        return
    _LOGP_EXC_SEEN.add(key)
    try:
        # RAW-space values: the worker has no raw_to_phys map (that lives on
        # the parent, which is what describe_proposal needs), and a
        # diagnostic that cannot be produced is worth less than one in the
        # wrong coordinates.
        where = ", ".join(
            f"{k}={np.asarray(v).ravel()[:4]}"
            for k, v in sorted(proposal.items())
        )
    except Exception:  # pragma: no cover - diagnostics must not raise
        where = "<unprintable proposal>"
    logger.error(
        f"logp evaluation raised {key}: {exc}.  The proposal is being "
        f"REJECTED (logp = -inf), i.e. treated as zero posterior density, "
        f"so this region is excluded from the posterior rather than "
        f"explored.  First occurrence in this worker; further {key} will "
        f"not be logged.  Proposal: {where}",
        exc_info=True,
    )


def _eval_logp(proposal):
    """Worker: evaluate logp for one raw-space proposal dict.

    Returns a bare float normally. When _PTDE_COLLECT_TIMING is set, returns
    (lp, elapsed_seconds) instead (diagnostic mode; see the comment above
    _PTDE_COLLECT_TIMING).
    """
    if _PTDE_COLLECT_TIMING:
        t0 = time.perf_counter()
        try:
            lp = float(_PTDE_LOGP_FN(proposal))
        except Exception as exc:
            _report_logp_exception(exc, proposal)
            lp = -np.inf
        return lp, time.perf_counter() - t0
    try:
        return float(_PTDE_LOGP_FN(proposal))
    except Exception as exc:
        _report_logp_exception(exc, proposal)
        return -np.inf


# ---------------------------------------------------------------------------
# Worker pool lifecycle
# ---------------------------------------------------------------------------


def _worker_init():
    """Pool worker: ignore SIGINT/SIGTERM so only the parent handles graceful
    stop. A batch scheduler typically signals the whole process group, and a
    worker that died mid pool.map() would break the parent's current step
    (BrokenProcessPool) instead of letting it finish and wrap up cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _shutdown_pool(pool, grace=1.0):
    """terminate() a Pool, escalating to SIGKILL for workers that outlive the
    grace period.

    _worker_init has the workers ignore SIGTERM (so a batch scheduler signalling
    the whole process group cannot break the parent mid-step). But pool.terminate()
    kills workers *by sending SIGTERM*, then joins them -- so a worker stuck in a
    pathological logp evaluation ignores the terminate and the join blocks forever.
    That is the "recycling worker pool" hang: a timed-out worker is exactly the
    one that can never be reaped this way.

    The SIGKILL is issued from a watchdog thread rather than up front, because it
    must land only AFTER terminate() is past its _help_stuff_finish step. That
    step drains inqueue while holding inqueue._rlock; a worker blocked in
    inqueue.get() holds that same lock, so SIGKILLing it before terminate()
    reaches _help_stuff_finish wedges the lock and deadlocks terminate() even
    earlier. Waiting `grace` seconds lets a well-behaved terminate() finish
    untouched (the escalation never fires), and only forces the issue when a
    worker is genuinely wedged.
    """
    done = threading.Event()

    def _reaper():
        if done.wait(grace):
            return  # terminate()/join() completed cleanly; no escalation needed
        for p in pool._pool:
            if p.exitcode is None:
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

    watchdog = threading.Thread(target=_reaper, daemon=True)
    watchdog.start()
    try:
        pool.terminate()
        pool.join()
    finally:
        done.set()
        watchdog.join()


def create_pool(cores, total_proposals, label, log):
    """Resolve the core count and fork the worker pool.

    Must be called AFTER set_worker_globals so fork children inherit the
    compiled logp function. Returns (pool_or_None, actual_cores); pool is
    None in serial mode (actual_cores <= 1).
    """
    phys_cores = mp.cpu_count()
    if cores is None:
        # Fallback if called directly (not via run.py): same 75% formula.
        cores = max(1, min(int(phys_cores * 0.75), phys_cores - 1))
    actual_cores = min(cores, total_proposals)
    if cores > phys_cores:
        log.warning(
            f"{label}: cores={cores} exceeds physical core count ({phys_cores}); "
            f"over-subscription will slow sampling via context switching."
        )
    pool = (
        mp.get_context("fork").Pool(actual_cores, initializer=_worker_init)
        if actual_cores > 1
        else None
    )
    return pool, actual_cores


def recycle_pool(pool, actual_cores):
    """Tear down a pool with a possibly-hung worker and fork a fresh one.

    Terminated Pools sit in reference cycles (handler threads, worker
    sentinels, queue pipes) that only the cyclic GC frees; without an
    explicit collect each recycle leaks ~2 fds per worker until the process
    hits EMFILE ("Too many open files") after enough timeouts.
    """
    _shutdown_pool(pool)
    gc.collect()
    return mp.get_context("fork").Pool(actual_cores, initializer=_worker_init)


def warn_serial_eval_timeout(eval_timeout, pool, actual_cores, label, log):
    """One-time startup warning: eval_timeout is unenforceable without a pool."""
    if eval_timeout is not None and pool is None:
        log.warning(
            f"{label}: eval_timeout={eval_timeout:.0f}s has no effect with a "
            f"single core (cores={actual_cores}) -- there is no worker process "
            f"to enforce a wall-clock timeout against a hung logp call."
        )


def _map_logp(pool, proposals):
    if pool is None:
        return [_eval_logp(p) for p in proposals]
    return pool.map(_eval_logp, proposals)


def _map_logp_timeout(pool, proposals, timeout):
    """Evaluate logps with a per-call wall-clock timeout.

    Each proposal individually gets up to `timeout` seconds (not a deadline
    shared across the whole batch -- a slow-but-legitimate early proposal must
    not eat into the budget of proposals evaluated later in the same step).
    A proposal that doesn't complete in time receives -inf, so the caller's
    normal Metropolis accept/reject logic rejects it automatically.

    A logp evaluation can call into external/compiled code that occasionally
    enters a genuine infinite loop for some pathological parameter
    combination. When that happens the worker process that drew the
    timed-out proposal is stuck forever and never becomes available again;
    this function has no way to kill a single worker without tearing down
    the whole Pool, so the caller is responsible for recycling `pool`
    whenever `timed_out` is non-empty. Without that, a long run slowly
    bleeds workers, one per hang, until the pool is exhausted.

    With no pool (single-core / serial mode), there is no subprocess to time
    out, so `timeout` cannot be enforced -- proposals run to completion as
    before. The caller should warn about this once at startup if cores<=1
    (warn_serial_eval_timeout).

    Returns (lps, timed_out) where timed_out is a list of indices into
    `proposals`.
    """
    if pool is None:
        return [_eval_logp(p) for p in proposals], []

    async_results = [pool.apply_async(_eval_logp, (p,)) for p in proposals]
    lps = []
    timed_out = []
    # Keep the timeout sentinel the same shape _eval_logp returns (a bare
    # float, or (lp, elapsed) in collect_rung_timing diagnostic mode) so
    # every entry in `lps` is uniformly typed for the caller to unpack.
    timeout_val = (-np.inf, timeout) if _PTDE_COLLECT_TIMING else -np.inf
    for idx, r in enumerate(async_results):
        try:
            lps.append(r.get(timeout=timeout))
        except mp.TimeoutError:
            lps.append(timeout_val)
            timed_out.append(idx)
    return lps, timed_out


# ---------------------------------------------------------------------------
# DE proposal construction
# ---------------------------------------------------------------------------

# ter Braak 2006 epsilon term: without it a DE proposal is a pure linear
# combination of population states, so sampling is confined to the affine
# hull of the initial T=1 starts FOREVER (swaps only exchange states within
# the same hull). With the default n_chains = 2*n_params the hull is almost
# surely full-rank and this never bites, but a user setting
# n_chains <= n_params would otherwise silently explore a proper subspace
# and get a wrong posterior. Raw (whitened) space has scales ~1 by
# construction, so a fixed 1e-4 is small against any posterior width --
# a correctness backstop, not a mixing mechanism (see
# warn_if_population_degenerate for the mixing side).
DE_JITTER = 1e-4

# Smallest DE population this code will run with.
#
# A DE proposal for member i is x_i + gamma*(x_j1 - x_j2) with j1 != j2 != i,
# so it needs two OTHER members: n = 2 has only one other and _pick_two used
# to die inside numpy with "Cannot take a larger sample than population when
# replace is False" (notes/code_review_20260808.txt 2.9.4).  n = 2 is not a
# perverse setting -- it is what the DEFAULT n_chains = 2 * n_params produces
# for a one-parameter model.  n = 3 is the smallest population that can move
# at all, but the two others are then FORCED, so every proposal for member i
# lies along the single direction +/-(x_j1 - x_j2); 4 is the smallest that
# offers any choice of difference vector, and is what the default is floored
# at.  This is a hard floor on the MOVE being defined, not a mixing
# recommendation -- warn_if_population_degenerate below owns that, and it
# asks for a great deal more (n_params + 2).
MIN_DE_CHAINS = 4


def _pick_two(rng, n, exclude):
    """Pick two distinct indices from [0, n) excluding `exclude`."""
    if n < 3:
        raise ValueError(
            f"A DE proposal needs two other population members, but n={n} "
            f"leaves only {max(n - 1, 0)}. There is no meaningful "
            f"difference vector to propose with -- raise n_chains to at "
            f"least {MIN_DE_CHAINS} (the default is 2 x n_params, floored "
            f"at {MIN_DE_CHAINS})."
        )
    idx = rng.choice(n - 1, 2, replace=False)
    return tuple(int(i + (1 if i >= exclude else 0)) for i in idx)


def resolve_n_chains(n_chains, n_params, label, log):
    """Resolve and validate the DE population size for one rung.

    ``None`` takes the default 2 * n_params, floored at MIN_DE_CHAINS so the
    default can never land on a population too small to form a difference
    vector (n_params = 1 -> 2; see MIN_DE_CHAINS).  An explicit value below 3
    RAISES rather than being silently bumped: the user asked for something
    the DE move cannot do, and quietly running a different sampler than the
    one requested is worse than stopping.  Also emits the
    warn_if_population_degenerate mixing warning.
    """
    if n_chains is None:
        n_chains = max(2 * n_params, MIN_DE_CHAINS)
        if n_chains > 2 * n_params:
            log.info(
                f"{label}: default n_chains = 2 x n_params = "
                f"{2 * n_params} is below the DE minimum; using "
                f"{n_chains} chains/rung."
            )
    else:
        n_chains = int(n_chains)
        if n_chains < 3:
            raise ValueError(
                f"{label}: n_chains={n_chains} is too small for a DE "
                f"proposal, which needs two OTHER population members "
                f"(x_i + gamma*(x_j1 - x_j2), j1 != j2 != i). Use "
                f"n_chains >= {MIN_DE_CHAINS}, or omit n_chains for the "
                f"default 2 x n_params."
            )
    warn_if_population_degenerate(n_chains, n_params, label, log)
    return n_chains


# ---------------------------------------------------------------------------
# Per-rung explored span
# ---------------------------------------------------------------------------


class SpanTracker:
    """Running min/max of every raw coordinate, per temperature rung.

    The cheapest honest answer to "how far can this ladder actually reach?".
    Raw space is whitened, so one unit is one measured posterior sigma and
    the span is directly comparable to the distance to a candidate
    alternative solution.

    Why this rather than an inline mode search: on DC2018 event 128 the known
    second basin (s = 0.854 against the posterior's 0.9755) sits 296 sigma
    away in log_s.  A rung at temperature T explores a width ~sqrt(T) sigma,
    so T_max=200 spans ~14 sigma -- 21 widths away, unreachable at any
    runtime -- while T_max=8500 spans ~92 sigma, 3.2 widths, rare per step
    but reachable across 54 chains.  Reporting the span per rung turns that
    from an after-the-fact calculation into something visible WHILE the run
    is going: if the top rung's span is not of order the distance to whatever
    you hoped to find, more runtime will not find it and T_max is too low.

    Updated only where a slot's state changes (first evaluation and accepted
    moves), so the cost is two numpy ops on one parameter vector per accepted
    proposal -- negligible against a logp evaluation.
    """

    def __init__(self, n_temps, raw_start, layout=None):
        # PACKED, matching what the samplers actually carry: since review
        # 6.4.2 a proposal is a (total,) float64 vector from RawLayout, not a
        # dict of per-parameter arrays.  This class was written against the
        # dict form and called state.items() on it, which merged cleanly with
        # the packing (different regions of the file) and then died at run
        # time with "'numpy.ndarray' object has no attribute 'items'" across
        # every ptde_async test.  Packing it here is also simply less work:
        # two whole-vector minimum/maximum calls per update instead of a
        # Python loop over the keys.
        self.layout = layout if layout is not None else RawLayout(raw_start)
        self.lo = np.full((n_temps, self.layout.total), np.inf)
        self.hi = np.full((n_temps, self.layout.total), -np.inf)
        self.n_temps = n_temps

    def update(self, k, vec):
        """Fold one packed state into rung ``k``'s running extremes."""
        np.minimum(self.lo[k], vec, out=self.lo[k])
        np.maximum(self.hi[k], vec, out=self.hi[k])

    def report(self):
        """[(widest span in sigma, which parameter)] per rung.

        A rung no slot has reported yet gives (0.0, "-") rather than nan or
        inf: an unvisited rung has no span, and a diagnostic that prints inf
        reads as a bug rather than as "no data yet".
        """
        out = []
        for k in range(self.n_temps):
            span = self.hi[k] - self.lo[k]
            finite = np.isfinite(span)
            if not finite.any():
                out.append((0.0, "-"))
                continue
            masked = np.where(finite, span, -np.inf)
            j = int(np.argmax(masked))
            best = float(masked[j])
            if best <= 0.0:
                out.append((0.0, "-"))
                continue
            # Map the winning element back to the parameter that owns it.
            best_key = "-"
            for key, (a, b, _) in zip(self.layout.keys, self.layout.slices):
                if a <= j < b:
                    best_key = key
                    break
            out.append((best, best_key))
        return out


# ---------------------------------------------------------------------------
# Hot-rung retention (store_hot_chains)
# ---------------------------------------------------------------------------

# Thinning applied when hot-rung retention is on but no factor was given.
DEFAULT_HOT_THIN = 20


def hot_chain_trace_share(n_temps, hot_thin, n_raw_elements, n_out_elements):
    """Fraction of the saved trace the ``posterior_hot`` group will occupy.

    Both groups scale with n_chains x draws, so those cancel exactly and the
    share is fixed by three things: the ladder height, the thinning, and --
    the term nobody expects -- the DERIVED-variable count.  The cold group
    stores PHYSICAL variables (sampled plus every pm.Deterministic), the hot
    group stores RAW ones only, so a model with many derived quantities
    dilutes the hot share even at the same n_temps.

    examples/ob140939: n_temps=24, hot_thin=20, 19 raw and 39 output
    elements -> 23 x 20/20 = 23 against 40, i.e. 36.5% of the file.
    """
    if n_temps < 2 or hot_thin <= 0:
        return 0.0
    hot = (n_temps - 1) * (n_raw_elements + 1) / float(hot_thin)
    cold = float(n_out_elements + 1)
    return hot / (hot + cold)


def resolve_store_hot_chains(
    spec, system, n_temps, n_raw_elements, n_out_elements, label, log
):
    """Resolve ``store_hot_chains`` to a thinning factor (0 = off).

    Vocabulary, matching the rest of the sampler block ('auto' as in
    `seed_polish`, `n_temps` and `mmexofast`): ``True``/'on' -> the default
    thinning, ``False``/None/'off' -> off, an int -> that thinning factor,
    and 'auto' (the default) -> decided by TOPOLOGY.

    'auto' turns retention on when any active component declares
    ``expects_suppressed_modes`` -- today the microlensing lens, whose
    degeneracies are structural rather than accidental.  Hot-rung draws are
    the only detector for a basin the T=1 posterior abandons
    (outputs.ledger.discover_hot_modes), and they cost real trace size, so
    the default follows the topology the way `chen` and
    `mass_parameterization` do on the planet component: resolved from the
    built system, overridable per fit, and never silent -- the decision and
    its price are logged where it is made.

    ``isinstance(spec, bool)`` comes first deliberately: ``spec is True``
    already guarded the ``store_hot_chains: 1`` case here, but the same
    ``1 == True`` collision cost `seed_polish` a whole value
    (notes/code_review_20260808.txt 2.9.1), so the guard is now explicit
    rather than incidental.

    An int <= 0 is OFF, the same as ``false``, and not the ``max(1, ...)``
    it used to take -- which resolved ``store_hot_chains: 0`` to thin=1,
    i.e. MAXIMUM retention (every hot iteration of every hot rung), the
    exact opposite of what the summary line above promises and of what
    anyone writing a zero means (review 1.4.2).  A negative factor has no
    other reading either, so it takes the same branch rather than raising:
    the vocabulary already spells off three ways and a fourth spelling of
    it costs nothing, while a raise would fail a fit at the sampler for a
    key that changes only what is STORED.
    """
    auto = False
    if isinstance(spec, bool):
        hot_thin = DEFAULT_HOT_THIN if spec else 0
    elif spec is None:
        hot_thin = 0
    elif isinstance(spec, str):
        key = spec.strip().lower()
        if key == "auto":
            auto = True
            hot_thin = 0
        elif key == "on":
            hot_thin = DEFAULT_HOT_THIN
        elif key == "off":
            hot_thin = 0
        else:
            raise ValueError(
                f"{label}: store_hot_chains={spec!r} is not recognized; use "
                f"auto (default), true/on, false/off, or an integer "
                f"thinning factor."
            )
    else:
        hot_thin = max(0, int(spec))

    named = sorted(
        name
        for name, comp in getattr(system, "active_components", {}).items()
        if getattr(comp, "expects_suppressed_modes", False)
    )
    if auto:
        hot_thin = DEFAULT_HOT_THIN if named else 0

    if n_temps < 2:
        if hot_thin:
            log.info(
                f"{label}: store_hot_chains is set but n_temps={n_temps} "
                f"leaves no hot rungs; nothing will be stored."
            )
        return 0

    share = hot_chain_trace_share(
        n_temps, hot_thin, n_raw_elements, n_out_elements
    )
    if auto and hot_thin:
        log.info(
            f"{label}: store_hot_chains: auto -> ON (thin {hot_thin}). "
            f"Component(s) {named} expect posterior-suppressed modes, so "
            f"thinned draws from the {n_temps - 1} hot rungs are kept to "
            f"search for solutions the T=1 posterior abandons "
            f"(outputs.ledger.discover_hot_modes). Cost: roughly "
            f"{100 * share:.0f}% of the trace file. Set "
            f"`store_hot_chains: false` to opt out."
        )
    elif auto:
        log.info(
            f"{label}: store_hot_chains: auto -> OFF. No active component "
            f"expects posterior-suppressed modes, so hot-rung draws are not "
            f"kept and NO search for suppressed modes will run. Set "
            f"`store_hot_chains: true` to enable it (adds roughly "
            f"{100 * hot_chain_trace_share(n_temps, DEFAULT_HOT_THIN, n_raw_elements, n_out_elements):.0f}% "
            f"to the trace file)."
        )
    elif hot_thin:
        log.info(
            f"{label}: store_hot_chains set explicitly -> ON (thin "
            f"{hot_thin}), roughly {100 * share:.0f}% of the trace file."
        )
    else:
        log.info(
            f"{label}: store_hot_chains set explicitly -> OFF; no search "
            f"for posterior-suppressed modes will run."
        )
    return hot_thin


def de_proposal(rng, pop, i, gamma, keys, jitter=DE_JITTER):
    """One ter Braak DE-MC proposal for population member `i`:
    x_i + gamma * (x_j1 - x_j2) + jitter * N(0, 1), j1 != j2 != i.

    THE REFERENCE IMPLEMENTATION, and no longer the production path: both
    samplers now hold their populations packed and call RawLayout.propose,
    which is the same move on a flat vector and 5x cheaper (review 6.4.2).
    This one is kept -- deliberately, do not delete it as unused -- because
    it is what the packed move is PINNED AGAINST: a test builds the same
    proposal both ways from the same seed and demands they agree bit for
    bit.  Written in the obvious dict-at-a-time way, it is the readable
    statement of the move, and the thing to change first if the move ever
    changes.
    """
    j1, j2 = _pick_two(rng, len(pop), i)
    prop = {}
    for key in keys:
        step = gamma * (pop[j1][key] - pop[j2][key])
        if jitter:
            step = step + jitter * rng.standard_normal(np.shape(pop[i][key]))
        prop[key] = pop[i][key] + step
    return prop


# Largest factor by which one adaptation window may move gamma.  A window is
# a few hundred proposals, so its acceptance estimate is noisy; the clip is
# what keeps one unlucky window from moving the step size by orders of
# magnitude.
GAMMA_CLIP_FACTOR = 10.0


def next_gamma(gamma, ar, target_accept, clip=GAMMA_CLIP_FACTOR):
    """The DE step-size update, shared by everything that adapts gamma.

    ``gamma *= (ar / target)**0.5``, clipped to a factor `clip` per update.
    The square root dampens the oscillation a proportional rule would set
    up: acceptance responds to gamma with a lag of one whole window, so a
    full correction consistently overshoots.

    ``ar <= 0`` -- nothing accepted at all -- has no signal for the ratio to
    use, and shrinks by the clip factor instead of stalling at a step size
    already proven too large.  ptde/ptde_async guard on `ar > 0` before
    calling and so never take that branch; polish_seed_starts relies on it,
    since a T=1 optimizer routinely sustains sub-1% acceptance.

    THE single owner of the rule, for the same reason de_proposal is the
    single owner of the move: the sampler's T=1 kernel and the polish's
    engine are the same move, so a second tuning story would be one more
    thing to keep in sync.
    """
    shrink = 1.0 / clip  # NOT gamma/clip below: the callers this replaced
    # multiplied by 0.1, and 1.0/10.0 IS 0.1 in float64 while gamma/10.0
    # need not equal gamma*0.1 in the last bit.
    scale = (ar / target_accept) ** 0.5 if ar > 0 else shrink
    return float(np.clip(gamma * scale, gamma * shrink, gamma * clip))


# Ensemble spread, in probe-scale units, below which the start population is
# reported as under-dispersed.  1.0 is not a tuning choice: the probe scale IS
# an estimate of the posterior width (the 0.5-nat step), so a spread below it
# means the chains start INSIDE the posterior, which is the one assumption
# Rhat cannot check for itself.
UNDERDISPERSED_SPREAD = 1.0


def start_spread_ratios(starts, scales):
    """Between-chain spread of a start population, in units of `scales`.

    Returns {key: ndarray} of ``std(starts, over chains) / scale``, i.e. one
    ratio per raw ELEMENT.  Elements whose scale is not finite and positive
    are dropped -- a flat probe direction (see _PROBE_FLAT_SCALE) has no
    posterior width to be dispersed against.
    """
    if len(starts) < 2:
        return {}
    ratios = {}
    for key in starts[0]:
        stacked = np.stack(
            [np.ravel(np.asarray(s[key], dtype=float)) for s in starts]
        )
        spread = stacked.std(axis=0, ddof=1)
        scale = np.ravel(
            np.broadcast_to(
                np.asarray(scales[key], dtype=float), stacked.shape[1:]
            )
        )
        ok = np.isfinite(scale) & (scale > 0)
        if not np.any(ok):
            continue
        ratios[key] = spread[ok] / scale[ok]
    return ratios


def warn_if_starts_underdispersed(starts, scales, label, log):
    """Warn when the chains start narrower than the posterior they sample.

    Rhat compares between-chain to within-chain variance, so it can only
    diagnose non-convergence if the chains START further apart than the
    posterior is wide.  Starts drawn from INSIDE the target -- a restart
    seeded from a previous run's posterior draws, or a hand-written
    initvals list -- make the between-chain term small from the first
    draw, and Rhat then reads ~1.00 while the chains have explored
    nothing.  convergence.good_chain_mask and converged_on_tail both
    inherit that assumption, and the early-stop check acts on it, so an
    under-dispersed restart can stop a fit that never mixed (review 2.4.5).

    Reported, never corrected: widening someone's deliberately tight start
    would move the fit they asked for, and a legitimate high-dimensional
    run is under-dispersed BY CONSTRUCTION -- _make_starts scatters at
    min(sqrt(500/D), 3) scale units, which drops below 1 past D = 500.
    The number is what matters, so the message quotes it.

    Returns the median ratio (NaN when it could not be measured).
    """
    if len(starts) < 2:
        return float("nan")
    ratios = start_spread_ratios(starts, scales)
    if not ratios:
        return float("nan")
    flat = np.concatenate([np.ravel(v) for v in ratios.values()])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan")
    median = float(np.median(flat))
    if median < UNDERDISPERSED_SPREAD:
        worst = min(ratios, key=lambda k: float(np.min(np.ravel(ratios[k]))))
        log.warning(
            f"{label}: the {len(starts)} chain starts are UNDER-DISPERSED -- "
            f"their between-chain spread is {median:.2f}x the measured "
            f"posterior scale (tightest: {worst} at "
            f"{float(np.min(np.ravel(ratios[worst]))):.2f}x). Rhat assumes "
            f"chains start FURTHER apart than the posterior is wide; below "
            f"that it reads ~1.00 whether or not the chains have mixed, and "
            f"the min_ess/max_rhat early stop acts on it. Seeding from a "
            f"previous run's posterior draws does exactly this. Widen the "
            f"seed spread, or judge convergence on a longer run."
        )
    return median


class RawLayout:
    """One flat float64 vector per state, instead of a dict of small arrays.

    A raw-space state is a dict of ~20 little arrays, one per free RV, and
    that shape is what made the DE move expensive: every proposal ran a
    Python loop over the keys doing three array operations and a dict
    insert, plus its own ``standard_normal`` draw per key.  Measured on a
    DC2018-shaped model (27 elements over 20 raw variables, 20k proposals):

        _pick_two (rng.choice)                8.8 us
        20 per-key standard_normal draws     18.3 us   -> 0.9 us as one draw
        20 per-key arithmetic + dict         29.2 us   -> 2.1 us packed
        loop/np.shape/branch overhead       ~28 us     -> gone
        ------------------------------------------------------------------
        de_proposal total                    85.2 us   -> ~20 us

    which on an 8 x 54 ladder is 36.8 ms of SERIAL parent time per step
    against ~8 ms -- and the parent is the bottleneck the async sampler
    exists to keep fed (review 6.4.2).

    BIT-IDENTICAL, not merely equivalent in distribution, and both halves of
    that are load-bearing:

    * The partner draw stays per member, exactly the ``_pick_two`` the dict
      path used.  A batched partner draw would consume the bit stream in a
      different order and change every subsequent number.
    * ONE ``standard_normal(total)`` per proposal is the same SEQUENCE as
      one draw per key in key order -- numpy's generator fills sequentially
      with no per-call buffering, verified over mixed shapes including 0-d
      and 2-d, and verified again after an intervening ``rng.choice``.
    * The arithmetic is elementwise, so concatenating the operands changes
      no float operation and no rounding.

    ``unpack`` returns views into ONE freshly copied buffer, so the dict a
    worker is handed cannot be aliased by a later proposal, and shapes come
    back exactly as the model declared them (a 0-d parameter comes back as
    a 0-d ARRAY, where the dict path produced a numpy scalar -- strictly
    closer to what the compiled logp wants; see PositionalLogp).
    """

    def __init__(self, raw_start, keys=None):
        self.keys = list(raw_start.keys() if keys is None else keys)
        self.shapes = [np.shape(raw_start[k]) for k in self.keys]
        self.sizes = [int(np.prod(s)) if s else 1 for s in self.shapes]
        offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        self.total = int(offsets[-1])
        self.slices = [
            (int(a), int(b), sh)
            for a, b, sh in zip(offsets[:-1], offsets[1:], self.shapes)
        ]

    def pack(self, state):
        """dict -> (total,) float64 vector."""
        out = np.empty(self.total, dtype=float)
        for key, (a, b, _) in zip(self.keys, self.slices):
            out[a:b] = np.ravel(np.asarray(state[key], dtype=float))
        return out

    def pack_many(self, states):
        """list of dicts -> (n, total) float64 array."""
        out = np.empty((len(states), self.total), dtype=float)
        for row, state in zip(out, states):
            for key, (a, b, _) in zip(self.keys, self.slices):
                row[a:b] = np.ravel(np.asarray(state[key], dtype=float))
        return out

    def unpack(self, vec):
        """(total,) vector -> dict of arrays in the model's own shapes."""
        buf = np.array(vec, dtype=float, copy=True)
        return {
            key: buf[a:b].reshape(sh)
            for key, (a, b, sh) in zip(self.keys, self.slices)
        }

    def store_draw(self, stored_raw, vec, *index):
        """Write one packed state into the per-variable draw buffers.

        ``index`` is whatever leads the buffer's draw axis -- (chain, draw)
        for the cold group, (rung, chain, draw) for the hot one.
        """
        for key, (a, b, sh) in zip(self.keys, self.slices):
            stored_raw[key][index] = vec[a:b].reshape(sh)

    def propose(self, rng, pop, i, gamma, jitter=DE_JITTER):
        """The ter Braak DE-MC move of ``de_proposal``, on packed rows.

        ``pop`` is an (n_chains, total) array; returns a new (total,)
        vector.  Draws exactly what de_proposal draws, in the same order.
        """
        j1, j2 = _pick_two(rng, len(pop), i)
        step = gamma * (pop[j1] - pop[j2])
        if jitter:
            step = step + jitter * rng.standard_normal(self.total)
        return pop[i] + step


def warn_if_population_degenerate(n_chains, n_params, label, log):
    """Warn when the DE population cannot span parameter space.

    With n_chains < n_params + 2 the difference vectors span a proper
    subspace, and the epsilon jitter (DE_JITTER) is the only escape from
    it -- ergodic in principle, hopeless in practice. The default
    n_chains = 2 * n_params never triggers this.
    """
    if n_chains < n_params + 2:
        log.warning(
            f"{label}: n_chains={n_chains} < n_params + 2 = {n_params + 2}; "
            f"DE difference vectors cannot span parameter space and mixing "
            f"across the missing directions relies on the tiny epsilon "
            f"jitter alone. Raise n_chains (default 2 x n_params)."
        )


# ---------------------------------------------------------------------------
# Compiled conversions (logp is compiled by the caller; see set_worker_globals)
# ---------------------------------------------------------------------------


def compile_conversions(model):
    """Compile the raw -> physical conversion functions ONCE.

    raw_to_phys is the single-sample form, kept for the (rare) eval_timeout
    diagnostic log path, which converts exactly one proposal at a time.
    raw_to_phys_batched vectorizes the same graph over an extra leading
    sample axis (pytensor's vectorize_graph -- adds the batch dim to every
    op in the graph rather than looping in Python) and is what the
    ensemble-start-plot and final posterior conversions use, since those
    can be tens of thousands to millions of samples: the free_RVs +
    deterministics graph is pure elementwise/indexing math (each
    Parameter's physical-unit conversion; verified empirically that no
    deterministic here touches the magnification Ops, which only feed the
    likelihood), so it vectorizes cleanly and cuts what was a
    Python-level per-sample loop (dominant cost: interpreter + pytensor
    call overhead, not the underlying math) down to a handful of batched
    calls. (Measured under notes/hpc_optimization.txt's PROMPT 7, which
    has since been pruned from that note.)

    Returns (raw_to_phys, raw_to_phys_batched, raw_var_names, out_var_names).
    """
    output_vars = model.free_RVs + model.deterministics
    raw_to_phys = pytensor.function(
        inputs=model.free_RVs,
        outputs=output_vars,
        on_unused_input="ignore",
    )
    batched_inputs = [
        pt.tensor(
            name=f"batched_{v.name}",
            dtype=v.type.dtype,
            shape=(None,) + v.type.shape,
        )
        for v in model.free_RVs
    ]
    raw_to_phys_batched = pytensor.function(
        inputs=batched_inputs,
        outputs=vectorize_graph(
            output_vars, replace=dict(zip(model.free_RVs, batched_inputs))
        ),
        on_unused_input="ignore",
    )
    raw_var_names = [v.name for v in model.free_RVs]  # ordered input names
    out_var_names = [v.name for v in output_vars]  # ordered output names
    return raw_to_phys, raw_to_phys_batched, raw_var_names, out_var_names


def describe_proposal(prop, raw_to_phys, raw_var_names, out_var_names):
    """Render one raw proposal as (physical_params, raw_params) dicts of
    plain lists, for the eval-timeout diagnostic log."""
    raw_vals = [prop[k] for k in raw_var_names]
    phys_vals = raw_to_phys(*raw_vals)
    phys_params = {
        name: np.asarray(val).tolist()
        for name, val in zip(out_var_names, phys_vals)
    }
    raw_params = {k: np.asarray(v).tolist() for k, v in prop.items()}
    return phys_params, raw_params


# ---------------------------------------------------------------------------
# Start-population generation + ensemble plots
# ---------------------------------------------------------------------------


def _make_starts(
    n_chains,
    raw_starts,
    logp_fn,
    rng,
    seed_indices=None,
    system=None,
    raw_scales=None,
):
    """Generate n_chains starting points near one or more seeds (P4).

    `raw_starts` is a single raw-start dict (legacy) or a LIST of K raw-start
    dicts (multi-seed sampling). Chains are assigned to seeds round-robin
    (chain j -> seed j % K); the first chain of each seed group starts exactly
    at that seed's solved point, the rest jitter around their seed's center.

    Mirrors EXOFASTv2: scatter chains by factor x scale where
    factor = min(sqrt(500/n_params), 3), accept any finite logp (no proximity
    threshold), and apply exponential decay only when proposals hit hard prior
    boundaries (lp=-inf).  Raises RuntimeError if a chain cannot be initialized
    within max_iter retries.

    The scatter is delegated to `system.jitter_raw_start` when the system
    provides it, which draws in physical space from a Gaussian truncated to
    each parameter's bounds.  Scattering in raw space instead saturates the
    logit transform and starts a large fraction of chains pinned at the bounds
    (31.5% within 1% of a bound for a parameter whose logp is flat out to them,
    against uniform's 2.0%); see System.jitter_raw_start.  Systems without that
    method (minimal test stubs) fall back to the historical raw-space jitter.

    ``raw_scales`` (optional) is the per-element dispersion scale in current
    raw units, as measured by the startup whitening pass (run.py passes
    whiten_report["raw_scales"]).  When given, the probe is skipped entirely
    -- the model was just whitened against the very same start, so re-probing
    would re-derive ~1.0 everywhere at n_elements x O(10) logp calls.  When
    absent (measure_scales: false, or standalone use), the probe runs as
    before.

    Returns (starts, chain_seed_index) where chain_seed_index[j] is the original
    seed index that chain j was drawn from (for trace-attr provenance).
    """
    if isinstance(raw_starts, dict):
        raw_starts = [raw_starts]
    K = len(raw_starts)
    if seed_indices is None:
        seed_indices = list(range(K))

    if raw_scales is not None:
        map_lp = float(logp_fn(raw_starts[0]))
        scales = {
            k: np.asarray(
                raw_scales.get(k, np.ones_like(np.asarray(v, dtype=float))),
                dtype=float,
            ).reshape(np.shape(v))
            for k, v in raw_starts[0].items()
        }
    else:
        # Probe scales once from seed 0 (the canonical MAP-ish start); the same
        # per-parameter jitter scale is reused around every seed.
        map_lp, scales = _probe_scales(raw_starts[0], logp_fn)
    n_params = sum(v.size for v in raw_starts[0].values())
    factor = min(np.sqrt(500.0 / max(n_params, 1)), 3.0)
    max_iter = 1000

    _jitter = (
        getattr(system, "jitter_raw_start", None)
        if system is not None
        else None
    )
    logger.info(
        f"PTDE init: MAP lp={map_lp:.1f}, n_params={n_params}, factor={factor:.2f}, "
        f"jitter={'physical (truncated)' if _jitter else 'raw (fallback)'}"
        + (
            f", {K} seeds (round-robin over {n_chains} chains)"
            if K > 1
            else ""
        )
    )

    starts = []
    chain_seed_index = []
    seed_seen = set()
    # Cap the number of chains that start EXACTLY at a seed. With K ~
    # n_chains (a params.2-style posterior-draw seed set), every chain
    # would otherwise start at a previous-run posterior draw with zero
    # jitter -- and since the DE population's spread IS the proposal
    # generator, the restart could then never explore beyond the previous
    # posterior, entrenching whatever that run missed. At least half the
    # chains always get the factor-scaled jitter.
    max_exact = max(1, n_chains // 2)
    n_exact = 0
    if K > max_exact:
        logger.info(
            f"PTDE init: {K} seeds > {max_exact} exact-start budget "
            f"(half the {n_chains} chains); the rest start jittered around "
            f"their seed to keep restart overdispersion."
        )
    for j in range(n_chains):
        s = j % K
        center = raw_starts[s]
        # First chain of each seed group starts exactly at the solved seed
        # (up to the overdispersion budget above).
        if s not in seed_seen and n_exact < max_exact:
            lp0 = float(logp_fn(center))
            if np.isfinite(lp0):
                starts.append({k: v.copy() for k, v in center.items()})
                chain_seed_index.append(seed_indices[s])
                seed_seen.add(s)
                n_exact += 1
                logger.debug(
                    f"PTDE init chain {j}: exact seed {seed_indices[s]} "
                    f"(lp={lp0:.1f})"
                )
                continue
            logger.warning(
                f"PTDE init: seed {seed_indices[s]} exact start has non-finite "
                f"lp; jittering to find a finite start."
            )
        for niter in range(max_iter):
            eff = factor / np.exp(niter / 1000.0)
            if _jitter is not None:
                prop = _jitter(center, scales, eff, rng)
            else:
                prop = {
                    k: v + eff * scales[k] * rng.standard_normal(v.shape)
                    for k, v in center.items()
                }
            lp = float(logp_fn(prop))
            if np.isfinite(lp):
                starts.append(prop)
                chain_seed_index.append(seed_indices[s])
                seed_seen.add(s)
                logger.debug(
                    f"PTDE init chain {j} (seed {seed_indices[s]}): accepted after "
                    f"{niter} retries (lp={lp:.1f}, dlp={lp - map_lp:.1f})"
                )
                break
            if niter % 200 == 0 and niter > 0:
                logger.warning(
                    f"PTDE init chain {j}: {niter} retries still seeking finite lp "
                    f"(eff={eff:.3g})"
                )
        else:
            raise RuntimeError(
                f"PTDE chain {j} initialization failed after {max_iter} retries. "
                f"Check initval/bounds in your params.yaml -- a parameter may be "
                f"starting outside its prior bounds."
            )
    # Over-dispersion is what makes Rhat mean anything; measure it once, on
    # the population that will actually run (review 2.4.5).
    warn_if_starts_underdispersed(starts, scales, "PTDE init", logger)
    return starts, chain_seed_index


def resolve_start_population(
    model,
    system,
    n_chains,
    logp_fn,
    rng,
    raw_start,
    initvals=None,
    raw_starts=None,
    seed_indices=None,
    raw_scales=None,
):
    """Resolve the T=1 chain starts: explicit initvals, or multi-seed
    round-robin via _make_starts (P4).

    raw_starts/seed_indices come from run.py when available; else fall back
    to system.get_raw_starts, and further to a bare raw_start (single start)
    for minimal test/system stubs that don't implement get_raw_starts at all.

    The explicit-``initvals`` bypass RAISES on a length mismatch rather than
    asserting it: `python -O` compiles an assert out entirely, and the list
    is consumed positionally (start j becomes chain j), so a wrong-length
    list would silently pair chains with the wrong starts -- or, shorter
    than n_chains, hand the sampler a population it then indexes past.

    Returns (t1_starts, chain_seed_index).
    """
    if initvals is not None:
        if len(initvals) != n_chains:
            raise ValueError(
                f"initvals has {len(initvals)} start(s) but there are "
                f"{n_chains} chains per rung; the list is consumed "
                f"positionally, so it must have exactly one entry per "
                f"chain. Pass n_chains={len(initvals)}, or omit initvals to "
                f"start from the solved point."
            )
        return initvals, [0] * n_chains
    if raw_starts is None:
        if hasattr(system, "get_raw_starts"):
            raw_starts, seed_indices = system.get_raw_starts(model)
        else:
            raw_starts, seed_indices = [raw_start], [0]
    return _make_starts(
        n_chains,
        raw_starts,
        logp_fn,
        rng,
        seed_indices,
        system=system,
        raw_scales=raw_scales,
    )


def plot_start_ensemble(
    system,
    t1_starts,
    raw_to_phys_batched,
    raw_var_names,
    out_var_names,
    plot_prefix,
    log,
):
    """Ensemble start plots (T=1 starts only; raw -> physical via the
    batched fn). No-op when plot_prefix is None."""
    if plot_prefix is None:
        return
    log.info("Generating ensemble start plots...")
    batched_vals = raw_to_phys_batched(
        *[np.stack([s[k] for s in t1_starts], axis=0) for k in raw_var_names]
    )
    internal_starts = [
        {
            name: np.asarray(val)[i]
            for name, val in zip(out_var_names, batched_vals)
        }
        for i in range(len(t1_starts))
    ]
    for comp in system.active_components.values():
        comp.plot(
            system,
            internal_starts,
            filename_prefix=plot_prefix + "_start_ensemble",
        )


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def install_stop_handlers(handler):
    """Route SIGINT/SIGTERM (and Windows SIGBREAK) to `handler`.

    SIGTERM gets the same handler as SIGINT so a batch scheduler (e.g.
    `qsig -s SIGTERM <job_id>` / `kill -TERM <pid>`) can request the same
    graceful stop-after-this-step behavior as a Ctrl+C at a terminal,
    instead of Python's default SIGTERM action (immediate termination,
    discarding whatever draws were already collected).  Windows delivers
    the GUI's stop request as CTRL_BREAK_EVENT, which arrives as SIGBREAK
    rather than SIGINT (see gui/runner.py); without wiring it the request
    is received and silently ignored.

    Returns an opaque token for restore_stop_handlers.
    """
    old_sigint = signal.signal(signal.SIGINT, handler)
    old_sigterm = signal.signal(signal.SIGTERM, handler)
    old_sigbreak = (
        signal.signal(signal.SIGBREAK, handler)
        if hasattr(signal, "SIGBREAK")
        else None
    )
    return old_sigint, old_sigterm, old_sigbreak


def restore_stop_handlers(token):
    old_sigint, old_sigterm, old_sigbreak = token
    signal.signal(signal.SIGINT, old_sigint)
    signal.signal(signal.SIGTERM, old_sigterm)
    if old_sigbreak is not None:
        signal.signal(signal.SIGBREAK, old_sigbreak)


# ---------------------------------------------------------------------------
# Runaway-lp guard
# ---------------------------------------------------------------------------


class LpPlausibilityGuard:
    """Warn ONCE when a T=1 chain's accepted lp exceeds the plausibility
    ceiling.

    A T=1 chain's lp this large always indicates a model bug (an unbounded/
    uncancelled logp term), never real physics -- no finite dataset's logp
    reaches 1e12. PTDE accepts on lp_new > lp_old, so such a bug is a
    ratchet: once a chain's lp is inflated this way it can only climb
    further, wasting the rest of the run (see examples/DC2018_128, a real
    occurrence of exactly this failure mode). Warn once so it's noticed
    immediately rather than discovered post-hoc via identify_modes, which
    rejects draws on the same ceiling (outputs.modes.DEFAULT_LP_ABS_MAX).
    """

    def __init__(self, ceiling, label, log):
        self.ceiling = DEFAULT_LP_ABS_MAX if ceiling is None else ceiling
        self.label = label
        self.log = log
        self.warned = False

    def check(self, chain_i, lp):
        """Call with every ACCEPTED T=1 (chain index, lp)."""
        if self.warned or abs(lp) <= self.ceiling:
            return
        self.log.warning(
            f"{self.label}: T=1 chain {chain_i} lp={lp:.3e} exceeds "
            f"the plausibility ceiling "
            f"(|lp| > {self.ceiling:g}); this "
            "almost always means a model bug (e.g. an "
            "unbounded logp term), not physics -- since "
            "PTDE only accepts lp increases, this chain "
            "will likely keep climbing for the rest of the "
            "run. See outputs.modes.identify_modes, which "
            "rejects draws on the same ceiling post-hoc."
        )
        self.warned = True


# ---------------------------------------------------------------------------
# Posterior assembly + diagnostics reports
# ---------------------------------------------------------------------------


def assemble_inference_data(
    stored_raw,
    stored_lp,
    actual_draws,
    n_chains,
    raw_start,
    raw_var_names,
    out_var_names,
    raw_to_phys_batched,
    chain_seed_index,
    label,
    log,
):
    """Convert the stored T=1 raw draws to physical space and build the
    arviz.InferenceData (posterior + sample_stats.lp + multi-seed attrs).

    Flattens (n_chains, draws) -> (n_total,) per raw variable and runs the
    batched converter in chunks (bounds memory for large n_params/draws;
    chunk_size is independent of param count/shape, only of sample count).
    """
    log.info(
        f"{label}: converting {n_chains} x {actual_draws} draws to "
        f"physical space..."
    )
    n_total = n_chains * actual_draws
    flat_raw = {
        k: stored_raw[k][:, :actual_draws].reshape(
            (n_total,) + raw_start[k].shape
        )
        for k in raw_var_names
    }
    chunk_size = 20000
    out_chunks = {name: [] for name in out_var_names}
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_out = raw_to_phys_batched(
            *[flat_raw[k][start:end] for k in raw_var_names]
        )
        for name, val in zip(out_var_names, chunk_out):
            out_chunks[name].append(np.asarray(val, dtype=float))

    # assemble posterior dict: (n_chains, draws, ...) per variable
    posterior_dict = {}
    for name in out_var_names:
        arr = np.concatenate(out_chunks[name], axis=0)  # (n_total, ...)
        arr = arr.reshape((n_chains, actual_draws) + arr.shape[1:])
        # old per-sample path ran every value through atleast_1d then squeezed
        # a trailing dim-1 for scalar params -- match that convention here.
        if arr.ndim > 2 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        posterior_dict[name] = arr

    idata = az.from_dict(
        {
            "posterior": posterior_dict,
            "sample_stats": {"lp": stored_lp[:, :actual_draws]},
        }
    )

    # Multi-seed provenance (P4): record which solved seed each T=1 chain was
    # started from.  With seeded starts, occupancy weights are initialization
    # artifacts BY DESIGN unless chains mix, so downstream reporting must be
    # able to say "chains 0-3 at seed 0, 4-7 at seed 1".
    # TODO(P4): surface this in outputs/modes.py ModeReport once chains->modes
    # attribution is wired; for now the per-chain attr is the source of truth.
    idata.posterior.attrs["chain_seed_index"] = list(chain_seed_index)
    if len(set(chain_seed_index)) > 1:
        log.info(
            f"{label} multi-seed provenance (chain -> seed): "
            f"{list(chain_seed_index)}"
        )
    return idata


# Draws allocated per growth step (see grow_draw_storage).  Large enough
# that the reallocation is rare (one per 10k draws/chain), small enough that
# a run stopped early has not reserved a trace it never wrote.
DRAW_CHUNK = 10000


def grow_draw_storage(stored_raw, stored_lp, needed, chunk=DRAW_CHUNK):
    """Ensure the T=1 draw buffers hold `needed` draws per chain; grow if not.

    The buffers used to be allocated at the FULL configured draw count up
    front, which is wrong in both directions once `draws` is large.  A run
    that stops on convergence, maxtime or a user interrupt has reserved --
    and, being np.zeros, touched -- the whole thing regardless: at
    draws=252600 on a 27-parameter model that is ~1.6 GB of resident memory
    for draws the run will never take (review 6.4.5).

    Grows by whole chunks with np.resize-free concatenation on the DRAW axis,
    and mutates the dict in place, because the same dict object is handed to
    the GUI progress callback and to _check_convergence by reference.  The
    caller must therefore re-read `stored_lp` from the return value -- it is
    a bare array, not a container.

    Returns the (possibly new) stored_lp array.
    """
    have = stored_lp.shape[1]
    if needed <= have:
        return stored_lp
    add = max(chunk, needed - have)
    n_chains = stored_lp.shape[0]
    for key, arr in stored_raw.items():
        pad = np.zeros((n_chains, add) + arr.shape[2:], dtype=arr.dtype)
        stored_raw[key] = np.concatenate([arr, pad], axis=1)
    return np.concatenate(
        [stored_lp, np.zeros((n_chains, add), dtype=stored_lp.dtype)], axis=1
    )


def stamp_and_log_run_summary(
    idata,
    label,
    log,
    *,
    actual_draws,
    draws,
    n_accept,
    n_propose,
    n_swap_accept,
    n_swap_propose,
    round_trips,
    n_swap_rounds,
    n_temps,
    swap_schedule,
    rate_unit,
    extras=(),
):
    """Stamp the ladder round-trip attrs and log the one-line run summary.

    Shared by both samplers, which had ~20 near-identical lines each and had
    already started to drift.  Deliberately stops SHORT of the ladder
    diagnostics: ladder_health_report is called by each sampler afterwards,
    because what its counters span differs between them (ptde zeroes at the
    tune -> draw boundary, ptde_async never resets and says so), and because
    that is the code the announced adaptive-rung-spacing work will reshape.

    ``rate_unit`` is "round" (a synchronous DEO round) or "swap" (one async
    swap event) -- the same denominator each sampler already used for its
    round-trip rate.  ``extras`` are pre-formatted trailing fragments (the
    async swap-discard count, either sampler's timeout count), appended in
    order and each already carrying its leading separator.

    THE ROUND TRIPS ARE TEMPERATURE round trips of a replica
    (T=1 -> T_max -> T=1), NOT mode changes: outputs.modes counts the latter
    itself from the stored T=1 labels and labels the two separately, because
    "swap" is ambiguous between them.
    """
    idata.posterior.attrs["ptde_ladder_round_trips"] = int(round_trips)
    idata.posterior.attrs["ptde_swap_rounds"] = int(n_swap_rounds)

    ar_T1 = float(n_accept[0] / max(n_propose[0], 1))
    sr_all = np.asarray(n_swap_accept) / np.maximum(n_swap_propose, 1)
    rt_rate = round_trips / max(n_swap_rounds, 1)
    log.info(
        f"{label} done: {actual_draws}/{draws} draws  "
        f"accept(T=1)={ar_T1:.3f}  "
        + (
            f"swap=[{', '.join(f'{r:.2f}' for r in sr_all)}]  "
            f"round_trips={round_trips} (rate={rt_rate:.3f}/{rate_unit}, "
            f"schedule={swap_schedule})"
            if n_temps > 1
            else ""
        )
        + "".join(extras)
    )


def log_rung_timing(rung_times, temperatures, label, log):
    """Per-rung logp wall-time summary (collect_rung_timing diagnostic)."""
    log.info(f"{label} per-rung logp timing (seconds):")
    for k, times in enumerate(rung_times):
        if not times:
            log.info(f"  rung {k} (T={temperatures[k]:.1f}): no calls")
            continue
        arr = np.asarray(times)
        n_slow = int((arr > 0.1).sum())
        log.info(
            f"  rung {k} (T={temperatures[k]:.1f}): n={len(arr)}  "
            f"median={np.median(arr):.3f}  mean={arr.mean():.3f}  "
            f"p90={np.percentile(arr, 90):.3f}  max={arr.max():.3f}  "
            f"n_slow(>0.1s)={n_slow}"
        )
