"""Data-driven whitening: measure each raw parameter's true local scale and
rescale the model's whitening in place.

The model is first built with PRELIMINARY whitening scales (defaults.yaml
init_scale, or a fraction of the bound span when absent -- see
Parameter.build_pymc).  probe_scales() then measures, per raw element, the
step in the current raw coordinate whose symmetric second difference of logp
reaches 1 nat -- 0.5 nats per side, the EXOFASTv2 exofast_getmcmcscale
convention (dlogp=0.5 <-> dchi2=1, the 1-sigma step for a locally Gaussian
posterior), made gradient-immune: the second difference cancels the linear
Taylor term identically, so the measurement returns the local curvature width
1/sqrt(h) even when the start is far below the mode (where the one-sided drop
it replaces measured 0.5/|gradient| instead, arbitrarily tight).  Directions
with no measurable positive curvature fall back to the one-sided nearer
contour; see probe_scales.  Because one raw unit is one preliminary scale by
construction, the measured step IS the multiplicative correction to the
preliminary scale.  apply_measured_whitening() feeds it to
Parameter.set_whitening(), which updates the pytensor.shared whitening scales
in place -- no model rebuild, and the posterior is provably unchanged.

That proof covers the LOGIT branch only -- an element with two finite bounds,
whose raw coordinate is flat (Parameter.build_pymc's _raw_logp) and whose
uniform prior is section C's Jacobian for any scale, so the scale affects
conditioning alone.  It is
exactly the set set_whitening rescales.  Every NON-logit element (anything
without two finite bounds) is left untouched, because there the raw N(0,1) IS
the prior: N(mu, sigma) when a sigma was given, and N(initval, init_scale)
when the bounds are infinite and no sigma was -- the one place init_scale is
a posterior term rather than conditioning, and one a data-measured multiplier
would turn into a data-dependent prior width.  build_pymc warns when it builds
such an element.  So the invariant holds for the pass as a whole, but only
because the branch it does not hold for is never rescaled.

After the rescale, every raw direction costs ~0.5 nats per unit step: the
"curvature = -1" conditioning the retired curvature check used to ask users
to approximate by hand-tuning init_scale, reached here without iteration.

The probe originated in the PTDE sampler's chain-initialization
(samplers/ptde.py imports it from here); the whitening pass uses a wider
dynamic range, since a preliminary scale can be off by many orders of
magnitude while PTDE re-probes an already-whitened model where every scale
is ~1.
"""

import json
import logging
import os
import queue
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# target = 0.5 nats (dlogp=0.5 <-> dchi2=1, the EXOFASTv2 convention); for a
# locally Gaussian posterior that step IS the 1-sigma width along the element.
PROBE_TARGET_DELTA = 0.5
_PROBE_MAX_STEP = 1.0e3  # raw units; beyond this a direction is "flat"
_PROBE_MIN_STEP = 1.0e-8  # raw units; floor on a returned step
_PROBE_RTOL = 0.05  # accept a step whose drop is within 5% of target
_PROBE_MAX_BISECT = 40
_PROBE_FLAT_SCALE = 1.0  # fallback when neither direction reaches target;
# one raw unit is one preliminary scale by construction, so 1.0
# is the natural unit for a direction logp ignores.

# An element whose gradient contributes more than this many nats over one
# measured scale unit is "gradient-dominated": for a locally quadratic logp,
# |g|*sigma is the start's displacement from its conditional optimum in
# sigmas, so 3.0 means "the start is >~3 sigma off along this direction".
# The curvature probe still measures the width correctly there (that is its
# point); the warning exists because everything else that uses the START --
# chain scatter, tempered burn-in, the reported start table -- degrades with
# displacement, and a seed polish would fix it at the source.
_GRAD_DOMINATED_NATS = 3.0

# The whitening pass widens the probe's dynamic range: a preliminary scale
# (defaults.yaml, or the span-fraction fallback) can be off by 10+ orders of
# magnitude in either direction, where PTDE's defaults -- tuned for a model
# that is already whitened -- would hit the 1e-8 step floor or declare a
# merely-distant contour "flat".  Bracketing is geometric and bisection
# halves, so the wider range costs only tens of extra logp calls per element,
# and only for elements whose preliminary scale was badly off.
_WHITEN_MAX_STEP = 1.0e9
_WHITEN_MIN_STEP = 1.0e-14
_WHITEN_MAX_BISECT = 90

# THE PROBE RUNS ON A WORKER POOL WHEN THE LOGP IS EXPENSIVE (review
# 2.6.41).  It used to call logp one element at a time in the main process:
# untimed, silent, one core.  OGLE-2013-BLG-0341 (three lenses, ~225 s per
# logp, 112 cores) sat 12.85 h in it with 111 cores idle and no line in the
# log, and 62 elements x tens of evaluations each cannot fit a 3-day wall
# serially at all.  The elements are independent given the start, so each
# element's bracket+bisect search runs as its own coroutine and every
# evaluation any of them asks for goes to the pool at once (_probe_on_pool),
# each under the sampler's `eval_timeout`.  The serial path stays for
# `cores: 1` and for a CHEAP logp: below this many seconds per call (timed
# on the start's own evaluation) forking a pool and shipping every probe
# point through IPC costs more than it saves, and nothing here can hang
# long enough to need a timeout -- unless the user set one, which always
# takes the pool, since a timeout cannot be enforced in-process.  The two
# paths return bit-identical scales (see _need).
PROBE_POOL_MIN_LOGP_S = 0.05

# Wall-clock interval between the probe's progress lines (elements done /
# total, logp evaluations, in flight, ETA).  A probe that finishes inside
# one interval logs nothing, which keeps a fast model's log (and the test
# suite) quiet; a slow one says it is alive every interval, so "computing"
# and "hung" are distinguishable from the log alone.
PROBE_PROGRESS_S = 60.0

# A probe element abandoned because one of its logp calls exceeded
# eval_timeout (see _probe_on_pool).
_TIMED_OUT = "timed out"

_BANNER = "!" * 70


class StaleWhiteningError(RuntimeError):
    """A saved trace's whitening file does not apply to this build.

    Raised only on the REUSE path (``recompute_trace: false`` with an
    existing trace).  See :func:`restore_whitening_for_trace`.
    """


def _need(cache, steps):
    """Generator: the deltas at ``steps``, asking the DRIVER for the ones not
    yet in ``cache``.

    The probe's step search is written as a coroutine so that one search
    can be driven either in-process (``_drive``: each requested step is
    evaluated on the spot, in order) or by the pool scheduler
    (``_probe_on_pool``: the requested steps of every element in flight at
    once, each under ``eval_timeout``).  A request is the list of uncached
    steps, in the order the old serial code evaluated them (the curvature
    probe asks for +s then -s); the driver sends back their deltas in the
    same order.  logp is a pure function of the point, so the two drivers
    see the same numbers and take the same branches -- the serial and the
    pooled probe are bit-identical by construction, which
    tests/test_whitening_pool.py checks on a shipped example.
    """
    missing = list(dict.fromkeys(s for s in steps if s not in cache))
    if missing:
        vals = yield missing
        for s, v in zip(missing, vals):
            cache[s] = v
    return [cache[s] for s in steps]


def _drive(gen, eval_delta):
    """Run a probe coroutine in-process: evaluate each request in order."""
    try:
        steps = next(gen)
        while True:
            steps = gen.send([eval_delta(s) for s in steps])
    except StopIteration as stop:
        return stop.value


def _probe_step_1d_gen(
    cache,
    sign,
    target=PROBE_TARGET_DELTA,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    rtol=_PROBE_RTOL,
    max_bisect=_PROBE_MAX_BISECT,
):
    """probe_step_1d as a coroutine over ``cache`` (see _need)."""
    # BRACKET: lo always has delta < target (delta(0) == 0); hi is the first
    # step at/above target, or non-finite (a hard prior wall counts as "past").
    lo, hi = 0.0, None
    step = 1.0
    while step <= max_step:
        (d,) = yield from _need(cache, (sign * step,))
        if (not np.isfinite(d)) or d >= target:
            hi = step
            break
        lo = step
        step *= 2.0
    if hi is None:
        return None

    # BISECT.
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        (d,) = yield from _need(cache, (sign * mid,))
        if np.isfinite(d) and abs(d - target) <= rtol * target:
            return mid
        if (not np.isfinite(d)) or d > target:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < min_step:
            break
    return max(0.5 * (lo + hi), min_step)


def probe_step_1d(
    eval_delta,
    sign,
    target=PROBE_TARGET_DELTA,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    rtol=_PROBE_RTOL,
    max_bisect=_PROBE_MAX_BISECT,
):
    """Step magnitude in one direction whose logp drop reaches the target.

    `eval_delta(step)` returns logp(start) - logp(start + step), so it is 0 at
    step=0 and grows as the step walks logp downhill.  Returns None when the
    direction is flat out to max_step.

    Brackets by geometric growth, then bisects.  delta may start NEGATIVE --
    raw_start is initval, not the MAP, so one direction is typically uphill --
    and growing through the turnover still lands on the step where logp has
    fallen `target` below the START, which is the quantity being asked for.
    """
    return _drive(
        _probe_step_1d_gen(
            {},
            sign,
            target=target,
            max_step=max_step,
            min_step=min_step,
            rtol=rtol,
            max_bisect=max_bisect,
        ),
        eval_delta,
    )


def _probe_element_gen(
    target=PROBE_TARGET_DELTA,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    rtol=_PROBE_RTOL,
    max_bisect=_PROBE_MAX_BISECT,
):
    """_probe_element as a coroutine (see _need); same return value."""
    cache = {}

    def curv(s):
        dp, dm = yield from _need(cache, (s, -s))
        if not (np.isfinite(dp) and np.isfinite(dm)):
            return np.inf  # a wall inside +/-s: treat as past-target (tight)
        return dp + dm

    target_c = 2.0 * target
    # BRACKET on the symmetric drop; same geometry as probe_step_1d.
    lo, hi = 0.0, None
    step = 1.0
    while step <= max_step:
        c = yield from curv(step)
        if c >= target_c:
            hi = step
            break
        lo = step
        step *= 2.0

    if hi is not None:
        s_h = None
        for _ in range(max_bisect):
            mid = 0.5 * (lo + hi)
            c = yield from curv(mid)
            if np.isfinite(c) and abs(c - target_c) <= rtol * target_c:
                s_h = mid
                break
            if c > target_c:
                hi = mid
            else:
                lo = mid
            if (hi - lo) < min_step:
                break
        if s_h is None:
            s_h = max(0.5 * (lo + hi), min_step)
        dp, dm = yield from _need(cache, (s_h, -s_h))
        grad_nats = (
            0.5 * abs(dp - dm)
            if np.isfinite(dp) and np.isfinite(dm)
            else np.nan
        )
        return min(s_h, max_step), "curvature", grad_nats

    # FALLBACK: no measurable positive curvature out to max_step.  The
    # one-sided searches reuse the cached evaluations from the bracket above.
    steps = []
    for sign in (1.0, -1.0):
        x = yield from _probe_step_1d_gen(
            cache,
            sign,
            target=target,
            max_step=max_step,
            min_step=min_step,
            rtol=rtol,
            max_bisect=max_bisect,
        )
        steps.append(x)
    steps = [x for x in steps if x is not None]
    if steps:
        return min(float(np.min(steps)), max_step), "linear", np.nan
    return None, "flat", np.nan


def _probe_element(
    eval_delta,
    target=PROBE_TARGET_DELTA,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    rtol=_PROBE_RTOL,
    max_bisect=_PROBE_MAX_BISECT,
):
    """Measure one element's scale: symmetric curvature first, then the
    one-sided nearer contour, then None (flat).

    The symmetric second difference C(s) = delta(+s) + delta(-s)
    (= 2*lp(x0) - lp(x0+s) - lp(x0-s)) cancels the gradient term of the
    Taylor expansion identically, so bracket+bisect on C(s) = 2*target
    returns 1/sqrt(h) -- the local Gaussian sigma -- whether or not the
    start is at the mode.  A non-finite side (hard prior wall inside +/-s)
    counts as past-target, so a wall bounds the scale at its own distance,
    matching the one-sided behavior.

    When C never reaches the target within max_step (h <= 0: a locally
    linear direction, a ridge, or a saddle) the curvature is meaningless
    and the measurement falls back to the one-sided probe of each sign,
    taking the NEARER contour -- which errs tight, the direction every
    downstream consumer absorbs.  When both signs are flat too, returns
    (None, "flat", nan) and the caller applies its flat fallback.

    Returns (scale_or_None, method, grad_nats) where method is one of
    "curvature" / "linear" / "flat" and grad_nats = |delta(+s) - delta(-s)|/2
    at the accepted curvature scale (nan otherwise): the gradient's nats over
    one scale unit, i.e. the start's displacement from its conditional
    optimum in sigmas for a locally quadratic logp.

    This is the in-process driver of _probe_element_gen; the whitening pass
    can also drive the same coroutine on a worker pool (_probe_on_pool).
    """
    return _drive(
        _probe_element_gen(
            target=target,
            max_step=max_step,
            min_step=min_step,
            rtol=rtol,
            max_bisect=max_bisect,
        ),
        eval_delta,
    )


class _ProbeProgress:
    """The probe's progress line, rate-limited on one wall clock.

    One line per PROBE_PROGRESS_S of wall clock: elements done / total,
    logp evaluations so far, evaluations in flight (pool path), elapsed and
    an ETA at the element rate so far.  Nothing is logged by a probe that
    finishes inside the first interval; one that logged at least once also
    logs its closing total, so the last line of a slow probe is never a
    stale "in progress".
    """

    def __init__(self, total, where):
        self.total = int(total)
        self.where = where
        self.t0 = time.monotonic()
        self.last = self.t0
        self.n_lines = 0

    def beat(self, n_done, n_evals, n_in_flight=None, force=False):
        now = time.monotonic()
        if not force and now - self.last < PROBE_PROGRESS_S:
            return
        self.last = now
        self.n_lines += 1
        elapsed = now - self.t0
        if n_done >= self.total:
            eta = "done"
        elif n_done > 0:
            eta = (
                f"ETA ~{elapsed / n_done * (self.total - n_done):.0f} s at "
                f"the element rate so far"
            )
        else:
            eta = "ETA unknown (no element finished yet)"
        flight = "" if n_in_flight is None else f", {n_in_flight} in flight"
        logger.info(
            f"Whitening probe: {n_done}/{self.total} element(s) done, "
            f"{n_evals} logp evaluation(s){flight} ({self.where}), "
            f"elapsed {elapsed:.0f} s, {eta}."
        )

    def finish(self, n_evals):
        if self.n_lines:
            self.beat(self.total, n_evals, force=True)


def _probe_elements(
    raw_start,
    logp_fn,
    elems,
    map_lp,
    t_logp,
    probe_kw,
    *,
    cores=1,
    eval_timeout=None,
):
    """Probe every (key, flat_index) in ``elems`` around ``raw_start``.

    Returns {(key, i): (scale_or_None, method, grad_nats)} -- _probe_element's
    triple -- or _TIMED_OUT for an element abandoned on `eval_timeout`.

    Serial (in this process) when the resolved grant is one core, or when
    the logp is cheap (``t_logp``, the start's own evaluation, below
    PROBE_POOL_MIN_LOGP_S) and no `eval_timeout` asks for enforcement;
    otherwise on a pool (_probe_on_pool).  Both drive the same coroutine per
    element, so their results are bit-identical.
    """
    from .samplers import _common

    if cores is None or cores <= 0:
        cores = _common.default_cores()
    want_pool = (
        int(cores) > 1
        and len(elems) > 0
        and (eval_timeout is not None or t_logp >= PROBE_POOL_MIN_LOGP_S)
    )
    if want_pool:
        return _probe_on_pool(
            raw_start,
            logp_fn,
            elems,
            map_lp,
            probe_kw,
            cores=int(cores),
            eval_timeout=eval_timeout,
        )

    progress = _ProbeProgress(len(elems), "serial, 1 core")
    n_evals = [0]
    results = {}
    for n_done, (key, i) in enumerate(elems):

        def eval_delta(step, key=key, i=i):
            probe = {k: v.copy() for k, v in raw_start.items()}
            probe[key].flat[i] += step
            n_evals[0] += 1
            return map_lp - float(logp_fn(probe))

        results[(key, i)] = _probe_element(eval_delta, **probe_kw)
        progress.beat(n_done + 1, n_evals[0])
    progress.finish(n_evals[0])
    return results


def _probe_on_pool(
    raw_start, logp_fn, elems, map_lp, probe_kw, *, cores, eval_timeout
):
    """The pooled probe: every element's search in flight at once.

    Each element's _probe_element_gen asks for its next one or two steps;
    those evaluations go to a worker pool forked here (so the workers see
    the model's CURRENT whitening -- the shared variables are copied at
    fork, which is also why the pool lives for one probe pass and is never
    reused across a rescale), and an element whose request comes back
    asks for the next.  At most one evaluation per worker is in flight, so
    the `eval_timeout` clock, which starts at submit, measures the call and
    not a queue wait (the ptde_async livelock in samplers.md cannot happen
    here).

    A call older than `eval_timeout` abandons ITS ELEMENT: the element is
    _TIMED_OUT, which the whitening pass treats exactly like a failed probe
    -- the element keeps its preliminary (declared or propagated) scale --
    and a WARNING names it.  A hung worker cannot be killed alone, so the
    pool is recycled and every other in-flight evaluation is resubmitted
    unchanged; evaluations are pure, so the other elements' results are
    unaffected.  A result from the old pool that arrives late finds its
    submission id gone and is dropped.
    """
    from .samplers import _common

    _common.set_worker_globals(logp_fn)
    pool, n_workers = _common.create_pool(
        cores, 2 * len(elems), "Whitening probe", logger
    )
    if pool is None:  # pragma: no cover - create_pool returns None only
        # for one worker, which _probe_elements already routed serially.
        raise RuntimeError(
            f"Whitening probe: create_pool gave no pool for cores={cores} "
            f"and {len(elems)} element(s)"
        )
    timeout_note = (
        f", each logp call under eval_timeout={float(eval_timeout):g}s"
        if eval_timeout is not None
        else " (no eval_timeout: a hung call is not written off)"
    )
    logger.info(
        f"Whitening probe: {len(elems)} element(s) on {n_workers} worker "
        f"process(es){timeout_note}."
    )
    progress = _ProbeProgress(len(elems), f"{n_workers} workers")

    results = {}
    gens = {}
    waiting = {}  # elem -> (requested steps, {step: delta})
    ready = deque()  # (elem, step) not yet submitted
    in_flight = {}  # sub_id -> (elem, step, t_submit)
    result_q = queue.Queue()
    seq = [0]
    n_evals = [0]

    def _advance(e, vals):
        try:
            steps = next(gens[e]) if vals is None else gens[e].send(vals)
        except StopIteration as stop:
            results[e] = stop.value
            del gens[e]
            return
        waiting[e] = (steps, {})
        ready.extend((e, st) for st in steps)

    def _submit(e, st):
        key, i = e
        probe = {k: v.copy() for k, v in raw_start.items()}
        probe[key].flat[i] += st
        seq[0] += 1
        sid = seq[0]
        in_flight[sid] = (e, st, time.monotonic())

        def _cb(lp, sid=sid):
            result_q.put((sid, lp, None))

        def _ecb(exc, sid=sid):
            result_q.put((sid, None, exc))

        pool.apply_async(
            _common._eval_logp, (probe,), callback=_cb, error_callback=_ecb
        )

    def _fill():
        while ready and len(in_flight) < n_workers:
            _submit(*ready.popleft())

    def _enforce_timeout():
        nonlocal pool
        if eval_timeout is None or not in_flight:
            return
        now = time.monotonic()
        stale = {
            e: st
            for e, st, t0 in in_flight.values()
            if now - t0 > float(eval_timeout)
        }
        if not stale:
            return
        for (key, i), st in stale.items():
            e = (key, i)
            logger.warning(
                f"Whitening probe: a logp call for {key}[{i}] exceeded "
                f"eval_timeout={float(eval_timeout):g}s (raw step "
                f"{st:+.6g} from the start); abandoning that element's "
                f"probe -- it keeps its preliminary (declared or "
                f"propagated) scale."
            )
            results[e] = _TIMED_OUT
            gens.pop(e).close()
            waiting.pop(e, None)
        # The pool goes down whole (no single worker can be killed), so
        # everything else in flight is resubmitted unchanged, ahead of the
        # queue it was already ahead of.
        survivors = [
            (e, st) for e, st, _t0 in in_flight.values() if e not in stale
        ]
        in_flight.clear()
        rest = [(e, st) for e, st in ready if e not in stale]
        ready.clear()
        ready.extend(survivors + rest)
        logger.warning(
            "Whitening probe: recycling the worker pool -- a hung worker "
            "never rejoins it on its own."
        )
        pool = _common.recycle_pool(pool, n_workers)

    wait_s = 1.0
    if eval_timeout is not None:
        wait_s = min(wait_s, max(0.05, float(eval_timeout) / 4.0))
    try:
        for e in elems:
            gens[e] = _probe_element_gen(**probe_kw)
            _advance(e, None)
        _fill()
        while in_flight or ready:
            try:
                sid, lp, exc = result_q.get(timeout=wait_s)
            except queue.Empty:
                sid = None
            if sid is not None:
                meta = in_flight.pop(sid, None)
                if meta is not None:
                    if exc is not None:
                        # _eval_logp catches the model's own exceptions; an
                        # error here is the pool's plumbing failing, which
                        # nothing downstream can repair.
                        raise RuntimeError(
                            f"Whitening probe: the worker pool failed on "
                            f"{meta[0][0]}[{meta[0][1]}]: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    e, st, _t0 = meta
                    n_evals[0] += 1
                    if e in waiting:
                        steps, got = waiting[e]
                        got[st] = map_lp - float(lp)
                        if len(got) == len(steps):
                            del waiting[e]
                            _advance(e, [got[x] for x in steps])
            _enforce_timeout()
            _fill()
            progress.beat(len(results), n_evals[0], len(in_flight))
    finally:
        _common._shutdown_pool(pool)
    progress.finish(n_evals[0])
    return results


def probe_scales(
    raw_start,
    logp_fn,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    max_bisect=_PROBE_MAX_BISECT,
    flat_scale=_PROBE_FLAT_SCALE,
    diagnostics=None,
    *,
    cores=1,
    eval_timeout=None,
):
    """Adaptive probe: per-element scale whose raw step costs ~0.5 nats.

    Per element the measurement is _probe_element's three-rung ladder:

    1. Symmetric curvature: bracket+bisect on the second difference
       C(s) = 2*lp(x0) - lp(x0+s) - lp(x0-s) = h*s^2 + O(s^4), whose target
       crossing is 1/sqrt(h) -- the local Gaussian sigma -- for ANY gradient.
       The one-sided drop this replaces contains the gradient term g*s, so
       from a start that is not the mode it measured 0.5/|g| instead of
       sigma: on examples/ob140939 a start ~5900 nats below the posterior
       returned scales ~1000x too small, the mass matrix re-widened the raw
       posterior to sd ~5e3, and the sampler diverged on 86% of draws
       against the raw-cancellation clip's wall at |raw|=1e4 (since
       removed: review 1.2.9, parameter.py's _raw_logp).
    2. When C never reaches the target (h <= 0: locally linear directions,
       ridges, saddles), fall back to EXOFASTv2's exofast_getmcmcscale
       convention of searching each sign for the one-sided 0.5-nat drop,
       taking the NEARER contour (not the average: EXOFASTv2 probes from an
       AMOEBA best fit where the directions are near-symmetric; off the mode
       the uphill contour only turns over on the far side, and averaging it
       in inflates the scale -- chains then jitter past the logit
       saturation walls and start pinned at the bounds).  The nearer contour
       only ever errs tight, which downstream consumers absorb
       (chain-scatter multipliers, DE ensemble expansion, NUTS adaptation).
    3. When both signs are flat too, `flat_scale`.

    The step search is bracket + bisect.  An earlier version instead walked a
    fixed probe ladder [0.003 ... 2.0], took the FIRST magnitude registering
    any drop at all, and extrapolated with a quadratic assumption,
    scale = |dp| * sqrt(target/delta) -- which only holds AT a maximum, and
    for the generic locally-linear response collapses to sqrt(0.5*|dp|/g),
    tracking the smallest rung of the ladder (examples/ogle0383: 0.048
    returned where the true 0.5-nat step is 0.566, under-dispersing every
    chain 12-18x).

    `diagnostics`, when given, is a dict filled with:
      "linear_fallback" -- ["key[i]", ...] elements measured by rung 2
      "flat"            -- ["key[i]", ...] elements that fell to rung 3
      "gradient_nats"   -- {"key[i]": g} rung-1 elements' |gradient| in nats
                           per measured scale unit (the start's displacement
                           from its conditional optimum in sigmas, locally)
      "timed_out"       -- ["key[i]", ...] elements whose probe was abandoned
                           because one logp call exceeded `eval_timeout`;
                           they get `flat_scale`, like a flat direction, but
                           are NOT listed under "flat"

    `cores` / `eval_timeout` choose where the evaluations run (see
    PROBE_POOL_MIN_LOGP_S and _probe_elements): serial in this process with
    the default `cores=1`, otherwise on a forked worker pool, every call
    under `eval_timeout` seconds.  `cores=None` (or <= 0) is the AUTO grant,
    as everywhere else (samplers._common.default_cores).  The scales are
    bit-identical either way.

    Returns (map_lp, scales) where scales maps each raw-start key to an array
    shaped like its entry.
    """
    t0 = time.monotonic()
    map_lp = float(logp_fn(raw_start))
    t_logp = time.monotonic() - t0

    elems = [
        (key, i) for key, val in raw_start.items() for i in range(val.size)
    ]
    results = _probe_elements(
        raw_start,
        logp_fn,
        elems,
        map_lp,
        t_logp,
        dict(max_step=max_step, min_step=min_step, max_bisect=max_bisect),
        cores=cores,
        eval_timeout=eval_timeout,
    )

    scales = {}
    tight = []
    flat = []
    linear_fallback = []
    gradient_nats = {}
    timed_out = []
    for key, val in raw_start.items():
        n = val.size
        s = np.full(n, flat_scale)
        for i in range(n):
            res = results[(key, i)]
            if res is _TIMED_OUT:
                # The failed-probe contract: no measurement, so the caller's
                # flat_scale (NaN on the whitening pass: keep the preliminary
                # scale) -- but booked separately, because "timed out" is not
                # "flat" and must be neither escalated nor reported as one.
                timed_out.append(f"{key}[{i}]")
                continue
            scale, method, g_nats = res
            if scale is not None:
                s.flat[i] = scale
            if method == "flat":
                flat.append(f"{key}[{i}]")
            elif method == "linear":
                linear_fallback.append(f"{key}[{i}]")
            elif np.isfinite(g_nats):
                gradient_nats[f"{key}[{i}]"] = float(g_nats)
            if s.flat[i] < 0.5:
                tight.append(f"{key}[{i}]: scale={s.flat[i]:.3g}")
        scales[key] = s.reshape(val.shape)

    if diagnostics is not None:
        diagnostics["linear_fallback"] = linear_fallback
        diagnostics["flat"] = flat
        diagnostics["gradient_nats"] = gradient_nats
        diagnostics["timed_out"] = timed_out

    if tight:
        logger.debug(
            "whitening probe: tightly-constrained params: " + "; ".join(tight)
        )
    if linear_fallback:
        logger.debug(
            "whitening probe: no measurable positive curvature (h <= 0) "
            "along: "
            + "; ".join(linear_fallback)
            + " -- using the nearer one-sided contour (errs tight)."
        )
    if flat:
        logger.debug(
            f"whitening probe: logp flat to within {PROBE_TARGET_DELTA} nats "
            f"out to +/-{max_step:g} for: "
            + "; ".join(flat)
            + f" -- using scale={flat_scale}"
        )
    return map_lp, scales


# A measured multiplier at/beyond these marks was CLIPPED by one of the
# probe's two limits -- the step floor or the max step -- so the true scale
# was only bounded, not resolved.  Each mark is half the corresponding limit,
# i.e. one bracketing step inside it.  Escalation applies the clipped value
# (already a huge improvement) and re-probes just those elements in the new
# raw coordinates, where the residual error is within range again; it repeats
# until nothing is clipped, so the reach is not a hardcoded number of orders
# of magnitude on either side.  Per round that is ~14 orders down
# (_WHITEN_MIN_STEP) and ~9 up (_WHITEN_MAX_STEP); the cap below only stops a
# pathological non-converging case, and reaching it warns.
#
# There used to be a THIRD limit on the up side, and it is gone for good
# (review 1.2.9).  parameter.py cancelled a raw N(0, 1) on every logit
# element with a +0.5*raw**2 term clipped at |raw| = 1e4, which put a
# near-vertical wall in logp there; the probe found THAT wall instead of the
# posterior's contour (review 1.2.1, the ob140939 divergences) and so raised
# the clip to 1e6 for its own duration, and this mark was
# 0.5 * min(_WHITEN_MAX_STEP, 1e6).  The raw coordinate of a logit element
# is now flat (parameter.py, _raw_logp), so there is no cancellation, no
# clip and no wall: the marks derive from the probe's own step limits only.
# If a wall ever reappears in raw space it must be added here too, or the
# probe will again report it as a measured scale.
#
# What the wall did that WAS needed, and what replaces it: it guaranteed
# every logit element a contour within reach, so an element too tight by
# more than the probe's reach measured "the wall", was clipped, and
# escalated.  Without it such an element measures FLAT, which on its own
# would keep the preliminary scale and stop.  But a logit element is never
# genuinely flat: its prior alone, the Jacobian log(q(1-q)), costs 0.5 nats
# within |delta lq| ~ 1.5 of any start (whitening.md, "What flat means"),
# so a flat probe of one proves only that the scale is beyond the reach.
# _flat_logit_is_clipped therefore books it AT the reach (_WHITEN_MAX_STEP),
# which _CLIP_HI reads as clipped, and escalation proceeds exactly as the
# wall used to make it -- but now from a statement about the measurement,
# not from a term in the density.  A NON-logit element that measures flat
# keeps NaN: there flat can be real.
_CLIP_LO = 2.0 * _WHITEN_MIN_STEP
_CLIP_HI = 0.5 * _WHITEN_MAX_STEP
_MAX_ESCALATION_ROUNDS = 8


def _param_for_raw(lookup, key):
    name = key[: -len("_raw")] if key.endswith("_raw") else key
    return lookup.get(name)


def _logit_raw_mask(par):
    """Per raw-vector element: is it on the logit path?  (Raw order is the
    parameter's sampled_idx order.)"""
    tf = par._raw_transform
    return np.asarray(tf["use_logit"], dtype=bool)[tf["sampled_idx"]]


def _flat_logit_is_clipped(mult, logit_mask):
    """Book a FLAT (NaN) probe of a logit element at the probe's reach.

    See the note above _CLIP_LO: a logit element cannot be flat, so a flat
    probe means its scale lies beyond _WHITEN_MAX_STEP, and the escalation
    must be told so.  Returns a new array; non-logit NaNs are left alone.
    """
    out = np.asarray(mult, dtype=float).reshape(-1).copy()
    out[logit_mask & ~np.isfinite(out)] = _WHITEN_MAX_STEP
    return out


def _refetch_raw_start(system, model, fallback):
    """The current canonical raw start, re-read after a rescale.

    set_whitening anchors raw = 0 and re-expresses a nonzero raw_initval (a
    pre-whitening seed polish moved the start off the anchor) in the new
    coordinates, so any raw-start dict captured BEFORE a rescale is stale
    afterwards unless it was all zeros -- which it was, historically,
    making this a no-op for unpolished runs.  Systems without get_raw_start
    (test stubs) keep the fallback."""
    getter = getattr(system, "get_raw_start", None)
    if getter is None:
        return fallback
    return getter(model)


def _clipped_elements(latest, applied):
    """The (raw_name, flat_index) pairs whose latest multiplier was clipped.

    Only elements that were actually rescaled (`applied`) can be re-probed;
    a non-finite multiplier is a flat direction, which is a different report
    and not something another round can resolve.
    """
    return [
        (key, i)
        for key, mult in latest.items()
        for i in range(mult.size)
        if applied[key][i]
        and np.isfinite(mult.flat[i])
        and not (_CLIP_LO < abs(mult.flat[i]) < _CLIP_HI)
    ]


def _probe_selected(raw_start, logp_fn, elems, *, cores=1, eval_timeout=None):
    """Re-probe only the given (raw_name, flat_index) elements.

    Returns {(key, i): measured multiplier}: NaN for a flat direction,
    _TIMED_OUT for an element whose probe hit `eval_timeout`.
    """
    t0 = time.monotonic()
    map_lp = float(logp_fn(raw_start))
    t_logp = time.monotonic() - t0
    results = _probe_elements(
        raw_start,
        logp_fn,
        list(elems),
        map_lp,
        t_logp,
        dict(
            max_step=_WHITEN_MAX_STEP,
            min_step=_WHITEN_MIN_STEP,
            max_bisect=_WHITEN_MAX_BISECT,
        ),
        cores=cores,
        eval_timeout=eval_timeout,
    )
    out = {}
    for e in elems:
        res = results[e]
        if res is _TIMED_OUT:
            out[e] = _TIMED_OUT
            continue
        scale, _method, _g = res
        out[e] = scale if scale is not None else np.nan
    return out


def apply_measured_whitening(
    system, model, raw_start=None, logp_fn=None, *, cores=1, eval_timeout=None
):
    """Measure every sampled element's true raw-space scale and rescale the
    model's whitening in place.

    Probes from the relaxation-engine start (raw_start), then routes each raw
    variable's measured 0.5-nat step to its Parameter.set_whitening(), which
    updates the shared whitening scales so the same compiled model is now
    whitened by the data-driven scales.  Elements whose probe failed (flat
    both ways) keep their preliminary scale; elements whose raw N(0,1) is the
    prior -- every NON-LOGIT element, i.e. anything without two finite
    bounds, whether its width came from a sigma or from init_scale -- are
    never rescaled.

    Elements whose measured multiplier hit the probe's dynamic-range limits
    (a preliminary scale off by more than ~9 orders of magnitude too tight or
    ~14 too loose, e.g. a period constrained to nanoseconds against day-scale
    bounds; a logit element that probes FLAT counts as too tight, see
    _flat_logit_is_clipped) are escalated: the clipped correction is applied, then just those
    elements are re-probed in the new raw coordinates, where the residual
    error is resolvable.  That repeats until nothing is clipped, so the reach
    is a product of rounds rather than a fixed number of orders of magnitude;
    an element still clipped at _MAX_ESCALATION_ROUNDS gets a warning naming
    it (fix its defaults.yaml init_scale).

    ``cores`` / ``eval_timeout`` are probe_scales' (where the evaluations
    run, and the per-call wall-clock budget there).  An element whose probe
    TIMED OUT is a failed probe: it keeps its preliminary scale (the
    declared or propagated one build_pymc used), is never escalated, and a
    WARNING names it.  Its multiplier is reported as 1.0 -- the scale it
    kept -- not NaN, so the startup table does not call a slow direction
    "flat"; ``probe_diagnostics["timed_out"]`` lists it.

    Returns a report dict:
      map_lp       -- logp at the start
      multipliers  -- {raw_name: array} CUMULATIVE measured step per element
                      in preliminary-scale units (~1 means the preliminary
                      scale was already right; NaN = flat direction)
      raw_scales   -- {raw_name: array} per-element scale in the FINAL raw
                      units (1.0 for rescaled elements by construction; the
                      measured value for the deliberately-untouched
                      non-logit elements, whose raw N(0,1) IS their prior)
                      -- what PTDE's chain dispersion uses instead of
                      re-probing.
    """
    if raw_start is None:
        raw_start = system.get_raw_start(model)
    if logp_fn is None:
        logp_fn = model.compile_logp()

    n_elements = sum(v.size for v in raw_start.values())
    logger.debug(
        f"Whitening: probing {n_elements} raw element(s) for their "
        f"data-driven scales..."
    )
    # flat_scale=NaN: a flat direction keeps its preliminary scale
    # (set_whitening skips non-finite entries) and stays flagged in the
    # report so the diagnostics table can warn about it.
    probe_diag = {}
    map_lp, scales = probe_scales(
        raw_start,
        logp_fn,
        max_step=_WHITEN_MAX_STEP,
        min_step=_WHITEN_MIN_STEP,
        max_bisect=_WHITEN_MAX_BISECT,
        flat_scale=np.nan,
        diagnostics=probe_diag,
        cores=cores,
        eval_timeout=eval_timeout,
    )
    timed_out = set(probe_diag.get("timed_out", []))
    if timed_out:
        logger.warning(
            f"Whitening: {len(timed_out)} element(s) kept their preliminary "
            f"(declared or propagated) scale because a probe call exceeded "
            f"eval_timeout={float(eval_timeout):g}s: "
            + "; ".join(sorted(timed_out))
            + " -- the run continues in those units; a logp that slow at "
            "the start is worth a look (or a larger eval_timeout)."
        )

    # Loud diagnostics: neither condition invalidates the measured scales
    # (curvature is gradient-immune, and the linear fallback errs tight),
    # but both are signatures of a start far from its conditional optimum,
    # which degrades everything that uses the START itself -- chain scatter,
    # tempering, the reported start table.  ob140939's 86%-divergence run
    # would have printed ~16 gradient-dominated elements here.
    grad_dom = {
        k: g
        for k, g in probe_diag.get("gradient_nats", {}).items()
        if g > _GRAD_DOMINATED_NATS
    }
    if grad_dom:
        worst = sorted(grad_dom.items(), key=lambda kv: -kv[1])
        logger.warning(
            "Whitening: gradient-dominated start -- displaced ~N sigma from "
            "its conditional optimum along: "
            + "; ".join(f"{k} (~{g:.0f} sigma)" for k, g in worst)
            + " -- scales are curvature-measured and stay valid, but "
            "consider a polished start (seed_polish) or closer initvals."
        )
    if probe_diag.get("linear_fallback"):
        logger.warning(
            "Whitening: no measurable positive curvature (ridge/saddle/"
            "linear logp) along: "
            + "; ".join(probe_diag["linear_fallback"])
            + " -- using the nearer one-sided contour, which errs tight."
        )

    lookup = {p.label: p for p in system.get_all_parameters()}
    multipliers = {
        k: np.asarray(v, dtype=float).copy() for k, v in scales.items()
    }
    raw_scales = {}
    applied = {}  # raw_name -> bool array: element was actually rescaled
    for key, mult in scales.items():
        flat_mult = np.asarray(mult, dtype=float).reshape(-1)
        par = _param_for_raw(lookup, key)
        if par is None or getattr(par, "_whiten_state", None) is None:
            logger.warning(
                f"Whitening: no parameter found for raw variable '{key}'; "
                f"its scale stays preliminary."
            )
            raw_scales[key] = np.ones_like(flat_mult).reshape(mult.shape)
            applied[key] = np.zeros(flat_mult.size, dtype=bool)
            continue
        timed = np.array(
            [f"{key}[{i}]" in timed_out for i in range(flat_mult.size)],
            dtype=bool,
        )
        # A timed-out element is NOT a flat logit direction: nothing was
        # measured, so it is neither booked at the reach nor escalated.
        flat_mult = _flat_logit_is_clipped(
            flat_mult, _logit_raw_mask(par) & ~timed
        )
        multipliers[key] = np.where(timed, 1.0, flat_mult).reshape(
            multipliers[key].shape
        )
        post = par.set_whitening(flat_mult)
        raw_scales[key] = np.asarray(post, dtype=float).reshape(mult.shape)
        # applied: finite multiplier that landed as 1.0 in the new raw units
        applied[key] = np.isfinite(flat_mult) & (
            np.asarray(post, dtype=float) == 1.0
        )

    # The rescale above re-expressed a polished (nonzero) start in the new
    # raw coordinates; re-read it so the escalation re-probe measures
    # around the same physical point.
    raw_start = _refetch_raw_start(system, model, raw_start)

    # Escalation: re-probe elements whose LATEST measured multiplier was
    # clipped by one of the probe's limits, until none is -- or the round cap
    # is reached.  Keyed on the latest measurement, never on the cumulative
    # product: after a successful escalation the CUMULATIVE is outside the
    # window by construction (that is what "the preliminary scale was off by
    # 14 orders" means), so testing it re-probed resolved elements forever
    # and then declared them "still unresolved" (review 1.2.2).
    latest = {
        key: np.asarray(m, dtype=float).copy()
        for key, m in multipliers.items()
    }
    round_i = 0
    while True:
        clipped = _clipped_elements(latest, applied)
        if not clipped or round_i >= _MAX_ESCALATION_ROUNDS:
            break
        round_i += 1
        logger.warning(
            f"Whitening: {len(clipped)} element(s) hit the probe's dynamic "
            f"range (preliminary scale off by >9 orders of magnitude too "
            f"tight, or >14 too loose); escalation round {round_i}: "
            + "; ".join(f"{k}[{i}]" for k, i in clipped)
        )
        res = _probe_selected(
            raw_start,
            logp_fn,
            clipped,
            cores=cores,
            eval_timeout=eval_timeout,
        )
        by_param = {}
        for (key, i), m in res.items():
            if m is _TIMED_OUT:
                # A failed re-probe: the clipped correction already applied
                # stands, and the element leaves the loop (a NaN latest is
                # never "clipped").  Said by name, like the first pass.
                logger.warning(
                    f"Whitening: escalation re-probe of {key}[{i}] exceeded "
                    f"eval_timeout={float(eval_timeout):g}s; it keeps the "
                    f"correction applied so far."
                )
                latest[key].flat[i] = np.nan
                continue
            by_param.setdefault(key, {})[i] = m
        for key, elems in by_param.items():
            par = _param_for_raw(lookup, key)
            n = multipliers[key].size
            mult2 = np.ones(n)
            logit_mask = _logit_raw_mask(par)
            for i, m in elems.items():
                # A flat re-probe of a LOGIT element is still "beyond the
                # reach" (see _flat_logit_is_clipped) and escalates again.
                # A flat (NaN) re-probe of anything else leaves the
                # cumulative alone and drops the element from the loop:
                # nothing more can be measured.
                if logit_mask[i] and not np.isfinite(m):
                    m = _WHITEN_MAX_STEP
                latest[key].flat[i] = m
                if np.isfinite(m) and m > 0:
                    mult2[i] = m
                    multipliers[key].flat[i] *= m
            par.set_whitening(mult2)
        raw_start = _refetch_raw_start(system, model, raw_start)

    still = [
        f"{key}[{i}]: last measured={latest[key].flat[i]:.3g}, "
        f"cumulative={multipliers[key].flat[i]:.3g}"
        for key, i in _clipped_elements(latest, applied)
    ]
    if still:
        logger.warning(
            f"Whitening: scale still unresolved after {round_i} escalation "
            f"round(s) for: "
            + "; ".join(still)
            + " -- set a closer init_scale in the component's defaults.yaml."
        )

    logger.debug(
        f"Whitening: rescaled in place (start lp={map_lp:.1f}); every raw "
        f"unit step now costs ~{PROBE_TARGET_DELTA} nats."
    )
    return {
        "map_lp": map_lp,
        "multipliers": multipliers,
        "raw_scales": raw_scales,
        "probe_diagnostics": probe_diag,
    }


def measure_barrier_scales(system, model, raw_start):
    """Measure and set the soft-bound barrier steepness scales in place.

    For every parameter element carrying a soft-bound barrier (derived, or
    the rare half-bounded sampled element), the natural steepness scale is
    its posterior 1-sigma width.  With the model freshly whitened, one raw
    unit IS one sigma along every sampled direction, so the measured width
    of element k is the quadrature sum of its responses to unit steps:
    sqrt(sum_j (D_k(e_j) - D_k(0))^2).  This replaces the old sympy forward
    Jacobian pass: n_sampled+1 forward evaluations of the (cheap) parameter
    transform graphs, no symbolic solving, no timeouts, and it captures
    nonlinearity over the actual 1-sigma step.

    User-pinned elements (bound_scale) are respected by set_barrier_scales.
    Returns {param_label: measured_scale_vector}.
    """
    import pytensor.tensor as pt

    params = [
        p
        for p in system.get_all_parameters()
        if getattr(p, "_barrier_state", None) is not None
    ]
    if not params:
        return {}

    outs = []
    for p in params:
        v = p.value
        outs.append(pt.flatten(v) if v.ndim > 0 else pt.reshape(v, (1,)))
    fn = model.compile_fn(
        model.replace_rvs_by_values(outs),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    base = [np.asarray(b, dtype=float) for b in fn(raw_start)]
    var = [np.zeros_like(b) for b in base]
    n_steps = 0
    for key, val in raw_start.items():
        for i in range(np.asarray(val).size):
            step_pt = {k: v.copy() for k, v in raw_start.items()}
            step_pt[key].flat[i] += 1.0
            resp = fn(step_pt)
            for m, (b, r) in enumerate(zip(base, resp)):
                var[m] += (np.asarray(r, dtype=float) - b) ** 2
            n_steps += 1

    measured = {}
    for p, v in zip(params, var):
        s = np.sqrt(v)
        p.set_barrier_scales(s)
        measured[p.label] = s
    logger.debug(
        f"Whitening: measured soft-bound barrier scales for "
        f"{len(params)} parameter(s) from {n_steps} unit-step responses."
    )
    return measured


# If updating the barrier steepness moved the START's logp by more than this
# many nats, the barriers were active where the whitening probe measured its
# contours -- the measured scales are stale by up to the barrier-scale
# change.  One correction round (re-probe at ~unit scales, cheap) removes
# the first-order coupling; the residual is second-order.
_BARRIER_FEEDBACK_NATS = 0.1


def measure_and_whiten(
    system, model, raw_start=None, logp_fn=None, *, cores=1, eval_timeout=None
):
    """Full startup whitening: probe + rescale, then measured barrier
    steepness, with one correction round if the two interact.

    The whitening probe measures logp contours that INCLUDE the preliminary
    soft-bound barriers; updating the barriers afterwards therefore changes
    the surface the scales were measured on.  In the healthy case the start
    sits more than one transition width (0.01 * scale) inside every bound,
    the barrier penalty at the start is ~0 before AND after the update, and
    the feedback is exactly zero.  When the update does move the start's
    logp, the whitening is re-measured once against the final barriers
    (cheap: the model is already ~whitened, so every bracket lands within a
    few evaluations) and the barriers are re-measured with the corrected
    unit steps.

    ``cores`` / ``eval_timeout``: where the probe's logp evaluations run and
    their per-call budget (probe_scales).  ``cores=1`` (the default) is the
    serial in-process probe; run.py and the seed polish pass the sampler's
    grant and ``sampler: eval_timeout:`` (review 2.6.41).
    """
    if raw_start is None:
        raw_start = system.get_raw_start(model)
    if logp_fn is None:
        logp_fn = model.compile_logp()

    report = apply_measured_whitening(
        system,
        model,
        raw_start,
        logp_fn,
        cores=cores,
        eval_timeout=eval_timeout,
    )
    # Each rescale re-expresses a polished (nonzero) start in the new raw
    # coordinates -- re-read it before evaluating or probing there again
    # (a no-op for the historical all-zeros start).
    raw_start = _refetch_raw_start(system, model, raw_start)
    lp_before = float(logp_fn(raw_start))
    measure_barrier_scales(system, model, raw_start)
    lp_after = float(logp_fn(raw_start))

    if abs(lp_after - lp_before) > _BARRIER_FEEDBACK_NATS:
        logger.debug(
            f"Whitening: barrier update moved the start logp by "
            f"{lp_after - lp_before:+.2f} nats (a soft bound is active at "
            f"the start); re-measuring scales against the final barriers."
        )
        report2 = apply_measured_whitening(
            system,
            model,
            raw_start,
            logp_fn,
            cores=cores,
            eval_timeout=eval_timeout,
        )
        raw_start = _refetch_raw_start(system, model, raw_start)
        # Fold the correction into the cumulative multipliers for reporting.
        for key, m2 in report2["multipliers"].items():
            m1 = report["multipliers"].get(key)
            if m1 is not None:
                keep = np.isfinite(m2)
                m1[keep] = m1[keep] * m2[keep]
        report["raw_scales"] = report2["raw_scales"]
        report["map_lp"] = report2["map_lp"]
        # ...and the diagnostics, which describe the SURFACE and so are the
        # part most affected by the barrier update (review 3.2.1).  Keeping
        # round 1's flat/linear/gradient lists here showed the user the
        # pre-barrier surface in exactly the case where a soft bound is
        # active at the start, i.e. exactly when they differ.
        report["probe_diagnostics"] = report2["probe_diagnostics"]
        measure_barrier_scales(system, model, raw_start)

    return report


# Schema version of <prefix>_whitening.json.
#
# 1 -- scales only (scale_logits, gaussian_scales, barrier_scales).  Written
#      by code in which the logit ANCHOR could not move: set_whitening left
#      logit_q_inits exactly where build_pymc put it, so the anchor was
#      derivable from a rebuilt model and did not need storing.
# 2 -- adds `logit_q_inits`, the anchor, because it CAN now move:
#      System.recenter_whitening_anchor folds the polished start into it
#      before the probe (review 4.3.1).  A raw draw decodes through
#      `lower + span*sigmoid(anchor + scale*raw)`, so a moving anchor that
#      was not persisted would make a reused trace decode against the wrong
#      center -- silently, since every number involved stays plausible.
#
# Version 1 files stay READABLE (their traces are real and their anchors are
# reproducible by construction) and are never written again.  The validator
# is what enforces that a version-2 file may not omit the anchor.
_WHITENING_SCHEMA_VERSION = 2
_WHITENING_READABLE_VERSIONS = (1, 2)


def save_whitening(system, path, map_lp=None):
    """Persist the ABSOLUTE whitening + barrier state next to the trace.

    The absolute logit-space scales and the logit-space ANCHOR (not
    multipliers, not displacements) are stored so a reload reproduces the
    sampled trace's raw coordinates exactly, independent of the rebuilt
    model's preliminary scales and of where the rebuild's own anchor sits.
    """
    data = {
        "version": _WHITENING_SCHEMA_VERSION,
        "map_lp": map_lp,
        "params": {},
    }
    for p in system.get_all_parameters():
        exporter = getattr(p, "export_whitening", None)
        state = exporter() if exporter is not None else None
        if state:
            data["params"][p.label] = state
    with open(path, "w") as f:
        json.dump(data, f)
    logger.debug(f"Whitening: state saved to {path}")


def _validate_whitening_state(system, saved, lookup, version):
    """Check a persisted params mapping against the current build.

    Returns None when it applies cleanly, or a human-readable reason string.
    Every vector the apply step will touch is checked here -- both whitening
    vectors, the ANCHOR, AND the barrier vector -- so the apply step below
    cannot fail part way through and leave the model in a half-restored
    state (they are different measures: rescaling the whitening is
    posterior-preserving, but the barrier IS a posterior term, so a model
    carrying one file's barriers and another's whitening has a logp that was
    never sampled).

    ``version`` decides only one thing: whether the ANCHOR
    (``logit_q_inits``) may be absent.  In a version-1 file it MUST be --
    the code that wrote it could not move the anchor, so the rebuilt model's
    own anchor is the one its trace was sampled under and there was nothing
    to record.  In a version-2 file it must be PRESENT and the right length,
    because a version-2 writer that omitted it would be describing a
    coordinate system it did not record the center of.  Both directions are
    checked: the absent-in-v2 case is the silent one, and the
    present-in-v1 case means the file is not what its version claims.
    """
    # Coverage: every parameter that carries restorable state must appear in
    # the file.  Barrier-ONLY parameters (derived ones: no _whiten_state, a
    # soft bound) count -- their barrier steepness is a posterior term, so a
    # file that predates them describes a different logp than the one that
    # produced the trace being reused.
    stateful_now = {
        p.label
        for p in system.get_all_parameters()
        if getattr(p, "_whiten_state", None) is not None
        or getattr(p, "_barrier_state", None) is not None
    }
    missing = sorted(stateful_now - set(saved))
    if missing:
        return f"persisted state does not cover {missing}"

    for label, state in saved.items():
        par = lookup.get(label)
        if par is None:
            return f"persisted parameter '{label}' no longer exists"
        ws = getattr(par, "_whiten_state", None)
        bs = getattr(par, "_barrier_state", None)

        has_whiten_keys = "scale_logits" in state or "gaussian_scales" in state
        if has_whiten_keys:
            if ws is None:
                return f"'{label}' is no longer whitened"
            # BOTH vectors are validated: Parameter.load_whitening applies
            # them together, and checking only scale_logits let a bad
            # gaussian_scales abort mid-loop after earlier parameters had
            # already been written.
            required = [
                ("scale_logits", ws["sv_scale_logits"]),
                ("gaussian_scales", ws["sv_gaussian_scales"]),
            ]
            if version >= 2:
                required.append(("logit_q_inits", ws["sv_logit_q_inits"]))
            elif "logit_q_inits" in state:
                return (
                    f"'{label}' carries an anchor ('logit_q_inits') that a "
                    f"version-1 file cannot have written"
                )
            for key, sv in required:
                if key not in state:
                    return f"'{label}' is missing '{key}'"
                if len(state[key]) != np.asarray(sv.get_value()).size:
                    return f"'{label}' {key} does not match the model"
        elif ws is not None:
            return f"'{label}' is whitened now but was not when saved"

        if "barrier_scales" in state:
            if bs is None:
                return f"'{label}' no longer has a soft bound"
            if (
                len(state["barrier_scales"])
                != np.asarray(bs["sv"].get_value()).size
            ):
                return f"'{label}' barrier state does not match the model"
        elif bs is not None:
            return f"'{label}' has a soft bound that the persisted state omits"

    return None


def _apply_whitening_file(system, path):
    """Read, validate and apply a persisted whitening state.

    Returns ``None`` on success, or a human-readable reason the file does
    not apply.  Nothing is written to any shared variable until the whole
    file has been validated, so a rejected file leaves the build untouched
    (never half-restored).

    This is the shared body of :func:`load_whitening` (fresh path: warn and
    re-measure) and :func:`restore_whitening_for_trace` (reuse path: raise).

    A version-1 file (scales, no anchor) still applies: it was written by
    code whose anchor could not move, so the rebuilt model's build-time
    anchor is the one its trace was sampled under.  Nothing is guessed --
    the absent key is a property of that schema, enforced in both directions
    by the validator.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return f"{path} could not be read ({e})"
    version = data.get("version")
    if version not in _WHITENING_READABLE_VERSIONS:
        return f"{path} has an unknown version ({version!r})"
    saved = data.get("params", {})
    lookup = {p.label: p for p in system.get_all_parameters()}

    reason = _validate_whitening_state(system, saved, lookup, version)
    if reason is not None:
        return reason

    for label, state in saved.items():
        if not lookup[label].load_whitening(state):
            # Unreachable: the validation above covers every shape
            # load_whitening checks.  Loud rather than silent if it ever is.
            return (
                f"applying persisted state for '{label}' failed AFTER "
                f"validation; the build may be partially restored"
            )
    logger.debug(
        f"Whitening: state restored from {path} (schema version {version}, "
        f"no probe needed)."
    )
    return None


def load_whitening(system, path):
    """Apply a persisted whitening state.  Returns True on success.

    Validates EVERYTHING up front -- every persisted parameter must exist,
    every vector it carries must match the built shape, and every parameter
    that carries whitening OR barrier state now must appear in the file --
    before touching any shared variable, so a mismatch (a changed model, a
    truncated file) leaves the build untouched and the caller falls back to
    a fresh probe.

    A mismatch WARNS and re-measures rather than raising: when nothing has
    been sampled yet the whitening is a free choice of coordinates, and
    unlike foreign posterior draws it can be honestly recomputed from the
    model at load time, so there is nothing to salvage by stopping the run.

    That reasoning holds ONLY while no existing draws depend on the answer.
    A caller that is about to REUSE a saved trace must call
    :func:`restore_whitening_for_trace` instead.
    """
    reason = _apply_whitening_file(system, path)
    if reason is not None:
        logger.warning(f"Whitening: {reason}; re-measuring.")
        return False
    return True


def restore_whitening_for_trace(system, path, trace_path):
    """Restore the whitening a REUSED trace was sampled under.

    Returns ``"restored"`` when the file applied, or ``"unverifiable"`` when
    no whitening file exists (a warning is logged and the build keeps its
    PRELIMINARY scales).  Raises :class:`StaleWhiteningError` when a file
    exists but does not apply to this build.

    Why this is not ``load_whitening``'s warn-and-re-measure: a trace's raw
    draws are coordinates in the whitened space, so they only decode to the
    physical values they were sampled at under the whitening that produced
    them.  Re-measuring is a probe of the CURRENT start, which the draws
    never saw; every posterior mode label, Laplace ledger width, recomputed
    lp and model curve drawn at a stored draw would then be computed at a
    different physical point than the sampler visited, with nothing
    downstream able to notice.  This is exactly
    :class:`~exozippy.trace_meta.StaleTraceError`'s situation, and the
    asymmetry that module documents ("whitening can honestly be
    re-measured") is an argument about a run that is about to sample, not
    about one decoding draws that already exist.

    Nothing is written to ``path`` on this path, ever: overwriting the
    whitening a trace was sampled under destroys the only record of how to
    read that trace, and a re-measured file would be indistinguishable from
    a genuine one on the NEXT reload -- laundering a detected mismatch into
    a silent one.
    """
    if not os.path.exists(path):
        # Same policy as trace_meta's absent-fingerprint case: this is NOT a
        # detected mismatch.  Traces sampled with 'measure_scales: false',
        # or before whitening was persisted at all, legitimately have no
        # file -- and for the former the preliminary scales in the fresh
        # build ARE the ones it was sampled with.  Keep them: a probe of
        # the current start is not more correct than they are, merely less
        # reproducible, and it would move the soft-bound barriers (a real
        # posterior term) away from the sampled model as well.
        logger.warning(
            f"{_BANNER}\n"
            f"UNVERIFIABLE WHITENING: reusing the saved trace {trace_path}, "
            f"but no whitening state was found at {path}, so the raw draws "
            f"cannot be checked against -- or decoded under -- the "
            f"coordinates they were sampled in. This is NOT a detected "
            f"mismatch; the model keeps its PRELIMINARY scales and nothing "
            f"is re-measured or written. If the trace was sampled with "
            f"measured scales, its physical Deterministics are still "
            f"correct, but anything recomputed from the raw draws (mode "
            f"labels, the seed ledger, a recomputed lp) may not be. To be "
            f"certain, re-sample with 'sampler: {{recompute_trace: true}}'."
            f"\n{_BANNER}"
        )
        return "unverifiable"

    reason = _apply_whitening_file(system, path)
    if reason is None:
        return "restored"

    raise StaleWhiteningError(
        f"{_BANNER}\n"
        f"STALE WHITENING: {path} does not describe the model this config "
        f"builds, and the saved trace {trace_path} is being REUSED "
        f"('sampler: {{recompute_trace: false}}').\n"
        f"  why it does not apply: {reason}\n"
        f"A trace's raw draws are coordinates in the whitened space: they "
        f"only decode to the physical values the sampler visited under the "
        f"whitening they were sampled with. Re-measuring it here would "
        f"probe the CURRENT start instead -- silently changing what every "
        f"stored draw means -- and would overwrite the only record of how "
        f"to read this trace. The draws cannot be repaired at load time.\n"
        f"REMEDY: re-sample with 'sampler: {{recompute_trace: true}}' in the "
        f"config, or restore the {path} that was written alongside "
        f"{trace_path} (and revert the model edit that invalidated it).\n"
        f"{_BANNER}"
    )


def prepare_whitening(
    system,
    model,
    raw_start,
    path,
    trace_path,
    *,
    reusing_trace,
    cores=1,
    eval_timeout=None,
):
    """Put the model into the whitened coordinates this run must use.

    The two situations this serves look alike and are not:

    * ``reusing_trace=False`` -- about to SAMPLE.  The whitening is a free
      choice of coordinates, so it is measured from the data and persisted
      to ``path``.  Whatever was there before is superseded, because the
      draws it described are about to be superseded too.  Returns the probe
      report (the caller must re-read its raw start: the rescale re-expressed
      it in the new coordinates).
    * ``reusing_trace=True`` -- about to DECODE draws that already exist.
      The whitening is then a property of those draws, not a choice, so it
      is restored and never re-measured or rewritten (see
      :func:`restore_whitening_for_trace`; a mismatch raises).  Returns
      ``None`` -- there is no probe and the raw start did not move.

    ``cores`` / ``eval_timeout`` reach the probe (measure_and_whiten): run.py
    passes the sampler's core grant and ``sampler: eval_timeout:``.
    """
    if reusing_trace:
        restore_whitening_for_trace(system, path, trace_path)
        return None
    report = measure_and_whiten(
        system, model, raw_start, cores=cores, eval_timeout=eval_timeout
    )
    save_whitening(system, path, map_lp=report["map_lp"])
    return report
