"""
Asynchronous (non-blocking) Parallel Tempering + Differential Evolution sampler.

The production default for Op-based (non-differentiable) models (YAML:
sampler.method: "ptde_async"). Kept as a separate sampler module so the
synchronous PTDE in ptde.py -- the reference implementation with fully
up-to-date DE partner states -- stays available for A/B validation. The
non-sampling scaffolding both share lives in exozippy.samplers._common.

MOTIVATION: ptde.py's synchronous design must wait for the SLOWEST of all
n_temps*n_chains proposals before ANY chain can advance to its next step,
because every chain's next DE proposal needs its rung-mates' CURRENT states,
and "current" is only well-defined once the whole step resolves. Production
runs on examples/DC2018_128 show this stalls the entire sampler behind a rare
but expensive near-caustic evaluation concentrated in the hottest 1-2 rungs
(the per-rung timing table at the top of notes/hpc_optimization.txt:
0.09%/0.7% of rung 6/7 calls exceed 0.1 s, and with 320 proposals/step the
odds NONE of them land in that tail across a whole run are essentially zero).
This module removes that
barrier: every (rung, chain) slot is its own continuous pipeline against a
shared worker pool -- as soon as one slot's evaluation resolves, it is
accepted/rejected and that SAME slot's next proposal is immediately
resubmitted, without waiting for any other slot. A slow evaluation ties up
only the one worker running it; every other core keeps advancing its own
chain.

STATISTICAL CAVEAT (read before trusting output): a chain's DE proposal is
built from whatever states its two partner chains CURRENTLY hold, which may
be "last accepted" rather than "just updated" if a partner's own evaluation
is still in flight elsewhere. This kind of stale-partner DE move is used in
the async-ensemble-MCMC literature, but it changes the move's detailed-
balance argument relative to the synchronous version in ptde.py, which always
uses each rung's fully up-to-date population. Swap moves are NOT affected by
this staleness concern -- logp(x) does not depend on when x was last
computed, only on x itself, and a swap always compares one (state, logp) pair
against another self-consistent pair. A swap CAN, however, replace a slot's
state while that slot's own DE proposal is in flight; accepting/rejecting
that proposal against the swapped-in state would violate detailed balance,
so every submission is stamped with its slot's state generation and a result
whose generation no longer matches (the state was swapped away underneath
it) is DISCARDED without an accept/reject transition and the slot resubmits
from its new state (the discard decision depends only on the swap event,
never on the proposal's logp, so it is a valid "do nothing" kernel mixture).
Before trusting a production posterior from this sampler on a new class of
model, validate against a synchronous PTDE run on the same model (see
tests/test_ptde_async.py for the toy-Gaussian recovery check this module
ships with).

Storage differs from ptde.py because chains are no longer synchronized on a
shared step counter: each T=1 chain accumulates its own `draws` post-tune
samples at its own pace (per-chain iteration counters `iter_count`,
per-chain draw counters `per_chain_draws`). A chain that reaches its target
`draws` count keeps evolving (so it does not go stale as a DE/swap partner
for chains still catching up) but stops being recorded. The run stops once
every chain has recorded `draws` samples, or maxtime/convergence/user-
interrupt fires first; at that point output is truncated to
min(per_chain_draws) across chains (the simplest correct option: every
chain then contributes the same number of draws, and no chain contributes
draws its peers have not caught up to).
"""

import logging
import queue
import signal
import time

import numpy as np

from exozippy.samplers import _common
from exozippy.samplers._common import (
    DE_JITTER,
    LpPlausibilityGuard,
    _eval_logp,
    next_gamma,
)
from exozippy.samplers.ptde import (
    _check_convergence,
    _convergence_check_schedule,
    _deo_pair_sequence,
    _geometric_ladder,
    _record_round_trips,
    _safe_progress,
    _update_ladder_barrier,
    ladder_health_report,
    resolve_n_temps,
)

logger = logging.getLogger(__name__)


def ptde_async_sample(
    model,
    system,
    draws,
    tune,
    *,
    n_temps=8,
    T_max=200.0,
    n_chains=None,
    cores=None,
    initvals=None,
    raw_starts=None,
    seed_indices=None,
    raw_scales=None,
    gamma=None,
    target_accept=0.20,
    adapt_gamma=True,
    gamma_adapt_window=None,
    adapt_ladder=False,
    de_jitter=DE_JITTER,
    swap_interval=None,
    swap_schedule="deo",
    seed=None,
    log_interval=None,
    plot_prefix=None,
    min_ess=1000,
    max_rhat=1.01,
    maxtime=None,
    eval_timeout=None,
    lp_plausibility_ceiling=None,
    collect_rung_timing=False,
    progress_callback=None,
    store_hot_chains="auto",
):
    """
    Asynchronous Parallel Tempering + Differential Evolution sampler.

    See the module docstring for the motivation and the statistical caveat
    around stale DE partners. Parameters mirror exozippy.samplers.ptde.
    ptde_sample where the meaning is identical; a few differ because chains
    are no longer synchronized on a shared step counter:

    swap_interval : int | None -- attempt one swap every `swap_interval`
               completed evaluations (any rung). None -> n_chains, chosen so
               the long-run ratio of evaluations-per-swap-attempt roughly
               matches ptde.py's one-full-round-of-swaps-per-step cadence.
    swap_schedule : {"deo", "random"} -- "deo" (default) walks the adjacent
               rung pairs in a deterministic non-reversible cycling order
               (even pairs (0,1),(2,3),... first, then odd pairs (1,2),(3,4),
               ..., repeating; see ptde._deo_pair_sequence), the event-time
               analog of the synchronous DEO schedule. "random" restores the
               legacy random-adjacent-pair draw for A/B comparison. The chains
               swapped within the chosen pair are random either way, and each
               swap is the identical MH test, so invariance is untouched.
    adapt_ladder : bool -- re-space the ladder during tuning to equalize the
               per-pair communication barrier (Syed et al. 2022). Default
               False, matching the synchronous sampler. Turn it on when the
               per-rung swap acceptances are non-uniform: a round trip must
               cross every pair, so transport is throttled by the WORST
               stretch, and a geometric ladder cannot fix that by getting
               longer (measured on DC2018 event 128 -- see the block that
               calls _update_ladder_barrier below).
    gamma_adapt_window : int | None -- adapt gamma once per this many
               completed T=1 proposals still within their own chain's tune
               phase. None -> max(n_chains, (tune * n_chains) // 20), i.e.
               roughly 20 adaptations over tune, matching ptde.py's cadence.
               Adaptation FREEZES the moment the first T=1 chain enters its
               draw phase: chains record at their own pace here, and a
               recorded draw must never come from a kernel a slower chain's
               tune-phase proposals are still mutating.
    log_interval : int | None -- steps between progress log lines, where one
               async "step" is n_slots (= n_temps x n_chains) completed
               evaluations -- the same amount of work as one ptde.py step, so
               the knob means the same thing for both samplers. None -> 5%.
    lp_plausibility_ceiling : float | None -- same runaway-lp warning as
               ptde.py (None -> outputs.modes.DEFAULT_LP_ABS_MAX).
    store_hot_chains : "auto" | False | True | int -- keep THINNED draws from
               the hot rungs (T > 1) as an extra ``posterior_hot`` group on
               the returned InferenceData: every int-th post-tune iteration of
               each hot chain (True -> 20), with its UNtempered logp and a
               per-chain ``temperature`` coordinate. "auto" (the default)
               decides from the TOPOLOGY -- on when any active component
               declares ``expects_suppressed_modes`` (today the microlensing
               lens), off otherwise; see
               _common.resolve_store_hot_chains, which also logs the decision
               and the resulting trace-size cost. Hot draws are not
               posterior draws -- a T-tempered mode is ~sqrt(T) too wide --
               but they visit posterior-suppressed basins (occupancy
               ~exp(-dlp/T)) that the T=1 chains abandon, so they are the
               mode DETECTOR for outputs.ledger.discover_hot_modes, which
               polishes and Laplace-characterizes anything they find.
               Memory: n_raw_elements x (n_temps-1) x n_chains x
               (draws/thin) float64.

    Returns
    -------
    arviz.InferenceData with posterior and sample_stats["lp"] from T=1 chains.
    """
    if swap_schedule not in ("deo", "random"):
        raise ValueError(
            f"swap_schedule must be 'deo' or 'random', got {swap_schedule!r}"
        )

    lp_guard = LpPlausibilityGuard(
        lp_plausibility_ceiling, "PTDE-async", logger
    )

    rng = np.random.default_rng(seed)

    # parameter bookkeeping -- before the ladder, since n_temps may be
    # "auto" (sized from the parameter count; see ptde.resolve_n_temps).
    raw_start = system.get_raw_start(model)
    model_keys = list(raw_start.keys())
    n_params = sum(v.size for v in raw_start.values())
    n_temps = resolve_n_temps(n_temps, n_params, T_max)
    temperatures = _geometric_ladder(n_temps, T_max)

    # compile logp ONCE; install in _common BEFORE forking workers so fork
    # children inherit it (copy-on-write; see _common.set_worker_globals)
    # PositionalLogp calls the compiled function by position instead of
    # through pymc's dict wrapper -- same value, ~3.5x less call overhead on
    # a 20-variable model, which at one call per proposal is the difference
    # between 35 us and 10 us of pure plumbing per evaluation (6.4.3).
    logp_fn = _common.PositionalLogp(model.compile_logp())
    _common.set_worker_globals(logp_fn, collect_rung_timing)

    # compile raw -> physical conversions ONCE (single-sample and batched;
    # see _common.compile_conversions for the rationale).
    raw_to_phys, raw_to_phys_batched, raw_var_names, out_var_names = (
        _common.compile_conversions(model)
    )
    # Element count of the PHYSICAL side (sampled + Deterministic), the
    # denominator of the hot-group trace share below.  One transform
    # evaluation at the start point; the graph is compiled either way.
    _n_out_elements = sum(
        int(np.asarray(v).size)
        for v in raw_to_phys(
            *[np.asarray(raw_start[k]) for k in raw_var_names]
        )
    )

    n_chains = _common.resolve_n_chains(
        n_chains, n_params, "PTDE-async", logger
    )
    if gamma is None:
        gamma = 2.38 / np.sqrt(2 * n_params)
    logger.info(
        f"PTDE-async: {n_params} params, {n_chains} chains/rung, gamma={gamma:.4f}"
    )

    t1_starts, chain_seed_index = _common.resolve_start_population(
        model,
        system,
        n_chains,
        logp_fn,
        rng,
        raw_start,
        initvals=initvals,
        raw_starts=raw_starts,
        seed_indices=seed_indices,
        raw_scales=raw_scales,
    )

    _common.plot_start_ensemble(
        system,
        t1_starts,
        raw_to_phys_batched,
        raw_var_names,
        out_var_names,
        plot_prefix,
        logger,
    )

    # Per-(rung, chain) slot state. current_lp[k][i] is None until that
    # slot's first evaluation completes -- doubles as "still initializing".
    # A rung's states are ONE (n_chains, n_raw_elements) array rather than a
    # list of dicts, so the DE move is three vector operations instead of a
    # Python loop over the free RVs; _common.RawLayout owns the packing and
    # the proof that it is bit-identical (review 6.4.2).
    layout = _common.RawLayout(raw_start, model_keys)
    current_state = [
        layout.pack_many([t1_starts[i % n_chains] for i in range(n_chains)])
        for _ in range(n_temps)
    ]
    current_lp = [[None] * n_chains for _ in range(n_temps)]
    iter_count = [[0] * n_chains for _ in range(n_temps)]
    # state_gen[k][i] counts the swaps that have replaced slot (k, i)'s
    # state. Every submission is stamped with it; a result whose stamp no
    # longer matches was proposed FROM a state that a swap has since moved
    # elsewhere, so accept/rejecting it would compare apples to oranges
    # (the detailed-balance violation of code_review_20260808.txt 1.15a).
    # Such results are discarded and the slot resubmits from its new state.
    state_gen = [[0] * n_chains for _ in range(n_temps)]

    slot_list = [(k, i) for k in range(n_temps) for i in range(n_chains)]
    n_slots = len(slot_list)

    pool, actual_cores = _common.create_pool(
        cores, n_slots, "PTDE-async", logger
    )
    logger.info(
        f"PTDE-async: {n_temps} rungs x {n_chains} chains = {n_slots} slots, "
        f"{actual_cores} cores  "
        f"T=[{', '.join(f'{t:.1f}' for t in temperatures)}]"
    )
    _common.warn_serial_eval_timeout(
        eval_timeout, pool, actual_cores, "PTDE-async", logger
    )

    swap_interval = (
        max(1, int(swap_interval)) if swap_interval else max(1, n_chains)
    )
    gamma_adapt_window = (
        int(gamma_adapt_window)
        if gamma_adapt_window
        else max(n_chains, (tune * n_chains) // 20)
    )
    # One async "step" = n_slots completed evaluations, the work of one
    # synchronous ptde.py step -- so log_interval means the same thing in
    # both samplers' configs.
    log_every_evals = (
        max(1, int(log_interval)) * n_slots
        if log_interval
        else max(n_slots, (n_slots * (tune + draws)) // 20)
    )

    # storage: raw values from T=1 chains only; each chain records exactly
    # `draws` samples (its own post-tune iterations), so no dynamic growth
    # is needed -- capacity is a hard per-chain cap by construction.
    stored_raw = {
        k: np.zeros((n_chains, draws) + raw_start[k].shape) for k in model_keys
    }
    stored_lp = np.zeros((n_chains, draws))
    per_chain_draws = np.zeros(n_chains, dtype=int)

    # Optional thinned hot-rung storage (store_hot_chains): detector data
    # for post-hoc discovery of posterior-suppressed modes; see the
    # parameter docstring. UNtempered logp is stored (current_lp holds the
    # raw logp; tempering happens in the acceptance rule), so hot lp values
    # are directly comparable to T=1.
    hot_thin = _common.resolve_store_hot_chains(
        store_hot_chains,
        system,
        n_temps,
        n_params,
        _n_out_elements,
        "PTDE-async",
        logger,
    )
    hot_cap = max(1, draws // hot_thin) if hot_thin else 0
    if hot_thin:
        stored_hot_raw = {
            k: np.zeros((n_temps - 1, n_chains, hot_cap) + raw_start[k].shape)
            for k in model_keys
        }
        stored_hot_lp = np.full((n_temps - 1, n_chains, hot_cap), np.nan)
        per_hot_draws = np.zeros((n_temps - 1, n_chains), dtype=int)

    n_accept = np.zeros(n_temps)
    n_propose = np.zeros(n_temps)
    n_swap_accept = np.zeros(max(n_temps - 1, 1))
    n_swap_propose = np.zeros(max(n_temps - 1, 1))
    n_eval_timeouts = [0]
    n_swap_discards = [0]  # in-flight proposals invalidated by a swap
    rung_times = [[] for _ in range(n_temps)]

    # DEO schedule + round-trip diagnostics. direction[k][i] tags the config
    # in slot (k, i) with the last extreme rung it visited; tags travel with
    # the state through swaps (see ptde._record_round_trips). _deo_seq is the
    # deterministic cycling order of adjacent-pair lower indices; _deo_pos
    # walks it one swap event at a time.
    direction = [[0] * n_chains for _ in range(n_temps)]
    round_trips = [0]
    n_swap_rounds = [0]
    _deo_seq = _deo_pair_sequence(n_temps)
    _deo_pos = [0]

    gamma_box = [gamma]
    gamma_frozen = [False]
    n_propose_T1_window = [0]
    n_accept_T1_window = [0]
    n_completed_total = [0]

    stop_requested = [False]
    start_time = time.time()

    def _stop_handler(sig, frame):
        if stop_requested[0]:
            raise KeyboardInterrupt
        stop_requested[0] = True
        logger.info(
            f"PTDE-async: stop requested ({signal.Signals(sig).name}) — finishing "
            "in-flight evaluations (send the signal again to abort immediately)"
        )

    _sig_token = _common.install_stop_handlers(_stop_handler)

    result_q = queue.Queue()
    # One record per live submission, keyed by a unique submission id:
    # sub_id -> (k, i, prop, t_submitted, state_gen_at_submit). A result
    # whose sub_id is no longer here was written off by the eval-timeout
    # recovery below; if it arrives anyway (it raced the write-off through
    # the queue) it is dropped on the floor, so one submission can never be
    # double-processed (code_review_20260808.txt 1.15b).
    in_flight_meta = {}
    in_flight = [0]
    _sub_seq = [0]

    def _build_proposal(k, i):
        """The slot's next PACKED proposal (see _common.RawLayout)."""
        if current_lp[k][i] is None:
            # First evaluation for this slot: evaluate the start state itself.
            return current_state[k][i].copy()
        return layout.propose(
            rng, current_state[k], i, gamma_box[0], jitter=de_jitter
        )

    def _submit(k, i):
        # The slot keeps the packed vector; the worker is handed the dict the
        # compiled logp wants.
        prop = _build_proposal(k, i)
        payload = layout.unpack(prop)
        _sub_seq[0] += 1
        sub_id = _sub_seq[0]
        in_flight_meta[sub_id] = (k, i, prop, time.time(), state_gen[k][i])
        in_flight[0] += 1
        if pool is None:
            # Serial fallback: no real concurrency, evaluate immediately.
            result_q.put((sub_id, _eval_logp(payload)))
            return

        def _cb(result, sub_id=sub_id):
            result_q.put((sub_id, result))

        def _ecb(exc, sub_id=sub_id, k=k, i=i):
            logger.error(
                f"PTDE-async: worker exception at rung {k} chain {i}: {exc}"
            )
            failure = (-np.inf, 0.0) if collect_rung_timing else -np.inf
            result_q.put((sub_id, failure))

        pool.apply_async(
            _eval_logp, (payload,), callback=_cb, error_callback=_ecb
        )

    def _attempt_swap():
        if n_temps <= 1:
            return
        if swap_schedule == "deo":
            # Deterministic cycling over adjacent pairs (event-time DEO); only
            # the pair-SELECTION order differs from the random schedule, the
            # MH test below is identical, so invariance is untouched.
            k = _deo_seq[_deo_pos[0] % len(_deo_seq)]
            _deo_pos[0] += 1
        else:
            k = int(rng.integers(n_temps - 1))
        i = int(rng.integers(n_chains))
        j = int(rng.integers(n_chains))
        lp_i, lp_j = current_lp[k][i], current_lp[k + 1][j]
        if lp_i is None or lp_j is None:
            return  # one side hasn't completed its first evaluation yet
        n_swap_propose[k] += 1
        log_a = (lp_j - lp_i) * (
            1.0 / temperatures[k] - 1.0 / temperatures[k + 1]
        )
        if rng.random() < np.exp(min(0.0, log_a)):
            # .copy() first: these are numpy ROWS, so the tuple form would
            # hold views and the second assignment would read the first back.
            _tmp = current_state[k][i].copy()
            current_state[k][i] = current_state[k + 1][j]
            current_state[k + 1][j] = _tmp
            current_lp[k][i], current_lp[k + 1][j] = lp_j, lp_i
            direction[k][i], direction[k + 1][j] = (
                direction[k + 1][j],
                direction[k][i],
            )
            # Invalidate any in-flight DE proposal generated from the states
            # that just moved: its accept/reject would otherwise run against
            # the swapped-in state (1.15a; see state_gen above).
            state_gen[k][i] += 1
            state_gen[k + 1][j] += 1
            n_swap_accept[k] += 1
        # Update round-trip tags after every swap event (idempotent): a config
        # sitting at either extreme rung is tagged accordingly and a completed
        # cold -> hot -> cold excursion is counted.
        _record_round_trips(direction, round_trips, n_temps)
        n_swap_rounds[0] += 1

    _do_convergence = (
        min_ess is not None or max_rhat is not None
    ) and n_chains >= 2
    _check_gen = _convergence_check_schedule() if _do_convergence else None
    _next_check = [next(_check_gen)] if _check_gen else [None]

    stopping = [False]
    stop_reason = [None]  # human-readable, for logging
    stop_category = [None]  # "abort" (user/maxtime) vs "complete" -- used
    # below to decide whether an empty run should
    # raise KeyboardInterrupt (user/time abort) or
    # just fall through to the "no draws" RuntimeError.

    def _maybe_stop():
        if stopping[0]:
            return
        if stop_requested[0]:
            stopping[0] = True
            stop_reason[0] = "user interrupt"
            stop_category[0] = "abort"
        elif maxtime is not None and (time.time() - start_time) > maxtime:
            stopping[0] = True
            stop_reason[0] = f"maxtime {maxtime:.0f}s reached"
            stop_category[0] = "abort"
        elif int(per_chain_draws.min()) >= draws:
            stopping[0] = True
            stop_reason[0] = "draws target reached"
            stop_category[0] = "complete"
        elif (
            _do_convergence
            and _next_check[0] is not None
            and int(per_chain_draws.min()) >= _next_check[0]
        ):
            n_check = int(per_chain_draws.min())
            converged, rhat_val, ess_val = _check_convergence(
                stored_raw, n_check, min_ess, max_rhat, stored_lp
            )
            logger.info(
                f"PTDE-async convergence @ min {n_check} draws/chain: "
                f"max_rhat={rhat_val:.4f}  min_ess={ess_val:.1f}"
            )
            # GUI progress hook (bounded: fires once per geometric check).
            # Passes the live T=1 draw buffers by reference for snapshotting.
            _safe_progress(
                progress_callback,
                {
                    "n_draws": n_check,
                    "n_chains": n_chains,
                    "max_rhat": rhat_val,
                    "min_ess": ess_val,
                    "elapsed_s": time.time() - start_time,
                    "stop_reason": "converged" if converged else None,
                    "stored_raw": stored_raw,
                    "stored_lp": stored_lp,
                    "raw_var_names": model_keys,
                },
            )
            _next_check[0] = next(_check_gen, None)
            if converged:
                stopping[0] = True
                stop_reason[0] = "convergence criterion met"
                stop_category[0] = "complete"

    # How often the stale scan below may run.  It is O(in-flight), so it is
    # paced by a WALL CLOCK rather than by a completion count: the run's
    # completion rate spans microseconds (a toy model) to seconds (a
    # near-caustic binary lens), and either extreme would make a
    # "scan every N results" rule either useless or expensive.
    timeout_scan_interval = (
        max(eval_timeout / 4.0, 0.05) if eval_timeout is not None else None
    )
    next_timeout_scan = [np.inf]
    if timeout_scan_interval is not None:
        next_timeout_scan[0] = time.time() + timeout_scan_interval

    # The main loop's blocking wait is ALWAYS bounded, even with no
    # eval_timeout (the default).  It used to be `timeout=None` there, so
    # the loop sat in result_q.get() until a result happened to arrive and
    # _maybe_stop ran only on the result path: a Ctrl+C, a SIGTERM from the
    # batch scheduler, or maxtime expiring were honored only when some slot
    # finished.  On a model where every remaining evaluation is slow that is
    # exactly when nothing finishes, so the user sent a SECOND signal, which
    # raises KeyboardInterrupt out of the handler and abandons every draw
    # already collected -- the abort save path is never reached (2.4.3).
    # One wakeup a second costs nothing; missing the graceful stop costs the
    # run.  A shorter eval_timeout tightens it further, since the stale scan
    # has to keep up too.
    POLL_CEILING = 1.0
    poll_timeout = (
        min(POLL_CEILING, timeout_scan_interval)
        if timeout_scan_interval is not None
        else POLL_CEILING
    )

    def _enforce_eval_timeout():
        """Write off every in-flight submission once one has gone stale.

        There is no way to kill a single hung worker in a
        multiprocessing.Pool without tearing down the whole pool (same
        limitation as ptde.py's _map_logp_timeout), so any OTHER
        legitimately-still-running slot is also abandoned and immediately
        resubmitted with a fresh proposal.

        THIS MUST BE REACHABLE WHILE THE QUEUE IS BUSY.  It used to live
        exclusively inside `except queue.Empty`, which is a state a healthy
        run essentially never reaches: with hundreds of slots resolving in
        milliseconds there is always another result waiting, so a genuinely
        hung logp -- the exact scenario eval_timeout exists for -- froze its
        slot for the whole run.  A frozen T=1 slot is worse than a lost
        proposal: its chain stops recording, so min(per_chain_draws) never
        reaches `draws`, the convergence checks stop firing, and only
        maxtime or Ctrl+C ends the run (review 1.4.1).

        No-op without a pool: in serial mode the evaluation happens inline
        in _submit, so every in_flight_meta entry is already COMPLETE and
        merely waiting to be read off the queue -- scanning their submission
        times would write off finished work and then try to recycle a pool
        that does not exist.  warn_serial_eval_timeout says so at startup.

        Returns True when anything was written off.
        """
        nonlocal pool
        if eval_timeout is None or pool is None:
            return False
        now = time.time()
        stale = [
            sid
            for sid, (_, _, _, t0, _) in in_flight_meta.items()
            if now - t0 > eval_timeout
        ]
        if not stale:
            return False
        n_eval_timeouts[0] += len(stale)
        for sid in stale:
            sk, si, stale_prop, _, _ = in_flight_meta[sid]
            phys_params, raw_params = _common.describe_proposal(
                layout.unpack(stale_prop),
                raw_to_phys,
                raw_var_names,
                out_var_names,
            )
            logger.error(
                f"PTDE-async: logp call exceeded "
                f"eval_timeout={eval_timeout:.0f}s at rung {sk} "
                f"chain {si} — rejecting this proposal.\n"
                f"  physical params: {phys_params}\n"
                f"  raw params: {raw_params}"
            )
        # Write off EVERY in-flight submission (the pool recycle kills the
        # workers running them). A written-off result that nevertheless
        # arrives (it raced us through the queue) finds its sub_id gone and
        # is dropped, so it can never be double-processed (1.15b).
        lost = list(in_flight_meta.items())
        in_flight_meta.clear()
        in_flight[0] -= len(lost)
        logger.warning(
            f"PTDE-async: recycling worker pool ({len(stale)} timeout(s))"
        )
        pool = _common.recycle_pool(pool, actual_cores)
        if not stopping[0]:
            for _, (sk, si, _, _, _) in lost:
                _submit(sk, si)
        return True

    try:
        for k, i in slot_list:
            _submit(k, i)

        while in_flight[0] > 0:
            try:
                sub_id, result = result_q.get(timeout=poll_timeout)
            except queue.Empty:
                # _enforce_eval_timeout is a no-op when eval_timeout is
                # unset, which is now a reachable state: the poll is bounded
                # either way (see POLL_CEILING).  _maybe_stop is what the
                # bounded poll is FOR -- on a quiet queue it is the only
                # place a user signal or maxtime can be noticed.  The draws
                # and convergence conditions it also tests cannot have
                # changed since the last result, since every result path
                # calls it too.
                _enforce_eval_timeout()
                if timeout_scan_interval is not None:
                    next_timeout_scan[0] = time.time() + timeout_scan_interval
                _maybe_stop()
                if stop_category[0] == "abort":
                    # Nothing completed within a whole poll, and the user (or
                    # maxtime) has asked to stop: the remaining evaluations
                    # are not coming back on any useful timescale, so stop
                    # waiting on them and go save what is already stored.
                    # The loop's normal exit is the RESULT path draining
                    # in_flight to zero, which never happens when every
                    # in-flight evaluation is wedged -- the state in which
                    # the user reaches for a second Ctrl+C, and a second
                    # signal raises straight out of the handler and throws
                    # the collected draws away.  In-flight proposals carry
                    # nothing that was recorded; the pool is torn down in
                    # the `finally`.
                    logger.info(
                        f"PTDE-async: {stop_reason[0]} — abandoning "
                        f"{in_flight[0]} in-flight evaluation(s) that did "
                        f"not return, and saving what is stored"
                    )
                    break
                continue

            meta = in_flight_meta.pop(sub_id, None)
            if meta is None:
                # Already written off by the timeout recovery above; the
                # bookkeeping (in_flight, resubmission) happened there.
                continue
            k, i, prop, _, gen_at_submit = meta
            in_flight[0] -= 1

            # Same scan on the RESULT path, paced by the wall clock so a
            # busy queue cannot hide a hung slot (review 1.4.1).  It runs
            # AFTER the result in hand has been taken off the books, so a
            # write-off never discards an evaluation that already finished.
            # A scan that finds nothing stale touches no state and draws no
            # random numbers, so a run in which no evaluation ever times out
            # is bit-for-bit the run this sampler produced before.
            if timeout_scan_interval is not None:
                now = time.time()
                if now >= next_timeout_scan[0]:
                    next_timeout_scan[0] = now + timeout_scan_interval
                    _enforce_eval_timeout()

            if collect_rung_timing:
                lp, elapsed = result
                rung_times[k].append(elapsed)
            else:
                lp = result

            if gen_at_submit != state_gen[k][i]:
                # A swap replaced this slot's state while the proposal was in
                # flight; the proposal no longer has a valid reference state
                # for an accept/reject. Discard it (1.15a) and resubmit from
                # the swapped-in state below. No counters advance: no MH
                # transition happened for this slot.
                n_swap_discards[0] += 1
            elif current_lp[k][i] is None:
                # First evaluation for this slot: the start state itself.
                current_state[k][i] = prop
                current_lp[k][i] = lp
            else:
                T = temperatures[k]
                n_propose[k] += 1
                accepted = np.isfinite(lp) and rng.random() < np.exp(
                    min(0.0, (lp - current_lp[k][i]) / T)
                )
                if k == 0 and iter_count[k][i] < tune:
                    n_propose_T1_window[0] += 1
                    if accepted:
                        n_accept_T1_window[0] += 1
                if accepted:
                    current_state[k][i] = prop
                    current_lp[k][i] = lp
                    n_accept[k] += 1
                    # Runaway-lp early detection (see LpPlausibilityGuard).
                    if k == 0:
                        lp_guard.check(i, lp)
                iter_count[k][i] += 1

                # Freeze gamma the moment the first T=1 chain finishes its
                # tune phase: from here on some chain may be RECORDING, and a
                # recorded draw must come from a fixed kernel -- slower
                # chains' tune-phase proposals must not keep mutating
                # gamma_box under it (1.15c). Their remaining tune iterations
                # simply run at the frozen gamma.
                if k == 0 and not gamma_frozen[0] and iter_count[k][i] >= tune:
                    gamma_frozen[0] = True
                    if adapt_gamma:
                        logger.info(
                            f"PTDE-async gamma: frozen at {gamma_box[0]:.4f} "
                            f"(chain {i} entered its draw phase)"
                        )

                # store T=1 post-tune draws (each chain caps at `draws`)
                if (
                    k == 0
                    and iter_count[k][i] > tune
                    and per_chain_draws[i] < draws
                ):
                    d = per_chain_draws[i]
                    layout.store_draw(stored_raw, current_state[k][i], i, d)
                    stored_lp[i, d] = current_lp[k][i]
                    per_chain_draws[i] = d + 1

                # thinned hot-rung storage (see store_hot_chains)
                if (
                    hot_thin
                    and k >= 1
                    and iter_count[k][i] > tune
                    and iter_count[k][i] % hot_thin == 0
                    and per_hot_draws[k - 1, i] < hot_cap
                ):
                    d = per_hot_draws[k - 1, i]
                    layout.store_draw(
                        stored_hot_raw, current_state[k][i], k - 1, i, d
                    )
                    stored_hot_lp[k - 1, i, d] = current_lp[k][i]
                    per_hot_draws[k - 1, i] = d + 1

            n_completed_total[0] += 1

            if n_completed_total[0] % swap_interval == 0:
                _attempt_swap()

            if (
                adapt_gamma
                and not gamma_frozen[0]
                and n_propose_T1_window[0] >= gamma_adapt_window
            ):
                ar_T1 = n_accept_T1_window[0] / max(n_propose_T1_window[0], 1)
                # Skipped rather than shrunk when the window accepted
                # nothing, exactly as in ptde.py; _common.next_gamma owns
                # the rule itself.
                if ar_T1 > 0:
                    gamma_new = next_gamma(gamma_box[0], ar_T1, target_accept)
                    if abs(gamma_new - gamma_box[0]) / gamma_box[0] > 0.01:
                        logger.info(
                            f"PTDE-async gamma: {gamma_box[0]:.4f} -> {gamma_new:.4f} "
                            f"(T=1 accept={ar_T1:.3f}, target={target_accept:.2f})"
                        )
                        gamma_box[0] = gamma_new
                n_propose_T1_window[0] = 0
                n_accept_T1_window[0] = 0

            # Communication-barrier ladder adaptation (Syed et al. 2022),
            # ported from the synchronous sampler, which had it and this one
            # did not -- so until now `ptde_async` could not reshape its
            # ladder AT ALL, only lengthen it via n_temps.
            #
            # Why it matters, measured on examples/DC2018 event 128 at
            # T_max=8500 with n_temps=48 (correctly provisioned: Lambda=18.9
            # against the 39 rungs the DEO criterion asks for, mean swap
            # acceptance 0.598 at its 0.5 target): swap acceptance was
            # strongly NON-uniform, 0.46-0.52 across the cold pairs against
            # 0.66-0.70 hot, and round trips stayed at ZERO for 21 hours.
            # The barrier concentrates where the posterior is sharp, while a
            # geometric ladder spaces rungs uniformly in ln T -- so it
            # over-resolves the easy hot end and under-resolves the stretch
            # that actually throttles a traversal.  A round trip must cross
            # EVERY pair, so the rate is set by the worst stretch, not the
            # mean, which is why adding rungs uniformly did not help.
            #
            # Gated on `not gamma_frozen[0]`, the same flag gamma uses: that
            # flips the moment the first T=1 chain enters its draw phase, and
            # re-spacing once any chain is RECORDING would break invariance.
            # `temperatures` is mutated IN PLACE because _attempt_swap closes
            # over it by reference and never rebinds it.
            if (
                adapt_ladder
                and not gamma_frozen[0]
                and n_temps > 2
                and n_swap_propose.sum() > 0
            ):
                new_T = _update_ladder_barrier(
                    temperatures, n_swap_accept, n_swap_propose
                )
                if not np.allclose(new_T, temperatures):
                    logger.info(
                        "PTDE-async ladder (barrier-equalized): "
                        f"T=[{', '.join(f'{t:.1f}' for t in new_T)}]"
                    )
                    temperatures[:] = new_T
                # Fresh measurement per adaptation window, as the sync
                # sampler does: a re-spaced ladder's acceptances say nothing
                # about the ladder that produced them.
                n_swap_accept[:] = 0
                n_swap_propose[:] = 0

            if n_completed_total[0] % log_every_evals == 0:
                ar = n_accept / np.maximum(n_propose, 1)
                sr = n_swap_accept / np.maximum(n_swap_propose, 1)
                rt_rate = round_trips[0] / max(n_swap_rounds[0], 1)
                logger.info(
                    f"PTDE-async: {n_completed_total[0]} evals  "
                    f"draws=[min={per_chain_draws.min()}, "
                    f"mean={per_chain_draws.mean():.0f}, "
                    f"max={per_chain_draws.max()}]/{draws}  "
                    f"accept=[{', '.join(f'{r:.2f}' for r in ar)}]  "
                    f"gamma={gamma_box[0]:.4f}  "
                    + (
                        f"swap=[{', '.join(f'{r:.2f}' for r in sr)}]  "
                        f"round_trips={round_trips[0]} "
                        f"(rate={rt_rate:.3f}/swap)"
                        if n_temps > 1
                        else ""
                    )
                )

            _maybe_stop()
            if not stopping[0]:
                _submit(k, i)
            elif in_flight[0] == 0:
                break

        if stop_category[0] == "abort" and int(per_chain_draws.min()) == 0:
            logger.warning(
                f"PTDE-async: stopped ({stop_reason[0]}) before any chain "
                "recorded a draw — nothing to save"
            )
            raise KeyboardInterrupt

    finally:
        _common.restore_stop_handlers(_sig_token)
        if pool is not None:
            # terminate() (not close()) at shutdown: close() waits for any
            # in-flight eval to finish, so a worker still stuck on a slow
            # proposal when the run stops (convergence/maxtime/interrupt)
            # would hang here. Stored draws are already saved; discard the
            # in-flight result. _shutdown_pool SIGKILLs a worker that is
            # wedged past the grace period so join() cannot block forever.
            _common._shutdown_pool(pool)

    actual_draws = int(per_chain_draws.min())
    if actual_draws == 0:
        raise RuntimeError(
            "PTDE-async: sampling stopped — no draws were collected"
        )
    if actual_draws < draws:
        logger.info(
            f"PTDE-async: early stop ({stop_reason[0]}) — "
            f"{actual_draws}/{draws} draws/chain collected "
            f"(some chains ran ahead: max={per_chain_draws.max()})"
        )

    idata = _common.assemble_inference_data(
        stored_raw,
        stored_lp,
        actual_draws,
        n_chains,
        raw_start,
        raw_var_names,
        out_var_names,
        raw_to_phys_batched,
        chain_seed_index,
        "PTDE-async",
        logger,
    )

    if hot_thin:
        import xarray as xr

        # Rectangular cut at the shortest hot chain (rungs run at slightly
        # different speeds); rungs x chains flatten into one 'chain' dim
        # with a per-chain temperature coordinate, which round-trips
        # through netcdf.
        n_hot = int(per_hot_draws.min())
        if n_hot > 0:
            n_hot_chains = (n_temps - 1) * n_chains
            data_vars = {}
            for key in model_keys:
                arr = stored_hot_raw[key][:, :, :n_hot].reshape(
                    (n_hot_chains, n_hot) + raw_start[key].shape
                )
                dims = ("chain", "draw") + tuple(
                    f"{key}_dim_{j}" for j in range(arr.ndim - 2)
                )
                data_vars[str(key)] = (dims, arr)
            data_vars["lp"] = (
                ("chain", "draw"),
                stored_hot_lp[:, :, :n_hot].reshape(n_hot_chains, n_hot),
            )
            idata["posterior_hot"] = xr.Dataset(
                data_vars,
                coords={
                    "chain": np.arange(n_hot_chains),
                    "draw": np.arange(n_hot),
                    "temperature": (
                        "chain",
                        np.repeat(np.asarray(temperatures[1:]), n_chains),
                    ),
                },
            )
            logger.info(
                f"PTDE-async: stored {n_hot} thinned hot draws/chain from "
                f"{n_temps - 1} rungs x {n_chains} chains "
                f"(store_hot_chains={store_hot_chains}, thin={hot_thin})"
            )
        else:
            logger.warning(
                "PTDE-async: store_hot_chains was set but no hot draws "
                "accumulated (draws too small for the thinning factor?)"
            )

    # Ladder communication statistics, stamped on the trace so the mode
    # report can quote them as context (see stamp_and_log_run_summary).
    _extras = []
    if n_swap_discards[0]:
        _extras.append(f"  swap_discards={n_swap_discards[0]}")
    if n_eval_timeouts[0]:
        _extras.append(f"  eval_timeouts={n_eval_timeouts[0]}")
    _common.stamp_and_log_run_summary(
        idata,
        "PTDE-async",
        logger,
        actual_draws=actual_draws,
        draws=draws,
        n_accept=n_accept,
        n_propose=n_propose,
        n_swap_accept=n_swap_accept,
        n_swap_propose=n_swap_propose,
        round_trips=round_trips[0],
        n_swap_rounds=n_swap_rounds[0],
        n_temps=n_temps,
        swap_schedule=swap_schedule,
        rate_unit="swap",
        extras=_extras,
    )
    # Async never resets the swap counters, so these stats span tune+draw;
    # still the right order of magnitude for the barrier check.
    ladder_health_report(temperatures, n_swap_accept, n_swap_propose)

    if collect_rung_timing:
        _common.log_rung_timing(rung_times, temperatures, "PTDE-async", logger)

    return idata
