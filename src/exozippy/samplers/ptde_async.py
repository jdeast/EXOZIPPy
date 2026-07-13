"""
Asynchronous (non-blocking) Parallel Tempering + Differential Evolution sampler.

EXPERIMENTAL -- see hpc_optimization.txt PROMPT 13. Kept as a fully separate
sampler module (YAML: sampler.method: "ptde_async") so the validated
synchronous PTDE in ptde.py is never modified or put at risk. Pick this
sampler explicitly; ptde.py's "ptde" remains the default/recommended choice.

MOTIVATION: ptde.py's synchronous design must wait for the SLOWEST of all
n_temps*n_chains proposals before ANY chain can advance to its next step,
because every chain's next DE proposal needs its rung-mates' CURRENT states,
and "current" is only well-defined once the whole step resolves. Production
runs on examples/DC2018_128 show this stalls the entire sampler behind a rare
but expensive near-caustic evaluation concentrated in the hottest 1-2 rungs
(see hpc_optimization.txt's 2026-07-07 status update: 0.4%/0.09% of rung 6/7
calls exceed 0.1s, and with 320 proposals/step the odds NONE of them land in
that tail across a whole run are essentially zero). This module removes that
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
against another self-consistent pair -- so swaps here remain as rigorous as
in ptde.py. Before trusting a production posterior from this sampler,
validate against a synchronous PTDE run on the same model (see
tests/test_ptde_async.py for the toy-Gaussian recovery check this module
ships with, and consider running both samplers on your real model and
comparing posterior means/std/ESS before relying on ptde_async alone).

Storage differs from ptde.py because chains are no longer synchronized on a
shared step counter: each T=1 chain accumulates its own `draws` post-tune
samples at its own pace (per-chain iteration counters `iter_count`,
per-chain draw counters `per_chain_draws`). A chain that reaches its target
`draws` count keeps evolving (so it does not go stale as a DE/swap partner
for chains still catching up) but stops being recorded. The run stops once
every chain has recorded `draws` samples, or maxtime/convergence/user-
interrupt fires first; at that point output is truncated to
min(per_chain_draws) across chains (the simplest correct option per
hpc_optimization.txt PROMPT 13 item 4).
"""
import gc
import logging
import queue
import signal
import time

import numpy as np
import arviz as az
import multiprocessing as mp
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import vectorize_graph

from exozippy.samplers import ptde as _ptde
from exozippy.samplers.ptde import (
    _check_convergence,
    _convergence_check_schedule,
    _eval_logp,
    _geometric_ladder,
    _make_starts,
    _pick_two,
    _worker_init,
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
    gamma=None,
    target_accept=0.20,
    adapt_gamma=True,
    gamma_adapt_window=None,
    swap_interval=None,
    seed=None,
    log_interval=None,
    plot_prefix=None,
    min_ess=1000,
    max_rhat=1.01,
    maxtime=None,
    eval_timeout=None,
    collect_rung_timing=False,
):
    """
    Asynchronous Parallel Tempering + Differential Evolution sampler.

    See the module docstring for the motivation and the statistical caveat
    around stale DE partners. Parameters mirror exozippy.samplers.ptde.
    ptde_sample where the meaning is identical; a few differ because chains
    are no longer synchronized on a shared step counter:

    swap_interval : int | None -- attempt one swap (random adjacent rung
               pair, random chain in each) every `swap_interval` completed
               evaluations (any rung). None -> n_chains, chosen so the
               long-run ratio of evaluations-per-swap-attempt roughly
               matches ptde.py's one-full-round-of-swaps-per-step cadence.
    gamma_adapt_window : int | None -- adapt gamma once per this many
               completed T=1 proposals still within their own chain's tune
               phase. None -> max(n_chains, (tune * n_chains) // 20), i.e.
               roughly 20 adaptations over tune, matching ptde.py's cadence.

    Returns
    -------
    arviz.InferenceData with posterior and sample_stats["lp"] from T=1 chains.
    """
    _ptde._PTDE_COLLECT_TIMING = collect_rung_timing

    rng = np.random.default_rng(seed)
    temperatures = _geometric_ladder(n_temps, T_max)

    # compile logp ONCE; store in ptde.py's module global BEFORE forking
    # workers -- _eval_logp (imported from ptde.py) reads it from ptde.py's
    # own module namespace, not this module's, since that is where the
    # function object's __globals__ point.
    logp_fn = model.compile_logp()
    _ptde._PTDE_LOGP_FN = logp_fn

    # compile raw -> physical functions ONCE (single-sample and batched);
    # see ptde.py's matching block for the rationale (PROMPT 7).
    output_vars = model.free_RVs + model.deterministics
    raw_to_phys = pytensor.function(
        inputs=model.free_RVs,
        outputs=output_vars,
        on_unused_input="ignore",
    )
    _batched_inputs = [
        pt.tensor(name=f"batched_{v.name}", dtype=v.type.dtype,
                  shape=(None,) + v.type.shape)
        for v in model.free_RVs
    ]
    raw_to_phys_batched = pytensor.function(
        inputs=_batched_inputs,
        outputs=vectorize_graph(output_vars,
                                 replace=dict(zip(model.free_RVs, _batched_inputs))),
        on_unused_input="ignore",
    )
    raw_var_names = [v.name for v in model.free_RVs]
    out_var_names = [v.name for v in output_vars]

    # parameter bookkeeping
    raw_start = system.get_raw_start(model)
    model_keys = list(raw_start.keys())
    n_params = sum(v.size for v in raw_start.values())
    if n_chains is None:
        n_chains = 2 * n_params
    if gamma is None:
        gamma = 2.38 / np.sqrt(2 * n_params)
    logger.info(f"PTDE-async: {n_params} params, {n_chains} chains/rung, gamma={gamma:.4f}")

    if initvals is not None:
        assert len(initvals) == n_chains, "len(initvals) must equal n_chains"
        t1_starts = initvals
        chain_seed_index = [0] * n_chains
    else:
        # Multi-seed starts (P4): round-robin the chain population across seeds.
        # Fall back to a single start for minimal system stubs that don't
        # implement get_raw_starts.
        if raw_starts is None:
            if hasattr(system, "get_raw_starts"):
                raw_starts, seed_indices = system.get_raw_starts(model)
            else:
                raw_starts, seed_indices = [raw_start], [0]
        t1_starts, chain_seed_index = _make_starts(
            n_chains, raw_starts, logp_fn, rng, seed_indices)

    if plot_prefix is not None:
        logger.info("Generating ensemble start plots...")
        batched_vals = raw_to_phys_batched(
            *[np.stack([s[k] for s in t1_starts], axis=0) for k in raw_var_names])
        internal_starts = [
            {name: np.asarray(val)[i] for name, val in zip(out_var_names, batched_vals)}
            for i in range(len(t1_starts))
        ]
        for comp in system.active_components.values():
            comp.plot(system, internal_starts,
                      filename_prefix=plot_prefix + "_start_ensemble")

    # Per-(rung, chain) slot state. current_lp[k][i] is None until that
    # slot's first evaluation completes -- doubles as "still initializing".
    current_state = [
        [{key: v.copy() for key, v in t1_starts[i % n_chains].items()}
         for i in range(n_chains)]
        for _ in range(n_temps)
    ]
    current_lp = [[None] * n_chains for _ in range(n_temps)]
    iter_count = [[0] * n_chains for _ in range(n_temps)]

    slot_list = [(k, i) for k in range(n_temps) for i in range(n_chains)]
    n_slots = len(slot_list)

    total_proposals = n_slots
    phys_cores = mp.cpu_count()
    if cores is None:
        cores = max(1, min(int(phys_cores * 0.75), phys_cores - 1))
    actual_cores = min(cores, total_proposals)
    if cores > phys_cores:
        logger.warning(
            f"PTDE-async: cores={cores} exceeds physical core count ({phys_cores}); "
            f"over-subscription will slow sampling via context switching.")
    logger.info(
        f"PTDE-async: {n_temps} rungs x {n_chains} chains = {total_proposals} slots, "
        f"{actual_cores} cores  "
        f"T=[{', '.join(f'{t:.1f}' for t in temperatures)}]")
    pool = (mp.get_context("fork").Pool(actual_cores, initializer=_worker_init)
            if actual_cores > 1 else None)

    if eval_timeout is not None and pool is None:
        logger.warning(
            f"PTDE-async: eval_timeout={eval_timeout:.0f}s has no effect with a "
            f"single core (cores={actual_cores}) — there is no worker process to "
            f"enforce a wall-clock timeout against a hung logp call.")

    swap_interval = max(1, int(swap_interval)) if swap_interval else max(1, n_chains)
    gamma_adapt_window = (int(gamma_adapt_window) if gamma_adapt_window
                           else max(n_chains, (tune * n_chains) // 20))
    log_every_evals = log_interval or max(n_slots, (n_slots * (tune + draws)) // 20)

    # storage: raw values from T=1 chains only; each chain records exactly
    # `draws` samples (its own post-tune iterations), so no dynamic growth
    # is needed -- capacity is a hard per-chain cap by construction.
    stored_raw = {k: np.zeros((n_chains, draws) + raw_start[k].shape)
                  for k in model_keys}
    stored_lp = np.zeros((n_chains, draws))
    per_chain_draws = np.zeros(n_chains, dtype=int)

    n_accept = np.zeros(n_temps)
    n_propose = np.zeros(n_temps)
    n_swap_accept = np.zeros(max(n_temps - 1, 1))
    n_swap_propose = np.zeros(max(n_temps - 1, 1))
    n_eval_timeouts = [0]
    rung_times = [[] for _ in range(n_temps)]

    gamma_box = [gamma]
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
            "in-flight evaluations (send the signal again to abort immediately)")

    old_sigint = signal.signal(signal.SIGINT, _stop_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _stop_handler)

    result_q = queue.Queue()
    submitted_at = {}     # (k, i) -> submission time, only while in flight
    in_flight_props = {}  # (k, i) -> proposal dict, only while in flight
    in_flight = [0]

    def _build_proposal(k, i):
        if current_lp[k][i] is None:
            # First evaluation for this slot: evaluate the start state itself.
            return {key: v.copy() for key, v in current_state[k][i].items()}
        j1, j2 = _pick_two(rng, n_chains, i)
        pop_k = current_state[k]
        return {key: pop_k[i][key] + gamma_box[0] * (pop_k[j1][key] - pop_k[j2][key])
                for key in model_keys}

    def _submit(k, i):
        prop = _build_proposal(k, i)
        submitted_at[(k, i)] = time.time()
        in_flight_props[(k, i)] = prop
        in_flight[0] += 1
        if pool is None:
            # Serial fallback: no real concurrency, evaluate immediately.
            result_q.put((k, i, prop, _eval_logp(prop)))
            return

        def _cb(result, k=k, i=i, prop=prop):
            result_q.put((k, i, prop, result))

        def _ecb(exc, k=k, i=i, prop=prop):
            logger.error(f"PTDE-async: worker exception at rung {k} chain {i}: {exc}")
            failure = (-np.inf, 0.0) if _ptde._PTDE_COLLECT_TIMING else -np.inf
            result_q.put((k, i, prop, failure))

        pool.apply_async(_eval_logp, (prop,), callback=_cb, error_callback=_ecb)

    def _recycle_pool(reason):
        nonlocal pool
        logger.warning(f"PTDE-async: recycling worker pool ({reason})")
        pool.terminate()
        pool.join()
        pool = None
        gc.collect()
        pool = mp.get_context("fork").Pool(actual_cores, initializer=_worker_init)

    def _attempt_swap():
        if n_temps <= 1:
            return
        k = int(rng.integers(n_temps - 1))
        i = int(rng.integers(n_chains))
        j = int(rng.integers(n_chains))
        lp_i, lp_j = current_lp[k][i], current_lp[k + 1][j]
        if lp_i is None or lp_j is None:
            return   # one side hasn't completed its first evaluation yet
        n_swap_propose[k] += 1
        log_a = (lp_j - lp_i) * (1.0 / temperatures[k] - 1.0 / temperatures[k + 1])
        if rng.random() < np.exp(min(0.0, log_a)):
            current_state[k][i], current_state[k + 1][j] = (
                current_state[k + 1][j], current_state[k][i])
            current_lp[k][i], current_lp[k + 1][j] = lp_j, lp_i
            n_swap_accept[k] += 1

    _do_convergence = (min_ess is not None or max_rhat is not None) and n_chains >= 2
    _check_gen = _convergence_check_schedule() if _do_convergence else None
    _next_check = [next(_check_gen)] if _check_gen else [None]

    stopping = [False]
    stop_reason = [None]     # human-readable, for logging
    stop_category = [None]   # one of "no_draws_ok", "no_draws_abort" -- used
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
        elif (_do_convergence and _next_check[0] is not None
                and int(per_chain_draws.min()) >= _next_check[0]):
            n_check = int(per_chain_draws.min())
            converged, rhat_val, ess_val = _check_convergence(
                stored_raw, n_check, min_ess, max_rhat)
            logger.info(
                f"PTDE-async convergence @ min {n_check} draws/chain: "
                f"max_rhat={rhat_val:.4f}  min_ess={ess_val:.1f}")
            _next_check[0] = next(_check_gen, None)
            if converged:
                stopping[0] = True
                stop_reason[0] = "convergence criterion met"
                stop_category[0] = "complete"

    poll_timeout = max(eval_timeout / 4.0, 0.05) if eval_timeout is not None else None

    try:
        for k, i in slot_list:
            _submit(k, i)

        while in_flight[0] > 0:
            try:
                k, i, prop, result = result_q.get(timeout=poll_timeout)
            except queue.Empty:
                # eval_timeout enforcement: scan for stale in-flight slots.
                # There is no way to kill a single hung worker in a
                # multiprocessing.Pool without tearing down the whole pool
                # (same limitation as ptde.py's _map_logp_timeout), so any
                # OTHER legitimately-still-running slot is also abandoned
                # and immediately resubmitted with a fresh proposal.
                now = time.time()
                stale = [slot for slot, t0 in submitted_at.items()
                         if now - t0 > eval_timeout]
                if stale:
                    n_eval_timeouts[0] += len(stale)
                    for (sk, si) in stale:
                        stale_prop = in_flight_props.get((sk, si))
                        raw_vals = [stale_prop[name] for name in raw_var_names]
                        phys_vals = raw_to_phys(*raw_vals)
                        phys_params = {name: np.asarray(val).tolist()
                                       for name, val in zip(out_var_names, phys_vals)}
                        raw_params = {name: np.asarray(val).tolist()
                                      for name, val in stale_prop.items()}
                        logger.error(
                            f"PTDE-async: logp call exceeded "
                            f"eval_timeout={eval_timeout:.0f}s at rung {sk} "
                            f"chain {si} — rejecting this proposal.\n"
                            f"  physical params: {phys_params}\n"
                            f"  raw params: {raw_params}")
                    lost_slots = list(submitted_at.keys())
                    submitted_at.clear()
                    in_flight_props.clear()
                    in_flight[0] -= len(lost_slots)
                    _recycle_pool(f"{len(stale)} timeout(s)")
                    if not stopping[0]:
                        for (sk, si) in lost_slots:
                            _submit(sk, si)
                continue

            submitted_at.pop((k, i), None)
            in_flight_props.pop((k, i), None)
            in_flight[0] -= 1

            if _ptde._PTDE_COLLECT_TIMING:
                lp, elapsed = result
                rung_times[k].append(elapsed)
            else:
                lp = result

            is_init = current_lp[k][i] is None
            if is_init:
                current_state[k][i] = prop
                current_lp[k][i] = lp
            else:
                T = temperatures[k]
                n_propose[k] += 1
                accepted = (np.isfinite(lp) and rng.random()
                            < np.exp(min(0.0, (lp - current_lp[k][i]) / T)))
                if k == 0 and iter_count[k][i] < tune:
                    n_propose_T1_window[0] += 1
                    if accepted:
                        n_accept_T1_window[0] += 1
                if accepted:
                    current_state[k][i] = prop
                    current_lp[k][i] = lp
                    n_accept[k] += 1
                iter_count[k][i] += 1

                # store T=1 post-tune draws (each chain caps at `draws`)
                if k == 0 and iter_count[k][i] > tune and per_chain_draws[i] < draws:
                    d = per_chain_draws[i]
                    for key in model_keys:
                        stored_raw[key][i, d] = current_state[k][i][key]
                    stored_lp[i, d] = current_lp[k][i]
                    per_chain_draws[i] = d + 1

            n_completed_total[0] += 1

            if n_completed_total[0] % swap_interval == 0:
                _attempt_swap()

            if (adapt_gamma and n_propose_T1_window[0] >= gamma_adapt_window):
                ar_T1 = n_accept_T1_window[0] / max(n_propose_T1_window[0], 1)
                if ar_T1 > 0:
                    scale = (ar_T1 / target_accept) ** 0.5
                    gamma_new = float(np.clip(gamma_box[0] * scale,
                                               gamma_box[0] * 0.1, gamma_box[0] * 10.0))
                    if abs(gamma_new - gamma_box[0]) / gamma_box[0] > 0.01:
                        logger.info(
                            f"PTDE-async gamma: {gamma_box[0]:.4f} -> {gamma_new:.4f} "
                            f"(T=1 accept={ar_T1:.3f}, target={target_accept:.2f})")
                        gamma_box[0] = gamma_new
                n_propose_T1_window[0] = 0
                n_accept_T1_window[0] = 0

            if n_completed_total[0] % log_every_evals == 0:
                ar = n_accept / np.maximum(n_propose, 1)
                sr = n_swap_accept / np.maximum(n_swap_propose, 1)
                logger.info(
                    f"PTDE-async: {n_completed_total[0]} evals  "
                    f"draws=[min={per_chain_draws.min()}, "
                    f"mean={per_chain_draws.mean():.0f}, "
                    f"max={per_chain_draws.max()}]/{draws}  "
                    f"accept=[{', '.join(f'{r:.2f}' for r in ar)}]  "
                    f"gamma={gamma_box[0]:.4f}  "
                    + (f"swap=[{', '.join(f'{r:.2f}' for r in sr)}]"
                       if n_temps > 1 else ""))

            _maybe_stop()
            if not stopping[0]:
                _submit(k, i)
            elif in_flight[0] == 0:
                break

        if stop_category[0] == "abort" and int(per_chain_draws.min()) == 0:
            logger.warning(
                f"PTDE-async: stopped ({stop_reason[0]}) before any chain "
                "recorded a draw — nothing to save")
            raise KeyboardInterrupt

    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if pool is not None:
            pool.close()
            pool.join()

    actual_draws = int(per_chain_draws.min())
    if actual_draws == 0:
        raise RuntimeError("PTDE-async: sampling stopped — no draws were collected")
    if actual_draws < draws:
        logger.info(
            f"PTDE-async: early stop ({stop_reason[0]}) — "
            f"{actual_draws}/{draws} draws/chain collected "
            f"(some chains ran ahead: max={per_chain_draws.max()})")

    logger.info(
        f"PTDE-async: converting {n_chains} x {actual_draws} draws to physical space...")

    n_total = n_chains * actual_draws
    flat_raw = {k: stored_raw[k][:, :actual_draws].reshape(
                    (n_total,) + raw_start[k].shape)
                for k in raw_var_names}
    chunk_size = 20000
    out_chunks = {name: [] for name in out_var_names}
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_out = raw_to_phys_batched(
            *[flat_raw[k][start:end] for k in raw_var_names])
        for name, val in zip(out_var_names, chunk_out):
            out_chunks[name].append(np.asarray(val, dtype=float))

    posterior_dict = {}
    for name in out_var_names:
        arr = np.concatenate(out_chunks[name], axis=0)
        arr = arr.reshape((n_chains, actual_draws) + arr.shape[1:])
        if arr.ndim > 2 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        posterior_dict[name] = arr

    idata = az.from_dict({
        "posterior": posterior_dict,
        "sample_stats": {"lp": stored_lp[:, :actual_draws]},
    })

    # Multi-seed provenance (P4): which solved seed each T=1 chain started from.
    # TODO(P4): surface in outputs/modes.py ModeReport; per-chain attr for now.
    idata.posterior.attrs["chain_seed_index"] = list(chain_seed_index)
    if len(set(chain_seed_index)) > 1:
        logger.info(f"PTDE-async multi-seed provenance (chain -> seed): "
                    f"{list(chain_seed_index)}")

    ar_T1 = float(n_accept[0] / max(n_propose[0], 1))
    sr_all = n_swap_accept / np.maximum(n_swap_propose, 1)
    logger.info(
        f"PTDE-async done: {actual_draws}/{draws} draws  accept(T=1)={ar_T1:.3f}  "
        + (f"swap=[{', '.join(f'{r:.2f}' for r in sr_all)}]"
           if n_temps > 1 else "")
        + (f"  eval_timeouts={n_eval_timeouts[0]}" if n_eval_timeouts[0] else ""))

    if collect_rung_timing:
        logger.info("PTDE-async per-rung logp timing (seconds):")
        for k in range(n_temps):
            times = rung_times[k]
            if not times:
                logger.info(f"  rung {k} (T={temperatures[k]:.1f}): no calls")
                continue
            arr = np.asarray(times)
            n_slow = int((arr > 0.1).sum())
            logger.info(
                f"  rung {k} (T={temperatures[k]:.1f}): n={len(arr)}  "
                f"median={np.median(arr):.3f}  mean={arr.mean():.3f}  "
                f"p90={np.percentile(arr, 90):.3f}  max={arr.max():.3f}  "
                f"n_slow(>0.1s)={n_slow}")

    return idata
