"""
Parallel Tempering + Differential Evolution (PTDE) sampler for PyMC models.

Bypasses pm.sample() to enable:
  - True parallel logp evaluation across all (temperature × chain) proposals
  - Parallel tempering for multi-modal posterior exploration
  - lp values stored during sampling (no post-hoc recomputation)

Default geometric ladder: ntemps=8, T_max=200 (EXOFASTv2 parity).
Hook for Robbins-Monro adaptive ladder via adapt_ladder=True (not yet implemented).

Fork-based parallelism: logp function is inherited by child processes via
copy-on-write, avoiding the picklability constraint that blocks cloudpickle
(PyMC's multiprocessing backend) from serializing PyTensor compiled functions.

Returns arviz.InferenceData compatible with the EXOZIPPy pipeline.
"""
import logging
import os
import signal
import time

import numpy as np
import arviz as az
import multiprocessing as mp
import pytensor

# Force single-threaded BLAS/OMP in every forked worker.  Without this,
# numpy (OpenBLAS/MKL) and C extensions (VBBinaryLensing) each spawn their
# own thread pool, producing n_workers × n_blas_threads threads on a fixed
# number of physical cores and causing catastrophic scheduler thrash.
# These must be set BEFORE fork so children inherit them.
for _tvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "BLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_tvar, "1")

logger = logging.getLogger(__name__)

# Module-level logp function: set in parent process before Pool.  Fork
# children inherit the compiled PyTensor function via copy-on-write without
# pickling.  Proposals (dicts of numpy arrays) are the only IPC payload.
_PTDE_LOGP_FN = None


def _eval_logp(proposal):
    """Worker: evaluate logp for one raw-space proposal dict."""
    try:
        return float(_PTDE_LOGP_FN(proposal))
    except Exception:
        return -np.inf


def _geometric_ladder(n_temps, T_max):
    """T_k = T_max^(k/(n_temps-1)), T_0=1 (target), T_{K-1}=T_max."""
    if n_temps == 1:
        return np.array([1.0])
    return T_max ** (np.arange(n_temps) / (n_temps - 1))


def _probe_scales(raw_start, logp_fn):
    """Adaptive probe: find per-element proposal scales where one step costs ~0.5 nats.

    Matches EXOFASTv2's exofast_getmcmcscale: bisects probe magnitude (both signs)
    until Δlogp ≈ 0.5 (= Δχ²=1).  Falls back to 0.1 when no signal is found,
    avoiding the old fixed-probe default of 1.0 that sent proposals outside tight
    priors and caused ~1000 retries per chain.
    """
    map_lp = float(logp_fn(raw_start))
    target_delta = 0.5   # nats; Δlogp=0.5 ↔ Δχ²=1 (EXOFASTv2 convention)

    scales = {}
    tight = []
    for key, val in raw_start.items():
        n = val.size
        s = np.full(n, 0.1)   # conservative fallback (old default 1.0 → many retries)
        for i in range(n):
            found = False
            for probe_mag in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0]:
                if found:
                    break
                for sign in (1.0, -1.0):
                    dp = sign * probe_mag
                    probe = {k: v.copy() for k, v in raw_start.items()}
                    probe[key] = probe[key].copy()
                    probe[key].flat[i] += dp
                    plp = float(logp_fn(probe))
                    delta = map_lp - plp
                    if delta > 0 and np.isfinite(delta):
                        # Quadratic approx: scale² × (delta/dp²) = target_delta
                        s.flat[i] = min(1.0, abs(dp) * np.sqrt(target_delta / delta))
                        found = True
                        break
            if s.flat[i] < 0.5:
                tight.append(f"{key}[{i}]: scale={s.flat[i]:.3g}")
        scales[key] = s.reshape(val.shape)

    if tight:
        logger.info("PTDE probe: tightly-constrained params: " + "; ".join(tight))
    return map_lp, scales


def _make_starts(n_chains, raw_start, logp_fn, rng):
    """Generate n_chains starting points near MAP using probe-based scaling.

    Mirrors EXOFASTv2: scatter chains by factor × scale where
    factor = min(sqrt(500/n_params), 3), accept any finite logp (no proximity
    threshold), and apply exponential decay only when proposals hit hard prior
    boundaries (lp=-inf).  Raises RuntimeError if a chain cannot be initialized
    within max_iter retries.
    """
    map_lp, scales = _probe_scales(raw_start, logp_fn)
    n_params = sum(v.size for v in raw_start.values())
    factor = min(np.sqrt(500.0 / max(n_params, 1)), 3.0)
    max_iter = 1000
    logger.info(
        f"PTDE init: MAP lp={map_lp:.1f}, n_params={n_params}, factor={factor:.2f}"
    )

    starts = [{k: v.copy() for k, v in raw_start.items()}]
    for j in range(1, n_chains):
        for niter in range(max_iter):
            eff = factor / np.exp(niter / 1000.0)
            prop = {k: v + eff * scales[k] * rng.standard_normal(v.shape)
                    for k, v in raw_start.items()}
            lp = float(logp_fn(prop))
            if np.isfinite(lp):
                starts.append(prop)
                logger.debug(
                    f"PTDE init chain {j}: accepted after {niter} retries "
                    f"(lp={lp:.1f}, Δlp={lp - map_lp:.1f})"
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
                f"Check init_scale values in your params.yaml — a parameter may be "
                f"starting outside its prior bounds."
            )
    return starts


def _worker_init():
    """Pool worker: ignore SIGINT so only the parent handles graceful stop."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _map_logp(pool, proposals):
    if pool is None:
        return [_eval_logp(p) for p in proposals]
    return pool.map(_eval_logp, proposals)


def _map_logp_timeout(pool, proposals, timeout):
    """Evaluate logps with a per-step wall-clock deadline.

    Proposals that don't complete within `timeout` seconds receive -inf so the
    parent loop can proceed without blocking on a single slow VBBinaryLensing
    caustic integration.  The worker continues running in the background and
    will be available for future tasks once it finishes.

    Returns (lps, n_timed_out).
    """
    if pool is None:
        return [_eval_logp(p) for p in proposals], 0

    async_results = [pool.apply_async(_eval_logp, (p,)) for p in proposals]
    deadline = time.time() + timeout
    lps = []
    n_timed_out = 0
    for r in async_results:
        remaining = max(0.001, deadline - time.time())
        try:
            lps.append(r.get(timeout=remaining))
        except mp.TimeoutError:
            lps.append(-np.inf)
            n_timed_out += 1
    return lps, n_timed_out


def _pick_two(rng, n, exclude):
    """Pick two distinct indices from [0, n) excluding `exclude`."""
    idx = rng.choice(n - 1, 2, replace=False)
    return tuple(int(i + (1 if i >= exclude else 0)) for i in idx)


def _convergence_check_schedule(min_draws=100, growth=0.9):
    """Yield cumulative draw counts at which to run a convergence check.

    Positions: round(min_draws / growth**j) for j=0,1,2,...
    Default (growth=0.9): 100, 111, 123, 137, 152, ...
    Gaps grow by ~11% each check, so we check frequently early and less often later.
    """
    j, prev = 0, 0
    while True:
        n = round(min_draws / growth ** j)
        if n > prev:
            yield n
            prev = n
        j += 1


def _check_convergence(stored_raw, n_draws, min_ess, max_rhat):
    """Compute R-hat and ESS on stored_raw[:, :n_draws].

    Returns (converged, max_rhat_val, min_ess_val).
    None thresholds are treated as "no limit" for that statistic.
    """
    data = {key: arr[:, :n_draws] for key, arr in stored_raw.items()}
    try:
        idata_partial = az.from_dict(posterior=data)
        rhat_ds = az.rhat(idata_partial)
        ess_ds = az.ess(idata_partial)
    except Exception:
        return False, float("nan"), float("nan")

    rhat_vals = [float(v.values.max()) for v in rhat_ds.data_vars.values()
                 if v.values.size and not np.all(np.isnan(v.values))]
    ess_vals = [float(v.values.min()) for v in ess_ds.data_vars.values()
                if v.values.size and not np.all(np.isnan(v.values))]

    if not rhat_vals or not ess_vals:
        return False, float("nan"), float("nan")

    max_rhat_val = max(rhat_vals)
    min_ess_val = min(ess_vals)
    converged = (
        (max_rhat is None or max_rhat_val <= max_rhat)
        and (min_ess is None or min_ess_val >= min_ess)
    )
    return converged, max_rhat_val, min_ess_val


def ptde_sample(
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
    gamma=None,
    target_accept=0.20,
    adapt_gamma=True,
    swap_interval=1,
    # Reserved for future Robbins-Monro temperature adaptation:
    target_swap_rate=None,
    adapt_ladder=False,
    seed=None,
    log_interval=None,
    plot_prefix=None,
    min_ess=1000,
    max_rhat=1.01,
    maxtime=None,
    eval_timeout=None,
):
    """
    Parallel Tempering + Differential Evolution sampler.

    Parameters
    ----------
    model : PyMC model (from system.build_model())
    system : EXOZIPPy System (MAP start + raw→physical conversion)
    draws, tune : int
    n_temps : int   — temperature rungs (default 8, EXOFASTv2 parity)
    T_max : float   — hottest temperature (default 200, EXOFASTv2 parity)
    n_chains : int | None  — chains per temperature rung;
               None → 2 × n_params (standard DE minimum for good mixing)
    cores : int | None  — CPU cores for parallel logp evaluation;
               None → n_temps × n_chains (fully parallel, one core per chain)
    initvals : list[dict] | None  — n_chains raw-space starting dicts for
               the T=1 rung; replicated across hotter rungs.
               None → probe-based from MAP (same logic as DEMetropolis block).
    gamma : float   — DE proposal scale; None → 2.38 / sqrt(2 × n_params)
    target_accept : float  — T=1 acceptance rate target for gamma adaptation (default 0.20)
    adapt_gamma : bool     — scale gamma toward target_accept during tune (default True)
    swap_interval : int  — attempt temperature swaps every N steps
    target_swap_rate, adapt_ladder  — reserved for Robbins-Monro (not yet
               implemented; hook is here so the API won't need to change)
    seed : int | None
    log_interval : int | None — steps between progress log lines (None → 5%)
    plot_prefix : str | None  — if set, generate ensemble-start plots at this path prefix

    Returns
    -------
    arviz.InferenceData with posterior and sample_stats["lp"] from T=1 chains.
    """
    global _PTDE_LOGP_FN

    rng = np.random.default_rng(seed)
    temperatures = _geometric_ladder(n_temps, T_max)

    # compile logp ONCE; store in module global BEFORE forking workers
    logp_fn = model.compile_logp()
    _PTDE_LOGP_FN = logp_fn

    # compile raw→physical function ONCE for the final output conversion
    output_vars = model.free_RVs + model.deterministics
    raw_to_phys = pytensor.function(
        inputs=model.free_RVs,
        outputs=output_vars,
        on_unused_input="ignore",
    )
    raw_var_names = [v.name for v in model.free_RVs]   # ordered input names
    out_var_names = [v.name for v in output_vars]       # ordered output names

    # parameter bookkeeping
    raw_start = system.get_raw_start(model)
    model_keys = list(raw_start.keys())
    n_params = sum(v.size for v in raw_start.values())
    if n_chains is None:
        n_chains = 2 * n_params   # standard DE minimum for good mixing
    if gamma is None:
        gamma = 2.38 / np.sqrt(2 * n_params)
    logger.info(f"PTDE: {n_params} params, {n_chains} chains/rung, γ={gamma:.4f}")

    # initialize populations
    if initvals is not None:
        assert len(initvals) == n_chains, "len(initvals) must equal n_chains"
        t1_starts = initvals
    else:
        t1_starts = _make_starts(n_chains, raw_start, logp_fn, rng)

    # ensemble start plots (T=1 starts only; raw→physical via already-compiled fn)
    if plot_prefix is not None:
        logger.info("Generating ensemble start plots...")
        internal_starts = []
        for start in t1_starts:
            raw_vals = [start[k] for k in raw_var_names]
            phys_dict = dict(zip(out_var_names, raw_to_phys(*raw_vals)))
            internal_starts.append(phys_dict)
        for comp in system.active_components.values():
            comp.plot(system, internal_starts,
                      filename_prefix=plot_prefix + "_ensemble_start")

    # Replicate T=1 starts to all rungs; hotter chains spread quickly during tune
    populations = [
        [{k: v.copy() for k, v in t1_starts[i % n_chains].items()}
         for i in range(n_chains)]
        for _ in range(n_temps)
    ]

    # start pool AFTER _PTDE_LOGP_FN is set so fork children inherit it
    total_proposals = n_temps * n_chains
    phys_cores = mp.cpu_count()
    if cores is None:
        # Fallback if called directly (not via run.py): same 75% formula
        cores = max(1, min(int(phys_cores * 0.75), phys_cores - 1))
    actual_cores = min(cores, total_proposals)
    if cores > phys_cores:
        logger.warning(
            f"PTDE: cores={cores} exceeds physical core count ({phys_cores}); "
            f"over-subscription will slow sampling via context switching.")
    logger.info(
        f"PTDE: {n_temps} rungs × {n_chains} chains = {total_proposals} proposals/step, "
        f"{actual_cores} cores  "
        f"T=[{', '.join(f'{t:.1f}' for t in temperatures)}]")
    pool = (mp.get_context("fork").Pool(actual_cores, initializer=_worker_init)
            if actual_cores > 1 else None)

    # Early-stop state: mutable list so the closure can write back to us.
    stop_requested = [False]
    actual_draws = 0
    start_time = time.time()

    _do_convergence = (min_ess is not None or max_rhat is not None) and n_chains >= 2
    _check_gen = _convergence_check_schedule() if _do_convergence else None
    _next_check = [next(_check_gen)] if _check_gen else [None]

    def _sigint_handler(sig, frame):
        if stop_requested[0]:
            raise KeyboardInterrupt   # second Ctrl+C: abort immediately
        stop_requested[0] = True
        logger.info(
            "PTDE: stop requested — finishing current step "
            "(Ctrl+C again to abort immediately)")

    old_sigint = signal.signal(signal.SIGINT, _sigint_handler)
    try:
        # initial logp evaluations
        flat_starts = [populations[k][i]
                       for k in range(n_temps) for i in range(n_chains)]
        all_lps = _map_logp(pool, flat_starts)
        logps = [
            [all_lps[k * n_chains + i] for i in range(n_chains)]
            for k in range(n_temps)
        ]
        logger.info(
            f"PTDE: T=1 initial lp  "
            f"min={min(logps[0]):.1f}  max={max(logps[0]):.1f}")

        # storage: raw values from T=1 chains only
        stored_raw = {k: np.zeros((n_chains, draws) + raw_start[k].shape)
                      for k in model_keys}
        stored_lp = np.zeros((n_chains, draws))

        n_accept = np.zeros(n_temps)
        n_propose = np.zeros(n_temps)
        n_swap_accept = np.zeros(max(n_temps - 1, 1))
        n_swap_propose = np.zeros(max(n_temps - 1, 1))

        total_steps = tune + draws
        log_every = log_interval or max(1, total_steps // 20)

        for step in range(total_steps):
            phase = "tune" if step < tune else "draw"
            draw_idx = step - tune
            _t0 = time.time()

            # 1. build DE proposals for every chain at every temperature
            props_flat = []
            prop_map = []
            for k in range(n_temps):
                pop_k = populations[k]
                for i in range(n_chains):
                    j1, j2 = _pick_two(rng, n_chains, i)
                    prop = {key: pop_k[i][key]
                                 + gamma * (pop_k[j1][key] - pop_k[j2][key])
                            for key in model_keys}
                    props_flat.append(prop)
                    prop_map.append((k, i))
            _t_build = time.time()

            # 2. evaluate all logps in parallel
            prop_lps = _map_logp(pool, props_flat)
            _t_eval = time.time()

            # 3. Metropolis accept/reject at effective temperature T_k
            for idx, (k, i) in enumerate(prop_map):
                T = temperatures[k]
                lp_new = prop_lps[idx]
                n_propose[k] += 1
                if (np.isfinite(lp_new)
                        and rng.random()
                            < np.exp(min(0.0, (lp_new - logps[k][i]) / T))):
                    populations[k][i] = props_flat[idx]
                    logps[k][i] = lp_new
                    n_accept[k] += 1

            # 4. temperature swaps (adjacent rungs, random chain pair)
            if n_temps > 1 and (step + 1) % swap_interval == 0:
                for k in range(n_temps - 1):
                    i = int(rng.integers(n_chains))
                    j = int(rng.integers(n_chains))
                    n_swap_propose[k] += 1
                    # Detailed balance: (logp_j - logp_i) * (1/T_k - 1/T_{k+1})
                    # T_k < T_{k+1}  →  factor > 0  →  accept when logp_j > logp_i
                    log_a = ((logps[k+1][j] - logps[k][i])
                             * (1.0 / temperatures[k] - 1.0 / temperatures[k + 1]))
                    if rng.random() < np.exp(min(0.0, log_a)):
                        (populations[k][i], populations[k+1][j]) = (
                            populations[k+1][j], populations[k][i])
                        (logps[k][i], logps[k+1][j]) = (
                            logps[k+1][j], logps[k][i])
                        n_swap_accept[k] += 1

            _t_step = time.time()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"PTDE step {step+1} ({phase})  "
                    f"total={_t_step-_t0:.3f}s  "
                    f"build={_t_build-_t0:.3f}s  "
                    f"eval={_t_eval-_t_build:.3f}s  "
                    f"rest={_t_step-_t_eval:.3f}s  "
                    f"T1_lp=[{min(logps[0]):.1f},{max(logps[0]):.1f}]")

            # 5. store T=1 draws
            if phase == "draw":
                for i in range(n_chains):
                    for key in model_keys:
                        stored_raw[key][i, draw_idx] = populations[0][i][key]
                    stored_lp[i, draw_idx] = logps[0][i]
                actual_draws = draw_idx + 1

            # 6. progress log + gamma adaptation during tune
            if (step + 1) % log_every == 0:
                ar = n_accept / np.maximum(n_propose, 1)
                sr = n_swap_accept / np.maximum(n_swap_propose, 1)

                if phase == "tune" and adapt_gamma:
                    # Scale gamma toward target_accept based on T=1 window rate.
                    # Multiplicative update: gamma *= (ar_T1 / target_accept)^0.5
                    # The sqrt dampens oscillation; clamp to [gamma/10, gamma*10].
                    ar_T1 = ar[0]
                    if ar_T1 > 0:
                        scale = (ar_T1 / target_accept) ** 0.5
                        gamma_new = float(np.clip(gamma * scale,
                                                   gamma * 0.1, gamma * 10.0))
                        if abs(gamma_new - gamma) / gamma > 0.01:
                            logger.info(f"PTDE gamma: {gamma:.4f} → {gamma_new:.4f} "
                                        f"(T=1 accept={ar_T1:.3f}, target={target_accept:.2f})")
                            gamma = gamma_new

                    # Reset window counters so next log period is a fresh measurement
                    n_accept[:] = 0
                    n_propose[:] = 0
                    n_swap_accept[:] = 0
                    n_swap_propose[:] = 0

                logger.info(
                    f"PTDE {step+1}/{total_steps} ({phase})  "
                    f"accept=[{', '.join(f'{r:.2f}' for r in ar)}]  "
                    f"γ={gamma:.4f}  "
                    + (f"swap=[{', '.join(f'{r:.2f}' for r in sr)}]"
                       if n_temps > 1 else ""))

            # 7. early-stop checks
            if stop_requested[0]:
                if actual_draws == 0:
                    logger.warning("PTDE: stop requested during tune — no draws to save")
                    raise KeyboardInterrupt
                logger.info(f"PTDE: stopping after {actual_draws} draws (user interrupt)")
                break

            if maxtime is not None and (time.time() - start_time) > maxtime:
                if actual_draws == 0:
                    logger.warning(f"PTDE: time limit {maxtime:.0f}s reached during tune — no draws to save")
                    raise KeyboardInterrupt
                logger.info(
                    f"PTDE: wall-clock limit {maxtime:.0f}s reached "
                    f"after {actual_draws} draws")
                break

            if (phase == "draw"
                    and _next_check[0] is not None
                    and actual_draws >= _next_check[0]):
                converged, rhat_val, ess_val = _check_convergence(
                    stored_raw, actual_draws, min_ess, max_rhat)
                logger.info(
                    f"PTDE convergence @ {actual_draws} draws: "
                    f"max_rhat={rhat_val:.4f}  min_ess={ess_val:.1f}")
                _next_check[0] = next(_check_gen, None)
                if converged:
                    logger.info("PTDE: convergence criterion met, wrapping up")
                    break

    finally:
        signal.signal(signal.SIGINT, old_sigint)
        if pool is not None:
            pool.close()
            pool.join()

    # convert raw → physical for every stored draw
    if actual_draws == 0:
        raise RuntimeError("PTDE: sampling stopped during tune — no draws were collected")
    if actual_draws < draws:
        logger.info(f"PTDE: early stop — {actual_draws}/{draws} draws collected")

    logger.info(
        f"PTDE: converting {n_chains} × {actual_draws} draws to physical space…")
    chain_lists = {name: [] for name in out_var_names}

    for i in range(n_chains):
        draw_lists = {name: [] for name in out_var_names}
        for d in range(actual_draws):
            raw_vals = [stored_raw[k][i, d] for k in raw_var_names]
            phys_vals = raw_to_phys(*raw_vals)
            for name, val in zip(out_var_names, phys_vals):
                draw_lists[name].append(np.atleast_1d(np.asarray(val, dtype=float)))
        for name in out_var_names:
            chain_lists[name].append(
                np.stack(draw_lists[name], axis=0))   # (draws, ...)

    # assemble posterior dict: (n_chains, draws, ...) per variable
    posterior_dict = {}
    for name in out_var_names:
        arr = np.stack(chain_lists[name], axis=0)   # (n_chains, draws, ...)
        # atleast_1d adds a trailing dim-1 for scalar params; squeeze it away
        if arr.ndim > 2 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        posterior_dict[name] = arr

    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={"lp": stored_lp[:, :actual_draws]},
    )

    ar_T1 = float(n_accept[0] / max(n_propose[0], 1))
    sr_all = n_swap_accept / np.maximum(n_swap_propose, 1)
    logger.info(
        f"PTDE done: {actual_draws}/{draws} draws  accept(T=1)={ar_T1:.3f}  "
        + (f"swap=[{', '.join(f'{r:.2f}' for r in sr_all)}]"
           if n_temps > 1 else ""))

    return idata
