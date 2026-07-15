"""
Parallel Tempering + Differential Evolution (PTDE) sampler for PyMC models.

Bypasses pm.sample() to enable:
  - True parallel logp evaluation across all (temperature × chain) proposals
  - Parallel tempering for multi-modal posterior exploration
  - lp values stored during sampling (no post-hoc recomputation)

Default geometric ladder: ntemps=8, T_max=200 (EXOFASTv2 parity).
Adaptive ladder via adapt_ladder=True re-spaces the rungs to equalize the
communication barrier during tuning (Syed et al. 2022).

Temperature swaps use the Deterministic Even-Odd (DEO) schedule by default
(swap_schedule="deo"; Syed et al. 2022, "Non-reversible parallel tempering"),
which turns round-trip transport across the ladder from O(n_temps^2) to
O(n_temps). Pass swap_schedule="random" to restore the legacy random-pair
schedule for A/B comparison.

Fork-based parallelism: logp function is inherited by child processes via
copy-on-write, avoiding the picklability constraint that blocks cloudpickle
(PyMC's multiprocessing backend) from serializing PyTensor compiled functions.

Returns arviz.InferenceData compatible with the EXOZIPPy pipeline.
"""
import gc
import logging
import os
import signal
import time

import numpy as np
import arviz as az
import multiprocessing as mp
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import vectorize_graph

# Force single-threaded BLAS/OMP in every forked worker.  Without this,
# numpy (OpenBLAS/MKL) and C extensions (VBBinaryLensing) each spawn their
# own thread pool, producing n_workers × n_blas_threads threads on a fixed
# number of physical cores and causing catastrophic scheduler thrash -- and,
# separately, each with a memory arena sized to the (over-subscribed) thread
# count, which is what actually blows up h_vmem once forked N-ways.
# exozippy/__init__.py already sets these before numpy/pytensor/pymc/arviz/
# jax are imported at all -- the only point where it's effective, since a
# native thread pool can't be shrunk after the fact by setting os.environ
# once numpy et al. are already loaded (import exozippy always runs
# __init__.py first, before this module). This block is redundant there;
# kept as a guard for any environment that imports ptde.py without ever
# importing the exozippy package proper.
for _tvar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "BLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_tvar, "1")

logger = logging.getLogger(__name__)

# Shared with outputs.modes.identify_modes: |lp| above this is numerically
# broken, not a real posterior mode (no realistic dataset's logp reaches
# 1e12). Imported (not duplicated) so the two ceilings can't drift apart.
from exozippy.outputs.modes import DEFAULT_LP_ABS_MAX as _DEFAULT_LP_ABS_MAX
from exozippy.samplers import convergence

# Module-level logp function: set in parent process before Pool.  Fork
# children inherit the compiled PyTensor function via copy-on-write without
# pickling.  Proposals (dicts of numpy arrays) are the only IPC payload.
_PTDE_LOGP_FN = None

# Diagnostic flag (see ptde_sample(collect_rung_timing=...) and
# hpc_optimization.txt P13): set in the parent before forking, like
# _PTDE_LOGP_FN, so workers inherit it via copy-on-write. When True,
# _eval_logp times its own call and returns (lp, elapsed_seconds) instead of
# a bare float, so the parent can attribute wall time to a rung via
# prop_map -- resolves whether slow evaluations concentrate in a few
# chains/rungs or spread across many, which determines how much benefit an
# async dispatch redesign (P13) would realistically deliver.
_PTDE_COLLECT_TIMING = False


def _eval_logp(proposal):
    """Worker: evaluate logp for one raw-space proposal dict.

    Returns a bare float normally. When _PTDE_COLLECT_TIMING is set, returns
    (lp, elapsed_seconds) instead (diagnostic mode; see module docstring
    above _PTDE_COLLECT_TIMING).
    """
    if _PTDE_COLLECT_TIMING:
        t0 = time.perf_counter()
        try:
            lp = float(_PTDE_LOGP_FN(proposal))
        except Exception:
            lp = -np.inf
        return lp, time.perf_counter() - t0
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


def _make_starts(n_chains, raw_starts, logp_fn, rng, seed_indices=None):
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

    Returns (starts, chain_seed_index) where chain_seed_index[j] is the original
    seed index that chain j was drawn from (for trace-attr provenance).
    """
    if isinstance(raw_starts, dict):
        raw_starts = [raw_starts]
    K = len(raw_starts)
    if seed_indices is None:
        seed_indices = list(range(K))

    # Probe scales once from seed 0 (the canonical MAP-ish start); the same
    # per-parameter jitter scale is reused around every seed.
    map_lp, scales = _probe_scales(raw_starts[0], logp_fn)
    n_params = sum(v.size for v in raw_starts[0].values())
    factor = min(np.sqrt(500.0 / max(n_params, 1)), 3.0)
    max_iter = 1000
    logger.info(
        f"PTDE init: MAP lp={map_lp:.1f}, n_params={n_params}, factor={factor:.2f}"
        + (f", {K} seeds (round-robin over {n_chains} chains)" if K > 1 else "")
    )

    starts = []
    chain_seed_index = []
    seed_seen = set()
    for j in range(n_chains):
        s = j % K
        center = raw_starts[s]
        # First chain of each seed group starts exactly at the solved seed.
        if s not in seed_seen:
            lp0 = float(logp_fn(center))
            if np.isfinite(lp0):
                starts.append({k: v.copy() for k, v in center.items()})
                chain_seed_index.append(seed_indices[s])
                seed_seen.add(s)
                logger.debug(f"PTDE init chain {j}: exact seed {seed_indices[s]} "
                             f"(lp={lp0:.1f})")
                continue
            logger.warning(
                f"PTDE init: seed {seed_indices[s]} exact start has non-finite "
                f"lp; jittering to find a finite start.")
        for niter in range(max_iter):
            eff = factor / np.exp(niter / 1000.0)
            prop = {k: v + eff * scales[k] * rng.standard_normal(v.shape)
                    for k, v in center.items()}
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
                f"Check init_scale values in your params.yaml -- a parameter may be "
                f"starting outside its prior bounds."
            )
    return starts, chain_seed_index


def _worker_init():
    """Pool worker: ignore SIGINT/SIGTERM so only the parent handles graceful
    stop. A batch scheduler typically signals the whole process group, and a
    worker that died mid pool.map() would break the parent's current step
    (BrokenProcessPool) instead of letting it finish and wrap up cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _map_logp(pool, proposals):
    if pool is None:
        return [_eval_logp(p) for p in proposals]
    return pool.map(_eval_logp, proposals)


def _map_logp_timeout(pool, proposals, timeout):
    """Evaluate logps with a per-call wall-clock timeout.

    Each proposal individually gets up to `timeout` seconds (not a deadline
    shared across the whole batch — a slow-but-legitimate early proposal must
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
    before. The caller should warn about this once at startup if cores<=1.

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


def _pick_two(rng, n, exclude):
    """Pick two distinct indices from [0, n) excluding `exclude`."""
    idx = rng.choice(n - 1, 2, replace=False)
    return tuple(int(i + (1 if i >= exclude else 0)) for i in idx)


# ---------------------------------------------------------------------------
# Deterministic Even-Odd (DEO) swap schedule + round-trip diagnostics
# (Syed et al. 2022, JRSS-B, "Non-reversible parallel tempering"; reference
# implementation Pigeons.jl). The DEO schedule changes only WHICH adjacent
# rung pairs are attempted on a given swap round, not the per-swap Metropolis
# test, so parallel-tempering invariance is untouched. Alternating the pair
# offset each round makes the temperature-index process non-reversible: a
# configuration that just moved up the ladder tends to keep moving up rather
# than immediately undoing the move, so round trips (cold -> hot -> cold, the
# excursions that transport a chain out of one posterior mode and into
# another) scale O(n_temps) instead of the O(n_temps^2) of random-pair swaps.
# ---------------------------------------------------------------------------

def _deo_pairs(round_idx, n_temps, active_rungs=None):
    """Adjacent rung pairs attempted simultaneously in one DEO swap round.

    Even rounds (round_idx even) attempt (0,1),(2,3),(4,5),...; odd rounds
    attempt (1,2),(3,4),(5,6),.... The pairs within a round are disjoint (no
    rung appears twice), so all can be attempted at once, and the alternating
    offset is what makes the index process non-reversible.

    active_rungs : iterable[int] | None -- when given, any pair touching a
        rung not in this set (e.g. one thinned out this round in ptde.py) is
        dropped, rather than reverting to a random pairing, which would break
        the non-reversible index flow. None -> every pair is returned.
    """
    start = 0 if round_idx % 2 == 0 else 1
    pairs = [(k, k + 1) for k in range(start, n_temps - 1, 2)]
    if active_rungs is not None:
        active = set(active_rungs)
        pairs = [(a, b) for (a, b) in pairs if a in active and b in active]
    return pairs


def _deo_pair_sequence(n_temps):
    """Deterministic cycling order of adjacent-pair lower indices for the
    async sampler: all even pairs (0,1),(2,3),... exhausted first, then all
    odd pairs (1,2),(3,4),..., then repeat. Async has no synchronized rounds,
    so it fires one swap per `swap_interval` completed evaluations and walks
    this fixed sequence instead of drawing a random rung pair -- same DEO
    idea (deterministic, non-reversible pair selection) adapted to event time.
    Returns the lower rung index k of each pair (the pair is (k, k+1)).
    """
    even = list(range(0, n_temps - 1, 2))
    odd = list(range(1, n_temps - 1, 2))
    return even + odd


def _record_round_trips(direction, round_trips, n_temps):
    """Update per-member direction tags at the extreme rungs and count
    completed cold -> hot -> cold round trips.

    direction : list[list[int]] -- direction[k][i] in {0, +1, -1} is the last
        extreme rung the configuration NOW occupying slot (k, i) has visited
        (+1 = cold end / heading up, -1 = hot end / heading down, 0 = neither
        yet). Tags travel WITH the configuration through swaps: the caller
        swaps direction[k][i] alongside the population state and its logp, so
        a counted round trip means one configuration was carried the full
        length of the ladder and back -- exactly the transport that moves a
        chain between posterior modes.
    round_trips : list[int] -- single-element mutable counter, incremented in
        place. Idempotent: a cold slot already tagged +1 is not recounted, so
        the synchronous sampler can call this once per DEO round and the async
        sampler after every swap event without double-counting.

    THE round-trip metric is the direct measure of whether the ladder is
    actually transporting mass between modes; report it next to per-rung swap
    acceptance.
    """
    if n_temps < 2:
        return
    hot = n_temps - 1
    n_chains = len(direction[0])
    for i in range(n_chains):
        # Any configuration currently at the hottest rung is now "heading
        # down" toward the cold end.
        direction[hot][i] = -1
    for i in range(n_chains):
        # A configuration back at the coldest rung that last touched the hot
        # end has completed a full cold -> hot -> cold round trip.
        if direction[0][i] == -1:
            round_trips[0] += 1
        direction[0][i] = 1


def _update_ladder_barrier(temperatures, swap_accept, swap_propose):
    """Re-space the temperature ladder to equalize the communication barrier
    (Syed et al. 2022). Returns a new temperature array.

    The per-pair swap REJECTION rate r_k approximates the local communication
    barrier between rungs k and k+1; the cumulative barrier up to rung k is
    Lambda_k = sum_{j<k} r_j, and the total barrier is Lambda_{K-1}. An
    optimally-tuned ladder carries an equal share of the barrier on every
    rung, so we place the interior rungs at equal barrier fractions by
    interpolating coldness beta = 1/T against the cumulative barrier. The two
    endpoints (T_0 = 1 target, T_{K-1} = T_max) are pinned so the ladder still
    spans the same temperature range (EXOFASTv2 parity at the ends).

    Only valid to call DURING the tuning phase -- re-spacing the ladder after
    tuning would break invariance, the same rule the DE gamma adaptation
    follows.
    """
    n_temps = len(temperatures)
    if n_temps < 3:
        return np.asarray(temperatures, dtype=float)
    r = 1.0 - swap_accept / np.maximum(swap_propose, 1)
    r = np.clip(r, 0.0, 1.0)
    # Cumulative barrier at each rung; Lambda[0] = 0, length n_temps.
    Lambda = np.concatenate([[0.0], np.cumsum(r)])
    total = float(Lambda[-1])
    if total <= 0.0:
        # Perfect mixing (or no swap data): nothing to equalize.
        return np.asarray(temperatures, dtype=float)
    # Lambda is monotonically non-decreasing in k (valid np.interp x); beta is
    # monotonically decreasing in k. Guard against flat segments (r_k == 0)
    # that would make Lambda non-strictly-increasing by nudging duplicates.
    for k in range(1, n_temps):
        if Lambda[k] <= Lambda[k - 1]:
            Lambda[k] = Lambda[k - 1] + 1e-9
    beta = 1.0 / np.asarray(temperatures, dtype=float)
    targets = np.linspace(0.0, Lambda[-1], n_temps)
    new_beta = np.interp(targets, Lambda, beta)
    new_beta[0] = beta[0]      # pin target rung (T=1)
    new_beta[-1] = beta[-1]    # pin hottest rung (T=T_max)
    return 1.0 / new_beta


def _active_rungs(step, n_temps, thin_start, thin_factor):
    """Rung indices that propose a DE move at this step.

    Rungs below thin_start always propose; rungs at or above it only
    propose every thin_factor-th step (thin_factor<=1 -> always, i.e. no
    thinning). Swaps are unaffected by thinning -- they only exchange
    already-cached (population, logp) pairs between adjacent rungs, so a
    rung that skipped its own DE move this step can still participate in
    a swap using its last-computed logp.

    Rationale (hpc_optimization.txt P12): PTDE's per-step wall time is
    gated by the SLOWEST of all n_temps*n_chains proposals. Hot rungs
    (large T) explore a heavily flattened target and routinely draw
    parameter combinations that are individually expensive to evaluate but
    scientifically irrelevant (only the T=1 rung's draws are kept). Thinning
    them directly cuts the number of chances per step to draw from that
    expensive tail, at the cost of slower mixing for swap partners.
    """
    if thin_factor <= 1:
        return list(range(n_temps))
    return [k for k in range(n_temps)
            if k < thin_start or step % thin_factor == 0]


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


def _check_convergence(stored_raw, n_draws, min_ess, max_rhat, stored_lp=None):
    """Live early-stop test on the first ``n_draws`` stored T=1 draws.

    Judges convergence on the trace AFTER dropping stuck chains and trimming
    a generous fixed burn-in (the last-half tail), so the transient can no
    longer poison the Rhat/ESS the stop decision reads -- the reason a run
    with a slow, likelihood-flat degenerate direction otherwise never
    auto-stops. Rank Rhat/bulk-ESS are transform-invariant, so computing on
    the raw draws matches the physical report. The precise (ESS-maximizing)
    burn-in is found once at wrap-up by convergence.find_burnin; here we only
    need the cheap pass/fail. See samplers/convergence.py.

    Returns (converged, max_rhat_val, min_ess_val). None thresholds are
    treated as "no limit" for that statistic.
    """
    posterior = {key: arr[:, :n_draws] for key, arr in stored_raw.items()}
    lp = stored_lp[:, :n_draws] if stored_lp is not None else None
    try:
        return convergence.converged_on_tail(
            posterior, lp, min_ess, max_rhat)
    except Exception:
        return False, float("nan"), float("nan")


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
    raw_starts=None,
    seed_indices=None,
    gamma=None,
    target_accept=0.20,
    adapt_gamma=True,
    swap_interval=1,
    swap_schedule="deo",
    target_swap_rate=None,
    adapt_ladder=False,
    rung_thin_factor=1,
    rung_thin_start=None,
    collect_rung_timing=False,
    seed=None,
    log_interval=None,
    plot_prefix=None,
    min_ess=1000,
    max_rhat=1.01,
    maxtime=None,
    eval_timeout=None,
    lp_plausibility_ceiling=None,
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
    swap_schedule : {"deo", "random"}  — "deo" (default) uses the
               Deterministic Even-Odd non-reversible schedule (Syed et al.
               2022): even swap rounds attempt rung pairs (0,1),(2,3),...;
               odd rounds (1,2),(3,4),.... Within an attempted pair, chain i
               of the colder rung is paired with chain perm[i] of the hotter
               rung under a fresh random permutation each round, so n_chains
               swaps are attempted per pair (each pairwise swap satisfies
               detailed balance at fixed pairing). "random" restores the
               legacy one-random-chain-pair-per-adjacent-rung schedule for
               A/B comparison.
    target_swap_rate  — reserved (kept so the API won't need to change).
    adapt_ladder : bool  — when True, re-space the ladder during tuning to
               equalize the per-rung communication barrier (Syed et al. 2022);
               adaptation stops when tuning ends (adapting afterward would
               break invariance -- same rule the gamma adaptation follows).
               Default False keeps the geometric ladder (EXOFASTv2 parity).
    rung_thin_factor : int  — update rungs >= rung_thin_start only every
               rung_thin_factor-th step (default 1 = no thinning, every rung
               proposes every step). Directly cuts the number of chances per
               step that a hot, heavily-flattened rung draws a parameter
               combination that is expensive to evaluate but scientifically
               irrelevant (see hpc_optimization.txt P12). Swaps are
               unaffected -- they exchange cached (population, logp) pairs
               and need no new evaluation.
    rung_thin_start : int | None  — first rung index subject to thinning;
               None -> n_temps // 2. Clamped to >= 1: the T=1 rung (index 0,
               the only one whose draws are kept) is never thinned.
    collect_rung_timing : bool  — diagnostic (see hpc_optimization.txt P13):
               record per-call wall time and attribute it to a rung, logging
               a summary (count/median/mean/p90/max per rung) when sampling
               finishes. Default False (zero overhead when off).
    seed : int | None
    log_interval : int | None — steps between progress log lines (None → 5%)
    plot_prefix : str | None  — if set, generate ensemble-start plots at this path prefix
    eval_timeout : float | None  — user-settable per-call wall-clock timeout (seconds)
               for a single logp evaluation (default None = no timeout; proposals run
               to completion no matter how long they take). Opt in with a value
               (e.g. 10.0) for models whose logp can call into a backend known to
               occasionally hang on pathological parameter combinations. A call that
               doesn't return within the timeout is treated as -inf (the proposal is
               rejected by the normal accept/reject logic) and the worker pool is
               recycled, since the stuck worker may never return.
               Has no effect when cores<=1 (no worker pool to enforce it against).
    maxtime : float | None  — wall-clock budget in seconds; sampling stops
               gracefully once exceeded, keeping whatever draws were already
               collected. SIGINT (Ctrl+C) or SIGTERM (e.g. `qsig -s SIGTERM
               <job_id>` / `kill -TERM <pid>`) trigger the same graceful
               stop-after-this-step behavior on demand, without waiting for
               maxtime.
    lp_plausibility_ceiling : float | None  — |lp| threshold above which a
               T=1 chain's logp is logged as a loud one-time warning, since
               no realistic dataset's logp reaches this scale: it always
               indicates a model bug (e.g. an unbounded/uncancelled logp
               term), not physics. None -> outputs.modes.DEFAULT_LP_ABS_MAX
               (the same constant identify_modes uses to reject runaway
               draws post-hoc).

    Returns
    -------
    arviz.InferenceData with posterior and sample_stats["lp"] from T=1 chains.
    """
    global _PTDE_LOGP_FN, _PTDE_COLLECT_TIMING
    _PTDE_COLLECT_TIMING = collect_rung_timing
    if lp_plausibility_ceiling is None:
        lp_plausibility_ceiling = _DEFAULT_LP_ABS_MAX
    warned_implausible_lp = False

    if swap_schedule not in ("deo", "random"):
        raise ValueError(
            f"swap_schedule must be 'deo' or 'random', got {swap_schedule!r}")

    rng = np.random.default_rng(seed)
    temperatures = _geometric_ladder(n_temps, T_max)

    _rung_thin_factor = max(1, int(rung_thin_factor))
    _rung_thin_start = (n_temps // 2 if rung_thin_start is None
                        else int(rung_thin_start))
    _rung_thin_start = max(1, min(_rung_thin_start, n_temps))  # never thin T=1
    if _rung_thin_factor > 1 and _rung_thin_start < n_temps:
        logger.info(
            f"PTDE: rung thinning enabled — rungs >= {_rung_thin_start} "
            f"(of {n_temps}) propose every {_rung_thin_factor} steps")

    # compile logp ONCE; store in module global BEFORE forking workers
    logp_fn = model.compile_logp()
    _PTDE_LOGP_FN = logp_fn

    # compile raw→physical function ONCE for the final output conversion.
    # raw_to_phys is the single-sample form, kept for the (rare) eval_timeout
    # diagnostic log path, which converts exactly one proposal at a time.
    # raw_to_phys_batched vectorizes the same graph over an extra leading
    # sample axis (pytensor's vectorize_graph -- adds the batch dim to every
    # op in the graph rather than looping in Python) and is what the
    # ensemble-start-plot and final posterior conversions use, since those
    # can be tens of thousands to millions of samples: the free_RVs +
    # deterministics graph is pure elementwise/indexing math (each
    # Parameter's physical-unit conversion; verified empirically that no
    # deterministic here touches the magnification Ops, which only feed the
    # likelihood), so it vectorizes cleanly and cuts what was a
    # Python-level per-sample loop (dominant cost: interpreter + pytensor
    # call overhead, not the underlying math) down to a handful of batched
    # calls. See hpc_optimization.txt PROMPT 7.
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
        chain_seed_index = [0] * n_chains
    else:
        # Multi-seed starts (P4): round-robin the chain population across every
        # solved seed. raw_starts/seed_indices come from run.py when available;
        # else fall back to system.get_raw_starts, and further to a bare
        # get_raw_start (single start) for minimal test/system stubs that don't
        # implement get_raw_starts at all.
        if raw_starts is None:
            if hasattr(system, "get_raw_starts"):
                raw_starts, seed_indices = system.get_raw_starts(model)
            else:
                raw_starts, seed_indices = [raw_start], [0]
        t1_starts, chain_seed_index = _make_starts(
            n_chains, raw_starts, logp_fn, rng, seed_indices)

    # ensemble start plots (T=1 starts only; raw→physical via the batched fn)
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

    if eval_timeout is not None and pool is None:
        logger.warning(
            f"PTDE: eval_timeout={eval_timeout:.0f}s has no effect with a single "
            f"core (cores={actual_cores}) — there is no worker process to enforce "
            f"a wall-clock timeout against a hung logp call.")

    # Early-stop state: mutable list so the closure can write back to us.
    stop_requested = [False]
    actual_draws = 0
    start_time = time.time()
    n_eval_timeouts = [0]  # mutable box, incremented by _eval_logps_safe
    rung_times = [[] for _ in range(n_temps)]  # per-rung wall times (collect_rung_timing only)

    def _eval_logps_safe(proposals, step_label, index_labels=None, rungs=None):
        """Evaluate logps for `proposals`, honoring eval_timeout if set.

        A logp call that exceeds eval_timeout is treated as -inf, which the
        normal Metropolis accept/reject logic rejects automatically. The full
        (raw and physical) parameter set that triggered the timeout is
        logged so the run can be reproduced and diagnosed offline. The
        worker process that was evaluating it may never return on its own,
        so the pool is recycled whenever a timeout occurs — otherwise a long
        run slowly bleeds workers, one per hang, until none are left.

        index_labels : list[str] | None — optional per-proposal identity
            (e.g. "rung 3 chain 12"), same length/order as `proposals`, used
            in place of a bare index in the timeout log message.
        rungs : list[int] | None — optional per-proposal rung index, same
            length/order as `proposals`. When collect_rung_timing is set,
            per-call wall times are attributed to these rungs.
        """
        nonlocal pool
        if eval_timeout is None:
            raw = _map_logp(pool, proposals)
            if _PTDE_COLLECT_TIMING:
                lps = [r[0] for r in raw]
                if rungs is not None:
                    for r, k in zip(raw, rungs):
                        rung_times[k].append(r[1])
                return lps
            return raw

        lps, timed_out = _map_logp_timeout(pool, proposals, eval_timeout)
        if timed_out:
            n_eval_timeouts[0] += len(timed_out)
            for idx in timed_out:
                raw_vals = [proposals[idx][k] for k in raw_var_names]
                phys_vals = raw_to_phys(*raw_vals)
                phys_params = {name: np.asarray(val).tolist()
                               for name, val in zip(out_var_names, phys_vals)}
                raw_params = {k: np.asarray(v).tolist()
                              for k, v in proposals[idx].items()}
                who = index_labels[idx] if index_labels is not None else f"proposal {idx}"
                logger.error(
                    f"PTDE: logp call exceeded eval_timeout={eval_timeout:.0f}s "
                    f"at {step_label} ({who}) — rejecting this proposal.\n"
                    f"  physical params: {phys_params}\n"
                    f"  raw params: {raw_params}")
            if pool is not None:
                logger.warning(
                    f"PTDE: recycling worker pool after {len(timed_out)} "
                    f"timeout(s) at {step_label} — a hung worker never "
                    f"rejoins the pool on its own.")
                pool.terminate()
                pool.join()
                # Terminated Pools sit in reference cycles (handler threads,
                # worker sentinels, queue pipes) that only the cyclic GC
                # frees; without an explicit collect each recycle leaks
                # ~2 fds per worker until the process hits EMFILE
                # ("Too many open files") after enough timeouts.
                pool = None
                gc.collect()
                pool = mp.get_context("fork").Pool(
                    actual_cores, initializer=_worker_init)
        if _PTDE_COLLECT_TIMING:
            if rungs is not None:
                for r, k in zip(lps, rungs):
                    rung_times[k].append(r[1])
            return [r[0] for r in lps]
        return lps

    _do_convergence = (min_ess is not None or max_rhat is not None) and n_chains >= 2
    _check_gen = _convergence_check_schedule() if _do_convergence else None
    _next_check = [next(_check_gen)] if _check_gen else [None]

    def _stop_handler(sig, frame):
        if stop_requested[0]:
            raise KeyboardInterrupt   # second signal: abort immediately
        stop_requested[0] = True
        logger.info(
            f"PTDE: stop requested ({signal.Signals(sig).name}) — finishing "
            "current step (send the signal again to abort immediately)")

    # SIGTERM gets the same handler as SIGINT so a batch scheduler (e.g.
    # `qsig -s SIGTERM <job_id>` / `kill -TERM <pid>`) can request the same
    # graceful stop-after-this-step behavior as a Ctrl+C at a terminal,
    # instead of Python's default SIGTERM action (immediate termination,
    # discarding whatever draws were already collected).
    old_sigint = signal.signal(signal.SIGINT, _stop_handler)
    old_sigterm = signal.signal(signal.SIGTERM, _stop_handler)
    try:
        # initial logp evaluations
        flat_starts = [populations[k][i]
                       for k in range(n_temps) for i in range(n_chains)]
        flat_start_labels = [f"rung {k} chain {i}"
                              for k in range(n_temps) for i in range(n_chains)]
        flat_start_rungs = [k for k in range(n_temps) for i in range(n_chains)]
        all_lps = _eval_logps_safe(flat_starts, "initial evaluation",
                                    index_labels=flat_start_labels,
                                    rungs=flat_start_rungs)
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

        # DEO schedule + round-trip diagnostics state. direction[k][i] tags the
        # configuration in slot (k, i) with the last extreme rung it visited;
        # tags travel with the state through swaps (see _record_round_trips).
        swap_round = 0
        n_swap_rounds = 0
        round_trips = [0]
        direction = [[0] * n_chains for _ in range(n_temps)]

        total_steps = tune + draws
        log_every = log_interval or max(1, total_steps // 20)

        for step in range(total_steps):
            phase = "tune" if step < tune else "draw"
            draw_idx = step - tune
            _t0 = time.time()

            # 1. build DE proposals for every chain at every ACTIVE temperature
            #    (rung thinning skips hot rungs on most steps; see _active_rungs)
            props_flat = []
            prop_map = []
            for k in _active_rungs(step, n_temps, _rung_thin_start, _rung_thin_factor):
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
            prop_labels = [f"rung {k} chain {i}" for k, i in prop_map]
            prop_lps = _eval_logps_safe(
                props_flat, f"step {step + 1} ({phase})",
                index_labels=prop_labels,
                rungs=[k for k, i in prop_map])
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
                    # A T=1 chain's lp this large always indicates a model
                    # bug (an unbounded/uncancelled logp term), never real
                    # physics -- no finite dataset's logp reaches 1e12. PTDE
                    # accepts on lp_new > lp_old, so such a bug is a ratchet:
                    # once a chain's lp is inflated this way it can only
                    # climb further, wasting the rest of the run (see
                    # examples/DC2018_128, a real occurrence of exactly this
                    # failure mode). Warn once so it's noticed immediately
                    # rather than discovered post-hoc via identify_modes.
                    if (k == 0 and not warned_implausible_lp
                            and abs(lp_new) > lp_plausibility_ceiling):
                        logger.warning(
                            f"PTDE: T=1 chain {i} lp={lp_new:.3e} exceeds "
                            f"the plausibility ceiling "
                            f"(|lp| > {lp_plausibility_ceiling:g}); this "
                            "almost always means a model bug (e.g. an "
                            "unbounded logp term), not physics -- since "
                            "PTDE only accepts lp increases, this chain "
                            "will likely keep climbing for the rest of the "
                            "run. See outputs.modes.identify_modes, which "
                            "rejects draws on the same ceiling post-hoc.")
                        warned_implausible_lp = True

            # 4. temperature swaps. DEO (deterministic even-odd) schedule by
            #    default -- see _deo_pairs / _record_round_trips. Each pairwise
            #    swap below is the exact same Metropolis test as the legacy
            #    random schedule, so PT invariance is untouched; only WHICH
            #    pairs (and how many) are attempted changes.
            if n_temps > 1 and (step + 1) % swap_interval == 0:
                if swap_schedule == "deo":
                    active = _active_rungs(
                        step, n_temps, _rung_thin_start, _rung_thin_factor)
                    deo_pairs = _deo_pairs(swap_round, n_temps, active)
                    if _rung_thin_factor > 1 and len(deo_pairs) < len(
                            _deo_pairs(swap_round, n_temps)):
                        logger.debug(
                            "PTDE DEO: some swap pairs skipped this round "
                            "(rung thinned inactive)")
                    # Fresh random chain pairing each round: rung-k chain i is
                    # swapped with rung-(k+1) chain perm[i]. Any fixed or
                    # randomized pairing is valid (each pairwise swap satisfies
                    # detailed balance at fixed pairing); a random permutation
                    # spreads swap attempts symmetrically over all n_chains.
                    perm = rng.permutation(n_chains)
                    for (k, kp1) in deo_pairs:
                        for i in range(n_chains):
                            j = int(perm[i])
                            n_swap_propose[k] += 1
                            # (logp_j - logp_i) * (1/T_k - 1/T_{k+1});
                            # T_k < T_{k+1} -> factor > 0 -> accept lp increase.
                            log_a = ((logps[kp1][j] - logps[k][i])
                                     * (1.0 / temperatures[k]
                                        - 1.0 / temperatures[kp1]))
                            if rng.random() < np.exp(min(0.0, log_a)):
                                (populations[k][i], populations[kp1][j]) = (
                                    populations[kp1][j], populations[k][i])
                                (logps[k][i], logps[kp1][j]) = (
                                    logps[kp1][j], logps[k][i])
                                (direction[k][i], direction[kp1][j]) = (
                                    direction[kp1][j], direction[k][i])
                                n_swap_accept[k] += 1
                    swap_round += 1
                else:
                    # Legacy random schedule: one random chain pair per
                    # adjacent rung. Kept for A/B comparison (swap_schedule=
                    # "random"). Round-trip tags are still tracked so the
                    # metric is reported the same way for both schedules.
                    for k in range(n_temps - 1):
                        i = int(rng.integers(n_chains))
                        j = int(rng.integers(n_chains))
                        n_swap_propose[k] += 1
                        log_a = ((logps[k+1][j] - logps[k][i])
                                 * (1.0 / temperatures[k]
                                    - 1.0 / temperatures[k + 1]))
                        if rng.random() < np.exp(min(0.0, log_a)):
                            (populations[k][i], populations[k+1][j]) = (
                                populations[k+1][j], populations[k][i])
                            (logps[k][i], logps[k+1][j]) = (
                                logps[k+1][j], logps[k][i])
                            (direction[k][i], direction[k+1][j]) = (
                                direction[k+1][j], direction[k][i])
                            n_swap_accept[k] += 1
                _record_round_trips(direction, round_trips, n_temps)
                n_swap_rounds += 1

            _t_step = time.time()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"PTDE step {step+1} ({phase})  "
                    f"n_props={len(props_flat)}  "
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

                # Communication-barrier ladder adaptation (Syed et al. 2022),
                # gated to the tuning phase like the gamma adaptation above --
                # re-spacing after tuning would break invariance. Uses this
                # window's swap accept/propose counts, so it must run before
                # the reset below. Independent of adapt_gamma.
                if (phase == "tune" and adapt_ladder and n_temps > 2
                        and n_swap_propose.sum() > 0):
                    new_T = _update_ladder_barrier(
                        temperatures, n_swap_accept, n_swap_propose)
                    if not np.allclose(new_T, temperatures):
                        logger.info(
                            "PTDE ladder (barrier-equalized): "
                            f"T=[{', '.join(f'{t:.1f}' for t in new_T)}]")
                        temperatures = new_T

                # Reset window counters during tune so each adaptation period
                # is a fresh measurement (only matters while adapting).
                if phase == "tune" and (adapt_gamma or adapt_ladder):
                    n_accept[:] = 0
                    n_propose[:] = 0
                    n_swap_accept[:] = 0
                    n_swap_propose[:] = 0

                rt_rate = round_trips[0] / max(n_swap_rounds, 1)
                logger.info(
                    f"PTDE {step+1}/{total_steps} ({phase})  "
                    f"accept=[{', '.join(f'{r:.2f}' for r in ar)}]  "
                    f"γ={gamma:.4f}  "
                    + (f"swap=[{', '.join(f'{r:.2f}' for r in sr)}]  "
                       f"round_trips={round_trips[0]} "
                       f"(rate={rt_rate:.3f}/round)"
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
                    stored_raw, actual_draws, min_ess, max_rhat, stored_lp)
                logger.info(
                    f"PTDE convergence @ {actual_draws} draws: "
                    f"max_rhat={rhat_val:.4f}  min_ess={ess_val:.1f}")
                _next_check[0] = next(_check_gen, None)
                if converged:
                    logger.info("PTDE: convergence criterion met, wrapping up")
                    break

    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
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

    # Flatten (n_chains, draws) -> (n_total,) per raw variable and run the
    # batched converter in chunks (bounds memory for large n_params/draws;
    # chunk_size is independent of param count/shape, only of sample count).
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

    # assemble posterior dict: (n_chains, draws, ...) per variable
    posterior_dict = {}
    for name in out_var_names:
        arr = np.concatenate(out_chunks[name], axis=0)   # (n_total, ...)
        arr = arr.reshape((n_chains, actual_draws) + arr.shape[1:])
        # old per-sample path ran every value through atleast_1d then squeezed
        # a trailing dim-1 for scalar params -- match that convention here.
        if arr.ndim > 2 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        posterior_dict[name] = arr

    idata = az.from_dict({
        "posterior": posterior_dict,
        "sample_stats": {"lp": stored_lp[:, :actual_draws]},
    })

    # Multi-seed provenance (P4): record which solved seed each T=1 chain was
    # started from.  With seeded starts, occupancy weights are initialization
    # artifacts BY DESIGN unless chains mix, so downstream reporting must be
    # able to say "chains 0-3 at seed 0, 4-7 at seed 1".
    # TODO(P4): surface this in outputs/modes.py ModeReport once chains->modes
    # attribution is wired; for now the per-chain attr is the source of truth.
    idata.posterior.attrs["chain_seed_index"] = list(chain_seed_index)
    if len(set(chain_seed_index)) > 1:
        logger.info(f"PTDE multi-seed provenance (chain -> seed): {list(chain_seed_index)}")

    ar_T1 = float(n_accept[0] / max(n_propose[0], 1))
    sr_all = n_swap_accept / np.maximum(n_swap_propose, 1)
    rt_rate = round_trips[0] / max(n_swap_rounds, 1)
    logger.info(
        f"PTDE done: {actual_draws}/{draws} draws  accept(T=1)={ar_T1:.3f}  "
        + (f"swap=[{', '.join(f'{r:.2f}' for r in sr_all)}]  "
           f"round_trips={round_trips[0]} (rate={rt_rate:.3f}/round, "
           f"schedule={swap_schedule})"
           if n_temps > 1 else "")
        + (f"  eval_timeouts={n_eval_timeouts[0]}" if n_eval_timeouts[0] else ""))

    if collect_rung_timing:
        logger.info("PTDE per-rung logp timing (seconds):")
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
