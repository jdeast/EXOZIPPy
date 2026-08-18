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

This is the SYNCHRONOUS dispatch loop: every chain advances in lockstep, so
each step waits for the slowest of all n_temps*n_chains evaluations. The
asynchronous variant (exozippy.samplers.ptde_async, sampler.method:
"ptde_async") removes that barrier and is the recommended default for
Op-based models; this module remains the reference implementation (fully
up-to-date DE partner states) for A/B validation. The non-sampling
scaffolding both share lives in exozippy.samplers._common.

Returns arviz.InferenceData compatible with the EXOZIPPy pipeline.
"""

import logging
import signal
import time

import numpy as np

from exozippy.samplers import _common, convergence
from exozippy.samplers._common import (  # noqa: F401
    DE_JITTER,
    LpPlausibilityGuard,
    _eval_logp,
    _make_starts,
    _map_logp,
    _map_logp_timeout,
    _pick_two,
    _shutdown_pool,
    _worker_init,
    de_proposal,
    next_gamma,
)

# Re-exported for compatibility: these historically lived here and are
# imported by tests and by polish.py. The single owner is _common (shared
# with ptde_async so the two samplers cannot drift).
from exozippy.samplers._common import (  # noqa: F401
    DEFAULT_LP_ABS_MAX as _DEFAULT_LP_ABS_MAX,
)

logger = logging.getLogger(__name__)

# The chain-initialization probe lives in exozippy.whitening now (it is also
# the engine behind the data-driven whitening rescale done at model setup);
# PTDE re-probes the already-whitened model, where every scale is ~1, so the
# default (narrow) dynamic range is the right one here.  The old private
# names are re-exported for compatibility.
from exozippy.whitening import (
    _PROBE_FLAT_SCALE,
)
from exozippy.whitening import (  # noqa: E402
    PROBE_TARGET_DELTA as _PROBE_TARGET_DELTA,
)
from exozippy.whitening import (
    probe_scales as _probe_scales,
)
from exozippy.whitening import (
    probe_step_1d as _probe_step_1d,
)


def _geometric_ladder(n_temps, T_max):
    """T_k = T_max^(k/(n_temps-1)), T_0=1 (target), T_{K-1}=T_max."""
    if n_temps == 1:
        return np.array([1.0])
    return T_max ** (np.arange(n_temps) / (n_temps - 1))


def resolve_n_temps(n_temps, n_params, T_max):
    """Resolve the sampler-config ``n_temps``, including ``"auto"``.

    ``auto`` sizes the ladder a priori for adjacent-rung energy overlap on
    a D-dimensional target: between rungs the mean logp shifts by
    ~(D/2)*ln(r) while fluctuating ~sqrt(D/2), so geometric spacing wants
    ln(r) ~ sqrt(2/D), i.e. n = ceil(sqrt(D/2) * ln(T_max)) rungs (floored
    at the historical EXOFASTv2-parity 8).

    THIS IS SELF-CONSISTENT WITH THE DEO CRITERION AT ITS DESIGN POINT, and
    it is worth seeing why before concluding the formula is wrong.  The
    communication barrier is the sum of per-pair REJECTION rates,
    Lambda = (n-1)*rho, so ladder_health_report's requirement
    n >= 2*Lambda + 1 = 2*(n-1)*rho + 1 holds exactly when rho = 0.5 --
    i.e. at the 0.5 adjacent-rung swap acceptance that the overlap argument
    above is chosen to produce.  At its design point this formula IS
    2*Lambda+1; it is not off by a factor of two.

    WHAT ACTUALLY GOES WRONG is that the achieved acceptance is not 0.5.
    Measured on examples/DC2018 event 128 (D = 27):

      T_max   n_temps   Lambda   rho   accept   needs 2L+1   round trips
        200      20     12.02   0.633   0.367       26            61
        200      20     11.52   0.606   0.394       25             8
       8500      34     19.77   0.599   0.401       41             0

    ~0.40 where the derivation implies 0.50, so Lambda lands 20-30% high
    and the ladder 20-30% short.  The gap is in the D/2 Gaussian assumption
    about how Var(logp) varies with T, which is model-specific -- so the
    shortfall is too, and no fixed coefficient here can absorb it.  (Do not
    "fix" this by doubling: 2*33.24+1 = 68 rungs against a measured need of
    41 is 63% waste.)

    For contrast, the SAME 20-rung ladder on the pre-parallax model had
    Lambda = 6.01, so 20 > 2*6.01+1 = 14 was comfortably provisioned, and
    it delivered 1427 round trips.  Nothing about the ladder changed;
    Lambda doubled when pi_E became a real likelihood direction, and round
    trips fell to 8.  That is the whole story of why a ladder that used to
    work stopped working.

    So Lambda is a property of the MODEL, is measured every run, and is the
    only problem-specific number in this decision.  Keep sqrt(D/2) --
    the dimension dependence is derived; an empirical multiplier fitted to
    one example would not be.  The honest upgrade is to MEASURE Lambda in a
    short pilot ladder (it is an average of swap rejection rates and
    converges in a few hundred swap rounds) and size the real ladder at
    2*Lambda+1; see notes/polish_todo.txt.
    """
    if isinstance(n_temps, str):
        if n_temps.strip().lower() != "auto":
            raise ValueError(
                f"n_temps must be an integer or 'auto', got {n_temps!r}"
            )
        n = max(8, int(np.ceil(np.sqrt(n_params / 2.0) * np.log(T_max))))
        logger.info(
            f"n_temps: auto -> {n} rungs "
            f"(D={n_params}, T_max={T_max:g}, sqrt(D/2)*ln(T_max))"
        )
        return n
    return int(n_temps)


def ladder_health_report(temperatures, n_swap_accept, n_swap_propose):
    """Log the measured communication barrier; warn if the ladder chokes.

    Lambda = sum over adjacent-rung pairs of their swap REJECTION rates --
    the empirical global communication barrier of Syed et al. 2022 (JRSS-B).
    Under the non-reversible DEO schedule the T_max<->T=1 round-trip rate
    approaches 1/(2 + 2*Lambda) once n_temps is comfortably above Lambda;
    with n_temps - 1 < ~2*Lambda the ladder itself is the mixing
    bottleneck, and the fix is more rungs -- an EXPLICIT sampler-config
    n_temps at the recommended value, not `n_temps: auto`, which cannot
    know the acceptance it will actually achieve (see resolve_n_temps).
    More draws do not help at all.
    """
    n_temps = len(temperatures)
    prop = np.asarray(n_swap_propose, dtype=float)
    if n_temps < 2 or prop.sum() <= 0:
        return None
    # A pair that was never PROPOSED is unmeasured, not 100%-rejecting.  The
    # guard above is on the TOTAL, so `np.maximum(prop, 1.0)` turned every
    # zero-proposal pair into r_k = 1, the largest barrier a link can have --
    # inflating Lambda and firing the "communication-limited, raise n_temps"
    # warning on a healthy ladder.  Zero proposals are routine (DEO
    # alternates parities; the counters reset every adaptation window).
    # Interpolate over the measured pairs, exactly as _update_ladder_barrier
    # already does for the same reason.
    acc = np.asarray(n_swap_accept, dtype=float)
    measured = prop > 0
    rej = np.zeros(prop.shape, dtype=float)
    rej[measured] = np.clip(1.0 - acc[measured] / prop[measured], 0.0, 1.0)
    if not measured.all():
        pair_idx = np.arange(prop.size)
        rej[~measured] = np.interp(
            pair_idx[~measured], pair_idx[measured], rej[measured]
        )
    lam = float(np.sum(np.clip(rej, 0.0, 1.0)))
    logger.info(
        f"PT ladder health: communication barrier Lambda={lam:.2f} with "
        f"n_temps={n_temps} (DEO round-trip ceiling ~ 1/(2+2*Lambda) = "
        f"{1.0 / (2.0 + 2.0 * lam):.3f} per swap round)"
    )
    recommended = int(np.ceil(2.0 * lam)) + 1
    if (n_temps - 1) < 2.0 * lam:
        logger.warning(
            f"PT ladder is communication-limited: n_temps={n_temps} is "
            f"below ~2*Lambda+1 = {recommended}. Round trips between T_max "
            f"and T=1 -- not draws -- are the mixing bottleneck; set "
            f"n_temps: {recommended} and rerun. 'n_temps: auto' will not "
            f"get you there: its spacing is self-consistent with this "
            f"criterion only at 0.50 adjacent-rung swap acceptance, and the "
            f"acceptance actually achieved here is "
            f"{1.0 - lam / max(n_temps - 1, 1):.2f}, which is what makes "
            f"Lambda higher than the spacing assumed. Lambda is measured, "
            f"so this recommendation is problem-specific; the formula "
            f"cannot be."
        )
    return lam


# OPT-IN improvement tolerance for the gradient-free polish: stop when the
# best point found has gained less than `tol` nats over the last `tol_window`
# sweeps.  DEFAULT OFF (tol=None), and that default is measured, not timid.
#
# The L-BFGS engine stops on its GRADIENT NORM by default (polish.py
# _LBFGS_GTOL) -- an actual statement about the local surface.  No such
# quantity exists here; this engine is gradient-free by construction.  The
# only observable is the best-lp history, and on a real binary-lens surface
# that history is a STAIRCASE: exactly-flat plateaus punctuated by jumps,
# because best_lp is the running maximum of a T=1 Metropolis population that
# spends many sweeps before one member escapes to a better region.  Measured
# on examples/DC2018_128 (2 seeds, 300 sweeps, pop 38):
#
#   window  seed 0 stops at   nats missed    seed 1 stops at   nats missed
#     10        sweep 11          73.2          sweep 20          136.5
#     20        sweep 36          38.8          sweep 30          136.5
#     30        sweep 46          38.8          sweep 40          136.5
#     50        sweep 66          38.8          sweep 158          11.3
#
# and the shortfall is IDENTICAL for tol = 0.05, 0.5 and 2.0 nats, which is
# the proof: the plateaus are exactly flat, so no threshold separates "has
# converged" from "has not jumped yet".  Widening the window only delays the
# same mistake.  A rule that costs 38-137 nats of start quality to save
# sweeps is the opposite of the point -- the polish exists because a start
# far below its basin optimum poisons the whitening probe (polish.py).
#
# So this engine's default stopping criterion stays the step CAP.  These
# constants are the values used when a caller does opt in (tol=..., e.g. for
# a surface known to be smooth, or a re-polish of an already-polished point).
# ABSOLUTE nats, never relative to |lp|: logp carries an arbitrary additive
# normalization, so a relative threshold means something different for every
# model -- the trap documented on polish._LBFGS_FTOL.
POLISH_TOL_NATS = 0.05
POLISH_TOL_WINDOW = 10


# Acceptance the polish adapts gamma toward, and how many proposals it
# measures over before each adjustment.  Same target and same
# (ar/target)**0.5 rule ptde_async uses, deliberately: this engine IS the
# sampler's T=1 move, so a second tuning story would be one more thing to
# keep in sync.
POLISH_TARGET_ACCEPT = 0.2
POLISH_GAMMA_WINDOW = 4  # sweeps between gamma updates


def polish_seed_starts(
    raw_starts,
    logp_fn,
    rng,
    scales,
    n_steps=150,
    pop_size=None,
    gamma=None,
    tol=None,
    tol_window=POLISH_TOL_WINDOW,
    pool=None,
    adapt_gamma=False,
    target_accept=POLISH_TARGET_ACCEPT,
    gamma_window=POLISH_GAMMA_WINDOW,
):
    """Parallel T=1 differential-evolution polish of each seed's raw start.

    For each seed: spawn a small population jittered at ONE scale unit
    around the seed (staying inside its own basin -- this is a local
    refiner, not a search), run DE-MC at T=1 (Metropolis acceptance on
    difference-vector proposals, the same move the sampler itself uses), and
    return the best-lp point visited as the new seed.

    Stopping: ``n_steps`` sweeps.  An improvement tolerance is available
    (``tol`` nats over the last ``tol_window`` sweeps) but is OFF by default
    -- see the POLISH_TOL_NATS comment for the measurement that says why a
    best-lp window cannot be trusted on this engine.

    Rationale: an unpolished solution-estimate seed (e.g. a raw MMEXOFAST
    fit) can start hundreds of nats below its own basin's optimum, and
    chains rationally defect to whichever basin LOOKS best at
    initialization -- on DC2018 event 128, 26 of 27 chains abandoned the
    true branch (ultimately 500 nats better once refined) because its seed
    started ~100 nats below the wrong branch's seed. Gradient-free by
    construction (the binary-lens magnification Op has no analytic
    gradient). Greedy DE optimizers converge similarly but collapse the
    population to a point; Metropolis-at-T=1 costs the same and the caller
    only takes the best point anyway, with _make_starts re-jittering
    chains around it as usual.

    GAMMA ADAPTATION IS AVAILABLE BUT OFF BY DEFAULT, and the default is
    measured, not cautious.  The rationale for wanting it was sound: the
    fixed 2.38/sqrt(2D) step sustains a T=1 acceptance of 0.003-0.004 here
    with 84-88% of sweeps accepting NOTHING, where ptde_async -- running
    this same move on this same model -- adapts to ~0.055 and sustains
    0.18-0.19.  But measured head to head on DC2018 event 128 at equal
    wall clock, adapting gamma makes the polish WORSE:

      sweeps   adaptive gamma        fixed gamma
         150   127090 / 125321      127628 / 124880
        1500   128553 / 126740      129294 / 128584   <- fixed wins by
                                                         +741 / +1844 nats

    The reason is that 0.2 target acceptance is a SAMPLING criterion and
    this stage is OPTIMIZING.  The oversized fixed step lands few proposals
    but the ones it lands are large, and those rare jumps are what climb
    the basin -- exactly the staircase POLISH_TOL_NATS documents.  Tuning
    to 0.2 trades a few big productive leaps for many small well-behaved
    ones: better mixing, worse hill-climbing.  Leave this off unless you
    have measured otherwise on YOUR model.

    The historical note the flag exists for:  ptde_async starts from this same
    2.38/sqrt(2D) rule of thumb and then tunes it; measured on DC2018 event
    128 (D = 27) it settles at ~0.055 against the 0.3239 the formula gives.
    Run fixed at 0.3239 the polish sustained a T=1 acceptance of 0.003-0.004
    with 84-88% of sweeps accepting NOTHING, so most of the population never
    moved off its (badly scaled, pre-whitening) birth position at all: the
    best-lp gains came from a handful of lucky members while the rest sat
    frozen, and best-minus-median stayed ~800-1000 nats instead of the ~D/2
    a converged population shows.  Same rule as the sampler:
    ``gamma *= (ar/target)**0.5``, clipped to a factor 10 per update.

    PARALLELISM.  Proposals are pooled ACROSS SEEDS, not partitioned between
    them.  Each sweep every seed generates all ``pop_size`` proposals from
    its population as frozen at the start of that sweep; the whole batch --
    ``n_seeds * pop_size`` of them -- goes to one shared pool, and workers
    take the next available item.  Two seeds on 64 cores therefore do NOT
    get 32 each, and nothing sits idle while the other seed finishes: the
    only limit is that a batch has n_seeds*pop_size items to hand out.
    Accept/reject is applied afterwards, in the same per-seed order the
    serial engine used.

    That frozen snapshot is the one behavioural change: the serial engine
    updated member i in place, so member i+1 saw it within the same sweep.
    Drawing every proposal in a sweep against the sweep's opening state is
    standard parallel DE-MC (each member's difference vector still comes
    from two OTHER members, and its accept/reject still uses only its own
    lp), and this routine returns the best point VISITED, so what matters is
    coverage rather than the exact chain.

    ``pool`` is anything with a ``map``; ``None`` keeps the serial path,
    which stays byte-for-byte the old behaviour when ``adapt_gamma=False``.

    Returns (polished_starts, dlp_per_seed).
    """
    if isinstance(raw_starts, dict):
        raw_starts = [raw_starts]
    keys = list(raw_starts[0].keys())
    n_params = sum(np.asarray(v).size for v in raw_starts[0].values())
    if pop_size is None:
        pop_size = int(max(8, min(2 * n_params, 64)))
    if gamma is None:
        gamma = 2.38 / np.sqrt(2 * max(n_params, 1))

    n_seeds = len(raw_starts)
    _map = pool.map if pool is not None else lambda f, xs: [f(x) for x in xs]

    def _lps(props):
        """Evaluate a flat list of proposals, pooled across seeds."""
        return [float(v) for v in _map(logp_fn, props)]

    # --- build every seed's population, then score them all in one batch ---
    pops, states = [], []
    for center in raw_starts:
        pop = [{k: np.array(v, dtype=float) for k, v in center.items()}]
        for _ in range(pop_size - 1):
            pop.append(
                {
                    k: center[k]
                    + scales[k] * rng.standard_normal(np.shape(center[k]))
                    for k in keys
                }
            )
        pops.append(pop)
    flat = [p for pop in pops for p in pop]
    flat_lps = _lps(flat)
    for s, center in enumerate(raw_starts):
        lps = np.array(flat_lps[s * pop_size : (s + 1) * pop_size])
        # Non-finite members re-center (a jitter may cross a hard bound).
        for i in np.nonzero(~np.isfinite(lps))[0]:
            pops[s][i] = {
                k: np.array(v, dtype=float) for k, v in center.items()
            }
            lps[i] = lps[0]
        best_i = int(np.nanargmax(lps))
        states.append(
            {
                "lps": lps,
                "best": {k: v.copy() for k, v in pops[s][best_i].items()},
                "best_lp": float(lps[best_i]),
                # member 0 is the exact center, so this is the seed's own lp
                "lp0": float(lps[0]),
                "gamma": float(gamma),
                "n_acc": 0,
                "n_prop": 0,
                "history": [],
                "done": False,
                "steps": 0,
                "stop": "cap",
            }
        )

    for _sweep in range(int(n_steps)):
        live = [s for s in range(n_seeds) if not states[s]["done"]]
        if not live:
            break

        # One batch of proposals from every live seed, against that seed's
        # population as frozen at the start of this sweep.
        batch, index = [], []
        for s in live:
            pop, g = pops[s], states[s]["gamma"]
            for i in range(pop_size):
                j1, j2 = _pick_two(rng, pop_size, i)
                batch.append(
                    {
                        k: pop[i][k]
                        + g * (pop[j1][k] - pop[j2][k])
                        + 1e-4
                        * scales[k]
                        * rng.standard_normal(np.shape(pop[i][k]))
                        for k in keys
                    }
                )
                index.append((s, i))
        lps_batch = _lps(batch)

        for (s, i), prop, lp in zip(index, batch, lps_batch):
            st = states[s]
            st["n_prop"] += 1
            if np.isfinite(lp) and np.log(rng.random()) < lp - st["lps"][i]:
                pops[s][i], st["lps"][i] = prop, lp
                st["n_acc"] += 1
                if lp > st["best_lp"]:
                    st["best_lp"] = lp
                    st["best"] = {k: v.copy() for k, v in prop.items()}

        for s in live:
            st = states[s]
            st["steps"] += 1

            if adapt_gamma and st["steps"] % gamma_window == 0:
                # Nothing accepted at all is next_gamma's shrink branch:
                # the (ar/target)**0.5 rule has no signal to use there.
                ar = st["n_acc"] / max(st["n_prop"], 1)
                st["gamma"] = next_gamma(st["gamma"], ar, target_accept)
                st["n_acc"] = st["n_prop"] = 0

            if tol is None or not tol_window:
                continue
            st["history"].append(st["best_lp"])
            if (
                len(st["history"]) > tol_window
                and st["history"][-1] - st["history"][-1 - tol_window] < tol
            ):
                st["stop"] = "tol"
                st["done"] = True

    polished, dlps = [], []
    for s, st in enumerate(states):
        polished.append(st["best"])
        dlps.append(st["best_lp"] - st["lp0"])
        reason = (
            f"converged: < {tol} nats over {tol_window} sweeps"
            if st["stop"] == "tol"
            else f"hit the {int(n_steps)}-step cap"
        )
        logger.info(
            f"PTDE seed polish: seed {s} lp {st['lp0']:.1f} -> "
            f"{st['best_lp']:.1f} (dlp=+{st['best_lp'] - st['lp0']:.1f}, "
            f"{st['steps']} steps x {pop_size} pop, gamma "
            f"{gamma:.4f}->{st['gamma']:.4f}, {reason})"
        )
    return polished, dlps


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


def _deo_pairs(round_idx, n_temps):
    """Adjacent rung pairs attempted simultaneously in one DEO swap round.

    Even rounds (round_idx even) attempt (0,1),(2,3),(4,5),...; odd rounds
    attempt (1,2),(3,4),(5,6),.... The pairs within a round are disjoint (no
    rung appears twice), so all can be attempted at once, and the alternating
    offset is what makes the index process non-reversible.

    Every pair is always returned -- in particular, rung thinning must NOT
    filter this list. Swaps only exchange already-cached (state, logp) pairs
    and need no fresh evaluation, and because the DEO round parity is
    deterministically coupled to the step counter, filtering by the thinning
    activity pattern permanently removed specific pairs from the schedule
    (e.g. rung_thin_factor=2, swap_interval=1, n_temps=8, thin_start=4 never
    attempted (3,4) or (5,6)), disconnecting the ladder
    (notes/code_review_20260808.txt bug 1.14).
    """
    start = 0 if round_idx % 2 == 0 else 1
    return [(k, k + 1) for k in range(start, n_temps - 1, 2)]


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

    Pairs with ZERO proposals in the window carry no measurement and are
    filled in from their measured neighbours, never scored.  The old
    `1 - accept/max(propose, 1)` read a never-proposed pair as 0/1 = fully
    REJECTING, r_k = 1, the largest barrier a link can have -- so an
    unmeasured link stole ladder resolution from the links that had actually
    been measured.  Zero proposals are routine: the DEO schedule alternates
    even and odd pairs by round, the counters are reset every adaptation
    window, and rung thinning lengthens the windows in which a given parity
    never came up.  Scoring the gap 0 instead is equally wrong in the other
    direction (it claims perfect mixing, collapsing those two rungs
    together) and, because the gap then drops out of the total, silently
    rescales every other pair's share.  Linear interpolation over the pair
    index keeps the total honest and preserves the barrier PROFILE, which
    varies smoothly along a smooth ladder; np.interp clamps at the ends, so
    an unmeasured end pair inherits its nearest measured neighbour.  With a
    single measured pair every r_k is that one value, the ladder is already
    equal-share, and the update is exactly a no-op -- the right answer from
    one datum.
    """
    n_temps = len(temperatures)
    if n_temps < 3:
        return np.asarray(temperatures, dtype=float)
    prop = np.asarray(swap_propose, dtype=float)
    acc = np.asarray(swap_accept, dtype=float)
    measured = prop > 0
    if not measured.any():
        return np.asarray(temperatures, dtype=float)
    r = np.zeros(prop.shape, dtype=float)
    r[measured] = np.clip(1.0 - acc[measured] / prop[measured], 0.0, 1.0)
    if not measured.all():
        pair_idx = np.arange(prop.size)
        r[~measured] = np.interp(
            pair_idx[~measured], pair_idx[measured], r[measured]
        )
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
    new_beta[0] = beta[0]  # pin target rung (T=1)
    new_beta[-1] = beta[-1]  # pin hottest rung (T=T_max)
    return 1.0 / new_beta


def _active_rungs(step, n_temps, thin_start, thin_factor):
    """Rung indices that propose a DE move at this step.

    Rungs below thin_start always propose; rungs at or above it only
    propose every thin_factor-th step (thin_factor<=1 -> always, i.e. no
    thinning). Swaps are unaffected by thinning -- they only exchange
    already-cached (population, logp) pairs between adjacent rungs, so a
    rung that skipped its own DE move this step can still participate in
    a swap using its last-computed logp.

    Rationale (the slow-evaluation tail measured at the top of
    notes/hpc_optimization.txt; its P12, cited here until 2026-08, has
    since been pruned from that note): PTDE's per-step wall time is
    gated by the SLOWEST of all n_temps*n_chains proposals. Hot rungs
    (large T) explore a heavily flattened target and routinely draw
    parameter combinations that are individually expensive to evaluate but
    scientifically irrelevant (only the T=1 rung's draws are kept). Thinning
    them directly cuts the number of chances per step to draw from that
    expensive tail, at the cost of slower mixing for swap partners.
    """
    if thin_factor <= 1:
        return list(range(n_temps))
    return [
        k for k in range(n_temps) if k < thin_start or step % thin_factor == 0
    ]


def _convergence_check_schedule(min_draws=100, growth=0.9):
    """Yield cumulative draw counts at which to run a convergence check.

    Positions: round(min_draws / growth**j) for j=0,1,2,...
    Default (growth=0.9): 100, 111, 123, 137, 152, ...
    Gaps grow by ~11% each check, so we check frequently early and less often later.
    """
    j, prev = 0, 0
    while True:
        n = round(min_draws / growth**j)
        if n > prev:
            yield n
            prev = n
        j += 1


def _safe_progress(progress_callback, state):
    """Invoke an optional GUI progress hook without ever letting it break the run.

    The callback (see exozippy.gui.status.GuiReporter.progress_callback) writes
    monitoring artifacts to disk; a filesystem hiccup there must never abort
    sampling, so any exception is logged and swallowed.
    """
    if progress_callback is None:
        return
    try:
        progress_callback(state)
    except Exception:
        logger.warning(
            "PTDE: progress_callback raised; continuing sampling",
            exc_info=True,
        )


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
        return convergence.converged_on_tail(posterior, lp, min_ess, max_rhat)
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
    raw_scales=None,
    gamma=None,
    target_accept=0.20,
    adapt_gamma=True,
    de_jitter=DE_JITTER,
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
    progress_callback=None,
):
    """
    Parallel Tempering + Differential Evolution sampler.

    Parameters
    ----------
    model : PyMC model (from system.build_model())
    system : EXOZIPPy System (MAP start + raw→physical conversion)
    draws, tune : int
    n_temps : int | "auto"  — temperature rungs (default 8, EXOFASTv2
               parity); "auto" → max(8, ceil(sqrt(D/2)·ln(T_max))), sized
               for adjacent-rung energy overlap (see resolve_n_temps).
               Every run also logs the measured communication barrier and
               warns when the ladder is the mixing bottleneck
               (ladder_health_report).
    T_max : float   — hottest temperature (default 200, EXOFASTv2 parity)
    n_chains : int | None  — chains per temperature rung;
               None → 2 × n_params (standard DE minimum for good mixing)
    cores : int | None  — CPU cores for parallel logp evaluation. NOT one
               core per chain: the pool is capped at the proposal count
               (n_temps x n_chains) but sized from the MACHINE, as
               max(1, min(0.75*n_cpu, n_cpu - 1)) -- see
               _common.create_pool, and run.py, which applies the same
               formula when the sampler block leaves `cores` unset. A
               64-core box therefore runs a 320-proposal step 47 at a time,
               not 320. Passing a value above the physical count is allowed
               and warns.
    initvals : list[dict] | None  — n_chains raw-space starting dicts for
               the T=1 rung; replicated across hotter rungs.
               None → probe-based from MAP (same logic as DEMetropolis block).
    gamma : float   — DE proposal scale; None → 2.38 / sqrt(2 × n_params)
    target_accept : float  — T=1 acceptance rate target for gamma adaptation (default 0.20)
    adapt_gamma : bool     — scale gamma toward target_accept during tune (default True)
    de_jitter : float  — epsilon term of the ter Braak 2006 DE move, in raw
               (whitened) units where every scale is ~1 (default
               _common.DE_JITTER = 1e-4; 0 disables). Without it, sampling
               is confined to the affine hull of the initial T=1 starts,
               which silently truncates the posterior when
               n_chains <= n_params.
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
               irrelevant (see _active_rungs for the measurement this
               is argued from). Swaps are
               unaffected -- they exchange cached (population, logp) pairs
               and need no new evaluation.
    rung_thin_start : int | None  — first rung index subject to thinning;
               None -> n_temps // 2. Clamped to >= 1: the T=1 rung (index 0,
               the only one whose draws are kept) is never thinned.
    collect_rung_timing : bool  — diagnostic: record per-call wall time and
               attribute it to a rung, logging a summary
               (count/median/mean/p90/max per rung) when sampling finishes.
               Default False (zero overhead when off). This is the
               measurement the sampler's optimization work is argued from --
               it is what localized the slow-evaluation tail to the top two
               rungs of DC2018_128 (see the per-rung timing table at the top
               of notes/hpc_optimization.txt, and the 6.4.x block of
               notes/code_review_20260814.txt, which supersedes the P13 this
               line used to cite; P13 itself has since been pruned from that
               note).
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
    progress_callback : callable | None  — optional GUI progress hook invoked
               at each geometric convergence check (bounded overhead) with a
               state dict {n_draws, n_chains, max_rhat, min_ess, elapsed_s,
               stop_reason?} plus, for snapshot writers, stored_raw/stored_lp/
               raw_var_names referencing the live T=1 draw buffers. Exceptions
               raised by the callback are logged and swallowed (see
               _safe_progress), so a monitoring failure never aborts the fit.

    Returns
    -------
    arviz.InferenceData with posterior and sample_stats["lp"] from T=1 chains.
    """
    lp_guard = LpPlausibilityGuard(lp_plausibility_ceiling, "PTDE", logger)

    if swap_schedule not in ("deo", "random"):
        raise ValueError(
            f"swap_schedule must be 'deo' or 'random', got {swap_schedule!r}"
        )

    rng = np.random.default_rng(seed)

    # parameter bookkeeping -- before the ladder, since n_temps may be
    # "auto" (sized from the parameter count; see resolve_n_temps).
    raw_start = system.get_raw_start(model)
    model_keys = list(raw_start.keys())
    n_params = sum(v.size for v in raw_start.values())
    n_temps = resolve_n_temps(n_temps, n_params, T_max)
    temperatures = _geometric_ladder(n_temps, T_max)

    _rung_thin_factor = max(1, int(rung_thin_factor))
    _rung_thin_start = (
        n_temps // 2 if rung_thin_start is None else int(rung_thin_start)
    )
    _rung_thin_start = max(1, min(_rung_thin_start, n_temps))  # never thin T=1
    if _rung_thin_factor > 1 and _rung_thin_start < n_temps:
        logger.info(
            f"PTDE: rung thinning enabled — rungs >= {_rung_thin_start} "
            f"(of {n_temps}) propose every {_rung_thin_factor} steps"
        )

    # compile logp ONCE; install in _common BEFORE forking workers so fork
    # children inherit it (copy-on-write; see _common.set_worker_globals)
    logp_fn = model.compile_logp()
    _common.set_worker_globals(logp_fn, collect_rung_timing)

    # compile raw -> physical conversions ONCE (single-sample and batched;
    # see _common.compile_conversions for the rationale).
    raw_to_phys, raw_to_phys_batched, raw_var_names, out_var_names = (
        _common.compile_conversions(model)
    )

    # 2 * n_params is the standard DE population for good mixing; the floor
    # and the warning both live in _common.resolve_n_chains.
    n_chains = _common.resolve_n_chains(n_chains, n_params, "PTDE", logger)
    if gamma is None:
        gamma = 2.38 / np.sqrt(2 * n_params)
    logger.info(
        f"PTDE: {n_params} params, {n_chains} chains/rung, γ={gamma:.4f}"
    )

    # initialize populations
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

    # ensemble start plots (T=1 starts only; raw→physical via the batched fn)
    _common.plot_start_ensemble(
        system,
        t1_starts,
        raw_to_phys_batched,
        raw_var_names,
        out_var_names,
        plot_prefix,
        logger,
    )

    # Replicate T=1 starts to all rungs; hotter chains spread quickly during tune
    populations = [
        [
            {k: v.copy() for k, v in t1_starts[i % n_chains].items()}
            for i in range(n_chains)
        ]
        for _ in range(n_temps)
    ]

    # start pool AFTER set_worker_globals so fork children inherit the logp fn
    total_proposals = n_temps * n_chains
    pool, actual_cores = _common.create_pool(
        cores, total_proposals, "PTDE", logger
    )
    logger.info(
        f"PTDE: {n_temps} rungs × {n_chains} chains = {total_proposals} proposals/step, "
        f"{actual_cores} cores  "
        f"T=[{', '.join(f'{t:.1f}' for t in temperatures)}]"
    )
    _common.warn_serial_eval_timeout(
        eval_timeout, pool, actual_cores, "PTDE", logger
    )

    # Early-stop state: mutable list so the closure can write back to us.
    stop_requested = False
    actual_draws = 0
    start_time = time.time()
    n_eval_timeouts = 0  # incremented by _eval_logps_safe (nonlocal)
    rung_times = [
        [] for _ in range(n_temps)
    ]  # per-rung wall times (collect_rung_timing only)

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
        nonlocal pool, n_eval_timeouts
        if eval_timeout is None:
            raw = _map_logp(pool, proposals)
            if collect_rung_timing:
                lps = [r[0] for r in raw]
                if rungs is not None:
                    for r, k in zip(raw, rungs):
                        rung_times[k].append(r[1])
                return lps
            return raw

        lps, timed_out = _map_logp_timeout(pool, proposals, eval_timeout)
        if timed_out:
            n_eval_timeouts += len(timed_out)
            for idx in timed_out:
                phys_params, raw_params = _common.describe_proposal(
                    proposals[idx], raw_to_phys, raw_var_names, out_var_names
                )
                who = (
                    index_labels[idx]
                    if index_labels is not None
                    else f"proposal {idx}"
                )
                logger.error(
                    f"PTDE: logp call exceeded eval_timeout={eval_timeout:.0f}s "
                    f"at {step_label} ({who}) — rejecting this proposal.\n"
                    f"  physical params: {phys_params}\n"
                    f"  raw params: {raw_params}"
                )
            if pool is not None:
                logger.warning(
                    f"PTDE: recycling worker pool after {len(timed_out)} "
                    f"timeout(s) at {step_label} — a hung worker never "
                    f"rejoins the pool on its own."
                )
                pool = _common.recycle_pool(pool, actual_cores)
        if collect_rung_timing:
            if rungs is not None:
                for r, k in zip(lps, rungs):
                    rung_times[k].append(r[1])
            return [r[0] for r in lps]
        return lps

    _do_convergence = (
        min_ess is not None or max_rhat is not None
    ) and n_chains >= 2
    _check_gen = _convergence_check_schedule() if _do_convergence else None
    _next_check = next(_check_gen) if _check_gen else None

    def _stop_handler(sig, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt  # second signal: abort immediately
        stop_requested = True
        logger.info(
            f"PTDE: stop requested ({signal.Signals(sig).name}) — finishing "
            "current step (send the signal again to abort immediately)"
        )

    _sig_token = _common.install_stop_handlers(_stop_handler)
    try:
        # initial logp evaluations
        flat_starts = [
            populations[k][i] for k in range(n_temps) for i in range(n_chains)
        ]
        flat_start_labels = [
            f"rung {k} chain {i}"
            for k in range(n_temps)
            for i in range(n_chains)
        ]
        flat_start_rungs = [k for k in range(n_temps) for i in range(n_chains)]
        all_lps = _eval_logps_safe(
            flat_starts,
            "initial evaluation",
            index_labels=flat_start_labels,
            rungs=flat_start_rungs,
        )
        logps = [
            [all_lps[k * n_chains + i] for i in range(n_chains)]
            for k in range(n_temps)
        ]
        logger.info(
            f"PTDE: T=1 initial lp  "
            f"min={min(logps[0]):.1f}  max={max(logps[0]):.1f}"
        )

        # storage: raw values from T=1 chains only
        stored_raw = {
            k: np.zeros((n_chains, draws) + raw_start[k].shape)
            for k in model_keys
        }
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

            # Acceptance/swap counters restart at the tune -> draw boundary,
            # UNCONDITIONALLY.  They are diagnostics -- nothing statistical
            # reads them outside the tune-gated gamma and ladder adaptations
            # below -- and the wrap-up ladder_health_report is a statement
            # about the FINAL ladder, which is only true if the window it
            # measures begins where adaptation ends.  The old rule reset
            # them inside the adapt branch, at log_every boundaries, so the
            # measured window carried however much of tune fell after the
            # last boundary, and with BOTH adaptations off it carried the
            # whole of tune (review 3.4.2).  ptde_async never resets and
            # says so where it reports.
            if step == tune:
                n_accept[:] = 0
                n_propose[:] = 0
                n_swap_accept[:] = 0
                n_swap_propose[:] = 0

            # 1. build DE proposals for every chain at every ACTIVE temperature
            #    (rung thinning skips hot rungs on most steps; see _active_rungs)
            props_flat = []
            prop_map = []
            for k in _active_rungs(
                step, n_temps, _rung_thin_start, _rung_thin_factor
            ):
                pop_k = populations[k]
                for i in range(n_chains):
                    props_flat.append(
                        de_proposal(
                            rng, pop_k, i, gamma, model_keys, jitter=de_jitter
                        )
                    )
                    prop_map.append((k, i))
            _t_build = time.time()

            # 2. evaluate all logps in parallel
            prop_labels = [f"rung {k} chain {i}" for k, i in prop_map]
            prop_lps = _eval_logps_safe(
                props_flat,
                f"step {step + 1} ({phase})",
                index_labels=prop_labels,
                rungs=[k for k, i in prop_map],
            )
            _t_eval = time.time()

            # 3. Metropolis accept/reject at effective temperature T_k
            for idx, (k, i) in enumerate(prop_map):
                T = temperatures[k]
                lp_new = prop_lps[idx]
                n_propose[k] += 1
                if np.isfinite(lp_new) and rng.random() < np.exp(
                    min(0.0, (lp_new - logps[k][i]) / T)
                ):
                    populations[k][i] = props_flat[idx]
                    logps[k][i] = lp_new
                    n_accept[k] += 1
                    # Runaway-lp early detection (see LpPlausibilityGuard).
                    if k == 0:
                        lp_guard.check(i, lp_new)

            # 4. temperature swaps. DEO (deterministic even-odd) schedule by
            #    default -- see _deo_pairs / _record_round_trips. Each pairwise
            #    swap below is the exact same Metropolis test as the legacy
            #    random schedule, so PT invariance is untouched; only WHICH
            #    pairs (and how many) are attempted changes. Swaps are never
            #    filtered by rung thinning: they exchange cached (state, logp)
            #    pairs and need no fresh evaluation (bug 1.14 -- filtering by
            #    the thinning pattern permanently disconnected the ladder).
            if n_temps > 1 and (step + 1) % swap_interval == 0:
                if swap_schedule == "deo":
                    deo_pairs = _deo_pairs(swap_round, n_temps)
                    # Fresh random chain pairing each round: rung-k chain i is
                    # swapped with rung-(k+1) chain perm[i]. Any fixed or
                    # randomized pairing is valid (each pairwise swap satisfies
                    # detailed balance at fixed pairing); a random permutation
                    # spreads swap attempts symmetrically over all n_chains.
                    perm = rng.permutation(n_chains)
                    for k, kp1 in deo_pairs:
                        for i in range(n_chains):
                            j = int(perm[i])
                            n_swap_propose[k] += 1
                            # (logp_j - logp_i) * (1/T_k - 1/T_{k+1});
                            # T_k < T_{k+1} -> factor > 0 -> accept lp increase.
                            log_a = (logps[kp1][j] - logps[k][i]) * (
                                1.0 / temperatures[k] - 1.0 / temperatures[kp1]
                            )
                            if rng.random() < np.exp(min(0.0, log_a)):
                                populations[k][i], populations[kp1][j] = (
                                    populations[kp1][j],
                                    populations[k][i],
                                )
                                logps[k][i], logps[kp1][j] = (
                                    logps[kp1][j],
                                    logps[k][i],
                                )
                                direction[k][i], direction[kp1][j] = (
                                    direction[kp1][j],
                                    direction[k][i],
                                )
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
                        log_a = (logps[k + 1][j] - logps[k][i]) * (
                            1.0 / temperatures[k] - 1.0 / temperatures[k + 1]
                        )
                        if rng.random() < np.exp(min(0.0, log_a)):
                            populations[k][i], populations[k + 1][j] = (
                                populations[k + 1][j],
                                populations[k][i],
                            )
                            logps[k][i], logps[k + 1][j] = (
                                logps[k + 1][j],
                                logps[k][i],
                            )
                            direction[k][i], direction[k + 1][j] = (
                                direction[k + 1][j],
                                direction[k][i],
                            )
                            n_swap_accept[k] += 1
                _record_round_trips(direction, round_trips, n_temps)
                n_swap_rounds += 1

            _t_step = time.time()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"PTDE step {step + 1} ({phase})  "
                    f"n_props={len(props_flat)}  "
                    f"total={_t_step - _t0:.3f}s  "
                    f"build={_t_build - _t0:.3f}s  "
                    f"eval={_t_eval - _t_build:.3f}s  "
                    f"rest={_t_step - _t_eval:.3f}s  "
                    f"T1_lp=[{min(logps[0]):.1f},{max(logps[0]):.1f}]"
                )

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
                    # Scale gamma toward target_accept on the T=1 window
                    # rate (_common.next_gamma owns the rule).  A window
                    # that accepted NOTHING is skipped rather than shrunk:
                    # here that is one log interval of a tempered sampler,
                    # where the T=1 rung can legitimately stall while the
                    # ladder is still equilibrating.
                    ar_T1 = ar[0]
                    if ar_T1 > 0:
                        gamma_new = next_gamma(gamma, ar_T1, target_accept)
                        if abs(gamma_new - gamma) / gamma > 0.01:
                            logger.info(
                                f"PTDE gamma: {gamma:.4f} → {gamma_new:.4f} "
                                f"(T=1 accept={ar_T1:.3f}, target={target_accept:.2f})"
                            )
                            gamma = gamma_new

                # Communication-barrier ladder adaptation (Syed et al. 2022),
                # gated to the tuning phase like the gamma adaptation above --
                # re-spacing after tuning would break invariance. Uses this
                # window's swap accept/propose counts, so it must run before
                # the reset below. Independent of adapt_gamma.
                if (
                    phase == "tune"
                    and adapt_ladder
                    and n_temps > 2
                    and n_swap_propose.sum() > 0
                ):
                    new_T = _update_ladder_barrier(
                        temperatures, n_swap_accept, n_swap_propose
                    )
                    if not np.allclose(new_T, temperatures):
                        logger.info(
                            "PTDE ladder (barrier-equalized): "
                            f"T=[{', '.join(f'{t:.1f}' for t in new_T)}]"
                        )
                        temperatures = new_T

                # Reset window counters during tune so each adaptation period
                # is a fresh measurement (only matters while adapting; the
                # tune -> draw boundary reset at the top of the loop is
                # unconditional and is what the wrap-up report relies on).
                if phase == "tune" and (adapt_gamma or adapt_ladder):
                    n_accept[:] = 0
                    n_propose[:] = 0
                    n_swap_accept[:] = 0
                    n_swap_propose[:] = 0

                rt_rate = round_trips[0] / max(n_swap_rounds, 1)
                logger.info(
                    f"PTDE {step + 1}/{total_steps} ({phase})  "
                    f"accept=[{', '.join(f'{r:.2f}' for r in ar)}]  "
                    f"γ={gamma:.4f}  "
                    + (
                        f"swap=[{', '.join(f'{r:.2f}' for r in sr)}]  "
                        f"round_trips={round_trips[0]} "
                        f"(rate={rt_rate:.3f}/round)"
                        if n_temps > 1
                        else ""
                    )
                )

            # 7. early-stop checks
            if stop_requested:
                if actual_draws == 0:
                    logger.warning(
                        "PTDE: stop requested during tune — no draws to save"
                    )
                    raise KeyboardInterrupt
                logger.info(
                    f"PTDE: stopping after {actual_draws} draws (user interrupt)"
                )
                break

            if maxtime is not None and (time.time() - start_time) > maxtime:
                if actual_draws == 0:
                    logger.warning(
                        f"PTDE: time limit {maxtime:.0f}s reached during tune — no draws to save"
                    )
                    raise KeyboardInterrupt
                logger.info(
                    f"PTDE: wall-clock limit {maxtime:.0f}s reached "
                    f"after {actual_draws} draws"
                )
                break

            if (
                phase == "draw"
                and _next_check is not None
                and actual_draws >= _next_check
            ):
                converged, rhat_val, ess_val = _check_convergence(
                    stored_raw, actual_draws, min_ess, max_rhat, stored_lp
                )
                logger.info(
                    f"PTDE convergence @ {actual_draws} draws: "
                    f"max_rhat={rhat_val:.4f}  min_ess={ess_val:.1f}"
                )
                # GUI progress hook: fires at each geometric check, so overhead
                # stays bounded. Passes the live T=1 draw buffers by reference
                # so a snapshot writer can downsample them itself.
                _safe_progress(
                    progress_callback,
                    {
                        "n_draws": actual_draws,
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
                _next_check = next(_check_gen, None)
                if converged:
                    logger.info("PTDE: convergence criterion met, wrapping up")
                    break

    finally:
        _common.restore_stop_handlers(_sig_token)
        if pool is not None:
            # _shutdown_pool, not close()+join(): close() waits for every
            # in-flight evaluation to finish, and _worker_init has the
            # workers IGNORE SIGTERM, so a worker wedged in a pathological
            # logp (exactly the case eval_timeout exists for, and the
            # default is eval_timeout=None) blocks join() forever -- on the
            # ABORT path, after Ctrl+C or maxtime, with the draws already
            # collected and nothing left to wait for.  _shutdown_pool
            # terminates and escalates to SIGKILL past a grace period.
            # Safe on the normal path too: every map has completed there,
            # so there is nothing for close() to have waited on (review
            # 2.4.1; ptde_async has done this since it was written).
            _common._shutdown_pool(pool)

    # convert raw → physical for every stored draw
    if actual_draws == 0:
        raise RuntimeError(
            "PTDE: sampling stopped during tune — no draws were collected"
        )
    if actual_draws < draws:
        logger.info(
            f"PTDE: early stop — {actual_draws}/{draws} draws collected"
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
        "PTDE",
        logger,
    )

    # Ladder communication statistics, stamped on the trace so the mode
    # report can quote them as context (see stamp_and_log_run_summary).
    _common.stamp_and_log_run_summary(
        idata,
        "PTDE",
        logger,
        actual_draws=actual_draws,
        draws=draws,
        n_accept=n_accept,
        n_propose=n_propose,
        n_swap_accept=n_swap_accept,
        n_swap_propose=n_swap_propose,
        round_trips=round_trips[0],
        n_swap_rounds=n_swap_rounds,
        n_temps=n_temps,
        swap_schedule=swap_schedule,
        rate_unit="round",
        extras=(
            [f"  eval_timeouts={n_eval_timeouts}"] if n_eval_timeouts else []
        ),
    )
    # Post-tune swap counters (they are zeroed at the tune -> draw boundary;
    # see the step loop), so this measures the FINAL ladder's communication
    # barrier and not the ladders that preceded it.
    ladder_health_report(temperatures, n_swap_accept, n_swap_propose)

    if collect_rung_timing:
        _common.log_rung_timing(rung_times, temperatures, "PTDE", logger)

    return idata
