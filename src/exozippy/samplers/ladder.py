"""The temperature ladder: spacing it, walking it, and measuring it.

Both PTDE loops need every function here and NEITHER should own them.  They
lived in `ptde.py`, and `ptde_async.py` imported nine names from it -- a
second sharing channel that `samplers.md` admits is "invisible if you only
grep for `_common`", and which made `ptde.py` simultaneously a sampler and
the other sampler's library.  Anything a sampler imports from its sibling is
a shared module that has not been written yet; this is that module.

Nothing here knows whether proposals are dispatched synchronously.  The
ladder is a property of the temperatures and the swap statistics, not of the
loop that produced them.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
            f"and T=1 -- not draws -- are the bottleneck for TEMPERATURE "
            f"transport; set n_temps: {recommended} and rerun. "
            f"'n_temps: auto' will not get you there: its spacing is "
            f"self-consistent with this criterion only at 0.50 "
            f"adjacent-rung swap acceptance, and the acceptance actually "
            f"achieved here is {1.0 - lam / max(n_temps - 1, 1):.2f}. "
            f"BUT DO NOT EXPECT ROUND TRIPS TO FIX MODE MIXING, which is "
            f"usually what you wanted them for: measured on a 27-D "
            f"Gaussian at a fixed ladder and 2M evaluations, the cold "
            f"chains' far-mode fraction is 0.27-0.35 at a 24-nat barrier "
            f"and 0.05-0.09 at a 78-nat one, at EVERY T_max from 16 to "
            f"8500 -- 0.5 is correct -- while an 8-nat barrier "
            f"equilibrates everywhere, including where there are zero "
            f"round trips, because the proposals cross it directly. A low "
            f"T_max transports and cannot cross; a high one crosses and "
            f"cannot transport. Where the basins are far apart, "
            f"between-mode traffic comes from multi-seed starts, the "
            f"hot-rung suppressed-mode search (`store_hot_chains`), "
            f"per-mode evidence weighting or explicit mode jumps -- and "
            f"note that the hot-rung search's reach is 10*T_max, so "
            f"shortening the ladder to buy round trips costs discovery "
            f"horizon. See notes/pt_round_trip_collapse.txt."
        )
    return lam


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
