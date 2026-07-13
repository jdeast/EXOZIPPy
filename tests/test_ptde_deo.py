"""Tests for the Deterministic Even-Odd (DEO) swap schedule, round-trip
diagnostics, and communication-barrier ladder adaptation added to the PTDE
samplers (Syed et al. 2022, "Non-reversible parallel tempering").

Mirrors tests/test_ptde.py's AAA / Given-When-Then style and its cheap
gradient-free toy targets. The comparative tests use a well-separated 1-D
bimodal mixture -- the regime where PT round trips (cold -> hot -> cold) are
the ONLY mechanism that transports a chain between modes, so the DEO
schedule's O(n_temps) vs random-pair O(n_temps^2) round-trip scaling is
directly observable.
"""
import io
import logging
import re

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from exozippy.samplers.ptde import (
    _deo_pair_sequence,
    _deo_pairs,
    _record_round_trips,
    _update_ladder_barrier,
    ptde_sample,
)
from exozippy.samplers.ptde_async import ptde_async_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MinimalSystem:
    """Minimal system stub: supplies raw_start from the model."""
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _bimodal_model(sep=4.0, s=1.0):
    """Symmetric two-mode target at x = +/- sep, each of width s.

    Uses an UNtransformed broad-normal base variable (so the raw sampling
    space name matches the free-RV name, as the toy _MinimalSystem assumes)
    plus a log-sum-exp Potential carrying the actual bimodal structure. The
    broad base prior (sigma=10) is nearly flat over the modes.
    """
    with pm.Model() as model:
        x = pm.Normal("x", mu=0.0, sigma=10.0)
        c1 = -0.5 * ((x + sep) / s) ** 2
        c2 = -0.5 * ((x - sep) / s) ** 2
        pm.Potential("mix", pm.math.logsumexp(pm.math.stack([c1, c2])))
    return model


def _run_and_capture(sched, seed, **kw):
    """Run ptde_sample with a given swap_schedule, capturing its log output.

    Returns (idata, round_trips, swap_rates) parsed from the 'PTDE done:'
    summary line -- the round-trip count and per-rung swap-acceptance array
    are only exposed through the log, so the test reads them back the same
    way an operator would.
    """
    logger = logging.getLogger("exozippy.samplers.ptde")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        idata = ptde_sample(
            _bimodal_model(), _MinimalSystem(), draws=400, tune=300,
            n_temps=8, T_max=100.0, n_chains=8, cores=1, seed=seed,
            log_interval=100000, swap_schedule=sched,
            min_ess=None, max_rhat=None, **kw)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    txt = buf.getvalue()
    rt = int(re.search(r"round_trips=(\d+)", txt).group(1))
    swaps = [float(v) for v in
             re.search(r"swap=\[([^\]]+)\]", txt).group(1).split(",")]
    return idata, rt, np.array(swaps)


# ---------------------------------------------------------------------------
# (a) Schedule generator: even/odd alternation, correct pairs, thinned skip
# ---------------------------------------------------------------------------

def test_deo_pairs_even_round_starts_at_zero():
    """
    Given an even swap round on an 8-rung ladder,
    When _deo_pairs is called,
    Then it returns the disjoint pairs (0,1),(2,3),(4,5),(6,7).
    """
    assert _deo_pairs(0, 8) == [(0, 1), (2, 3), (4, 5), (6, 7)]


def test_deo_pairs_odd_round_starts_at_one():
    """
    Given an odd swap round on an 8-rung ladder,
    When _deo_pairs is called,
    Then it returns the disjoint pairs (1,2),(3,4),(5,6) (rungs 0 and 7 idle).
    """
    assert _deo_pairs(1, 8) == [(1, 2), (3, 4), (5, 6)]


def test_deo_pairs_alternate_between_rounds():
    """
    Given consecutive swap rounds,
    When _deo_pairs is called,
    Then even and odd rounds produce different (alternating) pair sets --
      this alternation is what makes the index process non-reversible.
    """
    assert _deo_pairs(0, 8) != _deo_pairs(1, 8)
    assert _deo_pairs(0, 8) == _deo_pairs(2, 8)
    assert _deo_pairs(1, 8) == _deo_pairs(3, 8)


def test_deo_pairs_are_disjoint_within_a_round():
    """
    Given any round on any ladder size,
    When _deo_pairs is called,
    Then no rung index appears in more than one pair (so all pairs in the
      round can be attempted simultaneously).
    """
    for round_idx in (0, 1):
        for n_temps in (5, 6, 7, 8):
            flat = [r for pair in _deo_pairs(round_idx, n_temps) for r in pair]
            assert len(flat) == len(set(flat))


def test_deo_pairs_odd_ladder_drops_dangling_rung():
    """
    Given an odd number of rungs (7),
    When _deo_pairs is called on an even round,
    Then the last rung (6) has no partner and is simply omitted.
    """
    assert _deo_pairs(0, 7) == [(0, 1), (2, 3), (4, 5)]


def test_deo_pairs_skips_thinned_inactive_rungs():
    """
    Given a set of active rungs that excludes some thinned (inactive) rungs,
    When _deo_pairs is called with active_rungs,
    Then any pair touching an inactive rung is dropped rather than reverting
      to a random pairing (preserving the non-reversible index flow).
    """
    # Only rungs 0-3 active: on an even round, (4,5) and (6,7) must drop.
    assert _deo_pairs(0, 8, active_rungs={0, 1, 2, 3}) == [(0, 1), (2, 3)]
    # A pair with only one active endpoint is also dropped.
    assert _deo_pairs(0, 8, active_rungs={0, 1, 2}) == [(0, 1)]


def test_deo_pair_sequence_even_pairs_before_odd():
    """
    Given the async cycling sequence generator,
    When _deo_pair_sequence is called,
    Then every even-offset pair index is exhausted before any odd-offset one
      (the event-time analog of the synchronous even/odd rounds).
    """
    assert _deo_pair_sequence(8) == [0, 2, 4, 6, 1, 3, 5]
    assert _deo_pair_sequence(7) == [0, 2, 4, 1, 3, 5]


# ---------------------------------------------------------------------------
# Round-trip diagnostics helper
# ---------------------------------------------------------------------------

def test_record_round_trips_tags_extreme_rungs():
    """
    Given a fresh direction array (all neutral) on a 3-rung ladder,
    When _record_round_trips is called,
    Then the cold rung (0) is tagged +1 and the hot rung (2) is tagged -1,
      with no round trip counted yet.
    """
    direction = [[0], [0], [0]]
    round_trips = [0]
    _record_round_trips(direction, round_trips, n_temps=3)
    assert direction[0][0] == 1
    assert direction[2][0] == -1
    assert round_trips[0] == 0


def test_record_round_trips_counts_full_cold_hot_cold_excursion():
    """
    Given a configuration that has reached the hot end (tag -1) and is then
      transported (via a swap) into the cold rung,
    When _record_round_trips is called,
    Then exactly one round trip is counted and the config is re-tagged +1.
    """
    direction = [[0], [0], [0]]
    round_trips = [0]
    _record_round_trips(direction, round_trips, n_temps=3)   # cold=+1, hot=-1
    # Swap: move the hot-tagged config into the cold slot (as an accepted swap
    # would, carrying its direction tag with it).
    direction[0][0], direction[2][0] = direction[2][0], direction[0][0]
    _record_round_trips(direction, round_trips, n_temps=3)
    assert round_trips[0] == 1
    assert direction[0][0] == 1


def test_record_round_trips_is_idempotent():
    """
    Given a settled direction state,
    When _record_round_trips is called repeatedly with no swaps in between,
    Then no additional round trips are counted (safe for the async sampler to
      call after every swap event).
    """
    direction = [[0, 0], [0, 0], [0, 0]]
    round_trips = [0]
    for _ in range(5):
        _record_round_trips(direction, round_trips, n_temps=3)
    assert round_trips[0] == 0


# ---------------------------------------------------------------------------
# (b) Invariance smoke test: DEO and random agree on moments; DEO wins on RT
# ---------------------------------------------------------------------------

def test_deo_and_random_agree_on_bimodal_moments():
    """
    Given a symmetric 1-D bimodal target (modes at x = +/- 4),
    When ptde_sample runs with the DEO and the random swap schedules,
    Then both recover the same (symmetric) posterior moments within MC error
      -- the DEO schedule only changes WHICH swap pairs are attempted, never
      the per-swap Metropolis test, so it must not bias the posterior.
    """
    idata_deo, _, _ = _run_and_capture("deo", seed=1)
    idata_rnd, _, _ = _run_and_capture("random", seed=1)

    x_deo = idata_deo.posterior["x"].values
    x_rnd = idata_rnd.posterior["x"].values

    # True mixture std = sqrt(sep^2 + s^2) = sqrt(17) ~ 4.12; mean = 0.
    assert abs(float(x_deo.mean())) < 1.0
    assert abs(float(x_rnd.mean())) < 1.0
    assert abs(float(x_deo.std()) - float(x_rnd.std())) < 1.0
    # DEO mixes well enough to populate both modes roughly evenly.
    frac_pos = float((x_deo > 0).mean())
    assert 0.3 < frac_pos < 0.7, f"DEO mode balance off: frac_pos={frac_pos:.2f}"


def test_deo_achieves_higher_round_trip_rate_than_random():
    """
    Given the same bimodal target and seed,
    When ptde_sample runs under the DEO vs the random swap schedule,
    Then DEO completes substantially more cold -> hot -> cold round trips --
      THE metric for how fast the ladder transports mass between modes. On an
      8-rung ladder DEO's O(n_temps) round-trip scaling beats random-pair
      O(n_temps^2) by roughly an order of magnitude.
    """
    _, rt_deo, _ = _run_and_capture("deo", seed=1)
    _, rt_rnd, _ = _run_and_capture("random", seed=1)
    assert rt_deo > 0
    assert rt_deo > 3 * rt_rnd, (
        f"expected DEO round trips ({rt_deo}) to far exceed random "
        f"({rt_rnd}) on an 8-rung ladder")


# ---------------------------------------------------------------------------
# (c) Ladder adaptation: communication-barrier equalization
# ---------------------------------------------------------------------------

def test_update_ladder_barrier_equalizes_uneven_barrier():
    """
    Given a geometric ladder and a rejection-rate profile whose barrier is
      concentrated in the middle rungs,
    When _update_ladder_barrier re-spaces the ladder,
    Then the per-interval share of the communication barrier becomes uniform
      across the interior rungs, and the two endpoints (T=1, T=T_max) stay
      pinned (EXOFASTv2 parity at the ends).
    """
    T = 200.0 ** (np.arange(8) / 7)          # geometric, 8 rungs
    propose = np.full(7, 100.0)
    reject = np.array([0.02, 0.02, 0.6, 0.6, 0.02, 0.02, 0.02])
    accept = propose * (1.0 - reject)

    new_T = _update_ladder_barrier(T, accept, propose)

    # Endpoints pinned.
    assert np.isclose(new_T[0], 1.0)
    assert np.isclose(new_T[-1], 200.0)
    assert np.all(np.diff(new_T) > 0)        # still strictly increasing

    # Re-derive the barrier carried by each NEW interval by interpolating the
    # cumulative barrier Lambda(beta) at the new coldnesses.
    beta = 1.0 / T
    Lambda = np.concatenate([[0.0], np.cumsum(reject)])
    new_beta = 1.0 / new_T
    new_barrier = np.interp(new_beta[::-1], beta[::-1], Lambda[::-1])[::-1]
    per_interval_new = np.abs(np.diff(new_barrier))
    per_interval_old = np.diff(Lambda)

    assert per_interval_new.std() < 0.1 * per_interval_old.std()


def test_update_ladder_barrier_noop_on_perfect_mixing():
    """
    Given zero rejection everywhere (perfect swap mixing),
    When _update_ladder_barrier is called,
    Then the ladder is returned unchanged (no barrier to redistribute).
    """
    T = 200.0 ** (np.arange(8) / 7)
    propose = np.full(7, 100.0)
    accept = propose.copy()                  # 100% acceptance -> zero barrier
    new_T = _update_ladder_barrier(T, accept, propose)
    assert np.allclose(new_T, T)


def test_adapt_ladder_runs_end_to_end_and_recovers_moments():
    """
    Given adapt_ladder=True on the bimodal toy,
    When ptde_sample runs,
    Then it completes, returns well-formed InferenceData, and still recovers
      the symmetric posterior mean (adaptation is confined to tuning, so it
      must not bias the kept draws).
    """
    idata, rt, _ = _run_and_capture("deo", seed=2, adapt_ladder=True)
    assert isinstance(idata, xr.DataTree)
    assert idata.posterior.sizes["draw"] == 400
    assert abs(float(idata.posterior["x"].values.mean())) < 1.0
    assert rt > 0


# ---------------------------------------------------------------------------
# swap_schedule validation + async wiring
# ---------------------------------------------------------------------------

def test_invalid_swap_schedule_raises():
    """
    Given an unrecognized swap_schedule,
    When ptde_sample is called,
    Then it raises ValueError before doing any sampling.
    """
    with pytest.raises(ValueError, match="swap_schedule"):
        ptde_sample(_bimodal_model(), _MinimalSystem(), draws=5, tune=5,
                    n_temps=2, T_max=2.0, cores=1, seed=0,
                    swap_schedule="bogus")


@pytest.mark.parametrize("sched", ["deo", "random"])
def test_ptde_async_runs_under_both_schedules(sched):
    """
    Given each swap_schedule value,
    When ptde_async_sample runs on the bimodal toy,
    Then it completes and returns the expected number of draws per chain --
      the async event-time DEO cycling must not deadlock or break output.
    """
    idata = ptde_async_sample(
        _bimodal_model(), _MinimalSystem(), draws=40, tune=40,
        n_temps=4, T_max=20.0, n_chains=4, cores=1, seed=3,
        log_interval=100000, swap_schedule=sched,
        min_ess=None, max_rhat=None)
    assert idata.posterior.sizes["draw"] == 40
    assert idata.posterior.sizes["chain"] == 4


def test_ptde_async_invalid_swap_schedule_raises():
    """
    Given an unrecognized swap_schedule,
    When ptde_async_sample is called,
    Then it raises ValueError before doing any sampling.
    """
    with pytest.raises(ValueError, match="swap_schedule"):
        ptde_async_sample(_bimodal_model(), _MinimalSystem(), draws=5, tune=5,
                          n_temps=2, T_max=2.0, cores=1, seed=0,
                          swap_schedule="bogus")
