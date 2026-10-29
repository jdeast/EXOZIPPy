"""The early stop's ensemble-mean drift test (review 1.4.4).

`convergence.converged_on_tail` -- the min_ess/max_rhat early stop of both
PTDE samplers -- judged a run on rank Rhat and bulk ESS across the T=1
chains.  A differential-evolution population moves as ONE body: every
proposal is built from the other chains, so a drift of the whole ensemble
along a curved ridge leaves the between-chain variance ~0 and Rhat ~1 the
whole way.  Measured on ob140939 (sync ptde, 24 rungs, 34 chains): the stop
fired at 3995 of 50000 draws with max Rhat 1.009 while the median lens
distance was still climbing 2426 -> 2840 pc between the halves of the tail;
the full-length runs put it at 3.3-4.7 kpc.  `ensemble_drift` tests the
chain-AVERAGED value of every sampled element and of lp for a linear trend
against its autocorrelation-corrected error, and the stop now requires it.

The synthetic ensembles here are the cases the review asked for: a
population translating at constant speed must never stop, a stationary
autocorrelated one must stop at near-nominal false-alarm rate, and an
ensemble built to ob140939's numbers must not stop where the old rule did.
The calibration behind the constants is in samplers.md, "The early stop
also requires a stationary ensemble mean".
"""

import logging

import numpy as np
import pytest
from scipy.signal import lfilter

from exozippy.samplers import _common
from exozippy.samplers import convergence as C


def _ar1(rng, shape, n, tau):
    """Stationary unit-variance AR(1) along the last axis, IACT ``tau``."""
    phi = (tau - 1.0) / (tau + 1.0)
    burn = int(10 * tau) + 10
    e = rng.standard_normal(shape + (n + burn,))
    return lfilter([np.sqrt(1 - phi**2)], [1, -phi], e, axis=-1)[..., burn:]


def _population(
    n, seed, n_chains=24, n_elem=6, tau_chain=20.0, w=0.05, tau_pop=200.0
):
    """A DE-like population: fast per-chain decorrelation plus a slow
    component SHARED by every chain carrying a fraction ``w`` of the
    variance -- the shape measured on the ob140939 traces, where each
    chain's IACT was ~5-10 draws while the ensemble mean's was hundreds.

    Returns (posterior, lp) in converged_on_tail's layout.
    """
    rng = np.random.default_rng(seed)
    x = np.sqrt(1.0 - w) * _ar1(rng, (n_chains, n_elem), n, tau_chain)
    x += np.sqrt(w) * _ar1(rng, (n_elem,), n, tau_pop)[None]
    x = np.transpose(x, (0, 2, 1))  # (chain, draw, elem)
    lp = -0.5 * (x[..., :3] ** 2).sum(axis=-1)
    return x, lp


def _old_rule(rhat, ess, min_ess=1000, max_rhat=1.01):
    return rhat <= max_rhat and ess >= min_ess


def _walk_schedule(x, lp, n_max, min_ess=1000, max_rhat=1.01):
    """Run the live check schedule; return (old_stop, new_stop, last_drift)."""
    old_stop = new_stop = None
    drift = None
    for k in _common._convergence_check_schedule():
        if k > n_max:
            break
        conv, rhat, ess, drift = C.converged_on_tail(
            {"p.x_raw": x[:, :k]}, lp[:, :k], min_ess, max_rhat
        )
        if old_stop is None and _old_rule(rhat, ess, min_ess, max_rhat):
            old_stop = k
        if conv:
            new_stop = k
            break
    return old_stop, new_stop, drift


# --- the three cases the review asked for ---------------------------------


def test_an_ensemble_translating_at_constant_speed_never_stops():
    """
    Given 24 chains that all translate together at a constant 1 posterior
      sd per 12000 draws -- slow enough that split-Rhat passes (a faster
      translation splits each chain's halves apart and Rhat catches it on
      its own),
    When the live check schedule runs to 12000 draws,
    Then the old Rhat/ESS rule would have stopped it (at 4439 draws), and
      the new one never does, flagging the drift.

    The detection floor is the population's own Monte Carlo noise: a
    translation of ~0.15 sd across the tail at the stop (1 sd per 24000
    draws here) scores |z| ~ 2.5 and is not distinguishable from wander at
    that run length; samplers.md quotes the scan.
    """
    # ARRANGE
    n = 12000
    x, _ = _population(n, seed=11, w=0.0)
    x = x + (np.arange(n) / 12000.0)[None, :, None]
    lp = -0.5 * (x[..., :3] ** 2).sum(axis=-1)

    # ACT
    old_stop, new_stop, drift = _walk_schedule(x, lp, n)

    # ASSERT: not vacuous -- Rhat/ESS really were satisfied ...
    assert old_stop is not None
    # ... and the drift test held the run every time.
    assert new_stop is None
    assert drift["flagged"], C.format_drift(drift)
    assert "DRIFTING" in C.format_drift(drift)


def test_stationary_population_false_alarm_rate_is_near_nominal():
    """
    Given 300 independent stationary DE-like populations (24 chains, 6
      elements + lp, ensemble-mean IACT ~200, a window of 50 of those),
    When each window is tested,
    Then the family-wise false-alarm rate is near DRIFT_FAMILY_ALPHA (1%):
      measured 2-3% here and on 34-chain, 17-element ensembles (samplers.md);
      the bound below is 5x nominal, which a test that ignored the
      autocorrelation (per-draw spread) would exceed on every seed.
    """
    # ARRANGE / ACT
    n_seeds, hits = 300, 0
    for s in range(n_seeds):
        x, lp = _population(10000, seed=100 + s)
        d = C.ensemble_drift({"p.x_raw": x}, lp)
        hits += not d["stationary"]

    # ASSERT
    rate = hits / n_seeds
    assert rate <= 5 * C.DRIFT_FAMILY_ALPHA, rate


def test_a_stationary_population_still_stops_with_modest_overhead():
    """
    Given stationary DE-like populations that the old Rhat/ESS rule stops,
    When the live check schedule runs,
    Then the new rule also stops every one of them, at most a few check
      intervals later (each interval is ~11% more draws).
    """
    # ARRANGE / ACT
    ratios = []
    for s in range(4):
        x, lp = _population(40000, seed=500 + s)
        old_stop, new_stop, drift = _walk_schedule(x, lp, 40000)
        assert old_stop is not None
        assert new_stop is not None, C.format_drift(drift)
        ratios.append(new_stop / old_stop)

    # ASSERT: the window must hold >= _DRIFT_MIN_NEFF ensemble-mean
    # correlation times, which a population with IACT 200 reaches at a tail
    # of ~4000 draws -- so the stop moves by the population's own clock,
    # not by an arbitrary factor.
    assert max(ratios) <= 3.0, ratios


def _ob140939_like(seed=2):
    """34 chains, 3995 draws, built to review 1.4.4's measured numbers.

    The lens distance climbs along a curved ridge as one body: its ensemble
    median is ~2426 pc over the third quarter of the run and ~2840 pc over
    the fourth (the halves of the 50% tail the early stop judges), with a
    per-chain posterior scatter of 2500 pc (the lens distance is
    prior-dominated along the ridge; the converged runs span 3.3-4.7 kpc
    between priors) that hides the climb from Rhat, as it did there (max
    Rhat 1.006 here, 1.009 measured); lp rises with it (no T=1 draw beat
    the polished start).  Four nuisance
    elements are stationary.
    """
    rng = np.random.default_rng(seed)
    n_chains, n = 34, 3995
    t = np.arange(n) / n
    # Decelerating climb, ridge position in pc (a sqrt ramp from the start).
    ridge = 170.0 + 2855.0 * np.sqrt(t)
    scatter = 2500.0 * _ar1(rng, (n_chains,), n, 8.0)
    dist = ridge[None, :] + scatter
    nuisance = _ar1(rng, (n_chains, 4), n, 8.0).transpose(0, 2, 1)
    lp = (
        10650.0
        + 1.5 * np.sqrt(t)[None, :]
        + 2.5 * _ar1(rng, (n_chains,), n, 8.0)
    )
    raw = np.concatenate([dist[..., None] / 1000.0, nuisance], axis=-1)
    return raw, lp, dist


def test_an_ob140939_like_ridge_drift_does_not_stop_where_the_old_rule_did():
    """
    Given an ensemble built to ob140939's numbers (review 1.4.4),
    When it is judged at 3995 draws, where the old rule stopped that run,
    Then Rhat/ESS pass exactly as they did there, but the drift test flags
      the lens-distance element (and lp), so the run keeps sampling.
    """
    # ARRANGE
    raw, lp, dist = _ob140939_like()
    n = raw.shape[1]
    q3 = np.median(dist[:, n // 2 : 3 * n // 4])
    q4 = np.median(dist[:, 3 * n // 4 :])
    # the synthetic reproduces the measured climb between tail halves
    assert 2300 < q3 < 2550 and 2700 < q4 < 2950, (q3, q4)

    # ACT
    conv, rhat, ess, drift = C.converged_on_tail(
        {"star.distance_raw": raw}, lp, 1000, 1.01
    )

    # ASSERT
    assert _old_rule(rhat, ess), (rhat, ess)
    assert conv is False
    flagged = {name for name, *_ in drift["flagged"]}
    assert "star.distance_raw[0]" in flagged, C.format_drift(drift)
    assert "lp" in flagged, C.format_drift(drift)


# --- the statistic's edge cases --------------------------------------------


def test_a_window_too_short_in_population_time_is_unresolved_not_stationary():
    """
    Given a stationary population whose window spans only ~5 ensemble-mean
      correlation times,
    When it is tested,
    Then nothing is claimed stationary: the elements come back UNRESOLVED
      (or flagged), never as a pass -- the IACT of so short a series is
      not trustworthy, so stationarity has not been shown.
    """
    # ARRANGE: shared component only, IACT 400, window 2000
    rng = np.random.default_rng(3)
    common = _ar1(rng, (3,), 2000, 400.0)
    x = np.repeat(common.T[None], 8, axis=0) + 0.01 * rng.standard_normal(
        (8, 2000, 3)
    )

    # ACT
    d = C.ensemble_drift({"p.x_raw": x})

    # ASSERT
    assert d["stationary"] is False
    assert d["unresolved"] or d["flagged"]


def test_constant_elements_are_skipped_and_too_few_draws_never_pass():
    """
    Given an element that never moves beside one that does,
    When the window is long enough, the constant one is not counted;
    and when the window is shorter than the minimum series length,
      stationarity is not shown.
    """
    rng = np.random.default_rng(4)
    moving = rng.standard_normal((6, 400))
    post = {"a_raw": np.zeros((6, 400)), "b_raw": moving}
    d = C.ensemble_drift(post)
    assert d["n_tests"] == 1
    assert d["stationary"] is True

    short = C.ensemble_drift({"b_raw": moving[:, :10]})
    assert short["stationary"] is False
    assert "too few draws" in C.format_drift(short)


def test_a_non_finite_draw_raises_naming_the_element():
    """
    Given a stored draw that is NaN,
    When the drift test reads it,
    Then it raises and names the element: an accepted state cannot be
      non-finite, so this is an upstream bug, not a drift.
    """
    x = np.random.default_rng(5).standard_normal((4, 200))
    x[2, 50] = np.nan
    with pytest.raises(ValueError, match="p.bad_raw"):
        C.ensemble_drift({"p.bad_raw": x})


# --- the wrap-up report -----------------------------------------------------


def test_wrap_up_reports_the_drift_and_withholds_the_converged_verdict(
    caplog,
):
    """
    Given a finished trace whose chains all translate together (Rhat ~1),
    When analyze_idata and log_convergence run at wrap-up,
    Then the diag carries the drift, the verdict is NOT converged, and the
      log says the ensemble mean is not stationary.
    """
    az = pytest.importorskip("arviz")
    n = 6000
    x, _ = _population(n, seed=21, n_elem=2)
    x = x + (np.arange(n) / 2000.0)[None, :, None]
    idata = az.from_dict(
        {
            "posterior": {"p.x_raw": x, "p.x": 2.0 * x},
            "sample_stats": {"lp": -0.5 * (x**2).sum(axis=-1)},
        }
    )

    _, diag = C.analyze_idata(idata, min_ess=100, max_rhat=1.05)
    with caplog.at_level(logging.WARNING):
        C.log_convergence(diag)

    assert diag["drift"] is not None
    assert diag["drift"]["stationary"] is False
    assert diag["converged"] is False
    assert "ENSEMBLE MEAN NOT STATIONARY" in caplog.text


def test_wrap_up_on_a_stationary_trace_reports_a_stationary_ensemble(caplog):
    """
    Given a stationary trace of independent chains,
    When analyze_idata and log_convergence run,
    Then the drift is judged stationary and the OK line quotes it.
    """
    az = pytest.importorskip("arviz")
    x = np.random.default_rng(22).standard_normal((6, 3000, 2))
    idata = az.from_dict(
        {
            "posterior": {"p.x_raw": x, "p.x": 2.0 * x},
            "sample_stats": {"lp": -0.5 * (x**2).sum(axis=-1)},
        }
    )
    _, diag = C.analyze_idata(idata, min_ess=100, max_rhat=1.05)
    with caplog.at_level(logging.INFO):
        C.log_convergence(diag)
    assert diag["drift"]["stationary"] is True
    assert diag["converged"] is True
    assert "ensemble stationary" in caplog.text
