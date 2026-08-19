"""ptde_async can re-shape its ladder, not just lengthen it.

`_update_ladder_barrier` (Syed et al. 2022) lived only in the SYNCHRONOUS
sampler, and even there defaulted off -- so `ptde_async`, the recommended
default for Op-based models, could not reshape its ladder at all.

That is not academic.  On examples/DC2018 event 128 at T_max=8500 with a
correctly provisioned n_temps=48 (Lambda=18.9 against the 39 the DEO
criterion asks, mean swap acceptance 0.598 at its 0.5 target), per-rung
acceptance was strongly non-uniform -- 0.46-0.52 cold against 0.66-0.70 hot
-- and round trips stayed at ZERO for 21 hours.  A round trip must cross
every pair, so transport is set by the worst stretch; a geometric ladder
cannot fix that by getting longer.
"""

import logging

import numpy as np
import pytest

from exozippy.samplers.ptde import _update_ladder_barrier


def test_respacing_moves_rungs_toward_the_high_barrier_region():
    """
    Given per-pair rejection concentrated at the COLD end,
    When the ladder is re-spaced to equalize the barrier,
    Then interior rungs move so the cold end gets more resolution, while
      both endpoints stay pinned (T_0 = 1, T_max unchanged).
    """
    # ARRANGE: 9 rungs, geometric over T_max=200; cold pairs reject 0.9,
    # hot pairs 0.1 -- the shape measured on event 128, exaggerated.
    n = 9
    T = 200.0 ** (np.arange(n) / (n - 1))
    propose = np.full(n - 1, 100.0)
    reject = np.array([0.9, 0.9, 0.9, 0.5, 0.2, 0.1, 0.1, 0.1])
    accept = propose * (1.0 - reject)

    # ACT
    newT = _update_ladder_barrier(T, accept, propose)

    # ASSERT: endpoints pinned
    assert newT[0] == 1.0
    np.testing.assert_allclose(newT[-1], T[-1])
    # monotone ladder
    assert np.all(np.diff(newT) > 0)
    # the cold end gains resolution: more rungs below the geometric midpoint
    mid = np.sqrt(T[-1])
    assert (newT < mid).sum() > (T < mid).sum()


def test_respacing_is_a_noop_on_an_already_equal_barrier():
    """A ladder already carrying an equal share per pair should barely move."""
    n = 9
    T = 200.0 ** (np.arange(n) / (n - 1))
    propose = np.full(n - 1, 100.0)
    accept = propose * 0.5  # uniform rejection 0.5

    newT = _update_ladder_barrier(T, accept, propose)

    np.testing.assert_allclose(newT, T, rtol=1e-6)


def test_async_accepts_adapt_ladder_and_defaults_it_off():
    """The knob exists on ptde_async and is off unless asked for.

    Off by default deliberately: it matches the synchronous sampler, and
    re-spacing changes the sampled ladder, which is not something to switch
    on for every existing config without measurement.
    """
    import inspect

    from exozippy.samplers.ptde_async import ptde_async_sample

    sig = inspect.signature(ptde_async_sample)
    assert "adapt_ladder" in sig.parameters
    assert sig.parameters["adapt_ladder"].default is False


# ---------------------------------------------------------------------------
# Per-rung explored span (_common.SpanTracker)
# ---------------------------------------------------------------------------


def test_span_tracker_measures_per_rung_reach():
    """
    Given states visited at a cold and a hot rung,
    When SpanTracker records them,
    Then each rung reports its own widest span, in raw (whitened) sigma, and
      names the parameter responsible.

    This is what makes "can this ladder reach the thing I am looking for?"
    answerable DURING a run.  On DC2018 event 128 the known second basin
    (s = 0.854 against the posterior's 0.9755) sits 296 sigma away in log_s,
    while a rung at temperature T explores ~sqrt(T) sigma: 14 sigma at
    T_max=200 (21 widths -- unreachable at any runtime) against 92 sigma at
    T_max=8500 (3.2 widths -- reachable across 54 chains).  Comparing the
    reported span against that distance is the whole point.
    """
    # ARRANGE
    from exozippy.samplers._common import RawLayout, SpanTracker

    # States are PACKED vectors, not dicts: since review 6.4.2 that is what a
    # proposal is, and RawLayout is what maps a packed index back to the
    # parameter that owns it.
    start = {"a": np.zeros(2), "b": np.zeros(())}
    layout = RawLayout(start)
    tr = SpanTracker(n_temps=3, raw_start=start, layout=layout)

    # ACT: rung 0 wanders +-1, rung 2 wanders +-50 in "a"
    tr.update(0, layout.pack({"a": np.array([-1.0, 0.0]), "b": np.array(0.0)}))
    tr.update(0, layout.pack({"a": np.array([1.0, 0.5]), "b": np.array(0.2)}))
    tr.update(
        2, layout.pack({"a": np.array([-50.0, 0.0]), "b": np.array(0.0)})
    )
    tr.update(2, layout.pack({"a": np.array([50.0, 1.0]), "b": np.array(0.1)}))
    rep = tr.report()

    # ASSERT
    assert rep[0][0] == pytest.approx(2.0)  # a spans -1..1
    assert rep[0][1] == "a"
    assert rep[2][0] == pytest.approx(100.0)  # a spans -50..50
    assert rep[2][1] == "a"
    # the hot rung reaches 50x further than the cold one
    assert rep[2][0] / rep[0][0] == pytest.approx(50.0)


def test_span_tracker_reports_zero_for_an_unvisited_rung():
    """An unvisited rung has no span; it must not print inf or nan.

    A diagnostic that shows inf reads as a bug rather than as "no data yet",
    and rung 1 here is genuinely unreported at this point in a run.
    """
    from exozippy.samplers._common import RawLayout, SpanTracker

    start = {"a": np.zeros(())}
    layout = RawLayout(start)
    tr = SpanTracker(n_temps=2, raw_start=start, layout=layout)
    tr.update(0, layout.pack({"a": np.array(3.0)}))

    rep = tr.report()
    assert rep[1] == (0.0, "-")
    assert np.isfinite(rep[0][0])


def _packed(layout, **values):
    """One packed raw state, so these tests speak the samplers' own form."""
    return layout.pack(values)


def test_span_window_resets_while_the_cumulative_span_does_not():
    """
    Given a rung that ranged far early and is confined afterwards,
    When one tracker is reset and both then record only the confined region,
    Then the reset one reports the RECENT span and the other still reports
      the historical one.

    Both numbers are wanted and they answer different questions, which is why
    the samplers carry two trackers.  A running min/max only ever widens, so
    a cumulative span is permanently set by the widest excursion the rung
    ever made -- early in a run, where the chains were thrown at
    initialization.  On DC2018 event 128 still in its tune phase the
    cumulative T=1 span read 5995 sigma, which is a burn-in transient
    reported in the same field and units as a posterior width.
    """
    # ARRANGE
    from exozippy.samplers._common import RawLayout, SpanTracker

    start = {"a": np.zeros(())}
    layout = RawLayout(start)
    cum = SpanTracker(n_temps=1, raw_start=start, layout=layout)
    win = SpanTracker(n_temps=1, raw_start=start, layout=layout)
    for v in (-500.0, 500.0):  # the early excursion, seen by both
        cum.update(0, _packed(layout, a=np.array(v)))
        win.update(0, _packed(layout, a=np.array(v)))
    assert win.report()[0][0] == pytest.approx(1000.0)

    # ACT: a fresh window, then only the confined region
    win.reset()
    for v in (-1.0, 1.0):
        cum.update(0, _packed(layout, a=np.array(v)))
        win.update(0, _packed(layout, a=np.array(v)))

    # ASSERT
    assert win.report()[0][0] == pytest.approx(2.0)
    assert cum.report()[0][0] == pytest.approx(1000.0)


def test_span_report_on_one_key_makes_a_top_cold_ratio_meaningful():
    """
    Given two rungs whose widest-ranging parameters DIFFER,
    When the span is reported for one named parameter,
    Then both rungs report that parameter, so top/cold compares like with
      like.

    Each rung's independent maximum is generally a different parameter, and
    dividing one by the other is a ratio of nothing: on event 128 it divided
    the hot rung's log_f_total span by the cold rung's pm_ra span and printed
    "3x".
    """
    # ARRANGE
    from exozippy.samplers._common import RawLayout, SpanTracker

    start = {"tight": np.zeros(()), "loose": np.zeros(())}
    layout = RawLayout(start)
    tr = SpanTracker(n_temps=2, raw_start=start, layout=layout)
    # rung 0: "loose" ranges widest (60 vs 2).  rung 1: "tight" does (200 vs 10).
    for t, lo in ((-1.0, -30.0), (1.0, 30.0)):
        tr.update(0, _packed(layout, tight=np.array(t), loose=np.array(lo)))
    for t, lo in ((-100.0, -5.0), (100.0, 5.0)):
        tr.update(1, _packed(layout, tight=np.array(t), loose=np.array(lo)))

    # ACT
    free = tr.report()
    keyed = tr.report("tight")

    # ASSERT: unkeyed, the two ends name different parameters
    assert tr.widest_at(0) == "loose"
    assert tr.widest_at(1) == "tight"
    assert (free[0][1], free[1][1]) == ("loose", "tight")
    # keyed, both ends are the same parameter and the ratio is 200/2
    assert keyed[0][0] == pytest.approx(2.0)
    assert keyed[1][0] == pytest.approx(200.0)
    assert keyed[1][1] == keyed[0][1] == "tight"


def test_span_report_on_a_key_no_rung_has_visited_is_zero_not_inf():
    """An unvisited rung reports 0.0 for a named key too, never inf."""
    from exozippy.samplers._common import RawLayout, SpanTracker

    start = {"a": np.zeros(())}
    layout = RawLayout(start)
    tr = SpanTracker(n_temps=2, raw_start=start, layout=layout)
    tr.update(0, _packed(layout, a=np.array(3.0)))

    assert tr.report("a")[1][0] == 0.0
    assert tr.widest_at(1) is None


def test_widest_at_returns_none_before_any_state_is_recorded():
    """No rung has a span yet, so there is no widest parameter to name."""
    from exozippy.samplers._common import RawLayout, SpanTracker

    start = {"a": np.zeros(2)}
    tr = SpanTracker(n_temps=2, raw_start=start, layout=RawLayout(start))

    assert tr.widest_at(0) is None


# ---------------------------------------------------------------------------
# The adaptation window (ptde_async)
# ---------------------------------------------------------------------------


class _MinimalSystem:
    """Minimal system stub for ptde_async_sample (see test_ptde_async.py)."""

    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _spy_on_adaptation(monkeypatch):
    """Record the swap-proposal count behind every re-spacing measurement."""
    from exozippy.samplers import ptde_async

    seen = []
    real = ptde_async._update_ladder_barrier

    def spy(temperatures, accept, propose):
        seen.append(float(np.sum(propose)))
        return real(temperatures, accept, propose)

    monkeypatch.setattr(ptde_async, "_update_ladder_barrier", spy)
    return seen


def _run_async(**kwargs):
    import pymc as pm

    from exozippy.samplers.ptde_async import ptde_async_sample

    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=3.0, sigma=0.5)
    return ptde_async_sample(
        model,
        _MinimalSystem(),
        draws=20,
        tune=40,
        n_temps=4,
        T_max=8.0,
        n_chains=4,
        cores=1,
        seed=7,
        swap_interval=1,
        log_interval=1000,
        adapt_ladder=True,
        **kwargs,
    )


def test_ladder_adaptation_never_measures_a_barrier_from_a_short_window(
    monkeypatch,
):
    """
    Given adapt_ladder on,
    When the sampler runs,
    Then every re-spacing measurement is backed by at least
      ladder_adapt_window swap proposals.

    The gate is the correctness of the block, not a tuning preference.  The
    adaptation ZEROES the per-pair swap counters on its way out and runs per
    completed evaluation, so gating on `n_swap_propose.sum() > 0` -- as the
    original port did -- fired it again on the very next proposal, measuring
    a communication barrier from one or two swaps.  Observed on DC2018
    event 128: zero re-spacing log lines in 10 hours, because a single
    accepted swap says every pair it touched is already perfectly
    equalized.
    """
    # ARRANGE
    seen = _spy_on_adaptation(monkeypatch)
    window = 60

    # ACT
    _run_async(ladder_adapt_window=window)

    # ASSERT
    assert seen, "the adaptation never ran at all; the test proves nothing"
    assert min(seen) >= window, (
        f"re-spaced on {min(seen):.0f} swap proposals against a window of "
        f"{window}"
    )


def test_ladder_adaptation_does_not_run_at_all_below_one_window(
    monkeypatch, caplog
):
    """
    Given a window wider than the whole run,
    When the sampler runs with adapt_ladder on,
    Then the ladder is never re-spaced AND the run says so.

    Two halves of one contract.  The gate must be capable of suppressing the
    adaptation, otherwise passing it is decoration -- and windowing the gate
    makes "adapt_ladder did nothing at all" reachable, where the unwindowed
    version adapted (uselessly) on every proposal.  A knob that is silently
    inert is the failure this codebase warns about everywhere else, so the
    sampler warns and names the three ways out.
    """
    # ARRANGE
    seen = _spy_on_adaptation(monkeypatch)

    # ACT
    with caplog.at_level(
        logging.WARNING, logger="exozippy.samplers.ptde_async"
    ):
        _run_async(ladder_adapt_window=10**9)

    # ASSERT
    assert seen == []
    assert any(
        "adapt_ladder was requested but the ladder was never re-spaced"
        in r.message
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_async_exposes_ladder_adapt_window_and_defaults_it_to_none():
    """The knob exists and defaults to the measured-from-n_temps value."""
    import inspect

    from exozippy.samplers.ptde_async import ptde_async_sample

    sig = inspect.signature(ptde_async_sample)
    assert "ladder_adapt_window" in sig.parameters
    assert sig.parameters["ladder_adapt_window"].default is None
