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
