"""
The PTDE tune-phase finish forecast (samplers._common.TuneForecast).

DC2018 event 208 (2026-10) ran at ~22 s per step against ~3 s normally and
hit maxtime 72000 s during TUNE, at step 3250 of 13000, saving nothing. The
forecast projects the end of tune from the measured step rate after a
warmup window and, when it is past maxtime, shortens tune (the default) or
aborts at once. Every clock here is fake: the class takes ``now`` as an
argument, and the sampler-level tests replace the module's ``time``.
"""

import logging
import types

import numpy as np
import pymc as pm
import pytest

from exozippy.samplers import _common
from exozippy.samplers import ptde as ptde_mod
from exozippy.samplers._common import TuneForecast, TuneWontFitError
from exozippy.samplers.ptde import ptde_sample
from exozippy.samplers.ptde_async import ptde_async_sample


def _forecast(**kw):
    args = dict(
        tune=5000,
        draws=8000,
        maxtime=72000.0,
        after=250,
        t0=0.0,
        log=logging.getLogger("test.forecast"),
    )
    args.update(kw)
    return TuneForecast(**args)


def test_the_event_208_rate_shortens_tune_so_draws_fit(caplog):
    """
    Given event 208's shape (tune 5000, draws 8000, maxtime 72000 s) at
      22 s per step,
    When the forecast runs after its 250-step warmup,
    Then tune is cut to the configured tune:draws share of the steps that
      fit in 90% of maxtime, a WARNING says so, and the trace stamp
      records requested and used tune.
    """
    fc = _forecast()

    with caplog.at_level(logging.INFO, logger="test.forecast"):
        new_tune = fc.check(250, now=250 * 22.0)

    # 0.9 * 72000 s - 5500 s elapsed = 59300 s = 2695 more steps; of the
    # 2945 that fit, 5000/13000 are tune.
    n_fit = 250 + (0.9 * 72000 - 250 * 22.0) / 22.0
    assert new_tune == int(5000 / 13000 * n_fit) == 1132
    assert fc.shortened
    assert "TUNE SHORTENED 5000 -> 1132" in caplog.text
    assert "finish forecast after 250 steps" in caplog.text
    # projected end of the new tune is inside maxtime: no second cut
    assert fc.check(500, now=500 * 22.0) == 1132

    idata = types.SimpleNamespace(posterior=types.SimpleNamespace(attrs={}))
    fc.stamp(idata)
    assert idata.posterior.attrs == {
        "tune_requested": 5000,
        "tune_used": 1132,
        "tune_shortened": 1,
    }


def test_a_rate_that_fits_leaves_tune_alone():
    """
    Given the normal ~3 s per step on the same run,
    When the forecast runs,
    Then tune is untouched and the stamp says so.
    """
    fc = _forecast()
    assert fc.check(250, now=250 * 3.0) == 5000
    assert fc.check(4999, now=4999 * 3.0) == 5000
    assert not fc.shortened


def test_no_forecast_before_the_warmup_or_after_tune_or_without_maxtime():
    """
    Given a hopeless rate,
    When progress is under the warmup window, or past tune, or no maxtime
      is set, or the mode is "off",
    Then nothing changes.
    """
    assert _forecast().check(249, now=249 * 1e6) == 5000
    assert _forecast().check(5000, now=5000 * 1e6) == 5000
    assert _forecast(maxtime=None).check(250, now=250 * 1e6) == 5000
    assert _forecast(mode="off").check(250, now=250 * 1e6) == 5000
    assert _forecast(t0=None).check(250, now=250 * 1e6) == 5000


def test_abort_mode_raises_with_the_projection():
    """
    Given `tune_forecast: abort` and event 208's rate,
    When the forecast runs,
    Then it raises TuneWontFitError naming the projected tune end.
    """
    with pytest.raises(TuneWontFitError, match=r"22\.00 s/step"):
        _forecast(mode="abort").check(250, now=250 * 22.0)


def test_the_tune_never_shrinks_below_the_progress_made():
    """
    Given a run already past maxtime's budget at the forecast,
    When the forecast shortens tune,
    Then the new tune is the progress made: tune ends now, draws follow.
    """
    fc = _forecast()
    assert fc.check(250, now=80000.0) == 250


@pytest.mark.parametrize(
    "kw", [{"mode": "sometimes"}, {"after": 0}], ids=["mode", "after"]
)
def test_bad_knobs_are_refused(kw):
    with pytest.raises(ValueError):
        _forecast(**kw)


# ---------------------------------------------------------------------------
# The samplers
# ---------------------------------------------------------------------------


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _model():
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
    return model


class _FakeTime:
    """A clock that moves one second per reading, whatever the host does."""

    def __init__(self):
        self.t = 0.0

    def time(self):
        self.t += 1.0
        return self.t

    monotonic = time
    perf_counter = time

    @staticmethod
    def sleep(_):
        pass


_SAMPLER_KW = dict(
    n_temps=2,
    T_max=2.0,
    n_chains=4,
    cores=1,
    seed=5,
    log_interval=10**6,
    min_ess=None,
    max_rhat=None,
    store_hot_chains=False,
)


def test_sync_shortens_tune_and_keeps_draws_on_a_slow_clock(
    monkeypatch, caplog
):
    """
    Given a synchronous run whose (fake) clock puts the end of tune far past
      maxtime,
    When ptde_sample runs with the default forecast,
    Then tune is shortened, the run reaches its draw phase and returns
      draws before maxtime, and the trace is stamped -- where with the
      forecast off the same run stops in tune and saves nothing.
    """
    kw = dict(
        _SAMPLER_KW,
        draws=200,
        tune=400,
        maxtime=1500.0,
        tune_forecast_after=50,
    )
    monkeypatch.setattr(ptde_mod, "time", _FakeTime())
    with caplog.at_level(logging.INFO, logger="exozippy.samplers"):
        idata = ptde_sample(_model(), _MinimalSystem(), **kw)

    attrs = idata.posterior.attrs
    assert attrs["tune_shortened"] == 1
    assert attrs["tune_requested"] == 400
    assert 50 <= attrs["tune_used"] < 400
    assert idata.posterior.sizes["draw"] >= 1
    assert "TUNE SHORTENED" in caplog.text

    monkeypatch.setattr(ptde_mod, "time", _FakeTime())
    with pytest.raises(KeyboardInterrupt):
        ptde_sample(
            _model(), _MinimalSystem(), **dict(kw, tune_forecast="off")
        )


def test_sync_abort_mode_stops_at_the_forecast(monkeypatch):
    """
    Given the same slow clock and `tune_forecast: abort`,
    When ptde_sample runs,
    Then it raises TuneWontFitError at the forecast, not KeyboardInterrupt
      at maxtime.
    """
    monkeypatch.setattr(ptde_mod, "time", _FakeTime())
    with pytest.raises(TuneWontFitError):
        ptde_sample(
            _model(),
            _MinimalSystem(),
            **dict(
                _SAMPLER_KW,
                draws=200,
                tune=400,
                maxtime=1500.0,
                tune_forecast="abort",
                tune_forecast_after=50,
            ),
        )


@pytest.mark.parametrize(
    "sampler", [ptde_sample, ptde_async_sample], ids=["ptde", "ptde_async"]
)
def test_both_samplers_honor_a_shortened_tune(sampler, monkeypatch):
    """
    Given a forecast that cuts tune from 10**6 to 5 steps once 3 are done
      (TuneForecast.check replaced, so the wiring is tested on its own),
    When either PTDE loop runs,
    Then it moves its tune -> draw boundary, records its draws, finishes,
      and stamps the shortened tune: the parity rule for a shared knob.
    """

    def _check(self, progress, now):
        if progress >= 3 and not self.shortened:
            self.tune, self.shortened = 5, True
        return self.tune

    monkeypatch.setattr(_common.TuneForecast, "check", _check)
    idata = sampler(
        _model(),
        _MinimalSystem(),
        **dict(_SAMPLER_KW, draws=20, tune=10**6, maxtime=1e9),
    )

    assert idata.posterior.sizes["draw"] == 20
    assert idata.posterior.attrs["tune_used"] == 5
    assert idata.posterior.attrs["tune_shortened"] == 1
    assert np.isfinite(idata.posterior["x"].values).all()
