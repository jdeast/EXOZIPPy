"""The per-step wall-clock alarm in the samplers (review 2.4.14 (e)).

examples/ob09020's 2000-sweep arm spent 98,756 s on ONE synchronous PTDE
step with no eval_timeout message at all, and nothing said so until the
step ended a day later -- nor whether the time went inside the parallel
logp batch or in the serial work around it.  _common.StepAlarm says both,
while the step is still running.
"""

import logging
import time

import pymc as pm

from exozippy.samplers import _common
from exozippy.samplers.ptde import ptde_sample


def _alarm(caplog_logger, **kw):
    # A poll far longer than any test: the watchdog thread never fires on
    # its own here, so every look is the test's own check() call.
    return _common.StepAlarm("TEST", caplog_logger, poll_s=3600.0, **kw)


def test_a_slow_step_alarms_while_running_and_names_its_phase(caplog):
    """
    Given a step that has run past its alarm threshold,
    When the watchdog looks,
    Then ONE warning names the step, the phase it is in now and the time
      spent in each phase so far; it does not repeat until twice the elapsed
      time; and when the step ends its full phase split is logged.
    """
    # ARRANGE
    log = logging.getLogger("exozippy.test_step_alarm")
    alarm = _alarm(log, min_s=10.0, factor=20.0)
    try:
        alarm.begin(7, "build proposals")
        alarm.phase("eval (parallel logp batch)")
        t0 = alarm._t_step

        with caplog.at_level(logging.WARNING, logger=log.name):
            # ACT: below the threshold, at it, just after, and at 2x
            quiet = alarm.check(now=t0 + 5.0)
            first = alarm.check(now=t0 + 12.0)
            repeat = alarm.check(now=t0 + 13.0)
            second = alarm.check(now=t0 + 25.0)
            alarm.end()

        # ASSERT
        assert (quiet, first, repeat, second) == (False, True, False, True)
        text = caplog.text
        assert "step 7 has been running 12 s" in text
        assert "IN 'eval (parallel logp batch)'" in text
        assert "build proposals" in text
        assert "slow step 7 finished" in text
    finally:
        alarm.close()


def test_threshold_tracks_the_median_completed_step():
    """
    Given completed steps of ~1 s,
    When the threshold is asked for,
    Then it is max(floor, factor * median): an expensive model's ordinary
      step never alarms, and a fast model's jitter never does either.
    """
    alarm = _alarm(logging.getLogger("x"), min_s=0.0, factor=20.0)
    try:
        alarm._history.extend([1.0, 1.0, 3.0])
        assert alarm.threshold() == 20.0
        alarm2 = _alarm(logging.getLogger("x"), min_s=600.0, factor=20.0)
        alarm2._history.extend([1.0])
        assert alarm2.threshold() == 600.0
        alarm2.close()
    finally:
        alarm.close()


def test_a_normal_step_is_silent(caplog):
    """
    Given a step that ends well inside its threshold,
    When it ends,
    Then nothing is logged: the alarm costs a fast run no lines at all.
    """
    log = logging.getLogger("exozippy.test_step_alarm.quiet")
    alarm = _alarm(log, min_s=60.0)
    try:
        with caplog.at_level(logging.DEBUG, logger=log.name):
            alarm.begin(1, "build")
            alarm.phase("eval")
            alarm.end()
            assert alarm.check(now=time.monotonic() + 1e6) is False
        assert caplog.text == ""
    finally:
        alarm.close()


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def test_ptde_sample_brackets_every_step_with_the_alarm(monkeypatch):
    """
    Given the synchronous PTDE sampler,
    When it runs,
    Then every step opens the alarm in the proposal phase, passes through
      the parallel logp batch and the rest, and the alarm is closed at the
      end -- so a slow step on a real run is caught wherever it stalls.
    """
    # ARRANGE
    events = []

    class _Spy:
        def __init__(self, label, log, **kw):
            events.append(("init", label))

        def begin(self, step, phase):
            events.append(("begin", step, phase))

        def phase(self, phase):
            events.append(("phase", phase))

        def end(self):
            events.append(("end",))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(_common, "StepAlarm", _Spy)
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)

    # ACT
    ptde_sample(
        model,
        _MinimalSystem(),
        draws=3,
        tune=2,
        n_temps=2,
        T_max=2.0,
        n_chains=4,
        cores=1,
        seed=0,
        log_interval=100,
    )

    # ASSERT
    begins = [e for e in events if e[0] == "begin"]
    assert [b[1] for b in begins] == [1, 2, 3, 4, 5]
    phases = [e[1] for e in events if e[0] == "phase"]
    assert phases.count("eval (parallel logp batch)") == 5
    assert events[-1] == ("close",)
