"""The `_start` plots are written BEFORE the seed polish.

They are the quick look a user iterates the params file against ("is my
start in the ball park?"), so they must not wait for the polish, the
whitening probe or the startup table -- tens of minutes on a microlensing
fit (JDE, 2026-11-05).  run.md, "The user-start contract is judged at the
BUILD start".

Driven through the real `_run_fit` on a minimal orbit-only System (the
test_nuts_start.py harness): every component's `plot` is spied (Orbit
inherits Component.plot), and the polish is replaced by a stub that records
when it was reached and ends the run.
"""

import os

import pytest

from exozippy import run as run_module
from exozippy.components.component import Component


class _StopAtPolish(Exception):
    """Raised by the stubbed polish to end the run once it is reached."""


def test_start_plots_are_written_before_the_seed_polish(tmp_path):
    """
    Given a fit whose seed polish is on,
    When _run_fit runs up to the polish,
    Then every component has already been plotted at the build start
    (prefix `<prefix>_start`), and nothing at the polished start yet.
    """
    prefix = str(tmp_path / "earlystart")
    config = {
        "name": "early_start_plots",
        "orbit": [{"name": "test_orbit"}],
        "prefix": prefix,
        "sampler": {
            "method": "nuts",
            "tune": 1,
            "draws": 1,
            "chains": 1,
            "cores": 1,
            "recompute_trace": True,
            "seed_polish": 1,
        },
    }
    user_params = {
        "orbit.test_orbit.logP": {"initval": 1.0},
        "orbit.test_orbit.tc": {"initval": 0.0},
        "orbit.test_orbit.secosw": {"initval": 0.0},
        "orbit.test_orbit.sesinw": {"initval": 0.0},
    }
    events = []
    real_plot = Component.plot

    def spy_plot(self, system, points, filename_prefix="debug"):
        events.append(("plot", type(self).__name__, filename_prefix))
        return real_plot(self, system, points, filename_prefix)

    def stub_polish_rounds(*args, **kwargs):
        events.append(("polish",))
        raise _StopAtPolish

    orig_cwd = os.getcwd()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Component, "plot", spy_plot)
        mp.setattr(run_module, "polish_rounds", stub_polish_rounds)
        os.chdir(tmp_path)
        try:
            with pytest.raises(_StopAtPolish):
                run_module.run_fit(config, user_params=user_params)
        finally:
            os.chdir(orig_cwd)

    assert events[-1] == ("polish",)
    plots = events[:-1]
    assert ("plot", "Orbit", prefix + "_start") in plots
    assert all(e[2] == prefix + "_start" for e in plots)
