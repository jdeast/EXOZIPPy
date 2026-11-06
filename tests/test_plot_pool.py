"""Posterior plot_data on the wrap-up's pool (review 2.6.44).

The wrap-up's posterior plots, per-mode plots and summary figure call
``plot_data`` once per draw.  On examples/MOA-2008-BLG-379 the light-curve
plot_data costs ~74 logp-equivalents a draw (17.4 s CPU against a 0.23 s
logp), so a serial pass of 51 draws is ~15 CPU-minutes and scales with the
magnification cost.  Inside ``plotrender.plot_pool`` the draws after the
reference go out on a fork pool under eval_timeout.
"""

import logging
import multiprocessing as mp
import os
import time

import pytest

from exozippy import plotrender
from exozippy.chart import Chart, Trace
from exozippy.plotrender import plot_data_at_points, plot_pool

requires_fork = pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(), reason="the plot pool forks"
)

MAIN_PID = os.getpid()
_MAIN_CALLS = []


class _Comp:
    prefix = "toy"

    def __init__(self, hang_at=None, fail_at=None):
        self.hang_at = hang_at
        self.fail_at = fail_at

    def plot_data(self, system, point, scale=1.0):
        x = point["x"]
        if os.getpid() == MAIN_PID:
            _MAIN_CALLS.append(x)
        if x == self.hang_at and os.getpid() != MAIN_PID:
            time.sleep(60.0)
        if x == self.fail_at:
            raise ValueError(f"bad draw {x}")
        return [_chart(x, scale, node=object())]


def _chart(x, scale=1.0, node=None):
    """One chart; ``node`` stands in for the unpicklable pytensor graph."""
    return Chart(
        id="c",
        component={"yaml_key": "toy"},
        title="t",
        xlabel="x",
        ylabel="y",
        traces=[
            Trace(
                name="m",
                role="model",
                kind="line",
                x=[0.0, 1.0],
                y=[scale * x, scale * x * x],
                node=node,
            )
        ],
    )


@pytest.fixture(autouse=True)
def _reset():
    _MAIN_CALLS.clear()
    yield


POINTS = [{"x": float(i)} for i in range(30)]


def test_outside_a_plot_pool_everything_is_serial():
    out = plot_data_at_points(_Comp(), None, POINTS, t_first=100.0, scale=2.0)
    assert len(_MAIN_CALLS) == 30
    assert out == [("ok", [_chart(float(i), 2.0)]) for i in range(30)]
    # the graph node is dropped on both paths (it cannot leave a worker)
    assert all(r[1][0].traces[0].node is None for r in out)


@requires_fork
def test_inside_a_plot_pool_slow_draws_never_run_in_the_main_process():
    with plot_pool(4, 30.0):
        out = plot_data_at_points(
            _Comp(fail_at=7.0), None, POINTS, t_first=1.0, scale=2.0
        )
    assert _MAIN_CALLS == []
    assert out[7] == ("error", "ValueError: bad draw 7.0")
    assert [r for i, r in enumerate(out) if i != 7] == [
        ("ok", [_chart(float(i), 2.0)]) for i in range(30) if i != 7
    ]
    assert plotrender._PLOT_POOL == {"cores": 1, "eval_timeout": None}


def test_a_cheap_plot_stays_serial_even_inside_a_plot_pool():
    with plot_pool(4, 30.0):
        plot_data_at_points(_Comp(), None, POINTS, t_first=1e-4)
    assert len(_MAIN_CALLS) == 30


@requires_fork
def test_a_hung_plot_draw_times_out_and_is_reported(caplog):
    with plot_pool(4, 1.0):
        out = plot_data_at_points(
            _Comp(hang_at=12.0), None, POINTS, t_first=1.0
        )
    assert out[12] == ("timeout", None)
    assert sum(r[0] == "ok" for r in out) == 29


@requires_fork
def test_plot_via_specs_skips_and_counts_a_timed_out_draw(monkeypatch, caplog):
    seen = {}
    monkeypatch.setattr(
        plotrender,
        "render_spec_groups",
        lambda groups, filename_prefix: seen.setdefault("groups", groups),
    )
    monkeypatch.setattr(plotrender, "PLOT_POOL_MIN_S", 0.0)
    caplog.clear()
    with caplog.at_level(logging.WARNING), plot_pool(4, 1.0):
        plotrender.plot_via_specs(_Comp(hang_at=12.0), None, POINTS)
    assert len(seen["groups"]) == 29
    assert "1 of 29 posterior draw(s) of toy exceeded eval_timeout=1s" in (
        caplog.text
    )
