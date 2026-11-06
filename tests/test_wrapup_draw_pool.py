"""Wrap-up evaluations over posterior draws: where they run, how they are
chunked, and what a hung call costs (review 2.6.44).

Four wrap-up stages re-evaluate the model after sampling:

  - lp from the model, for a sampler that writes none (run._ensure_lp);
  - the branch weights of a V_c/V_e fit (branches.resolve_branch_draws);
  - the evidence bridge (outputs.evidence._batch_logp, opt-in);
  - the hot-chain discovery's midpoint test (outputs.ledger).

The first three forked one task per CHAIN (or per worker block) with no
eval_timeout, so one hung call held a finished fit's wrap-up forever; the
fourth ran serially in the main process.  At ~225 s per logp
(OGLE-2013-BLG-0341) a serial or untimed wrap-up of hours can lose the
whole run's products.  These tests count main-process calls and the task
ranges so a per-chain or serial loop regresses loudly, and hang one call
past eval_timeout to show it is written off, counted and reported -- and
that every other draw's value is exactly what it would have been.
"""

import logging
import multiprocessing as mp
import os
import time

import arviz as az
import numpy as np
import pytest

from exozippy.samplers import _common
from exozippy.samplers._common import map_draws

requires_fork = pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="the wrap-up pool forks",
)

MAIN_PID = os.getpid()
_MAIN_CALLS = []

# Fork-inherited state for the module-level range worker below.
_DATA = None
_HANG_AT = None
_SLEEP = 0.0


def _value(i):
    if os.getpid() == MAIN_PID:
        _MAIN_CALLS.append(i)
    if _SLEEP:
        time.sleep(_SLEEP)
    if _HANG_AT is not None and i in _HANG_AT and os.getpid() != MAIN_PID:
        time.sleep(60.0)  # a hung call; the pool recycle kills it
    return float(np.sin(_DATA[i]) * 1e3)


def _range_worker(lo, hi):
    return [_value(i) for i in range(lo, hi)]


@pytest.fixture(autouse=True)
def _reset():
    global _DATA, _HANG_AT, _SLEEP
    _MAIN_CALLS.clear()
    _DATA = np.linspace(0.0, 10.0, 200)
    _HANG_AT = None
    _SLEEP = 0.0
    yield
    _DATA, _HANG_AT, _SLEEP = None, None, 0.0


def _spy_ranges(monkeypatch):
    """Record every (lo, hi) range map_draws dispatches (parent side)."""
    seen = []
    real = _common._map_logp_timeout

    def _spy(pool, tasks, timeout, **kw):
        seen.extend((lo, hi) for _w, lo, hi in tasks)
        return real(pool, tasks, timeout, **kw)

    monkeypatch.setattr(_common, "_map_logp_timeout", _spy)
    return seen


# ---------------------------------------------------------------------------
# map_draws itself
# ---------------------------------------------------------------------------


def test_serial_map_draws_is_one_in_process_pass():
    """
    Given cores=1,
    When map_draws runs,
    Then every draw is evaluated once, in the main process, in order.
    """
    out, timed_out = map_draws(
        _range_worker, 200, cores=1, eval_timeout=None, label="t"
    )
    assert timed_out == []
    assert _MAIN_CALLS == list(range(200))
    assert out == [float(np.sin(x) * 1e3) for x in _DATA]


@requires_fork
def test_pooled_map_draws_makes_no_main_process_call_and_splits_below_a_chain(
    monkeypatch,
):
    """
    Given 200 draws (think 4 chains x 50) on 4 workers, ~10 ms per draw,
    When map_draws runs,
    Then the main process evaluates nothing, every draw is covered exactly
      once, no task holds a whole chain's 50 draws, and the values equal
      the serial pass bit for bit.
    """
    global _SLEEP
    _SLEEP = 0.01
    monkeypatch.setattr(_common, "DRAW_TASK_TARGET_S", 0.1)
    seen = _spy_ranges(monkeypatch)

    out, timed_out = map_draws(
        _range_worker, 200, cores=4, eval_timeout=30.0, label="t"
    )

    assert timed_out == []
    assert _MAIN_CALLS == []
    covered = sorted(i for lo, hi in seen for i in range(lo, hi))
    assert covered == list(range(200))
    assert max(hi - lo for lo, hi in seen) < 50
    # the first batch is one draw per worker (it measures the draw time)
    assert seen[:4] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert out == [float(np.sin(x) * 1e3) for x in _DATA]


@requires_fork
def test_a_slow_model_gets_one_draw_per_task(monkeypatch):
    """
    Given a model whose one call is longer than DRAW_TASK_TARGET_S,
    When map_draws runs,
    Then every task is a single draw, so eval_timeout bounds each call
      exactly.
    """
    global _SLEEP
    _SLEEP = 0.02
    monkeypatch.setattr(_common, "DRAW_TASK_TARGET_S", 0.01)
    seen = _spy_ranges(monkeypatch)

    map_draws(_range_worker, 24, cores=4, eval_timeout=30.0, label="t")

    assert all(hi - lo == 1 for lo, hi in seen)
    assert len(seen) == 24


@requires_fork
def test_a_hung_draw_is_written_off_counted_and_reported(monkeypatch, caplog):
    """
    Given one draw that hangs past eval_timeout, inside a multi-draw range,
    When map_draws runs,
    Then only THAT draw comes back None (its range is re-run one draw per
      task, so its neighbours are recovered), it is listed in timed_out,
      an ERROR names the count, and every other value is exact.
    """
    global _HANG_AT
    _HANG_AT = {137}
    seen = _spy_ranges(monkeypatch)
    caplog.clear()

    t0 = time.monotonic()
    with caplog.at_level(logging.INFO):
        out, timed_out = map_draws(
            _range_worker, 200, cores=4, eval_timeout=1.0, label="hangtest"
        )
    elapsed = time.monotonic() - t0

    assert timed_out == [137]
    assert out[137] is None
    exact = [float(np.sin(x) * 1e3) for x in _DATA]
    assert [v for i, v in enumerate(out) if i != 137] == [
        v for i, v in enumerate(exact) if i != 137
    ]
    # 137 was first dispatched inside a range, then alone
    assert any(lo <= 137 < hi and hi - lo > 1 for lo, hi in seen)
    assert (137, 138) in seen
    assert "hangtest: 1 of 200 draw(s) exceeded eval_timeout=1s" in caplog.text
    assert elapsed < 30.0  # bounded by the timeouts, not by the 60 s hang
    assert _MAIN_CALLS == []


# ---------------------------------------------------------------------------
# The three callers
# ---------------------------------------------------------------------------


class _FakeRV:
    def __init__(self, name):
        self.name = name


class _Model:
    """Just enough of a PyMC model for run._compute_lp_from_model."""

    def __init__(self, hang_at=None):
        self.free_RVs = [_FakeRV("x")]
        self.rvs_to_values = {rv: _FakeRV("x_value") for rv in self.free_RVs}
        self.hang_at = hang_at

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def compile_logp(self, jacobian=False):
        hang_at = self.hang_at

        def _lp(point):
            x = float(np.asarray(point["x_value"]).ravel()[0])
            if os.getpid() == MAIN_PID:
                _MAIN_CALLS.append(x)
            if hang_at is not None and x == hang_at:
                time.sleep(60.0)
            return -0.5 * x * x

        return _lp


def _trace(n_chains=4, n_draws=50):
    x = np.arange(n_chains * n_draws, dtype=float).reshape(n_chains, n_draws)
    return az.from_dict({"posterior": {"x": x}})


@requires_fork
def test_ensure_lp_pooled_is_exact_and_never_in_the_main_process():
    """
    Given a 4 x 50 trace with no lp and cores=4,
    When _ensure_lp computes it,
    Then the main process evaluates nothing, lp is exactly the serial
      answer, and the trace records zero timeouts.
    """
    from exozippy.run import _ensure_lp

    idata = _trace()
    assert _ensure_lp(idata, _Model(), cores=4, eval_timeout=30.0)
    assert _MAIN_CALLS == []
    x = idata.posterior["x"].values
    np.testing.assert_array_equal(
        idata.sample_stats["lp"].values, -0.5 * x * x
    )
    assert idata.sample_stats["lp"].attrs["n_eval_timeout"] == 0


@requires_fork
def test_ensure_lp_writes_a_hung_draw_off_as_nan_and_counts_it(caplog):
    """
    Given a draw whose logp hangs past eval_timeout,
    When _ensure_lp computes lp,
    Then that draw's lp is NaN (an invalid draw to the mode pass), the
      count is on the trace and in a WARNING, and the rest are exact.
    """
    from exozippy.run import _ensure_lp

    idata = _trace()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _ensure_lp(idata, _Model(hang_at=77.0), cores=4, eval_timeout=1.0)

    lp = idata.sample_stats["lp"].values
    x = idata.posterior["x"].values
    assert np.isnan(lp[1, 27]) and x[1, 27] == 77.0
    mask = np.ones_like(lp, dtype=bool)
    mask[1, 27] = False
    np.testing.assert_array_equal(lp[mask], (-0.5 * x * x)[mask])
    assert idata.sample_stats["lp"].attrs["n_eval_timeout"] == 1
    assert "lp: 1 of 200 draw(s) exceeded eval_timeout=1s" in caplog.text


def _toy_branch():
    """test_branch_draws' toy: d = x + 1 (primary) or x - 1 (alternative),
    both branches competing at every x."""
    import pymc as pm

    from exozippy.system import System

    system = System.__new__(System)
    system._branch_alternatives = []
    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 0.1)
        d = pm.Deterministic("d", x + 1.0)
        pm.Deterministic("g", 2.0 * d)
        pm.Normal("y", mu=d, sigma=1.0, observed=np.array([0.0]))
        system.register_branch_alternative("toy", {d: d - 2.0})
        system._add_branch_mixtures(model)
    xs = np.random.default_rng(3).normal(0.0, 0.1, size=(4, 50))
    idata = az.from_dict(
        {"posterior": {"x": xs, "d": xs + 1.0, "g": 2.0 * (xs + 1.0)}}
    )
    idata.posterior.attrs["random_seed"] = 1234
    return system, model, idata


@requires_fork
def test_branch_weights_pooled_equal_serial_with_no_main_process_eval(
    monkeypatch,
):
    """
    Given the toy branch model,
    When resolve_branch_draws runs on 4 workers and on 1,
    Then the drawn combinations and the regenerated Deterministics are
      identical, and the pooled run never evaluates in the main process.
    """
    from exozippy import branches

    real = branches._compile_evaluator

    def _counting(system, model):
        fn, names, dets = real(system, model)

        def _f(*args):
            if os.getpid() == MAIN_PID:
                _MAIN_CALLS.append(1)
            return fn(*args)

        return _f, names, dets

    monkeypatch.setattr(branches, "_compile_evaluator", _counting)

    system, model, idata_a = _toy_branch()
    _s, _m, idata_b = _toy_branch()
    branches.resolve_branch_draws(system, model, idata_a, cores=1)
    n_serial = len(_MAIN_CALLS)
    _MAIN_CALLS.clear()
    branches.resolve_branch_draws(
        system, model, idata_b, cores=4, eval_timeout=30.0
    )

    assert n_serial == 200
    assert _MAIN_CALLS == []
    np.testing.assert_array_equal(
        idata_a.sample_stats["branch_combination"].values,
        idata_b.sample_stats["branch_combination"].values,
    )
    np.testing.assert_array_equal(
        idata_a.posterior["d"].values, idata_b.posterior["d"].values
    )
    assert (
        idata_b.sample_stats["branch_combination"].attrs["n_eval_timeout"] == 0
    )


@requires_fork
def test_a_hung_branch_draw_is_unresolved_and_the_rest_unchanged(
    monkeypatch, caplog
):
    """
    Given one draw whose branch evaluation hangs past eval_timeout,
    When resolve_branch_draws runs,
    Then that draw is UNRESOLVED (combination -1, as-built Deterministics,
      out of the probabilities), counted on the trace, in a WARNING and in
      the summary lines, and every other draw's combination is exactly the
      one a clean run draws.
    """
    from exozippy import branches

    system, model, clean = _toy_branch()
    branches.resolve_branch_draws(system, model, clean, cores=1)

    system, model, idata = _toy_branch()
    hang_x = float(idata.posterior["x"].values[2, 11])
    d_before = float(idata.posterior["d"].values[2, 11])
    real = branches._compile_evaluator

    def _hanging(system, model):
        fn, names, dets = real(system, model)

        def _f(*args):
            if float(np.asarray(args[0]).ravel()[0]) == hang_x:
                time.sleep(60.0)
            return fn(*args)

        return _f, names, dets

    monkeypatch.setattr(branches, "_compile_evaluator", _hanging)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        branches.resolve_branch_draws(
            system, model, idata, cores=4, eval_timeout=1.0
        )

    z = idata.sample_stats["branch_combination"].values
    z_clean = clean.sample_stats["branch_combination"].values
    assert z[2, 11] == -1
    mask = np.ones_like(z, dtype=bool)
    mask[2, 11] = False
    np.testing.assert_array_equal(z[mask], z_clean[mask])
    assert float(idata.posterior["d"].values[2, 11]) == d_before
    attrs = idata.sample_stats["branch_combination"].attrs
    assert attrs["n_eval_timeout"] == 1
    assert "1 of 200 draw(s) exceeded eval_timeout=1s" in caplog.text
    assert any(
        "1 draw(s) UNRESOLVED" in ln
        for ln in branches.branch_summary_lines(idata)
    )


class _EvModel(_Model):
    def compile_logp(self, jacobian=True):
        return super().compile_logp(jacobian)


@requires_fork
def test_evidence_batch_logp_is_pooled_timed_and_counted():
    """
    Given 120 bridge points, one of which hangs past eval_timeout,
    When _batch_logp evaluates them on 4 workers,
    Then the main process evaluates nothing, the hung point is NaN (a
      failed evaluation to the bridge) and listed, and the rest are exact.
    """
    from exozippy.outputs.evidence import _batch_logp

    pts = np.arange(120, dtype=float).reshape(120, 1)
    layout = [("x", "x_value", 0, 1, (1,))]
    out, timed_out = _batch_logp(
        _EvModel(hang_at=42.0), layout, pts, cores=4, eval_timeout=1.0
    )
    assert _MAIN_CALLS == []
    assert timed_out == [42]
    assert np.isnan(out[42])
    keep = np.arange(120) != 42
    np.testing.assert_array_equal(out[keep], -0.5 * pts[keep, 0] ** 2)


# ---------------------------------------------------------------------------
# The hot-chain midpoint test
# ---------------------------------------------------------------------------


def _midpoint_lp(row):
    """Two basins at 0 and 10 with a valley between; flat inside each."""
    x = float(np.asarray(row).ravel()[0])
    if os.getpid() == MAIN_PID:
        _MAIN_CALLS.append(x)
    if _HANG_AT is not None and x in _HANG_AT and os.getpid() != MAIN_PID:
        time.sleep(60.0)
    return 0.0 if (abs(x) < 3.0 or abs(x - 10.0) < 3.0) else -100.0


def _candidates():
    # sorted best first, as discover_hot_modes hands them over
    xs = [0.0, 10.0, 1.0, 11.0, 0.5, 9.0, 2.0]
    return [(0.0, np.array([x])) for x in xs]


def test_midpoint_dedup_serial_is_the_one_at_a_time_loop():
    """
    Given cores=1,
    When the midpoint test runs,
    Then it keeps one candidate per basin, evaluating in the main process
      exactly the midpoints the old loop did (stopping at the first
      duplicate).
    """
    from exozippy.outputs.ledger import _dedup_by_midpoint

    kept = _dedup_by_midpoint(_candidates(), _midpoint_lp, 1, None, {})
    assert [float(x[0]) for _lp, x in kept] == [0.0, 10.0]
    # c=10 vs 0 (valley: kept); 1 vs 0 (dup); 11 vs 0, 10 (dup at 10);
    # 0.5 vs 0 (dup); 9 vs 0, 10; 2 vs 0
    assert _MAIN_CALLS == [5.0, 0.5, 5.5, 10.5, 0.25, 4.5, 9.5, 1.0]


@requires_fork
def test_midpoint_dedup_pooled_same_verdict_no_main_process_call():
    """
    Given cores=4,
    When the midpoint test runs,
    Then the kept set is the serial one and the main process evaluates
      nothing.
    """
    from exozippy.outputs.ledger import _dedup_by_midpoint

    kept = _dedup_by_midpoint(_candidates(), _midpoint_lp, 4, 30.0, {})
    assert [float(x[0]) for _lp, x in kept] == [0.0, 10.0]
    assert _MAIN_CALLS == []


@requires_fork
def test_a_hung_midpoint_keeps_the_candidate_and_is_counted(caplog):
    """
    Given the midpoint between candidates 0 and 1 hanging past eval_timeout,
    When the midpoint test runs,
    Then the call scores -inf (a valley), candidate 1 is KEPT -- the
      conservative direction -- and the timeout is counted in the status
      and logged.
    """
    global _HANG_AT
    from exozippy.outputs.ledger import _dedup_by_midpoint

    _HANG_AT = {0.5}
    status = {}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        kept = _dedup_by_midpoint(_candidates(), _midpoint_lp, 4, 1.0, status)
    assert [float(x[0]) for _lp, x in kept][:3] == [0.0, 10.0, 1.0]
    assert status["n_midpoint_eval_timeouts"] == 1
    assert "midpoint logp call(s) exceeded eval_timeout=1s" in caplog.text
