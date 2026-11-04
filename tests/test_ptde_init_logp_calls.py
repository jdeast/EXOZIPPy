"""PTDE start-population build: how many logp calls, and where (review 2.6.45).

Before the fix every PTDE run evaluated, before its first step and serially
in the main process with no eval_timeout:

  - the rung report (report_rung_dispersion): every chain of every rung
    again, n_chains x n_rungs calls, for a log line;
  - the build itself (_make_starts): every jittered candidate, plus the
    seed(s) once per rung, while the sampler's cores sat idle.

On OGLE-2013-BLG-0341 (124 chains x 8 rungs, ~225 s per logp) the report
alone was ~62 h.  These tests count the calls so a serial per-chain loop
regresses loudly:

  1. Serially the build costs exactly one call per jittered candidate plus
     one per seed for the WHOLE ladder; the report costs none.
  2. With a pool the main process makes NO logp call at all, and the
     populations are bit-identical to the serial build, retries included.
  3. End to end, ptde_sample and ptde_async_sample with cores > 1 make no
     main-process logp call before (or during) sampling.
  4. A candidate that outlasts eval_timeout is scored -inf and redrawn, the
     pool is recycled, and the build completes.
"""

import logging
import multiprocessing as mp
import os
import time

import numpy as np
import pymc as pm
import pytest

from exozippy.samplers import _common
from exozippy.samplers._common import (
    StartEvaluator,
    build_rung_populations,
    create_pool,
    set_worker_globals,
)
from exozippy.samplers.ptde import ptde_sample
from exozippy.samplers.ptde_async import ptde_async_sample

requires_fork = pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="the sampler pool forks",
)

LADDER = np.array([1.0, 4.0, 25.0])
MAIN_PID = os.getpid()
_MAIN_CALLS = []


def _counting(logp):
    """Wrap ``logp`` so each call made IN THIS PROCESS is recorded.

    A forked worker appends to its own copy of the list, so the parent's
    count is exactly the main-process evaluations.
    """

    def _f(point):
        if os.getpid() == MAIN_PID:
            _MAIN_CALLS.append(1)
        return logp(point)

    return _f


def _quad(p):
    return -0.5 * float(np.sum(np.asarray(p["a"], dtype=float) ** 2))


def _boxed(p):
    """A logp with hard walls, so hot-rung jitter draws need retries."""
    a = np.asarray(p["a"], dtype=float)
    if np.any(np.abs(a) > 2.0) or a[0] > 1.8:
        return -np.inf
    return -0.5 * float(np.sum(a**2))


def _sleepy_boxed(p):
    """_boxed, but a candidate with a[1] > 0.5 hangs past any timeout."""
    if float(np.asarray(p["a"])[1]) > 0.5:
        time.sleep(30.0)
    return _boxed(p)


# Seed 2 sits outside the support (a[0] > 1.8): its exact start is -inf, so
# its chains take the jitter branch and every rung exercises retries.
SEEDS = [
    {"a": np.zeros(5)},
    {"a": np.full(5, 0.3)},
    {"a": np.array([1.9, 0.0, 0.0, 0.0, 0.0])},
]
SCALES = {"a": np.full(5, 0.3)}


def _build(
    logp, evaluator=None, seeds=SEEDS, n_chains=12, seed=5, ladder=LADDER
):
    return build_rung_populations(
        None,
        None,
        n_chains,
        logp,
        np.random.default_rng(seed),
        seeds[0],
        ladder,
        None,
        raw_starts=seeds,
        seed_indices=list(range(len(seeds))),
        raw_scales=SCALES,
        evaluator=evaluator,
    )


@pytest.fixture(autouse=True)
def _reset_calls():
    _MAIN_CALLS.clear()
    yield
    _MAIN_CALLS.clear()


def test_serial_build_scores_each_candidate_once_and_the_report_none():
    """
    Given one seed, a logp finite everywhere and a 3-rung ladder,
    When the per-rung populations are built serially,
    Then the logp is called once for the seed (shared by every rung) and
      once per jittered chain -- 1 + 3 x 5 = 16 -- and NOT again for the
      rung report, which used to add n_chains x n_rungs = 18 more (plus a
      MAP-lp and an exact-seed call per rung: 39 in all).
    """
    # ARRANGE
    logp = _counting(_quad)

    # ACT
    _build(logp, seeds=[{"a": np.zeros(5)}], n_chains=6)

    # ASSERT
    assert len(_MAIN_CALLS) == 1 + len(LADDER) * 5


@requires_fork
def test_pooled_build_is_bit_identical_and_makes_no_main_process_call(caplog):
    """
    Given three seeds (one outside the support) and a walled logp, so that
      hot rungs need many retries,
    When the populations are built on a 4-worker pool,
    Then the main process evaluates the logp ZERO times, and every rung's
      population is bit-identical to the serial build: the speculative
      batches rewind the rng to exactly where the one-at-a-time loop
      would stand.
    """
    # ARRANGE
    logp = _counting(_boxed)
    serial = _build(logp)[2]
    n_serial = len(_MAIN_CALLS)
    _MAIN_CALLS.clear()
    set_worker_globals(logp)
    pool, cores = create_pool(4, 100, "test", logging.getLogger(__name__))
    evaluator = StartEvaluator(logp, pool=pool, cores=cores)
    # The logger can already be at INFO (test order), so the serial build
    # above may have left its own rung lines in caplog: count only ours.
    caplog.clear()

    # ACT
    try:
        with caplog.at_level(logging.INFO):
            pooled = _build(logp, evaluator=evaluator)[2]
    finally:
        _common._shutdown_pool(evaluator.pool)

    # ASSERT
    assert n_serial > 3 * 12, "the fixture must exercise retries"
    assert _MAIN_CALLS == []
    for k, (ps, pp) in enumerate(zip(serial, pooled, strict=True)):
        assert len(ps) == len(pp) == 12
        for a, b in zip(ps, pp, strict=True):
            np.testing.assert_array_equal(a["a"], b["a"], err_msg=f"rung {k}")
    assert caplog.text.count("start lp median") == len(LADDER)


@requires_fork
def test_build_scores_a_hung_candidate_minus_inf_and_recycles(caplog):
    """
    Given a logp that hangs on some jitter candidates,
    When the populations are built on a pool with eval_timeout,
    Then those candidates score -inf and are redrawn like bound hits, the
      pool is recycled with an ERROR naming the timeout, and every rung
      still gets a full population of finite-logp starts that avoid the
      hanging region.
    """
    # ARRANGE
    if mp.cpu_count() < 2:
        pytest.skip("eval_timeout needs a worker pool")
    set_worker_globals(_sleepy_boxed)
    pool, cores = create_pool(4, 100, "test", logging.getLogger(__name__))
    evaluator = StartEvaluator(
        _sleepy_boxed, pool=pool, cores=cores, eval_timeout=0.5
    )
    seeds = [{"a": np.zeros(5)}]

    # ACT
    try:
        with caplog.at_level(logging.INFO):
            pops = _build(
                _sleepy_boxed,
                evaluator,
                seeds=seeds,
                n_chains=6,
                ladder=np.array([1.0, 2.0]),
            )[2]
    finally:
        _common._shutdown_pool(evaluator.pool)

    # ASSERT
    assert evaluator.n_timed_out > 0
    assert evaluator.pool is not pool
    assert "exceeded eval_timeout=" in caplog.text
    for pop in pops:
        assert len(pop) == 6
        for p in pop:
            assert float(p["a"][1]) <= 0.5
            assert np.isfinite(_boxed(p))


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


@requires_fork
@pytest.mark.parametrize("sampler", [ptde_sample, ptde_async_sample])
def test_ptde_with_a_pool_never_evaluates_in_the_main_process(sampler):
    """
    Given a 2-parameter model whose compiled logp counts main-process calls,
    When ptde (sync or async) runs on 2 cores with 3 rungs,
    Then the main process evaluates the logp zero times: the start build,
      the rung report and the sampling all run on the pool.  A serial
      per-chain loop anywhere in the prologue shows up here as at least
      n_chains x n_rungs calls.
    """
    # ARRANGE
    if mp.cpu_count() < 2:
        pytest.skip("needs a worker pool")
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=3.0, sigma=0.5)
    compiled = model.compile_logp()
    model.compile_logp = lambda *a, **k: _counting(compiled)
    raw_start = model.initial_point()
    raw_scales = {
        k: np.ones_like(np.asarray(v, dtype=float))
        for k, v in raw_start.items()
    }

    # ACT
    idata = sampler(
        model,
        _MinimalSystem(),
        draws=5,
        tune=5,
        n_temps=3,
        T_max=4.0,
        n_chains=6,
        cores=2,
        seed=3,
        log_interval=10**6,
        raw_scales=raw_scales,
        min_ess=None,
        max_rhat=None,
    )

    # ASSERT
    assert idata.posterior.sizes["chain"] == 6
    assert _MAIN_CALLS == []
