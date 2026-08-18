"""Tests for the asynchronous PTDE sampler (ptde_async.py, hpc_optimization.txt PROMPT 13).

Mirrors tests/test_ptde.py's structure and toy model so the two samplers'
behavior can be compared directly. ptde_async is the recommended default for
Op-based models (see its module docstring for the stale-DE-partner caveat);
these tests validate that it (a) produces well-formed output, (b) recovers
known posterior moments on a toy model, and (c) survives edge cases (single
core, eval timeouts, rung timing) without crashing or deadlocking.
"""

import logging
import multiprocessing as mp
import os
import time

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from conftest import requires_fork
from exozippy.samplers.ptde_async import ptde_async_sample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinimalSystem:
    """Minimal system stub for ptde_async_sample: supplies raw_start from model."""

    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _simple_model():
    """2-D standard normal — fast, gradient-free friendly, known mean/std."""
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=3.0, sigma=0.5)
    return model


# ---------------------------------------------------------------------------
# n_chains default
# ---------------------------------------------------------------------------


def test_n_chains_defaults_to_twice_n_params():
    """
    Given a 2-parameter model and no explicit n_chains,
    When ptde_async_sample runs,
    Then the returned posterior has 2*2=4 chains.
    """
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_async_sample(
        model,
        system,
        draws=20,
        tune=20,
        n_temps=2,
        T_max=2.0,
        cores=1,
        seed=42,
        log_interval=1000,
    )
    assert idata.posterior.sizes["chain"] == 4


# ---------------------------------------------------------------------------
# End-to-end: InferenceData structure
# ---------------------------------------------------------------------------


def test_ptde_async_returns_inferencedata_with_expected_structure():
    """
    Given a simple 2-D normal model,
    When ptde_async_sample runs with minimal settings,
    Then the result is an InferenceData with posterior vars x and y,
      sample_stats contains lp, and every chain has exactly `draws` samples.
    """
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_async_sample(
        model,
        system,
        draws=30,
        tune=20,
        n_temps=2,
        T_max=2.0,
        n_chains=4,
        cores=1,
        seed=0,
        log_interval=1000,
    )

    assert isinstance(idata, xr.DataTree)
    assert hasattr(idata, "posterior")
    post = idata.posterior
    assert "x" in post.data_vars
    assert "y" in post.data_vars
    assert "chain" in post.dims
    assert "draw" in post.dims
    assert post.sizes["draw"] == 30
    assert post.sizes["chain"] == 4
    assert "lp" in idata.sample_stats.data_vars


@requires_fork
def test_ptde_async_runs_with_multiple_cores():
    """
    Given cores>1 (a real fork Pool, not the serial fallback),
    When ptde_async_sample runs,
    Then the event-driven dispatch loop completes without deadlocking and
      returns the expected number of draws per chain.
    """
    if mp.cpu_count() < 2:
        pytest.skip("needs at least 2 cores to exercise the pool path")
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_async_sample(
        model,
        system,
        draws=30,
        tune=20,
        n_temps=3,
        T_max=8.0,
        n_chains=4,
        cores=2,
        seed=5,
        log_interval=1000,
    )
    assert idata.posterior.sizes["draw"] == 30
    for i in range(4):
        assert not np.any(np.isnan(idata.posterior["x"].values[i]))


# ---------------------------------------------------------------------------
# collect_rung_timing diagnostic
# ---------------------------------------------------------------------------


def test_ptde_async_collect_rung_timing_runs_end_to_end(caplog):
    """
    Given collect_rung_timing=True on a multi-temp model,
    When ptde_async_sample runs,
    Then it completes normally and logs a per-rung timing summary line
      for every rung.
    """
    model = _simple_model()
    system = _MinimalSystem()
    with caplog.at_level("INFO", logger="exozippy.samplers.ptde_async"):
        idata = ptde_async_sample(
            model,
            system,
            draws=20,
            tune=20,
            n_temps=3,
            T_max=8.0,
            n_chains=4,
            cores=1,
            seed=3,
            log_interval=1000,
            collect_rung_timing=True,
        )
    assert idata.posterior.sizes["draw"] == 20
    messages = "\n".join(r.message for r in caplog.records)
    assert "PTDE-async per-rung logp timing" in messages
    for k in range(3):
        assert f"rung {k}" in messages


# ---------------------------------------------------------------------------
# eval_timeout smoke test (no hangs to trigger it here, just verify it
# doesn't break normal completion when enabled)
# ---------------------------------------------------------------------------


@requires_fork
def test_ptde_async_with_eval_timeout_runs_end_to_end():
    """
    Given eval_timeout set on a model whose logp always evaluates quickly,
    When ptde_async_sample runs,
    Then it completes normally with zero timeouts triggered.
    """
    if mp.cpu_count() < 2:
        pytest.skip("eval_timeout has no effect with a single core")
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_async_sample(
        model,
        system,
        draws=20,
        tune=20,
        n_temps=2,
        T_max=4.0,
        n_chains=4,
        cores=2,
        seed=11,
        log_interval=1000,
        eval_timeout=5.0,
    )
    assert idata.posterior.sizes["draw"] == 20


# ---------------------------------------------------------------------------
# Correctness: posterior recovery on a toy model with known moments
# ---------------------------------------------------------------------------


def test_ptde_async_posterior_mean_near_true_values():
    """
    Given a 2-D normal model with known mean (x=0, y=3),
    When ptde_async_sample runs with enough draws,
    Then the posterior mean of each variable is within tolerance of the
      truth -- validating that stale-DE-partner proposals (this sampler's
      core statistical caveat, see its module docstring) do not visibly
      bias recovery on this toy model.
    """
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_async_sample(
        model,
        system,
        draws=300,
        tune=150,
        n_temps=2,
        T_max=4.0,
        n_chains=6,
        cores=1,
        seed=7,
        log_interval=5000,
    )
    x_mean = float(idata.posterior["x"].values.mean())
    y_mean = float(idata.posterior["y"].values.mean())
    x_std = float(idata.posterior["x"].values.std())
    y_std = float(idata.posterior["y"].values.std())
    assert abs(x_mean) < 1.0, f"x posterior mean {x_mean:.2f} too far from 0"
    assert abs(y_mean - 3.0) < 0.5, (
        f"y posterior mean {y_mean:.2f} too far from 3"
    )
    assert abs(x_std - 1.0) < 0.5, (
        f"x posterior std {x_std:.2f} too far from 1"
    )
    assert abs(y_std - 0.5) < 0.3, (
        f"y posterior std {y_std:.2f} too far from 0.5"
    )


def test_ptde_async_early_stop_via_maxtime():
    """
    Given a very small maxtime on a model that would otherwise take longer,
    When ptde_async_sample runs,
    Then it stops early and still returns a valid (possibly shorter)
      InferenceData rather than hanging or crashing.
    """
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_async_sample(
        model,
        system,
        draws=5000,
        tune=100,
        n_temps=2,
        T_max=4.0,
        n_chains=4,
        cores=1,
        seed=13,
        log_interval=100000,
        maxtime=1.0,
        min_ess=None,
        max_rhat=None,
    )
    assert idata.posterior.sizes["draw"] >= 1
    assert idata.posterior.sizes["draw"] <= 5000


def test_ptde_async_warns_once_when_t1_lp_exceeds_plausibility_ceiling(
    caplog,
):
    """
    Given a plausibility ceiling set far below any lp this model can
    legitimately reach,
    When ptde_async_sample runs and a T=1 chain's accepted lp exceeds it,
    Then a single loud warning is logged (not one per evaluation) naming the
      offending chain and lp -- the same runaway-lp early-detection guard
      the synchronous sampler has (the two used to drift here: sync had the
      check, async silently lacked it; code_review_20260808.txt 1.15/sec 4).
    """
    model = _simple_model()
    system = _MinimalSystem()
    with caplog.at_level("WARNING", logger="exozippy.samplers.ptde_async"):
        ptde_async_sample(
            model,
            system,
            draws=20,
            tune=20,
            n_temps=2,
            T_max=2.0,
            n_chains=4,
            cores=1,
            seed=1,
            log_interval=100,
            lp_plausibility_ceiling=0.1,
        )
    warnings = [
        r.message
        for r in caplog.records
        if "plausibility ceiling" in r.message
    ]
    assert len(warnings) == 1, (
        f"expected exactly one plausibility-ceiling warning, got "
        f"{len(warnings)}: {warnings}"
    )
    assert "T=1 chain" in warnings[0]


def test_ptde_async_freezes_gamma_when_first_chain_starts_recording(caplog):
    """
    Given adapt_gamma=True (the default) and asynchronous per-chain pacing,
    When the first T=1 chain finishes its tune phase,
    Then gamma is frozen (logged once) so recorded draws never come from a
      kernel that slower chains' tune-phase proposals are still mutating
      (code_review_20260808.txt 1.15c).
    """
    model = _simple_model()
    system = _MinimalSystem()
    with caplog.at_level("INFO", logger="exozippy.samplers.ptde_async"):
        ptde_async_sample(
            model,
            system,
            draws=30,
            tune=30,
            n_temps=2,
            T_max=2.0,
            n_chains=4,
            cores=1,
            seed=2,
            log_interval=1000,
        )
    freeze_msgs = [
        r.message for r in caplog.records if "gamma: frozen at" in r.message
    ]
    assert len(freeze_msgs) == 1, (
        f"expected exactly one gamma-freeze message, got {freeze_msgs}"
    )


# ---------------------------------------------------------------------------
# review 2.9.4: n_chains = 2 (reachable at n_params = 1 with the default
# 2 * n_params) has no valid DE move.
# ---------------------------------------------------------------------------


def test_pick_two_raises_a_clear_error_below_three_chains():
    """
    Given a population of 2,
    When a DE proposal tries to pick two OTHER members,
    Then it raises with a message naming n_chains and the minimum.

    Regression (notes/code_review_20260808.txt 2.9.4): this died inside
    numpy as "Cannot take a larger sample than population when replace is
    False", which says nothing about n_chains.
    """
    from exozippy.samplers._common import _pick_two

    with pytest.raises(ValueError, match="two other population members"):
        _pick_two(np.random.default_rng(0), 2, 0)
    # three is enough to form a difference vector (degenerately: the two
    # others are forced) and must still work
    assert _pick_two(np.random.default_rng(0), 3, 0) in [(1, 2), (2, 1)]


def test_default_n_chains_is_floored_above_the_de_minimum():
    """
    Given a one-parameter model, where the default n_chains = 2 * n_params
      would be 2,
    When the population size is resolved,
    Then it is floored at MIN_DE_CHAINS so the DEFAULT can never land on a
      population too small to make a DE move.
    """
    import logging

    from exozippy.samplers._common import MIN_DE_CHAINS, resolve_n_chains

    log = logging.getLogger("test")
    assert resolve_n_chains(None, 1, "PTDE", log) == MIN_DE_CHAINS
    # the default is untouched wherever it is already large enough
    assert resolve_n_chains(None, 6, "PTDE", log) == 12


def test_explicitly_requesting_two_chains_raises_rather_than_being_bumped():
    """
    Given a user explicitly asking for n_chains = 2,
    When the population size is resolved,
    Then it RAISES with a message explaining why -- quietly running a
      different sampler than the one requested is worse than stopping, and
      unlike the default case there is a human to tell.
    """
    import logging

    from exozippy.samplers._common import resolve_n_chains

    log = logging.getLogger("test")
    with pytest.raises(ValueError, match="two OTHER population members"):
        resolve_n_chains(2, 1, "PTDE-async", log)


def test_one_parameter_model_samples_end_to_end_on_the_default_population():
    """
    Given a one-parameter model and no explicit n_chains -- exactly the
      configuration that used to crash inside numpy,
    When the async sampler runs,
    Then it completes and returns MIN_DE_CHAINS chains of draws.
    """
    from exozippy.samplers._common import MIN_DE_CHAINS

    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)

    idata = ptde_async_sample(
        model,
        _MinimalSystem(),
        draws=30,
        tune=10,
        n_temps=2,
        T_max=4.0,
        cores=1,
        seed=5,
        log_interval=10000,
    )

    assert idata.posterior.sizes["chain"] == MIN_DE_CHAINS
    assert idata.posterior.sizes["draw"] == 30


# ---------------------------------------------------------------------------
# eval_timeout enforcement while the result queue is BUSY (review 1.4.1)
# ---------------------------------------------------------------------------


def _one_shot_hanging_logp(hang_flag, threshold=-5.0):
    """A logp that hangs forever on the FIRST proposal below `threshold`.

    The one-shot budget is claimed with an atomic O_EXCL create, so it holds
    across the pool recycle the timeout recovery performs -- a per-process
    counter would be reset by the fresh fork and the slot would hang again
    on every retry, forever.
    """

    def _logp(point):
        x = float(np.asarray(point["x"]))
        if x < threshold:
            try:
                fd = os.open(
                    str(hang_flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError:
                fd = None
            if fd is not None:
                os.close(fd)
                while True:  # the unbreakable loop eval_timeout exists for
                    time.sleep(0.05)
        return -0.5 * x * x

    return _logp


@requires_fork
def test_hung_slot_is_written_off_while_other_slots_keep_the_queue_busy(
    tmp_path, caplog
):
    """
    Given one T=1 slot whose first logp call hangs forever while every other
      slot resolves in microseconds -- so the result queue is never empty and
      `queue.Empty` (the only place the stale scan used to run) never fires,
    When ptde_async_sample runs with eval_timeout well below maxtime,
    Then the hung submission is written off on the RESULT path, its slot
      resubmits, and every chain reaches the draw target -- instead of the
      run stalling until maxtime with min(per_chain_draws) stuck at zero
      (review 1.4.1).
    """
    # Arrange: chain 0 starts inside the hang region; the rest do not.
    if mp.cpu_count() < 2:
        pytest.skip("eval_timeout has no effect with a single core")
    hang_flag = tmp_path / "hang.claimed"
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
    model.compile_logp = lambda *a, **k: _one_shot_hanging_logp(hang_flag)
    n_chains = 6
    initvals = [{"x": np.array(-6.0)}] + [
        {"x": np.array(0.1 * j)} for j in range(1, n_chains)
    ]

    # Act
    t0 = time.time()
    with caplog.at_level(logging.ERROR, logger="exozippy.samplers.ptde_async"):
        idata = ptde_async_sample(
            model,
            _MinimalSystem(),
            draws=15,
            tune=5,
            n_temps=1,
            n_chains=n_chains,
            cores=2,
            initvals=initvals,
            seed=5,
            log_interval=100000,
            eval_timeout=0.5,
            maxtime=60.0,
            min_ess=None,
            max_rhat=None,
        )
    elapsed = time.time() - t0

    # Assert
    assert hang_flag.exists(), "the hang was never triggered; test is inert"
    assert "logp call exceeded eval_timeout" in caplog.text
    assert idata.posterior.sizes["draw"] == 15
    assert elapsed < 55.0, (
        f"the run took {elapsed:.1f}s, i.e. it stalled on the hung slot "
        f"instead of writing it off"
    )
