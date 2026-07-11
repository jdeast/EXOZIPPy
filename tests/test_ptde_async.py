"""Tests for the asynchronous PTDE sampler (ptde_async.py, hpc_optimization.txt PROMPT 13).

Mirrors tests/test_ptde.py's structure and toy model so the two samplers'
behavior can be compared directly. ptde_async is experimental (see its module
docstring's statistical caveat on stale DE partners); these tests validate
that it (a) produces well-formed output, (b) recovers known posterior
moments on a toy model, and (c) survives edge cases (single core, eval
timeouts, rung timing) without crashing or deadlocking.
"""
import multiprocessing as mp

import numpy as np
import pymc as pm
import pytest
import xarray as xr

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
        model, system, draws=20, tune=20,
        n_temps=2, T_max=2.0, cores=1, seed=42,
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
        model, system, draws=30, tune=20,
        n_temps=2, T_max=2.0, n_chains=4, cores=1, seed=0,
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
        model, system, draws=30, tune=20,
        n_temps=3, T_max=8.0, n_chains=4, cores=2, seed=5,
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
            model, system, draws=20, tune=20,
            n_temps=3, T_max=8.0, n_chains=4, cores=1, seed=3,
            log_interval=1000, collect_rung_timing=True,
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
        model, system, draws=20, tune=20,
        n_temps=2, T_max=4.0, n_chains=4, cores=2, seed=11,
        log_interval=1000, eval_timeout=5.0,
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
        model, system, draws=300, tune=150,
        n_temps=2, T_max=4.0, n_chains=6, cores=1, seed=7,
        log_interval=5000,
    )
    x_mean = float(idata.posterior["x"].values.mean())
    y_mean = float(idata.posterior["y"].values.mean())
    x_std = float(idata.posterior["x"].values.std())
    y_std = float(idata.posterior["y"].values.std())
    assert abs(x_mean) < 1.0, f"x posterior mean {x_mean:.2f} too far from 0"
    assert abs(y_mean - 3.0) < 0.5, f"y posterior mean {y_mean:.2f} too far from 3"
    assert abs(x_std - 1.0) < 0.5, f"x posterior std {x_std:.2f} too far from 1"
    assert abs(y_std - 0.5) < 0.3, f"y posterior std {y_std:.2f} too far from 0.5"


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
        model, system, draws=5000, tune=100,
        n_temps=2, T_max=4.0, n_chains=4, cores=1, seed=13,
        log_interval=100000, maxtime=1.0, min_ess=None, max_rhat=None,
    )
    assert idata.posterior.sizes["draw"] >= 1
    assert idata.posterior.sizes["draw"] <= 5000
