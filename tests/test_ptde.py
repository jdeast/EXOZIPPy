"""Tests for the PTDE (Parallel Tempering + Differential Evolution) sampler."""
import multiprocessing as mp

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from exozippy.samplers.ptde import _active_rungs, _geometric_ladder, ptde_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MinimalSystem:
    """Minimal system stub for ptde_sample: supplies raw_start from model."""
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
# _geometric_ladder
# ---------------------------------------------------------------------------

def test_geometric_ladder_single_temp_returns_one():
    """
    Given n_temps=1,
    When _geometric_ladder is called,
    Then it returns [1.0] regardless of T_max.
    """
    result = _geometric_ladder(1, T_max=200.0)
    assert list(result) == [1.0]


def test_geometric_ladder_first_is_one_last_is_T_max():
    """
    Given n_temps=8, T_max=200,
    When _geometric_ladder is called,
    Then the ladder starts at T=1 (target) and ends at T=T_max (hottest).
    """
    temps = _geometric_ladder(8, T_max=200.0)
    assert len(temps) == 8
    assert np.isclose(temps[0], 1.0)
    assert np.isclose(temps[-1], 200.0)


def test_geometric_ladder_is_monotonically_increasing():
    """
    Given any valid n_temps and T_max > 1,
    When _geometric_ladder is called,
    Then every rung is strictly hotter than the previous.
    """
    temps = _geometric_ladder(5, T_max=50.0)
    assert all(temps[i] < temps[i + 1] for i in range(len(temps) - 1))


# ---------------------------------------------------------------------------
# cores formula
# ---------------------------------------------------------------------------

def test_default_cores_is_within_physical_bounds():
    """
    Given no user-specified cores,
    When the default formula max(1, min(int(phys*0.75), phys-1)) is applied,
    Then the result is at least 1 and strictly less than the physical count.
    """
    phys = mp.cpu_count()
    cores = max(1, min(int(phys * 0.75), phys - 1))
    assert cores >= 1
    assert cores < phys or phys == 1


# ---------------------------------------------------------------------------
# n_chains default
# ---------------------------------------------------------------------------

def test_n_chains_defaults_to_twice_n_params():
    """
    Given a 2-parameter model and no explicit n_chains,
    When ptde_sample runs,
    Then the returned posterior has 2*2=4 chains.
    """
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_sample(
        model, system, draws=20, tune=20,
        n_temps=2, T_max=2.0, cores=1, seed=42,
        log_interval=100,
    )
    # 2 params → n_chains = 4; 2 temps × 4 chains = 8 total chains in the run,
    # but only T=1 chains are stored in posterior → 4 chains
    assert idata.posterior.sizes["chain"] == 4


# ---------------------------------------------------------------------------
# End-to-end: InferenceData structure
# ---------------------------------------------------------------------------

def test_ptde_returns_inferencedata_with_expected_structure():
    """
    Given a simple 2-D normal model,
    When ptde_sample runs with minimal settings,
    Then the result is an InferenceData with posterior vars x and y,
      sample_stats contains lp, and dims are (chain, draw).
    """
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_sample(
        model, system, draws=30, tune=20,
        n_temps=2, T_max=2.0, n_chains=4, cores=1, seed=0,
        log_interval=100,
    )

    assert isinstance(idata, xr.DataTree)
    assert hasattr(idata, "posterior")
    post = idata.posterior
    assert "x" in post.data_vars
    assert "y" in post.data_vars
    assert "chain" in post.dims
    assert "draw" in post.dims
    assert post.sizes["draw"] == 30
    assert "lp" in idata.sample_stats.data_vars


# ---------------------------------------------------------------------------
# _active_rungs (rung thinning)
# ---------------------------------------------------------------------------

def test_active_rungs_no_thinning_returns_all_rungs_every_step():
    """
    Given thin_factor=1 (default, disabled),
    When _active_rungs is called at any step,
    Then every rung is active, regardless of thin_start.
    """
    for step in range(5):
        assert _active_rungs(step, n_temps=8, thin_start=4, thin_factor=1) == list(range(8))


def test_active_rungs_cold_rungs_always_active():
    """
    Given thinning enabled (factor=2, start=4),
    When _active_rungs is called at any step,
    Then rungs below thin_start (0-3) are always in the result.
    """
    for step in range(6):
        active = _active_rungs(step, n_temps=8, thin_start=4, thin_factor=2)
        assert set(range(4)).issubset(active)


def test_active_rungs_hot_rungs_thinned_on_alternate_steps():
    """
    Given thinning enabled (factor=2, start=4, n_temps=8),
    When _active_rungs is called across steps 0 and 1,
    Then hot rungs (4-7) are active on step 0 (all 8 rungs) and absent on
      step 1 (only rungs 0-3).
    """
    assert _active_rungs(0, n_temps=8, thin_start=4, thin_factor=2) == list(range(8))
    assert _active_rungs(1, n_temps=8, thin_start=4, thin_factor=2) == [0, 1, 2, 3]
    assert _active_rungs(2, n_temps=8, thin_start=4, thin_factor=2) == list(range(8))


def test_ptde_rung_thinning_runs_end_to_end():
    """
    Given a simple 2-temp model with rung_thin_factor=2,
    When ptde_sample runs,
    Then it completes and returns the expected InferenceData structure
      (rung thinning must not break basic sampling).
    """
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_sample(
        model, system, draws=30, tune=20,
        n_temps=4, T_max=8.0, n_chains=4, cores=1, seed=1,
        log_interval=100, rung_thin_factor=2, rung_thin_start=2,
    )
    assert idata.posterior.sizes["draw"] == 30
    assert idata.posterior.sizes["chain"] == 4


def test_ptde_collect_rung_timing_runs_end_to_end(caplog):
    """
    Given collect_rung_timing=True on a multi-temp model,
    When ptde_sample runs,
    Then it completes normally and logs a per-rung timing summary line
      for every rung.
    """
    model = _simple_model()
    system = _MinimalSystem()
    with caplog.at_level("INFO", logger="exozippy.samplers.ptde"):
        idata = ptde_sample(
            model, system, draws=20, tune=20,
            n_temps=3, T_max=8.0, n_chains=4, cores=1, seed=3,
            log_interval=100, collect_rung_timing=True,
        )
    assert idata.posterior.sizes["draw"] == 20
    messages = "\n".join(r.message for r in caplog.records)
    assert "PTDE per-rung logp timing" in messages
    for k in range(3):
        assert f"rung {k}" in messages


def test_ptde_posterior_mean_near_true_values():
    """
    Given a 2-D normal model with known mean (x=0, y=3),
    When ptde_sample runs with enough draws,
    Then the posterior mean of each variable is within 1 sigma of the truth.
    """
    model = _simple_model()
    system = _MinimalSystem()
    idata = ptde_sample(
        model, system, draws=200, tune=100,
        n_temps=2, T_max=4.0, n_chains=6, cores=1, seed=7,
        log_interval=500,
    )
    x_mean = float(idata.posterior["x"].values.mean())
    y_mean = float(idata.posterior["y"].values.mean())
    assert abs(x_mean) < 1.0, f"x posterior mean {x_mean:.2f} too far from 0"
    assert abs(y_mean - 3.0) < 0.5, f"y posterior mean {y_mean:.2f} too far from 3"
