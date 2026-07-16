"""Tests for the PTDE (Parallel Tempering + Differential Evolution) sampler."""
import multiprocessing as mp
import threading
import time

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from scipy.stats import kstest

from exozippy.components.parameter import Parameter
from exozippy.system import System
from exozippy.samplers.ptde import (
    _active_rungs,
    _make_starts,
    _geometric_ladder,
    _probe_scales,
    _probe_step_1d,
    _PROBE_FLAT_SCALE,
    _shutdown_pool,
    _worker_init,
    ptde_sample,
)


def _hang_forever(_):
    """Stands in for a logp call that enters an unbreakable loop for some
    pathological proposal (module-level so fork workers inherit it)."""
    while True:
        time.sleep(0.01)


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


def test_ptde_warns_once_when_t1_lp_exceeds_plausibility_ceiling(caplog):
    """
    Given a plausibility ceiling set far below any lp this model can
    legitimately reach,
    When ptde_sample runs and a T=1 chain's accepted lp exceeds it,
    Then a single loud warning is logged (not one per step) naming the
    offending chain and lp -- this is the early-detection guard for the
    DC2018_128-style runaway (examples/DC2018_128), where a buggy unbounded
    logp term let a T=1 chain's lp climb to 1e15..1e39 unnoticed until
    post-hoc mode identification discarded the run.
    """
    model = _simple_model()
    system = _MinimalSystem()
    with caplog.at_level("WARNING", logger="exozippy.samplers.ptde"):
        ptde_sample(
            model, system, draws=20, tune=20,
            n_temps=2, T_max=2.0, n_chains=4, cores=1, seed=1,
            log_interval=100, lp_plausibility_ceiling=0.1,
        )
    warnings = [r.message for r in caplog.records
                if "plausibility ceiling" in r.message]
    assert len(warnings) == 1, (
        f"expected exactly one plausibility-ceiling warning, got "
        f"{len(warnings)}: {warnings}"
    )
    assert "T=1 chain" in warnings[0]


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


# ---------------------------------------------------------------------------
# _probe_scales: step search
# ---------------------------------------------------------------------------

def test_probe_scale_recovers_gaussian_sigma():
    """
    Given an exactly Gaussian logp with known per-element sigma,
    When _probe_scales probes it,
    Then each scale is that element's sigma -- for a Gaussian the step costing
      0.5 nats IS 1 sigma, so the correct answer is known in closed form.
    """
    # ARRANGE
    sigmas = np.array([0.25, 1.0, 4.0])
    start = {"x": np.zeros(3)}

    def logp_fn(p):
        return float(-0.5 * np.sum((p["x"] / sigmas) ** 2))

    # ACT
    _, scales = _probe_scales(start, logp_fn)

    # ASSERT
    np.testing.assert_allclose(scales["x"], sigmas, rtol=0.05)


def test_probe_step_1d_lands_on_the_target_drop():
    """
    Given a logp probed from a start OFFSET from its mode,
    When _probe_step_1d searches one direction,
    Then the returned step is where logp has fallen ~0.5 nats below the START
      in that direction -- including when the direction starts out UPHILL and
      the search must grow through the turnover to find the far side.
    """
    # ARRANGE: mode at 3.0, start at 0.0.  +x climbs to the mode then falls;
    # -x falls immediately.
    def logp(x):
        return -0.5 * (x - 3.0) ** 2

    map_lp = logp(0.0)

    def eval_delta(step):
        return map_lp - logp(step)

    # 0.5 nats below logp(0) is (x-3)^2 = 10, i.e. x = 3 +/- sqrt(10)
    for sign, expected in ((1.0, 3.0 + np.sqrt(10.0)),
                           (-1.0, np.sqrt(10.0) - 3.0)):
        # ACT
        step = _probe_step_1d(eval_delta, sign)

        # ASSERT
        assert step == pytest.approx(expected, rel=0.05)
        assert eval_delta(sign * step) == pytest.approx(0.5, abs=0.05)


def test_probe_scale_takes_nearer_contour_when_start_is_off_mode():
    """
    Given a Gaussian logp probed from a start offset from its mode,
    When _probe_scales probes it,
    Then the scale is the NEARER 0.5-nat contour, not the average of the two.

    EXOFASTv2 averages the two directions because it probes from an AMOEBA best
    fit, where they are near-symmetric.  Off the mode they are not: the uphill
    direction only turns over on the far side, and averaging it in inflates the
    scale enough to jitter chains past the logit transform's saturation walls,
    starting them pinned at the bounds.
    """
    # ARRANGE: mode at 3.0, unit width, start 3 sigma away at 0.0
    start = {"x": np.zeros(1)}

    def logp_fn(p):
        return float(-0.5 * (p["x"][0] - 3.0) ** 2)

    # ACT
    _, scales = _probe_scales(start, logp_fn)
    s = float(scales["x"][0])

    # ASSERT: logp(0) - logp(x) = 0.5 at (x-3)^2 = 10, so the contours are at
    # x = 3 - sqrt(10) (near) and x = 3 + sqrt(10) (far).  Take the near one.
    assert s == pytest.approx(np.sqrt(10.0) - 3.0, rel=0.05)

    # ASSERT: still far looser than the old quadratic extrapolation, which
    # collapsed to sqrt(0.5*|dp|/g) at the smallest rung of its probe ladder.
    old = 0.003 * np.sqrt(0.5 / (3.0 * 0.003))
    assert old < 0.03
    assert s > 5 * old


def test_probe_scale_linear_logp_is_not_extrapolated_quadratically():
    """
    Given a logp that is exactly LINEAR in one element,
    When _probe_scales probes it,
    Then the scale is the true 0.5-nat step (target/gradient), not the
      quadratic extrapolation from the smallest probe.
    """
    # ARRANGE: logp = -g*x  =>  drop of 0.5 nats at exactly x = 0.5/g
    g = 0.32
    start = {"x": np.zeros(1)}

    def logp_fn(p):
        return float(-g * p["x"][0])

    # ACT
    _, scales = _probe_scales(start, logp_fn)

    # ASSERT: true step is 0.5/0.32 = 1.5625.  Only one direction (+x) falls;
    # -x rises forever and is reported flat, so the scale is that one direction.
    assert scales["x"][0] == pytest.approx(0.5 / g, rel=0.05)


def test_probe_scale_flat_direction_falls_back():
    """
    Given an element logp does not depend on at all,
    When _probe_scales probes it,
    Then it falls back to _PROBE_FLAT_SCALE instead of hanging or returning 0.
    """
    # ARRANGE
    start = {"x": np.zeros(2)}

    def logp_fn(p):
        return float(-0.5 * p["x"][0] ** 2)      # x[1] unused

    # ACT
    _, scales = _probe_scales(start, logp_fn)

    # ASSERT
    assert scales["x"][0] == pytest.approx(1.0, rel=0.05)   # constrained
    assert scales["x"][1] == _PROBE_FLAT_SCALE              # flat


def test_probe_scale_respects_hard_prior_wall():
    """
    Given a hard prior wall (logp = -inf) close to the start on one side,
    When _probe_scales probes it,
    Then the returned scale stays inside the wall rather than proposing across
      it, which would make every jittered start non-finite.
    """
    # ARRANGE: wall at x = -0.5, gentle Gaussian otherwise
    start = {"x": np.zeros(1)}

    def logp_fn(p):
        x = p["x"][0]
        if x <= -0.5:
            return -np.inf
        return float(-0.5 * (x / 3.0) ** 2)

    # ACT
    _, scales = _probe_scales(start, logp_fn)

    # ASSERT: the wall-side search cannot exceed 0.5; averaged with the open
    # side the scale must stay finite and positive.
    s = float(scales["x"][0])
    assert 0.0 < s < 3.0
    assert np.isfinite(logp_fn({"x": np.array([-0.5 + 1e-9])}))


# ---------------------------------------------------------------------------
# Chain seeding: physical-space truncated jitter
# ---------------------------------------------------------------------------

class _ParamSystem:
    """Exposes the real System.jitter_raw_start against a bare Parameter."""

    def __init__(self, par):
        self._par = par

    def get_all_parameters(self):
        return [self._par]

    jitter_raw_start = System.jitter_raw_start


def _flat_param():
    """A parameter whose logp is flat in PHYSICAL space out to both bounds."""
    with pm.Model() as model:
        par = Parameter(label="p", initval=0.5, init_scale=0.05,
                        lower=0.0, upper=1.0)
        par.build_pymc()
    return par, model


def _narrow_param():
    """A parameter tightly constrained relative to its prior volume."""
    with pm.Model() as model:
        par = Parameter(label="p", initval=0.5, init_scale=0.05,
                        mu=0.5, sigma=0.02, lower=0.0, upper=1.0)
        par.build_pymc()
    return par, model


def _seed_physical(par, model, n, factor=3.0, seed=0):
    logp_fn = model.compile_logp()
    center = {"p_raw": np.zeros(1)}
    _, scales = _probe_scales(center, logp_fn)
    sysobj = _ParamSystem(par)
    rng = np.random.default_rng(seed)
    raws = [sysobj.jitter_raw_start(center, scales, factor, rng)["p_raw"][0]
            for _ in range(n)]
    return np.array([par.phys_from_raw(np.array([r]))[0] for r in raws])


def test_phys_from_raw_roundtrips_through_raw_from_initval():
    """
    Given a raw coordinate,
    When it is mapped to physical and back,
    Then the original raw value is recovered -- phys_from_raw must be the exact
      inverse of raw_from_initval, since seeding maps raw -> physical, draws,
      and maps the draw back.
    """
    # ARRANGE
    par, _ = _flat_param()

    for raw in (-2.5, -0.3, 0.0, 0.7, 3.1):
        # ACT
        phys = par.phys_from_raw(np.array([raw]))
        back = par.raw_from_initval(phys)

        # ASSERT
        assert back[0] == pytest.approx(raw, abs=1e-9)


def test_seeding_is_uniform_for_a_flat_parameter():
    """
    Given a parameter whose logp is flat out to its bounds,
    When chains are seeded,
    Then the physical starts are ~uniform rather than piled at the bounds.

    Jittering in RAW space saturates the logit transform: `lower +
    span*sigmoid(lq)` folds both tails onto the bounds once the jitter's width
    in lq approaches ~3, which for a flat parameter it always does (factor*1.41,
    independent of init_scale).  That started 31.5% of chains within 1% of a
    bound where uniform wants 2.0%.
    """
    # ARRANGE / ACT
    v = _seed_physical(*_flat_param(), n=4000)

    # ASSERT: no pileup -- at or under uniform's own 2% / 0.2%
    assert ((v < 0.01) | (v > 0.99)).mean() < 0.03
    assert ((v < 0.001) | (v > 0.999)).mean() < 0.005
    # ASSERT: actually uniform, not merely un-piled (uniform sd = 1/sqrt(12))
    assert v.std() == pytest.approx(1.0 / np.sqrt(12.0), rel=0.10)
    assert kstest(v, "uniform").statistic < 0.05


def test_seeding_preserves_overdispersion_when_curvature_exists():
    """
    Given a parameter tightly constrained relative to its prior volume,
    When chains are seeded,
    Then they keep EXOFASTv2's factor-x over-dispersion.

    The flat-case fix must not cost over-dispersion where the parameter does
    not feel its bounds: truncation simply never bites there.
    """
    # ARRANGE / ACT: sigma=0.02 on a span of 1, factor=3 -> target sd 0.06
    v = _seed_physical(*_narrow_param(), n=4000, factor=3.0)

    # ASSERT
    assert v.std() == pytest.approx(3.0 * 0.02, rel=0.15)
    assert ((v < 0.01) | (v > 0.99)).mean() == 0.0


def test_make_starts_falls_back_when_system_has_no_jitter():
    """
    Given a system stub that does not implement jitter_raw_start,
    When _make_starts runs,
    Then it still produces finite starts via the historical raw-space jitter.
    """
    # ARRANGE
    model = _simple_model()
    logp_fn = model.compile_logp()
    raw_start = model.initial_point()
    rng = np.random.default_rng(0)

    # ACT: system=None is the no-jitter path
    starts, _ = _make_starts(6, raw_start, logp_fn, rng, system=None)

    # ASSERT
    assert len(starts) == 6
    assert all(np.isfinite(float(logp_fn(s))) for s in starts)


def test_shutdown_pool_kills_workers_that_ignore_sigterm():
    """
    Given a fork Pool whose workers ignore SIGTERM (as _worker_init sets up)
      and a worker wedged in an unbreakable logp loop,
    When the pool is recycled via _shutdown_pool,
    Then it returns promptly by escalating to SIGKILL rather than blocking
      forever in join() -- the "recycling worker pool" hang.

    Regression: pool.terminate() reaps workers by sending SIGTERM, which these
    workers ignore, so the plain terminate()/join() the recycle path used hung
    on the very worker a timeout was trying to clear.
    """
    # ARRANGE: pool with a genuinely stuck, SIGTERM-ignoring worker
    pool = mp.get_context("fork").Pool(2, initializer=_worker_init)
    pool.apply_async(_hang_forever, (1,))
    time.sleep(0.5)  # let the worker enter the loop

    # ACT: run in a watchdog thread so a regression FAILS instead of hanging
    #      the whole suite; grace is 1.0s so allow generous headroom.
    done = threading.Event()

    def _run():
        _shutdown_pool(pool)
        done.set()

    t = threading.Thread(target=_run, daemon=True)
    t0 = time.time()
    t.start()
    finished = done.wait(timeout=15.0)
    elapsed = time.time() - t0

    # ASSERT
    assert finished, "_shutdown_pool did not return -- the recycle hang regressed"
    assert elapsed < 10.0, f"_shutdown_pool took {elapsed:.1f}s, expected ~1s"

    # a fresh pool is usable after the recycle
    pool2 = mp.get_context("fork").Pool(2, initializer=_worker_init)
    try:
        assert pool2.apply_async(abs, (-3,)).get(timeout=10) == 3
    finally:
        _shutdown_pool(pool2)
