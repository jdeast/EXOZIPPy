"""Every worker pool works whatever the platform's DEFAULT start method is.

Review 2.6.42.  A pool built with ``mp.Pool(...)`` takes the platform's
default start method: fork on Linux up to Python 3.13, but spawn on macOS
and forkserver on Linux from Python 3.14.  A spawn / forkserver worker
re-imports the module and inherits NONE of the parent's runtime globals --
so a pool whose workers read a compiled logp out of a module global (set by
the parent just before the pool starts) finds ``None`` there, every
evaluation raises, the caller scores it -inf, and the stage silently does
nothing.  That shipped for the DE seed polish (fixed in #536) and was still
true of the nested sampler.

On a Linux CI runner below 3.14 the default is fork, so a test that does not
FORCE another start method passes without exercising the bug.  Every test
here therefore runs with the process default switched to forkserver (the
3.14 Linux default; spawn behaves the same for this purpose), and asserts
FINITE results -- an exception scored -inf or floored must fail the test.

Each pool must either use an explicit fork context (``_common.create_pool``)
or hand its workers their state explicitly (the grid search's
``initializer`` / ``initargs``).
"""

import multiprocessing as mp

import numpy as np
import pymc as pm
import pytest

needs_cores = pytest.mark.skipif(
    mp.cpu_count() < 3, reason="a real pool needs >= 2 worker cores"
)
needs_forkserver = pytest.mark.skipif(
    "forkserver" not in mp.get_all_start_methods()
    or "fork" not in mp.get_all_start_methods(),
    reason="needs both the forkserver and fork start methods",
)


@pytest.fixture
def forkserver_default():
    """Make forkserver the process-wide DEFAULT start method for one test,
    then put back whatever was there (None = not yet chosen)."""
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("forkserver", force=True)
    assert mp.get_start_method() == "forkserver"
    try:
        yield
    finally:
        mp.set_start_method(previous, force=True)


@needs_cores
@needs_forkserver
def test_create_pool_workers_see_the_compiled_logp_under_forkserver(
    forkserver_default,
):
    """
    Given forkserver as the process default and a compiled logp handed to
      set_worker_globals,
    When create_pool builds a 2-worker pool and _map_logp evaluates 4 points,
    Then every value is finite and equals the serial value -- the workers
      inherited the logp (create_pool forks explicitly), so none raised and
      was scored -inf.
    """
    # ARRANGE
    from exozippy.samplers import _common

    with pm.Model() as model:
        pm.Normal("x", 0.0, 1.0)
    logp = model.compile_logp()
    points = [{"x": np.array(v)} for v in (-1.0, 0.0, 0.5, 2.0)]
    _common.set_worker_globals(logp)
    pool, actual = _common.create_pool(2, len(points), "test", _NullLog())

    # ACT
    try:
        lps = _common._map_logp(pool, points)
    finally:
        pool.close()
        pool.join()

    # ASSERT
    assert actual == 2
    assert np.isfinite(lps).all(), lps
    np.testing.assert_allclose(lps, [float(logp(p)) for p in points])


@needs_cores
@needs_forkserver
def test_nested_pool_evaluates_the_likelihood_under_forkserver(
    forkserver_default,
):
    """
    Given forkserver as the process default and a 1-D Gaussian model,
    When nested_sample runs on 2 worker cores,
    Then no likelihood call raised or returned NaN (the shared counters are
      0) and logZ is finite and near its analytic value log(1/10).  Under
      the old mp.Pool the workers' _NB was empty and every call died on
      _NB["bridge"].
    """
    pytest.importorskip("dynesty")
    # ARRANGE
    from test_nested_sampler import _make_model

    from exozippy.samplers.nested import nested_sample

    model = _make_model(
        [0.0],
        [10.0],
        lambda x: (
            -0.5 * ((x[0] - 3.0) / 0.5) ** 2 - np.log(0.5 * np.sqrt(2 * np.pi))
        ),
    )

    # ACT
    idata = nested_sample(model, None, nlive=100, dlogz=0.5, cores=2, seed=3)

    # ASSERT
    attrs = idata.posterior.attrs
    assert attrs["nested_n_logp_exceptions"] == 0
    assert np.isfinite(attrs["nested_logz"])
    assert attrs["nested_logz"] == pytest.approx(np.log(0.1), abs=0.5)
    assert np.isfinite(np.asarray(idata.posterior["x"])).all()


@pytest.mark.slow
@needs_cores
@needs_forkserver
def test_grid_search_pool_evaluates_every_node_under_forkserver(
    forkserver_default,
):
    """
    Given forkserver as the process default,
    When grid_search runs a 2-node grid on its own 2-worker pool,
    Then both nodes finish with a finite chi2.  `_node` catches EVERYTHING
      and scores it inf, so a worker that never received its curves would
      come back as an all-inf grid rather than an error; the grid hands the
      workers their state through initializer/initargs, which every start
      method delivers.
    """
    # ARRANGE
    from test_gridsearch import _blind_start, _injected_curves

    from exozippy.components.mulensing.gridsearch import GridSpec, grid_search

    curves = _injected_curves()
    pspl = _blind_start(curves)
    spec = GridSpec(
        log_s_lo=-0.1,
        log_s_hi=0.1,
        n_s=2,
        log_q_lo=-3.0,
        log_q_hi=-3.0,
        n_q=1,
        n_alpha_starts=1,
        trim_t_E=10.0,
    )

    # ACT
    res = grid_search(curves, pspl, spec, ncpu=2, progress_every=0)

    # ASSERT
    assert res.complete and res.n_done == res.n_tasks == 2
    assert np.isfinite(res.chi2).all(), res.chi2
    assert np.isfinite(res.alpha).all()


class _NullLog:
    def warning(self, *a, **k):
        pass
