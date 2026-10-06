"""A NaN logp is a model bug, not a rejection (reviews 2.4.23 and 2.4.22).

-inf is legal: a proposal outside a hard prior wall, rejected for free.  NaN
is the 0*inf / where-trap class (CLAUDE.md).  Every in-house sampler tested
only `np.isfinite`, so a NaN was absorbed into the acceptance rate exactly
like -inf -- no log line, no counter, no trace attr -- and nested sampling
went further, flooring every NaN AND every exception to -1e300 silently, so
a region that NaNs was missing from the posterior, logZ and the mode masses
with nothing in the log.

These tests pin the four places that now tell the two apart: the worker
funnel (_common._eval_logp), the PTDE loops' parent-side counter
(`n_nan_logp`), the start builder (_make_starts) and the polish, and the
nested likelihood (`nested_n_nan`).
"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from conftest import requires_fork
from exozippy.samplers import _common, nested
from exozippy.samplers._common import _make_starts
from exozippy.samplers.ptde import polish_seed_starts, ptde_sample
from exozippy.samplers.ptde_async import ptde_async_sample

NAN_LINE = "logp evaluation returned NaN"


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _nan_box_logp(point):
    """Standard normal in x, NaN on the sub-box 0.5 < x < 1.5 (the shape of
    a where-trap that fires on part of the support)."""
    x = float(np.asarray(point["x"]))
    if 0.5 < x < 1.5:
        return np.nan
    return -0.5 * x * x


def _one_param_model(logp):
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0)
    model.compile_logp = lambda *a, **k: logp
    return model


def _ptde_kwargs(cores):
    return dict(
        draws=60,
        tune=40,
        n_temps=2,
        T_max=4.0,
        n_chains=4,
        cores=cores,
        initvals=[{"x": np.array(-0.2 * j)} for j in range(4)],
        seed=3,
        log_interval=10**6,
        min_ess=None,
        max_rhat=None,
    )


# ---------------------------------------------------------------------------
# The worker funnel
# ---------------------------------------------------------------------------


def test_eval_logp_passes_nan_back_and_reports_it_once(caplog):
    """
    Given a logp that returns NaN,
    When the worker funnel evaluates two proposals,
    Then both come back as NaN (so the parent can count them, not as -inf),
      and exactly ONE ERROR names the first proposal.
    """
    # ARRANGE
    saved = _common._PTDE_LOGP_FN
    _common.set_worker_globals(lambda p: np.nan)
    try:
        # ACT
        with caplog.at_level(logging.ERROR):
            a = _common._eval_logp({"x_raw": np.array([1.25])})
            b = _common._eval_logp({"x_raw": np.array([2.5])})
    finally:
        _common.set_worker_globals(saved)

    # ASSERT
    assert np.isnan(a) and np.isnan(b)
    nan_errors = [r for r in caplog.records if NAN_LINE in r.message]
    assert len(nan_errors) == 1
    assert "1.25" in nan_errors[0].message
    assert _common.count_nan_logps([a, -np.inf, 0.0, b]) == 2


# ---------------------------------------------------------------------------
# The PTDE loops: counted in the parent, stamped on the trace
# ---------------------------------------------------------------------------


def test_sync_ptde_counts_nan_and_logs_one_error_serially(caplog):
    """
    Given a model whose logp is NaN on a sub-box of the support,
    When synchronous PTDE runs serially (the parent IS the worker),
    Then the trace carries a nonzero n_nan_logp, the run summary says
      nan_logp=, and the per-worker NaN report fires exactly once.
    """
    # ARRANGE
    model = _one_param_model(_nan_box_logp)

    # ACT
    with caplog.at_level(logging.INFO):
        idata = ptde_sample(model, _MinimalSystem(), **_ptde_kwargs(cores=1))

    # ASSERT
    n_nan = idata.posterior.attrs["n_nan_logp"]
    assert n_nan > 0
    assert f"nan_logp={n_nan}" in caplog.text
    assert len([r for r in caplog.records if NAN_LINE in r.message]) == 1
    # and no NaN box point was ever accepted
    x = np.ravel(idata.posterior["x"].values)
    assert not np.any((x > 0.5) & (x < 1.5))


@requires_fork
def test_sync_ptde_logs_one_nan_error_per_pool_worker(tmp_path):
    """
    Given the same NaN sub-box model and a forked worker pool,
    When synchronous PTDE runs,
    Then each worker logs AT MOST one NaN report (at least one overall) and
      the parent's count reaches the trace.

    caplog cannot see a forked child's records, so a FileHandler attached
    before the fork collects every process's lines in one file.
    """
    # ARRANGE
    model = _one_param_model(_nan_box_logp)
    cores = 2
    log_path = tmp_path / "workers.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(process)d %(message)s"))
    lg = logging.getLogger("exozippy.samplers._common")
    lg.addHandler(handler)
    old_level = lg.level
    lg.setLevel(logging.ERROR)
    try:
        # ACT
        idata = ptde_sample(
            model, _MinimalSystem(), **_ptde_kwargs(cores=cores)
        )
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)
        handler.close()

    # ASSERT
    pids = [
        line.split()[0]
        for line in log_path.read_text().splitlines()
        if NAN_LINE in line
    ]
    assert 1 <= len(pids) <= cores
    assert len(set(pids)) == len(pids), "a worker reported its NaN twice"
    assert idata.posterior.attrs["n_nan_logp"] > 0


def test_async_ptde_counts_nan(caplog):
    """
    Given the NaN sub-box model,
    When ptde_async runs (serially, so it is deterministic),
    Then its trace carries a nonzero n_nan_logp too.
    """
    model = _one_param_model(_nan_box_logp)
    with caplog.at_level(logging.INFO):
        idata = ptde_async_sample(
            model, _MinimalSystem(), **_ptde_kwargs(cores=1)
        )
    n_nan = idata.posterior.attrs["n_nan_logp"]
    assert n_nan > 0
    assert f"nan_logp={n_nan}" in caplog.text


def test_clean_model_stamps_zero_nan_count():
    """
    Given a model that never returns NaN,
    When synchronous PTDE runs,
    Then n_nan_logp is stamped, and is 0 (present, not absent).
    """
    model = _one_param_model(lambda p: -0.5 * float(np.asarray(p["x"])) ** 2)
    idata = ptde_sample(model, _MinimalSystem(), **_ptde_kwargs(cores=1))
    assert idata.posterior.attrs["n_nan_logp"] == 0


# ---------------------------------------------------------------------------
# The start builder
# ---------------------------------------------------------------------------


def test_make_starts_raises_on_nan_at_the_exact_seed():
    """
    Given a logp that is NaN exactly at the seed,
    When _make_starts evaluates the seed for the exact-start chain,
    Then it raises ValueError naming the seed, instead of jittering away
      from the very point the model cannot evaluate.
    """
    seed = {"x": np.array(1.0)}
    with pytest.raises(ValueError, match=r"NaN at seed 7"):
        _make_starts(
            4,
            [seed],
            _nan_box_logp,
            np.random.default_rng(0),
            seed_indices=[7],
            raw_scales={"x": np.array(1.0)},
        )


def test_make_starts_counts_nan_jitter_separately_from_bound_hits(caplog):
    """
    Given a logp that is NaN on one side of the seed and -inf beyond a wall
      on the other,
    When _make_starts jitters chains around the seed,
    Then the NaN draws are redrawn (the starts are all finite) but reported
      at ERROR with their own count, distinct from the bound hits.
    """

    def logp(p):
        x = float(np.asarray(p["x"]))
        if x > 0.3:
            return np.nan
        if x < -0.3:
            return -np.inf
        return -0.5 * x * x

    with caplog.at_level(logging.DEBUG):
        starts, _ = _make_starts(
            12,
            [{"x": np.array(0.0)}],
            logp,
            np.random.default_rng(1),
            raw_scales={"x": np.array(1.0)},
        )

    assert all(np.isfinite(logp(s)) for s in starts)
    errs = [
        r.message
        for r in caplog.records
        if r.levelno >= logging.ERROR and "NaN logp" in r.message
    ]
    assert len(errs) == 1
    assert "hit -inf" in errs[0]


# ---------------------------------------------------------------------------
# The DE polish
# ---------------------------------------------------------------------------


def test_polish_counts_nan_and_raises_on_nan_seed(caplog):
    """
    Given the NaN sub-box logp,
    When the DE polish runs from a clean seed, then from a seed inside the
      box,
    Then the first reports its NaN count at ERROR, and the second raises
      (a NaN seed would freeze the whole population: nothing beats NaN).
    """
    kw = dict(n_steps=15, pop_size=8, scales={"x": np.ones(())})
    with caplog.at_level(logging.ERROR):
        polish_seed_starts(
            [{"x": np.array(0.0)}],
            _nan_box_logp,
            np.random.default_rng(0),
            **kw,
        )
    assert any(
        "PTDE seed polish" in r.message and "returned NaN" in r.message
        for r in caplog.records
    )
    with pytest.raises(ValueError, match="NaN at seed 0"):
        polish_seed_starts(
            [{"x": np.array(1.0)}],
            _nan_box_logp,
            np.random.default_rng(0),
            **kw,
        )


# ---------------------------------------------------------------------------
# Nested sampling (2.4.22)
# ---------------------------------------------------------------------------


def _nested_model(loglike):
    """Exozippy-style build: x_raw ~ N(0,1), x logit-bounded on [0, 10], the
    logit correction that makes the prior uniform, plus `loglike(x)`."""
    with pm.Model() as model:
        raw = pm.Normal("x_raw", 0.0, 1.0, shape=1)
        q = pm.math.sigmoid(raw)
        x = pm.Deterministic("x", 10.0 * q)
        pm.Potential(
            "logit_correction",
            pt.sum(pt.log(q) + pt.log(1 - q))
            + 0.5 * pt.sum(raw**2)
            + 0.5 * np.log(2 * np.pi),
        )
        pm.Potential("loglike", loglike(x))
    return model


def _install_nested(logp_fn):
    model = _nested_model(lambda x: 0.0 * x[0])
    bridge = nested.UnitCubeBridge(model)
    _common.reset_logp_reports()
    nested._NB.update(
        bridge=bridge, logp_fn=logp_fn, pool=None, counts=nested._new_counts()
    )


def test_nested_loglike_tells_nan_exception_and_bound_apart(caplog):
    """
    Given logps that return NaN, raise, return -inf and return +inf,
    When the nested likelihood scores them,
    Then NaN and the exception are floored but COUNTED and logged at ERROR
      (once each), -inf is floored silently (it is zero density), and +inf
      raises (it is not a density at all).
    """
    u = np.array([0.5])
    try:
        with caplog.at_level(logging.ERROR):
            _install_nested(lambda p: np.nan)
            assert nested._loglike_u(u) == nested._LOGL_FLOOR
            assert nested._loglike_u(u) == nested._LOGL_FLOOR
            assert nested._NB["counts"][nested._N_NAN] == 2

            def boom(p):
                raise RuntimeError("magnification backend failed")

            _install_nested(boom)
            assert nested._loglike_u(u) == nested._LOGL_FLOOR
            assert nested._NB["counts"][nested._N_EXC] == 1

            _install_nested(lambda p: -np.inf)
            assert nested._loglike_u(u) == nested._LOGL_FLOOR
            assert list(nested._NB["counts"]) == [0, 0]

            _install_nested(lambda p: np.inf)
            with pytest.raises(FloatingPointError, match=r"\+inf"):
                nested._loglike_u(u)
    finally:
        nested._NB.clear()

    msgs = [r.message for r in caplog.records]
    assert len([m for m in msgs if NAN_LINE in m]) == 1
    assert len([m for m in msgs if "raised RuntimeError" in m]) == 1


def test_nested_sample_stamps_nonzero_nested_n_nan(caplog):
    """
    Given a model whose likelihood is NaN on a sub-box of [0, 10],
    When nested sampling runs,
    Then the trace carries a nonzero nested_n_nan and the run says so at
      ERROR, rather than silently dropping the box from logZ.
    """
    pytest.importorskip("dynesty")
    model = _nested_model(
        lambda x: pt.switch(
            pt.and_(x[0] > 6.0, x[0] < 8.0),
            np.nan,
            -0.5 * ((x[0] - 3.0) / 0.5) ** 2,
        )
    )
    with caplog.at_level(logging.ERROR):
        idata = nested.nested_sample(
            model, None, nlive=50, dlogz=1.0, cores=1, seed=2
        )
    assert idata.posterior.attrs["nested_n_nan"] > 0
    assert idata.posterior.attrs["nested_n_logp_exceptions"] == 0
    assert any("returned NaN" in r.message for r in caplog.records)
