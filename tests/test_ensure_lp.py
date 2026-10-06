"""One "make sure lp is in sample_stats" helper, not two (review 4.3.1).

The save path and the trace-plot path each carried their own copy of the
block, and they had drifted: the save path checked
``hasattr(idata, "sample_stats")`` and then assigned into
``idata.sample_stats`` regardless -- which would raise on a trace with no
such group -- while the later plotting copy had grown an ``add_groups``
guard for exactly that case.  ``_ensure_lp`` is the merge, and it turned up
that BOTH were broken for that trace: ``add_groups`` is arviz 0.x API that
no supported arviz has (the floor is 1.1.0, where ``InferenceData`` IS an
``xarray.DataTree``).
"""

import inspect

import numpy as np
import pytest

az = pytest.importorskip("arviz")

from exozippy.run import _ensure_lp


def _posterior_only():
    """A trace with a posterior and NO sample_stats group at all."""
    return az.from_dict({"posterior": {"x": np.zeros((2, 5))}})


def _with_lp():
    return az.from_dict(
        {
            "posterior": {"x": np.zeros((2, 5))},
            "sample_stats": {"lp": np.arange(10.0).reshape(2, 5)},
        }
    )


class _FakeModel:
    """Stands in for a PyMC model; only _compute_lp_from_model reads it."""


def test_an_existing_lp_is_reported_and_left_alone():
    """
    Given a trace that already carries lp,
    When _ensure_lp runs,
    Then it reports True and does not touch the values.
    """
    idata = _with_lp()
    before = idata.sample_stats["lp"].values.copy()

    assert _ensure_lp(idata, model=None) is True
    np.testing.assert_array_equal(idata.sample_stats["lp"].values, before)


def test_no_lp_and_no_model_reports_false():
    """
    Given a trace with no lp and no model to compute one from,
    When _ensure_lp runs,
    Then it reports False rather than raising.

    The plotting path can be handed a trace with no model; it simply gets no
    lp page.
    """
    assert _ensure_lp(_posterior_only(), model=None) is False


def test_lp_is_computed_and_inserted_when_a_model_is_available(monkeypatch):
    """
    Given a trace with no lp but a model to compute one from,
    When _ensure_lp runs,
    Then lp lands in sample_stats with the posterior's own chain/draw coords.
    """
    import exozippy.run as run_module

    lp = np.arange(10.0).reshape(2, 5)
    monkeypatch.setattr(
        run_module,
        "_compute_lp_from_model",
        lambda model, idata, cores=None, trace_path=None: lp,
    )
    idata = _with_lp()
    del idata.sample_stats["lp"]

    assert _ensure_lp(idata, model=_FakeModel()) is True
    np.testing.assert_array_equal(idata.sample_stats["lp"].values, lp)
    assert idata.sample_stats["lp"].dims == ("chain", "draw")


def test_a_trace_with_no_sample_stats_group_gets_one(monkeypatch):
    """
    Given a trace carrying NO sample_stats group,
    When _ensure_lp computes an lp,
    Then the group is created rather than the assignment raising.

    This is the drift the merge resolves, and BOTH copies were broken here:
    the save path would have raised on the assignment, and the plotting
    path's guard called `idata.add_groups(...)`, arviz 0.x API that no
    supported arviz has (the floor is 1.1.0, where InferenceData IS an
    xarray.DataTree).
    """
    import exozippy.run as run_module

    lp = np.full((2, 5), -3.0)
    monkeypatch.setattr(
        run_module,
        "_compute_lp_from_model",
        lambda model, idata, cores=None, trace_path=None: lp,
    )
    idata = _posterior_only()
    assert getattr(idata, "sample_stats", None) is None

    assert _ensure_lp(idata, model=_FakeModel()) is True
    assert "sample_stats" in [g.lstrip("/") for g in idata.groups]
    np.testing.assert_array_equal(idata.sample_stats["lp"].values, lp)


class _FakeRV:
    def __init__(self, name):
        self.name = name


class _ModelWithFreeRVs:
    """Just enough of a PyMC model for _compute_lp_from_model's mapping."""

    def __init__(self, names, values=None):
        self.free_RVs = [_FakeRV(n) for n in names]
        self.rvs_to_values = (
            {rv: _FakeRV(rv.name + "_value") for rv in self.free_RVs}
            if values is None
            else values
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def compile_logp(self, jacobian=False):
        def _boom(point):
            raise FloatingPointError("logp evaluation exploded")

        return _boom


def test_a_failed_lp_computation_raises_naming_the_trace(monkeypatch):
    """
    Given a model whose lp evaluation fails,
    When _ensure_lp runs with a model,
    Then it RAISES RuntimeError naming the trace and the cause, and writes
      nothing (review 2.3.21).

    It used to report False and the save path wrote the trace without lp,
    so mkparam seeded the restart file from the last draw of chain 0 behind
    one warning line.  The model and trace come from one run, so a failure
    is a bug; the computation now runs after the trace is saved
    (run._finish_saved_trace), where a raise loses nothing.
    """
    from exozippy import run as run_module

    monkeypatch.setattr(run_module, "default_cores", lambda: 1)
    idata = _posterior_only()
    model = _ModelWithFreeRVs(["x"])

    # serial pool: the fork context is real, so stub it to run in-process
    class _InlinePool:
        def __init__(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, args):
            return [fn(a) for a in args]

    class _Ctx:
        def Pool(self, n):  # noqa: N802 - mirrors mp API
            return _InlinePool(n)

    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_context", lambda _m: _Ctx())

    with pytest.raises(RuntimeError) as err:
        _ensure_lp(idata, model=model, trace_path="fit_trace.nc")

    assert "fit_trace.nc" in str(err.value)
    assert "logp evaluation exploded" in str(err.value)
    ss = getattr(idata, "sample_stats", None)
    assert ss is None or "lp" not in ss.data_vars


def test_a_trace_holding_none_of_the_free_rvs_raises():
    """
    Given a trace whose posterior holds none of the model's free RVs,
    When lp is computed,
    Then it raises naming the missing RVs instead of returning None.
    """
    from exozippy import run as run_module

    with pytest.raises(RuntimeError, match="star.logmass_raw"):
        run_module._compute_lp_from_model(
            _ModelWithFreeRVs(["star.logmass_raw"]),
            _posterior_only(),
            trace_path="t.nc",
        )


def test_a_free_rv_with_no_value_variable_raises():
    """
    Given a model whose free RV has no value variable,
    When lp is computed,
    Then it raises naming the RV rather than skipping it (the old
      `vv is None: continue`).
    """
    from exozippy import run as run_module

    model = _ModelWithFreeRVs(["x"], values={})
    with pytest.raises(RuntimeError, match="'x' has no value variable"):
        run_module._compute_lp_from_model(
            model, _posterior_only(), trace_path="t.nc"
        )


def test_compute_lp_honours_the_core_grant(monkeypatch):
    """review 2.3.9: the wrap-up pool must not size itself from the NODE.

    Reading mp.cpu_count() here forked one worker per chain up to the whole
    box -- 78 on a 128-CPU node -- and it forks AFTER the caller has the
    trace resident, so each worker inherits a multi-GB parent.  That
    destroyed the wrap-up of three completed multi-day runs at 771, 613 and
    705 GB.  Pin the grant so it cannot regress.
    """
    import multiprocessing as mp

    from exozippy import run as run_module

    seen = {}

    class _FakeCtx:
        def Pool(self, n):  # noqa: N802 - mirrors mp API
            seen["n_workers"] = n
            raise RuntimeError("stop here: the worker count is the assertion")

    monkeypatch.setattr(mp, "get_context", lambda _m: _FakeCtx())
    monkeypatch.setattr(run_module, "default_cores", lambda: 999)

    # CODE, not prose: the fix's own comment names mp.cpu_count() as the
    # thing it replaced, so a naive substring test over the raw source
    # fails on the explanation rather than on a regression.
    src = inspect.getsource(run_module._compute_lp_from_model)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "mp.cpu_count()" not in code, (
        "the wrap-up pool is sizing itself from the NODE again (2.3.9)"
    )
    assert "default_cores() if cores is None" in code
