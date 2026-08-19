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
        run_module, "_compute_lp_from_model", lambda model, idata: lp
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
        run_module, "_compute_lp_from_model", lambda model, idata: lp
    )
    idata = _posterior_only()
    assert getattr(idata, "sample_stats", None) is None

    assert _ensure_lp(idata, model=_FakeModel()) is True
    assert "sample_stats" in [g.lstrip("/") for g in idata.groups]
    np.testing.assert_array_equal(idata.sample_stats["lp"].values, lp)


def test_a_failed_computation_reports_false(monkeypatch):
    """
    Given a model whose lp evaluation fails (_compute_lp_from_model returns
      None, which is how it reports failure),
    When _ensure_lp runs,
    Then it reports False and writes nothing.
    """
    import exozippy.run as run_module

    monkeypatch.setattr(
        run_module, "_compute_lp_from_model", lambda model, idata: None
    )
    idata = _posterior_only()

    assert _ensure_lp(idata, model=_FakeModel()) is False
    ss = getattr(idata, "sample_stats", None)
    assert ss is None or "lp" not in ss.data_vars
