"""The hot-chain suppressed-mode search travels with the trace (run.md).

The search reads only the saved hot draws, so a `recompute_trace: false`
rerun used to redo -- at 41 of 47 wrap-up minutes on the KMT-2021-BLG-1122L
replot -- exactly what the run that wrote the trace had already found.  A
COMPLETED search is now stamped on the trace in place
(trace_meta.HOT_SEARCH_ATTR, trace_meta.stamp_trace_attr) and the reuse
path reads it back (run._hot_search_from_trace) instead of searching.
"""

import dataclasses
import logging

import numpy as np
import pymc as pm
import pytest

from exozippy import run as run_module
from exozippy.components.parameter import Parameter
from exozippy.outputs.ledger import (
    HOT_FAILED,
    HOT_FOUND,
    HOT_NONE_FOUND,
    HOT_NOT_SEARCHED,
    build_seed_ledger,
    hot_search_from_json,
    hot_search_to_json,
    ledger_to_text,
)
from exozippy.trace_meta import (
    HOT_SEARCH_ATTR,
    SEED_LEDGER_ATTR,
    stamp_trace_attr,
)


def _two_basin_model():
    """One bounded parameter with two Gaussian basins (as test_mode_ledger)."""
    p = Parameter(
        label="toy.x", init_scale=1.0, initval=2.0, lower=0.0, upper=10.0
    )
    with pm.Model() as model:
        xv = p.build_pymc()
        lp1 = -0.5 * ((xv - 2.0) / 0.05) ** 2
        lp2 = -0.5 * ((xv - 7.0) / 0.05) ** 2 - 6.0
        pm.Potential("like", pm.math.logsumexp(pm.math.stack([lp1, lp2])))
    return model, p


class _StubSystem:
    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params


def _searched_ledger():
    """A ledger as run_hot_mode_discovery leaves it: one seeded record and
    one the hot-chain search appended (source 'hot-chain')."""
    model, p = _two_basin_model()
    seeds = [
        {"toy.x_raw": np.asarray(p.raw_from_initval(np.array([2.0])))},
        {"toy.x_raw": np.asarray(p.raw_from_initval(np.array([7.0])))},
    ]
    ledger = build_seed_ledger(_StubSystem([p]), model, seeds, [0, 1])
    ledger[1] = dataclasses.replace(ledger[1], source="hot-chain")
    return ledger


_FOUND_STATUS = {
    "state": HOT_FOUND,
    "detail": "",
    "n_new": 1,
    "n_hot_draws": 400,
    "n_viable": 390,
    "n_clusters": 2,
    "margin_nats": 50.0,
    "t_max": 5.0,
}


def _trace_on_disk(tmp_path):
    """A saved trace carrying the pre-sampling seed-ledger stamp, as the
    fresh save path writes it; returns (idata, path)."""
    import arviz as az

    idata = az.from_dict(
        {"posterior": {"toy.x_raw": np.arange(10.0).reshape(2, 5)}}
    )
    idata.attrs[SEED_LEDGER_ATTR] = "null"
    path = tmp_path / "trace.nc"
    idata.to_netcdf(str(path))
    return idata, path


def test_a_completed_search_round_trips_through_the_in_place_stamp(tmp_path):
    """
    Given a trace already on disk and a completed hot-chain search (the
      ledger it extended, with a 'hot-chain' record, and its status),
    When the search is stamped on the trace IN PLACE and the file reloaded,
    Then the stamp decodes to the same ledger (text-identical, arrays equal)
      and the same status, and the trace's draws and earlier stamps are
      untouched -- the file was annotated, not rewritten.
    """
    import arviz as az

    # ARRANGE
    _, path = _trace_on_disk(tmp_path)
    ledger = _searched_ledger()

    # ACT
    stamp_trace_attr(
        path, HOT_SEARCH_ATTR, hot_search_to_json(ledger, _FOUND_STATUS)
    )
    back = az.from_netcdf(str(path))
    back_ledger, back_status = hot_search_from_json(
        back.attrs[HOT_SEARCH_ATTR]
    )

    # ASSERT
    assert back_status == _FOUND_STATUS
    assert ledger_to_text(back_ledger) == ledger_to_text(ledger)
    assert [r.source for r in back_ledger] == ["seed", "hot-chain"]
    for a, b in zip(ledger, back_ledger):
        for k in a.raw_point:
            np.testing.assert_array_equal(a.raw_point[k], b.raw_point[k])
    assert back.attrs[SEED_LEDGER_ATTR] == "null"
    np.testing.assert_array_equal(
        back.posterior["toy.x_raw"].values, np.arange(10.0).reshape(2, 5)
    )


def test_the_reuse_path_takes_the_stamp_and_the_rerun_key_declines_it(caplog):
    """
    Given a reloaded trace whose attrs carry a completed search,
    When the wrap-up asks what the trace already knows,
    Then it gets the ledger and status back and says so at INFO, and with
      `modes: {rerun_hot_search: true}` it gets None (search afresh) and
      says that instead.
    """
    import arviz as az

    # ARRANGE
    ledger = _searched_ledger()
    idata = az.from_dict({"posterior": {"toy.x_raw": np.zeros((2, 5))}})
    idata.attrs[HOT_SEARCH_ATTR] = hot_search_to_json(ledger, _FOUND_STATUS)

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        reused = run_module._hot_search_from_trace(idata, {})
        fresh = run_module._hot_search_from_trace(
            idata, {"rerun_hot_search": True}
        )

    # ASSERT
    assert reused is not None
    got_ledger, got_status = reused
    assert got_status == _FOUND_STATUS
    assert ledger_to_text(got_ledger) == ledger_to_text(ledger)
    assert fresh is None
    assert "reusing the completed search stamped on the trace" in caplog.text
    assert "1 hot-chain record(s)" in caplog.text
    assert "rerun_hot_search: true} asks for a fresh one" in caplog.text


def test_a_trace_without_the_stamp_is_searched():
    """
    Given a trace with no hot-search stamp (a fresh fit, or one written
      before the stamp existed),
    When the wrap-up asks what it already knows,
    Then it gets None: the search runs.
    """
    import arviz as az

    idata = az.from_dict({"posterior": {"toy.x_raw": np.zeros((2, 5))}})

    assert run_module._hot_search_from_trace(idata, {}) is None


@pytest.mark.parametrize("state", [HOT_FAILED, HOT_NOT_SEARCHED])
def test_only_a_completed_search_is_stamped(tmp_path, state):
    """
    Given a search that failed or never ran,
    When the wrap-up would persist it,
    Then nothing is written -- the next rerun must try again (failed) or
      loses nothing by repeating it (not searched) -- while a completed
      none-found search IS stamped, on disk and in memory.
    """
    import arviz as az

    # ARRANGE
    idata, path = _trace_on_disk(tmp_path)

    # ACT
    run_module._persist_hot_search(
        path, idata, None, {"state": state, "detail": "x"}
    )
    skipped = run_module._load_trace(str(path)).attrs.get(HOT_SEARCH_ATTR)
    done = {"state": HOT_NONE_FOUND, "detail": "", "n_new": 0}
    run_module._persist_hot_search(path, idata, None, done)
    stamped = run_module._load_trace(str(path)).attrs[HOT_SEARCH_ATTR]

    # ASSERT
    assert skipped is None
    assert hot_search_from_json(stamped) == (None, done)
    assert idata.attrs[HOT_SEARCH_ATTR] == stamped


def test_a_stamp_recording_a_failed_search_raises():
    """
    Given attrs carrying a hot-search stamp whose state is 'failed' (which
      _persist_hot_search never writes),
    When the reuse path reads it,
    Then it raises naming the attr, rather than reporting a failed search as
      the answer.
    """
    import arviz as az

    idata = az.from_dict({"posterior": {"toy.x_raw": np.zeros((2, 5))}})
    idata.attrs[HOT_SEARCH_ATTR] = hot_search_to_json(
        None, {"state": HOT_FAILED, "detail": "boom"}
    )

    with pytest.raises(RuntimeError, match=HOT_SEARCH_ATTR):
        run_module._hot_search_from_trace(idata, {})


def test_a_foreign_blob_raises_instead_of_decoding_partially():
    """
    Given JSON this code did not write,
    When it is decoded as a hot-search stamp,
    Then it raises naming what is wrong: the shape, or a status without a
      state -- never a partial answer.
    """
    with pytest.raises(ValueError, match="not \\{'ledger'"):
        hot_search_from_json('{"x": 1}')
    with pytest.raises(ValueError, match="lacks \\['state'\\]"):
        hot_search_from_json('{"ledger": null, "status": {"detail": ""}}')
    with pytest.raises(ValueError, match="status must be the dict"):
        hot_search_to_json(None, {"detail": "no state"})


def test_the_stamp_wants_a_string(tmp_path):
    """
    Given a value that is not JSON text,
    When it is stamped on a trace,
    Then stamp_trace_attr refuses: every exozippy stamp is a string, so a
      reload sees it exactly as idata.attrs[name].
    """
    _, path = _trace_on_disk(tmp_path)

    with pytest.raises(TypeError, match="must be a str"):
        stamp_trace_attr(path, HOT_SEARCH_ATTR, {"state": "found"})


def test_the_reuse_loader_releases_the_file_so_the_stamp_can_write(tmp_path):
    """
    Given a trace read the way the reuse path reads it (run._load_trace),
    When the wrap-up stamps a completed search on that same file,
    Then the write succeeds -- a lazy az.from_netcdf keeps the HDF5 handle
      open and HDF5 then refuses the read-write open ("file is already open
      for read-only"), which is how the first version of this stamp failed
      after a 40-minute search -- and the loaded trace still carries its
      draws and its earlier stamps.
    """
    import arviz as az

    # ARRANGE
    _, path = _trace_on_disk(tmp_path)
    lazy = az.from_netcdf(str(path))
    with pytest.raises(OSError, match="already open"):
        stamp_trace_attr(path, HOT_SEARCH_ATTR, "{}")  # the failure mode
    lazy.close()
    loaded = run_module._load_trace(str(path))

    # ACT
    run_module._persist_hot_search(
        path, loaded, None, {"state": HOT_NONE_FOUND, "detail": ""}
    )

    # ASSERT
    assert loaded.attrs[SEED_LEDGER_ATTR] == "null"
    np.testing.assert_array_equal(
        loaded.posterior["toy.x_raw"].values, np.arange(10.0).reshape(2, 5)
    )
    reread = run_module._load_trace(str(path))
    assert hot_search_from_json(reread.attrs[HOT_SEARCH_ATTR])[1]["state"] == (
        HOT_NONE_FOUND
    )
