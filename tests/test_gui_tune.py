"""Tests for the GUI Tune tab endpoints (G10).

Two things are pinned here, per the G10 verification list:

  * the ``/api/tune/eval`` round trip, exercised through the full HTTP surface
    with a STUBBED evaluator worker so no real pytensor compile is needed in
    the fast test (the worker is monkeypatched to a canned in-process fake);
  * a params.yaml written after a slider-style edit (an undoable RANK_USER
    ``set_param_field`` initval override through the G8 document) round-trips
    through ``yaml.safe_load`` and still ``prepare()``s -- a smoke test that
    the edit the Tune tab emits produces a runnable configuration.

Tests follow AAA with Given/When/Then docstrings.  The endpoint tests are
skipped when the optional 'gui' extra is absent.
"""

import os
import queue
import shutil
import time
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "kelt4"


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    return TestClient(create_app())


@pytest.fixture
def rvonly_project(tmp_path):
    """A working copy of the cheap RV-only KELT-4 example."""
    if not EXAMPLE_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    for name in (
        "kelt4_rvonly.yaml",
        "kelt4.params.yaml",
        "KELT-4b.HIRES.rv",
        "KELT-4b.TRES.rv",
    ):
        shutil.copy(EXAMPLE_DIR / name, tmp_path / name)
    return tmp_path


# ---------------------------------------------------------------------------
# Stubbed evaluator worker
# ---------------------------------------------------------------------------

_STUB_PLOT = {
    "id": "rv.unphased",
    "component": {"yaml_key": "rvinstrument", "instance": "HIRES"},
    "title": "RV",
    "xlabel": "BJD",
    "ylabel": "RV (m/s)",
    "traces": [
        {
            "name": "data",
            "role": "data",
            "kind": "scatter",
            "x": [1, 2],
            "y": [3, 4],
        },
        {
            "name": "model",
            "role": "model",
            "kind": "line",
            "x": [1, 2],
            "y": [3.0, 4.0],
        },
    ],
    "param_deps": ["orbit.logP"],
    "meta": {},
}


class _StubWorker:
    """A drop-in for EvaluatorWorker that returns canned data (no compile)."""

    def start(self):
        self.started = True

    def solve(self, config, params, workdir, on_progress=None):
        if on_progress:
            # The real worker forwards the full progress dict (phase +
            # data-only plots); exercise that path here.
            data_only = {
                "id": _STUB_PLOT["id"],
                "traces": [_STUB_PLOT["traces"][0]],
            }
            on_progress({"progress": "compiling", "data_plots": [data_only]})
        return {
            "parameters": {
                "orbit.b.logP": {
                    "value": 0.4801,
                    "unit": "dex(d)",
                    "internal_unit": "dex(d)",
                    "lower": 0.0,
                    "upper": 1.0,
                    "init_scale": 0.01,
                    "sigma": None,
                    "mu": None,
                    "fixed": False,
                    "derived": False,
                    "provenance": {
                        "rank": 100,
                        "label": "user",
                        "relation": None,
                    },
                },
            },
            "seeds": None,
            "plots": [_STUB_PLOT],
            "structural_hash": "deadbeef",
        }

    def set_and_eval(self, path, value):
        # Echo the value back through the model trace so the round trip is
        # observable end to end.
        return {"plots": {"rv.unphased": {"model": [value, value * 2.0]}}}

    def close(self):
        pass


def _wait_live(client, timeout=10.0):
    deadline = time.time() + timeout
    st = {}
    while time.time() < deadline:
        st = client.get("/api/tune/status").json()
        if st["phase"] in ("live", "error"):
            return st
        time.sleep(0.05)
    return st


def test_tune_eval_round_trip_with_stubbed_worker(client, monkeypatch):
    """
    Given a stubbed evaluator worker (no real compile),
    When a solve completes and a slider value is posted to /api/tune/eval,
    Then the endpoint returns the updated model-trace arrays for that value.
    """
    monkeypatch.setattr("exozippy.gui.tune.EvaluatorWorker", _StubWorker)

    resp = client.post(
        "/api/tune/solve",
        json={
            "config": {"star": [{"name": "A"}]},
            "params": {},
            "workdir": None,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["phase"] in ("solving", "compiling", "live")

    st = _wait_live(client)
    assert st["phase"] == "live", st
    assert st["structural_hash"] == "deadbeef"

    result = client.get("/api/tune/result").json()
    assert "orbit.b.logP" in result["parameters"]
    assert result["plots"][0]["id"] == "rv.unphased"

    ev = client.post(
        "/api/tune/eval", json={"path": "orbit.b.logP", "value": 0.6}
    )
    assert ev.status_code == 200
    body = ev.json()
    assert body["plots"]["rv.unphased"]["model"] == [0.6, 1.2]


def test_tune_data_plots_available_from_compiling(client, monkeypatch):
    """
    Given a stubbed worker that ships data-only plots with its progress
    message, When a solve runs, Then /api/tune/plots/data serves them and the
    status advertises has_data_plots (the plots persist after live so a
    late-arriving client can still fetch them).
    """
    # With no session yet the endpoint is empty, not an error.
    assert client.get("/api/tune/plots/data").json() == {"plots": []}
    assert client.get("/api/tune/status").json()["has_data_plots"] is False

    monkeypatch.setattr("exozippy.gui.tune.EvaluatorWorker", _StubWorker)
    client.post(
        "/api/tune/solve",
        json={
            "config": {"star": [{"name": "A"}]},
            "params": {},
            "workdir": None,
        },
    )
    st = _wait_live(client)
    assert st["phase"] == "live", st
    assert st["has_data_plots"] is True

    plots = client.get("/api/tune/plots/data").json()["plots"]
    assert plots[0]["id"] == "rv.unphased"
    assert [t["role"] for t in plots[0]["traces"]] == ["data"]


def test_tune_eval_before_solve_is_409(client):
    """
    Given no solve has run,
    When /api/tune/eval is posted,
    Then the server refuses with 409 (nothing is live to evaluate).
    """
    resp = client.post(
        "/api/tune/eval", json={"path": "orbit.b.logP", "value": 0.6}
    )
    assert resp.status_code == 409
    assert "error" in resp.json()


def test_tune_hash_reports_staleness(client, monkeypatch, rvonly_project):
    """
    Given a live solve and an open document,
    When a bound is changed and /api/tune/hash is queried,
    Then it reports the document as stale relative to the live evaluator.
    """
    monkeypatch.setattr("exozippy.gui.tune.EvaluatorWorker", _StubWorker)
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})

    client.post("/api/tune/solve", json={})  # uses the open doc
    st = _wait_live(client)
    assert st["phase"] == "live", st

    # Fresh (unedited) doc hash is not equal to the stub's fake live hash, so
    # "stale" is driven purely by the comparison; make the change observable by
    # asserting a structural edit keeps it stale.
    client.post(
        "/api/doc/command",
        json={
            "op": "set_param_field",
            "args": {"path": "orbit.b.logP", "field": "lower", "value": 0.2},
        },
    )
    resp = client.get("/api/tune/hash").json()
    assert resp["live_hash"] == "deadbeef"
    assert resp["stale"] is True


@pytest.mark.slow
def test_slider_edit_params_roundtrip_and_prepare(client, rvonly_project):
    """
    Given an open document,
    When a slider-style initval override is committed and saved to params.yaml,
    Then the file round-trips through yaml.safe_load and the config still
        prepare()s successfully.
    """
    from exozippy.system import System

    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})

    # A slider release commits one undoable RANK_USER initval override.
    resp = client.post(
        "/api/doc/command",
        json={
            "op": "set_param_field",
            "args": {"path": "star.A.teff", "field": "initval", "value": 6250},
        },
    )
    assert resp.status_code == 200
    client.post("/api/doc/save")

    params_file = rvonly_project / "kelt4.params.yaml"
    with open(params_file) as fh:
        data = yaml.safe_load(fh)
    # The slider path is the NAME form (what the GUI displays); this params
    # file spells that element by INDEX, and the writer updates that entry in
    # place instead of appending a twin -- one element, one spelling, which is
    # also what the System() smoke test below now enforces.
    assert data["star.0.teff"]["initval"] == 6250
    assert "star.A.teff" not in data

    # Smoke: the written params still drive a successful prepare().
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    prev = os.getcwd()
    os.chdir(rvonly_project)
    try:
        system = System(config, user_params=data)
        system.prepare()
    finally:
        os.chdir(prev)


def test_opening_a_project_resets_the_tune_session(
    client, monkeypatch, tmp_path
):
    """
    Given a solved (live) Tune session for project A,
    When a different project is opened,
    Then the session is dropped -- status is idle again and A's solved
        parameters are no longer served under B.

    Reproduces review 2.11.1: the stale session left project A's solved values
    and plots on screen under B's name, and an edit made against them was
    committed into B's params file.
    """
    monkeypatch.setattr("exozippy.gui.tune.EvaluatorWorker", _StubWorker)
    client.post(
        "/api/tune/solve",
        json={"config": {"star": [{"name": "A"}]}, "params": {}},
    )
    assert _wait_live(client)["phase"] == "live"
    assert (
        "orbit.b.logP" in client.get("/api/tune/result").json()["parameters"]
    )

    other = tmp_path / "projectB"
    other.mkdir()
    assert (
        client.post("/api/project/open", json={"path": str(other)}).status_code
        == 200
    )

    st = client.get("/api/tune/status").json()
    assert st["phase"] == "idle"
    assert st["has_result"] is False
    assert st["structural_hash"] is None
    assert client.get("/api/tune/result").status_code == 409
    assert (
        client.post(
            "/api/tune/eval", json={"path": "orbit.b.logP", "value": 0.6}
        ).status_code
        == 409
    )


def test_opening_a_project_drops_a_foreign_document(client, tmp_path):
    """
    Given a document open from project A,
    When project B is opened,
    Then the stale document is dropped, so an edit or a Solve cannot be
        applied to A's files while B is on screen (a doc inside the newly
        opened project is kept).
    """
    proj_a = tmp_path / "A"
    proj_a.mkdir()
    (proj_a / "a.yaml").write_text(
        "prefix: out/A\nparameter_file: a.params.yaml\n"
    )
    (proj_a / "a.params.yaml").write_text("star.A.teff: {initval: 5800}\n")
    proj_b = tmp_path / "B"
    proj_b.mkdir()

    client.post("/api/doc/open", json={"config_path": str(proj_a / "a.yaml")})
    assert client.get("/api/doc").status_code == 200

    # Re-opening A keeps its document.
    client.post("/api/project/open", json={"path": str(proj_a)})
    assert client.get("/api/doc").status_code == 200

    # Opening B drops it: the next doc/open decides what B edits.
    client.post("/api/project/open", json={"path": str(proj_b)})
    assert client.get("/api/doc").status_code == 404


def test_worker_await_raises_when_the_process_dies(monkeypatch):
    """
    Given a worker whose process is gone with no reply queued,
    When the parent waits for a response,
    Then it raises instead of blocking forever (a closed-mid-solve worker
        would otherwise wedge the server's single-slot tune pool).
    """
    from exozippy.gui import tune

    monkeypatch.setattr(tune, "_AWAIT_POLL_S", 0.01)
    worker = tune.EvaluatorWorker()
    worker._resp_q = queue.Queue()  # a plain queue: same get(timeout=) API
    assert worker.is_alive() is False  # never started -> no process

    with pytest.raises(RuntimeError, match="worker exited"):
        worker.solve({}, {}, None)
