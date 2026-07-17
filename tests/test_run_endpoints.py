"""Tests for the G11 run-control endpoints on the GUI app.

Two layers:

* Fast endpoint tests drive POST /api/run, GET /api/run/status,
  POST /api/run/stop, GET /api/run/image and POST /api/utilities/run with a
  *fake* RunHandle (no subprocess), so the HTTP wiring, guard rails, status
  payload, and path safety are covered in milliseconds.
* One slow end-to-end test launches a real, tiny PTDE fit through the endpoints
  (reusing the kelt4 RV-only example the way tests/test_runner.py does) and
  confirms the start -> running -> stop -> clean-exit round trip.

The endpoint tests are skipped when the optional 'gui' extra is absent.
"""

import shutil
import time
from pathlib import Path

import pytest
import yaml

from exozippy.gui import TERMINAL_PHASES

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "kelt4"

REACH_SAMPLING_TIMEOUT = 360.0
POLL_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    return TestClient(create_app())


class _FakeHandle:
    """A stand-in RunHandle whose phase the test controls, no subprocess."""

    def __init__(self, cwd, prefix="out/RUN", config_path="cfg.yaml"):
        self.cwd = str(cwd)
        self.prefix = prefix
        self.config_path = config_path
        self.snapshot_dir = str(Path(cwd) / (prefix + "_gui_snapshot"))
        self._phase = "starting"
        self._alive = True
        self.stop_calls = []

    def is_alive(self):
        return self._alive

    def status(self):
        return {
            "phase": self._phase,
            "state": {"n_draws": 128, "max_rhat": 1.02, "min_ess": 210.0},
            "pid": 4321,
            "alive": self._alive,
        }

    def stop(self, force=False, **_kw):
        self.stop_calls.append(force)
        self._phase = "stopped"
        self._alive = False
        return 0


def _poll_until(predicate, timeout, interval=POLL_INTERVAL):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


def _write_ptde_config(work_dir, out_prefix, *, draws=100_000):
    """Write a fast, never-auto-converging PTDE config; return its filename."""
    with open(EXAMPLE_DIR / "kelt4_rvonly.yaml") as fh:
        config = yaml.safe_load(fh)
    config["prefix"] = str(out_prefix)
    config["sampler"] = {
        "method": "ptde",
        "tune": 30,
        "draws": draws,
        "n_temps": 2,
        "T_max": 5.0,
        "n_chains": 4,
        "cores": 1,
        "check_curvatures": False,
        "recompute_trace": True,
        "min_ess": 100_000_000,
        "max_rhat": 1.0000001,
    }
    config_name = "run_ptde.yaml"
    with open(work_dir / config_name, "w") as fh:
        yaml.safe_dump(config, fh)
    return config_name


@pytest.fixture
def kelt4_workdir(tmp_path):
    """A throwaway copy of the kelt4 example directory (data + params)."""
    work_dir = tmp_path / "kelt4"
    shutil.copytree(
        EXAMPLE_DIR, work_dir,
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
    )
    return work_dir


# ---------------------------------------------------------------------------
# Fast endpoint tests (fake handle -- no subprocess)
# ---------------------------------------------------------------------------

def test_status_idle_when_no_run(client):
    """
    Given a fresh app with no run started,
    When GET /api/run/status,
    Then it reports an inactive, idle run.
    """
    resp = client.get("/api/run/status")

    assert resp.status_code == 200
    assert resp.json() == {"active": False, "phase": "idle"}


def test_run_start_status_stop_roundtrip(client, monkeypatch, tmp_path):
    """
    Given start_run is stubbed with a fake handle,
    When a run is started, polled, and stopped through the endpoints,
    Then status reflects the phase and stop drives it to a terminal phase.
    """
    from exozippy.gui import runner

    fake = _FakeHandle(tmp_path)
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)

    started = client.post("/api/run", json={"config": "cfg.yaml",
                                            "project_dir": str(tmp_path)})
    assert started.status_code == 200
    body = started.json()
    assert body["active"] is True
    assert body["phase"] == "starting"
    assert body["log_path"].endswith("out/RUN.log")
    assert body["results_dir"].endswith("out")

    fake._phase = "sampling"
    status = client.get("/api/run/status").json()
    assert status["phase"] == "sampling"
    assert status["state"]["n_draws"] == 128

    stopped = client.post("/api/run/stop", json={"force": False})
    assert stopped.status_code == 200
    assert stopped.json()["phase"] in TERMINAL_PHASES
    assert fake.stop_calls == [False]


def test_run_guard_rail_only_one_active(client, monkeypatch, tmp_path):
    """
    Given a run is already active,
    When a second POST /api/run arrives,
    Then it is rejected with 409 (one run per project).
    """
    from exozippy.gui import runner

    fake = _FakeHandle(tmp_path)
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)

    first = client.post("/api/run", json={"config": "cfg.yaml",
                                          "project_dir": str(tmp_path)})
    assert first.status_code == 200

    second = client.post("/api/run", json={"config": "cfg.yaml",
                                           "project_dir": str(tmp_path)})
    assert second.status_code == 409
    assert "error" in second.json()


def test_run_snapshots_config_into_output_dir(client, monkeypatch, tmp_path):
    """
    Given a run started from a real config file,
    When POST /api/run runs,
    Then a frozen '<stem>.used.yaml' copy lands in the output directory.
    """
    from exozippy.gui import runner

    (tmp_path / "cfg.yaml").write_text("prefix: out/RUN\n")
    fake = _FakeHandle(tmp_path, config_path="cfg.yaml")
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)

    client.post("/api/run", json={"config": "cfg.yaml",
                                  "project_dir": str(tmp_path)})

    assert (tmp_path / "out" / "cfg.used.yaml").is_file()


def test_run_image_rejects_outside_tree(client, monkeypatch, tmp_path):
    """
    Given an active run,
    When GET /api/run/image asks for a path outside the run's cwd,
    Then it is forbidden (403).
    """
    from exozippy.gui import runner

    fake = _FakeHandle(tmp_path)
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)
    client.post("/api/run", json={"config": "cfg.yaml",
                                  "project_dir": str(tmp_path)})

    resp = client.get("/api/run/image", params={"path": "/etc/passwd"})
    assert resp.status_code == 403


def test_run_image_serves_file_inside_tree(client, monkeypatch, tmp_path):
    """
    Given an active run with a plot image on disk,
    When GET /api/run/image asks for it,
    Then the image bytes are served.
    """
    from exozippy.gui import runner

    img = tmp_path / "out" / "RUN_start_rv.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake png bytes")
    fake = _FakeHandle(tmp_path)
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)
    client.post("/api/run", json={"config": "cfg.yaml",
                                  "project_dir": str(tmp_path)})

    resp = client.get("/api/run/image", params={"path": str(img)})
    assert resp.status_code == 200
    assert resp.content.startswith(b"\x89PNG")


def test_stop_without_active_run_is_400(client):
    """
    Given no active run,
    When POST /api/run/stop,
    Then it returns 400 rather than crashing.
    """
    resp = client.post("/api/run/stop", json={"force": False})
    assert resp.status_code == 400


def test_utilities_run_unknown_name_is_400(client, tmp_path):
    """
    Given an unknown utility name,
    When POST /api/utilities/run,
    Then it returns 400 with an error.
    """
    resp = client.post("/api/utilities/run",
                       json={"name": "nope-not-real", "args": {},
                             "cwd": str(tmp_path)})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_utilities_schema_form_roundtrip(client):
    """
    Given the /api/utilities argument schema for a real utility,
    When a form's arg dict is marshalled back to argv,
    Then the utility's own parser accepts it (schema is faithful).
    """
    from exozippy.utilities.registry import all_utilities, args_dict_to_argv

    schema = client.get("/api/utilities").json()
    assert "getdata" in schema
    args = schema["getdata"]["arguments"]
    assert isinstance(args, list) and args

    # Build a minimal form-values dict: satisfy required args with placeholders.
    form = {}
    for arg in args:
        if arg["required"]:
            form[arg["name"]] = 1 if arg["type"] in ("int", "float") else "x"

    spec = all_utilities()["getdata"]
    parser = spec.build_parser()
    argv = args_dict_to_argv(parser, form)
    # The parser accepts the marshalled argv (it may add derived defaults).
    parsed, _unknown = parser.parse_known_args(argv)
    assert parsed is not None


# ---------------------------------------------------------------------------
# Slow end-to-end lifecycle through the endpoints
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.timeout(900)
def test_endpoint_run_lifecycle_start_sampling_stop(kelt4_workdir, tmp_path):
    """
    Given the run endpoints backed by the real subprocess runner,
    When a tiny PTDE fit is started, reaches sampling, and is stopped,
    Then status advances to 'sampling' and stop drives it to a terminal phase.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    out_prefix = tmp_path / "out" / "RUN"
    config_name = _write_ptde_config(kelt4_workdir, out_prefix)

    client = TestClient(create_app())
    started = client.post("/api/run", json={"config": config_name,
                                            "project_dir": str(kelt4_workdir)})
    assert started.status_code == 200

    try:
        def _sampling_with_progress():
            st = client.get("/api/run/status").json()
            if not st.get("alive") and st.get("phase") not in ("sampling",):
                return True
            return (st.get("phase") == "sampling"
                    and st.get("state", {}).get("n_draws", 0) >= 100)

        assert _poll_until(_sampling_with_progress, REACH_SAMPLING_TIMEOUT), \
            "run never reported n_draws>=100 during sampling"

        status = client.get("/api/run/status").json()
        assert status["phase"] == "sampling", f"unexpected phase {status}"

        # A frozen copy of the config was stashed for reproducibility.
        assert (out_prefix.parent / (Path(config_name).stem + ".used.yaml")).is_file()

        stopped = client.post("/api/run/stop", json={"force": False})
        assert stopped.status_code == 200

        def _terminal():
            st = client.get("/api/run/status").json()
            return st["phase"] if st.get("phase") in TERMINAL_PHASES else None

        final_phase = _poll_until(_terminal, timeout=600.0)
        if final_phase is None:
            client.post("/api/run/stop", json={"force": True})
            final_phase = _poll_until(_terminal, timeout=120.0)
    finally:
        client.post("/api/run/stop", json={"force": True})

    assert final_phase in {"stopped", "done"}, f"non-terminal end: {final_phase}"
