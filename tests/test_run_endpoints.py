"""Tests for the G11 run-control endpoints on the GUI app.

Two layers:

* Fast endpoint tests drive POST /api/run, GET /api/run/status,
  POST /api/run/stop, GET /api/run/plots, GET /api/run/image and
  POST /api/utilities/run with a *fake* RunHandle (no subprocess), so the HTTP
  wiring, guard rails, status payload, and path safety are covered in
  milliseconds.
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

# Poll budgets for the slow end-to-end test at the bottom. Their sum must sit
# comfortably UNDER that test's @pytest.mark.timeout(900): if the guard fires
# first, pytest-timeout kills the test instead of letting the poll report what
# it was waiting for (and under xdist that used to surface as a nameless dead
# worker). Warm, that test measures 52-84 s end to end; a cold pytensor compile
# cache alone has been seen to multiply it ~6x, so the one budget that spans a
# compile stays large and the post-stop ones -- which only cover wrap-up and
# process reaping -- are sized to that work, not to the compile.
#
#   REACH_SAMPLING_TIMEOUT 360 s  subprocess + imports + model build + a COLD
#                                 pytensor compile + tune + 100 draws
#   GRACEFUL_EXIT_TIMEOUT  240 s  wrap-up: save partial trace + reports/plots
#                                 (~20-30 s warm, so ~8x headroom)
#   FORCE_EXIT_TIMEOUT      45 s  only reached after stop(force=True), which
#                                 itself blocks ~50 s through second-SIGINT and
#                                 SIGKILL; the terminal phase is then written
#                                 (or synthesized by RunHandle.status) at once
#
# Worst case, counting the ~50 s each force stop blocks internally:
# 360 + 240 + 50 + 45 + 50 = 745 s < 900 s.
REACH_SAMPLING_TIMEOUT = 360.0
GRACEFUL_EXIT_TIMEOUT = 240.0
FORCE_EXIT_TIMEOUT = 45.0
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
        "measure_scales": False,
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
        EXAMPLE_DIR,
        work_dir,
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

    started = client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )
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

    first = client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )
    assert first.status_code == 200

    second = client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )
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

    client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )

    assert (tmp_path / "out" / "cfg.used.yaml").is_file()


def test_run_snapshots_the_params_file_the_fit_will_read(
    client, monkeypatch, tmp_path
):
    """
    Given a config naming a parameter_file,
    When POST /api/run runs with no explicit params argument,
    Then that params file is snapshotted into the output dir too.

    Reproduces review 2.11.2: every caller omitted the argument, so the params
    branch of _snapshot_run_inputs never ran and the promised '.used' copy of
    the file that actually sets the start values was never written.
    """
    from exozippy.gui import runner

    (tmp_path / "cfg.yaml").write_text(
        "prefix: out/RUN\nparameter_file: cfg.params.yaml\n"
    )
    (tmp_path / "cfg.params.yaml").write_text("star.A.teff: {initval: 5800}\n")
    fake = _FakeHandle(tmp_path, config_path="cfg.yaml")
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)

    client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )

    used = tmp_path / "out" / "cfg.params.used.yaml"
    assert used.is_file()
    assert used.read_text() == (tmp_path / "cfg.params.yaml").read_text()


def test_run_snapshot_prefers_an_explicit_params_argument(
    client, monkeypatch, tmp_path
):
    """
    Given a request that names its own params file,
    When POST /api/run runs,
    Then that file is snapshotted instead of the config's parameter_file.
    """
    from exozippy.gui import runner

    (tmp_path / "cfg.yaml").write_text(
        "prefix: out/RUN\nparameter_file: cfg.params.yaml\n"
    )
    (tmp_path / "cfg.params.yaml").write_text("star.A.teff: {initval: 5800}\n")
    (tmp_path / "other.params.yaml").write_text(
        "star.A.teff: {initval: 6100}\n"
    )
    fake = _FakeHandle(tmp_path, config_path="cfg.yaml")
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)

    client.post(
        "/api/run",
        json={
            "config": "cfg.yaml",
            "params": "other.params.yaml",
            "project_dir": str(tmp_path),
        },
    )

    assert (tmp_path / "out" / "other.params.used.yaml").is_file()
    assert not (tmp_path / "out" / "cfg.params.used.yaml").exists()


def test_run_snapshot_survives_a_config_without_a_params_file(
    client, monkeypatch, tmp_path
):
    """
    Given a config with no parameter_file key (or an unreadable one),
    When POST /api/run runs,
    Then the config snapshot still happens and nothing raises.
    """
    from exozippy.gui import runner

    (tmp_path / "cfg.yaml").write_text("prefix: out/RUN\n")
    fake = _FakeHandle(tmp_path, config_path="cfg.yaml")
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)

    resp = client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )
    assert resp.status_code == 200
    assert (tmp_path / "out" / "cfg.used.yaml").is_file()


def test_run_plots_lists_raster_images_but_not_pdfs(
    client, monkeypatch, tmp_path
):
    """
    Given an active run with start/mcmc plot files beside its prefix,
    When GET /api/run/plots is polled,
    Then raster images are listed under the phase their filename names, while
    PDFs and non-image files are not.

    This endpoint has no frontend caller today -- it is a kept seam (the
    run-plot gallery; see gui.md), so its contract is pinned here. The PDF
    exclusion is the CURRENT behavior, and it is also the reason a restored
    gallery would show nothing: ``plotrender`` writes ``{prefix}_{tag}.pdf``,
    so a real fit produces no file this endpoint will list. Fixing that (a
    raster copy, or serving PDFs and rendering them as links/embeds) is part
    of restoring the gallery, and this test is where the change shows up.
    """
    from exozippy.gui import runner

    out = tmp_path / "out"
    out.mkdir()
    (out / "RUN_start_rv.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (out / "RUN_mcmc_rv.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (out / "RUN_mcmc_transit.pdf").write_bytes(b"%PDF-1.4")
    (out / "RUN_start_notes.txt").write_text("not an image")
    fake = _FakeHandle(tmp_path)
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)
    client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )

    body = client.get("/api/run/plots").json()

    assert [Path(p).name for p in body["start"]] == ["RUN_start_rv.png"]
    assert [Path(p).name for p in body["progress"]] == ["RUN_mcmc_rv.png"]


def test_run_plots_without_active_run_is_empty(client):
    """
    Given no active run,
    When GET /api/run/plots is polled,
    Then both lists are empty rather than an error (the poll is best-effort).
    """
    assert client.get("/api/run/plots").json() == {
        "start": [],
        "progress": [],
    }


def test_run_image_rejects_outside_tree(client, monkeypatch, tmp_path):
    """
    Given an active run,
    When GET /api/run/image asks for a path outside the run's cwd,
    Then it is forbidden (403).
    """
    from exozippy.gui import runner

    fake = _FakeHandle(tmp_path)
    monkeypatch.setattr(runner, "start_run", lambda *a, **k: fake)
    client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )

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
    client.post(
        "/api/run", json={"config": "cfg.yaml", "project_dir": str(tmp_path)}
    )

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
    resp = client.post(
        "/api/utilities/run",
        json={"name": "nope-not-real", "args": {}, "cwd": str(tmp_path)},
    )
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
    started = client.post(
        "/api/run",
        json={"config": config_name, "project_dir": str(kelt4_workdir)},
    )
    assert started.status_code == 200

    try:

        def _sampling_with_progress():
            st = client.get("/api/run/status").json()
            if not st.get("alive") and st.get("phase") not in ("sampling",):
                return True
            return (
                st.get("phase") == "sampling"
                and st.get("state", {}).get("n_draws", 0) >= 100
            )

        assert _poll_until(_sampling_with_progress, REACH_SAMPLING_TIMEOUT), (
            "run never reported n_draws>=100 during sampling within "
            f"{REACH_SAMPLING_TIMEOUT}s; last status: "
            f"{client.get('/api/run/status').json()}"
        )

        status = client.get("/api/run/status").json()
        assert status["phase"] == "sampling", f"unexpected phase {status}"

        # A frozen copy of the config was stashed for reproducibility.
        assert (
            out_prefix.parent / (Path(config_name).stem + ".used.yaml")
        ).is_file()

        stopped = client.post("/api/run/stop", json={"force": False})
        assert stopped.status_code == 200

        # Keep the last full status doc so a failure message carries the
        # child's recorded error/traceback, not just the bare phase name.
        final_status = {}

        def _terminal():
            st = client.get("/api/run/status").json()
            final_status.clear()
            final_status.update(st)
            return st["phase"] if st.get("phase") in TERMINAL_PHASES else None

        final_phase = _poll_until(_terminal, timeout=GRACEFUL_EXIT_TIMEOUT)
        escalated = final_phase is None
        if escalated:
            client.post("/api/run/stop", json={"force": True})
            final_phase = _poll_until(_terminal, timeout=FORCE_EXIT_TIMEOUT)
    finally:
        client.post("/api/run/stop", json={"force": True})

    how = (
        f"graceful stop timed out after {GRACEFUL_EXIT_TIMEOUT}s and was "
        "force-escalated"
        if escalated
        else "graceful stop was honored"
    )
    assert final_phase in {"stopped", "done"}, (
        f"non-terminal end: {final_phase}; {how}; last status: {final_status}"
    )
