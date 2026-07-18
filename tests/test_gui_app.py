"""Tests for the GUI application shell (G7).

Covers the import-safety contract (plain `import exozippy` must not drag in
FastAPI), the project-directory listing helper, and the JSON API endpoints via
FastAPI's TestClient. The endpoint tests are skipped when the optional 'gui'
extra is not installed, so the suite stays green in a bare environment.
"""

import subprocess
import sys

import pytest


def test_import_exozippy_does_not_import_fastapi():
    """Given a bare interpreter, When exozippy is imported, Then fastapi is not.

    The GUI is optional; the core package and CLI must import without the
    'gui' extra present. We run a fresh interpreter so an already-imported
    fastapi in the test process cannot mask a regression.
    """
    code = (
        "import sys, exozippy\n"
        "assert 'fastapi' not in sys.modules, sorted(m for m in sys.modules if 'fastapi' in m)\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_open_project_lists_and_classifies(tmp_path):
    """Given a dir of files, When open_project runs, Then files are classified."""
    from exozippy.gui.app import open_project

    # A config is recognized by content (a known component/global block, here
    # the global 'prefix'); a params file by its all-dotted keys or its name.
    (tmp_path / "system.yaml").write_text("prefix: out\n")
    (tmp_path / "system.params.yaml").write_text("star.0.mass: {initval: 1}\n")
    (tmp_path / "lc.rv").write_text("0 1 2\n")
    (tmp_path / "notes.md").write_text("hi\n")
    (tmp_path / ".hidden.yaml").write_text("skip: me\n")

    result = open_project(str(tmp_path))

    assert [f["name"] for f in result["configs"]] == ["system.yaml"]
    assert [f["name"] for f in result["params"]] == ["system.params.yaml"]
    assert [f["name"] for f in result["data_files"]] == ["lc.rv"]
    assert [f["name"] for f in result["other"]] == ["notes.md"]
    # Hidden files are skipped and everything is JSON-serializable.
    import json

    json.dumps(result)


def test_open_project_classifies_by_content_not_only_name(tmp_path):
    """Given YAMLs whose names mislead, When open_project runs, Then content wins.

    A params override whose name does not end in .params.yaml (all-dotted keys)
    is still 'params'; a YAML that is neither a config nor params (a component's
    own input file) is 'other', not a spurious config.
    """
    from exozippy.gui.app import open_project

    (tmp_path / "real.yaml").write_text("prefix: out\nstar: {}\n")
    # Misnamed params file: dotted keys, but not the *.params.yaml convention.
    (tmp_path / "overrides.3.yaml").write_text("orbit.b.cosi: {initval: 0}\n")
    # A component input file: a mapping, but no config/global blocks.
    (tmp_path / "star_input.yaml").write_text("model: NextGen\nnstars: 3\n")

    result = open_project(str(tmp_path))

    assert [f["name"] for f in result["configs"]] == ["real.yaml"]
    assert [f["name"] for f in result["params"]] == ["overrides.3.yaml"]
    assert "star_input.yaml" in [f["name"] for f in result["other"]]


def test_open_project_rejects_missing_dir(tmp_path):
    """Given a nonexistent path, When open_project runs, Then it raises ValueError."""
    from exozippy.gui.app import open_project

    with pytest.raises(ValueError):
        open_project(str(tmp_path / "nope"))


# --- endpoint tests (require the 'gui' extra) --------------------------------

@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    return TestClient(create_app())


def test_health_endpoint(client):
    """Given the app, When GET /api/health, Then it reports ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_schema_endpoint_lists_components(client):
    """Given the app, When GET /api/schema, Then known components appear."""
    resp = client.get("/api/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "components" in data and "global" in data
    for expected in ("star", "planet", "orbit", "transit"):
        assert expected in data["components"]


def test_utilities_endpoint(client):
    """Given the app, When GET /api/utilities, Then declared utilities appear."""
    resp = client.get("/api/utilities")
    assert resp.status_code == 200
    names = resp.json()
    assert "getdata" in names
    # Each entry is a JSON argument schema (has an 'arguments' list).
    assert "arguments" in names["getdata"]


def test_project_open_endpoint(client, tmp_path):
    """Given a real dir, When POST /api/project/open, Then it lists files."""
    (tmp_path / "cfg.yaml").write_text("prefix: out\n")
    resp = client.post("/api/project/open", json={"path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["configs"][0]["name"] == "cfg.yaml"


def test_project_open_endpoint_bad_dir(client):
    """Given a bad path, When POST /api/project/open, Then it returns 400."""
    resp = client.post("/api/project/open", json={"path": "/no/such/dir/here"})
    assert resp.status_code == 400
    assert "error" in resp.json()


# --- config document endpoints (G8) ------------------------------------------

import shutil
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "kelt4"


@pytest.fixture
def rvonly_project(tmp_path):
    """A working copy of the cheap RV-only KELT-4 example."""
    for name in (
        "kelt4_rvonly.yaml",
        "kelt4.params.yaml",
        "KELT-4b.HIRES.rv",
        "KELT-4b.TRES.rv",
    ):
        shutil.copy(EXAMPLE_DIR / name, tmp_path / name)
    return tmp_path


def test_doc_get_without_open_is_404(client):
    """Given no open document, When GET /api/doc, Then it returns 404."""
    resp = client.get("/api/doc")
    assert resp.status_code == 404


def test_doc_open_command_undo_save_flow(client, rvonly_project):
    """Given an opened document, When a command runs, undoes, and saves, Then
    the endpoints report dirty state and the file changes on save."""
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")

    # open
    resp = client.post("/api/doc/open", json={"config_path": config_path})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dirty"] is False
    assert body["config"]["star"][0]["name"] == "A"

    # command: set a param field
    resp = client.post(
        "/api/doc/command",
        json={"op": "set_param_field",
              "args": {"path": "star.A.teff", "field": "initval", "value": 6300}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dirty"] is True
    assert body["undo_depth"] == 1
    assert body["params"]["star.A.teff"]["initval"] == 6300

    # undo
    resp = client.post("/api/doc/undo")
    assert resp.json()["undo_depth"] == 0
    assert resp.json()["redo_depth"] == 1

    # redo then save
    client.post("/api/doc/redo")
    params_file = rvonly_project / "kelt4.params.yaml"
    resp = client.post("/api/doc/save")
    assert resp.status_code == 200
    assert resp.json()["dirty"] is False
    assert "6300" in params_file.read_text()


def test_doc_command_bad_op_is_400(client, rvonly_project):
    """Given an open doc, When an unknown command op is posted, Then 400."""
    client.post(
        "/api/doc/open",
        json={"config_path": str(rvonly_project / "kelt4_rvonly.yaml")},
    )
    resp = client.post("/api/doc/command", json={"op": "nonsense", "args": {}})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_doc_validate_job_lifecycle(client, rvonly_project):
    """Given an open doc, When validation is requested, Then a job id is
    returned and polling eventually reports a terminal status with a
    diagnostics list."""
    client.post(
        "/api/doc/open",
        json={"config_path": str(rvonly_project / "kelt4_rvonly.yaml")},
    )
    resp = client.post("/api/doc/validate")
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == "running"

    # unknown job -> 404
    assert client.get("/api/doc/validate/deadbeef").status_code == 404

    # poll to completion (validation runs the relaxation engine off-thread)
    deadline = time.time() + 120
    status = "running"
    diagnostics = None
    while time.time() < deadline:
        poll = client.get(f"/api/doc/validate/{job_id}").json()
        status = poll["status"]
        diagnostics = poll["diagnostics"]
        if status != "running":
            break
        time.sleep(0.5)

    assert status in ("done", "error")
    assert isinstance(diagnostics, list)
