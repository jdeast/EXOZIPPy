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

    (tmp_path / "system.yaml").write_text("a: 1\n")
    (tmp_path / "system.params.yaml").write_text("b: 2\n")
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
    (tmp_path / "cfg.yaml").write_text("x: 1\n")
    resp = client.post("/api/project/open", json={"path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["configs"][0]["name"] == "cfg.yaml"


def test_project_open_endpoint_bad_dir(client):
    """Given a bad path, When POST /api/project/open, Then it returns 400."""
    resp = client.post("/api/project/open", json={"path": "/no/such/dir/here"})
    assert resp.status_code == 400
    assert "error" in resp.json()
