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


# --- CLI arg resolution (`exozippy-gui [project]`) ----------------------------


def test_resolve_project_arg_defaults_to_cwd(tmp_path):
    """Given no positional arg, When resolved, Then it falls back to cwd."""
    from exozippy.gui.app import resolve_project_arg

    project_dir, initial_config = resolve_project_arg(None, cwd=str(tmp_path))

    assert project_dir == str(tmp_path)
    assert initial_config is None


def test_resolve_project_arg_directory(tmp_path):
    """Given a directory arg, When resolved, Then it opens as the project with no config."""
    from exozippy.gui.app import resolve_project_arg

    project_dir, initial_config = resolve_project_arg(str(tmp_path))

    assert project_dir == str(tmp_path)
    assert initial_config is None


def test_resolve_project_arg_config_file(tmp_path):
    """Given a config file arg (e.g. 'kelt4.yaml'), When resolved, Then the
    parent dir becomes the project and the file is the config to pre-select.
    """
    from exozippy.gui.app import resolve_project_arg

    config = tmp_path / "kelt4.yaml"
    config.write_text("prefix: out\n")

    project_dir, initial_config = resolve_project_arg(str(config))

    assert project_dir == str(tmp_path)
    assert initial_config == str(config)


def test_resolve_project_arg_relative_path(tmp_path, monkeypatch):
    """Given a relative config path, When resolved, Then it resolves against cwd."""
    from exozippy.gui.app import resolve_project_arg

    (tmp_path / "kelt4.yaml").write_text("prefix: out\n")
    monkeypatch.chdir(tmp_path)

    project_dir, initial_config = resolve_project_arg("kelt4.yaml")

    assert project_dir == str(tmp_path)
    assert initial_config == str(tmp_path / "kelt4.yaml")


def test_resolve_project_arg_missing_path_raises(tmp_path):
    """Given a path that does not exist, When resolved, Then it raises ValueError."""
    from exozippy.gui.app import resolve_project_arg

    with pytest.raises(ValueError):
        resolve_project_arg(str(tmp_path / "nope.yaml"))


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


def test_config_endpoint_reports_initial_project_and_config(tmp_path):
    """Given create_app(project_dir=..., initial_config=...), When GET
    /api/config, Then the client bootstrap payload carries both, resolved.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    config = tmp_path / "kelt4.yaml"
    config.write_text("prefix: out\n")

    app = create_app(project_dir=str(tmp_path), initial_config=str(config))
    resp = TestClient(app).get("/api/config")

    assert resp.status_code == 200
    assert resp.json() == {
        "initial_project": str(tmp_path.resolve()),
        "initial_config": str(config.resolve()),
    }


def test_config_endpoint_defaults_to_none(client):
    """Given create_app() with no args, When GET /api/config, Then both are null."""
    resp = client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json() == {"initial_project": None, "initial_config": None}


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
        json={
            "op": "set_param_field",
            "args": {"path": "star.A.teff", "field": "initval", "value": 6300},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dirty"] is True
    assert body["undo_depth"] == 1
    # The edit is addressed by NAME (what the GUI displays) but this params
    # file spells that element by INDEX, and the writer updates the entry the
    # file already has rather than appending a second spelling of one element
    # -- which ConfigManager rejects outright.  See
    # test_gui_document.py::test_slider_edit_updates_the_users_own_spelling.
    assert body["params"]["star.0.teff"]["initval"] == 6300

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


def test_doc_reopen_same_path_keeps_unsaved_edits(client, rvonly_project):
    """Given a dirty open document, When the same config is opened again (as
    every tab does on mount), Then the in-memory edits and undo stack survive;
    and Given a clean document, When reopened, Then it reloads from disk."""
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})
    client.post(
        "/api/doc/command",
        json={
            "op": "set_param_field",
            "args": {"path": "star.A.teff", "field": "initval", "value": 6300},
        },
    )

    # Re-open (e.g. a tab remount): the dirty doc must be returned untouched.
    resp = client.post("/api/doc/open", json={"config_path": config_path})
    body = resp.json()
    assert body["dirty"] is True
    assert body["undo_depth"] == 1
    # Index form: the writer preserves the spelling the file already uses.
    assert body["params"]["star.0.teff"]["initval"] == 6300

    # After save the doc is clean; a re-open now reloads from disk.
    client.post("/api/doc/save")
    resp = client.post("/api/doc/open", json={"config_path": config_path})
    body = resp.json()
    assert body["dirty"] is False
    assert body["undo_depth"] == 0
    assert body["params"]["star.0.teff"]["initval"] == 6300


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

    # A job read to completion is retired, so the dict does not grow one entry
    # per edit for the life of the server (review 6.12.1).
    assert client.get(f"/api/doc/validate/{job_id}").status_code == 404


def test_abandoned_validate_jobs_are_bounded(
    client, rvonly_project, monkeypatch
):
    """
    Given many validations started and never polled to completion,
    When more are requested,
    Then the oldest are evicted rather than retained forever.

    Pop-on-terminal-read retires the jobs the frontend follows to completion;
    this covers the ones it abandons (a tab unmounted, a poll loop that hit its
    own deadline first), which nothing else would ever remove -- and each holds
    a full diagnostics list.  The real validator is stubbed out: the point is
    the bookkeeping, and 30 relaxation-engine solves are not.
    """
    import exozippy.solve_api
    from exozippy.gui import app as gui_app

    monkeypatch.setattr(exozippy.solve_api, "validate", lambda *a, **k: [])
    monkeypatch.setattr(gui_app, "_MAX_VALIDATE_JOBS", 4)

    client.post(
        "/api/doc/open",
        json={"config_path": str(rvonly_project / "kelt4_rvonly.yaml")},
    )
    first = client.post("/api/doc/validate").json()["job_id"]
    for _ in range(6):
        assert client.post("/api/doc/validate").status_code == 200

    assert client.get(f"/api/doc/validate/{first}").status_code == 404


# --- classification of edge-case YAML (review 1.12.6) -------------------------


def test_empty_and_comment_only_yaml_are_selectable_as_configs(tmp_path):
    """
    Given a freshly created (empty or comment-only) config file,
    When the project is listed,
    Then it is classified as a config, not dumped in 'other'.

    Both parse to ``None``, so the key-based rules had nothing to match and the
    file landed in 'other' -- where the config picker cannot see it at all,
    which is exactly when a user needs to select it.
    """
    from exozippy.gui.app import open_project

    (tmp_path / "fresh.yaml").write_text("")
    (tmp_path / "sketch.yaml").write_text("# nothing here yet\n")
    (tmp_path / "blank.params.yaml").write_text("# no overrides yet\n")

    result = open_project(str(tmp_path))

    assert sorted(f["name"] for f in result["configs"]) == [
        "fresh.yaml",
        "sketch.yaml",
    ]
    assert [f["name"] for f in result["params"]] == ["blank.params.yaml"]


def test_a_transient_introspection_failure_is_not_cached_forever(monkeypatch):
    """
    Given one failing call into the introspection layer,
    When it later succeeds,
    Then the real key set is used.

    An ``@lru_cache`` froze the four-key literal fallback for the whole process
    after a single transient import failure, and every config in every project
    then classified as 'other' with no way back short of a restart.
    """
    from exozippy.gui import app as gui_app

    monkeypatch.setattr(gui_app, "_CONFIG_TOP_KEYS", None)
    real_import = __import__

    def boom(name, *args, **kwargs):
        if name.endswith("introspect"):
            raise ImportError("transient")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as failing:
        failing.setattr("builtins.__import__", boom)
        degraded = gui_app._config_top_keys()

    assert "star" not in degraded  # the literal fallback was used ...
    assert gui_app._CONFIG_TOP_KEYS is None  # ... and NOT remembered
    assert "star" in gui_app._config_top_keys()  # the retry recovers


def test_a_huge_yaml_is_classified_without_being_parsed(tmp_path, monkeypatch):
    """
    Given a .yaml far larger than any config,
    When the project is listed,
    Then it is classified by name rather than parsed.
    """
    from exozippy.gui import app as gui_app

    monkeypatch.setattr(gui_app, "_YAML_PARSE_MAX_BYTES", 64)
    monkeypatch.setattr(gui_app, "_CLASSIFY_CACHE", {})
    huge = tmp_path / "huge.params.yaml"
    huge.write_text("# " + "x" * 500 + "\n")

    result = gui_app.open_project(str(tmp_path))

    assert [f["name"] for f in result["params"]] == ["huge.params.yaml"]


# --- file browser confinement (review 2.12.2 / 2.12.6) ------------------------


def test_files_browser_refuses_a_directory_outside_the_project(tmp_path):
    """
    Given a server whose open project is one directory,
    When /api/files is asked for a directory outside it,
    Then it is refused.

    The root used to gate only the parent LINK, so ``GET /api/files?dir=/etc``
    happily listed /etc and the documented sandbox was cosmetic.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")

    client = TestClient(create_app(project_dir=str(project)))

    resp = client.get("/api/files", params={"dir": str(outside)})

    assert resp.status_code == 400
    assert "outside" in resp.json()["error"]


def test_files_browser_root_follows_the_open_project(tmp_path):
    """
    Given a server launched on one project that then opens another,
    When /api/files lists the new project,
    Then it is allowed (and the old one is not).

    ``root`` was a closure over the LAUNCH project that /api/project/open never
    updated, so after a switch the browser could not navigate above the config
    directory of the project actually open.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    (second / "data").mkdir(parents=True)

    client = TestClient(create_app(project_dir=str(first)))
    assert (
        client.get("/api/files", params={"dir": str(second)}).status_code
        == 400
    )

    client.post("/api/project/open", json={"path": str(second)})

    assert (
        client.get("/api/files", params={"dir": str(second)}).status_code
        == 200
    )
    assert (
        client.get("/api/files", params={"dir": str(first)}).status_code == 400
    )


def test_browse_reports_an_unreadable_directory_as_a_client_error(
    client, tmp_path
):
    """
    Given a directory the server cannot read,
    When /api/browse lists it,
    Then it answers 400 rather than raising a bare 500.
    """
    import os

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        if os.access(locked, os.R_OK):  # running as root: nothing to test
            pytest.skip("cannot make a directory unreadable as this user")
        resp = client.get("/api/browse", params={"dir": str(locked)})
        assert resp.status_code == 400
        assert "error" in resp.json()
    finally:
        locked.chmod(0o755)


# --- document command error mapping (review 2.12.4) ---------------------------


def test_a_bad_path_in_a_command_is_a_400_not_a_500(client, rvonly_project):
    """
    Given a config-key path that indexes past a list or traverses a scalar,
    When the command runs,
    Then the server answers 400 and the document is unchanged.

    Both raise IndexError/TypeError out of ``_set_nested``; they escaped as
    500s, so the UI showed a generic failure for an edit the command's own
    snapshot restore had already cleanly rolled back.
    """
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})
    before = client.get("/api/doc").json()

    for path in ("star.99.teff", "prefix.nested.key"):
        resp = client.post(
            "/api/doc/command",
            json={
                "op": "set_config_key",
                "args": {"path": path, "value": 1},
            },
        )
        assert resp.status_code == 400, path
        assert "error" in resp.json()

    assert client.get("/api/doc").json()["config"] == before["config"]


def test_the_unwired_autosave_endpoint_is_gone(client):
    """
    Given the removed POST /api/doc/autosave,
    When it is called,
    Then it does not exist.

    It had no api.ts client, no test and no entry in gui.md's known-unwired
    list; server-side autosave is invoked directly by the project-switch path.
    """
    assert client.post("/api/doc/autosave").status_code in (404, 405)
