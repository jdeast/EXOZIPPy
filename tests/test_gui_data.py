"""Tests for the GUI data-file manager (G9).

Covers the SCHEMA-DRIVEN association eligibility helper/endpoint (with a fake
component to prove no component names are hardcoded), the current-association
mapping, the directory-listing helper, and the preview endpoint returning
PlotSpec JSON for a real kelt4 RV instance.

Follows AAA with Given/When/Then docstrings.
"""

import json
import shutil
from pathlib import Path

import pytest

from exozippy.gui import datafiles

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "kelt4"


# --- schema-driven eligibility (no component names hardcoded) ----------------

def _fake_schema():
    """A schema with a made-up component declaring a custom datafile glob."""
    return {
        "components": {
            "gadget": {
                "config": [
                    {
                        "key": "trace",
                        "kind": "datafile",
                        "accepts": "*.widget",
                        "required": True,
                        "doc": "A widget trace file.",
                    },
                    {
                        "key": "mode",
                        "kind": "option",
                        "accepts": None,
                        "required": False,
                        "doc": "not a datafile",
                    },
                ]
            }
        }
    }


def test_eligible_associations_matches_custom_glob():
    """
    Given a fake component whose schema declares a '*.widget' datafile key
        and one instance of it in the config,
    When eligible_associations is asked about a matching filename,
    Then that instance/key pair is returned (purely from the schema).
    """
    config = {"gadget": [{"name": "g1"}]}

    eligible = datafiles.eligible_associations(
        "run3.widget", config, _fake_schema()
    )

    assert eligible == [
        {
            "comp_type": "gadget",
            "name": "g1",
            "key": "trace",
            "glob": "*.widget",
            "doc": "A widget trace file.",
        }
    ]


def test_eligible_associations_excludes_nonmatching_file():
    """
    Given the same fake component and instance,
    When a file that does NOT match the glob is checked,
    Then no eligible pair is returned.
    """
    config = {"gadget": [{"name": "g1"}]}

    eligible = datafiles.eligible_associations(
        "run3.rv", config, _fake_schema()
    )

    assert eligible == []


def test_eligible_associations_skips_component_with_no_instances():
    """
    Given a matching filename but no instance of the declaring component,
    When eligibility is computed,
    Then nothing is eligible (there is nothing to associate with).
    """
    eligible = datafiles.eligible_associations(
        "run3.widget", {}, _fake_schema()
    )

    assert eligible == []


def test_current_associations_maps_basename_to_instances():
    """
    Given a config where an instance references a datafile by path,
    When current_associations runs,
    Then the file basename maps to that instance/key/path.
    """
    config = {"gadget": [{"name": "g1", "trace": "data/run3.widget"}]}

    assoc = datafiles.current_associations(config, _fake_schema())

    assert assoc == {
        "run3.widget": [
            {"comp_type": "gadget", "name": "g1", "key": "trace",
             "path": "data/run3.widget"}
        ]
    }


def test_list_directory_lists_files_and_dirs(tmp_path):
    """
    Given a directory with a file, a subdir, and a dotfile,
    When list_directory runs rooted at that dir,
    Then it lists the file and subdir, skips the dotfile, and has no parent.
    """
    (tmp_path / "a.rv").write_text("0 1 2\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").write_text("x\n")

    result = datafiles.list_directory(str(tmp_path), root=str(tmp_path))

    names = [e["name"] for e in result["entries"]]
    assert "a.rv" in names and "sub" in names
    assert ".hidden" not in names
    assert result["parent"] is None  # cannot escape the project root
    json.dumps(result)  # JSON-serializable


# --- endpoint tests (require the 'gui' extra) --------------------------------

@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from exozippy.gui.app import create_app

    return TestClient(create_app())


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


def test_files_eligible_endpoint_is_schema_driven(client, rvonly_project):
    """
    Given an opened RV-only project (rvinstrument HIRES + TRES),
    When POST /api/files/eligible asks about an .rv file,
    Then both rvinstrument instances appear with the datafile key 'file'.
    """
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})

    resp = client.post("/api/files/eligible", json={"filename": "KELT-4b.HIRES.rv"})

    assert resp.status_code == 200
    eligible = resp.json()["eligible"]
    pairs = {(e["comp_type"], e["name"], e["key"]) for e in eligible}
    assert ("rvinstrument", "HIRES", "file") in pairs
    assert ("rvinstrument", "TRES", "file") in pairs
    # A non-datafile key (star_ndx ref) never appears as an association target.
    assert all(e["key"] == "file" for e in eligible)


def test_files_eligible_endpoint_rejects_wrong_extension(client, rvonly_project):
    """
    Given the opened RV-only project,
    When an .sed file is checked for eligibility,
    Then no rvinstrument instance is eligible (glob is '*.rv').
    """
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})

    resp = client.post("/api/files/eligible", json={"filename": "kelt4.sed.yaml"})

    assert resp.status_code == 200
    assert resp.json()["eligible"] == []


def test_files_and_associations_endpoints(client, rvonly_project):
    """
    Given the opened RV-only project,
    When the browser lists files and asks for current associations,
    Then the .rv files appear and each maps to its rvinstrument instance.
    """
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})

    files = client.get("/api/files").json()
    names = [e["name"] for e in files["entries"]]
    assert "KELT-4b.HIRES.rv" in names

    assoc = client.get("/api/files/associations").json()["associations"]
    assert assoc["KELT-4b.HIRES.rv"][0]["comp_type"] == "rvinstrument"
    assert assoc["KELT-4b.HIRES.rv"][0]["name"] == "HIRES"


@pytest.mark.slow
def test_preview_endpoint_returns_plotspec_json(client, rvonly_project):
    """
    Given the opened RV-only kelt4 project,
    When POST /api/preview requests the rvinstrument data-only preview,
    Then it returns >= 1 PlotSpec with observed RV data traces.
    """
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})

    resp = client.post("/api/preview", json={"comp_type": "rvinstrument"})

    assert resp.status_code == 200
    body = resp.json()
    assert "specs" in body, body
    specs = body["specs"]
    assert len(specs) >= 1
    # data-only: every trace is observational, no model curves
    roles = {t["role"] for s in specs for t in s["traces"]}
    assert roles == {"data"}
    # and the payload is real JSON with numeric arrays
    first = specs[0]
    assert first["traces"][0]["x"] and first["traces"][0]["y"]


def test_preview_endpoint_without_doc_is_400(client):
    """Given no open document, When POST /api/preview, Then it returns 400."""
    resp = client.post("/api/preview", json={"comp_type": "rvinstrument"})
    assert resp.status_code == 400
    assert "error" in resp.json()
