"""Tests for the GUI data-file manager (G9).

Covers the SCHEMA-DRIVEN association eligibility helper/endpoint (with a fake
component to prove no component names are hardcoded), the current-association
mapping, and the directory-listing helper.

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
            {
                "comp_type": "gadget",
                "name": "g1",
                "key": "trace",
                "path": "data/run3.widget",
            }
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

    resp = client.post(
        "/api/files/eligible", json={"filename": "KELT-4b.HIRES.rv"}
    )

    assert resp.status_code == 200
    eligible = resp.json()["eligible"]
    pairs = {(e["comp_type"], e["name"], e["key"]) for e in eligible}
    assert ("rvinstrument", "HIRES", "file") in pairs
    assert ("rvinstrument", "TRES", "file") in pairs
    # A non-datafile key (star_ndx ref) never appears as an association target.
    assert all(e["key"] == "file" for e in eligible)


def test_files_eligible_endpoint_rejects_wrong_extension(
    client, rvonly_project
):
    """
    Given the opened RV-only project,
    When an .sed file is checked for eligibility,
    Then no rvinstrument instance is eligible (glob is '*.rv').
    """
    config_path = str(rvonly_project / "kelt4_rvonly.yaml")
    client.post("/api/doc/open", json={"config_path": config_path})

    resp = client.post(
        "/api/files/eligible", json={"filename": "kelt4.sed.yaml"}
    )

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


# --- confinement (review 2.12.2 / 2.12.6 / 4.12.2) ---------------------------


def test_list_directory_refuses_a_path_outside_the_root(tmp_path):
    """
    Given a root,
    When a directory outside it is listed,
    Then it raises instead of listing it.

    ``root`` used to gate only whether ``parent`` was reported, so the
    "sandboxed to the open project" claim was cosmetic: a caller who passed an
    absolute path got whatever it named.
    """
    from exozippy.gui import datafiles

    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    with pytest.raises(ValueError, match="outside"):
        datafiles.list_directory(str(outside), root=str(root))


def test_list_directory_allows_a_subdirectory_of_the_root(tmp_path):
    """Given a nested directory inside the root, When listed, Then it works
    and reports its parent."""
    from exozippy.gui import datafiles

    nested = tmp_path / "data" / "night1"
    nested.mkdir(parents=True)

    listing = datafiles.list_directory(str(nested), root=str(tmp_path))

    assert listing["parent"] == str(tmp_path / "data")


def test_list_directory_reports_an_unreadable_directory_as_a_value_error(
    tmp_path,
):
    """
    Given a directory the process cannot read,
    When it is listed,
    Then a ValueError names it -- the endpoint turns that into a 400 rather
    than letting a raw PermissionError escape as a 500.
    """
    import os

    from exozippy.gui import datafiles

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        if os.access(locked, os.R_OK):  # running as root: nothing to test
            pytest.skip("cannot make a directory unreadable as this user")
        with pytest.raises(ValueError, match="Cannot list directory"):
            datafiles.list_directory(str(locked))
    finally:
        locked.chmod(0o755)


def test_list_directory_tolerates_an_unstattable_child(tmp_path):
    """
    Given a dangling symlink among the entries,
    When the directory is listed,
    Then the rest still lists (the sort key used to be able to raise).
    """
    from exozippy.gui import datafiles

    (tmp_path / "real.rv").write_text("1 2 3\n")
    (tmp_path / "dangling").symlink_to(tmp_path / "gone")

    names = [
        e["name"] for e in datafiles.list_directory(str(tmp_path))["entries"]
    ]

    assert "real.rv" in names


def test_is_within_is_the_one_containment_predicate(tmp_path):
    """
    Given a path and a root,
    When containment is asked,
    Then the answer covers the same cases the two former copies did.

    There used to be two implementations of this security question -- one on
    realpath+commonpath in app.py, one on resolve+relative_to here -- with
    different symlink behavior depending on which endpoint you asked.
    """
    from exozippy.gui import datafiles

    inside = tmp_path / "a" / "b"
    inside.mkdir(parents=True)

    assert datafiles.is_within(inside, tmp_path)
    assert datafiles.is_within(tmp_path, tmp_path)
    assert not datafiles.is_within(tmp_path.parent, tmp_path)
    assert not datafiles.is_within(None, tmp_path)
    assert not datafiles.is_within(inside, None)
    # A sibling whose NAME merely prefixes the root is not inside it.
    sibling = tmp_path.parent / (tmp_path.name + "-other")
    assert not datafiles.is_within(sibling, tmp_path)
