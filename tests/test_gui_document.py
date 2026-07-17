"""Tests for the GUI project-document model (G8).

The document holds the system config and the params-override file as ruamel
round-trip trees. These tests pin the behaviour the GUI relies on: comment- and
order-preserving saves, schema-driven cross-reference rewriting on rename, exact
undo, and autosave sidecar lifecycle.
"""

import shutil
from pathlib import Path

import pytest

from exozippy.gui.document import (
    AddComponentInstance,
    DeleteInstance,
    DuplicateInstance,
    ProjectDocument,
    RenameInstance,
    SetParamField,
    command_from_json,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "kelt4"
CONFIG_NAME = "kelt4_rv+transit+sed.yaml"
PARAMS_NAME = "kelt4_rv+transit+sed.params.yaml"


@pytest.fixture
def project(tmp_path):
    """A working copy of the KELT-4 rv+transit+sed example in a temp dir."""
    for name in (
        CONFIG_NAME,
        PARAMS_NAME,
        "kelt4.sed.yaml",
    ):
        shutil.copy(EXAMPLE_DIR / name, tmp_path / name)
    return tmp_path


def _open(project_dir):
    return ProjectDocument.open(project_dir / CONFIG_NAME)


def _norm(text):
    """Split into lines with trailing whitespace stripped (EOL-agnostic)."""
    return [line.rstrip() for line in text.splitlines()]


def test_set_value_preserves_comments_and_order(project):
    """Given the example config, When one param value is set and saved, Then
    every line except the edited value is byte-identical (comments, order)."""
    # Arrange
    doc = _open(project)
    original = (project / PARAMS_NAME).read_text()

    # Act
    doc.execute(SetParamField("star.A.radius", "initval", 1.999))
    doc.save()
    saved = (project / PARAMS_NAME).read_text()

    # Assert -- the only differing line is the one we edited.
    orig_lines = _norm(original)
    saved_lines = _norm(saved)
    diffs = [
        (o, s) for o, s in zip(orig_lines, saved_lines) if o != s
    ]
    assert len(orig_lines) == len(saved_lines)
    assert diffs == [("  initval: 1.610", "  initval: 1.999")]


def test_config_roundtrips_unedited(project):
    """Given the example config, When saved without edits, Then the file is
    unchanged modulo trailing whitespace (round-trip fidelity)."""
    # Arrange
    doc = _open(project)
    original = (project / CONFIG_NAME).read_text()

    # Act
    doc.save()
    saved = (project / CONFIG_NAME).read_text()

    # Assert
    assert _norm(saved) == _norm(original)


def test_rename_rewrites_orbit_body_lists_and_links(project):
    """Given a star referenced by an orbit body list and by link expressions,
    When the star is renamed, Then both the body list and the link
    expressions are rewritten to the new name."""
    # Arrange
    doc = _open(project)
    assert doc.config["orbit"][0]["primary"][0] == "A"
    assert doc.params["star.B.distance"]["initval"] == "star.A.distance"

    # Act
    doc.execute(RenameInstance("star", "A", "Alpha"))

    # Assert -- system-config cross-reference (orbit primary body group)
    assert doc.config["orbit"][0]["primary"][0] == "Alpha"
    # the star's own name field
    assert doc.config["star"][0]["name"] == "Alpha"
    # torres star: ref (schema-declared) rewritten too
    assert doc.config["torres"][0]["star"] == "Alpha"
    # params keys keyed by the star were renamed
    assert "star.Alpha.radius" in doc.params
    assert "star.A.radius" not in doc.params
    # link expressions inside params values were rewritten
    assert doc.params["star.B.distance"]["initval"] == "star.Alpha.distance"
    assert doc.params["star.C.av"]["initval"] == "star.Alpha.av"


def test_rename_leaves_unrelated_bare_names_alone(project):
    """Given a planet 'b' and an orbit 'b', When the star 'A' is renamed,
    Then no unrelated bare name is touched (schema scoping)."""
    # Arrange
    doc = _open(project)

    # Act
    doc.execute(RenameInstance("star", "A", "Alpha"))

    # Assert -- planet b and orbit b untouched; companion body list untouched.
    assert doc.config["planet"][0]["name"] == "b"
    assert doc.config["orbit"][0]["name"] == "b"
    assert doc.config["orbit"][0]["companion"][0] == "b"


def test_undo_restores_exact_prior_tree(project):
    """Given a rename that touches many places, When undone, Then both trees
    are byte-identical to before the edit."""
    # Arrange
    doc = _open(project)
    before_config = doc.config_text()
    before_params = doc.params_text()

    # Act
    doc.execute(RenameInstance("star", "A", "Alpha"))
    assert doc.config_text() != before_config  # sanity: the edit did something
    doc.undo()

    # Assert
    assert doc.config_text() == before_config
    assert doc.params_text() == before_params


def test_redo_reapplies(project):
    """Given an undone edit, When redone, Then the edited state returns."""
    doc = _open(project)
    doc.execute(SetParamField("star.A.teff", "initval", 6300))
    after = doc.params_text()
    doc.undo()
    doc.redo()
    assert doc.params_text() == after


def test_autosave_sidecar_appears_and_is_cleaned_on_save(project):
    """Given a dirty document, When autosaved then saved, Then the sidecar
    appears on autosave and is removed on real save."""
    # Arrange
    doc = _open(project)
    doc.execute(SetParamField("star.A.teff", "initval", 6300))

    # Act -- autosave writes sidecars
    written = doc.autosave()

    # Assert -- sidecars exist and are named .<stem>.autosave.yaml
    assert written
    for sidecar in written:
        assert sidecar.exists()
        assert sidecar.name.endswith(".autosave.yaml")

    # Act -- a real save clears them
    doc.save()

    # Assert
    for sidecar in written:
        assert not sidecar.exists()
    assert doc.dirty is False


def test_autosave_noop_when_clean(project):
    """Given a clean document, When autosave is called, Then nothing writes."""
    doc = _open(project)
    assert doc.autosave() == []


def test_autosave_recovery_detects_newer_sidecar(project):
    """Given an autosave sidecar newer than the real file, When recovery is
    queried, Then the file is reported as recoverable."""
    doc = _open(project)
    doc.execute(SetParamField("star.A.teff", "initval", 6300))
    doc.autosave()
    recovery = doc.autosave_recovery()
    assert any(r["file"].endswith(PARAMS_NAME) for r in recovery)


def test_add_and_delete_instance(project):
    """Given the config, When a band is added then deleted, Then the block
    reflects the change and params keys for a deleted instance are dropped."""
    # Arrange
    doc = _open(project)
    n_bands = len(doc.config["band"])

    # Act -- add
    doc.execute(AddComponentInstance("band", "Sloani", {"filter": "Sloani"}))
    assert len(doc.config["band"]) == n_bands + 1
    assert doc.config["band"][-1]["name"] == "Sloani"

    # Act -- delete an rvinstrument and confirm its params keys go away
    assert any(k.startswith("rvinstrument.TRES.") for k in doc.params)
    doc.execute(DeleteInstance("rvinstrument", "TRES"))

    # Assert
    assert all(e["name"] != "TRES" for e in doc.config["rvinstrument"])
    assert not any(k.startswith("rvinstrument.TRES.") for k in doc.params)


def test_duplicate_instance_copies_block_and_params(project):
    """Given an rvinstrument with params, When duplicated, Then a new block
    entry and copied params keys appear under the new name."""
    doc = _open(project)
    doc.execute(DuplicateInstance("rvinstrument", "FIES", "FIES2"))
    assert any(e["name"] == "FIES2" for e in doc.config["rvinstrument"])
    assert "rvinstrument.FIES2.gamma" in doc.params
    # original untouched
    assert "rvinstrument.FIES.gamma" in doc.params


def test_command_from_json_dispatch(project):
    """Given a JSON command payload, When dispatched, Then it produces the
    matching Command and applies."""
    doc = _open(project)
    cmd = command_from_json(
        {"op": "set_param_field",
         "args": {"path": "star.A.teff", "field": "initval", "value": 6100}}
    )
    doc.execute(cmd)
    assert doc.params["star.A.teff"]["initval"] == 6100


def test_to_json_shape(project):
    """Given a document, When serialized, Then it exposes trees + undo depth."""
    doc = _open(project)
    doc.execute(SetParamField("star.A.teff", "initval", 6300))
    blob = doc.to_json()
    assert blob["dirty"] is True
    assert blob["undo_depth"] == 1
    assert blob["config"]["star"][0]["name"] == "A"
    # booleans survive as real JSON booleans
    assert blob["config"]["star"][0]["mist"] is False
