"""Tests for the GUI project-document model (G8).

The document holds the system config and the params-override file as ruamel
round-trip trees. These tests pin the behaviour the GUI relies on: comment- and
order-preserving saves, schema-driven cross-reference rewriting on rename, exact
undo, and autosave sidecar lifecycle.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from exozippy.config import ConfigManager
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
    diffs = [(o, s) for o, s in zip(orig_lines, saved_lines) if o != s]
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
        {
            "op": "set_param_field",
            "args": {"path": "star.A.teff", "field": "initval", "value": 6100},
        }
    )
    doc.execute(cmd)
    assert doc.params["star.A.teff"]["initval"] == 6100


@pytest.fixture
def indexed_project(tmp_path):
    """A working copy of the RV-only KELT-4 example.

    Its shipped params file spells star 0 in the INDEX form (`star.0.teff`)
    while the config names that star "A" -- the combination the GUI writer
    used to break.  No data files are copied: ProjectDocument only reads the
    two YAML trees.
    """
    for name in ("kelt4_rvonly.yaml", "kelt4.params.yaml"):
        shutil.copy(EXAMPLE_DIR / name, tmp_path / name)
    return tmp_path


def test_slider_edit_updates_the_users_own_spelling(indexed_project):
    """
    Given a params file that spells an element in the index form
      (`star.0.teff`, carrying a `sigma` prior) while the config names that
      star "A",
    When a slider-style edit is committed against the GUI's name-form path
      `star.A.teff`,
    Then the existing entry is updated in place -- one spelling only, the
      user's own, with the rest of their entry intact.

    The GUI addresses parameters by the NAME form because that is what
    introspect and export_solution display, so a literal params[path] write
    APPENDED a twin.  Both spellings name the same element and neither is
    more specific, so nothing downstream could adjudicate them:
    standardize_param_names kept whichever key came last -- the GUI's, since
    it appends -- and silently discarded the whole original entry.  Here that
    is `sigma: 100` on the stellar effective temperature: a real prior, still
    visible in the saved file, no longer applied to the fit.
    """
    # Arrange
    doc = ProjectDocument.open(indexed_project / "kelt4_rvonly.yaml")
    assert "star.0.teff" in doc.params  # fixture precondition
    assert doc.params["star.0.teff"]["sigma"] == 100

    # Act
    doc.execute(SetParamField("star.A.teff", "initval", 6250))
    doc.save()

    # Assert -- in the tree...
    assert "star.A.teff" not in doc.params
    assert doc.params["star.0.teff"]["initval"] == 6250
    assert doc.params["star.0.teff"]["sigma"] == 100

    # ...and on disk, where exactly one spelling of the element survives.
    saved = yaml.safe_load((indexed_project / "kelt4.params.yaml").read_text())
    assert [k for k in saved if k.endswith(".teff")] == ["star.0.teff"]
    assert saved["star.0.teff"] == {"initval": 6250, "sigma": 100}

    # ...and the result is a config ConfigManager will accept: naming one
    # element twice is a fatal error, so an appended twin would fail the very
    # next fit rather than merely losing the prior.
    config = yaml.safe_load(
        (indexed_project / "kelt4_rvonly.yaml").read_text()
    )
    ConfigManager(saved, system_config=config)


def test_slider_edit_keeps_the_name_form_when_the_file_uses_it(project):
    """
    Given a params file that already spells the element in the NAME form,
    When the same edit is committed,
    Then that entry is updated in place and no index-form twin appears.

    The other half of "preserve the user's spelling": resolving to an
    existing key must not canonicalize name-form files into index form.
    """
    # Arrange
    doc = _open(project)
    assert "star.A.teff" in doc.params  # fixture precondition

    # Act
    doc.execute(SetParamField("star.A.teff", "initval", 6250))

    # Assert
    assert doc.params["star.A.teff"]["initval"] == 6250
    assert "star.0.teff" not in doc.params


def test_slider_edit_does_not_consume_a_broadcast_entry(tmp_path):
    """
    Given a params file whose only entry for a parameter is the 2-part
      BROADCAST spelling,
    When one element is edited by name,
    Then a new specific entry is created and the broadcast entry is left
      alone.

    Broadcast + specific is the legitimate "set every element, refine one"
    idiom, not a duplicate -- so the writer must not treat the broadcast
    entry as the element's existing key and overwrite everyone else's value.
    """
    # Arrange
    config_path = tmp_path / "sys.yaml"
    config_path.write_text(
        "star:\n  - name: A\n  - name: B\nparameter_file: p.params.yaml\n"
    )
    (tmp_path / "p.params.yaml").write_text("star.teff:\n  initval: 5800\n")
    doc = ProjectDocument.open(config_path)

    # Act
    doc.execute(SetParamField("star.A.teff", "initval", 6250))

    # Assert
    assert doc.params["star.teff"]["initval"] == 5800
    assert doc.params["star.A.teff"]["initval"] == 6250


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


# --- structural edits vs. the index spelling (review 11.10) -------------------
#
# An element has three legal spellings and only ONE of them survives a change
# to the instance list: the NAME form.  `star.A.teff` follows its star wherever
# the list moves it; `star.0.teff` follows the INDEX, so deleting an earlier
# instance silently repoints it at a different body.  These tests pin what the
# two structural commands do about that.


def _three_stars(tmp_path, params_text):
    """A three-star project (A, B, C) with a hand-written params file."""
    config_path = tmp_path / "sys.yaml"
    config_path.write_text(
        "star:\n"
        "  - name: A\n"
        "  - name: B\n"
        "  - name: C\n"
        "parameter_file: p.params.yaml\n"
    )
    (tmp_path / "p.params.yaml").write_text(params_text)
    return config_path


def _teff_by_star_name(doc):
    """{star name: initval} as the FIT would read the saved params file."""
    config = yaml.safe_load(doc.config_text())
    params = yaml.safe_load(doc.params_text())
    names = [e["name"] for e in config["star"]]
    out = {}
    for key, entry in params.items():
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "star" or parts[2] != "teff":
            continue
        instance = parts[1]
        if instance.isdigit():
            i = int(instance)
            name = names[i] if i < len(names) else f"<no star at index {i}>"
        else:
            name = instance
        out[name] = entry["initval"]
    return out


def test_delete_instance_does_not_leave_index_keys_on_another_star(tmp_path):
    """
    Given a params file that spells all three stars by INDEX
      (`star.0/1/2.teff`),
    When the first star is deleted,
    Then no surviving entry addresses a different star than it did before --
      B still carries 6000 and C still carries 7000, and nothing carries the
      deleted star's 5000.

    This is the wrong-element half of review 11.10.  The old name-prefix scan
    saw no `star.A.` key at all, so it deleted NOTHING: `star.0.teff` (the
    deleted star's own entry, 5000) stayed in the file and, with A gone,
    addressed B; `star.1.teff` addressed C; `star.2.teff` addressed nobody.
    Every value silently moved one star over.
    """
    # Arrange
    config_path = _three_stars(
        tmp_path,
        "star.0.teff:\n  initval: 5000\n"
        "star.1.teff:\n  initval: 6000\n"
        "star.2.teff:\n  initval: 7000\n",
    )
    doc = ProjectDocument.open(config_path)

    # Act
    doc.execute(DeleteInstance("star", "A"))

    # Assert -- each star still carries the value it carried before.
    assert _teff_by_star_name(doc) == {"B": 6000, "C": 7000}
    assert 5000 not in _teff_by_star_name(doc).values()

    # ...spelled in the one form that survives a list edit...
    assert set(doc.params) == {"star.B.teff", "star.C.teff"}

    # ...and the result is a params file ConfigManager accepts: a rewrite
    # that produced two spellings of one element would be a fatal error.
    ConfigManager(
        yaml.safe_load(doc.params_text()),
        system_config=yaml.safe_load(doc.config_text()),
    )


def test_delete_instance_drops_both_spellings_of_its_own_entries(tmp_path):
    """
    Given a params file naming the doomed star by NAME for one parameter and
      by INDEX for another,
    When that star is deleted,
    Then both entries go.

    `star.A.teff` and `star.0.teff` are two spellings of one element, so a
    delete that honours only the first leaves a live entry behind.
    """
    # Arrange
    config_path = _three_stars(
        tmp_path,
        "star.A.teff:\n  initval: 5000\n"
        "star.0.radius:\n  initval: 0.9\n"
        "star.B.teff:\n  initval: 6000\n",
    )
    doc = ProjectDocument.open(config_path)

    # Act
    doc.execute(DeleteInstance("star", "A"))

    # Assert
    assert set(doc.params) == {"star.B.teff"}


def test_delete_instance_leaves_the_broadcast_form_alone(tmp_path):
    """
    Given a params file mixing all three spellings -- a 2-part BROADCAST
      entry, a name-form entry and index-form entries on either side of the
      doomed star,
    When the first star is deleted,
    Then the broadcast entry is untouched, entries that already spelled their
      own element correctly are byte-identical, and only the shifted
      index-form entry is respelled.

    The broadcast form is not specific to any instance: it covers whatever
    elements no specific entry claims, which is exactly what it still says
    with one fewer star.
    """
    # Arrange
    config_path = _three_stars(
        tmp_path,
        "star.teff:\n  initval: 5800\n"  # broadcast: every star
        "star.A.radius:\n  initval: 0.9\n"  # doomed star, name form
        "star.0.mass:\n  initval: 0.8\n"  # doomed star, index form
        "star.2.teff:\n  initval: 7000\n"  # C, index form -- SHIFTS
        "star.B.radius:\n  initval: 1.1\n",  # B, name form -- stable
    )
    doc = ProjectDocument.open(config_path)

    # Act
    doc.execute(DeleteInstance("star", "A"))

    # Assert
    assert dict(doc.params["star.teff"]) == {"initval": 5800}
    assert set(doc.params) == {"star.teff", "star.C.teff", "star.B.radius"}
    assert doc.params["star.C.teff"]["initval"] == 7000
    assert doc.params["star.B.radius"]["initval"] == 1.1
    ConfigManager(
        yaml.safe_load(doc.params_text()),
        system_config=yaml.safe_load(doc.config_text()),
    )


def test_delete_instance_keeps_index_keys_that_do_not_shift(tmp_path):
    """
    Given index-form entries BEFORE and AFTER the doomed star,
    When the star is deleted,
    Then only the one whose index shifts is respelled; the earlier one keeps
      the user's own spelling.

    The minimum intervention that guarantees correctness: an index below the
    deleted one still means the same element, so rewriting it would churn the
    user's file for nothing.
    """
    # Arrange
    config_path = _three_stars(
        tmp_path,
        "star.0.teff:\n  initval: 5000\n"
        "star.1.teff:\n  initval: 6000\n"
        "star.2.teff:\n  initval: 7000\n",
    )
    doc = ProjectDocument.open(config_path)

    # Act -- delete the MIDDLE star
    doc.execute(DeleteInstance("star", "B"))

    # Assert
    assert set(doc.params) == {"star.0.teff", "star.C.teff"}
    assert _teff_by_star_name(doc) == {"A": 5000, "C": 7000}


def test_delete_instance_reindexes_when_a_survivor_has_no_name(tmp_path):
    """
    Given a survivor with no `name:` key at all,
    When an earlier instance is deleted,
    Then its index-form entry is RE-INDEXED, since there is no name form to
      rewrite it to.

    Same guarantee (the entry still addresses the same instance), by the only
    means available.
    """
    # Arrange
    config_path = tmp_path / "sys.yaml"
    config_path.write_text(
        "star:\n  - name: A\n  - teff: 6000\nparameter_file: p.params.yaml\n"
    )
    (tmp_path / "p.params.yaml").write_text("star.1.radius:\n  initval: 1.1\n")
    doc = ProjectDocument.open(config_path)

    # Act
    doc.execute(DeleteInstance("star", "A"))

    # Assert
    assert set(doc.params) == {"star.0.radius"}
    assert doc.params["star.0.radius"]["initval"] == 1.1


def test_delete_instance_retargets_index_form_links(tmp_path):
    """
    Given a link expression that references a surviving star by INDEX,
    When an earlier star is deleted,
    Then the reference is respelled to that star's name.

    A link VALUE is a parameter reference too: `star.2.teff` in an expression
    retargets on a deletion exactly as a key does, and just as silently.
    """
    # Arrange
    config_path = _three_stars(
        tmp_path,
        "star.B.teff:\n  initval: star.2.teff\n  sigma: 0\n",
    )
    doc = ProjectDocument.open(config_path)

    # Act
    doc.execute(DeleteInstance("star", "A"))

    # Assert
    assert doc.params["star.B.teff"]["initval"] == "star.C.teff"


def test_duplicate_instance_copies_index_form_entries(tmp_path):
    """
    Given a params file that spells an instance's entries by INDEX,
    When that instance is duplicated,
    Then the clone carries copies of them, under its own NAME.

    The mirror of the delete bug: the old name-prefix scan matched no
    `star.A.` key, so the clone silently arrived with none of the parameters
    it was supposed to be a copy of.  The clone's entries are written in the
    name form -- never its index form, which is exactly the spelling the
    delete path exists to repair -- and one element ends up with exactly one
    spelling, since naming it twice is fatal in ConfigManager.
    """
    # Arrange
    config_path = _three_stars(
        tmp_path,
        "star.0.teff:\n  initval: 5000\n  sigma: 100\n",
    )
    doc = ProjectDocument.open(config_path)

    # Act
    doc.execute(DuplicateInstance("star", "A", "A2"))

    # Assert
    assert set(doc.params) == {"star.0.teff", "star.A2.teff"}
    assert dict(doc.params["star.A2.teff"]) == {"initval": 5000, "sigma": 100}
    # The copy is deep: editing the clone must not touch the original.
    doc.execute(SetParamField("star.A2.teff", "initval", 9999))
    assert doc.params["star.0.teff"]["initval"] == 5000
    ConfigManager(
        yaml.safe_load(doc.params_text()),
        system_config=yaml.safe_load(doc.config_text()),
    )


def test_structural_edit_refuses_a_pre_existing_duplicate_spelling(tmp_path):
    """
    Given a params file that already names one element under BOTH specific
      spellings (which ConfigManager itself refuses),
    When a delete would merge them onto one key,
    Then the command raises and the document is left exactly as it was.

    Silently keeping one of the two would bury a fault that carries a value,
    a bound or a prior.  The restore is `Command.apply`'s: a command that
    raises is not on the undo stack, so it must leave nothing behind.
    """
    # Arrange
    config_path = _three_stars(
        tmp_path,
        "star.1.teff:\n  initval: 6000\n"
        "star.B.teff:\n  initval: 6100\n  sigma: 50\n",
    )
    doc = ProjectDocument.open(config_path)
    before = doc.params_text()

    # Act / Assert
    with pytest.raises(ValueError, match="two spellings"):
        doc.execute(DeleteInstance("star", "A"))
    assert doc.params_text() == before
    assert [e["name"] for e in doc.config["star"]] == ["A", "B", "C"]


# --- emptying the params tree (review 1.12.1) --------------------------------


def _blank_every_override(doc):
    """Delete every params entry the way the UI does: one blanked field at a
    time, through the same command the form's onBlur issues."""
    for key in list(doc.params.keys()):
        for field in list(doc.params[key].keys()):
            doc.execute(SetParamField(key, field, None))


def test_saving_an_emptied_params_tree_rewrites_the_file(project):
    """
    Given a params file whose every override is deleted through the GUI,
    When the document is saved,
    Then the file on disk no longer carries them.

    The write used to be gated on ``len(self.params) > 0``, so removing the
    LAST override -- "Reset to solved" on the only tuned parameter, or blanking
    its fields -- left the deleted entry, ``sigma`` prior included, sitting on
    disk while the GUI reported the document clean.  RunControl then saved and
    launched, and the fit read exactly the entry the user had just removed,
    with no cue anywhere.
    """
    # Arrange
    doc = _open(project)
    params_path = project / PARAMS_NAME
    assert yaml.safe_load(params_path.read_text())  # non-empty to begin with

    # Act
    _blank_every_override(doc)
    assert len(doc.params) == 0
    doc.save()

    # Assert
    assert not (yaml.safe_load(params_path.read_text()) or {})
    assert doc.dirty is False


def test_saving_an_empty_tree_does_not_create_a_params_file(tmp_path):
    """
    Given a document whose params file has never existed,
    When it is saved with an empty params tree,
    Then no params file is created.

    The complement of the fix above: "write the empty tree" must mean "keep the
    on-disk file honest", not "materialize a file nobody asked for".
    """
    config_path = tmp_path / "sys.yaml"
    config_path.write_text("prefix: out\nparameter_file: sys.params.yaml\n")

    doc = ProjectDocument.open(config_path)
    doc.save()

    assert not (tmp_path / "sys.params.yaml").exists()


def test_autosave_of_an_emptied_params_tree_rewrites_the_sidecar(project):
    """
    Given a document whose params overrides were all deleted,
    When it autosaves,
    Then the sidecar reflects the emptied tree rather than being skipped.
    """
    doc = _open(project)
    _blank_every_override(doc)

    written = doc.autosave()

    params_sidecars = [p for p in written if "params" in p.name]
    assert params_sidecars
    for path in params_sidecars:
        assert not (yaml.safe_load(path.read_text()) or {})


# --- settable param fields (review 1.12.3) -----------------------------------


def _frontend_param_fields():
    """The field list ConfigTab.tsx renders a column for.

    Parsed out of the frontend source because the frontend has no test harness
    of its own; a server-side assertion that ``set_param_field`` accepts every
    advertised field is what stops the two drifting.
    """
    import re

    source = (
        Path(__file__).resolve().parents[1]
        / "gui"
        / "frontend"
        / "src"
        / "components"
        / "ConfigTab.tsx"
    ).read_text()
    match = re.search(r"const PARAM_FIELDS = \[(.*?)\];", source, re.S)
    assert match, "ConfigTab.tsx no longer declares a PARAM_FIELDS literal"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_every_field_the_config_tab_offers_is_settable(project):
    """
    Given the parameter columns ConfigTab renders,
    When each is set through the document command,
    Then none is rejected.

    ``bound_scale`` was rendered and rejected: ``_PARAM_FIELDS`` was defined as
    ``set(LINKABLE_FIELDS)``, and ``bound_scale`` is deliberately NOT linkable,
    so every blur in that column 400'd and the one user-facing scale knob could
    not be set from the GUI at all.
    """
    doc = _open(project)
    fields = _frontend_param_fields()
    assert "bound_scale" in fields  # the column that regressed

    for field in fields:
        doc.execute(SetParamField("star.A.radius", field, 1.25))

    assert set(fields) <= set(doc.params["star.A.radius"].keys())


def test_bound_scale_is_settable_but_not_linkable():
    """
    Given the two field sets,
    When they are compared,
    Then bound_scale is settable and NOT linkable -- they are decoupled.
    """
    from exozippy.gui.document import _PARAM_FIELDS
    from exozippy.linking import LINKABLE_FIELDS

    assert "bound_scale" in _PARAM_FIELDS
    assert "bound_scale" not in LINKABLE_FIELDS
    assert set(LINKABLE_FIELDS) < _PARAM_FIELDS


# --- instance names (review 2.12.7) ------------------------------------------


@pytest.mark.parametrize("bad", ["0", "12", "", "  ", "a.b", "has space"])
def test_rename_refuses_a_name_that_breaks_path_parsing(bad):
    """
    Given an instance rename to a name the fit would reject,
    When the command is built,
    Then it raises instead of writing an unusable config.

    The all-digit case is the dangerous one: renaming star "A" to "0" rewrote
    ``star.A.teff`` into ``star.0.teff``, which every later command -- and the
    fit -- reads as the INDEX form, potentially addressing a different star.
    """
    with pytest.raises(ValueError):
        RenameInstance("star", "A", bad)


@pytest.mark.parametrize("bad", ["0", "", "a.b"])
def test_add_and_duplicate_refuse_the_same_names(bad):
    """
    Given an add or duplicate under a name that breaks path parsing,
    When the command is built,
    Then it raises -- all three creation paths agree.
    """
    with pytest.raises(ValueError):
        AddComponentInstance("star", bad)
    with pytest.raises(ValueError):
        DuplicateInstance("star", "A", bad)


def test_valid_names_are_still_accepted(project):
    """Given ordinary names, When used, Then nothing is refused."""
    doc = _open(project)
    doc.execute(AddComponentInstance("star", "Companion-2"))
    doc.execute(RenameInstance("star", "Companion-2", "star_2"))

    assert "star_2" in [e["name"] for e in doc.config["star"]]


# --- autosave recovery (review 1.12.10) --------------------------------------


def test_restore_autosave_loads_the_sidecar_back_and_is_undoable(project):
    """
    Given a document with an autosave sidecar holding unsaved edits,
    When the restore_autosave command runs,
    Then the edits are back in the document -- and undo takes them away again.

    The server has always REPORTED recoverable sidecars; nothing could act on
    them, so the edits were reachable only by finding hidden dotfiles by hand.
    """
    # Arrange: make an edit, flush it to the sidecar, then throw the document
    # away (what a project switch does) and re-open from disk.
    doc = _open(project)
    doc.execute(SetParamField("star.A.radius", "initval", 1.999))
    doc.autosave()
    reopened = _open(project)
    assert reopened.autosave_recovery()
    assert float(reopened.params["star.A.radius"]["initval"]) != 1.999

    # Act
    reopened.execute(command_from_json({"op": "restore_autosave", "args": {}}))

    # Assert
    assert float(reopened.params["star.A.radius"]["initval"]) == 1.999
    reopened.undo()
    assert float(reopened.params["star.A.radius"]["initval"]) != 1.999


def test_restore_autosave_without_a_sidecar_raises(project):
    """Given no sidecar, When restore runs, Then it says so rather than
    silently doing nothing."""
    doc = _open(project)

    with pytest.raises(ValueError, match="no autosaved edits"):
        doc.execute(command_from_json({"op": "restore_autosave", "args": {}}))
