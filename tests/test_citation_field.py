"""The params-file ``citation`` field: what justifies a prior, carried from
the params file (or mkticsed) into the parameter table's notes and through
mkparam restart files.

Each citation is a references.bib key, cited with \\citet, or free text
("email from XX 9/9/2026") passed through verbatim for the user to replace
by hand when writing the paper.
"""

import re
from pathlib import Path

import astropy.units as u
import pytest

from exozippy.components.parameter import Parameter
from exozippy.config import ConfigManager, normalize_citation
from exozippy.mkparam import _apply_existing_constraints
from exozippy.outputs.texutils import known_bib_keys

MKTICSED = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "exozippy"
    / "utilities"
    / "mkticsed.py"
)


# --- normalization at the params-file boundary --------------------------------


def test_a_string_is_one_citation_and_is_never_split():
    """
    Given free text carrying commas,
    When it is normalized,
    Then it stays ONE citation (a string is never parsed).
    """
    # Act / Assert
    assert normalize_citation("Smith, Jones & Lee 2023", "p") == (
        "Smith, Jones & Lee 2023",
    )


def test_a_list_is_several_citations():
    """
    Given a list of citations,
    When it is normalized,
    Then each item is one citation, stripped.
    """
    # Act / Assert
    assert normalize_citation([" Lindegren:2021", "email 9/9/2026 "], "p") == (
        "Lindegren:2021",
        "email 9/9/2026",
    )


@pytest.mark.parametrize("bad", [3.0, ["ok", 3], [], [""], "  "])
def test_a_malformed_citation_raises_naming_the_parameter(bad):
    """
    Given a citation that is not a string or a list of non-empty strings,
    When it is normalized,
    Then it raises, naming the parameter path.
    """
    # Act / Assert
    with pytest.raises(ValueError, match="star.A.av"):
        normalize_citation(bad, "star.A.av")


# --- resolve() carries it per element ----------------------------------------


def test_resolve_carries_a_citation_per_element():
    """
    Given two stars, only one of which cites its prior,
    When star.distance is resolved,
    Then the citation lands on that element only.
    """
    # Arrange
    cm = ConfigManager(
        {"star.B.distance": {"mu": 100.0, "sigma": 1.0, "citation": "a note"}},
        system_config={"star": [{"name": "A"}, {"name": "B"}]},
    )

    # Act
    resolved = cm.resolve("star", "distance", shape=(2,))

    # Assert
    assert resolved["citation"] == [(), ("a note",)]


def test_a_citation_on_a_start_value_only_entry_is_ignored_with_a_warning(
    caplog,
):
    """
    Given a citation on an entry that states only an initval (a start value,
    not a prior),
    When star.distance is resolved,
    Then the citation is dropped with a warning naming the entry -- it would
    otherwise print "Prior from ..." against the defaults' uniform prior.
    """
    # Arrange
    cm = ConfigManager(
        {"star.A.distance": {"initval": 100.0, "citation": "Smith 2020"}},
        system_config={"star": [{"name": "A"}]},
    )

    # Act
    with caplog.at_level("WARNING"):
        resolved = cm.resolve("star", "distance", shape=(1,))

    # Assert
    assert resolved["citation"] == ()
    assert "'citation' ignored" in caplog.text


# --- rendered as a Prior-column table note ------------------------------------


def _gaussian_param(citation):
    """A scalar Parameter with a Gaussian prior and the given citation."""
    return Parameter(
        label="star.A.parallax",
        unit=u.mas,
        internal_unit=u.mas,
        initval=4.6,
        mu=4.6,
        sigma=0.02,
        citation=citation,
    )


def test_a_known_bib_key_is_cited_in_a_prior_note():
    """
    Given a prior citing keys that exist in references.bib,
    When its Prior-column cell is built,
    Then the cell keeps the Gaussian and one note cites every key.
    """
    # Arrange
    p = _gaussian_param(("Lindegren:2021", "ElBadry:2021"))

    # Act
    cell, notes = p.prior_cell_and_notes(0)

    # Assert
    assert r"\mathcal{N}" in cell
    assert notes == [r"Prior from \citet{Lindegren:2021,ElBadry:2021}"]


def test_free_text_is_passed_through_escaped_not_cited():
    """
    Given a citation that is not a references.bib key,
    When its note is built,
    Then the text is passed through LaTeX-escaped, never inside a \\citet
    (which would print "?"), after any known keys.
    """
    # Arrange
    p = _gaussian_param(("Stassun:2019", "email from XX_YY 9/9/2026"))

    # Act
    _, notes = p.prior_cell_and_notes(0)

    # Assert
    assert notes == [
        r"Prior from \citet{Stassun:2019}; email from XX\_YY 9/9/2026"
    ]


def test_no_citation_means_no_note():
    """
    Given a prior with no citation,
    When its cell is built,
    Then no note is added.
    """
    # Act
    _, notes = _gaussian_param(()).prior_cell_and_notes(0)

    # Assert
    assert notes == []


# --- carried through restart files --------------------------------------------


def test_mkparam_keeps_a_prior_and_its_citation_together():
    """
    Given an existing params entry with a prior and its citation,
    When mkparam layers it onto a fresh restart entry,
    Then the citation travels with the prior.
    """
    # Arrange
    existing = {"mu": 0.1, "sigma": 0.08, "citation": ["Stassun:2019"]}

    # Act
    entry = _apply_existing_constraints({"initval": 0.12}, existing)

    # Assert
    assert entry["citation"] == ["Stassun:2019"]
    assert entry["initval"] == 0.12


def test_a_restart_file_keeps_every_citation_and_they_resolve_again(tmp_path):
    """
    Given a params file citing a sampled Gaussian prior (free text with a
    comma), a sampled bound-only prior (a bib key) and a DERIVED parameter's
    prior (a list) -- the two derived and sampled copy paths in mkparam --
    When mkparam writes a restart file from a trace,
    Then each citation is written back unchanged, and reading the restart
    file resolves them to the same citations as the original.
    """
    # Arrange
    import yaml
    from test_mkparam import _make_idata

    from exozippy.mkparam import write_param_file

    existing = {
        "star.Host.teff": {
            "mu": 5800.0,
            "sigma": 100.0,
            "citation": "spectroscopy, email from XX 9/9/2026",
        },
        "star.Host.av": {"upper": 0.1, "citation": "Schlegel:1998"},
        "star.Host.parallax": {
            "mu": 7.45,
            "sigma": 0.02,
            "citation": ["GaiaCollaboration:2023", "ElBadry:2021"],
        },
    }
    (tmp_path / "star.params.yaml").write_text(yaml.safe_dump(existing))
    trace = _make_idata(
        {"star.teff": 5750.0, "star.av": 0.05, "star.parallax": 7.44},
        tmpdir=tmp_path,
        derived_vars={"star.parallax"},
    )
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    # Act
    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "restart.yaml",
    )
    restart = yaml.safe_load(open(out))

    # Assert -- written back unchanged, on both copy paths
    for key, entry in existing.items():
        assert restart[key]["citation"] == entry["citation"], key

    # ... and they resolve again exactly as the original did
    def _resolved(params):
        cm = ConfigManager(params, system_config={"star": [{"name": "Host"}]})
        return {
            p: cm.resolve("star", p, shape=(1,))["citation"]
            for p in ("teff", "av", "parallax")
        }

    assert _resolved(restart) == _resolved(existing)


# --- the GUI document ----------------------------------------------------------


def test_blanking_a_cited_priors_last_field_in_the_gui_removes_the_entry(
    tmp_path,
):
    """
    Given a params entry with a sigma prior and its citation,
    When the GUI blanks the sigma (its table has no citation column),
    Then the entry is removed -- a citation left alone would justify nothing
    and survive in the saved file as a dangling, ignored field.
    """
    # Arrange
    pytest.importorskip("ruamel.yaml")
    from exozippy.gui.document import ProjectDocument, SetParamField

    (tmp_path / "fit.yaml").write_text(
        'star:\n  - name: "A"\nparameter_file: "fit.params.yaml"\n'
    )
    (tmp_path / "fit.params.yaml").write_text(
        "star.A.feh:\n  sigma: 0.08\n  mu: 0.27\n  citation: email 9/9/2026\n"
    )
    doc = ProjectDocument.open(tmp_path / "fit.yaml")

    # Act
    doc.execute(SetParamField("star.A.feh", "mu", None))
    doc.execute(SetParamField("star.A.feh", "sigma", None))

    # Assert
    assert "star.A.feh" not in doc.params


# --- mkticsed only writes keys the bibliography has ---------------------------


def test_every_key_mkticsed_can_write_exists_in_references_bib():
    """
    Given every Author:Year key literal in mkticsed.py,
    When checked against the shipped references.bib,
    Then each one exists -- so no generated note can print "?".
    """
    # Arrange
    keys = set(
        re.findall(r"[\"']([A-Z][A-Za-z]+:\d{4})[\"']", MKTICSED.read_text())
    )

    # Act
    missing = sorted(keys - known_bib_keys())

    # Assert
    assert keys, "the scan found no citation keys at all"
    assert missing == []
