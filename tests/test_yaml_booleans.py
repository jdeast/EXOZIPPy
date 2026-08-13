"""The shared YAML-1.1 boolean guard (``exozippy.yamlio``).

The fit reads the config/params files with PyYAML (YAML 1.1) and the GUI reads
the SAME files with ruamel's round-trip loader (YAML 1.2). The two disagree
about ``yes/no/on/off``, so ``finite_source: no`` is ``False`` to the fit and
the truthy string ``"no"`` to the GUI -- the GUI shows, and can write out, the
opposite of what the fit will do. The guard refuses those spellings on BOTH
paths; these tests pin the disagreement itself, the guard, and the fact that
no shipped example uses an ambiguous spelling.
"""

from pathlib import Path

import pytest
import yaml
from ruamel.yaml import YAML

from exozippy.yamlio import (
    ACCEPTED_BOOLEANS,
    AmbiguousBooleanError,
    check_yaml_booleans,
    load_yaml,
    load_yaml_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# The spellings PyYAML's YAML 1.1 resolver calls booleans and ruamel's YAML
# 1.2 resolver does not. This is the whole defect surface.
AMBIGUOUS = [
    "yes",
    "Yes",
    "YES",
    "no",
    "No",
    "NO",
    "on",
    "On",
    "ON",
    "off",
    "Off",
    "OFF",
]


def _ruamel_load(text):
    return YAML().load(text)


# ---------------------------------------------------------------------------
# The root cause, pinned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", AMBIGUOUS)
def test_ambiguous_spelling_means_different_things_to_the_two_loaders(
    spelling,
):
    """
    Given a plain scalar spelled yes/no/on/off,
    When the fit's loader (PyYAML, YAML 1.1) and the GUI's loader (ruamel,
      YAML 1.2) both read it,
    Then one produces a bool and the other a truthy string -- which is why
      the guard exists.
    """
    text = f"finite_source: {spelling}\n"

    from_fit = yaml.safe_load(text)["finite_source"]
    from_gui = _ruamel_load(text)["finite_source"]

    assert isinstance(from_fit, bool)
    assert isinstance(from_gui, str)
    # And for the false-ish half the disagreement inverts the meaning.
    if spelling.lower() in ("no", "off"):
        assert from_fit is False
        assert bool(from_gui) is True


@pytest.mark.parametrize("spelling", ACCEPTED_BOOLEANS)
def test_accepted_spellings_agree_in_both_loaders(spelling):
    """
    Given one of the accepted boolean spellings,
    When both loaders read it,
    Then both produce the same bool -- the accepted set is defined by what
      the loaders actually do, not by taste.
    """
    text = f"finite_source: {spelling}\n"

    from_fit = yaml.safe_load(text)["finite_source"]
    from_gui = _ruamel_load(text)["finite_source"]

    assert isinstance(from_fit, bool)
    assert bool(from_gui) is from_fit
    assert from_fit is (spelling.lower() == "true")


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", AMBIGUOUS)
def test_check_raises_naming_key_spelling_and_accepted_form(spelling):
    """
    Given a config spelling a boolean the YAML 1.1 way,
    When the shared guard checks it,
    Then it raises, naming the key, the offending spelling, and the
      accepted spellings.
    """
    text = f"lens:\n  - name: L\n    finite_source: {spelling}\n"

    with pytest.raises(AmbiguousBooleanError) as excinfo:
        check_yaml_booleans(text, source="demo.yaml")

    message = str(excinfo.value)
    assert "finite_source" in message
    assert spelling in message
    assert "true" in message and "false" in message
    assert "demo.yaml" in message


def test_error_names_every_offender_with_its_line():
    """
    Given several ambiguous booleans in one file,
    When the guard checks it,
    Then all of them are reported, each with its line number.
    """
    text = "a: no\nb: 1\nc:\n  d: ON\n"

    with pytest.raises(AmbiguousBooleanError) as excinfo:
        check_yaml_booleans(text)

    message = str(excinfo.value)
    assert "line 1: a: no" in message
    assert "line 4: c.d: ON" in message


def test_quoted_string_is_not_a_boolean():
    """
    Given a quoted value that looks like a YAML 1.1 boolean,
    When the guard checks it,
    Then it passes: a quoted scalar is a string to both loaders, so quoting
      is the documented escape hatch.
    """
    text = "sampler:\n  seed_polish: \"on\"\nname: 'no'\n"

    check_yaml_booleans(text)  # must not raise
    assert load_yaml_text(text)["sampler"]["seed_polish"] == "on"


def test_single_letter_y_n_are_untouched():
    """
    Given band/filter names like Y or N,
    When the guard checks them,
    Then nothing is flagged -- neither loader treats them as booleans, and
      flagging them would break real filter names.
    """
    check_yaml_booleans("band:\n  - filter: Y\n  - filter: N\n")


def test_accepted_booleans_load_through_the_shared_loader():
    """
    Given a config that spells its booleans the accepted way,
    When the shared loader reads it,
    Then it loads normally.
    """
    assert load_yaml_text("a: true\nb: FALSE\n") == {"a": True, "b": False}


# ---------------------------------------------------------------------------
# Both paths: the fit's loader and the GUI's document
# ---------------------------------------------------------------------------


def test_fit_path_refuses_the_file(tmp_path):
    """
    Given a config file on disk with `finite_source: no`,
    When the fit's shared loader reads it (what exozippy.cli calls),
    Then it raises instead of silently resolving the value to False.
    """
    path = tmp_path / "bad.yaml"
    path.write_text("lens:\n  - finite_source: no\n")

    with pytest.raises(AmbiguousBooleanError) as excinfo:
        load_yaml(path)

    assert "finite_source" in str(excinfo.value)


def test_gui_document_open_refuses_the_same_file(tmp_path):
    """
    Given the same config file,
    When the GUI opens it as a round-trip document,
    Then it raises the same error -- the GUI cannot accept a spelling the
      fit rejects, or vice versa.
    """
    from exozippy.gui.document import ProjectDocument

    path = tmp_path / "bad.yaml"
    path.write_text("lens:\n  - finite_source: no\n")

    with pytest.raises(AmbiguousBooleanError) as excinfo:
        ProjectDocument.open(path)

    assert "finite_source" in str(excinfo.value)


def test_gui_document_opens_an_accepted_file(tmp_path):
    """
    Given a config spelling its boolean `false`,
    When the GUI opens it,
    Then it loads and the value is a real boolean.
    """
    from exozippy.gui.document import ProjectDocument

    path = tmp_path / "ok.yaml"
    path.write_text("lens:\n  - finite_source: false\n")

    doc = ProjectDocument.open(path)

    assert bool(doc.config["lens"][0]["finite_source"]) is False


# ---------------------------------------------------------------------------
# Anti-drift
# ---------------------------------------------------------------------------


def test_boolean_option_keys_come_from_the_component_schemas():
    """
    Given components that declare boolean config keys as accepts [True, False],
    When introspect.boolean_option_keys() collects them,
    Then the declared keys are reported with their owning components -- the
      guard's message needs no hand-maintained list.
    """
    from exozippy.introspect import boolean_option_keys

    keys = boolean_option_keys()

    assert "lens" in keys.get("finite_source", [])
    assert "planet" in keys.get("chen", [])


def test_no_shipped_example_uses_an_ambiguous_boolean():
    """
    Given every YAML file shipped under examples/,
    When the guard checks it,
    Then none of them uses a YAML-1.1-only boolean spelling, so the guard is
      safe to be fatal.
    """
    offenders = []
    for path in sorted(REPO_ROOT.glob("examples/**/*.yaml")):
        try:
            check_yaml_booleans(path.read_text(), source=str(path))
        except AmbiguousBooleanError as exc:
            offenders.append(str(exc))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue  # not our problem here; the loader reports it in situ

    assert not offenders, "\n".join(offenders)
