"""Tests for exozippy.introspect: static, JSON-serializable component schemas.

These tests exercise the introspection layer that a GUI, documentation
generator, or scripting helper can use to learn "what components exist and
what can be configured on them" WITHOUT building a System or having any data
files on disk.
"""
import json

import pytest

from exozippy import introspect
from exozippy.components.component import Component

# The core component set every EXOZIPPy checkout ships.
CORE_COMPONENTS = {
    "star",
    "planet",
    "orbit",
    "band",
    "transit",
    "rvinstrument",
    "sed",
    "lens",
    "mulensinstrument",
    "astrometryinstrument",
    "galacticmodel",
}


def test_list_components_finds_the_core_component_set():
    """
    Given the standard EXOZIPPy component tree,
    When list_components is called,
    Then every known core component key is present with class/module/doc.
    """
    # Act
    comps = introspect.list_components()

    # Assert
    missing = CORE_COMPONENTS - set(comps)
    assert not missing, f"Missing components: {missing}"
    for key, info in comps.items():
        assert set(info) == {"class", "module", "doc"}, key
        assert isinstance(info["class"], str) and info["class"]
        assert info["module"].startswith("exozippy.components")


def test_component_schema_reports_class_and_parameters():
    """
    Given the star component,
    When component_schema('star') is built,
    Then it exposes the class summary and a non-empty parameter dict.
    """
    # Act
    schema = introspect.component_schema("star")

    # Assert
    assert schema["yaml_key"] == "star"
    assert schema["class"] == "Star"
    assert "mass" in schema["parameters"]
    assert "logmass" in schema["parameters"]


def test_star_sampled_parameters_have_bounds_and_scale():
    """
    Given star's defaults.yaml,
    When every parameter marked as sampled is inspected,
    Then each carries lower, upper, and init_scale (per CLAUDE.md).
    """
    # Arrange
    schema = introspect.component_schema("star")

    # Act
    sampled = {
        name: p for name, p in schema["parameters"].items() if p["sampled"]
    }

    # Assert
    assert sampled, "star should have at least one sampled parameter"
    for name, p in sampled.items():
        for field in ("lower", "upper", "init_scale"):
            assert field in p, f"sampled star.{name} missing {field}"


def test_derived_parameter_exposes_expression_and_deps():
    """
    Given star.mass, which is derived from logmass,
    When its schema entry is inspected,
    Then it is flagged derived, not sampled, and lists its dependency.
    """
    # Act
    mass = introspect.component_schema("star")["parameters"]["mass"]

    # Assert
    assert mass["derived"] is True
    assert mass["sampled"] is False
    assert mass["deps"] == ["logmass"]
    assert mass["expressions"]["default"]["func_name"] == "calc_mass"


def test_transit_config_schema_has_band_ref_and_data_file():
    """
    Given the transit component,
    When its config schema is inspected,
    Then it declares a required band reference and a data-file key.
    """
    # Act
    config = introspect.component_schema("transit")["config"]
    by_key = {e["key"]: e for e in config}

    # Assert
    assert by_key["band"]["kind"] == "ref"
    assert "band" in by_key["band"]["accepts"]
    assert by_key["band"]["required"] is True

    assert by_key["file"]["kind"] == "datafile"
    assert isinstance(by_key["file"]["accepts"], str)  # a glob pattern
    assert by_key["file"]["required"] is True


def test_config_schema_entries_have_the_declared_shape():
    """
    Given every component that overrides config_schema,
    When each entry is checked,
    Then it exposes key/kind/accepts/required/doc with valid kinds.
    """
    # Arrange
    valid_kinds = {"datafile", "ref", "option"}

    # Act / Assert
    for yaml_key in introspect.list_components():
        for entry in introspect.component_schema(yaml_key)["config"]:
            assert set(entry) >= {"key", "kind", "accepts", "required", "doc"}
            assert entry["kind"] in valid_kinds
            assert isinstance(entry["required"], bool)
            if entry["kind"] == "ref":
                assert isinstance(entry["accepts"], list)
            elif entry["kind"] == "datafile":
                assert isinstance(entry["accepts"], str)


def test_component_schema_exposes_utilities():
    """
    Given the transit component,
    When its schema is built,
    Then a JSON-serializable 'utilities' list surfaces getdata and bls.
    """
    # Act
    schema = introspect.component_schema("transit")
    utilities = schema["utilities"]

    # Assert
    assert json.loads(json.dumps(utilities)) == utilities
    by_name = {u["name"]: u for u in utilities}
    assert "getdata" in by_name
    assert by_name["getdata"]["available"] is True
    # Arguments carry the JSON argument schema for form rendering.
    assert any(a["name"] == "id" for a in by_name["getdata"]["arguments"])
    assert by_name["bls"]["available"] is False


def test_every_component_utility_list_is_json_serializable():
    """
    Given every discoverable component,
    When its schema's utilities list is inspected,
    Then each entry has the declared shape and survives json.dumps.
    """
    # Act / Assert
    for yaml_key in introspect.list_components():
        utilities = introspect.component_schema(yaml_key)["utilities"]
        assert json.loads(json.dumps(utilities)) == utilities
        for entry in utilities:
            assert set(entry) >= {
                "name", "label", "description", "available", "arguments"}
            assert isinstance(entry["available"], bool)
            assert isinstance(entry["arguments"], list)


def test_base_component_config_schema_is_empty():
    """
    Given the base Component class,
    When config_schema is called on it,
    Then the generic default is an empty list.
    """
    # Act / Assert
    assert Component.config_schema() == []


def test_full_schema_round_trips_through_json():
    """
    Given the full introspection schema,
    When it is serialized with json.dumps and parsed back,
    Then the object is unchanged (no numpy scalars, no Path objects).
    """
    # Arrange
    schema = introspect.full_schema()

    # Act
    round_tripped = json.loads(json.dumps(schema))

    # Assert
    assert round_tripped == schema
    assert set(schema["components"]) >= CORE_COMPONENTS
    assert "sampler" in schema["global"]
    # KNOWN_SAMPLER_KEYS from run.py surfaces as the sampler block's accepts.
    assert "draws" in schema["global"]["sampler"]["accepts"]
    assert "method" in schema["global"]["sampler"]["accepts"]


def test_unknown_component_raises_keyerror():
    """
    Given a yaml_key that no component declares,
    When component_schema is called with it,
    Then a KeyError is raised.
    """
    # Act / Assert
    with pytest.raises(KeyError):
        introspect.component_schema("does_not_exist")
