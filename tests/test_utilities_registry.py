"""Tests for exozippy.utilities.registry: component-owned utility discovery.

These exercise the layer a GUI uses to (a) discover per-component helper
programs generically, (b) render an argument form from an argparse parser
without importing argparse semantics, and (c) run a utility headless while
detecting the files it produced. No test hits the network: real downloaders
are only introspected, never run; run_utility is exercised with a tiny fake.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from exozippy.components.component import Component
from exozippy.components.transit.transit import Transit
from exozippy.utilities import registry
from exozippy.utilities.registry import UtilitySpec, parser_to_schema, run_utility
from exozippy.utilities import getdata, mkticsed, mmexofast_to_params

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


# --- parser -> JSON schema ----------------------------------------------------

def test_parser_to_schema_is_json_serializable_for_all_real_parsers():
    """
    Given the three real utility argparse parsers,
    When each is converted with parser_to_schema,
    Then the result survives json.dumps unchanged.
    """
    # Arrange
    parsers = [getdata.build_parser(), mkticsed.build_parser(),
               mmexofast_to_params.build_parser()]

    # Act / Assert
    for parser in parsers:
        schema = parser_to_schema(parser)
        assert json.loads(json.dumps(schema)) == schema
        for entry in schema:
            assert set(entry) == {
                "name", "type", "default", "required", "choices", "help"}


def test_mkticsed_schema_exposes_expected_argument_names():
    """
    Given mkticsed's parser,
    When converted to a schema,
    Then the positional 'ticid' and options '--sedfile'/'--priorfile' appear.
    """
    # Act
    names = {e["name"] for e in parser_to_schema(mkticsed.build_parser())}

    # Assert
    assert "ticid" in names
    assert "--sedfile" in names
    assert "--priorfile" in names


def test_getdata_schema_marks_positional_required_and_flags_boolean():
    """
    Given getdata's parser,
    When converted to a schema,
    Then the positional id is required and the verbose flag is a bool.
    """
    # Act
    by_name = {e["name"]: e for e in parser_to_schema(getdata.build_parser())}

    # Assert
    assert by_name["id"]["required"] is True
    assert by_name["id"]["type"] == "str"
    assert by_name["--verbose"]["type"] == "bool"
    assert by_name["--verbose"]["required"] is False
    assert by_name["--depth"]["type"] == "float"


def test_mmexofast_schema_exposes_json_and_options():
    """
    Given mmexofast_to_params's parser,
    When converted to a schema,
    Then the positional 'json' and options '--lens-name'/'--out' appear.
    """
    # Act
    names = {e["name"] for e in parser_to_schema(mmexofast_to_params.build_parser())}

    # Assert
    assert "json" in names
    assert "--lens-name" in names
    assert "--out" in names


# --- component-declared utilities ---------------------------------------------

def test_transit_declares_getdata_and_a_disabled_bls():
    """
    Given the transit component,
    When its utilities are listed,
    Then getdata is available and bls is a disabled placeholder.
    """
    # Act
    utils = {u.name: u for u in Transit.get_utilities()}

    # Assert
    assert utils["getdata"].available is True
    assert utils["getdata"].build_parser is not None
    assert utils["bls"].available is False
    assert utils["bls"].build_parser is None


def test_base_component_get_utilities_is_empty():
    """
    Given the base Component class,
    When get_utilities is called,
    Then the generic default is an empty list.
    """
    # Act / Assert
    assert Component.get_utilities() == []


def test_all_utilities_gathers_expected_names():
    """
    Given the standard component tree,
    When all_utilities aggregates every component's declarations,
    Then the known utility names are present and uniquely keyed.
    """
    # Act
    utils = registry.all_utilities()

    # Assert
    for name in ("getdata", "bls", "mkticsed", "mmexofast_to_params",
                 "lomb_scargle"):
        assert name in utils, name
    assert utils["mkticsed"].component_keys == ["sed"]
    assert utils["mmexofast_to_params"].component_keys == ["lens"]


def test_utility_to_schema_round_trips_through_json():
    """
    Given a real, available utility spec,
    When to_schema is serialized and parsed back,
    Then it is unchanged and carries its argument schema.
    """
    # Arrange
    spec = {u.name: u for u in Transit.get_utilities()}["getdata"]

    # Act
    schema = spec.to_schema()
    round_tripped = json.loads(json.dumps(schema))

    # Assert
    assert round_tripped == schema
    assert schema["available"] is True
    assert any(a["name"] == "id" for a in schema["arguments"])


# --- headless run with a fake spec (no network) -------------------------------

@pytest.fixture
def fake_utility():
    """A tiny in-process utility that writes one file and returns rc=7."""
    def _run(args_dict, cwd):
        out = Path(cwd) / args_dict["outfile"]
        out.write_text("produced by fake utility\n")
        print("fake ran")
        return {"returncode": 7, "output": "fake ran"}

    return UtilitySpec(
        name="fake",
        label="Fake",
        description="A test-only utility.",
        component_keys=["transit"],
        available=True,
        build_parser=None,
        run=_run,
    )


def test_run_utility_captures_produced_files_and_returncode(tmp_path, fake_utility):
    """
    Given a fake utility that writes a file,
    When run_utility executes it in a working directory,
    Then the produced file and the return code are reported.
    """
    # Arrange
    reg = {fake_utility.name: fake_utility}

    # Act
    result = run_utility("fake", {"outfile": "new.txt"}, tmp_path, registry=reg)

    # Assert
    assert result["returncode"] == 7
    assert "fake ran" in result["output"]
    produced = [Path(p).name for p in result["produced_files"]]
    assert produced == ["new.txt"]
    assert json.loads(json.dumps(result)) == result


def test_run_utility_rejects_placeholder(fake_utility, tmp_path):
    """
    Given a disabled placeholder utility,
    When run_utility is asked to run it,
    Then it raises rather than executing anything.
    """
    # Arrange
    placeholder = UtilitySpec(name="ph", label="ph", description="",
                              available=False)
    reg = {"ph": placeholder}

    # Act / Assert
    with pytest.raises(ValueError):
        run_utility("ph", {}, tmp_path, registry=reg)


def test_run_utility_unknown_name_raises(tmp_path):
    """
    Given an empty registry,
    When run_utility is called with an unknown name,
    Then a KeyError is raised.
    """
    # Act / Assert
    with pytest.raises(KeyError):
        run_utility("nope", {}, tmp_path, registry={})


# --- scripts wrappers still work ----------------------------------------------

@pytest.mark.parametrize("script", ["getdata.py", "mkticsed.py",
                                    "mmexofast_to_params.py"])
def test_script_wrapper_responds_to_help(script):
    """
    Given a thin scripts/*.py wrapper,
    When invoked with --help as a subprocess,
    Then it exits 0 and prints usage (backward-compatible CLI).
    """
    # Act
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)

    # Assert
    assert proc.returncode == 0, proc.stderr
    assert "usage" in (proc.stdout + proc.stderr).lower()
