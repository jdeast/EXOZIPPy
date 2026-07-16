"""Tests for the headless solve / validate API (exozippy.solve_api).

The solve_api runs only lifecycle stages 1-3 (System.prepare) and reads the
in-memory relaxation-engine solution back out -- no PyMC model is built.  These
tests use the RV-only KELT-4 example (examples/kelt4/kelt4_rvonly.yaml).

Tests follow AAA (Arrange / Act / Assert) with Given/When/Then docstrings.
"""

import copy
import json
from pathlib import Path

import pytest
import yaml

from exozippy.solve_api import solve, validate, SolveResult

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "kelt4"


@pytest.fixture(scope="module")
def kelt4_inputs():
    """Given the RV-only KELT-4 example, load its config and params dicts.

    Returns (config, user_params, workdir) where workdir is the example
    directory the config's data-file paths are relative to.
    """
    with open(EXAMPLE_DIR / "kelt4_rvonly.yaml") as f:
        config = yaml.safe_load(f)
    with open(EXAMPLE_DIR / "kelt4.params.yaml") as f:
        user_params = yaml.safe_load(f)
    return config, user_params, str(EXAMPLE_DIR)


@pytest.fixture(scope="module")
def solved(kelt4_inputs):
    """When solve() runs on the example, return the SolveResult (shared)."""
    config, user_params, workdir = kelt4_inputs
    return solve(config, user_params, workdir)


def test_solve_result_is_json_serializable(solved):
    """
    Given the KELT-4 RV-only example,
    When solve() returns a SolveResult,
    Then as_dict() round-trips through json.dumps without error.
    """
    # Arrange / Act
    payload = solved.as_dict()

    # Assert
    text = json.dumps(payload)
    assert isinstance(text, str) and len(text) > 0
    assert set(payload) >= {"parameters", "warnings", "diagnostics", "elapsed_s"}
    assert isinstance(solved, SolveResult)
    assert solved.elapsed_s >= 0.0


def test_user_set_parameter_reports_user_provenance(solved):
    """
    Given star.A.radius is set explicitly in the params file,
    When solve() resolves the config,
    Then that parameter reports provenance label "user".
    """
    # Arrange
    info = solved.parameters["star.A.radius"]

    # Assert
    assert info["provenance"]["label"] == "user"
    assert info["value"] == pytest.approx(1.610)


def test_untouched_default_reports_default_provenance(solved):
    """
    Given star.A.distance is never set by the user or any relation,
    When solve() resolves the config,
    Then it reports provenance label "default" carrying the defaults.yaml value.
    """
    # Arrange
    info = solved.parameters["star.A.distance"]

    # Assert
    assert info["provenance"]["label"] == "default"
    assert info["unit"] == "pc"
    assert info["value"] is not None


def test_derived_quantity_reports_solved_provenance(solved):
    """
    Given stellar density is derived from mass and radius (a physics relation),
    When solve() resolves the config,
    Then it reports provenance label "solved", the derived flag, and a value.
    """
    # Arrange
    info = solved.parameters["star.A.density"]

    # Assert
    assert info["provenance"]["label"] == "solved"
    assert info["derived"] is True
    assert info["value"] is not None
    # The engine records which relation produced a solved value.
    assert info["provenance"]["relation"] is not None


def test_clean_config_has_no_diagnostics(solved):
    """
    Given a consistent config,
    When solve() runs,
    Then no contradiction diagnostics are reported.
    """
    # Assert
    assert solved.diagnostics == []


def test_bounds_excluding_initval_yields_diagnostic(kelt4_inputs):
    """
    Given a user initval placed below a tightened lower bound,
    When validate() runs,
    Then it returns an error diagnostic naming the offending parameter instead
    of raising.
    """
    # Arrange -- teff initval 6207 with a lower bound of 7000 excludes it.
    config, user_params, workdir = kelt4_inputs
    bad_params = copy.deepcopy(user_params)
    bad_params["star.0.teff"] = {"initval": 6207, "lower": 7000}

    # Act
    diagnostics = validate(config, bad_params, workdir)

    # Assert
    assert isinstance(diagnostics, list)
    assert any(
        d["severity"] == "error" and "star.A.teff" in d["param_paths"]
        for d in diagnostics
    )


def test_overconstrained_relation_yields_diagnostic(kelt4_inputs):
    """
    Given mass, radius, and density all set to mutually inconsistent values,
    When validate() runs,
    Then the over-constrained density relation surfaces as an error diagnostic.
    """
    # Arrange -- density=mass/radius^3 approx 1.7; force an inconsistent 99.
    config, user_params, workdir = kelt4_inputs
    bad_params = copy.deepcopy(user_params)
    bad_params["star.0.density"] = {"initval": 99.0}

    # Act
    diagnostics = validate(config, bad_params, workdir)

    # Assert
    assert any(
        d["severity"] == "error" and "density" in d["message"].lower()
        for d in diagnostics
    )


def test_solve_is_repeatable_in_one_process(kelt4_inputs):
    """
    Given solve() must not leak module-level state,
    When it is called twice consecutively in one process,
    Then both calls succeed and report the same parameter set.
    """
    # Arrange
    config, user_params, workdir = kelt4_inputs

    # Act
    first = solve(config, user_params, workdir)
    second = solve(config, user_params, workdir)

    # Assert
    assert set(first.parameters) == set(second.parameters)
    assert first.parameters["star.A.radius"]["provenance"]["label"] == "user"
    assert second.parameters["star.A.radius"]["provenance"]["label"] == "user"


def test_solve_accepts_user_params_from_file(kelt4_inputs):
    """
    Given user_params is None,
    When solve() runs,
    Then it loads the config's parameter_file relative to workdir and still
    reports the user-set radius as "user".
    """
    # Arrange
    config, _user_params, workdir = kelt4_inputs

    # Act
    result = solve(config, None, workdir)

    # Assert
    assert result.parameters["star.A.radius"]["provenance"]["label"] == "user"
