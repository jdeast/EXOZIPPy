"""Tests for the Band component lifecycle (load_data, build_maps, register_parameters)."""
import numpy as np
import pytest

from exozippy.components.band.band import Band
from conftest import _DummyConfigManager


def _make_band(config):
    return Band(config, _DummyConfigManager())


def test_load_data_populates_lists_from_config():
    """
    Given a config with filter, star_ndx, and ld_law,
    When load_data is called,
    Then each attribute list is populated from the corresponding config key.
    """
    band = _make_band([{"filter": "Cousins.I", "star_ndx": 1, "ld_law": "linear"}])
    band.load_data(system=None)
    assert band.filter_names == ["Cousins.I"]
    assert band.star_indices == [1]
    assert band.ld_laws == ["linear"]


def test_load_data_applies_defaults_for_missing_keys():
    """
    Given a config with no optional keys,
    When load_data is called,
    Then defaults are used: empty filter, star_ndx=0, quadratic law.
    """
    band = _make_band([{}])
    band.load_data(system=None)
    assert band.filter_names == [""]
    assert band.star_indices == [0]
    assert band.ld_laws == ["quadratic"]


def test_build_maps_creates_integer_star_map():
    """
    Given a two-instance Band with star_ndx values [0, 2],
    When build_maps is called,
    Then star_map is a numpy int array with those indices.
    """
    band = _make_band([{"star_ndx": 0}, {"star_ndx": 2}])
    band.load_data(system=None)
    band.build_maps()
    np.testing.assert_array_equal(band.star_map, [0, 2])
    assert band.star_map.dtype == int


def test_register_parameters_quadratic_law_uses_kipping():
    """
    Given a Band with the default quadratic limb-darkening law,
    When register_parameters is called,
    Then the manifest contains q1, q2 (free) and u1, u2 (derived).
    """
    band = _make_band([{"ld_law": "quadratic"}])
    band.load_data(system=None)
    band.register_parameters(system=None)
    assert "q1" in band.manifest
    assert "q2" in band.manifest
    assert "u1" in band.manifest
    assert "u2" in band.manifest
    # q1/q2 are free; u1/u2 derive from them
    assert band.manifest["q1"] is None
    assert band.manifest["q2"] is None
    assert band.manifest["u1"] == "default"
    assert band.manifest["u2"] == "default"


def test_register_parameters_linear_law_samples_u1_directly():
    """
    Given a Band with ld_law = 'linear',
    When register_parameters is called,
    Then the manifest contains only u1 as a free parameter (no q1/q2/u2).
    """
    band = _make_band([{"ld_law": "linear"}])
    band.load_data(system=None)
    band.register_parameters(system=None)
    assert "u1" in band.manifest
    assert band.manifest["u1"] is None
    assert "q1" not in band.manifest
    assert "q2" not in band.manifest
    assert "u2" not in band.manifest


def test_build_likelihood_adds_no_potentials():
    """
    Given any Band configuration,
    When build_likelihood is called inside a pm.Model,
    Then no potentials are added (table LD penalty is not yet implemented).
    """
    import pymc as pm

    band = _make_band([{"ld_law": "quadratic"}])
    band.load_data(system=None)
    band.build_maps()
    band.register_parameters(system=None)

    with pm.Model() as model:
        band.build_likelihood(model, system=None)

    assert list(model.named_vars) == [], (
        f"Expected no model variables from build_likelihood, found: {list(model.named_vars)}"
    )
