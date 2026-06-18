"""Tests for the Band component lifecycle (load_data, build_maps, register_parameters)."""
import logging
import numpy as np
import pytest

from exozippy.components.band.band import Band
from conftest import _DummyConfigManager


def _make_band(config):
    return Band(config, _DummyConfigManager())


def test_load_data_populates_lists_from_config():
    """
    Given a config with filter, star_ndx, ld_law, and claret_sigma,
    When load_data is called,
    Then each attribute list is populated from the corresponding config key.
    """
    band = _make_band([{"filter": "Cousins.I", "star_ndx": 1, "ld_law": "linear", "claret_sigma": 0.0}])
    band.load_data(system=None)
    assert band.filter_names == ["Cousins.I"]
    assert band.star_indices == [1]
    assert band.ld_laws == ["linear"]
    assert band.claret_sigmas == [0.0]


def test_load_data_applies_defaults_for_missing_keys():
    """
    Given a config with no optional keys,
    When load_data is called,
    Then defaults are used: empty filter, star_ndx=0, quadratic law, sigma=0.
    """
    band = _make_band([{}])
    band.load_data(system=None)
    assert band.filter_names == [""]
    assert band.star_indices == [0]
    assert band.ld_laws == ["quadratic"]
    assert band.claret_sigmas == [0.0]


def test_load_data_no_claret_grid_when_sigma_is_zero():
    """
    Given a config with claret_sigma = 0,
    When load_data is called,
    Then no Claret grid is loaded (grid entry remains None).
    """
    band = _make_band([{"filter": "Cousins.I", "claret_sigma": 0.0}])
    band.load_data(system=None)
    assert band._claret_grids[0] is None


def test_load_data_missing_filter_suppresses_claret_grid_even_with_sigma():
    """
    Given a config with claret_sigma > 0 but no filter name,
    When load_data is called,
    Then no Claret grid is loaded (can't look up grid without filter name).
    """
    band = _make_band([{"claret_sigma": 0.1}])  # no "filter" key
    band.load_data(system=None)
    assert band._claret_grids[0] is None


def test_load_data_missing_claret_file_logs_warning_and_returns_none(caplog):
    """
    Given a config pointing to a non-existent filter grid,
    When load_data is called with claret_sigma > 0,
    Then a WARNING is emitted and the grid entry is None.
    """
    band = _make_band([{"filter": "NonExistent.Filter", "claret_sigma": 0.05}])
    with caplog.at_level(logging.WARNING):
        band.load_data(system=None)
    assert band._claret_grids[0] is None
    assert any("Claret grid not found" in r.message for r in caplog.records)


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


def test_register_parameters_quadratic_law_sets_u1_u2():
    """
    Given a Band with the default quadratic limb-darkening law,
    When register_parameters is called,
    Then the manifest contains u1 and u2 (and not gamma).
    """
    band = _make_band([{"ld_law": "quadratic"}])
    band.load_data(system=None)
    band.register_parameters(system=None)
    assert "u1" in band.manifest
    assert "u2" in band.manifest
    assert "gamma" not in band.manifest


def test_register_parameters_linear_law_sets_gamma():
    """
    Given a Band with ld_law = 'linear',
    When register_parameters is called,
    Then the manifest contains gamma (and not u1/u2).
    """
    band = _make_band([{"ld_law": "linear"}])
    band.load_data(system=None)
    band.register_parameters(system=None)
    assert "gamma" in band.manifest
    assert "u1" not in band.manifest
    assert "u2" not in band.manifest


def test_build_likelihood_no_op_when_no_claret_grids():
    """
    Given a Band with no Claret grids (sigma=0 for all instances),
    When build_likelihood is called inside a pm.Model,
    Then no Potentials are added to the model.
    """
    import pymc as pm

    band = _make_band([{"ld_law": "quadratic", "claret_sigma": 0.0}])
    band.load_data(system=None)
    band.build_maps()
    band.register_parameters(system=None)

    with pm.Model() as model:
        band.build_likelihood(model, system=None)

    claret_potentials = [k for k in model.named_vars if "claret_prior" in k]
    assert claret_potentials == [], (
        f"Expected no Claret potentials, found: {claret_potentials}"
    )
