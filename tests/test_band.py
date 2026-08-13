"""Tests for the Band component lifecycle (load_data, build_maps, register_parameters)."""

import numpy as np
import pytest

from conftest import _DummyConfigManager
from exozippy.components.band.band import Band


def _make_band(config):
    return Band(config, _DummyConfigManager())


def _is_free(entry):
    """True when a manifest entry declares a SAMPLED (non-derived) parameter.

    The manifest's own rule (Component.add_parameter, System.derived_params):
    a string names an expression, a dict names one only via "expr_key", and
    anything else -- including a dict carrying just an "overrides" pin, which
    is how Band pins limb darkening no consumer reads -- is free.
    """
    if isinstance(entry, str):
        return False
    if isinstance(entry, dict):
        return entry.get("expr_key") is None
    return entry is None


def test_load_data_populates_lists_from_config():
    """
    Given a config with filter, star_ndx, and ld_law,
    When load_data is called,
    Then each attribute list is populated from the corresponding config key.
    """
    band = _make_band(
        [{"filter": "Cousins.I", "star_ndx": 1, "ld_law": "linear"}]
    )
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
    # q1/q2 are free; u1/u2 derive from them.  These bands have no consumer in
    # this (empty) topology, so the free pair also carries the auto-pin -- a
    # manifest dict with no "expr_key" is still a free parameter.
    assert _is_free(band.manifest["q1"])
    assert _is_free(band.manifest["q2"])
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
    assert _is_free(band.manifest["u1"])
    assert "q1" not in band.manifest
    assert "q2" not in band.manifest
    assert "u2" not in band.manifest


@pytest.mark.parametrize("bad", ["quadratik", "Quadratic LD", "kipping", ""])
def test_unknown_ld_law_raises_naming_value_and_accepted_set(bad):
    """
    Given a Band whose ld_law is not one of the implemented laws,
    When load_data is called,
    Then it raises, naming the offending value and every accepted law.

    Pre-fix, `has_quadratic = any(law != "linear")` made any typo silently
    select the quadratic law -- the same silent-ignore class as `IMF:
    Salpeter` (review 2.4.4).
    """
    band = _make_band([{"name": "V", "ld_law": bad}])
    with pytest.raises(ValueError) as excinfo:
        band.load_data(system=None)
    msg = str(excinfo.value)
    assert repr(bad) in msg, f"error does not name the bad value: {msg}"
    assert "quadratic" in msg and "linear" in msg, (
        f"error does not name the accepted set: {msg}"
    )


@pytest.mark.parametrize("law", ["quadratic", "linear", " LINEAR "])
def test_known_ld_law_is_accepted_and_normalized(law):
    """
    Given a Band whose ld_law is a recognized law (in any case/whitespace),
    When load_data is called,
    Then it is accepted and stored in canonical lowercase form.
    """
    band = _make_band([{"ld_law": law}])
    band.load_data(system=None)
    assert band.ld_laws == [law.strip().lower()]


def test_mixed_ld_laws_raise_instead_of_promoting_linear_bands():
    """
    Given two bands declaring different limb-darkening laws,
    When load_data is called,
    Then it raises, naming each band and its law.

    Pre-fix this configuration was accepted and `any(law != "linear")` chose
    the quadratic manifest for the whole vector, so the band the user declared
    linear silently got a free u2 and was modelled as quadratic (review
    2.4.4). Per-element derivation is not expressible in the manifest, so the
    honest behavior is a hard error.
    """
    band = _make_band(
        [
            {"name": "V", "ld_law": "linear"},
            {"name": "Sloani", "ld_law": "quadratic"},
        ]
    )
    with pytest.raises(ValueError) as excinfo:
        band.load_data(system=None)
    msg = str(excinfo.value)
    assert "ld_law" in msg
    assert "V" in msg and "Sloani" in msg, (
        f"error does not name the conflicting bands: {msg}"
    )


def test_uniform_ld_law_across_several_bands_is_allowed():
    """
    Given several bands that all declare the same law,
    When load_data and register_parameters run,
    Then no error is raised and the manifest matches that one law.
    """
    band = _make_band(
        [
            {"name": "V", "ld_law": "linear"},
            {"name": "Sloani", "ld_law": "linear"},
        ]
    )
    band.load_data(system=None)
    band.register_parameters(system=None)
    assert "u2" not in band.manifest
    assert _is_free(band.manifest["u1"])


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
