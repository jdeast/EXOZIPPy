"""Tests for the shared Instrument base class (components/instrument.py).

These cover the behavior-preserving extraction from notes/instrument_todo.txt:
the two noise parameterizations (additive jitter_variance vs multiplicative
err_scale), the optional detrend block-diagonal builder, the shared jitter
floor, the optional per-instrument plot styling, and the factory's skipping of
the abstract base.  Physics/likelihood behavior of the four children is covered
by their own integration suites; here we pin the shared scaffolding directly.
"""
import inspect

import numpy as np
import pytest

from exozippy.components.instrument import Instrument
from exozippy.components.component import Component
from exozippy.components.factory import discover_components
from exozippy.plotspec import Trace


class _DummyInstrument(Instrument):
    """Minimal concrete Instrument for unit-testing the shared helpers."""

    @property
    def prefix(self):
        return "dummy"

    def register_parameters(self, system):  # pragma: no cover - not exercised
        pass

    def build_likelihood(self, model, system):  # pragma: no cover
        pass


class _DummyErrScale(_DummyInstrument):
    noise_model = "err_scale"


def _make(config):
    return _DummyInstrument(config, config_manager=None)


# ---------------------------------------------------------------------------
# Factory: the abstract base must not be a discoverable component
# ---------------------------------------------------------------------------
def test_instrument_base_is_abstract():
    """
    Given the Instrument base leaves Component's abstract methods unimplemented,
    When inspect.isabstract is queried,
    Then Instrument is abstract (so the factory will skip it).
    """
    assert inspect.isabstract(Instrument)


def test_factory_does_not_register_instrument_base():
    """
    Given the factory scans every module (Instrument is imported into each child),
    When discover_components runs,
    Then no 'instrument' key is registered and every value is concrete.
    """
    registry = discover_components()
    assert "instrument" not in registry
    for key, cls in registry.items():
        assert not inspect.isabstract(cls), f"{key} -> {cls} is abstract"
        assert issubclass(cls, Component) and cls is not Component


def test_four_children_are_instrument_subclasses_with_expected_noise_model():
    """
    Given the four data components,
    When their class attributes are inspected,
    Then all subclass Instrument and mulens uses err_scale while the rest use
    the additive jitter_variance model.
    """
    registry = discover_components()
    expected = {
        "rvinstrument": "jitter_variance",
        "transit": "jitter_variance",
        "astrometryinstrument": "jitter_variance",
        "mulensinstrument": "err_scale",
    }
    for key, noise in expected.items():
        cls = registry[key]
        assert issubclass(cls, Instrument)
        assert cls.noise_model == noise


# ---------------------------------------------------------------------------
# Shared jitter floor
# ---------------------------------------------------------------------------
def test_jitter_floor_matches_formula():
    """
    Given an error array,
    When _jitter_floor is applied,
    Then it returns -0.95 * min(err)**2.
    """
    err = np.array([0.3, 0.1, 0.25])
    assert Instrument._jitter_floor(err) == pytest.approx(-0.95 * 0.1 ** 2)


def test_jitter_floor_applies_unit_factor_before_squaring():
    """
    Given a unit conversion factor (rv scales the error column to m/s first),
    When _jitter_floor is applied with that factor,
    Then the floor is computed on the converted minimum.
    """
    err = np.array([2.0, 0.5, 1.0])
    factor = 3.0
    assert Instrument._jitter_floor(err, factor=factor) == pytest.approx(
        -0.95 * (0.5 * factor) ** 2)


# ---------------------------------------------------------------------------
# Shared detrend block-diagonal builder
# ---------------------------------------------------------------------------
def test_build_block_detrend_places_columns_on_the_diagonal():
    """
    Given two instruments with 3 and 2 observations and 1 and 2 detrend columns,
    When _build_block_detrend runs,
    Then the design matrix is block-diagonal so coefficients never mix across
    instruments, and the column counts are reported.
    """
    a = np.array([[1.0], [2.0], [3.0]])          # inst 0: 3 obs, 1 col
    b = np.array([[4.0, 5.0], [6.0, 7.0]])       # inst 1: 2 obs, 2 cols
    matrix, per_inst, total = Instrument._build_block_detrend([a, b], 5)

    assert per_inst == [1, 2]
    assert total == 3
    assert matrix.shape == (5, 3)
    # inst 0 column only populated in its own rows/cols
    assert np.array_equal(matrix[:3, 0], [1.0, 2.0, 3.0])
    assert np.all(matrix[:3, 1:] == 0.0)         # no cross-instrument leakage
    assert np.all(matrix[3:, 0] == 0.0)
    assert np.array_equal(matrix[3:, 1:], b)


def test_build_block_detrend_handles_no_detrend_columns():
    """
    Given instruments with zero detrend columns,
    When _build_block_detrend runs,
    Then it returns a (n_obs, 0) matrix and total 0.
    """
    empties = [np.empty((3, 0)), np.empty((2, 0))]
    matrix, per_inst, total = Instrument._build_block_detrend(empties, 5)
    assert total == 0
    assert per_inst == [0, 0]
    assert matrix.shape == (5, 0)


# ---------------------------------------------------------------------------
# Noise registration into a manifest
# ---------------------------------------------------------------------------
def test_register_noise_additive_adds_jitter_variance_and_jitter():
    """
    Given the default jitter_variance noise model,
    When _register_noise runs with a floor,
    Then the manifest gains jitter_variance (with the lower floor) and jitter.
    """
    inst = _make([{"file": "a.dat"}])
    manifest = {}
    inst._register_noise(manifest, jittervar_lower=[-0.1])
    assert manifest["jitter_variance"] == {"lower": [-0.1]}
    assert manifest["jitter"] == "default"
    assert "err_scale" not in manifest


def test_register_noise_multiplicative_adds_err_scale_only():
    """
    Given the err_scale noise model,
    When _register_noise runs,
    Then the manifest gains only err_scale (no jitter_variance/jitter).
    """
    inst = _DummyErrScale([{"file": "a.dat"}], config_manager=None)
    manifest = {}
    inst._register_noise(manifest)
    assert manifest == {"err_scale": None}


def test_register_noise_additive_requires_a_floor():
    """
    Given the additive model,
    When _register_noise is called without a jittervar_lower floor,
    Then it raises (the floor keeps the total variance positive).
    """
    inst = _make([{"file": "a.dat"}])
    with pytest.raises(ValueError):
        inst._register_noise({})


# ---------------------------------------------------------------------------
# Optional per-instrument plot styling (config, not Parameters)
# ---------------------------------------------------------------------------
def test_plot_styles_default_to_none():
    """
    Given instruments without a plot: block,
    When the base __init__ loads styles,
    Then plot_color/plot_marker are None and the trace style carries only the
    series index (so the theme default by index still applies).
    """
    inst = _make([{"file": "a.dat"}, {"file": "b.dat"}])
    assert inst.plot_color == [None, None]
    assert inst.plot_marker == [None, None]
    assert inst._data_trace_style(1) == {"series_index": 1}


def test_plot_styles_read_user_overrides():
    """
    Given an instrument with plot: {color, marker},
    When styles are loaded,
    Then the overrides surface in _data_trace_style alongside series_index.
    """
    inst = _make([
        {"file": "a.dat", "plot": {"color": "#1f77b4", "marker": "s"}},
        {"file": "b.dat", "plot": {"color": "red"}},
        {"file": "c.dat"},
    ])
    assert inst.plot_color == ["#1f77b4", "red", None]
    assert inst.plot_marker == ["s", None, None]
    assert inst._data_trace_style(0) == {
        "series_index": 0, "color": "#1f77b4", "marker": "s"}
    assert inst._data_trace_style(1) == {"series_index": 1, "color": "red"}
    assert inst._data_trace_style(2) == {"series_index": 2}


def test_plot_style_schema_entry_shape():
    """
    Given the shared plot config-schema entry,
    When inspected,
    Then it declares the standard option shape consumed by introspection.
    """
    entry = Instrument._plot_style_config_schema()
    assert entry["key"] == "plot"
    assert entry["kind"] == "option"
    assert entry["required"] is False
    assert set(entry) >= {"key", "kind", "accepts", "required", "doc"}


# ---------------------------------------------------------------------------
# Trace.style serialization (plumbing target for the plot styles)
# ---------------------------------------------------------------------------
def test_trace_style_serializes_when_present():
    """
    Given a data Trace carrying a style dict,
    When to_json runs,
    Then the style is included (and absent when None).
    """
    t = Trace(name="TESS", role="data", kind="scatter",
              x=[1.0], y=[2.0], style={"series_index": 0, "marker": "s"})
    d = t.to_json()
    assert d["style"] == {"series_index": 0, "marker": "s"}

    plain = Trace(name="m", role="model", kind="line", x=[1.0], y=[2.0])
    assert "style" not in plain.to_json()
