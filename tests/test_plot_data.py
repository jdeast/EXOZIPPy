"""
Tests for Component.plot_data (GUI PlotSpec pathway, prompt G4).

The GUI consumes plot DATA (arrays + labels) rather than rendered
matplotlib figures. These tests check that:
  * each implemented component returns >= 1 PlotSpec after
    prepare()+build_model();
  * every spec is JSON-serializable (json.dumps(spec.to_json()));
  * data-only mode (point=None) works after prepare() WITHOUT
    build_model();
  * model-trace y-values at the start point are finite and match the
    arrays the refactored shared helper feeds the legacy plot() path;
  * the legacy plot() still renders figures at the start point.

They follow AAA with Given/When/Then docstrings.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.plotspec import PlotSpec, Trace
from exozippy.system import System

pytestmark = pytest.mark.slow

_KELT4_DIR = Path(__file__).parent.parent / "examples" / "kelt4"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rvonly_prepared():
    """kelt4 RV-only system, prepared but NOT built (data-only regime)."""
    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    cwd = os.getcwd()
    os.chdir(_KELT4_DIR)
    try:
        with open("kelt4_rvonly.yaml") as f:
            config = yaml.safe_load(f)
        system = System(config)
        system.prepare()
    finally:
        os.chdir(cwd)
    return system


@pytest.fixture(scope="module")
def rvonly_built():
    """kelt4 RV-only system, prepared and built, with a start point."""
    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    cwd = os.getcwd()
    os.chdir(_KELT4_DIR)
    try:
        with open("kelt4_rvonly.yaml") as f:
            config = yaml.safe_load(f)
        system = System(config)
        system.prepare()
        model = system.build_model()
        with model:
            point = system.get_internal_point(model, system.get_raw_start(model))
    finally:
        os.chdir(cwd)
    return system, model, point


@pytest.fixture(scope="module")
def transit_built():
    """A transit-only kelt4 build (star A + planet b + TESS band), so the
    transit component's plot_data can be tested without the SED's model
    spectra download."""
    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    config = {
        "run": {"name": "kelt4"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{
            "name": "TESS_S48",
            "file": "n20220130.TESS.TESS.TIC165297570.S48.0120.SPOC.dat",
            "band": "TESS", "exptime": 2.0, "ninterp": 1.0,
        }],
    }
    user_params = {
        "star.0.radius": {"initval": 1.610, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": 2.9895933},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.11996},
        "planet.0.radius": {"initval": 1.706},
    }
    cwd = os.getcwd()
    os.chdir(_KELT4_DIR)
    try:
        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
        with model:
            point = system.get_internal_point(model, system.get_raw_start(model))
    finally:
        os.chdir(cwd)
    return system, model, point


# A minimal star-only SED build (the kelt4 three-star A/B/C deblend), so
# the SED component's plot_data can be tested without the full kelt4.yaml's
# astrometry/transit/rv machinery. Mirrors the star portion of
# kelt4_sed.params.yaml.
_SED_CONFIG = {
    "run": {"name": "kelt4"},
    "star": [{"name": "A", "mist": False},
             {"name": "B", "mist": False},
             {"name": "C", "mist": False}],
    "sed": {"file": "kelt4.sed.yaml"},
}
_SED_PARAMS = {
    "star.0.radius": {"initval": 1.610, "sigma": 0.05},
    "star.0.mass": {"initval": 1.204, "sigma": 0.05},
    "star.0.teff": {"initval": 6207, "sigma": 100},
    "star.0.feh": {"initval": -0.116, "sigma": 0.08},
    "star.A.distance": {"initval": 218.055, "mu": 218.055, "sigma": 0.889},
    "star.A.av": {"upper": 0.0521},
    "star.B.distance": {"initval": "star.A.distance", "sigma": 0},
    "star.B.av": {"initval": "star.A.av", "sigma": 0},
    "star.B.feh": {"initval": "star.A.feh", "sigma": 0},
    "star.B.logmass": {"initval": -0.125},
    "star.B.radius": {"initval": 0.7},
    "star.B.teff": {"initval": 4800},
    "star.C.distance": {"initval": "star.A.distance", "sigma": 0},
    "star.C.av": {"initval": "star.A.av", "sigma": 0},
    "star.C.feh": {"initval": "star.A.feh", "sigma": 0},
    "star.C.logmass": {"initval": -0.125},
    "star.C.radius": {"initval": 0.7},
    "star.C.teff": {"initval": 4800},
}


@pytest.fixture(scope="module")
def sed_built():
    """A star-only kelt4 SED build. Skips if the model spectra are
    unavailable (they are fetched from Zenodo on first load)."""
    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    cwd = os.getcwd()
    os.chdir(_KELT4_DIR)
    try:
        try:
            system = System(dict(_SED_CONFIG), user_params=dict(_SED_PARAMS))
            system.prepare()
            model = system.build_model()
        except Exception as e:  # noqa: BLE001 - network/data availability
            pytest.skip(f"SED example data unavailable: {e}")
        with model:
            point = system.get_internal_point(model, system.get_raw_start(model))
    finally:
        os.chdir(cwd)
    return system, model, point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_json_roundtrip(specs):
    for spec in specs:
        assert isinstance(spec, PlotSpec)
        payload = spec.to_json()
        text = json.dumps(payload)  # must not raise
        assert isinstance(text, str) and len(text) > 0


def _model_traces(specs):
    return [t for s in specs for t in s.traces if t.role == "model"]


# ---------------------------------------------------------------------------
# rvinstrument
# ---------------------------------------------------------------------------

def test_rvinstrument_plot_data_returns_serializable_specs(rvonly_built):
    """
    Given a prepared+built RV-only kelt4 system and a start point,
    When rvinstrument.plot_data(system, point) is called,
    Then it returns >= 1 PlotSpec and every spec is JSON-serializable.
    """
    system, model, point = rvonly_built

    specs = system.rvinstrument.plot_data(system, point)

    assert len(specs) >= 1
    _assert_json_roundtrip(specs)


def test_rvinstrument_model_trace_matches_shared_helper(rvonly_built):
    """
    Given the built RV-only system,
    When the unphased model trace from plot_data is compared to the arrays
    the shared _eval_unphased_model helper feeds the legacy plot() path,
    Then the y-values are finite and identical.
    """
    system, model, point = rvonly_built
    rv = system.rvinstrument

    specs = rv.plot_data(system, point)
    unphased = [s for s in specs if not s.meta["phase_folded"]][0]
    model_trace = [t for t in unphased.traces if t.role == "model"][0]

    _, y_expected = rv._eval_unphased_model(system, point)
    assert np.all(np.isfinite(model_trace.y))
    np.testing.assert_allclose(model_trace.y, y_expected)


def test_rvinstrument_param_deps_are_populated(rvonly_built):
    """
    Given the built RV-only system,
    When plot_data returns the unphased spec,
    Then its param_deps names sampled parameters that feed the RV model.
    """
    system, model, point = rvonly_built

    specs = system.rvinstrument.plot_data(system, point)
    unphased = [s for s in specs if not s.meta["phase_folded"]][0]

    assert len(unphased.param_deps) >= 1
    valid = {p.label for p in system.plot_params}
    assert set(unphased.param_deps) <= valid


def test_rvinstrument_data_only_without_build_model(rvonly_prepared):
    """
    Given an RV-only system that is only prepared (no build_model),
    When rvinstrument.plot_data(system, point=None) is called,
    Then it returns >= 1 serializable, data-only spec (no model traces).
    """
    system = rvonly_prepared

    specs = system.rvinstrument.plot_data(system, point=None)

    assert len(specs) >= 1
    _assert_json_roundtrip(specs)
    assert _model_traces(specs) == []
    # every trace is observational data
    assert all(t.role == "data" for s in specs for t in s.traces)


# ---------------------------------------------------------------------------
# transit
# ---------------------------------------------------------------------------

def test_transit_plot_data_returns_serializable_specs(transit_built):
    """
    Given a prepared+built transit kelt4 system and a start point,
    When transit.plot_data(system, point) is called,
    Then it returns >= 1 serializable PlotSpec with finite model traces.
    """
    system, model, point = transit_built

    specs = system.transit.plot_data(system, point)

    assert len(specs) >= 1
    _assert_json_roundtrip(specs)
    model_traces = _model_traces(specs)
    assert len(model_traces) >= 1
    for t in model_traces:
        assert np.all(np.isfinite(t.y))


def test_transit_model_trace_matches_shared_helper(transit_built):
    """
    Given the built transit system,
    When the unphased model trace from plot_data is compared to the arrays
    the shared _eval_unphased_lc helper feeds the legacy plot() path,
    Then the y-values are identical.
    """
    system, model, point = transit_built
    transit = system.transit

    specs = transit.plot_data(system, point)
    unphased = [s for s in specs if not s.meta["phase_folded"]][0]
    model_trace = [t for t in unphased.traces if t.role == "model"][0]

    inst_name = unphased.meta["instrument"]
    i = transit.names.index(inst_name)
    _, y_expected = transit._eval_unphased_lc(system, point, i)
    np.testing.assert_allclose(model_trace.y, y_expected)


def test_transit_data_only_without_model(transit_built):
    """
    Given the transit system,
    When plot_data is called with point=None,
    Then only data traces are returned and they are serializable.
    """
    system, model, point = transit_built

    specs = system.transit.plot_data(system, point=None)

    assert len(specs) >= 1
    _assert_json_roundtrip(specs)
    assert _model_traces(specs) == []


# ---------------------------------------------------------------------------
# sed
# ---------------------------------------------------------------------------

# The model-trace path loads the NextGen spectra table, whose per-row
# json.loads parse takes minutes and is not cached between runs -- it can
# exceed the 300s global pytest timeout on a cold or slow filesystem.
@pytest.mark.timeout(1200)
def test_sed_plot_data_returns_serializable_specs(sed_built):
    """
    Given a prepared+built rv+transit+sed kelt4 system and a start point,
    When sed.plot_data(system, point) is called,
    Then it returns >= 1 serializable PlotSpec with finite model traces
    matching the shared plot-object helper.
    """
    system, model, point = sed_built

    specs = system.sed.plot_data(system, point)

    assert len(specs) >= 1
    _assert_json_roundtrip(specs)
    model_traces = _model_traces(specs)
    assert len(model_traces) >= 1
    for t in model_traces:
        assert np.all(np.isfinite(t.y))

    # model spectra match the shared _make_plot_obj helper. The GUI spec plots
    # log10(lambda * F_lambda) (the standard SED representation, matching the
    # matplotlib plot() path) rather than raw flux, so compare in that space.
    plot_obj = system.sed._make_plot_obj(system, [point])
    wave_ang = np.asarray(plot_obj.df_wave["wavelength_angstrom"], dtype=float)
    star0 = [t for t in model_traces if t.name.endswith(str(plot_obj.star_names[0]))][0]
    np.testing.assert_allclose(
        star0.y, np.log10(plot_obj.flux_model_draws[0][0] * wave_ang))


def test_sed_data_only_without_build_model():
    """
    Given a prepared (not built) rv+transit+sed system,
    When sed.plot_data(system, point=None) is called,
    Then it returns a serializable data-only spec (observed photometry).
    """
    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    cwd = os.getcwd()
    os.chdir(_KELT4_DIR)
    try:
        try:
            system = System(dict(_SED_CONFIG), user_params=dict(_SED_PARAMS))
            system.prepare()
        except Exception as e:  # noqa: BLE001 - network/data availability
            pytest.skip(f"SED example data unavailable: {e}")

        specs = system.sed.plot_data(system, point=None)
    finally:
        os.chdir(cwd)

    assert len(specs) >= 1
    _assert_json_roundtrip(specs)
    assert _model_traces(specs) == []


# ---------------------------------------------------------------------------
# Regression: legacy plot() still renders at the start point
# ---------------------------------------------------------------------------

def test_legacy_plot_still_renders_at_start(rvonly_built, tmp_path):
    """
    Given the built RV-only system and a start point,
    When the legacy rvinstrument.plot() is invoked at that point,
    Then it produces PDF figures without error (refactor preserved it).
    """
    system, model, point = rvonly_built
    prefix = str(tmp_path / "kelt4_start")

    system.rvinstrument.plot(system, [point], filename_prefix=prefix)

    produced = list(Path(tmp_path).glob("kelt4_start*.pdf"))
    assert len(produced) >= 1
