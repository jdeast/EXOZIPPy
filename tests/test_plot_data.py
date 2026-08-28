"""
Tests for Component.plot_data (GUI Chart pathway, prompt G4).

The GUI consumes plot DATA (arrays + labels) rather than rendered
matplotlib figures. These tests check that:
  * each implemented component returns >= 1 Chart after
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

from exozippy.chart import Chart, Trace
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
            point = system.get_internal_point(
                model, system.get_raw_start(model)
            )
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
        "transit": [
            {
                "name": "TESS_S48",
                "file": "n20220130.TESS.TESS.TIC165297570.S48.0120.SPOC.dat",
                "band": "TESS",
                "exptime": 2.0,
                "ninterp": 1.0,
            }
        ],
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
            point = system.get_internal_point(
                model, system.get_raw_start(model)
            )
    finally:
        os.chdir(cwd)
    return system, model, point


# A minimal star-only SED build (the kelt4 three-star A/B/C deblend), so
# the SED component's plot_data can be tested without the full kelt4.yaml's
# astrometry/transit/rv machinery. Mirrors the star portion of
# kelt4_sed.params.yaml.
_SED_CONFIG = {
    "run": {"name": "kelt4"},
    "star": [
        {"name": "A", "mist": False},
        {"name": "B", "mist": False},
        {"name": "C", "mist": False},
    ],
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
            point = system.get_internal_point(
                model, system.get_raw_start(model)
            )
    finally:
        os.chdir(cwd)
    return system, model, point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_json_roundtrip(specs):
    for spec in specs:
        assert isinstance(spec, Chart)
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
    Then it returns >= 1 Chart and every spec is JSON-serializable.
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
    Then it returns >= 1 serializable Chart with finite model traces.
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


def test_transit_spec_meta_carries_pdf_file_tags(transit_built):
    """
    Given the built transit system and a start point,
    When plot_data returns its specs,
    Then every spec's meta carries the file_tag that reproduces the
    historical PDF filenames: LC_unphased_{inst} for the unphased chart
    and LC_phased_{inst}_{planet} for each phased chart.
    """
    system, model, point = transit_built

    specs = system.transit.plot_data(system, point)

    unphased = [s for s in specs if not s.meta["phase_folded"]]
    phased = [s for s in specs if s.meta["phase_folded"]]
    assert len(unphased) >= 1 and len(phased) >= 1
    for s in unphased:
        assert s.meta["file_tag"] == f"LC_unphased_{s.meta['instrument']}"
    for s in phased:
        assert "file_tag" in s.meta
        assert s.meta["file_tag"] == (
            f"LC_phased_{s.meta['instrument']}_{s.meta['planet']}"
        )


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
    Then it returns >= 1 serializable Chart with finite model traces
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

    # the spec's file_tag names the PDF the hand-drawn plot() saves
    # ({prefix}_SED.pdf), keeping the two descriptions tied together
    sed_spec = [s for s in specs if s.id == "sed.sed"][0]
    assert sed_spec.meta["file_tag"] == "SED"

    # model spectra match the shared _make_plot_obj helper. The GUI spec plots
    # log10(lambda * F_lambda) (the standard SED representation, matching the
    # matplotlib plot() path) rather than raw flux, so compare in that space.
    plot_obj = system.sed._make_plot_obj(system, [point])
    wave_ang = np.asarray(plot_obj.df_wave["wavelength_angstrom"], dtype=float)
    star0 = [
        t for t in model_traces if t.name.endswith(str(plot_obj.star_names[0]))
    ][0]
    np.testing.assert_allclose(
        star0.y, np.log10(plot_obj.flux_model_draws[0][0] * wave_ang)
    )


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


# ---------------------------------------------------------------------------
# Regression: a PINNED gamma must reach the RV plots (not plot as zero)
# ---------------------------------------------------------------------------

# A large, unmistakable offset: nothing else in the model can produce it, so
# any leak of the old 0.0 default shows up as the whole instrument sitting
# _GAMMA_PINNED m/s away from the model curve.
_PIN = dict(gamma=-12345.0, K=100.0, P=10.0, tc=2450005.0)


@pytest.fixture(scope="module")
def pinned_gamma_rv_system(tmp_path_factory):
    """One-instrument RV system whose gamma is PINNED (sigma: 0).

    A gamma vector with no free element never becomes a pm.Deterministic,
    so its label is absent from both model.deterministics and the
    posterior -- which is exactly the condition the bug needed.
    """
    tmp_dir = tmp_path_factory.mktemp("pinned_gamma")
    path = tmp_dir / "pinned.rv"

    t = np.linspace(2450000.0, 2450100.0, 40)
    rv = _PIN["gamma"] + _PIN["K"] * np.sin(
        2 * np.pi * (t - _PIN["tc"]) / _PIN["P"]
    )
    np.savetxt(path, np.column_stack([t, rv, np.full_like(t, 5.0)]))

    config = {
        "run": {"name": "pinned_gamma"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": [{"name": "HIRES", "file": str(path)}],
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "orbit.b.period": {"initval": _PIN["P"]},
        "orbit.b.tc": {"initval": _PIN["tc"]},
        "rvinstrument.HIRES.gamma": {"initval": _PIN["gamma"], "sigma": 0},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.compile_plotter_functions(model)
    return system, model, point, t, rv


def test_pinned_gamma_is_absent_from_the_point(pinned_gamma_rv_system):
    """
    Given an RV fit whose gamma is pinned with sigma: 0,
    When the plotting point is built from the model,
    Then gamma is not in it -- so any plot helper reading the point needs a
    real fallback, and this fixture genuinely exercises that path.
    """
    system, _, point, _, _ = pinned_gamma_rv_system

    label = system.rvinstrument.gamma.label

    assert label not in point


def test_unphased_rv_data_uses_the_pinned_gamma(pinned_gamma_rv_system):
    """
    Given an RV fit whose gamma is PINNED at a large nonzero value,
    When plot_data builds the unphased chart,
    Then the plotted (gamma-subtracted) data carry that pinned offset.

    Regression: _point_value's predecessor read the point with
    point.get(label, 0.0), and a pinned parameter is always absent from the
    draws, so the whole instrument plotted -12345 m/s away from the model
    curve while the likelihood used the real offset.
    """
    system, _, point, _, rv_ms = pinned_gamma_rv_system
    comp = system.rvinstrument

    specs = comp.plot_data(system, point)
    unphased = [s for s in specs if not s.meta["phase_folded"]][0]
    data_trace = [t for t in unphased.traces if t.role == "data"][0]

    # independent reference: the file's own RVs minus the pinned gamma
    expected = rv_ms - _PIN["gamma"]
    np.testing.assert_allclose(data_trace.y, expected, atol=1e-6)
    # and the offset dominates the signal, so plotting gamma as 0 is not a
    # rounding-level difference: it is a 123x shift of the whole series
    assert abs(_PIN["gamma"]) > 100 * np.ptp(expected) / 2.0
    # the pre-fix value, explicitly excluded
    assert not np.allclose(data_trace.y, rv_ms, atol=1.0)


def test_gamma_helper_returns_the_pinned_value(pinned_gamma_rv_system):
    """
    Given the same pinned-gamma fit,
    When _point_value is asked for the instrument's offset,
    Then it returns the pinned value (in internal units), not zero.
    """
    system, _, point, _, _ = pinned_gamma_rv_system
    comp = system.rvinstrument

    g_ms = comp._point_value(point, comp.gamma, 0) * comp._rv_factor()

    assert g_ms == pytest.approx(_PIN["gamma"], rel=1e-9)


def test_phased_rv_data_uses_the_pinned_gamma(pinned_gamma_rv_system):
    """
    Given the same pinned-gamma fit,
    When plot_data builds the phased chart for the single orbit,
    Then its cleaned data are gamma-subtracted with the pinned value too
    (the phased panel calls the same helper, one orbit -> no other signal).
    """
    system, _, point, _, rv_ms = pinned_gamma_rv_system
    comp = system.rvinstrument

    specs = comp.plot_data(system, point)
    phased = [s for s in specs if s.meta["phase_folded"]]
    assert len(phased) == 1
    data_trace = [t for t in phased[0].traces if t.role == "data"][0]

    expected = rv_ms - _PIN["gamma"]
    # the phase fold reorders nothing here (traces keep data order), so
    # compare the sorted values -- the offset is what is under test
    np.testing.assert_allclose(
        np.sort(data_trace.y), np.sort(expected), atol=1e-6
    )


# ---------------------------------------------------------------------------
# Regression: a PINNED baseline must reach the transit plots (not plot as 1.0)
# ---------------------------------------------------------------------------

# An UN-NORMALIZED light curve: raw counts, so the baseline the likelihood
# uses is nowhere near the old 1.0 default and any leak of it shifts both
# panels by the entire flux scale.
_LC = dict(baseline=20000.0, depth=0.01, P=3.0, tc=2459000.0, err=20.0)


@pytest.fixture(scope="module")
def pinned_baseline_transit_system(tmp_path_factory):
    """One-instrument transit system on raw counts whose baseline is PINNED.

    A baseline vector with no free element never becomes a
    pm.Deterministic, so its label is absent from both
    model.deterministics and the posterior -- the condition the bug needed.
    """
    tmp_dir = tmp_path_factory.mktemp("pinned_baseline")
    path = tmp_dir / "pinned.TESS.dat"

    # One transit, sampled densely enough that the median flux is still the
    # out-of-transit level (so baseline_init is the pinned value too).
    t = np.linspace(_LC["tc"] - 0.25, _LC["tc"] + 0.25, 200)
    flux = np.full_like(t, _LC["baseline"])
    in_transit = np.abs(t - _LC["tc"]) < 0.05
    flux[in_transit] -= _LC["baseline"] * _LC["depth"]
    np.savetxt(path, np.column_stack([t, flux, np.full_like(t, _LC["err"])]))

    config = {
        "run": {"name": "pinned_baseline"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "TESS_S48", "file": str(path), "band": "TESS"}],
    }
    user_params = {
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.b.period": {"initval": _LC["P"]},
        "orbit.b.tc": {"initval": _LC["tc"]},
        "orbit.b.cosi": {"initval": 0.0},
        "planet.b.radius": {"initval": 1.0},
        "transit.TESS_S48.baseline": {
            "initval": _LC["baseline"],
            "sigma": 0,
        },
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system, model, point, t, flux


def test_pinned_baseline_is_absent_from_the_point(
    pinned_baseline_transit_system,
):
    """
    Given a transit fit whose baseline is pinned with sigma: 0,
    When the plotting point is built from the model,
    Then the baseline is not in it -- so any plot helper reading the point
    needs a real fallback, and this fixture exercises that path.
    """
    system, _, point, _, _ = pinned_baseline_transit_system

    label = system.transit.baseline.label

    assert label not in point


def test_baseline_helper_returns_the_pinned_value(
    pinned_baseline_transit_system,
):
    """
    Given the same pinned-baseline fit,
    When _point_value is asked for the instrument's baseline,
    Then it returns the pinned raw-count value, not 1.0.
    """
    system, _, point, _, _ = pinned_baseline_transit_system

    base = system.transit._point_value(point, system.transit.baseline, 0)

    assert base == pytest.approx(_LC["baseline"], rel=1e-9)


def test_unphased_transit_model_uses_the_pinned_baseline(
    pinned_baseline_transit_system,
):
    """
    Given a transit fit on RAW COUNTS whose baseline is PINNED,
    When plot_data builds the unphased chart,
    Then the plotted model curve sits at that baseline, not at 1.0.

    Regression: _point_value's predecessor read the point with
    point.get(label, 1.0), and a pinned parameter is always absent from the
    draws, so the model curve plotted at unity -- 20000 counts below the
    data the likelihood actually fit.
    """
    system, _, point, _, flux = pinned_baseline_transit_system
    comp = system.transit

    specs = comp.plot_data(system, point)
    unphased = [s for s in specs if not s.meta["phase_folded"]][0]
    model_trace = [t for t in unphased.traces if t.role == "model"][0]

    # out-of-transit the curve is the baseline itself; in transit it dips by
    # at most the depth, so the whole curve lives within a percent of it
    assert np.max(model_trace.y) == pytest.approx(_LC["baseline"], rel=1e-9)
    assert np.min(model_trace.y) > 0.9 * _LC["baseline"]
    # the pre-fix value, explicitly excluded: the data are 20000x above it
    assert np.max(model_trace.y) > 0.5 * np.median(flux)


def test_phased_transit_data_uses_the_pinned_baseline(
    pinned_baseline_transit_system,
):
    """
    Given the same pinned-baseline fit,
    When plot_data builds the phased chart for the single planet,
    Then the cleaned flux is the data minus that baseline (one planet and
    no GP, so nothing else is removed) rather than the data minus 1.0.
    """
    system, _, point, _, flux = pinned_baseline_transit_system
    comp = system.transit

    specs = comp.plot_data(system, point)
    phased = [s for s in specs if s.meta["phase_folded"]]
    assert len(phased) == 1
    data_trace = [t for t in phased[0].traces if t.role == "data"][0]

    expected = flux - _LC["baseline"]
    np.testing.assert_allclose(
        np.sort(data_trace.y), np.sort(expected), atol=1e-6
    )
    # the pre-fix cleaned flux (data - 1.0) is off by the whole flux scale
    assert not np.allclose(data_trace.y, flux - 1.0, atol=1.0)


# ---------------------------------------------------------------------------
# 2.5.1: the phased panels must survive a point that does not carry tc/period
# ---------------------------------------------------------------------------
#
# The review item claimed a pinned (sigma: 0) orbit.tc is ABSENT from the
# point and so crashed the RV phased panel with float(None).  Re-verified
# 2026-08-18: it is not, because `orbit.tc` and `orbit.period` are declared
# `force_node: True`, which makes them pm.Deterministics -- present in
# model.deterministics, in get_internal_point and in the posterior -- even
# with every element pinned.  What the fix removes is the no-fallback read
# itself: both panels now go through Instrument._point_value, which is the
# same "value from the point, else the Parameter's initval" rule gamma and
# baseline already used.  These tests pin that contract on a point with the
# keys genuinely removed, which is the only state that could still reach it
# (a partial point, or a build where force_node is dropped).


def test_pinned_ephemeris_is_still_in_the_point(pinned_gamma_rv_system):
    """
    Given an RV fit built from a config with tc and period seeded,
    When the plotting point is built from the model,
    Then both are in it -- force_node makes them Deterministics, so the
    crash the item described cannot happen through that door.
    """
    system, _, point, _, _ = pinned_gamma_rv_system

    assert system.orbit.tc.label in point
    assert system.orbit.period.label in point


def test_phased_rv_panel_survives_a_point_without_tc(pinned_gamma_rv_system):
    """
    Given a point with orbit.tc and orbit.period removed entirely,
    When plot_data builds the RV charts,
    Then the phased panel is still produced, folded on the Parameters'
    own initvals rather than crashing on float(None).
    """
    system, _, point, _, _ = pinned_gamma_rv_system
    partial = {
        k: v
        for k, v in point.items()
        if k not in (system.orbit.tc.label, system.orbit.period.label)
    }

    specs = system.rvinstrument.plot_data(system, partial)

    phased = [s for s in specs if s.meta["phase_folded"]]
    assert len(phased) == 1
    assert phased[0].meta["period"] == pytest.approx(
        float(np.atleast_1d(system.orbit.period.initval)[0]), rel=1e-9
    )
    assert phased[0].meta["tc"] == pytest.approx(
        float(np.atleast_1d(system.orbit.tc.initval)[0]), rel=1e-9
    )


def test_point_value_falls_back_to_the_initval(pinned_gamma_rv_system):
    """
    Given a Parameter whose label is absent from the point,
    When Instrument._point_value is asked for one of its elements,
    Then it returns that element's initval, in internal units.
    """
    system, _, _, _, _ = pinned_gamma_rv_system
    comp = system.rvinstrument

    got = comp._point_value({}, system.orbit.tc, 0)

    assert got == pytest.approx(
        float(np.atleast_1d(system.orbit.tc.initval)[0]), rel=1e-12
    )
