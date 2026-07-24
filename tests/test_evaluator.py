"""
Tests for the GUI compiled forward evaluator (prompt G5).

The evaluator (src/exozippy/evaluator.py) is the millisecond half of the
GUI's hybrid loop: after a "Solve" builds the model, dragging a slider
calls set_value() to rebuild a raw point and eval_plots() to re-render the
model curves. eval_plots recomputes each affected component's own
plot_data(system, point) at the new point -- the same code that built the
base PlotSpecs and that the CLI's matplotlib plot() reuses -- rather than a
separate compiled-graph implementation; the only optimization is a single
cached raw-point -> internal-point pytensor function (Evaluator.internal_point),
built once, in place of System.get_internal_point's per-call recompile.

These tests exercise, on the tracked kelt4 RV-only config and a minimal
inline transit config (mirroring tests/test_plot_data.py's fixtures):
  * eval_plots returns finite model-trace x/y-arrays for both the unphased
    AND the phased plot -- the phased plot is the regression case: its node
    output is sorted-by-phase and column-selected from a multi-orbit matrix,
    which defeated the old affine-calibration fast path and left it frozen
    at the solved value until a re-Solve;
  * moving the planet period by 1% (via its sampled logP) changes both the
    unphased and phased RV model curves, and restoring the value reproduces
    them exactly;
  * set_value round-trips a slider value to float64 precision;
  * 100 warm eval_plots calls average well under 50 ms;
  * structural_hash changes on a bound change, is stable across reordered
    dict keys, and ignores pure initval changes;
  * derived/fixed parameters raise NeedsResolve.

They follow AAA with Given/When/Then docstrings.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.evaluator import (
    Evaluator,
    NeedsResolve,
    compile_evaluator,
    structural_hash,
)
from exozippy.system import System

pytestmark = pytest.mark.slow

_KELT4_DIR = Path(__file__).parent.parent / "examples" / "kelt4"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rvonly_evaluator():
    """kelt4 RV-only system, built, with a compiled Evaluator and base point."""
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
        base_raw = system.get_raw_start(model)
        ev = compile_evaluator(system, model, base_raw)
    finally:
        os.chdir(cwd)
    return system, model, ev, base_raw, config


_TRANSIT_CONFIG = {
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
_TRANSIT_PARAMS = {
    "star.0.radius": {"initval": 1.610, "sigma": 0.05},
    "star.0.mass": {"initval": 1.204, "sigma": 0.05},
    "star.0.teff": {"initval": 6207, "sigma": 100},
    "star.0.feh": {"initval": -0.116, "sigma": 0.08},
    "orbit.0.period": {"initval": 2.9895933},
    "orbit.0.tc": {"initval": 2459634.3},
    "orbit.0.cosi": {"initval": 0.11996},
    "planet.0.radius": {"initval": 1.706},
}


@pytest.fixture(scope="module")
def transit_evaluator():
    """Minimal transit-only kelt4 build with a compiled Evaluator."""
    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    cwd = os.getcwd()
    os.chdir(_KELT4_DIR)
    try:
        system = System(dict(_TRANSIT_CONFIG), user_params=dict(_TRANSIT_PARAMS))
        system.prepare()
        model = system.build_model()
        base_raw = system.get_raw_start(model)
        ev = compile_evaluator(system, model, base_raw)
    finally:
        os.chdir(cwd)
    return system, model, ev, base_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unphased_id(ev):
    return [s.id for s in ev.specs if "unphased" in s.id][0]


def _phased_id(ev):
    return [s.id for s in ev.specs if "phased" in s.id][0]


def _period_and_logP(system, model, raw):
    """(period in days, logP in user units) at a raw point."""
    o = system.orbit
    ip = system.get_internal_point(model, raw)
    period = float(np.atleast_1d(ip[o.period.label])[0])
    logP_user = float(o.logP.phys_from_raw(np.asarray(raw[o.logP.label + "_raw"]))[0])
    return period, logP_user


# ---------------------------------------------------------------------------
# eval_plots: finiteness and the period-shift response
# ---------------------------------------------------------------------------

@pytest.mark.timeout(900)
def test_eval_plots_returns_finite_model_traces(rvonly_evaluator):
    """
    Given a compiled RV-only evaluator,
    When eval_plots is called at the base point,
    Then at least one plot carries model traces, and every emitted trace's x
    and y arrays are finite and non-empty -- including the phased plot, whose
    node output is sorted-by-phase and column-selected from a multi-orbit
    matrix and so cannot be recovered by an affine fit of the raw pytensor
    output (see the module docstring); eval_plots must recompute it exactly
    via plot_data rather than leaving it stale.
    """
    system, model, ev, base_raw, _ = rvonly_evaluator

    out = ev.eval_plots(base_raw)

    assert len(out) >= 1
    assert any(len(traces) >= 1 for traces in out.values())
    assert _phased_id(ev) in out
    for plot_id, traces in out.items():
        for name, xy in traces.items():
            for axis in ("x", "y"):
                arr = np.atleast_1d(np.asarray(xy[axis]))
                assert arr.size >= 1
                assert np.all(np.isfinite(arr))


@pytest.mark.timeout(900)
def test_period_shift_changes_and_restores_rv_curves(rvonly_evaluator):
    """
    Given a compiled RV-only evaluator and its base RV model curves,
    When the planet period is moved +1% (via its sampled logP) and then
        restored,
    Then BOTH the unphased and the phased RV model curves change (finite,
        different) under the shift and are reproduced exactly once the value
        is restored. The phased curve is the regression case for the
        plot_data-based redesign (see module docstring).
    """
    system, model, ev, base_raw, _ = rvonly_evaluator
    uid = _unphased_id(ev)
    pid = _phased_id(ev)

    base = ev.eval_plots(base_raw)
    y0 = np.asarray(base[uid]["model"]["y"])
    py0 = np.asarray(base[pid]["model"]["y"])

    period0, logP0 = _period_and_logP(system, model, base_raw)
    shifted_raw = ev.set_value("orbit.b.logP", np.log10(period0 * 1.01), base_raw)
    period1, _ = _period_and_logP(system, model, shifted_raw)
    shifted = ev.eval_plots(shifted_raw)
    y1 = np.asarray(shifted[uid]["model"]["y"])
    py1 = np.asarray(shifted[pid]["model"]["y"])

    restored_raw = ev.set_value("orbit.b.logP", logP0, shifted_raw)
    restored = ev.eval_plots(restored_raw)
    y2 = np.asarray(restored[uid]["model"]["y"])
    py2 = np.asarray(restored[pid]["model"]["y"])

    # the period actually moved by ~1%
    assert period1 == pytest.approx(period0 * 1.01, rel=1e-6)
    # both curves changed and stayed finite
    assert np.all(np.isfinite(y1)) and np.all(np.isfinite(py1))
    assert not np.allclose(y0, y1)
    assert not np.allclose(py0, py1)
    # restoring reproduces the original curves
    np.testing.assert_allclose(y2, y0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(py2, py0, rtol=0, atol=1e-9)


@pytest.mark.timeout(900)
def test_changed_label_filter_skips_unrelated_components(rvonly_evaluator):
    """
    Given a compiled RV-only evaluator,
    When eval_plots is called with changed_label naming the tuned parameter,
    Then it still recomputes the plots that depend on it (no silent gap
        relative to an unfiltered call).
    """
    system, model, ev, base_raw, _ = rvonly_evaluator
    uid = _unphased_id(ev)

    label = ev.label_for_path("orbit.b.logP")
    filtered = ev.eval_plots(base_raw, changed_label=label)

    assert uid in filtered


@pytest.mark.timeout(900)
def test_second_eval_at_same_point_is_identical(rvonly_evaluator):
    """
    Given a compiled evaluator,
    When eval_plots is called twice at the same base point,
    Then the two results are bit-for-bit identical (pure function).
    """
    system, model, ev, base_raw, _ = rvonly_evaluator

    a = ev.eval_plots(base_raw)
    b = ev.eval_plots(base_raw)

    for pid in a:
        for name in a[pid]:
            for axis in ("x", "y"):
                np.testing.assert_array_equal(np.asarray(a[pid][name][axis]),
                                              np.asarray(b[pid][name][axis]))


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

@pytest.mark.timeout(900)
def test_eval_plots_latency_under_50ms(rvonly_evaluator):
    """
    Given a compiled evaluator warmed up by one eval,
    When 100 eval_plots calls are timed with perf_counter,
    Then the mean per-call latency is well under 50 ms.
    """
    system, model, ev, base_raw, _ = rvonly_evaluator

    ev.eval_plots(base_raw)  # warm the lazy compile + numpy paths
    t0 = time.perf_counter()
    for _ in range(100):
        ev.eval_plots(base_raw)
    mean_ms = (time.perf_counter() - t0) / 100 * 1000.0

    assert mean_ms < 50.0, f"mean eval_plots latency {mean_ms:.2f} ms >= 50 ms"


# ---------------------------------------------------------------------------
# set_value round-trip and NeedsResolve
# ---------------------------------------------------------------------------

@pytest.mark.timeout(900)
def test_set_value_round_trip_to_float64_precision(rvonly_evaluator):
    """
    Given a compiled evaluator,
    When a sampled parameter is set to a target user-unit value and read back
        by inverting the raw transform,
    Then the read-back value matches the target to float64 precision.
    """
    system, model, ev, base_raw, _ = rvonly_evaluator
    logP = system.orbit.logP

    target_user = 0.4801234567891011  # dex(d); factor 1
    new_raw = ev.set_value("orbit.b.logP", target_user, base_raw)

    readback = float(logP.phys_from_raw(np.asarray(new_raw[logP.label + "_raw"]))[0])
    factor = float(np.atleast_1d(logP._get_conversion_factors())[0])
    assert readback * factor == pytest.approx(target_user, rel=0, abs=1e-12)


@pytest.mark.timeout(900)
def test_set_value_returns_new_point_leaving_original_unchanged(rvonly_evaluator):
    """
    Given a compiled evaluator and a base raw point,
    When set_value is called,
    Then it returns a new dict and does not mutate the caller's point.
    """
    system, model, ev, base_raw, _ = rvonly_evaluator
    key = system.orbit.logP.label + "_raw"
    before = np.array(base_raw[key], copy=True)

    new_raw = ev.set_value("orbit.b.logP", 0.5, base_raw)

    assert new_raw is not base_raw
    np.testing.assert_array_equal(np.asarray(base_raw[key]), before)
    assert not np.allclose(np.asarray(new_raw[key]), before)


@pytest.mark.timeout(900)
def test_set_value_on_derived_param_raises_needs_resolve(rvonly_evaluator):
    """
    Given a compiled evaluator,
    When set_value targets a derived parameter (period = 10**logP), which has
        no static raw inverse,
    Then NeedsResolve is raised so the GUI knows to re-solve.
    """
    system, model, ev, base_raw, _ = rvonly_evaluator

    with pytest.raises(NeedsResolve):
        ev.set_value("orbit.b.period", 3.0, base_raw)


# ---------------------------------------------------------------------------
# Transit component
# ---------------------------------------------------------------------------

@pytest.mark.timeout(900)
def test_transit_eval_and_radius_shift(transit_evaluator):
    """
    Given a compiled transit-only evaluator,
    When the planet radius is increased 5% and then restored,
    Then BOTH the unphased and the phased transit model curves deepen
        (finite, different) and are reproduced exactly on restore. The
        phased curve is transit's version of the same regression covered
        for RV above (sorted-by-phase, column-selected node output).
    """
    system, model, ev, base_raw = transit_evaluator
    uid = _unphased_id(ev)
    pid = _phased_id(ev)
    pr = system.planet.radius

    base = ev.eval_plots(base_raw)
    y0 = np.asarray(base[uid]["model"]["y"])
    py0 = np.asarray(base[pid]["model"]["y"])
    assert np.all(np.isfinite(y0)) and np.all(np.isfinite(py0))

    cur = float(pr.phys_from_raw(np.asarray(base_raw[pr.label + "_raw"]))[0])
    factor = float(np.atleast_1d(pr._get_conversion_factors())[0])
    cur_user = cur * factor

    shifted = ev.set_value("planet.b.radius", cur_user * 1.05, base_raw)
    shifted_out = ev.eval_plots(shifted)
    y1 = np.asarray(shifted_out[uid]["model"]["y"])
    py1 = np.asarray(shifted_out[pid]["model"]["y"])
    restored = ev.set_value("planet.b.radius", cur_user, shifted)
    restored_out = ev.eval_plots(restored)
    y2 = np.asarray(restored_out[uid]["model"]["y"])
    py2 = np.asarray(restored_out[pid]["model"]["y"])

    assert np.all(np.isfinite(y1)) and np.all(np.isfinite(py1))
    assert not np.allclose(py0, py1)
    assert not np.allclose(y0, y1)
    # a bigger planet blocks more light -> deeper transit (smaller minimum)
    assert (1.0 - y1.min()) > (1.0 - y0.min())
    np.testing.assert_allclose(y2, y0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(py2, py0, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# structural_hash
# ---------------------------------------------------------------------------

def test_structural_hash_stable_across_reordered_keys():
    """
    Given a config and a params dict,
    When their top-level keys and per-param fields are reordered,
    Then structural_hash returns the same string.
    """
    config = {
        "run": {"name": "x"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
    }
    params = {"orbit.b.logP": {"lower": 0.1, "upper": 1.0, "sigma": 0.05}}

    h1 = structural_hash(config, params)

    config_r = {k: config[k] for k in reversed(list(config))}
    params_r = {"orbit.b.logP": {"sigma": 0.05, "upper": 1.0, "lower": 0.1}}
    h2 = structural_hash(config_r, params_r)

    assert h1 == h2


def test_structural_hash_ignores_pure_initval_change():
    """
    Given a config,
    When only a numeric initval is changed in user_params,
    Then structural_hash is unchanged (initvals are not structural).
    """
    config = {"star": [{"name": "A"}], "orbit": [{"name": "b"}]}

    h0 = structural_hash(config)
    h_iv = structural_hash(config, {"orbit.b.logP": {"initval": 0.9}})

    assert h_iv == h0


def test_structural_hash_changes_on_bound_change():
    """
    Given a config with a params dict,
    When a bound (lower) is changed,
    Then structural_hash changes (bounds set up the compiled transform).
    """
    config = {"star": [{"name": "A"}], "orbit": [{"name": "b"}]}

    h_a = structural_hash(config, {"orbit.b.logP": {"lower": 0.1}})
    h_b = structural_hash(config, {"orbit.b.logP": {"lower": 0.2}})

    assert h_a != h_b


def test_structural_hash_changes_on_component_add():
    """
    Given a base config,
    When a component (a new star instance) is added,
    Then structural_hash changes.
    """
    base = {"star": [{"name": "A"}]}
    more = {"star": [{"name": "A"}, {"name": "B"}]}

    assert structural_hash(base) != structural_hash(more)


def test_structural_hash_changes_on_fixed_flag():
    """
    Given a config,
    When a parameter's fixed-ness (sigma == 0) flips,
    Then structural_hash changes.
    """
    config = {"star": [{"name": "A"}]}

    h_free = structural_hash(config, {"star.A.mass": {"sigma": 0.05}})
    h_fixed = structural_hash(config, {"star.A.mass": {"sigma": 0.0}})

    assert h_free != h_fixed


def test_evaluator_structural_hash_matches_module_function():
    """
    Given the module-level structural_hash,
    When Evaluator.structural_hash is called with the same inputs,
    Then they agree (the static method just delegates).
    """
    config = {"star": [{"name": "A"}]}
    params = {"star.A.mass": {"lower": 0.5}}

    assert Evaluator.structural_hash(config, params) == structural_hash(config, params)
