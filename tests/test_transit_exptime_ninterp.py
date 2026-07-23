"""
Tests for exptime/ninterp exposure-time smearing on the transit component:
  - exptime/ninterp are fixed per-instrument properties read straight off
    the config (like self.files/self.band_names), not fitted parameters
  - ninterp==1 must reproduce the pre-existing instantaneous model exactly,
    regardless of exptime
  - a long exptime with ninterp>1 measurably smooths ingress/egress
  - mixed ninterp/exptime across instruments in one component is handled
    per-instrument (padded to the largest ninterp, inert padding weighted
    to zero), not by assuming a single uniform ninterp

All light curves here are synthetic (not the kelt4 example).
"""

import numpy as np
import pytensor
import pytest

from exozippy.system import System


def _write_lc(path, t, flux=None, err=None):
    """Write a whitespace-separated time/flux/err light curve file."""
    n = len(t)
    if flux is None:
        flux = np.ones(n)
    if err is None:
        err = np.full(n, 1e-3)
    np.savetxt(path, np.column_stack([t, flux, err]))
    return str(path)


def _config(lc_files, transit_extra=None):
    """One star/planet/orbit, one V band, N transit instruments on lc_files."""
    transit_extra = transit_extra or [{}] * len(lc_files)
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "V", "filter": "V", "ld_law": "quadratic"}],
        "transit": [
            {"name": f"inst{i}", "file": f, "band": "V", **extra}
            for i, (f, extra) in enumerate(zip(lc_files, transit_extra))
        ],
    }


def _params():
    """A short-period, near-central, hot-Jupiter-like synthetic system.

    Short period + near-zero impact parameter gives a fast, sharp-edged
    transit (ingress/egress ~11.5 min, total duration ~100 min at these
    values) so exposure smearing over tens of minutes has a clearly
    measurable effect.
    """
    return {
        "star.0.radius": {"initval": 1.0, "sigma": 0.05},
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": 0.5},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.001},
        "planet.0.radius": {"initval": 1.2},
    }


TC = 2459634.3
T14 = 0.0696  # days; total transit duration at the _params() geometry


def _plot_param_values(system):
    """Reproduce transit.py's own plot_params -> compiled-function argument
    conversion (see Transit.plot_unphased), so we can call the untouched
    compile_plotters path (no oversampling) as an independent reference."""
    return [
        float(np.squeeze(np.asarray(p.initval))) if getattr(p.value, "ndim", 0) == 0
        else np.atleast_1d(p.initval) for p in system.plot_params
    ]


def _model_flux_at_initial_point(system, model):
    """Evaluate transit's per-observation model-flux tensor at the model's
    initial point (same givens-substitution pattern already used in
    test_transit_band.py for transit.dilution). This is a plain attribute
    (Transit._model_flux_node), not a Deterministic -- it's (N_obs,), and
    registering it as a Deterministic would add N_obs floats per draw per
    chain to every trace, so it's deliberately kept out of the model graph's
    named/traced variables and only compiled here for testing."""
    ip = model.initial_point()
    givens = [(rv, np.asarray(ip[rv.name])) for rv in model.free_RVs if rv.name in ip]
    fn = pytensor.function(
        [], system.transit._model_flux_node, givens=givens,
        on_unused_input="ignore", mode="FAST_COMPILE")
    return fn()


def test_ninterp_one_matches_instantaneous_model(tmp_path_factory):
    """
    Given a transit instrument with ninterp=1 and a nonzero exptime,
    When the model is built,
    Then transit.model_flux exactly matches the instantaneous model from
    compile_plotters (which never oversamples) -- ninterp=1 must ignore
    exptime and short-circuit to the original single-point evaluation.
    """
    d = tmp_path_factory.mktemp("ninterp_one")
    t = np.linspace(TC - 0.55 * T14, TC + 0.55 * T14, 401)
    lc = _write_lc(d / "lc.dat", t)

    config = _config([lc], [{"exptime": 30.0, "ninterp": 1}])
    system = System(config, user_params=_params())
    system.prepare()
    model = system.build_model()

    model_flux = _model_flux_at_initial_point(system, model)

    decrement = system.transit._compiled_full_lc(
        system.transit.time, 0, *_plot_param_values(system))
    baseline = float(np.atleast_1d(system.transit.baseline.initval)[0])
    reference = baseline + decrement

    np.testing.assert_allclose(model_flux, reference, atol=1e-8)


def test_long_exptime_with_ninterp_smooths_ingress_egress(tmp_path_factory):
    """
    Given two otherwise-identical systems -- one with ninterp=1, one with a
    long exptime and ninterp=21 -- evaluated on the same fine time grid
    spanning one transit,
    When comparing the discrete slope (d flux / d t) of transit.model_flux,
    Then the oversampled model's steepest slope (at ingress/egress) is
    substantially smaller than the instantaneous model's: exposure smearing
    measurably softens the sharp ingress/egress edges.
    """
    d = tmp_path_factory.mktemp("ninterp_smooth")
    t = np.linspace(TC - 0.55 * T14, TC + 0.55 * T14, 801)

    lc_instant = _write_lc(d / "lc_instant.dat", t)
    config_instant = _config([lc_instant], [{"exptime": 30.0, "ninterp": 1}])
    system_instant = System(config_instant, user_params=_params())
    system_instant.prepare()
    model_instant = system_instant.build_model()
    flux_instant = _model_flux_at_initial_point(system_instant, model_instant)

    lc_smeared = _write_lc(d / "lc_smeared.dat", t)
    config_smeared = _config([lc_smeared], [{"exptime": 60.0, "ninterp": 21}])
    system_smeared = System(config_smeared, user_params=_params())
    system_smeared.prepare()
    model_smeared = system_smeared.build_model()
    flux_smeared = _model_flux_at_initial_point(system_smeared, model_smeared)

    dt = np.diff(t)
    max_slope_instant = np.max(np.abs(np.diff(flux_instant) / dt))
    max_slope_smeared = np.max(np.abs(np.diff(flux_smeared) / dt))

    assert max_slope_smeared < 0.5 * max_slope_instant


def test_mixed_ninterp_across_instruments_is_padded_per_instrument(tmp_path_factory):
    """
    Given two transit instruments in one component with different
    exptime/ninterp (inst0: ninterp=1; inst1: exptime=60min, ninterp=3),
    When load_data builds the oversampling grid,
    Then max_ninterp is the largest ninterp among instruments, inst0's rows
    use only their first column at weight 1 (padding weighted to zero), and
    inst1's rows get 3 evenly-spaced sub-times each weighted 1/3 -- each
    instrument's own ninterp/exptime is honored rather than assuming one
    uniform value for the whole component.
    """
    d = tmp_path_factory.mktemp("mixed_ninterp")
    t0 = np.linspace(TC - 0.1, TC + 0.1, 5)
    t1 = np.linspace(TC - 0.1, TC + 0.1, 5)
    lc0 = _write_lc(d / "lc0.dat", t0)
    lc1 = _write_lc(d / "lc1.dat", t1)

    config = _config(
        [lc0, lc1],
        [{"ninterp": 1}, {"exptime": 60.0, "ninterp": 3}],
    )
    system = System(config, user_params=_params())
    system.prepare()

    tr = system.transit
    assert tr.max_ninterp == 3
    assert tr.oversample_time.shape == (10, 3)
    assert tr.oversample_weights.shape == (10, 3)

    inst0_rows = tr.inst_map == 0
    inst1_rows = tr.inst_map == 1

    np.testing.assert_allclose(
        tr.oversample_weights[inst0_rows], np.tile([1.0, 0.0, 0.0], (5, 1)))
    np.testing.assert_allclose(
        tr.oversample_time[inst0_rows, 0], t0)
    # Padding columns collapse onto the timestamp itself (finite, inert).
    np.testing.assert_allclose(
        tr.oversample_time[inst0_rows, 1:], np.tile(t0[:, None], (1, 2)))

    exptime_days = 60.0 / 1440.0
    expected_offsets = np.array([-0.5, 0.0, 0.5]) * exptime_days
    np.testing.assert_allclose(
        tr.oversample_weights[inst1_rows], np.tile([1 / 3, 1 / 3, 1 / 3], (5, 1)))
    np.testing.assert_allclose(
        tr.oversample_time[inst1_rows], t1[:, None] + expected_offsets[None, :])


def test_mixed_ninterp_model_flux_matches_instantaneous_only_for_ninterp_one(
        tmp_path_factory):
    """
    Given one mixed-ninterp component (inst0: ninterp=1; inst1: long
    exptime, ninterp=21), both instruments sampling the same fine time grid
    across one transit,
    When transit.model_flux is evaluated,
    Then inst0's rows match its own instantaneous (compile_plotters)
    reference to floating-point precision, while inst1's rows do not --
    proving the padded/weighted oversampling actually reaches the model's
    per-observation output correctly for each instrument independently,
    not just the grid/weight arrays checked in the test above.
    """
    d = tmp_path_factory.mktemp("mixed_ninterp_flux")
    t = np.linspace(TC - 0.55 * T14, TC + 0.55 * T14, 401)
    lc0 = _write_lc(d / "lc0.dat", t)
    lc1 = _write_lc(d / "lc1.dat", t)

    config = _config(
        [lc0, lc1],
        [{"exptime": 30.0, "ninterp": 1}, {"exptime": 60.0, "ninterp": 21}],
    )
    system = System(config, user_params=_params())
    system.prepare()
    model = system.build_model()

    model_flux = _model_flux_at_initial_point(system, model)

    tr = system.transit
    inst0_rows = tr.inst_map == 0
    inst1_rows = tr.inst_map == 1

    param_values = _plot_param_values(system)
    baseline = np.atleast_1d(tr.baseline.initval)

    ref0 = baseline[0] + tr._compiled_full_lc(t, 0, *param_values)
    ref1 = baseline[1] + tr._compiled_full_lc(t, 1, *param_values)

    # inst0 (ninterp=1): exact match, exptime is ignored.
    np.testing.assert_allclose(model_flux[inst0_rows], ref0, atol=1e-8)

    # inst1 (ninterp=21, exptime=60min): smeared, so it must differ
    # measurably from its own instantaneous reference.
    max_diff_inst1 = np.max(np.abs(model_flux[inst1_rows] - ref1))
    assert max_diff_inst1 > 1e-4
