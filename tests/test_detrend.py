"""Detrending: the fitted trend in the plots, and the whitened design matrix.

Two review items live here.

1.5.1 -- the plotted model omitted the fitted detrend model entirely: the
likelihood adds ``pt.dot(X, c)`` per observation, but no plotted model node
carried it and the plotted data were never corrected, so any fit with active
detrend columns showed a systematic data-vs-model mismatch equal to the whole
fitted trend.  The fix is EXOFASTv2's: subtract the fitted trend from the
plotted DATA (a pretty-grid model curve cannot carry a per-observation
quantity).

6.5.2 -- the design matrix columns were used as read from the file.  A
nonzero-mean column is exactly degenerate with the instrument offset along
its mean direction, so the columns are now WHITENED per (instrument, column)
at ingestion, and the coefficient is reported back in raw units.
"""

import numpy as np
import pytest

from exozippy.system import System

# One detrend column with a large nonzero mean (an airmass-like basis
# vector, exactly the shape examples/gj1214 ships): the mean is what makes
# the raw column degenerate with the offset, and the amplitude is what makes
# an omitted trend visible in a plot.
_X_MEAN = 1.25
_X_AMP = 0.08
_COEFF = 300.0  # m/s per unit column, big enough to dominate the RV scatter
_RV = dict(gamma=25.0, K=60.0, P=8.0, tc=2450004.0, err=5.0)


def _write_rv_file(path, n=48):
    t = np.linspace(2450000.0, 2450080.0, n)
    x = _X_MEAN + _X_AMP * np.sin(2 * np.pi * np.arange(n) / 11.0)
    rv = (
        _RV["gamma"]
        + _RV["K"] * np.sin(2 * np.pi * (t - _RV["tc"]) / _RV["P"])
        + _COEFF * (x - _X_MEAN)
    )
    np.savetxt(path, np.column_stack([t, rv, np.full_like(t, _RV["err"]), x]))
    return t, rv, x


@pytest.fixture(scope="module")
def detrended_rv(tmp_path_factory):
    """Built one-instrument RV system with a single detrend column.

    The coefficient is PINNED so its value is known exactly; that also
    exercises _point_value's fallback, since a fully pinned vector never
    becomes a pm.Deterministic.
    """
    tmp_dir = tmp_path_factory.mktemp("detrend_rv")
    path = tmp_dir / "detrended.rv"
    t, rv, x = _write_rv_file(path)

    config = {
        "run": {"name": "detrend_rv"},
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
        "orbit.b.period": {"initval": _RV["P"]},
        "orbit.b.tc": {"initval": _RV["tc"]},
        "rvinstrument.detrend_coeffs": {"initval": 0.5, "sigma": 0},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.compile_plotter_functions(model)
    return system, point, t, rv, x


# ---------------------------------------------------------------------------
# 6.5.2: the design matrix is whitened per (instrument, column)
# ---------------------------------------------------------------------------


def test_detrend_columns_are_whitened_at_ingestion(detrended_rv):
    """
    Given a detrend column with a large nonzero mean and a nonunit scale,
    When the instrument's design matrix is built,
    Then that column has mean 0 and standard deviation 1 -- the mean is
    what was exactly degenerate with the instrument offset, and the scale
    is what left the coefficient badly conditioned for the sampler.
    """
    system, _, _, _, x = detrended_rv
    col = system.rvinstrument.detrend_matrix[:, 0]

    assert np.mean(col) == pytest.approx(0.0, abs=1e-12)
    assert np.std(col) == pytest.approx(1.0, rel=1e-12)
    # ... and it is still the SAME basis vector, only rescaled
    np.testing.assert_allclose(col, (x - np.mean(x)) / np.std(x), atol=1e-12)


def test_whitening_is_per_instrument_block(detrended_rv):
    """
    Given two instruments whose single detrend columns have very different
    moments,
    When the block-diagonal matrix is built,
    Then each block is whitened against its OWN moments -- a global moment
    would couple blocks the design is block-diagonal precisely to keep
    independent.
    """
    system, _, _, _, _ = detrended_rv
    a = np.array([[1.0], [3.0]])  # mean 2, std 1
    b = np.array([[100.0], [140.0]])  # mean 120, std 20

    matrix, per_inst, total, scales = system.rvinstrument._build_block_detrend(
        [a, b], 4
    )

    assert per_inst == [1, 1]
    assert total == 2
    np.testing.assert_allclose(matrix[:2, 0], [-1.0, 1.0])
    np.testing.assert_allclose(matrix[2:, 1], [-1.0, 1.0])
    np.testing.assert_allclose(scales, [1.0, 20.0])


def test_a_constant_detrend_column_is_refused(detrended_rv):
    """
    Given a detrend column with zero variance,
    When the block-diagonal matrix is built,
    Then it RAISES naming the column.

    A constant column carries no information and is exactly degenerate with
    the instrument offset, so there is nothing to estimate; mean-subtracting
    it instead would leave an all-zero basis vector whose coefficient the
    likelihood never sees, and dividing by an epsilon would invent one.
    """
    system, _, _, _, _ = detrended_rv

    with pytest.raises(ValueError, match="constant"):
        system.rvinstrument._build_block_detrend([np.full((5, 1), 2.5)], 5)


def test_detrend_coefficient_is_reported_in_raw_units(detrended_rv):
    """
    Given a whitened design matrix,
    When the coefficient Parameter converts internal -> user,
    Then the reported number is the coefficient per RAW column unit --
    the sampled one divided by that column's standard deviation.

    Sample whitened, report un-whitened: the conversion goes through
    Parameter.from_internal like every other unit change, so nothing hand
    writes the factor at a call site.
    """
    system, _, _, _, x = detrended_rv
    coeffs = system.rvinstrument.detrend_coeffs

    reported = coeffs.from_internal(np.array([1.0]))

    assert float(np.atleast_1d(reported)[0]) == pytest.approx(
        1.0 / np.std(x), rel=1e-12
    )


def test_user_bounds_are_pushed_through_the_same_map(detrended_rv):
    """
    Given the coefficient's raw-unit bounds from defaults.yaml,
    When the Parameter stores them internally,
    Then they are the raw bounds times the column's standard deviation --
    a stated prior keeps its meaning under the change of coordinate.
    """
    system, _, _, _, x = detrended_rv
    coeffs = system.rvinstrument.detrend_coeffs

    assert float(np.atleast_1d(coeffs.lower)[0]) == pytest.approx(
        -1.0e6 * np.std(x), rel=1e-12
    )
    assert float(np.atleast_1d(coeffs.upper)[0]) == pytest.approx(
        1.0e6 * np.std(x), rel=1e-12
    )


# ---------------------------------------------------------------------------
# 1.5.1: the fitted trend reaches the plots
# ---------------------------------------------------------------------------


def test_detrend_at_data_is_the_fitted_trend(detrended_rv):
    """
    Given a point carrying (here, pinning) the detrend coefficient,
    When Instrument.detrend_at_data is evaluated,
    Then it is the design matrix times that coefficient -- nonzero, and an
    exact affine image of the raw column.
    """
    system, point, _, _, x = detrended_rv
    comp = system.rvinstrument

    trend = comp.detrend_at_data(point)

    assert trend.shape == (comp.n_total_obs,)
    assert np.ptp(trend) > 0.0
    np.testing.assert_allclose(
        trend, comp.detrend_matrix @ comp.detrend_coeffs.initval, atol=1e-14
    )
    # ... and it is exactly the raw column up to an affine map
    assert abs(np.corrcoef(trend, x)[0, 1]) == pytest.approx(1.0, abs=1e-12)


def test_unphased_rv_data_are_detrend_corrected(detrended_rv):
    """
    Given an RV fit with an active detrend column,
    When plot_data builds the unphased chart,
    Then the plotted data have the fitted trend removed as well as gamma.

    Regression: they had only gamma removed, so the panel showed the whole
    fitted trend as unmodeled residual structure.
    """
    system, point, _, rv, _ = detrended_rv
    comp = system.rvinstrument
    factor = comp._rv_factor()

    specs = comp.plot_data(system, point)
    unphased = [s for s in specs if not s.meta["phase_folded"]][0]
    data_trace = [t for t in unphased.traces if t.role == "data"][0]

    g = comp._point_value(point, comp.gamma, 0)
    trend = comp.detrend_at_data(point)
    np.testing.assert_allclose(
        data_trace.y, (comp.rv - g - trend) * factor, atol=1e-9
    )
    # the pre-fix value, explicitly excluded
    assert not np.allclose(data_trace.y, (comp.rv - g) * factor, atol=1e-3)


def test_phased_rv_data_are_detrend_corrected(detrended_rv):
    """
    Given the same fit,
    When plot_data builds the phased chart,
    Then its cleaned data have the fitted trend removed too (one orbit, no
    GP, so gamma plus the trend is the whole cleaning).
    """
    system, point, _, _, _ = detrended_rv
    comp = system.rvinstrument
    factor = comp._rv_factor()

    specs = comp.plot_data(system, point)
    phased = [s for s in specs if s.meta["phase_folded"]]
    assert len(phased) == 1
    data_trace = [t for t in phased[0].traces if t.role == "data"][0]

    g = comp._point_value(point, comp.gamma, 0)
    trend = comp.detrend_at_data(point)
    expected = (comp.rv - g - trend) * factor
    np.testing.assert_allclose(
        np.sort(data_trace.y), np.sort(expected), atol=1e-9
    )


def test_detrend_coefficients_are_a_param_dep(detrended_rv):
    """
    Given the detrend correction is applied to the data in numpy,
    When plot_data declares its param_deps,
    Then the coefficient label is among them -- the graph walk over the
    symbolic model node cannot see a numpy correction, so without this a
    GUI slider on the coefficients would never refresh the chart.
    """
    system, point, _, _, _ = detrended_rv
    comp = system.rvinstrument

    label = comp.detrend_coeffs.label
    specs = comp.plot_data(system, point)

    assert comp.detrend_dep_labels() == [label]
    for spec in specs:
        assert label in spec.param_deps


_LC = dict(baseline=1.0, depth=0.01, P=3.0, tc=2459000.0, err=3.0e-4)


@pytest.fixture(scope="module")
def detrended_transit(tmp_path_factory):
    """Built one-instrument transit system with a single detrend column.

    Same shape as examples/gj1214's ground-based light curves: time, flux,
    error, then an airmass-like column with a large nonzero mean.
    """
    tmp_dir = tmp_path_factory.mktemp("detrend_lc")
    path = tmp_dir / "detrended.TESS.dat"

    n = 200
    t = np.linspace(_LC["tc"] - 0.15, _LC["tc"] + 0.15, n)
    x = _X_MEAN + _X_AMP * np.sin(2 * np.pi * np.arange(n) / 37.0)
    in_transit = np.abs(t - _LC["tc"]) < 0.03
    flux = _LC["baseline"] - _LC["depth"] * in_transit + 0.02 * (x - _X_MEAN)
    np.savetxt(
        path, np.column_stack([t, flux, np.full_like(t, _LC["err"]), x])
    )

    config = {
        "run": {"name": "detrend_lc"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "TESS", "file": str(path), "band": "TESS"}],
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "orbit.b.period": {"initval": _LC["P"]},
        "orbit.b.tc": {"initval": _LC["tc"]},
        "transit.detrend_coeffs": {"initval": 0.01, "sigma": 0},
    }

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.compile_plotter_functions(model)
    return system, point, flux, x


def test_unphased_transit_data_are_detrend_corrected(detrended_transit):
    """
    Given a transit fit with an active detrend column,
    When plot_data builds the unphased chart,
    Then the plotted flux has the fitted trend removed.

    Regression: the raw flux was plotted against a model curve that omitted
    the trend, so the whole fitted trend read as residual structure.
    """
    system, point, flux, _ = detrended_transit
    comp = system.transit

    specs = comp.plot_data(system, point)
    unphased = [s for s in specs if not s.meta["phase_folded"]][0]
    data_trace = [t for t in unphased.traces if t.role == "data"][0]

    trend = comp.detrend_at_data(point)
    np.testing.assert_allclose(data_trace.y, comp.flux - trend, atol=1e-12)
    assert not np.allclose(data_trace.y, comp.flux, atol=1e-6)
    # the data trace now moves with the point, so the renderer must re-ship it
    assert unphased.meta["dynamic_data"] is True


def test_phased_transit_data_are_detrend_corrected(detrended_transit):
    """
    Given the same fit,
    When plot_data builds the phased chart,
    Then the cleaned flux has the fitted trend removed along with the
    baseline (one planet, no GP, so that is the whole cleaning).
    """
    system, point, _, _ = detrended_transit
    comp = system.transit

    specs = comp.plot_data(system, point)
    phased = [s for s in specs if s.meta["phase_folded"]]
    assert len(phased) == 1
    data_trace = [t for t in phased[0].traces if t.role == "data"][0]

    baseline = comp._point_value(point, comp.baseline, 0)
    trend = comp.detrend_at_data(point)
    expected = comp.flux - baseline - trend
    np.testing.assert_allclose(
        np.sort(data_trace.y), np.sort(expected), atol=1e-9
    )
