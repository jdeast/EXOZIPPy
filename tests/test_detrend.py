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

3.14.10 -- and because the whitened matrix is DIMENSIONLESS, the coefficient
carries the units of whatever the dot product feeds.  rvinstrument declared
none at all while adding its term to a model in the internal solRad/d, so an
RV detrend coefficient was reported 8052.0833x too small and in the wrong
unit.  The rule, and the fact that the three declaring components correctly
disagree with each other, are pinned at the bottom of this file.

1.6.7 -- mulensinstrument scored its (multiplicative) trend and plotted
neither it nor its removal.  The fix is architectural: one Instrument-base
mechanism (DETREND_SPACES / _detrended_model / detrend_corrected), and a
parity test that runs over every detrending child.  Last section.
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
    the sampled one divided by that column's standard deviation, and in
    m/s (see the unit test below).

    Sample whitened, report un-whitened: the conversion goes through
    Parameter.from_internal like every other unit change, so nothing hand
    writes the factor at a call site.
    """
    system, _, _, _, x = detrended_rv
    comp = system.rvinstrument
    coeffs = comp.detrend_coeffs

    reported = coeffs.from_internal(np.array([1.0]))

    assert float(np.atleast_1d(reported)[0]) == pytest.approx(
        comp._rv_factor() / np.std(x), rel=1e-12
    )


def test_user_bounds_are_pushed_through_the_same_map(detrended_rv):
    """
    Given the coefficient's raw-unit bounds from defaults.yaml,
    When the Parameter stores them internally,
    Then they are the raw bounds times the column's standard deviation and
    divided by the m/s-per-internal factor -- a stated prior keeps its
    meaning under BOTH halves of the change of coordinate.
    """
    system, _, _, _, x = detrended_rv
    comp = system.rvinstrument
    coeffs = comp.detrend_coeffs
    expected = 1.0e6 * np.std(x) / comp._rv_factor()

    assert float(np.atleast_1d(coeffs.lower)[0]) == pytest.approx(
        -expected, rel=1e-12
    )
    assert float(np.atleast_1d(coeffs.upper)[0]) == pytest.approx(
        expected, rel=1e-12
    )


# ---------------------------------------------------------------------------
# 3.14.10: the coefficient's unit is the unit of what the dot product feeds
# ---------------------------------------------------------------------------


def test_an_rv_detrend_coefficient_is_reported_in_m_per_s(detrended_rv):
    """
    Given a detrend column carrying an injected trend of a KNOWN size in
      m/s per unit of the raw column,
    When the Parameter converts the internal coefficient that reproduces
      that trend back to user units,
    Then the reported number is that size, in m/s.

    Review 3.14.10.  `rv_model += pt.dot(detrend, c)` and rv_model is in
    the internal solRad/d, so the coefficient is too -- but it was declared
    `unit: ""` / `internal_unit: ""`, and the table therefore printed the
    coefficient in solRad/d.  The error is exactly the m/s-per-solRad/d
    factor 8052.0833...: this fit reported 0.0373 for a real 300 m/s.
    """
    system, _, _, _, x = detrended_rv
    comp = system.rvinstrument
    coeffs = comp.detrend_coeffs

    # The internal coefficient that reproduces the injected trend: the
    # trend is _COEFF * (x - mean) m/s = _COEFF * std(x) * whitened_col,
    # and rv_model carries it in solRad/d.
    internal = _COEFF * np.std(x) / comp._rv_factor()

    reported = float(
        np.atleast_1d(coeffs.from_internal(np.array([internal])))[0]
    )

    assert reported == pytest.approx(_COEFF, rel=1e-12)
    # The pre-fix value, explicitly excluded.
    assert reported != pytest.approx(_COEFF / comp._rv_factor(), rel=1e-6)


def test_the_declared_units_name_what_the_dot_product_feeds(detrended_rv):
    """
    Given the three components that declare detrend_coeffs,
    When their declarations are read,
    Then each names the unit of the model term its dot product is added
      to -- which is NOT the same unit for all three.

    The rule, pinned as a fact rather than as prose: the design matrix is
    whitened at ingestion and so dimensionless, which leaves the
    coefficient carrying the units of whatever the dot product feeds.
    mulensinstrument's `mag` is the trap this guards -- its coefficient
    sits inside 10**(-0.4*...) so it is in MAGNITUDES even though the
    component models flux, and an audit that "fixes" it to flux breaks
    mulens detrending silently.
    """
    import pathlib

    import exozippy.components
    from exozippy.yamlio import load_yaml

    root = pathlib.Path(exozippy.components.__file__).parent
    declared = {}
    for path in root.rglob("defaults.yaml"):
        for prefix, block in (load_yaml(str(path)) or {}).items():
            entry = (block or {}).get("detrend_coeffs")
            if isinstance(entry, dict):
                declared[prefix] = (
                    entry.get("unit"),
                    entry.get("internal_unit"),
                )

    assert declared == {
        # rv_model is internal solRad/d
        "rvinstrument": ("m/s", "solRad/d"),
        # lc_model is normalized flux: genuinely dimensionless
        "transit": ("", ""),
        # the coefficient is inside a magnitude exponent
        "mulensinstrument": ("mag", "mag"),
    }


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


def test_a_detrending_fits_captions_say_the_data_were_corrected(
    detrended_rv, detrended_transit
):
    """
    Given a fit with active detrend columns,
    When plot_data builds its specs,
    Then every caption says the fitted trend was subtracted from the
    plotted points -- the reader is looking at corrected data, not the raw
    file, and the figure has to say so.
    """
    rv_system, rv_point = detrended_rv[0], detrended_rv[1]
    lc_system, lc_point = detrended_transit[0], detrended_transit[1]
    for comp, system, point in (
        (rv_system.rvinstrument, rv_system, rv_point),
        (lc_system.transit, lc_system, lc_point),
    ):
        specs = comp.plot_data(system, point)
        assert specs
        for spec in specs:
            assert "detrend columns" in spec.meta["caption"]


def test_a_fit_without_detrend_columns_says_nothing_extra(detrended_rv):
    """
    Given an instrument with no detrend columns,
    When the shared caption sentence is asked for,
    Then it is empty, so no shipped example's caption changes.
    """
    comp = detrended_rv[0].rvinstrument
    saved = comp.total_detrend_cols
    try:
        comp.total_detrend_cols = 0
        assert comp.detrend_caption() == ""
    finally:
        comp.total_detrend_cols = saved


def test_an_all_nan_detrend_column_is_refused_as_non_finite(detrended_rv):
    """
    Given a detrend column that is entirely NaN,
    When the block-diagonal matrix is built,
    Then it RAISES saying the column is NON-FINITE, not "constant (value
    nan)" -- which sent the user hunting for a repeated value that does not
    exist (review 2.14.3).  The constant-column raise above is the same
    site with the other cause named.
    """
    system, _, _, _, _ = detrended_rv

    with pytest.raises(ValueError, match=r"non-finite") as excinfo:
        system.rvinstrument._build_block_detrend([np.full((5, 1), np.nan)], 5)
    assert "rvinstrument[HIRES]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 1.6.7: ONE detrend mechanism, owned by the Instrument base
#
# mulensinstrument scored 10**(-0.4 * X.c) in its likelihood while neither
# its plotted model nor its plotted data carried it -- the 1.5.1 symptom,
# because 1.5.1 had fixed rv/transit by hand.  Now every detrending child
# declares a DETREND_SPACE, builds its likelihood mu through
# Instrument._detrended_model, and its plots remove the trend through
# Instrument.detrend_corrected -- the forward map and its inverse from the
# same DETREND_SPACES entry.  The generic test below runs over EVERY
# Instrument child that declares a space, so a new one is covered (or the
# coverage test fails until it is).
# ---------------------------------------------------------------------------

_MU_T0 = 2460025.0
_MU_TE = 30.0
_MU_U0 = 0.1
_MU_COEFF = 0.3  # mag per raw column unit -- a 0.024 mag rms trend


def _write_mulens_detrend_lc(path, n=80):
    """PSPL light curve in magnitudes plus one airmass-like detrend column
    with a large nonzero mean, the trend injected IN MAGNITUDES."""
    t = np.linspace(_MU_T0 - 2 * _MU_TE, _MU_T0 + 2 * _MU_TE, n)
    u = np.sqrt(_MU_U0**2 + ((t - _MU_T0) / _MU_TE) ** 2)
    amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    x = _X_MEAN + _X_AMP * np.sin(2 * np.pi * np.arange(n) / 13.0)
    mag = 18.0 - 2.5 * np.log10(amp) + _MU_COEFF * (x - _X_MEAN)
    np.savetxt(path, np.column_stack([t, mag, np.full(n, 0.01), x]))
    return x


@pytest.fixture(scope="module")
def detrended_mulens(tmp_path_factory):
    """Built one-instrument PSPL microlensing system with a single detrend
    column -- the mulens twin of ``detrended_rv``.  The coefficient is
    PINNED (in mag per raw column unit) so its value is known exactly."""
    path = tmp_path_factory.mktemp("detrend_mu") / "lc.dat"
    x = _write_mulens_detrend_lc(path)
    config = {
        "run": {"name": "detrend_mu"},
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [
            {"finite_source": False, "t0_par": _MU_T0, "use_op": False}
        ],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": str(path)}],
    }
    user_params = {
        "source.Source.t_0": {"initval": _MU_T0},
        "source.Source.u_0": {"initval": _MU_U0},
        "mulensevent.t_E": {"initval": _MU_TE},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
        "mulensinstrument.detrend_coeffs": {
            "initval": _MU_COEFF,
            "sigma": 0,
        },
    }
    for nm in ("Lens", "Source"):
        user_params[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        user_params[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.compile_plotter_functions(model)
    return system, point, x


def test_unphased_mulens_data_are_detrend_corrected(detrended_mulens):
    """
    Given a microlensing fit with an active detrend column,
    When plot_data builds the light-curve chart,
    Then the plotted points are the fluxes with the fitted factor
    10**(-0.4 * X.c) DIVIDED out (the inverse of what the likelihood
    multiplied the model by), aligned and shown in magnitudes as before.

    Regression (review 1.6.7): the raw fluxes were aligned and plotted,
    so the whole fitted trend read as residual structure -- the 1.5.1
    symptom, on the one child 1.5.1's fix never reached.
    """
    system, point, _ = detrended_mulens
    comp = system.mulensinstrument

    specs = comp.plot_data(system, point)
    data_trace = [t for t in specs[0].traces if t.role == "data"][0]

    aln = comp._flux_alignment(comp._point_to_plot_params(point, system))
    factor = 10.0 ** (-0.4 * comp.detrend_at_data(point))
    assert np.ptp(factor) > 1e-3  # the trend is live
    np.testing.assert_allclose(
        data_trace.y, aln["align"](comp.flux / factor, 0), atol=1e-12
    )
    # the pre-fix value, explicitly excluded
    assert not np.allclose(data_trace.y, aln["align"](comp.flux, 0), atol=1e-3)
    # the error bars are divided by the same factor
    lo = data_trace.y - aln["align"]((comp.flux + comp.err) / factor, 0)
    np.testing.assert_allclose(data_trace.yerr[0], lo, atol=1e-12)


def test_corrected_mulens_data_carry_no_trend(detrended_mulens):
    """
    Given data generated as an exact PSPL times a magnitude trend, and a
    fit pinned at the injected trend coefficient,
    When the corrected plotted points are compared with the plotted model
    at the data times,
    Then what is left is free of the detrend column: the correction removes
    the very trend the likelihood fitted, not an additive approximation of
    it.
    """
    system, point, x = detrended_mulens
    comp = system.mulensinstrument
    flux_c, _ = comp.detrend_corrected(comp.flux, comp.err, point)
    # The corrected magnitudes minus the noiseless PSPL are a constant
    # (the flux zero point), with no residual correlation with x.
    t = comp.time
    u = np.sqrt(_MU_U0**2 + ((t - _MU_T0) / _MU_TE) ** 2)
    amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    resid = -2.5 * np.log10(flux_c) - (18.0 - 2.5 * np.log10(amp))
    assert np.ptp(resid) < 1e-9
    raw = -2.5 * np.log10(comp.flux) - (18.0 - 2.5 * np.log10(amp))
    assert abs(np.corrcoef(raw, x)[0, 1]) == pytest.approx(1.0, abs=1e-9)


def test_a_mulens_detrending_fit_says_so_and_depends_on_the_coefficients(
    detrended_mulens,
):
    """
    Given the same fit,
    When plot_data builds its specs,
    Then every caption says the trend was DIVIDED out (the multiplicative
    wording, not the additive "subtracted"), and the coefficient label is a
    param_dep -- the correction is numpy, invisible to the graph walk.
    """
    system, point, _ = detrended_mulens
    comp = system.mulensinstrument
    label = comp.detrend_coeffs.label
    specs = comp.plot_data(system, point)
    assert len(specs) == 2
    assert "divided out" in specs[0].meta["caption"]
    assert "detrend columns" in specs[0].meta["caption"]
    for spec in specs:
        assert label in spec.param_deps
    assert specs[0].meta["dynamic_data"] is True


_DETRENDING_FIXTURES = {
    "RVInstrument": ("detrended_rv", "rv"),
    "Transit": ("detrended_transit", "flux"),
    "MulensInstrument": ("detrended_mulens", "flux"),
}


def _eval_plot_node(system, node, point, comp):
    """``node`` compiled against ``system.plot_params`` (the plotters'
    inputs) at ``point``."""
    import pytensor

    fn = pytensor.function(
        [p.value for p in system.plot_params], node, on_unused_input="ignore"
    )
    return np.asarray(
        fn(*comp._point_to_plot_params(point, system)), dtype=float
    )


def test_every_detrending_child_is_covered():
    """
    Given every discoverable Instrument child,
    When the ones that declare a DETREND_SPACE are listed,
    Then each has a fixture in ``_DETRENDING_FIXTURES`` -- so the parity
    test below runs over a new detrending child automatically, or this
    test fails until someone gives it one.
    """
    from exozippy.components.factory import discover_components
    from exozippy.components.instrument import DETREND_SPACES, Instrument

    detrending = {
        cls.__name__: cls.DETREND_SPACE
        for cls in discover_components().values()
        if issubclass(cls, Instrument) and cls.DETREND_SPACE is not None
    }
    assert set(detrending) == set(_DETRENDING_FIXTURES)
    assert set(detrending.values()) <= set(DETREND_SPACES)


@pytest.mark.parametrize("cls_name", sorted(_DETRENDING_FIXTURES))
def test_every_detrending_child_inverts_its_own_likelihood(cls_name, request):
    """
    Given a built fit of a detrending Instrument child with a live trend,
    When the likelihood's own mu (the node add_observation_likelihood was
    handed -- it raises on any other) and its detrend-free model are
    evaluated at the point, and the plots' correction is applied,
    Then (1) a datum the likelihood fits EXACTLY (y = mu) is plotted
    exactly ON the detrend-free model the plots draw, and (2) for the real
    data, every plotted residual in units of its plotted error bar equals
    the likelihood's own normalized residual (y - mu) / err.

    This is the architecture of review 1.6.7: one forward map and its
    inverse, never a child's own spelling of either.
    """
    fixture, observable = _DETRENDING_FIXTURES[cls_name]
    system, point = request.getfixturevalue(fixture)[:2]
    comp = next(
        c
        for c in system.active_components.values()
        if type(c).__name__ == cls_name
    )
    assert comp.total_detrend_cols > 0
    assert np.ptp(comp.detrend_at_data(point)) > 0.0

    mu = _eval_plot_node(system, comp._likelihood_mu_node, point, comp)
    free = _eval_plot_node(system, comp._detrend_free_node, point, comp)
    # the trend is live in the likelihood (relative: mulens fluxes are ~1e-8)
    assert np.max(np.abs(mu - free) / np.abs(free)) > 1e-4

    # (1) a perfect fit plots on the model
    y_c, _ = comp.detrend_corrected(mu, comp.err, point)
    np.testing.assert_allclose(
        y_c, free, rtol=1e-12, atol=1e-12 * np.max(np.abs(free))
    )

    # (2) the plotted normalized residuals are the likelihood's
    y = getattr(comp, observable)
    y_c, err_c = comp.detrend_corrected(y, comp.err, point)
    np.testing.assert_allclose(
        (y_c - free) / err_c, (y - mu) / comp.err, rtol=1e-9, atol=1e-9
    )

    # (3) a signal fitted to the likelihood's RESIDUAL (what celerite2's GP
    # mean is) keeps that property once mapped by detrend_corrected_signal:
    # the plotted (y_c - free - s_c) / err_c is the likelihood's
    # (y - mu - s) / err.  Identity for additive spaces, / f for magnitude.
    s = 0.3 * (y - mu) * np.sin(np.arange(y.size))
    s_c = comp.detrend_corrected_signal(s, point)
    np.testing.assert_allclose(
        (y_c - free - s_c) / err_c,
        (y - mu - s) / comp.err,
        rtol=1e-9,
        atol=1e-9,
    )
    if comp.DETREND_SPACE == "additive":
        np.testing.assert_array_equal(s_c, s)
        assert comp.detrend_grid_exact(0)
    else:
        assert not comp.detrend_grid_exact(0)


def test_a_child_that_skips_the_shared_mechanism_is_refused(detrended_rv):
    """
    Given a detrending child,
    When its likelihood is handed a mu that did NOT come from
    Instrument._detrended_model (a child spelling its own trend),
    Then add_observation_likelihood raises naming the component -- the
    drift 1.6.7 found is unrepresentable rather than merely tested for.
    """
    import pymc as pm
    import pytensor.tensor as pt

    comp = detrended_rv[0].rvinstrument
    with pm.Model():
        with pytest.raises(RuntimeError, match=r"rvinstrument.*_detrended"):
            comp.add_observation_likelihood(
                "x", mu=pt.zeros(3), sigma=1.0, observed=np.zeros(3)
            )


def test_reading_detrend_columns_needs_a_declared_space(detrended_rv):
    """
    Given an Instrument child that declares no DETREND_SPACE,
    When it asks the shared reader for detrend columns,
    Then the read raises -- a child cannot acquire detrend columns without
    saying how its likelihood applies them.
    """
    comp = detrended_rv[0].rvinstrument
    saved = type(comp).DETREND_SPACE
    try:
        comp.DETREND_SPACE = None
        with pytest.raises(TypeError, match=r"DETREND_SPACE"):
            comp._read_data(0, roles=("time", "rv", "err"), detrend=True)
    finally:
        del comp.DETREND_SPACE
    assert comp.DETREND_SPACE == saved


try:
    import celerite2.pymc  # noqa: F401

    _HAS_CELERITE2_PYMC = True
except ImportError:  # pragma: no cover - platform dependent
    _HAS_CELERITE2_PYMC = False


@pytest.fixture(scope="module")
def detrended_mulens_gp(tmp_path_factory):
    """The mulens detrend fixture with a GP on the light curve."""
    if not _HAS_CELERITE2_PYMC:
        pytest.skip("celerite2's PyMC backend is unimportable here")
    path = tmp_path_factory.mktemp("detrend_mu_gp") / "lc.dat"
    _write_mulens_detrend_lc(path)
    config = {
        "run": {"name": "detrend_mu_gp"},
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [
            {"finite_source": False, "t0_par": _MU_T0, "use_op": False}
        ],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": str(path), "gp": "sho"}],
    }
    user_params = {
        "source.Source.t_0": {"initval": _MU_T0},
        "source.Source.u_0": {"initval": _MU_U0},
        "mulensevent.t_E": {"initval": _MU_TE},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
        "mulensinstrument.detrend_coeffs": {
            "initval": _MU_COEFF,
            "sigma": 0,
        },
    }
    for nm in ("Lens", "Source"):
        user_params[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        user_params[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    system.compile_plotter_functions(model)
    return system, point


def test_mulens_model_plus_gp_is_exact_on_a_detrended_curve(
    detrended_mulens_gp,
):
    """
    Given a microlensing light curve with BOTH a GP and a (multiplicative)
    detrend trend,
    When plot_data draws that file's model+GP companion,
    Then it is drawn at the DATA epochs and equals, aligned, the
    likelihood's own (mu + gp) / f -- i.e. m + gp / f, the exact companion
    of the corrected points y / f = m + r / f, where celerite2 conditioned
    the GP on r = y - mu.  The old grid curve m + gp was first-order only
    (JDE 2026-10-05: "let's make the plots exact").
    """
    system, point = detrended_mulens_gp
    comp = system.mulensinstrument
    assert not comp.detrend_grid_exact(0)

    vals = comp._point_to_plot_params(point, system)
    mu = _eval_plot_node(system, comp._likelihood_mu_node, point, comp)
    free = _eval_plot_node(system, comp._detrend_free_node, point, comp)
    gp = comp.gp_mean_at_data(system, point)
    f = 10.0 ** (-0.4 * comp.detrend_at_data(point))
    aln = comp._flux_alignment(vals)

    specs = comp.plot_data(system, point)
    tr = next(t for t in specs[0].traces if t.name == "OGLE model+GP")
    np.testing.assert_array_equal(tr.x, comp.time)
    np.testing.assert_allclose(
        tr.y, aln["align"]((mu + gp) / f, 0), rtol=1e-12, atol=0
    )
    # ... and it is measurably NOT the first-order m + gp
    assert np.max(np.abs(gp)) > 0.0
    old = aln["align"](free + gp, 0)
    assert np.max(np.abs(np.asarray(tr.y) - old)) > 1e-9
