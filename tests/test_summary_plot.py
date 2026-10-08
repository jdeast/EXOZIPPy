"""
Tests for the system summary figure (outputs/summary_plot.py) and its
console script (cli_summary.py).

The figure is a third renderer of the components' Charts: it adds no
physics, so what is tested here is what it OWNS -- the header's number
formatting, the choice of the median draw, the page layout, the name
checks on its keywords, and the end-to-end path from a real trace on disk.
The O-C it draws is the components' meta["residuals"], whose numbers are
pinned in tests/test_plot_data.py.

They follow AAA with Given/When/Then docstrings.
"""

import dataclasses
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from exozippy import cli_summary
from exozippy.components.parameter import PosteriorSummary
from exozippy.outputs import summary_plot as sp

_KELT4_DIR = Path(__file__).parent.parent / "examples" / "kelt4"


# ---------------------------------------------------------------------------
# The header's numbers
# ---------------------------------------------------------------------------


def test_format_value_writes_the_errors_at_the_median_precision():
    """
    Given a median of 0.0841 with errors -0.0612/+0.1104, which the results
      table writes 0.084^{+0.11}_{-0.061},
    When format_value renders it,
    Then all three numbers carry the median's three decimal places, the
      trailing zero of the upper error kept.
    """
    summary = PosteriorSummary(
        median=0.0841, err_minus=0.0612, err_plus=0.1104
    )

    assert sp.format_value(summary) == "0.084^{+0.110}_{-0.061}"


@pytest.mark.parametrize(
    "median, err_minus, err_plus",
    [
        (2.95248098, 9.57e-7, 9.55e-7),
        (10.834, 17.256, 14.410),
        (1234.5, 560.0, 780.0),
        (0.74727, 0.01445, 0.03827),
    ],
)
def test_format_value_rounds_the_median_as_the_table_does(
    median, err_minus, err_plus
):
    """
    Given posterior summaries across many orders of magnitude,
    When format_value renders each,
    Then its median is the number PosteriorSummary.format puts in the table,
      and each error has exactly the median's number of decimal places.
    """
    summary = PosteriorSummary(median, err_minus, err_plus)

    text = sp.format_value(summary)

    if r" \pm " in text:
        med, err = text.split(r" \pm ")
        errs = [err]
    else:
        med, rest = text.split("^{+")
        hi, lo = rest.rstrip("}").split("}_{-")
        errs = [hi, lo]
    assert float(med) == float(summary.format().median)
    places = len(med.split(".")[1]) if "." in med else 0
    for err in errs:
        assert (len(err.split(".")[1]) if "." in err else 0) == places


def test_format_value_collapses_equal_errors_and_marks_fixed_values():
    """
    Given a symmetric posterior, a zero-spread (fixed) one and a NaN one,
    When format_value renders them,
    Then the first is written with \\pm, the second with \\equiv as the table
      writes a fixed value, and the third has nothing to report (None).
    """
    assert sp.format_value(PosteriorSummary(5.0, 0.1, 0.1)) == r"5.00 \pm 0.10"
    assert sp.format_value(PosteriorSummary(0.0, 0.0, 0.0)) == r"\equiv 0"
    assert sp.format_value(PosteriorSummary(float("nan"), 1.0, 1.0)) is None


# ---------------------------------------------------------------------------
# The reported posterior
# ---------------------------------------------------------------------------


class _RecordingSystem:
    """The System calls reported_posterior makes, recorded."""

    def __init__(self):
        self.calls = []

    def report_only_labels(self):
        return []

    def fold_degenerate_draws(self, idata, model):
        self.calls.append(("fold", model))
        return []

    def distribute_posterior(self, idata):
        self.calls.append(("distribute", idata))


def _trace_with_stale_modes():
    rng = np.random.default_rng(1)
    return az.from_dict(
        {
            "posterior": {
                "x": rng.normal(size=(2, 50)),
                "mode": np.full((2, 50), 3),
            }
        }
    )


@pytest.fixture
def no_trim(monkeypatch):
    """analyze_idata as the identity, so the test sees what it hands on.

    Imports exozippy.run first: reported_posterior imports it, and a FIRST
    import while a test has modes.identify_modes patched would bind the
    fake into report_pipeline (`from .modes import identify_modes`) for
    every later test in the process -- the kelt4 fit's wrap-up included.
    """
    import exozippy.run  # noqa: F401
    from exozippy.samplers import convergence

    monkeypatch.setattr(
        convergence, "analyze_idata", lambda idata, exclude=(): (idata, {})
    )


def test_reported_posterior_folds_and_drops_stale_mode_labels(
    monkeypatch, no_trim
):
    """
    Given a saved trace carrying posterior['mode'] labels from an earlier
      mode pass,
    When reported_posterior rebuilds the reported posterior from it,
    Then it folds the declared degeneracies first (as run._wrap_up does),
      the mode pass never sees the stale labels, and the posterior is
      distributed afterwards (review 2.11.8).
    """
    from exozippy.outputs import modes

    seen = {}

    def _identify(idata):
        seen["vars"] = set(idata.posterior.data_vars)
        return SimpleNamespace(n_modes=1)

    monkeypatch.setattr(modes, "identify_modes", _identify)
    system = _RecordingSystem()
    model = object()

    sp.reported_posterior(system, model, _trace_with_stale_modes())

    assert "mode" not in seen["vars"]
    assert [c[0] for c in system.calls] == ["fold", "distribute"]
    assert system.calls[0][1] is model


def test_reported_posterior_raises_on_a_mode_pass_failure(
    monkeypatch, no_trim
):
    """
    Given a mode pass that fails for a reason other than "no valid draws",
    When reported_posterior runs it,
    Then the failure propagates -- the figure is not drawn from an
      unlabelled posterior behind a warning.
    """
    from exozippy.outputs import modes

    def _identify(idata):
        raise RuntimeError("mode pass bug")

    monkeypatch.setattr(modes, "identify_modes", _identify)

    with pytest.raises(RuntimeError, match="mode pass bug"):
        sp.reported_posterior(
            _RecordingSystem(), object(), _trace_with_stale_modes()
        )


# ---------------------------------------------------------------------------
# The median draw
# ---------------------------------------------------------------------------


class _Param:
    """A Parameter stand-in: user -> internal divides by ``factor``."""

    def __init__(self, factor):
        self.factor = factor

    def to_internal(self, val):
        return np.asarray(val) / self.factor


def _stub_system():
    return SimpleNamespace(
        get_parameter_lookup=lambda: {
            "orbit.period": _Param(2.0),
            "orbit.logP": _Param(1.0),
        },
        report_only_labels=lambda: ["planet.t14"],
    )


def _stub_idata(log_p_raw, mode=None):
    log_p_raw = np.asarray(log_p_raw, dtype=float)
    period = np.arange(log_p_raw.size, dtype=float).reshape(log_p_raw.shape)
    posterior = {
        "orbit.period": period + 10.0,
        "orbit.logP_raw": log_p_raw,
        # A pinned coordinate: no width, so it cannot be scaled by one.
        "orbit.tc_raw": np.full(log_p_raw.shape, 3.0),
        "planet.t14": period * 0.0,
    }
    if mode is not None:
        posterior["mode"] = np.asarray(mode)
    return az.from_dict({"posterior": posterior})


def test_median_draw_is_the_valid_draw_nearest_the_median():
    """
    Given draws whose valid ones have their median exactly at draw (1, 0),
      an invalid draw (mode -1) that would tie for nearest if it counted,
      and a pinned (zero-width) coordinate,
    When median_draw_point picks the draw,
    Then it takes (1, 0) at distance 0 -- the invalid draw enters neither
      the median nor the choice, the pinned coordinate is skipped rather
      than divided by -- converts physical values to internal units, passes
      *_raw through, and leaves out the mode label and report-only
      Deterministics.
    """
    idata = _stub_idata(
        log_p_raw=[[0.0, -2.0, -1.0, 1.0, 2.0], [0.05, -1.5, -0.5, 0.5, 1.5]],
        mode=[[-1, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    )

    point, (chain, draw), distance = sp.median_draw_point(
        _stub_system(), idata
    )

    assert (chain, draw) == (1, 0)
    assert distance == pytest.approx(0.0)
    period = idata.posterior["orbit.period"].values[1, 0]
    assert point["orbit.period"] == pytest.approx(period / 2.0)
    assert point["orbit.logP_raw"] == pytest.approx(0.05)
    assert "mode" not in point
    assert "planet.t14" not in point


def test_median_draw_refuses_a_trace_with_nothing_to_choose():
    """
    Given a trace with no sampled (*_raw) variables, and one whose every
      draw was rejected by the mode pass,
    When median_draw_point is asked for the draw,
    Then it raises rather than guessing one.
    """
    no_raw = az.from_dict({"posterior": {"orbit.period": np.ones((1, 3))}})
    with pytest.raises(ValueError, match="no sampled"):
        sp.median_draw_point(_stub_system(), no_raw)

    rejected = _stub_idata(log_p_raw=[[1.0, 2.0]], mode=[[-1, -1]])
    with pytest.raises(ValueError, match="No valid draw"):
        sp.median_draw_point(_stub_system(), rejected)


# ---------------------------------------------------------------------------
# Layout and keyword checks
# ---------------------------------------------------------------------------


def test_place_reproduces_the_classic_one_planet_page():
    """
    Given the panels of a one-planet transit + RV + SED + MIST fit, in
      reading order,
    When _place lays them out,
    Then the transits sit over the SED on the left, and the RVs against
      time, the folded RVs and the Kiel diagram run down the right, on a
      two-column, three-row grid.
    """
    panels = [
        sp._Panel("stacked_fold", [], 2),
        sp._Panel("time_series", [], 1),
        sp._Panel("phase_fold", [], 1),
        sp._Panel("spectrum", [], 1),
        sp._Panel("track", [], 1),
    ]

    placed, ncols, nrows = sp._place(panels)

    assert (ncols, nrows) == (2, 3)
    assert [(p.kind, col, row) for p, col, row in placed] == [
        ("stacked_fold", 0, 0),
        ("time_series", 1, 0),
        ("phase_fold", 1, 1),
        ("spectrum", 0, 2),
        ("track", 1, 2),
    ]
    _, ncols, _ = sp._place([sp._Panel("spectrum", [], 1)])
    assert ncols == 1


def test_bin_averages_within_each_bin_and_drops_empty_ones():
    """
    Given points falling in bins 0, 1 and 3 of width 1 (bin 2 empty),
    When _bin averages them,
    Then it returns the three occupied bins' centers and means.
    """
    x = np.array([0.0, 0.1, 0.2, 1.05, 1.1, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 10.0])

    centers, means = sp._bin(x, y, 1.0)

    np.testing.assert_allclose(centers, [0.5, 1.5, 3.5])
    np.testing.assert_allclose(means, [2.0, 5.0, 10.0])


def test_an_unknown_instrument_name_raises_with_a_suggestion():
    """
    Given a keyword naming an instrument with the wrong case,
    When the summary checks the names against the fit's instruments,
    Then it raises, naming the near miss -- suggested, never accepted, as
      every user-facing name in EXOZIPPy is case-sensitive.
    """
    with pytest.raises(
        ValueError, match=r"'TESS_s16' \(did you mean 'TESS_S16'"
    ):
        sp._check_names("labels", {"TESS_s16": "x"}, ["TESS_S16", "HIRES"])


def _transits(rows, stated=True):
    """A transit component stand-in: (name, band, exptime_min, label), each
    file's exptime stated in its config and accepted unless ``stated`` is
    False."""
    from exozippy.components.transit.transit import Transit

    labels = [r[3] for r in rows]
    return SimpleNamespace(
        names=[r[0] for r in rows],
        band_names=[r[1] for r in rows],
        exptime_min=[r[2] for r in rows],
        plot_label=labels,
        exptime_stated=[stated] * len(rows),
        display_label=lambda i: labels[i] or rows[i][0],
        SUMMARY_TESS_FILTER=Transit.SUMMARY_TESS_FILTER,
    )


_BANDS = SimpleNamespace(
    names=["TESS", "Sloang", "Kepler"],
    filter_names=["TESS", "Sloang", "Kepler"],
)


def _default_groups(transit):
    """The default grouping end to end: Transit.summary_rows declares each
    file's row on its stacked chart, and declared_row_groups reads them."""
    from exozippy.chart import Chart
    from exozippy.components.transit.transit import Transit

    rows = Transit.summary_rows(transit, SimpleNamespace(band=_BANDS))
    charts = [
        Chart(
            id=f"transit.{name}",
            component={"yaml_key": "transit", "instance": None},
            title="",
            xlabel="",
            ylabel="",
            traces=[],
            meta={
                "summary": {
                    "layout": "stacked_fold",
                    "stack": "b",
                    "instrument": name,
                    "row_key": rows[name][0],
                    "row_label": rows[name][1],
                }
            },
        )
        for name in transit.names
    ]
    return sp.declared_row_groups(charts)


def test_tess_files_are_grouped_by_cadence_and_nothing_else_is():
    """
    Given TOI-5432's transits -- two 10-min and two 2-min TESS sectors, and
      ground-based files of which two share a filter and exposure time --
      with no labels in the config,
    When the default grouping is formed,
    Then each TESS cadence is one row, labelled with its cadence in
      seconds, and every ground-based file is left on its own row.
    """
    transit = _transits(
        [
            ("TESS_UT20210916", "TESS", 10.0, None),
            ("TESS_UT20211107", "TESS", 10.0, None),
            ("Acton-Sky-Portal_UT20221219", "Sloang", 1.0, None),
            ("LO_OSC_UT20221222", "Kepler", 1.0, None),
            ("TESS_UT20231016", "TESS", 2.0, None),
            ("LCO-McD-0m35_UT20231019", "Sloang", 1.0, None),
            ("TESS_UT20231112", "TESS", 2.0, None),
        ]
    )

    groups = _default_groups(transit)

    assert groups == {
        "TESS 600 s": ["TESS_UT20210916", "TESS_UT20211107"],
        "TESS 120 s": ["TESS_UT20231016", "TESS_UT20231112"],
    }


def test_a_tess_group_takes_its_members_common_label():
    """
    Given TESS sectors all labelled "TESS" at two cadences, and a 200 s
      sector labelled "TESS FFI",
    When the default grouping is formed,
    Then the label two cadences share gets the cadence appended so the rows
      stay distinct, while the label only one cadence uses is kept as is.
    """
    transit = _transits(
        [
            ("S43", "TESS", 10.0, "TESS"),
            ("S45", "TESS", 10.0, "TESS"),
            ("S71", "TESS", 2.0, "TESS"),
            ("S98", "TESS", 200.0 / 60.0, "TESS FFI"),
        ]
    )

    groups = _default_groups(transit)

    assert groups == {
        "TESS 600 s": ["S43", "S45"],
        "TESS 120 s": ["S71"],
        "TESS FFI": ["S98"],
    }


def test_a_tess_group_with_no_stated_exptime_is_not_given_a_cadence():
    """
    Given two TESS sectors whose config sets no exptime (so the fit uses
      the inert 1-minute default, which is not their cadence),
    When the default grouping is formed,
    Then they share one row labelled just "TESS", not "TESS 60 s".
    """
    transit = _transits(
        [("S16", "TESS", 1.0, None), ("S22", "TESS", 1.0, None)],
        stated=False,
    )

    assert _default_groups(transit) == {"TESS": ["S16", "S22"]}


def test_two_tess_groups_with_one_label_raise_rather_than_overwrite():
    """
    Given a 60 s TESS sector, labelled "TESS 60 s" by its cadence, and a
      120 s sector whose config labels it "TESS 60 s" too,
    When the default grouping is formed,
    Then it raises naming both groups' files -- one row is never silently
      replaced by the other.
    """
    transit = _transits(
        [("S16", "TESS", 1.0, None), ("S22", "TESS", 2.0, "TESS 60 s")]
    )

    with pytest.raises(ValueError, match=r"\['S16'\] and \['S22'\]"):
        _default_groups(transit)


def test_two_single_file_rows_may_share_a_label():
    """
    Given two ground-based files a user labelled alike ("LCO"),
    When the default grouping is formed,
    Then neither is a group and nothing raises: each keeps its own row
      under the same label, as two rows of one telescope.
    """
    transit = _transits(
        [("LCO_1", "Sloang", 1.0, "LCO"), ("LCO_2", "Sloang", 1.0, "LCO")]
    )

    assert _default_groups(transit) == {}


def test_a_chart_is_on_the_page_only_by_declaring_a_layout():
    """
    Given a chart that declares no meta["summary"] (the SED's data-only
      photometry, say), one that declares a layout, and one that declares a
      layout the summary does not have,
    When each is assigned a panel,
    Then the first has none, the second its layout, and the third raises
      naming the chart -- a declaration is a contract, not a guess.
    """
    from exozippy.chart import Chart, Trace

    photometry = Chart(
        id="sed.photometry",
        component={"yaml_key": "sed", "instance": None},
        title="",
        xlabel="",
        ylabel="",
        traces=[Trace("observed", "data", "scatter", [0.5], [10.0])],
    )
    spectrum = dataclasses.replace(
        photometry, id="sed.sed", meta={"summary": {"layout": "spectrum"}}
    )
    typo = dataclasses.replace(
        photometry, id="sed.typo", meta={"summary": {"layout": "spectra"}}
    )

    assert sp._panel_kind(photometry) is None
    assert sp._panel_kind(spectrum) == "spectrum"
    with pytest.raises(ValueError, match=r"'sed.typo' declares layout"):
        sp._panel_kind(typo)


@pytest.mark.parametrize(
    "teff, logg",
    [
        (np.full(50, 5800.0), np.linspace(4.3, 4.5, 50)),
        (np.linspace(5700.0, 5900.0, 50), 4.4 + np.linspace(0, 0.2, 50)),
        (np.array([5800.0, 5810.0]), np.array([4.4, 4.5])),
    ],
    ids=["constant-teff", "collinear", "two-draws"],
)
def test_a_degenerate_kiel_posterior_gets_no_contour(teff, logg, caplog):
    """
    Given Kiel samples with no 2-D density -- a pinned Teff, a logg that
      moves exactly with Teff, or too few draws -- for both the fit and MIST,
    When the contours are drawn,
    Then none is drawn and none is in the legend, the panel's window falls
      back to the star and the track (None is returned), and the reason is
      logged -- instead of gaussian_kde's LinAlgError costing the whole page.
    """
    contours = [("MIST", teff, logg), ("Global fit", teff, logg)]
    fig, ax = plt.subplots()
    try:
        with caplog.at_level("INFO", logger=sp.logger.name):
            extent = sp._contours(ax, contours)
        handles, _ = ax.get_legend_handles_labels()
    finally:
        plt.close(fig)

    assert extent is None
    assert handles == []
    assert caplog.text.count("no 2-D density") == 2


def test_sed_in_flux_converts_the_points_and_their_oc_together():
    """
    Given an SED chart in log10(lambda F_lambda) whose O-C is in dex,
    When sed_in_flux converts it for the summary panel,
    Then the points become fluxes on a log axis with their error bars
      converted on each side, and each O-C becomes the flux difference it
      encodes (observed minus model) with its point's error bars.
    """
    from exozippy.chart import Chart, Trace

    y = np.array([-10.0, -11.0])
    err = np.array([[0.1, 0.2], [0.1, 0.2]])
    oc_dex = np.array([0.05, -0.02])
    chart = Chart(
        id="sed.sed",
        component={"yaml_key": "sed", "instance": None},
        title="SED",
        xlabel="Wavelength [micron]",
        ylabel="log10(lambda F_lambda [erg/s/cm2])",
        traces=[
            Trace("Star A", "model", "line", [0.5, 2.0], [-9.0, -12.0]),
            Trace("A", "data", "scatter", [0.6, 1.2], y, yerr=err),
        ],
        y_range=[-13.0, -9.0],
        meta={
            "residuals": [
                Trace("A", "residual", "scatter", [0.6, 1.2], oc_dex, err)
            ],
            "summary": {
                "layout": "spectrum",
                "y_log10": True,
                "xlabel": "wave",
                "ylabel": "flux",
            },
        },
    )

    flux = sp.sed_in_flux(chart)

    data = [t for t in flux.traces if t.role == "data"][0]
    (oc,) = flux.meta["residuals"]
    np.testing.assert_allclose(data.y, 10**y)
    np.testing.assert_allclose(data.yerr[0], 10**y - 10 ** (y - err[0]))
    np.testing.assert_allclose(oc.y, 10**y - 10 ** (y - oc_dex))
    np.testing.assert_allclose(oc.yerr, data.yerr)
    assert flux.y_log and (flux.xlabel, flux.ylabel) == ("wave", "flux")
    np.testing.assert_allclose(flux.y_range, [1e-13, 1e-9])

    chart.meta["residuals"][0].name = "B"
    with pytest.raises(ValueError, match="'B'"):
        sp.sed_in_flux(chart)


def test_smooth_is_the_pipeline_boxcar_and_leaves_gaps_alone():
    """
    Given a spectrum with a NaN in it,
    When it is smoothed with a 9-point boxcar,
    Then away from the edges and the gap every point is the plain 9-point
      running mean (np.convolve with ones(9)/9, the pipeline's smooth), the
      NaN stays NaN without spreading, and a window below 3 is a no-op.
    """
    rng = np.random.default_rng(1)
    y = rng.normal(size=60)
    plain = np.convolve(y, np.ones(9) / 9, mode="same")
    y_gap = y.copy()
    y_gap[40] = np.nan

    smooth = sp._smooth(y_gap, 9)

    np.testing.assert_allclose(smooth[4:35], plain[4:35])
    assert np.isnan(smooth[40]) and np.isfinite(smooth[36:40]).all()
    np.testing.assert_array_equal(sp._smooth(y, 1), y)


def _phased_chart(depth, x_range=(-0.1, 0.1)):
    from exozippy.chart import Chart, Trace

    x = np.linspace(-0.1, 0.1, 200)
    model = np.where(np.abs(x) < 0.04, -depth, 0.0)
    return Chart(
        id="transit.phased.T.b",
        component={"yaml_key": "transit", "instance": "T"},
        title="",
        xlabel="",
        ylabel="",
        traces=[
            Trace("model", "model", "line", x, model),
            Trace("T", "data", "scatter", x, model + 1e-4, yerr=x * 0),
        ],
        x_range=list(x_range),
        meta={
            "instrument": "T",
            "planet": "b",
            "phase_folded": True,
            "summary": {
                "layout": "stacked_fold",
                "stack": "b",
                "instrument": "T",
                "label": "T",
                "color": None,
                "row_key": "T",
                "row_label": "T",
                "x_scale": 24.0,
                "xlabel": "Time from Mid-Transit [hr]",
                "ylabel": "Normalized Flux + Constant",
            },
        },
    )


def test_transit_stack_limits_are_symmetric_about_the_stack():
    """
    Given two stacked rows 0.02 apart with a 0.01-deep transit,
    When the stack is drawn,
    Then the margin above the first row's baseline equals the margin below
      the bottom of the last row's transit (half a row each), and the
      x-axis is the chart's window in hours.
    """
    rows = [
        ("A", "A", [_phased_chart(0.01)], None, "#009B77"),
        ("B", "B", [_phased_chart(0.01)], None, "#821EA6"),
    ]
    fig, ax = plt.subplots()
    try:
        sp._draw_transit_stack(ax, rows, spacing=0.02)
        bottom, top = ax.get_ylim()
        xlim = ax.get_xlim()
    finally:
        plt.close(fig)

    assert top - 1.0 == pytest.approx(0.01)
    assert (1.0 - 0.02 - 0.01) - bottom == pytest.approx(top - 1.0)
    np.testing.assert_allclose(xlim, [-2.4, 2.4])


def test_each_transit_label_sits_just_above_its_own_row():
    """
    Given two rows 0.02 apart with a 0.01-deep transit, the first
      scattered by +-0.0005 about its model and the second by +-0.003,
    When the stack is drawn,
    Then each label is anchored at the left, _LABEL_SIGMAS times its row's
      robust scatter above that row's baseline -- 2 x 1.4826 x 0.0005 for
      the first -- but no more than _LABEL_MAX_GAP of the 0.01 gap to the
      row above, where the noisier second row's is capped.
    """
    from exozippy.chart import Trace

    def noisy(chart, amplitude):
        model = sp._one(chart, "model")
        wiggle = amplitude * np.where(np.arange(np.size(model.y)) % 2, 1, -1)
        data = Trace(
            "T", "data", "scatter", model.x, np.asarray(model.y) + wiggle
        )
        return dataclasses.replace(chart, traces=[model, data])

    rows = [
        ("A", "A", [noisy(_phased_chart(0.01), 0.0005)], None, "#009B77"),
        ("B", "B", [noisy(_phased_chart(0.01), 0.003)], None, "#821EA6"),
    ]
    fig, ax = plt.subplots()
    try:
        sp._draw_transit_stack(ax, rows, spacing=0.02)
        anchors = {t.get_text(): t.xy for t in ax.texts}
        left = ax.get_xlim()[0]
    finally:
        plt.close(fig)

    assert anchors["A"][1] == pytest.approx(1.0 + 2.0 * 1.4826 * 0.0005)
    assert anchors["B"][1] == pytest.approx(0.98 + sp._LABEL_MAX_GAP * 0.01)
    assert anchors["A"][0] == pytest.approx(left + 0.02 * 4.8)


def test_posterior_draws_share_one_alpha_with_the_reference_curve():
    """
    Given one transit row grouping two files whose model curves are
      identical at every point, and two other posterior draws for each --
      the first of them identical to the reference point,
    When the stack is drawn without and then with the draws,
    Then alone the reference curve is drawn once, at full weight; with
      them, the reference and each draw are drawn once apiece, all at
      _DRAWS_ALPHA (as EXOZIPPy's own plots draw spaghetti): the files'
      repeat within a draw is left out so overlapping files do not darken
      it, while the draw repeating the reference is kept, as plotrender
      keeps every draw.
    """
    from exozippy.chart import Trace

    first = _phased_chart(0.01)
    second = dataclasses.replace(first, id="transit.phased.T2.b")
    reference = sp._one(first, "model")
    extra = [
        [Trace("model", "model", "line", reference.x, reference.y)],
        [Trace("model", "model", "line", reference.x, reference.y * 0.9)],
    ]
    draws = {first.id: extra, second.id: extra}
    rows = [("TESS", "TESS", [first, second], None, "#009B77")]

    alphas = []
    for draw_models in (None, draws):
        fig, ax = plt.subplots()
        try:
            sp._draw_transit_stack(ax, rows, 0.02, draw_models)
            alphas.append(
                [ln.get_alpha() for ln in ax.get_lines() if ln.get_ls() == "-"]
            )
        finally:
            plt.close(fig)

    assert alphas == [[None], [sp._DRAWS_ALPHA] * 3]


def _rv_hints(layout):
    hints = {
        "layout": layout,
        "labels": {"HIRES": "HIRES"},
        "colors": {0: None},
        "ylabel": "RV [m/s]",
    }
    if layout == "time_series":
        hints["x_offsets"] = [2457000, 2450000]
    return hints


def _rv_chart(n=50):
    from exozippy.chart import Chart, Trace

    x = np.linspace(0.0, 1.0, n)
    data_x = np.array([0.1, 0.5, 0.9])
    return Chart(
        id="rvinstrument.phased.b",
        component={"yaml_key": "rvinstrument", "instance": None},
        title="",
        xlabel="Phase",
        ylabel="RV",
        traces=[
            Trace("Model", "model", "line", x, 50.0 * np.sin(2 * np.pi * x)),
            Trace(
                "HIRES",
                "data",
                "scatter",
                data_x,
                np.zeros(3),
                yerr=np.ones(3),
                style={"series_index": 0},
            ),
        ],
        meta={"phase_folded": True, "summary": _rv_hints("phase_fold")},
    )


def test_time_segments_split_at_gaps_longer_than_the_threshold():
    """
    Given observations with gaps of 101, 35 and 60 days,
    When they are split at gaps over 50 days, and then with no threshold,
    Then the 101- and 60-day gaps break the axis and the 35-day one does
      not; with no threshold there is one piece.
    """
    times = [4017.0, 4118.0, 4130.0, 4165.0, 4200.0, 4260.0, 4261.0]

    assert sp.time_segments(times, 50.0) == [
        (4017.0, 4017.0),
        (4118.0, 4200.0),
        (4260.0, 4261.0),
    ]
    assert sp.time_segments(times, None) == [(4017.0, 4261.0)]


def test_piece_ticks_stay_clear_of_the_breaks():
    """
    Given a wide piece (4113-4222, 5.5 in) and a narrow one (4013-4021,
      0.8 in) of a broken axis,
    When their ticks are chosen, trimming the edges that face a break,
    Then the wide piece's ticks are round and none lies within
      _BREAK_EDGE_IN of its trimmed left edge, and the narrow piece gets
      one round tick inside its trimmed window.
    """
    wide = sp.piece_ticks(4113.0, 4222.0, 5.5, True, False)
    narrow = sp.piece_ticks(4013.0, 4021.0, 0.8, False, True)

    per_in = (4222.0 - 4113.0) / 5.5
    assert len(wide) >= 3
    assert min(wide) >= 4113.0 + sp._BREAK_EDGE_IN * per_in
    assert all(t % 10 == 0 for t in wide)
    per_in = 8.0 / 0.8
    assert narrow == [4015.0]
    assert narrow[0] <= 4021.0 - sp._BREAK_EDGE_IN * per_in


def _rv_time_chart():
    """An RV-against-time chart: two seasons 100 days apart, with O-C."""
    from exozippy.chart import Chart, Trace

    times = np.array([10.0, 12.0, 15.0, 115.0, 118.0, 130.0])
    model_x = np.linspace(0.0, 140.0, 2000)
    data = Trace(
        "HIRES",
        "data",
        "scatter",
        times,
        np.zeros(6),
        yerr=np.ones(6),
        style={"series_index": 0},
    )
    oc = dataclasses.replace(data, role="residual", y=np.ones(6))
    return Chart(
        id="rvinstrument.unphased",
        component={"yaml_key": "rvinstrument", "instance": None},
        title="",
        xlabel="Time [BJD - 2457000]",
        ylabel="RV",
        traces=[
            Trace("Model", "model", "line", model_x, np.sin(model_x)),
            data,
        ],
        meta={"residuals": [oc], "summary": _rv_hints("time_series")},
    )


def test_a_long_gap_breaks_the_rv_time_axis_and_its_oc():
    """
    Given an RV-against-time chart with two seasons 100 days apart,
    When the panel is drawn with breaks at 50 days, and again with none,
    Then it has two panel and two O-C pieces, each framing its own season;
      the facing spines are hidden, only the first piece carries the y
      labels, and one x label is left; without breaks it is one piece.
    """
    chart = sp._prepared(_rv_time_chart(), {})

    fig = plt.figure(figsize=(8, 6))
    try:
        axes, oc_axes = sp._draw_rv(
            fig, fig.add_gridspec(1, 1)[0, 0], chart, {0: "b"}, False, (), 50
        )
        xlims = [ax.get_xlim() for ax in axes]
        ylabels = [ax.get_ylabel() for ax in axes + oc_axes]
        xlabels = [ax.get_xlabel() for ax in axes + oc_axes]
        facing = (
            axes[0].spines["right"].get_visible(),
            axes[1].spines["left"].get_visible(),
            oc_axes[0].spines["right"].get_visible(),
            oc_axes[1].spines["left"].get_visible(),
        )
    finally:
        plt.close(fig)
    fig = plt.figure(figsize=(8, 6))
    try:
        unbroken = sp._draw_rv(
            fig, fig.add_gridspec(1, 1)[0, 0], chart, {0: "b"}, False, (), None
        )
    finally:
        plt.close(fig)

    assert (len(axes), len(oc_axes)) == (2, 2)
    assert xlims[0][0] < 10.0 and 15.0 < xlims[0][1] < 115.0
    assert 15.0 < xlims[1][0] < 115.0 and xlims[1][1] > 130.0
    assert facing == (False, False, False, False)
    assert ylabels == ["RV [m/s]", "", sp.OC_YLABEL, ""]
    assert [x for x in xlabels if x] == ["Time [BJD - 2457000]"]
    assert [len(a) for a in unbroken] == [1, 1]


def test_rv_draws_are_faint_but_their_legend_swatch_is_not():
    """
    Given a folded RV chart and two other posterior draws' model curves,
    When the RV panel is drawn with them,
    Then all three curves are at _DRAWS_ALPHA, and the legend's "Model"
      swatch is raised to _LEGEND_ALPHA -- the plotted curves untouched --
      as plotrender's legend does.
    """
    from exozippy.chart import Trace

    chart = _rv_chart()
    x = np.linspace(0.0, 1.0, 50)
    draws = [
        [Trace("Model", "model", "line", x, a * np.sin(2 * np.pi * x))]
        for a in (45.0, 55.0)
    ]
    fig = plt.figure()
    try:
        (ax,), _ = sp._draw_rv(
            fig, fig.add_gridspec(1, 1)[0, 0], chart, {0: "b"}, True, draws
        )
        curves = [
            ln.get_alpha() for ln in ax.get_lines() if ln.get_ls() == "-"
        ]
        legend = ax.get_legend()
        swatch = {
            t.get_text(): h
            for t, h in zip(legend.get_texts(), legend.legend_handles)
        }["Model"]
    finally:
        plt.close(fig)

    assert curves == [sp._DRAWS_ALPHA] * 3
    assert swatch.get_alpha() == sp._LEGEND_ALPHA


def test_draw_model_traces_gathers_the_panel_charts_model_traces():
    """
    Given a component with two charts at each of two draws, and a panel
      drawing one of them,
    When the draws' model traces are gathered,
    Then the panel's chart gets the model traces (not the data) of both
      draws, and the other chart nothing.
    """
    chart = _rv_chart()
    other = dataclasses.replace(chart, id="rvinstrument.unphased")
    component = SimpleNamespace(plot_data=lambda system, point: [chart, other])
    system = SimpleNamespace(active_components={"rvinstrument": component})
    panels = [sp._Panel("phase_fold", [chart], 1)]

    models = sp._draw_model_traces(system, panels, [{}, {}])

    assert list(models) == [chart.id]
    assert [[t.role for t in traces] for traces in models[chart.id]] == [
        ["model"],
        ["model"],
    ]


def test_draw_model_traces_raises_on_a_failing_draw():
    """
    Given a component whose plot_data raises at one posterior draw,
    When the draws' model traces are gathered,
    Then the failure propagates: a component that cannot draw a draw of
      its own posterior is a bug to surface, not a curve to drop from the
      spaghetti behind a warning.
    """
    chart = _rv_chart()

    def plot_data(system, point):
        if point["bad"]:
            raise RuntimeError("non-finite draw")
        return [chart]

    component = SimpleNamespace(plot_data=plot_data)
    system = SimpleNamespace(active_components={"rvinstrument": component})
    panels = [sp._Panel("phase_fold", [chart], 1)]

    with pytest.raises(RuntimeError, match="non-finite draw"):
        sp._draw_model_traces(system, panels, [{"bad": False}, {"bad": True}])


def test_the_kiel_and_rv_time_panels_are_not_drawn_with_posterior_draws():
    """
    Given a folded RV panel, an RV-against-time panel and a Kiel panel, and
      two posterior draws,
    When the draws' model traces are gathered,
    Then only the folded RV chart gets them, and the evolutionary model is
      never evaluated at a draw -- the Kiel diagram keeps the reference
      point's one track, its contours showing the posterior, and the RVs
      against time the reference point's one curve, legible over many
      orbits.
    """
    rv_chart = _rv_chart()
    time_chart = _rv_time_chart()
    kiel_chart = _kiel_chart()
    calls = []

    def kiel_plot_data(system, point):
        calls.append(point)
        return [kiel_chart]

    system = SimpleNamespace(
        active_components={
            "rvinstrument": SimpleNamespace(
                plot_data=lambda system, point: [time_chart, rv_chart]
            ),
            "evolutionarymodel": SimpleNamespace(plot_data=kiel_plot_data),
        }
    )
    panels = [
        sp._Panel("time_series", [time_chart], 1),
        sp._Panel("phase_fold", [rv_chart], 1),
        sp._Panel("track", [kiel_chart], 1),
    ]

    models = sp._draw_model_traces(system, panels, [{}, {}])

    assert list(models) == [rv_chart.id]
    assert len(models[rv_chart.id]) == 2
    assert calls == []


def test_sed_panel_spans_the_photometry_and_marks_the_model():
    """
    Given an SED chart (already in flux) whose bandpasses run from 0.4 to
      15 micron,
    When the SED panel is drawn,
    Then the flux axis spans the photometry padded by _SED_Y_PAD; the
      wavelength axis spans the bandpasses padded by _SED_X_PAD, widened to
      the model spectrum where it lies within that flux axis (0.05 micron)
      but not where it lies below it (30 micron); the axes carry the
      compact labels; and each model photometry point sits at the observed
      flux less its O-C.
    """
    from exozippy.chart import Chart, Trace

    x = np.array([0.5, 12.0])
    xerr = np.array([[0.1, 4.0], [0.1, 3.0]])
    y = np.array([1e-10, 1e-12])
    yerr = np.array([[1e-12, 1e-14], [1e-12, 1e-14]])
    chart = Chart(
        id="sed.sed",
        component={"yaml_key": "sed", "instance": None},
        title="",
        xlabel="wave",
        ylabel="flux",
        traces=[
            Trace("Star A", "model", "line", [0.05, 30.0], [1e-11, 1e-13]),
            Trace("A", "data", "scatter", x, y, yerr=yerr, xerr=xerr),
        ],
        x_log=True,
        y_log=True,
        x_range=[0.05, 30.0],
        y_range=[1e-15, 1e-9],
        meta={
            "residuals": [
                Trace("A", "residual", "scatter", x, y * 0.1, yerr, xerr)
            ],
            "summary": {
                "layout": "spectrum",
                "identity": {"Star A": "A", "A": "A"},
            },
        },
    )
    fig = plt.figure()
    try:
        sp._draw_sed(fig, fig.add_gridspec(1, 1)[0, 0], chart)
        ax, ax_oc = fig.get_axes()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        model = [ln for ln in ax.get_lines() if ln.get_label() == "Model"]
        model_y = model[0].get_ydata()
        labels = (ax.get_ylabel(), ax_oc.get_ylabel(), ax_oc.get_xlabel())
    finally:
        plt.close(fig)

    y_lo = (1e-12 - 1e-14) / 2.0
    np.testing.assert_allclose(ylim, [y_lo, (1e-10 + 1e-12) * 2.0])
    np.testing.assert_allclose(xlim, [0.05, 15.0 * 1.5])
    np.testing.assert_allclose(model_y, y * 0.9)
    assert labels == ("flux", sp.OC_YLABEL, "wave")


def test_the_renderer_names_no_component():
    """
    Given every component the factory discovers,
    When the summary renderer's source is read,
    Then no string literal in it is a component's yaml_key: it reaches a
      component only through what a chart declares (meta["summary"]) and
      the Component hooks, so a rename inside a component cannot break the
      page, and a component from another field joins it by declaring a
      layout.
    """
    import ast
    import inspect

    from exozippy.components.factory import discover_components

    keys = set(discover_components())
    tree = ast.parse(inspect.getsource(sp))
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert keys and not (literals & keys), sorted(literals & keys)


def test_a_component_from_another_field_joins_the_page_by_declaring():
    """
    Given a system whose only component is not an astronomy one (yaml_key
      "assay"), whose chart declares the track layout with a window and a
      mark, whose summary_header headlines one line and whose
      summary_posterior marks the reported point,
    When the summary figure is drawn,
    Then it has that one panel, framed by the declared window and the
      marker, with the mark labelled, under the header line -- nothing in
      the renderer had to know the component.
    """
    from exozippy.chart import Chart, Trace

    chart = Chart(
        id="assay.curve",
        component={"yaml_key": "assay", "instance": None},
        title="",
        xlabel="dose",
        ylabel="response",
        traces=[Trace("fit", "model", "line", [0.0, 10.0], [0.0, 1.0])],
        meta={
            "summary": {
                "layout": "track",
                "window": [2.0, 8.0, 0.2, 0.8],
                "marks": [{"x": 5.0, "y": 0.5, "text": "EC50"}],
            }
        },
    )
    assay = SimpleNamespace(
        plot_data=lambda system, point: [chart],
        summary_header=lambda system: ["$EC_{50} = 5.0$"],
        summary_posterior=lambda system: {
            "assay.curve": {"marker": ((5.0, 0.5, 0.5), (0.5, 0.1, 0.1))}
        },
    )
    system = SimpleNamespace(active_components={"assay": assay})

    fig = sp.summary_figure(
        system, {}, header_lines=sp.summary_header_lines(system)
    )
    try:
        (ax,) = fig.get_axes()
        texts = [t.get_text() for t in fig.texts]
        marks = [t.get_text() for t in ax.texts]
        (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    finally:
        plt.close(fig)

    assert texts == ["$EC_{50} = 5.0$"]
    assert marks == ["EC50"]
    assert left < 2.0 and right > 8.0 and bottom < 0.2 and top > 0.8


def test_a_fit_with_nothing_to_draw_raises_its_own_error():
    """
    Given a system none of whose components makes a summary panel (as a
      microlensing fit's do not),
    When the summary figure is asked for,
    Then it raises NoSummaryPanels, a ValueError of its own class.
    """
    system = SimpleNamespace(active_components={})

    with pytest.raises(sp.NoSummaryPanels):
        sp.summary_figure(system, {})
    assert issubclass(sp.NoSummaryPanels, ValueError)


def test_the_wrapup_writer_skips_a_fit_with_nothing_to_draw(
    tmp_path, monkeypatch, caplog
):
    """
    Given a fit none of whose components makes a summary panel,
    When write_summary_plot is called, as run.py's wrap-up calls it for
      every fit,
    Then it writes nothing, returns None and says why at INFO, without
      raising -- the wrap-up catches nothing, so a microlensing fit must
      not fail there.
    """
    monkeypatch.setattr(
        sp, "median_draw_point", lambda system, posterior: ({}, (0, 0), 0.0)
    )
    system = SimpleNamespace(active_components={})
    out = tmp_path / "fit_mcmc_summary.pdf"

    with caplog.at_level("INFO", logger=sp.logger.name):
        written = sp.write_summary_plot(system, None, out)

    assert written is None
    assert not out.exists()
    assert "declares a summary chart" in caplog.text


def _kiel_chart():
    """A Kiel chart whose track runs from EEP 150 to 600, cooling and
    dropping in gravity with age; the main sequence is EEP 202-454.  Its
    summary hints are the evolutionary model's own (summary_track_marks)."""
    from exozippy.chart import Chart, Trace
    from exozippy.components.evolutionarymodel.plot import MISTPlot

    eep = np.linspace(150.0, 600.0, 451)
    teff = 4900.0 + 1.2 * (eep - 150.0)
    logg = 4.7 - 0.0015 * (eep - 150.0)
    age = 0.02 * (eep - 150.0)
    rows = {"eep": eep, "teff": teff, "logg": logg, "age": age}
    return Chart(
        id="evolutionarymodel.kiel.star.A",
        component={"yaml_key": "evolutionarymodel", "instance": None},
        title="",
        xlabel="Teff",
        ylabel="logg",
        traces=[
            Trace("Star A MIST track", "model", "line", teff, logg),
            Trace(
                "Star A MIST model point", "data", "scatter", [5300.0], [4.5]
            ),
            Trace("Star A fit value", "data", "scatter", [5200.0], [4.55]),
        ],
        x_range=[4000.0, 7000.0],
        y_range=[3.0, 5.0],
        x_inverted=True,
        y_inverted=True,
        meta={
            "summary": {
                "layout": "track",
                **MISTPlot.summary_track_marks(MISTPlot, rows),
            }
        },
    )


def test_the_kiel_panel_frames_the_main_sequence_star_and_contours():
    """
    Given a Kiel chart, contour samples clustered about 5200 K and 4.55, and
      the star's reported median 5210 K and 4.56 with asymmetric errors (the
      evolutionary model's summary_posterior),
    When the track panel is drawn,
    Then the reversed axes span the main sequence up to the turnoff (EEP
      202-454, the declared window), the star and the contours, padded; the
      star is a red cross at its reported median with those errors; the
      MIST model point is not drawn; three reference ages are labelled along
      the main sequence; and the legend holds only the two contour entries.
    """
    from matplotlib.contour import ContourSet

    rng = np.random.default_rng(5)
    teff = rng.normal(5200.0, 40.0, 1500)
    logg = rng.normal(4.55, 0.015, 1500)
    overlay = {
        "contours": [
            ("MIST", teff + 20.0, logg - 0.01),
            ("Global fit", teff, logg),
        ],
        "marker": ((5210.0, 50.0, 70.0), (4.56, 0.02, 0.03)),
    }
    fig = plt.figure()
    try:
        sp._draw_track(
            fig, fig.add_gridspec(1, 1)[0, 0], _kiel_chart(), overlay
        )
        (ax,) = fig.get_axes()
        contours = [c for c in ax.get_children() if isinstance(c, ContourSet)]
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        ages = [t.get_text() for t in ax.texts]
        (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
        (cross,) = ax.containers
        # The horizontal bar: from median - minus to median + plus.
        xbar = cross.lines[2][0].get_segments()[0]
        markers = [ln for ln in ax.get_lines() if ln.get_marker() == "D"]
    finally:
        plt.close(fig)

    # Main sequence: EEP 202 -> Teff 4962.4, logg 4.622; EEP 454 -> 5264.8,
    # 4.244; padded by 6% of the span either way.
    ms_lo_t, ms_hi_t = 4900.0 + 1.2 * 52.0, 4900.0 + 1.2 * 304.0
    ms_lo_g, ms_hi_g = 4.7 - 0.0015 * 304.0, 4.7 - 0.0015 * 52.0
    assert left > ms_hi_t and right < ms_lo_t
    assert bottom > ms_hi_g and top < ms_lo_g
    assert left > 5210.0 + 70.0 and bottom > 4.56 + 0.03
    np.testing.assert_allclose(xbar, [[5160.0, 4.56], [5280.0, 4.56]])
    assert markers == []
    assert len(contours) == 2
    assert legend == [
        r"MIST ($1\sigma$, $2\sigma$)",
        r"Global fit ($1\sigma$, $2\sigma$)",
    ]
    # Main-sequence ages run 0.02 * (202 - 150) = 1.04 to 6.08 Gyr; the
    # quarter points 2.30, 3.56 and 4.82 round to 2, 4 and 5.
    assert ages == ["2 Gyr", "4 Gyr", "5 Gyr"]


def test_a_track_without_a_main_sequence_declares_no_window_or_marks():
    """
    Given a drawn track that never reaches the main sequence (EEP < 202),
    When the evolutionary model builds its summary hints,
    Then it declares no window and no marks, and the panel still draws.
    """
    from exozippy.components.evolutionarymodel.plot import MISTPlot

    eep = np.linspace(100.0, 180.0, 50)
    rows = {"eep": eep, "teff": eep * 30.0, "logg": 4.0 + 0 * eep, "age": eep}

    hints = MISTPlot.summary_track_marks(MISTPlot, rows)

    assert hints == {"window": None, "marks": []}


def test_reference_ages_are_round_and_spread_through_the_span():
    """
    Given main-sequence age spans from tens of Myr to past the age of the
      universe,
    When three reference ages are chosen,
    Then they sit near the quarter points, rounded to one significant figure
      unless that would merge two of them, and an empty span has none.
    """
    from exozippy.components.evolutionarymodel.plot import MISTPlot

    ages = MISTPlot.reference_ages
    assert ages(0.04, 13.8, 3) == [3.0, 7.0, 10.0]
    assert ages(0.02, 1.2, 3) == [0.3, 0.6, 0.9]
    assert ages(1.0, 1.4, 3) == [1.1, 1.2, 1.3]
    assert ages(2.0, 2.0, 3) == []


def test_cli_refuses_an_unknown_options_key(tmp_path):
    """
    Given an --options file with a misspelled keyword,
    When exozippy-summary runs,
    Then it exits with a usage error naming the key, before building
      anything.
    """
    options = tmp_path / "options.yaml"
    options.write_text("titel: TOI-1234\n")

    result = CliRunner().invoke(
        cli_summary.main,
        [str(tmp_path / "no_such_fit.yaml"), "--options", str(options)],
    )

    assert result.exit_code == 2
    assert "titel" in result.output


# ---------------------------------------------------------------------------
# End to end, from a real trace on disk (slow)
# ---------------------------------------------------------------------------


def _kelt4_config():
    """kelt4: two RV instruments plus one TESS sector, the smallest fit that
    makes a transit stack and both RV panels."""
    return {
        "run": {"name": "KELT-4A"},
        "prefix": "fitresults/KELT-4A",
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "transit": [
            {
                "name": "TESS_S48",
                "file": "n20220130.TESS.TESS.TIC165297570.S48.0120.SPOC.dat",
                "band": "TESS",
                "exptime": 2.0,
                "ninterp": 1.0,
            }
        ],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "rvinstrument": [
            # A display label in the config (display only: it does not
            # enter the trace's structural hash).
            {
                "name": "HIRES",
                "file": "KELT-4b.HIRES.rv",
                "label": "Keck/HIRES",
            },
            {"name": "TRES", "file": "KELT-4b.TRES.rv"},
        ],
        "parameter_file": "summary_test.params.yaml",
        # nutpie, not PyMC NUTS: it compiles through numba, while the C
        # backend's code for this model's gradient exceeds Apple clang's
        # 256-deep bracket nesting limit.  A core dependency on every
        # platform, like the hat3 example's sampler.
        "sampler": {
            "method": "nutpie",
            "tune": 5,
            "draws": 4,
            "chains": 1,
            "cores": 1,
            "measure_scales": False,
            "recompute_trace": True,
            # Reproducible on one platform (not across them: the build is
            # platform-dependent at ~1e-9).
            "seed": 418,
        },
        # Four draws after five tuning steps are not a converged posterior,
        # and the mode pass's robust-z filter (median and MAD of 4 draws)
        # flags one of them as "raw-z" invalid on some seeds -- 25%, over
        # the 1% default, which fails the fit's wrap-up (seen on macOS CI).
        # These tests are about the figure, not the sampler, so tolerate
        # it; a fit where most draws are invalid still fails.
        "modes": {"max_invalid_frac": 0.5},
        "modeling": {"compile": False},
    }


# Star A's literature values as priors (there is no SED or relation to
# constrain it here) and the transit geometry of the example's own params.
# Not kelt4_rv+transit+sed.params.yaml, which also names stars B and C.
_KELT4_PARAMS = {
    "star.A.radius": {"initval": 1.610, "mu": 1.610, "sigma": 0.05},
    "star.A.mass": {"initval": 1.204, "mu": 1.204, "sigma": 0.05},
    "star.A.teff": {"initval": 6207, "mu": 6207, "sigma": 100},
    "star.A.feh": {"initval": -0.116, "mu": -0.116, "sigma": 0.08},
    "planet.b.radius": {"initval": 1.706},
    "orbit.b.period": {"initval": 2.9895933},
    "orbit.b.tc": {"initval": 2459634.3},
    "orbit.b.cosi": {"initval": 0.11996},
}


@pytest.fixture(scope="module")
def kelt4_fit(tmp_path_factory):
    """A finished (tiny) kelt4 fit in a copy of the example directory:
    returns the path of its config.  run_fit is called once per module."""
    if not _KELT4_DIR.is_dir():
        pytest.skip("kelt4 example not present")
    from exozippy.run import run_fit

    work_dir = tmp_path_factory.mktemp("summary") / "kelt4"
    shutil.copytree(
        _KELT4_DIR,
        work_dir,
        ignore=shutil.ignore_patterns("fitresults*", ".#*", "#*#"),
    )
    config = _kelt4_config()
    config_path = work_dir / "summary_test.yaml"
    config_path.write_text(yaml.safe_dump(config))
    (work_dir / config["parameter_file"]).write_text(
        yaml.safe_dump(_KELT4_PARAMS)
    )

    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        run_fit(config)
    finally:
        os.chdir(cwd)
    # Kept aside before any test redraws the default file, so the wrap-up
    # test below sees what the fit itself wrote.
    wrapup = work_dir / "fitresults" / "KELT-4A_mcmc_summary.pdf"
    if wrapup.exists():
        shutil.copy(wrapup, work_dir / "summary_written_by_the_fit.pdf")
    return config_path


@pytest.fixture(scope="module")
def kelt4_posterior(kelt4_fit):
    """The kelt4 fit's System, rebuilt, with its reported posterior and
    median-draw point."""
    from exozippy.system import System

    cwd = os.getcwd()
    os.chdir(kelt4_fit.parent)
    try:
        system = System(_kelt4_config())
        system.prepare()
        model = system.build_model()
        idata = az.from_netcdf("fitresults/KELT-4A_trace.nc")
        posterior = sp.reported_posterior(system, model, idata)
        point, where, _distance = sp.median_draw_point(system, posterior)
    finally:
        os.chdir(cwd)
    return system, posterior, point, where


@pytest.mark.slow
def test_every_fit_writes_its_summary_figure_at_wrapup(kelt4_fit):
    """
    Given a fit run through run_fit,
    When its wrap-up finishes,
    Then <prefix>_mcmc_summary.pdf is among its outputs, with no call to
      create_summary_plot.
    """
    written = kelt4_fit.parent / "summary_written_by_the_fit.pdf"

    assert written.exists() and written.stat().st_size > 0


@pytest.mark.slow
def test_create_summary_plot_writes_the_default_file(kelt4_fit):
    """
    Given a finished fit, and a working directory that is NOT the config's,
    When create_summary_plot is called with just the config path,
    Then it resolves the fit's relative paths against the config's
      directory and writes <prefix>_mcmc_summary.pdf there.
    """
    assert Path.cwd() != kelt4_fit.parent

    out = sp.create_summary_plot(kelt4_fit)

    expected = kelt4_fit.parent / "fitresults" / "KELT-4A_mcmc_summary.pdf"
    assert out == expected.resolve()
    assert out.stat().st_size > 0


@pytest.mark.slow
def test_median_draw_point_is_a_valid_draw_in_internal_units(
    kelt4_posterior,
):
    """
    Given the fit's reported (post-burn-in) posterior,
    When median_draw_point picks the draw,
    Then it is a valid draw of that posterior, and its physical values
      convert back through the Parameters to exactly the trace's user-unit
      values at that draw.
    """
    system, posterior, point, (chain, draw) = kelt4_posterior
    labels = posterior.posterior["mode"].values

    assert labels[chain, draw] >= 0
    lookup = system.get_parameter_lookup()
    for label in ("orbit.period", "planet.mass", "rvinstrument.gamma"):
        np.testing.assert_allclose(
            lookup[label].from_internal(point[label]),
            posterior.posterior[label].values[chain, draw],
            rtol=1e-12,
        )


@pytest.mark.slow
def test_summary_figure_draws_one_panel_per_chart_kind(kelt4_posterior):
    """
    Given the fit at its median draw, with a `label:` on HIRES in the
      config and no labels or groups passed in,
    When summary_figure draws it with a binned transit and three other
      posterior draws,
    Then the page has the transit stack, the RVs against time and the
      folded RVs (each of the last two with an O-C axis), the header lines
      it was given, the TESS sector on its cadence row "TESS 120 s", the
      folded panel's legend showing HIRES by its configured label and TRES
      by its name; the transit and folded RV curves are the median draw's
      and one per extra draw, all at one faint alpha, while the RVs
      against time show the median draw's curve alone, at full weight.
    """
    from exozippy.run import get_draws

    system, posterior, point, _ = kelt4_posterior
    header = sp.summary_header_lines(system)
    draws = get_draws(
        posterior,
        n_draws=3,
        param_lookup=system.get_parameter_lookup(),
        exclude=set(system.report_only_labels()),
    )

    fig = sp.summary_figure(
        system,
        point,
        title="KELT-4A",
        header_lines=header,
        transit_bin=10,
        draws=draws,
    )
    try:
        texts = [t.get_text() for t in fig.texts]
        axes = fig.get_axes()
        ylabels = [ax.get_ylabel() for ax in axes]
        transit_ax = axes[ylabels.index("Normalized Flux + Constant")]
        legend_texts = [
            t.get_text()
            for ax in axes
            if ax.get_legend() is not None
            for t in ax.get_legend().get_texts()
        ]
        # Model curves are the solid lines; points and break marks have no
        # line style.  Transit, then the first RV time piece, then folded.
        curves = [
            [ln.get_alpha() for ln in ax.get_lines() if ln.get_ls() == "-"]
            for ax in axes
            if ax.get_ylabel() in ("RV [m/s]", "Normalized Flux + Constant")
        ]
    finally:
        plt.close(fig)

    # The RV time axis is broken at the >50-day gaps in the HIRES and TRES
    # seasons: two axes (panel and O-C) per piece.
    (rv_time,) = [
        c
        for c in sp._collect_charts(system, point)
        if c.id.startswith("rvinstrument") and not c.meta.get("phase_folded")
    ]
    times = np.concatenate([np.ravel(t.x) for t in sp._role(rv_time, "data")])
    pieces = len(sp.time_segments(times, sp.RV_BREAK_DAYS))
    assert pieces > 1
    assert len(axes) == 3 + 2 * pieces
    assert ylabels.count(sp.OC_YLABEL) == 2
    rv_axes = [ax for ax in axes if ax.get_ylabel() == "RV [m/s]"]
    assert len(rv_axes) == 2
    assert rv_axes[0].get_ylim() == rv_axes[1].get_ylim()
    assert texts[0] == "KELT-4A" and texts[1:] == header
    assert len(header) == 1
    for symbol in ("$P = ", "$R_P = ", "$M_P = ", "$e = "):
        assert symbol in header[0]
    assert [t.get_text() for t in transit_ax.texts] == ["TESS 120 s"]
    assert {"Keck/HIRES", "TRES", "Model"} <= set(legend_texts)
    spaghetti = [sp._DRAWS_ALPHA] * (1 + len(draws))
    assert curves == [spaghetti, [sp._RV_TIME_MODEL["alpha"]], spaghetti]


@pytest.mark.slow
def test_summary_figure_rejects_keywords_that_do_not_fit_the_fit(
    kelt4_posterior,
):
    """
    Given the fit (transit TESS_S48; RVs HIRES and TRES),
    When the summary is asked to group an RV instrument as a transit, to
      bin a row that does not exist, or to label an unknown instrument,
    Then each raises, naming the offending name.
    """
    system, _, point, _ = kelt4_posterior

    with pytest.raises(ValueError, match="'HIRES'"):
        sp.summary_figure(system, point, transit_groups={"x": ["HIRES"]})
    with pytest.raises(ValueError, match="'TESS'"):
        sp.summary_figure(system, point, transit_bin={"TESS": 10})
    with pytest.raises(ValueError, match="'FIES'"):
        sp.summary_figure(system, point, labels={"FIES": "FIES"})
    plt.close("all")


@pytest.mark.slow
def test_a_trace_from_another_model_is_refused(kelt4_fit):
    """
    Given the fit's trace and a config that builds a DIFFERENT model (one RV
      instrument dropped) with the same prefix,
    When create_summary_plot is called on that config,
    Then it raises StaleTraceError, as every trace reload does.
    """
    from exozippy.trace_meta import StaleTraceError

    config = _kelt4_config()
    config["rvinstrument"] = config["rvinstrument"][:1]
    other = kelt4_fit.parent / "summary_test_other.yaml"
    other.write_text(yaml.safe_dump(config))

    with pytest.raises(StaleTraceError):
        sp.create_summary_plot(other, kelt4_fit.parent / "never.pdf")


@pytest.mark.slow
def test_cli_writes_the_figure_with_an_options_file(kelt4_fit, tmp_path):
    """
    Given the finished fit and an --options file of labels and a per-row bin,
    When exozippy-summary runs with --output and a --title flag,
    Then it exits cleanly and writes the requested file.
    """
    options = tmp_path / "options.yaml"
    options.write_text(
        yaml.safe_dump(
            {
                "title": "overridden by the flag",
                "labels": {"TRES": "Tillinghast/TRES"},
                "transit_bin": {"TESS 120 s": 10},
                "figsize": [17, 14],
            }
        )
    )
    out = tmp_path / "summary.png"

    result = CliRunner().invoke(
        cli_summary.main,
        [
            str(kelt4_fit),
            "--output",
            str(out),
            "--title",
            "KELT-4A",
            "--options",
            str(options),
        ],
    )

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert out.stat().st_size > 0
