"""The system summary figure: a finished fit's model-bearing panels on one page.

Every fit writes ``<prefix>_mcmc_summary.pdf`` at wrap-up (``run.py`` calls
``write_summary_plot``), and ``create_summary_plot(config_file)`` redraws it
from what a fit leaves behind -- the config it ran with and
``<prefix>_trace.nc`` -- in the layout of the one-page system figure of an
exoplanet discovery paper:
each planet's phase-folded transits stacked with offsets, the RVs against time
and folded on each orbit (each with its O-C), the SED (with its O-C) and each
star's Kiel diagram, under a header quoting every planet's P, R_P, M_P and e.
A panel is drawn only when the fit has the component behind it, so an RV-only
or a star-only fit gets a shorter page rather than empty axes.

THIS MODULE ADDS NO PHYSICS AND KNOWS NO COMPONENT.  It is a third renderer
of the components' Charts (``plotrender.py`` and the GUI's
``plotly-adapter.ts`` are the other two): every curve and point is what
``Component.plot_data(system, point)`` returned, and every O-C is the
component's own ``meta["residuals"]`` -- data minus the likelihood's model
at the observations, which only the component can evaluate, given in the
chart's own units.  A chart is on the page when it declares
``meta["summary"]``: one of the ``LAYOUTS`` and the hints that layout reads
(display labels, colors, rows, axis labels, marks); the header is every
component's ``summary_header`` and a track panel's contours and marked point
its component's ``summary_posterior`` (both on ``Component``).  Nothing here
names a component, a filter or a unit, so a component from another field
joins the page by declaring a layout.  What it owns is the presentation:
stacking the folded rows with offsets (optionally binned), an offset on a
time axis and a break in it at every gap of over ``RV_BREAK_DAYS``, a
log10 chart drawn as the quantity on a log axis, and the page layout.
It draws in its own style, that of the one-page EXOFASTv2-era figures it
replaces (``PALETTE`` and the style constants below), not the role encodings
the per-component PDFs and the GUI share.  A new panel kind is a new
component chart, not new arithmetic here.

THE REFERENCE POINT IS THE MEDIAN DRAW OF THE REPORTED POSTERIOR.
``reported_posterior`` reproduces what ``run.py`` reports from -- the declared
label degeneracies folded, burn-in and stuck chains trimmed
(``convergence.analyze_idata``), draws the mode pass rejects as numerically
invalid labelled -1 -- and ``median_draw_point`` takes
the valid draw nearest its median: one joint draw, never a vector of
per-parameter medians, which need not be a point the posterior contains, and
a typical draw rather than the highest-lp one (see that function for why).
It plays the part ``points[0]`` plays for ``plotrender``: the data and its
cleaning, every O-C, the SED's model photometry, and the Kiel track, window
and ages are its.  The other model curves are EXOZIPPy's spaghetti: the
median draw's and those of other posterior draws (``run.get_draws``, the
same 50 the component PDFs drew when the fit writes the figure), all at one
low alpha, none darker.  The RVs against time show the median draw's curve
alone, legible over many orbits, and the Kiel diagram shows its posterior
as contours instead.  The header and the Kiel diagram quote the medians and
credible intervals of that same posterior, at the run's credible-interval
width.
"""

import contextlib
import dataclasses
import difflib
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: The layouts a chart can declare in ``meta["summary"]["layout"]``, in the
#: page's reading order: a stack of phase-folded rows (one panel per
#: ``stack``), a time series (its axis broken at long gaps), a phase fold, a
#: spectrum (y declared as log10 of the drawn quantity) and a track with
#: posterior contours.  The vocabulary and every hint each layout reads are
#: in outputs.md, "The system summary figure".
LAYOUTS = ("stacked_fold", "time_series", "phase_fold", "spectrum", "track")

#: The run.py default, for a config that names no prefix.
DEFAULT_PREFIX = "fitresults/planet"


class NoSummaryPanels(ValueError):
    """No component of the fit declares a summary chart
    (``meta["summary"]``) -- a microlensing fit, say, whose components
    declare none.  ``summary_figure``
    and an explicit ``create_summary_plot`` raise it.  ``write_summary_plot``
    instead returns None for such a fit, so the live fit's wrap-up, which
    catches nothing (run.md, "A fit has three phases"), skips the figure
    with an INFO line."""


#: Every O-C axis is in the units of the panel above it, so it names none.
OC_YLABEL = "O-C"

#: A time series' axis is broken wherever consecutive observations are
#: more than this many days apart (``summary_figure(rv_break_days=)``).
RV_BREAK_DAYS = 50.0

# Page geometry.  A transit stack is two grid rows tall, every other panel
# one; a panel with an O-C gives it a quarter of its height.
_TRANSIT_ROWS = 2
_PANEL_ROWS = 1
_ROW_HEIGHT_IN = 6.0
_COLUMN_WIDTH_IN = 8.5
_OC_HEIGHT_RATIOS = (3, 1)
# A broken RV time axis: each piece padded by this fraction of the time the
# pieces cover (at least _BREAK_MIN_PAD_DAYS), as wide as its padded span
# but never narrower than _BREAK_MIN_WIDTH of the whole, and the pieces
# _BREAK_WSPACE apart.  Its x ticks are about _BREAK_TICK_IN apart, none
# within _BREAK_EDGE_IN of an edge facing another piece, where its label
# would run into the neighbor's.
_BREAK_PAD_FRAC = 0.04
_BREAK_MIN_PAD_DAYS = 1.0
_BREAK_MIN_WIDTH = 0.12
_BREAK_WSPACE = 0.06
_BREAK_TICK_IN = 0.9
_BREAK_EDGE_IN = 0.3
# The diagonal marks on the facing edges of two pieces.
_BREAK_MARK = {
    "marker": [(-0.5, -1.0), (0.5, 1.0)],
    "markersize": 12,
    "ls": "none",
    "color": "k",
    "mec": "k",
    "mew": 1.5,
    "clip_on": False,
    "zorder": 20,
}
# An O-C axis's x10^n multiplier: its right end, in axes fraction, and how
# far (points) it is dropped from matplotlib's place above the axes -- both
# clear of the spines' inward ticks.
_OC_OFFSET_X = 0.98
_OC_OFFSET_DROP_PT = 10.0
_TITLE_HEIGHT_IN = 0.55
_LINE_HEIGHT_IN = 0.38

_FONT_RC = {
    "font.size": 13,
    "axes.labelsize": 15,
    "legend.fontsize": 13,
}
_TICK_LABELSIZE = 13
# The dense transit points are rasterized inside the vector page; this is
# their resolution.
_SAVE_DPI = 200

#: The palette, after the one-page system figures this layout comes from.
#: Transit rows take it from the front; RV instruments and SED identities
#: from the back, so the first row and the first RV instrument differ.  A
#: user's per-instrument ``plot: color:`` wins over it.
PALETTE = (
    "#009B77",
    "#821EA6",
    "#34568B",
    "#D1AF19",
    "#95DEE3",
    "#88B04B",
    "#955251",
    "#5B5EA6",
    "#9B2335",
    "#E6AF91",
    "#D65076",
    "#422C7A",
    "#378011",
    "#FF00FF",
    "#EA832F",
    "#2BD27E",
    "#000000",
    "#AA2871",
    "#A4BC1A",
    "#0661AC",
)

# Line weights and marks, after the same figures.  Ticks point in and out,
# long and heavy; points are filled circles with black edges.
_TICK_MAJOR = {"direction": "inout", "length": 10, "width": 2}
_TICK_MINOR = {"direction": "inout", "length": 6, "width": 1}
_EDGE_WIDTH = 0.8
_ZERO_LINE = {"color": "grey", "ls": "--", "lw": 2.0}
# Transit stack.  Points under a binned set are fainter, not faint.
_RAW_MARKERSIZE = 9.0
_RAW_ALPHA = 0.7
_RAW_ALPHA_UNDER_BINS = 0.3
_BIN_MARKERSIZE = 10.0
_BIN_ALPHA = 0.9
_TRANSIT_MODEL = {"color": "k", "lw": 3.0}
# The automatic offset between rows: the deepest transit plus this many
# times the typical (median over rows) scatter of the points drawn.
_SPACING_SIGMAS = 4.0
# Each row's label sits at the left, this many times the row's own scatter
# above its baseline (plus a few points of padding), but no more than this
# fraction of the way to the row above.
_LABEL_SIGMAS = 2.0
_LABEL_MAX_GAP = 0.2
_LABEL_PAD_PT = 2.0
# RVs: the folded model dark grey; against time, where it runs over many
# orbits, a thin dark line beneath the points.
_RV_MARKERSIZE = 6.0
_CAPSIZE = 4
_RV_MODEL = {"color": "k", "lw": 2.0, "alpha": 0.7}
_RV_TIME_MODEL = {"color": "k", "lw": 0.6, "alpha": 0.8}
_MODEL_LABEL = "Model"
# Model curves follow EXOZIPPy's own posterior plots (plotrender's rule):
# with other posterior draws, EVERY model curve -- the median draw's
# included -- is drawn at one low alpha and none is darker, since no one
# draw is "the" fit; a lone point's curves keep the weights above.  The
# SED's spectra take the SED component's own spaghetti alpha.  A legend
# swatch is raised to a legible alpha, as plotrender's
# _LEGEND_MIN_ALPHA does.  Two panels are exceptions and draw the median
# draw alone.  The Kiel diagram's contours show the posterior, and its one
# track carries the reference ages, which fifty overlapping tracks would
# leave on no track the eye can follow.  The RVs against time span many
# orbits, where fifty faint curves blur into a grey band; the folded panel
# shows their spread.
_PANELS_WITH_DRAWS = frozenset({"stacked_fold", "phase_fold", "spectrum"})
_N_DRAWS = 50
_DRAWS_ALPHA = 0.1
_SED_DRAWS_ALPHA = 0.15
_LEGEND_ALPHA = 0.8
# SED: the model spectra are smoothed with a boxcar of this many points;
# the axes span the bandpasses and the photometry, padded by these factors.
_SED_MARKERSIZE = 8.0
_SED_SPECTRUM_LW = 1.0
_SED_SMOOTH = 9
_SED_X_PAD = 1.5
_SED_Y_PAD = 2.0
# Track (the Kiel diagram): the track heavy blue with its marks, the
# reported point a red cross with its (asymmetric) errors, and the 1- and
# 2-sigma contours (highest-density regions enclosing 68.27% and 95.45% of
# the draws) of each posterior sample set, the first in black and the second
# in green.  Only the contours have legend entries.
_TRACK_LINE = {"color": "b", "lw": 3.0}
_TRACK_MARKER = {"ecolor": "r", "elinewidth": 2.0, "capsize": 3}
_CONTOUR_PROBS = (0.9545, 0.6827)
_CONTOUR_COLORS = ("k", "g")
_CONTOUR_LW = 1.5
# The window frames the chart's declared window, the reported point (with
# its error bars) and the 2-sigma contours, padded by this fraction of its
# span on each side.
_TRACK_PAD_FRAC = 0.06
_TRACK_MARK = {"color": "b", "marker": "o", "ms": 6}


# ---------------------------------------------------------------------------
# The posterior and the point
# ---------------------------------------------------------------------------


def reported_posterior(system, model, idata):
    """The posterior a fit's tables and plots describe, distributed onto
    ``system``'s Parameters.

    ``idata`` is the trace as saved -- run.py keeps the FULL, untrimmed trace
    on disk and derives every report from it -- so this repeats what
    ``run._wrap_up`` does between that trace and the tables:
    ``run.fold_degeneracies`` collapses the declared label degeneracies
    (not written to disk), ``convergence.analyze_idata`` drops burn-in and
    stuck chains, and ``identify_modes`` labels the draws it rejects as
    numerically invalid -1, which ``distribute_posterior`` then leaves out
    of every summary and ``median_draw_point`` out of its choice.  Labels
    an earlier mode pass left on the trace are dropped first, as
    ``report_pipeline`` drops them (review 2.11.8).  Any mode-pass failure
    raises, ``NoValidDrawsError`` included: a figure of rejected draws is
    meaningless, and every other failure is a bug ``report_pipeline``
    raises on too.

    Returns the trimmed InferenceData, ``posterior["mode"]`` attached.
    ``system`` must be built (``prepare()`` + ``build_model()``) and
    ``model`` is what ``build_model`` returned.
    """
    from ..run import fold_degeneracies
    from ..samplers import convergence
    from .modes import identify_modes

    fold_degeneracies(system, model, idata)
    report_only = set(system.report_only_labels())
    trimmed, _ = convergence.analyze_idata(idata, exclude=report_only)
    if "mode" in trimmed.posterior.data_vars:
        del trimmed.posterior["mode"]
    report = identify_modes(trimmed)
    if report.n_modes > 1:
        logger.warning(
            f"summary plot: the posterior has {report.n_modes} modes.  "
            "The panels are drawn at the median draw, which lies in "
            "one of them, and the header quotes the COMBINED posterior; "
            "the per-mode values are in the results table."
        )
    system.distribute_posterior(trimmed)
    return trimmed


def median_draw_point(system, idata):
    """The valid draw of ``idata`` nearest its posterior median, as a
    plotting point.

    Returns ``(point, (chain, draw), distance)``.  Nearest in the SAMPLED
    coordinates -- every ``*_raw`` variable, the unconstrained space the
    sampler moves in -- with each element centered on its median and scaled
    by its posterior standard deviation; ``distance`` is the chosen draw's
    root-sum-square of those.  One joint draw, so every curve on the page
    comes from a point the posterior contains (a vector of per-parameter
    medians need not be one), and a TYPICAL draw, agreeing with the medians
    the header and the Kiel diagram quote.

    It is deliberately not the highest-lp draw.  With dozens of parameters,
    the best-scoring of thousands of draws is picked out by noise in the
    photometry's thousands of terms, and can sit well out in the parameters
    the eye checks: measured on a TOI-7475 fit (16000 draws), the highest-lp
    draw had its period 2.4 sigma off and both RV jitters inflated to
    150-190 m/s, and its RV chi-square on the quoted errors was 303 against
    a median of 19 over 200 random draws -- a phased RV curve visibly off
    the data that every random draw fits.

    ``point`` maps every posterior variable but ``mode`` and the report-only
    Deterministics to its value at that draw, in INTERNAL units -- the form
    ``Component.plot_data`` takes, built the way ``run.get_draws`` builds
    its spaghetti draws, but through ``Parameter.to_internal`` since a
    single draw is the element vector the owner's conversion expects.  Draws
    the mode pass labelled -1 (numerically invalid) are never chosen, and do
    not enter the median.
    """
    posterior = idata.posterior
    raw = [v for v in posterior.data_vars if v.endswith("_raw")]
    if not raw:
        raise ValueError(
            "The trace has no sampled (*_raw) variables to find its median "
            "draw in."
        )
    shape = (posterior.sizes["chain"], posterior.sizes["draw"])
    valid = np.ones(shape, dtype=bool)
    if "mode" in posterior:
        valid = np.asarray(posterior["mode"].values) >= 0
    if not valid.any():
        raise ValueError("No valid draw in the trace to plot.")

    distance2 = np.zeros(shape)
    for var in raw:
        values = np.asarray(posterior[var].values, dtype=float)
        values = values.reshape(shape + (-1,))
        median = np.median(values[valid], axis=0)
        scale = np.std(values[valid], axis=0)
        moving = scale > 0  # a pinned element has no width to scale by
        z = (values[..., moving] - median[moving]) / scale[moving]
        distance2 += np.sum(z**2, axis=-1)
    distance2 = np.where(valid, distance2, np.inf)
    chain, draw = np.unravel_index(int(np.argmin(distance2)), shape)

    lookup = system.get_parameter_lookup()
    skip = set(system.report_only_labels()) | {"mode"}
    point = {}
    for var in posterior.data_vars:
        if var in skip:
            continue
        val = posterior[var].isel(chain=chain, draw=draw).values
        if var in lookup and not var.endswith("_raw"):
            val = lookup[var].to_internal(val)
        point[var] = val
    return (
        point,
        (int(chain), int(draw)),
        float(np.sqrt(distance2[chain, draw])),
    )


# ---------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------


def _decimals(err, sigfigs):
    """Decimal places that keep ``sigfigs`` significant figures of ``err``
    (negative for an error of tens or more) -- PosteriorSummary.format's rule."""
    return -int(math.floor(math.log10(err))) + (sigfigs - 1)


def format_value(summary, sigfigs=2):
    """Mathtext for one PosteriorSummary: ``median^{+err}_{-err}``.

    The median is rounded as the results table rounds it -- to the decimal
    place of the more precise error at ``sigfigs`` significant figures -- and
    the errors are written to that SAME decimal place, trailing zeros kept,
    so the three numbers carry one precision (a median of 0.084 with errors
    0.11 and 0.061 reads ``0.084^{+0.110}_{-0.061}``).  Equal errors collapse
    to ``\\pm``; a zero-spread (fixed) value renders ``\\equiv`` as the table
    renders it.  Returns None for a non-finite summary, which has nothing to
    report.
    """
    med = float(summary.median)
    lo = abs(float(summary.err_minus))
    hi = abs(float(summary.err_plus))
    if not all(math.isfinite(v) for v in (med, lo, hi)):
        return None
    if lo == 0 and hi == 0:
        return rf"\equiv {med:.6g}"
    places = max(_decimals(e, sigfigs) for e in (lo, hi) if e > 0)

    def fmt(v):
        return f"{round(v, places):.{max(places, 0)}f}"

    med_s, lo_s, hi_s = fmt(med), fmt(lo), fmt(hi)
    if lo_s == hi_s:
        return rf"{med_s} \pm {hi_s}"
    return rf"{med_s}^{{+{hi_s}}}_{{-{lo_s}}}"


def reported_summary(param, index):
    """Element ``index``'s PosteriorSummary -- the median and credible
    interval the results table and ``<prefix>_results.csv`` report -- or None
    when ``param`` has no posterior (before one is distributed)."""
    if param is None or param.posterior is None:
        return None
    param.ensure_summary()
    summary = param.summary
    return summary[index] if isinstance(summary, list) else summary


def summary_header_lines(system):
    """The header under the title: every active component's
    ``summary_header`` lines (the planet's P, R_P, M_P and e, say), in
    component order.  Called once a posterior is distributed."""
    lines = []
    for comp in system.active_components.values():
        lines.extend(comp.summary_header(system))
    return lines


# ---------------------------------------------------------------------------
# Charts -> panels
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Panel:
    kind: str  # one of LAYOUTS
    charts: list
    rows: int


def _collect_charts(system, point):
    charts = []
    for comp in system.active_components.values():
        charts.extend(comp.plot_data(system, point))
    return charts


def _draw_model_traces(system, panels, draws):
    """``{chart id: [model traces of one draw, ...]}``: the model curves of
    every chart a panel draws, at each of ``draws``.

    The spaghetti ``plotrender.render_spec_groups`` overlays, gathered the
    same way: each draw's ``plot_data``, matched to the panel's chart by id,
    model traces only (the data and its cleaning are the reference point's).
    A draw whose ``plot_data`` raises is a component bug at a posterior
    draw and propagates; it is not dropped from the spaghetti.  Only the
    panels in
    ``_PANELS_WITH_DRAWS`` are gathered for, so no other component is
    evaluated at the draws.
    """
    panels = [p for p in panels if p.kind in _PANELS_WITH_DRAWS]
    wanted = {chart.id for panel in panels for chart in panel.charts}
    keys = {
        chart.component.get("yaml_key")
        for panel in panels
        for chart in panel.charts
    }
    components = [system.active_components[key] for key in sorted(keys)]
    models = {}
    for extra in draws:
        for comp in components:
            for chart in comp.plot_data(system, extra):
                traces = _role(chart, "model")
                if chart.id in wanted and traces:
                    models.setdefault(chart.id, []).append(traces)
    return models


def _hints(chart):
    """The chart's ``meta["summary"]``, or None for a chart not on the
    page."""
    return (chart.meta or {}).get("summary")


def _panel_kind(chart):
    """The layout ``chart`` declares (``meta["summary"]["layout"]``), or
    None for a chart that declares none.  An unknown layout raises, naming
    the chart: a component's declaration is a contract, not a guess."""
    hints = _hints(chart)
    if hints is None:
        return None
    layout = hints["layout"]
    if layout not in LAYOUTS:
        raise ValueError(
            f"summary plot: chart {chart.id!r} declares layout {layout!r}; "
            f"the layouts are {list(LAYOUTS)}."
        )
    return layout


def _panels(charts):
    """Sort the charts into panels, in reading order (``LAYOUTS``).

    One stacked panel per declared ``stack`` (a planet's phase-folded
    transits), every other chart a panel of its own.  A component none of
    whose charts declares a layout has no panel; it is logged, so a missing
    panel is not a mystery.
    """
    by_kind = {layout: [] for layout in LAYOUTS}
    stacks = {}
    owners, drawn = set(), set()
    for chart in charts:
        key = chart.component["yaml_key"]
        owners.add(key)
        kind = _panel_kind(chart)
        if kind is None:
            continue
        drawn.add(key)
        if kind == "stacked_fold":
            stacks.setdefault(_hints(chart)["stack"], []).append(chart)
        else:
            by_kind[kind].append(chart)
    if owners - drawn:
        logger.info(
            "summary plot: no summary panel for component(s) %s; their "
            "charts are in their own PDFs.",
            ", ".join(sorted(owners - drawn)),
        )
    panels = []
    for kind in LAYOUTS:
        if kind == "stacked_fold":
            panels += [_Panel(kind, c, _TRANSIT_ROWS) for c in stacks.values()]
        else:
            panels += [_Panel(kind, [c], _PANEL_ROWS) for c in by_kind[kind]]
    return panels


def _place(panels):
    """Assign each panel a ``(column, first_row)`` on the page grid.

    Greedy, in reading order: each panel goes to the shorter column (the
    left on a tie).  For the common one-planet transit+RV+SED+MIST fit that
    reproduces the classic page -- transits over the SED on the left; RVs
    against time, the folded RVs and the Kiel diagram on the right -- and it
    degrades sensibly for any subset.  One panel gets one column.
    """
    ncols = 1 if len(panels) <= 1 else 2
    heights = [0] * ncols
    placed = []
    for panel in panels:
        col = int(np.argmin(heights))
        placed.append((panel, col, heights[col]))
        heights[col] += panel.rows
    return placed, ncols, max(heights) if heights else 0


def _suggest(name, known):
    close = difflib.get_close_matches(name, known, n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _check_names(what, names, known):
    """Raise on names that are not instruments of this fit.  Case-sensitive,
    like every user-facing name in EXOZIPPy; a near miss is suggested, never
    accepted."""
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(
            f"{what} names "
            + ", ".join(f"{n!r}{_suggest(n, known)}" for n in unknown)
            + f", which this fit does not have.  Known: {sorted(known)}."
        )


def declared_row_groups(charts):
    """The default ``transit_groups``: the rows the stacked charts declare.

    ``{row label: [instruments]}`` for every declared row that is more than
    one file under its own name (``meta["summary"]["row_key"]``, labelled
    ``row_label``) -- the TESS sectors of one cadence, say
    (``Transit.summary_rows``).  Two groups that would share a label
    raise, naming both: one group is never silently replaced by another.
    """
    keys = {}
    for chart in charts:
        hints = _hints(chart)
        if hints is None or hints["layout"] != "stacked_fold":
            continue
        members = keys.setdefault(
            hints["row_key"], {"label": hints["row_label"], "names": []}
        )
        if hints["instrument"] not in members["names"]:
            members["names"].append(hints["instrument"])
    groups = {}
    for key, row in keys.items():
        # A row is a group when it is more than its one file: several
        # files, or one file under the row's own key (a TESS cadence).  A
        # file on its own row is labelled on its own, and two such files
        # may share a label (both "LCO", say); two GROUPS may not, since
        # the label is how transit_groups and transit_bin name a group.
        if row["names"] == [key]:
            continue
        label = row["label"]
        if label in groups:
            raise ValueError(
                f"summary plot: the rows {groups[label]} and "
                f"{row['names']} would both be labelled {label!r}.  Give "
                "them distinct `label:`s in the config, or pass "
                "transit_groups."
            )
        groups[label] = row["names"]
    return groups


def _transit_rows(stack, labels, groups, bins):
    """The rows of one stack, ``[(bin key, label, charts, bin_minutes,
    color)]`` in chart order, a group placed where its first member is.
    ``labels`` maps an instrument to a user's display name (else its
    declared ``label``), ``groups`` a group label to its instruments;
    ``bins`` is None, minutes, or ``{group label or instrument: minutes}``.
    A row's color is its first member's declared ``color`` (a user's
    ``plot: color:``), else the palette by row.
    """
    member_of = {m: g for g, members in groups.items() for m in members}
    rows, seen = [], {}
    for chart in stack:
        hints = _hints(chart)
        inst = hints["instrument"]
        key = (
            ("group", member_of[inst]) if inst in member_of else ("file", inst)
        )
        if key in seen:
            rows[seen[key]][2].append(chart)
            continue
        seen[key] = len(rows)
        name = key[1]
        label = name if key[0] == "group" else labels.get(inst, hints["label"])
        minutes = bins.get(name) if isinstance(bins, dict) else bins
        rows.append([name, label, [chart], minutes, None])
    for k, row in enumerate(rows):
        # A user's per-instrument plot: color (the first member's) wins;
        # otherwise the theme palette by row, so neighbors differ.
        user = _hints(row[2][0])["color"]
        row[4] = user or PALETTE[k % len(PALETTE)]
    return [tuple(r) for r in rows]


def _role(chart, role):
    return [t for t in chart.traces if t.role == role]


def _one(chart, role):
    """The chart's single trace of ``role``; a phased transit chart has
    exactly one data and one model trace, and a row of the stack is drawn
    from them, so anything else is a chart this module does not know how to
    stack -- named rather than guessed at."""
    traces = _role(chart, role)
    if len(traces) != 1:
        raise ValueError(
            f"summary plot: chart {chart.id!r} has {len(traces)} {role} "
            f"traces; a transit row is drawn from exactly one."
        )
    return traces[0]


def _bin(x, y, width):
    """Mean of ``y`` in bins of ``width`` in ``x``; empty bins dropped."""
    idx = np.floor((x - x.min()) / width).astype(int)
    counts = np.bincount(idx)
    sums = np.bincount(idx, weights=y)
    keep = counts > 0
    centers = x.min() + (np.arange(counts.size) + 0.5) * width
    return centers[keep], sums[keep] / counts[keep]


def _robust_sigma(r):
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(r - np.median(r))))


def _transit_row_arrays(row):
    """Per member chart: data (x, y), the binned set or None, and the model
    (x, y) -- x times the chart's declared ``x_scale`` (days to hours for a
    transit), ``y`` relative to the baseline, as the chart has it.  The bin
    width is in minutes of that time axis."""
    out = []
    for chart in row[2]:
        scale = _hints(chart)["x_scale"]
        data, model = _one(chart, "data"), _one(chart, "model")
        x = np.asarray(data.x, dtype=float) * scale
        y = np.asarray(data.y, dtype=float)
        xm = np.asarray(model.x, dtype=float) * scale
        ym = np.asarray(model.y, dtype=float)
        binned = _bin(x, y, row[3] * scale / 1440.0) if row[3] else None
        out.append((x, y, binned, xm, ym))
    return out


def _auto_spacing(arrays):
    """``(spacing, depth, sigmas)``: the automatic row spacing -- the
    deepest transit plus ``_SPACING_SIGMAS`` times the median of
    ``sigmas`` -- that depth, and each row's scatter about its model, of
    the points it draws most prominently (its bins if binned)."""
    depth, sigmas = 0.0, []
    for members in arrays:
        row_sigmas = []
        for x, y, binned, xm, ym in members:
            depth = max(depth, -float(np.min(ym)))
            bx, by = binned if binned is not None else (x, y)
            row_sigmas.append(_robust_sigma(by - np.interp(bx, xm, ym)))
        sigmas.append(max(row_sigmas))
    spacing = depth + _SPACING_SIGMAS * float(np.median(sigmas))
    return spacing, depth, sigmas


def _row_model_curves(row, members, draw_models):
    """Every model curve of one stack row, ``(x, y)`` with x scaled as in
    ``_transit_row_arrays``: the members' at the reference point, then at
    each draw (``draw_models``, by chart id).  Within one draw, a curve repeated exactly -- grouped TESS files
    share their model -- is kept once, so overlapping files do not darken
    it; a draw repeating another draw is kept, as every draw is in
    ``plotrender``.
    """
    per_draw = {}
    for chart, (_, _, _, xm, ym) in zip(row[2], members):
        per_draw.setdefault(0, []).append((xm, ym))
        scale = _hints(chart)["x_scale"]
        for k, traces in enumerate(draw_models.get(chart.id, ()), start=1):
            per_draw.setdefault(k, []).extend(
                (
                    scale * np.asarray(trace.x, dtype=float),
                    np.asarray(trace.y, dtype=float),
                )
                for trace in traces
            )
    curves = []
    for candidates in per_draw.values():
        seen = set()
        for x, y in candidates:
            key = (x.tobytes(), y.tobytes())
            if key not in seen:
                seen.add(key)
                curves.append((x, y))
    return curves


def _draw_transit_stack(ax, rows, spacing, draw_models=None):
    """Draw the rows of one stack, ``spacing`` apart (None: automatic).

    ``draw_models`` (``_draw_model_traces``) are the other posterior draws'
    model curves: with any, every curve in the stack is drawn at
    ``_DRAWS_ALPHA``, the reference point's included.  The vertical limits
    are symmetric about the stack: the same margin, half a row, above the
    first row's baseline as below the bottom of the last row's transit.
    """
    draw_models = draw_models or {}
    alpha = _DRAWS_ALPHA if draw_models else None
    arrays = [_transit_row_arrays(row) for row in rows]
    auto, depth, sigmas = _auto_spacing(arrays)
    spacing = auto if spacing is None else float(spacing)

    first = rows[0][2][0]
    hints = _hints(first)
    x_range = first.x_range
    for k, (row, members) in enumerate(zip(rows, arrays)):
        offset = 1.0 - k * spacing
        color = row[4]
        for xm, ym in _row_model_curves(row, members, draw_models):
            ax.plot(
                xm, ym + offset, "-", alpha=alpha, zorder=3, **_TRANSIT_MODEL
            )
        for x, y, binned, xm, ym in members:
            ax.plot(
                x,
                y + offset,
                "o",
                ms=_RAW_MARKERSIZE,
                mfc=color,
                mec="k",
                mew=_EDGE_WIDTH,
                ls="none",
                alpha=_RAW_ALPHA_UNDER_BINS if binned else _RAW_ALPHA,
                zorder=1,
                rasterized=True,
            )
            if binned is not None:
                ax.plot(
                    binned[0],
                    binned[1] + offset,
                    "o",
                    ms=_BIN_MARKERSIZE,
                    mfc=color,
                    mec="k",
                    mew=_EDGE_WIDTH,
                    ls="none",
                    alpha=_BIN_ALPHA,
                    zorder=2,
                )

    if x_range is not None:
        ax.set_xlim(
            hints["x_scale"] * x_range[0], hints["x_scale"] * x_range[1]
        )
    lo, hi = ax.get_xlim()
    gap = spacing - depth
    for k, (row, sigma) in enumerate(zip(rows, sigmas)):
        # Just above the row's own points, never past _LABEL_MAX_GAP of the
        # way to the row above, so it reads as this row's.
        lift = min(_LABEL_SIGMAS * sigma, _LABEL_MAX_GAP * gap)
        ax.annotate(
            row[1],
            (lo + 0.02 * (hi - lo), 1.0 - k * spacing + lift),
            xytext=(0.0, _LABEL_PAD_PT),
            textcoords="offset points",
            fontsize=_FONT_RC["legend.fontsize"],
            va="bottom",
            zorder=5,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )
    margin = 0.5 * spacing
    ax.set_ylim(
        1.0 - (len(rows) - 1) * spacing - depth - margin,
        1.0 + margin,
    )
    ax.set_xlabel(hints["xlabel"])
    ax.set_ylabel(hints["ylabel"])
    _ticks(ax)


def _shifted(trace, offset, labels):
    return dataclasses.replace(
        trace,
        x=np.asarray(trace.x, dtype=float) - offset,
        name=labels.get(trace.name, trace.name),
    )


def _offset_label(label, offset):
    if not offset:
        return label
    text = f" - {offset}"
    return label[:-1] + text + "]" if label.endswith("]") else label + text


def _x_offset(chart):
    """The largest of the chart's declared ``x_offsets`` (a time axis's
    zero points, largest first: BTJD, then the older 2450000 convention)
    that every x of the chart is past, subtracted so the tick labels are
    short numbers rather than a matplotlib offset; 0 for none declared or
    none passed (an axis its data already offsets)."""
    offsets = _hints(chart).get("x_offsets") or ()
    x_min = min(float(np.min(t.x)) for t in chart.traces if np.size(t.x) > 0)
    return next((o for o in offsets if x_min > o), 0)


def _prepared(chart, labels, offset=0):
    """The chart with instrument names relabelled, ``offset`` (an
    ``_x_offset``) subtracted from its x axis, and its declared ``ylabel``
    (if any) as its y label."""
    meta = dict(chart.meta or {})
    if "residuals" in meta:
        meta["residuals"] = [
            _shifted(t, offset, labels) for t in meta["residuals"]
        ]
    return dataclasses.replace(
        chart,
        traces=[_shifted(t, offset, labels) for t in chart.traces],
        xlabel=_offset_label(chart.xlabel, offset),
        ylabel=_hints(chart).get("ylabel", chart.ylabel),
        x_range=None
        if chart.x_range is None
        else [float(v) - offset for v in chart.x_range],
        meta=meta,
    )


def _log_errors_to_linear(y, yerr):
    """``(2, N)`` linear error bars for points at ``10**y`` whose errors are
    ``yerr`` in dex (symmetric ``(N,)`` or asymmetric ``(2, N)``)."""
    e = np.asarray(yerr, dtype=float)
    lo, hi = (e[0], e[1]) if e.ndim == 2 else (e, e)
    return np.vstack([10**y - 10 ** (y - lo), 10 ** (y + hi) - 10**y])


def trace_in_flux(trace):
    """A trace of log10 of a quantity (errors in dex) at ``10**y``, its
    error bars converted on each side."""
    y = np.asarray(trace.y, dtype=float)
    yerr = None if trace.yerr is None else _log_errors_to_linear(y, trace.yerr)
    return dataclasses.replace(trace, y=10**y, yerr=yerr)


def sed_in_flux(chart):
    """A spectrum chart as the quantity itself on a log axis, O-C converted
    with it.

    A chart that declares ``y_log10`` carries log10 of what is drawn on a
    linear axis (the SED's log10(lambda*F_lambda): the log is taken
    component-side so its JSON payload stays at normal scale) and its
    ``meta["residuals"]`` in that unit (dex).  Every trace becomes
    ``10**y`` (error bars converted on each side), and each residual the
    difference it encodes, ``obs - model = 10**y_obs - 10**(y_obs - oc)``,
    carrying its point's linear error bars.  A residual is paired with the
    data trace of the same name, point for point; anything else is a chart
    this function does not know how to convert, and raises.  The declared
    ``xlabel``/``ylabel`` replace the chart's, which name the log.
    """
    hints = _hints(chart)
    if not hints.get("y_log10"):
        return chart

    data = {t.name: t for t in chart.traces if t.role == "data"}
    residuals = []
    for oc in (chart.meta or {}).get("residuals", []):
        obs = data.get(oc.name)
        if obs is None or not np.array_equal(
            np.asarray(obs.x, dtype=float), np.asarray(oc.x, dtype=float)
        ):
            raise ValueError(
                f"summary plot: residual trace {oc.name!r} has no data "
                f"trace with the same points in chart {chart.id!r}."
            )
        y_obs = np.asarray(obs.y, dtype=float)
        residuals.append(
            dataclasses.replace(
                oc,
                y=10**y_obs - 10 ** (y_obs - np.asarray(oc.y, dtype=float)),
                yerr=_log_errors_to_linear(y_obs, obs.yerr),
            )
        )
    lo, hi = chart.y_range
    return dataclasses.replace(
        chart,
        traces=[trace_in_flux(t) for t in chart.traces],
        y_range=[10.0**lo, 10.0**hi],
        y_log=True,
        xlabel=hints.get("xlabel", chart.xlabel),
        ylabel=hints.get("ylabel", chart.ylabel),
        meta={**(chart.meta or {}), "residuals": residuals},
    )


def _ticks(ax):
    ax.tick_params(
        which="major",
        top=True,
        right=True,
        labelsize=_TICK_LABELSIZE,
        **_TICK_MAJOR,
    )
    ax.tick_params(which="minor", top=True, right=True, **_TICK_MINOR)


def _geometry(ax, chart):
    """The chart's own axis geometry and labels (log axes, ranges, inverted
    axes: the first-class Chart fields both renderers honor)."""
    if chart.x_log:
        ax.set_xscale("log")
    if chart.y_log:
        ax.set_yscale("log")
    if chart.x_range is not None:
        ax.set_xlim(*[float(v) for v in chart.x_range])
    if chart.y_range is not None:
        ax.set_ylim(*[float(v) for v in chart.y_range])
    if chart.x_inverted:
        ax.invert_xaxis()
    if chart.y_inverted:
        ax.invert_yaxis()
    ax.set_xlabel(chart.xlabel)
    ax.set_ylabel(chart.ylabel)


def _axes(fig, cell, with_oc):
    """``(ax, ax_oc)`` in one grid cell; ``ax_oc`` is None without an O-C."""
    if not with_oc:
        return fig.add_subplot(cell), None
    sub = cell.subgridspec(2, 1, height_ratios=_OC_HEIGHT_RATIOS, hspace=0.0)
    ax = fig.add_subplot(sub[0])
    return ax, fig.add_subplot(sub[1], sharex=ax)


def _finish(ax, ax_oc):
    """Ticks on both axes; the O-C axis takes the x label and a zero line,
    and is in the units of the panel above it, so it names none."""
    from matplotlib.transforms import ScaledTranslation

    _ticks(ax)
    if ax_oc is None:
        return
    ax_oc.axhline(0.0, zorder=0, **_ZERO_LINE)
    ax_oc.set_xlabel(ax.get_xlabel())
    ax_oc.set_ylabel(OC_YLABEL)
    # A flux O-C (~1e-12) is written with a x10^n offset, not "1e-12", in
    # the O-C axes' top right corner: clear of the left-hand tick labels,
    # and of the panel above, which it would sit on where matplotlib puts
    # it (standing on the axes, re-anchored there at every draw -- hence a
    # shift of its transform, not of its position).
    ax_oc.yaxis.get_major_formatter().set_useMathText(True)
    ax_oc.yaxis.set_offset_position("right")
    offset = ax_oc.yaxis.get_offset_text()
    offset.set_verticalalignment("top")
    offset.set_x(_OC_OFFSET_X)
    offset.set_transform(
        offset.get_transform()
        + ScaledTranslation(
            0.0, -_OC_OFFSET_DROP_PT / 72.0, ax_oc.figure.dpi_scale_trans
        )
    )
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)
    _ticks(ax_oc)


def _errorbar(ax, trace, color, marker, ms, label=None, zorder=10):
    """One trace's points: filled ``color``, black edge, colored error bars."""
    ax.errorbar(
        np.asarray(trace.x, dtype=float),
        np.asarray(trace.y, dtype=float),
        yerr=None if trace.yerr is None else np.asarray(trace.yerr, float),
        xerr=None if trace.xerr is None else np.asarray(trace.xerr, float),
        fmt=marker,
        ms=ms,
        mfc=color,
        mec="k",
        mew=_EDGE_WIDTH,
        ecolor=color,
        capsize=_CAPSIZE,
        ls="none",
        zorder=zorder,
        label=label,
    )


def _legend(ax, **kwargs):
    """One entry per label, first occurrence winning; a swatch fainter than
    ``_LEGEND_ALPHA`` (a model curve among posterior draws) is raised to it,
    leaving the plotted curves untouched, as ``plotrender._legend`` does."""
    unique = {}
    for handle, label in zip(*ax.get_legend_handles_labels()):
        unique.setdefault(label, handle)
    if not unique:
        return
    legend = ax.legend(list(unique.values()), list(unique.keys()), **kwargs)
    for proxy, handle in zip(legend.legend_handles, unique.values()):
        # An errorbar handle is a Container, with no alpha to read.
        alpha = getattr(handle, "get_alpha", lambda: None)()
        if alpha is not None and alpha < _LEGEND_ALPHA:
            proxy.set_alpha(_LEGEND_ALPHA)


def time_segments(times, max_gap):
    """``[(first, last), ...]``: the observation ``times`` split wherever
    consecutive ones are more than ``max_gap`` days apart -- the pieces of
    a broken time axis.  ``max_gap=None`` never splits."""
    t = np.unique(np.asarray(times, dtype=float))
    t = t[np.isfinite(t)]
    if t.size == 0:
        return []
    if max_gap is None:
        return [(float(t[0]), float(t[-1]))]
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    return [(float(p[0]), float(p[-1])) for p in np.split(t, cuts)]


def _segment_limits(segments):
    """Each piece's x limits, all padded alike, and its width ratio: its
    padded span, floored at ``_BREAK_MIN_WIDTH`` of the whole so a lone
    observation still has an axis to sit on."""
    covered = sum(hi - lo for lo, hi in segments)
    pad = max(_BREAK_PAD_FRAC * covered, _BREAK_MIN_PAD_DAYS)
    limits = [(lo - pad, hi + pad) for lo, hi in segments]
    spans = np.array([hi - lo for lo, hi in limits])
    return limits, np.maximum(spans, _BREAK_MIN_WIDTH * spans.sum())


def _broken_axes(fig, cell, with_oc, widths):
    """``[(ax, ax_oc), ...]``: one column per piece of a broken x axis in
    one grid cell, ``widths`` apart in ratio; every panel shares the first
    one's y axis, and each O-C its panel's x axis."""
    sub = cell.subgridspec(
        2 if with_oc else 1,
        len(widths),
        width_ratios=list(widths),
        height_ratios=_OC_HEIGHT_RATIOS if with_oc else None,
        hspace=0.0,
        wspace=_BREAK_WSPACE,
    )
    columns = []
    for i in range(len(widths)):
        first = columns[0] if columns else (None, None)
        ax = fig.add_subplot(sub[0, i], sharey=first[0])
        ax_oc = (
            fig.add_subplot(sub[1, i], sharex=ax, sharey=first[1])
            if with_oc
            else None
        )
        columns.append((ax, ax_oc))
    return columns


def piece_ticks(lo, hi, width_in, trim_left, trim_right):
    """Round x ticks for one piece ``(lo, hi)`` of a broken axis,
    ``width_in`` inches wide: as many as fit ``_BREAK_TICK_IN`` apart, none
    within ``_BREAK_EDGE_IN`` of an edge marked to be trimmed (one facing
    another piece), and never none -- a piece too narrow for those gets the
    roundest value near its middle."""
    from matplotlib.ticker import MaxNLocator

    per_in = (hi - lo) / width_in
    keep_lo = lo + (_BREAK_EDGE_IN * per_in if trim_left else 0.0)
    keep_hi = hi - (_BREAK_EDGE_IN * per_in if trim_right else 0.0)
    if keep_lo >= keep_hi:
        keep_lo, keep_hi = lo, hi
    nbins = max(1, int(width_in / _BREAK_TICK_IN))
    ticks = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(
        lo, hi
    )
    ticks = [float(t) for t in ticks if keep_lo <= t <= keep_hi]
    if ticks:
        return ticks
    center = 0.5 * (keep_lo + keep_hi)
    exponent = int(np.floor(np.log10(hi - lo)))
    for e in range(exponent, exponent - 6, -1):
        for mantissa in (5.0, 2.0, 1.0):
            step = mantissa * 10.0**e
            tick = round(center / step) * step
            if keep_lo <= tick <= keep_hi:
                return [float(tick)]
    return [center]


def _join_broken(columns, limits):
    """Make ``columns`` (``_broken_axes``, each already finished) read as
    one broken axis: each piece at its ``limits`` with ``piece_ticks``, the
    facing spines and their ticks removed and marked with diagonals, the y
    labels on the first piece only, and one x label, centered under the
    whole."""
    from matplotlib.ticker import FixedLocator

    last = len(columns) - 1
    for i, ((ax, ax_oc), lim) in enumerate(zip(columns, limits)):
        ax.set_xlim(*lim)
        width_in = ax.get_position().width * ax.figure.get_figwidth()
        ax.xaxis.set_major_locator(
            FixedLocator(piece_ticks(*lim, width_in, i > 0, i < last))
        )
        for a in (ax, ax_oc):
            if a is None:
                continue
            if i > 0:
                a.spines["left"].set_visible(False)
                a.tick_params(which="both", left=False, labelleft=False)
                a.set_ylabel("")
                a.plot([0, 0], [0, 1], transform=a.transAxes, **_BREAK_MARK)
            if i < last:
                a.spines["right"].set_visible(False)
                a.tick_params(which="both", right=False)
                a.plot([1, 1], [0, 1], transform=a.transAxes, **_BREAK_MARK)
    bottom = [ax_oc if ax_oc is not None else ax for ax, ax_oc in columns]
    for a in bottom[1:]:
        a.set_xlabel("")
    left, right = bottom[0].get_position(), bottom[-1].get_position()
    bottom[0].xaxis.label.set_x(
        ((left.x0 + right.x1) / 2.0 - left.x0) / left.width
    )


def _draw_rv(
    fig, cell, chart, colors, phased, draw_models=(), break_days=None
):
    """An RV panel and its O-C; returns ``(axes, oc_axes)``, lists of the
    panel's axes and of its O-C's (empty without one).

    ``colors`` maps an instrument's index (its traces' ``series_index``) to
    its color.  A model curve that belongs to one instrument (its own RM or
    GP) takes that instrument's color; the shared model is drawn in the
    folded or the against-time model style.  ``draw_models`` are the other
    posterior draws' model traces for this chart (``_draw_model_traces``):
    with them, every curve, the reference point's included, is drawn at
    ``_DRAWS_ALPHA``.  Against time, the axis is broken wherever the
    observations are more than ``break_days`` apart (``time_segments``):
    one column of axes per piece, everything drawn in each and clipped to
    it, so a season-long gap takes no room on the page.
    """
    residuals = (chart.meta or {}).get("residuals")
    with_oc = residuals is not None
    data = _role(chart, "data")
    segments = (
        []
        if phased
        else time_segments(
            np.concatenate([np.ravel(t.x) for t in data] or [[]]),
            break_days,
        )
    )
    if len(segments) > 1:
        limits, widths = _segment_limits(segments)
        columns = _broken_axes(fig, cell, with_oc, widths)
    else:
        columns = [_axes(fig, cell, with_oc)]
    style = _RV_MODEL if phased else _RV_TIME_MODEL
    alpha = _DRAWS_ALPHA if draw_models else style["alpha"]
    for ax, ax_oc in columns:
        for traces in [_role(chart, "model"), *draw_models]:
            for trace in traces:
                owner = (trace.style or {}).get("series_index")
                ax.plot(
                    np.asarray(trace.x, dtype=float),
                    np.asarray(trace.y, dtype=float),
                    color=style["color"] if owner is None else colors[owner],
                    lw=style["lw"],
                    alpha=alpha,
                    zorder=2 if phased else 0,
                    label=_MODEL_LABEL if owner is None else trace.name,
                )
        for trace in data:
            style_i = trace.style or {}
            _errorbar(
                ax,
                trace,
                colors[style_i["series_index"]],
                style_i.get("marker") or "o",
                _RV_MARKERSIZE,
                label=trace.name,
            )
        for trace in residuals or []:
            style_i = trace.style or {}
            _errorbar(
                ax_oc,
                trace,
                colors[style_i["series_index"]],
                style_i.get("marker") or "o",
                _RV_MARKERSIZE,
            )
        _geometry(ax, chart)
        if phased:
            ax.set_xlim(0.0, 1.0)
            _legend(ax, loc="upper right")
        _finish(ax, ax_oc)
    if len(columns) > 1:
        _join_broken(columns, limits)
    return (
        [ax for ax, _ in columns],
        [ax_oc for _, ax_oc in columns if ax_oc is not None],
    )


def _smooth(y, window):
    """A ``window``-point boxcar (moving) mean of ``y``, NaN-aware: a NaN is
    left out of its neighbors' means and stays NaN.  ``window < 3``, or a
    curve shorter than the window, is returned unchanged."""
    y = np.asarray(y, dtype=float)
    if window < 3 or y.size < window:
        return y
    kernel = np.ones(int(window))
    good = np.isfinite(y)
    total = np.convolve(np.where(good, y, 0.0), kernel, mode="same")
    count = np.convolve(good.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    return np.where(good, out, np.nan)


def _draw_sed(fig, cell, chart, draw_models=()):
    """The SED panel and its O-C, ``chart`` already in flux (``sed_in_flux``).

    Each star (or measured combination) takes one color for its photometry
    and its spectrum; the spectra are smoothed (``_SED_SMOOTH``) and drawn
    beneath, and the model's photometry -- the observed flux less its O-C --
    is a black point on each.  ``draw_models`` are the other posterior
    draws' spectra, also in flux: with them, every spectrum, the reference
    point's included, is drawn at ``_SED_DRAWS_ALPHA``.  The flux axis spans
    the photometry; the wavelength axis spans the bandpasses, widened to
    wherever a spectrum lies within the flux axis.
    """
    residuals = (chart.meta or {}).get("residuals") or []
    identity = _hints(chart)["identity"]
    ax, ax_oc = _axes(fig, cell, bool(residuals))
    data = _role(chart, "data")
    colors = {}
    for trace in data:
        colors[identity[trace.name]] = PALETTE[
            -(len(colors) + 1) % len(PALETTE)
        ]
    alpha = _SED_DRAWS_ALPHA if draw_models else None
    spectra = []
    for traces in [_role(chart, "model"), *draw_models]:
        for trace in traces:
            who = identity[trace.name]
            if who not in colors:
                colors[who] = PALETTE[-(len(colors) + 1) % len(PALETTE)]
            spectra.append(
                (
                    np.asarray(trace.x, dtype=float),
                    _smooth(trace.y, _SED_SMOOTH),
                )
            )
            ax.plot(
                *spectra[-1],
                color=colors[who],
                lw=_SED_SPECTRUM_LW,
                alpha=alpha,
                zorder=0,
            )
    single = len(data) == 1
    by_name = {t.name: t for t in data}
    for trace in data:
        _errorbar(
            ax,
            trace,
            colors[identity[trace.name]],
            "o",
            _SED_MARKERSIZE,
            label="Observations" if single else trace.name,
            zorder=3,
        )
    for oc in residuals:
        obs = by_name[oc.name]
        ax.plot(
            np.asarray(obs.x, dtype=float),
            np.asarray(obs.y, dtype=float) - np.asarray(oc.y, dtype=float),
            "o",
            color="k",
            ms=_RV_MARKERSIZE,
            ls="none",
            zorder=4,
            label=_MODEL_LABEL,
        )
        _errorbar(ax_oc, oc, colors[identity[oc.name]], ".", _SED_MARKERSIZE)

    _geometry(ax, chart)
    y = np.concatenate([np.asarray(t.y, float) for t in data])
    yerr = np.hstack([np.asarray(t.yerr, float).reshape(2, -1) for t in data])
    span = np.concatenate([y - yerr[0], y + yerr[1]])
    span = span[np.isfinite(span) & (span > 0)]
    y_lo, y_hi = np.min(span) / _SED_Y_PAD, np.max(span) * _SED_Y_PAD
    ax.set_ylim(y_lo, y_hi)
    # The wavelength axis: the bandpasses, widened to every stretch of a
    # model spectrum that lies within the flux axis, so no part of the
    # atmosphere the y range shows is cut off at the sides.  The flux axis
    # is the photometry's and does not move.
    x = np.concatenate([np.asarray(t.x, float) for t in data])
    xerr = np.hstack([np.asarray(t.xerr, float).reshape(2, -1) for t in data])
    x_lo = np.min(x - xerr[0]) / _SED_X_PAD
    x_hi = np.max(x + xerr[1]) * _SED_X_PAD
    for wave, flux in spectra:
        shown = np.isfinite(flux) & (flux >= y_lo) & (wave > 0)
        if shown.any():
            x_lo = min(x_lo, float(wave[shown].min()))
            x_hi = max(x_hi, float(wave[shown].max()))
    ax.set_xlim(x_lo, x_hi)
    _legend(ax, loc="upper right")
    _finish(ax, ax_oc)


def _degeneracy(x, y):
    """Why the draws ``(x, y)`` have no 2-D density to contour, or None
    when they have one.  A pinned quantity, or two that move together
    exactly (one derived from the other alone), has a singular covariance:
    ``gaussian_kde`` cannot be built on it."""
    if x.size < 3:
        return f"{x.size} draw(s)"
    if not (np.std(x) > 0 and np.std(y) > 0):
        return "a constant coordinate"
    r = np.corrcoef(x, y)[0, 1]
    if not abs(r) < 1.0 - 1e-9:
        return f"perfectly correlated coordinates (r = {r:.12f})"
    return None


def _contours(ax, contours):
    """Draw the 1- and 2-sigma contours of each ``(label, x, y)`` sample
    set, in ``_CONTOUR_COLORS`` order, returning the x and y extent of the
    drawn lines, or None when no set has a density to contour.  A Gaussian
    KDE (Scott's bandwidth) of the draws, contoured at the densities
    enclosing ``_CONTOUR_PROBS`` of it: the shared ``contour_plot.Contour``.
    A sample set with no 2-D density (``_degeneracy``) gets no contour and
    no legend entry, and says so at INFO; the reported point and its error
    bars still show where it is."""
    from matplotlib.lines import Line2D

    from .contour_plot import Contour

    if len(contours) > len(_CONTOUR_COLORS):
        raise ValueError(
            f"summary plot: {len(contours)} contour sets, more than the "
            f"{len(_CONTOUR_COLORS)} colors a track panel draws."
        )
    extent = []
    for (label, x, y), color in zip(contours, _CONTOUR_COLORS):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        why = _degeneracy(x, y)
        if why is not None:
            logger.info(
                "summary plot: no %s contour -- its posterior has no 2-D "
                "density (%s).",
                label,
                why,
            )
            continue
        contour = Contour(
            x,
            y,
            x_err=np.std(x),
            y_err=np.std(y),
            bw_method="scott",
            probs=_CONTOUR_PROBS,
        )
        drawn = ax.contour(
            contour.X,
            contour.Y,
            contour.Z,
            levels=contour.levels,
            colors=color,
            linewidths=_CONTOUR_LW,
            zorder=6,
        )
        # What was drawn, the 2-sigma line included: the KDE's region
        # reaches past the draws' own central 95%, so the window has to
        # follow the lines rather than the draws.
        polygons = [
            poly
            for path in drawn.get_paths()
            for poly in path.to_polygons(closed_only=False)
        ]
        if not polygons:
            # Contour's levels are densities of its own KDE grid, so a
            # non-degenerate sample set always yields a line.
            raise RuntimeError(
                f"summary plot: the {label} KDE of {x.size} draws drew no "
                f"contour line at levels {list(contour.levels)}."
            )
        ax.add_line(
            Line2D(
                [],
                [],
                color=color,
                lw=_CONTOUR_LW,
                label=rf"{label} ($1\sigma$, $2\sigma$)",
            )
        )
        vertices = np.concatenate(polygons)
        extent.append(
            (
                vertices[:, 0].min(),
                vertices[:, 0].max(),
                vertices[:, 1].min(),
                vertices[:, 1].max(),
            )
        )
    if not extent:
        return None
    lo_x, hi_x, lo_y, hi_y = zip(*extent)
    return min(lo_x), max(hi_x), min(lo_y), max(hi_y)


def _draw_track(fig, cell, chart, overlay=None):
    """A track panel (the Kiel diagram).

    The chart's model track heavy blue with its declared ``marks`` (each
    ``{"x", "y", "text"}``: reference ages along the main sequence, say);
    and from ``overlay`` (the owning component's ``summary_posterior`` for
    this chart): the reported ``marker`` a red cross with its asymmetric
    errors, and the 1- and 2-sigma ``contours`` of each sample set, the
    only legend entries.  The window frames the declared ``window``, the
    marker and the contours, padded by ``_TRACK_PAD_FRAC``; an axis the
    chart declares inverted (the Kiel convention: hot on the left, low
    gravity at the top) stays inverted.
    """
    hints = _hints(chart)
    overlay = overlay or {}
    ax = fig.add_subplot(cell)
    for trace in _role(chart, "model"):
        ax.plot(
            np.asarray(trace.x, dtype=float),
            np.asarray(trace.y, dtype=float),
            zorder=7,
            **_TRACK_LINE,
        )
    _geometry(ax, chart)

    boxes = []
    if hints.get("window") is not None:
        boxes.append(tuple(hints["window"]))
    if "marker" in overlay:
        (x, x_lo, x_hi), (y, y_lo, y_hi) = overlay["marker"]
        ax.errorbar(
            [x],
            [y],
            xerr=[[x_lo], [x_hi]],
            yerr=[[y_lo], [y_hi]],
            fmt="none",
            zorder=9,
            **_TRACK_MARKER,
        )
        boxes.append((x - x_lo, x + x_hi, y - y_lo, y + y_hi))
    if overlay.get("contours"):
        drawn = _contours(ax, overlay["contours"])
        if drawn is not None:
            boxes.append(drawn)
    for mark in hints.get("marks") or ():
        ax.plot(mark["x"], mark["y"], ls="none", zorder=8, **_TRACK_MARK)
        ax.annotate(
            mark["text"],
            (mark["x"], mark["y"]),
            xytext=(7, -3),
            textcoords="offset points",
            color=_TRACK_MARK["color"],
            fontweight="bold",
            fontsize=_TICK_LABELSIZE - 2,
            zorder=10,
        )
    if boxes:
        lo_x, hi_x, lo_y, hi_y = (
            f(v) for f, v in zip((min, max, min, max), zip(*boxes))
        )
        pad_x = _TRACK_PAD_FRAC * (hi_x - lo_x)
        pad_y = _TRACK_PAD_FRAC * (hi_y - lo_y)
        x_lim = (lo_x - pad_x, hi_x + pad_x)
        y_lim = (lo_y - pad_y, hi_y + pad_y)
        ax.set_xlim(*(x_lim[::-1] if chart.x_inverted else x_lim))
        ax.set_ylim(*(y_lim[::-1] if chart.y_inverted else y_lim))
    _legend(ax, loc="best", fontsize="small")
    _finish(ax, None)


def _in_linear(chart, trace):
    """A draw's trace of a spectrum chart, in the units ``sed_in_flux``
    draws the chart in."""
    return trace_in_flux(trace) if _hints(chart).get("y_log10") else trace


def _posterior_overlays(system, panels):
    """``{chart id: overlay}`` from the ``summary_posterior`` of every
    component owning a track panel (``Component.summary_posterior``)."""
    keys = {
        chart.component["yaml_key"]
        for panel in panels
        if panel.kind == "track"
        for chart in panel.charts
    }
    overlays = {}
    for key in sorted(keys):
        overlays.update(
            system.active_components[key].summary_posterior(system)
        )
    return overlays


def _share_ylims(axes):
    """One y range for every axis in ``axes``: the union of their own."""
    if not axes:
        return
    lows, highs = zip(*(ax.get_ylim() for ax in axes))
    for ax in axes:
        ax.set_ylim(min(lows), max(highs))


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def summary_figure(
    system,
    point,
    *,
    title=None,
    header_lines=(),
    labels=None,
    transit_groups=None,
    transit_bin=None,
    transit_spacing=None,
    figsize=None,
    rv_break_days=RV_BREAK_DAYS,
    draws=(),
):
    """Draw the summary figure for ``system`` at ``point``; return the Figure.

    ``system`` must be built and ``point`` a plotting point (internal units,
    e.g. from ``median_draw_point``): the reference, whose data, cleaning
    and O-C every panel draws.  The caller owns the figure: save it, then
    ``plt.close`` it.

    Parameters
    ----------
    title : str, optional
        Bold title across the top.
    header_lines : sequence of str
        Lines under the title -- ``summary_header_lines(system)`` once a
        posterior is distributed.  Mathtext is rendered.
    labels : dict, optional
        Display names that override the fit's own, keyed by instrument name
        (transit or RV).  Not needed: by default each instrument is shown by
        its ``label:`` in the config, else its ``name:``.  Unknown names
        raise.
    transit_groups : dict, optional
        ``{row label: [transit names]}``: files drawn on ONE row of the
        stack, each with its own model curve (identical curves overlap).
        Default: the rows the charts declare (``declared_row_groups``) --
        for transits, the TESS files one row per exposure time, every other
        file on its own row (``Transit.summary_rows``).  Passing a dict
        replaces that rule (``{}`` puts every file on its own row).  A name
        in two groups, or not a transit of this fit, raises.
    transit_bin : float or dict, optional
        Bin the phased transit points to this many minutes, drawn over the
        unbinned points.  A number bins every row; a dict bins only the rows
        it names, keyed by transit name or, for a grouped row, its label
        (e.g. ``{"TESS 120 s": 10}``).
    transit_spacing : float, optional
        Vertical offset between stacked transits, in normalized flux.
        Default: the deepest transit plus four times the typical scatter.
    figsize : (float, float), optional
        Inches.  Default scales with the number of panels.
    rv_break_days : float or None
        Break the RV time axis (and its O-C) wherever consecutive
        observations are more than this many days apart; None never
        breaks it.  Default ``RV_BREAK_DAYS`` (50).
    draws : sequence of dict, optional
        Other posterior draws, as plotting points (``run.get_draws``).  The
        transit, folded RV and SED model curves are drawn at ``point`` and
        at each of these, all at one low alpha, as EXOZIPPy's own posterior
        plots draw them; the data, its cleaning, the O-C, the RV curve
        against time and the Kiel track stay ``point``'s.  Without any, ``point``'s curves are drawn alone at
        full weight.
    """
    import matplotlib.pyplot as plt

    labels = dict(labels or {})

    charts = _collect_charts(system, point)
    panels = _panels(charts)
    if not panels:
        raise NoSummaryPanels(
            "No component of this fit declares a summary chart "
            '(meta["summary"]): no transit, RV, SED or evolutionary model, '
            "or any other component that draws one."
        )

    # Each instrument's display name: an override, else what its chart
    # declares (an instrument's `label:`, else its `name:`).
    stacked = [c for p in panels if p.kind == "stacked_fold" for c in p.charts]
    stacked_names = list(
        dict.fromkeys(_hints(c)["instrument"] for c in stacked)
    )
    series = [
        c
        for p in panels
        if p.kind in ("time_series", "phase_fold")
        for c in p.charts
    ]
    series_labels = {}
    for chart in series:
        series_labels.update(_hints(chart)["labels"])
    _check_names("labels", labels, stacked_names + list(series_labels))
    display = {**series_labels, **labels}

    if transit_groups is None:
        groups = declared_row_groups(stacked)
    else:
        groups = {str(g): list(m) for g, m in transit_groups.items()}
    grouped = [m for members in groups.values() for m in members]
    _check_names("transit_groups", grouped, stacked_names)
    twice = sorted({m for m in grouped if grouped.count(m) > 1})
    if twice:
        raise ValueError(
            f"transit_groups puts {twice} in more than one group; a file is "
            "drawn on one row."
        )
    if isinstance(transit_bin, dict):
        ungrouped = [n for n in stacked_names if n not in grouped]
        _check_names("transit_bin", transit_bin, ungrouped + list(groups))
        bad = {k: v for k, v in transit_bin.items() if not v > 0}
    elif transit_bin is not None:
        bad = {} if transit_bin > 0 else {"transit_bin": transit_bin}
    else:
        bad = {}
    if bad:
        raise ValueError(f"Bin widths must be positive minutes, got {bad}.")
    if transit_spacing is not None and not transit_spacing > 0:
        raise ValueError(
            f"transit_spacing must be positive, got {transit_spacing}."
        )
    if rv_break_days is not None and not rv_break_days > 0:
        raise ValueError(
            f"rv_break_days must be positive days or None, got "
            f"{rv_break_days}."
        )

    placed, ncols, nrows = _place(panels)
    header = list(header_lines)
    header_in = (
        (_TITLE_HEIGHT_IN if title else 0.0)
        + _LINE_HEIGHT_IN * len(header)
        + (0.25 if (title or header) else 0.0)
    )
    if figsize is None:
        figsize = (
            _COLUMN_WIDTH_IN * ncols,
            _ROW_HEIGHT_IN * nrows + header_in,
        )
    width, height = figsize

    with plt.rc_context(_FONT_RC):
        fig = plt.figure(figsize=(width, height))
        try:
            y = 1.0 - 0.1 / height
            if title:
                fig.text(
                    0.5,
                    y,
                    title,
                    ha="center",
                    va="top",
                    fontsize=22,
                    fontweight="bold",
                )
                y -= _TITLE_HEIGHT_IN / height
            for line in header:
                fig.text(0.5, y, line, ha="center", va="top", fontsize=16)
                y -= _LINE_HEIGHT_IN / height

            grid = fig.add_gridspec(
                nrows,
                ncols,
                left=0.09 if ncols == 2 else 0.14,
                right=0.97,
                bottom=0.05,
                top=1.0 - header_in / height,
                hspace=0.2,
                wspace=0.24,
            )
            # Each series' color, by its series_index: the declared user
            # `plot: color:`, else the palette from the back.
            declared = {}
            for chart in series:
                declared.update(_hints(chart)["colors"])
            series_colors = {
                int(i): c or PALETTE[-(int(i) + 1) % len(PALETTE)]
                for i, c in declared.items()
            }
            draw_models = _draw_model_traces(system, panels, draws)
            overlays = _posterior_overlays(system, panels)
            series_axes, series_oc_axes = [], []
            for panel, col, row in placed:
                cell = grid[row : row + panel.rows, col]
                if panel.kind == "stacked_fold":
                    ax = fig.add_subplot(cell)
                    rows = _transit_rows(
                        panel.charts, labels, groups, transit_bin
                    )
                    _draw_transit_stack(ax, rows, transit_spacing, draw_models)
                elif panel.kind in ("time_series", "phase_fold"):
                    phased = panel.kind == "phase_fold"
                    chart = panel.charts[0]
                    offset = 0 if phased else _x_offset(chart)
                    extra = [
                        [_shifted(t, offset, display) for t in traces]
                        for traces in draw_models.get(chart.id, ())
                    ]
                    axes, oc_axes = _draw_rv(
                        fig,
                        cell,
                        _prepared(chart, display, offset),
                        series_colors,
                        phased,
                        extra,
                        rv_break_days,
                    )
                    series_axes += axes
                    series_oc_axes += oc_axes
                elif panel.kind == "spectrum":
                    chart = panel.charts[0]
                    _draw_sed(
                        fig,
                        cell,
                        sed_in_flux(chart),
                        [
                            [_in_linear(chart, t) for t in traces]
                            for traces in draw_models.get(chart.id, ())
                        ],
                    )
                else:
                    chart = panel.charts[0]
                    _draw_track(fig, cell, chart, overlays.get(chart.id))
            # Every time-series and phase-fold panel on one scale (they
            # plot one quantity: the RVs), and every such O-C on another.
            _share_ylims(series_axes)
            _share_ylims(series_oc_axes)
        except BaseException:
            plt.close(fig)
            raise
    return fig


def write_summary_plot(
    system, posterior, path, *, title=None, draws=None, **kwargs
):
    """Draw the summary figure at the median draw of ``posterior`` and save
    it to ``path`` (the format follows its extension); returns ``path``, or
    None, writing nothing, for a fit with no chart to draw.

    ``posterior`` is the REPORTED posterior, already distributed onto
    ``system`` -- what ``reported_posterior`` returns, and what run.py's
    wrap-up holds after its result tables, which is why both the live fit
    and ``create_summary_plot`` come through here.  ``draws`` are the
    posterior draws whose model curves are overlaid (``summary_figure``):
    run.py passes the ones its component PDFs drew; None takes
    ``_N_DRAWS`` of ``posterior`` with ``run.get_draws``.  The other
    keywords are ``summary_figure``'s.

    A fit with nothing to draw (no component declares a summary chart) is
    skipped with an INFO line rather than raising: run.py's wrap-up
    calls this for every fit and catches nothing, so a microlensing fit
    must not fail there.  Anything else that goes wrong raises.
    """
    import matplotlib.pyplot as plt

    point, (chain, draw), distance = median_draw_point(system, posterior)
    if not any(_panel_kind(c) for c in _collect_charts(system, point)):
        logger.info(
            "summary plot: none written -- no component of this fit "
            "declares a summary chart."
        )
        return None
    logger.info(
        f"summary plot: median draw chain {chain}, draw {draw} of the "
        f"post-burn-in trace ({distance:.1f} posterior widths from the "
        "median, summed in quadrature over the sampled coordinates)."
    )
    if draws is None:
        from ..run import get_draws

        draws = get_draws(
            posterior,
            n_draws=_N_DRAWS,
            param_lookup=system.get_parameter_lookup(),
            exclude=set(system.report_only_labels()),
        )
    fig = summary_figure(
        system,
        point,
        title=title,
        header_lines=summary_header_lines(system),
        draws=draws,
        **kwargs,
    )
    try:
        fig.savefig(
            path, bbox_inches="tight", facecolor="white", dpi=_SAVE_DPI
        )
    finally:
        plt.close(fig)
    logger.info(f"summary plot: wrote {path}")
    return path


def create_summary_plot(
    config_file,
    output=None,
    *,
    title=None,
    labels=None,
    transit_groups=None,
    transit_bin=None,
    transit_spacing=None,
    figsize=None,
    rv_break_days=RV_BREAK_DAYS,
):
    """Draw a finished fit's summary figure from its config and saved trace.

    ``config_file`` is the system YAML the fit ran with (and, through its
    ``parameter_file:``, its params): it rebuilds the System -- the data and
    the compiled model the panels are drawn from -- and its ``prefix:``
    locates ``<prefix>_trace.nc``.  Relative paths in it resolve against the
    config's own directory, wherever this is called from.  The trace must
    belong to the model the config builds (``check_trace_freshness``, the
    check every trace reload makes; a mismatch raises ``StaleTraceError``).

    Writes ``output`` (default ``<prefix>_mcmc_summary.pdf``; the format
    follows its extension) and returns its path.  The remaining keywords are
    ``summary_figure``'s; ``title`` defaults to the config's ``run: name:``.

    Everything else comes from the fit: each instrument is shown by its
    ``label:`` in the config (else its name), and the TESS files are grouped
    by cadence.  Example::

        from exozippy.outputs.summary_plot import create_summary_plot

        create_summary_plot("toi5432.yaml", transit_bin={"TESS 120 s": 10})
    """
    import arviz as az

    from ..system import System
    from ..trace_meta import check_trace_freshness
    from ..yamlio import load_system_config

    config_path = Path(config_file).expanduser().resolve()
    out = None if output is None else Path(output).expanduser().resolve()

    # The fit resolved its data files and prefix against its working
    # directory, which is where its config lives in every documented
    # workflow; resolving them there again makes the call site irrelevant.
    with contextlib.chdir(config_path.parent):
        config = load_system_config(config_path.name)
        prefix = Path(config.get("prefix", DEFAULT_PREFIX))
        trace_path = Path(f"{prefix}_trace.nc")
        if not trace_path.exists():
            raise FileNotFoundError(
                f"No saved trace at {trace_path.resolve()}.  The summary "
                f"plot is drawn from a finished fit; run `exozippy "
                f"{config_path.name}` first."
            )

        system = System(config)
        system.prepare()
        model = system.build_model()

        idata = az.from_netcdf(str(trace_path))
        check_trace_freshness(idata, system, trace_path)
        posterior = reported_posterior(system, model, idata)
        if title is None:
            title = (config.get("run") or {}).get("name")
        if out is None:
            out = Path(f"{prefix}_mcmc_summary.pdf").resolve()
        written = write_summary_plot(
            system,
            posterior,
            out,
            title=title,
            labels=labels,
            transit_groups=transit_groups,
            transit_bin=transit_bin,
            transit_spacing=transit_spacing,
            figsize=figsize,
            rv_break_days=rv_break_days,
        )
    if written is None:
        raise NoSummaryPanels(
            f"{config_path.name}: no component of this fit declares a "
            'summary chart (meta["summary"]; the transit, RV, SED and '
            "evolutionary-model components do), so there is no summary "
            "figure to draw."
        )
    return out
