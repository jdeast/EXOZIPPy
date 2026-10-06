"""The system summary figure: a finished fit's model-bearing panels on one page.

``create_summary_plot(config_file)`` draws ``<prefix>_mcmc_summary.pdf`` from
what a fit leaves behind -- the config it ran with and ``<prefix>_trace.nc`` --
in the layout of the one-page system figure of an exoplanet discovery paper:
each planet's phase-folded transits stacked with offsets, the RVs against time
and folded on each orbit (each with its O-C), the SED (with its O-C) and each
star's Kiel diagram, under a header quoting every planet's P, R_P, M_P and e.
A panel is drawn only when the fit has the component behind it, so an RV-only
or a star-only fit gets a shorter page rather than empty axes.

THIS MODULE ADDS NO PHYSICS.  It is a third renderer of the components' Charts
(``plotrender.py`` and the GUI's ``plotly-adapter.ts`` are the other two):
every curve and point is what ``Component.plot_data(system, point)`` returned,
and every O-C is the component's own ``meta["residuals"]`` -- data minus the
likelihood's model at the observations, which only the component can
evaluate.  What it owns is the layout: which charts make a panel, stacking
the transits with offsets (in hours, optionally binned), a BJD offset on the
RV time axis, and the header.  A new panel kind is a new component chart, not
new arithmetic here.

THE POINT IS THE BEST-FIT DRAW OF THE REPORTED POSTERIOR.  ``reported_posterior``
reproduces what ``run.py`` reports from -- burn-in and stuck chains trimmed
(``convergence.analyze_idata``), draws the mode pass rejects as numerically
invalid labelled -1 -- and ``best_fit_point`` takes the highest-lp valid draw
of that: one joint draw, never a vector of per-parameter medians, which need
not be a point the posterior contains (mkparam seeds from the MAP for the
same reason).  The header quotes the medians and credible intervals of that
same posterior, at the run's credible-interval width.
"""

import contextlib
import dataclasses
import difflib
import logging
import math
from pathlib import Path

import numpy as np

from .. import plot_theme
from ..plotrender import _draw_data, draw_chart

logger = logging.getLogger(__name__)

#: The yaml_key of each component whose charts make summary panels.
TRANSIT_KEY = "transit"
RV_KEY = "rvinstrument"
SED_KEY = "sed"
KIEL_KEY = "evolutionarymodel"

#: Header entries, per planet: the Parameter label and which element it is
#: read at -- the planet's own index, or the index of the orbit it sits on.
HEADER_PARAMETERS = (
    ("orbit.period", "orbit"),
    ("planet.radius", "planet"),
    ("planet.mass", "planet"),
    ("orbit.ecc", "orbit"),
)

#: The run.py default, for a config that names no prefix.
DEFAULT_PREFIX = "fitresults/planet"

#: Subtracted from a BJD time axis: the largest of these that every time on
#: the axis exceeds -- TESS's BTJD zero point, else the older 2450000
#: convention -- so the tick labels are short numbers rather than a
#: matplotlib offset.  A time axis already offset by its data is untouched.
BJD_OFFSETS = (2457000, 2450000)

# Page geometry.  A transit stack is two grid rows tall, every other panel
# one; a panel with an O-C gives it a quarter of its height.
_TRANSIT_ROWS = 2
_PANEL_ROWS = 1
_ROW_HEIGHT_IN = 6.0
_COLUMN_WIDTH_IN = 8.5
_OC_HEIGHT_RATIOS = (3, 1)
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

# Transit stack styling.  Raw points are faint when a binned set is drawn
# over them; the model is black so it reads against every row's color.
_RAW_ALPHA = 0.6
_RAW_ALPHA_UNDER_BINS = 0.15
_RAW_MARKERSIZE = 3.0
_BIN_MARKERSIZE = 6.5
_TRANSIT_MODEL_COLOR = "k"
_TRANSIT_MODEL_LW = 1.8
# The RV model against time, drawn under the points (see _draw_main).
_UNDER_MODEL_LW = 0.6
_UNDER_MODEL_ALPHA = 0.4
# The automatic offset between rows: the deepest transit plus this many
# times the typical (median over rows) scatter of the points drawn.
_SPACING_SIGMAS = 4.0


# ---------------------------------------------------------------------------
# The posterior and the point
# ---------------------------------------------------------------------------


def reported_posterior(system, idata):
    """The posterior a fit's tables and plots describe, distributed onto
    ``system``'s Parameters.

    ``idata`` is the trace as saved -- run.py keeps the FULL, untrimmed trace
    on disk and trims a view for every report -- so this repeats that
    trimming: ``convergence.analyze_idata`` drops burn-in and stuck chains,
    and ``identify_modes`` labels the draws it rejects as numerically
    invalid -1, which ``distribute_posterior`` then leaves out of every
    summary and ``best_fit_point`` out of the ranking.  A mode pass that
    finds NO valid draw raises (a figure of rejected draws is meaningless);
    any other mode-pass failure is the one ``report_pipeline`` tolerates --
    the tables then describe the combined posterior -- and is tolerated here
    for the same reason, with the same warning.

    Returns the trimmed InferenceData, ``posterior["mode"]`` attached.
    ``system`` must be built (``prepare()`` + ``build_model()``).
    """
    from ..samplers import convergence
    from .modes import NoValidDrawsError, identify_modes

    report_only = set(system.report_only_labels())
    trimmed, _ = convergence.analyze_idata(idata, exclude=report_only)
    try:
        report = identify_modes(trimmed)
    except NoValidDrawsError:
        raise
    except Exception:  # noqa: BLE001 - report_pipeline's tolerance, see above
        logger.warning(
            "summary plot: mode identification failed; ranking every "
            "post-burn-in draw (the fit's tables fall back the same way).",
            exc_info=True,
        )
    else:
        if report.n_modes > 1:
            logger.warning(
                f"summary plot: the posterior has {report.n_modes} modes.  "
                "The panels are drawn at the best-fit draw, which lies in "
                "one of them, and the header quotes the COMBINED posterior; "
                "the per-mode values are in the results table."
            )
    system.distribute_posterior(trimmed)
    return trimmed


def best_fit_point(system, idata):
    """The highest-lp valid draw of ``idata``, as a plotting point.

    Returns ``(point, (chain, draw), lp)``.  ``point`` maps every posterior
    variable but ``mode`` and the report-only Deterministics to its value at
    that draw, in INTERNAL units -- the form ``Component.plot_data`` takes,
    built the way ``run.get_draws`` builds its spaghetti draws, but through
    ``Parameter.to_internal`` since a single draw is the element vector the
    owner's conversion expects.

    A draw labelled -1 by the mode pass is never chosen: the runaway-lp
    failure produces finite, enormous lp values that win any ranking by
    construction (mkparam's ``_map_draw_from_lp`` masks them for the same
    reason).  A trace with no ``sample_stats["lp"]`` has no best fit and
    raises rather than guessing one.
    """
    stats = idata.get("sample_stats")
    if stats is None or "lp" not in stats.data_vars:
        raise ValueError(
            "The trace has no sample_stats['lp'], so it has no best-fit draw "
            "to plot.  Re-run the fit with a current EXOZIPPy, which stores "
            "lp for every sampler."
        )
    lp = np.asarray(stats["lp"].values, dtype=float)
    posterior = idata.posterior
    if "mode" in posterior:
        labels = np.asarray(posterior["mode"].values)
        lp = np.where(labels < 0, np.nan, lp)
    if not np.isfinite(lp).any():
        raise ValueError(
            "No valid draw in the trace has a finite lp, so there is no "
            "best-fit draw to plot."
        )
    chain, draw = np.unravel_index(int(np.nanargmax(lp)), lp.shape)

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
    return point, (int(chain), int(draw)), float(lp[chain, draw])


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


def planet_header_lines(system):
    """One header line per planet: P, R_P, M_P and e from the posterior.

    Read off the distributed posterior (``reported_posterior``), each value
    formatted by ``format_value`` with the Parameter's own table symbol and
    unit.  A quantity this fit does not report -- no such Parameter, or one
    with no posterior -- is left out of the line rather than shown as a
    blank; a system with no planet has no lines.  With more than one planet
    each line is prefixed by the planet's name.
    """
    planet = getattr(system, "planet", None)
    if planet is None:
        return []
    lookup = system.get_parameter_lookup()
    lines = []
    for p_idx, pname in enumerate(planet.names):
        o_idx = int(planet.orbit_map[p_idx])
        parts = []
        for label, owner in HEADER_PARAMETERS:
            param = lookup.get(label)
            if param is None or param.posterior is None:
                continue
            param.ensure_summary()
            index = o_idx if owner == "orbit" else p_idx
            summ = (
                param.summary[index]
                if isinstance(param.summary, list)
                else param.summary
            )
            value = format_value(summ)
            if value is None:
                continue
            unit = (param.unit_latex or "").replace("$", "")
            unit_text = rf"\,{unit}" if unit else ""
            parts.append(f"${param.latex} = {value}{unit_text}$")
        if parts:
            prefix = f"{pname}:  " if len(planet.names) > 1 else ""
            lines.append(prefix + "   |   ".join(parts))
    return lines


# ---------------------------------------------------------------------------
# Charts -> panels
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Panel:
    kind: str  # "transit", "rv_time", "rv_phase", "sed", "kiel"
    charts: list
    rows: int


def _collect_charts(system, point):
    charts = []
    for comp in system.active_components.values():
        charts.extend(comp.plot_data(system, point))
    return charts


def _panels(charts):
    """Sort the charts into panels, in reading order.

    Transit stacks (one per planet, its phase-folded charts), the RVs against
    time, one folded RV panel per orbit, the SED, one Kiel diagram per star.
    The unphased transit charts and every other component's charts are not
    part of the summary; the latter are logged so a missing panel is not a
    mystery.
    """
    stacks = {}
    rv_time, rv_phase, sed, kiel, unused = [], [], [], [], []
    for chart in charts:
        key = chart.component.get("yaml_key")
        meta = chart.meta or {}
        if key == TRANSIT_KEY:
            if meta.get("phase_folded"):
                stacks.setdefault(meta["planet"], []).append(chart)
        elif key == RV_KEY:
            (rv_phase if meta.get("phase_folded") else rv_time).append(chart)
        elif key == SED_KEY:
            sed.append(chart)
        elif key == KIEL_KEY:
            kiel.append(chart)
        else:
            unused.append(chart.id)
    if unused:
        logger.info(
            "summary plot: no summary panel for chart(s) %s; they are in "
            "the component's own PDFs.",
            ", ".join(unused),
        )
    panels = [_Panel("transit", c, _TRANSIT_ROWS) for c in stacks.values()]
    panels += [_Panel("rv_time", [c], _PANEL_ROWS) for c in rv_time]
    panels += [_Panel("rv_phase", [c], _PANEL_ROWS) for c in rv_phase]
    panels += [_Panel("sed", [c], _PANEL_ROWS) for c in sed]
    panels += [_Panel("kiel", [c], _PANEL_ROWS) for c in kiel]
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


def _transit_rows(stack, labels, groups, bins, system):
    """The rows of one transit stack: ``[(key, label, charts, bin_minutes,
    color)]`` in config order, a group placed where its first member is."""
    member_of = {m: g for g, members in groups.items() for m in members}
    rows, seen = [], {}
    for chart in stack:
        inst = chart.meta["instrument"]
        key = member_of.get(inst, inst)
        if key in seen:
            rows[seen[key]][2].append(chart)
            continue
        seen[key] = len(rows)
        label = key if inst in member_of else labels.get(inst, inst)
        if isinstance(bins, dict):
            minutes = bins.get(key)
        else:
            minutes = bins
        rows.append([key, label, [chart], minutes, None])
    transit = getattr(system, TRANSIT_KEY)
    for k, row in enumerate(rows):
        # A user's per-instrument plot: color (the first member's) wins;
        # otherwise the theme palette by row, so neighbors differ.
        first = row[2][0].meta["instrument"]
        user = transit.plot_color[transit.names.index(first)]
        row[4] = user or plot_theme.PALETTE[k % len(plot_theme.PALETTE)]
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
    """Per member chart: data (x hours, y), the binned set or None, and the
    model (x hours, y) -- ``y`` relative to the baseline, as the chart has it."""
    out = []
    for chart in row[2]:
        data, model = _one(chart, "data"), _one(chart, "model")
        x = np.asarray(data.x, dtype=float) * 24.0
        y = np.asarray(data.y, dtype=float)
        xm = np.asarray(model.x, dtype=float) * 24.0
        ym = np.asarray(model.y, dtype=float)
        binned = _bin(x, y, row[3] / 60.0) if row[3] else None
        out.append((x, y, binned, xm, ym))
    return out


def _auto_spacing(arrays):
    """The deepest transit plus ``_SPACING_SIGMAS`` times the median, over
    rows, of the scatter of the points each row draws (its bins if binned)."""
    depth, sigmas = 0.0, []
    for members in arrays:
        row_sigmas = []
        for x, y, binned, xm, ym in members:
            depth = max(depth, -float(np.min(ym)))
            bx, by = binned if binned is not None else (x, y)
            row_sigmas.append(_robust_sigma(by - np.interp(bx, xm, ym)))
        sigmas.append(max(row_sigmas))
    return depth + _SPACING_SIGMAS * float(np.median(sigmas)), depth


def _draw_transit_stack(ax, rows, spacing):
    arrays = [_transit_row_arrays(row) for row in rows]
    auto, depth = _auto_spacing(arrays)
    spacing = auto if spacing is None else float(spacing)

    x_range = rows[0][2][0].x_range
    for k, (row, members) in enumerate(zip(rows, arrays)):
        offset = 1.0 - k * spacing
        color = row[4]
        for x, y, binned, xm, ym in members:
            ax.plot(
                x,
                y + offset,
                "o",
                ms=_RAW_MARKERSIZE,
                color=color,
                mec="none",
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
                    color=color,
                    mec="k",
                    mew=0.6,
                    zorder=2,
                )
            ax.plot(
                xm,
                ym + offset,
                "-",
                color=_TRANSIT_MODEL_COLOR,
                lw=_TRANSIT_MODEL_LW,
                zorder=3,
            )

    if x_range is not None:
        ax.set_xlim(24.0 * x_range[0], 24.0 * x_range[1])
    lo, hi = ax.get_xlim()
    gap = spacing - depth
    for k, row in enumerate(rows):
        ax.text(
            lo + 0.02 * (hi - lo),
            1.0 - k * spacing + 0.35 * gap,
            row[1],
            fontsize=_FONT_RC["legend.fontsize"],
            va="bottom",
            zorder=5,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )
    ax.set_ylim(
        1.0 - (len(rows) - 1) * spacing - depth - 0.5 * gap,
        1.0 + 0.5 * spacing,
    )
    ax.set_xlabel("Time from Mid-Transit [hr]")
    ax.set_ylabel("Normalized Flux + Constant")


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


def _prepared(chart, labels, time_offset=False):
    """The chart with instrument names relabelled and, for a time axis whose
    every value is past one of ``BJD_OFFSETS``, the largest such subtracted."""
    offset = 0
    if time_offset:
        x_min = min(
            float(np.min(t.x)) for t in chart.traces if np.size(t.x) > 0
        )
        offset = next((o for o in BJD_OFFSETS if x_min > o), 0)
    meta = dict(chart.meta or {})
    if "residuals" in meta:
        meta["residuals"] = {
            "ylabel": meta["residuals"]["ylabel"],
            "traces": [
                _shifted(t, offset, labels)
                for t in meta["residuals"]["traces"]
            ],
        }
    return dataclasses.replace(
        chart,
        traces=[_shifted(t, offset, labels) for t in chart.traces],
        xlabel=_offset_label(chart.xlabel, offset),
        x_range=None
        if chart.x_range is None
        else [float(v) - offset for v in chart.x_range],
        meta=meta,
    )


def _style(ax):
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=_TICK_LABELSIZE,
    )
    ax.minorticks_on()


def _draw_main(ax, chart, legend, models_under):
    """``draw_chart``, or -- ``models_under`` -- the model curves first,
    thin and beneath the points.  That is the RVs against time: over a
    baseline of many orbits the model is a band rather than a curve, and at
    the renderer's model weight it hides the data it is compared with."""
    if not models_under:
        draw_chart(ax, chart, legend=legend)
        return
    models = [t for t in chart.traces if t.role == "model"]
    rest = [t for t in chart.traces if t.role != "model"]
    draw_chart(
        ax,
        dataclasses.replace(chart, traces=models),
        model_alpha=_UNDER_MODEL_ALPHA,
        legend=False,
    )
    for line in ax.get_lines():
        line.set_zorder(0)
        line.set_linewidth(_UNDER_MODEL_LW)
    draw_chart(ax, dataclasses.replace(chart, traces=rest), legend=legend)


def _draw_chart_panel(fig, cell, chart, legend, models_under=False):
    """One chart in one grid cell, with an O-C sub-panel when the chart
    declares ``meta["residuals"]``."""
    residuals = (chart.meta or {}).get("residuals")
    if residuals is None:
        ax = fig.add_subplot(cell)
        _draw_main(ax, chart, legend, models_under)
        _style(ax)
        return [ax]
    sub = cell.subgridspec(2, 1, height_ratios=_OC_HEIGHT_RATIOS, hspace=0.0)
    ax = fig.add_subplot(sub[0])
    ax_oc = fig.add_subplot(sub[1], sharex=ax)
    _draw_main(ax, chart, legend, models_under)
    ax.set_xlabel("")
    for trace in residuals["traces"]:
        _draw_data(ax_oc, trace)
    ax_oc.axhline(0.0, color="0.4", ls="--", lw=1.0, zorder=0)
    if chart.x_log:
        ax_oc.set_xscale("log")
    ax_oc.set_xlabel(chart.xlabel)
    ax_oc.set_ylabel(residuals["ylabel"])
    for a in (ax, ax_oc):
        _style(a)
    ax.tick_params(labelbottom=False)
    return [ax, ax_oc]


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
):
    """Draw the summary figure for ``system`` at ``point``; return the Figure.

    ``system`` must be built and ``point`` a plotting point (internal units,
    e.g. from ``best_fit_point``).  The caller owns the figure: save it, then
    ``plt.close`` it.

    Parameters
    ----------
    title : str, optional
        Bold title across the top.
    header_lines : sequence of str
        Lines under the title -- ``planet_header_lines(system)`` once a
        posterior is distributed.  Mathtext is rendered.
    labels : dict, optional
        Display name per instrument (transit or RV), keyed by the name in the
        config, e.g. ``{"TCS_MuSCAT2_UT20251116_9": "MuSCAT2 ($i'$)"}``.
        Unknown names raise.
    transit_groups : dict, optional
        ``{display label: [transit names]}``: files drawn on ONE row of the
        stack, each with its own model curve (identical curves overlap) --
        e.g. several TESS sectors at one cadence.  A name in two groups, or
        not a transit of this fit, raises.
    transit_bin : float or dict, optional
        Bin the phased transit points to this many minutes, drawn over the
        unbinned points.  A number bins every row; a dict bins only the rows
        it names, keyed by transit name or, for a grouped row, group label.
    transit_spacing : float, optional
        Vertical offset between stacked transits, in normalized flux.
        Default: the deepest transit plus four times the typical scatter.
    figsize : (float, float), optional
        Inches.  Default scales with the number of panels.
    """
    import matplotlib.pyplot as plt

    labels = dict(labels or {})
    groups = {str(g): list(m) for g, m in (transit_groups or {}).items()}

    charts = _collect_charts(system, point)
    panels = _panels(charts)
    if not panels:
        raise ValueError(
            "This fit has no chart a summary panel is made of (transits, "
            "RVs, an SED or a Kiel diagram)."
        )

    transit_names = (
        list(getattr(system, TRANSIT_KEY).names)
        if any(p.kind == "transit" for p in panels)
        else []
    )
    rv_names = [
        t.name
        for p in panels
        if p.kind == "rv_time"
        for t in _role(p.charts[0], "data")
    ]
    _check_names("labels", labels, transit_names + rv_names)
    grouped = [m for members in groups.values() for m in members]
    _check_names("transit_groups", grouped, transit_names)
    twice = sorted({m for m in grouped if grouped.count(m) > 1})
    if twice:
        raise ValueError(
            f"transit_groups puts {twice} in more than one group; a file is "
            "drawn on one row."
        )
    if isinstance(transit_bin, dict):
        ungrouped = [n for n in transit_names if n not in grouped]
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
            for panel, col, row in placed:
                cell = grid[row : row + panel.rows, col]
                if panel.kind == "transit":
                    ax = fig.add_subplot(cell)
                    rows = _transit_rows(
                        panel.charts, labels, groups, transit_bin, system
                    )
                    _draw_transit_stack(ax, rows, transit_spacing)
                    _style(ax)
                elif panel.kind == "rv_time":
                    chart = _prepared(
                        panel.charts[0], labels, time_offset=True
                    )
                    _draw_chart_panel(
                        fig, cell, chart, legend=False, models_under=True
                    )
                elif panel.kind == "rv_phase":
                    chart = _prepared(panel.charts[0], labels)
                    axes = _draw_chart_panel(fig, cell, chart, legend=True)
                    axes[0].set_xlim(0.0, 1.0)
                elif panel.kind == "sed":
                    chart = panel.charts[0]
                    # One star's spectrum and photometry need no key; several
                    # stars' identities do.
                    n_ids = len(_role(chart, "data"))
                    _draw_chart_panel(fig, cell, chart, legend=n_ids > 1)
                else:
                    _draw_chart_panel(fig, cell, panel.charts[0], legend=True)
        except BaseException:
            plt.close(fig)
            raise
    return fig


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

    Example::

        from exozippy.outputs.summary_plot import create_summary_plot

        create_summary_plot(
            "toi5432.yaml",
            title="TOI-5432",
            transit_groups={
                "TESS 600 s": ["TESS_UT20210916", "TESS_UT20211107"],
                "TESS 120 s": ["TESS_UT20231016", "TESS_UT20231112"],
            },
            transit_bin={"TESS 120 s": 10},
        )
    """
    import arviz as az
    import matplotlib.pyplot as plt

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
        system.build_model()

        idata = az.from_netcdf(str(trace_path))
        check_trace_freshness(idata, system, trace_path)
        posterior = reported_posterior(system, idata)
        point, (chain, draw), lp = best_fit_point(system, posterior)
        logger.info(
            f"summary plot: best-fit draw chain {chain}, draw {draw} of the "
            f"post-burn-in trace (lp = {lp:.2f})."
        )

        if title is None:
            title = (config.get("run") or {}).get("name")
        fig = summary_figure(
            system,
            point,
            title=title,
            header_lines=planet_header_lines(system),
            labels=labels,
            transit_groups=transit_groups,
            transit_bin=transit_bin,
            transit_spacing=transit_spacing,
            figsize=figsize,
        )
        if out is None:
            out = Path(f"{prefix}_mcmc_summary.pdf").resolve()
        try:
            fig.savefig(
                out, bbox_inches="tight", facecolor="white", dpi=_SAVE_DPI
            )
        finally:
            plt.close(fig)
    logger.info(f"summary plot: wrote {out}")
    return out
