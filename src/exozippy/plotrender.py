"""Render Charts to matplotlib figures -- the saved-PDF counterpart of the
GUI's plotly-adapter.ts.

A component describes each of its plots ONCE, as the Chart list returned by
``Component.plot_data(system, point)``.  The GUI renders those specs with
plotly; this module renders the same specs with matplotlib for the pre-flight
and posterior PDFs.  ``Component.plot`` implementations reduce to::

    def plot(self, system, points, filename_prefix="debug"):
        from exozippy.plotrender import plot_via_specs
        plot_via_specs(self, system, points, filename_prefix)

so there is no second, hand-drawn version of any plot to keep in sync.  The
two renderers must stay visually equivalent: when you extend the meta or
style vocabulary here, mirror it in ``gui/frontend/src/plotly-adapter.ts``.

Vocabulary
----------
Trace ``role`` drives the mark (mirroring plotly-adapter):

* ``"data"``    -> errorbar markers (alpha 0.6, zorder 1, legend label).
* ``"model"``   -> line (red by default) when ``kind == "line"``; red point
  markers when ``kind == "scatter"`` (e.g. a model sampled only at the
  observation epochs).  Model alpha is set ONCE PER FIGURE from the number
  of posterior draws, not per draw: a single draw is opaque (alpha 0.8),
  and as soon as there is more than one EVERY model trace is spaghetti
  (alpha 0.1) -- including the reference draw's, which is otherwise
  privileged (it supplies the data traces and decorations).  That is
  deliberate: the reference point is only "first" in the list, not a
  best-fit, so drawing it darker would advertise a distinction the fit
  does not make.
* ``"residual"`` -> gray markers around zero.

Trace ``style`` (all optional): ``series_index`` (fixed categorical color
``C{i}`` -- assigned per instrument at load, never re-cycled per chart),
``color`` / ``marker`` user overrides, ``lw`` line width, ``legend`` to
force a legend entry on a non-data trace.

Chart ``meta`` presentation keys (all optional):

* ``file_tag``   -- output basename: ``{prefix}_{file_tag}.pdf``.  Falls back
  to the spec ``id`` with dots replaced by underscores.
* ``figsize``    -- ``(w, h)`` inches.
* ``hline_y``    -- draw a dotted horizontal reference line at this y.
* ``aspect_equal`` -- equal-aspect axes (sky-plane plots).  Read only here;
  the plotly adapter ignores it, which is why it was NOT promoted to a Chart
  field in review 4.11.3 while the six axis-geometry keys were.
* ``caption``    -- LaTeX figure caption for the generated paper draft
  (``outputs/modeling.py``, the third consumer of these specs).  Neither
  renderer draws it; without one the draft falls back to a generic
  caption built from the spec title.  It is emitted verbatim into
  ``\\caption{...}``, so escape any non-LaTeX pieces (instrument names!)
  with ``latex_escape`` when composing it.
* ``residuals``  -- the chart's O-C at the plotted point: a list of
  ``Trace(role="residual")`` in the chart's OWN x and y units (m/s on an
  RV chart, dex on the SED's log10 lambda*F_lambda), declared by the
  component that owns the model (it alone can evaluate the likelihood's
  model at the observations).  Like ``caption`` it is drawn by NEITHER
  renderer: its consumer is the system summary figure
  (``outputs/summary_plot.py``), which gives each chart that carries one an
  O-C sub-panel.

Axis GEOMETRY is no longer here: ``x_range``/``y_range``, ``x_log``/``y_log``
and ``x_inverted``/``y_inverted`` are first-class ``Chart`` attributes (review
4.11.3).  They moved because both renderers must consult them to lay out an
axis at all, so a typo in a stringly-typed key silently produced a linear axis
where a log one was meant.  ``meta`` now carries annotations only.

Per-trace opacity: ``Trace.alpha`` overrides the role default, and the role
defaults here are the ones the GUI mirrors (review 4.11.6).
"""

from __future__ import annotations

import dataclasses
import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from . import plot_theme

logger = logging.getLogger(__name__)

# Role encodings come from the ONE shared table (review 4.11.4). They used to
# be spelled here as matplotlib shorthands -- "r", "0.5", and an implicit
# f"C{n}" cycle -- and the plotly adapter carried its own hex copies that
# matched by convention only. Each substitution below is exactly equal to the
# shorthand it replaces (pinned in tests/test_plot_theme.py), so the saved
# PDFs do not move; what moved is the GUI, onto these values.
_DATA_ALPHA = plot_theme.ROLE_ALPHA["data"]
_MODEL_COLOR = plot_theme.ROLE_COLORS["model"]
_RESIDUAL_COLOR = plot_theme.ROLE_COLORS["residual"]


def _trace_alpha(trace, default):
    """The trace's own alpha if it set one, else the role default.

    One resolver so a per-trace override and the role default cannot drift
    apart, and so the plotly adapter has a single rule to mirror (review
    4.11.6: the two renderers had silently disagreed, this file drawing data
    at 0.6 and the GUI drawing it opaque).
    """
    return default if trace.alpha is None else float(trace.alpha)


def _trace_color(trace, fallback=None):
    """Explicit style color, else the shared palette, else fallback.

    The palette lookup replaces an f"C{n}" handoff to matplotlib's own
    property cycle. That cycle IS tab10 and so resolved to exactly these
    values, but only by coincidence of matplotlib's defaults -- indexing the
    shared table makes the agreement with the GUI a fact rather than a
    coincidence, and survives a matplotlib that ships a different cycle.
    """
    style = trace.style or {}
    if style.get("color") is not None:
        return style["color"]
    if style.get("series_index") is not None:
        idx = int(style["series_index"]) % len(plot_theme.PALETTE)
        return plot_theme.PALETTE[idx]
    return fallback


def _as_err(err):
    """Pass (N,) or (2, N) error arrays through as numpy for errorbar."""
    if err is None:
        return None
    return np.asarray(err, dtype=float)


def _draw_data(ax, trace):
    style = trace.style or {}
    ax.errorbar(
        np.asarray(trace.x, dtype=float),
        np.asarray(trace.y, dtype=float),
        yerr=_as_err(trace.yerr),
        xerr=_as_err(trace.xerr),
        fmt=style.get("marker") or "o",
        color=_trace_color(trace),
        alpha=_trace_alpha(trace, _DATA_ALPHA),
        zorder=1,
        label=trace.name or None,
    )


def _draw_model(ax, trace, alpha):
    style = trace.style or {}
    color = _trace_color(trace, fallback=_MODEL_COLOR)
    label = trace.name if style.get("legend") else None
    if trace.kind == "scatter":
        ax.plot(
            np.asarray(trace.x, dtype=float),
            np.asarray(trace.y, dtype=float),
            style.get("marker") or ".",
            color=color,
            alpha=_trace_alpha(trace, alpha),
            zorder=2,
            label=label,
        )
    else:
        ax.plot(
            np.asarray(trace.x, dtype=float),
            np.asarray(trace.y, dtype=float),
            "-",
            color=color,
            lw=style.get("lw", plot_theme.DEFAULT_LINEWIDTH),
            alpha=_trace_alpha(trace, alpha),
            zorder=2,
            label=label,
        )


def _draw_residual(ax, trace):
    ax.errorbar(
        np.asarray(trace.x, dtype=float),
        np.asarray(trace.y, dtype=float),
        yerr=_as_err(trace.yerr),
        fmt=".",
        color=_RESIDUAL_COLOR,
        alpha=_trace_alpha(trace, _DATA_ALPHA),
        zorder=1,
        label=trace.name or None,
    )


def _apply_axes(ax, spec):
    """Axis geometry from the chart's first-class fields, plus meta extras.

    The six geometry fields are attributes on the Chart (review 4.11.3); only
    `hline_y` and `aspect_equal` are still read from `meta`, because each is
    honored by exactly this renderer and promoting them would advertise a
    field the plotly adapter silently ignores.
    """
    meta = spec.meta or {}
    if meta.get("hline_y") is not None:
        ax.axhline(
            float(meta["hline_y"]), color="black", linestyle=":", alpha=0.5
        )
    if spec.x_log:
        ax.set_xscale("log")
    if spec.y_log:
        ax.set_yscale("log")
    if spec.x_range is not None:
        ax.set_xlim(*[float(v) for v in spec.x_range])
    if spec.y_range is not None:
        ax.set_ylim(*[float(v) for v in spec.y_range])
    # Invert AFTER any explicit range so [lo, hi] semantics stay ascending.
    if spec.x_inverted:
        ax.invert_xaxis()
    if spec.y_inverted:
        ax.invert_yaxis()
    if meta.get("aspect_equal"):
        ax.set_aspect("equal", adjustable="datalim")


# Legend proxies for spaghetti traces are drawn at this alpha rather than
# the traces' own: fifty draws at alpha 0.1 read as a curve, but one legend
# swatch at 0.1 is invisible.
_LEGEND_MIN_ALPHA = 0.8


def _legend(ax):
    """One legend entry per LABEL, first occurrence winning, swatches legible.

    A spaghetti figure overlays the same named model trace once per draw --
    fifty identical "MIST track" rows would otherwise swamp the panel -- and
    a component may legitimately name the same series in more than one
    trace.  De-duplicating by label keeps matplotlib's handle order (lines
    before errorbar containers), first occurrence winning.  This lived in a
    private copy of ``render_spec_groups`` inside the evolutionarymodel
    component until 2026-09 (review 4.11.7); it belongs to every chart.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    legend = ax.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="best",
        fontsize="small",
    )
    # legend_handles are the legend's own proxy artists, so raising their
    # alpha changes the swatch and leaves the plotted spaghetti untouched.
    for proxy, handle in zip(legend.legend_handles, unique.values()):
        # An errorbar handle is a Container, not an Artist: no alpha to read,
        # and a data swatch at the role default is legible anyway.
        alpha = getattr(handle, "get_alpha", lambda: None)()
        if alpha is not None and alpha < _LEGEND_MIN_ALPHA:
            proxy.set_alpha(_LEGEND_MIN_ALPHA)
    return legend


def render_spec_groups(spec_groups, filename_prefix="debug"):
    """Render one figure per spec, overlaying model traces from every group.

    Parameters
    ----------
    spec_groups : list[list[Chart]]
        One ``plot_data`` result per posterior point.  The FIRST group is the
        reference: it supplies the data traces, labels, and decorations
        (matching the historical convention that data offsets/cleaning use
        ``points[0]``).  Later groups contribute only their model traces.
        The reference group is NOT privileged in the model layer: with more
        than one group every model trace is drawn at the same spaghetti
        alpha, the reference's included.
    filename_prefix : str
        Output files are ``{filename_prefix}_{file_tag}.pdf``.
    """
    if not spec_groups:
        return []
    ref_specs = spec_groups[0]
    # Model spaghetti from the later groups, matched to the reference spec by
    # id (a draw whose plot_data failed simply contributes nothing).
    extra_models = {}
    for group in spec_groups[1:]:
        for spec in group:
            models = [t for t in spec.traces if t.role == "model"]
            if models:
                extra_models.setdefault(spec.id, []).append(models)

    # One alpha for the whole figure, applied to the reference group's model
    # traces below as well as to extra_models -- see the module docstring.
    model_alpha = 0.8 if len(spec_groups) == 1 else 0.1
    written = []
    for spec in ref_specs:
        meta = spec.meta or {}
        fig, ax = plt.subplots(figsize=tuple(meta.get("figsize") or (10, 6)))
        try:
            for trace in spec.traces:
                if trace.role == "model":
                    _draw_model(ax, trace, model_alpha)
                elif trace.role == "residual":
                    _draw_residual(ax, trace)
                else:
                    _draw_data(ax, trace)
            for models in extra_models.get(spec.id, []):
                for trace in models:
                    _draw_model(ax, trace, model_alpha)

            _apply_axes(ax, spec)
            ax.set_xlabel(spec.xlabel)
            ax.set_ylabel(spec.ylabel)
            ax.set_title(spec.title)
            _legend(ax)
            fig.tight_layout()

            tag = meta.get("file_tag") or spec.id.replace(".", "_")
            path = f"{filename_prefix}_{tag}.pdf"
            fig.savefig(path)
            written.append(path)
        finally:
            plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Posterior-draw plot_data on the sampler's pool (review 2.6.44)
# ---------------------------------------------------------------------------

# The wrap-up's posterior plots call plot_data once per draw (51 per pass,
# once more per mode, and again for the one-page figure), serially.  On a
# finite-source binary lens that is the dominant wrap-up cost: measured on
# examples/MOA-2008-BLG-379 (CPU time, 51 draws), the light-curve plot_data
# is 17.4 s per draw against a 0.23 s logp -- 74 logp-equivalents a draw,
# ~15 CPU-minutes a pass -- and it scales with the magnification cost, so at
# OGLE-2013-BLG-0341's ~225 s logp one serial pass would be days.  Inside
# ``plot_pool(cores, eval_timeout)`` (run.py opens it around the wrap-up's
# plot stages) the draws after the reference go out on a fork pool through
# samplers._common.map_draws, each under eval_timeout; outside it (the
# pre-flight plots, a GUI preview, a test) everything is serial, as before.
_PLOT_POOL = {"cores": 1, "eval_timeout": None}
# Pool only when the serial remainder would take longer than this (the
# reference draw's own time x the draws left): a fork per component is not
# free, and an RV or transit plot_data is milliseconds.
PLOT_POOL_MIN_S = 5.0

_PD_COMPONENT = None
_PD_SYSTEM = None
_PD_POINTS = None
_PD_KWARGS = None


class plot_pool:  # noqa: N801 - used as a context manager, like a function
    """Context in which plot_data_at_points may use ``cores`` workers."""

    def __init__(self, cores, eval_timeout=None):
        self.grant = {"cores": cores, "eval_timeout": eval_timeout}

    def __enter__(self):
        self.saved = dict(_PLOT_POOL)
        _PLOT_POOL.update(self.grant)
        return self

    def __exit__(self, *exc):
        _PLOT_POOL.clear()
        _PLOT_POOL.update(self.saved)
        return False


def _without_nodes(charts):
    """``charts`` with every trace's ``node`` dropped.

    ``Trace.node`` is the symbolic pytensor graph behind a model curve, kept
    for the GUI's live re-evaluation; it is not serialized, not compared,
    and no PDF renderer reads it -- and it does not pickle (a pymc
    CustomDist class), so a chart carrying one cannot come back from a pool
    worker.  Dropped on both paths, so serial and pooled charts are equal.
    """
    return [
        dataclasses.replace(
            chart,
            traces=[dataclasses.replace(tr, node=None) for tr in chart.traces],
        )
        for chart in charts
    ]


def _plot_data_range(lo, hi):
    """Worker: ("ok", charts) or ("error", text) for points [lo, hi)."""
    out = []
    for i in range(lo, hi):
        try:
            out.append(
                (
                    "ok",
                    _without_nodes(
                        _PD_COMPONENT.plot_data(
                            _PD_SYSTEM, _PD_POINTS[i], **_PD_KWARGS
                        )
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - reported by the parent
            out.append(("error", f"{type(exc).__name__}: {exc}"))
    return out


def plot_data_at_points(component, system, points, t_first=None, **kwargs):
    """``component.plot_data`` at each of ``points``, in order.

    Returns ``[("ok", charts) | ("error", text) | ("timeout", None)]``.
    Serial unless a ``plot_pool`` grant is open AND the serial cost
    (``t_first`` -- the caller's measured time for one call -- times the
    number of points) exceeds ``PLOT_POOL_MIN_S``; pooled results are the
    same charts, computed in forked workers.  A draw whose call outlasts
    eval_timeout is ("timeout", None), and the caller reports it.
    """
    global _PD_COMPONENT, _PD_SYSTEM, _PD_POINTS, _PD_KWARGS
    from .samplers._common import map_draws

    cores = _PLOT_POOL["cores"]
    pooled = (
        cores != 1
        and len(points) > 1
        and t_first is not None
        and t_first * len(points) > PLOT_POOL_MIN_S
    )
    _PD_COMPONENT, _PD_SYSTEM, _PD_POINTS, _PD_KWARGS = (
        component,
        system,
        list(points),
        kwargs,
    )
    try:
        results, _timed_out = map_draws(
            _plot_data_range,
            len(points),
            cores=cores if pooled else 1,
            eval_timeout=_PLOT_POOL["eval_timeout"],
            label=f"plot_data ({getattr(component, 'prefix', component)})",
            log=logger,
        )
    finally:
        _PD_COMPONENT = _PD_SYSTEM = _PD_POINTS = _PD_KWARGS = None
    return [("timeout", None) if r is None else r for r in results]


def plot_via_specs(
    component, system, points, filename_prefix="debug", **plot_data_kwargs
):
    """The standard ``Component.plot`` body: plot_data per point -> PDFs.

    ``points`` may be a single point dict (pre-flight) or a list of posterior
    draws (spaghetti).  A draw whose ``plot_data`` raises is skipped with a
    warning, matching the per-draw tolerance of the old hand-drawn loops --
    except the first (reference) point, whose failure aborts the plot since
    it supplies the data traces themselves.  ``plot_data_kwargs`` go to
    every ``plot_data`` call: a component whose charts' layout depends on
    the point (mulensinstrument's zoom windows) fixes it at the reference
    point and passes it here, so every draw lands on the same charts.
    """
    if isinstance(points, dict):
        points = [points]
    if not points:
        logger.warning("No points provided for plotting.")
        return []

    # The reference point in-process: its failure aborts the plot (it
    # supplies the data traces), and its time decides whether the rest are
    # worth a pool (plot_data_at_points).
    t0 = time.perf_counter()
    spec_groups = [component.plot_data(system, points[0], **plot_data_kwargs)]
    t_first = time.perf_counter() - t0
    rest = plot_data_at_points(
        component, system, points[1:], t_first=t_first, **plot_data_kwargs
    )
    n_timeout = 0
    failed = []
    for idx, (state, value) in enumerate(rest, start=1):
        if state == "ok":
            spec_groups.append(value)
        elif state == "timeout":
            n_timeout += 1
        else:  # a bad posterior draw is skipped
            failed.append((idx, value))
            logger.debug(
                "plot_data failed for draw %d of %s: %s",
                idx,
                getattr(component, "prefix", component),
                value,
            )
    if failed:
        # One line for the lot (docs/logging.md, "logged ONCE"); each
        # draw's error is at DEBUG above.
        logger.warning(
            "plot_data failed for %d of %d posterior draw(s) of %s, which are "
            "left out of its plots (first: draw %d: %s).",
            len(failed),
            len(points) - 1,
            getattr(component, "prefix", component),
            failed[0][0],
            failed[0][1],
        )
    if n_timeout:
        logger.warning(
            "%d of %d posterior draw(s) of %s exceeded eval_timeout=%gs in "
            "plot_data and are left out of its plots; raise the sampler's "
            "eval_timeout to draw them.",
            n_timeout,
            len(points) - 1,
            getattr(component, "prefix", component),
            float(_PLOT_POOL["eval_timeout"]),
        )
    return render_spec_groups(spec_groups, filename_prefix=filename_prefix)
