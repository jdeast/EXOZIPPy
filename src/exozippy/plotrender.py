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

Axis GEOMETRY is no longer here: ``x_range``/``y_range``, ``x_log``/``y_log``
and ``x_inverted``/``y_inverted`` are first-class ``Chart`` attributes (review
4.11.3).  They moved because both renderers must consult them to lay out an
axis at all, so a typo in a stringly-typed key silently produced a linear axis
where a log one was meant.  ``meta`` now carries annotations only.

Per-trace opacity: ``Trace.alpha`` overrides the role default, and the role
defaults here are the ones the GUI mirrors (review 4.11.6).
"""

from __future__ import annotations

import logging

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
            handles, _labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize="small")
            fig.tight_layout()

            tag = meta.get("file_tag") or spec.id.replace(".", "_")
            path = f"{filename_prefix}_{tag}.pdf"
            fig.savefig(path)
            written.append(path)
        finally:
            plt.close(fig)
    return written


def plot_via_specs(component, system, points, filename_prefix="debug"):
    """The standard ``Component.plot`` body: plot_data per point -> PDFs.

    ``points`` may be a single point dict (pre-flight) or a list of posterior
    draws (spaghetti).  A draw whose ``plot_data`` raises is skipped with a
    warning, matching the per-draw tolerance of the old hand-drawn loops --
    except the first (reference) point, whose failure aborts the plot since
    it supplies the data traces themselves.
    """
    if isinstance(points, dict):
        points = [points]
    if not points:
        logger.warning("No points provided for plotting.")
        return []

    spec_groups = []
    for idx, point in enumerate(points):
        try:
            spec_groups.append(component.plot_data(system, point))
        except Exception as exc:  # noqa: BLE001 - skip a bad posterior draw
            if idx == 0:
                raise
            logger.warning(
                "plot_data failed for draw %d of %s: %s",
                idx,
                getattr(component, "prefix", component),
                exc,
            )
    return render_spec_groups(spec_groups, filename_prefix=filename_prefix)
