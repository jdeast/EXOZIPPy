"""Render PlotSpecs to matplotlib figures -- the saved-PDF counterpart of the
GUI's plotly-adapter.ts.

A component describes each of its plots ONCE, as the PlotSpec list returned by
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
  observation epochs).  Model traces from posterior draws after the first
  are drawn as spaghetti (alpha 0.1).
* ``"residual"`` -> gray markers around zero.

Trace ``style`` (all optional): ``series_index`` (fixed categorical color
``C{i}`` -- assigned per instrument at load, never re-cycled per chart),
``color`` / ``marker`` user overrides, ``lw`` line width, ``legend`` to
force a legend entry on a non-data trace.

PlotSpec ``meta`` presentation keys (all optional):

* ``file_tag``   -- output basename: ``{prefix}_{file_tag}.pdf``.  Falls back
  to the spec ``id`` with dots replaced by underscores.
* ``figsize``    -- ``(w, h)`` inches.
* ``hline_y``    -- draw a dotted horizontal reference line at this y.
* ``x_range`` / ``y_range``     -- explicit ``[lo, hi]`` axis windows.
* ``x_log`` / ``y_log``         -- logarithmic axes.
* ``x_inverted`` / ``y_inverted`` -- reversed axes (magnitudes, RA).
* ``aspect_equal`` -- equal-aspect axes (sky-plane plots).
* ``caption``    -- LaTeX figure caption for the generated paper draft
  (``outputs/modeling.py``, the third consumer of these specs).  Neither
  renderer draws it; without one the draft falls back to a generic
  caption built from the spec title.  It is emitted verbatim into
  ``\\caption{...}``, so escape any non-LaTeX pieces (instrument names!)
  with ``latex_escape`` when composing it.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Mirror of plotly-adapter's role encodings.
_DATA_ALPHA = 0.6
_MODEL_COLOR = "r"
_RESIDUAL_COLOR = "0.5"


def _trace_color(trace, fallback=None):
    """Explicit style color, else the fixed C{series_index}, else fallback."""
    style = trace.style or {}
    if style.get("color") is not None:
        return style["color"]
    if style.get("series_index") is not None:
        return f"C{int(style['series_index'])}"
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
        alpha=_DATA_ALPHA,
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
            alpha=alpha,
            zorder=2,
            label=label,
        )
    else:
        ax.plot(
            np.asarray(trace.x, dtype=float),
            np.asarray(trace.y, dtype=float),
            "-",
            color=color,
            lw=style.get("lw", 1.5),
            alpha=alpha,
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
        alpha=_DATA_ALPHA,
        zorder=1,
        label=trace.name or None,
    )


def _apply_meta(ax, meta):
    """Axis decorations from the spec's meta presentation keys."""
    if meta.get("hline_y") is not None:
        ax.axhline(
            float(meta["hline_y"]), color="black", linestyle=":", alpha=0.5
        )
    if meta.get("x_log"):
        ax.set_xscale("log")
    if meta.get("y_log"):
        ax.set_yscale("log")
    if meta.get("x_range") is not None:
        ax.set_xlim(*[float(v) for v in meta["x_range"]])
    if meta.get("y_range") is not None:
        ax.set_ylim(*[float(v) for v in meta["y_range"]])
    # Invert AFTER any explicit range so [lo, hi] semantics stay ascending.
    if meta.get("x_inverted"):
        ax.invert_xaxis()
    if meta.get("y_inverted"):
        ax.invert_yaxis()
    if meta.get("aspect_equal"):
        ax.set_aspect("equal", adjustable="datalim")


def render_spec_groups(spec_groups, filename_prefix="debug"):
    """Render one figure per spec, overlaying model traces from every group.

    Parameters
    ----------
    spec_groups : list[list[PlotSpec]]
        One ``plot_data`` result per posterior point.  The FIRST group is the
        reference: it supplies the data traces, labels, and decorations
        (matching the historical convention that data offsets/cleaning use
        ``points[0]``).  Later groups contribute only their model traces,
        drawn as low-alpha spaghetti.
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

            _apply_meta(ax, meta)
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
