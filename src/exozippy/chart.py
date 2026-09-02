"""
JSON-serializable plot descriptions for the EXOZIPPy GUI.

Components draw matplotlib figures directly via ``Component.plot`` for
the CLI, but a browser GUI needs the plot *data* (arrays plus labels),
not a rendered figure, so it can draw pan/zoomable charts and re-render
model curves when parameter sliders move.  ``Component.plot_data``
returns a list of :class:`Chart`; this module defines that container
and its ``to_json`` helper (numpy arrays -> rounded Python lists, with
non-finite values mapped to ``None`` so the payload is valid JSON).

Each :class:`Chart` carries one or more :class:`Trace` objects.  A
trace's ``role`` is ``"data"`` (observations), ``"model"`` (a model
curve evaluated at a parameter point) or ``"residual"``.  A trace may
also carry, in its non-serialized ``node`` field, the symbolic pytensor
graph node behind a model curve -- this is what a later prompt (G5) uses
to compile fast re-evaluation functions when a slider moves.  ``node``
is deliberately excluded from ``to_json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Default number of decimal places kept when serializing float arrays.
# Six places is ample for time (BJD), RV and magnitude payloads while trimming
# the JSON size relative to full float64 repr. Values that span many orders of
# magnitude (e.g. SED flux at Earth ~1e-13, which would round to 0.0 here) are
# plotted in log10 space by the component, so they too arrive as normal-scale
# numbers -- no special-casing needed in this serializer.
_FLOAT_ROUND = 6


def _finite_or_none(x):
    """Return float(x) if finite, else None (JSON has no NaN/inf)."""
    xf = float(x)
    return xf if np.isfinite(xf) else None


def _array_to_list(arr, round_to=_FLOAT_ROUND):
    """Convert a numpy array (or None) to nested Python lists of floats.

    1-D arrays become a flat list; 2-D arrays (e.g. asymmetric
    ``yerr`` of shape ``(2, N)``) become a list of lists.  Non-finite
    entries are mapped to ``None`` so ``json.dumps`` produces strictly
    valid JSON.
    """
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.ndim == 0:
        return _finite_or_none(round(float(a), round_to))
    a = np.round(a, round_to)
    if a.ndim == 1:
        return [_finite_or_none(v) for v in a.tolist()]
    return [[_finite_or_none(v) for v in row] for row in a.tolist()]


def _jsonify(obj):
    """Recursively coerce numpy scalars/arrays inside meta dicts to JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _array_to_list(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _finite_or_none(float(obj))
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@dataclass
class Trace:
    """One data or model curve within a :class:`Chart`.

    Parameters
    ----------
    name : str
        Legend label for this curve (e.g. an instrument or star name).
    role : str
        ``"data"``, ``"model"`` or ``"residual"``.
    kind : str
        Preferred mark: ``"scatter"`` or ``"line"``.
    x, y : array-like
        The curve's abscissa/ordinate in the spec's axis units.
    yerr : array-like, optional
        Symmetric ``(N,)`` or asymmetric ``(2, N)`` vertical errors.
    xerr : array-like, optional
        Symmetric ``(N,)`` or asymmetric ``(2, N)`` horizontal errors
        (sky-plane astrometry has uncertainties on both axes).
    node : object, optional
        The symbolic pytensor node behind a model curve, kept for G5's
        compiled re-evaluation.  Not serialized by ``to_json``.
    style : dict, optional
        Style identity + optional overrides for this series: ``series_index``
        (drives categorical color), ``legend`` to force a legend entry on a
        non-data trace, ``lw`` line width, and optional ``color`` / ``marker``
        user overrides (see ``Instrument._data_trace_style`` and
        ``notes/gui_todo.txt``).  ``role`` still implies everything derivable
        (data = markers, model = line); this carries only identity/overrides.
        Serialized by ``to_json`` when present.
    alpha : float, optional
        Per-trace opacity override.  ``None`` means "use the role's default",
        which is where the two renderers had silently disagreed: the saved
        PDF drew data at 0.6 while the GUI drew it fully opaque (review
        4.11.6).  The default now lives in one place per renderer and is the
        PDF's, so the GUI matches what gets published.
    """

    name: str
    role: str
    kind: str
    x: Any
    y: Any
    yerr: Optional[Any] = None
    xerr: Optional[Any] = None
    node: Any = field(default=None, repr=False, compare=False)
    style: Optional[dict] = None
    alpha: Optional[float] = None

    def to_json(self) -> dict:
        d = {
            "name": self.name,
            "role": self.role,
            "kind": self.kind,
            "x": _array_to_list(self.x),
            "y": _array_to_list(self.y),
        }
        if self.yerr is not None:
            d["yerr"] = _array_to_list(self.yerr)
        if self.xerr is not None:
            d["xerr"] = _array_to_list(self.xerr)
        if self.style is not None:
            d["style"] = _jsonify(self.style)
        if self.alpha is not None:
            d["alpha"] = float(self.alpha)
        return d


@dataclass
class Chart:
    """A single GUI-renderable chart: data plus optional model curves.

    Parameters
    ----------
    id : str
        Stable identifier, unique within a system (used as a GUI key).
    component : dict
        ``{"yaml_key": ..., "instance": ...}`` naming the owning
        component and, where relevant, its instance.
    title, xlabel, ylabel : str
        Human-facing chart labels.
    traces : list[Trace]
        The data/model curves to draw.
    param_deps : list[str]
        Parameter paths (``system.plot_params`` labels, plus any
        declared cross-component deps) whose change affects this chart's
        model traces.  A GUI highlights the affected charts when a
        slider moves.  Empty for data-only charts.
    x_range, y_range : list[float], optional
        Explicit ``[lo, hi]`` axis windows.
    x_log, y_log : bool
        Logarithmic axes.
    x_inverted, y_inverted : bool
        Reversed axes -- magnitudes, and RA increasing to the left.
    meta : dict
        Free-form ANNOTATIONS, and nothing a renderer needs to lay out an
        axis: ``caption`` (the LaTeX figure caption the generated paper
        draft, ``outputs/modeling.py``, pairs with this chart's saved PDF),
        ``phase_folded``, ``file_tag``, ``md5``, ``url``, ``size``,
        ``dynamic_data``.

        THE SIX AXIS-GEOMETRY FIELDS ABOVE USED TO LIVE HERE (review
        4.11.3).  They were promoted because they are not annotations: both
        renderers have to consult them to lay out an axis at all, so a typo
        in a ``meta`` key silently produced a linear axis where a log one
        was meant, with nothing to catch it -- a stringly-typed field that
        two independent renderers must agree on is a contract, and a
        contract belongs in the dataclass.  ``aspect_equal`` and
        ``hline_y`` deliberately STAYED: each is honored by exactly one
        renderer, so promoting them would advertise a field the other
        silently ignores.  Promote them if and when both read them.
    """

    id: str
    component: dict
    title: str
    xlabel: str
    ylabel: str
    traces: list = field(default_factory=list)
    param_deps: list = field(default_factory=list)
    x_range: Optional[Any] = None
    y_range: Optional[Any] = None
    x_log: bool = False
    y_log: bool = False
    x_inverted: bool = False
    y_inverted: bool = False
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = {
            "id": self.id,
            "component": _jsonify(self.component),
            "title": self.title,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "traces": [t.to_json() for t in self.traces],
            "param_deps": list(self.param_deps),
            "meta": _jsonify(self.meta),
        }
        # Emitted only when set, the same way `yerr` and `style` are: the
        # default state of an axis is "whatever the renderer would do
        # anyway", so a payload full of nulls and falses would be noise.
        if self.x_range is not None:
            d["x_range"] = _array_to_list(self.x_range)
        if self.y_range is not None:
            d["y_range"] = _array_to_list(self.y_range)
        for name in ("x_log", "y_log", "x_inverted", "y_inverted"):
            if getattr(self, name):
                d[name] = True
        return d
