"""The one style table both chart renderers read (review 4.11.4).

There are two renderers for the same :class:`~exozippy.chart.Chart` -- this
package's ``plotrender.py`` (matplotlib, the saved PDFs) and the GUI's
``plotly-adapter.ts`` (plotly, the browser). Until this module existed each
carried its own copy of the palette and the role colors, and they agreed **by
convention only**. Measured 2026-08-27, that convention had already broken in
one place: the residual color was matplotlib's ``"0.5"`` (``#808080``) in the
PDF and ``#6e7781`` in the GUI, so the same fit looked different depending on
where you viewed it.

WHY THIS LIVES IN THE CORE PACKAGE AND NOT BEHIND A GUI ENDPOINT. The obvious
alternative -- serve the theme from the GUI server and let both sides fetch it
-- was considered and REJECTED, because it would make the CLI depend on the
GUI server: ``exozippy <config>`` writes PDFs with no browser and no FastAPI
process anywhere. So the source of truth is this module, which imports nothing
beyond the standard library; the GUI *serves a copy* of it (``GET
/api/theme``) rather than owning it.

THE VALUES ARE MATPLOTLIB'S, and deliberately: they are what gets published.
JDE ruled 2026-08-27 "match the pdf". Each was verified equal to the
matplotlib spelling it replaces, so the saved PDFs do not move:

    PALETTE == matplotlib's default ``axes.prop_cycle``  (it IS tab10, so the
              old ``f"C{n}"`` mapping resolves to exactly these hex values)
    "#ff0000" == matplotlib "r"      (model curves)
    "#808080" == matplotlib "0.5"    (residuals -- the one the GUI moves to)

Adding a role, a marker or a palette entry here changes BOTH renderers, which
is the point. ``tests/test_plot_theme.py`` pins the equivalences above, so a
future matplotlib whose default cycle is not tab10 fails loudly rather than
silently re-coloring every published figure.
"""

from __future__ import annotations

#: Categorical series colors, indexed by ``Trace.style["series_index"]``.
#: Identical to matplotlib's default property cycle (tab10). The index is
#: assigned once per instrument at load and never re-cycled per chart, so an
#: instrument keeps its color across every panel.
PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

#: Fallback color per trace role, used when a trace carries no
#: ``series_index`` and no explicit override.
ROLE_COLORS = {
    "model": "#ff0000",
    "residual": "#808080",
    "data": "#24292f",
}

#: Opacity per role. Data and residuals are drawn semi-transparent so
#: overlapping points remain readable; model curves are opaque.
#: ``Trace.alpha`` overrides these per trace (review 4.11.6).
ROLE_ALPHA = {
    "model": 1.0,
    "residual": 0.6,
    "data": 0.6,
}

#: Default line width, in points (plotly wants px: 1 pt = 4/3 px).
DEFAULT_LINEWIDTH = 1.5

#: matplotlib marker code -> (plotly symbol, plotly size). The "." case is
#: the small dot used for dense photometry.
MARKERS = {
    ".": ("circle", 3),
    "*": ("star", 10),
}

#: Symbol/size for any marker code not in MARKERS.
DEFAULT_MARKER = ("circle", 5)


def resolve(role, series_index=None, overrides=None):
    """The style for one trace: ``(role, series_index, overrides) -> dict``.

    Precedence, highest first -- and the first rung is the point of the
    function: a user's ``plot: {color: ..., marker: ...}`` in the config wins
    over everything this module would otherwise choose.

    1. ``overrides`` (the trace's ``style`` dict: ``color``, ``marker``,
       ``lw``, and ``alpha`` if the caller merged one in),
    2. ``series_index`` -> ``PALETTE`` (categorical identity),
    3. ``role`` -> ``ROLE_COLORS`` / ``ROLE_ALPHA``.

    Returns a dict with ``color``, ``alpha``, ``marker`` and ``lw``. Callers
    translate to their own renderer's spelling; nothing here imports either
    renderer.
    """
    overrides = overrides or {}

    color = overrides.get("color")
    if color is None:
        if series_index is not None:
            color = PALETTE[int(series_index) % len(PALETTE)]
        else:
            color = ROLE_COLORS.get(role, ROLE_COLORS["data"])

    alpha = overrides.get("alpha")
    if alpha is None:
        alpha = ROLE_ALPHA.get(role, ROLE_ALPHA["data"])

    return {
        "color": color,
        "alpha": float(alpha),
        "marker": overrides.get("marker"),
        "lw": float(overrides.get("lw", DEFAULT_LINEWIDTH)),
    }


def marker_symbol(code):
    """``(plotly_symbol, size)`` for a matplotlib marker code."""
    return MARKERS.get(code, DEFAULT_MARKER)


def as_json():
    """The whole table, JSON-serializable -- what the GUI serves as a copy.

    A copy, never the source: see this module's header for why the CLI must
    not have to ask a server what color a residual is.
    """
    return {
        "palette": list(PALETTE),
        "role_colors": dict(ROLE_COLORS),
        "role_alpha": dict(ROLE_ALPHA),
        "default_linewidth": DEFAULT_LINEWIDTH,
        "markers": {k: list(v) for k, v in MARKERS.items()},
        "default_marker": list(DEFAULT_MARKER),
    }
