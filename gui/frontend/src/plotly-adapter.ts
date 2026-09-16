// Chart -> plotly translation. The single place that maps the backend's
// trace roles to visual encodings -- the GUI counterpart of
// src/exozippy/plotrender.py, which renders the SAME specs with matplotlib
// for the saved PDFs. The two must stay visually equivalent: when you extend
// the meta or style vocabulary in one, mirror it in the other.
//
// One meta key is deliberately ignored by BOTH renderers: meta.caption is
// the LaTeX figure caption for the generated paper draft
// (src/exozippy/outputs/modeling.py, the specs' third consumer). If the
// GUI ever surfaces it (e.g. as a card subtitle), strip the LaTeX first.
//
// Plots render on WHITE cards (matching the saved figures), even though the
// surrounding UI is dark -- every color here is chosen for a white background.

import type { Chart, Trace } from "./chart";

// matplotlib's default categorical cycle (C0..C9, tab10). A trace's
// style.series_index picks from it by FIXED index -- assigned once per
// instrument at load, never re-cycled per chart -- so an instrument keeps
// its color across every panel, exactly as in the PDFs.
//
// THIS BLOCK IS A COPY OF src/exozippy/plot_theme.py. Do not hand-edit it:
// tests/test_plot_theme.py parses this file and fails if it drifts from the
// Python table, which is the source of truth (review 4.11.4). The GUI also
// serves the same table at GET /api/theme for any consumer that wants it at
// runtime.
//
// It is a compiled-in copy rather than a fetch on purpose. The reason
// endpoint-as-source was rejected for the CLI -- a renderer must not need a
// server to know what color a residual is -- applies to the browser too: a
// first paint blocked on a network round-trip is worse than a copy a test
// keeps honest.
//
// The residual color moved here (#6e7781 -> #808080, matplotlib's "0.5") when
// the tables were unified: the two renderers had genuinely disagreed, and JDE
// ruled 2026-08-27 "match the pdf".
const TAB10 = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
];

const MODEL_COLOR = "#ff0000";
const DATA_COLOR = "#24292f";
const RESIDUAL_COLOR = "#808080";

interface TraceStyle {
  series_index?: number;
  color?: string;
  marker?: string;
  lw?: number;
  legend?: boolean;
}

function styleOf(trace: Trace): TraceStyle {
  return ((trace as unknown as { style?: TraceStyle }).style || {}) as TraceStyle;
}

function traceColor(trace: Trace): string {
  const style = styleOf(trace);
  if (style.color) return style.color;
  if (style.series_index != null) return TAB10[style.series_index % TAB10.length];
  if (trace.role === "model") return MODEL_COLOR;
  if (trace.role === "residual") return RESIDUAL_COLOR;
  return DATA_COLOR;
}

// matplotlib marker codes -> plotly symbol/size (mirroring plotrender.py's
// use of the style.marker override; "." is the small dense-photometry dot).
function markerFor(code: string | undefined): { symbol: string; size: number } {
  if (code === ".") return { symbol: "circle", size: 3 };
  if (code === "*") return { symbol: "star", size: 10 };
  return { symbol: "circle", size: 5 };
}

function errorBar(arr: unknown, color: string): Record<string, unknown> {
  // Symmetric (N,) or asymmetric (2, N) errors, as chart serializes them.
  const isAsym = Array.isArray(arr) && Array.isArray((arr as unknown[])[0]);
  return isAsym
    ? {
        type: "data",
        symmetric: false,
        arrayminus: (arr as number[][])[0],
        array: (arr as number[][])[1],
        visible: true,
        color,
        thickness: 1,
        width: 0,
      }
    : {
        type: "data",
        array: arr,
        visible: true,
        color,
        thickness: 1,
        width: 0,
      };
}

// Role opacity defaults, mirroring plotrender.py's _DATA_ALPHA. These were
// the two renderers' one silent disagreement (review 4.11.6): the saved PDF
// drew data at 0.6 while this drew it fully opaque, so the same fit looked
// different depending on where you viewed it. JDE ruled "match the pdf", so
// the PDF's values are canonical here.
const DATA_ALPHA = 0.6; // plot_theme.ROLE_ALPHA["data"]

/** The trace's own alpha if it set one, else its role default. */
function traceAlpha(trace: Trace): number {
  if (trace.alpha != null) return trace.alpha;
  return trace.role === "model" ? 1.0 : DATA_ALPHA;
}

/** Convert one Chart trace into a plotly trace object. */
export function traceToPlotly(trace: Trace): Record<string, unknown> {
  const style = styleOf(trace);
  const color = traceColor(trace);
  // Model curves are lines unless the spec says scatter (e.g. a model
  // sampled only at the observation epochs); data/residuals are markers.
  const isLine = trace.role === "model" && trace.kind !== "scatter";

  const out: Record<string, unknown> = {
    name: trace.name,
    x: trace.x,
    y: trace.y,
    // SVG scatter, not "scattergl": the native pywebview window (QtWebEngine)
    // runs with software GL over X/remote displays, where WebGL is unavailable
    // ("WebGL is not supported by your browser") and every plot came up blank.
    // These plots have modest point counts, so SVG rendering is plenty fast and
    // works in every environment. Revisit only if a trace pushes >~10k points.
    type: "scatter",
    mode: isLine ? "lines" : "markers",
    // Legend parity with the PDFs: data traces are labeled; model traces
    // only when the spec asks (style.legend).
    showlegend: trace.role === "data" ? true : Boolean(style.legend),
    opacity: traceAlpha(trace),
  };

  if (isLine) {
    // matplotlib lw is points; plotly width is px (1 pt = 4/3 px).
    out.line = { color, width: (style.lw ?? 1.5) * (4 / 3) };
  } else {
    const mark = markerFor(style.marker);
    out.marker = { color, symbol: mark.symbol, size: mark.size };
    const t = trace as unknown as { yerr?: unknown; xerr?: unknown };
    if (t.yerr != null) out.error_y = errorBar(t.yerr, color);
    if (t.xerr != null) out.error_x = errorBar(t.xerr, color);
  }
  return out;
}

/** plotly layout for a Chart: a white figure card, like the saved plots. */
export function specToLayout(spec: Chart): Record<string, unknown> {
  // Axis scaling comes from the Chart's first-class geometry fields (review
  // 4.11.3; they were `meta` keys until then). The SED, for one, needs a log
  // wavelength axis and an inverted magnitude axis -- without honoring these
  // the points collapse to what looks like a flat line. `meta` is still read
  // for aspect_equal, which only this renderer honors. See plotrender.py for
  // the full shared vocabulary.
  const meta = (spec.meta || {}) as Record<string, unknown>;
  const xaxis: Record<string, unknown> = {
    title: { text: spec.xlabel },
    gridcolor: "#d8dee4",
    zerolinecolor: "#d8dee4",
    linecolor: "#57606a",
  };
  const yaxis: Record<string, unknown> = {
    title: { text: spec.ylabel },
    gridcolor: "#d8dee4",
    zerolinecolor: "#d8dee4",
    linecolor: "#57606a",
  };
  if (spec.x_log) xaxis.type = "log";
  if (spec.y_log) yaxis.type = "log";
  if (spec.x_inverted) xaxis.autorange = "reversed";
  if (spec.y_inverted) yaxis.autorange = "reversed";
  // Explicit [lo, hi] windows (e.g. the SED focuses on the observed data rather
  // than autoranging to the model's numerically-tiny spectral tails). On a log
  // axis plotly expects the range endpoints in log10 units. A reversed axis
  // wants its range descending.
  if (Array.isArray(spec.x_range)) {
    let r = spec.x_range as number[];
    if (spec.x_log) r = r.map((v) => Math.log10(v));
    xaxis.range = spec.x_inverted ? [r[1], r[0]] : r;
    if (spec.x_inverted) delete xaxis.autorange;
  }
  if (Array.isArray(spec.y_range)) {
    let r = spec.y_range as number[];
    if (spec.y_log) r = r.map((v) => Math.log10(v));
    yaxis.range = spec.y_inverted ? [r[1], r[0]] : r;
    if (spec.y_inverted) delete yaxis.autorange;
  }
  // Sky-plane plots (astrometry) need x and y in the same physical scale.
  if (meta.aspect_equal) {
    yaxis.scaleanchor = "x";
    yaxis.scaleratio = 1;
  }

  const layout: Record<string, unknown> = {
    title: { text: spec.title, font: { color: "#24292f", size: 15 } },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: { color: "#24292f" },
    margin: { l: 60, r: 20, t: 40, b: 50 },
    xaxis,
    yaxis,
    legend: { orientation: "h", y: -0.2 },
    // Persist user pan/zoom (and legend toggles) across the Plotly.react calls
    // that a slider drag triggers: while uirevision is unchanged, Plotly keeps
    // the user's view and lets it override the supplied axis ranges. Keyed by
    // the plot id so it stays constant for live model-trace updates but a
    // genuinely new plot (different id) still starts from its default view.
    uirevision: spec.id,
  };

  // Dotted reference line (e.g. zero in phased-RV panels), as in the PDFs.
  if (meta.hline_y != null) {
    layout.shapes = [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: meta.hline_y,
        y1: meta.hline_y,
        line: { color: "#24292f", width: 1, dash: "dot" },
        opacity: 0.5,
      },
    ];
  }
  return layout;
}

/** Full (data, layout) pair ready for Plotly.react. */
export function specToPlotly(spec: Chart): {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
} {
  // Draw data first, models last, so the model curves sit above the data
  // markers -- matching the renderer's zorder (plotly draws in array order).
  const models = spec.traces.filter((t) => t.role === "model");
  const rest = spec.traces.filter((t) => t.role !== "model");
  const data = [...rest, ...models].map((t) => traceToPlotly(t));
  const layout = specToLayout(spec);
  layout.showlegend = data.some((d) => d.showlegend);
  return { data, layout };
}

export const PLOTLY_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};
