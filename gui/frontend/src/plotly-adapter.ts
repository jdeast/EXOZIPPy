// PlotSpec -> plotly translation. The single place that maps the backend's
// trace roles to visual encodings, so every tab (preview in G9, Tune plots in
// G10, results in G13) renders identically. Data = markers with error bars;
// model = line; residual = markers around zero.

import type { PlotSpec, Trace } from "./plotspec";

// A theme-friendly categorical palette for model curves (data stays neutral).
const MODEL_COLORS = [
  "#6cb6ff", "#f2a65a", "#7ee787", "#d2a8ff", "#ff7b72", "#79c0ff",
];

function traceColor(trace: Trace, index: number): string {
  if (trace.role === "data") return "#c9d1d9";
  if (trace.role === "residual") return "#8b949e";
  return MODEL_COLORS[index % MODEL_COLORS.length];
}

/** Convert one PlotSpec trace into a plotly trace object. */
export function traceToPlotly(trace: Trace, index: number): Record<string, unknown> {
  const color = traceColor(trace, index);
  const isLine = trace.kind === "line" || trace.role === "model";

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
  };

  if (isLine) {
    out.line = { color, width: 2 };
  } else {
    out.marker = { color, size: 5 };
    if (trace.yerr && trace.yerr.length === trace.y.length) {
      out.error_y = {
        type: "data",
        array: trace.yerr,
        visible: true,
        color,
        thickness: 1,
        width: 0,
      };
    }
  }
  return out;
}

/** plotly layout for a PlotSpec, styled for a dark UI. */
export function specToLayout(spec: PlotSpec): Record<string, unknown> {
  // Axis scaling comes from the spec's `meta` hints (set by the component's
  // plot_data). The SED, for one, needs a log wavelength axis and an inverted
  // magnitude axis -- without honoring these the points collapse to what looks
  // like a flat line. Keys: x_log/y_log -> logarithmic; x_inverted/y_inverted
  // -> reversed (magnitudes increase downward).
  const meta = (spec.meta || {}) as Record<string, unknown>;
  const xaxis: Record<string, unknown> = {
    title: { text: spec.xlabel },
    gridcolor: "#30363d",
    zerolinecolor: "#30363d",
  };
  const yaxis: Record<string, unknown> = {
    title: { text: spec.ylabel },
    gridcolor: "#30363d",
    zerolinecolor: "#30363d",
  };
  if (meta.x_log) xaxis.type = "log";
  if (meta.y_log) yaxis.type = "log";
  if (meta.x_inverted) xaxis.autorange = "reversed";
  if (meta.y_inverted) yaxis.autorange = "reversed";
  // Explicit [lo, hi] windows (e.g. the SED focuses on the observed data rather
  // than autoranging to the model's numerically-tiny spectral tails). On a log
  // axis plotly expects the range endpoints in log10 units.
  if (Array.isArray(meta.x_range))
    xaxis.range = meta.x_log ? (meta.x_range as number[]).map((v) => Math.log10(v)) : meta.x_range;
  if (Array.isArray(meta.y_range))
    yaxis.range = meta.y_log ? (meta.y_range as number[]).map((v) => Math.log10(v)) : meta.y_range;

  return {
    title: { text: spec.title, font: { color: "#e6e6e6", size: 15 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#c9d1d9" },
    margin: { l: 60, r: 20, t: 40, b: 50 },
    xaxis,
    yaxis,
    legend: { orientation: "h", y: -0.2 },
    showlegend: spec.traces.length > 1,
    // Persist user pan/zoom (and legend toggles) across the Plotly.react calls
    // that a slider drag triggers: while uirevision is unchanged, Plotly keeps
    // the user's view and lets it override the supplied axis ranges. Keyed by
    // the plot id so it stays constant for live model-trace updates but a
    // genuinely new plot (different id) still starts from its default view.
    uirevision: spec.id,
  };
}

/** Full (data, layout) pair ready for Plotly.react. */
export function specToPlotly(spec: PlotSpec): {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
} {
  // Draw data first, models on top; index models independently for coloring.
  let modelIndex = 0;
  const data = spec.traces.map((t) => {
    const idx = t.role === "model" ? modelIndex++ : 0;
    return traceToPlotly(t, idx);
  });
  return { data, layout: specToLayout(spec) };
}

export const PLOTLY_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};
