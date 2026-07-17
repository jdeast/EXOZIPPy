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
    type: "scattergl",
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
  return {
    title: { text: spec.title, font: { color: "#e6e6e6", size: 15 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#c9d1d9" },
    margin: { l: 60, r: 20, t: 40, b: 50 },
    xaxis: { title: { text: spec.xlabel }, gridcolor: "#30363d", zerolinecolor: "#30363d" },
    yaxis: { title: { text: spec.ylabel }, gridcolor: "#30363d", zerolinecolor: "#30363d" },
    legend: { orientation: "h", y: -0.2 },
    showlegend: spec.traces.length > 1,
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
