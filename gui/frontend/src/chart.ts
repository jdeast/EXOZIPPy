// TypeScript mirror of the Chart contract emitted by
// src/exozippy/chart.py (G4). The GUI consumes these JSON payloads and
// renders them with plotly; later prompts (G9/G10) re-render model traces as
// sliders move. Keep this in sync with chart.py's to_json() output.

export type TraceRole = "data" | "model" | "residual";
export type TraceKind = "scatter" | "line";

// Style identity + optional overrides (mirrors chart.py's Trace.style):
// series_index drives the fixed categorical color; color/marker/lw are user
// or component overrides; legend forces a legend entry on a model trace.
export interface TraceStyle {
  series_index?: number;
  color?: string;
  marker?: string;
  lw?: number;
  legend?: boolean;
}

export interface Trace {
  name: string;
  role: TraceRole;
  kind: TraceKind;
  x: number[];
  y: number[];
  // Symmetric (N,) or asymmetric (2, N) errors.
  yerr?: number[] | number[][] | null;
  xerr?: number[] | number[][] | null;
  style?: TraceStyle | null;
  // Per-trace opacity override; absent means "use the role default", which
  // must match plotrender.py's (review 4.11.6 -- the two renderers had
  // silently disagreed, the PDF drawing data at 0.6 and this one opaque).
  alpha?: number | null;
}

export interface Chart {
  id: string;
  // Emitted by chart.py as {yaml_key, instance}; older specs used a string.
  component: { yaml_key: string; instance: string | null } | string;
  title: string;
  xlabel: string;
  ylabel: string;
  traces: Trace[];
  param_deps: string[];
  // Axis geometry: first-class fields since review 4.11.3, emitted by
  // chart.py only when set. They used to be stringly-typed `meta` keys, but
  // both renderers must consult them to lay out an axis at all, so a typo
  // silently produced a linear axis where a log one was meant. `meta` now
  // carries annotations only (caption, file_tag, aspect_equal, ...).
  x_range?: number[] | null;
  y_range?: number[] | null;
  x_log?: boolean;
  y_log?: boolean;
  x_inverted?: boolean;
  y_inverted?: boolean;
  meta?: Record<string, unknown>;
}
