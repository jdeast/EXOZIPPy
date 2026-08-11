// TypeScript mirror of the PlotSpec contract emitted by
// src/exozippy/plotspec.py (G4). The GUI consumes these JSON payloads and
// renders them with plotly; later prompts (G9/G10) re-render model traces as
// sliders move. Keep this in sync with plotspec.py's to_json() output.

export type TraceRole = "data" | "model" | "residual";
export type TraceKind = "scatter" | "line";

// Style identity + optional overrides (mirrors plotspec.py's Trace.style):
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
}

export interface PlotSpec {
  id: string;
  // Emitted by plotspec.py as {yaml_key, instance}; older specs used a string.
  component: { yaml_key: string; instance: string | null } | string;
  title: string;
  xlabel: string;
  ylabel: string;
  traces: Trace[];
  param_deps: string[];
  meta?: Record<string, unknown>;
}
