// TypeScript mirror of the PlotSpec contract emitted by
// src/exozippy/plotspec.py (G4). The GUI consumes these JSON payloads and
// renders them with plotly; later prompts (G9/G10) re-render model traces as
// sliders move. Keep this in sync with plotspec.py's to_json() output.

export type TraceRole = "data" | "model" | "residual";
export type TraceKind = "scatter" | "line";

export interface Trace {
  name: string;
  role: TraceRole;
  kind: TraceKind;
  x: number[];
  y: number[];
  yerr?: number[] | null;
}

export interface PlotSpec {
  id: string;
  component: string; // "<yaml_key>.<instance>"
  title: string;
  xlabel: string;
  ylabel: string;
  traces: Trace[];
  param_deps: string[];
  meta?: Record<string, unknown>;
}
