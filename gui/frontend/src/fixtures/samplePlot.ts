import type { PlotSpec } from "../plotspec";

// A canned PlotSpec used by the Welcome tab's demo panel to exercise the
// PlotSpec -> plotly adapter without a running fit. Shape matches
// src/exozippy/plotspec.py's to_json(). Values are synthetic (a noisy sine
// "RV" plus a smooth "model" curve) purely to show data + model roles render.
function build(): PlotSpec {
  const n = 60;
  const x: number[] = [];
  const yData: number[] = [];
  const yErr: number[] = [];
  const xModel: number[] = [];
  const yModel: number[] = [];
  const k = 42; // amplitude, m/s
  for (let i = 0; i < n; i++) {
    const phase = i / (n - 1);
    x.push(phase);
    const clean = k * Math.sin(2 * Math.PI * phase);
    // deterministic pseudo-noise so the demo is stable across renders
    const noise = k * 0.12 * Math.sin(37.0 * phase + 1.3);
    yData.push(clean + noise);
    yErr.push(k * 0.1);
  }
  const m = 200;
  for (let i = 0; i < m; i++) {
    const phase = i / (m - 1);
    xModel.push(phase);
    yModel.push(k * Math.sin(2 * Math.PI * phase));
  }
  return {
    id: "demo.rv.phase",
    component: "rvinstrument.demo",
    title: "Demo: phase-folded radial velocity",
    xlabel: "orbital phase",
    ylabel: "RV (m/s)",
    traces: [
      { name: "observed", role: "data", kind: "scatter", x, y: yData, yerr: yErr },
      { name: "model", role: "model", kind: "line", x: xModel, y: yModel },
    ],
    param_deps: ["orbit.b.period", "orbit.b.K"],
    meta: { phase_folded: true, demo: true },
  };
}

export const SAMPLE_PLOT: PlotSpec = build();
