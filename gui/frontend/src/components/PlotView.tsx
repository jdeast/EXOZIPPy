import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { PlotSpec } from "../plotspec";
import { specToPlotly, PLOTLY_CONFIG } from "../plotly-adapter";

// Thin React wrapper over plotly.js-dist-min. Uses Plotly.react so repeated
// renders (slider moves in G10) patch the existing chart instead of tearing it
// down. No react-plotly.js dependency -- keeps the bundle lean and typing simple.
export default function PlotView({ spec }: { spec: PlotSpec }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { data, layout } = specToPlotly(spec);
    Plotly.react(el, data, layout, PLOTLY_CONFIG);
    return () => {
      if (el) Plotly.purge(el);
    };
  }, [spec]);

  return <div className="plot-view" ref={ref} />;
}
