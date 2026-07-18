import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { PlotSpec } from "../plotspec";
import { specToPlotly, PLOTLY_CONFIG } from "../plotly-adapter";

// Thin React wrapper over plotly.js-dist-min. Uses Plotly.react so repeated
// renders (slider moves in G10) patch the existing chart in place instead of
// tearing it down -- which, combined with the layout's uirevision, preserves
// the user's pan/zoom across redraws. No react-plotly.js dependency -- keeps the
// bundle lean and typing simple.
export default function PlotView({ spec }: { spec: PlotSpec }) {
  const ref = useRef<HTMLDivElement>(null);

  // Patch the chart in place on every spec change. Crucially this effect does
  // NOT purge on cleanup: purging destroys the DOM plot (and its zoom state) on
  // each update, which would defeat Plotly.react + uirevision.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { data, layout } = specToPlotly(spec);
    Plotly.react(el, data, layout, PLOTLY_CONFIG);
  }, [spec]);

  // Tear the chart down only when the component actually unmounts.
  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div className="plot-view" ref={ref} />;
}
