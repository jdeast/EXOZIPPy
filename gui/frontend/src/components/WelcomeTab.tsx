import { useEffect, useState } from "react";
import logo from "../assets/exozippy_logo.svg";
import PlotView from "./PlotView";
import { SAMPLE_PLOT } from "../fixtures/samplePlot";
import { api } from "../api";

// The initial workspace tab. Shows the logo, a live backend health/schema
// check (proves the API wiring), and a demo plot fed by a canned PlotSpec
// fixture (proves the PlotSpec -> plotly adapter). Later prompts add real tabs
// (Config, Data, Tune, Run) alongside this one.
export default function WelcomeTab() {
  const [componentCount, setComponentCount] = useState<number | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);

  useEffect(() => {
    api
      .schema()
      .then((s) => {
        const comps = (s.components || {}) as Record<string, unknown>;
        setComponentCount(Object.keys(comps).length);
      })
      .catch((e) => setSchemaError(String(e)));
  }, []);

  return (
    <div className="welcome">
      <div className="welcome-hero">
        <img className="logo" src={logo} alt="EXOZIPPy logo" />
        <div>
          <h1>EXOZIPPy</h1>
          <p className="muted">
            A component-agnostic modeling GUI. Open a project from the sidebar to
            begin; editing, tuning, and running arrive in later builds.
          </p>
          <p className="backend-status">
            {schemaError
              ? `backend error: ${schemaError}`
              : componentCount === null
                ? "checking backend..."
                : `backend connected -- ${componentCount} components discovered`}
          </p>
        </div>
      </div>

      <section className="demo-panel">
        <h2>PlotSpec adapter demo</h2>
        <p className="muted">
          Rendered from a canned PlotSpec fixture (no fit required). Data points
          show markers with error bars; the model shows a line -- the same
          adapter every plotting tab uses.
        </p>
        <PlotView spec={SAMPLE_PLOT} />
      </section>
    </div>
  );
}
