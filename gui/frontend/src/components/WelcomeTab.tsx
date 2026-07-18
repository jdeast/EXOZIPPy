import { useEffect, useState } from "react";
import logo from "../assets/exozippy_logo.svg";
import PlotView from "./PlotView";
import { api, type PlotSpec } from "../api";

// The landing tab. With no project open it shows the logo + a backend health
// check. With a project open it renders data-only plots (no fit required) of
// every data-bearing component in the loaded config -- the real "what's in this
// project" overview -- and points the user to the Tune tab, where the same
// plots gain live sliders.

interface Props {
  configPath: string | null;
  setActiveTab: (id: string) => void;
}

// A component is worth previewing on the overview if its schema declares a
// datafile config key (rvinstrument, transit, sed, ...). Purely-parametric
// components (star, orbit) have no data to draw here. Schema-driven, so no
// component name is hardcoded.
function dataBearingKeys(schema: any): Set<string> {
  const out = new Set<string>();
  const comps = (schema && schema.components) || {};
  for (const [key, cs] of Object.entries<any>(comps)) {
    const cfg = (cs && cs.config) || [];
    if (Array.isArray(cfg) && cfg.some((k: any) => k && k.kind === "datafile")) {
      out.add(key);
    }
  }
  return out;
}

interface ComponentPreview {
  compType: string;
  specs: PlotSpec[];
  error: string | null;
}

export default function WelcomeTab({ configPath, setActiveTab }: Props) {
  const [componentCount, setComponentCount] = useState<number | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [previews, setPreviews] = useState<ComponentPreview[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api
      .schema()
      .then((s) => {
        const comps = (s.components || {}) as Record<string, unknown>;
        setComponentCount(Object.keys(comps).length);
      })
      .catch((e) => setSchemaError(String(e)));
  }, []);

  // When a project's config is open, build data-only previews of each
  // data-bearing component in it. Each preview loads the component's data in a
  // worker subprocess (no fit), so this is safe before any Solve/Run.
  useEffect(() => {
    if (!configPath) {
      setPreviews(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setPreviews([]);
    (async () => {
      try {
        const schema = await api.schema();
        const dataKeys = dataBearingKeys(schema);
        // Opening the doc makes the config+params available to the preview
        // endpoint. It is idempotent for a given path (the Config tab may also
        // have opened the same file).
        const doc = await api.docOpen(configPath);
        const present = Object.keys(doc.config || {}).filter((k) =>
          dataKeys.has(k),
        );
        // Each preview builds a System in a worker subprocess (seconds each),
        // so append results as they arrive rather than blocking on the whole
        // set -- the first plot shows in ~seconds, not after every component.
        for (const compType of present) {
          if (cancelled) return;
          let entry: ComponentPreview | null = null;
          try {
            const res = await api.preview(compType);
            if (res.error) entry = { compType, specs: [], error: res.error };
            else if (res.specs && res.specs.length)
              entry = { compType, specs: res.specs, error: null };
          } catch (e) {
            entry = { compType, specs: [], error: String(e) };
          }
          if (cancelled) return;
          if (entry) setPreviews((prev) => [...(prev || []), entry as ComponentPreview]);
        }
      } catch (e) {
        if (!cancelled) setPreviews([{ compType: "", specs: [], error: String(e) }]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [configPath]);

  const hasPlots = previews && previews.some((p) => p.specs.length);

  return (
    <div className="welcome">
      <div className="welcome-hero">
        <img className="logo" src={logo} alt="EXOZIPPy logo" />
        <div>
          <h1>EXOZIPPy</h1>
          <p className="muted">
            A component-agnostic modeling GUI. Open a project from the sidebar to
            load its data; use the Tune tab to Solve and drag parameters with the
            plots updating live.
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

      {!configPath && (
        <section className="demo-panel">
          <p className="muted">
            No project open. Enter a project directory in the sidebar (or pass one
            on the command line) to see its data plotted here.
          </p>
        </section>
      )}

      {configPath && (
        <section className="welcome-previews">
          <div className="welcome-previews-head">
            <h2>Loaded data</h2>
            <button className="link-button" onClick={() => setActiveTab("tune")}>
              Tune these parameters with live sliders -&gt;
            </button>
          </div>
          {loading && <p className="muted">Loading component data...</p>}
          {!loading && previews && !hasPlots && (
            <p className="muted">
              This config has no plottable data components, or their data files
              could not be loaded. Check the sidebar and the Config tab.
            </p>
          )}
          {previews &&
            previews.map((p) =>
              p.specs.length ? (
                <div key={p.compType} className="welcome-preview">
                  <h3>{p.compType}</h3>
                  {p.specs.map((spec, i) => (
                    <PlotView key={i} spec={spec} />
                  ))}
                </div>
              ) : p.error ? (
                <div key={p.compType || "err"} className="welcome-preview">
                  <h3>{p.compType || "preview"}</h3>
                  <p className="sidebar-error">{p.error}</p>
                </div>
              ) : null,
            )}
        </section>
      )}
    </div>
  );
}
