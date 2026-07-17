import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  runImageUrl,
  type ProjectListing,
  type RunPlots,
  type RunStatus,
} from "../api";

// Phases that mean the run is over -- mirrors the backend TERMINAL_PHASES set.
const TERMINAL = new Set(["done", "stopped", "error"]);

interface RunTabProps {
  listing: ProjectListing | null;
  // Auto-attach the bottom log terminal to the run's log file for its duration.
  setLogFile: (file: string | null) => void;
}

// Tiny inline SVG sparkline of the rhat history collected while polling.
function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <span className="muted">--</span>;
  const w = 120;
  const h = 24;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / span) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="sparkline" width={w} height={h} aria-label="rhat history">
      <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth="1.5" />
    </svg>
  );
}

function fmt(n: number | undefined | null, digits = 3): string {
  if (n === undefined || n === null || Number.isNaN(n)) return "--";
  return typeof n === "number" ? n.toFixed(digits) : String(n);
}

export default function RunTab({ listing, setLogFile }: RunTabProps) {
  const configs = listing?.configs ?? [];
  const params = listing?.params ?? [];

  const [config, setConfig] = useState<string>("");
  const [paramFile, setParamFile] = useState<string>("");
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [plots, setPlots] = useState<RunPlots>({ start: [], progress: [] });
  const [rhatHistory, setRhatHistory] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [stopping, setStopping] = useState(false);
  const pollRef = useRef<number | null>(null);

  // Default the config selection to the first config in the project.
  useEffect(() => {
    if (!config && configs.length) setConfig(configs[0].path);
  }, [configs, config]);

  // G8 (sibling branch) owns document dirty/validation state; until it merges,
  // treat the doc as always ready. TODO(G8): thread real docReady in here so
  // the Run button disables with "unsaved changes" / "validation errors".
  const docReady = true;

  const active = status?.active && !TERMINAL.has(status?.phase ?? "");

  const refreshPlots = useCallback(async () => {
    try {
      setPlots(await api.runPlots());
    } catch {
      /* plots are best-effort; ignore transient failures */
    }
  }, []);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const poll = useCallback(async () => {
    try {
      const st = await api.runStatus();
      setStatus(st);
      if (st.log_path) setLogFile(st.log_path);
      const rhat = st.state?.max_rhat ?? st.snapshot?.max_rhat ?? undefined;
      if (typeof rhat === "number" && !Number.isNaN(rhat)) {
        setRhatHistory((prev) => [...prev.slice(-119), rhat]);
      }
      refreshPlots();
      if (!st.active || TERMINAL.has(st.phase)) {
        stopPolling();
        setStopping(false);
      }
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, [setLogFile, refreshPlots, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    poll();
    pollRef.current = window.setInterval(poll, 1500);
  }, [poll, stopPolling]);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const onRun = async () => {
    if (!config || !listing) return;
    setError(null);
    setBusy(true);
    setRhatHistory([]);
    setPlots({ start: [], progress: [] });
    try {
      // TODO(G8): save (or prompt to save) the edited doc before launching.
      const st = await api.startRun(config, listing.dir, paramFile || null);
      setStatus(st);
      if (st.log_path) setLogFile(st.log_path);
      startPolling();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  const doStop = async (force: boolean) => {
    setConfirming(false);
    setStopping(true);
    try {
      const st = await api.stopRun(force);
      setStatus(st);
      startPolling();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  };

  const disabledReason = !listing
    ? "Open a project first"
    : !config
      ? "Select a config file"
      : !docReady
        ? "Resolve unsaved changes / validation errors"
        : active
          ? "A run is already active"
          : null;

  const state = status?.state ?? {};

  return (
    <div className="run-tab">
      <div className="run-controls">
        <div className="run-selects">
          <label>
            Config
            <select value={config} onChange={(e) => setConfig(e.target.value)}>
              {configs.length === 0 && <option value="">(no config in project)</option>}
              {configs.map((c) => (
                <option key={c.path} value={c.path}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Params (optional)
            <select value={paramFile} onChange={(e) => setParamFile(e.target.value)}>
              <option value="">(none)</option>
              {params.map((p) => (
                <option key={p.path} value={p.path}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="run-buttons">
          <button
            className="run-button"
            onClick={onRun}
            disabled={!!disabledReason || busy}
            title={disabledReason ?? "Start the fit"}
          >
            {busy ? "Starting..." : "Run"}
          </button>
          {active && (
            <button
              className="stop-button"
              onClick={() => setConfirming(true)}
              disabled={stopping && !status?.alive}
            >
              {stopping ? "Stopping..." : "Stop"}
            </button>
          )}
        </div>
        {disabledReason && !active && (
          <p className="run-hint muted">{disabledReason}</p>
        )}
        {error && <p className="run-error">{error}</p>}
      </div>

      {status && status.active && (
        <div className="status-strip">
          <div className="status-cell">
            <span className="status-label">phase</span>
            <span className={`status-phase phase-${status.phase}`}>{status.phase}</span>
          </div>
          <div className="status-cell">
            <span className="status-label">draws</span>
            <span>{state.n_draws ?? "--"}</span>
          </div>
          <div className="status-cell">
            <span className="status-label">max rhat</span>
            <span>{fmt(state.max_rhat)}</span>
          </div>
          <div className="status-cell">
            <span className="status-label">min ess</span>
            <span>{fmt(state.min_ess, 0)}</span>
          </div>
          <div className="status-cell">
            <span className="status-label">elapsed</span>
            <span>{state.elapsed_s ? `${fmt(state.elapsed_s, 0)}s` : "--"}</span>
          </div>
          <div className="status-cell">
            <span className="status-label">rhat</span>
            <Sparkline values={rhatHistory} />
          </div>
        </div>
      )}

      {status && TERMINAL.has(status.phase) && (
        <div className={`run-final phase-${status.phase}`}>
          <strong>Run {status.phase}.</strong>{" "}
          {status.error ? <span>({status.error}) </span> : null}
          {status.results_dir && (
            <span className="results-link" title={status.results_dir}>
              Results: {status.results_dir}
            </span>
          )}
        </div>
      )}

      {plots.start.length > 0 && (
        <section className="plot-gallery">
          <h3>Start plots</h3>
          <div className="gallery-grid">
            {plots.start.map((p) => (
              <a key={p} href={runImageUrl(p)} target="_blank" rel="noreferrer">
                <img src={runImageUrl(p)} alt={p.split("/").pop() || p} />
              </a>
            ))}
          </div>
        </section>
      )}

      {plots.progress.length > 0 && (
        <section className="plot-gallery">
          <h3>Progress</h3>
          <div className="gallery-grid">
            {plots.progress.map((p) => (
              <a key={p} href={runImageUrl(p)} target="_blank" rel="noreferrer">
                <img src={runImageUrl(p)} alt={p.split("/").pop() || p} />
              </a>
            ))}
          </div>
        </section>
      )}

      {confirming && (
        <div className="modal-backdrop" onClick={() => setConfirming(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Stop this fit?</h3>
            <p>This will terminate the current fit and cannot be undone. Are you sure?</p>
            <div className="modal-buttons">
              <button onClick={() => setConfirming(false)}>Cancel</button>
              <button className="stop-button" onClick={() => doStop(false)}>
                Stop (graceful)
              </button>
              {stopping && (
                <button className="force-button" onClick={() => doStop(true)}>
                  Force kill
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
