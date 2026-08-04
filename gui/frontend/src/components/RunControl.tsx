import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ProjectListing, type RunStatus } from "../api";

// Pinned run control (bottom-right of the workspace), replacing the old Run
// tab. "Run fit" runs THE CONFIGURATION ON SCREEN: it saves the open document
// first (the fit subprocess reads from disk), so there is never a stale-file
// ambiguity about what is being fit. While a run is active it shows the
// sampler's progress (draws / rhat / ess from the snapshot the samplers emit
// at each convergence check) and a Stop button; the run's log is attached to
// the bottom terminal.

interface Props {
  configPath: string | null;
  listing: ProjectListing | null;
  setLogFile: (file: string | null) => void;
}

export default function RunControl({ configPath, listing, setLogFile }: Props) {
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.runStatus());
    } catch {
      /* server briefly unavailable; keep the last status */
    }
  }, []);

  useEffect(() => {
    refresh();
    pollRef.current = window.setInterval(refresh, 2000);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [refresh]);

  const active = Boolean(status?.active);

  const run = async () => {
    if (!configPath || !listing) return;
    setBusy(true);
    setError(null);
    try {
      // Run what you see: flush the document's unsaved edits to disk first.
      // (No open doc -- e.g. a config never opened in the Config tab -- just
      // runs the file as it is on disk.)
      try {
        await api.docSave();
      } catch {
        /* no document open */
      }
      const st = await api.startRun(configPath, listing.dir);
      setStatus(st);
      if (st.log_path) setLogFile(st.log_path);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    try {
      setStatus(await api.stopRun(false));
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  };

  const snap = status?.snapshot;
  const state = status?.state;
  const draws = snap?.n_draws ?? state?.n_draws;
  const rhat = snap?.max_rhat ?? state?.max_rhat;
  const ess = snap?.min_ess ?? state?.min_ess;

  return (
    <div className="run-control">
      {error && <span className="run-control-error" title={error}>run failed</span>}
      {active ? (
        <>
          <span className="run-control-status">
            {status?.phase || "running"}
            {draws != null ? ` -- ${draws} draws` : ""}
            {rhat != null ? ` -- rhat ${Number(rhat).toFixed(3)}` : ""}
            {ess != null ? ` -- ess ${Math.round(Number(ess))}` : ""}
          </span>
          <button className="run-control-stop" onClick={stop}>
            Stop
          </button>
        </>
      ) : (
        <button
          className="run-control-run"
          disabled={!configPath || !listing || busy}
          title={
            configPath
              ? "Save the current configuration and start the fit"
              : "Open a project first"
          }
          onClick={run}
        >
          {busy ? "Starting..." : "Run fit"}
        </button>
      )}
    </div>
  );
}
