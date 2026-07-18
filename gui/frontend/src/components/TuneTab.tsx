import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type DocCommand,
  type PlotSpec,
  type TuneParam,
  type TuneResult,
  type TuneStatus,
} from "../api";
import PlotView from "./PlotView";

// Tune tab (G10): the signature interaction. Press Solve -> a worker runs the
// relaxation engine + compiles the forward evaluator; the app then enters LIVE
// mode where dragging a slider re-renders the affected model curves in ms.
//
// Layout: left = searchable/filterable parameter tree grouped by component
// instance; center = the PlotSpec plots (highlighted when they depend on the
// selected parameter); right = a detail panel with slider / bounds / prior /
// fix-free controls. Bound/prior/fix edits are structural and flip the
// evaluator's structural_hash -> a "re-Solve" banner + stale sliders.

const PROV_COLORS: Record<string, string> = {
  user: "#6cb6ff",
  data: "#7ee787",
  solved: "#f2a65a",
  default: "#8b949e",
};

const PROV_HELP: Record<string, string> = {
  user: "from params.yaml (user override)",
  data: "derived from the data",
  solved: "solved by the relaxation engine",
  default: "from the component defaults",
};

// A display path (comp.instance.param) reduced to its plot_params label form
// (comp.param), which is what PlotSpec.param_deps carries.
function labelForm(path: string): string {
  const parts = path.split(".");
  return parts.length === 3 ? `${parts[0]}.${parts[2]}` : path;
}

function instanceKey(path: string): string {
  const parts = path.split(".");
  return parts.length >= 3 ? `${parts[0]}.${parts[1]}` : parts[0];
}

function paramName(path: string): string {
  return path.split(".").pop() || path;
}

export default function TuneTab({ configPath }: { configPath: string | null }) {
  const [status, setStatus] = useState<TuneStatus | null>(null);
  const [result, setResult] = useState<TuneResult | null>(null);
  const [specs, setSpecs] = useState<PlotSpec[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  const [staleReason, setStaleReason] = useState<string | null>(null);

  // Filters
  const [search, setSearch] = useState("");
  const [sampledOnly, setSampledOnly] = useState(false);
  const [userOnly, setUserOnly] = useState(false);
  const [compFilter, setCompFilter] = useState("");

  const pollTimer = useRef<number | null>(null);

  // Ensure a document is open on the server (Solve and the edit commands both
  // read/write it). Opening is idempotent enough for one project.
  const ensureDoc = useCallback(async () => {
    if (!configPath) return;
    try {
      await api.doc();
    } catch {
      await api.docOpen(configPath);
    }
  }, [configPath]);

  // Poll solve status until it leaves the transient phases.
  const startPolling = useCallback(() => {
    if (pollTimer.current) window.clearInterval(pollTimer.current);
    pollTimer.current = window.setInterval(async () => {
      try {
        const st = await api.tuneStatus();
        setStatus(st);
        if (st.phase === "live" && st.has_result) {
          if (pollTimer.current) window.clearInterval(pollTimer.current);
          pollTimer.current = null;
          const res = await api.tuneResult();
          setResult(res);
          setSpecs(res.plots);
          setStale(false);
          setStaleReason(null);
        } else if (st.phase === "error") {
          if (pollTimer.current) window.clearInterval(pollTimer.current);
          pollTimer.current = null;
          setError(st.error || "solve failed");
        }
      } catch (e) {
        setError(String(e instanceof Error ? e.message : e));
      }
    }, 400);
  }, []);

  useEffect(
    () => () => {
      if (pollTimer.current) window.clearInterval(pollTimer.current);
    },
    []
  );

  // Restore from the server-side session on (re)mount. The solve runs in a
  // background thread on the server; its phase/result/hash outlive this
  // component, so switching tabs mid-solve must NOT lose it. We pick an
  // in-flight solve back up, re-hydrate a finished one, or (when idle) prewarm
  // the evaluator worker so the eventual first Solve is faster.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await ensureDoc();
      let st: TuneStatus;
      try {
        st = await api.tuneStatus();
      } catch {
        return; // no session -> idle
      }
      if (cancelled) return;
      setStatus(st);
      if (st.phase === "solving" || st.phase === "compiling") {
        startPolling(); // keep watching the background solve
      } else if (st.phase === "live" && st.has_result) {
        const res = await api.tuneResult();
        if (cancelled) return;
        setResult(res);
        setSpecs(res.plots);
        // A structural edit made elsewhere while we were away may have
        // invalidated the live evaluator; surface the re-Solve banner.
        try {
          const h = await api.tuneHash();
          if (!cancelled && h.stale) {
            setStale(true);
            setStaleReason("Config changed -- re-Solve to refresh.");
          }
        } catch {
          /* no doc open: leave as-is */
        }
      } else if (st.phase === "error") {
        setError(st.error || "solve failed");
      } else {
        // idle: warm the worker subprocess now so the first Solve is faster.
        api.tunePrewarm().catch(() => {});
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ensureDoc, startPolling]);

  // Once a solve populates parameters, auto-select the first slider-tunable one
  // so the detail panel shows a working slider immediately -- otherwise the
  // right pane just says "Select a parameter" and it is not obvious that
  // clicking a row in the tree is how you tune.
  useEffect(() => {
    if (!result || selected) return;
    const first = Object.entries(result.parameters).find(
      ([, p]) =>
        !p.derived &&
        !p.fixed &&
        p.lower != null &&
        p.upper != null &&
        p.upper > p.lower
    );
    if (first) setSelected(first[0]);
  }, [result, selected]);

  const solve = useCallback(async () => {
    setError(null);
    await ensureDoc();
    try {
      const st = await api.tuneSolve();
      setStatus(st);
      startPolling();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, [ensureDoc, startPolling]);

  // Send a document command (undoable, RANK_USER) then refresh staleness.
  const runCommand = useCallback(async (cmd: DocCommand, structural: boolean) => {
    try {
      await api.docCommand(cmd);
      if (structural) {
        const h = await api.tuneHash();
        setStale(h.stale);
        if (h.stale) setStaleReason("Config changed -- re-Solve to refresh.");
      }
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  // Live eval: patch the affected model traces into the current specs.
  const applyEval = useCallback(
    (updated: Record<string, Record<string, (number | null)[]>>) => {
      setSpecs((prev) =>
        prev.map((s) => {
          const upd = updated[s.id];
          if (!upd) return s;
          return {
            ...s,
            traces: s.traces.map((t) =>
              t.role === "model" && upd[t.name] !== undefined
                ? { ...t, y: upd[t.name] as number[] }
                : t
            ),
          };
        })
      );
    },
    []
  );

  const doEval = useCallback(
    async (path: string, value: number) => {
      try {
        const res = await api.tuneEval(path, value);
        if (res.needs_resolve) {
          setStale(true);
          setStaleReason(res.reason || "This parameter needs a re-Solve.");
          return;
        }
        if (res.out_of_bounds) {
          setStaleReason(res.reason || "Value outside bounds.");
          return;
        }
        if (res.plots) applyEval(res.plots);
      } catch {
        // transient; sliders stay usable
      }
    },
    [applyEval]
  );

  const parameters = result?.parameters || {};
  const live = status?.phase === "live";

  // Group filtered parameters by component instance for the tree.
  const grouped = useMemo(() => {
    const groups = new Map<string, string[]>();
    const q = search.trim().toLowerCase();
    for (const [path, p] of Object.entries(parameters)) {
      if (q && !path.toLowerCase().includes(q)) continue;
      if (sampledOnly && (p.derived || p.fixed)) continue;
      if (userOnly && p.provenance.label !== "user") continue;
      if (compFilter && !path.startsWith(compFilter + ".")) continue;
      const key = instanceKey(path);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(path);
    }
    return Array.from(groups.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [parameters, search, sampledOnly, userOnly, compFilter]);

  const compTypes = useMemo(() => {
    const set = new Set<string>();
    for (const path of Object.keys(parameters)) set.add(path.split(".")[0]);
    return Array.from(set).sort();
  }, [parameters]);

  const selectedLabel = selected ? labelForm(selected) : null;

  if (!configPath) {
    return (
      <div className="tune-empty muted">
        Open a project from the sidebar to tune its parameters here.
      </div>
    );
  }

  const phaseText: Record<string, string> = {
    idle: "Not solved yet",
    solving: "Solving (relaxation engine)...",
    compiling: "Compiling evaluator...",
    live: "Live",
    error: "Error",
  };

  return (
    <div className="tune-tab">
      <div className="tune-toolbar">
        <button
          className="tune-solve-btn"
          onClick={solve}
          disabled={status?.phase === "solving" || status?.phase === "compiling"}
        >
          {status?.phase === "solving" || status?.phase === "compiling"
            ? "Solving..."
            : "Solve"}
        </button>
        <span className={`tune-phase phase-${status?.phase || "idle"}`}>
          {phaseText[status?.phase || "idle"]}
        </span>
        <ProvenanceLegend />
        {error && <span className="tune-error">{error}</span>}
      </div>

      {stale && (
        <div className="tune-stale-banner">
          {staleReason || "Config changed -- re-Solve to refresh."}
          <button onClick={solve}>Re-Solve</button>
        </div>
      )}

      <div className="tune-body">
        {/* LEFT: searchable parameter tree */}
        <div className="tune-tree">
          <div className="tune-filters">
            <input
              className="tune-search"
              placeholder="search path..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <label>
              <input
                type="checkbox"
                checked={sampledOnly}
                onChange={(e) => setSampledOnly(e.target.checked)}
              />
              sampled only
            </label>
            <label>
              <input
                type="checkbox"
                checked={userOnly}
                onChange={(e) => setUserOnly(e.target.checked)}
              />
              user-touched
            </label>
            <select value={compFilter} onChange={(e) => setCompFilter(e.target.value)}>
              <option value="">all components</option>
              {compTypes.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
          {!result ? (
            <div className="muted tune-tree-empty">
              Press Solve to populate parameters.
            </div>
          ) : (
            grouped.map(([key, paths]) => (
              <div key={key} className="tune-tree-group">
                <div className="tune-tree-comp">{key}</div>
                {paths.map((path) => (
                  <ParamRow
                    key={path}
                    path={path}
                    param={parameters[path]}
                    active={selected === path}
                    stale={stale}
                    onClick={() => setSelected(path)}
                  />
                ))}
              </div>
            ))
          )}
        </div>

        {/* CENTER: plots, highlighted by dependency on the selected param */}
        <div className="tune-plots">
          {specs.length === 0 ? (
            <div className="muted tune-plots-empty">
              Model plots appear here after Solve.
            </div>
          ) : (
            specs.map((spec) => {
              const affected =
                selectedLabel !== null &&
                (spec.param_deps || []).includes(selectedLabel);
              const dimmed = selectedLabel !== null && !affected;
              return (
                <div
                  key={spec.id}
                  className={`tune-plot ${affected ? "affected" : ""} ${
                    dimmed ? "dimmed" : ""
                  }`}
                >
                  <PlotView spec={spec} />
                </div>
              );
            })
          )}
        </div>

        {/* RIGHT: detail panel for the selected parameter */}
        <div className="tune-detail">
          {selected && parameters[selected] ? (
            <DetailPanel
              path={selected}
              param={parameters[selected]}
              live={live}
              stale={stale}
              onEval={doEval}
              onCommand={runCommand}
            />
          ) : (
            <div className="muted tune-detail-empty">
              Select a parameter to edit its value, bounds, and prior.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- parameter tree row ------------------------------------------------------

function ParamRow({
  path,
  param,
  active,
  stale,
  onClick,
}: {
  path: string;
  param: TuneParam;
  active: boolean;
  stale: boolean;
  onClick: () => void;
}) {
  const color = PROV_COLORS[param.provenance.label] || PROV_COLORS.default;
  return (
    <button
      className={`tune-row ${active ? "active" : ""} ${stale ? "stale" : ""}`}
      onClick={onClick}
    >
      <span
        className="prov-dot"
        style={{ background: color }}
        title={PROV_HELP[param.provenance.label] || param.provenance.label}
      />
      <span className="tune-row-name">{paramName(path)}</span>
      <PriorGlyph param={param} />
      <span className="tune-row-value">
        {formatValue(param.value)}
        {param.unit ? <span className="tune-row-unit"> {param.unit}</span> : null}
      </span>
    </button>
  );
}

// --- prior glyph -------------------------------------------------------------

function PriorGlyph({ param }: { param: TuneParam }) {
  const kind = priorKind(param);
  if (kind === "fixed") {
    return (
      <svg className="prior-glyph" width="18" height="12">
        <title>fixed</title>
        <line x1="9" y1="1" x2="9" y2="11" stroke="#8b949e" strokeWidth="2" />
        <circle cx="9" cy="3" r="2" fill="#8b949e" />
      </svg>
    );
  }
  if (kind === "gaussian") {
    return (
      <svg className="prior-glyph" width="18" height="12">
        <title>gaussian prior</title>
        <path
          d="M1 11 C6 11, 6 2, 9 2 C12 2, 12 11, 17 11"
          fill="none"
          stroke="#f2a65a"
          strokeWidth="1.5"
        />
      </svg>
    );
  }
  return (
    <svg className="prior-glyph" width="18" height="12">
      <title>uniform prior (bounds)</title>
      <rect x="2" y="4" width="14" height="5" fill="none" stroke="#6cb6ff" strokeWidth="1.5" />
    </svg>
  );
}

type PriorKind = "fixed" | "gaussian" | "uniform";

function priorKind(param: TuneParam): PriorKind {
  if (param.fixed || param.sigma === 0) return "fixed";
  if (param.sigma != null && param.sigma > 0) return "gaussian";
  return "uniform";
}

// --- detail panel ------------------------------------------------------------

function DetailPanel({
  path,
  param,
  live,
  stale,
  onEval,
  onCommand,
}: {
  path: string;
  param: TuneParam;
  live: boolean;
  stale: boolean;
  onEval: (path: string, value: number) => void;
  onCommand: (cmd: DocCommand, structural: boolean) => void;
}) {
  const [value, setValue] = useState<number>(param.value ?? 0);
  const debounce = useRef<number | null>(null);

  // Reset the local value whenever the selection or solved value changes.
  useEffect(() => {
    setValue(param.value ?? 0);
  }, [path, param.value]);

  const lower = param.lower;
  const upper = param.upper;
  const sampled = !param.derived && !param.fixed;
  const hasBounds = lower != null && upper != null && upper > lower;
  const logScale = hasBounds && lower! > 0 && upper! > 0 &&
    Math.log10(upper! / lower!) > 3;

  const canLiveEdit = live && !stale && sampled && hasBounds;

  // Debounced live eval on slider/number change (~50 ms).
  const liveEval = useCallback(
    (v: number) => {
      if (!canLiveEdit) return;
      if (debounce.current) window.clearTimeout(debounce.current);
      debounce.current = window.setTimeout(() => onEval(path, v), 50);
    },
    [canLiveEdit, onEval, path]
  );

  // Commit the value to params.yaml as an undoable RANK_USER initval override
  // (one entry per slider release -- coalesces the whole drag).
  const commit = useCallback(
    (v: number) => {
      onCommand(
        { op: "set_param_field", args: { path, field: "initval", value: v } },
        false
      );
    },
    [onCommand, path]
  );

  const setField = useCallback(
    (field: string, v: number | null) => {
      onCommand({ op: "set_param_field", args: { path, field, value: v } }, true);
    },
    [onCommand, path]
  );

  const sliderPos = hasBounds ? toSlider(value, lower!, upper!, logScale) : 0.5;
  const kind = priorKind(param);

  return (
    <div className="detail-panel">
      <h3 className="detail-title">{path}</h3>
      <div className="detail-meta">
        <span
          className="prov-dot"
          style={{ background: PROV_COLORS[param.provenance.label] }}
        />
        <span className="muted">
          {param.provenance.label}
          {param.provenance.relation ? ` (${param.provenance.relation})` : ""}
        </span>
        {param.unit && <span className="detail-unit">unit: {param.unit}</span>}
      </div>

      {param.derived && <p className="muted">Derived -- not a free slider.</p>}
      {param.fixed && !param.derived && (
        <p className="muted">Fixed (sigma = 0) -- free it to slide.</p>
      )}

      {/* Slider + numeric entry */}
      <div className="detail-slider-row">
        <input
          type="range"
          className={`detail-slider ${stale ? "stale" : ""}`}
          min={0}
          max={1000}
          step={1}
          value={Math.round(sliderPos * 1000)}
          disabled={!canLiveEdit}
          onChange={(e) => {
            const t = Number(e.target.value) / 1000;
            const v = fromSlider(t, lower!, upper!, logScale);
            setValue(v);
            liveEval(v);
          }}
          onPointerUp={() => commit(value)}
        />
        <input
          type="number"
          className="detail-value-input"
          value={Number.isFinite(value) ? value : ""}
          disabled={!sampled}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v)) {
              setValue(v);
              liveEval(v);
            }
          }}
          onBlur={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v)) commit(v);
          }}
        />
        <span className="muted detail-unit-inline">{param.unit || ""}</span>
      </div>
      {logScale && <div className="muted detail-scale-note">log-scaled slider</div>}

      {/* Bounds */}
      <div className="detail-fields">
        <label>
          lower
          <input
            type="number"
            defaultValue={lower ?? ""}
            key={`lo-${path}-${lower}`}
            onBlur={(e) =>
              setField("lower", e.target.value === "" ? null : Number(e.target.value))
            }
          />
        </label>
        <label>
          upper
          <input
            type="number"
            defaultValue={upper ?? ""}
            key={`hi-${path}-${upper}`}
            onBlur={(e) =>
              setField("upper", e.target.value === "" ? null : Number(e.target.value))
            }
          />
        </label>
      </div>

      {/* Prior fields */}
      <div className="detail-fields">
        <label>
          mu
          <input
            type="number"
            defaultValue={param.mu ?? ""}
            key={`mu-${path}-${param.mu}`}
            onBlur={(e) =>
              setField("mu", e.target.value === "" ? null : Number(e.target.value))
            }
          />
        </label>
        <label>
          sigma
          <input
            type="number"
            defaultValue={param.sigma ?? ""}
            key={`sig-${path}-${param.sigma}`}
            onBlur={(e) =>
              setField("sigma", e.target.value === "" ? null : Number(e.target.value))
            }
          />
        </label>
      </div>

      {/* Fix / free + reset */}
      <div className="detail-actions">
        <button
          className="detail-toggle"
          onClick={() =>
            kind === "fixed" ? setField("sigma", null) : setField("sigma", 0)
          }
        >
          {kind === "fixed" ? "Free" : "Fix"}
        </button>
        <button
          className="detail-reset"
          title="Remove the user initval override and revert to the solved value"
          onClick={() => {
            onCommand(
              { op: "set_param_field", args: { path, field: "initval", value: null } },
              false
            );
            setValue(param.value ?? 0);
          }}
        >
          Reset to solved
        </button>
      </div>

      <div className="detail-prior-preview">
        <span className="muted">prior: </span>
        <PriorGlyph param={param} />
        <span className="muted">
          {" "}
          {kind === "fixed"
            ? "fixed"
            : kind === "gaussian"
            ? `N(mu, sigma)`
            : "uniform in [lower, upper]"}
        </span>
      </div>
    </div>
  );
}

// --- provenance legend -------------------------------------------------------

function ProvenanceLegend() {
  return (
    <span className="prov-legend">
      {(["user", "data", "solved", "default"] as const).map((k) => (
        <span key={k} className="prov-legend-item" title={PROV_HELP[k]}>
          <span className="prov-dot" style={{ background: PROV_COLORS[k] }} />
          {k}
        </span>
      ))}
    </span>
  );
}

// --- helpers -----------------------------------------------------------------

function formatValue(v: number | null): string {
  if (v == null || !Number.isFinite(v)) return "--";
  const a = Math.abs(v);
  if (a !== 0 && a < 1e-3) return v.toExponential(3);
  // Large values are typically times (BJD ~2.46e6); keep 6 decimals rather than
  // collapsing to sci-notation, which drops the sub-day (tc) precision.
  if (a >= 1e5) return v.toFixed(6);
  return String(Number(v.toPrecision(6)));
}

function toSlider(value: number, lo: number, hi: number, log: boolean): number {
  const clamp = (t: number) => Math.max(0, Math.min(1, t));
  if (log) {
    const l = Math.log10(lo);
    const h = Math.log10(hi);
    const v = Math.log10(Math.max(value, lo));
    return clamp((v - l) / (h - l));
  }
  return clamp((value - lo) / (hi - lo));
}

function fromSlider(t: number, lo: number, hi: number, log: boolean): number {
  if (log) {
    const l = Math.log10(lo);
    const h = Math.log10(hi);
    return Math.pow(10, l + t * (h - l));
  }
  return lo + t * (hi - lo);
}
