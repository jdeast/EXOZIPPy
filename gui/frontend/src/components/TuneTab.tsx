import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  ApiError,
  type DocCommand,
  type Chart,
  type TuneEvalTrace,
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
// instance; center = the Chart plots (highlighted when they depend on the
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
// (comp.param), which is what Chart.param_deps carries.
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

const HASH_STALE_REASON = "Config changed -- re-Solve to refresh.";

// How often the tab re-asks the server whether the open document still matches
// the live evaluator, while it is visible and live.
const HASH_POLL_MS = 2000;

export default function TuneTab({
  configPath,
  active = true,
}: {
  configPath: string | null;
  active?: boolean;
}) {
  const [status, setStatus] = useState<TuneStatus | null>(null);
  const [result, setResult] = useState<TuneResult | null>(null);
  const [specs, setSpecs] = useState<Chart[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Staleness has TWO independent sources and they must not overwrite each
  // other. `hashStale` is "the document no longer matches the compiled
  // evaluator", answered by the server and re-answerable at any time.
  // `evalStale` is "this evaluator cannot serve you", raised by an eval that
  // came back needs_resolve or by a worker that died -- a fact no hash check
  // knows about. One boolean carried both, so any hash re-check silently
  // cleared a needs_resolve banner (and vice versa).
  const [hashStale, setHashStale] = useState(false);
  const [evalStale, setEvalStale] = useState<string | null>(null);
  const stale = hashStale || evalStale !== null;
  const staleReason = evalStale ?? (hashStale ? HASH_STALE_REASON : null);
  // A transient, NON-blocking complaint about the last eval (an out-of-bounds
  // value). Distinct from `stale`: nothing about the compiled evaluator is
  // invalid, so demanding a re-Solve would be wrong -- moving back inside the
  // bounds resumes at once, and the notice clears itself on the next good eval.
  const [notice, setNotice] = useState<string | null>(null);
  const [docDirty, setDocDirty] = useState(false);

  // Filters
  const [search, setSearch] = useState("");
  const [sampledOnly, setSampledOnly] = useState(false);
  const [userOnly, setUserOnly] = useState(false);
  const [compFilter, setCompFilter] = useState("");

  const pollTimer = useRef<number | null>(null);

  // Ensure OUR config is the document open on the server (Solve and every edit
  // command read/write that one slot). Checking the path matters: after a
  // project switch the server may still hold the previous project's document,
  // and reusing it would solve -- and save edits into -- the wrong files.
  // Re-opening the same path is edit-preserving on the server, so this is
  // idempotent for the config we are actually tuning.
  const ensureDoc = useCallback(async () => {
    if (!configPath) return;
    try {
      const d = await api.doc();
      if (d.config_path === configPath) {
        setDocDirty(d.dirty);
        return;
      }
    } catch {
      /* nothing open -- fall through and open ours */
    }
    const d = await api.docOpen(configPath);
    setDocDirty(d.dirty);
  }, [configPath]);

  // One data-plots fetch per solve: the worker ships data-only specs with its
  // "compiling" progress message so the observations render while the model
  // compiles; this ref stops the 400 ms poll from refetching them.
  const dataPlotsLoaded = useRef(false);

  // Ask the server whether the open document still matches the live
  // evaluator. Never asserts freshness it has not verified: a failure (no
  // document open, a transient blip) leaves the banner exactly as it was.
  const refreshHashStale = useCallback(async () => {
    try {
      const h = await api.tuneHash();
      setHashStale(h.stale);
    } catch {
      /* no doc open, or transient -- leave the current verdict alone */
    }
  }, []);

  // Load the live solve result AND re-derive staleness from the server's own
  // hash. The poller and the mount-restore effect used to be two copies of
  // this with OPPOSITE verdicts -- the poller hard-reset `stale`, restore
  // consulted the hash -- and the divergence was load-bearing: pressing
  // Re-Solve from the stale banner while a solve was already in flight let
  // the poller hydrate the PRE-EDIT result and then clear the very banner
  // that would have said so.
  const hydrateResult = useCallback(async () => {
    const res = await api.tuneResult();
    setResult(res);
    setSpecs(res.plots);
    await refreshHashStale();
  }, [refreshHashStale]);

  // Poll solve status until it leaves the transient phases.
  const startPolling = useCallback(() => {
    if (pollTimer.current) window.clearInterval(pollTimer.current);
    pollTimer.current = window.setInterval(async () => {
      try {
        const st = await api.tuneStatus();
        setStatus(st);
        if (
          st.has_data_plots &&
          !dataPlotsLoaded.current &&
          (st.phase === "solving" || st.phase === "compiling")
        ) {
          dataPlotsLoaded.current = true;
          try {
            const dp = await api.tuneDataPlots();
            // Only fill an empty canvas: during a re-Solve the previous full
            // (data+model) plots are better than a data-only downgrade.
            if (dp.plots.length) {
              setSpecs((prev) => (prev.length ? prev : dp.plots));
            }
          } catch {
            dataPlotsLoaded.current = false; // transient; retry next poll
          }
        }
        if (st.phase === "live" && st.has_result) {
          if (pollTimer.current) window.clearInterval(pollTimer.current);
          pollTimer.current = null;
          await hydrateResult();
        } else if (st.phase === "error") {
          if (pollTimer.current) window.clearInterval(pollTimer.current);
          pollTimer.current = null;
          setError(st.error || "solve failed");
        } else {
          // A poll that SUCCEEDS clears whatever the last failed one left
          // behind. Nothing else did: the catch below set `error` and no
          // success path ever unset it, so one transient blip pinned a
          // message on screen until the next manual Solve.
          setError(null);
        }
      } catch (e) {
        setError(String(e instanceof Error ? e.message : e));
      }
    }, 400);
  }, [hydrateResult]);

  useEffect(
    () => () => {
      if (pollTimer.current) window.clearInterval(pollTimer.current);
    },
    []
  );

  const solve = useCallback(async () => {
    setError(null);
    setNotice(null);
    // A fresh solve supersedes any needs_resolve/dead-worker verdict; the
    // hash verdict is re-derived from the server when the result lands.
    setEvalStale(null);
    dataPlotsLoaded.current = false;
    try {
      // ensureDoc INSIDE the try. It was outside, so a docOpen rejection (a
      // config `check_yaml_booleans` refuses -- `finite_source: no` -- is a
      // 400) escaped as an unhandled promise rejection from the auto-solve
      // effect below: no setError, no phase change, "Not solved yet" forever.
      // Single-config projects land straight here, so the user may never see
      // ConfigTab's correct rendering of the same error.
      await ensureDoc();
      const st = await api.tuneSolve();
      setStatus(st);
      startPolling();
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, [ensureDoc, startPolling]);

  // Restore from the server-side session on (re)mount. The solve runs in a
  // background thread on the server; its phase/result/hash outlive this
  // component, so switching tabs mid-solve must NOT lose it. We pick an
  // in-flight solve back up, re-hydrate a finished one, or -- when idle with a
  // config open -- kick the first Solve off automatically: the Tune tab is the
  // landing page, and the natural first render is the data with the solved
  // model over it, not an empty pane waiting for a button.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      // The whole body is guarded. `ensureDoc` can reject (a config the YAML
      // boolean guard refuses), and so can `tuneResult` -- and an unhandled
      // rejection here left the tab on "Live" with no parameters, no plots
      // and no error. Anything unexpected falls through to the idle path,
      // which auto-solves and reports its own failure properly.
      try {
        await ensureDoc();
        const st = await api.tuneStatus();
        if (cancelled) return;
        setStatus(st);
        if (st.phase === "solving" || st.phase === "compiling") {
          startPolling(); // keep watching the background solve
          return;
        }
        if (st.phase === "live" && st.has_result) {
          // hydrateResult consults the hash, so a structural edit made
          // elsewhere while we were away raises the re-Solve banner.
          await hydrateResult();
          return;
        }
        if (st.phase === "error") {
          setError(st.error || "solve failed");
          return;
        }
      } catch (e) {
        if (cancelled) return;
        if (!configPath) {
          setError(String(e instanceof Error ? e.message : e));
          return;
        }
        // fall through to the auto-solve, which surfaces its own errors
      }
      // idle (or an unreadable session) + a config to work with: auto-Solve.
      if (!cancelled && configPath) solve();
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ensureDoc, startPolling, solve, hydrateResult, configPath]);

  // Structural edits made in OTHER tabs never came through `runCommand` here,
  // and both tabs stay mounted, so nothing re-checked the hash after a
  // ConfigTab edit: the sliders stayed live and kept committing PRECEDENCE_USER
  // initvals against a model the document no longer described. Re-check on
  // reveal, and keep checking while visible and live -- an edit can also
  // arrive without a tab switch (Ctrl+Z, an external write picked up on
  // re-open).
  useEffect(() => {
    const isLive = status?.phase === "live";
    if (!active || !isLive) return;
    refreshHashStale();
    const timer = window.setInterval(refreshHashStale, HASH_POLL_MS);
    return () => window.clearInterval(timer);
  }, [active, status?.phase, refreshHashStale]);

  // Once a solve populates parameters, auto-select the first slider-tunable
  // one so a working slider shows immediately -- otherwise it is not obvious
  // that clicking a row in the tree is how you tune. Once only: clicking the
  // active row deselects it, and re-auto-selecting would fight that.
  const autoSelected = useRef(false);
  useEffect(() => {
    if (!result || selected || autoSelected.current) return;
    const first = Object.entries(result.parameters).find(
      ([, p]) =>
        !p.derived &&
        !p.fixed &&
        p.lower != null &&
        p.upper != null &&
        p.upper > p.lower
    );
    if (first) {
      autoSelected.current = true;
      setSelected(first[0]);
    }
  }, [result, selected]);

  // Send a document command (undoable, PRECEDENCE_USER) then refresh staleness.
  const runCommand = useCallback(
    async (cmd: DocCommand, structural: boolean) => {
      try {
        const next = await api.docCommand(cmd);
        setDocDirty(next.dirty);
        if (structural) await refreshHashStale();
      } catch (e) {
        setError(String(e instanceof Error ? e.message : e));
      }
    },
    [refreshHashStale]
  );

  // Live eval: patch the affected traces (both x and y -- a phased curve's
  // x-grid moves too when period/tc is tuned) into the current specs. The
  // payload carries model traces always, and data traces (with their errors)
  // for dynamic_data specs -- phase folds, gamma offsets, flux alignment --
  // so those panels track the slider instead of freezing.
  const applyEval = useCallback(
    (updated: Record<string, Record<string, TuneEvalTrace>>) => {
      setSpecs((prev) =>
        prev.map((s) => {
          const upd = updated[s.id];
          if (!upd) return s;
          return {
            ...s,
            traces: s.traces.map((t) => {
              const u = upd[t.name];
              if (u === undefined) return t;
              const next = { ...t, x: u.x as number[], y: u.y as number[] };
              if (u.yerr !== undefined) next.yerr = u.yerr;
              return next;
            }),
          };
        })
      );
    },
    []
  );

  // Eval requests are debounced but still overlap, and nothing makes the server
  // answer them in order: a slower early response landing last would repaint
  // the charts at a value the user has already dragged away from, leaving the
  // panel showing a curve that is not the one the slider says. Each request
  // takes a monotonically increasing id and a response older than the newest
  // one already applied is dropped -- including its notice/staleness verdict,
  // which describes a superseded value.
  const evalSeq = useRef(0);
  const appliedSeq = useRef(0);

  const doEval = useCallback(
    async (path: string, value: number) => {
      const seq = ++evalSeq.current;
      try {
        const res = await api.tuneEval(path, value);
        if (seq < appliedSeq.current) return; // a newer response already landed
        appliedSeq.current = seq;
        if (res.needs_resolve) {
          setEvalStale(res.reason || "This parameter needs a re-Solve.");
          return;
        }
        if (res.out_of_bounds) {
          // Visible feedback, and NOT via the stale banner (see `notice`): the
          // value was rejected, the plots still show the last good point.
          setNotice(res.reason || "Value outside bounds -- plots not updated.");
          return;
        }
        setNotice(null);
        if (res.plots) applyEval(res.plots);
      } catch (e) {
        const message = String(e instanceof Error ? e.message : e);
        if (e instanceof ApiError && e.status === 409) {
          // NOT transient. A 409 means the evaluator is gone -- a wedged
          // worker was terminated and respawned, or the session was reset --
          // so every further slider move is a no-op until the next Solve.
          // This used to end in a bare catch, and the status poll had already
          // been cleared once the solve went live, so a timed-out eval was
          // invisible until the user happened to press Solve again.
          setEvalStale(message);
          return;
        }
        // Anything else (a network blip) really is transient: say so once,
        // and leave the sliders usable.
        setNotice(message);
      }
    },
    [applyEval]
  );

  const parameters = result?.parameters || {};
  const live = status?.phase === "live";
  // A solve is already running: Solve and the banner's Re-Solve must both be
  // dead. Leaving Re-Solve armed let a second solve be queued against the
  // first, and the poller then hydrated whichever finished first.
  const solving =
    status?.phase === "solving" || status?.phase === "compiling";

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
        <button className="tune-solve-btn" onClick={solve} disabled={solving}>
          {solving ? "Solving..." : "Solve"}
        </button>
        <span className={`tune-phase phase-${status?.phase || "idle"}`}>
          {phaseText[status?.phase || "idle"]}
        </span>
        <button
          className="save-btn"
          disabled={!docDirty}
          title="Write the tuned values to the params file"
          onClick={async () => {
            try {
              const d = await api.docSave();
              setDocDirty(d.dirty);
            } catch (e) {
              setError(String(e instanceof Error ? e.message : e));
            }
          }}
        >
          Save
        </button>
        <span
          className={`dirty-dot ${docDirty ? "on" : ""}`}
          title={docDirty ? "Unsaved changes" : "Saved"}
        />
        <ProvenanceLegend />
        {notice && (
          <span className="tune-notice" title={notice}>
            {notice}
          </span>
        )}
        {error && <span className="tune-error">{error}</span>}
      </div>

      {stale && (
        <div className="tune-stale-banner">
          {staleReason || HASH_STALE_REASON}
          <button onClick={solve} disabled={solving}>
            {solving ? "Solving..." : "Re-Solve"}
          </button>
        </div>
      )}

      <div className="tune-body">
        {/* LEFT: searchable parameter tree; the selected row expands into
            the detail editor (slider/bounds/prior) right below itself, so
            the rest of the window is all plots. */}
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
              {status?.phase === "solving" || status?.phase === "compiling"
                ? "Solving -- parameters appear when it finishes."
                : "Press Solve to populate parameters."}
            </div>
          ) : (
            grouped.map(([key, paths]) => (
              <div key={key} className="tune-tree-group">
                <div className="tune-tree-comp">{key}</div>
                {paths.map((path) => (
                  <div key={path}>
                    <ParamRow
                      path={path}
                      param={parameters[path]}
                      active={selected === path}
                      stale={stale}
                      onClick={() =>
                        setSelected(selected === path ? null : path)
                      }
                    />
                    {selected === path && (
                      <DetailPanel
                        path={path}
                        param={parameters[path]}
                        live={live}
                        stale={stale}
                        onEval={doEval}
                        onCommand={runCommand}
                      />
                    )}
                  </div>
                ))}
              </div>
            ))
          )}
        </div>

        {/* RIGHT: the plot grid, highlighted by dependency on the selected
            parameter. A grid (instead of one wide column) keeps several
            panels visible at once. */}
        <div className="tune-plots">
          {specs.length === 0 ? (
            <div className="muted tune-plots-empty">
              {status?.phase === "solving"
                ? "Loading data..."
                : "Plots appear here after Solve."}
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
  // The value the user has actually moved to, readable synchronously. `value`
  // is React state, so a handler that closes over it sees the value as of its
  // own render: the pointerup that ends a drag can run in the same batch as
  // the last pointermove and commit the move BEFORE it -- one move stale, and
  // the params file then disagrees with both the slider and the plots. Every
  // write goes through applyValue so the ref and the state move together.
  const latest = useRef<number>(param.value ?? 0);
  // The value last written to the params file (starting at the solved one,
  // which is not an override). The number input commits onBlur, which fires on
  // a plain click-through, so this is what tells a real edit from a tab-past.
  const committed = useRef<number | null>(param.value ?? null);

  // Reset the local value whenever the selection or solved value changes.
  useEffect(() => {
    latest.current = param.value ?? 0;
    committed.current = param.value ?? null;
    setValue(param.value ?? 0);
  }, [path, param.value]);

  const lower = param.lower;
  const upper = param.upper;
  const sampled = !param.derived && !param.fixed;

  // Default slider window: +/-10*init_scale around the last solved value --
  // NOT the full prior range, which for a wide uniform bound would make one
  // slider tick a huge, useless jump. Falls back to the full [lower, upper]
  // range when init_scale isn't available. The window auto-expands to keep
  // the live value inside it (see the drag handlers below, which let the
  // user keep moving the value past the nominal rail by continuing to drag).
  const initScale = param.init_scale != null && param.init_scale > 0 ? param.init_scale : null;
  const solvedValue = param.value ?? 0;
  const windowHalf = initScale != null ? 10 * initScale : null;
  const baseLo = windowHalf != null
    ? (lower != null ? Math.max(lower, solvedValue - windowHalf) : solvedValue - windowHalf)
    : lower;
  const baseHi = windowHalf != null
    ? (upper != null ? Math.min(upper, solvedValue + windowHalf) : solvedValue + windowHalf)
    : upper;
  const hasBounds = baseLo != null && baseHi != null && baseHi > baseLo;
  const winLo = hasBounds ? Math.min(baseLo!, value) : value - 1;
  const winHi = hasBounds ? Math.max(baseHi!, value) : value + 1;
  const logScale = hasBounds && winLo > 0 && winHi > 0 &&
    Math.log10(winHi / winLo) > 3;
  const stepSize = initScale != null ? 0.3 * initScale : undefined;

  const canLiveEdit = live && !stale && sampled && hasBounds;

  const clampToHardBounds = (v: number) => {
    let out = v;
    if (lower != null) out = Math.max(lower, out);
    if (upper != null) out = Math.min(upper, out);
    return out;
  };

  // Debounced live eval on slider/number change (~50 ms).
  const liveEval = useCallback(
    (v: number) => {
      if (!canLiveEdit) return;
      if (debounce.current) window.clearTimeout(debounce.current);
      debounce.current = window.setTimeout(() => onEval(path, v), 50);
    },
    [canLiveEdit, onEval, path]
  );

  // Move the value: state (for the render) + ref (for the handlers) + a
  // debounced live eval. The single writer for slider, drag and typed input.
  const applyValue = useCallback(
    (v: number) => {
      latest.current = v;
      setValue(v);
      liveEval(v);
    },
    [liveEval]
  );

  // Commit the value to params.yaml as an undoable PRECEDENCE_USER initval override
  // (one entry per slider release -- coalesces the whole drag).
  const commit = useCallback(
    (v: number) => {
      committed.current = v;
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

  // Bounds/prior fields commit onBlur, which fires on a plain click-through
  // with nothing typed. Writing the field back unchanged would turn whatever
  // the panel happens to be showing -- a component default, or another
  // project's solved bound -- into a PRECEDENCE_USER override in the params file.
  // Only a real edit is a command.
  const setFieldIfChanged = useCallback(
    (field: string, raw: string, current: number | null | undefined) => {
      const v = raw === "" ? null : Number(raw);
      if (v !== null && !Number.isFinite(v)) return;
      if (v === (current ?? null)) return;
      setField(field, v);
    },
    [setField]
  );

  // Pointer-driven drag: the anchor window is snapshotted at drag start and
  // stays fixed for the whole gesture, so continuing to drag past the rail's
  // visual edge keeps extending the value smoothly (unclamped fraction t)
  // instead of getting stuck once the native [0, 1000] track saturates.
  const dragRef = useRef<{
    startX: number; startT: number; lo: number; hi: number; log: boolean; width: number;
  } | null>(null);

  const beginDrag = (clientX: number, rectLeft: number, width: number) => {
    const w = Math.max(1, width);
    const clickT = (clientX - rectLeft) / w;
    applyValue(clampToHardBounds(fromSlider(clickT, winLo, winHi, logScale)));
    dragRef.current = { startX: clientX, startT: clickT, lo: winLo, hi: winHi, log: logScale, width: w };
  };

  const moveDrag = (clientX: number) => {
    const drag = dragRef.current;
    if (!drag) return;
    const t = drag.startT + (clientX - drag.startX) / drag.width; // unclamped
    applyValue(clampToHardBounds(fromSlider(t, drag.lo, drag.hi, drag.log)));
  };

  const endDrag = () => {
    if (!dragRef.current) return;
    dragRef.current = null;
    // latest.current, not `value`: the last pointermove may not have been
    // rendered yet, and committing the render's value would write the
    // second-to-last position of the drag.
    commit(latest.current);
  };

  const sliderPos = hasBounds ? toSlider(value, winLo, winHi, logScale) : 0.5;
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
          value={Math.round(Math.max(0, Math.min(1, sliderPos)) * 1000)}
          disabled={!canLiveEdit}
          onPointerDown={(e) => {
            if (!canLiveEdit) return;
            e.preventDefault();
            e.currentTarget.setPointerCapture(e.pointerId);
            const rect = e.currentTarget.getBoundingClientRect();
            beginDrag(e.clientX, rect.left, rect.width);
          }}
          onPointerMove={(e) => moveDrag(e.clientX)}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onChange={(e) => {
            // Keyboard (arrow/Home/End) fallback -- pointer drags are handled
            // above and don't go through this (the native track stays clamped
            // to [0, 1000], but dragRef's frozen window lets the value itself
            // keep moving past it).
            if (dragRef.current) return;
            const t = Number(e.target.value) / 1000;
            applyValue(
              clampToHardBounds(fromSlider(t, winLo, winHi, logScale))
            );
          }}
        />
        <input
          type="number"
          className="detail-value-input"
          value={Number.isFinite(value) ? value : ""}
          step={stepSize}
          // Same gate as the slider (canLiveEdit), NOT just `sampled`: this
          // input commits a PRECEDENCE_USER initval into the params file, so it must
          // be dead whenever the number on screen does not belong to the model
          // the server would write it against -- not live yet, gone stale, or
          // (before the remount fix) left over from another project.
          disabled={!canLiveEdit}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v)) applyValue(v);
          }}
          onBlur={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v) && v !== committed.current) commit(v);
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
            onBlur={(e) => setFieldIfChanged("lower", e.target.value, lower)}
          />
        </label>
        <label>
          upper
          <input
            type="number"
            defaultValue={upper ?? ""}
            key={`hi-${path}-${upper}`}
            onBlur={(e) => setFieldIfChanged("upper", e.target.value, upper)}
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
            onBlur={(e) => setFieldIfChanged("mu", e.target.value, param.mu)}
          />
        </label>
        <label>
          sigma
          <input
            type="number"
            defaultValue={param.sigma ?? ""}
            key={`sig-${path}-${param.sigma}`}
            onBlur={(e) => setFieldIfChanged("sigma", e.target.value, param.sigma)}
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
            // applyValue, so the ref the drag/commit handlers read cannot
            // drift from the number on screen (and the plots follow it back).
            committed.current = param.value ?? null;
            applyValue(param.value ?? 0);
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
