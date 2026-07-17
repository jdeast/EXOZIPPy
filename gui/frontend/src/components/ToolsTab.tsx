import { useEffect, useMemo, useState } from "react";
import {
  api,
  type ProjectListing,
  type UtilityArg,
  type UtilityResult,
  type UtilitySchema,
} from "../api";

interface ToolsTabProps {
  listing: ProjectListing | null;
  // Attach the bottom terminal to the utility's captured-output log file.
  setLogFile: (file: string | null) => void;
}

// One auto-generated form field, typed from the argparse-derived arg schema.
function ArgField({
  arg,
  value,
  onChange,
}: {
  arg: UtilityArg;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const label = (
    <span className="arg-label">
      {arg.name}
      {arg.required && <span className="arg-required">*</span>}
    </span>
  );

  if (arg.type === "bool") {
    return (
      <label className="arg-field arg-bool" title={arg.help}>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
        />
        {label}
      </label>
    );
  }

  if (arg.choices && arg.choices.length) {
    return (
      <label className="arg-field" title={arg.help}>
        {label}
        <select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>
          <option value="">(default)</option>
          {arg.choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>
    );
  }

  const inputType = arg.type === "int" || arg.type === "float" ? "number" : "text";
  return (
    <label className="arg-field" title={arg.help}>
      {label}
      <input
        type={inputType}
        value={value === undefined || value === null ? "" : String(value)}
        placeholder={arg.default !== null && arg.default !== undefined ? String(arg.default) : ""}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function UtilityCard({
  util,
  cwd,
  setLogFile,
}: {
  util: UtilitySchema;
  cwd: string | null;
  setLogFile: (file: string | null) => void;
}) {
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [result, setResult] = useState<UtilityResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canRun = util.available && !!cwd && !running;

  const run = async () => {
    if (!cwd) return;
    setError(null);
    setRunning(true);
    setResult(null);
    try {
      // Drop empty strings so the backend falls back to argparse defaults.
      const args: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(values)) {
        if (v !== "" && v !== undefined && v !== null) args[k] = v;
      }
      const res = await api.runUtility(util.name, args, cwd);
      setResult(res);
      if (res.log_path) setLogFile(res.log_path);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className={`utility-card ${util.available ? "" : "unavailable"}`}>
      <div className="utility-head">
        <h3>{util.label}</h3>
        {!util.available && <span className="coming-soon">coming soon</span>}
        {util.component_keys.length > 0 && (
          <span className="utility-comps">{util.component_keys.join(", ")}</span>
        )}
      </div>
      <p className="muted">{util.description}</p>

      {util.available && (
        <>
          <div className="arg-grid">
            {util.arguments.map((arg) => (
              <ArgField
                key={arg.name}
                arg={arg}
                value={values[arg.name]}
                onChange={(v) => setValues((prev) => ({ ...prev, [arg.name]: v }))}
              />
            ))}
          </div>
          <div className="utility-actions">
            <button onClick={run} disabled={!canRun} title={cwd ? "" : "Open a project first"}>
              {running ? "Running..." : "Run"}
            </button>
          </div>
        </>
      )}

      {error && <p className="run-error">{error}</p>}

      {result && (
        <div className="utility-result">
          <p className={result.returncode === 0 ? "ok" : "run-error"}>
            exit code {result.returncode}
          </p>
          {result.produced_files.length > 0 ? (
            <ul className="produced-files">
              {result.produced_files.map((f) => (
                <li key={f}>
                  <span className="file-path" title={f}>
                    {f.split("/").pop()}
                  </span>
                  {/* TODO(G9): wire the associate-with-component flow. */}
                  <button
                    className="associate-button"
                    title="Associate with a component (coming in G9)"
                    disabled
                  >
                    Associate
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No new files produced.</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function ToolsTab({ listing, setLogFile }: ToolsTabProps) {
  const [utils, setUtils] = useState<Record<string, UtilitySchema> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .utilities()
      .then(setUtils)
      .catch((e) => setError(String(e)));
  }, []);

  const sorted = useMemo(
    () =>
      utils
        ? Object.values(utils).sort((a, b) => {
            if (a.available !== b.available) return a.available ? -1 : 1;
            return a.label.localeCompare(b.label);
          })
        : [],
    [utils],
  );

  return (
    <div className="tools-tab">
      <p className="muted">
        Component-declared utilities, discovered from the registry (no component
        names are hardcoded). Runs execute server-side in a subprocess against
        the open project directory.
      </p>
      {!listing && <p className="run-hint muted">Open a project to run utilities.</p>}
      {error && <p className="run-error">{error}</p>}
      {utils === null && !error && <p className="muted">Loading utilities...</p>}
      <div className="utility-list">
        {sorted.map((u) => (
          <UtilityCard key={u.name} util={u} cwd={listing?.dir ?? null} setLogFile={setLogFile} />
        ))}
      </div>
    </div>
  );
}
