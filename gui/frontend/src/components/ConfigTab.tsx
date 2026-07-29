import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  type DocCommand,
  type DocState,
  type Diagnostic,
} from "../api";

// Config tab: a form editor over the two user YAML files. Left = tree of
// component instances (grouped by type) + global keys; right = an
// auto-generated form for the selection, driven entirely by the G1 schema (no
// component names are hardcoded). Edits go through the document command API so
// undo/redo and dirty tracking live on the server.

type Schema = {
  components: Record<string, ComponentSchema>;
  global: Record<string, GlobalKey>;
};

interface ComponentSchema {
  yaml_key: string;
  doc: string;
  parameters: Record<string, ParamSchema>;
  config: ConfigKey[];
}

interface ParamSchema {
  name: string;
  unit?: string;
  latex?: string;
  description?: string;
  derived: boolean;
  sampled: boolean;
  initval?: number;
  lower?: number;
  upper?: number;
  sigma?: number;
}

interface ConfigKey {
  key: string;
  kind: "datafile" | "ref" | "option";
  accepts: string[] | string | null;
  required: boolean;
  doc: string;
}

interface GlobalKey {
  key: string;
  kind: string;
  accepts: string[] | null;
  required: boolean;
  doc: string;
}

type Selection =
  | { kind: "global" }
  | { kind: "instance"; comp: string; name: string };

const PARAM_FIELDS = ["initval", "lower", "upper", "sigma", "mu", "init_scale"];

export default function ConfigTab({ configPath }: { configPath: string | null }) {
  const [schema, setSchema] = useState<Schema | null>(null);
  const [doc, setDoc] = useState<DocState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>({ kind: "global" });
  const [diagnostics, setDiagnostics] = useState<Diagnostic[]>([]);
  const [validating, setValidating] = useState(false);

  // Load the schema once.
  useEffect(() => {
    api
      .schema()
      .then((s) => setSchema(s as unknown as Schema))
      .catch((e) => setError(String(e)));
  }, []);

  // Open the document whenever the selected config file changes.
  useEffect(() => {
    if (!configPath) return;
    api
      .docOpen(configPath)
      .then((d) => {
        setDoc(d);
        setError(null);
      })
      .catch((e) => setError(String(e)));
  }, [configPath]);

  const runCommand = useCallback(async (cmd: DocCommand) => {
    try {
      const next = await api.docCommand(cmd);
      setDoc(next);
      setError(null);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  const save = useCallback(async () => {
    try {
      setDoc(await api.docSave());
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  const undo = useCallback(async () => setDoc(await api.docUndo()), []);
  const redo = useCallback(async () => setDoc(await api.docRedo()), []);

  // Debounced validation after edits (1.2 s idle). Poll the job to completion.
  const revision = doc ? `${doc.undo_depth}:${doc.redo_depth}` : "";
  useEffect(() => {
    if (!doc) return;
    const handle = setTimeout(async () => {
      try {
        setValidating(true);
        const job = await api.docValidateStart();
        let poll = job;
        const deadline = Date.now() + 120000;
        while (poll.status === "running" && Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 500));
          poll = await api.docValidatePoll(job.job_id);
        }
        setDiagnostics(poll.diagnostics || []);
      } catch {
        // leave prior diagnostics on transient failure
      } finally {
        setValidating(false);
      }
    }, 1200);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [revision, configPath]);

  // Keyboard: Ctrl+S save, Ctrl+Z undo, Ctrl+Shift+Z redo.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      const key = e.key.toLowerCase();
      if (key === "s") {
        e.preventDefault();
        save();
      } else if (key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((key === "z" && e.shiftKey) || key === "y") {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [save, undo, redo]);

  const instancesByType = useMemo(() => buildTree(doc), [doc]);

  if (!configPath) {
    return (
      <div className="config-empty muted">
        Open a project from the sidebar, then a config file, to edit it here.
      </div>
    );
  }
  if (error && !doc) {
    return <div className="config-empty config-error">{error}</div>;
  }
  if (!schema || !doc) {
    return <div className="config-empty muted">Loading configuration...</div>;
  }

  return (
    <div className="config-tab">
      <div className="config-toolbar">
        <button disabled={doc.undo_depth === 0} onClick={undo} title={doc.undo_label || "Undo"}>
          Undo
        </button>
        <button disabled={doc.redo_depth === 0} onClick={redo} title={doc.redo_label || "Redo"}>
          Redo
        </button>
        <button className="save-btn" disabled={!doc.dirty} onClick={save}>
          Save
        </button>
        <span className={`dirty-dot ${doc.dirty ? "on" : ""}`} title={doc.dirty ? "Unsaved changes" : "Saved"} />
        <span className="config-validate-status">
          {validating ? "validating..." : diagnostics.length ? `${diagnostics.length} issue(s)` : "no issues"}
        </span>
        {error && <span className="config-inline-error">{error}</span>}
      </div>

      <div className="config-body">
        <div className="config-tree">
          <TreeItem
            label="Global"
            active={selection.kind === "global"}
            onClick={() => setSelection({ kind: "global" })}
          />
          {instancesByType.map(([comp, names]) => (
            <div key={comp} className="config-tree-group">
              <div className="config-tree-comp">{comp}</div>
              {names.map((name) => (
                <TreeItem
                  key={`${comp}.${name}`}
                  label={name || "(default)"}
                  indent
                  active={
                    selection.kind === "instance" &&
                    selection.comp === comp &&
                    selection.name === name
                  }
                  onClick={() => setSelection({ kind: "instance", comp, name })}
                />
              ))}
            </div>
          ))}
        </div>

        <div className="config-form">
          {selection.kind === "global" ? (
            <GlobalForm schema={schema} doc={doc} run={runCommand} />
          ) : (
            <InstanceForm
              schema={schema}
              doc={doc}
              comp={selection.comp}
              name={selection.name}
              diagnostics={diagnostics}
              run={runCommand}
              onRenamed={(newName) =>
                setSelection({ kind: "instance", comp: selection.comp, name: newName })
              }
            />
          )}
        </div>
      </div>

      {diagnostics.length > 0 && (
        <div className="problems-strip">
          {diagnostics.map((d, i) => (
            <div key={i} className={`problem sev-${d.severity}`}>
              <span className="problem-sev">{d.severity}</span>
              {d.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TreeItem({
  label,
  active,
  indent,
  onClick,
}: {
  label: string;
  active: boolean;
  indent?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className={`config-tree-item ${active ? "active" : ""} ${indent ? "indent" : ""}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

// Build [comp, [names...]] pairs from the config tree. List components list
// their named instances; singleton dict blocks get a single "" entry; scalar
// globals are excluded (they live under Global).
function buildTree(doc: DocState | null): Array<[string, string[]]> {
  if (!doc) return [];
  const out: Array<[string, string[]]> = [];
  for (const [key, value] of Object.entries(doc.config)) {
    if (Array.isArray(value)) {
      const names = value.map((e: any) =>
        e && typeof e === "object" ? String(e.name ?? "") : ""
      );
      out.push([key, names]);
    } else if (value && typeof value === "object") {
      out.push([key, [""]]);
    }
  }
  return out;
}

function instanceIndex(doc: DocState, comp: string, name: string): number {
  const block = doc.config[comp];
  if (!Array.isArray(block)) return 0;
  return block.findIndex((e: any) => String(e?.name ?? "") === name);
}

function GlobalForm({
  schema,
  doc,
  run,
}: {
  schema: Schema;
  doc: DocState;
  run: (c: DocCommand) => void;
}) {
  // Scalar top-level config keys the user edits directly (prefix, logger_level,
  // and any other non-component scalar keys present in the file).
  const scalarKeys = Object.entries(doc.config).filter(
    ([, v]) => v === null || typeof v !== "object"
  );
  return (
    <div>
      <h3>Global</h3>
      <table className="field-table">
        <tbody>
          {scalarKeys.map(([key, value]) => {
            const g = schema.global[key];
            const options = g && Array.isArray(g.accepts) ? g.accepts : null;
            return (
              <tr key={key}>
                <td className="field-name">{key}</td>
                <td>
                  {options ? (
                    <select
                      value={String(value ?? "")}
                      onChange={(e) =>
                        run({ op: "set_config_key", args: { path: key, value: e.target.value } })
                      }
                    >
                      {options.map((o) => (
                        <option key={String(o)} value={String(o)}>
                          {String(o)}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      defaultValue={String(value ?? "")}
                      onBlur={(e) =>
                        run({
                          op: "set_config_key",
                          args: { path: key, value: coerce(e.target.value) },
                        })
                      }
                    />
                  )}
                </td>
                <td className="field-doc muted">{g?.doc || ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function InstanceForm({
  schema,
  doc,
  comp,
  name,
  diagnostics,
  run,
  onRenamed,
}: {
  schema: Schema;
  doc: DocState;
  comp: string;
  name: string;
  diagnostics: Diagnostic[];
  run: (c: DocCommand) => void;
  onRenamed: (n: string) => void;
}) {
  const cs = schema.components[comp];
  const idx = instanceIndex(doc, comp, name);
  const block = doc.config[comp];
  const instance: any = Array.isArray(block) ? block[idx] : block;

  const diagFor = (paramPath: string) =>
    diagnostics.filter((d) => (d.param_paths || []).includes(paramPath));

  return (
    <div>
      <div className="instance-header">
        <h3>
          {comp}.{name || "(default)"}
        </h3>
        {Array.isArray(block) && (
          <input
            className="rename-input"
            defaultValue={name}
            title="Rename instance (rewrites cross-references)"
            onBlur={(e) => {
              const nv = e.target.value.trim();
              if (nv && nv !== name) {
                run({
                  op: "rename_instance",
                  args: { comp_type: comp, old_name: name, new_name: nv },
                });
                onRenamed(nv);
              }
            }}
          />
        )}
      </div>
      {cs?.doc && <p className="muted">{cs.doc}</p>}

      {/* Component-level config keys (files, refs, options). */}
      {cs && cs.config.length > 0 && (
        <section>
          <h4>Settings</h4>
          <table className="field-table">
            <tbody>
              {cs.config.map((ck) => (
                <ConfigKeyRow
                  key={ck.key}
                  ck={ck}
                  value={instance ? instance[ck.key] : undefined}
                  doc={doc}
                  onSet={(v) =>
                    run({
                      op: "set_config_key",
                      args: { path: `${comp}.${idx}.${ck.key}`, value: v },
                    })
                  }
                />
              ))}
            </tbody>
          </table>
        </section>
      )}

      {/* Per-parameter fields (initval / bounds / sigma) from params.yaml. */}
      {cs && Object.keys(cs.parameters).length > 0 && (
        <section>
          <h4>Parameters</h4>
          <table className="field-table param-table">
            <thead>
              <tr>
                <th>parameter</th>
                {PARAM_FIELDS.map((f) => (
                  <th key={f}>{f}</th>
                ))}
                <th>unit</th>
              </tr>
            </thead>
            <tbody>
              {Object.values(cs.parameters).map((p) => {
                const paramPath = `${comp}.${name}.${p.name}`;
                const entry = doc.params[paramPath] || {};
                const diags = diagFor(paramPath);
                return (
                  <tr key={p.name} className={diags.length ? "row-error" : ""}>
                    <td className="field-name" title={p.description || ""}>
                      {p.name}
                      {p.derived && <span className="derived-tag">derived</span>}
                    </td>
                    {PARAM_FIELDS.map((f) => (
                      <td key={f}>
                        <input
                          className="num-input"
                          defaultValue={entry[f] ?? ""}
                          placeholder={
                            f === "initval" && p.initval != null
                              ? String(p.initval)
                              : ""
                          }
                          key={`${paramPath}.${f}.${JSON.stringify(entry[f])}`}
                          onBlur={(e) => {
                            const raw = e.target.value.trim();
                            const value = raw === "" ? null : coerce(raw);
                            run({
                              op: "set_param_field",
                              args: { path: paramPath, field: f, value },
                            });
                          }}
                        />
                      </td>
                    ))}
                    <td className="field-unit muted">{p.unit || ""}</td>
                    {diags.length > 0 && (
                      <td className="inline-diag danger">{diags[0].message}</td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

function ConfigKeyRow({
  ck,
  value,
  doc,
  onSet,
}: {
  ck: ConfigKey;
  value: any;
  doc: DocState;
  onSet: (v: unknown) => void;
}) {
  let control: JSX.Element;
  if (ck.kind === "datafile") {
    // Read-only path with a Browse hook reserved for G9's file manager.
    control = (
      <div className="datafile-field">
        <input readOnly value={value ?? ""} placeholder="(no file)" title={String(value ?? "")} />
        <button className="browse-btn" disabled title="File browser arrives in G9">
          Browse
        </button>
      </div>
    );
  } else if (ck.kind === "ref") {
    const accepts = Array.isArray(ck.accepts) ? ck.accepts : [];
    const choices = refChoices(doc, accepts);
    if (Array.isArray(value)) {
      // Body-group / multi-value ref: comma-separated names.
      control = (
        <input
          defaultValue={value.join(", ")}
          placeholder={choices.join(", ")}
          onBlur={(e) =>
            onSet(
              e.target.value
                .split(",")
                .map((s) => s.trim())
                .filter(Boolean)
            )
          }
        />
      );
    } else {
      control = (
        <select value={String(value ?? "")} onChange={(e) => onSet(e.target.value)}>
          <option value="">(none)</option>
          {choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      );
    }
  } else if (Array.isArray(ck.accepts)) {
    control = (
      <select value={String(value ?? "")} onChange={(e) => onSet(coerce(e.target.value))}>
        <option value="">(default)</option>
        {ck.accepts.map((o) => (
          <option key={String(o)} value={String(o)}>
            {String(o)}
          </option>
        ))}
      </select>
    );
  } else {
    control = (
      <input
        defaultValue={value ?? ""}
        onBlur={(e) => onSet(e.target.value === "" ? null : coerce(e.target.value))}
      />
    );
  }
  return (
    <tr>
      <td className="field-name">
        {ck.key}
        {ck.required && <span className="req-tag">*</span>}
      </td>
      <td>{control}</td>
      <td className="field-doc muted">{ck.doc}</td>
    </tr>
  );
}

// Names of every instance of the accepted component types (for ref dropdowns).
function refChoices(doc: DocState, accepts: string[]): string[] {
  const out: string[] = [];
  for (const comp of accepts) {
    const block = doc.config[comp];
    if (Array.isArray(block)) {
      for (const e of block) {
        const nm = e && typeof e === "object" ? e.name : undefined;
        if (nm != null) out.push(String(nm));
      }
    }
  }
  return out;
}

// Turn a form string into a number when it parses cleanly, else keep the
// string (so bools/paths/names pass through unchanged).
function coerce(raw: string): unknown {
  if (raw === "") return "";
  if (raw === "true" || raw === "True") return true;
  if (raw === "false" || raw === "False") return false;
  const n = Number(raw);
  if (!Number.isNaN(n) && /^-?\d*\.?\d+(e-?\d+)?$/i.test(raw.trim())) return n;
  return raw;
}
