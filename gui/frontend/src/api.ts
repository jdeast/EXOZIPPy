// Typed client for the FastAPI backend. All calls are relative to the current
// origin so the same bundle works served by FastAPI (production) or behind the
// Vite dev proxy.

export interface FileEntry {
  name: string;
  path: string;
  size: number | null;
  kind: string;
}

export interface ProjectListing {
  dir: string;
  configs: FileEntry[];
  params: FileEntry[];
  data_files: FileEntry[];
  other: FileEntry[];
}

export interface GuiConfig {
  initial_project: string | null;
}

// A single config-editing document (system config + params files).
export interface DocState {
  config: Record<string, any>;
  params: Record<string, any>;
  config_path: string | null;
  params_path: string | null;
  dirty: boolean;
  undo_depth: number;
  redo_depth: number;
  undo_label: string | null;
  redo_label: string | null;
  recovery?: Array<{ file: string; autosave: string }>;
}

export interface Diagnostic {
  severity: string;
  message: string;
  param_paths: string[];
}

export interface ValidateJob {
  job_id: string;
  status: string;
  diagnostics: Diagnostic[];
}

// A command applied to the document. `op` names the edit; `args` are its
// parameters. The server is the single authority on which ops exist.
export interface DocCommand {
  op: string;
  args: Record<string, unknown>;
}

async function getJson<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: ${resp.status} ${resp.statusText}`);
  return (await resp.json()) as T;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error((data && data.error) || resp.statusText);
  return data as T;
}

export const api = {
  health: () => getJson<{ status: string }>("/api/health"),
  config: () => getJson<GuiConfig>("/api/config"),
  schema: () => getJson<Record<string, unknown>>("/api/schema"),
  utilities: () => getJson<Record<string, unknown>>("/api/utilities"),
  openProject: (path: string) =>
    postJson<ProjectListing>("/api/project/open", { path }),

  // --- config document (G8) ---
  docOpen: (config_path: string, params_path?: string | null) =>
    postJson<DocState>("/api/doc/open", { config_path, params_path }),
  doc: () => getJson<DocState>("/api/doc"),
  docCommand: (cmd: DocCommand) => postJson<DocState>("/api/doc/command", cmd),
  docUndo: () => postJson<DocState>("/api/doc/undo", {}),
  docRedo: () => postJson<DocState>("/api/doc/redo", {}),
  docSave: () => postJson<DocState>("/api/doc/save", {}),
  docValidateStart: () =>
    postJson<ValidateJob>("/api/doc/validate", {}),
  docValidatePoll: (jobId: string) =>
    getJson<ValidateJob>(`/api/doc/validate/${jobId}`),
};

/** Open the log-tail WebSocket for a given file path on the server. */
export function openLogSocket(file: string): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${window.location.host}/api/logs?file=${encodeURIComponent(file)}`;
  return new WebSocket(url);
}
