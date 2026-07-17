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

// --- run controls (G11) -----------------------------------------------------

export interface RunState {
  n_draws?: number;
  n_chains?: number;
  max_rhat?: number;
  min_ess?: number;
  elapsed_s?: number;
  stop_reason?: string | null;
}

export interface SnapshotMeta {
  n_draws?: number;
  n_kept?: number;
  max_rhat?: number | null;
  min_ess?: number | null;
  n_chains?: number;
  updated_at?: number;
}

export interface RunStatus {
  active: boolean;
  phase: string;
  state?: RunState;
  alive?: boolean;
  pid?: number;
  returncode?: number | null;
  error?: string | null;
  prefix?: string;
  config_path?: string;
  cwd?: string;
  log_path?: string;
  results_dir?: string;
  snapshot?: SnapshotMeta | null;
}

export interface RunPlots {
  start: string[];
  progress: string[];
}

// A single utility argument, straight from the argparse-derived schema.
export interface UtilityArg {
  name: string;
  type: string;
  default: unknown;
  required: boolean;
  choices: string[] | null;
  help: string;
}

export interface UtilitySchema {
  name: string;
  label: string;
  description: string;
  component_keys: string[];
  available: boolean;
  arguments: UtilityArg[];
}

export interface UtilityResult {
  returncode: number;
  produced_files: string[];
  output?: string;
  log_path?: string;
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
  utilities: () => getJson<Record<string, UtilitySchema>>("/api/utilities"),
  openProject: (path: string) =>
    postJson<ProjectListing>("/api/project/open", { path }),

  // Run controls (G11).
  startRun: (config: string, project_dir: string, params?: string | null) =>
    postJson<RunStatus>("/api/run", { config, project_dir, params: params ?? null }),
  runStatus: () => getJson<RunStatus>("/api/run/status"),
  stopRun: (force: boolean) => postJson<RunStatus>("/api/run/stop", { force }),
  runPlots: () => getJson<RunPlots>("/api/run/plots"),
  runUtility: (name: string, args: Record<string, unknown>, cwd: string) =>
    postJson<UtilityResult>("/api/utilities/run", { name, args, cwd }),
};

/** URL that serves a plot image from the active run's output directory. */
export function runImageUrl(path: string): string {
  return `/api/run/image?path=${encodeURIComponent(path)}`;
}

/** Open the log-tail WebSocket for a given file path on the server. */
export function openLogSocket(file: string): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${window.location.host}/api/logs?file=${encodeURIComponent(file)}`;
  return new WebSocket(url);
}
