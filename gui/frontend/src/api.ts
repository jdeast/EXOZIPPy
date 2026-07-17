// Typed client for the FastAPI backend. All calls are relative to the current
// origin so the same bundle works served by FastAPI (production) or behind the
// Vite dev proxy.

import type { PlotSpec } from "./plotspec";
export type { PlotSpec } from "./plotspec";

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

// --- Tune tab: solve + live evaluator (G10) ---------------------------------

export interface Provenance {
  rank: number | null;
  label: "user" | "data" | "solved" | "default";
  relation: string | null;
}

export interface TuneParam {
  value: number | null;
  unit: string | null;
  internal_unit?: string | null;
  lower: number | null;
  upper: number | null;
  init_scale: number | null;
  sigma: number | null;
  mu: number | null;
  fixed: boolean;
  derived: boolean;
  provenance: Provenance;
}

export interface TuneStatus {
  phase: "idle" | "solving" | "compiling" | "live" | "error";
  error: string | null;
  structural_hash: string | null;
  has_result: boolean;
}

export interface TuneResult {
  parameters: Record<string, TuneParam>;
  seeds: Array<Record<string, number>> | null;
  plots: PlotSpec[];
}

// One eval response: updated model-trace y-arrays per plot, OR a signal that a
// full re-solve is required (linked/dynamic bounds, fixed/derived element).
export interface TuneEvalResult {
  plots?: Record<string, Record<string, (number | null)[]>>;
  needs_resolve?: boolean;
  out_of_bounds?: boolean;
  reason?: string;
}

export interface TuneHash {
  structural_hash: string;
  live_hash: string | null;
  stale: boolean;
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

  // Run controls (G11).
  startRun: (config: string, project_dir: string, params?: string | null) =>
    postJson<RunStatus>("/api/run", { config, project_dir, params: params ?? null }),
  runStatus: () => getJson<RunStatus>("/api/run/status"),
  stopRun: (force: boolean) => postJson<RunStatus>("/api/run/stop", { force }),
  runPlots: () => getJson<RunPlots>("/api/run/plots"),
  runUtility: (name: string, args: Record<string, unknown>, cwd: string) =>
    postJson<UtilityResult>("/api/utilities/run", { name, args, cwd }),

  // --- Tune tab (G10) ---
  tuneSolve: () => postJson<TuneStatus>("/api/tune/solve", {}),
  tuneStatus: () => getJson<TuneStatus>("/api/tune/status"),
  tuneResult: () => getJson<TuneResult>("/api/tune/result"),
  tuneEval: (path: string, value: number) =>
    postJson<TuneEvalResult>("/api/tune/eval", { path, value }),
  tuneHash: () => getJson<TuneHash>("/api/tune/hash"),
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
