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
};

/** Open the log-tail WebSocket for a given file path on the server. */
export function openLogSocket(file: string): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${window.location.host}/api/logs?file=${encodeURIComponent(file)}`;
  return new WebSocket(url);
}
