import { useEffect, useState } from "react";
import { api, type ProjectListing, type FileEntry, type DirListing } from "../api";

// Left sidebar: open a project directory and show its files grouped by kind.
// Selecting a data/log file bubbles up so the log terminal can tail it (any
// file works; the terminal simply follows it); selecting a config opens it in
// the Config tab (handled by the parent).
interface Props {
  listing: ProjectListing | null;
  onOpen: (path: string) => void;
  onSelectFile: (entry: FileEntry) => void;
  error: string | null;
}

// A server-side directory browser. Uses the same /api/files endpoint the Data
// tab uses, so it works identically whether the GUI runs in the native window
// or a plain browser tab (a native OS dialog would only work in the former).
function FolderPicker({
  start,
  onPick,
  onClose,
}: {
  start: string | null;
  onPick: (dir: string) => void;
  onClose: () => void;
}) {
  const [dir, setDir] = useState<DirListing | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = (path?: string | null) => {
    api
      .browse(path)
      .then((d) => {
        setDir(d);
        setError(null);
      })
      .catch((e) => setError(String(e instanceof Error ? e.message : e)));
  };

  useEffect(() => {
    load(start);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="folderpicker-backdrop" onClick={onClose}>
      <div className="folderpicker" onClick={(e) => e.stopPropagation()}>
        <div className="folderpicker-head">
          <span className="folderpicker-path" title={dir?.dir}>
            {dir?.dir ?? "..."}
          </span>
          <button className="folderpicker-close" onClick={onClose}>
            x
          </button>
        </div>
        {error && <div className="sidebar-error">{error}</div>}
        <ul className="folderpicker-list">
          {dir?.parent && (
            <li>
              <button className="folderpicker-row" onClick={() => load(dir.parent)}>
                <span className="kind-dot kind-dir" /> ..
              </button>
            </li>
          )}
          {dir?.entries
            .filter((e) => e.is_dir)
            .map((e) => (
              <li key={e.path}>
                <button className="folderpicker-row" onClick={() => load(e.path)}>
                  <span className="kind-dot kind-dir" /> {e.name}
                </button>
              </li>
            ))}
        </ul>
        <div className="folderpicker-actions">
          <button
            className="folderpicker-open"
            disabled={!dir}
            onClick={() => dir && onPick(dir.dir)}
          >
            Open this folder
          </button>
        </div>
      </div>
    </div>
  );
}

function FileGroup({
  label,
  files,
  onSelectFile,
}: {
  label: string;
  files: FileEntry[];
  onSelectFile: (e: FileEntry) => void;
}) {
  if (files.length === 0) return null;
  return (
    <div className="file-group">
      <div className="file-group-label">{label}</div>
      <ul>
        {files.map((f) => (
          <li key={f.path}>
            <button className="file-row" title={f.path} onClick={() => onSelectFile(f)}>
              <span className={`kind-dot kind-${f.kind}`} />
              {f.name}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Sidebar({ listing, onOpen, onSelectFile, error }: Props) {
  const [path, setPath] = useState("");
  const [picking, setPicking] = useState(false);

  return (
    <aside className="sidebar">
      <div className="sidebar-open">
        <input
          type="text"
          placeholder="project directory path"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && path.trim()) onOpen(path.trim());
          }}
        />
        <button onClick={() => path.trim() && onOpen(path.trim())}>Open</button>
        <button className="sidebar-browse" onClick={() => setPicking(true)}>
          Browse...
        </button>
      </div>
      {picking && (
        <FolderPicker
          start={listing?.dir ?? null}
          onClose={() => setPicking(false)}
          onPick={(d) => {
            setPicking(false);
            setPath(d);
            onOpen(d);
          }}
        />
      )}
      {error && <div className="sidebar-error">{error}</div>}
      {listing && (
        <div className="file-tree">
          <div className="file-tree-root" title={listing.dir}>
            {listing.dir}
          </div>
          <FileGroup label="Config" files={listing.configs} onSelectFile={onSelectFile} />
          <FileGroup label="Params" files={listing.params} onSelectFile={onSelectFile} />
          <FileGroup label="Data" files={listing.data_files} onSelectFile={onSelectFile} />
          <FileGroup label="Other" files={listing.other} onSelectFile={onSelectFile} />
        </div>
      )}
    </aside>
  );
}
