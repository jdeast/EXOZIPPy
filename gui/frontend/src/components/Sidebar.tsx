import { useState } from "react";
import type { ProjectListing, FileEntry } from "../api";

// Left sidebar: open a project directory and show its files grouped by kind.
// Selecting a data/log file bubbles up so the log terminal can tail it (any
// file works; the terminal simply follows it).
interface Props {
  listing: ProjectListing | null;
  onOpen: (path: string) => void;
  onSelectFile: (entry: FileEntry) => void;
  error: string | null;
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
      </div>
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
