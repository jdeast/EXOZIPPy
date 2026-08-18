import { useEffect, useState } from "react";
import type { DirListing } from "../api";

// The one server-side path browser. Two callers, one implementation: the
// sidebar picks a project DIRECTORY (unconfined `api.browse`, confirmed with a
// button) and the Config form picks a data FILE (project-rooted `api.files`,
// picked by clicking the row). Everything else -- the modal chrome, the parent
// row, navigation, error handling -- is identical, so it lives here once.
//
// It browses through the server rather than a native OS dialog so it works the
// same in the pywebview window and in a plain browser tab.
export interface PathPickerProps {
  /** Directory lister: `api.browse` (unconfined) or `api.files` (sandboxed). */
  list: (dir?: string | null) => Promise<DirListing>;
  /** Directory to list on mount; null lets the lister pick its own default. */
  start?: string | null;
  /**
   * What the picker returns. "file": file rows are shown and clicking one
   * picks it. "dir": only directories are listed and the confirm button picks
   * the directory currently being browsed.
   */
  select: "file" | "dir";
  /** Confirm-button label. Required by (and only used in) select="dir". */
  confirmLabel?: string;
  /**
   * Return the picked path relative to the FIRST directory listed. Config
   * datafile paths are relative to the project (= config) directory; a project
   * directory is absolute.
   */
  relativeToRoot?: boolean;
  onPick: (path: string) => void;
  onClose: () => void;
}

export default function PathPicker({
  list,
  start = null,
  select,
  confirmLabel,
  relativeToRoot = false,
  onPick,
  onClose,
}: PathPickerProps) {
  const [root, setRoot] = useState<string | null>(null);
  const [dir, setDir] = useState<DirListing | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = (path?: string | null) => {
    list(path)
      .then((d) => {
        setDir(d);
        setRoot((r) => r ?? d.dir);
        setError(null);
      })
      .catch((e) => setError(String(e instanceof Error ? e.message : e)));
  };

  useEffect(() => {
    load(start);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pick = (abs: string) =>
    onPick(
      relativeToRoot && root && abs.startsWith(root + "/")
        ? abs.slice(root.length + 1)
        : abs
    );

  const entries = (dir?.entries ?? []).filter(
    (e) => select === "file" || e.is_dir
  );

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
          {entries.map((e) => (
            <li key={e.path}>
              <button
                className="folderpicker-row"
                onClick={() => (e.is_dir ? load(e.path) : pick(e.path))}
              >
                <span className={`kind-dot ${e.is_dir ? "kind-dir" : "kind-data"}`} />
                {e.name}
              </button>
            </li>
          ))}
        </ul>
        {select === "dir" && (
          <div className="folderpicker-actions">
            <button
              className="folderpicker-open"
              disabled={!dir}
              onClick={() => dir && pick(dir.dir)}
            >
              {confirmLabel}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
