import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type AssociationMap,
  type DirEntry,
  type DirListing,
  type EligiblePair,
  type ProjectListing,
} from "../api";
import type { PlotSpec } from "../plotspec";
import PlotView from "./PlotView";

// G9 Data tab: browse project files, multi-select, right-click to associate a
// selection with the component instances whose schema datafile glob matches
// (menu built ENTIRELY from the schema -- no hardcoded component names), and
// preview a file's raw data. Associations are shown as chips per row and issued
// as undoable associate_datafile document commands.

interface DataTabProps {
  listing: ProjectListing | null;
  // The project's config file, so we can ensure a document is open before
  // issuing undoable associate_datafile commands (shared with the Config tab).
  configPath: string | null;
}

interface ContextMenu {
  x: number;
  y: number;
  files: DirEntry[];
  eligible: EligiblePair[];
}

// A single filesystem row: an eligible-file basename plus its current chips.
function chipsFor(name: string, assoc: AssociationMap) {
  return assoc[name] || [];
}

export default function DataTab({ listing, configPath }: DataTabProps) {
  const [cwd, setCwd] = useState<string | null>(null);
  const [dir, setDir] = useState<DirListing | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [assoc, setAssoc] = useState<AssociationMap>({});
  const [menu, setMenu] = useState<ContextMenu | null>(null);
  const [previewComp, setPreviewComp] = useState<string | null>(null);
  const [previewSpecs, setPreviewSpecs] = useState<PlotSpec[] | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastClicked = useRef<string | null>(null);

  // Follow the open project. The Data tab is rooted at the project dir.
  useEffect(() => {
    if (listing) setCwd(listing.dir);
  }, [listing]);

  const loadDir = useCallback(async (target?: string | null) => {
    try {
      setError(null);
      const d = await api.files(target ?? cwd);
      setDir(d);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, [cwd]);

  // Ensure a document is open so associate_datafile commands and associations
  // work even if the user never visited the Config tab. Shared server state, so
  // this is a no-op reuse when Config already opened it.
  const ensureDoc = useCallback(async () => {
    if (!configPath) return;
    try {
      await api.doc();
    } catch {
      try {
        await api.docOpen(configPath);
      } catch {
        // leave associations empty; the browser still works read-only
      }
    }
  }, [configPath]);

  const loadAssociations = useCallback(async () => {
    try {
      await ensureDoc();
      const res = await api.filesAssociations();
      setAssoc(res.associations);
    } catch {
      // No open document yet -> no associations. Not an error for the browser.
      setAssoc({});
    }
  }, [ensureDoc]);

  useEffect(() => {
    if (cwd) {
      loadDir(cwd);
      loadAssociations();
    }
  }, [cwd, loadDir, loadAssociations]);

  const entries = dir?.entries ?? [];

  const toggleSelect = (entry: DirEntry, shiftKey: boolean, metaKey: boolean) => {
    if (entry.is_dir) {
      setDir(null);
      setCwd(entry.path);
      return;
    }
    const path = entry.path;
    setSelected((prev) => {
      const next = new Set(metaKey || shiftKey ? prev : []);
      if (shiftKey && lastClicked.current) {
        // Range select over the current file entries.
        const files = entries.filter((e) => !e.is_dir).map((e) => e.path);
        const a = files.indexOf(lastClicked.current);
        const b = files.indexOf(path);
        if (a >= 0 && b >= 0) {
          const [lo, hi] = a < b ? [a, b] : [b, a];
          for (let i = lo; i <= hi; i++) next.add(files[i]);
        } else {
          next.add(path);
        }
      } else if (metaKey) {
        if (next.has(path)) next.delete(path);
        else next.add(path);
      } else {
        next.add(path);
      }
      return next;
    });
    lastClicked.current = path;
  };

  const openMenu = async (e: React.MouseEvent, entry: DirEntry) => {
    e.preventDefault();
    if (entry.is_dir) return;
    // Right-clicking a row that is not in the selection selects just that row.
    let files: DirEntry[];
    if (selected.has(entry.path)) {
      files = entries.filter((x) => !x.is_dir && selected.has(x.path));
    } else {
      setSelected(new Set([entry.path]));
      files = [entry];
    }
    // Eligibility is per file; intersect so the menu only offers targets that
    // accept EVERY selected file (schema-driven, no component names here).
    try {
      const perFile = await Promise.all(
        files.map((f) => api.filesEligible(f.name).then((r) => r.eligible)),
      );
      const key = (p: EligiblePair) => `${p.comp_type}.${p.name}.${p.key}`;
      let common = perFile[0] ?? [];
      for (const list of perFile.slice(1)) {
        const keys = new Set(list.map(key));
        common = common.filter((p) => keys.has(key(p)));
      }
      setMenu({ x: e.clientX, y: e.clientY, files, eligible: common });
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    }
  };

  // Issue associate_datafile commands (undoable via G8) for every selected file
  // against a chosen instance/key, then refresh chips and preview.
  const associate = async (pair: EligiblePair, files: DirEntry[]) => {
    setMenu(null);
    try {
      for (const f of files) {
        await api.docCommand({
          op: "associate_datafile",
          args: {
            comp_type: pair.comp_type,
            name: pair.name,
            key: pair.key,
            path: f.name,
          },
        });
      }
      await loadAssociations();
      runPreview(pair.comp_type);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  };

  const runPreview = useCallback(async (comp_type: string) => {
    setPreviewComp(comp_type);
    setPreviewSpecs(null);
    setPreviewError(null);
    setPreviewLoading(true);
    try {
      const res = await api.preview(comp_type);
      if (res.error) setPreviewError(res.error);
      else setPreviewSpecs(res.specs ?? []);
    } catch (e) {
      setPreviewError(String(e instanceof Error ? e.message : e));
    } finally {
      setPreviewLoading(false);
    }
  }, []);

  // Clicking an associated file's chip previews that component.
  const previewFile = (name: string) => {
    const refs = chipsFor(name, assoc);
    if (refs.length) runPreview(refs[0].comp_type);
  };

  const dismiss = useMemo(() => () => setMenu(null), []);
  useEffect(() => {
    if (!menu) return;
    window.addEventListener("click", dismiss);
    return () => window.removeEventListener("click", dismiss);
  }, [menu, dismiss]);

  if (!listing) {
    return <p className="run-hint muted">Open a project to browse its data files.</p>;
  }

  return (
    <div className="data-tab">
      <div className="data-browser">
        <div className="data-path">
          {dir?.parent && (
            <button className="crumb" onClick={() => { setDir(null); setCwd(dir.parent); }}>
              .. up
            </button>
          )}
          <span className="muted" title={dir?.dir}>{dir?.dir ?? cwd}</span>
        </div>
        {error && <p className="run-error">{error}</p>}
        <ul className="file-list">
          {entries.map((entry) => {
            const chips = entry.is_dir ? [] : chipsFor(entry.name, assoc);
            const isSel = selected.has(entry.path);
            return (
              <li
                key={entry.path}
                className={`file-row ${isSel ? "selected" : ""} ${entry.is_dir ? "is-dir" : ""}`}
                onClick={(e) => toggleSelect(entry, e.shiftKey, e.metaKey || e.ctrlKey)}
                onContextMenu={(e) => openMenu(e, entry)}
                onDoubleClick={() => !entry.is_dir && previewFile(entry.name)}
              >
                <span className="file-icon">{entry.is_dir ? "[dir]" : ""}</span>
                <span className="file-name">{entry.name}</span>
                <span className="file-chips">
                  {chips.map((c, i) => (
                    <span
                      key={i}
                      className="assoc-chip"
                      title={`${c.comp_type}.${c.name} (${c.key})`}
                      onClick={(e) => { e.stopPropagation(); runPreview(c.comp_type); }}
                    >
                      {c.comp_type}.{c.name}
                    </span>
                  ))}
                </span>
                {!entry.is_dir && entry.size != null && (
                  <span className="file-size">{entry.size} B</span>
                )}
              </li>
            );
          })}
        </ul>
        <p className="muted data-hint">
          Click to select (shift/ctrl for multi-select). Right-click to associate
          with a matching component instance. Double-click an associated file to
          preview its data.
        </p>
      </div>

      <div className="data-preview">
        {previewComp ? (
          <>
            <h3 className="preview-head">Preview: {previewComp}</h3>
            {previewLoading && <p className="muted">Loading preview...</p>}
            {previewError && (
              <pre className="preview-error">{previewError}</pre>
            )}
            {previewSpecs && previewSpecs.length === 0 && (
              <p className="muted">No plottable data for this component.</p>
            )}
            {previewSpecs &&
              previewSpecs.map((spec) => <PlotView key={spec.id} spec={spec} />)}
          </>
        ) : (
          <p className="muted preview-placeholder">
            Associate a file and it previews here, or double-click an associated
            file to catch a wrong-file mistake immediately.
          </p>
        )}
      </div>

      {menu && (
        <div
          className="context-menu"
          style={{ left: menu.x, top: menu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="context-title">
            Associate {menu.files.length > 1 ? `${menu.files.length} files` : menu.files[0].name}
          </div>
          {menu.eligible.length === 0 ? (
            <div className="context-empty muted">
              No matching component instance. Add one in the Config tab.
            </div>
          ) : (
            menu.eligible.map((pair) => (
              <button
                key={`${pair.comp_type}.${pair.name}.${pair.key}`}
                className="context-item"
                title={pair.doc}
                onClick={() => associate(pair, menu.files)}
              >
                {pair.comp_type}.{pair.name}
                <span className="context-key">{pair.key}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
