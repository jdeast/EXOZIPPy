"""Component-agnostic data-file association helpers for the GUI (G9).

These back the project file browser and the schema-driven association menu
(the Config form's Browse button; the Tools tab's "associate produced file"
affordance is still unwired -- see gui.md). Which instances a given file may be
associated with is decided SOLELY by the declarative schema
(:func:`exozippy.introspect.full_schema`): every component config key with
``kind == "datafile"`` carries a glob pattern in its ``accepts`` field, and a
file is eligible for an instance when its basename matches that glob. No
component names are hardcoded here -- add a component with a datafile key and
it participates automatically.

These helpers are pure functions over (filename, config-tree, schema) so they
are trivially unit-testable with a fake component schema.
"""

import fnmatch
import os
from pathlib import Path


def _datafile_keys(comp_schema):
    """Yield (key, glob, doc) for every datafile config entry of a component.

    ``accepts`` is normally a single glob string (e.g. ``"*.rv"``); a list of
    globs is tolerated so a component may accept several extensions.
    """
    for entry in comp_schema.get("config", []) or []:
        if entry.get("kind") != "datafile":
            continue
        accepts = entry.get("accepts")
        if isinstance(accepts, str):
            globs = [accepts]
        elif isinstance(accepts, (list, tuple)):
            globs = [g for g in accepts if isinstance(g, str)]
        else:
            globs = []
        for glob in globs:
            yield entry.get("key"), glob, entry.get("doc", "")


def named_instances(config, comp_type):
    """Return [(name, instance_dict), ...] for a component block in a config.

    Supports both the list form (``rvinstrument: [{name: HIRES, ...}]``) and
    the single-dict form (``sed: {file: ...}``, which has no name -> index 0).

    This is the ONE answer to "what are this component's instances" in the GUI:
    :mod:`exozippy.gui.document` builds its own (name-less) view on top of it
    rather than carrying a second copy that can drift.
    """
    block = config.get(comp_type)
    out = []
    if isinstance(block, list):
        for i, entry in enumerate(block):
            if isinstance(entry, dict):
                out.append((str(entry.get("name", i)), entry))
    elif isinstance(block, dict):
        out.append((str(block.get("name", 0)), block))
    return out


def _matches(filename, glob):
    """True if the basename of ``filename`` matches a shell glob (case-fold)."""
    base = os.path.basename(str(filename))
    return fnmatch.fnmatch(base.lower(), str(glob).lower())


def eligible_associations(filename, config, schema):
    """Return the (instance, datafile-key) pairs a file may be associated with.

    Purely schema-driven: for every component type that declares a datafile
    key, every instance of that type present in ``config`` becomes eligible
    when the file basename matches the key's glob.

    Returns a list of dicts, sorted for stable UI ordering::

        {"comp_type", "name", "key", "glob", "doc"}
    """
    out = []
    components = schema.get("components", {}) if schema else {}
    for comp_type, comp_schema in components.items():
        keys = list(_datafile_keys(comp_schema))
        if not keys:
            continue
        instances = named_instances(config or {}, comp_type)
        if not instances:
            continue
        for name, _inst in instances:
            for key, glob, doc in keys:
                if _matches(filename, glob):
                    out.append(
                        {
                            "comp_type": comp_type,
                            "name": name,
                            "key": key,
                            "glob": glob,
                            "doc": doc,
                        }
                    )
    out.sort(key=lambda e: (e["comp_type"], e["name"], e["key"]))
    return out


def current_associations(config, schema):
    """Map each associated data file to the instances that reference it.

    Reads the config tree for datafile-key values actually set on instances,
    keyed by the file's basename so the Data tab can render chips on each file
    row. Returns ``{basename: [{comp_type, name, key, path}, ...]}``.
    """
    out = {}
    components = schema.get("components", {}) if schema else {}
    for comp_type, comp_schema in components.items():
        keys = [k for (k, _g, _d) in _datafile_keys(comp_schema)]
        if not keys:
            continue
        for name, inst in named_instances(config or {}, comp_type):
            for key in keys:
                val = inst.get(key)
                if not isinstance(val, str) or not val:
                    continue
                base = os.path.basename(val)
                out.setdefault(base, []).append(
                    {
                        "comp_type": comp_type,
                        "name": name,
                        "key": key,
                        "path": val,
                    }
                )
    return out


def _sortable_children(d):
    """``sorted(d.iterdir())`` with directories first, tolerant of I/O errors.

    Both halves can raise: ``iterdir`` on a directory the server may not read,
    and ``is_dir`` inside the sort key on any child (a dangling symlink, a
    vanished entry, an unreadable mount point). Neither is a programming error
    -- a file browser walks whatever the filesystem hands it -- so an
    unreadable directory raises the same ``ValueError`` every other bad path
    here raises, and an unstattable child is simply dropped.
    """
    try:
        children = list(d.iterdir())
    except OSError as exc:
        raise ValueError(f"Cannot list directory: {d} ({exc.strerror})")

    def sort_key(p):
        try:
            is_dir = p.is_dir()
        except OSError:
            is_dir = False
        return (not is_dir, p.name.lower())

    return sorted(children, key=sort_key)


def list_directory(dirpath, root=None):
    """List a directory for the file browser, rooted at the project dir.

    Returns ``{dir, parent, entries}`` where each entry is
    ``{name, path, size, is_dir}``. Hidden entries (dotfiles) are skipped.

    ``root``, when given, is a real CONFINEMENT and not just a hint about the
    parent link: a ``dirpath`` outside it is refused outright. It used to gate
    only whether ``parent`` was reported, so ``GET /api/files?dir=/etc``
    happily listed ``/etc`` and the documented "sandboxed to the open project"
    invariant was cosmetic. ``parent`` is still None at the root so the
    browser cannot walk out of the tree one level at a time either.
    """
    d = Path(dirpath).expanduser().resolve()
    root_resolved = Path(root).expanduser().resolve() if root else None
    if root_resolved is not None and not is_within(d, root_resolved):
        raise ValueError(f"Path is outside the project directory: {d}")
    if not d.exists():
        raise ValueError(f"No such path: {d}")
    if not d.is_dir():
        raise ValueError(f"Not a directory: {d}")

    entries = []
    for child in _sortable_children(d):
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            size = None if is_dir else child.stat().st_size
        except OSError:
            continue
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "size": size,
                "is_dir": is_dir,
            }
        )

    parent = None
    if d.parent != d:
        if root_resolved is None or is_within(d.parent, root_resolved):
            parent = str(d.parent)

    return {"dir": str(d), "parent": parent, "entries": entries}


def is_within(path, root):
    """True if ``path`` is ``root`` or nested inside it (both fully resolved).

    The ONE path-containment predicate in the GUI. There used to be two --
    this one and an ``app._is_inside`` built on ``realpath`` + ``commonpath``
    -- which answered the same security question with different symlink
    behavior depending on which endpoint you asked.
    """
    if path is None or root is None:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False
