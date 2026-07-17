"""Editable project document for the EXOZIPPy GUI.

A :class:`ProjectDocument` holds the two user-editable files -- the system
config (``*.yaml``) and the parameter-override file (``*.params.yaml``) -- as
ruamel.yaml ROUND-TRIP trees, so hand-written comments, key order, and
formatting survive GUI edits. All mutations go through :class:`Command`
objects with ``apply``/``revert`` so the GUI gets undo/redo for free.

Design notes
------------
- Booleans are wrapped in :class:`_PreservedBool` on load and re-emitted with
  their ORIGINAL spelling (``False`` stays ``False``, not ``false``). Combined
  with a fixed indent style this makes an unedited load->dump a byte-for-byte
  round trip on the example configs.
- Undo/redo is snapshot-based: every command captures a deep copy of both
  trees before it mutates them, so ``revert`` restores the exact prior state
  (comments included). This is more robust than hand-written inverse edits.
- This module is component-AGNOSTIC. The only component knowledge it uses is
  the declarative schema (``exozippy.introspect.full_schema``): specifically
  the ``kind: "ref"`` config keys, which tell it -- without any hardcoded
  component names -- which YAML keys reference which component type. That is
  how ``rename_instance`` rewrites cross-references generically.
"""

from __future__ import annotations

import copy
import io
import re
import time
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from ..linking import LINKABLE_FIELDS, is_link_expression

# --- YAML round-trip machinery ------------------------------------------------


class _PreservedBool(int):
    """A boolean that remembers its original YAML spelling.

    ruamel's round-trip constructor turns ``False`` into a plain ``bool`` and
    then dumps it lowercased (``false``), which would rewrite unedited lines.
    We keep the original text and represent it verbatim. It is an ``int``
    subclass (like ``bool``) so ``bool(x)`` and arithmetic behave normally.
    """

    def __new__(cls, truthy, text):
        obj = int.__new__(cls, 1 if truthy else 0)
        obj.text = text
        return obj

    def __bool__(self):
        return int(self) == 1

    def __reduce__(self):
        # int subclasses default to a reconstructor that passes only the int
        # value, dropping ``text``; teach copy/pickle the full signature.
        return (_PreservedBool, (int(self) == 1, self.text))

    def __deepcopy__(self, memo):
        return _PreservedBool(int(self) == 1, self.text)


def _construct_bool(constructor, node):
    text = constructor.construct_scalar(node)
    return _PreservedBool(text.lower() in ("true", "yes", "on"), text)


def _represent_bool(representer, data):
    return representer.represent_scalar("tag:yaml.org,2002:bool", data.text)


def make_yaml():
    """Return a YAML instance configured for lossless round-tripping."""
    yaml = YAML()  # round-trip is the default typ
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096  # do not wrap long scalars/comments
    yaml.constructor.add_constructor("tag:yaml.org,2002:bool", _construct_bool)
    yaml.representer.add_representer(_PreservedBool, _represent_bool)
    return yaml


_YAML = make_yaml()


def _load_yaml_text(text):
    return _YAML.load(text)


def _dump_yaml_text(tree):
    buf = io.StringIO()
    _YAML.dump(tree, buf)
    return buf.getvalue()


def _jsonable(node):
    """Convert a ruamel tree into plain JSON-serializable Python objects."""
    if isinstance(node, _PreservedBool):
        return bool(node)
    if isinstance(node, dict):
        return {str(k): _jsonable(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(v) for v in node]
    if isinstance(node, bool):
        return node
    # ruamel scalar subclasses (ScalarFloat/ScalarInt/str) are JSON-native.
    if isinstance(node, float):
        return float(node)
    if isinstance(node, int):
        return int(node)
    if node is None or isinstance(node, str):
        return node
    return str(node)


# --- schema-driven reference discovery ---------------------------------------

_SCHEMA_CACHE = None


def _default_schema():
    """Load and cache the component schema (heavy: imports every component)."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        from ..introspect import full_schema

        _SCHEMA_CACHE = full_schema()
    return _SCHEMA_CACHE


def _ref_keys_for(comp_type, schema):
    """Find every declared ref key that may point at ``comp_type``.

    Returns {owner_comp_type: [config_entry, ...]} where each config_entry is
    a ``kind: "ref"`` schema dict whose ``accepts`` list includes comp_type.
    Purely schema-driven -- no component names are hardcoded.
    """
    out = {}
    for owner, cschema in schema.get("components", {}).items():
        for entry in cschema.get("config", []) or []:
            if entry.get("kind") != "ref":
                continue
            if comp_type in (entry.get("accepts") or []):
                out.setdefault(owner, []).append(entry)
    return out


# --- tree helpers -------------------------------------------------------------


def _instances_of(config, comp_type):
    """Return the list of instance dicts for a component block (or [])."""
    block = config.get(comp_type)
    if isinstance(block, list):
        return [e for e in block if isinstance(e, dict)]
    if isinstance(block, dict):
        return [block]
    return []


def _find_instance(config, comp_type, name):
    """Return (list, index) of the named instance, or raise KeyError."""
    block = config.get(comp_type)
    if not isinstance(block, list):
        raise KeyError(f"'{comp_type}' is not a list component")
    for i, entry in enumerate(block):
        if isinstance(entry, dict) and str(entry.get("name")) == str(name):
            return block, i
    raise KeyError(f"no instance '{name}' in component '{comp_type}'")


def _set_nested(tree, path, value):
    """Set a dotted path into a config tree; integer segments index lists."""
    parts = path.split(".")
    node = tree
    for seg in parts[:-1]:
        if isinstance(node, list):
            node = node[int(seg)]
        elif seg in node:
            node = node[seg]
        else:
            child = CommentedMap()
            node[seg] = child
            node = child
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _get_nested(tree, path):
    node = tree
    for seg in path.split("."):
        if isinstance(node, list):
            node = node[int(seg)]
        else:
            node = node[seg]
    return node


def _rename_top_keys(cmap, rename_fn):
    """Rebuild a CommentedMap with top-level keys renamed, comments preserved."""
    new = CommentedMap()
    for key in list(cmap.keys()):
        new_key = rename_fn(key)
        new[new_key] = cmap[key]
    # Carry over per-key and block comments where the key survived.
    old_ca = getattr(cmap, "ca", None)
    if old_ca is not None:
        new.ca.comment = old_ca.comment
        for key, comment in old_ca.items.items():
            new.ca.items[rename_fn(key)] = comment
    return new


# --- commands -----------------------------------------------------------------


class Command:
    """A reversible edit. Subclasses implement ``_do``; revert uses a snapshot.

    ``label`` is a short human-readable description for the undo UI. ``apply``
    captures a full snapshot of both trees, runs the edit, and captures the
    result so ``revert`` and re-``apply`` (redo) restore exact states.
    """

    label = "edit"

    def apply(self, doc):
        self._before = doc._snapshot()
        self._do(doc)
        self._after = doc._snapshot()

    def revert(self, doc):
        doc._restore(self._before)

    def reapply(self, doc):
        doc._restore(self._after)

    def _do(self, doc):  # pragma: no cover - abstract
        raise NotImplementedError


class SetConfigKey(Command):
    def __init__(self, path, value):
        self.path = path
        self.value = value
        self.label = f"set {path}"

    def _do(self, doc):
        _set_nested(doc.config, self.path, doc._wrap(self.value))


class SetParamField(Command):
    def __init__(self, path, field, value):
        if field not in _PARAM_FIELDS:
            raise ValueError(
                f"'{field}' is not a settable param field {sorted(_PARAM_FIELDS)}"
            )
        self.path = path
        self.field = field
        self.value = value
        self.label = f"set {path}.{field}"

    def _do(self, doc):
        entry = doc.params.get(self.path)
        if not isinstance(entry, dict):
            entry = CommentedMap()
            doc.params[self.path] = entry
        if self.value is None:
            entry.pop(self.field, None)
            if len(entry) == 0:
                doc.params.pop(self.path, None)
        else:
            entry[self.field] = doc._wrap(self.value)


class AddComponentInstance(Command):
    def __init__(self, comp_type, name, fields=None):
        self.comp_type = comp_type
        self.name = name
        self.fields = fields or {}
        self.label = f"add {comp_type} '{name}'"

    def _do(self, doc):
        block = doc.config.get(self.comp_type)
        if not isinstance(block, list):
            block = CommentedSeq()
            doc.config[self.comp_type] = block
        for entry in block:
            if isinstance(entry, dict) and str(entry.get("name")) == str(self.name):
                raise ValueError(
                    f"{self.comp_type} already has an instance '{self.name}'"
                )
        entry = CommentedMap()
        entry["name"] = self.name
        for k, v in self.fields.items():
            entry[k] = doc._wrap(v)
        block.append(entry)


class DeleteInstance(Command):
    def __init__(self, comp_type, name):
        self.comp_type = comp_type
        self.name = name
        self.label = f"delete {comp_type} '{name}'"

    def _do(self, doc):
        block, idx = _find_instance(doc.config, self.comp_type, self.name)
        del block[idx]
        doc._drop_param_keys(self.comp_type, self.name)


class RenameInstance(Command):
    def __init__(self, comp_type, old_name, new_name):
        self.comp_type = comp_type
        self.old_name = old_name
        self.new_name = new_name
        self.label = f"rename {comp_type} '{old_name}' -> '{new_name}'"

    def _do(self, doc):
        if str(self.old_name) == str(self.new_name):
            return
        block, idx = _find_instance(doc.config, self.comp_type, self.old_name)
        for entry in block:
            if isinstance(entry, dict) and str(entry.get("name")) == str(self.new_name):
                raise ValueError(
                    f"{self.comp_type} already has an instance '{self.new_name}'"
                )
        # 1. the instance's own name field
        block[idx]["name"] = self.new_name
        # 2. cross-references in the system config (schema-driven, agnostic)
        doc._rewrite_refs(self.comp_type, self.old_name, self.new_name)
        # 3. params keys keyed by the old instance name
        doc._rename_param_keys(self.comp_type, self.old_name, self.new_name)
        # 4. link expressions inside params values
        doc._rewrite_param_links(self.comp_type, self.old_name, self.new_name)


class DuplicateInstance(Command):
    def __init__(self, comp_type, name, new_name):
        self.comp_type = comp_type
        self.name = name
        self.new_name = new_name
        self.label = f"duplicate {comp_type} '{name}' -> '{new_name}'"

    def _do(self, doc):
        block, idx = _find_instance(doc.config, self.comp_type, self.name)
        for entry in block:
            if isinstance(entry, dict) and str(entry.get("name")) == str(self.new_name):
                raise ValueError(
                    f"{self.comp_type} already has an instance '{self.new_name}'"
                )
        clone = copy.deepcopy(block[idx])
        clone["name"] = self.new_name
        block.append(clone)
        doc._copy_param_keys(self.comp_type, self.name, self.new_name)


class AssociateDatafile(Command):
    def __init__(self, comp_type, name, key, path):
        self.comp_type = comp_type
        self.name = name
        self.key = key
        self.path = path
        self.label = f"associate {key} on {comp_type} '{name}'"

    def _do(self, doc):
        block, idx = _find_instance(doc.config, self.comp_type, self.name)
        block[idx][self.key] = self.path


_PARAM_FIELDS = set(LINKABLE_FIELDS)

_COMMANDS = {
    "set_config_key": lambda a: SetConfigKey(a["path"], a["value"]),
    "set_param_field": lambda a: SetParamField(a["path"], a["field"], a["value"]),
    "add_component_instance": lambda a: AddComponentInstance(
        a["comp_type"], a["name"], a.get("fields")
    ),
    "delete_instance": lambda a: DeleteInstance(a["comp_type"], a["name"]),
    "rename_instance": lambda a: RenameInstance(
        a["comp_type"], a["old_name"], a["new_name"]
    ),
    "duplicate_instance": lambda a: DuplicateInstance(
        a["comp_type"], a["name"], a["new_name"]
    ),
    "associate_datafile": lambda a: AssociateDatafile(
        a["comp_type"], a["name"], a["key"], a["path"]
    ),
}


def command_from_json(payload):
    """Build a Command from a ``{"op": ..., "args": {...}}`` dict."""
    op = payload.get("op")
    if op not in _COMMANDS:
        raise ValueError(f"unknown command op '{op}'")
    return _COMMANDS[op](payload.get("args", {}))


# --- the document ------------------------------------------------------------


class ProjectDocument:
    """The two user files as editable, undoable, round-trip YAML trees."""

    def __init__(
        self,
        config,
        params=None,
        config_path=None,
        params_path=None,
        schema=None,
    ):
        self.config = config
        self.params = params if params is not None else CommentedMap()
        self.config_path = Path(config_path) if config_path else None
        self.params_path = Path(params_path) if params_path else None
        self._schema = schema
        self.dirty = False
        self.undo_stack = []
        self.redo_stack = []

    # -- construction ---------------------------------------------------------

    @classmethod
    def open(cls, config_path, params_path=None, schema=None):
        """Load a config file (and its params file) into a document.

        If ``params_path`` is None, the config's ``parameter_file`` key is
        used, resolved relative to the config file's directory.
        """
        config_path = Path(config_path)
        config = _load_yaml_text(config_path.read_text())
        if params_path is None:
            pf = config.get("parameter_file")
            if pf:
                params_path = config_path.parent / str(pf)
        params = CommentedMap()
        if params_path is not None and Path(params_path).exists():
            params = _load_yaml_text(Path(params_path).read_text())
        return cls(
            config,
            params,
            config_path=config_path,
            params_path=params_path,
            schema=schema,
        )

    # -- schema ---------------------------------------------------------------

    def schema(self):
        if self._schema is None:
            self._schema = _default_schema()
        return self._schema

    # -- snapshots / undo -----------------------------------------------------

    def _snapshot(self):
        # Snapshot via the serialized text, not deepcopy: ruamel's CommentedMap
        # deepcopy is fragile about re-attaching key comments, so an
        # undo-then-dump could silently drop a comment. The dumped text is the
        # canonical tree and reloads exactly (load->dump is identity here), so
        # text snapshots make undo byte-for-byte faithful.
        return (self.config_text(), self.params_text())

    def _restore(self, snap):
        self.config = _load_yaml_text(snap[0])
        self.params = _load_yaml_text(snap[1])

    def execute(self, command):
        command.apply(self)
        self.undo_stack.append(command)
        self.redo_stack.clear()
        self.dirty = True
        return command

    def undo(self):
        if not self.undo_stack:
            return None
        command = self.undo_stack.pop()
        command.revert(self)
        self.redo_stack.append(command)
        self.dirty = True
        return command

    def redo(self):
        if not self.redo_stack:
            return None
        command = self.redo_stack.pop()
        command.reapply(self)
        self.undo_stack.append(command)
        self.dirty = True
        return command

    # -- value wrapping -------------------------------------------------------

    def _wrap(self, value):
        """Wrap a plain Python bool so it round-trips with its spelling."""
        if isinstance(value, bool):
            return _PreservedBool(value, "True" if value else "False")
        return value

    # -- params key operations ------------------------------------------------

    def _drop_param_keys(self, comp_type, name):
        prefix = f"{comp_type}.{name}."
        for key in [k for k in self.params if str(k).startswith(prefix)]:
            del self.params[key]

    def _rename_param_keys(self, comp_type, old, new):
        prefix = f"{comp_type}.{old}."
        new_prefix = f"{comp_type}.{new}."

        def rename(key):
            k = str(key)
            return new_prefix + k[len(prefix):] if k.startswith(prefix) else key

        self.params = _rename_top_keys(self.params, rename)

    def _copy_param_keys(self, comp_type, name, new_name):
        prefix = f"{comp_type}.{name}."
        new_prefix = f"{comp_type}.{new_name}."
        additions = []
        for key in list(self.params.keys()):
            k = str(key)
            if k.startswith(prefix):
                additions.append((new_prefix + k[len(prefix):], self.params[key]))
        for new_key, value in additions:
            self.params[new_key] = copy.deepcopy(value)

    def _rewrite_param_links(self, comp_type, old, new):
        pat = re.compile(
            r"(?<![\w.])" + re.escape(comp_type) + r"\." + re.escape(str(old))
            + r"\.([A-Za-z_]\w*)"
        )

        def repl(m):
            return f"{comp_type}.{new}.{m.group(1)}"

        for entry in self.params.values():
            if not isinstance(entry, dict):
                continue
            for fld in LINKABLE_FIELDS:
                val = entry.get(fld)
                if isinstance(val, str) and is_link_expression(val, self.config):
                    new_val = pat.sub(repl, val)
                    if new_val != val:
                        entry[fld] = new_val

    # -- config reference rewriting (schema-driven) ---------------------------

    def _bare_unambiguous(self, comp_type, name, accepts):
        """A bare name is safe to rewrite only if no OTHER accepted component
        type also has an instance with that name."""
        for other in accepts:
            if other == comp_type:
                continue
            for entry in _instances_of(self.config, other):
                if str(entry.get("name")) == str(name):
                    return False
        return True

    def _rewrite_ref_value(self, value, comp_type, old, new, accepts):
        path_old = f"{comp_type}.{old}"
        path_new = f"{comp_type}.{new}"
        bare_ok = self._bare_unambiguous(comp_type, old, accepts)

        def one(v):
            if not isinstance(v, str):
                return v
            if v == path_old:
                return path_new
            if bare_ok and v == str(old):
                return str(new)
            return v

        if isinstance(value, list):
            for i in range(len(value)):
                value[i] = one(value[i])
            return value
        return one(value)

    def _rewrite_refs(self, comp_type, old, new):
        ref_map = _ref_keys_for(comp_type, self.schema())
        for owner, entries in ref_map.items():
            for inst in _instances_of(self.config, owner):
                for entry in entries:
                    key = entry["key"]
                    if key not in inst:
                        continue
                    accepts = entry.get("accepts") or []
                    inst[key] = self._rewrite_ref_value(
                        inst[key], comp_type, old, new, accepts
                    )

    # -- serialization --------------------------------------------------------

    def to_json(self):
        return {
            "config": _jsonable(self.config),
            "params": _jsonable(self.params),
            "config_path": str(self.config_path) if self.config_path else None,
            "params_path": str(self.params_path) if self.params_path else None,
            "dirty": self.dirty,
            "undo_depth": len(self.undo_stack),
            "redo_depth": len(self.redo_stack),
            "undo_label": self.undo_stack[-1].label if self.undo_stack else None,
            "redo_label": self.redo_stack[-1].label if self.redo_stack else None,
        }

    def config_text(self):
        return _dump_yaml_text(self.config)

    def params_text(self):
        return _dump_yaml_text(self.params)

    # -- saving / autosave ----------------------------------------------------

    def _autosave_path(self, path):
        path = Path(path)
        return path.parent / f".{path.stem}.autosave.yaml"

    def save(self, config_path=None, params_path=None):
        """Write both files, preserving comments/order, and clear autosaves."""
        cpath = Path(config_path) if config_path else self.config_path
        if cpath is None:
            raise ValueError("no config path to save to")
        cpath.write_text(self.config_text())
        self.config_path = cpath

        ppath = Path(params_path) if params_path else self.params_path
        if ppath is not None and len(self.params) > 0:
            ppath.write_text(self.params_text())
            self.params_path = ppath

        self._remove_autosave()
        self.dirty = False
        return cpath, ppath

    def autosave(self):
        """Write ``.<name>.autosave.yaml`` sidecars when dirty; else no-op.

        Returns the list of sidecar paths written (empty when not dirty).
        """
        if not self.dirty:
            return []
        written = []
        if self.config_path is not None:
            sp = self._autosave_path(self.config_path)
            sp.write_text(self.config_text())
            written.append(sp)
        if self.params_path is not None and len(self.params) > 0:
            sp = self._autosave_path(self.params_path)
            sp.write_text(self.params_text())
            written.append(sp)
        return written

    def _remove_autosave(self):
        for path in (self.config_path, self.params_path):
            if path is None:
                continue
            sp = self._autosave_path(path)
            if sp.exists():
                sp.unlink()

    def autosave_paths(self):
        out = []
        for path in (self.config_path, self.params_path):
            if path is not None:
                out.append(self._autosave_path(path))
        return out

    def autosave_recovery(self):
        """Report any autosave sidecar newer than its real file.

        Returns a list of {file, autosave, real_mtime, autosave_mtime}; empty
        when nothing recoverable exists. The GUI uses this on open to offer
        recovery.
        """
        recoverable = []
        for path in (self.config_path, self.params_path):
            if path is None:
                continue
            sp = self._autosave_path(path)
            if not sp.exists():
                continue
            real_mtime = path.stat().st_mtime if path.exists() else 0.0
            sp_mtime = sp.stat().st_mtime
            if sp_mtime > real_mtime:
                recoverable.append(
                    {
                        "file": str(path),
                        "autosave": str(sp),
                        "real_mtime": real_mtime,
                        "autosave_mtime": sp_mtime,
                    }
                )
        return recoverable


def now():
    """Wall-clock seconds (indirection makes autosave timing testable)."""
    return time.time()
