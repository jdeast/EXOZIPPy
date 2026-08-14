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

from ..config import canonical_param_key
from ..linking import LINKABLE_FIELDS, is_link_expression
from ..yamlio import check_yaml_booleans

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


def _load_user_yaml(path):
    """Load a user file into a round-trip tree, refusing ambiguous booleans.

    ruamel (YAML 1.2) reads ``finite_source: no`` as the truthy STRING "no"
    while the fit's PyYAML (YAML 1.1) reads it as ``False`` -- so the editor
    would show, and could save, the opposite of what the fit does. The shared
    ``exozippy.yamlio`` guard refuses those spellings for both paths; it is
    applied here, on the read from disk, and not in ``_load_yaml_text``, whose
    other caller is the undo/redo round trip of text this module itself
    dumped.
    """
    text = Path(path).read_text()
    check_yaml_booleans(text, source=str(path))
    return _load_yaml_text(text)


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


def _instances_of_raw(config, comp_type):
    """The component's list block VERBATIM (or []).

    Unlike ``_instances_of`` this filters nothing, so positions here are the
    element indices ``comp.<i>.param`` means -- dropping a stray non-dict
    entry would shift every index after it.
    """
    block = config.get(comp_type)
    return block if isinstance(block, list) else []


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
    """Rebuild a CommentedMap with top-level keys renamed, comments preserved.

    ``rename_fn`` returns the new key, or ``None`` to DROP the entry -- so one
    pass can rename and delete together, which is what retargeting a params
    file across an instance deletion needs (some keys go, some are respelled,
    most are left exactly as the user wrote them).
    """
    new = CommentedMap()
    dropped = set()
    for key in list(cmap.keys()):
        new_key = rename_fn(key)
        if new_key is None:
            dropped.add(key)
            continue
        new[new_key] = cmap[key]
    # Carry over per-key and block comments where the key survived.
    old_ca = getattr(cmap, "ca", None)
    if old_ca is not None:
        new.ca.comment = old_ca.comment
        for key, comment in old_ca.items.items():
            if key in dropped:
                continue
            new_key = rename_fn(key)
            if new_key is not None:
                new.ca.items[new_key] = comment
    return new


def _split_param_key(key, comp_type):
    """``(instance, param)`` for a 3-part params key of ``comp_type``, else None.

    The 2-part BROADCAST spelling (``star.teff``) deliberately returns None:
    it is not specific to any instance, so no instance-level edit may touch
    it -- it covers whatever elements remain, which is exactly what it said
    before.
    """
    parts = str(key).split(".")
    if len(parts) != 3 or parts[0] != comp_type:
        return None
    return parts[1], parts[2]


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
        try:
            self._do(doc)
        except Exception:
            # A command that raises PART WAY THROUGH must leave nothing
            # behind: it is not on the undo stack, so the user would have no
            # way to take back (say) an appended clone whose params copy then
            # refused. The snapshot restore we already own makes every
            # command atomic for free.
            doc._restore(self._before)
            raise
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
        # Update the entry the file already has for this element, under
        # whatever spelling the user chose -- never append a second one.
        key = doc.param_key_for(self.path)
        entry = doc.params.get(key)
        if not isinstance(entry, dict):
            entry = CommentedMap()
            doc.params[key] = entry
        if self.value is None:
            entry.pop(self.field, None)
            if len(entry) == 0:
                doc.params.pop(key, None)
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
            if isinstance(entry, dict) and str(entry.get("name")) == str(
                self.name
            ):
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
        # Retarget the params file FIRST. Every decision there needs the
        # PRE-deletion index of each entry's element, and the ``del`` below
        # destroys exactly that: afterwards `star.A.teff` resolves to nothing
        # and `star.0.teff` resolves to whichever star moved up.
        doc._retarget_params_for_delete(self.comp_type, self.name)
        block, idx = _find_instance(doc.config, self.comp_type, self.name)
        del block[idx]


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
            if isinstance(entry, dict) and str(entry.get("name")) == str(
                self.new_name
            ):
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
            if isinstance(entry, dict) and str(entry.get("name")) == str(
                self.new_name
            ):
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
    "set_param_field": lambda a: SetParamField(
        a["path"], a["field"], a["value"]
    ),
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
        config = _load_user_yaml(config_path)
        if params_path is None:
            pf = config.get("parameter_file")
            if pf:
                params_path = config_path.parent / str(pf)
        params = CommentedMap()
        if params_path is not None and Path(params_path).exists():
            params = _load_user_yaml(Path(params_path))
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

    def param_key_for(self, path):
        """The spelling THIS params file already uses for ``path``'s element.

        The GUI addresses parameters by the NAME form (`star.A.teff`) because
        that is what `introspect` and `export_solution` display, while a
        params file may equally well spell the same element in the INDEX form
        (`star.0.teff`) -- `examples/kelt4/kelt4.params.yaml`, the GUI test
        fixture, does exactly that.  A literal `params[path]` write then
        APPENDS a twin instead of updating the entry, and the two spellings
        are equally specific, so nothing downstream can adjudicate them:
        `ConfigManager` now refuses such a file outright, and before it did
        it silently kept whichever key came LAST -- the GUI's -- discarding
        the user's entire original entry, `sigma` prior included.

        So resolve to the existing key and edit in place.  Only the two
        SPECIFIC spellings are matched: a 2-part broadcast entry is a
        different, coarser statement, and refining one element of it with a
        specific entry is the legitimate "most specific wins" idiom, not a
        duplicate.  With no existing entry the caller's own spelling is used,
        so a params file written by the GUI alone stays in the name form.
        """
        if path in self.params:
            return path
        if len(str(path).split(".")) != 3:
            return path
        canon = canonical_param_key(str(path), self.config)
        for key in self.params:
            if len(str(key).split(".")) != 3:
                continue
            if canonical_param_key(str(key), self.config) == canon:
                return key
        return path

    def _instance_indices(self, comp_type):
        """``(names_by_index, index_by_name)`` for a list component.

        Only names that can actually be written back into a params key are
        listed: a non-string or all-digit ``name:`` cannot serve as the NAME
        spelling (``validate_instance_names`` bans both outright), and an
        instance with no ``name:`` at all has only its index.
        """
        names_by_index = {}
        index_by_name = {}
        for i, entry in enumerate(_instances_of_raw(self.config, comp_type)):
            if not isinstance(entry, dict):
                continue
            nm = entry.get("name")
            if not isinstance(nm, str) or nm.isdigit():
                continue
            names_by_index[i] = nm
            index_by_name.setdefault(nm, i)
        return names_by_index, index_by_name

    @staticmethod
    def _element_index(instance, index_by_name, n_inst):
        """Which instance INDEX an ``<instance>`` path segment addresses.

        Returns ``(index, is_index_form)``, or ``(None, ...)`` when the
        spelling names no instance this config defines (an orphan entry left
        over from an earlier edit -- not ours to retarget).
        """
        if instance.isdigit():
            idx = int(instance)
            return (idx if idx < n_inst else None), True
        return index_by_name.get(instance), False

    def _retarget_params_for_delete(self, comp_type, name):
        """Retarget the params file across the deletion of one instance.

        MUST run while the instance is still in the config -- see the comment
        at the ``DeleteInstance`` call site.

        Three kinds of entry, three answers:

        * **The deleted instance's own entries go, under BOTH specific
          spellings.** ``star.A.teff`` and ``star.0.teff`` name the same
          element; a name-prefix scan saw only the first, so every index-form
          entry survived the delete and then silently applied to whichever
          star moved up. That is a WRONG-ELEMENT bug, not litter.
        * **A survivor's index-form entry whose index shifts is rewritten to
          the NAME form** (``star.1.teff`` -> ``star.B.teff`` when B moves
          from index 1 to 0). The name form means the same element before and
          after any list mutation, so this converts a fragile spelling into a
          stable one exactly where the fragility would otherwise bite.
          Re-indexing would fix today's delete and leave the same trap set
          for the next one; refusing the delete would gate an edit the GUI
          can repair exactly. An UNNAMED instance has no name form, so its
          keys are re-indexed instead -- the same guarantee by the only means
          available.
        * **Everything else is left byte-identical**: a survivor at an index
          BELOW the deleted one still spells its own element correctly, the
          name form was never index-dependent, and the 2-part BROADCAST form
          (``star.teff``) was never specific to the deleted instance.

        Link EXPRESSIONS inside the surviving entries get the same treatment
        (they are parameter references too, and an index-form one retargets
        just as silently), with one difference: a reference to the DELETED
        instance is rewritten to its name form rather than removed, turning a
        silent mis-address into the loud "no instance named 'A'" the name
        form has always produced.
        """
        block, idx = _find_instance(self.config, comp_type, name)
        n_inst = len(block)
        names_by_index, index_by_name = self._instance_indices(comp_type)

        def respelled(i, is_index_form):
            """New instance segment, or None when the spelling still holds."""
            if not is_index_form or i < idx:
                return None
            nm = names_by_index.get(i)
            return nm if nm is not None else str(i - 1)

        def rewrite_key(key):
            split = _split_param_key(key, comp_type)
            if split is None:
                return key
            instance, param = split
            i, is_index_form = self._element_index(
                instance, index_by_name, n_inst
            )
            if i is None:
                return key
            if i == idx:
                return None  # the deleted instance, under either spelling
            new_instance = respelled(i, is_index_form)
            if new_instance is None:
                return key
            return f"{comp_type}.{new_instance}.{param}"

        plan = {str(k): rewrite_key(k) for k in self.params}
        self._reject_colliding_rewrites(plan)
        self.params = _rename_top_keys(self.params, rewrite_key)

        def rewrite_ref(instance, param):
            i, is_index_form = self._element_index(
                instance, index_by_name, n_inst
            )
            if i is None:
                return None
            if i == idx:
                # Dangling either way now; spell it so it FAILS rather than
                # quietly addressing the instance that took this index.
                return str(name) if is_index_form else None
            return respelled(i, is_index_form)

        self._rewrite_param_link_paths(comp_type, rewrite_ref)

    def _reject_colliding_rewrites(self, plan):
        """Refuse a retarget that would leave one element spelled twice.

        Only reachable from a params file that ALREADY names one element
        under both specific spellings -- which ``ConfigManager`` refuses
        outright (see CLAUDE.md's "Parameter naming convention"), so the
        collision is a pre-existing fault this edit would otherwise bury.
        """
        landed = {}
        for src, dst in plan.items():
            if dst is None:
                continue
            prev = landed.get(dst)
            if prev is not None:
                raise ValueError(
                    f"cannot retarget the params file: '{prev}' and '{src}' "
                    f"would both become '{dst}', i.e. they are two spellings "
                    f"of one parameter element. Keep exactly one of them "
                    f"(merging any fields you need from both) and retry."
                )
            landed[dst] = src

    def _copy_param_keys(self, comp_type, name, new_name):
        """Copy one instance's param entries onto its freshly added duplicate.

        The mirror image of the delete case, and it was wrong the same way:
        a name-prefix scan reads only the NAME spelling, so duplicating an
        instance whose entries the file spells by INDEX produced a clone with
        none of its parameters -- silently, since a missing entry is a legal
        params file.

        Entries are read under both specific spellings and written under the
        clone's NAME form. Never its index form: that is the fragile spelling
        ``_retarget_params_for_delete`` exists to remove, correct only until
        someone deletes an earlier instance. And exactly one spelling per
        element -- naming one element twice is fatal in ``ConfigManager``.
        """
        block, idx = _find_instance(self.config, comp_type, name)
        _, new_idx = _find_instance(self.config, comp_type, new_name)
        n_inst = len(block)
        _, index_by_name = self._instance_indices(comp_type)

        source = {}
        for key in list(self.params.keys()):
            split = _split_param_key(key, comp_type)
            if split is None:
                continue
            instance, param = split
            i, _ = self._element_index(instance, index_by_name, n_inst)
            if i == idx:
                if param in source:
                    raise ValueError(
                        f"cannot duplicate '{name}': its '{param}' is named "
                        f"twice in the params file ('{source[param][0]}' and "
                        f"'{key}' are two spellings of one element). Keep "
                        f"exactly one of them and retry."
                    )
                source[param] = (str(key), self.params[key])
            elif i == new_idx:
                raise ValueError(
                    f"cannot duplicate '{name}' as '{new_name}': the params "
                    f"file already has '{key}' for the new instance. Remove "
                    f"or rename that entry and retry."
                )

        for param, (_, value) in source.items():
            self.params[f"{comp_type}.{new_name}.{param}"] = copy.deepcopy(
                value
            )

    def _rename_param_keys(self, comp_type, old, new):
        prefix = f"{comp_type}.{old}."
        new_prefix = f"{comp_type}.{new}."

        def rename(key):
            k = str(key)
            return (
                new_prefix + k[len(prefix) :] if k.startswith(prefix) else key
            )

        self.params = _rename_top_keys(self.params, rename)

    def _rewrite_param_links(self, comp_type, old, new):
        # An INDEX-form reference to the renamed instance still addresses it
        # (a rename moves nothing), so only the name form is rewritten.
        self._rewrite_param_link_paths(
            comp_type,
            lambda instance, param: str(new) if instance == str(old) else None,
        )

    def _rewrite_param_link_paths(self, comp_type, rewrite_ref):
        """Rewrite ``comp.<instance>.<param>`` references inside link values.

        ``rewrite_ref(instance, param)`` returns the new instance segment, or
        None to leave that reference exactly as written. The character
        classes are ``linking._PATH_3``'s, so this sees precisely the
        references the link parser will -- index form included.
        """
        pat = re.compile(
            r"(?<![\w.])"
            + re.escape(comp_type)
            + r"\.([A-Za-z0-9_][A-Za-z0-9_\-]*)\.([A-Za-z_]\w*)(?![\w.(])"
        )

        def repl(m):
            new_instance = rewrite_ref(m.group(1), m.group(2))
            if new_instance is None:
                return m.group(0)
            return f"{comp_type}.{new_instance}.{m.group(2)}"

        for entry in self.params.values():
            if not isinstance(entry, dict):
                continue
            for fld in LINKABLE_FIELDS:
                val = entry.get(fld)
                if isinstance(val, str) and is_link_expression(
                    val, self.config
                ):
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
            "undo_label": (
                self.undo_stack[-1].label if self.undo_stack else None
            ),
            "redo_label": (
                self.redo_stack[-1].label if self.redo_stack else None
            ),
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
