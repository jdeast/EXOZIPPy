"""Shared loading of the user-facing YAML files (config + params).

Why this module exists: the fit and the GUI parse the SAME two files with
DIFFERENT YAML libraries, and the two libraries disagree about booleans.

  * The fit reads them with PyYAML (``yaml.safe_load``), which implements
    YAML **1.1**: ``yes/Yes/YES/no/No/NO/on/On/ON/off/Off/OFF`` all resolve
    to ``bool``.
  * The GUI reads them with ruamel.yaml's round-trip loader (a documented
    invariant -- it must preserve comments and key order on save), which
    implements YAML **1.2**: those same twelve spellings resolve to ``str``.

So ``finite_source: no`` is ``False`` to the fit and the truthy string
``"no"`` to the GUI: the GUI shows -- and may write out -- the OPPOSITE of
what the fit will do. Only ``true/True/TRUE/false/False/FALSE`` mean the same
thing to both loaders; that intersection is ``ACCEPTED_BOOLEANS``, and it is
derived from what the two loaders actually resolve, not from taste.

Rather than pick a winner (either choice leaves one of the two paths reading
something the user did not write), this module refuses the ambiguous
spellings outright: ``check_yaml_booleans`` raises ``AmbiguousBooleanError``
naming the key, the offending spelling, and the accepted form. Both paths
call it -- the fit through ``load_yaml`` (cli, System's params read), the GUI
through ``ProjectDocument``'s ruamel load -- so neither can accept a file the
other would read differently.

The test is PyYAML's own implicit resolver: a plain scalar PyYAML tags
``tag:yaml.org,2002:bool`` whose text is not in ``ACCEPTED_BOOLEANS`` is
exactly a spelling the two libraries disagree about. No list of "which keys
are boolean" is maintained anywhere, so nothing can drift; quoting the value
(``"no"``) is the escape hatch when a string really was meant, because a
quoted scalar is a string to both libraries.
"""

import yaml

_BOOL_TAG = "tag:yaml.org,2002:bool"

ACCEPTED_BOOLEANS = ("true", "True", "TRUE", "false", "False", "FALSE")
"""Boolean spellings PyYAML (YAML 1.1) and ruamel (YAML 1.2) both accept."""


class AmbiguousBooleanError(ValueError):
    """A YAML file spells a boolean in a way the two loaders disagree about."""


def _join_path(parts):
    """Join key/index path parts into ``a.b[0].c``."""
    out = ""
    for part in parts:
        if part.startswith("["):
            out += part
        elif out:
            out += "." + part
        else:
            out = part
    return out or "<root>"


def _is_ambiguous(node):
    return (
        isinstance(node, yaml.ScalarNode)
        and node.tag == _BOOL_TAG
        and node.value not in ACCEPTED_BOOLEANS
    )


def _scan(node, path, out):
    """Collect (path, node) for every ambiguously spelled boolean below node."""
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            name = str(getattr(key_node, "value", "?"))
            if _is_ambiguous(key_node):
                out.append((path + [name], key_node))
            _scan(value_node, path + [name], out)
    elif isinstance(node, yaml.SequenceNode):
        for i, item in enumerate(node.value):
            _scan(item, path + [f"[{i}]"], out)
    elif _is_ambiguous(node):
        out.append((path, node))


def _boolean_key_note(key):
    """'is a boolean option of component x' note, or '' -- best effort.

    Driven off ``Component.config_schema()`` (an entry declares a boolean key
    with ``"accepts": [True, False]``), so the note follows the components and
    is never a hand-maintained list. Only ever used to sharpen an error
    message, and only on the raising path: import failures are swallowed so a
    lightweight caller (the GUI runner imports no components) still gets the
    error, just without the extra clause.
    """
    try:
        from .introspect import boolean_option_keys

        owners = boolean_option_keys().get(key)
    except Exception:  # pragma: no cover - defensive; message-only
        return ""
    if not owners:
        return ""
    return f"  <- a boolean option of {', '.join(sorted(owners))}"


def check_yaml_booleans(text, source=None):
    """Raise if `text` spells any boolean in a YAML-1.1-only way.

    `source` is the file name used in the message. Returns None. A file that
    is not valid YAML raises the underlying ``yaml.YAMLError``; callers that
    must stay tolerant of unreadable files should keep their own try/except.
    """
    offenders = []
    for doc in yaml.compose_all(text, Loader=yaml.SafeLoader):
        if doc is not None:
            _scan(doc, [], offenders)
    if not offenders:
        return

    where = f"{source}: " if source else ""
    lines = [
        f"{where}YAML 1.1 boolean spelling(s) that the fit and the GUI "
        f"read differently:"
    ]
    for path, node in offenders:
        lines.append(
            f"  line {node.start_mark.line + 1}: "
            f"{_join_path(path)}: {node.value}"
            f"{_boolean_key_note(path[-1] if path else '')}"
        )
    lines.append(
        "Write booleans as one of: " + ", ".join(ACCEPTED_BOOLEANS) + "."
    )
    lines.append(
        'If you meant the string, quote it (e.g. seed_polish: "on") -- a '
        "quoted scalar is a string to both loaders."
    )
    lines.append(
        "Reason: the fit parses this file with PyYAML (YAML 1.1), where "
        "yes/no/on/off are booleans; the GUI parses it with ruamel.yaml "
        "(YAML 1.2), where they are strings."
    )
    raise AmbiguousBooleanError("\n".join(lines))


def load_yaml_text(text, source=None):
    """``yaml.safe_load`` the text after the ambiguous-boolean check."""
    check_yaml_booleans(text, source=source)
    return yaml.safe_load(text)


def load_yaml(path):
    """Load a user YAML file, refusing YAML-1.1-only boolean spellings."""
    with open(path, "r") as fh:
        text = fh.read()
    return load_yaml_text(text, source=str(path))
