"""Component-owned utility declarations and a headless runner.

A "utility" is a user-facing helper program that logically belongs to a
component (getdata for transit light curves, mkticsed for an SED, etc.). A
component declares its utilities by overriding ``Component.get_utilities()``
to return a list of :class:`UtilitySpec`. The GUI (and any other consumer)
discovers them generically -- no component names are hardcoded outside the
component-owned code.

The design keeps utilities definable by argparse alone: an argparse parser is
the single source of truth for a utility's arguments, and
:func:`parser_to_schema` converts it into a JSON-serializable argument schema
so a browser can render an input form without importing argparse semantics.

Public API
----------
UtilitySpec            -- dataclass declaring one utility
parser_to_schema(p)    -- argparse.ArgumentParser -> JSON-serializable arg list
argparse_subprocess_runner(module) -- build a ``run`` callable for a module
all_utilities()        -- {name: UtilitySpec} gathered from every component
run_utility(name, ...) -- execute a utility headless, report produced files
"""

import argparse
import io
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


# --- the declaration ----------------------------------------------------------

@dataclass
class UtilitySpec:
    """Declares one component-owned utility program.

    Attributes
    ----------
    name : str
        Stable, unique machine identifier (e.g. "getdata").
    label : str
        Short human-readable name for a GUI menu (e.g. "Download TESS/Kepler").
    description : str
        One-line explanation of what the utility does.
    component_keys : list[str]
        yaml_keys of the components that surface this utility.
    available : bool
        True when the utility is implemented and runnable; False marks a
        placeholder the GUI can show as "coming soon" (build_parser/run may
        be None).
    build_parser : callable or None
        Zero-argument callable returning an argparse.ArgumentParser. Used both
        to render the argument schema and, for the default subprocess runner,
        to marshal an args dict into a command line.
    run : callable or None
        ``run(args_dict, cwd) -> dict`` executing the utility. The dict should
        carry at least a "returncode"; run_utility adds "produced_files". None
        for placeholders.
    """

    name: str
    label: str
    description: str
    component_keys: List[str] = field(default_factory=list)
    available: bool = True
    build_parser: Optional[Callable[[], argparse.ArgumentParser]] = None
    run: Optional[Callable[[dict, str], dict]] = None

    def argument_schema(self):
        """JSON-serializable argument schema for this utility (or [])."""
        if self.build_parser is None:
            return []
        return parser_to_schema(self.build_parser())

    def to_schema(self):
        """Full JSON-serializable description of this utility for the GUI."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "component_keys": list(self.component_keys),
            "available": bool(self.available),
            "arguments": self.argument_schema(),
        }


# --- argparse -> JSON-serializable schema -------------------------------------

def _type_name(action):
    """Map an argparse action to a JSON-friendly type string."""
    # store_true / store_false take no argument -> boolean flag.
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"
    t = action.type
    if t is int:
        return "int"
    if t is float:
        return "float"
    # Everything else (including None and str) presents to the GUI as text.
    return "str"


def _arg_name(action):
    """Public argument name: dest for positionals, long option otherwise."""
    if not action.option_strings:
        return action.dest
    # Prefer a long option ("--sedfile") over a short one ("-s").
    for opt in action.option_strings:
        if opt.startswith("--"):
            return opt
    return action.option_strings[0]


def _is_required(action):
    """Whether the GUI must collect a value for this action."""
    if action.option_strings:
        return bool(action.required)
    # Positional: required unless it accepts zero occurrences.
    return action.nargs not in ("?", "*")


def parser_to_schema(parser):
    """Convert an argparse.ArgumentParser into a JSON-serializable arg list.

    Each entry is a dict with keys: name, type, default, required, choices,
    help. The help action is skipped. The result survives json.dumps().
    """
    schema = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        choices = list(action.choices) if action.choices is not None else None
        default = action.default
        if default is argparse.SUPPRESS:
            default = None
        schema.append({
            "name": _arg_name(action),
            "type": _type_name(action),
            "default": default,
            "required": _is_required(action),
            "choices": choices,
            "help": action.help or "",
        })
    return schema


# --- turning an args dict into a command line ---------------------------------

def args_dict_to_argv(parser, args_dict):
    """Marshal a plain args dict into an argv list the parser accepts.

    Positionals are emitted first (in declaration order), then options.
    store_true/store_false flags are emitted only when truthy. Keys may be
    given with or without leading dashes; unknown keys are ignored.
    """
    def lookup(action):
        name = _arg_name(action)
        if name in args_dict:
            return args_dict[name], True
        if action.dest in args_dict:
            return args_dict[action.dest], True
        return None, False

    positionals = []
    optionals = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        value, present = lookup(action)
        if not present:
            continue
        if not action.option_strings:
            if value is not None:
                positionals.append(str(value))
            continue
        opt = _arg_name(action)
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if value:
                optionals.append(opt)
        elif value is not None:
            optionals.extend([opt, str(value)])
    return positionals + optionals


def argparse_subprocess_runner(module_name):
    """Build a ``run(args_dict, cwd)`` that runs a utility module as a subprocess.

    The module must be runnable as ``python -m <module_name>`` (i.e. expose a
    ``main()`` under ``if __name__ == '__main__'``). Running in a subprocess
    isolates the host from utilities that call sys.exit or print heavily, and
    makes the working directory unambiguous for produced-file detection.
    """
    def run(args_dict, cwd):
        # Import lazily so schema/introspection never imports the module.
        from importlib import import_module
        module = import_module(module_name)
        argv = args_dict_to_argv(module.build_parser(), args_dict)
        proc = subprocess.run(
            [sys.executable, "-m", module_name, *argv],
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        return {
            "returncode": proc.returncode,
            "output": proc.stdout + proc.stderr,
        }
    return run


def inprocess_runner(func):
    """Build a ``run(args_dict, cwd)`` that calls ``func(args_dict, cwd)`` in-process.

    Captures stdout/stderr and chdir's into cwd for the call. Intended for
    lightweight, well-behaved utilities (and test fakes) that neither call
    sys.exit nor spawn a heavy import stack.
    """
    def run(args_dict, cwd):
        buf = io.StringIO()
        prev = os.getcwd()
        returncode = 0
        try:
            os.chdir(str(cwd))
            with redirect_stdout(buf), redirect_stderr(buf):
                rc = func(dict(args_dict), str(cwd))
            if isinstance(rc, int):
                returncode = rc
        except SystemExit as e:
            returncode = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
        finally:
            os.chdir(prev)
        return {"returncode": returncode, "output": buf.getvalue()}
    return run


# --- discovery ----------------------------------------------------------------

def all_utilities():
    """Gather every component-declared utility, keyed by unique name.

    Returns {name: UtilitySpec}. Discovery is generic: it asks every
    component class for ``get_utilities()``; nothing here knows component
    names. Duplicate names raise (utility names must be globally unique).
    """
    # Lazy import to avoid an import cycle (components import this module for
    # UtilitySpec) and to keep the heavy component stack out of light imports.
    from ..components.factory import discover_components

    out = {}
    for cls in dict.fromkeys(discover_components().values()):
        for spec in cls.get_utilities():
            if spec.name in out and out[spec.name] is not spec:
                raise ValueError(
                    f"Duplicate utility name '{spec.name}' declared by "
                    f"{cls.__name__} and elsewhere; names must be unique.")
            out[spec.name] = spec
    return out


# --- headless execution -------------------------------------------------------

def _snapshot(cwd):
    """Return the set of file paths (recursive) currently under cwd."""
    cwd = Path(cwd)
    if not cwd.exists():
        return set()
    return {p for p in cwd.rglob("*") if p.is_file()}


def run_utility(name, args_dict, cwd, registry=None):
    """Execute a declared utility headlessly and report what it produced.

    Snapshots the files under ``cwd`` before and after, so the caller (a GUI)
    can offer to associate newly produced files with the owning component.

    Parameters
    ----------
    name : str
        UtilitySpec.name to run.
    args_dict : dict
        Plain argument values (see parser_to_schema for the arg names).
    cwd : str or Path
        Working directory to run in and snapshot for produced files.
    registry : dict, optional
        {name: UtilitySpec} to look up in; defaults to all_utilities().

    Returns
    -------
    dict with keys:
        returncode      : int
        output          : captured stdout+stderr (str)
        produced_files  : sorted list of absolute paths created under cwd
    """
    registry = registry if registry is not None else all_utilities()
    if name not in registry:
        raise KeyError(f"Unknown utility '{name}'. Known: {sorted(registry)}")
    spec = registry[name]
    if not spec.available or spec.run is None:
        raise ValueError(f"Utility '{name}' is a placeholder and cannot be run.")

    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    before = _snapshot(cwd)
    result = spec.run(dict(args_dict), str(cwd)) or {}
    after = _snapshot(cwd)

    produced = sorted(str(p.resolve()) for p in (after - before))

    out = {
        "returncode": int(result.get("returncode", 0)),
        "produced_files": produced,
    }
    if "log_path" in result:
        out["log_path"] = result["log_path"]
    else:
        out["output"] = result.get("output", "")
    return out
