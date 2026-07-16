"""Static introspection of EXOZIPPy components and their configuration.

This module produces JSON-serializable descriptions of "what components
exist and what can be configured on them" WITHOUT building a System or
having any data files on disk. It is deliberately component-agnostic: all
component knowledge flows through the schemas that the components declare
(their ``defaults.yaml`` parameter blocks and the ``config_schema()``
classmethod on the component classes).

Intended consumers are a future GUI, documentation generators, and
scripting/validation helpers. Everything returned here is guaranteed to
survive ``json.dumps()`` unchanged -- no numpy scalars, no ``Path`` objects.

Public API
----------
list_components()          -> {yaml_key: {"class", "module", "doc"}}
component_schema(yaml_key) -> full description of one component
full_schema()              -> every component plus global config keys
"""

import inspect
import re
from pathlib import Path

import yaml

from .components.factory import discover_components

# Numeric parameter fields (in defaults.yaml) that a GUI cares about.
# Order here is the order they are emitted in the schema.
_NUMERIC_FIELDS = (
    "initval",
    "init_scale",
    "lower",
    "upper",
    "sigma",
    "mu",
)

# Descriptive (non-numeric) fields we pass through verbatim when present.
_DESCRIPTIVE_FIELDS = (
    "unit",
    "internal_unit",
    "latex",
    "description",
    "table_note",
    "rank",
)


def _first_paragraph(docstring):
    """Return the first paragraph of a docstring as a single clean line.

    Collapses internal whitespace and stops at the first blank line. Returns
    an empty string when there is no docstring.
    """
    if not docstring:
        return ""
    text = inspect.cleandoc(docstring)
    # First paragraph = everything up to the first blank line.
    paragraph = text.split("\n\n", 1)[0]
    return re.sub(r"\s+", " ", paragraph).strip()


def _expression_info(raw):
    """Extract derived-parameter expression info from a defaults entry.

    Returns (derived, expressions) where ``expressions`` maps each
    expression key (e.g. "default") to {"func_name", "deps"}.
    """
    expr_block = raw.get("expressions")
    if not isinstance(expr_block, dict) or not expr_block:
        return False, {}
    out = {}
    for expr_key, cfg in expr_block.items():
        cfg = cfg or {}
        out[expr_key] = {
            "func_name": cfg.get("func_name"),
            "deps": list(cfg.get("deps", []) or []),
        }
    return True, out


def _param_schema(name, raw):
    """Build the JSON-serializable schema for one parameter entry."""
    raw = raw or {}
    derived, expressions = _expression_info(raw)

    entry = {"name": name, "derived": derived}

    for field in _DESCRIPTIVE_FIELDS:
        if field in raw:
            entry[field] = raw[field]

    for field in _NUMERIC_FIELDS:
        if field in raw:
            entry[field] = raw[field]

    if derived:
        entry["expressions"] = expressions
        # Convenience: flatten the "default" dependency list to top level.
        default_expr = expressions.get("default")
        if default_expr is not None:
            entry["deps"] = list(default_expr["deps"])

    # A parameter is "sampled" when it is a free parameter (no derivation),
    # is not fixed (sigma != 0), and carries the init_scale required of every
    # sampled parameter (see CLAUDE.md). Fixed/derived parameters have none.
    fixed = raw.get("sigma") == 0
    entry["sampled"] = bool(
        (not derived) and (not fixed) and ("init_scale" in raw)
    )
    return entry


def _defaults_path(cls):
    """Return the defaults.yaml Path living beside a component class."""
    return Path(inspect.getfile(cls)).parent / "defaults.yaml"


def _load_param_block(cls, yaml_key):
    """Load a component's parameter block from its defaults.yaml.

    A single defaults.yaml may declare several component blocks (e.g. the
    mulensing directory declares both ``lens`` and ``mulensinstrument``), so
    we index into the block matching this component's yaml_key. Returns an
    empty dict when the file or block is absent (e.g. galacticmodel, which
    declares no sampled parameters).
    """
    path = _defaults_path(cls)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    block = data.get(yaml_key) or {}
    return block if isinstance(block, dict) else {}


def list_components():
    """Return a summary of every discoverable component.

    {yaml_key: {"class": ClassName, "module": dotted.module.path,
                "doc": first paragraph of the class docstring}}
    """
    registry = discover_components()
    out = {}
    for yaml_key, cls in sorted(registry.items()):
        out[yaml_key] = {
            "class": cls.__name__,
            "module": cls.__module__,
            "doc": _first_paragraph(cls.__doc__),
        }
    return out


def component_schema(yaml_key):
    """Return a JSON-serializable description of one component.

    Includes the class summary, every parameter from the component's
    defaults.yaml block, and the component-level config keys declared by
    ``Component.config_schema()`` (data files, references, options).
    """
    registry = discover_components()
    if yaml_key not in registry:
        raise KeyError(
            f"Unknown component '{yaml_key}'. "
            f"Known components: {sorted(registry)}"
        )
    cls = registry[yaml_key]
    block = _load_param_block(cls, yaml_key)

    parameters = {
        name: _param_schema(name, raw) for name, raw in block.items()
    }

    return {
        "yaml_key": yaml_key,
        "class": cls.__name__,
        "module": cls.__module__,
        "doc": _first_paragraph(cls.__doc__),
        "parameters": parameters,
        "config": list(cls.config_schema()),
    }


def _global_schema():
    """Describe global (non-component) config keys recognized by run.py."""
    # KNOWN_SAMPLER_KEYS is the single source of truth in run.py. Import it
    # lazily so this module stays importable in lightweight contexts even if
    # the heavy sampling stack is unavailable.
    try:
        from .run import KNOWN_SAMPLER_KEYS
        sampler_keys = sorted(KNOWN_SAMPLER_KEYS)
    except Exception:  # pragma: no cover - defensive fallback
        sampler_keys = []

    return {
        "prefix": {
            "key": "prefix",
            "kind": "option",
            "accepts": None,
            "required": False,
            "doc": (
                "Output path prefix for all result files "
                "(default 'fitresults/planet')."
            ),
        },
        "logger_level": {
            "key": "logger_level",
            "kind": "option",
            "accepts": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "required": False,
            "doc": "Logging verbosity (default 'INFO').",
        },
        "sampler": {
            "key": "sampler",
            "kind": "block",
            "accepts": sampler_keys,
            "required": False,
            "doc": (
                "Sampler configuration block. Recognized keys are listed "
                "in 'accepts'; unrecognized keys are warned about and "
                "ignored by run.py."
            ),
        },
    }


def full_schema():
    """Return the complete schema for every component plus global config.

    {"components": {yaml_key: component_schema(...)},
     "global": {...global config keys...}}

    The returned object is guaranteed to survive json.dumps() unchanged.
    """
    registry = discover_components()
    components = {
        yaml_key: component_schema(yaml_key) for yaml_key in sorted(registry)
    }
    return {
        "components": components,
        "global": _global_schema(),
    }
