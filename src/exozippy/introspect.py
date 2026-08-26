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
from .config import NUMERIC_KEYS

# Numeric parameter fields (in defaults.yaml) that a GUI cares about.
# Exactly the numeric sub-keys ConfigManager.resolve() absorbs -- this used
# to be a third hand-maintained copy and had already lost ``bound_scale``,
# so a defaults.yaml that set a barrier width would not have shown it in the
# schema.  Order here is the order they are emitted in the schema.
_NUMERIC_FIELDS = NUMERIC_KEYS

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

    ``derived`` means derived BY DEFAULT: the block carries a "default"
    expression, the one a bare-string manifest entry selects.  A block
    holding only mode-selected expressions (the surgical coordinate
    swaps' ``from_mulens_*`` blocks on pm_ra/pm_dec/distance/logmass;
    ``fitvcve``'s alternates) describes a parameter that is SAMPLED
    unless a flag flips it, and a static schema answers for the default
    configuration.  The expressions are still reported either way, so a
    GUI can show what the modes would do.
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
    return "default" in out, out


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

    if expressions:
        entry["expressions"] = expressions
        # Convenience: flatten the "default" dependency list to top level.
        default_expr = expressions.get("default")
        if default_expr is not None:
            entry["deps"] = list(default_expr["deps"])

    # A parameter is "sampled" when it is a free parameter (no derivation),
    # is not fixed (sigma != 0), and carries the lower/upper bounds required
    # of every sampled parameter (rule 4 of the defaults.yaml contract in
    # src/exozippy/components/components.md).  Bounds are the marker --
    # init_scale is only a preliminary whitening seed and is optional.
    fixed = raw.get("sigma") == 0
    entry["sampled"] = bool(
        (not derived) and (not fixed) and ("lower" in raw) and ("upper" in raw)
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


def _load_shared_defaults():
    """Load the root-level (component-agnostic) parameter defaults.

    ``src/exozippy/components/defaults.yaml`` declares parameters by bare name
    instead of under a component key; ConfigManager.resolve() treats it as the
    blueprint any component's own block is layered over. Mirroring that here
    keeps introspection's view of a parameter identical to the one the model
    is actually built from.
    """
    path = Path(__file__).parent / "components" / "defaults.yaml"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _merged_param_block(cls, yaml_key):
    """A component's parameter block, including the shared parameters it uses.

    ``Component.shared_parameter_names()`` names root-level parameters the
    component may register but does not fully redeclare (Instrument's optional
    GP hyperparameters are the current case: the shared file carries the
    blueprint, the component overrides only the amplitude's unit and bounds).
    Those are layered in here so a GUI sees every parameter a fit can produce,
    not just the ones a component happened to override.
    """
    block = _load_param_block(cls, yaml_key)
    shared_names = list(cls.shared_parameter_names())
    if not shared_names:
        return dict(block)

    shared = _load_shared_defaults()
    merged = dict(block)
    for name in shared_names:
        if name not in shared and name not in merged:
            continue
        entry = dict(shared.get(name, {}))
        entry.update(block.get(name, {}))  # the component's own block wins
        merged[name] = entry
    return merged


def boolean_option_keys():
    """Config keys the components declare as boolean -> the components owning them.

    A ``config_schema()`` entry marks its key boolean by declaring
    ``"accepts": [True, False]``; this reads that declaration rather than any
    hand-maintained list, so a newly declared boolean option is covered the
    moment it is declared. Returns ``{key: [yaml_key, ...]}``.
    """
    registry = discover_components()
    out = {}
    for yaml_key, cls in sorted(registry.items()):
        for entry in cls.config_schema() or []:
            accepts = entry.get("accepts")
            if not isinstance(accepts, (list, tuple)):
                continue
            # The isinstance pass matters: [1, 0] equals [True, False] as a set.
            if not all(isinstance(v, bool) for v in accepts):
                continue
            if set(accepts) != {True, False}:
                continue
            out.setdefault(str(entry.get("key")), []).append(yaml_key)
    return out


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
    block = _merged_param_block(cls, yaml_key)

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
        "utilities": [spec.to_schema() for spec in cls.get_utilities()],
    }


# Per-key colour for the global schema below: is it a scalar option or a
# block, what does it accept, and what does it do.  The KEY SET is not here --
# it is system.RESERVED_CONFIG_KEYS, the one set System itself validates
# against, so a key added there appears in the schema (and therefore in the
# GUI's config-file detection) without a second edit.  This table only
# annotates; a key with no entry still gets a schema row.
_GLOBAL_KEY_INFO = {
    "run": ("block", None, "Documentation/bookkeeping block; inert."),
    "name": ("option", None, "Human-readable name for the fit."),
    "parameter_file": (
        "option",
        None,
        "Path to the parameter-override YAML, relative to the config.",
    ),
    "prefix": (
        "option",
        None,
        "Output path prefix for all result files "
        "(default 'fitresults/planet').",
    ),
    "logger_level": (
        "option",
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "Logging verbosity (default 'INFO').",
    ),
    "sampler": (
        "block",
        None,  # filled from run.KNOWN_SAMPLER_KEYS below
        "Sampler configuration block. Recognized keys are listed in "
        "'accepts'; unrecognized keys are warned about and ignored by run.py.",
    ),
    "modes": (
        "block",
        ["ledger", "max_invalid_frac", "force", "weights"],
        "Multimode reporting block (outputs/modes.py).",
    ),
    "mkparam": (
        "block",
        ["n_seeds", "force"],
        "Restart-file writer block (mkparam.write_param_file).",
    ),
    "gui": ("block", ["snapshot"], "GUI status block (gui.status)."),
    "modeling": (
        "block",
        ["compile"],
        "Modeling-draft block: {compile} for <prefix>_paper.tex.",
    ),
}


def _global_schema():
    """Describe global (non-component) config keys recognized by run.py.

    The key set is ``system.RESERVED_CONFIG_KEYS`` -- the same frozenset
    System validates a config against, and the one tests/test_known_keys.py
    already cross-checks against the source.  It used to be a third literal
    copy of that vocabulary and had drifted to three of the ten keys, while
    claiming to describe run.py's keys; the GUI's config-file detection reads
    this function's keys, so its answer was short by the same seven.  The
    remaining literal copy is the GUI's own degraded fallback, which is
    commented as such and is only reached when introspection is unavailable.
    """
    # Both imports are lazy so this module stays importable in lightweight
    # contexts even if the heavy sampling stack is unavailable -- and, for
    # system, so introspect keeps no module-level edge into the component
    # stack it describes.
    try:
        from .run import KNOWN_SAMPLER_KEYS

        sampler_keys = sorted(KNOWN_SAMPLER_KEYS)
    except Exception:  # pragma: no cover - defensive fallback
        sampler_keys = []

    try:
        from .system import RESERVED_CONFIG_KEYS

        keys = sorted(RESERVED_CONFIG_KEYS)
    except Exception:  # pragma: no cover - defensive fallback
        keys = sorted(_GLOBAL_KEY_INFO)

    schema = {}
    for key in keys:
        kind, accepts, doc = _GLOBAL_KEY_INFO.get(
            key, ("option", None, "Global configuration key.")
        )
        if key == "sampler":
            accepts = sampler_keys
        schema[key] = {
            "key": key,
            "kind": kind,
            "accepts": accepts,
            "required": False,
            "doc": doc,
        }
    return schema


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
