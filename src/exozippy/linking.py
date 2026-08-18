# src/exozippy/linking.py
"""
User-defined parameter links.

Any of the five numeric fields in a params.yaml entry (initval, mu, sigma,
lower, upper) may be a string expression referencing other parameters, with
or without algebraic manipulation:

    star.A.age:    {initval: star.B.age, sigma: 0}          # hard link (derived)
    star.A.age:    {initval: star.B.age, sigma: 1}          # soft link (Gaussian potential)
    star.A.av:     {lower: star.B.av}                       # dynamic hard bound
    orbit.b.omega: {initval: "orbit.c.omega + pi", sigma: 0}  # algebraic link

Semantics
---------
- Referenced parameters contribute their values in THEIR OWN user-facing
  units; the expression result is interpreted in the TARGET's user unit.
  All unit conversions to internal math units happen under the hood.
- initval link + sigma: 0   -> the target element is never sampled; its value
  is a deterministic function of the referenced parameters at all times.
- initval link + sigma > 0  -> the target element is sampled normally, and a
  Gaussian pm.Potential penalizes (target - expression) / sigma.
- initval link, no sigma    -> initialization seeding only (the relaxation
  engine solves the starting value from the expression; no runtime tie).
- mu link (sigma > 0)       -> same runtime behavior as a soft initval link.
- lower/upper link          -> the sampling transform maps into the dynamic
  interval, so the bound can never be violated.  No normalization potential
  is added: the logit reparameterization is already span-normalized (the
  sigmoid Jacobian supplies the 1/span), so the induced conditional is
  exactly U(lower, upper) and the marginal of the bound source is exactly
  its own prior -- for any span, dynamic or static.
- sigma link                -> evaluated once, numerically, from the
  relaxation-engine solution (a static snapshot, not a runtime tie).
  (init_scale is not linkable: it is obsolete user-side -- whitening scales
  are measured from the data at startup and the key is warn-ignored.)

This module is component-agnostic: it only knows about dotted parameter
paths and the system config's component keys / instance names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import sympy as sp

LINKABLE_FIELDS = ("initval", "mu", "sigma", "lower", "upper")

# Tokens like math.pi / np.pi are rewritten to sympy constants before any
# path matching so they are never mistaken for parameter references.
_CONSTANT_ALIASES = [
    (re.compile(r"(?<![\w.])(?:math|np|numpy)\.pi(?![\w.])"), "pi"),
    (re.compile(r"(?<![\w.])(?:math|np|numpy)\.e(?![\w.])"), "E"),
]

# comp.instance.param -- instance may contain hyphens (validated names allow
# them), which is why 3-part matching runs before the tokenizer ever sees a
# minus sign in that position.  A hyphenated match is only accepted if the
# instance name actually exists in the config; otherwise we fall back and
# the hyphen parses as subtraction.
_PATH_3 = re.compile(
    r"(?<![\w.])([A-Za-z_]\w*)\.([A-Za-z0-9_][A-Za-z0-9_\-]*)\.([A-Za-z_]\w*)(?![\w.(])"
)
_PATH_2 = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\.([A-Za-z_]\w*)(?![\w.(])")

_SYMPY_LOCALS_BASE = {"pi": sp.pi, "E": sp.E}

# The functions a link expression may call: sympy head -> the pytensor.tensor
# name that builds it.  ONE table, two consumers -- parse_link_expression
# rejects everything not in it (below) and sympy_to_pytensor builds from it --
# because the two must agree by construction.  They did not: the parser
# validated only free SYMBOLS, so an unknown head (a typo like sqr(), or
# log10(), which sympy does not define) sailed through as an AppliedUndef and
# only sympy_to_pytensor's "Unsupported operation" complained -- and only on
# the two link kinds that build a graph.  A seed-only initval link and a mu
# link never do: they are consumed by _apply_directed_links, whose
# float(expr.evalf()) can never evaluate an undefined function, and whose
# failure is caught as "not evaluable yet" at DEBUG.  The seed simply never
# applied, silently, forever.
# Pow (x**y, sqrt) and the arithmetic heads are structural in sympy, not
# Function subclasses, so they are handled directly by _eval and are
# deliberately absent here.
_SUPPORTED_FUNCS = {
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.asin: "arcsin",
    sp.acos: "arccos",
    sp.atan: "arctan",
    sp.atan2: "arctan2",
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.exp: "exp",
    sp.log: "log",
    sp.Abs: "abs",
    sp.sign: "sign",
    sp.floor: "floor",
    sp.ceiling: "ceil",
    sp.Min: "minimum",
    sp.Max: "maximum",
}


@dataclass
class ParamLink:
    """One user-defined link: <target_path>.<field> = f(dep_paths...)."""

    target_path: str  # standardized index form, e.g. "star.0.age"
    field: str  # one of LINKABLE_FIELDS
    expr_str: str  # original user string (for error messages)
    expr: sp.Expr  # symbols named by standardized dep paths (user units)
    dep_paths: List[str] = field(default_factory=list)


def _resolve_instance(comp_key, instance, system_config):
    """Return the integer index for an instance reference, or None."""
    comp_list = system_config.get(comp_key)
    if not isinstance(comp_list, list):
        return None
    for i, entry in enumerate(comp_list):
        if isinstance(entry, dict) and str(entry.get("name")) == instance:
            return i
    if instance.isdigit() and int(instance) < len(comp_list):
        return int(instance)
    return None


def is_link_expression(value, system_config):
    """True if `value` is a string that references at least one parameter path."""
    if not isinstance(value, str):
        return False
    try:
        float(value)
        return False  # plain numeric string; not a link
    except ValueError:
        pass
    s = value
    for pattern, repl in _CONSTANT_ALIASES:
        s = pattern.sub(repl, s)
    for m in _PATH_3.finditer(s):
        # Any 3-part reference on a KNOWN component key counts as a link
        # attempt, even if the instance doesn't resolve -- that way the
        # parser raises the specific "no instance named" error instead of a
        # generic not-a-number complaint.
        if m.group(1) in system_config:
            return True
    for m in _PATH_2.finditer(s):
        if m.group(1) in system_config:
            return True
    return False


def parse_link_expression(expr_str, system_config, context=""):
    """
    Parse a user link expression into a SymPy expression whose free symbols
    are standardized full parameter paths (comp.idx.param).

    Returns (sympy_expr, dep_paths).
    """
    s = expr_str
    for pattern, repl in _CONSTANT_ALIASES:
        s = pattern.sub(repl, s)

    placeholders = {}  # placeholder name -> standardized path

    def _register(path):
        name = f"_LNK{len(placeholders)}_"
        placeholders[name] = path
        return name

    # Pass 1: 3-part paths (comp.instance.param).
    def _sub3(m):
        comp_key, instance, param = m.group(1), m.group(2), m.group(3)
        if comp_key not in system_config:
            return m.group(0)
        idx = _resolve_instance(comp_key, instance, system_config)
        if idx is None:
            raise ValueError(
                f"Link expression '{expr_str}'{context}: '{m.group(0)}' looks like a "
                f"parameter reference, but '{comp_key}' has no instance named "
                f"'{instance}'."
            )
        return _register(f"{comp_key}.{idx}.{param}")

    s = _PATH_3.sub(_sub3, s)

    # Pass 2: 2-part paths (comp.param) -- only unambiguous for single-instance
    # components.
    def _sub2(m):
        comp_key, param = m.group(1), m.group(2)
        if comp_key not in system_config:
            return m.group(0)
        comp_list = system_config.get(comp_key)
        if not isinstance(comp_list, list):
            raise ValueError(
                f"Link expression '{expr_str}'{context}: component '{comp_key}' "
                f"is not a list component; links only support list components."
            )
        if len(comp_list) != 1:
            raise ValueError(
                f"Link expression '{expr_str}'{context}: '{m.group(0)}' is ambiguous "
                f"because '{comp_key}' has {len(comp_list)} instances. Use the "
                f"3-part form (e.g. '{comp_key}.<name>.{param}')."
            )
        return _register(f"{comp_key}.0.{param}")

    s = _PATH_2.sub(_sub2, s)

    # Any remaining dotted token is an unresolvable reference.
    leftover = re.search(r"(?<![\w.])([A-Za-z_]\w*\.[A-Za-z0-9_.\-]+)", s)
    if leftover:
        raise ValueError(
            f"Link expression '{expr_str}'{context}: cannot resolve "
            f"'{leftover.group(1)}'. References must use an active component "
            f"key from the system config (or math.pi / np.pi for constants)."
        )

    local_dict = dict(_SYMPY_LOCALS_BASE)
    for name, path in placeholders.items():
        local_dict[name] = sp.Symbol(path)

    try:
        expr = sp.sympify(s, locals=local_dict)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise ValueError(
            f"Link expression '{expr_str}'{context} could not be parsed: {e}"
        ) from e

    # A FUNCTION CALL THE BUILDER CANNOT BUILD IS A PARSE ERROR, HERE.
    # sympify happily turns any unknown name into an AppliedUndef, so `sqr(x)`
    # (a typo) and `log10(x)` (not a sympy name) parse cleanly.  Deferring the
    # complaint to sympy_to_pytensor only covers the hard/soft link kinds; the
    # seed-only and mu links never reach it and silently never applied.  See
    # _SUPPORTED_FUNCS.
    bad_funcs = sorted(
        {
            str(a.func)
            for a in expr.atoms(sp.Function)
            if a.func not in _SUPPORTED_FUNCS
        }
    )
    if bad_funcs:
        supported = ", ".join(sorted(f.__name__ for f in _SUPPORTED_FUNCS))
        raise ValueError(
            f"Link expression '{expr_str}'{context} calls "
            f"{', '.join(repr(f) for f in bad_funcs)}, which is not a "
            f"function link expressions support. Supported: {supported}, "
            f"plus sqrt/** and the arithmetic operators. "
            f"(For a base-10 logarithm write log(x)/log(10).)"
        )

    dep_paths = sorted(str(f) for f in expr.free_symbols)
    allowed = set(placeholders.values())
    unknown = [d for d in dep_paths if d not in allowed]
    if unknown:
        raise ValueError(
            f"Link expression '{expr_str}'{context} contains unrecognized "
            f"symbols: {unknown}. Only parameter paths, numbers, pi/E, and "
            f"standard math functions are allowed."
        )

    if not dep_paths:
        raise ValueError(
            f"Link expression '{expr_str}'{context} references no parameters; "
            f"use a plain number instead."
        )

    return expr, dep_paths


def extract_links(user_params, system_config):
    """
    Scan standardized user_params for link expressions in the five
    ``LINKABLE_FIELDS`` (initval, mu, sigma, lower, upper), REMOVE them from
    the entries (so downstream numeric code never sees strings), and return
    {target_path: {field: ParamLink}}.  ``init_scale`` is deliberately not
    among them -- see the module docstring.

    Must be called after standardize_param_names, so all list-component keys
    are in index form (comp.idx.param).
    """
    links: Dict[str, Dict[str, ParamLink]] = {}
    if not user_params or not system_config:
        return links

    for key, entry in user_params.items():
        if not isinstance(entry, dict):
            continue
        for fld in LINKABLE_FIELDS:
            val = entry.get(fld)
            if not isinstance(val, str):
                continue
            if not is_link_expression(val, system_config):
                # A non-numeric, non-link string in a numeric field is a
                # config error; fail loudly here rather than deep in numpy.
                try:
                    float(val)
                except ValueError:
                    raise ValueError(
                        f"Parameter '{key}' field '{fld}' has value '{val}', "
                        f"which is neither a number nor a recognizable "
                        f"parameter link expression."
                    )
                continue

            parts = key.split(".")
            if len(parts) != 3 or not parts[1].isdigit():
                raise ValueError(
                    f"Link on '{key}' could not be resolved to a single "
                    f"component instance. Use '<comp>.<name>.<param>' with a "
                    f"name defined in the system config."
                )

            context = f" (on {key}.{fld})"
            expr, dep_paths = parse_link_expression(
                val, system_config, context
            )

            if key in dep_paths:
                raise ValueError(
                    f"Link expression '{val}'{context} references its own "
                    f"target '{key}'."
                )

            links.setdefault(key, {})[fld] = ParamLink(
                target_path=key,
                field=fld,
                expr_str=val,
                expr=expr,
                dep_paths=dep_paths,
            )
            del entry[fld]

    return links


# ----------------------------
# SymPy -> PyTensor evaluation
# ----------------------------


def sympy_to_pytensor(expr, sym_values):
    """
    Recursively evaluate a SymPy expression as a PyTensor graph.

    sym_values maps symbol names (full parameter paths) to PyTensor scalars
    (or floats).  Supports +, -, *, /, **, and common math functions.
    """
    import pytensor.tensor as pt

    # Built from the one table the parser validates against, so the set of
    # functions a link may CONTAIN and the set it can be BUILT from cannot
    # drift apart (they had, in the direction that fails silently -- see
    # _SUPPORTED_FUNCS).
    _FUNC_MAP = {
        head: getattr(pt, attr) for head, attr in _SUPPORTED_FUNCS.items()
    }

    def _eval(node):
        if node.is_Symbol:
            try:
                return sym_values[node.name]
            except KeyError:
                raise ValueError(
                    f"No value available for link symbol '{node.name}'."
                )
        if node.is_number:
            return float(node)
        args = [_eval(a) for a in node.args]
        if node.is_Add:
            out = args[0]
            for a in args[1:]:
                out = out + a
            return out
        if node.is_Mul:
            out = args[0]
            for a in args[1:]:
                out = out * a
            return out
        if node.is_Pow:
            return args[0] ** args[1]
        func = _FUNC_MAP.get(node.func)
        if func is not None:
            return func(*args)
        raise ValueError(
            f"Unsupported operation '{node.func}' in link expression '{expr}'."
        )

    return _eval(expr)
