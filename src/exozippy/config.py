# src/exozippy/config.py
import logging
import numpy as np
import yaml
import copy
from pathlib import Path
import sympy as sp
import astropy.units as u
import importlib
import signal
import time

logger = logging.getLogger(__name__)
import re

from exozippy.linking import extract_links

class SymbolicTimeout(Exception): pass


import contextlib

@contextlib.contextmanager
def _sympy_time_limit(seconds=2):
    """Hard wall-clock limit for a block of symbolic work.

    sp.solve (and evalf on its solutions) can hang effectively forever on
    certain equation/target pairs, and which pairs get attempted depends on
    hash-seed-sensitive iteration order -- so an unguarded call is a latent
    intermittent hang.  Raises SymbolicTimeout when the limit is hit.

    The handler re-arms the alarm before raising: if the exception fires
    while the interpreter is inside a C-level frame that discards it (seen
    in practice with JAX's gc callback -- "Exception ignored in
    _xla_gc_callback" -- after which the guarded solve ran unbounded), the
    next alarm gets another chance to land in interpretable bytecode.
    """
    def handler(signum, frame):
        signal.alarm(1)
        raise SymbolicTimeout()
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Instance names appear as the middle part of dotted parameter paths
# (star.MyName.teff), so they must be safe to split on "." and must not
# collide with the internal index notation (star.0.teff).
_VALID_INSTANCE_NAME = re.compile(r'^[A-Za-z0-9_-]+$')


def validate_instance_names(system_config):
    """Fatal-error check on user-supplied component instance names.

    Rejects names that would corrupt parameter-path parsing:
      - non-string names (YAML ``name: 128`` arrives as an int)
      - characters outside [A-Za-z0-9_-] ("." splits paths; whitespace,
        brackets, etc. break resolve()/mkparam key matching)
      - all-digit names, which alias the internal index notation
        (a component named "1" would silently resolve to instance index 1)
    """
    for comp_key, entries in (system_config or {}).items():
        if not isinstance(entries, list):
            continue
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            name = entry["name"]
            where = f"{comp_key}[{i}]"
            if not isinstance(name, str):
                raise ValueError(
                    f"Invalid name {name!r} for {where}: names must be strings. "
                    f"Quote it in your config YAML (name: \"{name}\")."
                )
            if not _VALID_INSTANCE_NAME.match(name):
                raise ValueError(
                    f"Invalid name '{name}' for {where}: names may only contain "
                    f"letters, digits, underscores, and hyphens. Characters like "
                    f"'.' or spaces would break parameter-path parsing "
                    f"(e.g. '{comp_key}.{name}.param')."
                )
            if name.isdigit():
                raise ValueError(
                    f"Invalid name '{name}' for {where}: purely numeric names "
                    f"collide with the internal index notation "
                    f"({comp_key}.0, {comp_key}.1, ...). Add a non-digit "
                    f"character (e.g. name: \"{comp_key}_{name}\")."
                )

# Provenance Ranks
RANK_USER = 100  # Explicitly in params.yaml
RANK_DERIVED_USER = 80  # Solved using ONLY Rank 100s
RANK_DERIVED_DATA = 60  # auto-estimated (e.g., K-band mass, RV offsets)
RANK_DERIVED_MIXED = 40  # Solved using a mix of User and Defaults
RANK_DEFAULT = 20  # From defaults.yaml

class ProvenanceState:
    def __init__(self):
        self.values = {}
        self.ranks = {}

    def set(self, path, value, rank):
        if rank > self.ranks.get(path, 0):
            self.values[path] = value
            self.ranks[path] = rank
            return True
        return False

""" 
This class is the reconciliation engine to determine the sole source of truth 
with which to initialize the parameters (values, scales, bounds) of the model. Specifically, it

1) determines the appropriate limits of each parameter, such that the bounds are the strictest among 
  a) the default bounds (typically very conservatively limited by physics, e.g., abs(velocity) < c), 
  b) any component specific bounds (e.g., bounds of the underlying grids), 
  c) and any user supplied bounds.

2) merges conflicting constraints, respecting a hierarchy of trust, such that

user-specified values (from user_params.yaml) > data-derived estimates > global defaults (from defaults.yaml)

It makes use of each component's symbolic_physics.py to derive the sampled parameter initvals from the set of values 
above, iteratively replacing the lowest ranked parameter with the re-derived value, and updating its weight the 
average of its constituent weights. 

It repeats the process independently for init_scale, automatically differentiating the relations in 
symbolic_physics.py to propagate uncertainties, and recognizing that any parameter may have an override for the initval, 
init_scale, or both.

3) warns the user if there are conflicting user-specified constraints that cannot be reconciled.

The logic of the config manager is independent of any specific components.
"""


def _meaningful_change(new_val, old_val, new_rank, old_rank, tolerance,
                       resolved, provenance, target_str):
    """Return True iff _execute_solve should apply this update and signal progress.

    Propagates a rank improvement silently (updates provenance but returns False)
    when the value itself hasn't changed.  Returning False when the value is
    unchanged prevents the relaxation loop from running to max_iter on systems
    that have converged but still have two competing derivation paths for the
    same variable.
    """
    if old_val is None:
        return True   # Condition A: variable was previously unknown — always an update
    ref = max(abs(new_val), abs(old_val), 1e-9)
    if abs(new_val - old_val) / ref >= tolerance:
        return True   # value changed meaningfully
    # Value unchanged; propagate rank silently if it improved
    if new_rank > old_rank:
        provenance[target_str] = new_rank
    return False


class ConfigManager:
    def __init__(self, user_params, system_config=None):
        self.raw_user_params = user_params
        self.custom_solvers = {}
        self.standalone_solvers = set()

        # User-defined parameter links (expression strings in numeric fields).
        # Populated by extract_links, which also strips the strings from
        # user_params so downstream numeric code never sees them.
        self.links = {}

        # If config is provided, validate names then standardize right away
        if system_config is not None:
            validate_instance_names(system_config)
            self.user_params = self.standardize_param_names(user_params, system_config)
            self.links = extract_links(self.user_params, system_config)
        else:
            self.user_params = user_params

        self.system_config = system_config or {}
        self.base_defaults = {}
        self.all_relations = []
        self.master_symbol_map = {}

        # Storage for hints passed by components during Registration Sweep
        self.hints = {}
        self.hint_ranks = {}

        # Multi-seed sampling (P4).  seed_resolved holds K fully-solved start
        # points (list of {internal_path: internal_value} dicts) after
        # finalize_user_params runs the relaxation engine once per seed; it
        # stays None for the ordinary single-start case (K == 1).  seed_hint_sets
        # is a per-seed observable channel that components (e.g. the MMEXOFAST
        # loader) push into; it feeds the relaxation engine at a rank between
        # RANK_DERIVED_DATA and RANK_USER so an explicit user initval list wins.
        self.seed_resolved = None
        self.seed_hint_sets = []
        self.seed_hint_rank = RANK_DERIVED_USER  # 80: below RANK_USER, above data
        self.scale_hints = {}   # path -> init_scale in internal units
        self.propagated_scales = {}  # path -> init_scale (internal) from Jacobian forward pass
        self.dependencies = {}
        self.symbolic_blacklist = set()

        # Structured diagnostics collected by the relaxation engine (e.g.
        # over-constrained contradictions).  Each entry is a dict
        # {severity, message, param_paths}.  Consumed by the solve/validate
        # API (solve_api.py) without parsing log text.  Populated only when
        # the engine detects a contradiction; empty for a clean solve.
        self.diagnostics = []

        # Snapshots of the last relaxation solve (seed 0, which is solved
        # last in finalize_user_params).  Exposed via export_solution() so a
        # caller can report each parameter's solved value, scale, and
        # provenance without rebuilding the PyMC model.
        self._last_provenance = {}        # internal_path -> rank
        self._last_scale_provenance = {}  # internal_path -> rank
        self._last_resolved = {}          # internal_path -> internal value
        self._last_solved_by = {}         # internal_path -> relation string

        components_dir = Path(__file__).parent / "components"

        for py_file in components_dir.rglob("symbolic_physics.py"):
            module_name = f"exozippy.components.{py_file.parent.name}.symbolic_physics"

            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None: continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "register_solvers"):
                module.register_solvers(self)

            yaml_key = getattr(module, "comp_key", py_file.parent.name)

            if hasattr(module, "get_symbol_map") and yaml_key in self.system_config:
                comp_section = self.system_config[yaml_key]

                if not isinstance(comp_section, list):
                    comp_section = [comp_section]

                for i, entry_cfg in enumerate(comp_section):
                    if not isinstance(entry_cfg, dict):
                        continue

                    # A component may return a single symbol map, or a list of
                    # maps when one config entry instantiates the relations
                    # multiple times (e.g. a lens with N sources instantiates
                    # the per-source parameter chain once per source).
                    raw_maps = module.get_symbol_map(entry_cfg)
                    if not isinstance(raw_maps, list):
                        raw_maps = [raw_maps]

                    for raw_map in raw_maps:
                        instance_map = {}
                        for sym_name, path in raw_map.items():
                            if "." in str(path):
                                instance_map[sym_name] = path
                            else:
                                instance_map[sym_name] = f"{yaml_key}.{i}.{path}"

                        for _, full_path in instance_map.items():
                            self.master_symbol_map[full_path] = sp.Symbol(full_path)

                        # Extract the exact SymPy objects (with all their assumptions) from the equations
                        module_symbols = set()
                        for rel in getattr(module, "RELATIONS", []):
                            module_symbols.update(rel.free_symbols)

                        subs = {}
                        for sym in module_symbols:
                            if sym.name in instance_map:
                                subs[sym] = sp.Symbol(instance_map[sym.name])

                        for rel in getattr(module, "RELATIONS", []):
                            rel_inst = rel.subs(subs)
                            # Maps sharing symbols (e.g. per-source maps with a
                            # common lens mass) produce identical instances of
                            # the shared relations; keep one copy.
                            if rel_inst not in self.all_relations:
                                self.all_relations.append(rel_inst)

        for defaults_file in components_dir.rglob("defaults.yaml"):
            with open(defaults_file, "r") as f:
                comp_defaults = yaml.safe_load(f) or {}
                self._deep_merge(self.base_defaults, comp_defaults)

        # Add this inside ConfigManager.__init__ after filling all_relations
        for rel in self.all_relations:
            logger.debug(f"Relation: {rel}")

    def register_custom_solver(self, target_str, solver_func, standalone=False):
        """
        Register a shortcut solver for 'comp.param' targets.

        By default a custom solver only runs when the relaxation engine
        attacks an equation containing the target (see _execute_solve).
        standalone=True additionally runs it once per iteration on its own:
        required when the target's defining relation always holds a second
        derived unknown (e.g. orbit.m_total in the Kepler relation, whose
        other side is the equally-unknown arsun), so the equation path can
        never get down to one unknown by itself.
        """
        self.custom_solvers[target_str] = solver_func
        if standalone:
            self.standalone_solvers.add(target_str)

    # ----------------------------
    # User-defined parameter links
    # ----------------------------

    def _link_internal_expr(self, plink):
        """
        Convert a ParamLink's user-unit expression to internal units.

        The user writes f(deps) where each dep contributes its value in its
        own user unit and the result is in the target's user unit.  The
        relaxation engine works in internal units, so substitute
        dep -> dep_internal / factor_dep and multiply by factor_target.
        """
        subs = {}
        for dep in plink.dep_paths:
            parts = dep.split('.')
            f = self.get_conversion_factor(parts[0], parts[-1], full_path=dep)
            if f != 1.0:
                subs[sp.Symbol(dep)] = sp.Symbol(dep) / f
        tparts = plink.target_path.split('.')
        tf = self.get_conversion_factor(tparts[0], tparts[-1],
                                        full_path=plink.target_path)
        expr = plink.expr.subs(subs) if subs else plink.expr
        return tf * expr if tf != 1.0 else expr

    def get_element_links(self, comp_type, param_name):
        """
        Return the user-defined links targeting elements of one parameter:
        {field: {element_index: ParamLink}}.  Used by Component.add_parameter
        to wire dynamic (runtime) links into the PyMC graph.
        """
        out = {}
        for target, fields in self.links.items():
            parts = target.split('.')
            if len(parts) == 3 and parts[0] == comp_type and parts[2] == param_name:
                for fld, plink in fields.items():
                    out.setdefault(fld, {})[int(parts[1])] = plink
        return out

    def get_resolved_path(self, component_prefix, dep_str, resolved_maps):
        """
        Handles 'star.mass[star_map]' -> 'star.1.mass'
        """
        if "[" in dep_str and "]" in dep_str:
            param, map_name = dep_str.replace("]", "").split("[")
            # resolved_maps is a dict of the component's maps (e.g., {'star_map': 1})
            map_idx = resolved_maps.get(map_name)
            base_name, p_name = param.split(".")
            return f"{base_name}.{map_idx}.{p_name}"
        return f"{component_prefix}.{dep_str}"

    def attach_config(self, config):
        """Call this during Stage 1/Pre-flight when the System config is loaded."""
        self.system_config = config
        # Modernize the names to strict index format before running any staging logic
        self.user_params = self.standardize_param_names(self.raw_user_params, config)
        self.links = extract_links(self.user_params, config)

    def _translate_and_scale(self, path, value):
        """Standardize a human-readable path to internal-index form and convert
        its value to internal units.  Returns (translated_path, internal_value).
        Shared by add_hint / add_scale_hint / add_seed_hints so they all agree
        on nomenclature and unit handling."""
        translated_path = path
        parts = path.split('.')

        # 1. Standardize the Nomenclature
        if len(parts) == 3:
            comp_type, name, param = parts
            try:
                # Resolve human name (e.g., 'LENS') to internal index (e.g., '0')
                idx = next(i for i, c in enumerate(self.system_config.get(comp_type, []))
                           if str(c.get("name")) == str(name) or str(i) == str(name))
                translated_path = f"{comp_type}.{idx}.{param}"
            except StopIteration:
                pass

        # 2. Scale to Internal Units
        final_parts = translated_path.split('.')
        if len(final_parts) >= 2:
            c_type, p_name = final_parts[0], final_parts[-1]
            # Use the original path to check if the user provided an explicit unit in yaml
            factor = self.get_conversion_factor(c_type, p_name, full_path=path)
            internal_value = float(value) * factor
        else:
            internal_value = float(value)

        return translated_path, internal_value

    def add_hint(self, path, value, rank=RANK_DERIVED_DATA):
        """
        API for components to register their data-driven guesses.
        Converts human-readable paths to strict indices and scales to internal units.
        """
        translated_path, internal_value = self._translate_and_scale(path, value)

        # Store the fully processed, ready-to-use hint
        self.hints[translated_path] = internal_value
        self.hint_ranks[translated_path] = rank

    def add_seed_hints(self, seed_dicts, rank=None):
        """Register K per-seed observable sets for multi-seed sampling (P4).

        `seed_dicts` is a list of length K; each entry maps a parameter path
        (human-readable or index form) to a value in that parameter's user
        unit.  These feed the relaxation engine as one complete start point per
        seed (see finalize_user_params).  Rank sits between RANK_DERIVED_DATA
        and RANK_USER by default so an explicit user initval list still wins;
        the MMEXOFAST loader is the primary caller.  Paths absent from a given
        seed fall back to the base (defaults/hints/user) solution for that seed.
        """
        processed = []
        for d in seed_dicts:
            pd = {}
            for path, value in d.items():
                tpath, ival = self._translate_and_scale(path, value)
                pd[tpath] = ival
            processed.append(pd)
        self.seed_hint_sets = processed
        if rank is not None:
            self.seed_hint_rank = rank

    def add_scale_hint(self, path, scale):
        """
        Register a context-appropriate init_scale for a parameter.

        Overrides the defaults.yaml init_scale but yields to an explicit
        init_scale provided by the user in their params file.  Use this to
        set physically meaningful sampling scales that differ from the generic
        stellar defaults (e.g. bulge distances need ~500 pc, not 0.1 pc).
        """
        translated_path = path
        parts = path.split('.')
        if len(parts) == 3:
            comp_type, name, param = parts
            try:
                idx = next(i for i, c in enumerate(self.system_config.get(comp_type, []))
                           if str(c.get("name")) == str(name) or str(i) == str(name))
                translated_path = f"{comp_type}.{idx}.{param}"
            except StopIteration:
                pass

        final_parts = translated_path.split('.')
        if len(final_parts) >= 2:
            c_type, p_name = final_parts[0], final_parts[-1]
            factor = self.get_conversion_factor(c_type, p_name, full_path=path)
            internal_scale = float(scale) * factor
        else:
            internal_scale = float(scale)

        self.scale_hints[translated_path] = internal_scale

    def _deep_merge(self, base, overrides):
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def resolve(self, component_type, param_name, shape=(), internal_overrides=None, names=None,
                element=None):
        # `element` targets a single specific instance when resolving with
        # shape=(): the per-element user-param keys are built with that index
        # instead of the local loop index.  Without it, resolving star.1.age
        # one element at a time would read star.0.age's user entry and bleed
        # element-0 overrides (e.g. sigma: 0) into its siblings.

        # 1. Grab an isolated copy of the global root default blueprint
        base = copy.deepcopy(self.base_defaults.get(param_name, {}))

        # 2. Grab an isolated copy of the component-specific override block if it exists
        comp_defaults = self.base_defaults.get(component_type, {})
        comp_override = copy.deepcopy(comp_defaults.get(param_name, {}))

        if comp_override:
            # Safely layer in the math blueprint without mutating base_defaults
            if "expressions" not in comp_override and "expressions" in base:
                comp_override["expressions"] = base["expressions"]
            self._deep_merge(base, comp_override)

        # --- EXPLICIT ROOT LINEAGE BACKUP ---
        # If both base and child overrides are missing expressions due to startup file load order,
        # reach out directly to check the root-level dictionary block.
        if "expressions" not in base or not base["expressions"]:
            root_cfg = self.base_defaults.get(param_name, {})
            if "expressions" in root_cfg:
                base["expressions"] = copy.deepcopy(root_cfg["expressions"])

        n_elements = int(np.prod(shape)) if shape != () else 1

        def _eff_idx(i):
            return element if (element is not None and n_elements == 1) else i

        base_unit_str = base.get("unit", "")
        new_unit_str = None

        keys_to_check_global = []
        for i in range(n_elements):
            keys = [f"{component_type}.{param_name}", f"{component_type}.{_eff_idx(i)}.{param_name}"]
            if names and i < len(names):
                keys.append(f"{component_type}.{names[i]}.{param_name}")
            keys_to_check_global.extend(keys)

        for k in keys_to_check_global:
            if k in self.user_params and isinstance(self.user_params[k], dict):
                if "unit" in self.user_params[k]:
                    new_unit_str = self.user_params[k]["unit"]
                    break

        unit_scaling = 1.0
        if new_unit_str and base_unit_str and new_unit_str != base_unit_str:
            try:
                unit_scaling = u.Unit(base_unit_str).to(u.Unit(new_unit_str))
            except Exception:
                unit_scaling = 1.0

        resolved = {
            "shape": shape,
            "user_modified": False,
            "user_prior_modified": False,
            "unit": new_unit_str if new_unit_str else base_unit_str,
            "internal_unit": base.get("internal_unit"),
            "latex": base.get("latex", ""),
            "description": base.get("description", ""),
            "expressions": base.get("expressions", {}),
            "print_to_table": base.get("print_to_table", True),
            "debug_print": base.get("debug_print", None),
        }

        tuning_keys = ["initval", "init_scale"]
        physics_keys = ["lower", "upper", "mu", "sigma"]
        all_numeric = tuning_keys + physics_keys

        for key in all_numeric:
            val = base.get(key)
            if val is not None:
                resolved[key] = np.full(n_elements, float(val) * unit_scaling, dtype=float)
            else:
                resolved[key] = None

        def apply_value(key, current_arr, idx, new_val):
            if new_val is None: return current_arr
            if current_arr is None:
                current_arr = np.full(n_elements, np.nan, dtype=float)
                resolved[key] = current_arr
            v = float(new_val)
            if np.isnan(current_arr[idx]):
                current_arr[idx] = v
            elif key == "lower":
                current_arr[idx] = max(current_arr[idx], v)
            elif key == "upper":
                current_arr[idx] = min(current_arr[idx], v)
            else:
                current_arr[idx] = v
            return current_arr

        resolved["auto_estimated"] = False
        if internal_overrides:
            resolved["auto_estimated"] = True
            for key in all_numeric:
                if key in internal_overrides:
                    val = internal_overrides[key]
                    for i in range(n_elements):
                        v = val[i] if isinstance(val, (list, np.ndarray)) else val
                        v_scaled = float(v) * unit_scaling
                        apply_value(key, resolved[key], i, v_scaled)

        # propagated_scales and scale_hints are stored in internal units.
        # Divide by get_conversion_factor (user→internal) to recover user units
        # before passing to Parameter, which will re-apply the same factor.
        # This is distinct from unit_scaling (base→user), which only applies to
        # default values read from defaults.yaml.
        internal_factor = self.get_conversion_factor(component_type, param_name) or 1.0

        # Apply Jacobian-propagated scales (lowest priority after defaults,
        # overridden by scale hints and user params below).
        if self.propagated_scales:
            for i in range(n_elements):
                prop_keys = [f"{component_type}.{param_name}",
                             f"{component_type}.{_eff_idx(i)}.{param_name}"]
                if names and i < len(names):
                    prop_keys.append(f"{component_type}.{names[i]}.{param_name}")
                for k in prop_keys:
                    if k in self.propagated_scales:
                        apply_value("init_scale", resolved["init_scale"], i,
                                    self.propagated_scales[k] / internal_factor)
                        break

        # Apply scale hints: context-appropriate init_scales from components
        # (e.g. bulge distances). They override defaults.yaml but yield to the
        # user's explicit init_scale below.
        for i in range(n_elements):
            scale_keys = [f"{component_type}.{param_name}", f"{component_type}.{_eff_idx(i)}.{param_name}"]
            if names and i < len(names):
                scale_keys.append(f"{component_type}.{names[i]}.{param_name}")
            for k in scale_keys:
                if k in self.scale_hints:
                    apply_value("init_scale", resolved["init_scale"], i,
                                self.scale_hints[k] / internal_factor)
                    break

        for i in range(n_elements):
            keys_to_check = [f"{component_type}.{param_name}", f"{component_type}.{_eff_idx(i)}.{param_name}"]
            if names and i < len(names):
                keys_to_check.append(f"{component_type}.{names[i]}.{param_name}")

            for k in keys_to_check:
                if k in self.user_params:
                    ov = self.user_params[k]
                    if ov is None: continue
                    if not isinstance(ov, dict): ov = {"initval": ov}

                    resolved["user_modified"] = True
                    if any(pk in ov for pk in physics_keys):
                        resolved["user_prior_modified"] = True

                    for key in all_numeric:
                        if key in ov:
                            v_ov = ov[key]
                            # List-valued initvals are per-seed start points
                            # (P4 multi-seed sampling); bounds/scales and the
                            # canonical single start derive from seed 0.
                            if isinstance(v_ov, (list, tuple)):
                                v_ov = v_ov[0]
                            apply_value(key, resolved[key], i, v_ov)

                    for str_key in ["unit", "latex", "description"]:
                        if str_key in ov:
                            if n_elements > 1:
                                if not isinstance(resolved[str_key], list):
                                    resolved[str_key] = [resolved[str_key]] * n_elements
                                resolved[str_key][i] = ov[str_key]
                            else:
                                resolved[str_key] = ov[str_key]

                    for bool_key in ["print_to_table", "debug_print"]:
                        if bool_key in ov:
                            resolved[bool_key] = ov[bool_key]

                    if "sigma" in ov and "init_scale" not in ov and resolved["sigma"] is not None:
                        apply_value("init_scale", resolved.get("init_scale"), i, ov["sigma"])

                    # If user gave mu but not initval, start the chain at mu rather
                    # than the defaults.yaml value — the user's prior center is always
                    # a better starting point than an arbitrary global default.
                    if "mu" in ov and "initval" not in ov:
                        apply_value("initval", resolved["initval"], i, ov["mu"])

        return resolved

    def get_conversion_factor(self, component_type, param_name, full_path=None):
        u_str = None
        # 1. Check if the user explicitly provided a unit in their config
        if full_path and full_path in self.user_params and isinstance(self.user_params[full_path], dict):
            u_str = self.user_params[full_path].get("unit")

        # 2. Fallback to defaults
        comp_cfg = self.base_defaults.get(component_type, {})
        param_cfg = comp_cfg.get(param_name, {})
        if not u_str:
            u_str = param_cfg.get("unit", "")

        i_str = param_cfg.get("internal_unit", "")

        if not u_str or not i_str: return 1.0

        try:
            if "dex" in u_str or "dex" in i_str: return 1.0
            # Returns the multiplier to convert FROM user TO internal
            return u.Unit(u_str).to(u.Unit(i_str))
        except Exception:
            return 1.0

    @staticmethod
    def standardize_param_names(user_params, config):
        """
        Translate all user-facing parameter keys to canonical internal index form.

        Three input forms are accepted, processed in two passes so that explicit
        per-instance values always win over broadcast values regardless of file order:

          Pass 1 — 3-part keys (highest precedence):
            star.A.teff   →  star.0.teff   (name looked up in config list)
            star.0.teff   →  star.0.teff   (already indexed; stored as-is)

          Pass 2 — 2-part keys (broadcast to all instances):
            star.teff     →  star.0.teff, star.1.teff, …
            sed.errscale  →  sed.errscale  (flat-dict component; kept as-is)

        After this function, self.user_params contains only indexed or flat-dict
        keys internally.  The 2-part form is purely a user convenience.
        """
        if not user_params:
            return {}

        standardized = {}

        # Pass 1: resolve 3-part keys to index form.
        for key, val in user_params.items():
            parts = key.split(".", 2)
            if len(parts) != 3:
                continue

            comp_type, comp_name, param_name = parts

            if comp_type not in config:
                raise ValueError(
                    f"\n!!! STRICT NAMING ERROR !!!\n"
                    f"Parameter '{key}' uses the prefix '{comp_type}', but '{comp_type}' "
                    f"is not defined in your system configuration.\n"
                    f"Ensure your YAML block names match your parameter prefixes exactly."
                )

            comp_list = config[comp_type]
            if not isinstance(comp_list, list):
                standardized[key] = val  # flat-dict component 3-part key: keep as-is
                continue

            try:
                idx = next(
                    i for i, c in enumerate(comp_list)
                    if isinstance(c, dict) and c.get("name") == comp_name
                )
                standardized[f"{comp_type}.{idx}.{param_name}"] = val
            except StopIteration:
                standardized[key] = val  # numeric index or unknown name: keep as-is

        # Pass 2: expand 2-part keys for list components.
        # Indexed entries written by Pass 1 are never overwritten (explicit beats broadcast).
        for key, val in user_params.items():
            parts = key.split(".", 2)
            if len(parts) != 2:
                continue

            comp_type, param_name = parts
            comp_list = config.get(comp_type)

            if not isinstance(comp_list, list):
                standardized[key] = val  # flat-dict or unknown component: keep as-is
                continue

            for i in range(len(comp_list)):
                indexed_key = f"{comp_type}.{i}.{param_name}"
                if indexed_key not in standardized:
                    # Each instance must get its OWN dict: downstream code
                    # (finalize_user_params inject-back, init_scale sync)
                    # mutates these entries per-instance, and a shared object
                    # would let the last instance's write clobber all others
                    # (e.g. per-source radii solved from rho).
                    standardized[indexed_key] = copy.deepcopy(val)

        # Pass 3: 1-part and other unhandled keys (e.g. 'run').
        for key, val in user_params.items():
            if "." not in key and key not in standardized:
                standardized[key] = val

        return standardized

    def finalize_user_params(self):
        """
        Called by the System object AFTER all components have registered their hints.
        """
        # Reset structured diagnostics for this solve.  resolve_and_validate
        # runs once per seed and appends contradictions here; _record_diagnostic
        # dedupes so repeated seeds do not multiply identical entries.
        self.diagnostics = []

        flat_params = {}
        name_to_index = {}

        # Build index mapping directly from the config keys (No translations!)
        for comp_key, entries in self.system_config.items():
            if isinstance(entries, list):
                for i, c in enumerate(entries):
                    if isinstance(c, dict) and "name" in c:
                        name_to_index[(comp_key, c["name"])] = i

        logger.debug("=" * 50 + "\nSYMBOL MAP DEBUGGING\n" + "=" * 50)

        for path, data in self.user_params.items():
            if data is None:
                continue
            val = data.get("initval") if isinstance(data, dict) else data
            # A list-valued initval is a set of per-seed start points (P4
            # multi-seed sampling).  The base flat_params below seeds the
            # relaxation engine with seed 0; _build_seed_overrides re-injects
            # each seed's element as a RANK_USER override in the K-solve loop.
            if isinstance(val, (list, tuple)):
                val = val[0]
            if val is not None:
                parts = path.split('.')
                translated_path = path

                if len(parts) == 3:
                    comp_type, name, param = parts
                    if (comp_type, name) in name_to_index:
                        idx = name_to_index[(comp_type, name)]
                        translated_path = f"{comp_type}.{idx}.{param}"

                sym = self.master_symbol_map.get(translated_path)
                if sym:
                    c_type = translated_path.split('.')[0]
                    p_name = translated_path.split('.')[-1]
                    factor = self.get_conversion_factor(c_type, p_name, full_path=path)
                    internal_val = float(val) * factor
                    logger.debug(f"Mapped: {path} -> {translated_path} -> {sym} = {internal_val} (internal)")
                    flat_params[sym] = internal_val
                else:
                    if "." in path:
                        logger.debug(f"Unmapped: {path} (tried translated: {translated_path})")

        # --- REGISTER LINK TARGETS AND DEPENDENCIES AS SYMBOLS ---
        # A link target/dep may not appear in any component's symbol map
        # (e.g. star.age has no symbolic relations); registering it here lets
        # the relaxation engine seed it from defaults.yaml and solve the
        # directed link assignments.
        for target, fields in self.links.items():
            for plink in fields.values():
                for path in [target] + list(plink.dep_paths):
                    if path not in self.master_symbol_map:
                        self.master_symbol_map[path] = sp.Symbol(path)
                        logger.debug(f"Registered link symbol: {path}")

        # --- FALLBACK TO LEAFS ---
        for path in list(self.user_params.keys()):
            if path not in self.master_symbol_map:
                self.master_symbol_map[path] = sp.Symbol(path)
                logger.debug(f"Registered as leaf: {path}")

                # Push the user's value into the solver's initial state
                data = self.user_params[path]
                val = data.get("initval") if isinstance(data, dict) else data
                if isinstance(val, (list, tuple)):
                    val = val[0]  # seed 0; see per-seed handling below
                if val is not None:
                    c_type = path.split('.')[0]
                    p_name = path.split('.')[-1]
                    factor = self.get_conversion_factor(c_type, p_name, full_path=path)
                    flat_params[self.master_symbol_map[path]] = float(val) * factor

        base_flat = {str(k): v for k, v in flat_params.items()}

        # --- MULTI-SEED SOLVE (P4) ---
        # Build K per-seed RANK_USER override sets (user initval lists win over
        # MMEXOFAST-style seed hints; both fall back to the shared base_flat for
        # any path they do not touch), then run the relaxation engine once per
        # seed inside this single prepare() call so every seed shares one symbol
        # environment (guards against the known cross-build nondeterminism).
        # Bounds/scales are taken from seed 0 only -- seeds move the START, never
        # the bounds -- so self.propagated_scales is restored to seed 0's after
        # the loop.
        K, seed_overrides = self._build_seed_overrides(name_to_index)

        # Solve seed 0 LAST so the final self.propagated_scales and any
        # init_scale synced back into self.user_params by _execute_solve both
        # reflect seed 0 -- the seed whose bounds/scales the model actually
        # uses.  Only start positions vary between seeds; bounds/scales do not.
        seed_resolved = [None] * K
        for k in list(range(1, K)) + [0]:
            flat_k = dict(base_flat)
            flat_k.update(seed_overrides[k])
            seed_resolved[k] = self.resolve_and_validate_parameters(flat_k)

        # seed 0 remains the canonical single start injected back into
        # user_params below; the full K-set is stored for get_raw_starts.
        resolved_flat = seed_resolved[0]
        self.seed_resolved = seed_resolved if K > 1 else None
        if K > 1:
            logger.info(f"Multi-seed sampling: solved {K} seed start points.")

        logger.debug("Solver finished.")

        # 4. INJECT BACK SAFELY
        for sym_node, val in resolved_flat.items():
            path = str(sym_node)
            parts = path.split('.')
            final_path = path

            if len(parts) == 3:
                comp_type, idx, param = parts
                for (c_type, c_name), i in name_to_index.items():
                    if c_type == comp_type and str(i) == str(idx):
                        final_path = f"{comp_type}.{c_name}.{param}"
                        break
                factor = self.get_conversion_factor(comp_type, param, full_path=final_path)
                user_val = val / factor
            else:
                user_val = val

            # standardize_param_names stores entries under the index form
            # (star.0.teff) while final_path uses the name form (star.A.teff).
            # Check both so we don't create a spurious duplicate entry.
            existing_key = None
            for try_key in (final_path, path):
                if try_key in self.user_params and isinstance(self.user_params[try_key], dict):
                    existing_key = try_key
                    break

            if existing_key is None:
                self.user_params[final_path] = {"initval": user_val, "derived": True}
            else:
                existing = self.user_params[existing_key]
                # Don't clobber a user-specified Gaussian prior: if the user gave
                # mu but no initval, resolve() will use mu as the starting point.
                # Injecting the default-derived initval here would undo that.
                if "mu" in existing and "initval" not in existing:
                    existing["derived"] = True
                else:
                    existing["initval"] = user_val
                    existing["derived"] = True

        # --- SNAPSHOT USER-LINK EXPRESSIONS FOR STATIC FIELDS ---
        # sigma / init_scale links are static by design; lower / upper links
        # additionally need a numeric snapshot so the logit transform can be
        # set up (the dynamic tensor bound replaces it at runtime).
        for target, fields in self.links.items():
            entry = self.user_params.get(target)
            if not isinstance(entry, dict):
                entry = {}
                self.user_params[target] = entry
            for fld, plink in fields.items():
                if fld in ("initval", "mu"):
                    continue  # handled by the directed relaxation pass
                if not all(d in resolved_flat for d in plink.dep_paths):
                    logger.warning(
                        f"Link '{target}.{fld} = {plink.expr_str}' could not be "
                        f"snapshot: unresolved dependencies "
                        f"{[d for d in plink.dep_paths if d not in resolved_flat]}.")
                    continue
                try:
                    val_int = float(self._link_internal_expr(plink).evalf(subs=resolved_flat))
                except Exception as e:
                    logger.warning(
                        f"Link '{target}.{fld} = {plink.expr_str}' snapshot "
                        f"evaluation failed: {e}")
                    continue
                tparts = target.split('.')
                tf = self.get_conversion_factor(tparts[0], tparts[-1], full_path=target)
                entry[fld] = val_int / tf
                logger.debug(f"Link snapshot: {target}.{fld} = {entry[fld]:.6g} (user units)")

    def _build_seed_overrides(self, name_to_index):
        """Assemble the per-seed RANK_USER override sets for multi-seed sampling.

        Two sources feed the seeds, in priority order:
          1. User initval lists in params.yaml (`initval: [v0, v1, ...]`) --
             highest priority (an explicit user list always wins).
          2. Component seed hints (config_manager.seed_hint_sets), e.g. the
             MMEXOFAST loader -- lower priority.

        Returns (K, overrides) where K is the seed count and overrides is a
        length-K list of {internal_path_str: internal_value} dicts.  Every list
        must have length K or 1 (length-1 broadcasts to all seeds).  When no
        list initvals and no seed hints exist, K == 1 and overrides == [{}],
        exactly reproducing the legacy single-solve behavior.
        """
        # 1. User initval lists -> {sym_path: [internal values]}
        user_lists = {}
        for path, data in self.user_params.items():
            if not isinstance(data, dict):
                continue
            iv = data.get("initval")
            if not isinstance(iv, (list, tuple)):
                continue
            sym_path = self._to_symbol_path(path, name_to_index)
            if sym_path is None:
                logger.warning(
                    f"List initval on '{path}' is not a known parameter path; "
                    f"multi-seed override ignored.")
                continue
            c_type, p_name = sym_path.split('.')[0], sym_path.split('.')[-1]
            factor = self.get_conversion_factor(c_type, p_name, full_path=path)
            user_lists[sym_path] = [float(x) * factor for x in iv]

        mm_sets = self.seed_hint_sets or []

        # 2. Determine K and validate list lengths.
        Ku = max((len(v) for v in user_lists.values()), default=1)
        Km = len(mm_sets)
        K = max(Ku, Km, 1)

        for p, v in user_lists.items():
            if len(v) not in (1, Ku):
                raise ValueError(
                    f"Inconsistent seed count: initval list for '{p}' has "
                    f"length {len(v)}, expected {Ku} or 1. All initval lists in "
                    f"a params file must share one length K (or be length 1).")
        if Ku > 1 and Km > 1 and Ku != Km:
            raise ValueError(
                f"Seed-count mismatch: {Ku} user initval seeds vs {Km} "
                f"component seed hints. Provide matching counts (or length 1).")

        # 3. Merge per seed (mm first, user lists override).
        overrides = []
        for k in range(K):
            d = {}
            if mm_sets:
                src = mm_sets[k] if len(mm_sets) > 1 else mm_sets[0]
                d.update(src)
            for p, vals in user_lists.items():
                d[p] = vals[k] if len(vals) > 1 else vals[0]
            overrides.append(d)

        return K, overrides

    def _to_symbol_path(self, path, name_to_index):
        """Translate a user_params key to the internal-index path string used by
        the relaxation engine (e.g. 'lens.Lens.t_0' -> 'lens.0.t_0').  Returns
        None if the path does not correspond to a registered symbol."""
        translated = path
        parts = path.split('.')
        if len(parts) == 3:
            comp_type, name, param = parts
            if (comp_type, name) in name_to_index:
                translated = f"{comp_type}.{name_to_index[(comp_type, name)]}.{param}"
        if translated in self.master_symbol_map:
            return translated
        if path in self.master_symbol_map:
            return path
        return None

    def prune_dependency_cycles(self, cycle_nodes):
        """
        Forcefully removes nodes from the dependency graph that are known
        to cause cyclic build-order crashes.
        """
        for node in cycle_nodes:
            if node in self.dependencies:
                # Keep the node, but remove the parents that create the cycle
                # This treats the node as a 'Leaf' from the perspective of the graph builder
                self.dependencies[node] = []
                logger.debug(f"[Pruning] Severed dependency cycle at: {node}")

    def audit_scales(self):
        """
        Validates that all SAMPLED parameters have a valid init_scale.
        Warns if the user explicitly provided an init_scale for a DERIVED, FIXED, or AUXILIARY parameter.
        """
        missing_scales = []

        # 1. Identify which paths the user explicitly provided an init_scale for
        user_scale_paths = []
        standardized_user_params = self.standardize_param_names(self.raw_user_params, self.system_config)
        for k, v in standardized_user_params.items():
            if isinstance(v, dict) and "init_scale" in v:
                user_scale_paths.append(k)

        # 2. Audit all parameters in the symbol map
        for path_str in self.master_symbol_map.keys():
            parts = path_str.split('.')
            comp_type = parts[0]
            param_name = parts[-1]

            el = int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None
            cfg = self.resolve(comp_type, param_name, element=el)

            # --- Replicate Parameter.is_sampled logic ---
            # 1. is_config_present: Internal math variables (logradius, etc.) won't exist in defaults.yaml
            # 2. is_fixed: Has a hardcoded 'value'
            # 3. is_derived: Has 'expressions' to calculate it
            is_config_present = bool(cfg)
            is_fixed = cfg.get("value") is not None
            is_derived = bool(cfg.get("expressions"))

            is_sampled = is_config_present and not is_fixed and not is_derived

            # Check for a valid scale
            scale_val = cfg.get("init_scale")
            has_scale = scale_val is not None and not np.any(np.isnan(np.atleast_1d(scale_val)))

            # Issue targeted warnings based on exactly why the scale is being ignored
            if path_str in user_scale_paths:
                if is_derived:
                    logger.debug(f"init_scale for '{path_str}' will be back-propagated to sampled parents.")
                elif is_fixed:
                    logger.warning(f"init_scale for '{path_str}' ignored — it is FIXED; the sampler will not step here.")
                elif not is_config_present:
                    logger.warning(f"init_scale for '{path_str}' ignored — it is an internal AUXILIARY variable.")

            # We ONLY halt if it is a truly sampled parameter missing its scale
            if is_sampled and not has_scale:
                missing_scales.append(path_str)

        if missing_scales:
            raise ValueError(
                f"\n!!! CRITICAL CONFIGURATION ERROR !!!\n"
                f"The following SAMPLED parameters lack a valid 'init_scale'.\n"
                f"Missing: {missing_scales}\n"
                f"Check 'defaults.yaml' or provide an 'init_scale' in your config file."
            )

    def _record_diagnostic(self, severity, message, param_paths):
        """Append a structured diagnostic (deduped) for the solve/validate API.

        severity is one of "error" | "warning" | "info"; param_paths is a list
        of the parameter paths involved.  Duplicate entries (same severity,
        message, and paths) -- e.g. the same contradiction seen once per seed
        -- collapse to a single record.
        """
        entry = {
            "severity": severity,
            "message": message,
            "param_paths": list(param_paths),
        }
        if entry not in self.diagnostics:
            self.diagnostics.append(entry)

    def _provenance_label(self, rank):
        """Map a numeric provenance rank to a coarse source label."""
        if rank is None:
            return "default"
        if rank >= RANK_USER:
            return "user"
        # Microlensing distance hint (rank 30) and data-derived estimates
        # (RANK_DERIVED_DATA = 60) both come from the data channel.
        if rank == RANK_DERIVED_DATA or rank == 30:
            return "data"
        if rank > RANK_DEFAULT:
            return "solved"
        return "default"

    def export_solution(self):
        """Export the resolved parameter solution as JSON-friendly dicts.

        Returns a dict with:
          - "parameters": {user_path: {value, unit, internal_unit, lower,
            upper, init_scale, sigma, mu, fixed, derived, provenance}} where
            provenance is {rank, label, relation}.  All numeric fields are in
            the parameter's user unit (as reported by resolve()).
          - "seeds": a list of {user_path: value} start points, present only
            when multi-seed sampling produced more than one seed.

        Reads only in-memory state left behind by finalize_user_params; it does
        NOT build the PyMC model.  Must be called after System.prepare().

        Note: the relaxation engine has a known cross-build nondeterminism (two
        identical prepares may pick different derived bounds), so the exported
        bounds/values for solved quantities are one valid solution, not a
        canonical one.  See the solve_api module docstring.
        """
        def _clean(x):
            if x is None:
                return None
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(xf):
                return None
            return xf

        # Build an index -> name map per component for readable paths.
        idx_to_name = {}
        for comp_key, entries in (self.system_config or {}).items():
            if isinstance(entries, list):
                for i, c in enumerate(entries):
                    if isinstance(c, dict) and "name" in c:
                        idx_to_name[(comp_key, str(i))] = str(c["name"])

        def _display_path(internal_path):
            parts = internal_path.split('.')
            if len(parts) == 3 and (parts[0], parts[1]) in idx_to_name:
                return f"{parts[0]}.{idx_to_name[(parts[0], parts[1])]}.{parts[2]}"
            return internal_path

        parameters = {}
        for internal_path in self.master_symbol_map:
            parts = internal_path.split('.')
            c_type = parts[0]
            p_name = parts[-1]
            el = int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None
            cfg = self.resolve(c_type, p_name, element=el)

            def _first(key):
                arr = cfg.get(key)
                if arr is None:
                    return None
                try:
                    return _clean(np.atleast_1d(arr)[0])
                except (IndexError, TypeError):
                    return _clean(arr)

            sigma = _first("sigma")
            # Prefer the engine's solved value (index-form paths that match the
            # symbol map exactly, in internal units) over resolve()'s initval:
            # finalize injects derived initvals under the name-form path, which
            # a nameless resolve() call cannot see.
            if internal_path in self._last_resolved:
                factor = self.get_conversion_factor(
                    c_type, p_name, full_path=internal_path) or 1.0
                value = _clean(self._last_resolved[internal_path] / factor)
            else:
                value = _first("initval")
            derived = bool(cfg.get("expressions"))
            # A parameter is fixed when it has a hardcoded value or sigma == 0.
            fixed = (cfg.get("value") is not None) or (sigma is not None and sigma == 0)

            rank = self._last_provenance.get(internal_path)
            relation = self._last_solved_by.get(internal_path)
            label = self._provenance_label(rank)

            parameters[_display_path(internal_path)] = {
                "value": value,
                "unit": cfg.get("unit"),
                "internal_unit": cfg.get("internal_unit"),
                "lower": _first("lower"),
                "upper": _first("upper"),
                "init_scale": _first("init_scale"),
                "sigma": sigma,
                "mu": _first("mu"),
                "fixed": bool(fixed),
                "derived": derived,
                "provenance": {
                    "rank": rank,
                    "label": label,
                    "relation": relation if label == "solved" else None,
                },
            }

        result = {"parameters": parameters}

        # Multi-seed start points, converted to user units and readable paths.
        if self.seed_resolved and len(self.seed_resolved) > 1:
            seeds = []
            for seed in self.seed_resolved:
                seed_out = {}
                for internal_path, internal_val in seed.items():
                    parts = internal_path.split('.')
                    c_type = parts[0]
                    p_name = parts[-1]
                    factor = self.get_conversion_factor(
                        c_type, p_name, full_path=internal_path) or 1.0
                    seed_out[_display_path(internal_path)] = _clean(
                        internal_val / factor)
                seeds.append(seed_out)
            result["seeds"] = seeds

        return result

    def resolve_and_validate_parameters(self, user_provided_params, tolerance=1e-3):
        resolved = {str(k): float(v) for k, v in user_provided_params.items()}
        provenance = {str(k): RANK_USER for k in user_provided_params.keys()}
        resolved_scales = {}
        scale_provenance = {}
        # Per-solve record of which relation last set each variable (seed 0's
        # values win because it is solved last).  Reset each call.
        self._last_solved_by = {}

        # 1. Initialize Default Armor (Rank 20)
        def to_scalar(val):
            return val.item() if hasattr(val, 'item') else float(val)

        for path_str, sym in self.master_symbol_map.items():
            parts = path_str.split('.')
            c_type, p_name = parts[0], parts[-1]
            el = int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None
            cfg = self.resolve(c_type, p_name, element=el)

            # Read rank directly from base_defaults (not from resolve() return dict)
            param_rank = (self.base_defaults.get(c_type, {}).get(p_name, {}).get('rank')
                          or self.base_defaults.get(p_name, {}).get('rank')
                          or RANK_DEFAULT)

            if path_str not in resolved and cfg.get('initval') is not None:
                factor = self.get_conversion_factor(c_type, p_name)
                resolved[path_str] = to_scalar(cfg['initval']) * factor
                provenance[path_str] = param_rank

            if cfg.get('init_scale') is not None:
                factor = self.get_conversion_factor(c_type, p_name, full_path=path_str)
                resolved_scales[path_str] = to_scalar(cfg['init_scale']) * factor
                scale_provenance[path_str] = param_rank

        # 1.5 LAYER IN COMPONENT HINTS
        for path_str, val in self.hints.items():
            if provenance.get(path_str, 0) < RANK_USER:
                resolved[path_str] = val
                provenance[path_str] = self.hint_ranks.get(path_str, RANK_DERIVED_DATA)

        # 1.6 LAYER IN SCALE HINTS (correct indexed paths)
        # The initialization loop above calls resolve() without an index, so a hint
        # for star.0.logmass can bleed into star.1.logmass.  Apply scale_hints
        # directly using their already-normalized full paths to fix that.
        for hint_path, hint_scale in self.scale_hints.items():
            if hint_path in self.master_symbol_map:
                resolved_scales[hint_path] = hint_scale
                scale_provenance[hint_path] = RANK_DERIVED_DATA

        # 1.7 LAYER IN USER-SPECIFIED SCALES (RANK_USER)
        # When the user provides sigma or init_scale for a parameter, that scale
        # must propagate via the Jacobian (e.g. mass sigma → logmass init_scale).
        # The initialization loop records these with RANK_DEFAULT provenance because
        # it uses the parameter's rank field, not the source of the scale value.
        # Re-apply them here with RANK_USER so they win over hints and defaults
        # and their Jacobian derivatives can correctly update dependent parameters.
        for path_str in self.master_symbol_map:
            up = self.user_params.get(path_str)
            if not isinstance(up, dict):
                continue
            parts = path_str.split('.')
            c_type, p_name = parts[0], parts[-1]
            if 'init_scale' in up:
                factor = self.get_conversion_factor(c_type, p_name, full_path=path_str)
                resolved_scales[path_str] = float(up['init_scale']) * factor
                scale_provenance[path_str] = RANK_USER
            elif 'sigma' in up and up.get('sigma') is not None and float(up['sigma']) > 0:
                factor = self.get_conversion_factor(c_type, p_name, full_path=path_str)
                resolved_scales[path_str] = float(up['sigma']) * factor
                scale_provenance[path_str] = RANK_USER

        # 1.8 PREPARE USER-DEFINED LINK ASSIGNMENTS (directed, RANK_USER)
        # initval/mu links are directed: the target is defined in terms of its
        # dependencies, never the reverse.  They are re-asserted every
        # iteration so downstream physics relations always see the linked
        # value, and RANK_USER provenance protects it from being overridden.
        directed_links = []
        for link_target, link_fields in self.links.items():
            for link_field, plink in link_fields.items():
                if link_field in ("initval", "mu"):
                    directed_links.append(
                        (link_target, link_field, plink, self._link_internal_expr(plink)))

        # 2. The Relaxation Engine
        logger.info("Solving for starting values/scales of sampled parameters given user/data/default initialization....")
        iteration = 0
        max_iter = 100  # Failsafe
        _CYCLE_HIST = 6  # how many recent values to keep per variable
        value_history = {}  # {var_name: [recent rounded values]}
        pinned_vars = set()  # variables locked out of further updates due to cycle

        while iteration < max_iter:
            iteration += 1
            resolved_snapshot = dict(resolved)

            self._apply_directed_links(directed_links, resolved, provenance,
                                       resolved_scales, scale_provenance,
                                       tolerance, pinned_vars)

            self._run_standalone_solvers(resolved, provenance, tolerance,
                                         pinned_vars)

            for eq in self.all_relations:
                self._relax_equation(eq, resolved, provenance, resolved_scales, scale_provenance, tolerance,
                                     pinned_vars)

            # Convergence check: compare end-of-iteration state to start-of-iteration.
            # This correctly handles intra-iteration oscillation (two equations fighting
            # over the same variable within one pass): individual updates may fire on
            # each equation, but if the net state is unchanged the loop should stop.
            net_changed = False
            for k, v in resolved.items():
                old = resolved_snapshot.get(k)
                if old is None:
                    net_changed = True
                    break
                ref = max(abs(v), abs(old), 1e-9)
                if abs(v - old) / ref >= tolerance:
                    net_changed = True
                    break
            if not net_changed:
                break

            # Cycle detection: track per-variable history; pin any that oscillate.
            # Pinning lets other variables keep converging instead of stopping the loop.
            for k, v in resolved.items():
                if k in pinned_vars:
                    continue
                if k not in resolved_snapshot or resolved_snapshot[k] != v:
                    hist = value_history.setdefault(k, [])
                    hist.append(round(v, 8))
                    if len(hist) > _CYCLE_HIST:
                        hist.pop(0)

            for k, hist in value_history.items():
                if k in pinned_vars:
                    continue
                if len(hist) >= 4 and hist[-1] == hist[-3] and hist[-2] == hist[-4] and hist[-1] != hist[-2]:
                    logger.warning(
                        f"Cycle: '{k}' oscillates between {hist[-2]:.6g} and {hist[-1]:.6g} — "
                        f"pinned to {hist[-1]:.6g} (conflicting equal-rank constraints)."
                    )
                    pinned_vars.add(k)

        if iteration == max_iter:
            logger.warning("Relaxation engine reached max iterations — check for unstable circular dependencies.")

        logger.info(f"Done solving after {iteration} iterations.")

        # Forward scale pass: propagate scales using the same rank semantics as
        # initvals — higher rank always wins, regardless of value.  A hint at
        # rank 60 overrides a default at rank 20; a user override at rank 100
        # overrides a hint at rank 60.  This fills in derived-parameter scales
        # (e.g. mass from logmass) that the engine never solved for directly.
        for eq in self.all_relations:
            syms = [str(s) for s in eq.free_symbols]
            if not all(s in resolved for s in syms):
                continue
            for target_str in syms:
                if scale_provenance.get(target_str, 0) >= RANK_USER:
                    continue
                inputs = [s for s in syms if s != target_str]
                if not all(s in resolved_scales for s in inputs):
                    continue
                try:
                    with _sympy_time_limit(2):
                        sols = sp.solve(eq, sp.Symbol(target_str),
                                        simplify=False, check=False)
                        if not sols:
                            continue
                        sol = sols[0]
                        var = 0.0
                        active_inputs = []
                        for inp in inputs:
                            inp_sym = sp.Symbol(inp)
                            if not sol.has(inp_sym):
                                continue
                            d = float(sp.diff(sol, inp_sym).evalf(subs=resolved))
                            if np.isfinite(d) and abs(d) < 1e15:
                                var += (d * resolved_scales[inp]) ** 2
                                active_inputs.append(inp)
                    if var <= 0 or not active_inputs:
                        continue
                    new_scale = float(np.sqrt(var))
                    new_rank = sum(scale_provenance.get(s, 0) for s in active_inputs) / len(active_inputs)
                    if new_rank > scale_provenance.get(target_str, 0):
                        resolved_scales[target_str] = new_scale
                        scale_provenance[target_str] = new_rank
                except SymbolicTimeout:
                    logger.debug(f"Forward scale pass timed out solving {eq} for {target_str}; skipped.")
                except Exception as e:
                    pass

        # Backward scale pass: if the user specified init_scale on a derived parameter,
        # back-propagate it to sampled parents via the inverse Jacobian:
        #   σ_parent ≈ σ_derived / |∂derived/∂parent|
        # Rank and update semantics are identical to the initval relaxation engine:
        # a back-propagated scale gets RANK_DERIVED_USER (one tier below the user's
        # RANK_USER source), and only wins if its rank exceeds the parent's current rank.
        for eq in self.all_relations:
            syms = [str(s) for s in eq.free_symbols]
            if not all(s in resolved for s in syms):
                continue
            for derived_str in syms:
                if scale_provenance.get(derived_str, 0) < RANK_USER:
                    continue
                sigma_derived = resolved_scales.get(derived_str)
                if sigma_derived is None:
                    continue
                new_rank = min(RANK_DERIVED_USER, scale_provenance[derived_str])
                parents = [s for s in syms if s != derived_str]
                # Skip the expensive sp.solve if no parent can benefit
                if not any(
                    scale_provenance.get(p, 0) < RANK_USER and new_rank > scale_provenance.get(p, 0)
                    for p in parents
                ):
                    continue
                try:
                    with _sympy_time_limit(2):
                        sols = sp.solve(eq, sp.Symbol(derived_str),
                                        simplify=False, check=False)
                        if not sols:
                            continue
                        sol = sols[0]
                        for parent_str in parents:
                            if scale_provenance.get(parent_str, 0) >= RANK_USER:
                                continue
                            if new_rank <= scale_provenance.get(parent_str, 0):
                                continue
                            parent_sym = sp.Symbol(parent_str)
                            if not sol.has(parent_sym):
                                continue
                            d = float(sp.diff(sol, parent_sym).evalf(subs=resolved))
                            if not np.isfinite(d) or abs(d) < 1e-15:
                                continue
                            implied = sigma_derived / abs(d)
                            resolved_scales[parent_str] = implied
                            scale_provenance[parent_str] = new_rank
                            logger.debug(
                                f"Scale for {parent_str} informed by {derived_str} "
                                f"(σ={sigma_derived:.4g} → {implied:.4g})"
                            )
                except SymbolicTimeout:
                    logger.debug(f"Backward scale pass timed out solving {eq} for {derived_str}; skipped.")
                except Exception:
                    pass

        # Expose final scales for resolve() to use as a low-priority default
        # (hints and user init_scale still win; see resolve()).
        self.propagated_scales = dict(resolved_scales)

        # Snapshot provenance/scales/values for export_solution().  In the
        # multi-seed loop seed 0 is solved last, so these end up reflecting the
        # canonical seed-0 solution whose bounds/scales the model uses.
        self._last_provenance = dict(provenance)
        self._last_scale_provenance = dict(scale_provenance)
        self._last_resolved = dict(resolved)

        return resolved

    def _run_standalone_solvers(self, resolved, provenance, tolerance,
                                pinned_vars=None):
        """
        Run standalone-registered custom solvers once per relaxation
        iteration, for every instance path of their target in the symbol
        map.  A solver that raises (missing dependencies) is retried next
        iteration; results carry RANK_DERIVED_MIXED so user values and
        data-derived hints always win, and re-fire as inputs refine.
        """
        for lookup_key in self.standalone_solvers:
            solver_func = self.custom_solvers[lookup_key]
            comp, param = lookup_key.split('.')[0], lookup_key.split('.')[-1]
            for path in list(self.master_symbol_map):
                parts = path.split('.')
                if (len(parts) != 3 or parts[0] != comp or parts[2] != param
                        or not parts[1].isdigit()):
                    continue
                if pinned_vars and path in pinned_vars:
                    continue
                try:
                    val = float(solver_func(resolved, self.system_config,
                                            int(parts[1])))
                except Exception as e:
                    logger.debug(f"Standalone solver for {path} deferred: {e}")
                    continue
                if not _meaningful_change(val, resolved.get(path),
                                          RANK_DERIVED_MIXED,
                                          provenance.get(path, 0), tolerance,
                                          resolved, provenance, path):
                    continue
                resolved[path] = val
                provenance[path] = RANK_DERIVED_MIXED
                self._last_solved_by[path] = f"{lookup_key} (standalone solver)"
                logger.debug(f"Updated {path} = {val:.4g} (standalone solver)")

    def _apply_directed_links(self, directed_links, resolved, provenance,
                              resolved_scales, scale_provenance, tolerance,
                              pinned_vars=None):
        """
        Assert user-defined link assignments (target := f(deps), internal units).

        An 'initval' link IS the user's value for the target, so it always
        wins (RANK_USER).  A 'mu' link only seeds the starting point, so it
        yields to an explicit numeric initval (which also carries RANK_USER).
        Scales propagate through the link Jacobian at RANK_DERIVED_USER so an
        explicit user init_scale still wins.
        """
        for target, fld, plink, expr_int in directed_links:
            if pinned_vars and target in pinned_vars:
                continue
            if not all(d in resolved for d in plink.dep_paths):
                continue
            if fld == "mu" and provenance.get(target, 0) >= RANK_USER:
                continue
            try:
                val = float(expr_int.evalf(subs=resolved))
            except (TypeError, ValueError) as e:
                logger.debug(f"Link '{target} := {plink.expr_str}' not evaluable yet: {e}")
                continue

            if _meaningful_change(val, resolved.get(target), RANK_USER,
                                  provenance.get(target, 0), tolerance,
                                  resolved, provenance, target):
                resolved[target] = val
                provenance[target] = RANK_USER
                logger.debug(f"Updated {target} = {val:.4g} (user link: {plink.expr_str})")

            # Scale propagation through the link Jacobian
            if scale_provenance.get(target, 0) >= RANK_USER:
                continue
            var = 0.0
            any_input = False
            for dep in plink.dep_paths:
                dep_scale = resolved_scales.get(dep)
                if dep_scale is None:
                    continue
                try:
                    d = float(sp.diff(expr_int, sp.Symbol(dep)).evalf(subs=resolved))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(d) and abs(d) < 1e15:
                    var += (d * dep_scale) ** 2
                    any_input = True
            if any_input and var > 0 and RANK_DERIVED_USER > scale_provenance.get(target, 0):
                resolved_scales[target] = float(np.sqrt(var))
                scale_provenance[target] = RANK_DERIVED_USER

    def _relax_equation(self, eq, resolved, provenance, resolved_scales, scale_provenance, tolerance,
                        pinned_vars=None):

        if not isinstance(eq, sp.Eq):
            return False

        symbols_in_eq = [str(s) for s in eq.free_symbols]

        # 1. Gatekeeper: Skip if equation contains undefined variables
        if not all(s in self.master_symbol_map for s in symbols_in_eq):
            return False

        def get_rank(s):
            return provenance.get(s, 0)

        unknowns = [s for s in symbols_in_eq if s not in resolved]
        target = None
        is_contradiction = False

        # 2. Trigger Condition A: Missing Information
        if len(unknowns) == 1:
            target = unknowns[0]

        # 3. Trigger Condition B: Physics Verification
        elif len(unknowns) == 0:
            try:
                lhs = float(eq.lhs.evalf(subs=resolved))
                rhs = float(eq.rhs.evalf(subs=resolved))
                error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-9)
            except TypeError:
                return False

            if error <= tolerance:
                return False  # Equation is perfectly satisfied. Stop here.

            # Equation is broken. Find the weakest armor.
            # Sort by Rank (Ascending), then Alphabetically (Deterministic Tie-Breaker)
            candidates = sorted(symbols_in_eq, key=lambda s: (get_rank(s), s))
            target = candidates[0]

            # 4. The Contradiction Clause
            if get_rank(target) >= RANK_USER:
                is_contradiction = True

        if not target or len(unknowns) > 1:
            return False

        # Skip pinned variables so other variables can keep converging
        if pinned_vars and target in pinned_vars:
            return False

        # 5. Calculate New Armor
        # Use min(input_ranks) so a chain is only as strong as its weakest link.
        # Condition A floor is RANK_DEFAULT-1 so an indirectly-derived value
        # (e.g. ecc from K when secosw/sesinw are unavailable) always yields to
        # a proper expression-path derivation or a default in Condition B.
        # Condition B floor is RANK_DEFAULT so conflict resolution never produces
        # armor weaker than an unseeded default.
        inputs = [s for s in symbols_in_eq if s != target]
        min_input_rank = min(get_rank(s) for s in inputs) if inputs else RANK_DEFAULT
        # Condition A floor: RANK_DEFAULT-1 so indirect derivations always yield
        #   to expression-path derivations or defaults in Condition B.
        # Condition B floor: 0 so low-rank inputs (e.g. pm defaults at rank 10)
        #   produce low-rank results — preventing them from blocking higher-rank
        #   derivations from the t_E/theta_E/pi_rel chain.
        rank_floor = RANK_DEFAULT - 1 if len(unknowns) == 1 else 0
        new_rank = max(rank_floor, min(RANK_DERIVED_USER, min_input_rank))

        if is_contradiction:
            # All variables in this equation were explicitly set by the user.
            # Overriding any of them would silently discard user intent — this
            # commonly happens with "default identity" relations like
            # Eq(radiussed, radius) when the user intentionally gave the two
            # parameters different MAP values on a second-iteration run.
            # Leave every value untouched and let the sampler and likelihood
            # sort out any inconsistency.
            logger.debug(
                f"Over-constrained: all variables in '{eq}' have user rank "
                f"but equation is violated (error={error:.4g}). "
                f"Leaving all user values unchanged.")
            self._record_diagnostic(
                "error",
                f"Over-constrained relation '{eq.lhs} = {eq.rhs}' is violated "
                f"(relative error {error:.4g}): every parameter it links was "
                f"set explicitly, so no value can be adjusted to satisfy it.",
                symbols_in_eq,
            )
            return False

        return self._execute_solve(eq, target, resolved, provenance, new_rank, resolved_scales, scale_provenance,
                                   inputs, pinned_vars=pinned_vars, tolerance=tolerance)

    def _execute_solve(self, eq, target_str, resolved, provenance, new_rank, resolved_scales, scale_provenance, inputs,
                       pinned_vars=None, tolerance=1e-3):
        if pinned_vars and target_str in pinned_vars:
            return False

        # 1. ALWAYS check custom solvers FIRST
        parts = target_str.split('.')
        lookup_key = f"{parts[0]}.{parts[-1]}"
        idx = int(parts[1]) if len(parts) >= 3 else 0

        if lookup_key in self.custom_solvers:
            solver_func = self.custom_solvers[lookup_key]
            try:
                valid_val = float(solver_func(resolved, self.system_config, idx))
                if not _meaningful_change(valid_val, resolved.get(target_str), new_rank,
                                          provenance.get(target_str, 0), tolerance,
                                          resolved, provenance, target_str):
                    return False
                resolved[target_str] = valid_val
                provenance[target_str] = new_rank
                self._last_solved_by[target_str] = f"{eq.lhs} = {eq.rhs}"
                logger.debug(f"Updated {target_str} = {valid_val:.4g} (custom solver)")
                return True
            except Exception as e:
                # A custom solver is a shortcut for one specific relation
                # (e.g. K -> companion mass).  If it can't run (missing
                # dependencies), fall through to the generic symbolic solver
                # so OTHER equations targeting this parameter (e.g.
                # q * M_primary = M_companion) still get their chance.
                logger.debug(f"Custom solver failed for {target_str}: {e}; "
                             f"falling back to symbolic solve.")

        # 2. Skip equations whose symbolic inversion has previously timed out
        if target_str in self.symbolic_blacklist:
            return False

        # 3. Diagnostic Timing for the Symbolic Solver
        logger.debug(f"Attempting to solve: {eq} for target: {target_str}")

        # Print the equation with substituted numerical values ---
        try:
            # Format as "lhs = rhs" instead of "Eq(lhs, rhs)"
            eq_str = f"{eq.lhs} = {eq.rhs}"

            # Replace only the known symbols so the math structure is preserved
            for s in eq.free_symbols:
                s_str = str(s)
                if s_str in resolved:
                    # Format to 5 sig figs (handles scientific notation automatically)
                    val_str = f"{float(resolved[s_str]):.5g}"
                    # Use regex with word boundaries to replace exact variable names
                    eq_str = re.sub(rf'\b{re.escape(s_str)}\b', val_str, eq_str)

            logger.debug(f"  Substituted: {eq_str}")
        except Exception as e:
            pass

        start_time = time.time()

        def handler(signum, frame):
            raise TimeoutError("Symbolic solver timed out!")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(2)  # 2-second hard limit

        solutions = []
        used_nsolve = False

        try:
            target_sym = next(s for s in eq.free_symbols if str(s) == target_str)
            # simplify=False + check=False: sp.solve's default post-simplify and
            # its checksol() verification pass together dominate prepare()
            # (sympy.simplify + sympy.checksol) and are redundant here -- every
            # candidate solution is evaluated numerically, bounds-checked, and
            # (for multiple roots) scored against the other relations below, so
            # the code already does its own root validation. simplify only
            # changes the expression's form (identical numeric value); check
            # only pre-filters roots the numeric bounds/scoring pass re-filters.
            solutions = sp.solve(eq, target_sym, dict=False,
                                 simplify=False, check=False)
            elapsed = time.time() - start_time
            logger.debug(f"sp.solve finished in {elapsed:.4f}s for {target_str}")
        except TimeoutError:
            logger.debug(f"sp.solve timed out for {target_str} — blacklisting.")
            self.symbolic_blacklist.add(target_str)
            signal.alarm(0)
            return False
        except Exception as e:
            logger.debug(f"sp.solve exception for {target_str}: {e}")
            signal.alarm(0)
            return False
        finally:
            signal.alarm(0)

        # 3. Fallback to nsolve if analytical failed
        if not solutions:
            try:
                with _sympy_time_limit(2):
                    guess = float(resolved.get(target_str, 1.0))
                    sub_dict = {s: resolved[str(s)] for s in inputs}
                    expr = (eq.lhs - eq.rhs).subs(sub_dict).evalf()
                    solutions = [sp.nsolve(expr, target_sym, guess)]
                    used_nsolve = True
            except Exception:
                return False

        # 4. Validation — collect all in-bounds solutions
        cfg = self.resolve(parts[0], parts[-1], shape=(), element=idx)
        lower = cfg['lower'][0] if cfg.get('lower') is not None else -np.inf
        upper = cfg['upper'][0] if cfg.get('upper') is not None else np.inf

        valid_solutions = []
        for sol in solutions:
            try:
                val = float(sol.evalf(subs=resolved)) if not used_nsolve else float(sol)
                if lower <= val <= upper:
                    valid_solutions.append((val, sol))
            except (TypeError, ValueError):
                continue

        if not valid_solutions:
            return False

        # When multiple roots exist (e.g. ± from a quadratic), pick the one that
        # best satisfies other equations sharing this variable.  This prevents
        # sign-ambiguity failures (e.g. mu_ra_rel getting the wrong sign and
        # making pi_E_E irreconcilable).
        if len(valid_solutions) == 1:
            valid_val, valid_sol = valid_solutions[0]
        else:
            best_val, best_sol, best_score = None, None, float('inf')
            for val, sol in valid_solutions:
                temp = {**resolved, target_str: val}
                score = 0.0
                for other_eq in self.all_relations:
                    other_syms = [str(s) for s in other_eq.free_symbols]
                    if target_str not in other_syms or other_eq is eq:
                        continue
                    if not all(s in temp for s in other_syms):
                        continue
                    try:
                        lhs = float(other_eq.lhs.evalf(subs=temp))
                        rhs = float(other_eq.rhs.evalf(subs=temp))
                        ref = max(abs(lhs), abs(rhs), 1e-9)
                        score += ((lhs - rhs) / ref) ** 2
                    except Exception:
                        pass
                if score < best_score:
                    best_score = score
                    best_val, best_sol = val, sol
            valid_val, valid_sol = best_val, best_sol

        # 5. Apply Value and Armor
        if valid_val is not None:
            if not _meaningful_change(valid_val, resolved.get(target_str), new_rank,
                                      provenance.get(target_str, 0), tolerance,
                                      resolved, provenance, target_str):
                return False
            # 5b. Jacobian-filtered rank refinement.
            # Replace raw min(input_ranks) with min(active_input_ranks), where
            # "active" means the input has a non-negligible Jacobian contribution.
            # This prevents incidentally-zero inputs (e.g. mu_ra_rel ≈ 0 in the
            # mu_dec_rel = sqrt(mu_rel_mag² - mu_ra_rel²) equation) from dragging
            # down the trustworthiness of the result.
            jac_active_inputs = []
            if not used_nsolve and hasattr(valid_sol, 'free_symbols') and inputs:
                for parent_str in inputs:
                    parent_sym = sp.Symbol(parent_str)
                    if not valid_sol.has(parent_sym):
                        continue
                    try:
                        d = float(sp.diff(valid_sol, parent_sym).evalf(subs=resolved))
                        if np.isfinite(d) and abs(d) > 1e-6:
                            jac_active_inputs.append(parent_str)
                    except Exception:
                        jac_active_inputs.append(parent_str)  # conservative: include on error

            if jac_active_inputs:
                min_jac_rank = min(provenance.get(s, RANK_DEFAULT) for s in jac_active_inputs)
                # Refine upward only: never reduce a rank that was already determined
                # by a valid floor (e.g. Condition A floor of RANK_DEFAULT-1).
                new_rank = max(new_rank, min(RANK_DERIVED_USER, min_jac_rank))

            resolved[target_str] = valid_val
            provenance[target_str] = new_rank
            self._last_solved_by[target_str] = f"{eq.lhs} = {eq.rhs}"
            logger.debug(f"Updated {target_str} = {valid_val:.4g} (rank: {new_rank})")

            # 6. Independent Scale Propagation via Jacobian
            if not used_nsolve and hasattr(valid_sol, 'free_symbols') and inputs:
                scale_variance = 0.0
                valid_scale_inputs = []

                for parent_str in inputs:
                    parent_sym = sp.Symbol(parent_str)
                    if not valid_sol.has(parent_sym):
                        continue

                    parent_scale = resolved_scales.get(parent_str, 1e-9)
                    valid_scale_inputs.append(parent_str)

                    try:
                        derivative = sp.diff(valid_sol, parent_sym)
                        sensitivity = float(derivative.evalf(subs=resolved))

                        # Only update variance if the sensitivity is a sane number
                        if np.isfinite(sensitivity) and abs(sensitivity) < 1e15:
                            scale_variance += (sensitivity * parent_scale) ** 2

                    except (OverflowError, FloatingPointError, TypeError):
                        # If the derivative is explosive, we have reached a physical regime
                        # where variance propagation is numerically invalid.
                        # Treat the variance contribution as undefined/maxed out.
                        scale_variance = np.inf
                        break

                if valid_scale_inputs:
                    new_scale_rank = sum(scale_provenance.get(s, 0) for s in valid_scale_inputs) / len(
                        valid_scale_inputs)
                    new_scale = float(np.sqrt(scale_variance))
                    if new_scale_rank > scale_provenance.get(target_str, 0):
                        resolved_scales[target_str] = new_scale
                        scale_provenance[target_str] = new_scale_rank

                    if target_str in self.user_params and isinstance(self.user_params[target_str], dict):
                        factor = self.get_conversion_factor(parts[0], parts[-1], full_path=target_str)
                        self.user_params[target_str]["init_scale"] = resolved_scales[target_str] / factor

            return True

        return False

    def _attempt_rank_upgrade(self, eq, resolved, provenance, resolved_scales, scale_provenance):
        symbols_in_eq = [str(s) for s in eq.free_symbols]

        def get_rank(s):
            return provenance.get(s, RANK_DEFAULT)

        # 1. We need ALL symbols to be known except at most one
        # If any symbol is NOT in master_symbol_map, we can't solve this equation.
        if not all(s in self.master_symbol_map for s in symbols_in_eq):
            return False

        # 2. Identify the target:
        # Pick the symbol with the lowest provenance.
        target = min(symbols_in_eq, key=get_rank)

        # 3. Check dependencies: Are all inputs known?
        inputs = [s for s in symbols_in_eq if s != target]
        if not all(s in resolved for s in inputs):
            return False

        # 4. Calculate bottleneck rank
        input_ranks = [get_rank(s) for s in inputs]
        new_rank = min(input_ranks) if input_ranks else RANK_DEFAULT

        # 5. Monotonic upgrade check
        if new_rank > get_rank(target):
            logger.debug(f"Rank upgrade: {target} ({get_rank(target)} -> {new_rank}) via {eq}")
            return self._solve_and_update(eq, target, resolved, provenance, new_rank, resolved_scales, scale_provenance)

        return False

    def _solve_and_update(self, eq, target_str, resolved, provenance_map, new_rank, resolved_scales, scale_provenance):
        target_sym = next(s for s in eq.free_symbols if str(s) == target_str)

        # 1. Setup bounds and solver lookups
        parts = target_str.split('.')
        el = int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else None
        cfg = self.resolve(parts[0], parts[-1], shape=(), element=el)
        lower = cfg['lower'][0] if cfg.get('lower') is not None else -np.inf
        upper = cfg['upper'][0] if cfg.get('upper') is not None else np.inf

        # 2. Logic to isolate target and solve
        solutions = []
        valid_val = None
        valid_sol = None

        # Trivial Isolation
        if eq.lhs == target_sym:
            solutions = [eq.rhs]
        elif eq.rhs == target_sym:
            solutions = [eq.lhs]
        else:
            try:
                solutions = sp.solve(eq, target_sym, dict=False)
            except Exception:
                pass

        # Fallback to nsolve
        if not solutions:
            try:
                guess = float(resolved.get(target_str, 1.0))
                sub_dict = {s: resolved[str(s)] for s in eq.free_symbols if str(s) != target_str}
                expr = (eq.lhs - eq.rhs).subs(sub_dict).evalf()
                solutions = [sp.nsolve(expr, target_sym, guess)]
            except Exception:
                return False

        # 3. Validate Solution
        for sol in solutions:
            try:
                val = float(sol.evalf(subs=resolved))
                if lower <= val <= upper:
                    valid_val = val
                    valid_sol = sol
                    break
            except (TypeError, ValueError):
                continue

        # 4. Final Update Guard (The "Provenance Engine")
        if valid_val is not None:
            # Update Value
            resolved[target_str] = valid_val
            provenance_map[target_str] = new_rank

            # Propagate Scale (Calculus-based)
            if hasattr(valid_sol, 'free_symbols'):
                scale_variance = 0.0
                for parent_sym in valid_sol.free_symbols:
                    parent_str = str(parent_sym)
                    # Fallback to a small epsilon if parent scale is missing
                    parent_scale = resolved_scales.get(parent_str, 1e-9)

                    derivative = sp.diff(valid_sol, parent_sym)
                    sensitivity = float(derivative.evalf(subs=resolved))
                    scale_variance += (sensitivity * parent_scale) ** 2

                # Apply scale update only if rank allows
                if new_rank >= scale_provenance.get(target_str, 0):
                    resolved_scales[target_str] = float(np.sqrt(scale_variance))
                    scale_provenance[target_str] = new_rank

                    # Sync scale back to user_params if necessary
                    if target_str in self.user_params and isinstance(self.user_params[target_str], dict):
                        factor = self.get_conversion_factor(parts[0], parts[-1], full_path=target_str)
                        self.user_params[target_str]["init_scale"] = resolved_scales[target_str] / factor

            return True

        return False

    def print_contradiction_warning(self, eq, error):
        logger.warning(
            "!" * 60 + "\n"
            "WARNING: PHYSICAL CONTRADICTION DETECTED\n"
            f"Relation: {eq}\n"
            f"Relative Error: {error:.2%}\n"
            + "-" * 60 + "\n"
            "The parameters provided in your config do not satisfy this equation.\n"
            "Verify your starting values; a bad initialization will destroy NUTS efficiency.\n"
            + "!" * 60)