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

class SymbolicTimeout(Exception): pass

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
class ConfigManager:
    def __init__(self, user_params, system_config=None):
        self.raw_user_params = user_params
        self.custom_solvers = {}

        # If config is provided, standardize right away
        if system_config is not None:
            self.user_params = self.standardize_param_names(user_params, system_config)
        else:
            self.user_params = user_params

        self.system_config = system_config or {}
        self.base_defaults = {}
        self.all_relations = []
        self.master_symbol_map = {}

        # Storage for hints passed by components during Registration Sweep
        self.hints = {}
        self.hint_ranks = {}
        self.scale_hints = {}   # path -> init_scale in internal units
        self.propagated_scales = {}  # path -> init_scale (internal) from Jacobian forward pass
        self.dependencies = {}
        self.symbolic_blacklist = set()

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

                    raw_map = module.get_symbol_map(entry_cfg)

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
                        self.all_relations.append(rel.subs(subs))

        for defaults_file in components_dir.rglob("defaults.yaml"):
            with open(defaults_file, "r") as f:
                comp_defaults = yaml.safe_load(f) or {}
                self._deep_merge(self.base_defaults, comp_defaults)

        # Add this inside ConfigManager.__init__ after filling all_relations
        for rel in self.all_relations:
            logger.debug(f"Relation: {rel}")

    def register_custom_solver(self, target_str, solver_func):
        self.custom_solvers[target_str] = solver_func

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

    def add_hint(self, path, value, rank=RANK_DERIVED_DATA):
        """
        API for components to register their data-driven guesses.
        Converts human-readable paths to strict indices and scales to internal units.
        """
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

        # 3. Store the fully processed, ready-to-use hint
        self.hints[translated_path] = internal_value
        self.hint_ranks[translated_path] = rank

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

    def resolve(self, component_type, param_name, shape=(), internal_overrides=None, names=None):
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
        base_unit_str = base.get("unit", "")
        new_unit_str = None

        keys_to_check_global = []
        for i in range(n_elements):
            keys = [f"{component_type}.{param_name}", f"{component_type}.{i}.{param_name}"]
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
                             f"{component_type}.{i}.{param_name}"]
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
            scale_keys = [f"{component_type}.{param_name}", f"{component_type}.{i}.{param_name}"]
            if names and i < len(names):
                scale_keys.append(f"{component_type}.{names[i]}.{param_name}")
            for k in scale_keys:
                if k in self.scale_hints:
                    apply_value("init_scale", resolved["init_scale"], i,
                                self.scale_hints[k] / internal_factor)
                    break

        for i in range(n_elements):
            keys_to_check = [f"{component_type}.{param_name}", f"{component_type}.{i}.{param_name}"]
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
                            apply_value(key, resolved[key], i, ov[key])

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
        Transforms user-facing component names into strict internal indices.
        Example: 'rvinstrument.EXPERT.gamma' -> 'rvinstrument.0.gamma'
        """
        standardized = {}
        if not user_params:
            return standardized

        for key, val in user_params.items():
            parts = key.split(".")
            if len(parts) == 3:
                comp_type, comp_name, param_name = parts

                # STRICT CHECK: Ensure the prefix actually exists in the YAML
                if comp_type not in config:
                    raise ValueError(
                        f"\n!!! STRICT NAMING ERROR !!!\n"
                        f"Parameter '{key}' uses the prefix '{comp_type}', but '{comp_type}' "
                        f"is not defined in your system configuration.\n"
                        f"Ensure your YAML block names match your parameter prefixes exactly."
                    )

                try:
                    idx = next(i for i, c in enumerate(config[comp_type]) if c.get("name") == comp_name)
                    standard_key = f"{comp_type}.{idx}.{param_name}"
                    standardized[standard_key] = val
                    continue
                except StopIteration:
                    pass  # Name wasn't found in the config block, leave it as-is
            standardized[key] = val
        return standardized

    def finalize_user_params(self):
        """
        Called by the System object AFTER all components have registered their hints.
        """
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

        # --- FALLBACK TO LEAFS ---
        for path in list(self.user_params.keys()):
            if path not in self.master_symbol_map:
                self.master_symbol_map[path] = sp.Symbol(path)
                logger.debug(f"Registered as leaf: {path}")

                # Push the user's value into the solver's initial state
                data = self.user_params[path]
                val = data.get("initval") if isinstance(data, dict) else data
                if val is not None:
                    c_type = path.split('.')[0]
                    p_name = path.split('.')[-1]
                    factor = self.get_conversion_factor(c_type, p_name, full_path=path)
                    flat_params[self.master_symbol_map[path]] = float(val) * factor

        resolved_flat = self.resolve_and_validate_parameters({str(k): v for k, v in flat_params.items()})

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

            # Prevent overwriting dictionaries
            if final_path not in self.user_params or not isinstance(self.user_params[final_path], dict):
                self.user_params[final_path] = {"initval": user_val, "derived": True}
            else:
                self.user_params[final_path]["initval"] = user_val
                self.user_params[final_path]["derived"] = True

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

            cfg = self.resolve(comp_type, param_name)

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
                    logger.warning(f"init_scale for '{path_str}' ignored — it is a DERIVED parameter; scale propagates from parents.")
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

    def resolve_and_validate_parameters(self, user_provided_params, tolerance=1e-3):
        resolved = {str(k): float(v) for k, v in user_provided_params.items()}
        provenance = {str(k): RANK_USER for k in user_provided_params.keys()}
        resolved_scales = {}
        scale_provenance = {}

        # 1. Initialize Default Armor (Rank 20)
        def to_scalar(val):
            return val.item() if hasattr(val, 'item') else float(val)

        for path_str, sym in self.master_symbol_map.items():
            parts = path_str.split('.')
            c_type, p_name = parts[0], parts[-1]
            cfg = self.resolve(c_type, p_name)

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

        # 2. The Relaxation Engine
        logger.info("Solving for starting values/scales of sampled parameters given user/data initialization....")
        updated = True
        iteration = 0
        max_iter = 100  # Failsafe

        while updated and iteration < max_iter:
            updated = False
            iteration += 1
            for eq in self.all_relations:
                if self._relax_equation(eq, resolved, provenance, resolved_scales, scale_provenance, tolerance):
                    updated = True

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
                inputs = [s for s in syms if s != target_str]
                if not all(s in resolved_scales for s in inputs):
                    continue
                try:
                    sols = sp.solve(eq, sp.Symbol(target_str))
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
                except Exception as e:
                    pass

        # Expose final scales for resolve() to use as a low-priority default
        # (hints and user init_scale still win; see resolve()).
        self.propagated_scales = dict(resolved_scales)

        return resolved

    def _relax_equation(self, eq, resolved, provenance, resolved_scales, scale_provenance, tolerance):

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

        # 5. Calculate New Armor
        # Use min(input_ranks) so a chain is only as strong as its weakest link.
        # Condition A floor is RANK_DEFAULT-1 so an indirectly-derived value
        # (e.g. ecc from K when secosw/sesinw are unavailable) always yields to
        # a proper expression-path derivation or a default in Condition B.
        # Condition B floor is RANK_DEFAULT so conflict resolution never produces
        # armor weaker than an unseeded default.
        inputs = [s for s in symbols_in_eq if s != target]
        min_input_rank = min(get_rank(s) for s in inputs) if inputs else RANK_DEFAULT
        rank_floor = RANK_DEFAULT - 1 if len(unknowns) == 1 else RANK_DEFAULT
        new_rank = max(rank_floor, min(RANK_DERIVED_USER, min_input_rank))

        if is_contradiction:
            logger.warning(
                f"Over-constrained contradiction detected.\n"
                f"  Equation: {eq}\n"
                f"  Action: sacrificing '{target}' to enforce physical consistency.")

        return self._execute_solve(eq, target, resolved, provenance, new_rank, resolved_scales, scale_provenance,
                                   inputs)

    def _execute_solve(self, eq, target_str, resolved, provenance, new_rank, resolved_scales, scale_provenance, inputs):
        # 1. ALWAYS check custom solvers FIRST
        parts = target_str.split('.')
        lookup_key = f"{parts[0]}.{parts[-1]}"
        idx = int(parts[1]) if len(parts) >= 3 else 0

        if lookup_key in self.custom_solvers:
            solver_func = self.custom_solvers[lookup_key]
            try:
                valid_val = float(solver_func(resolved, self.system_config, idx))
                resolved[target_str] = valid_val
                provenance[target_str] = new_rank
                logger.debug(f"Updated {target_str} = {valid_val:.4g} (custom solver)")
                return True
            except Exception as e:
                logger.debug(f"Custom solver failed for {target_str}: {e}")
                return False

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
            solutions = sp.solve(eq, target_sym, dict=False)
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
                guess = float(resolved.get(target_str, 1.0))
                sub_dict = {s: resolved[str(s)] for s in inputs}
                expr = (eq.lhs - eq.rhs).subs(sub_dict).evalf()
                solutions = [sp.nsolve(expr, target_sym, guess)]
                used_nsolve = True
            except Exception:
                return False

        # 4. Validation — collect all in-bounds solutions
        cfg = self.resolve(parts[0], parts[-1], shape=())
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
            resolved[target_str] = valid_val
            provenance[target_str] = new_rank
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
        cfg = self.resolve(parts[0], parts[-1], shape=())
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