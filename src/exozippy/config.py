# src/exozippy/config.py
import numpy as np
import yaml
import copy
from pathlib import Path

class ConfigManager:
    def __init__(self, user_params):
        self.user_params = user_params
        self.base_defaults = {}

        # 1. Search the components directory dynamically
        components_dir = Path(__file__).parent / "components"

        # 2. Recursively find and load ALL defaults.yaml files
        for defaults_file in components_dir.rglob("defaults.yaml"):
            with open(defaults_file, "r") as f:
                comp_defaults = yaml.safe_load(f) or {}
                self._deep_merge(self.base_defaults, comp_defaults)

    def _deep_merge(self, base, overrides):
        """
        Recursively merges dictionaries so that nested keys (like 'expressions')
        are preserved unless explicitly overwritten.
        """
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def resolve(self, component_type, param_name, shape=(), internal_overrides=None, names=None):
        import astropy.units as u  # Ensure u is available for conversion

        # 1. Start with the global/DNA defaults
        base = copy.deepcopy(self.base_defaults.get(param_name, {}))

        # 2. Deep-merge the component-specific defaults
        comp_defaults = self.base_defaults.get(component_type, {})
        comp_override = comp_defaults.get(param_name, {})
        if comp_override:
            self._deep_merge(base, copy.deepcopy(comp_override))

        n_elements = int(np.prod(shape)) if shape != () else 1

        # 3. Detect if the user is overriding the unit
        base_unit_str = base.get("unit", "")
        new_unit_str = None

        # Check all possible YAML keys for a unit override
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

        # Calculate unit scaling factor (Default Unit -> User Unit)
        unit_scaling = 1.0
        if new_unit_str and base_unit_str and new_unit_str != base_unit_str:
            try:
                unit_scaling = u.Unit(base_unit_str).to(u.Unit(new_unit_str))
            except Exception:
                unit_scaling = 1.0

        # 4. Initialize the resolved dictionary with scaled defaults
        resolved = {
            "shape": shape,
            "user_modified": False,
            "user_prior_modified": False,
            "unit": new_unit_str if new_unit_str else base_unit_str,
            "internal_unit": base.get("internal_unit"),
            "latex": base.get("latex", ""),
            "description": base.get("description", ""),
            "expressions": base.get("expressions", {})
        }

        tuning_keys = ["initval", "init_scale"]
        physics_keys = ["lower", "upper", "mu", "sigma", "gaussian_width"]
        all_numeric = tuning_keys + physics_keys

        for key in all_numeric:
            val = base.get(key)
            if val is not None:
                # SCALE the default value to the new unit!
                resolved[key] = np.full(n_elements, float(val) * unit_scaling, dtype=float)
            else:
                resolved[key] = None

        def apply_value(key, current_arr, idx, new_val):
            if new_val is None:
                return current_arr

            # Initialize array with NaNs if it was empty
            if current_arr is None:
                current_arr = np.full(n_elements, np.nan, dtype=float)
                resolved[key] = current_arr

            v = float(new_val)

            if np.isnan(current_arr[idx]):
                current_arr[idx] = v
            elif key == "lower":
                current_arr[idx] = max(current_arr[idx], v)  # Most restrictive lower bound wins
            elif key == "upper":
                current_arr[idx] = min(current_arr[idx], v)  # Most restrictive upper bound wins
            else:
                current_arr[idx] = v  # initval, sigma, etc., just overwrite

            return current_arr

        # 5. Apply Internal Heuristics (e.g., K-based mass, RV mean)
        if internal_overrides:
            for key in all_numeric:
                if key in internal_overrides:
                    val = internal_overrides[key]
                    for i in range(n_elements):
                        v = val[i] if isinstance(val, (list, np.ndarray)) else val

                        # FIX: Scale the internal heuristic (which assumes default units)
                        # to the user's chosen unit.
                        v_scaled = float(v) * unit_scaling

                        apply_value(key, resolved[key], i, v_scaled)

        # 6. Apply User Overrides
        for i in range(n_elements):
            keys_to_check = [f"{component_type}.{param_name}", f"{component_type}.{i}.{param_name}"]

            # Teach the config manager to look for named components!
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
                                # Convert the base scalar string to a list if it isn't one already
                                if not isinstance(resolved[str_key], list):
                                    resolved[str_key] = [resolved[str_key]] * n_elements
                                resolved[str_key][i] = ov[str_key]
                            else:
                                resolved[str_key] = ov[str_key]

                    # Optional: Auto-sync init_scale if user changed sigma but forgot scale
                    if "sigma" in ov and "init_scale" not in ov and resolved["sigma"] is not None:
                        apply_value("init_scale", resolved.get("init_scale"), i, ov["sigma"])

        return resolved

    def get_conversion_factor(self, component_type, param_name):
        """
        Looks up the units in the master YAML and returns the
        numeric multiplier to go from User -> Internal.
        """
        import astropy.units as u

        comp_cfg = self.base_defaults.get(component_type, {})
        param_cfg = comp_cfg.get(param_name, {})

        u_str = param_cfg.get("unit", "")
        i_str = param_cfg.get("internal_unit", "")

        # Default to 1.0 if units aren't specified
        if not u_str or not i_str:
            return 1.0

        try:
            # Handle dex units (log space)
            if "dex" in u_str or "dex" in i_str:
                # If they are both dex, the factor is 1.0
                # We assume the user isn't doing dex(m/s) to dex(km/s)
                return 1.0

            user_u = u.Unit(u_str)
            internal_u = u.Unit(i_str)
            return user_u.to(internal_u)
        except Exception:
            return 1.0