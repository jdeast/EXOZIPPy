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

    def resolve(self, component_type, param_name, shape=()):
        """
        Merges defaults and user overrides, then applies unit conversion.
        """
        # 1. Start with the global/DNA defaults (from celestial_body)
        base = copy.deepcopy(self.base_defaults.get(param_name, {}))

        # 2. Deep-merge the component-specific overrides (from star, planet, etc.)
        comp_defaults = self.base_defaults.get(component_type, {})
        comp_override = comp_defaults.get(param_name, {})
        if comp_override:
            self._deep_merge(base, copy.deepcopy(comp_override))

        n_elements = int(np.prod(shape)) if shape != () else 1

        # 3. Initialize the resolved dictionary
        resolved = {
            "shape": shape,
            "user_modified": False,
            "user_prior_modified": False,
            "latex": base.get("latex", ""),
            "description": base.get("description", ""),
            "unit": base.get("unit"),
            "internal_unit": base.get("internal_unit"),
            "expressions": base.get("expressions", {})
        }

        tuning_keys = ["initval", "init_scale"]
        physics_keys = ["lower", "upper", "mu", "sigma", "gaussian_width"]
        all_numeric = tuning_keys + physics_keys

        # 4. Process Numeric Keys from Defaults
        for key in all_numeric:
            val = base.get(key)
            resolved[key] = np.full(n_elements, float(val), dtype=float) if val is not None else None

        # 5. Apply Overrides
        for i in range(n_elements):
            # Check both global (star.mass) and indexed (star.0.mass)
            keys_to_check = [f"{component_type}.{param_name}", f"{component_type}.{i}.{param_name}"]
            for k in keys_to_check:
                if k in self.user_params:
                    ov = self.user_params[k]

                    # 1. If the YAML key is blank, PyYAML parses it as None. Skip it.
                    if ov is None:
                        continue

                    # 2. If the user used shorthand (e.g. `planet.mass: 5.0`), assume it's an initval
                    if not isinstance(ov, dict):
                        ov = {"initval": ov}

                    resolved["user_modified"] = True

                    # Check if any physics-altering key was touched
                    if any(pk in ov for pk in physics_keys):
                        resolved["user_prior_modified"] = True

                    for key in all_numeric:
                        if key in ov and ov[key] is not None:
                            if resolved[key] is None:
                                resolved[key] = np.full(n_elements, np.nan, dtype=float)
                            resolved[key][i] = float(ov[key])
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