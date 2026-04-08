# src/exozippy/config.py
import numpy as np
import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, user_params):
        self.user_params = user_params

        # Load the internal physics defaults (This creates self.base_defaults)
        base_path = Path(__file__).parent / "exozippy_params.yaml"
        with open(base_path, "r") as f:
            self.base_defaults = yaml.safe_load(f)

        # src/exozippy/config.py

    def resolve(self, component_type, param_name, shape=()):
        """
        Merges defaults and user overrides, then applies unit conversion
        to move everything into Internal Math Space.
        """
        # 1. Access the master defaults for this component/parameter
        base = self.base_defaults.get(component_type, {}).get(param_name, {}).copy()
        n_elements = int(np.prod(shape)) if shape != () else 1

        # 2. Determine the conversion factor (User Unit -> Internal Unit)
        #factor = self.get_conversion_factor(component_type, param_name)

        # 3. Initialize the resolved dictionary
        resolved = {
            "shape": shape,
            "user_modified": False,
            "latex": base.get("latex", ""),
            "description": base.get("description", ""),
            "unit": base.get("unit"),
            "internal_unit": base.get("internal_unit"),
            "expressions": base.get("expressions", {})
        }

        # 4. Process Numeric Keys
        # These keys MUST be converted to internal units to prevent the Curvature Explosion
        numeric_keys = ["lower", "upper", "initval", "init_scale", "mu", "sigma"]
        for key in numeric_keys:
            val = base.get(key)
            if val is not None:
                # Convert default value from exozippy_params.yaml
                resolved[key] = np.full(n_elements, float(val), dtype=float)
            else:
                resolved[key] = None

        # 5. Apply Global User Overrides (e.g., 'star.mass: 1.2')
        global_key = f"{component_type}.{param_name}"
        if global_key in self.user_params:
            ov = self.user_params[global_key]
            resolved["user_modified"] = True
            for key in numeric_keys:
                if key in ov:
                    # If the key wasn't in defaults, create the array first
                    if resolved[key] is None:
                        resolved[key] = np.full(n_elements, np.nan, dtype=float)
                    # Convert user override to internal units
                    resolved[key][:] = float(ov[key])

        # 6. Apply Indexed Overrides (e.g., 'star.0.mass: 1.1')
        # This handles vector parameters like multiple planets or stars
        for i in range(n_elements):
            indexed_key = f"{component_type}.{i}.{param_name}"
            if indexed_key in self.user_params:
                ov = self.user_params[indexed_key]
                resolved["user_modified"] = True
                for key in numeric_keys:
                    if key in ov:
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