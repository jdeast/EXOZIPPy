import numpy as np
import pytensor.tensor as pt
from typing import Dict, Any, List


class ModelAuditor:
    def __init__(self, model, system, transformed_inits):
        self.model = model
        self.system = system
        self.transformed_inits = transformed_inits
        self.param_lookup = system.get_parameter_lookup()
        self.user_params = system.user_params
        self.all_params = system.get_all_parameters()

        # Internal Filter Suffixes
        self.hidden_suffixes = ["_raw", "_raw_n", "_raw_u", "_interval__", "_log__", "__"]

    def get_aggregated_logps(self) -> tuple[Dict[str, float], Dict[str, float]]:
        model_input_names = [v.name for v in self.model.value_vars]
        filtered_point = {k: v for k, v in self.transformed_inits.items() if k in model_input_names}
        raw_logps = self.model.point_logps(filtered_point)

        param_logps = {}
        other_nodes = {}

        # ONLY group logps for parameters that are actively being sampled
        sampled_labels = [p.label for p in self.all_params if p.expression is None]

        for node_name, lp in raw_logps.items():
            if any(node_name.endswith(s) for s in self.hidden_suffixes):
                continue

            clean_name = node_name
            for prefix in ["low_bound.", "up_bound.", "prior.", "user_prior."]:
                clean_name = clean_name.replace(prefix, "")

            if "." in clean_name and not clean_name.replace(".", "").isdigit():
                clean_name = ".".join([p for p in clean_name.split(".") if not p.isdigit()])

            # If it's a bound/prior on a SAMPLED parameter, group it
            if clean_name in sampled_labels:
                param_logps[clean_name] = param_logps.get(clean_name, 0.0) + lp
            else:
                # Derived bounds, Likelihoods, and System constraints fall through to the bottom table
                other_nodes[node_name] = lp

        return param_logps, other_nodes

    def get_curvatures(self) -> Dict[str, np.ndarray]:
        """Calculates diagonal curvature in Unity space."""
        from .run import get_diagonal_curvature  # Keep circular import local
        curvature_map = {}
        curv_vec = get_diagonal_curvature(self.model, self.transformed_inits)

        idx = 0
        for var in self.model.value_vars:
            var_size = self.transformed_inits[var.name].size
            clean_name = var.name.split('_raw')[0]
            curvature_map[clean_name] = curv_vec[idx: idx + var_size]
            idx += var_size
        return curvature_map

    def check_unused_yaml(self) -> List[str]:
        """Returns keys in YAML that didn't match any built Parameter."""
        used_keys = set()
        for p in self.all_params:
            used_keys.add(p.label)
            n = np.prod(p.shape).astype(int) if p.shape != () else 1
            for i in range(n):
                used_keys.add(p.get_display_label(i))
                # Add index fallback (star.0.radius)
                parts = p.label.split('.')
                used_keys.add(f"{parts[0]}.{i}.{parts[-1]}")

        return [k for k in self.user_params.keys() if k not in used_keys and k != 'run']