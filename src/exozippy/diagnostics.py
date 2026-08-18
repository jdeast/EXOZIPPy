from typing import Dict

import numpy as np

from .config import USER_PARAM_KEYS

# Sub-keys that resolve() does NOT read but that legitimately appear in a
# params file, so warning about them would be a false positive:
#
#   derived -- ConfigManager.finalize_user_params injects it into its OWN
#   deepcopy of the entries when it writes a solved initval back.  The
#   auditor reads system.user_params, the file as written, so this normally
#   never appears there; it is accepted for the case where a caller of
#   run_fit(config, user_params=...) round-trips an exported dict.
_INERT_SUBKEYS = ("derived",)

# What check_unused_yaml accepts.  Derived from config.py's own vocabulary,
# never restated -- the two used to drift (see USER_PARAM_KEYS' comment).
VALID_SUBKEYS = frozenset(USER_PARAM_KEYS) | frozenset(_INERT_SUBKEYS)


class ModelAuditor:
    def __init__(self, model, system, transformed_inits):
        self.model = model
        self.system = system
        self.transformed_inits = transformed_inits
        self.param_lookup = system.get_parameter_lookup()
        self.user_params = system.user_params
        self.all_params = system.get_all_parameters()

        # Internal Filter Suffixes
        self.hidden_suffixes = [
            "_raw",
            "_raw_n",
            "_raw_u",
            "_interval__",
            "_log__",
            "__",
        ]

    def get_aggregated_logps(
        self,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        model_input_names = [v.name for v in self.model.value_vars]
        filtered_point = {
            k: v
            for k, v in self.transformed_inits.items()
            if k in model_input_names
        }
        raw_logps = self.model.point_logps(filtered_point)

        param_logps = {}
        other_nodes = {}

        # ONLY group logps for parameters that are actively being sampled.
        # Read per element (a vector whose instances chose different
        # parameterizations is derived on some elements and sampled on others,
        # and it does have a _raw node), from the build's own role masks rather
        # than from `expression is None` -- a fully derived vector declared per
        # element leaves that field None.
        sampled_labels = [
            p.label
            for p in self.all_params
            if not bool(np.all(np.atleast_1d(p.is_derived)))
        ]

        for node_name, lp in raw_logps.items():
            if any(node_name.endswith(s) for s in self.hidden_suffixes):
                continue

            clean_name = node_name
            for prefix in ["low_bound.", "up_bound.", "prior.", "user_prior."]:
                clean_name = clean_name.replace(prefix, "")

            if "." in clean_name and not clean_name.replace(".", "").isdigit():
                clean_name = ".".join(
                    [p for p in clean_name.split(".") if not p.isdigit()]
                )

            # If it's a bound/prior on a SAMPLED parameter, group it
            if clean_name in sampled_labels:
                param_logps[clean_name] = param_logps.get(clean_name, 0.0) + lp
            else:
                # Derived bounds, Likelihoods, and System constraints fall through to the bottom table
                other_nodes[node_name] = lp

        return param_logps, other_nodes

    def check_unused_yaml(self):
        """Returns keys in YAML that didn't match any built Parameter."""
        used_keys = set()
        for p in self.all_params:
            used_keys.add(p.label)
            n = np.prod(p.shape).astype(int) if p.shape != () else 1
            for i in range(n):
                used_keys.add(p.get_display_label(i))
                # Add index fallback (star.0.radius)
                parts = p.label.split(".")
                used_keys.add(f"{parts[0]}.{i}.{parts[-1]}")

        unused_items = []

        # 1. Top-Level Unused Keys (e.g., misspelled component names: "inst.HIRES.gama")
        for k in self.user_params.keys():
            if k not in used_keys and k != "run":
                unused_items.append(k)

        # 2. Ignored Sub-Keys (e.g., spelled 'intival' instead of 'initval')
        # VALID_SUBKEYS is config.py's own vocabulary, not a second copy.
        for k, ov in self.user_params.items():
            if k in used_keys and isinstance(ov, dict):
                for sub_k in ov.keys():
                    if sub_k not in VALID_SUBKEYS:
                        unused_items.append(f"{k} -> '{sub_k}'")

        return unused_items
