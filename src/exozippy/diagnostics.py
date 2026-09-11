from typing import Dict

import numpy as np
import pytensor

from .config import USER_PARAM_KEYS

# check_user_starts' noise floor.  A derived quantity reassembled through a
# different float path than the seed differs in the last bits; reporting
# that on every run would train people to skip the block that matters.
USER_START_RTOL = 1e-4
USER_START_ATOL = 1e-12

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

    def _engine_consumed(self, key):
        """True when a params key names a quantity the relaxation engine's
        relations know, even though no built Parameter carries the label.

        The concrete case (pre-existing, surfaced by the mulensevent split's
        stage-3 gate): a `source.<star>.rho` seed on a point-source event.
        finite_source is off so rho is not a model parameter -- but the
        symbolic map wires rho = theta_star/theta_E unconditionally, so the
        seed back-solves into the stellar chain and moves the starting
        theta_E (measured on DC2018_128: dropping the seed moved the start's
        data logp by 1.3 nats through the flux decomposition).  Reporting it
        "unused" invites deleting a load-bearing seed, which is worse than
        the noise it saves.

        Membership is tested against `relation_symbol_paths` -- the subset
        of the symbol map built from the components' get_symbol_map
        discovery.  NOT against `master_symbol_map`: finalize_user_params'
        fallback registers every unmapped user key there as a leaf symbol,
        so full-map membership is vacuously true for any typo and would
        destroy the check (measured: two injected typos both reported
        "consumed" through the full map).
        """
        cm = getattr(self.system, "config_manager", None)
        symbol_paths = getattr(cm, "relation_symbol_paths", None) or set()
        if key in symbol_paths:
            return True
        # Fold the name spelling (source.Source.rho) to the index form the
        # symbol map stores (source.0.rho).
        parts = str(key).split(".")
        if len(parts) != 3 or parts[1].isdigit():
            return False
        comp, inst, par = parts
        entries = (getattr(self.system, "config", None) or {}).get(comp)
        if not isinstance(entries, list):
            return False
        names = [
            e.get("name") if isinstance(e, dict) else None for e in entries
        ]
        if inst not in names:
            return False
        return f"{comp}.{names.index(inst)}.{par}" in symbol_paths

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
            if (
                k not in used_keys
                and k != "run"
                and not self._engine_consumed(k)
            ):
                unused_items.append(k)

        # 2. Ignored Sub-Keys (e.g., spelled 'intival' instead of 'initval')
        # VALID_SUBKEYS is config.py's own vocabulary, not a second copy.
        for k, ov in self.user_params.items():
            if k in used_keys and isinstance(ov, dict):
                for sub_k in ov.keys():
                    if sub_k not in VALID_SUBKEYS:
                        unused_items.append(f"{k} -> '{sub_k}'")

        return unused_items

    @staticmethod
    def _wrap_if_angle(diff, unit):
        """Degrees are periodic: 352.57 and -7.43 are the SAME start.

        Without this the report cried wolf on four shipped examples --
        bigomega and alpha pairs differing by exactly 360 -- and a warning
        block with false positives in it is one nobody reads.
        """
        if "deg" in str(unit or ""):
            return (diff + 180.0) % 360.0 - 180.0
        return diff

    def check_user_starts(self):
        """Every user-set initval, against what the built model PRODUCES.

        THE CONTRACT: when a user sets a value, the model produces that
        value or says why it cannot.  Nothing enforced that, so the failure
        was silent -- ob09020 pins t_E = 76.9 and starts at 74.48, and the
        only way to find out was to compile the graph by hand.

        Two reasons, and the ledger separates them without guessing:

          ledger == user, built != user -> the DERIVATION cannot preserve
              it.  The pin was recorded at rank 100 and never touched; the
              graph still computes something else because the seed path
              approximates the chain.  (Worked example: mulensevent.t_E is
              seeded through mu_rel_HELIO while the graph evaluates
              mu_rel_GEO -- symbolic_physics.py's SEEDING APPROXIMATION.)

          ledger != user -> OVERSPECIFIED; something had to give, and
              _last_solved_by names the equation that overwrote it.

        Read off the COMPILED GRAPH.  Not ``p.value`` (a draw from the
        prior) and not ``p.initval`` (the ledger, which is half of what is
        under test) -- both look right while the model starts elsewhere.

        Returns a list of dicts sorted worst-first, empty when the model
        delivers everything asked of it.
        """
        cm = getattr(self.system, "config_manager", None)
        ledger = dict(getattr(cm, "_last_resolved", None) or {})
        solved_by = dict(getattr(cm, "_last_solved_by", None) or {})

        # Invert get_display_label -- the SAME mapping check_unused_yaml
        # builds, so the two checks agree on which key names which element
        # and all three user spellings (star.L1.mass / star.0.mass /
        # star.mass) land on one parameter.
        index = {}
        for p in self.all_params:
            n = int(np.prod(p.shape)) if p.shape != () else 1
            for i in range(n):
                index.setdefault(p.get_display_label(i), (p, i))
                parts = p.label.split(".")
                index.setdefault(f"{parts[0]}.{i}.{parts[-1]}", (p, i))
            index.setdefault(p.label, (p, 0))

        targets = []
        for key, entry in (self.user_params or {}).items():
            if not isinstance(entry, dict) or "initval" not in entry:
                continue
            requested = entry["initval"]
            if not isinstance(
                requested, (int, float, np.floating, np.integer)
            ) or isinstance(requested, bool):
                continue
            hit = index.get(key)
            if hit is None:
                continue  # check_unused_yaml owns unmatched keys
            p, i = hit
            # An inactive element is not built, so a mismatch there is a
            # statement about a parameterization the user did not choose.
            # The accessor, not the raw mask: the masks start as SCALAR and
            # only become vectors once build_pymc writes them.
            if not p.element_is_active(i):
                continue
            targets.append((key, p, i, float(requested)))

        if not targets:
            return []

        # ONE compiled function over the distinct value nodes.  Compiling
        # per parameter would add a PyTensor compile to startup for every
        # seed in the file.
        nodes, node_of = [], {}
        for _key, p, _i, _req in targets:
            if id(p) not in node_of:
                node_of[id(p)] = len(nodes)
                nodes.append(p.value)

        try:
            fn = pytensor.function(
                self.model.value_vars,
                self.model.replace_rvs_by_values(nodes),
                on_unused_input="ignore",
            )
            point = [
                self.transformed_inits[v.name] for v in self.model.value_vars
            ]
            produced = fn(*point)
        except Exception:
            # A diagnostic must never be the reason a fit does not start.
            return []

        findings = []
        for key, p, i, requested in targets:
            arr = np.atleast_1d(
                np.asarray(produced[node_of[id(p)]], dtype=float)
            )
            j = min(i, arr.size - 1)
            unit = getattr(p, "unit", "")
            got = float(p.from_internal(arr[j], index=j))

            diff = self._wrap_if_angle(got - requested, unit)
            denom = abs(requested) if requested else 1.0
            rel = diff / denom
            if abs(diff) <= USER_START_ATOL or abs(rel) <= USER_START_RTOL:
                continue

            held = ledger.get(key)
            who = solved_by.get(key)
            if held is None:
                parts = key.split(".")
                if len(parts) == 3:
                    idx_key = f"{parts[0]}.{i}.{parts[-1]}"
                    held = ledger.get(idx_key)
                    who = who or solved_by.get(idx_key)

            # THE LEDGER IS IN INTERNAL UNITS (config.py:890,
            # "internal_path -> internal value") and `requested` came from
            # the file, in USER units.  Comparing them raw reports every
            # unit-converted parameter as overwritten -- star.ra would read
            # 4.6095 against 264.105.  Convert through the Parameter, which
            # is the only thing allowed to know the factor, and let the
            # method name carry the direction.
            held_user = (
                None if held is None else float(p.from_internal(held, index=j))
            )
            kept = held_user is not None and abs(
                self._wrap_if_angle(held_user - requested, unit)
            ) <= max(USER_START_ATOL, denom * USER_START_RTOL)

            if held_user is not None and not kept:
                findings.append(
                    {
                        "key": key,
                        "requested": requested,
                        "produced": got,
                        "rel": rel,
                        "reason": "overspecified",
                        "detail": (
                            f"the system is overspecified and this value "
                            f"had to give; it resolved to {held_user:.6g} "
                            f"via {who or 'the relaxation engine'}"
                        ),
                    }
                )
            else:
                findings.append(
                    {
                        "key": key,
                        "requested": requested,
                        "produced": got,
                        "rel": rel,
                        "reason": "approximate",
                        "detail": (
                            "your value was kept, but the derivation "
                            "reproduces it only approximately"
                        ),
                    }
                )

        findings.sort(key=lambda f: -abs(f["rel"]))
        return findings
