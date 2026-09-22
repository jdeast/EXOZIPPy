import logging
from typing import Dict

import numpy as np
import pytensor
import pytensor.graph.basic
import pytensor.graph.traversal

from .components.parameter import derived_constraint_message
from .config import RESERVED_PARAM_KEYS, USER_PARAM_KEYS

logger = logging.getLogger(__name__)

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

# The pile-at-cap rule (review 8.6.3): an element whose upper bound is a
# flagged modelling cap (``Parameter.cap_alarm``) is reported when MORE than
# CAP_PILE_FRAC of its draws lie within the top CAP_TOP_FRAC of its
# [lower, upper] range.  Half the posterior in the top twentieth of the
# support is not a bounded parameter being sampled, it is a parameter that
# would leave if it could.
CAP_PILE_FRAC = 0.5
CAP_TOP_FRAC = 0.05


def cap_alarm_findings(system, pile_frac=CAP_PILE_FRAC, top_frac=CAP_TOP_FRAC):
    """Every sampled element whose posterior piles against a flagged cap.

    Component-agnostic and manifest-driven: the check reads nothing but
    ``Parameter.cap_alarm`` (set by the owning component through a manifest
    option) and the distributed posterior, so it never names a component
    or a parameter.  Call after ``System.distribute_posterior``.  Returns a
    list of dicts -- ``display`` (the user path, e.g.
    ``mulensinstrument.Roman_W149.out_scale``), ``label``, ``index``,
    ``cap`` and ``lower`` (USER units), ``unit`` and ``frac`` -- sorted by
    label then element, for ``log_cap_alarms`` and the modeling prose.
    """
    findings = []
    for p in system.get_all_parameters():
        n = int(np.prod(p.shape)) if getattr(p, "shape", ()) else 1
        n = max(
            n,
            int(np.atleast_1d(p.cap_alarm).size)
            if p.cap_alarm is not None
            else 1,
        )
        for i in range(n):
            if not p.element_cap_alarm(i):
                continue
            # A pinned or derived element cannot pile anywhere; only ask
            # once the build has written the roles (pre-build, assume free
            # so a synthetic posterior can be audited).
            if p._built_roles() and not p.element_is_sampled(i):
                continue
            frac = p.cap_saturation(i, top_frac=top_frac)
            if frac is None or frac <= pile_frac:
                continue
            findings.append(
                {
                    "display": p.get_display_label(i),
                    "label": p.label,
                    "index": i,
                    "cap": float(
                        p.from_internal(np.atleast_1d(p.upper)[i], index=i)
                    ),
                    "lower": float(
                        p.from_internal(np.atleast_1d(p.lower)[i], index=i)
                    ),
                    "unit": str(p.unit) if p.unit is not None else "",
                    "frac": frac,
                    "top_frac": top_frac,
                }
            )
    return findings


def log_cap_alarms(findings, log, top_frac=CAP_TOP_FRAC):
    """One WARNING per finding, naming the parameter, cap and remedy."""
    for f in findings:
        unit = f" {f['unit']}" if f.get("unit") else ""
        log.warning(
            "PILED AT CAP: %s has %.0f%% of its posterior draws within the "
            "top %.0f%% of its allowed range [%.4g, %.4g%s] -- the noise "
            "model wants more freedom than the cap allows.  Inspect the "
            "residuals of that data set; loosen `%s: {upper: ...}` in the "
            "params file only if the excess is NOT a real signal the model "
            "is missing (review 8.6.3: on DC2018 event 128 the uncapped "
            "mixture absorbed the caustic crossing).",
            f["display"],
            100.0 * f["frac"],
            100.0 * f.get("top_frac", top_frac),
            f["lower"],
            f["cap"],
            unit,
            f["display"],
        )


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
        #
        # RESERVED_PARAM_KEYS is exempt for the same reason "run" is, and the
        # exemption has to be HERE rather than upstream: this auditor reads
        # `system.user_params` -- the file exactly as written -- while
        # `ConfigManager` reads its own copy with the reserved keys already
        # split off.  Without this, `overdisperse:` (which mkparam writes into
        # every restart file) would be reported as a key that matched no
        # parameter, on every restart, forever.
        for k in self.user_params.keys():
            if (
                k not in used_keys
                and k != "run"
                and k not in RESERVED_PARAM_KEYS
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

    def values_at_start(self, params):
        """What the model PRODUCES for each Parameter at the start point.

        ``{id(parameter): 1-D array in INTERNAL units}``, empty when the
        graph cannot be evaluated -- a diagnostic must never be the reason
        a fit does not start.

        This is the one place allowed to answer "what value does this node
        actually have at the start", and the reason it is a method rather
        than two copies of ten lines is that the obvious spellings are both
        wrong in the silent direction: ``p.value.eval()`` DRAWS FROM THE
        PRIOR (it evaluates the graph with its RVs still in place) and
        ``p.initval`` is the ledger, which is exactly the thing a caller
        here is usually checking the graph against.  Read at the start
        point means: replace the RVs by their value variables and feed
        ``transformed_inits``.

        ONE compiled function over the distinct nodes, because compiling
        per parameter would add a PyTensor compile to startup for every
        parameter asked about.  ``id()`` keys rather than labels: two
        parameters can share a label across models, and the caller already
        holds the objects.
        """
        distinct, nodes = [], []
        for p in params:
            node = getattr(p, "value", None)
            if node is None or any(q is p for q in distinct):
                continue
            distinct.append(p)
            nodes.append(node)
        if not nodes:
            return {}
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
            return {}
        return {
            id(p): np.atleast_1d(np.asarray(v, dtype=float))
            for p, v in zip(distinct, produced)
        }

    def _derived_sources(self, p):
        """The SAMPLED parameters a derived value is computed FROM.

        Read off the GRAPH, like every other answer in
        ``check_user_starts``: the tensor ``build_pymc`` assembled is what
        the value actually consumes, so the names it reaches are the ones
        that can move it.  A manifest ``deps`` list would be the other
        candidate and is the wrong one -- it is what a component DECLARED,
        which is a statement about the build order rather than about this
        vector's value.

        Scope is the WHOLE VECTOR, deliberately.  A mixed vector (some
        instances derived, some sampled) has one built tensor, so a
        per-element answer would mean calling each ``ElementExpression``'s
        closure again outside the model context -- a rebuild, for a
        diagnostic, on a graph that may not be reconstructible there.  The
        superset is honest advice ("fix the sampled parameter(s) it is
        derived from") and costs one graph walk.

        Returns [] on anything unexpected: with no names the message falls
        back to naming the CLASS, which is what the ``sigma: 0`` warning
        this shares its wording with has always done.
        """
        try:
            node = getattr(p, "value", None)
            if not isinstance(node, pytensor.graph.basic.Variable):
                return []
            sampled = set()
            for q in self.all_params:
                if q is p:
                    continue
                n = int(np.prod(q.shape)) if q.shape != () else 1
                if any(q.element_is_sampled(i) for i in range(n)):
                    sampled.add(q.label)
            found = set()
            for anc in pytensor.graph.traversal.ancestors([node]):
                name = getattr(anc, "name", None)
                if not name:
                    continue
                # A sampled parameter reaches the graph as its own
                # Deterministic (named for the label) or, one step lower, as
                # the `<label>_raw` coordinate; the same suffix list the
                # logp grouping strips.
                for suffix in self.hidden_suffixes:
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        break
                if name in sampled:
                    found.add(name)
            return sorted(found)
        except Exception:
            # A diagnostic must never be the reason a fit does not start.
            return []

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

        Three reasons, and the ROLE plus the ledger separate them without
        guessing:

          the element is DERIVED -> its value is an expression, so no
              channel `initval` has can hold it; the remedy is to set the
              SAMPLED parameter(s) it is derived from, which is
              `parameter.py`'s own sentence for the `sigma: 0` twin of
              this mistake (review 2.3.17).

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

        # THE ENGINE ALREADY KNOWS about one whole class of these, and it
        # names the relation: when every symbol of a violated relation is
        # user-set, nothing can be adjusted and _relax_equation records an
        # "over-constrained" diagnostic.  Without reading it, such a miss
        # reports as "the derivation cannot preserve your value", which is
        # true of the mechanism and useless as advice -- the fix is to drop
        # one of two pins, and the user has to be told WHICH.  Measured on
        # examples/galactic_model, which pinned mass = 0.5 and
        # logmass = -0.3 (log10(0.5) = -0.30103) and started at 10**-0.3.
        #
        # NOT every double pin lands here: the clause needs ALL symbols at
        # user rank, so a relation carrying one derived symbol is invisible
        # to it (KMT-2019-BLG-1806 pins both rho and the source radius, and
        # rho's relation also holds theta_E at rank 60).  This reads what
        # the engine recorded; it does not re-derive the class.
        contradiction_of = {}
        for entry in getattr(cm, "diagnostics", None) or []:
            if entry.get("severity") != "error":
                continue
            paths = list(entry.get("param_paths") or [])
            for p in paths:
                contradiction_of.setdefault(
                    str(p), (entry.get("message", ""), paths)
                )

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

        produced = self.values_at_start([p for _key, p, _i, _req in targets])
        if not produced:
            return []

        findings = []
        for key, p, i, requested in targets:
            arr = produced.get(id(p))
            if arr is None:
                continue
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
            parts = key.split(".")
            idx_key = f"{parts[0]}.{i}.{parts[-1]}" if len(parts) == 3 else key
            if held is None and idx_key != key:
                held = ledger.get(idx_key)
                who = who or solved_by.get(idx_key)

            # An engine-recorded contradiction outranks every other reading:
            # it already knows the relation and the other pin.
            clash = contradiction_of.get(key) or contradiction_of.get(idx_key)
            if clash is not None:
                message, paths = clash
                others = [q for q in paths if q not in (key, idx_key)]
                findings.append(
                    {
                        "key": key,
                        "requested": requested,
                        "produced": got,
                        "rel": rel,
                        "reason": "overspecified",
                        "detail": (
                            "you also set "
                            + (
                                ", ".join(others)
                                if others
                                else "a linked parameter"
                            )
                            + ", which the same relation fixes; both cannot "
                            "hold, so drop one. " + message
                        ),
                    }
                )
                continue

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

            # A DERIVED element outranks both readings below, and the
            # codebase already knows what to say about it: its value IS an
            # expression, so `initval` has no channel that can hold it (the
            # engine can only back-solve the request into whatever is
            # sampled, and here that did not deliver it).  "your value was
            # kept, but the derivation reproduces it only approximately"
            # understated exactly this case by three orders of magnitude --
            # a derived planet.mass came back at 1/1047 of the request
            # (review 2.3.17) -- and pointed the user nowhere.  The
            # sentence is `parameter.py`'s own, shared rather than
            # paraphrased.
            #
            # An engine-recorded contradiction still wins (above): there the
            # user wrote a second pin, and naming THAT is better advice than
            # naming the class of the first.
            if p.element_is_derived(i):
                detail = derived_constraint_message(
                    "initval", self._derived_sources(p)
                ) + (
                    " A value written here reaches the model only by "
                    "back-solving into those, and here that did not "
                    "reproduce it"
                )
                if held_user is not None and not kept:
                    detail += (
                        f" (the relaxation engine resolved it to "
                        f"{held_user:.6g} via "
                        f"{who or 'the relaxation engine'})"
                    )
                findings.append(
                    {
                        "key": key,
                        "requested": requested,
                        "produced": got,
                        "rel": rel,
                        "reason": "derived",
                        "detail": detail,
                    }
                )
                continue

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


# ---------------------------------------------------------------------------
# Posterior against a wall (review 8.2.2)
# ---------------------------------------------------------------------------

# How close to a bound, as a fraction of the bound-to-bound distance, counts
# as "against the wall".  Measured in LOG space when both bounds are positive
# -- err_scale on [0.01, 100] is four decades, and 2% of that is a factor 1.2
# from either end -- and linearly otherwise.  A linear rule on a positive
# scale would call err_scale = 1 "near the lower bound 0.01" on a span of
# 100, which is the wrong reading of every scale-like parameter.
NEAR_BOUND_MARGIN = 0.02

# The MEDIAN test above misses a posterior that is sculpted by a wall without
# being centred on it: DC2018 event 152's source teffsed had its whole lower
# tail on the 2600 K edge of the bolometric-correction grid with a median at
# 3163 K, so nothing warned, and the reported temperature was a distribution
# the grid had cut in half.  An element is therefore ALSO reported when this
# fraction of its draws lies within NEAR_BOUND_MARGIN of either bound.  It is
# looser than the cap rule's 50% (CAP_PILE_FRAC) on purpose: that rule asks
# "would this parameter leave if it could", which wants a majority, while
# this one asks "is the answer shaped by the edge", which a tenth of the mass
# already does.
EDGE_MASS_FRAC = 0.10


def near_bound_position(value, lower, upper):
    """Where ``value`` sits in [lower, upper]: 0.0 at lower, 1.0 at upper.

    Log space when both bounds are positive (a scale), linear otherwise.
    NaN when the interval is degenerate or the value is not finite, so a
    caller comparing against a margin gets False rather than an exception.
    """
    value, lower, upper = float(value), float(lower), float(upper)
    if not (np.isfinite(value) and np.isfinite(lower) and np.isfinite(upper)):
        return np.nan
    if upper <= lower:
        return np.nan
    if lower > 0.0 and upper > 0.0 and value > 0.0:
        return (np.log(value) - np.log(lower)) / (
            np.log(upper) - np.log(lower)
        )
    return (value - lower) / (upper - lower)


def _positions(values, lower, upper):
    """``near_bound_position`` over an array, same log/linear rule."""
    v = np.asarray(values, dtype=float)
    lower, upper = float(lower), float(upper)
    if not (np.isfinite(lower) and np.isfinite(upper)) or upper <= lower:
        return np.full(v.shape, np.nan)
    if lower > 0.0 and upper > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (np.log(v) - np.log(lower)) / (np.log(upper) - np.log(lower))
        return np.where(v > 0.0, out, np.nan)
    return (v - lower) / (upper - lower)


def grid_bounded_paths(system):
    """{parameter path: info} for bounds that ARE a model grid's extent.

    Component-agnostic by duck typing, the same way ``cap_alarm_findings``
    stays component-agnostic through a Parameter field: any component that
    bounds someone else's parameter by the reach of an interpolation grid
    may declare it with a ``grid_bound_paths()`` method, and nothing here
    knows which component that is or what the grid interpolates.

    It matters because the two kinds of wall mean opposite things.  A
    physical bound (an error scale of 100x, a negative flux) is a statement
    about what is possible, and a posterior against it is usually the fit
    telling you something.  A grid extent is a statement about what the
    MODEL can evaluate: past it the interpolator has no data, and a
    posterior against it is reporting that the grid ran out.
    """
    out = {}
    for comp in getattr(system, "active_components", None) or []:
        fn = getattr(comp, "grid_bound_paths", None)
        if not callable(fn):
            continue
        try:
            out.update(fn() or {})
        except Exception:  # noqa: BLE001 -- a broken hook must not kill the check
            logger.debug(
                "grid_bound_paths() failed on %s",
                type(comp).__name__,
                exc_info=True,
            )
    return out


def warn_posterior_near_bounds(
    system,
    margin=NEAR_BOUND_MARGIN,
    log=None,
    mass_frac=EDGE_MASS_FRAC,
):
    """Warn, per sampled element, when the posterior median sits against a
    hard bound -- and say what the component thinks that means.

    Runs after ``System.distribute_posterior`` so every Parameter carries its
    posterior (user units, sample axis last).  Only elements on the logit
    transform have two finite bounds to be near; the frozen transform
    (``_raw_transform``) is the one owner of which elements those are and of
    their bounds in internal units, so nothing here re-derives a bound.

    The generic half of the message is the same sentence the post-polish
    wall warning uses (a value on a wall usually means the BOUND is the
    thing to revisit); the component-specific half is
    ``Parameter.near_bound_remedy``, declared in defaults.yaml, because for
    a nuisance scale the generic advice is wrong -- err_scale at its upper
    bound means the data are being rescaled instead of fitted, and the
    remedy is the errors or the starting model, not a wider bound (review
    8.2.2, measured on DC2018-226 where both bands sat at 300-460x).

    Two triggers, because a wall shapes an answer in two different ways:
    the posterior MEDIAN against the bound (the original rule, review
    8.2.2), or more than ``mass_frac`` of the DRAWS within ``margin`` of
    either bound (``EDGE_MASS_FRAC``).  A bound that is a model grid's
    extent (``grid_bounded_paths``) says so in the message, because there
    the values on the wall are extrapolation rather than measurement.

    Returns the list of hits (dicts) so a caller or a test can read them.
    """
    log = log or logger
    hits = []
    grid_paths = grid_bounded_paths(system)
    for par in system.get_all_parameters():
        tf = getattr(par, "_raw_transform", None)
        post = getattr(par, "posterior", None)
        if not tf or post is None:
            continue
        arr = np.asarray(getattr(post, "values", post), dtype=float)
        if arr.ndim == 0 or arr.size == 0:
            continue
        with np.errstate(all="ignore"):
            med = np.atleast_1d(np.nanmedian(arr, axis=-1))
        for i in tf["sampled_idx"]:
            if not tf["use_logit"][i] or i >= med.size:
                continue
            lower, upper = tf["lowers"][i], tf["uppers"][i]
            try:
                v_int = float(par.to_internal(med[i], index=i))
            except Exception:  # noqa: BLE001 -- a conversion that fails is not a wall
                continue
            pos = near_bound_position(v_int, lower, upper)
            # Every draw, not just the median: the mass criterion.
            try:
                # `med` is taken over the LAST axis, so the element axis
                # is the first one -- and a scalar parameter's posterior is
                # 1-D, where arr[i] would be one draw rather than the
                # element's draws.
                draws_u = arr if arr.ndim == 1 else arr[i]
                draws_int = np.asarray(
                    par.to_internal(np.asarray(draws_u, dtype=float), index=i),
                    dtype=float,
                )
            except Exception:  # noqa: BLE001 -- as above, a failed conversion is not a wall
                draws_int = np.array([])
            dpos = _positions(draws_int, lower, upper)
            n_ok = int(np.isfinite(dpos).sum())
            frac_low = float(np.nansum(dpos <= margin) / n_ok) if n_ok else 0.0
            frac_high = (
                float(np.nansum(dpos >= 1.0 - margin) / n_ok) if n_ok else 0.0
            )
            med_hit = np.isfinite(pos) and not (margin < pos < 1.0 - margin)
            mass_hit = max(frac_low, frac_high) >= mass_frac
            if not (med_hit or mass_hit):
                continue
            if med_hit:
                side = "lower" if pos <= margin else "upper"
            else:
                side = "lower" if frac_low >= frac_high else "upper"
            lo_u = float(np.atleast_1d(par.from_internal(lower, index=i))[0])
            hi_u = float(np.atleast_1d(par.from_internal(upper, index=i))[0])
            # `unit` is an astropy Unit (or a per-element list of them) after
            # __post_init__; rendered for the message only, never compared.
            u = par.unit
            if isinstance(u, (list, tuple)):
                u = u[i] if i < len(u) else None
            unit = "" if u is None else str(u).strip()
            grid = grid_paths.get(par.label)
            frac = frac_low if side == "lower" else frac_high
            hit = {
                "label": par.get_display_label(i),
                "median": float(med[i]),
                "side": side,
                "position": float(pos),
                "lower": lo_u,
                "upper": hi_u,
                "frac": frac,
                "trigger": "median" if med_hit else "mass",
                "grid": (grid or {}).get("source"),
            }
            hits.append(hit)
            if med_hit:
                where = (
                    f"posterior median {med[i]:.4g}"
                    f"{(' ' + unit) if unit else ''} sits against its {side} "
                    f"bound [{lo_u:.4g}, {hi_u:.4g}] ({pos:.1%} of the way "
                    f"across, within the {margin:.0%} margin)"
                )
            else:
                where = (
                    f"{frac:.0%} of its posterior draws sit within the "
                    f"{margin:.0%} margin of its {side} bound "
                    f"[{lo_u:.4g}, {hi_u:.4g}], though the median "
                    f"({med[i]:.4g}{(' ' + unit) if unit else ''}) does not"
                )
            if hit["grid"]:
                why = (
                    f" That bound IS the {hit['grid']}'s extent, so this is "
                    f"the MODEL running out, not the data preferring the "
                    f"edge: values there are the edge cell carried outward, "
                    f"and the reported interval is cut off rather than "
                    f"measured."
                )
            else:
                why = (
                    " A posterior piled on a wall usually means the bound, "
                    "not the fit, is the thing to revisit."
                )
            log.warning(
                f"Parameter '{hit['label']}': {where}.{why}"
                + par.remedy_suffix()
            )
    return hits
