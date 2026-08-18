# import ipdb
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

import arviz as az
import pymc as pm
import pytensor
import pytensor.tensor as pt

try:
    # Moved from pytensor.graph.basic in newer pytensor releases (the old
    # location warns and is scheduled for removal).  Same fallback as
    # Component._model_trace_param_deps.
    from pytensor.graph.traversal import ancestors
except ImportError:  # pragma: no cover - older pytensor
    from pytensor.graph.basic import ancestors

from exozippy.components.component import Component
from exozippy.components.factory import discover_components, import_failures
from exozippy.components.parameter import Parameter, SeedBoundViolation, to_vec
from exozippy.config import ConfigManager
from exozippy.evaluator import structural_hash, structural_payload
from exozippy.graph import determine_pymc_build_order
from exozippy.manifest import interpret_manifest_entry, normalize_selector
from exozippy.outputs.prose import ProseCollector
from exozippy.yamlio import load_yaml

"""
The System Class builds an entire system to model from its components.
Critically, it contains no component-specific logic, so it
can generally construct any model containing arbitrary components.
"""

# Top-level config keys that are NOT component blocks. Every other top-level
# key is looked up in the component registry and, failing that, warned about
# as "will be ignored" -- so a key that some part of the codebase honors but
# that is missing here produces a warning which is actively false, training
# users to disbelieve the warning system. Each entry names its consumer:
#
#   run            -- documentation/bookkeeping block; deliberately inert.
#                     evaluator._NON_STRUCTURAL_CONFIG_KEYS excludes it from
#                     the structural hash, which is its only mention in code.
#   name           -- System.name (below), read back by e.g.
#                     astrometryinstrument's sky-plot title.
#   parameter_file -- System.__init__, mkparam.write_param_file, gui.document.
#   prefix         -- run.py, cli_modes.py, mkparam.py, gui/status.py,
#                     gui/runner.py, mulensinstrument's mmexofast cache path.
#   logger_level   -- run.py, cli.py, cli_modes.py.
#   sampler        -- run.py (see run.KNOWN_SAMPLER_KEYS for its own block).
#   modes          -- run.py: {ledger, max_invalid_frac, force, weights}.
#   mkparam        -- mkparam.write_param_file: {n_seeds, force}.  `force`
#                     is deliberately NOT `modes: {force: true}`: that one
#                     authorizes forensic REPORTING off a known-bad
#                     trace, this one authorizes seeding the NEXT fit
#                     from one.  See mkparam._refuse_invalid_seed_draws.
#   gui            -- gui.status.gui_enabled: {snapshot}.
#   modeling       -- run.py: {compile} for the generated paper-draft
#                     scaffold (<prefix>_paper.tex).  Output-only, so
#                     evaluator._NON_STRUCTURAL_CONFIG_KEYS excludes it
#                     from the structural hash: adding the block or
#                     flipping `compile` must not stale a finished trace.
#
# tests/test_known_keys.py cross-checks this set against the top-level-config
# accesses in the source, in both directions, so it cannot silently drift.
RESERVED_CONFIG_KEYS = frozenset(
    {
        "run",
        "parameter_file",
        "prefix",
        "sampler",
        "name",
        "logger_level",
        "modes",
        "mkparam",
        "gui",
        "modeling",
    }
)


class System(Component):
    def __init__(self, config, user_params=None):
        self.config = config
        self.name = self.config.get("name", "system")

        if user_params is not None:
            self.user_params = user_params
        else:
            user_params_file = self.config.get("parameter_file", None)
            if user_params_file is None:
                # An OMITTED key is legal, and says "I have no overrides".
                # A params file may already be EMPTY -- defaults.yaml, the
                # component hints and the relaxation engine between them can
                # start a fit on their own, and since the global search
                # (8.3.1) a blind fit seeds its own period and epoch from
                # BLS/Lomb-Scargle -- so requiring the KEY while the FILE it
                # names may be empty is a distinction with no content, and
                # exactly the friction a blind fit should not have to pay.
                # An INFO line, not a warning: this is a supported way to
                # write a config, not a mistake to apologize for.  A key that
                # IS present still has to name a file that exists, which is
                # the real typo (below) and stays fatal.
                logger.info(
                    "No 'parameter_file' in the config; proceeding with no "
                    "user parameter overrides. Start values come from each "
                    "component's defaults.yaml, its data-derived hints and "
                    "the relaxation engine. Add "
                    "'parameter_file: myfit.params.yaml' to override any of "
                    "them."
                )
                self.user_params = {}
            else:
                if not os.path.exists(user_params_file):
                    raise FileNotFoundError(
                        f"parameter_file '{user_params_file}' not found. "
                        f"This path is resolved relative to the directory from which you "
                        f"run exozippy (currently: {os.getcwd()}). "
                        f"Check that the file exists and the path in your config YAML is correct."
                    )
                # `or {}`: yaml.safe_load returns None for an EMPTY file, and
                # an empty params file has to mean exactly what an omitted
                # key does -- "no overrides" -- rather than a third state
                # every consumer has to spell `user_params or {}` for.
                # mkparam's own loader has always normalized it this way.
                self.user_params = load_yaml(str(user_params_file)) or {}

        self.config_manager = ConfigManager(
            self.user_params, system_config=self.config
        )
        # The modeling-prose collector (outputs/prose.py): components add
        # sentences at the code sites that implement each feature (stages
        # 1-6), run.py adds the sampling/results sentences, and
        # outputs/modeling.py regenerates <prefix>_paper.tex from it at
        # each checkpoint.  add() is idempotent, so a second build_model()
        # on one System (the GUI) cannot accumulate copies.
        self.prose = ProseCollector()
        # Sliced per-element expressions awaiting their start-point check
        # (see register_element_slice_check / verify_element_slices).  Rebuilt
        # per build_model, so a second build on one System cannot accumulate.
        self._element_slice_checks = []
        # Many-to-one parameterizations' alternative branches, declared during
        # stage 7 and marginalized over at the end of it (see
        # register_branch_alternative / _add_branch_mixtures).  Rebuilt per
        # build_model for the same reason.
        self._branch_alternatives = []
        # Record the params file ONLY when one was really read: an in-memory
        # user_params dict must not be blamed on a parameter_file the config
        # happens to name but System never opened -- and neither must a config
        # that names none at all, whose (empty) overrides come from nowhere on
        # disk and so have no file for an error message to point at.
        if user_params is None and user_params_file is not None:
            self.config_manager.param_file = str(user_params_file)
        self.registry = discover_components()
        self.active_components = {}

        # 1. AGNOSTIC INSTANTIATION
        reserved_keys = RESERVED_CONFIG_KEYS
        for key in self.config.keys():
            if key in self.registry:
                CompClass = self.registry[key]
                inst = CompClass(self.config[key], self.config_manager)
                self.active_components[key] = inst
                setattr(self, key, inst)
            elif key not in reserved_keys:
                # Distinguish "you typo'd a key" from "the component you
                # asked for failed to import".  The old code said the
                # former for both and then fitted a model missing an
                # entire component and its data, quietly.
                failed = import_failures().get(key)
                if failed is not None:
                    module_path, exc = failed
                    raise ImportError(
                        f"YAML key '{key}' names the component module "
                        f"{module_path}, which failed to import: "
                        f"{type(exc).__name__}: {exc}.  A missing optional "
                        f"dependency is the usual cause.  Fix the import or "
                        f"remove the '{key}' block -- continuing would fit a "
                        f"model without it."
                    ) from exc
                logger.warning(
                    f"YAML key '{key}' does not match any registered component and will be ignored."
                )

        logger.info("Modeling the following components:")
        for key, comp in self.active_components.items():
            logger.info(f"  {key} ({comp.n_elements})")

        # Structural fingerprint of the inputs, snapshotted HERE: after the
        # components have normalized their own config blocks (Mann/Torres
        # derive `name:` from their `star:` key in __init__), and before
        # prepare() runs.  Both halves of that placement were measured, not
        # assumed -- see the note on the recomputation in
        # mkparam.write_param_file.  Taking it any earlier fingerprints a
        # config spelling that exists only for the first few lines of
        # __init__, so a fingerprint recomputed later would never match;
        # taking it later would fold in whatever stages 1-7 might one day
        # write.  The params half is safe at either point: ConfigManager
        # deepcopies before it standardizes keys, strips links and injects
        # solved initvals, so self.user_params stays exactly the file that
        # was read.
        self._structural_payload = structural_payload(
            self.config, self.user_params
        )
        self._structural_hash = structural_hash(self.config, self.user_params)

    def structural_fingerprint(self):
        """``(hash, payload)`` of the config + params this System was built from.

        The hash is ``evaluator.structural_hash``; the payload is the dict it
        was taken over, kept so a mismatch can name what changed.  Both are
        snapshotted at the END of ``__init__`` -- after the components have
        normalized their own config blocks, before ``prepare()`` -- so that a
        fingerprint recomputed from the same inputs later in the run
        (mkparam.write_param_file) reproduces it exactly.
        """
        return self._structural_hash, self._structural_payload

    def prepare(self):
        # ==========================================================
        # PRE-FLIGHT SEQUENCE
        # ==========================================================
        # Stages 1-2: DATA & LOGICAL MAPS
        for comp in self.active_components.values():
            if hasattr(comp, "load_data"):
                comp.load_data(self)
            if hasattr(comp, "build_maps"):
                comp.build_maps()

        # Stage 3: REGISTRATION (The Blueprint)
        for comp in self.active_components.values():
            if hasattr(comp, "register_parameters"):
                comp.register_parameters(self)

        # After stage 3: the REPORTED-element invariant, checked before anything is
        # built (see _validate_reported_not_consumed).
        self._validate_reported_not_consumed()

        # Stage 4: RECONCILIATION (The Solver)
        self.config_manager.finalize_user_params()

    def _validate_reported_not_consumed(self):
        """Refuse a manifest where something CONSUMES a reported element.

        Manifest role 3 rests on one property: a reported element is consumed by
        nothing.  That is what lets its expression be applied in a second phase
        (Parameter.finalize_deferred) after the parameter it reads has been
        built, and what makes the per-parameter cycle such a pair would
        otherwise form dissolve.  Break it, and the consumer silently reads the
        PRE-PATCH placeholder: a plausible number that is not the quantity it
        claims to be, with no error anywhere.

        That is not hypothetical -- it is how ``orbit.tp`` behaved the first
        time the V_c/V_e parameterization was wired, because `calc_tp` consumes
        `secosw`/`sesinw`, which a V_c/V_e orbit reports rather than samples.
        The fix was to give `tp` its own (e, omega) expression on those orbits;
        the check is here so the next such pairing is a startup error rather
        than a wrong posterior.

        Checked per ELEMENT, because that is the real condition: in a system
        with one orbit of each kind, `ecc`'s sqrt(e)cos/sin expression legally
        reads `secosw` for the hk orbit's elements, which are exactly the
        elements the other orbit does not report.  Only an overlap is an error.
        """
        reported = {}
        selections = {}
        for comp in self.active_components.values():
            manifest = getattr(comp, "manifest", {}) or {}
            for name, raw in manifest.items():
                entry = interpret_manifest_entry(raw)
                n_elements = comp.n_elements
                if entry.shape:
                    shape = entry.shape
                    n_elements = int(
                        np.prod(shape)
                        if isinstance(shape, tuple)
                        else int(shape)
                    )
                out_sel = entry.output_expr_selectors or {}
                if entry.output_expr_key is not None:
                    out_sel = {entry.output_expr_key: None}
                if out_sel:
                    mask = np.zeros(n_elements, dtype=bool)
                    for sel in out_sel.values():
                        mask |= normalize_selector(sel, n_elements)
                    reported[(comp.prefix, name)] = mask
                selections[(comp.prefix, name)] = (comp, entry, n_elements)

        if not reported:
            return

        for (prefix, name), (comp, entry, n_elements) in selections.items():
            cfg = self.config_manager.resolve(
                comp.prefix, name, shape=(comp.n_elements,)
            )
            try:
                sels = entry.expression_configs(
                    cfg.get("expressions", {}),
                    n_elements=n_elements,
                    where=f"{prefix}.{name}",
                )
            except Exception:  # a broken manifest is another check's error
                continue
            for sel in sels:
                if sel.output_only:
                    continue
                consumer_mask = (
                    np.ones(n_elements, dtype=bool)
                    if sel.mask is None
                    else sel.mask
                )
                for dep in entry.dep_names(sel.config):
                    dep_key = self._dep_parameter_key(comp, dep)
                    if dep_key is None or dep_key not in reported:
                        continue
                    dep_mask = reported[dep_key]
                    # Element-for-element only where the two vectors are the
                    # same length; a mapped dep (a different length) is
                    # compared conservatively, as any overlap at all.
                    if dep_mask.size == consumer_mask.size:
                        clash = np.nonzero(dep_mask & consumer_mask)[0]
                    else:
                        clash = np.nonzero(dep_mask)[0]
                    if clash.size:
                        raise ValueError(
                            f"[{prefix}.{name}] its '{sel.key}' expression "
                            f"consumes '{dep}', whose element(s) "
                            f"{clash.tolist()} are REPORTED "
                            f"({dep_key[0]}.{dep_key[1]}, manifest role 3). A "
                            f"reported element is applied in a second build "
                            f"phase, after every parameter exists, so a "
                            f"consumer would read its pre-patch placeholder -- "
                            f"a number that looks fine and is not the quantity "
                            f"it names. Give this parameter an expression in "
                            f"coordinates the instance actually has (as "
                            f"orbit.tp does with its 'from_ecc' block), or stop "
                            f"reporting that element."
                        )

    @staticmethod
    def _dep_parameter_key(comp, dep):
        """``(prefix, param)`` a dependency string names, or None."""
        name = dep.split("[", 1)[0]
        if "." not in name:
            return (comp.prefix, name)
        parts = name.split(".")
        return (parts[0], parts[-1])

    def derived_params(self):
        """`(component_prefix, param_name)` pairs the manifests actually derive.

        The static `expressions:` block in a defaults.yaml is not the answer:
        a component may declare the same parameter free in one topology and
        derived in another (planet.mass is sampled linearly when RV or
        astrometry measures it, and derived from log_q otherwise). The rule
        is `manifest.interpret_manifest_entry`'s, shared with
        `Component.add_parameter` (stage 6) and
        `graph.determine_pymc_build_order` (the build order) -- a manifest value that
        is a string, or a dict carrying an "expr_key", names an expression; a
        bare None is a free parameter.  Valid after stage 3.

        The question is asked through `expression_config` against the
        resolved config, exactly as the two build-time consumers ask it, and
        NOT through the structural `names_expression`.  The two can only
        differ when an entry names a block the config does not define, which
        `expression_config` raises on -- but only the build path would ever
        reach that raise, and this method's callers (`solve_api`, the GUI's
        Tune tab) never build a model.  Answering structurally there would
        keep exactly the silence this raise exists to remove: a parameter
        reported "derived" is excused from `solve_api._bounds_diagnostics`,
        so a broken expr_key would go on hiding an out-of-bounds start in
        the one tool whose job is to find them.  That is not hypothetical --
        it is what `rvinstrument.gamma` did until 2026-08.
        """
        return {
            key
            for key, mask in self.derived_elements().items()
            if bool(np.all(mask))
        }

    def derived_elements(self):
        """``(component_prefix, param_name) -> boolean mask`` of derived elements.

        The per-element form of :meth:`derived_params`, and the one every
        reporting consumer should ask: a vector whose instances chose different
        parameterizations is derived for SOME elements and sampled for others,
        and answering per parameter forces a choice between excusing a sampled
        element from the checks a sampled element needs (``derived_params``'s
        consumers skip derived parameters) and subjecting a derived one to them.

        ``derived_params`` keeps its historical meaning -- every element derived
        -- so a partially derived vector no longer counts as derived there.
        That is the conservative direction: its consumers treat "derived" as
        "exempt", and a mixed vector has sampled elements that must not be
        exempt.  Valid after stage 3.
        """
        out = {}
        for comp in self.active_components.values():
            for name, raw in getattr(comp, "manifest", {}).items():
                entry = interpret_manifest_entry(raw)
                if not entry.names_expression:
                    continue
                n_elements = comp.n_elements
                if entry.shape:
                    shape = entry.shape
                    n_elements = int(
                        np.prod(shape)
                        if isinstance(shape, tuple)
                        else int(shape)
                    )
                cfg = self.config_manager.resolve(
                    comp.prefix, name, shape=(comp.n_elements,)
                )
                mask = np.zeros(n_elements, dtype=bool)
                for sel in entry.expression_configs(
                    cfg.get("expressions", {}),
                    n_elements=n_elements,
                    where=f"{comp.prefix}.{name}",
                ):
                    mask |= (
                        np.ones(n_elements, dtype=bool)
                        if sel.mask is None
                        else sel.mask
                    )
                if mask.any():
                    out[(comp.prefix, name)] = mask
        return out

    def active_elements(self):
        """``(component_prefix, param_name) -> boolean mask`` of ACTIVE elements.

        The complement is manifest role 4: elements that are not parameters of
        their instance's parameterization (a non-MIST star's EEP).  Only entries
        that actually mask something appear, so a caller can treat a missing key
        as "every element active".  Valid after stage 3, and the reporting
        layer's authority for what to leave out of a table.
        """
        out = {}
        for comp in self.active_components.values():
            for name, raw in getattr(comp, "manifest", {}).items():
                entry = interpret_manifest_entry(raw)
                if entry.options.get("mask") is None:
                    continue
                n_elements = comp.n_elements
                if entry.shape:
                    shape = entry.shape
                    n_elements = int(
                        np.prod(shape)
                        if isinstance(shape, tuple)
                        else int(shape)
                    )
                out[(comp.prefix, name)] = entry.activity_mask(
                    n_elements, where=f"{comp.prefix}.{name}"
                )
        return out

    def manifest_overrides(self):
        """``(component_prefix, param_name) -> the manifest "overrides" dict``.

        The third of the after-prepare() tables a reporting consumer needs
        (with :meth:`derived_elements` and :meth:`active_elements`), and for
        the same reason: what the BUILD does is decided by the manifest, and a
        report that re-derives it from the config alone disagrees.
        ``"overrides"`` is how a component supplies per-element defaults the
        user may still beat -- including the ``sigma: 0`` pin on a GP
        hyperparameter of a file that asked for no GP, a robust-likelihood
        parameter of a file with no ``likelihood:``, or an unread
        limb-darkening coefficient.  ``ConfigManager.resolve`` applies them
        only when handed them (``Component.add_parameter`` does), so
        ``export_solution`` reported every such element as free until it was
        given this.  Only entries that carry overrides appear.  Valid after
        stage 2.
        """
        out = {}
        for comp in self.active_components.values():
            for name, raw in getattr(comp, "manifest", {}).items():
                overrides = interpret_manifest_entry(raw).options.get(
                    "overrides"
                )
                if overrides:
                    out[(comp.prefix, name)] = overrides
        return out

    def build_likelihood(self, model, system):
        pass

    @property
    def prefix(self) -> str:
        return "system"

    def register_parameters(self, system):
        pass

    def register_element_slice_check(
        self, where, func_name, idx, sliced_expr, full_expr
    ):
        """Record a sliced per-element expression for start-point verification.

        Called by ``Component._element_expression`` when an expression supplies
        a SUBSET of a parameter's elements and its dependencies were therefore
        sliced.  Slicing is only sound if the physics is elementwise in those
        deps; the alignment of the deps themselves is proven statically, but
        "elementwise" is a property of the function, and a function that sums or
        contracts over the element axis would return something else entirely
        from sliced inputs.  So both graphs are kept and compared numerically at
        the start point (``verify_element_slices``), where real values exist --
        dummy inputs could agree by accident, and evaluating a random variable
        would draw from its prior instead of reading the start.
        """
        self._element_slice_checks.append(
            {
                "where": where,
                "func_name": func_name,
                "idx": np.asarray(idx, dtype=int),
                "sliced": sliced_expr,
                "full": full_expr,
            }
        )

    def verify_element_slices(self, model, rtol=1e-9, atol=1e-12):
        """Check every sliced per-element expression against the full one.

        One compiled function over all the checks, evaluated at the start point.
        A mismatch RAISES, naming the parameter and the physics function: it
        means the function is not elementwise in its dependencies, so the
        elements one instance's parameterization computed were derived from
        another instance's numbers -- wrong values with no other symptom.

        NaN is treated as agreement ONLY when both sides are NaN at the same
        entry; a NaN that appears in just one of them is a real disagreement.
        """
        checks = self._element_slice_checks
        if not checks:
            return 0
        outputs = []
        for chk in checks:
            full = pt.as_tensor_variable(chk["full"]())
            sliced = pt.as_tensor_variable(chk["sliced"]())
            take = pt.as_tensor_variable(chk["idx"].astype("int32"))
            outputs.append(full[take] if full.ndim else full)
            outputs.append(sliced if sliced.ndim else sliced)
        # The expressions were built against the model's random variables, but
        # a point maps the VALUE variables; without this substitution the
        # compiled function asks for an unnamed RV input the point cannot fill.
        outputs = model.replace_rvs_by_values(outputs)
        # inputs=model.value_vars, not the default (whatever the outputs need):
        # a point carries EVERY value variable, and a function compiled for a
        # subset rejects the rest ("Too many parameter passed").
        fn = model.compile_fn(
            outputs,
            inputs=model.value_vars,
            point_fn=True,
            on_unused_input="ignore",
        )
        values = fn(self.get_raw_start(model))
        for k, chk in enumerate(checks):
            a = np.atleast_1d(np.asarray(values[2 * k], dtype=float))
            b = np.atleast_1d(np.asarray(values[2 * k + 1], dtype=float))
            both_nan = np.isnan(a) & np.isnan(b)
            if a.shape != b.shape or not np.allclose(
                np.where(both_nan, 0.0, a),
                np.where(both_nan, 0.0, b),
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            ):
                raise ValueError(
                    f"[{chk['where']}] the physics function "
                    f"'{chk['func_name']}' is not elementwise in its "
                    f"dependencies: evaluated on the dependencies sliced to "
                    f"element(s) {chk['idx'].tolist()} it gives {b.tolist()}, "
                    f"but evaluated on the full vectors and then indexed it "
                    f"gives {a.tolist()}. A per-element parameterization "
                    f"switch slices the dependencies so an instance that did "
                    f"not choose this parameterization cannot poison the "
                    f"result, which is only valid for an elementwise "
                    f"function. Give the expression a form that treats each "
                    f"element independently."
                )
        return len(checks)

    def register_branch_alternative(self, label, replacements, weight=0.5):
        """Declare one alternative value of a many-to-one parameterization.

        A component calls this when the coordinate it samples does not determine
        its physical quantity uniquely -- today only the V_c/V_e eccentricity,
        whose inversion is a quadratic with two roots (Eastman 2024 eq 5).
        ``replacements`` maps the node the model was BUILT with to the node the
        alternative branch would have used; ``label`` names the branch for logs.

        Two branches may name the SAME node -- two V_c/V_e orbits are two
        elements of one ``ecc`` vector, so they do -- and the mixture then has to
        apply both at once.  That works only if each replacement is written
        RELATIVE to the node it replaces (``set_subtensor(ecc[i], ...)``, not a
        node built from scratch), which is what lets ``_add_branch_mixtures``
        compose them by substituting one after the other.  It checks, because
        the failure is silent: merging the two into one dict, as the first
        version did, kept whichever was declared last and quietly marginalized
        over 3 of the 4 combinations.

        ``_add_branch_mixtures`` then marginalizes the likelihood over the
        branches instead of letting the component choose one.  That is the whole
        point: the paper picks a root with a discrete sign parameter, which for a
        gradient sampler means a piecewise-constant coordinate and a logp that
        jumps -- and picking the "physical" root instead means a choice that
        depends on the current parameters, i.e. the same discontinuity wearing a
        different hat.  Marginalizing is smooth, and it is also the honest
        statement: the data do not say which root is real.
        """
        replacements = dict(replacements)
        claimed = {
            key
            for branch in self._branch_alternatives
            for key in branch["replacements"]
        }
        for key, value in replacements.items():
            if key in claimed and key not in ancestors([value]):
                raise ValueError(
                    f"[system] branch '{label}' replaces node "
                    f"'{key.name or key}', which another branch also replaces, "
                    f"with an expression that does not read it. Two branches "
                    f"can only be applied together when each is written "
                    f"relative to the node it replaces (e.g. "
                    f"set_subtensor(node[i], ...)); as written, one of the two "
                    f"substitutions would be lost and the mixture would cover "
                    f"fewer than 2^k combinations."
                )
        self._branch_alternatives.append(
            {
                "label": label,
                "replacements": replacements,
                "weight": weight,
            }
        )

    def _add_branch_mixtures(self, model):
        """Marginalize the likelihood over every declared branch alternative.

        For k declared branches this adds ONE potential covering all 2^k
        combinations::

            total = logsumexp_over_combinations(log w_c + L_c)

        where ``L_c`` is the sum of every model term with that combination's
        nodes substituted, and ``w_c`` the product of the branch weights.  It is
        built as ``logsumexp(...) - L_ref``, because PyMC has already added
        ``L_ref`` (the as-built terms): the difference cancels it exactly, so no
        component has to know this is happening and no term is counted twice.

        Two properties fall out of substituting into the WHOLE term sum, and
        both are wanted:

        * Any term that does not depend on the substituted nodes appears in
          every branch identically and factors straight out of the logsumexp --
          so the mixture is over exactly the terms that care.
        * Any PRIOR term that does depend on them (the V_c/V_e Jacobian, the
          eccentricity collision bound, and a future orbit-crossing penalty
          coupling two orbits) is replicated per branch, which makes each
          branch's weight ``log w + log|J| + barriers`` -- the form review 8.4.4
          specifies for folded likelihoods, for free.

        Cost is 2^k evaluations of the whole logp, so it is logged, and a
        warning names the multiplier past two branches.
        """
        from pytensor.graph.replace import graph_replace

        branches = list(self._branch_alternatives)
        if not branches:
            return 0

        # RV-level terms, deliberately not model.logp(): that graph has already
        # had the random variables rewritten into value variables, so the nodes
        # a component handed us are no longer in it and graph_replace would find
        # nothing to replace.  Potentials and observed logps are stored
        # RV-level, and PyMC converts them consistently at logp time.
        terms = [
            pm.logp(rv, model.rvs_to_values[rv]).sum()
            for rv in model.observed_RVs
        ]
        terms += [pt.sum(p) for p in model.potentials]
        if not terms:
            logger.warning(
                "[system] branch alternatives were declared (%s) but the model "
                "has no likelihood terms to marginalize; skipping the mixture.",
                ", ".join(b["label"] for b in branches),
            )
            return 0
        l_ref = terms[0] if len(terms) == 1 else pt.add(*terms)

        n_comb = 2 ** len(branches)
        labels = ", ".join(b["label"] for b in branches)
        if len(branches) > 2:
            logger.warning(
                "[system] %d branch alternatives (%s) means the likelihood is "
                "evaluated %d times per step (2^%d) -- every combination of "
                "roots. Set 'fitvcve: false' on the orbits that do not need it "
                "to bring that down.",
                len(branches),
                labels,
                n_comb,
                len(branches),
            )
        else:
            logger.info(
                "[system] marginalizing the likelihood over %d branch "
                "combination(s) (%s).",
                n_comb,
                labels,
            )

        pieces = []
        for combo in range(n_comb):
            term = l_ref
            log_w = 0.0
            for bit, branch in enumerate(branches):
                if combo & (1 << bit):
                    # One substitution at a time, NOT one merged dict: two
                    # branches routinely name the same node (two V_c/V_e orbits
                    # are two elements of one `ecc` vector), and merging kept
                    # only the last, silently marginalizing over 3 of the 4
                    # combinations.  Applied in sequence, the second pass
                    # rewrites the first's `set_subtensor` base too, so both
                    # elements land -- which is why register_branch_alternative
                    # requires a replacement to read the node it replaces.
                    term = graph_replace(term, branch["replacements"])
                    log_w += float(np.log(branch["weight"]))
                else:
                    log_w += float(np.log1p(-branch["weight"]))
            pieces.append(term + log_w)

        total = pieces[0]
        for piece in pieces[1:]:
            total = pt.logaddexp(total, piece)
        pm.Potential("branch_mixture", total - l_ref)
        return n_comb

    def build_model(self):
        """Constructs the PyMC probabilistic model for the entire system."""
        self._element_slice_checks = []
        self._branch_alternatives = []
        with pm.Model() as model:
            # Stage 5: Automatic PyTensor Map Conversion
            # Convert logical numpy arrays into PyTensor variables for the graph
            for comp in self.active_components.values():
                comp.build_tensor_maps()

            # Build order for stage 6: topological sort
            # Fetch the dynamic, component-agnostic build order driven by the physics dependency graph
            pymc_build_order = determine_pymc_build_order(
                self.active_components, self.config_manager
            )

            # Stage 6: Linearly materialize the nodes node-by-node
            for param_path in pymc_build_order:
                comp_name, param_name = param_path.split(".", 1)
                if comp_name in self.active_components:
                    comp = self.active_components[comp_name]
                    if param_name in getattr(comp, "manifest", {}):
                        comp.add_parameter(model, param_name, self)

            # Warn about user-defined parameter links whose target was never
            # materialized (e.g. star.A.age linked in the params file, but the
            # current model configuration does not build 'age').
            for target in getattr(self.config_manager, "links", {}):
                t_comp, t_param = target.split(".")[0], target.split(".")[-1]
                comp = self.active_components.get(t_comp)
                if comp is None or not isinstance(
                    getattr(comp, t_param, None), Parameter
                ):
                    logger.warning(
                        f"Parameter link on '{target}' had no effect: "
                        f"'{t_comp}.{t_param}' is not built by the current model "
                        f"configuration."
                    )

            # Stage 7: LIKELIHOOD
            for comp in self.active_components.values():
                if hasattr(comp, "build_likelihood"):
                    comp.build_likelihood(model, system=self)

            # After stage 7: REPORTED elements (manifest role 3).  Deliberately
            # after stage 7: a reported element is consumed by nothing, so
            # every consumer in stages 6-7 has already read the phase-1 tensor
            # -- which is what makes the per-parameter cycle these expressions
            # would otherwise create dissolve.  See
            # Component.finalize_reported and Parameter.finalize_deferred.
            for comp in self.active_components.values():
                finalize = getattr(comp, "finalize_reported", None)
                if callable(finalize):
                    finalize(model, system=self)

            # After stage 7: BRANCH MIXTURES.  A component whose parameterization is
            # many-to-one (today: the V_c/V_e eccentricity, whose inversion is
            # quadratic) declares its alternative nodes and the likelihood is
            # marginalized over them here.  Must come last: it snapshots every
            # term the model has, so every term has to exist first.
            self._add_branch_mixtures(model)

        # Mixed-parameterization vectors only: verify that each sliced
        # expression agrees with the unsliced one on the elements it supplies,
        # at the start point.  No-op (and no compile) for a model without one.
        self.verify_element_slices(model)

        self.compile_plotter_functions(model)
        return model

    def get_all_parameters(self):
        """
        Extracts a flat list of all Parameter objects, respecting
        both Component and Parameter insertion order.
        """
        params = []
        for comp in self.get_all_components():
            # Use __dict__.values() to preserve the definition order from __init__/build_parameters
            for attr in comp.__dict__.values():
                if isinstance(attr, Parameter):
                    params.append(attr)
        return params

    def get_raw_start(self, model):
        """
        Build the raw-space starting point explicitly from each Parameter's
        stored raw_initval (set in build_pymc): 0 for logit-transformed
        elements (raw=0 maps to initval by construction) and
        (initval - mu)/sigma for Gaussian-path elements, so the physical
        starting value is always initval even when an explicit prior mean
        mu != initval.

        We override model.initial_point() here to guarantee the physical
        starting value is always our initval.
        """
        raw_start = model.initial_point()
        lookup = {p.label: p for p in self.get_all_parameters()}
        for key in raw_start:
            name = key[: -len("_raw")] if key.endswith("_raw") else key
            par = lookup.get(name)
            raw_init = (
                getattr(par, "raw_initval", None) if par is not None else None
            )
            if raw_init is not None and np.size(raw_init) == np.size(
                raw_start[key]
            ):
                raw_start[key] = np.asarray(raw_init, dtype=float).reshape(
                    np.shape(raw_start[key])
                )
            else:
                raw_start[key] = np.zeros_like(raw_start[key])
        return raw_start

    def jitter_raw_start(self, center, raw_scales, factor, rng):
        """One over-dispersed start, drawn in PHYSICAL space, returned as raw.

        Jittering directly in raw space -- center + factor*scale*N(0,1) -- looks
        natural but saturates the logit transform.  A raw element reaches its
        bound through `lower + span*sigmoid(lq)`, so what governs saturation is
        the jitter's width in lq (= factor * scale * init_scale_logit), not its
        width in raw.  Once that approaches ~3 the sigmoid folds both tails onto
        the bounds instead of spreading across them: measured on a [0,1]
        parameter, pileup within 1% of a bound goes 0.0000 -> 0.105 -> 0.276 as
        sigma/span goes 0.1 -> 0.3 -> 1.0, and a parameter whose logp is flat out
        to its bounds starts 31.5% of its chains within 1% of a bound where
        uniform wants 2.0%.  The flat case cannot be fixed by correcting the
        scale: its jitter width in lq is factor * 1.41 regardless of init_scale.

        Drawing in physical space from a Gaussian TRUNCATED to [lower, upper]
        reproduces what rejecting out-of-bounds draws would give, in closed form
        and with no rejection loop, and self-adapts across the whole range:
          - scale << span: truncation never bites -> plain factor-x
            over-dispersed Gaussian, unchanged.
          - scale ~ span:  truncated Gaussian, no pileup.
          - flat:          a Gaussian far wider than the interval, truncated,
                           IS uniform -- which is max entropy on a bounded
                           range, so over-dispersion correctly runs out there
                           rather than folding onto the bounds.
        Pileup never exceeds uniform's at any scale (worst measured 0.0196 vs
        uniform's 0.0200) because a truncated normal's density is monotone
        toward each bound.

        `raw_scales` holds the probe's per-element 0.5-nat step
        (whitening.probe_scales) in raw units; it is converted to a physical
        half-width through this parameter's own transform.  The probe scale
        measures the ACTUAL local posterior width including the likelihood --
        for a flat parameter it lands at 0.304*span independent of the
        whitening scale, which is exactly what makes the flat case come out
        uniform.  (After the startup whitening rescale these steps are ~1 by
        construction, but PTDE re-probes rather than assuming it.)

        Elements without a finite [lower, upper] pair are not logit-transformed
        (their raw -> physical map is linear and unbounded), so they keep plain
        Gaussian jitter; build_pymc's soft barriers handle any one-sided bound.
        """
        from scipy.stats import truncnorm

        lookup = {p.label: p for p in self.get_all_parameters()}
        out = {}
        for key, cval in center.items():
            name = key[: -len("_raw")] if key.endswith("_raw") else key
            par = lookup.get(name)
            tf = (
                getattr(par, "_raw_transform", None)
                if par is not None
                else None
            )
            shape = np.shape(cval)
            c = np.asarray(cval, dtype=float).reshape(-1)
            sc = np.asarray(raw_scales[key], dtype=float).reshape(-1)

            if tf is None or len(tf["sampled_idx"]) != c.size:
                # No frozen transform (or a shape we cannot line up): fall back
                # to the historical raw-space jitter.
                out[key] = np.asarray(cval, dtype=float) + (
                    factor
                    * np.asarray(raw_scales[key], dtype=float)
                    * rng.standard_normal(shape)
                )
                continue

            idx = tf["sampled_idx"]
            # Physical center, and the physical half-width spanned by one probe
            # scale either side of it -- measured through the real transform
            # rather than linearized, since the point is that it is nonlinear.
            p0 = par.phys_from_raw(c)
            p_hi = par.phys_from_raw(c + sc)
            p_lo = par.phys_from_raw(c - sc)

            new_phys = np.array(p0, dtype=float, copy=True)
            for j, i in enumerate(idx):
                half = 0.5 * abs(p_hi[i] - p_lo[i])
                s = factor * half
                if not np.isfinite(s) or s <= 0:
                    continue  # degenerate scale: start at the seed
                if not tf["use_logit"][i]:
                    new_phys[i] = p0[i] + s * rng.standard_normal()
                    continue
                lower, upper = tf["lowers"][i], tf["uppers"][i]
                a, b = (lower - p0[i]) / s, (upper - p0[i]) / s
                new_phys[i] = truncnorm.rvs(
                    a, b, loc=p0[i], scale=s, random_state=rng
                )

            raw_vec = par.raw_from_initval(new_phys)
            out[key] = np.asarray(raw_vec, dtype=float).reshape(shape)
        return out

    def get_raw_starts(self, model):
        """Multi-seed variant of get_raw_start (P4).

        Returns (starts, seed_indices): a list of raw-space start dicts (one per
        usable seed) and the parallel list of the original seed indices they
        came from.  seed 0 is always the canonical single start (identical to
        get_raw_start); additional seeds are built from the relaxation engine's
        per-seed solutions (config_manager.seed_resolved) by re-mapping each
        parameter's solved physical initval through its frozen forward transform
        (Parameter.raw_from_initval), so bounds/scale stay fixed at seed 0 and
        only the start position moves.

        A seed whose solved start violates a hard bound is logged and skipped
        entirely (a clipped start would sit in no posterior basin).  Falls back
        to a single-element list when no multi-seed solutions exist.
        """
        base = self.get_raw_start(model)
        seed_resolved = getattr(self.config_manager, "seed_resolved", None)
        if not seed_resolved or len(seed_resolved) <= 1:
            return [base], [0]

        lookup = {p.label: p for p in self.get_all_parameters()}
        starts, seed_indices = (
            [base],
            [0],
        )  # seed 0 == the canonical base start

        for k in range(1, len(seed_resolved)):
            resolved = seed_resolved[k]
            raw = {
                key: np.array(v, dtype=float, copy=True)
                for key, v in base.items()
            }
            violated = False
            for key in base:
                name = key[: -len("_raw")] if key.endswith("_raw") else key
                par = lookup.get(name)
                if par is None:
                    continue
                iv = self._seed_initvals_for(par, resolved)
                if iv is None:
                    continue  # not seeded: keep the seed-0 raw start
                try:
                    raw_vec = par.raw_from_initval(iv)
                except SeedBoundViolation as e:
                    logger.warning(
                        f"Multi-seed: seed {k} start violates a bound ({e}); "
                        f"skipping this seed (a clipped start is in no basin)."
                    )
                    violated = True
                    break
                if np.size(raw_vec) == np.size(base[key]):
                    raw[key] = raw_vec.reshape(np.shape(base[key]))
            if not violated:
                starts.append(raw)
                seed_indices.append(k)

        logger.info(
            f"Multi-seed starts: {len(starts)}/{len(seed_resolved)} seeds "
            f"usable (seed indices {seed_indices})."
        )
        return starts, seed_indices

    def apply_polished_starts(self, polished_raws, seed_indices):
        """Adopt polished raw starts (polish.polish_raw_starts) as the
        canonical starts.

        Seed 0 is written into each Parameter's ``raw_initval`` -- which
        get_raw_start, get_mcmc_init, and the sampler initvals all read --
        and its physical values into ``Parameter.initval`` so the startup
        table and diagnostics report the polished start.  set_whitening
        keeps a nonzero raw_initval pinned to the same physical point
        through later rescales.  Seeds k > 0 are written back into
        config_manager.seed_resolved as physical (internal-unit) values, so
        get_raw_starts re-derives them through the frozen transform in
        whatever raw coordinates are current.
        """
        lookup = {p.label: p for p in self.get_all_parameters()}
        seed_resolved = getattr(self.config_manager, "seed_resolved", None)

        for s, raw in enumerate(polished_raws):
            for key, vec in raw.items():
                name = key[: -len("_raw")] if key.endswith("_raw") else key
                par = lookup.get(name)
                tf = (
                    getattr(par, "_raw_transform", None)
                    if par is not None
                    else None
                )
                if tf is None:
                    continue
                new_raw = np.asarray(vec, dtype=float).reshape(-1)
                if new_raw.size != len(tf["sampled_idx"]):
                    continue
                phys = np.asarray(par.phys_from_raw(new_raw), dtype=float)

                if s == 0:
                    par.raw_initval = new_raw.copy()
                    n_elements = (
                        int(np.prod(par.shape))
                        if par.shape not in ((), None)
                        else 1
                    )
                    iv = np.asarray(
                        to_vec(par.initval, n_elements, fill=np.nan),
                        dtype=float,
                    )
                    for i in tf["sampled_idx"]:
                        iv[i] = phys[i]
                    par.initval = (
                        float(iv[0]) if par.shape in ((), None) else iv
                    )
                else:
                    k = seed_indices[s]
                    if (
                        not seed_resolved
                        or k >= len(seed_resolved)
                        or seed_resolved[k] is None
                    ):
                        continue
                    comp_type = par.label.split(".")[0]
                    param_name = par.label.split(".", 1)[1]
                    for i in tf["sampled_idx"]:
                        seed_resolved[k][f"{comp_type}.{i}.{param_name}"] = (
                            float(phys[i])
                        )

    def _seed_initvals_for(self, par, resolved):
        """Internal-unit initval vector for one Parameter under one seed's solved
        state, or None if that seed does not touch any of its elements."""
        comp_type = par.label.split(".")[0]
        param_name = par.label.split(".", 1)[1]
        n_elements = (
            int(np.prod(par.shape)) if par.shape not in ((), None) else 1
        )
        base_iv = to_vec(par.initval, n_elements, fill=np.nan)
        vals = np.array(base_iv, dtype=float).reshape(-1).copy()
        found = False
        for i in range(n_elements):
            path = f"{comp_type}.{i}.{param_name}"
            if path in resolved:
                vals[i] = resolved[path]
                found = True
        return vals if found else None

    def get_internal_point(self, model, raw_point):
        """Evaluates graph deterministics for plotting/physics without user-unit conversion."""
        output_vars = model.free_RVs + model.deterministics

        eval_fn = pytensor.function(
            inputs=model.free_RVs,
            outputs=output_vars,
            on_unused_input="ignore",
        )

        # Pull the values in the exact order the function expects them
        input_values = [raw_point[v.name] for v in model.free_RVs]

        physical_values = eval_fn(*input_values)

        return {
            var.name: val for var, val in zip(output_vars, physical_values)
        }

    def get_physical_point(self, model, raw_point):
        output_vars = model.free_RVs + model.deterministics

        eval_fn = pytensor.function(
            inputs=model.free_RVs,
            outputs=output_vars,
            on_unused_input="ignore",
        )

        # Pull the values in the exact order the function expects them
        input_values = [raw_point[v.name] for v in model.free_RVs]

        physical_values = eval_fn(*input_values)
        param_lookup = self.get_parameter_lookup()

        results = {}
        for var, val in zip(output_vars, physical_values):
            if var.name in param_lookup:
                # Standardize: Always use from_internal to ensure we return User Units
                results[var.name] = param_lookup[var.name].from_internal(val)
            else:
                results[var.name] = val

        return results

    def distribute_posterior(self, idata):
        """Maps the traces from idata back to the individual Parameter objects."""
        posterior = az.extract(idata, keep_dataset=True)
        param_lookup = self.get_parameter_lookup()

        # Mode labels (outputs.modes.identify_modes) ride along in the
        # posterior group; keep them sample-aligned with every Parameter's
        # posterior so per-mode summaries can be computed downstream.
        # Draws labeled -1 are invalid (runaway/stuck chains rejected by
        # identify_modes) and must not contaminate any reported summary, so
        # they are dropped from the distributed posterior entirely.
        if "mode" in posterior:
            labels = np.asarray(posterior["mode"].values, dtype=int)
            if (labels < 0).any():
                keep = labels >= 0
                posterior = posterior.isel(sample=keep)
                labels = labels[keep]
                logger.info(
                    f"distribute_posterior: dropped {int((~keep).sum())} "
                    f"invalid/unassigned draws flagged by mode identification"
                )
            self.mode_labels = labels
            self.n_modes = int(
                posterior["mode"].attrs.get("n_modes", labels.max() + 1)
            )
        else:
            self.mode_labels = None
            self.n_modes = 1

        # Dynamically discover all components (Stars, Planets, Orbits, Instruments, etc.)
        for attr_name, comp in self.__dict__.items():
            if isinstance(comp, Component) and comp is not self:
                self._set_comp_posterior(comp, posterior, param_lookup)

    def _set_comp_posterior(self, component, posterior, param_lookup):
        for attr_name in dir(component):
            attr = getattr(component, attr_name)

            if isinstance(attr, Parameter):
                if attr.label in posterior:
                    # Case A: Named Deterministic in the trace (user units).
                    attr.posterior = posterior[attr.label]
                elif attr.expression is not None:
                    # Case B: Not in the trace; evaluate the PyTensor expression.
                    # Pass param_lookup so generate_posterior converts user-unit
                    # inputs → internal → evaluates → back to user units.
                    attr.posterior = attr.generate_posterior(
                        posterior, param_lookup=param_lookup
                    )

            # Recurse to children (Stars, Planets, etc.)
            elif isinstance(attr, Component) and attr is not component:
                self._set_comp_posterior(attr, posterior, param_lookup)

    def get_parameter_lookup(self):
        """
        Recursively finds all Parameter objects in the system and
        returns a flat dictionary mapped by their labels.
        """
        lookup = {}

        def walk(obj):
            # 1. Check if the object itself is a Parameter
            if isinstance(obj, Parameter):
                lookup[obj.label] = obj

            # 2. If it's a list (like self.planets), walk each item
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

            # 3. If it's a Component, look at all its attributes
            elif isinstance(obj, Component):
                # We use __dict__.values() to see everything inside the component
                for attr in obj.__dict__.values():
                    walk(attr)

        # Start the walk from the system itself
        walk(self)
        return lookup

    def get_all_components(self):
        """
        Yields each component in the system exactly once.
        This maintains the Star -> Orbit -> Planet -> Instrument order.
        """
        seen_components = set()

        def crawl(obj):
            # We only want to crawl the top-level attributes of the System
            # or the internal structure of a Component.

            # Use __dict__ to respect insertion order instead of dir() (alphabetical)
            for attr_name, attr in obj.__dict__.items():
                if attr_name.startswith("_"):
                    continue

                # 1. If it's a Component, yield it and its children
                if isinstance(attr, Component):
                    if id(attr) not in seen_components:
                        seen_components.add(id(attr))
                        yield attr
                        yield from crawl(attr)

                # 2. If it's a list of components (future-proofing)
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if (
                            isinstance(item, Component)
                            and id(item) not in seen_components
                        ):
                            seen_components.add(id(item))
                            yield item
                            yield from crawl(item)

        yield from crawl(self)

    def get_mcmc_init(self, model):
        """The sampler's start point, keyed by PyMC value-variable name.

        The whitened start is 0.0 for every logit element and
        ``(initval - mu)/sigma`` for a Gaussian-path one (see
        ``get_raw_start``); this forwards it through each RV's transform so
        PyMC can take it as an ``initvals`` dict.

        Returns only that dict.  It used to return three more things -- a
        vector of 1.0s sized by the total transformed dimension (for a NUTS
        ``scaling`` argument nothing has passed since PTDE replaced
        DEMetropolis), plus ``{label: initval}`` and ``{label: init_scale}``
        maps -- and the two physical maps were dead by construction: the one
        caller handed them straight to ``inspect_start``, which reads
        ``p.initval`` / ``p.init_scale`` off the Parameters itself.
        """
        transformed_inits = {}

        # 1. Map Unity Space (the sampler's world) to PyMC transformed values.
        # Only iterate free_RVs — observed RVs also appear in rvs_to_values but
        # are not sampled, and their (potentially large) shapes would corrupt
        # total_dims and the Metropolis proposal covariance.
        raw_start = self.get_raw_start(model)
        free_rvs = set(model.free_RVs)
        for rv, value_var in model.rvs_to_values.items():
            if rv not in free_rvs:
                continue
            # The unity start is 0.0 for logit params, (initval-mu)/sigma for
            # Gaussian-path params (see get_raw_start).
            unity_start = raw_start.get(
                value_var.name, np.zeros(rv.shape.eval(), dtype=float)
            )
            transform = model.rvs_to_transforms.get(rv)

            if transform is not None:
                # Forward the 0.0 through the interval/log math
                t_node = transform.forward(
                    pt.as_tensor_variable(unity_start), *rv.owner.inputs
                )
                transformed_inits[value_var.name] = t_node.eval()
            else:
                # No transform, raw == value
                transformed_inits[value_var.name] = unity_start

        return transformed_inits

    def compile_plotter_functions(self, model):
        """
        Gathers the global sampling parameters so components know
        the exact input signature required for their compiled functions,
        then tells each component to compile its own plotters.
        """
        all_params = self.get_all_parameters()
        # The compiled plotters take the NON-derived parameters as inputs.  A
        # vector whose instances chose different parameterizations is derived on
        # only some elements, and it belongs here: its sampled elements have no
        # other input, and its derived ones are read from the point (which
        # carries the whole Deterministic vector) rather than recomputed.
        # (Read from the build's own role masks, not from `expression is None`:
        # a fully derived vector may be declared per element too, and then its
        # `expression` field is None while every element is derived.)
        self.plot_params = [
            p
            for p in all_params
            if not bool(np.all(np.atleast_1d(p.is_derived)))
        ]

        # Delegate the actual compilation to the components
        for comp in self.active_components.values():
            comp.compile_plotters(model, system=self)
