import weakref
from abc import ABC, abstractmethod

import numpy as np
import pytensor.tensor as pt

from ..manifest import interpret_manifest_entry
from ..physics_registry import PHYSICS_REGISTRY
from .parameter import ElementExpression, OwnPrePatchRef, Parameter


def in_topology(system, name):
    """Is component ``name`` part of this system?  Returns instance or config.

    THE answer to "is component X in the topology?", which was re-derived
    four ways -- three in ``star.py`` and two in ``orbit.py``, each with a
    different holder chain (review 4.8.1).  They were not equivalent, and the
    disagreements were exactly where a partial construction lives:

    * ``Star.register_parameters``' local ``in_topology`` read ``system.config``
      OR ``config_manager.system_config`` as an ``elif``, so a system carrying
      both but whose ``config`` lacked the key never consulted the second.
    * ``Orbit._topology`` and ``Orbit.register_parameters``' inline check
      consulted ``config_manager.system_config`` not at all.
    * Only ``Star._galactic_imf`` looked at ``active_components`` first, which
      is the sole holder that carries the BUILT instance.

    The chain here is the union, in the order of decreasing authority:

    1. ``system.active_components[name]`` -- the built instance, populated in
       ``System.__init__``, so it is available from stage 1 onward.
    2. ``getattr(system, name)`` -- the same instance under its attribute,
       which is what the mock systems in the test suite provide.
    3. the raw config block, from ``system.config`` then
       ``system.config_manager.system_config`` -- for a component whose
       instance does not exist yet or at all (a premature
       ``evolutionarymodel:`` block that no component backs still counts as
       topology, deliberately: see ``Star.register_parameters``).

    Returns the first hit, or ``None``.  Truthiness is the common use
    (``if in_topology(system, "sed")``), and the value is there for the
    caller that wants the instance (``Star._galactic_imf`` reads its
    ``IMF:``).  **An empty config block is a real answer** -- ``sed: {}`` is
    a system WITH an SED -- so this returns the block itself and callers must
    test ``is not None``, not truthiness, when an empty block matters.  The
    two shipped callers that care (``_galactic_imf``, ``structure_consumers``)
    read a key off it or only need the boolean, so both are safe with either.

    A module function rather than only a ``System`` method because every
    caller is a component holding a ``system`` that may be a test double;
    ``System.in_topology`` delegates here so there is still exactly one
    implementation.
    """
    components = getattr(system, "active_components", None)
    if isinstance(components, dict) and name in components:
        return components[name]

    inst = getattr(system, name, None)
    if inst is not None:
        return inst

    for holder in (
        getattr(system, "config", None),
        getattr(
            getattr(system, "config_manager", None), "system_config", None
        ),
    ):
        if isinstance(holder, dict) and name in holder:
            return holder[name]
    return None


def resolve_star_ref(ref, star_names, where):
    """Star INDEX from a name, a ``star.<name>``/``star.<i>`` path, or an index.

    The one translator behind every user-facing "index or name" star
    reference: an instrument's or a band's ``star_ndx:``, an SED
    ``photType`` entry, a relation component's ``star:`` key.  It existed
    three times with three messages and, worse, was simply absent from the
    two schemas that advertised it -- ``rvinstrument`` and ``band`` both
    documented ``star_ndx`` as "Index or name" while every consumer called
    ``int()`` on it, so a name crashed with a raw ValueError (review 3.5.1).

    ``where`` names the offending config location in the error, since only
    the caller knows it ("band 'I' star_ndx", "photType", "mann 'B'").
    ``star_names`` may be empty, in which case only integers and digit
    strings resolve -- a caller running before the star instances are known
    still gets the historical behaviour rather than a spurious failure.
    """
    names = list(star_names or [])
    n = len(names)

    def _bad(reason):
        known = f" Known stars: {names}." if names else ""
        return ValueError(
            f"{where}: {reason}.{known}"
            + (f" Valid indices are 0..{n - 1}." if n else "")
        )

    # bool is an int in Python; `star_ndx: true` is a typo, not star 1.
    if isinstance(ref, bool):
        raise _bad(f"invalid star reference {ref!r}")
    if isinstance(ref, (int, np.integer)):
        idx = int(ref)
    elif isinstance(ref, str):
        # A path spelling ("star.B", "star.1") names the same element as the
        # bare one; only the last segment selects.
        key = ref.split(".")[-1]
        if key in names:
            idx = names.index(key)
        else:
            try:
                idx = int(key)
            except ValueError:
                raise _bad(f"unknown star '{ref}'") from None
    else:
        raise _bad(f"invalid star reference {ref!r}")

    if n and not 0 <= idx < n:
        raise _bad(f"star index {idx} is out of range")
    return idx


class Component(ABC):
    """
    Base class for all physical and instrumental components in the system.

    This framework utilizes a "Lazy DAG" (Directed Acyclic Graph) architecture to
    safely construct complex PyMC models without deadlocks. The orchestration
    happens in the following distinct lifecycle stages:

    Stage 1: load_data()           - Ingests CSVs and calculates data-driven parameter estimates.
    Stage 2: build_maps()          - Generates Numpy integer arrays linking children to parents.
    Stage 3: register_parameters() - Declares the component's mathematical manifest.
    Stage 4: [System-Level]        - The ConfigManager symbolically solves the universe.
    Stage 5: build_tensor_maps()   - Auto-converts Numpy maps to PyTensor variables.
    Stage 6: add_parameter()       - Materializes PyMC nodes safely, one at a time.
    Stage 7: build_likelihood()    - Defines observational Likelihoods and Potentials.
    """

    # Does this component's parameter space routinely carry posterior-
    # SUPPRESSED but physically plausible alternative solutions -- distinct
    # basins with near-zero T=1 occupancy that a referee will still ask
    # about?  Declared per component and read only in aggregate ("does any
    # active component say yes"), so the sampler layer can default hot-rung
    # retention (samplers._common.resolve_store_hot_chains) ON for the
    # topologies where the suppressed-mode search earns its trace size and
    # off for the ones where it does not.
    #
    # A flag here rather than a component-name test up in run.py for the
    # same reason `supports_gp` is one: the higher-level code is
    # component-agnostic by design, and a future component with degenerate
    # solutions must be able to opt in without anyone editing the sampler.
    expects_suppressed_modes = False

    # Which of this component's injected context deps (see `context_dep_names`)
    # carry ONE ENTRY PER ELEMENT of the parameter they feed.  Only consulted
    # when an expression supplies a subset of a vector's elements and its deps
    # must therefore be sliced (see `_element_expression`): a context node is an
    # arbitrary tensor the component built, so nothing outside the component can
    # prove its alignment, and guessing wrong pairs the wrong instances
    # silently.  Declaring one here is a promise, so declare only what is true:
    # `orbit`'s per-orbit group masses are aligned; a per-observation vector is
    # not.
    aligned_context_deps = frozenset()

    # Human-readable heading for this component's block of the results table
    # (outputs/latex.py's \sidehead).  DECLARED here, rather than only being
    # assigned in ten component __init__s, so a generic consumer can read
    # `comp.label` without a getattr guard -- that guard was the tell that
    # the attribute was not part of the contract (review 4.2.3).  A component
    # that sets none gets its class name, filled in by __init__ below;
    # setting it as a CLASS attribute also works and is not overwritten.
    label = None

    def __init__(self, component_config, config_manager):
        """Standardized constructor for ALL components."""
        self.config = component_config
        self.config_manager = config_manager
        if type(self).label is None:
            self.label = type(self).__name__

        # Determine how many of this thing we are building
        self.n_elements = len(self.config)

        # Grab names for labeling PyMC nodes
        self.names = [c.get("name", f"{i}") for i, c in enumerate(self.config)]

        # Enforce unique names
        if len(set(self.names)) != len(self.names):
            raise ValueError(
                f"Duplicate names found in {self.__class__.__name__} configuration: {self.names}. "
                f"All component names must be unique."
            )

    @property
    @abstractmethod
    def prefix(self):
        """Naming prefix for the model (e.g., 'star', 'planet', 'inst')."""
        pass

    @classmethod
    def config_schema(cls):
        """Describe component-LEVEL config keys that are not parameters.

        These are the keys a user may set on a component's YAML block that
        are not sampled/derived parameters: data-file references, references
        to other components (bands, orbits, star indices), and free-form
        options. They are consumed by the introspection layer (see
        ``exozippy.introspect``) to drive documentation and a GUI without
        building a System.

        Returns a JSON-serializable list of dicts, each with keys:
          key      : the YAML key name
          kind     : "datafile" | "ref" | "option"
          accepts  : for "ref", the list of component yaml_keys it may point
                     at; for "datafile", a glob pattern (string); for
                     "option", a list of allowed values or None (free-form)
          required : bool, whether the key must be present
          doc      : human-readable description

        The base implementation returns an empty schema; components with
        such keys override this.
        """
        return []

    @classmethod
    def shared_parameter_names(cls):
        """Names of root-level parameters this component may register.

        Most parameters are declared in the component's own ``defaults.yaml``
        block and need no announcement.  A few live at the root level of
        ``components/defaults.yaml`` because several components share the
        blueprint (Instrument's optional GP hyperparameters are the current
        case), and a component's own block then overrides only what differs.
        Static consumers -- ``introspect``, and through it the GUI -- cannot
        infer those from the component's block alone, so this classmethod
        names them.

        The base implementation returns an empty list; mirrors
        ``config_schema()`` and ``get_utilities()``.
        """
        return []

    @classmethod
    def get_utilities(cls):
        """Declare the user-facing utility programs this component surfaces.

        A "utility" is a helper CLI that logically belongs to this component
        (downloading light curves, building an SED file, converting an
        external fit to a params file, ...). Returns a list of
        ``exozippy.utilities.registry.UtilitySpec``; the introspection layer
        (and a GUI) discover them generically, so no component names are ever
        hardcoded outside component-owned code.

        The base implementation returns an empty list; components with
        utilities override this. Mirrors ``config_schema()``.
        """
        return []

    def load_data(self, system):
        """
        Stage 1: Data Ingestion.
        Override this to load CSV files and push data-driven parameter guesses (like RV offsets)
        to the ConfigManager.
        """
        pass

    def build_maps(self):
        """
        Stage 2: Logical Mapping.
        Override this to define Numpy integer arrays (ending in '_map') that establish
        vectorized relationships between this component and its parents.
        """
        pass

    @abstractmethod
    def register_parameters(self, system):
        """
        Stage 3: The Blueprint.
        Define `self.manifest` (a dictionary) mapping parameter names to their physics
        dependencies, and push those symbols to the ConfigManager.
        """
        pass

    # Attribute names holding something that BELONGS TO ONE BUILD: a
    # pytensor node the component stashed, or a function compiled against
    # one.  ``System.build_model`` clears every one of them before stage 5,
    # so a second build on a live System cannot be handed the first model's
    # graph (reviews 1.5.2, 3.14.12).  Declare the name here rather than
    # writing another ad-hoc reset: a stage-6 cache (the SED's predicted
    # apparent magnitudes, read while parameters are still being
    # materialized) cannot be cleared at the top of stage 7, which is where
    # the first round of these resets landed.
    per_build_caches = ()

    def reset_build_caches(self):
        """Drop this component's per-build node caches.  See ``per_build_caches``."""
        for name in self.per_build_caches:
            setattr(self, name, None)

    def build_tensor_maps(self):
        """
        Stage 5: Automatic PyTensor Conversion.
        Scans the component's attributes. Any numpy array ending in '_map'
        is automatically converted to a PyTensor variable ending in '_map_tensor'.
        """
        for attr_name in list(self.__dict__.keys()):
            if attr_name.endswith("_map"):
                logical_array = getattr(self, attr_name)
                # Only convert if it's actually an array/list (safeguard)
                if isinstance(logical_array, (np.ndarray, list)):
                    tensor_name = attr_name + "_tensor"
                    tensor_var = pt.as_tensor_variable(logical_array).astype(
                        "int32"
                    )
                    setattr(self, tensor_name, tensor_var)

    def finalize_reported(self, model, system, context_nodes=None):
        """Wire and apply this component's REPORTED elements (manifest role 3).

        The second phase of the two-phase build, called by
        ``System.build_model`` after stage 7 for every component, inside the
        model context.  Every parameter exists by now, so the dependency of a
        reported expression resolves to an already-built node instead of
        recursing back into the parameter being built (see add_parameter).

        Returns the number of parameters finalized; zero -- and no work at all
        -- for a component with no reported elements, which is every component
        that does not flip a parameterization.
        """
        pending = getattr(self, "_pending_reported", None)
        if not pending:
            return 0

        context_nodes = context_nodes or {}
        finalized = 0
        for param_name, items in list(pending.items()):
            param = getattr(self, param_name, None)
            if not isinstance(param, Parameter):
                continue
            specs = [
                self._element_expression(
                    model, system, context_nodes, entry, sel, where
                )
                for (entry, sel, where) in items
            ]
            param.finalize_deferred(specs)
            finalized += 1
        self._pending_reported = {}
        return finalized

    @staticmethod
    def _has_built_parameter(comp, name):
        """Has ``comp`` already materialized ``name`` as a Parameter?

        The one predicate for "this node exists, do not build it again".  It
        must test the TYPE, not merely the attribute's presence: a component
        class attribute or method sharing a manifest parameter's name would
        otherwise be mistaken for the built node and either crash on
        ``.value`` or wire the wrong thing into the graph (review 2.2.2).
        Three of the four call sites already tested the type; the
        external-dependency one asked ``hasattr`` alone.  There are no
        collisions in the tree today -- which is exactly why the odd one out
        never showed.
        """
        return isinstance(getattr(comp, name, None), Parameter)

    @staticmethod
    def _parameter_is_current(comp, name, model):
        """Is ``comp.name`` a Parameter built for **this** ``model``?

        The build-time predicate, and the one every "do not build it again"
        site asks.  ``_has_built_parameter`` answers the narrower question
        "is there a Parameter here at all", which was the whole predicate
        until review 3.14.12: a component persists on the System, so a SECOND
        ``system.build_model()`` found every parameter still holding the FIRST
        model's node and handed it straight back -- the second model then
        contained the first model's random variables and its logp compile
        raised "Random variables detected in the logp graph".  The guard that
        makes a recursive dependency resolve once per build was also the thing
        that made a rebuild impossible.

        The stamp is a weakref, so a discarded model is not kept alive by the
        component that outlived it.

        Absent provenance counts as CURRENT, deliberately: a component that
        never went through ``add_parameter`` (a test double, or a Parameter
        set by hand) has no stamp registry and no name in it, and the only
        behaviour that may change here is the rebuild one.  A stamp naming a
        DIFFERENT model is the sole stale verdict.
        """
        if not Component._has_built_parameter(comp, name):
            return False
        stamps = getattr(comp, "_built_for_model", None)
        if not stamps:
            return True
        ref = stamps.get(name)
        if ref is None:
            return True
        return ref() is model

    def declared_star_names(self):
        """Star instance names from the raw system config, or ``[]``.

        Read from ``config_manager.system_config`` rather than from
        ``system.star``, so a NAME resolves at construction and at stage 1 --
        before the Star component exists.  Empty when the config manager has
        no system config (a test stub), which ``resolve_star_ref`` treats as
        "indices only".
        """
        cfg = getattr(self.config_manager, "system_config", None) or {}
        entries = cfg.get("star")
        if not isinstance(entries, list):
            return []
        return [
            str(e.get("name", i))
            for i, e in enumerate(entries)
            if isinstance(e, dict)
        ]

    def resolve_star_ndx(self, ref, where, default=0):
        """This component's ``star_ndx``-style reference as a star index.

        ``None`` (the key absent) takes ``default``; anything else goes
        through :func:`resolve_star_ref`, so the name form the schemas
        advertise actually works.
        """
        if ref is None:
            return int(default)
        return resolve_star_ref(ref, self.declared_star_names(), where)

    def add_parameter(self, model, param_name, system, context_nodes=None):
        context_nodes = context_nodes or {}
        # Reported (role 3) selections park here until finalize_reported; keyed
        # per parameter name, so a second build_model on one System starts clean
        # (the GUI builds more than once).
        if not hasattr(self, "_pending_reported"):
            self._pending_reported = {}

        # 0. Prevent double-building nodes -- within THIS build.  A node
        # stamped for an earlier model is stale and must be rebuilt; see
        # _parameter_is_current (review 3.14.12).
        if self._parameter_is_current(self, param_name, model):
            return getattr(self, param_name).value

        if not hasattr(self, "manifest"):
            raise ValueError(
                f"[{self.prefix}] has no manifest. Did register_parameters run?"
            )
        if param_name not in self.manifest:
            raise KeyError(
                f"[{self.prefix}] System requested '{param_name}', but it is not in the manifest."
            )

        # manifest.py is the single interpreter of the manifest vocabulary --
        # the same one graph.determine_pymc_build_order (the build order) and
        # System.derived_params read, so the build order can never disagree
        # with what gets built.  `entry.options` is already a copy, so the
        # pops below cannot mutate the live manifest.
        entry = interpret_manifest_entry(self.manifest[param_name])
        options = dict(entry.options)

        # Manifest entries may override the shape for parameters that are not
        # one-per-element (e.g. one (s, alpha) per lens companion), and the
        # per-element names used for user-param resolution and display labels
        # (e.g. per-source lens params named after the source stars).
        shape = tuple(options.pop("shape", None) or (self.n_elements,))
        names = options.pop("names", None) or getattr(self, "names", None)

        # Component-computed per-element defaults ("overrides") are layered in
        # BELOW the user's params file, unlike the remaining manifest options,
        # which are merged over the resolved config and so win outright.  Use
        # this whenever a component derives a value from its own configuration
        # or data but the user must still be able to override it (see
        # Instrument._register_gp, which pins the GP hyperparameters of files
        # that did not request a GP).  Per-element arrays may carry NaN for
        # "leave this element alone".
        overrides = options.pop("overrides", None)

        # 1. Grab configuration properties agnostically
        cfg = self.config_manager.resolve(
            self.prefix,
            param_name,
            shape=shape,
            names=names,
            internal_overrides=overrides,
        )

        expressions_dict = cfg.pop("expressions", {})
        where = f"{self.prefix}.{param_name}"
        n_elements = int(np.prod(shape)) if shape else 1
        selections = entry.expression_configs(
            expressions_dict, n_elements=n_elements, where=where
        )
        expression = None
        element_expressions = None

        # --- AGNOSTIC CONDITIONAL WIRE-UP ---
        # Only parse dependencies if an expression block actively exists for
        # this parameter role.  One selection covering every element is the
        # historical whole-vector case and keeps its own path; several
        # selections (or one covering a subset) mean the instances chose
        # different parameterizations, and each gets its own closure.
        if selections:
            options.pop("deps", None)
            uniform = (
                len(selections) == 1
                and selections[0].mask is None
                and not selections[0].output_only
            )
            built = []
            for sel in selections:
                if sel.output_only:
                    # REPORTED elements (role 3) defer their WHOLE wiring, not
                    # just the patch.  Resolving their dependencies here would
                    # recurse: the dep is a parameter that, on other elements,
                    # is derived from this one, and this parameter is not yet
                    # bound on the component (`setattr` happens below), so
                    # add_parameter's already-built guard could not stop it.
                    # Only the MASK is needed now, so build_pymc can mark the
                    # role and hold back the Deterministic; the expression is
                    # wired in finalize_reported, once every parameter exists.
                    built.append(
                        ElementExpression(
                            mask=sel.mask, expr=None, output_only=True
                        )
                    )
                    self._pending_reported.setdefault(param_name, []).append(
                        (entry, sel, where)
                    )
                    continue
                built.append(
                    self._element_expression(
                        model, system, context_nodes, entry, sel, where
                    )
                )
            if uniform:
                expression = built[0].expr
            else:
                element_expressions = built

        # 2b. Wire up user-defined parameter links (initval/mu/lower/upper
        # expressions from the params file referencing other parameters).
        element_links = self._wire_user_links(
            model, param_name, system, cfg, expression
        )

        # 3. Create Parameter Node
        full_params = {**cfg, **options}
        param_obj = Parameter(
            label=f"{self.prefix}.{param_name}",
            names=names,
            expression=expression,
            element_expressions=element_expressions,
            element_links=element_links,
            user_params=self.config_manager.user_params,
            source_file=getattr(self.config_manager, "param_file", None),
            # Bound method: (component, param, element=i) -> "user" | "data" |
            # "solved" | "default".  build_pymc quotes it when it refuses an
            # out-of-bounds start, so the message blames the right input.
            initval_source=getattr(
                self.config_manager, "initval_source", None
            ),
            **full_params,
        )

        setattr(self, param_name, param_obj)
        # Stamp which model this node belongs to, so the guard above can tell
        # "already built in this build" from "left over from the last one".
        if model is not None:
            if not hasattr(self, "_built_for_model"):
                self._built_for_model = {}
            self._built_for_model[param_name] = weakref.ref(model)
        return param_obj.build_pymc()

    def _element_expression(
        self, model, system, context_nodes, entry, sel, where
    ):
        """Wire ONE ``expressions:`` block into an :class:`ElementExpression`.

        The whole-vector case (``sel.mask is None``, or a mask covering every
        element) builds exactly the closure this method's predecessor built:
        ``lambda: func(*dep_nodes)`` over the full dependency vectors.

        A mask covering a SUBSET slices the dependencies down to those elements
        first, so the instances that did not choose this parameterization never
        enter the expression at all.  That is not an optimization: their values
        are bookkeeping pins that the other parameterization's physics makes no
        promise about, and an expression evaluated there can legitimately be
        NaN (sqrt of a negative eccentricity).  Keeping them out is the only way
        the mixed vector's gradient is guaranteed clean, since a NaN sitting in
        a discarded slot still reaches the input's gradient as 0*NaN.

        Slicing is only safe where the dependency is element-ALIGNED, so an
        unproven alignment raises rather than guessing -- see
        ``_resolve_dep_node``.  Whether the sliced expression really equals the
        full one on those elements (i.e. whether the physics is elementwise at
        all) is verified numerically at the start point, once the model exists:
        ``System.verify_element_slices``.
        """
        func_name = sel.config.get("func_name")
        if func_name not in PHYSICS_REGISTRY:
            raise NotImplementedError(
                f"[{where}] Function '{func_name}' not in PHYSICS_REGISTRY."
            )
        func = PHYSICS_REGISTRY[func_name]

        n_elements = np.size(sel.mask) if sel.mask is not None else None
        deps = [
            self._resolve_dep_node(
                model, system, context_nodes, d, where, n_elements
            )
            for d in entry.dep_names(sel.config)
        ]
        nodes = [node for _d, node, _aligned in deps]
        own_refs = [n for n in nodes if isinstance(n, OwnPrePatchRef)]

        def full_expr(nodes=nodes, prepatch=None):
            resolved = [
                prepatch[pt.as_tensor_variable(n.idx.astype("int32"))]
                if isinstance(n, OwnPrePatchRef)
                else n
                for n in nodes
            ]
            return func(*resolved)

        if own_refs:
            full_expr._own_ref_idx = np.concatenate([r.idx for r in own_refs])

        if sel.mask is None or bool(np.all(sel.mask)):
            return ElementExpression(
                mask=True if sel.mask is None else sel.mask,
                expr=full_expr,
                output_only=sel.output_only,
            )

        idx = np.nonzero(sel.mask)[0]
        sliced = []
        for d, node, aligned in deps:
            if isinstance(node, OwnPrePatchRef):
                # The sentinel's map has one entry per element of THIS
                # parameter (enforced below via `aligned`); slice the
                # INDEX ARRAY to the selected elements -- the tensor it
                # points into does not exist yet.
                if not aligned:
                    raise ValueError(
                        f"[{where}] same-parameter dep '{d}' must carry "
                        f"an index map with one entry per element of "
                        f"this parameter."
                    )
                sliced.append(OwnPrePatchRef(node.idx[idx]))
                continue
            if getattr(node, "ndim", 0) == 0:
                sliced.append(node)  # a scalar applies to every element
                continue
            if not aligned:
                raise ValueError(
                    f"[{where}] the expression '{func_name}' supplies only "
                    f"element(s) {idx.tolist()} of this parameter, so its "
                    f"dependencies are sliced to those elements -- but "
                    f"dependency '{d}' cannot be PROVEN to be element-aligned, "
                    f"and slicing a vector that is indexed by something else "
                    f"pairs the wrong instances silently (a different star's "
                    f"mass into this orbit's Kepler relation). Fix: give the "
                    f"dep an explicit index map ('{d}[<map_name>]', built in "
                    f"build_maps with one entry per element of this "
                    f"parameter), or -- for a dep injected as a context node "
                    f"-- name it in the component's 'aligned_context_deps' to "
                    f"declare that it already has one entry per element."
                )
            sliced.append(node[pt.as_tensor_variable(idx.astype("int32"))])

        own_sliced = [s for s in sliced if isinstance(s, OwnPrePatchRef)]

        def sliced_expr(sliced=sliced, prepatch=None):
            resolved = [
                prepatch[pt.as_tensor_variable(s.idx.astype("int32"))]
                if isinstance(s, OwnPrePatchRef)
                else s
                for s in sliced
            ]
            return func(*resolved)

        if own_sliced:
            sliced_expr._own_ref_idx = np.concatenate(
                [s.idx for s in own_sliced]
            )

        register = getattr(system, "register_element_slice_check", None)
        if callable(register) and not own_refs:
            # An own-ref expression cannot be evaluated without the
            # parameter's own pre-patch tensor, which the slice checker
            # does not have; correctness there is covered by the
            # sampled-elements-only guard in build_pymc instead.
            register(where, func_name, idx, sliced_expr, full_expr)
        return ElementExpression(
            mask=sel.mask,
            expr=sliced_expr,
            output_only=sel.output_only,
            sliced=True,
        )

    def _resolve_dep_node(
        self, model, system, context_nodes, d, where, n_elements=None
    ):
        """``(dep name, node, is_element_aligned)`` for one dependency string.

        The dependency vocabulary is unchanged: a context node injected by the
        component, a cross-component path with an optional index map
        (``star.mass[lens_map]``), or a bare local parameter name.

        ``is_element_aligned`` answers "does entry i of this node belong to
        element i of the parameter being built?" and is only ever True when the
        answer can be PROVEN from how the dep resolved -- a local parameter of
        the same length, a map with one entry per element, or a context node the
        component declared aligned.  It is False for a bare cross-component
        vector, whose entries are indexed by the OTHER component's elements and
        line up only by coincidence.  Only the per-element slicing path consults
        it; the whole-vector path never slices and so never cares.
        """
        if d in context_nodes:
            return (
                d,
                context_nodes[d],
                d in getattr(self, "aligned_context_deps", frozenset()),
            )

        if "." not in d:
            # Local tracking recursive lookup
            if not self._parameter_is_current(self, d, model):
                self.add_parameter(model, d, system, context_nodes)
            local = getattr(self, d)
            aligned = n_elements is not None and local._n_elements() == int(
                n_elements
            )
            return d, local.value, aligned

        # Parse universal cross-component strings: "star.density[star_map]"
        custom_slice = None
        if "[" in d and d.endswith("]"):
            path_part, slice_part = d.split("[", 1)
            custom_slice = slice_part.rstrip("]")
            d_lookup = path_part
        else:
            d_lookup = d

        ext_comp_name, ext_param_name = d_lookup.split(".", 1)
        ext_comp = getattr(system, ext_comp_name, None)
        if not ext_comp:
            raise ValueError(
                f"[{where}] Component '{ext_comp_name}' is not active."
            )

        # A dep naming the parameter BEING BUILT (fitmurel's
        # pm[lens] <- pm[source] + mu_rel) cannot recurse into
        # add_parameter -- the tensor does not exist yet.  Return an
        # OwnPrePatchRef sentinel; _patch_elements substitutes the
        # pre-patch tensor's elements at patch time, and build_pymc
        # refuses any referenced element that is not SAMPLED (only
        # sampled elements are final pre-patch).  `where` is
        # f"{prefix}.{param_name}" at both call sites, which is what
        # identifies the parameter under construction.
        building = None
        prefix_dot = f"{self.prefix}."
        if where.startswith(prefix_dot):
            building = where[len(prefix_dot) :]
        if ext_comp is self and ext_param_name == building:
            if custom_slice is None:
                raise ValueError(
                    f"[{where}] same-parameter element dep '{d}' needs "
                    f"an explicit index map ('{d}[<map_name>]', one "
                    f"entry per element of this parameter): without "
                    f"one there is no way to say WHICH of its own "
                    f"elements the expression reads."
                )
            idx = np.asarray(getattr(self, custom_slice), dtype=int)
            aligned = n_elements is not None and idx.size == int(n_elements)
            return d, OwnPrePatchRef(idx), aligned

        # Ensure the dependency node is built lazily on demand
        if not self._parameter_is_current(ext_comp, ext_param_name, model):
            ext_comp.add_parameter(
                model, ext_param_name, system, context_nodes
            )

        ext_param = getattr(ext_comp, ext_param_name)

        # Dynamically slice via requested map name or component fallback name
        map_name = custom_slice or f"{ext_comp_name}_map"
        map_attr = f"{map_name}_tensor"
        if hasattr(self, map_attr):
            map_tensor = getattr(self, map_attr)
            raw_map = getattr(self, map_name, None)
            aligned = (
                n_elements is not None
                and raw_map is not None
                and np.size(raw_map) == int(n_elements)
            )
            return d, ext_param.value[map_tensor], aligned
        if custom_slice:
            # A dep that NAMES its map ("star.mass[lens_map]")
            # asked for specific elements.  Falling back to the
            # unsliced vector does not mean "no slice" -- where
            # the lengths happen to match it broadcasts silently
            # and pairs the wrong bodies (a different star's mass
            # into a lens's theta_E).  The unnamed
            # "{comp}_map_tensor" convenience path keeps its
            # fallback.
            raise AttributeError(
                f"[{where}] dependency '{d}' "
                f"names the index map '{custom_slice}', but "
                f"{self.prefix} has no '{map_attr}'.  Build it in "
                f"build_maps() (build_tensor_maps converts "
                f"'{custom_slice}' automatically) or drop the "
                f"[...] from the dep."
            )
        return d, ext_param.value, False

    def _wire_user_links(self, model, param_name, system, cfg, expression):
        """
        Translate the ConfigManager's user-defined links targeting this
        parameter into per-element PyTensor closures for Parameter.build_pymc.

        Each closure receives this parameter's own physical vector (internal
        units) so same-parameter references (star.A.age -> star.B.age) resolve
        without leaving the node; external references are materialized lazily
        through add_parameter, exactly like physics expression dependencies.
        Unit convention: referenced parameters contribute their values in
        their own user units; the result is taken in the target's user unit.
        """
        cm = self.config_manager
        get_links = getattr(cm, "get_element_links", None)
        if get_links is None:
            return None
        links = get_links(self.prefix, param_name)
        if not links:
            return None

        from ..linking import sympy_to_pytensor

        sigma_arr = cfg.get("sigma")
        out = {}
        for fld, per_elem in links.items():
            for idx, plink in per_elem.items():
                # Classify the runtime role of this link.
                if fld == "initval":
                    s = sigma_arr[idx] if sigma_arr is not None else np.nan
                    if s == 0:
                        key = "hard"  # derived element: tracks the expression exactly
                    elif s > 0:
                        key = "mu"  # soft link: Gaussian penalty on the difference
                    else:
                        continue  # initialization-only; solver already applied it
                elif fld == "mu":
                    key = "mu"
                elif fld in ("lower", "upper"):
                    key = fld
                else:
                    continue  # sigma / init_scale: static snapshots

                if expression is not None and key != "mu":
                    raise ValueError(
                        f"[{self.prefix}.{param_name}] link '{plink.expr_str}' targets "
                        f"field '{fld}', but this parameter is derived from a physics "
                        f"expression; only soft (mu) links are supported there."
                    )

                ext_vals = {}  # dep path -> tensor in the dep's USER units
                self_refs = {}  # dep path -> (element index, user->internal factor)
                for dep in plink.dep_paths:
                    dparts = dep.split(".")
                    dcomp, didx, dparam = dparts[0], int(dparts[1]), dparts[2]
                    dfactor = cm.get_conversion_factor(
                        dcomp, dparam, full_path=dep
                    )
                    if dcomp == self.prefix and dparam == param_name:
                        self_refs[dep] = (didx, dfactor)
                        continue
                    comp = (
                        self
                        if dcomp == self.prefix
                        else getattr(system, dcomp, None)
                    )
                    if comp is None:
                        raise ValueError(
                            f"[{self.prefix}.{param_name}] link '{plink.expr_str}' "
                            f"references component '{dcomp}', which is not active."
                        )
                    if not self._parameter_is_current(comp, dparam, model):
                        comp.add_parameter(model, dparam, system)
                    node = getattr(comp, dparam).value
                    if getattr(node, "ndim", 0) >= 1:
                        node = node[didx]
                    elif didx != 0:
                        raise ValueError(
                            f"[{self.prefix}.{param_name}] link '{plink.expr_str}': "
                            f"'{dep}' indexes element {didx} of a scalar parameter."
                        )
                    ext_vals[dep] = node / dfactor if dfactor != 1.0 else node

                tfactor = cm.get_conversion_factor(
                    self.prefix,
                    param_name,
                    full_path=f"{self.prefix}.{idx}.{param_name}",
                )

                def make_fn(
                    plink=plink,
                    ext_vals=ext_vals,
                    self_refs=self_refs,
                    tfactor=tfactor,
                ):
                    def fn(phys_internal):
                        vals = dict(ext_vals)
                        for dep, (j, f) in self_refs.items():
                            v = phys_internal[j]
                            vals[dep] = v / f if f != 1.0 else v
                        user_val = sympy_to_pytensor(plink.expr, vals)
                        return (
                            user_val * tfactor if tfactor != 1.0 else user_val
                        )

                    return fn

                out.setdefault(key, {})[idx] = {
                    "fn": make_fn(),
                    "intra_deps": {j for (j, _) in self_refs.values()},
                }

        return out or None

    @abstractmethod
    def build_likelihood(self, model, system):
        """
        Stage 7: The Objective Function.
        Construct the PyMC Likelihoods (`pm.Normal`, etc.) or custom `pm.Potential`
        penalties that constrain the model against data.
        """
        pass

    def compile_plotters(self, model, system):
        """
        Compile fast PyTensor functions for plotting.
        Translates PyTensor graphs into numpy functions to ensure consistency
        between the likelihood calculation and the final figures.
        """
        pass

    def plot(self, system, points, filename_prefix="debug"):
        """
        Plot the model and data. Called twice:
          - Pre-flight: To visually verify the initialization logic.
          - Post-flight: To generate publication-quality posterior models.
        """
        pass

    def plot_data(self, system, point=None):
        """
        Stage: GUI plot description (the data behind plot()).

        Return a list of exozippy.chart.Chart objects -- the arrays
        and labels a browser GUI needs to draw pan/zoomable charts and
        re-render model curves when parameter sliders move. This is the
        data-only counterpart to plot(), which renders matplotlib figures.

        Semantics
        ---------
        point is None
            Return data-only specs (observations, no model curves). These
            are usable right after load_data()/prepare(), BEFORE any PyMC
            model or compiled plotter exists -- a raw file preview.
        point is a start/posterior point dict
            Include model traces evaluated at that point, reusing the
            functions compiled by compile_plotters() (no physics is
            duplicated here). Requires build_model() to have run.

        The default returns []; components that own observational data
        override it. See chart.Chart for the payload contract.
        """
        return []

    def _point_to_plot_params(self, point, system):
        """
        Marshal a point dict into the positional argument list the
        compiled plotter functions expect (one entry per
        system.plot_params, scalars squeezed, vectors kept 1-D).

        This is the single source of truth shared by the matplotlib
        plot() path and the GUI plot_data() path, so both feed the
        compiled functions the exact same values.
        """
        values = []
        for p in system.plot_params:
            val = np.asarray(point.get(p.label, p.initval), dtype=np.float64)
            if getattr(p.value, "ndim", 0) == 0:
                values.append(float(np.squeeze(val)))
            else:
                values.append(np.atleast_1d(val))
        return values

    def _model_trace_param_deps(self, node, system):
        """
        Sampled-parameter labels (a subset of system.plot_params) that a
        symbolic model-trace node depends on, found by walking the
        pytensor graph. Used to populate Chart.param_deps so a GUI can
        highlight the charts a moved slider affects. Returns [] when the
        node or plot_params are unavailable (e.g. data-only mode).
        """
        if node is None or not hasattr(system, "plot_params"):
            return []
        try:
            # Moved from pytensor.graph.basic in newer pytensor releases.
            from pytensor.graph.traversal import ancestors
        except ImportError:
            try:
                from pytensor.graph.basic import ancestors
            except Exception:
                return []
        wanted = {id(p.value): p.label for p in system.plot_params}
        deps = []
        for anc in ancestors([node]):
            label = wanted.get(id(anc))
            if label is not None and label not in deps:
                deps.append(label)
        return deps

    def plot_corner(self, idata, filename_prefix="debug"):
        """Optional: draw a component-specific posterior corner plot.

        Called once, after sampling, when the full posterior (idata) is
        available -- unlike plot(), which also runs pre-flight on a single
        point where a corner plot would be meaningless. Default: no-op;
        override in components that want one (see mulensing.Lens).
        """
        pass

    def sampler_requirements(self):
        """Return sampler constraints imposed by this component.

        Returns a dict with optional keys:
          'incompatible' : set of method names that cannot be used
          'recommended'  : preferred method name (str)
          'reason'       : human-readable explanation for warnings (str)

        The default implementation returns no constraints.  Override when a
        component uses non-differentiable Ops (e.g. MulensModel) that are
        incompatible with gradient-based samplers.
        """
        return {}
