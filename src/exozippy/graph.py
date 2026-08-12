import graphlib

"""
This builds a graph of the model and returns a topologically sorted list of parameters, ensuring that dependencies are built prior to things that depend on them.
This must contain no component-specific logic.
"""


def determine_pymc_build_order(active_components, config_manager):
    """
    Agnostically derives the forward topological build order for PyMC.
    Purely string-based. Zero component-level domain knowledge.
    """
    forward_graph = {}

    # 1. Initialize every manifest parameter across all components as a graph node
    for comp_name, comp in active_components.items():
        for param_name in getattr(comp, "manifest", {}):
            global_key = f"{comp_name}.{param_name}"
            forward_graph[global_key] = set()

    # 2. Parse forward expressions to populate graph edges (Child -> Parents)
    for comp_name, comp in active_components.items():
        for param_name in getattr(comp, "manifest", {}):
            global_key = f"{comp_name}.{param_name}"

            cfg = config_manager.resolve(
                comp.prefix, param_name, shape=(comp.n_elements,)
            )
            raw = comp.manifest[param_name]
            if isinstance(raw, str):
                expr_key = raw  # e.g. "default"
            elif isinstance(raw, dict):
                # A dict WITHOUT "expr_key" is a free parameter carrying only
                # options -- an "overrides" pin, a shape, a table note.  Same
                # rule Component.add_parameter (which needs a truthy expr_key
                # to build an expression) and System.derived_params use.  This
                # used to fall back to "default", which made graph.py the only
                # place that read such an entry as derived: harmless while no
                # pinned free parameter had an UNUSED `expressions:` block in
                # its defaults.yaml, and a hard "Dependency Error" the moment
                # one did (Band's linear-law u1, whose Kipping expression the
                # manifest deliberately ignores).  The fallback could only ever
                # add edges add_parameter does not use; it could never supply
                # a needed one, since a parameter it applied to is free.
                expr_key = raw.get("expr_key")  # explicit key or None
            else:
                expr_key = None  # None → free parameter, no expression
            expressions_dict = cfg.get("expressions", {})

            if expr_key is not None and expr_key in expressions_dict:
                manifest_deps = (
                    raw.get("deps") if isinstance(raw, dict) else None
                )
                dep_names = (
                    manifest_deps
                    if manifest_deps is not None
                    else expressions_dict[expr_key].get("deps", [])
                )
                # Deps a component declares in context_dep_names are
                # satisfied by context-node injection in its add_parameter
                # override (constants, not manifest parameters) -- they are
                # excluded from the build-order graph.
                context_deps = getattr(comp, "context_dep_names", frozenset())
                for d in dep_names:
                    if d in context_deps:
                        continue
                    if "." in d:
                        # Strip off any bracket indicators to get the raw structural key (e.g., "star.mass")
                        clean_dep = d.split("[")[0] if "[" in d else d
                        forward_graph[global_key].add(clean_dep)
                    else:
                        forward_graph[global_key].add(f"{comp_name}.{d}")

    # 2b. Add edges for user-defined parameter links (params-file expressions
    # referencing other parameters).  Same-parameter element links (star.A.age
    # -> star.B.age) are resolved inside build_pymc and would appear here as
    # self-loops, so they are skipped.
    for target, fields in (
        getattr(config_manager, "links", None) or {}
    ).items():
        tparts = target.split(".")
        tkey = f"{tparts[0]}.{tparts[-1]}"
        if tkey not in forward_graph:
            continue
        for plink in fields.values():
            for dep in plink.dep_paths:
                dparts = dep.split(".")
                dkey = f"{dparts[0]}.{dparts[-1]}"
                if dkey != tkey and dkey in forward_graph:
                    forward_graph[tkey].add(dkey)

    # 3. Validate that all dependencies are actually registered nodes
    for node, deps in forward_graph.items():
        for d in deps:
            if d not in forward_graph:
                # This makes the error helpful rather than a cryptic KeyError during sort
                raise ValueError(
                    f"Dependency Error: {node} depends on {d}, which is not defined in any manifest."
                )

    # 4. Sort agnostically
    sorter = graphlib.TopologicalSorter(forward_graph)
    try:
        return list(sorter.static_order())
    except graphlib.CycleError as e:
        raise ValueError(
            f"Circular reference detected in forward defaults.yaml graph: {e}"
        )
