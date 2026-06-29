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
        for param_name in getattr(comp, 'manifest', {}):
            global_key = f"{comp_name}.{param_name}"
            forward_graph[global_key] = set()

    # 2. Parse forward expressions to populate graph edges (Child -> Parents)
    for comp_name, comp in active_components.items():
        for param_name in getattr(comp, 'manifest', {}):
            global_key = f"{comp_name}.{param_name}"

            cfg = config_manager.resolve(comp.prefix, param_name, shape=(comp.n_elements,))
            raw = comp.manifest[param_name]
            if isinstance(raw, str):
                expr_key = raw          # e.g. "default"
            elif isinstance(raw, dict):
                expr_key = raw.get("expr_key")  # explicit key or None
            else:
                expr_key = None         # None → free parameter, no expression
            # Fall back to "default" only when the manifest explicitly requested it
            # via the "default" string shorthand.  A bare None means free parameter.
            if expr_key is None:
                expr_key = "default" if raw is not None else None
            expressions_dict = cfg.get("expressions", {})

            if expr_key is not None and expr_key in expressions_dict:
                manifest_deps = raw.get("deps") if isinstance(raw, dict) else None
                dep_names = manifest_deps if manifest_deps is not None else expressions_dict[expr_key].get("deps", [])
                for d in dep_names:
                    if "." in d:
                        # Strip off any bracket indicators to get the raw structural key (e.g., "star.mass")
                        clean_dep = d.split("[")[0] if "[" in d else d
                        forward_graph[global_key].add(clean_dep)
                    else:
                        forward_graph[global_key].add(f"{comp_name}.{d}")

    # 3. Validate that all dependencies are actually registered nodes
    for node, deps in forward_graph.items():
        for d in deps:
            if d not in forward_graph:
                # This makes the error helpful rather than a cryptic KeyError during sort
                raise ValueError(f"Dependency Error: {node} depends on {d}, which is not defined in any manifest.")

    # 4. Sort agnostically
    sorter = graphlib.TopologicalSorter(forward_graph)
    try:
        return list(sorter.static_order())
    except graphlib.CycleError as e:
        raise ValueError(f"Circular reference detected in forward defaults.yaml graph: {e}")