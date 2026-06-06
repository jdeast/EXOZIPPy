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
            options = comp.manifest[param_name] or {}
            expr_key = options if isinstance(options, str) else options.get("expr_key", "default")
            expressions_dict = cfg.get("expressions", {})

            if expr_key in expressions_dict:
                dep_names = expressions_dict[expr_key].get("deps", [])
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