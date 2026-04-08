from .parameter import Parameter

from ..physics import PHYSICS_REGISTRY

class Component:
    def __init__(self, component_config, config_manager):
        """
        Standardized constructor for ALL components.
        """
        self.config = component_config  # The list from kelt4.yaml (e.g. [{'name': 'Kelt-4A'}])
        self.config_manager = config_manager  # The resolver for exozippy_params.yaml

        # Determine how many of this thing we are building
        self.n_elements = len(self.config)

        # Grab names for labeling PyMC nodes
        self.names = [c.get("name", f"{i}") for i, c in enumerate(self.config)]

    def add_parameter(self, p_name, config_manager, shape=(), prefix=None, expr_key=None, context_nodes=None,
                      **kwargs):
        context_nodes = context_nodes or {}
        cfg = config_manager.resolve(prefix, p_name, shape=shape)

        if expr_key is None:
            expr_key = kwargs.pop("expr_key", None)

        expressions_dict = cfg.pop("expressions", {})
        expression = None

        if expr_key:
            expr_cfg = expressions_dict.get(expr_key)
            func = PHYSICS_REGISTRY[expr_cfg["func_name"]]
            dep_names = expr_cfg.get("deps", [])

            dep_nodes = []
            numeric_deps = []

            for d in dep_names:
                if d in context_nodes:
                    node = context_nodes[d]
                    dep_nodes.append(node)
                    # context_nodes are usually raw tensors, try to eval them
                    try:
                        numeric_deps.append(node.eval())
                    except:
                        numeric_deps.append(0.0)
                else:
                    param = getattr(self, d)
                    dep_nodes.append(param.value)
                    # THIS IS THE KEY: grab the numeric initval from the parent Parameter
                    numeric_deps.append(param.initval)

            # Pre-calculate the numeric init for the Auditor/Build process
            try:
                # Only overwrite if not explicitly provided in kwargs
                if 'initval' not in kwargs:
                    kwargs['initval'] = func(*numeric_deps)
            except Exception as e:
                # Fallback to a zero-array of correct shape if math fails
                n_elements = np.prod(shape).astype(int) if shape != () else 1
                kwargs['initval'] = np.zeros(n_elements)

            expression = lambda: func(*dep_nodes)

        full_params = {**cfg, **kwargs}

        param_obj = Parameter(
            label=f"{prefix}.{p_name}",
            names=getattr(self, 'names', None),
            expression=expression,
            user_params=self.config_manager.user_params,
            **full_params
        )

        setattr(self, p_name, param_obj)
        return param_obj.build_pymc()

    # Make sure build_pars_from_dict can pass it along!
    def build_pars_from_dict(self, par_dict, shape, prefix=None, context_nodes=None):
        for p_name, options in par_dict.items():
            if not options:
                self.add_parameter(p_name, self.config_manager, shape=shape, prefix=prefix,
                                   context_nodes=context_nodes)
            elif isinstance(options, str):
                self.add_parameter(p_name, self.config_manager, shape=shape, expr_key=options, prefix=prefix,
                                   context_nodes=context_nodes)
            elif isinstance(options, dict):
                self.add_parameter(p_name, self.config_manager, shape=shape, prefix=prefix,
                                   context_nodes=context_nodes, **options)