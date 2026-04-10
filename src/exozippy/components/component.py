from .parameter import Parameter

from ..physics_registry import PHYSICS_REGISTRY
from abc import ABC, abstractmethod

class Component:
    def __init__(self, component_config, config_manager):
        """
        Standardized constructor for ALL components.
        """
        self.config = component_config
        self.config_manager = config_manager

        # Determine how many of this thing we are building
        self.n_elements = len(self.config)

        # Grab names for labeling PyMC nodes
        self.names = [c.get("name", f"{i}") for i, c in enumerate(self.config)]

    """ building the model takes 4 distinct steps, all of which are required but some may be trivially empty:
    1) build_parameters - this defines the base parameters, both sampled and derived
    2) load_data - this loads data into the component (e.g., RV data)
    3) build_dependent_parameters - this defines additional parameters that depend on other components or have data-driven initializations, constraints, or definitions.
    4) build_likelihood - this builds the likelihood of the component
    """
    @abstractmethod
    def build_parameters(self, model):
        """
        Step 1: Define the base parameters, both sampled and derived
        """
        pass

    @abstractmethod
    def load_data(self):
        """
        Step 2: load any data required for this class (pass if none)
        """
        pass

    @abstractmethod
    def build_map(self, system):
        """
        Step 3: Define the indexing relationship between components.
        Converts YAML configuration indices into PyTensor tensor variables.
        """
        pass

    @abstractmethod
    def build_dependent_parameters(self, model, system):
        """
        Step 4: Define additional parameters that depend on other components or have data-driven initializations
        e.g. gamma=mean(RV), constraints (user errors => jitter lower bound), or definitions (planet.K needs star.mass)
        """
        pass

    @abstractmethod
    def build_likelihood(self, model, system):
        """
        Step 5: Build the likelihood of the component
        """
        pass

    @abstractmethod
    def compile_plotters(self, model, system):
        """
        Step 6 (Optional): Compile fast PyTensor functions for plotting.
        Compile the pytensor code that builds the model as a numpy function to use in plotting
        That reduces effort and ensures consistency between the likelihood calculation and the figure
        """
        pass

    @abstractmethod
    def plot(self, system, points, filename_prefix="debug"):
        """
        Plot the model and data. This will be called
          - at the beginning to verify the initialization
          - at the end to make publication quality plots
        """
        pass

    # this method adds a Parameter to the model
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

    def _is_sampling_param(self, attr):
        """Helper to identify parameters that need to be passed to the compiled function."""
        from .parameter import Parameter
        return isinstance(attr, Parameter) and attr.expression is None

    def get_parameters(self, sampling_only=False):
        """
        Returns all Parameter objects belonging to this component.
        If sampling_only=True, filters for parameters without expressions.
        """
        params = []
        # We look through __dict__ to preserve the order they were built in
        for attr in self.__dict__.values():
            if isinstance(attr, Parameter):
                if sampling_only:
                    if attr.expression is None:
                        params.append(attr)
                else:
                    params.append(attr)
        return params