from .parameter import Parameter
from ..physics_registry import PHYSICS_REGISTRY
from abc import ABC, abstractmethod
import numpy as np

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

        # Enforce unique names
        if len(set(self.names)) != len(self.names):
            raise ValueError(
                f"Duplicate names found in {self.__class__.__name__} configuration: {self.names}. All component names must be unique.")

    @property
    @abstractmethod
    def prefix(self):
        """Naming prefix for the model (e.g., 'star', 'planet')"""
        pass

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
    def add_parameter(self, p_name, config_manager, shape=(), expr_key=None, context_nodes=None,
                      **kwargs):
        context_nodes = context_nodes or {}

        # 1. INTERCEPT NUMERICAL HEURISTICS
        # Pull physics bounds out of kwargs so they don't clobber the resolved config!
        physics_keys = ["initval", "init_scale", "lower", "upper", "mu", "sigma", "gaussian_width"]
        internal_overrides = {}
        filtered_kwargs = {}

        for k, v in kwargs.items():
            if k in physics_keys:
                internal_overrides[k] = v
            else:
                filtered_kwargs[k] = v

        # 2. PASS TO CONFIG CAGE
        # Hand the names and the intercepted heuristics down to the config manager
        cfg = config_manager.resolve(
            self.prefix, p_name, shape=shape,
            internal_overrides=internal_overrides if internal_overrides else None,
            names=getattr(self, 'names', None)
        )

        if expr_key is None:
            expr_key = filtered_kwargs.pop("expr_key", None)

        expressions_dict = cfg.pop("expressions", {})
        expression = None

        if expr_key:
            # 1. Check if we even HAVE an expressions dictionary
            if not expressions_dict:
                raise KeyError(
                    f"[{self.prefix}.{p_name}] Requested expression key '{expr_key}', but no "
                    f"'expressions' were defined for this parameter in defaults.yaml. "
                    f"Did you forget to add this parameter to the component's defaults.yaml?"
                )

            # 2. Check if the specific key exists in that dictionary
            expr_cfg = expressions_dict.get(expr_key)
            if expr_cfg is None:
                raise ValueError(
                    f"[{self.prefix}.{p_name}] Requested expression key '{expr_key}' "
                    f"not found in expressions dictionary for this component."
                )

            # 3. Check for the Registry (The "Incomplete physics.py" check)
            func_name = expr_cfg.get("func_name")
            if func_name not in PHYSICS_REGISTRY:
                available = ", ".join(PHYSICS_REGISTRY.keys()) if PHYSICS_REGISTRY else "EMPTY"
                raise NotImplementedError(
                    f"[{self.prefix}.{p_name}] Physics function '{func_name}' is not in PHYSICS_REGISTRY. "
                    f"Verify it's decorated with @register_physics in physics.py. "
                    f"Available: {available}"
                )

            func = PHYSICS_REGISTRY[expr_cfg["func_name"]]
            dep_names = expr_cfg.get("deps", [])

            dep_nodes = []
            numeric_deps = []

            for d in dep_names:
                if d in context_nodes:
                    node = context_nodes[d]
                    dep_nodes.append(node)
                    try:
                        numeric_deps.append(node.eval())
                    except:
                        numeric_deps.append(0.0)
                else:
                    param = getattr(self, d)
                    dep_nodes.append(param.value)
                    numeric_deps.append(param.initval)

            # Calculate the numeric init so downstream parameters aren't hitting None
            try:
                # 1. Use user-provided initval if it exists
                # 2. Otherwise, calculate it from the physics function
                calculated_init = func(*numeric_deps)

                if internal_overrides.get('initval') is None and cfg.get('initval') is None:
                    filtered_kwargs['initval'] = calculated_init

                # We also need to store this on the expression object or config
                # so the Parameter class doesn't override it with None later
                cfg['initval'] = filtered_kwargs.get('initval')

            except Exception as e:
                # Fallback to a safe numeric array instead of None
                n_elements = np.prod(shape).astype(int) if shape != () else 1
                filtered_kwargs['initval'] = np.zeros(n_elements)

            expression = lambda: func(*dep_nodes)

        # 3. MERGE METADATA
        # Now it is safe to merge filtered_kwargs (units, latex names, etc.)
        full_params = {**cfg, **filtered_kwargs}

        param_obj = Parameter(
            label=f"{self.prefix}.{p_name}",
            names=getattr(self, 'names', None),
            expression=expression,
            user_params=self.config_manager.user_params,
            **full_params
        )

        setattr(self, p_name, param_obj)
        return param_obj.build_pymc()

    # Make sure build_pars_from_dict can pass it along!
    def build_pars_from_dict(self, par_dict, shape, context_nodes=None):
        for p_name, options in par_dict.items():
            if not options:
                self.add_parameter(p_name, self.config_manager, shape=shape,
                                   context_nodes=context_nodes)
            elif isinstance(options, str):
                self.add_parameter(p_name, self.config_manager, shape=shape, expr_key=options,
                                   context_nodes=context_nodes)
            elif isinstance(options, dict):
                self.add_parameter(p_name, self.config_manager, shape=shape,
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