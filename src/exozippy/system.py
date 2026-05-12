#import ipdb
import yaml
import numpy as np
import os

import pytensor
import pytensor.tensor as pt
import pymc as pm
import arviz as az

# local imports
from exozippy.components import Star, Orbit, Planet, RVInstrument
from exozippy.components.component import Component
from exozippy.components.parameter import Parameter
from exozippy.config import ConfigManager
from exozippy.components.star.star import Star
from exozippy.components.orbit.orbit import Orbit
from exozippy.components.planet.planet import Planet
from exozippy.components.rv_instrument.rv_instrument import RVInstrument
from exozippy.components.factory import discover_components

class System(Component):
    def __init__(self, config, user_params=None):

        self.config = config
        self.name = self.config.get("name","system")

        # load the user parameter file
        if user_params is not None:
            self.user_params = user_params
        else:
            user_params_file = self.config.get("parameter_file",None)
            if not os.path.exists(user_params_file):
                raise ValueError("The user must specify a valid parameter_file")
            with open(str(user_params_file), 'r') as f:
                self.user_params = yaml.safe_load(f)

        self.config_manager = ConfigManager(self.user_params)

        self.registry = discover_components()
        self.active_components = {}

        # 2. AGNOSTIC INSTANTIATION
        # We look at the YAML keys. If a key matches a registered Component, we build it.
        # Python >= 3.7 dicts preserve insertion order, so the user's YAML order
        # implicitly dictates the order in the output table.
        # note it also implicitly dictates the component processing order
        # but the model building recipe should be robust to build order
        # that might lead to mysterious bugs if that turns out to be false
        for key in self.config.keys():
            if key in self.registry:
                CompClass = self.registry[key]
                inst = CompClass(self.config[key], self.config_manager)
                self.active_components[key] = inst
                setattr(self, key, inst)

        print("Based on your config.yaml file, we are modeling the following components:")
        for key in self.active_components.keys(): print(f"{key} ({self.active_components[key].n_elements})")

        reserved_keys = {"run", "parameter_file", "prefix", "sampler", "name"}
        for key in self.config.keys():
            if key in self.registry:
                CompClass = self.registry[key]
                inst = CompClass(self.config[key], self.config_manager)
                self.active_components[key] = inst
                setattr(self, key, inst)
            elif key not in reserved_keys:
                print(
                    f"\033[93m" + f"!!!! WARNING: YAML key '{key}' does not match any registered component (Star, Planet, etc.) and will be ignored." + "\033[0m")

    def build_model(self):
        with pm.Model() as model:
            # 1. CORE PARAMETERS (The Foundations)
            for comp in self.active_components.values():
                comp.build_parameters(model)

            # 2. LOAD DATA (The Heuristics & Context)
            for comp in self.active_components.values():
                comp.load_data()

            # 3. Structural Mapping (PyTensor Graph Topology)
            for comp in self.active_components.values():
                comp.build_map(system=self)

            # --- BRIDGE: INITIALIZE MAPS ---
            for comp in self.active_components.values():
                comp.build_dependent_parameters(model, system=self)

            # 4. LIKELIHOOD (The Comparison)
            for comp in self.active_components.values():
                comp.build_likelihood(model, system=self)

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

    def get_internal_point(self, model, raw_point):
        """Evaluates graph deterministics for plotting/physics without user-unit conversion."""
        output_vars = model.free_RVs + model.deterministics

        eval_fn = pytensor.function(
            inputs=model.free_RVs,
            outputs=output_vars,
            on_unused_input='ignore'
        )

        # Pull the values in the exact order the function expects them
        input_values = [raw_point[v.name] for v in model.free_RVs]

        physical_values = eval_fn(*input_values)

        return {var.name: val for var, val in zip(output_vars, physical_values)}

    def get_physical_point(self, model, raw_point):
        output_vars = model.free_RVs + model.deterministics

        eval_fn = pytensor.function(
            inputs=model.free_RVs,
            outputs=output_vars,
            on_unused_input='ignore'
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
        #posterior = idata.posterior
        posterior = az.extract(idata)

        # Dynamically discover all components (Stars, Planets, Orbits, Instruments, etc.)
        for attr_name, comp in self.__dict__.items():
            if isinstance(comp, Component) and comp is not self:
                self._set_comp_posterior(comp, posterior)

    def _set_comp_posterior(self, component, posterior):
        for attr_name in dir(component):
            attr = getattr(component, attr_name)

            if isinstance(attr, Parameter):
                if attr.label in posterior:
                    # Case A: It was a named node in the graph
                    attr.posterior = posterior[attr.label]
                elif attr.expression is not None:
                    # Case B: It was 'Floating' math; calculate it now
                    attr.posterior = attr.generate_posterior(posterior)

            # Recurse to children (Stars, Planets, etc.)
            elif isinstance(attr, Component) and attr is not component:
                self._set_comp_posterior(attr, posterior)

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
                if attr_name.startswith('_'):
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
                        if isinstance(item, Component) and id(item) not in seen_components:
                            seen_components.add(id(item))
                            yield item
                            yield from crawl(item)

        yield from crawl(self)

    def get_mcmc_init(self, model):
        """
        Generalized initialization for the whitened parameters.
        Uses the agnostic parameter list to build metadata dictionaries.
        """
        transformed_inits = {}

        # 1. Map Unity Space (the sampler's world) to PyMC transformed values
        for rv, value_var in model.rvs_to_values.items():
            # Our 'Unity Start' is 0.0 in the raw space (N(0,1))
            unity_start = np.zeros(rv.shape.eval(), dtype=float)
            transform = model.rvs_to_transforms.get(rv)

            if transform is not None:
                # Forward the 0.0 through the interval/log math
                t_node = transform.forward(pt.as_tensor_variable(unity_start), *rv.owner.inputs)
                transformed_inits[value_var.name] = t_node.eval()
            else:
                # No transform, raw == value
                transformed_inits[value_var.name] = unity_start

        # 2. Extract Physical Metadata using the Master Parameter List
        # This now uses the simplified get_all_parameters() which relies on the generator
        all_params = self.get_all_parameters()

        # Filter for only the 'Free' parameters (those being sampled, no expression)
        sampling_params = [p for p in all_params if p.expression is None]

        ordered_inits = {p.label: p.initval for p in sampling_params}
        ordered_scales = {p.label: p.init_scale for p in sampling_params}

        # Calculate total dimensions for the NUTS step scaling
        total_dims = sum(np.size(val) for val in transformed_inits.values())

        # Return order: NUTS scales (all 1.0), physical scales, physical inits, transformed dict
        return np.ones(total_dims), ordered_scales, ordered_inits, transformed_inits

    def compile_plotter_functions(self, model):
        """
        Gathers the global sampling parameters so components know
        the exact input signature required for their compiled functions,
        then tells each component to compile its own plotters.
        """
        all_params = self.get_all_parameters()
        self.plot_params = [p for p in all_params if p.expression is None]

        # Delegate the actual compilation to the components
        for comp in self.active_components.values():
            comp.compile_plotters(model, system=self)
