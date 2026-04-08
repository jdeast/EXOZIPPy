import ipdb
import yaml
import numpy as np

import pytensor
import pytensor.tensor as pt
import pymc as pm
import arviz as az

from exozippy.components import Star, Orbit, Planet, RVInstrument
from .component import Component
from .parameter import Parameter
from ..config import ConfigManager

#from ..data.resolver import resolve_datasets
#from transit import Transit
#from astrometry import Astrometry
#from mulensing import mulensing


class StellarSystem(Component):
    def __init__(self, config):

        self.config = config
        self.name = self.config.get("name","planet")

        user_params_file = self.config.get("parameter_file",None)
        with open(str(user_params_file), 'r') as f:
            self.user_params = yaml.safe_load(f)

        self.config_manager = ConfigManager(self.user_params)

        # these are the objects we're modeling
        self.stars = Star(self.config.get("stars"), self.config_manager)
        self.orbits = Orbit(self.config.get("orbits"), self.config_manager)
        self.planets = Planet(self.config.get("planets"), self.config_manager)
        self.instruments = RVInstrument(self.config.get("rv").get("instruments"), self.config_manager)

        self.instruments.load_data()

        #self.stars = Star(self.config.get("stars"), self.user_params)
        #self.orbits = Orbit(self.config.get("orbits"),self.user_params)
        #self.planets = Planet(self.config.get("planets"),self.user_params)

        # data sets that constrain the above
        #self.instruments = RVInstrument(self.config.get("rv").get("instruments"),self.user_params)
        #self.transits = Transit(self.config.get("transits"),self.user_params)
        #self.astrometry = Astrometry(self.config.get("astrometry"),self.user_params)
        #self.mulensing = mulensing(self.config.get("mulensing"),self.user_params)

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

    def build_model(self):

        """
        Build the PyMC Graph.
        """
        with pm.Model() as model:

            # 1. first pass, build the foundational parameters
            self.stars.build_parameters()
            self.orbits.build_parameters()
            self.planets.build_parameters()
            self.instruments.build_parameters()

            self.orbit_map = pt.as_tensor_variable(
                np.array([p_cfg.get("orbit_ndx", 0) for p_cfg in self.planets.config])
            ).astype("int32")
            self.star_map = pt.as_tensor_variable(
                np.array([p_cfg.get("star_ndx", 0) for p_cfg in self.planets.config])
            ).astype("int32")
            self.instruments.inst_map_tensor = pt.as_tensor_variable(
                self.instruments.inst_map
            ).astype("int32")

            self.planets.build_dependent_parameters(stars=self.stars, orbits=self.orbits,
                                                    star_map=self.star_map, orbit_map=self.orbit_map)

            """ for vcve parameterization, this will be more complicated... punt for now
             
            # for transit-only fits (vcve parameterization), we must build the planet first and pass it to orbit
            # for other fits, we must build the orbit first and pass it to planet
            # 2. Build Planets/Orbits (Physics & Derived relations; requires Star and Orbit)
            for p in self.planets:
                host_star = self.stars[p.config.get(star_ndx,0)]

                if not p.fitvcve:
                    # 2. Build Orbits first (Timing/Geometry; no dependencies)
                    for o in self.orbits:
                        o.build_parameters(model)
                    planet_orbit = self.orbits[p.config.get('orbit_ndx', 0)]
                    p.build_parameters(model, host_star, planet_orbit)
                else:
                    pass
                    # planet_orbit = self.orbits[p.config.get('orbit_ndx', 0)]
                    # p.build_parameters(model, host_star, planet_orbit)
            """

            # third pass, build likelihoods
            self.instruments.build_likelihood(model, stars=self.stars, orbits=self.orbits, planets=self.planets,
                                                    star_map=self.star_map, orbit_map=self.orbit_map)

            """
            # 5. Build RV Instruments
            for t in self.transits:
                t.build_likelihood(model, self.planets, user_params)

            # 6. Build Astrometry Instruments
            for a in self.astrometry:
                a.build_likelihood(model)

            # 7. Build mulensing Instruments
            for m in self.mulensing:
                m.build_likelihood(model)
            """

            # now apply globally dependent parameters/constraints
            #self.build_dependent_parameters()

            self.compile_plotter_functions(model)

        return model

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

    """Global system-wide physical constraints."""
    def build_dependent_parameters(self):

        if len(self.planets) < 2: return

        # 1. Sort planets by semi-major axis (using the PyMC variables)
        # Note: Since these are tensors, we usually assume the user
        # provided them in order, or we use their 'initval' to sort.
        sorted_planets = sorted(self.planets, key=lambda p: p.a.initval)

        for i in range(len(sorted_planets) - 1):
            inner = sorted_planets[i]
            outer = sorted_planets[i + 1]

            # Get the symbolic apastron (furthest point) of the inner planet
            # Q = a * (1 + e)
            inner_apastron = inner.orbit.a_val * (1.0 + inner.orbit.e_val)

            # Get the symbolic periastron (closest point) of the outer planet
            # q = a * (1 - e)
            outer_periastron = outer.orbit.a_val * (1.0 - outer.orbit.e_val)

            # Potential: If they cross, log-probability goes to -inf
            pm.Potential(
                f"crossing_penalty_{inner.name}_{outer.name}",
                pt.switch(outer_periastron > inner_apastron, 0.0, -np.inf)
            )

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

    # FILE: components/stellarsystem.py
    def compile_plotter_functions(self, model):
        t_input = pt.vector("t_input")
        all_params = self.get_all_parameters()
        self.plot_params = [p for p in all_params if p.expression is None]

        # Pass them as the raw PyMC variables (Vectors)
        param_symbols = [p.value for p in self.plot_params]

        # Orbit map aligns planet parameters to the correct orbits
        K_mapped = self.planets.K.value[self.orbit_map]

        # This will now correctly return (N_times, N_planets)
        rv_matrix_node = self.orbits.get_radial_velocity(t_input, K_mapped, self.orbit_map)

        self._compiled_full_rv = pytensor.function(
            inputs=[t_input] + param_symbols,
            outputs=pt.sum(rv_matrix_node, axis=1),  # Sum over planets
            on_unused_input='ignore'
        )


        self._compiled_rv_matrix = pytensor.function(
            inputs=[t_input] + param_symbols,
            outputs=rv_matrix_node,
            on_unused_input='ignore'
        )