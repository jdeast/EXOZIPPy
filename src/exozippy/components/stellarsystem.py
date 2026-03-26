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

        # these are the objects we're modeling
        self.stars = Star(self.config.get("stars"), self.user_params)
        self.orbits = Orbit(self.config.get("orbits"),self.user_params)
        self.planets = Planet(self.config.get("planets"),self.user_params)

        # data sets that constrain the above
        self.instruments = RVInstrument(self.config.get("rv").get("instruments"),self.user_params)
        #self.transits = Transit(self.config.get("transits"),self.user_params)
        #self.astrometry = Astrometry(self.config.get("astrometry"),self.user_params)
        #self.mulensing = mulensing(self.config.get("mulensing"),self.user_params)

    def get_all_parameters(self):
        """
        Recursively gathers every Parameter object from all components.
        """
        all_params = []

        # Helper to crawl any component
        def crawl(component):
            for attr_name in dir(component):
                attr = getattr(component, attr_name)
                if isinstance(attr, Parameter):
                    all_params.append(attr)
                elif isinstance(attr, Component):
                    crawl(attr)

        # Start crawling from the high-level lists
        for comp in [self.stars, self.planets, self.orbits, self.instruments]:
            crawl(comp)

        return all_params

    import pytensor
    import numpy as np

    # FILE: components/stellarsystem.py

    def get_physical_point(self, model, raw_point):
        # 1. We want every node that has a name (Free RVs AND Deterministics)
        output_vars = model.free_RVs + model.deterministics

        # 2. Compile the evaluator
        eval_fn = pytensor.function(
            inputs=model.value_vars,
            outputs=output_vars,
            on_unused_input='ignore'
        )

        # 3. Pass the interval-space values from the raw point
        input_values = [raw_point[v.name] for v in model.value_vars]
        physical_values = eval_fn(*input_values)

        # 4. Build the dictionary mapping labels to physical numbers
        results = {}
        for var, val in zip(output_vars, physical_values):
            # var.name will be 'orbit.period', 'star.mass', etc.
            results[var.name] = val

        return results

    def get_physical_point_old(self, model, raw_point):
        """
        General map: (Sampler Space Dict) -> (Physical Space Dict)
        Returns every Free RV and Deterministic in the model.
        """
        # 1. Gather all variables we want to see (Physical RVs + Derived math)
        # This includes 'star.A.radius', 'orbit.b.ecc', 'planet.b.K', etc.
        output_vars = model.free_RVs + model.deterministics

        # 2. Compile a single evaluator
        # inputs=model.value_vars are the '..._interval__' variables in raw_point
        eval_fn = pytensor.function(
            inputs=model.value_vars,
            outputs=output_vars,
            on_unused_input='ignore'
        )

        # 3. Map the raw_point dictionary to a positional list for the function
        # This is the 'Magic Step' that avoids KeyErrors
        input_values = [raw_point[v.name] for v in model.value_vars]

        # 4. Execute and rebuild the dictionary with clean names
        physical_values = eval_fn(*input_values)

        # Create a lookup of all Parameter objects by their label
        # (Assuming you have a helper like get_all_parameters() in StellarSystem)
        param_lookup = {p.label: p for p in self.get_all_parameters()}

        results = {}
        for var, val in zip(output_vars, physical_values):
            name = var.name

            # 4. Check if this is a Parameter we know about
            if name in param_lookup:
                p_obj = param_lookup[name]
                # Use the existing logic to convert back to User Units
                # Handles scalars or vectors (e.g., [i] for multiple planets)
                results[name] = p_obj.from_internal(val)
            else:
                # Fallback for purely internal PyMC deterministics
                results[name] = float(np.atleast_1d(val)[0]) if np.isscalar(val) or val.size == 1 else val

        return results

        #return {var.name: float(val) for var, val in zip(output_vars, physical_values)}

    def build_model(self):

        """
        Build the PyMC Graph.
        """
        with pm.Model() as model:

            # 1. first pass, build the foundational parameters
            self.stars.build_parameters(model)
            self.orbits.build_parameters(model)
            self.planets.build_parameters(model)
            self.instruments.build_parameters(model)

            self.orbit_map = np.array([p_cfg.get("orbit_ndx", 0) for p_cfg in self.planets.config])
            self.star_map = np.array([p_cfg.get("star_ndx", 0) for p_cfg in self.planets.config])

            self.planets.build_dependent_parameters(model,
                                                    stars=self.stars, orbits=self.orbits,
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

    def _set_comp_posterior_old(self, component, posterior):
        # 1. Assign to Parameters on this component
        for attr_name in dir(component):
            attr = getattr(component, attr_name)
            if isinstance(attr, Parameter):
                if attr.label in posterior:
                    attr.posterior = posterior[attr.label]
                # Handle derived parameters (Deterministics) too!
                elif attr.label in posterior:
                    attr.posterior = posterior[attr.label]

        # 2. Recurse to children
        for attr_name in dir(component):
            attr = getattr(component, attr_name)
            if isinstance(attr, Component):
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

    import pytensor.tensor as pt
    import numpy as np

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


    def get_mcmc_init(self, model):
        """
        Returns an array of initial scales mapped to model.free_RVs,
        a dictionary of physical initial values, and a dictionary of
        interval-transformed initial values for PyMC logp evaluation.
        """
        # Create lookup dictionaries of all parameters we've built
        all_scales = {}
        all_inits = {}

        for component in [self.stars, self.planets, self.orbits, self.instruments]:
            all_scales.update(component.get_scales())
            all_inits.update(component.get_inits())

        ordered_scales = []
        ordered_initvals = {}
        transformed_inits = {}

        for random_var in model.free_RVs:
            name = random_var.name

            # 1. Match the Scale
            scale = all_scales.get(name, 1.0)
            ordered_scales.append(scale)

            # 2. Match the Initval and Build the Transform
            if name in all_inits:
                physical_val = all_inits[name]
                ordered_initvals[name] = physical_val

                # 3. Check PyMC 5's internal transform dictionary
                transform = model.rvs_to_transforms.get(random_var)

                if transform is not None:
                    # Get the actual internal value variable (this has the _interval__ name)
                    value_var = model.rvs_to_values[random_var]
                    transformed_name = value_var.name

                    # Push the physical value through PyMC's exact interval math
                    t_node = transform.forward(
                        pt.as_tensor_variable(physical_val),
                        *random_var.owner.inputs
                    )
                    transformed_inits[transformed_name] = t_node.eval()
                else:
                    # If no bounds/transform exist, physical space == internal space
                    transformed_inits[name] = physical_val
            else:
                print(f"WARNING: No initval found for {name}. Using PyMC default.")

        return np.array(ordered_scales), ordered_initvals, transformed_inits

    def get_mcmc_init_old(self, model):
        """
        Returns an array of initial scales mapped to model.free_RVs.
        """
        # Create a lookup dictionary of all parameters we've built
        # { 'planet.b.radius': 0.5, 'orbit.b.logp': 0.01, ... }
        all_scales = {}
        all_inits = {}

        for component in [self.stars,self.planets,self.orbits,self.instruments]:
            all_scales.update(component.get_scales())
            all_inits.update(component.get_inits())

        ordered_scales = []
        ordered_initvals = {}
        for random_var in model.free_RVs:
            name = random_var.name

            scale = all_scales.get(name, 1.0)
            ordered_scales.append(scale)

            # Match the Initval
            if name in all_inits:
                ordered_initvals[name] = all_inits[name]
            else:
                print(f"WARNING: No initval found for {name}. Using PyMC default.")

        return np.array(ordered_scales), ordered_initvals

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