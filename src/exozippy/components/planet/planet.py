import numpy as np
import pytensor.tensor as pt
import pymc as pm

from exozippy.components.component import Component

import ipdb

from exozippy.components.celestial_body.celestial_body import CelestialBody
# this import is required even though it's not used explicitly
# it registers all the mathematical relations
from . import physics

class Planet(CelestialBody):
    def __init__(self, config, config_manager):
        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        super().__init__(config, config_manager)
        self.label = "Planet Parameters"
        self.prefix = f"planet"


    def build_parameters(self, model):
        pass

    def load_data(self):
        # no data for planet component
        pass

    def build_map(self, system):
        # Resolve which star each planet belongs to
        star_indices = np.array([p.get("star_ndx", 0) for p in self.config])
        self.star_map = pt.as_tensor_variable(star_indices).astype("int32")

        # Resolve which orbit each planet belongs to
        orbit_indices = np.array([p.get("orbit_ndx", 0) for p in self.config])
        self.orbit_map = pt.as_tensor_variable(orbit_indices).astype("int32")


    def build_dependent_parameters(self, model, system): #stars, orbits, star_map, orbit_map):

        # 1. Prepare the cross-component PyTensor nodes with maps applied
        stars = system.star
        orbits = system.orbit

        # --- 1. DATA-DRIVEN HEURISTIC (RV K_init -> Planet Mass) ---
        cfg = self.config_manager.resolve(self.prefix, "mass", shape=(self.n_elements,))
        mass_inits = cfg.get("initval", np.ones(self.n_elements)).copy()

        rv_comps = [c for c in system.active_components.values() if hasattr(c, 'k_init')]

        for i in range(self.n_elements):
            # Only apply the heuristic if the user DID NOT explicitly set the mass in the yaml
            user_keys = [f"planet.{self.names[i]}.mass", f"planet.{i}.mass", "planet.mass"]
            has_initval_override = False
            for k in user_keys:
                ov = self.config_manager.user_params.get(k)
                if isinstance(ov, dict) and "initval" in ov and ov["initval"] is not None:
                    has_initval_override = True
                elif ov is not None and not isinstance(ov, dict):
                    has_initval_override = True  # Shorthand override (e.g., planet.0.mass: 5.0)

            if has_initval_override:
                continue

            if rv_comps:
                k_ms = rv_comps[0].k_init/np.sqrt(self.n_elements)
                s_idx = self.star_map.eval()[i]
                o_idx = self.orbit_map.eval()[i]

                # Fetch the initialized star mass and period for this specific planet
                m_star_arr = np.atleast_1d(stars.mass.initval)
                period_arr = np.atleast_1d(orbits.period.initval)
                m_star = m_star_arr[s_idx] if s_idx < len(m_star_arr) else m_star_arr[0]
                period = period_arr[o_idx] if o_idx < len(period_arr) else period_arr[0]

                # Invert the RV Semi-amplitude equation for a rough mass estimate:
                # M_p ≈ K / 28.43 * M_star^(2/3) * (P/365.25)^(1/3)
                m_p_guess = (k_ms / 28.43) * (m_star ** (2.0 / 3.0)) * ((period / 365.25) ** (1.0 / 3.0))
                mass_inits[i] = m_p_guess

        # Build the core parameters using our new heuristic dict
        self.build_core_parameters(model, self.prefix, overrides={"mass": {"initval": mass_inits}})

        context_nodes = {
            "star_mass": stars.mass.value[self.star_map],
            "star_radius": stars.radius.value[self.star_map],
            "orbit_period": orbits.period.value[self.orbit_map],
            "orbit_ecc": orbits.ecc.value[self.orbit_map],
            "orbit_cosi": orbits.cosi.value[self.orbit_map],
            "orbit_esinw": orbits.esinw.value[self.orbit_map],
            "orbit_sini": orbits.sini.value[self.orbit_map],
        }

        # 2. Back to the elegant YAML manifest!
        parameters = {
            "m_total": "default",
            "p": "default",
            "arsun": "default",
            "ar": "default",
            "b": "default",
            "K": "default",
            "max_ecc":"default",
        }

        self.build_pars_from_dict(parameters, shape=(self.n_elements,), prefix=self.prefix, context_nodes=context_nodes)

        """
        self.a = Parameter(f"{prefix}.a",
                           expression=lambda: self.arsun.value / AU, unit=u.au,
                           latex='a', description='Semi-major axis', latex_unit='au',
                           user_params=self.user_params)
        self.a.build_pymc()



        # Winn, 2010, eq 8
        self.bs = Parameter(f"{prefix}.bs",
                           expression=self.ar.value * orbit.cosi.value * (
                                   1.0 - orbit.ecc.value ** 2) / (1.0 + orbit.esinw.value),
                           latex='b_S', description='secondary impact parameter',
                           latex_unit='', user_params=self.user_params)
        self.bs.build_pymc()

        # Winn, 2010, eq 14 (with safeguards for grazing/non transit)
        sin_arg = pt.sqrt((1.0 + self.p.value) ** 2 - self.b.value ** 2) / (orbit.sini.value * self.ar.value)
        t14_math = (orbit.period.value / np.pi) * \
                   pt.arcsin(sin_arg) * \
                   pt.sqrt(1.0 - orbit.ecc.value ** 2) / (1.0 + orbit.esinw.value)
        is_transiting = (self.b.value < (1.0 + self.p.value)) & (sin_arg <= 1.0)

        self.t14 = Parameter(f"{prefix}.t14",
                           expression=pt.switch(is_transiting, t14_math, 0.0),
                           latex='T_{14}', description='Transit duration',
                           latex_unit='days', user_params=self.user_params)
        self.t14.build_pymc()

        # Winn, 2010, eq 15 (with safeguards for grazing/non transit)
        t23_chord_sq = (1.0 - self.p.value) ** 2 - self.b.value ** 2
        sin_arg_23 = pt.sqrt(pt.maximum(t23_chord_sq, 0.0)) / (orbit.sini.value * self.ar.value)
        t23_math = (orbit.period.value / np.pi) * \
                   pt.arcsin(pt.clip(sin_arg_23, 0.0, 1.0)) * \
                   pt.sqrt(1.0 - orbit.ecc.value ** 2) / (1.0 + orbit.esinw.value)
        is_full_transit = (self.b.value < (1.0 - self.p.value))
        self.t23 = Parameter(f"{prefix}.t23",
                           expression=pt.switch(is_full_transit, t23_math, 0.0),
                           latex='T_{23}', print_to_table=False, description='Transit duration',
                           latex_unit='days', user_params=self.user_params)
        self.t23.build_pymc()

        self.tau = Parameter(f"{prefix}.tau",
                           expression=(self.t14.value - self.t23.value)/2.0,
                           latex=r"\tau", description='Ingress/egress duration',
                           latex_unit='days', user_params=self.user_params)
        self.tau.build_pymc()

        self.tfwhm = Parameter(f"{prefix}.tfwhm",
                           expression=self.t14.value - self.tau.value,
                           latex='T_{FWHM}', description='Transit duration',
                           latex_unit='days', user_params=self.user_params)
        self.tfwhm.build_pymc()
        """
    def build_likelihood(self, model, system):
        # (Keep the soft boundary Potentials here, as they are system-level likelihoods, not parameters)
        steepness = 500.0
        pm.Potential(f"{self.prefix}.m_pos_constraint", pm.math.log(pt.sigmoid(self.m_total.value * steepness)))

        orbits = system.orbit
        diff = self.max_ecc.value - orbits.ecc.value[self.orbit_map]
        pm.Potential(f"{self.prefix}.e_collision_bound", pm.math.log(pt.sigmoid(diff * steepness)))


        if self.n_elements < 2: return

        print("planet collision penalty isn't tested!!!")

        # 1. Sort planets by semi-major axis (using the PyMC variables)
        # Note: Since these are tensors, we usually assume the user
        # provided them in order, or we use their 'initval' to sort.
        sorted_planets = sorted(self.planets, key=lambda p: p.a.initval)

        for i in range(len(sorted_planets) - 1):
            inner = sorted_planets[i]
            outer = sorted_planets[i + 1]

            # Get the symbolic apastron (furthest point) of the inner planet
            # Q = a * (1 + e)
            inner_apastron = inner.orbit.a.value * (1.0 + inner.orbit.ecc.value)

            # Get the symbolic periastron (closest point) of the outer planet
            # q = a * (1 - e)
            outer_periastron = outer.orbit.a_val * (1.0 - outer.orbit.ecc.value)

            # Potential: If they cross, log-probability goes to -inf
            pm.Potential(
                f"crossing_penalty_{inner.name}_{outer.name}",
                pt.switch(outer_periastron > inner_apastron, 0.0, -np.inf)
            )

    def plot(self, system, points, filename_prefix="debug"):
        pass