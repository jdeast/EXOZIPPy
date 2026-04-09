import numpy as np
import pytensor.tensor as pt
import pymc as pm

from .component import Component

import ipdb

class Planet(Component):
    def __init__(self, config, config_manager):
        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        super().__init__(config, config_manager)
        self.label = "Planet Parameters"
        self.prefix = f"planet"

    def build_parameters(self, model):
        # This encapsulates the logic in mkss.pro for each planet

        # 1. Fundamental & Internal parameters
        parameters = {
            "radius": None,
            "mass": None,
            "density": "default",
            "logg": "default"
        }

        self.build_pars_from_dict(parameters, shape=(self.n_elements,), prefix=self.prefix)

    def load_data(self):
        # no data for planet component
        pass

    def build_dependent_parameters(self, model, system): #stars, orbits, star_map, orbit_map):

        # 1. Prepare the cross-component PyTensor nodes with maps applied
        stars = system.star
        orbits = system.orbit
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

    def plot(self, system, points, filename_prefix="debug"):
        pass