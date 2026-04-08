import numpy as np
import pytensor.tensor as pt
import pymc as pm

from .component import Component
from .parameter import Parameter
from ..physics import calc_density, calc_logg, calc_b, calc_arsun, calc_arstar

import ipdb

class Planet(Component):
    def __init__(self, config, config_manager):
        # 1. Initialize the base Component
        # sets self.config and self.config_manager
        super().__init__(config, config_manager)
        self.label = "Planet Parameters"

    def build_parameters(self):
        # This encapsulates the logic in mkss.pro for each planet
        prefix = f"planet"

        # 1. Fundamental & Internal parameters
        parameters = {
            "radius": None,
            "mass": None,
            "density": "default",
            "logg": "default"
        }

        self.build_pars_from_dict(parameters, shape=(self.n_elements,), prefix=prefix)

    def build_dependent_parameters(self, stars, orbits, star_map, orbit_map):
        prefix = "planet"

        # 1. Prepare the cross-component PyTensor nodes with maps applied
        m_total_node = pt.maximum(stars.mass.value[star_map] + self.mass.value, 1e-9)

        context_nodes = {
            "m_total": m_total_node,
            "star_radius": stars.radius.value[star_map],
            "orbit_period": orbits.period.value[orbit_map],
            "orbit_ecc": orbits.ecc.value[orbit_map],
            "orbit_cosi": orbits.cosi.value[orbit_map],
            "orbit_esinw": orbits.esinw.value[orbit_map],
            "orbit_sini": orbits.sini.value[orbit_map],
        }

        # 2. Back to the elegant YAML manifest!
        parameters = {
            "p": "default",
            "arsun": "default",
            "ar": "default",
            "b": "default",
            "K": "default"
        }

        self.build_pars_from_dict(parameters, shape=(self.n_elements,), prefix=prefix, context_nodes=context_nodes)

        # (Keep the soft boundary Potentials here, as they are system-level likelihoods, not parameters)
        steepness = 500.0
        pm.Potential(f"{prefix}.m_pos_constraint", pm.math.log(pt.sigmoid(m_total_node * steepness)))

        maxe = 1.0 - 1.0 / self.ar.value - (self.radius.value / context_nodes["star_radius"]) / self.ar.value
        diff = maxe - orbits.ecc.value[orbit_map]
        pm.Potential(f"{prefix}.e_collision_bound", pm.math.log(pt.sigmoid(diff * steepness)))


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

    def get_rv_signal(self, t):
        """This is what the RVInstrument calls."""
        # This passes the derived K down to the Orbit's exoplanet-core solver
        return self.orbit.get_radial_velocity(t, self.K.value)