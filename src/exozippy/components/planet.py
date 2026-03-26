import numpy as np
import pytensor.tensor as pt
import pymc as pm
import astropy.units as u

from .component import Component
from exozippy.constants import G, AU, mjup, rjup, pc, rsun, msun, sigmasb, Gmsun, Gmjup, meter
from .parameter import Parameter

import ipdb

class Planet(Component):
    def __init__(self, config, user_params):
        self.label = "Planet Parameters"
        self.config = config
        self.nplanets = len(config)
        self.names = [c.get("name") for c in self.config]
        self.user_params = user_params

    def build_parameters(self, model):
        # This encapsulates the logic in mkss.pro for each planet
        prefix = f"planet"

        # fundamental parameters
        self.radius = Parameter(f"{prefix}.radius",
                                lower=1e-6, upper=100,
                                initval=1.0, init_scale=0.05,
                                unit=u.jupiterRad, internal_unit=u.solRad,
                                latex='R_J', description='Radius',
                                user_params=self.user_params, shape=(self.nplanets,))
        self.radius.build_pymc()

        # allow negative mass to reduce bias of marginal detections (-1 msun to 250 msun -- allow stellar "planets")
        self.mass = Parameter(f"{prefix}.mass",
                              lower=-1e3, upper=2.6e5,
                              initval=1.0, init_scale=0.1,
                              unit=u.jupiterMass, internal_unit=u.solMass,
                              latex='M_P', description='Mass',
                              user_params=self.user_params, shape=(self.nplanets,))
        self.mass.build_pymc()

        density_const = 3.0/(4.0*np.pi)
        self.density = Parameter(f"{prefix}.density",
                                 expression=lambda: density_const*self.mass.value/(self.radius.value*pt.sqr(self.radius.value)),
                                 unit=u.gram/u.cm**3, internal_unit=u.gram/u.cm**3,
                                 latex=r"\rho_P", description='Density',
                                 user_params=self.user_params)
        self.density.build_pymc()

        logg_const = np.log10(Gmsun/rsun**2)
        self.logg = Parameter(f"{prefix}.logg",
                              expression=lambda: logg_const + pt.log10(self.mass.value) - 2.0 * pt.log10(self.radius.value),
                              unit=u.dex(u.cm/u.s**2), internal_unit=u.dex(u.cm/u.s**2),
                              latex=r"\log{g_P}", description='Surface gravity',
                              user_params=self.user_params)
        self.logg.build_pymc()

    # some parameters/constraints require objects from another class. build them here
    def build_dependent_parameters(self, model, stars, orbits, star_map, orbit_map):
        prefix = f"planet"

        #self.orbit = orbits[orbit_map]

        self.p = Parameter(f"{prefix}.p",
                           expression=lambda: self.radius.value / stars.radius.value[star_map],
                           unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                           latex="R_P/R_*", description="",
                           user_params=self.user_params)
        self.p.build_pymc()

        const = (G/ (4.0 * np.pi ** 2))**(1.0/3.0)
        m_total = pt.maximum(stars.mass.value[star_map] + self.mass.value,1e-9) # numerical shield (values at 1e-9 will be rejected below, but we can't compute a NaN)
        m13 = pt.power(m_total, 1.0 / 3.0)
        p2 = pt.sqr(orbits.period.value[orbit_map])
        p23 = pt.power(p2,1.0/3.0)
        arsun = const * m13 * p23
        #self.arsun = Parameter(f"{prefix}.arsun",
        #                       expression = const * m13 * p23,
        #                       latex=r"a/R_{\sun}", print_to_table=False, description=r"Semi-major axis in \rsun", latex_unit=r"\rsun",
        #                       user_params=self.user_params)
        #self.arsun.build_pymc()

        self.ar = Parameter(f"{prefix}.ar",
                            expression=lambda: arsun / stars.radius.value[star_map],
                            unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                            latex="a/R_*", description="",
                            user_params=self.user_params)
        self.ar.build_pymc()

        # Winn, 2010, eq 7
        self.b = Parameter(f"{prefix}.b",
                           expression=self.ar.value * orbits.cosi.value[orbit_map] * (
                                   1.0 - pt.sqr(orbits.ecc.value[orbit_map])) / (1.0 - orbits.esinw.value[orbit_map]),
                           unit=u.dimensionless_unscaled, internal_unit=u.dimensionless_unscaled,
                           latex='b', description='impact parameter',
                           user_params=self.user_params)
        self.b.build_pymc()

        """
        self.a = Parameter(f"{prefix}.a",
                           expression=self.arsun.value / AU, unit=u.au,
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

        # we can leverage the computation of a/rstar to simplify this:
        const = 2.0 * np.pi
        mass_ratio = self.mass.value/m_total
        ecc_factor = 1.0 / pt.sqrt(1.0 - pt.sqr(orbits.ecc.value[orbit_map]))
        self.K = Parameter(f"{prefix}.K",
                           expression=lambda: const*(arsun * orbits.sini.value[orbit_map] * mass_ratio * ecc_factor) / orbits.period.value[orbit_map],
                           unit=u.meter / u.second, internal_unit=u.solRad/u.d,
                           latex='K', description='RV semi-amplitude',
                           user_params=self.user_params)
        #self.K = Parameter(f"{prefix}.K",
        #                   expression=(2.0 * np.pi * G / (orbit.period.value * (
        #                           star.mass.value + self.mass.value * mjup) ** 2.0)) ** (1.0 / 3.0) *
        #                              self.mass.value * mjup * orbit.sini.value /
        #                              pt.sqrt(1.0 - orbit.ecc.value ** 2.0) * rsun / meter / 86400.0,
        #                   unit=u.meter / u.second, latex_unit='m~s$^{-1}$',
        #                   latex='K', description='RV semi-amplitude',
        #                   user_params=self.user_params)
        self.K.build_pymc()

        # check parameters
        #self.sini = f"{prefix}.K",
        #                   expression=(2.0 * np.pi * G / (orbit.period.value * (
        ##                           star.mass.value + self.mass.value * mjup) ** 2.0)) ** (1.0 / 3.0) *
        #                             self.mass.value * mjup * pt.sin(orbit.inc.value) /
        #                              pt.sqrt(1.0 - orbit.ecc.value ** 2.0) * rsun / meter / 86400.0,
        #                   unit=u.meter / u.second, latex_unit='m~s$^{-1}$',
        #                   latex='K', description='RV semi-amplitude',
        #                   user_params=self.user_params)

        # still need things like secondary transit duration,


        ###### system constraints ######

        # we allow negative planet mass to avoid a Lucy-Sweeny bias on the planet mass
        # but we break our equations with a negative system mass. reject that here
        pm.Potential(f"{prefix}.m_pos_penalty", pt.log(m_total))

        # hard wall
        # massbound = pm.Potential(f"{prefix}.mp_bound", pt.switch(m_total > 0.0, 0.0, -np.inf))



        # planet cannot collide with star
        steepness = 500.0
        maxe = 1.0 - 1.0 / self.ar.value - self.p.value / self.ar.value  # 1 - (Rp+Rstar)/a
        diff = maxe - orbits.ecc.value
        # soft wall
        ebound = pm.Potential(f"{prefix}.soft_e_bound",-steepness * pt.sigmoid(-diff * steepness))

        # hard wall
        #ebound = pm.Potential(f"{prefix}.e_bound", pt.switch(orbit.ecc.value < maxe, 0.0, -np.inf))

    def get_rv_signal(self, t):
        """This is what the RVInstrument calls."""
        # This passes the derived K down to the Orbit's exoplanet-core solver
        return self.orbit.get_radial_velocity(t, self.K.value)