import numpy as np

import pytensor.tensor as pt

import astropy.units as u
from exozippy.constants import G, AU, mjup, rjup, pc, rsun, msun, sigmasb, Gmsun, meter, lsun
from pathlib import Path
from .parameter import Parameter
from .component import Component

# debugging imports
import ipdb

class Star(Component):
    def __init__(self, config, user_params):
        self.label = "Stellar Parameters"
        self.config = config
        self.user_params = user_params
        self.nstars = len(self.config)
        self.names = [c.get("name") for c in self.config]


        # this belongs at the system level
        #self.sedfile = Path(self.config.get("sedfile", None))
        #self.teffsedfloor = self.config.get("teffsedfloor", 0.020)
        #self.fbolsedfloor = self.config.get("fbolsedfloor", 0.024)

        self.mannmass = [c.get("mannmass",False) for c in self.config]
        self.mannrad = [c.get("mannrad",False) for c in self.config]
        self.mist = [c.get("mist",True) for c in self.config]
        self.parsec = [c.get("parsec",False) for c in self.config]

    def build_parameters(self, model):

        prefix = f"star"

        # the lower limits mean it doesn't actually have to be a star
        # the upper limits are defined by the most extreme stars known (plus some padding)

        # fundamental parameters
        self.radius = Parameter(f"{prefix}.radius",
                                lower=1e-1, upper=2000,
                                initval=1.0, init_scale=0.02,
                                unit=u.solRad, internal_unit=u.solRad,
                                latex='R_*', description='Radius',
                                user_params=self.user_params, shape=(self.nstars,))
        self.radius.build_pymc()

        self.mass = Parameter(f"{prefix}.mass",
                              lower=1e-1, upper=250,
                              initval=1.0, init_scale=0.02,
                              unit=u.solMass, internal_unit=u.solMass,
                              latex='M_*', description='Mass',
                              user_params=self.user_params, shape=(self.nstars,))
        self.mass.build_pymc()

        self.teff = Parameter(f"{prefix}.teff",
                              lower=1.0, upper=5e5,
                              initval=5778, init_scale=80.0,
                              unit=u.K, internal_unit=u.K,
                              latex=r"T_{\rm eff}", description="Effective Temperature",
                              user_params=self.user_params, shape=(self.nstars,))
        self.teff.build_pymc()

        self.feh = Parameter(f"{prefix}.feh",
                             lower=-5.0, upper=5.0,
                             initval=0.0, init_scale = 0.08,
                             unit=u.dex, internal_unit=u.dex,
                             latex=r"[{\rm Fe/H}]", description='Metallicity',
                             user_params=self.user_params, shape=(self.nstars,))
        self.feh.build_pymc()

        # derived parameters
        tsun4 = 5778**4
        self.luminosity = Parameter(f"{prefix}.luminosity",
                                    expression=lambda: pt.sqr(self.radius.value) * pt.sqr(pt.sqr(self.teff.value))/tsun4,
                                    unit=u.solLum, internal_unit=u.solLum,
                                    latex='L_*', description='Luminosity',
                                    user_params=self.user_params)
        self.luminosity.build_pymc()

        density_const = 3.0 / (4.0 * np.pi)
        self.density = Parameter(f"{prefix}.density",
                                    expression=lambda: density_const*self.mass.value/(pt.sqr(self.radius.value)*self.radius.value),
                                    unit=u.gram/u.cm**3, internal_unit=u.solMass/u.solRad**3,
                                    latex=r"\rho_*", description='Density',
                                    user_params=self.user_params)
        self.density.build_pymc()

        # note the lambda construction of the expression means the expression is only used for computing the posteriors at the end for the output table
        # otherwise, it would create unused nodes in the graph during sampling that would slow it down
        logg_const = np.log10(Gmsun/rsun**2)
        self.logg = Parameter(f"{prefix}.logg",
                              expression=lambda: logg_const + pt.log10(self.mass.value) - 2.0 * pt.log10(self.radius.value),
                              unit=u.dex(u.cm/u.s**2), internal_unit=u.dex(u.cm/u.s**2),
                              latex=r"\log{g_*}", description='Surface gravity',
                              user_params=self.user_params)
        self.logg.build_pymc()


        if self.parsec or self.mist:
            self.age = Parameter(f"{prefix}.age",
                                 lower=0.0, upper=13.77,
                                 initval=4.603, init_scale=3.0,
                                 unit=u.Gyr, internal_unit=u.Gyr,
                                 latex=r"{\rm Age}", description="Age",
                                 user_params=self.user_params, shape=(self.nstars,))
            self.age.build_pymc()

            self.initfeh = Parameter(f"{prefix}.initfeh",
                                     lower=-5.0, upper=5.0,
                                     initval=self.feh.initval,init_scale=0.08,
                                     unit=u.dex,internal_unit=u.dex,
                                     latex=r"[{\rm Fe/H}]_{0}", description='Initial Metallicity',
                                     user_params=self.user_params, shape=(self.nstars,))
            self.initfeh.build_pymc()

        return

        # punt on the SED for now, but much of this SED block needs to be moved to the system level
        # if we need the distance, add it as a parameter
        if self.sedfile != None or self.mannrad or self.mannmass:
            self.distance = Parameter(f"{prefix}.distance",
                                      lower=1e-6, upper=1.426e10,
                                      initval=10.0, init_scale=1000.0,
                                      unit=u.pc, internal_unit=u.solRad,
                                      latex='d', description="Distance",
                                      user_params=self.user_params, shape=(self.nstars,))
            self.distance.build_pymc()

            self.fbol = Parameter(f"{prefix}.fbol",
                                  expression= self.luminosity.value / (4.0 * np.pi * (self.distance.value * pc) ** 2.0),
                                  unit=u.erg/u.second/u.cm**2, internal_unit=u.erg/u.second/u.cm**2,
                                  latex="F_{Bol}", description='Bolometric Flux',
                                  user_params=self.user_params)
            self.fbol.build_pymc()

            self.parallax = Parameter(f"{prefix}.parallax",
                                      expression= 1e3 / self.distance.value,
                                      unit=u.mas, internal_unit=u.mas,
                                      latex=r'\varpi', description='Parallax',
                                      user_params=self.user_params)
            self.parallax.build_pymc()

            if self.mannmass or self.mannrad:
                self.appks = Parameter(f"{prefix}.appks",
                                       lower=-30,upper=30,
                                       unit=u.mag, internal_unit=u.mag,
                                       latex='ks', description="Apparent ks mag",
                                       user_params=self.user_params, shape=(self.nstars,))
                self.appks.build_pymc()

                self.absks = Parameter(f"{prefix}.absks",
                                       expression= self.appks.value - 2.5*pt.log10((self.distance.value/10.0)**2),
                                       unit=u.mag, internal_unit=u.mag,
                                       latex='Ks', description="Absolute Ks mag",
                                       user_params=self.user_params)
                self.absks.build_pymc()

#                mann_mstar, mann_rstar, mann_sigma_mstar, mann_sigma_rstar = massradius_mann(self.ks0.value, self.feh.value,
#                                                                                             distance=self.distance.value)
#                    if self.mannmass:
#                        mannmass_prior = pm.Potential(f"{prefix}."mannmass_prior", -0.5 *
#                                                      ((self.mass.value - mann_mstar) / mann_sigma_mstar) ** 2)
#                    if self.mannrad:
#                        mannrad_prior = pm.Potential(f"{prefix}.mannrad_prior", -0.5 *
#                                                     ((star.radius.value - mann_rstar) / mann_sigma_rstar) ** 2)

        # add SED parameters for each star
        if self.sedfile != None:
            self.av = Parameter(f"{prefix}.av",
                                lower=0.0, upper=100.0,
                                initval=0.0, init_scale=0.3,
                                unit=u.mag, internal_unit=u.mag,
                                latex='A_V', description='V-band extinction',
                                user_params=self.user_params, shape=(self.nstars,))
            self.av.build_pymc()

            self.rstarsed = Parameter(f"{prefix}.rstarsed",
                                      lower=1e-9, upper=2000.0,
                                      initval=1.0, init_scale=0.1,
                                      unit=u.solRad, internal_unit=u.solRad,
                                      latex='R_{*,SED}',description="Stellar radius",
                                      user_params=self.user_params, shape=(self.nstars,),
                                      table_note="This value ignores the systematic error and is for reference only")
            self.rstarsed.build_pymc()

            self.teffsed = Parameter(f"{prefix}.teffsed",
                                     lower=1.0, upper=5e6,
                                     initval=5778.0, init_scale = 300.0,
                                     unit=u.K, internal_unit=u.K,
                                     latex='T_{eff,SED}', description="Stellar temperature",
                                     user_params=self.user_params,
                                     table_note = "This value ignores the systematic error and is for reference only")
            self.teffsed.build_pymc()

            # these potentials link teff/teffsed and fbol/fbolsed within the sedfloors
            # so the SED cannot impact teff/fbol better than the floors
            # see eq ?? of Eastman+ 2019
            self.fbolsed = Parameter(f"{prefix}.fbolsed",
                                     expression=self.luminosity.value / (4.0 * np.pi * (self.distance.value * pc) ** 2.0),
                                     unit=u.erg/u.second/u.cm**2,internal_unit=u.erg/u.second/u.cm**2,
                                     latex="F_{Bol,SED}", description='Bolometric Flux',
                                     user_params=self.user_params, shape=(self.nstars,),
                                     table_note="This value ignores the systematic error and is for reference only")
            self.fbolsed.build_pymc()

            # this links the two with a user settable error floor
            self.teffsed_floor_prior = pm.Potential(f"{prefix}.teffsed_floor_prior",
                                                    -0.5 * ((self.teff.value - self.teffsed.value) / (self.teff.value * self.teffsedfloor)) ** 2)
            self.fbolsed_floor_prior = pm.Potential(f"{prefix}.fbolsed_floor_prior",
                                                    -0.5 * ((self.fbol.value - self.fbolsed.value) / (self.fbol.value * self.fbolsedfloor)) ** 2)