# general imports
import math
import glob
import ipdb

# astro/science imports
import numpy as np
from astropy import units as u
import astropy.constants as const

# pymc imports
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# exoplanet imports
# import exoplanet as xo

# exozippy imports
from summarize_model import summarize_model
from build_latex_table import build_latex_table
from trace_to_event import trace_to_event
from massradius_mann import massradius_mann
from parameter import Parameter
from readtran import readtran
from read_parfile import read_parfile
from exozippy_getmcmcscale import exozippy_getmcmcscale

class Star():

    def __init__(self, name,
                 parsec=False, mist=True, mannrad=False, mannmass=False,
                 sedfile=None,
                 ulenspath=None, astrompath=None):

        # constants easier to digest variable names
        G = const.GM_sun.value / const.R_sun.value ** 3 * 86400.0 ** 2
        AU = const.au.value / const.R_sun.value
        mjup = const.GM_jup.value / const.GM_sun.value
        rjup = const.R_jup.cgs.value / const.R_sun.cgs.value  # rsun/rjup
        pc = const.pc.cgs.value  # cm/pc
        rsun = const.R_sun.cgs.value  # cm/r_sun
        msun = const.M_sun.cgs.value  # g/m_sun
        sigmasb = const.sigma_sb.cgs.value
        Gmsun = const.GM_sun.cgs.value
        meter = 100.0  # cm/m

        self.rootlabel = "Stellar Parameters"
        self.name = name
        self.radius = Parameter("rstar_" + name, lower=1e-1, upper=2000, initval=1.0, unit=u.solRad,
                                latex='R_*', description='Radius', latex_unit="\\rsun",
                                user_params=user_params)

        self.mass = Parameter("mstar_" + name, lower=1e-1, upper=250, initval=1.0, unit=u.solMass,
                              latex='M_*', description='Mass', latex_unit="\msun",
                              user_params=user_params)

        self.teff = Parameter("teff_" + name, lower=1.0, upper=5e5, initval=5778, unit=u.K,
                                  latex='T_{\\rm eff}', description="Effective Temperature", latex_unit='K',
                                  user_params=user_params)

        self.feh = Parameter("feh_" + name, lower=-5.0, upper=5.0, initval=0.0,
                                 latex="[{\\rm Fe/H}]", description='Metallicity', latex_unit='dex',
                                 user_params=user_params)

        self.lstar = Parameter('lstar_' + name,
                               expression=4.0 * math.pi * rsun * rsun * sigmasb *
                               self.radius.value ** 2 * self.teff.value ** 4,
                               unit=u.erg / u.second,
                               latex='L_*', description='Luminosity', latex_unit='\lsun',
                               user_params=user_params)

        self.rhostar = Parameter('rhostar_' + name,
                                 expression=3.0 * self.mass.value * msun /
                                 (4.0 * math.pi * (self.radius.value * rsun) ** 3),
                                 unit=u.gram / u.cm ** 3,
                                 latex="\\rho_*", description='Density', latex_unit='g~cm$^{-3}$',
                                 user_params=user_params)

        self.logg = Parameter('logg_' + name, unit=u.dex(u.cm / u.s ** 2),
                              expression=pt.log10(
                              Gmsun * self.mself.value / (self.radius.value * rsun) ** 2),
                              latex="\log{g_*}", description='Surface gravity', latex_unit='cgs',
                              user_params=user_params)

        self.age = Parameter("age_" + name, lower=0.0, upper=13.77, initval=4.603, unit=u.year * 1e9,
                                    latex="Age", description="Age", latex_unit="Gyr", user_params=user_params)

        if parsec or mist:
            self.initfeh = Parameter("initfeh_" + name, lower=-5.0, upper=5.0, initval=0.0,
                                     latex='[{\\rm Fe/H}]_{0}', description='Initial Metallicity',
                                     latex_unit='dex', user_params=user_params)

        if ulenspath != None or astrompath != None or sedfile != None or mannrad or mannmass:
            self.distance = Parameter("distance_" + name, lower=1e-6, upper=1.426e10, initval=10.0,
                                      unit=u.pc, latex='d', description="Distance", latex_unit='pc',
                                      user_params=user_params)

            self.parallax = Parameter("parallax_" + name,
                                     expression=1e3 / star["distance"].value,
                                     unit=u.mas,
                                     latex='\varpi', description='Parallax', latex_unit='mas',
                                     user_params=user_params)

        # if we need the distance, add it as a parameter
        if sedfile != None or mannrad or mannmass:
            self.fbol = Parameter("fbol_" + name,
                                  expression=self.lself.value /
                                  (4.0 * math.pi * (self.distance.value * pc) ** 2.0),
                                  unit=u.erg / u.second / u.cm ** 2, latex_unit='erg~s$^{-1}$~cm$^{-2}$',
                                  latex="F_{Bol}", description='Bolometric Flux', user_params=user_params)


            # apply a potential to constrain the stellar radius according to the Mann+ 2015 K-Rstar relation (mannrad) or
            # constrain the stellar mass according to the Mann+ 2019 K-Mstar relation (mannmass)
            # only applies for 0.1 < mstar/msun < 0.7
            if mannmass or mannrad:
                self.appks = Parameter("appks_" + name, lower=-30, upper=30, unit=u.mag,
                                       latex='ks', description="Apparent ks mag", latex_unit='mag',
                                       user_params=user_params)

                self.absks = Parameter("absks_" + name,
                                       expression=self.appks.value - 2.5 * pt.log10(
                                       (stardistance.value / 10.0) ** 2),
                                       unit=u.mag, latex='Ks', description="Absolute Ks mag", latex_unit='mag',
                                       user_params=user_params)

                mann_mstar, mann_rstar, mann_sigma_mstar, mann_sigma_rstar = (
                    massradius_mann(self.absks.value, self.feh.value, distance=self.distance.value))

                if mannmass:
                    mannmass_prior = pm.Potential("mannmass_prior_" + name, -0.5 *
                                                  ((star["mass"].value - mann_mstar) / mann_sigma_mstar) ** 2)
                if mannrad:
                    mannrad_prior = pm.Potential("mannrad_prior_" + str(i), -0.5 *
                                                 ((star["radius"].value - mann_rstar) / mann_sigma_rstar) ** 2)

        # add SED parameters for each star
        if sedfile != None:
            self.av = Parameter("av_" + name, lower=0.0, upper=100.0, initval=0.0,
                                unit=u.mag,
                                latex='A_V', description='V-band extinction', latex_unit='mag',
                                user_params=user_params)

            # respect systematic error floors on fbol and teff as determined from the SED
            self.rstarsed = Parameter("rstarsed_" + name, lower=1e-9, upper=2000.0, initval=1.0,
                                      unit=u.solRad, latex='R_{*,SED}', latex_unit='\\rsun',
                                      user_params=user_params)

            self.teffsed = Parameter("teffsed_" + name, lower=1.0, upper=5e6, initval=5778.0,
                                     unit=u.K, latex='T_{eff,SED}', latex_unit='K', user_params=user_params)

            # these potentials link teff/teffsed and fbol/fbolsed within the sedfloors
            # so the SED cannot impact teff/fbol better than the floors
            # see eq ?? of Eastman+ 2019
            self.fbolsed = Parameter("fbolsed_" + name,
                                     expression=self.lstar.value / (4.0 * math.pi * (self.distance.value * pc) ** 2.0),
                                     unit=u.erg / u.second / u.cm ** 2, latex_unit='erg~s$^{-1}$~cm$^{-2}$',
                                     latex="F_{Bol,SED}", description='Bolometric Flux', user_params=user_params,
                                     table_note="This may be systematics dominated and is only for reference")

            teffsed_floor_prior = pm.Potential("teffsed_floor_prior_" + name, -0.5 * (
                    (self.teff.value - self.teffsed.value) / (self.teff.value * teffsedfloor)) ** 2)

            fbolsed_floor_prior = pm.Potential("fbolsed_floor_prior" + name, -0.5 * (
                    (star["fbol"].value - fbolsed) / (star["fbol"].value * fbolsedfloor)) ** 2)

        if ulenspath != None or astrompath != None:
            self.ra = Parameter("ra_icrs" + name, lower=0.0, upper=360.0, initval=180.0,
                                unit=u.deg, latex='RA (ICRS)', latex_unit='^\circ',
                                user_params=user_params)

            self.dec = Parameter("dec_icrs" + name, lower=-90.0, upper=90.0, initval=0.0,
                                 unit=u.deg, latex='Dec (ICRS)', latex_unit='^\circ',
                                 user_params=user_params)

            self.pmra = Parameter("pmra" + name, lower=-10000.0, upper=10000.0, initval=0.0,
                                 unit=u.arcsec/1000.0, latex='PM RA', latex_unit='mas',
                                 user_params=user_params)

            self.dec = Parameter("pmdec" + name, lower=-90.0, upper=90.0, initval=0.0,
                                 unit=u.arcsec/1000.0, latex='PM Dec', latex_unit='mas',
                                 user_params=user_params)

            dec_ngp = (27.0 + 7.0/60.0 + 42.01/3600.0)*math.pi/180.0
            ra_ngp = (12.0 + 51.0/60.0 + 26.282/3600.0)*15.0*math.pi/180.0
            l_ngp = 2.145570
            self.b = Parameter("galactic_latitude" + name,
                               expression=pt.arcsin(
                                   pt.sin(dec_ngp)*pt.sin(self.dec*math.pi/180.0) + pt.cos(dec_ngp)*pt.cos(self.dec*math.pi/180.0)*pt.cos(self.ra*math.pi/180.0-ra_ngp)
                               ),
                                     unit=u.rad, latex_unit='',
                                     latex="b", description='Galactic Latitude', user_params=user_params)
            self.l = Parameter("galactic_longitude" + name,
                               expression=l_ngp - pt.arcsin(
                                   pt.cos(self.dec*math.pi/180.0)*pt.sin(self.ra*math.pi/180.0-ra_ngp)/pt.cos(self.b)
                               ),
                                     unit=u.rad, latex_unit='',
                                     latex="l", description='Galactic Longitude', user_params=user_params)


        # we're gonna need this for mulensmodel...
        # https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html
        event["ulens"] = []
        for i, ulensfile in enumerate(ulensfiles):
            event["ulens"].append(readulens(ulensfile))

        # pm.traceplot(trace)

        # model = pm

        # default method='L-BFGS-B', other options are powell, amoeba, etc
        # find the initial best-fit
        # map_estimate = pm.find_MAP(model=model)

        #        model.debug()

        # MCMC sampling with pymc defaults
        trace = pm.sample(200, chains=4, cores=4, target_accept=0.9, discard_tuned_samples=False, tune=1000)

        ipdb.set_trace()

        # pymc generally assumes a small, well-behaved prior volume around the best-fit solution
        # their auto-tuning is poor for multi-modal parameters with an unknown scale (period)

        # our model parameters have wide uniform priors that doesn't play well with that assumption. Use our own step scaling.

        # this initializes the model based on MAP optimization
        map = pm.find_MAP(return_raw=True)
        # TODO: fit_params needs to be defined
        for p in fit_params:
            model.initial_values[p] = np.array(map[0][p])

        map_dict = dict()
        for p in fit_params:
            map_dict[p] = map[0][p]

        # chi2func should use model instead
        scale, bestpars = exozippy_getmcmcscale(map_dict, chi2func)

        # Turn each value in to a single element array... Why?
        for key, val in bestpars.items():
            bestpars[key] = np.array(val)

        # huh?
        apoint = DictToArrayBijection.map(bestpars)
        n = len(apoint.data)
        scaling = fmt_scale(model, scale)

        # this initializes the step scaling
        potential = quadpotential.QuadPotentialDiagAdapt(n, apoint.data, scaling, 10)
        nuts = pm.NUTS(target_accept=0.99, potential=potential, max_treedepth=12)

        # now I can actually sample
        trace = pm.sample(tune=1500, draws=1000, initvals=init_points, step=nuts, chains=36, cores=12,
                          target_accept=0.99, discard_tuned_samples=False)

        # copy posteriors to event dictionary
        trace_to_event(trace, event)
        build_latex_table(event)
        ipdb.set_trace()
