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
import star

'''This function is analagous to exofastv2's mkss.pro, but also defines
some of the fundamental relationships between parameters done in
step2pars.pro and imposes boundaries as in exofastv2_chi2.pro.

It builds the event dictionary (analagous to the stellar structure
(SS) in exofast, which has arrays of dictionaries for each object we
might care about (stars, planets, light curves, instruments,
astrometry, etc)

Each of those objects has the appropriate parameters associated with
them and can be replicated with ease for trivial generalization

'''


def build_model(nstars=1, nplanets=1,
                parfile=None,
                tranpath=None, fittran=False,
                rvpath=None, fitrv=False,
                ulenspath=None,
                fitlogmp=False,
                circular=False,
                mist=True, parsec=False, mannrad=False, mannmass=False,
                sedfile=None, teffsedfloor=0.02, fbolsedfloor=0.024,
                ttvs=False, tivs=False, tdeltavs=False,
                novcve=False, nochord=False, i180=False
                ):
    # override some defaults for development/testing
    mist = False
    nastrom = 0

    if tranpath is not None:
        tranfiles = glob.glob(tranpath)
    else:
        tranfiles = []
    ntransits = len(tranfiles)

    if rvpath is not None:
        rvfiles = glob.glob(rvpath)
    else:
        rvfiles = []
    ninstruments = len(rvfiles)

    if ulenspath is not None:
        ulensfiles = glob.glob(ulenspath)
    else:
        ulensfiles = []
    nulens = len(ulensfiles)

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

    # set defaults
    if fitrv is None:
        if ninstruments > 0:
            fitrv = np.zeros((nplanets), dtype=bool) + True
        else:
            fitrv = np.zeros((nplanets), dtype=bool) + False
    elif np.isscalar(fitrv):
        fitrv = np.zeros((nplanets), dtype=bool) + fitrv
    elif len(fitrv) != nplanets:
        print("fitrv must be a scalar or a list with NPLANETS elements")

    if fittran is None:
        if ntransits > 0:
            fittran = np.zeros((nplanets), dtype=bool) + True
        else:
            fittran = np.zeros((nplanets), dtype=bool) + False
    elif np.isscalar(fittran):
        fittran = np.zeros((nplanets), dtype=bool) + fittran
    elif len(fittran) != nplanets:
        print("fittran must be a scalar or a list with NPLANETS elements")

    if i180 is None:
        if nastrom > 0:
            i180 = np.zeros((nplanets), dtype=bool) + True
        else:
            fittran = np.zeros((nplanets), dtype=bool) + False
    elif np.isscalar(i180):
        i180 = np.zeros((nplanets), dtype=bool) + i180
    elif len(i180) != nplanets:
        print("i180 must be a scalar or a list with NPLANETS elements")

    if np.isscalar(circular):
        circular = np.zeros((nplanets), dtype=bool) + circular
    elif len(circular) != nplanets:
        print("circular must be a scalar or a list with NPLANETS elements")

    if np.isscalar(ttvs):
        ttvs = np.zeros((ntransits), dtype=bool) + ttvs
    elif len(ttvs) != ntransits:
        print("ttvs must be a scalar or a list with NTRANSITS elements")

    if np.isscalar(tdeltavs):
        tdeltavs = np.zeros((ntransits), dtype=bool) + tdeltavs
    elif len(tdeltavs) != ntransits:
        print("tdeltavs must be a scalar or a list with NTRANSITS elements")

    if np.isscalar(tivs):
        tivs = np.zeros((ntransits), dtype=bool) + tivs
    elif len(tivs) != ntransits:
        print("tivs must be a scalar or a list with NTRANSITS elements")

    if np.isscalar(mannrad):
        mannrad = np.zeros((nstars), dtype=bool) + mannrad
    elif len(mannrad) != nstars:
        print("mannrad must be a scalar or a list with NSTARS elements")

    if np.isscalar(mannmass):
        mannmass = np.zeros((nstars), dtype=bool) + mannmass
    if len(mannmass) != nstars:
        print("mannmass must be a scalar or a list with NSTARS elements")

    if np.isscalar(mist):
        mist = np.zeros((nstars), dtype=bool) + mist
    if len(mist) != nstars:
        print("mist must be a scalar or a list with NSTARS elements")

    if np.isscalar(parsec):
        parsec = np.zeros((nstars), dtype=bool) + parsec
    if len(parsec) != nstars:
        print("parsec must be a scalar or a list with NSTARS elements")

    if parfile is not None:
        user_params = read_parfile(parfile)
    else:
        user_params = {}

    event = {}
    with pm.Model() as model:

        event = Event()
        for i in range(nstars):

        # add per event parameters
        if sedfile is not None:
            event["sigmased"] = Parameter("sigmased", lower=1e-3, upper=1000.0, initval=1.0,  # unitless
                                          latex='\sigma_{SED}', description='photometric error scaling',
                                          latex_unit="", user_params=user_params)
        # add per star parameters
        starnames = list(string.ascii_uppercase)  # [A,B,C... Z]
        event["star"] = []
        for i in range(nstars):
            # the lower limits mean it doesn't actually have to be a star
            # the upper limits are defined by the most extreme stars known (plus some padding)
            star = {"rootlabel": "Stellar Parameters:",
                    "radius": Parameter("rstar_" + str(i), lower=1e-1, upper=2000, initval=1.0, unit=u.solRad,
                                        latex='R_*', description='Radius', latex_unit="\\rsun",
                                        user_params=user_params),
                    "mass": Parameter("mstar_" + str(i), lower=1e-1, upper=250, initval=1.0, unit=u.solMass,
                                      latex='M_*', description='Mass', latex_unit="\msun", user_params=user_params),
                    "teff": Parameter("teff_" + str(i), lower=1.0, upper=5e5, initval=5778, unit=u.K,
                                      latex='T_{\\rm eff}', description="Effective Temperature", latex_unit='K',
                                      user_params=user_params),
                    "feh": Parameter("feh_" + str(i), lower=-5.0, upper=5.0, initval=0.0,
                                     latex="[{\\rm Fe/H}]", description='Metallicity', latex_unit='dex',
                                     user_params=user_params),
                    }

            star["lstar"] = Parameter('lstar_' + str(i),
                                      expression=4.0 * math.pi * rsun * rsun * sigmasb *
                                                 star["radius"].value ** 2 * star["teff"].value ** 4,
                                      unit=u.erg / u.second,
                                      latex='L_*', description='Luminosity', latex_unit='\lsun',
                                      user_params=user_params)

            star["rhostar"] = Parameter('rhostar_' + str(i),
                                        expression=3.0 * star["mass"].value * msun /
                                                   (4.0 * math.pi * (star["radius"].value * rsun) ** 3),
                                        unit=u.gram / u.cm ** 3,
                                        latex="\\rho_*", description='Density', latex_unit='g~cm$^{-3}$',
                                        user_params=user_params)

            star["logg"] = Parameter('logg_' + str(i), unit=u.dex(u.cm / u.s ** 2),
                                     expression=pt.log10(
                                         Gmsun * star["mass"].value / (star["radius"].value * rsun) ** 2),
                                     latex="\log{g_*}", description='Surface gravity', latex_unit='cgs',
                                     user_params=user_params)

            if parsec[i] or mist[i]:
                star["age"] = Parameter("age_" + str(i), lower=0.0, upper=13.77, initval=4.603, unit=u.year * 1e9,
                                        latex="Age", description="Age", latex_unit="Gyr", user_params=user_params)
                star["initfeh"] = Parameter("initfeh_" + str(i), lower=-5.0, upper=5.0, initval=0.0,
                                            latex='[{\\rm Fe/H}]_{0}', description='Initial Metallicity',
                                            latex_unit='dex', user_params=user_params)

            # if we need the distance, add it as a parameter
            if sedfile != None or mannrad[i] or mannmass[i]:
                star["distance"] = Parameter("distance_" + str(i), lower=1e-6, upper=1.426e10, initval=10.0,
                                             unit=u.pc, latex='d', description="Distance", latex_unit='pc',
                                             user_params=user_params)
                star["fbol"] = Parameter("fbol_" + str(i),
                                         expression=star["lstar"].value /
                                                    (4.0 * math.pi * (star["distance"].value * pc) ** 2.0),
                                         unit=u.erg / u.second / u.cm ** 2, latex_unit='erg~s$^{-1}$~cm$^{-2}$',
                                         latex="F_{Bol}", description='Bolometric Flux', user_params=user_params)
                star["parallax"] = Parmeter("parallax_" + str(i),
                                            expression=1e3 / star["distance"].value,
                                            unit=u.mas,
                                            latex='\varpi', description='Parallax', latex_unit='mas',
                                            user_params=user_params)

                # apply a potential to constrain the stellar radius according to the Mann+ 2015 K-Rstar relation (mannrad) or
                # constrain the stellar mass according to the Mann+ 2019 K-Mstar relation (mannmass)
                # only applies for 0.1 < mstar/msun < 0.7
                if mannmass[i] or mannrad[i]:
                    star["appks"] = Parameter("appks_" + str(i), lower=-30, upper=30, unit=u.mag,
                                              latex='ks', description="Apparent ks mag", latex_unit='mag',
                                              user_params=user_params)
                    star["absks"] = Parameter("absks_" + str(i),
                                              expression=star["appks"].value - 2.5 * pt.log10(
                                                  (star["distance"].value / 10.0) ** 2),
                                              unit=u.mag, latex='Ks', description="Absolute Ks mag", latex_unit='mag',
                                              user_params=user_params)

                    mann_mstar, mann_rstar, mann_sigma_mstar, mann_sigma_rstar = massradius_mann(star["ks0"].value,
                                                                                                 star["feh"].value,
                                                                                                 distance=star[
                                                                                                     "distance"].value)
            #                    if mannmass[i]:
            #                        mannmass_prior = pm.Potential("mannmass_prior_" + str(i), -0.5 *
            #                                                      ((star["mass"].value - mann_mstar) / mann_sigma_mstar) ** 2)
            #                    if mannrad[i]:
            #                        mannrad_prior = pm.Potential("mannrad_prior_" + str(i), -0.5 *
            #                                                     ((star["radius"].value - mann_rstar) / mann_sigma_rstar) ** 2)

            # add SED parameters for each star
            if sedfile != None:
                star["av"] = Parameter("av_" + str(i), lower=0.0, upper=100.0, initval=0.0,
                                       unit=u.mag,
                                       latex='A_V', description='V-band extinction', latex_unit='mag',
                                       user_params=user_params)

                # respect systematic error floors on fbol and teff as determined from the SED
                star["rstarsed"] = Parameter("rstarsed_" + str(i), lower=1e-9, upper=2000.0, initval=1.0,
                                             unit=u.solRad, latex='R_{*,SED}', latex_unit='\\rsun',
                                             user_params=user_params)
                star["teffsed"] = Parameter("teffsed_" + str(i), lower=1.0, upper=5e6, initval=5778.0,
                                            unit=u.K, latex='T_{eff,SED}', latex_unit='K', user_params=user_params)

                # these potentials link teff/teffsed and fbol/fbolsed within the sedfloors
                # so the SED cannot impact teff/fbol better than the floors
                # see eq ?? of Eastman+ 2019
                fbolsed = Parameter("fbolsed_" + str(i),
                                    expression=star["lstar"].value / (
                                                4.0 * math.pi * (star["distance"].value * pc) ** 2.0),
                                    unit=u.erg / u.second / u.cm ** 2, latex_unit='erg~s$^{-1}$~cm$^{-2}$',
                                    latex="F_{Bol,SED}", description='Bolometric Flux', user_params=user_params,
                                    table_note="This may be systematics dominated and is only for reference",
                                    )
            #                teffsed_floor_prior = pm.Potential("teffsed_floor_prior_" + str(i), -0.5 * (
            #                    (star["teff"].value - star["teffsed"].value) / (star["teff"].value * teffsedfloor)) ** 2)
            #                fbolsed_floor_prior = pm.Potential("fbolsed_floor_prior" + str(i), -0.5 * (
            #                    (star["fbol"].value - fbolsed) / (star["fbol"].value * fbolsedfloor)) ** 2)

            event["star"].append(star)

        # for each planet
        event["planet"] = []
        starndx = 0  # place holder -- this should be a user input with 0 as default
        for i in range(nplanets):

            # physical range is -1 <= cosi <=1
            # but unless we have astrometry or mutual transits, bound should be 0 <= cosi <=1
            if i180[i]:
                cosilower = -1.0
            else:
                cosilower = 0.0

            planet = {"rootlabel": "Planet Parameters",
                      "logp": Parameter("logp_" + str(i), lower=1e-9, upper=13.7, initval=1.0, unit=u.dex(u.day),
                                        latex='P', description='log(Period/day)', latex_unit='',
                                        user_params=user_params),
                      "radius": Parameter("rp_" + str(i), lower=1e-9, upper=2e4, initval=1.0, unit=u.jupiterRad,
                                          latex='R_P', description='Radius', latex_unit=r'\rj',
                                          user_params=user_params),
                      "tco": Parameter("tc_" + str(i), lower=0.0, upper=9e9, initval=2460000.0, unit=u.day,
                                       latex='T_C', description='Time of conjunction', latex_unit=r'\bjdtdb',
                                       user_params=user_params)
                      }

            # choose planet mass parameterization
            if fitlogmp:
                # more physical prior, bad for marginal detections
                planet["logmp"] = Parameter("logmp" + str(i), lower=-10.0, upper=10.0, initval=0.0,
                                            unit=u.dex(u.jupiterMass),
                                            latex='\log{M_P}', description='Log of mp', latex_unit='',
                                            user_params=user_params)
                planet["mp"] = Parameter("mp_" + str(i),
                                         expression=10 ** planet["logmp"], unit=u.jupiterMass,
                                         latex='M_P', description='Mass', latex_unit="\mj", user_params=user_params)
            else:
                # allow negative mass to reduce bias of marginal detections
                planet["mp"] = Parameter("mp_" + str(i), lower=-1e9, upper=1e9, initval=1.0,
                                         latex='M_P', description='Mass', latex_unit="\mj", user_params=user_params)
                # can't have total negative mass
                total_mass = event["star"][starndx]["mass"].value + planet["mp"].value * mjup
            #                massbound = pm.Potential("mp_bound_" + str(i), pt.switch(total_mass > 0.0, 0.0, -np.inf))

            planet["period"] = Parameter("period_" + str(i),
                                         expression=10 ** planet["logp"].value, unit=u.day,
                                         latex='P', description='Period', latex_unit='days', user_params=user_params)

            planet["p"] = Parameter("p_" + str(i),
                                    expression=planet["radius"].value * rjup / event["star"][starndx]["radius"].value,
                                    latex="R_P/R_*", description="Radius of planet in stellar radii",
                                    latex_unit="", user_params=user_params)

            # use Kepler's law to derive semi-major axis
            planet["arsun"] = Parameter("arsun_" + str(i),
                                        expression=pt.power((G * (event["star"][starndx]["mass"].value +
                                                                  planet["mp"].value * mjup) * planet[
                                                                 "period"].value ** 2 /
                                                             (4.0 * math.pi ** 2)), 1.0 / 3.0),
                                        latex="a/R_{\sun}", description="Semi-major axis in \rsun", latex_unit='\\rsun',
                                        user_params=user_params)

            planet["ar"] = Parameter("ar_" + str(i),
                                     expression=planet["arsun"].value / event["star"][starndx]["radius"].value,
                                     latex='a/R_*', description='Semi-major axis in $R_*$', latex_unit='',
                                     user_params=user_params)
            planet["a"] = Parameter("a_" + str(i),
                                    expression=planet["arsun"].value / AU, unit=u.au,
                                    latex='a', description='Semi-major axis', latex_unit='au', user_params=user_params)

            ######### choose eccentricity parameterization ###########
            if circular[i]:
                planet["cosi"] = Parameter("cosi_" + str(i), lower=cosilower, upper=1.0,
                                           latex='\cos{i}', description='cos of inclination',
                                           latex_unit='', user_params=user_params)
            elif (not fitrv[i]) and fittran[i] and (not novcve[i]):
                # transit only parameterization
                # vcve, sin(omega), cos(omega), sign, chord
                planet["vcve"] = Parameter("vcve_" + str(i), lower=0.0, initval=1.0,  # upper limit implicit from e
                                           latex='V_c/V_e', description='Scaled velocity',
                                           latex_unit='', user_params=user_params)

                planet["cosw"] = Parameter("cosw_" + str(i), lower=-1.0, upper=1.0, initval=0.0,
                                           latex='\cos{\omega_*}', description='Cos of arg of periastron',
                                           latex_unit='', user_params=user_params)

                planet["sinw"] = Parameter("sinw_" + str(i), lower=-1.0, upper=1.0, initval=0.0,
                                           latex='\sin{\omega_*}', description='Sin of arg of periastron',
                                           latex_unit='', user_params=user_params)

                # bound omega
                L = planet["cosw"].value ** 2 + planet["sinw"].value ** 2
                # ensures uniform omega distribution
                #                wbound = pm.Potential("wbound_" + str(i), pt.switch(L > 1.0,-np.inf,0.0))

                planet["omega"] = Parameter("omega_" + str(i), unit=u.rad,
                                            expression=pt.arctan2(planet["sinw"], planet["cosw"]),
                                            latex='\omega_*', description='Arg of periastron',
                                            latex_unit='rad', user_params=user_params)
                planet["sign"] = Parameter("sign_" + str(i), lower=-1.0, upper=1.0,  # binary?
                                           latex='Sign', description='Sign of quadratic solution',
                                           latex_unit='', user_params=user_params)

                # solve quadratic for e (Eastman 2024, eq 5)
                a = planet["vcve"].value ** 2 * planet["sinw"].value ** 2 + 1.0
                b = 2.0 * planet["vcve"].value ** 2 * planet["sinw"].value
                c = planet["vcve"].value ** 2 - 1.0
                planet["e"] = Parameter('e_' + str(i),
                                        expression=-b + pt.sign(planet["sign"].value) * pt.sqrt(
                                            b ** 2 - 4.0 * a * c) / (2.0 * a),
                                        latex='e', description='eccentricity',
                                        latex_unit='', user_params=user_params)

                planet["esinw"] = Parameter("esinw_" + str(i),
                                            expression=planet["e"].value * planet["sinw"].value,
                                            latex='e\sin{\omega_*}', description='e times sin of arg of periastron',
                                            latex_unit='', user_params=user_params)

                jacobian = 1.0
                if not nochord[i]:
                    planet["chord"] = Parameter("chord_" + str(i), lower=0.0,
                                                latex='chord', description='transit chord',
                                                latex_unit='', user_params=user_params)
                    planet["b"] = Parameter("b_" + str(i),
                                            expression=pt.sqrt((1.0 + planet["p"]) ** 2 - planet["chord"] ** 2),
                                            latex='b', description='impact parameter',
                                            latex_unit='', user_params=user_params)
                    planet["cosi"] = Parameter("cosi_" + str(i),
                                               expression=planet["b"].value /
                                                          (planet["ar"].value * (1.0 - planet["e"].value ** 2) / (
                                                                      1.0 + planet["esinw"].value)),
                                               latex='\cos{i}', description='cos of inclination',
                                               latex_unit='', user_params=user_params)

                    jacobian *= planet["b"].value ** 2 / (
                                planet["cosi"].value * planet["chord"].value)  # d(chord)/d(cosi)
                else:
                    planet["cosi"] = Parameter("cosi_" + str(i), lower=0.0, upper=1.0,
                                               latex='\cos{i}', description='cos of inclination',
                                               latex_unit='', user_params=user_params)
                    planet["b"] = Parameter("b_" + str(i),
                                            expression=planet["ar"].value * planet["cosi"].value * (
                                                        1.0 - planet["e"].value ** 2) / (1.0 - planet["esinw"].value),
                                            latex='b', description='impact parameter',
                                            latex_unit='', user_params=user_params)

                # correct the prior to be uniform in e/omega (Eastman 2024, eq 6)
                jacobian *= (planet["e"].value + planet["sinw"].value) / (
                        pt.sqrt(1.0 - planet["e"].value ** 2) * (1.0 + planet["esinw"].value) ** 2)  # d(vcve)/d(e)

            #                planet["jacobian"] = pm.Potential("jacobian_" + str(i), pt.abs(jacobian))
            else:
                # sqrt(e)*cos(omega), sqrt(e)*sin(omega), cosi
                planet["secosw"] = Parameter("secosw_" + str(i), lower=-1.0, upper=1.0, initval=0.0,
                                             latex='e^{1/2}\cos{\omega_*}', description='sqrt(e) cos(omega)',
                                             latex_unit='', user_params=user_params)

                planet["sesinw"] = Parameter("sesinw_" + str(i), lower=-1.0, upper=1.0, initval=0.0,
                                             latex='e^{1/2}\sin{\omega_*}', description='sqrt(e) sin(omega)',
                                             latex_unit='', user_params=user_params)
                planet["cosi"] = Parameter("cosi_" + str(i), lower=cosilower, upper=1.0, initval=0.0,
                                           latex='\cos{i}', description='cos of inclination',
                                           latex_unit='', user_params=user_params)
                planet["e"] = Parameter("e_" + str(i),
                                        expression=planet["secosw"].value ** 2 + planet["sesinw"].value ** 2,
                                        latex='e', description='eccentricity',
                                        latex_unit='', user_params=user_params)
                planet["omega"] = Parameter("omega_" + str(i),
                                            expression=pt.arctan2(planet["sesinw"].value, planet["secosw"].value),
                                            unit=u.rad,
                                            latex='\omega_*', description='Arg of periastron',
                                            latex_unit='rad', user_params=user_params)
                planet["esinw"] = Parameter("esinw_" + str(i),
                                            expression=planet["e"].value * pt.sin(planet["omega"].value),
                                            latex='e\sin{\omega_*}', description='e times sin of arg of periastron',
                                            latex_unit='', user_params=user_params)
                planet["b"] = Parameter("b_" + str(i),
                                        expression=planet["ar"].value * planet["cosi"].value * (
                                                    1.0 - planet["e"].value ** 2) / (1.0 - planet["esinw"].value),
                                        latex='b', description='impact parameter',
                                        latex_unit='', user_params=user_params)

            # reject models where the planet collides with the star at periastron
            # this does four important things
            #   1) bounds e within the unit circle (uniform omega prior)
            #   2) rejects non-physical (NaN) solutions to the quadratic for vcve parameterization
            #   3) explicit physical upper limit on e
            #   4) implicit physical lower limit on a, period
            maxe = 1.0 - 1.0 / planet["ar"].value - planet["p"].value / planet["ar"].value  # 1 - (Rp+Rstar)/a
            #            ebound = pm.Potential("ebound_" + str(i), pt.switch(planet["e"].value < maxe, 0.0, -np.inf))

            # derive other parameters of interest
            planet["inc"] = Parameter("inc_" + str(i),
                                      expression=pt.arccos(planet["cosi"].value),
                                      latex='i', unit=u.rad, description='inclination',
                                      latex_unit='rad', user_params=user_params)
            # check units here!
            planet["k"] = Parameter("k_" + str(i),
                                    expression=(2.0 * math.pi * G / (planet["period"].value * (
                                                event["star"][starndx]["mass"].value + planet[
                                            "mp"].value * mjup) ** 2.0)) ** (1.0 / 3.0) *
                                               planet["mp"].value * mjup * pt.sin(planet["inc"].value) / pt.sqrt(
                                        1.0 - planet["e"].value ** 2.0) *
                                               rsun / meter / 86400.0,
                                    unit=u.meter / u.second, latex_unit='m~s$^{-1}$',
                                    latex='K', description='RV semi-amplitude',
                                    user_params=user_params)

            event["planet"].append(planet)

        # read in transit files
        event["transit"] = []
        for i, tranfile in enumerate(tranfiles):
            event["transit"].append(
                readtran(tranfile, ndx=i, ttv=ttvs[i], tdeltav=tdeltavs[i], tiv=tivs[i]))  # , user_params=user_params)

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
