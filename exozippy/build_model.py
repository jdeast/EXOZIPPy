import pymc as pm
import numpy as np
from astropy import units as u
import astropy.constants as const
from parameter import Parameter
# import exoplanet as xo
import arviz as az
import math
from summarize_model import summarize_model
import pytensor.tensor as pt

import ipdb

sigma = 0.1
X = np.linspace(0, 10, num=11) * np.random.normal() * sigma
m = 0.1
b = 0.5
Y = m * X + b

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


def build_transit_model():
    pass


def build_rv_model():
    pass


def build_dt_model():
    pass


def build_sed_model():
    pass


def build_mist_model():
    pass


def build_parsec_model():
    pass


def build_astrometry_model():
    pass


def build_torres_model():
    pass


def build_mann_model():
    pass


def build_model(nstars=1, nplanets=1, transit_only=False, fitlogmp=False, mist=True, parsec=False, mannrad=False, mannmass=False,
                sedfile=None, teffsedfloor=0.02, fbolsedfloor=0.024, ):
    # override some defaults for development/testing
    mist=False
    nplanets=0

    # constants easier to digest variable names
    G = const.GM_sun.value / const.R_sun.value ** 3 * 86400.0 ** 2
    AU = const.au.value / const.R_sun.value
    mjup = const.GM_jup.value / const.GM_sun.value
    pc = const.pc.cgs.value  # cm/pc
    rsun = const.R_sun.cgs.value  # cm/r_sun
    msun = const.M_sun.cgs.value # g/m_sun
    sigmasb = const.sigma_sb.cgs.value
    Gmsun = const.GM_sun.cgs.value

    # set defaults
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

    event = {}
    with pm.Model() as model:

        # add per event parameters
        event["sigmased"] = pm.Uniform("sigmased", lower=1e-3, upper=1000.0)  # unitless

        # add per star parameters
        event["star"] = []
        for i in range(nstars):
            # the lower limits mean it doesn't actually have to be a star
            # the upper limits are defined by the most extreme stars known (plus some padding)
            star = {"radius": Parameter(label="rstar_" + str(i), lower=1e-1, upper=2, initval=1.0, unit=u.solRad,
                                        latex='R_*', description='Radius', latex_unit="\rsun"),
                    "mass": Parameter(label="mstar_" + str(i), lower=1e-1, upper=2, initval=1.0, unit=u.solMass,
                                      latex='M_*', description='Mass', latex_unit="\msun"),
                    "teff": Parameter(label="teff_" + str(i), lower=1.0, upper=5e5, initval=5778, unit=u.K,
                                      latex='T_{\rm eff}', description="Effective Temperature", latex_unit='K'),
                    "feh": Parameter(label="feh_" + str(i), lower=-5.0, upper=5.0, initval=0.0,
                                     latex="[{\rm Fe/H}]", description='Metallicity', latex_unit='dex'),
                    }

            star["lstar"] = Parameter(label='lstar_' + str(i), Deterministic=True,
                                      expression=4.0 * math.pi * rsun * rsun * sigmasb *
                                                 star["radius"].value ** 2 * star["teff"].value ** 4,
                                      unit=u.erg / u.second,
                                      latex='L_*', description='Luminosity', latex_unit='\lsun')

            star["rhostar"] = Parameter(label='rhostar_' + str(i), Deterministic=True,
                                        expression=3.0*star["mass"].value*msun/
                                                   (4.0*math.pi*(star["radius"].value*rsun) ** 3),
                                        unit=u.gram/u.cm**3,
                                        latex="\rho_*", description='Density', latex_unit='g~cm$^{-3}$')

            star["logg"] = Parameter(label='logg_' + str(i), Deterministic=True,
                                     expression = pt.log10(Gmsun * star["mass"].value / (star["radius"].value * rsun) ** 2),
                                     latex="\log{g_*}", description='Surface gravity', latex_unit='cgs')

            if parsec[i] or mist[i]:
                star["age"] = Parameter(label="age_" + str(i), lower=0.0, upper=13.77, initval=4.603,unit=u.year*1e9,
                                        latex="Age",description="Age",latex_unit="Gyr")
                star["initfeh"] = Parameter(label="initfeh_" + str(i), lower=-5.0, upper=5.0, initval=0.0,
                                            latex='[{\rm Fe/H}]_{0}', description='Initial Metallicity', latex_unit='dex')

            # if we need the distance, add it as a parameter
            if sedfile != None or mannrad[i] or mannmass[i]:
                star["distance"] = Parameter(label="distance_" + str(i), lower=1e-6, upper=1.426e10, initval=10.0,
                                             unit=u.pc, latex='d',description="Distance",latex_unit='pc')
                star["fbol"] = Parameter(label="fbol_" + str(i), Deterministic=True,
                                         expression= star["lstar"].value /
                                         (4.0 * math.pi * (star["distance"].value * pc) ** 2.0),
                                         unit=u.erg/u.second/u.cm**2,latex_unit='erg~s$^{-1}$~cm$^{-2}$',
                                         latex="F_{Bol}",description='Bolometric Flux')
                star["parallax"] = Parmeter(label="parallax_" + str(i),Deterministic=True,
                                            expression= 1e3 / star["distance"].value,
                                            unit=u.mas,
                                            latex='\varpi', description='Parallax', latex_unit='mas')

            # add SED parameters for each star
            if sedfile != None:
                star["av"] = Parameter(label="av_" + str(i), lower=0.0, upper=100.0, initval=0.0,
                                       unit=u.mag,
                                       latex='A_V', description='V-band extinction', latex_unit='mag')

                # respect systematic error floors on fbol and teff as determined from the SED
                star["rstarsed"] = Parameter(label="rstarsed_" + str(i), lower=1e-9, upper=2000.0, initval=1.0,
                                             unit=u.solRad, latex='R_{*,SED}',latex_unit='\rsun')
                star["teffsed"] = Parameter(label="teffsed_" + str(i), lower=1.0, upper=5e6,initval=5778.0,
                                            unit=u.K, latex='T_{eff,SED}', latex_unit='K')

                # these potentials link teff/teffsed and fbol/fbolsed within the sedfloors
                # so the SED cannot impact teff/fbol better than the floors
                # see eq ?? of Eastman+ 2019
                fbolsed = star["lstar"].value / (4.0 * math.pi * (star["distance"].value * pc) ** 2.0)  # erg/s/cm^2
                teffsed_floor_prior = pm.Potential("teffsed_floor_prior_" + str(i), -0.5 * (
                        (star["teff"].value - star["teffsed"].value) / (star["teff"].value * teffsedfloor)) ** 2)
                fbolsed_floor_prior = pm.Potential("fbolsed_floor_prior" + str(i), -0.5 * (
                        (star["fbol"].value - fbolsed) / (star["fbol"].value * fbolsedfloor)) ** 2)

            event["star"].append(star)

        # for each planet
        event["planet"] = []
        starndx = 0
        for i in range(nplanets):
            planet = {
                # 15 minutes to age of the universe
                "logp": pm.Uniform("logp_" + str(i), lower=1e-9, upper=13.7),
                # stricter lower limit imposed by e, upper limit is age of universe
                "radius": pm.Uniform("rp_" + str(i), lower=1e-9, upper=2e4),  # r_jupiter
                # "tco" : pm.Uniform("tc_"+str(i), lower=0.0, upper=9e9), # days, need to bound to +/- period/2
            }

            # choose planet mass parameterization
            if fitlogmp:
                # more physical prior, bad for marginal detections
                planet["logmp"] = pm.Uniform("logmp" + str(i), lower=-10.0, upper=10.0)  # log10(mp/m_jupiter)
                planet["mp"] = pm.Deterministic("mp_" + str(i), 10 ** planet["logmp"])  # m_jupiter
            else:
                # allow negative mass to reduce bias of marginal detections
                planet["mp"] = pm.Uniform("mp_" + str(i), lower=-1e9, upper=1e9, initval=1.0)  # m_jupiter
                # can't have total negative mass
                massbound = pm.Potential("mp_bound_" + str(i),
                                         pt.switch(pt.lt(planet["mp"] * mjup + event["star"][starndx]["mass"], 0),
                                                   -np.inf, 0.0))

            planet["period"] = pm.Deterministic("period_" + str(i), 10 ** planet["logp"])
            planet["p"] = pm.Deterministic("p_" + str(i), planet["radius"] / event["star"][starndx]["radius"])

            # use Kepler's law to derive semi-major axis
            planet["arsun"] = pm.Deterministic("arsun_" + str(i), pt.power((G * (
                    event["star"][starndx]["mass"] + planet["mp"] * mjup) * planet["period"] ** 2 / (
                                                                                    4.0 * math.pi ** 2)),
                                                                           1.0 / 3.0))
            planet["ar"] = pm.Deterministic("ar_" + str(i), planet["arsun"] / event["star"][starndx]["radius"])
            planet["a"] = pm.Deterministic("a_" + str(i), planet["arsun"] / AU)

            ######### choose eccentricity parameterization ###########
            if transit_only:
                # vcve, sin(omega), cos(omega), sign, chord
                planet["vcve"] = pm.Uniform("vcve_" + str(i), lower=0.0)  # upper limit imposed by e
                planet["cosw"] = pm.Uniform("cosw_" + str(i), lower=-1.0, upper=1.0)
                planet["sinw"] = pm.Uniform("sinw_" + str(i), lower=-1.0, upper=1.0)

                # bound omega
                L = planet["cosw"] ** 2 + planet["sinw"] ** 2
                # ensures uniform omega distribution
                wbound = pm.Potential("wbound_" + str(i), pt.switch(pt.lt(L, 1.0), 0.0, -np.inf))
                planet["omega"] = pm.Deterministic("omega_" + str(i), pt.arctan2(planet["sinw"], planet["cosw"]))

                planet["chord"] = pm.Uniform("chord_" + str(i), lower=0.0, upper=10.0)
                planet["sign"] = pm.Uniform("sign_" + str(i), lower=-1.0, upper=1.0)  # binary?
                planet["esinw"] = pm.Deterministic("esinw_" + str(i), planet["e"] * planet["sinw"])

                # solve quadratic for e (Eastman+ 2024, eq 5)
                a = planet["vcve"] ** 2 * planet["sinw"] ** 2 + 1.0
                b = 2.0 * planet["vcve"] ** 2 * planet["sinw"]
                c = planet["vcve"] ** 2 - 1.0
                planet["e"] = pm.Deterministic(-b + pt.sign(planet["sign"]) * pt.sqrt(b ** 2 - 4.0 * a * c) / (2.0 * a))
                planet["omega"] = pm.Deterministic("omega_" + str(i), pt.arctan2(planet["sinw"], planet["cosw"]))
                planet["b"] = pm.Deterministic("b_" + str(i), pt.sqrt((1.0 + planet["p"]) ** 2 - planet["chord"] ** 2))
                planet["cosi"] = pm.Deterministic("cosi_" + str(i), planet["b"] / (
                        planet["ar"] * (1.0 - planet["e"] ** 2) / (1.0 + planet["esinw"])))

                # correct the prior to be uniform in e/omega (Eastman+ 2024, eq 6)
                dvcvede = (planet["e"] + planet["sinw"]) / (
                        pt.sqrt(1.0 - planet["e"] ** 2) * (1.0 + planet["esinw"]) ** 2)
                dchorddcosi = planet["b"] ** 2 / (planet["cosi"] * planet["chord"])
                planet["jacobian"] = pm.Potential("jacobian_" + str(i), pt.abs(dvcvede * dchorddcosi))
            else:
                # sqrt(e)*cos(omega), sqrt(e)*sin(omega), cosi
                planet["secosw"] = pm.Uniform("secosw_" + str(i), lower=-1.0, upper=1.0)
                planet["sesinw"] = pm.Uniform("sesinw_" + str(i), lower=-1.0, upper=1.0)
                planet["e"] = pm.Deterministic("e_" + str(i), planet["secosw"] ** 2 + planet["sesinw"] ** 2)
                planet["omega"] = pm.Deterministic("omega_" + str(i), pt.arctan2(planet["sesinw"], planet["secosw"]))
                # physical range is -1 <= cosi <=1
                # but unless we have astrometry or mutual transits, bound should be 0 <= cosi <=1
                planet["cosi"] = pm.Uniform("cosi_" + str(i), lower=0.0, upper=1.0)

            # reject models where the planet collides with the star at periastron 
            # this does four important things
            #   1) bounds e within the unit circle (uniform omega prior)
            #   2) rejects non-physical (NaN) solutions to the quadratic for vcve parameterization
            #   3) explicit physical upper limit on e
            #   4) implicit physical lower limit on a, period
            maxe = 1.0 - 1.0 / planet["ar"] - planet["p"] / planet["ar"]
            ebound = pm.Potential("ebound_" + str(i), pt.switch(pt.lt(planet["e"], maxe), 0.0, -np.inf))

            # derive other parameters of interest
            planet["inc"] = pm.Deterministic("inc_" + str(i), pt.arccos(planet["cosi"]))

            event["planet"].append(planet)

        # apply priors (eventually this will come from the parfile)
        #periodprior = pm.Potential("period_prior", -0.5 * ((event["planet"][0]["period"].value - 3.0) / 0.00001) ** 2)
        mstarprior = pm.Potential("mstar_prior", -0.5 * ((event["star"][0]["mass"].value - 1.0) / 0.03) ** 2)

        # pm.traceplot(trace)

        # model = pm

        # default method='L-BFGS-B', other options are powell, amoeba, etc
        # find the initial best-fit
        # map_estimate = pm.find_MAP(model=model) 

        # MCMC sampling
        trace = pm.sample(1000, chains=4, cores=4, target_accept=0.9)  # ,return_inferencedata=True)
        ipdb.set_trace()

        summarize_model(trace, prefix='test.')
        ipdb.set_trace()

    if 0:
        a = pm.Deterministic("a", np.power(period ** 2 * const.G * (mstar + mp) / (4.0 * math.pi ** 2), 1.0 / 3.0))

        # for each light curve
        # limb darkening
        u_star = xo.QuadLimbDark("u_star")
        star = xo.LimbDarkLightCurve(u_star)
        mean_flux = pm.Uniform("mean_flux", lower=0.0, upper=1e9)

        # add priors with Potential

        # Orbital parameters for the planets
        t0 = pm.Normal("t0", mu=np.array(t0s), sd=1, shape=2)
        log_m_pl = pm.Normal("log_m_pl", mu=np.log(msini.value), sd=1, shape=2)
        log_period = pm.Normal("log_period", mu=np.log(periods), sd=1, shape=2)

        # Fit in terms of transit depth (assuming b<1)
        b = pm.Uniform("b", lower=0, upper=1, shape=2)
        log_depth = pm.Normal(
            "log_depth", mu=np.log(depths), sigma=2.0, shape=2
        )
        ror = pm.Deterministic(
            "ror",
            star.get_ror_from_approx_transit_depth(
                1e-3 * tt.exp(log_depth), b
            ),
        )
        r_pl = pm.Deterministic("r_pl", ror * r_star)

        m_pl = pm.Deterministic("m_pl", tt.exp(log_m_pl))
        period = pm.Deterministic("period", tt.exp(log_period))

        ecs = pmx.UnitDisk("ecs", shape=(2, 2), testval=0.01 * np.ones((2, 2)))
        ecc = pm.Deterministic("ecc", tt.sum(ecs ** 2, axis=0))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
        xo.eccentricity.vaneylen19(
            "ecc_prior", multi=True, shape=2, fixed=True, observed=ecc
        )

        # RV jitter & a quadratic RV trend
        log_sigma_rv = pm.Normal(
            "log_sigma_rv", mu=np.log(np.median(yerr_rv)), sd=5
        )
        trend = pm.Normal(
            "trend", mu=0, sd=10.0 ** -np.arange(3)[::-1], shape=3
        )

        # Transit jitter & GP parameters
        log_sigma_lc = pm.Normal(
            "log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10
        )
        log_rho_gp = pm.Normal("log_rho_gp", mu=0.0, sd=10)
        log_sigma_gp = pm.Normal(
            "log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10
        )

        # Orbit models
        orbit = xo.orbits.KeplerianOrbit(
            r_star=r_star,
            m_star=m_star,
            period=period,
            t0=t0,
            b=b,
            m_planet=xo.units.with_unit(m_pl, msini.unit),
            ecc=ecc,
            omega=omega,
        )

        # Compute the model light curve
        light_curves = (
                star.get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp)
                * 1e3
        )
        light_curve = pm.math.sum(light_curves, axis=-1) + mean_flux
        resid = y[mask] - light_curve

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            Q=1 / np.sqrt(2),
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
        gp.marginal("transit_obs", observed=resid)

        # And then include the RVs as in the RV tutorial
        x_rv_ref = 0.5 * (x_rv.min() + x_rv.max())

        def get_rv_model(t, name=""):
            # First the RVs induced by the planets
            vrad = orbit.get_radial_velocity(t)
            pm.Deterministic("vrad" + name, vrad)

            # Define the background model
            A = np.vander(t - x_rv_ref, 3)
            bkg = pm.Deterministic("bkg" + name, tt.dot(A, trend))

            # Sum over planets and add the background to get the full model
            return pm.Deterministic(
                "rv_model" + name, tt.sum(vrad, axis=-1) + bkg
            )

        # Define the model
        rv_model = get_rv_model(x_rv)
        get_rv_model(t_rv, name="_pred")

        # The likelihood for the RVs
        err = tt.sqrt(yerr_rv ** 2 + tt.exp(2 * log_sigma_rv))
        pm.Normal("obs", mu=rv_model, sd=err, observed=y_rv)

        # Compute and save the phased light curve models
        pm.Deterministic(
            "lc_pred",
            1e3
            * tt.stack(
                [
                    star.get_light_curve(
                        orbit=orbit, r=r_pl, t=t0[n] + phase_lc, texp=texp
                    )[..., n]
                    for n in range(2)
                ],
                axis=-1,
            ),
        )

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        map_soln = pmx.optimize(start=start, vars=[trend])
        map_soln = pmx.optimize(start=map_soln, vars=[log_sigma_lc])
        map_soln = pmx.optimize(start=map_soln, vars=[log_depth, b])
        map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
        map_soln = pmx.optimize(
            start=map_soln, vars=[log_sigma_lc, log_sigma_gp]
        )
        map_soln = pmx.optimize(start=map_soln, vars=[log_rho_gp])
        map_soln = pmx.optimize(start=map_soln)

        extras = dict(
            zip(
                ["light_curves", "gp_pred"],
                pmx.eval_in_model([light_curves, gp.predict(resid)], map_soln),
            )
        )

        return model, map_soln, extras


model0, map_soln0, extras0 = build_model()


class Event:
    pass
