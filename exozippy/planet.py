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

from parameter import Parameter

class Planet(name, star, i180=False, fitlogmp=False, circular=False,
             fitrv=False, fittran=False,
             fitvcve=False, fitchord=False, fitsign=False):

    # constants easier to digest variable names
    G = const.GM_sun.value / const.R_sun.value ** 3 * 86400.0 ** 2
    AU = const.au.value / const.R_sun.value
    mjup = const.GM_jup.value / const.GM_sun.value
    rjup = const.R_jup.cgs.value / const.R_sun.cgs.value  # rsun/rjup
    meter = 100.0  # cm/m

    # physical range is -1 <= cosi <=1
    # but unless we have astrometry or mutual transits, bound should be 0 <= cosi <=1
    if i180:
        cosilower = -1.0
    else:
        cosilower = 0.0

    planet.rootlabel = "Planet Parameters"

    planet.logp = Parameter("logp_" + name, lower=1e-9, upper=13.7, initval=1.0, unit=u.dex(u.day),
                            latex='P', description='log(Period/day)', latex_unit='',
                            user_params=user_params),
    planet.radius = Parameter("rp_" + name, lower=1e-9, upper=2e4, initval=1.0, unit=u.jupiterRad,
                              latex='R_P', description='Radius', latex_unit=r'\rj',
                              user_params=user_params),
    planet.tco = Parameter("tc_" + name, lower=0.0, upper=9e9, initval=2460000.0, unit=u.day,
                           latex='T_C', description='Time of conjunction', latex_unit=r'\bjdtdb',
                           user_params=user_params)

    # choose planet mass parameterization
    if fitlogmp:
        # more physical prior, bad for marginal detections
        planet.logmp = Parameter("logmp" + name, lower=-10.0, upper=10.0, initval=0.0,
                                 unit=u.dex(u.jupiterMass), latex='\log{M_P}', description='Log of mp', latex_unit='',
                                 user_params=user_params)
        planet.mp = Parameter("mp_" + name,
                              expression=10 ** planet.logmp, unit=u.jupiterMass,
                              latex='M_P', description='Mass', latex_unit="\mj", user_params=user_params)
    else:
        # allow negative mass to reduce bias of marginal detections
        planet.mp = Parameter("mp_" + name, lower=-1e9, upper=1e9, initval=1.0,
                             latex='M_P', description='Mass', latex_unit="\mj", user_params=user_params)

        # can't have negative total (star+planet) mass
        total_mass = star.mass.value + planet.mp.value * mjup
        massbound = pm.Potential("mp_bound_" + name, pt.switch(total_mass > 0.0, 0.0, -np.inf))

    planet.period = Parameter("period_" + name,
                              expression=10 ** planet.logp.value, unit=u.day,
                              latex='P', description='Period', latex_unit='days', user_params=user_params)

    planet.p = Parameter("p_" + name,
                         expression=planet.radius.value * rjup / star.radius.value,
                         latex="R_P/R_*", description="Radius of planet in stellar radii",
                         latex_unit="", user_params=user_params)

    # use Kepler's law to derive semi-major axis
    planet.arsun = Parameter("arsun_" + name,
                             expression=pt.power((G * (star.mass.value + planet.mp.value * mjup) *
                                                  planet.period.value ** 2 / (4.0 * math.pi ** 2)), 1.0 / 3.0),
                             latex="a/R_{\sun}", description="Semi-major axis in \rsun", latex_unit='\\rsun',
                             user_params=user_params)

    planet.ar = Parameter("ar_" + name,
                          expression=planet.arsun.value / star.radius.value,
                          latex='a/R_*', description='Semi-major axis in $R_*$', latex_unit='',
                          user_params=user_params)

    planet.a = Parameter("a_" + name,
                         expression=planet.arsun.value / AU, unit=u.au,
                         latex='a', description='Semi-major axis', latex_unit='au',
                         user_params=user_params)

    ######### choose eccentricity parameterization ###########
    if circular:
        planet.cosi = Parameter("cosi_" + name, lower=cosilower, upper=1.0,
                                latex='\cos{i}', description='cos of inclination',
                                latex_unit='', user_params=user_params)

    elif fitvcve:
        # transit only parameterization
        # vcve, sin(omega), cos(omega), sign, chord
        planet.vcve = Parameter("vcve_" + name, lower=0.0, initval=1.0,
                                latex='V_c/V_e', description='Scaled velocity',
                                latex_unit='', user_params=user_params)

        planet.cosw = Parameter("cosw_" + name, lower=-1.0, upper=1.0, initval=0.0,
                                latex='\cos{\omega_*}', description='Cos of arg of periastron',
                                latex_unit='', user_params=user_params)

        planet.sinw = Parameter("sinw_" + name, lower=-1.0, upper=1.0, initval=0.0,
                                latex='\sin{\omega_*}', description='Sin of arg of periastron',
                                latex_unit='', user_params=user_params)

        # bound omega
        L = planet.cosw.value ** 2 + planet.sinw.value ** 2

        # ensures uniform omega distribution
        wbound = pm.Potential("wbound_" + name, pt.switch(L > 1.0,-np.inf,0.0))

        planet.omega = Parameter("omega_" + name, unit=u.rad,
                                 expression=pt.arctan2(planet.sinw, planet.cosw),
                                 latex='\omega_*', description='Arg of periastron',
                                 latex_unit='rad', user_params=user_params)

        planet.sign = Parameter("sign_" + name, lower=-1.0, upper=1.0,  # boolean?
                                latex='Sign', description='Sign of quadratic solution',
                                latex_unit='', user_params=user_params)

        # solve quadratic for e (Eastman 2024, eq 5)
        a = planet.vcve.value ** 2 * planet.sinw.value ** 2 + 1.0
        b = 2.0 * planet.vcve.value ** 2 * planet.sinw.value
        c = planet.vcve.value ** 2 - 1.0

        planet.e = Parameter('e_' + name,
                            expression=-b + pt.sign(planet.sign.value) * pt.sqrt(b ** 2 - 4.0 * a * c) / (2.0 * a),
                            latex='e', description='eccentricity',
                            latex_unit='', user_params=user_params)

        planet.esinw = Parameter("esinw_" + name,
                                 expression=planet["e"].value * planet["sinw"].value,
                                 latex='e\sin{\omega_*}', description='e times sin of arg of periastron',
                                 latex_unit='', user_params=user_params)

        jacobian = 1.0
        if fitchord:
            planet.chord = Parameter("chord_" + name, lower=0.0,
                                     latex='chord', description='transit chord',
                                     latex_unit='', user_params=user_params)

            planet.b = Parameter("b_" + name,
                                 expression=pt.sqrt((1.0 + planet.p) ** 2 - planet.chord ** 2),
                                 latex='b', description='impact parameter',
                                 latex_unit='', user_params=user_params)

            planet.cosi = Parameter("cosi_" + name,
                                    expression=planet.b.value /
                                               (planet.ar.value * (1.0 - planet.e.value ** 2) / (
                                                       1.0 + planet.esinw.value)),
                                    latex='\cos{i}', description='cos of inclination',
                                    latex_unit='', user_params=user_params)

            jacobian *= planet.b.value ** 2 / (
                planet.cosi.value * planet.chord.value)  # d(chord)/d(cosi)

        else:
            planet.cosi = Parameter("cosi_" + name, lower=0.0, upper=1.0,
                                    latex='\cos{i}', description='cos of inclination',
                                    latex_unit='', user_params=user_params)
            planet.b = Parameter("b_" + name,
                                 expression=planet.ar.value * planet.cosi.value * (
                                     1.0 - planet.e.value ** 2) / (1.0 - planet.esinw.value),
                                 latex='b', description='impact parameter',
                                 latex_unit='', user_params=user_params)

        # correct the prior to be uniform in e/omega (Eastman 2024, eq 6)
        jacobian *= (planet.e.value + planet.sinw.value) / (
                pt.sqrt(1.0 - planet.e.value ** 2) * (1.0 + planet.esinw.value) ** 2)  # d(vcve)/d(e)

        planet.jacobian = pm.Potential("jacobian_" + name, pt.abs(jacobian))

    else:
        # sqrt(e)*cos(omega), sqrt(e)*sin(omega), cosi
        planet.secosw = Parameter("secosw_" + name, lower=-1.0, upper=1.0, initval=0.0,
                                  latex='e^{1/2}\cos{\omega_*}', description='sqrt(e) cos(omega)',
                                  latex_unit='', user_params=user_params)

        planet.sesinw = Parameter("sesinw_" + name, lower=-1.0, upper=1.0, initval=0.0,
                                  latex='e^{1/2}\sin{\omega_*}', description='sqrt(e) sin(omega)',
                                  latex_unit='', user_params=user_params)

        planet.cosi = Parameter("cosi_" + name, lower=cosilower, upper=1.0, initval=0.0,
                                latex='\cos{i}', description='cos of inclination',
                                latex_unit='', user_params=user_params)

        planet.e = Parameter("e_" + name,
                             expression=planet.secosw.value ** 2 + planet.sesinw.value ** 2,
                             latex='e', description='eccentricity',
                             latex_unit='', user_params=user_params)

        planet.omega = Parameter("omega_" + name,
                                 expression=pt.arctan2(planet.sesinw.value, planet.secosw.value),
                                 unit=u.rad,
                                 latex='\omega_*', description='Arg of periastron',
                                 latex_unit='rad', user_params=user_params)

        planet.esinw = Parameter("esinw_" + name,
                                 expression=planet.e.value * pt.sin(planet.omega.value),
                                 latex='e\sin{\omega_*}', description='e times sin of arg of periastron',
                                 latex_unit='', user_params=user_params)

        planet["b"] = Parameter("b_" + name,
                                expression=planet.ar.value * planet.cosi.value * (
                                        1.0 - planet.e.value ** 2) / (1.0 - planet.esinw.value),
                                latex='b', description='impact parameter',
                                latex_unit='', user_params=user_params)

    # reject models where the planet collides with the star at periastron
    # this does four important things
    #   1) bounds e within the unit circle (uniform omega prior)
    #   2) rejects non-physical (NaN) solutions to the quadratic for vcve parameterization
    #   3) explicit physical upper limit on e
    #   4) implicit physical lower limit on a, period
    maxe = 1.0 - 1.0 / planet.ar.value - planet.p.value / planet.ar.value  # 1 - (Rp+Rstar)/a
    ebound = pm.Potential("ebound_" + name, pt.switch(planet.e.value < maxe, 0.0, -np.inf))

    # derive other parameters of interest
    planet.inc = Parameter("inc_" + name,
                           expression=pt.arccos(planet.cosi.value),
                           latex='i', unit=u.rad, description='inclination',
                           latex_unit='rad', user_params=user_params)
    # check units here!
    planet["k"] = Parameter("k_" + name,
                            expression=(2.0 * math.pi * G / (planet.period.value * (
                                    star.mass.value + planet.mp.value * mjup) ** 2.0)) ** (1.0 / 3.0) *
                                       planet.mp.value * mjup * pt.sin(planet.inc.value) /
                                       pt.sqrt(1.0 - planet.e.value ** 2.0) * rsun / meter / 86400.0,
                            unit=u.meter / u.second, latex_unit='m~s$^{-1}$',
                            latex='K', description='RV semi-amplitude',
                            user_params=user_params)