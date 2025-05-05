'''
The high level code for exozippy (equivalent to exofastv2.pro)
'''

import numpy as np
# import pymc as pm
import matplotlib as plt
import ipdb
from astropy import units as u
import astropy.constants as const
from .build_model import build_model
from .build_latex_table import build_latex_table

def exozippy(parfile=None, \
             prefix='fitresults/planet.',\
             # data file inputs
             rvpath=None, tranpath=None, \
             astrompath=None, dtpath=None, \
             # SED model inputs
             fluxfile=None,mistsedfile=None, \
             sedfile=None,specphotpath=None, \
             noavprior=False,\
             fbolsedfloor=0.024,teffsedfloor=0.02,\
             fehsedfloor=0.08, oned=False,\
             # evolutionary model inputs
             yy=False, nomist=False, parsec=False, \
             torres=False, mannrad=False,mannmass=False, \
             teffemfloor=0.02, fehemfloor=0.08, \
             rstaremfloor=0.042,ageemfloor=0.01,\
             # BEER model inputs
             fitthermal=False, fitellip=False, \
             fitreflect=False, fitphase=False, \
             fitbeam=False, derivebeam=False, \
             # star inputs
             nstars=1, starndx=0, \
             diluted=False, fitdilute=False, \
             # planet inputs
             nplanets=1, \
             fittran=None, fitrv=None, \
             rossiter=False, fitdt=None, \
             circular=False, tides=False, \
             alloworbitcrossing=False, \
             chen=False, i180=False, \
             # RV inputs
             fitslope=False, fitquad=False, rvepoch=None,\
             # transit inputs
             noclaret=False, \
             ttvs=False, tivs=False, tdeltavs=False,\
             longcadence=False, exptime=False, ninterp=False, \
             rejectflatmodel=False,\
             fitspline=False, splinespace=0.75, \
             fitwavelet=False, \
             # reparameterization inputs
             fitlogmp=False,\
             novcve=False, nochord=False, fitsign=False, \
             fittt=False, earth=False, \
             # plotting inputs
             transitrange=[None,None],rvrange=[None,None],\
             sedrange=[None,None],emrange=[None,None], \
             # debugging inputs
             debug=False, verbose=False, delay=0.0, \
             # MCMC inputs
             maxsteps=None, nthin=None, maxtime=None, \
             maxgr=1.01, mintz=1000, \
             dontstop=False, \
             ntemps=1, tf=200, keephot=False, \
             seed=None, \
             stretch=False, \
             nthreads=None, \
             # General inputs
             skiptt=False, \
             usernote=None, \
             mksummarypg=False, \
             nocovar=False, \
             plotonly=False, bestonly=False, \
             badstart=False):

    # this is a user supplied parameter file that modifies the default
    # initial values and imposes limits and priors on any parameter
    # (fit or derived)

    event = build_model(parfile=parfile, tranpath=tranpath)
    return event
    # Define the orbit
    # orbit = xo.orbits.KeplerianOrbit(
    #     period=10.0,  # All times are in days
    #     t0=0.5,
    #     incl=0.5 * np.pi,  # All angles are in radians
    #     ecc=0.3,
    #     omega=-2.5,
    #     Omega=1.2,
    #     m_planet=0.05,  # All masses and distances are in Solar units
    # )
    
    # Get the position and velocity as a function of time
    # t = np.linspace(0, 20, 5000)
    # x, y, z = orbit.get_relative_position(t)
    # vx, vy, vz = orbit.get_star_velocity(t)
    
    # Plot the coordinates
    # Note the use of `.eval()` throughout since the coordinates are all
    # Aesara/Theano objects
    # fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    # ax = axes[0]
    # ax.plot(t, x.eval(), label="x")
    # ax.plot(t, y.eval(), label="y")
    # ax.plot(t, z.eval(), label="z")
    # ax.set_ylabel("position of orbiting body [$R_*$]")
    # ax.legend(fontsize=10, loc=1)
    
    # ax = axes[1]
    # ax.plot(t, vx.eval(), label="$v_x$")
    # ax.plot(t, vy.eval(), label="$v_y$")
    # ax.plot(t, vz.eval(), label="$v_z$")
    # ax.set_xlim(t.min(), t.max())
    # ax.set_xlabel("time [days]")
    # ax.set_ylabel("velocity of central [$R_*$/day]")
    # _ = ax.legend(fontsize=10, loc=1)
    
    # with pm.Model():
    #     log_period = pm.Normal("log_period", mu=np.log(10), sigma=2.0)
    #     orbit = xo.orbits.KeplerianOrbit(
    #         period=pm.math.exp(log_period),  # ...
    #     )
        
    # Define the rest of you model using `orbit`...

if __name__ == "__main__":
    exozippy(parfile="../examples/gj1132/101955023.priors.json",
             tranpath="../examples/gj1132/*.dat")
