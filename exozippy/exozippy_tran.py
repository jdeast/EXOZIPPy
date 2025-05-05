import numpy as np
from .utils import *


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
# -------------------------------------------------------------------
# NAME:
#   exozippy_TRAN
#
# PURPOSE:
#   Computes a transit model given a complete set of physical parameters
#
# CALLING SEQUENCE:
#   model = exozippy_tran(time, inc, ar, tp, period, e, omega, p, u1, u2, f0,
#                        rstar=rstar, thermal=thermal, reflect=reflect,
#                        dilute=dilute, tc=tc, q=q, x1=x1, y1=y1, z1=z1,
#                        au=au, c=c)
#
# INPUTS:
#   time   - The BJD_TDB time of the model to compute. Scalar or array.
#   inc    - Inclination of the planetary orbit, in radians
#   ar     - a/Rstar, semi-major axis in units of stellar radii
#   tp     - Time of periastron (BJD_TDB)
#   period - Orbital period (days)
#   e      - Eccentricity
#   omega  - Argument of periastron (radians)
#   p      - Rp/Rstar, planetary radius in stellar radii
#   u1, u2 - Limb darkening parameters (linear and quadratic)
#   f0     - Baseline flux level
#
# OPTIONAL INPUTS:
#   rstar   - Stellar radius in AU. Used to apply light travel time correction
#   thermal - Thermal emission contribution from the planet, in ppm
#   reflect - Reflected light from the planet, in ppm
#   dilute  - Fraction of baseline flux from contaminating sources
#   tc      - Time of conjunction, for reflected light phase. Computed if not given
#   q       - M1/M2, stellar mass ratio. Applies reflex motion if given
#   x1, y1, z1 - Motion of the star due to other bodies, in Rstar (unused here)
#   au      - Value of AU in solar radii (default 215.094177)
#   c       - Speed of light in AU/day (used in bjd2target)
#
# OUTPUTS:
#   model  - The transit model flux as a function of time
#
# REVISION HISTORY:
#   2015     - Written by Jason Eastman (CfA)
#   2018/10  - Documented (JDE)
#   2019/01/28 - Replaced incorrect arg_present check with n_elements check.
#                Didn't convert to target frame before (as called by chi2v2)
# -------------------------------------------------------------------

def exozippy_tran(
    time, inc, ar, tp, period, e, omega, p, u1, u2, f0,
    rstar=None, thermal=0.0, reflect=0.0, beam=0.0, ellipsoidal=0.0,
    phaseshift=0.0, dilute=0.0, tc=None, q=None, x1=None, y1=None, z1=None,
    au=215.094177, c=None
):
    # If we have stellar radius, convert time to target's barycentric frame
    if rstar is not None:
        transitbjd = bjd2target(time, inclination=inc, a=ar * rstar, tp=tp,
                                period=period, e=e, omega=omega, q=q, c=c)
    else:
        transitbjd = time

    # The impact parameter for each BJD
    z = exozippy_getb2(transitbjd, inc=inc, a=ar, tperiastron=tp, period=period,
                      e=e, omega=omega, q=q)
    ntime = len(time)

    # Primary transit
    modelflux = np.ones(ntime)
    depth = np.zeros_like(z) - 1.0  # Placeholder for z2
    primary = depth < 0
    secondary = ~primary

    if np.any(primary):
        mu1,mu0 = exozippy_occultquad_cel(z[primary], u1, u2, p)
        modelflux[primary] = mu1

    # Calculate fraction of the planet visible at each time
    if thermal != 0.0 or reflect != 0.0:
        planetvisible = np.ones_like(time)
        if np.any(secondary):
            mu1,mu0 = exozippy_occultquad_cel(z[secondary] / abs(p), 0.0, 0.0, 1.0 / p)
            planetvisible[secondary] = mu1
    else:
        planetvisible = 1.0

    # Thermal emission from planet (isotropic)
    if thermal != 0.0:
        modelflux += 1e-6 * thermal * planetvisible

    # Phase-dependent reflection off the planet
    if reflect != 0.0:
        if tc is None:
            phase = exozippy_getphase(e, omega, primary=True)
            tc0 = tp - phase * period
        else:
            tc0 = tc

        # This makes flux = 0 during primary transit (Thanks Sam Quinn!)
        modelflux -= (1e-6 * reflect / 2.0) * (np.cos(2.0 * np.pi * (transitbjd - tc0) / period) - 1.0) * planetvisible

    # Normalize and apply dilution from neighboring star
    if dilute != 0.0:
        modelflux = f0 * (modelflux * (1.0 - dilute) + dilute)
    else:
        modelflux *= f0

    # Add beaming and ellipsoidal variations
    if ellipsoidal != 0.0:
        if tc is None:
            phase = exozippy_getphase(e, omega, primary=True)
            tc0 = tp - phase * period
        else:
            tc0 = tc

        modelflux *= (1.0 - ellipsoidal / 1e6 * np.cos(2.0 * np.pi * (transitbjd - tc0) / (period / 2.0)))

    if beam != 0.0:
        if tc is None:
            phase = exozippy_getphase(e, omega, primary=True)
            tc0 = tp - phase * period
        else:
            tc0 = tc

        modelflux += beam / 1e6 * np.sin(2.0 * np.pi * (transitbjd - tc0) / period)

    return modelflux


