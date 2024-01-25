'''
;+
; NAME:
;   exofast_rv
;
; PURPOSE:
;   This function returns the radial velocity at each BJD. Optionally
;   includes the Rossiter McLaughlin effect, using the Ohta et al 2005
;   approximation good to ~1 m/s.
;
; CALLING SEQUENCE:
;   result = exofast_rv(bjd, TPeriastron, period, V0, K, e, omega, $
;                       i=i, a=a, u1=u1, p=p, vsini=vsini, lambda=lambda, $
;                       r_star=r_star, slope=slope)
;
; INPUTS:
;   bjd         - Barycentric Julians dates for desired impact parameters
;   tperiastron - periastron passage time (BJD)
;   Period      - Period of orbit (days)
;   V0          - systemic velocity (m/s)
;   K           - radial velocity semi-amplitude (m/s)
;
; OPTIONAL INPUTS:
;   e           - eccentricity of orbit (0 if not specified)
;   omega       - orbit's argument of periastron (radians)
;                 - required if e is specified
;                 - assumed to be pi/2 if e not specified
;                 - omega of the orbit being measured (for exoplanets, omega_*)
;
;   slope       - the slope of the RV signal
;   t0          - the time at which V0 is referenced, if a slope is
;                 given. If not given, (maxtime + mintime)/2 is
;                 assumed.
;                 NOTE: if comparing two data sets with a slope and
;                 different times, this must be specified.
;
; OPTIONAL KEYWORDS:
;   rossiter    - If set, calculates the RM effect. All values below
;                 must also be specified.
;
; Optional parameters (required for Rossiter-McLaughlin effect)
;   i           - inclination of orbit (radians)
;   a           - semi-major axis (R_star)
;   u1          - linear limb-darkening coefficient
;   p           - occulting star size (stellar radii)
;   vsini       - projected stellar surface velocity (m/s)
;   lambda      - projected spin-orbit angle (radians)
;
; OUTPUTS:
;    result     - the radial velocity of the star at each input time
;
; MODIFICATION HISTORY
;  2024/01 -- Translated from exofast_rv, Jason Eastman (CfA)
;-
'''

import numpy as np
from . import exozippy_keplereq
from . import exozippy_rossiter
#from radvel import kepler

def exozippy_rv(bjd, tp, period, gamma, K, e=None, omega=None,
               slope=None, quad=None, t0=None, rossiter=False, i=None, a=None,
               u1=None, p=None, vsini=None, _lambda=None, deltarv=None):

    # Calculate the mean anomaly corresponding to each observed time
    meananom = 2.0 * np.pi * (1.0 + np.mod((bjd - tp)/period,1.0))

    # If eccentricity is given, integrate the orbit
    if e is not None:
        if omega is None:
            raise ValueError('ERROR: omega must be specified if e is specified')

        if e < 0:
            e0 = -e
            omega0 = omega + np.pi
        else:
            e0 = e
            omega0 = omega

        eccanom = exozippy_keplereq.exozippy_keplereq(meananom, e0)
        trueanom = 2.0 * np.arctan(np.sqrt((1.0 + e0) / (1.0 - e0)) * np.tan(eccanom / 2.0))
    else:
        trueanom = meananom
        e0 = 0.0
        # Standard definition of omega for circular orbits
        if omega is None:
            omega0 = np.pi / 2.0
        else:
            omega0 = omega

    # RV signal with no RM effect
    rv = K * (np.cos(trueanom + omega0) + e0 * np.cos(omega0)) + gamma

    # Add a slope if desired
    if slope is not None or quad is not None:
        if t0 is None:
            mintime = np.min(bjd)
            maxtime = np.max(bjd)
            t0 = (maxtime + mintime) / 2.0
        if slope is not None:
            rv += (bjd - t0) * slope
        if quad is not None:
            rv += (bjd - t0)**2 * quad

    # Calculate the RM effect
    if rossiter:
        if i is None or a is None or u1 is None or p is None or vsini is None or _lambda is None:
            raise ValueError('ERROR: a, i, u1, p, vsini, and _lambda must be specified '
                             'to calculate the Rossiter McLaughlin effect')

        # Calculate the corresponding (x,y) coordinates of the planet
        r = a * (1 - e ** 2) / (1 + e * np.cos(trueanom))

        # As seen from the observer
        x = -r * np.cos(trueanom + omega)
        tmp = r * np.sin(trueanom + omega)
        y = -tmp * np.cos(i)
        z = tmp * np.sin(i)

        exozippy_rossiter.exozippy_rossiter(x, y, u1, p, vsini, _lambda, deltarv, z=z)
        rv += deltarv

    return rv
