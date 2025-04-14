import numpy as np
from .exozippy_keplereq import *
import requests
import re
import os
from numba import njit

# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def parse_param_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    params = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # skip comments and blank lines

        # Remove inline comments and split on whitespace
        parts = re.split(r'\s+', line.split("#")[0].strip())
        if not parts:
            continue

        label = parts[0]
        try:
            values = list(map(float, parts[1:]))
        except ValueError:
            # Handle lines with malformed numbers
            continue

        # Assign values safely
        mu      = values[0] if len(values) > 0 else None
        sigma   = values[1] if len(values) > 1 else None
        lower   = values[2] if len(values) > 2 else None
        upper   = values[3] if len(values) > 3 else None
        initval = values[4] if len(values) > 4 else None

        params[label] = {
            "mu": mu,
            "sigma": sigma,
            "lower": lower,
            "upper": upper,
            "initval": initval
        }

    return params


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def vcve2e(vcve0, omega=None, lsinw=None, lcosw=None, sign=None):
    """
    Convert vcve to eccentricity e using omega or sin(omega), cos(omega).
    
    Parameters
    ----------
    vcve0 : float or array_like
        The quantity sqrt((1 - e^2) / (1 + e sin(omega))) to be inverted.
    omega : float or array_like, optional
        Argument of periastron in radians.
    lsinw, lcosw : float or array_like, optional
        Sine and cosine of omega multiplied by sqrt(e), used to recover omega if not given.
    sign : int or array_like, optional
        Sign to select between two solutions (0 = positive root, 1 = negative root).

    Returns
    -------
    e : float or np.ndarray
        Eccentricity.
    """
    vcve = np.atleast_1d(vcve0).astype(float)

    if omega is None:
        if lsinw is None or lcosw is None:
            raise ValueError("Must specify either omega or lsinw and lcosw")
        omega = np.arctan2(lsinw, lcosw)
    else:
        omega = np.atleast_1d(omega).astype(float)

    # Broadcast shapes
    if vcve.size == 1 and omega.size > 1:
        vcve = np.full_like(omega, vcve[0])
    elif omega.size == 1 and vcve.size > 1:
        omega = np.full_like(vcve, omega[0])
    
    if sign is None:
        if lsinw is not None and lcosw is not None:
            L2 = lsinw**2 + lcosw**2
            useneg = L2 >= 0.5
        else:
            a = vcve**2 * np.sin(omega)**2 + 1
            b = 2 * vcve**2 * np.sin(omega)
            c = vcve**2 - 1
            disc = b**2 - 4 * a * c
            epos = (-b + np.sqrt(disc)) / (2 * a)
            eneg = (-b - np.sqrt(disc)) / (2 * a)

            good_neg = np.isfinite(eneg) & (eneg >= 0) & (eneg < 1) & (eneg < epos)
            sign = np.zeros_like(vcve, dtype=int)
            sign[good_neg] = 1
            if vcve.size == 1:
                sign = int(sign[0])
            useneg = sign == 1
    else:
        sign = np.atleast_1d(sign)
        if sign.size == 1 and vcve.size > 1:
            sign = np.full_like(vcve, sign[0])
        useneg = np.floor(sign).astype(bool)

    # Solve quadratic
    a = vcve**2 * np.sin(omega)**2 + 1
    b = 2 * vcve**2 * np.sin(omega)
    c = vcve**2 - 1
    disc = b**2 - 4 * a * c

    epos = (-b + np.sqrt(disc)) / (2 * a)
    eneg = (-b - np.sqrt(disc)) / (2 * a)

    e = np.zeros_like(vcve)
    usepos = ~useneg
    e[usepos] = epos[usepos]
    e[useneg] = eneg[useneg]

    return e[0] if e.size == 1 else e


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def exozippy_like(residuals, var_r, sigma_w0, chi2=False, truechi2=False):
    """
    Compute the log likelihood for a given residual and noise model. 
    A simpler (and faster) alternative to the wavelet analysis of Carter & Winn 2009.

    Parameters
    ----------
    residuals : array_like
        The residuals of a fit (data - model).
    var_r : float
        Red noise amplitude variance (sigma_r^2).
    sigma_w0 : float or array_like
        White noise amplitude (or error array).
    chi2 : bool, optional
        If True, returns the "effective chi2" used for EXOFAST_DEMC.
    truechi2 : bool, optional
        If True, returns the true chi2 value.

    Returns
    -------
    float
        Log-likelihood, effective chi2, or true chi2 depending on keywords.
    """
    residuals = np.asarray(residuals)
    if np.isscalar(sigma_w0):
        sigma_w = np.full_like(residuals, sigma_w0)
    else:
        sigma_w = np.asarray(sigma_w0)

    good = np.isfinite(sigma_w)
    if not np.any(good):
        return np.inf if chi2 else -np.inf

    denom = sigma_w[good]**2 + var_r
    chisq = np.sum(residuals[good]**2 / denom)

    if truechi2:
        return chisq

    loglike = -0.5 * (np.sum(np.log(2.0 * np.pi * denom)) + chisq)

    if not np.isfinite(loglike):
        return np.inf if chi2 else -np.inf

    return -2.0 * loglike if chi2 else loglike


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
# ;+
# ; NAME:
# ;  angsep
# ; PURPOSE:
# ;  Compute the angular distance between two spherical coordinates.
# ; DESCRIPTION:
# ;
# ; CATEGORY:
# ;  Mathematical
# ; CALLING SEQUENCE:
# ;  ans=angsep(ra1,dec1,ra2,dec2)
# ;
# ; INPUTS:
# ;  ra1  - RA of first position (radians)
# ;  dec1 - Dec of first position (radians)
# ;  ra2  - RA of second position (radians)
# ;  dec2 - Dec of second position (radians)
# ;
# ; OPTIONAL INPUT PARAMETERS:
# ;
# ; KEYWORD INPUT PARAMETERS:
# ;
# ; OUTPUTS:
# ;  return value is the angular distance in radians.
# ;
# ; KEYWORD OUTPUT PARAMETERS:
# ;
# ; COMMON BLOCKS:
# ;
# ; SIDE EFFECTS:
# ;
# ; RESTRICTIONS:
# ;
# ; PROCEDURE:
# ;
# ; MODIFICATION HISTORY:
# ;  Written by Marc W. buie, Lowell Observatory, 1997/09/08
# ;  2009/02/26, MWB, added protection against round off error generating NaN
# ;-
def angsep(ra1, dec1, ra2, dec2):
    arg = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    arg = np.clip(arg, -1.0, 1.0)
    return np.arccos(arg)



# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def get_av_prior(ra=None, dec=None, object_name=None):
    """
    Returns the maximum V-band extinction (Av) from Schlafly and Finkbeiner (2011).

    Parameters
    ----------
    ra : float, optional
        J2000 right ascension in decimal degrees.
    dec : float, optional
        J2000 declination in decimal degrees.
    object_name : str, optional
        Object name to resolve via SIMBAD or IRSA.

    Returns
    -------
    maxav : float
        Maximum Av value.
    line : str
        String of the form 'av 0 -1 0 <maxav>' for EXOFASTv2.
    """

    if object_name:
        url = f"https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={object_name}"
    elif ra is not None and dec is not None:
        url = f"https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={ra}+{dec}+equ+j2000"
    else:
        raise ValueError("Must specify either object_name or both ra and dec.")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.text.splitlines()
    except Exception as e:
        print(f"[WARNING] Failed to query dust map: {e}")
        return 99.0, "av 0 -1 0 99"

    if len(content) == 0 or "Invalid object name" in content[2]:
        print("[WARNING] Invalid object name or empty response.")
        return 99.0, "av 0 -1 0 99"

    try:
        # Find the last line before the closing tag
        match_index = next(i for i, line in enumerate(content) if '</maxValueSandF>' in line)
        raw_value = content[match_index - 1]
        ebv = float(re.split(r'\(|\s', raw_value.strip())[0])
        maxav = ebv * 3.1 * 1.5 * 0.87
    except Exception as e:
        print(f"[WARNING] Parsing failed: {e}")
        return 99.0, "av 0 -1 0 99"

    line = f"av 0 -1 0 {maxav:.6f}"
    return maxav, line


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
# -------------------------------------------------------------------
# NAME:
#   target2bjd
#
# PURPOSE:
#   Converts a BJD in the target barycenter time to BJD in Barycentric
#   Dynamical Time (BJD_TDB).
#
# DESCRIPTION:
#   Corrects for the Roemer delay (light travel time) in
#   the target system (~30 seconds for typical Hot Jupiters). Most
#   importantly, this will naturally make primary, secondary, RV
#   observations, and phase variations self-consistent.
#
#   Additionally, for a typical Hot Jupiter (3 hour transit in a 3 day
#   orbit), the ingress is delayed by ~0.15 seconds relative to the mid
#   transit time. For circular-orbits, the delay between mid transit
#   time and ingress is:
#
#   dt = a/c*(1-cos((sqrt((R_*+R_P)^2 - b^2)/a))
#
#   Falling for larger semi-major axis.
#
# INPUTS:
#   bjd_target  - BJD times in target barycentric frame (scalar or array, float64)
#   inclination - Inclination of the orbit (radians)
#   a           - Semi-major axis (AU)
#   tp          - Time of periastron (same frame as bjd_target)
#   period      - Orbital period (days)
#   e           - Eccentricity
#   omega       - Argument of periastron (radians)
#
# OPTIONAL INPUTS:
#   q           - Mass ratio (M1/M2). If not given, M1 is stationary.
#   c           - Speed of light in AU/day. Default = 173.144483
#
# OPTIONAL KEYWORDS:
#   primary     - If True, light-time correction from primary to barycenter.
#                 Otherwise, correction from planet to barycenter.
#
# OUTPUT:
#   bjd_tdb     - Corrected BJD_TDB times (float64)
#
# LIMITATIONS:
#   - Ignores distance to object (plane-parallel wave approximation): <1 ms
#   - Ignores systemic velocity (gamma/c compression)
#
# REVISION HISTORY:
#   2011/06: Written by Jason Eastman (OSU)
# -------------------------------------------------------------------

def target2bjd(
    bjd_target,
    inclination,
    a,
    tp,
    period,
    e,
    omega,
    q=np.inf,
    primary=False,
    c=173.144483  # AU/day
):
    bjd_target = np.array(bjd_target, dtype=np.float64)

    # No correction necessary: already in SSB frame
    if not np.isfinite(q) and primary:
        return bjd_target

    # Mean anomaly
    mean_anom = 2.0 * np.pi * ((bjd_target - tp) / period)
    mean_anom = np.mod(mean_anom, 2 * np.pi)

    # Solve Kepler's Equation
    if np.ndim(e) == 0:
        ecc_anom = exozippy_keplereq(mean_anom, e)
    else:
        if e.shape != bjd_target.shape:
            raise ValueError("e must be scalar or same shape as bjd_target")
        ecc_anom = np.array([exozippy_keplereq(M, ei) for M, ei in zip(mean_anom, e)])

    # True anomaly
    true_anom = 2.0 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(ecc_anom / 2.0))

    # Displacement from barycenter
    if np.isfinite(q):
        if primary:
            factor = 1.0 / (1.0 + q)  # a1 = a * factor
        else:
            factor = q / (1.0 + q)   # a2 = a * factor
    else:
        factor = 1.0

    # Distance from barycenter to target
    r = a * (1 - e**2) / (1 + e * np.cos(true_anom)) * factor

    # Rotate orbit by omega
    if not primary:
        om = omega + np.pi
    else:
        om = omega

    # Line-of-sight component
    z = r * np.sin(true_anom + om) * np.sin(inclination)

    return bjd_target - z / c

# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
# -------------------------------------------------------------------
# NAME:
#   bjd2target
#
# PURPOSE:
#   Iteratively calls TARGET2BJD to convert a BJD in Barycentric
#   Dynamical Time (BJD_TDB) to a BJD in the target barycenter
#   time (BJD_TARGET) within TOL days.
#
# DESCRIPTION:
#   The opposite of TARGET2BJD; see description there.
#
# INPUTS:
#   bjd_tdb     - A scalar or array of BJDs in TDB. Must be double precision.
#   inclination - Inclination of the orbit (radians)
#   a           - Semi-major axis (AU)
#   tp          - Time of periastron (BJD_TARGET)
#   period      - Orbital period (days)
#   e           - Orbital eccentricity
#   omega       - Argument of periastron (radians)
#
# OPTIONAL INPUTS:
#   q           - Mass ratio (M1/M2). If not specified, M1 is stationary at barycenter.
#   tol         - Tolerance in days (default 1e-8 = 1 ms)
#
# OPTIONAL KEYWORDS:
#   primary     - If True, correction is for light travel time from primary
#                 to barycenter (as in RVs). Otherwise from planet to barycenter
#                 (as in transits).
#
# OUTPUT:
#   bjd_target  - The time as it would flow in the barycenter of the target.
#
# LIMITATIONS:
#   - Ignores distance to object (plane parallel waves): <1 ms error
#   - Ignores systemic velocity (gamma/c compression)
#
# REVISION HISTORY:
#   2011/06: Written by Jason Eastman (OSU)
# -------------------------------------------------------------------
@njit
def bjd2target(
    bjd_tdb,
    inclination,
    a,
    tp,
    period,
    e,
    omega,
    q=None,
    tol=1e-8,
    primary=False,
    pars=None,
    c=None
):
    bjd_target = np.array(bjd_tdb, dtype=np.float64)
    niter = 0

    while True:
        # Iterative process to find BJD_TARGET corresponding to BJD_TDB
        # Typically completes in ~3 iterations
        target_new = target2bjd(
            bjd_target,
            inclination=inclination,
            a=a,
            tp=tp,
            period=period,
            e=e,
            omega=omega,
            q=q,
            primary=primary,
            c=c
        )

        diff = bjd_tdb - target_new
        bjd_target += diff
        niter += 1

        if niter > 100:
            print(f"i={inclination:.20f}, a={a:.20f}, tp={tp:.20f}, period={period:.20f}, e={e:.20f}, omega={omega:.20f}")
            raise RuntimeError(
                "Not converging. This is a rare bug usually associated with poorly constrained parameters. "
                "Try again or consider imposing priors on poorly constrained parameters. Especially if you "
                "have parallel tempering enabled, you should have loose, uniform priors on Period and Tc."
            )

        if np.max(np.abs(diff)) < tol:
            break

    return bjd_target


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
# -------------------------------------------------------------------
# NAME:
#   exozippy_getb2
#
# PURPOSE:
#   Computes the impact parameter as a function of BJD in the target
#   barycentric frame, given Keplerian orbital elements for N planets.
#   Optionally returns barycentric 3D coordinates.
#
# INPUTS:
#   bjd          - Time array (scalar, 1D, or 2D for long cadence sampling)
#   inc          - Array of NPLANETS inclinations (radians)
#   a            - Semi-major axes in R_star units (length = NPLANETS)
#   tperiastron  - Time of periastron for each planet (same units as bjd)
#   period       - Orbital period for each planet (days)
#
# OPTIONAL INPUTS:
#   e            - Eccentricity (default 0)
#   omega        - Argument of periastron (default π/2 if e not given)
#   lonascnode   - Longitude of ascending node (default π)
#   q            - Mass ratio M1/M2 (default = inf, i.e., stationary star)
#
# OUTPUT:
#   b            - Impact parameter in R_star units
#
# OPTIONAL OUTPUTS:
#   x0, y0, z0   - Relative position (planet - star), R_star units
#   x1, y1, z1   - Star position in barycentric frame
#   x2, y2, z2   - Planet position in barycentric frame
# -------------------------------------------------------------------
def exozippy_getb2(
    bjd, inc, a, tperiastron, period, e=None, omega=None,
    lonascnode=None, q=None
):
    bjd = np.array(bjd, dtype=np.float64)
    inc = np.atleast_1d(inc); inc = np.asarray(inc, dtype=np.float64)
    a = np.atleast_1d(a); a = np.asarray(a, dtype=np.float64)
    tperiastron = np.atleast_1d(tperiastron); tperiastron = np.asarray(tperiastron, dtype=np.float64)
    period = np.atleast_1d(period); period = np.asarray(period, dtype=np.float64)
    nplanets = len(inc)

    shape = bjd.shape
    if bjd.ndim == 0:
        ntimes, ninterp = 1, 1
        bjd = bjd[None]
    elif bjd.ndim == 1:
        ntimes, ninterp = len(bjd), 1
        bjd = bjd[:, None]
    elif bjd.ndim == 2:
        ntimes, ninterp = bjd.shape
    else:
        raise ValueError("Incompatible dimensions for BJD")

    # Default eccentricity, omega, q
    e = np.atleast_1d(e if e is not None else np.zeros(nplanets))
    omega = np.atleast_1d(omega if omega is not None else np.ones(nplanets) * (np.pi / 2))
    q = np.atleast_1d(q if q is not None else np.full(nplanets, np.inf))

    # Allocate arrays
    x1 = np.zeros((ntimes, ninterp))
    y1 = np.zeros((ntimes, ninterp))
    z1 = np.zeros((ntimes, ninterp))

    x2 = np.zeros((nplanets, ntimes, ninterp))
    y2 = np.zeros((nplanets, ntimes, ninterp))
    z2 = np.zeros((nplanets, ntimes, ninterp))

    x0 = np.zeros((nplanets, ntimes, ninterp))
    y0 = np.zeros((nplanets, ntimes, ninterp))
    z0 = np.zeros((nplanets, ntimes, ninterp))

    isinf = ~np.isfinite(q)
    isfinite = np.isfinite(q)
    na = a.shape
    a1 = np.zeros(na)
    a2 = np.zeros(na)

    a2[isinf] = a[isinf]
    a1[isinf] = 0.0
    a2[isfinite] = a[isfinite] * q[isfinite] / (1.0 + q[isfinite])
    a1[isfinite] = a2[isfinite] / q[isfinite]

    for i in range(nplanets):
        # Mean anomaly
        meananom = 2.0 * np.pi * ((bjd - tperiastron[i]) / period[i])
        meananom = np.mod(meananom, 2.0 * np.pi)

        if e[i] != 0.0:
            eccanom = exozippy_keplereq(meananom, e[i])
            trueanom = 2.0 * np.arctan(np.sqrt((1 + e[i]) / (1 - e[i])) * np.tan(eccanom / 2.0))
        else:
            trueanom = meananom

        # Distance and coordinates (planet in barycentric frame)
        r2 = -a2[i] * (1 - e[i] ** 2) / (1 + e[i] * np.cos(trueanom))

        x2[i] = (r2 * np.cos(trueanom + omega[i])).reshape(ntimes, ninterp)
        tmp = r2 * np.sin(trueanom + omega[i])
        y2[i] = (tmp * np.cos(inc[i])).reshape(ntimes, ninterp)
        z2[i] = (tmp * np.sin(inc[i])).reshape(ntimes, ninterp)

        # Rotate by longitude of ascending node if provided
        if lonascnode is not None and len(lonascnode) == nplanets:
            lon = lonascnode[i]
            x_old = x2[i].copy()
            y_old = y2[i].copy()
            x2[i] = x_old * np.cos(lon) - y_old * np.sin(lon)
            y2[i] = x_old * np.sin(lon) + y_old * np.cos(lon)

        # Star position in barycentric frame
        r1 = a1[i] * (1 - e[i] ** 2) / (1 + e[i] * np.cos(trueanom))
        x1tmp = (r1 * np.cos(trueanom + omega[i])).reshape(ntimes, ninterp)
        tmp = r1 * np.sin(trueanom + omega[i])
        y1tmp = (tmp * np.cos(inc[i])).reshape(ntimes, ninterp)
        z1tmp = (tmp * np.sin(inc[i])).reshape(ntimes, ninterp)

        if lonascnode is not None and len(lonascnode) == nplanets:
            lon = lonascnode[i]
            x1 += x1tmp * np.cos(lon) - y1tmp * np.sin(lon)
            y1 += x1tmp * np.sin(lon) + y1tmp * np.cos(lon)
        else:
            x1 += x1tmp
            y1 += y1tmp
        z1 += z1tmp

    # Convert to stellar frame (planet position relative to star)
    for i in range(nplanets):
        x0[i] = x2[i] - x1
        y0[i] = y2[i] - y1
        z0[i] = z2[i] - z1

    # Impact parameter = projected sky-plane separation
    b = np.sqrt(x0**2 + y0**2)
    return b.squeeze()  # Return 1D if possible



# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
# -------------------------------------------------------------------
# NAME:
#   exozippy_getphase
#
# PURPOSE:
#   Calculates the phase (mean anomaly / 2π) corresponding to a
#   particular true anomaly.
#
# CALLING EXAMPLES:
#   phase = exozippy_getphase(e, omega, primary=True)
#
# INPUTS:
#   eccen         - Eccentricity (0 < eccen < 1)
#   omega         - Argument of periastron (radians)
#
# OPTIONAL:
#   trueanom      - Directly specify true anomaly
#   primary       - Compute phase of primary transit
#   secondary     - Compute phase of secondary eclipse
#   l4, l5        - Compute phase of L4 or L5 point
#   periastron    - Phase of periastron (by definition = 0)
#   ascendingnode - Phase of max RV (ascending node)
#   descendingnode- Phase of min RV (descending node)
#
# OUTPUT:
#   phase         - Orbital phase (0 to 1)
#
# MOD HISTORY:
#   2010/06 - Rewritten by Jason Eastman (OSU)
# -------------------------------------------------------------------
@njit
def exozippy_getphase(
    eccen,
    omega,
    trueanom=None,
    primary=False,
    secondary=False,
    l4=False,
    l5=False,
    periastron=False,
    ascendingnode=False,
    descendingnode=False
):
    eccen = np.array(eccen, dtype=np.float64)
    omega = np.array(omega, dtype=np.float64)

    # Handle common special-case phase positions
    if periastron:
        trueanom = 0.0
    elif l5:
        trueanom = (5.0 * np.pi / 6.0) - omega
    elif l4:
        trueanom = (1.0 * np.pi / 6.0) - omega
    elif secondary:
        trueanom = (3.0 * np.pi / 2.0) - omega
    elif primary:
        trueanom = (1.0 * np.pi / 2.0) - omega
    elif ascendingnode:
        trueanom = -omega
    elif descendingnode:
        trueanom = np.pi - omega

    if trueanom is None:
        raise ValueError("Must specify trueanom or one of the special-case keywords")

    # Convert true anomaly to eccentric anomaly
    eccanom = 2.0 * np.arctan(np.sqrt((1.0 - eccen) / (1.0 + eccen)) * np.tan(trueanom / 2.0))

    # Mean anomaly
    M = eccanom - eccen * np.sin(eccanom)

    # Phase = M / 2π
    phase = M / (2.0 * np.pi)

    # Normalize to [0, 1]
    phase = np.mod(phase, 1.0)

    return phase




# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
@njit
def cel_bulirsch_vec(k2, kc, p, a1, a2, a3, b1, b2, b3, f1, f2, f3):
    """
    Vectorized version of the Bulirsch-Stoer integration for computing
    limb darkening parameters during occultations with eccentric orbits.
    Implements the efficient recursive steps as per Mandel & Agol (2002).
    """

    ca = np.sqrt(k2 * 2.2e-16)

    # Avoid undefined k2 = 1 or kc = 0 cases
    mask = (k2 == 1.0) | (kc == 0.0)
    kc = np.where(mask, 2.22e-16, kc)

    ee = kc.copy()
    m = np.ones_like(kc)

    pos = np.where(p >= 0.0)[0]
    neg = np.where(p < 0.0)[0]

    pinv = np.zeros_like(k2)
    if pos.size:
        p[pos] = np.sqrt(p[pos])
        pinv[pos] = 1.0 / p[pos]
        b1[pos] *= pinv[pos]

    if neg.size:
        q = k2[neg].copy()
        g = 1.0 - p[neg]
        f = g - k2[neg]
        q *= (b1[neg] - a1[neg] * p[neg])
        ginv = 1.0 / g
        p[neg] = np.sqrt(f * ginv)
        a1[neg] = (a1[neg] - b1[neg]) * ginv
        pinv[neg] = 1.0 / p[neg]
        b1[neg] = -q * ginv**2 * pinv[neg] + a1[neg] * p[neg]

    # Compute recursion
    f1 = a1.copy()
    a1 += b1 * pinv
    g = ee * pinv
    b1 += f1 * g
    b1 *= 2
    p += g

    # Compute remainder with p = 1
    p1 = np.ones_like(p)
    g1 = ee.copy()

    f2 = a2.copy()
    f3 = a3.copy()

    a2 += b2
    b2 += f2 * g1
    b2 *= 2

    a3 += b3
    b3 += f3 * g1
    b3 *= 2

    p1 += g1

    g = m.copy()
    m += kc

    iter = 0
    itmax = 50

    while np.max(np.abs(g - kc) > g * ca) and (iter < itmax):
        kc = np.sqrt(ee)
        kc *= 2
        ee = kc * m

        f1 = a1.copy()
        f2 = a2.copy()
        f3 = a3.copy()

        pinv = 1.0 / p
        pinv1 = 1.0 / p1

        a1 += b1 * pinv
        a2 += b2 * pinv1
        a3 += b3 * pinv1

        g = ee * pinv
        g1 = ee * pinv1

        b1 += f1 * g
        b2 += f2 * g1
        b3 += f3 * g1

        b1 *= 2
        b2 *= 2
        b3 *= 2

        p += g
        p1 += g1

        g = m.copy()
        m += kc

        iter += 1

    dpi = np.pi
    f1 = 0.5 * dpi * (a1 * m + b1) / (m * (m + p))
    f2 = 0.5 * dpi * (a2 * m + b2) / (m * (m + p1))
    f3 = 0.5 * dpi * (a3 * m + b3) / (m * (m + p1))

    return f1, f2, f3

# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
@njit
def ellke(k):
    """
    Computes the complete elliptic integrals of the first (kk) and
    second (ek) kind using Hastings' polynomial approximation.

    Parameters
    ----------
    k : float or array-like
        The elliptic modulus.

    Returns
    -------
    ek : float or ndarray
        Elliptic integral of the second kind.
    kk : float or ndarray
        Elliptic integral of the first kind.
    
    References
    ----------
    - Jason Eastman (2009), EXOFAST/IDL implementation
    - Hastings (1955) polynomial approximation
    """
    k = np.asarray(k, dtype=np.float64)
    m1 = 1.0 - k**2
    logm1 = np.log(m1)

    # Elliptic integral of the second kind
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    ee1 = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * (-logm1)
    ek = ee1 + ee2

    # Elliptic integral of the first kind
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * logm1
    kk = ek1 - ek2

    return ek, kk


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def sqarea_triangle(z0, p0):
    """
    Computes sixteen times the square of the area of a triangle
    with sides 1, z0, and p0 using the Kahan method (Goldberg 1991).

    Parameters:
        z0 (array-like): Lengths of side z0.
        p0 (array-like): Lengths of side p0.

    Returns:
        numpy.ndarray: Array containing 16 times the squared areas.
    """
    z0 = np.array(z0, dtype=np.float64)
    sqarea = np.zeros_like(z0)

    # Six cases to consider
    pz1 = np.where((p0 <= z0) & (z0 <= 1))[0]
    if pz1.size:
        sqarea[pz1] = (p0 + (z0[pz1] + 1)) * (1 - (p0 - z0[pz1])) * \
                      (1 + (p0 - z0[pz1])) * (p0 + (z0[pz1] - 1))

    zp1 = np.where((z0 <= p0) & (p0 <= 1))[0]
    if zp1.size:
        sqarea[zp1] = (z0[zp1] + (p0 + 1)) * (1 - (z0[zp1] - p0)) * \
                      (1 + (z0[zp1] - p0)) * (z0[zp1] + (p0 - 1))

    p1z = np.where((p0 <= 1) & (1 <= z0))[0]
    if p1z.size:
        sqarea[p1z] = (p0 + (1 + z0[p1z])) * (z0[p1z] - (p0 - 1)) * \
                      (z0[p1z] + (p0 - 1)) * (p0 + (1 - z0[p1z]))

    z1p = np.where((z0 <= 1) & (1 <= p0))[0]
    if z1p.size:
        sqarea[z1p] = (z0[z1p] + (1 + p0)) * (p0 - (z0[z1p] - 1)) * \
                      (p0 + (z0[z1p] - 1)) * (z0[z1p] + (1 - p0))

    onepz = np.where((1 <= p0) & (p0 <= z0))[0]
    if onepz.size:
        sqarea[onepz] = (1 + (p0 + z0[onepz])) * (z0[onepz] - (1 - p0)) * \
                        (z0[onepz] + (1 - p0)) * (1 + (p0 - z0[onepz]))

    onezp = np.where((1 <= z0) & (z0 <= p0))[0]
    if onezp.size:
        sqarea[onezp] = (1 + (z0[onezp] + p0)) * (p0 - (1 - z0[onezp])) * \
                        (p0 + (1 - z0[onezp])) * (1 + (z0[onezp] - p0))

    return sqarea



# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def exozippy_occultquad_cel(z0, u1, u2, p0, return_coeffs=False):
    """
    Full translation of exozippy_OCCULTQUAD_CEL from IDL.
    Computes flux for quadratically limb-darkened occultation.
    """
    # Ensure inputs are numpy arrays with double precision
    z = np.asarray(z0, dtype=np.float64) # Checked, good
    p = np.abs(np.float64(p0)) # Checked, good

    nz = len(z)  # Checked, good
    lambdad = np.zeros(nz, dtype=np.float64) # Checked, good
    etad = np.zeros(nz, dtype=np.float64) # Checked, good
    lambdae = np.zeros(nz, dtype=np.float64) # Checked, good

    # Intermediate terms (not used yet in logic, but included for completeness)
    # x1 = (p - z) ** 2
    # x2 = (p + z) ** 2
    # x3 = p ** 2 - z ** 2

    # Case 1: star is unocculted — only consider z < 1 + p and p > 0
    notusedyet = np.where((z < (1.0 + p)) & (p > 0.0))[0]
    if notusedyet.size == 0:
        # goto final (in Python, you can just skip further computation or wrap the rest in an if block)
        pass
    else:
        # Case 11: source completely occulted — if p >= 1 and z <= p - 1
        if p >= 1.0:
            z_notused = z[notusedyet]
            occulted = np.where(z_notused <= (p - 1.0))[0]
            mask = np.ones_like(z_notused, dtype=bool)
            mask[occulted] = False
            notused2 = np.where(mask)[0]


            if occulted.size > 0:
                ndxuse = notusedyet[occulted]
                etad[ndxuse] = 0.5  # corrected typo in paper
                lambdae[ndxuse] = 1.0
                # lambdad stays 0

            if len(notused2) == 0:
                # goto final
                pass
            else:
                notusedyet = notusedyet[notused2]
                
    # Case 2, 7, 8 - ingress/egress (uniform disk only)
    # Ingress/egress region: abs(1 - p) ≤ z < 1 + p
    z_notused = z[notusedyet]
    inegressuni = np.where((z_notused >= np.abs(1.0 - p)) & (z_notused < 1.0 + p))[0]

    if inegressuni.size > 0:
        ndxuse = notusedyet[inegressuni]
        z_ndx = z[ndxuse]

        # Compute triangle area
        sqarea = sqarea_triangle(z_ndx, p)
        kite_area2 = np.sqrt(sqarea)

        # Compute angles
        kap1 = np.arctan2(kite_area2, (1.0 - p) * (1.0 + p) + z_ndx ** 2)
        kap0 = np.arctan2(kite_area2, (p - 1.0) * (1.0 + p) + z_ndx ** 2)

        # lambda_e: uniform disk flux
        lambdae[ndxuse] = (p ** 2 * kap0 + kap1 - 0.5 * kite_area2) / np.pi

        # eta_d
        etad[ndxuse] = (
            1.0 / (2.0 * np.pi) * (
                kap1 +
                p ** 2 * (p ** 2 + 2.0 * z_ndx ** 2) * kap0 -
                0.25 * (1.0 + 5.0 * p ** 2 + z_ndx ** 2) * kite_area2
            )
        )
        
        
    # Case 5, 6, 7 — z == p (edge of planet at origin of star)
    z_notused = z[notusedyet]
    ocltor = np.where(z_notused == p)[0]  # indices where z == p
    # notused3 = np.setdiff1d(np.arange(len(z_notused)), ocltor)
    # notused5 = np.delete(np.arange(len(z_notused)), inside)
    notused3 = np.delete(np.arange(len(z_notused)), ocltor)

    if ocltor.size > 0:
        ndxuse = notusedyet[ocltor]
        z_ndx = z[ndxuse]

        if p < 0.5:
            # Case 5
            q = 2.0 * p
            Ek, Kk = ellke(q)  # complete elliptic integral of the first kind
            # lambda_4
            lambdad[ndxuse] = (1.0 / 3.0 +
                2.0 / (9.0 * np.pi) *
                (4.0 * (2.0 * p**2 - 1.0) * Ek + (1.0 - 4.0 * p**2) * Kk)
            )

            # eta_2
            etad[ndxuse] = 1.5 * p**4  # = 3*p^4/2

            # Uniform disk
            lambdae[ndxuse] = p**2
        elif p > 0.5:
            # Case 7
            q = 0.5 / p
            Ek, Kk = ellke(q)  # complete elliptic integral of the first kind


            # lambda_3
            lambdad[ndxuse] = (
                1.0 / 3.0 +
                (16.0 * p / (9.0 * np.pi)) * (2.0 * p**2 - 1.0) * Ek -
                ((32.0 * p**4 - 20.0 * p**2 + 3.0) / (9.0 * np.pi * p)) * Kk
            )

            # etad = already computed elsewhere (eta_1), so nothing to do

        else:
            # Case 6: p == 0.5 exactly
            lambdad[ndxuse] = 1.0 / 3.0 - (4.0 / (9.0 * np.pi))
            etad[ndxuse] = 3.0 / 32.0
    # Update notusedyet by removing indices we just processed
    if notused3.size == 0:
        # goto final
        pass
    else:
        notusedyet = notusedyet[notused3]
        
        
    # ;; Case 3, 4, 9, 10 - planet completely inside star
    z_notused = z[notusedyet]
    inside = np.where((p < 1.0) & (z_notused <= (1.0 - p)))[0]
    # notused5 = np.setdiff1d(np.arange(len(z_notused)), inside)
    notused5 = np.delete(np.arange(len(z_notused)), inside)

    if inside.size > 0:
        ndxuse = notusedyet[inside]
        z_ndx = z[ndxuse]

        # eta_2
        etad[ndxuse] = 0.5 * p ** 2 * (p ** 2 + 2 * z_ndx ** 2)
        # uniform disk
        lambdae[ndxuse] = p ** 2
        # Case 4: edge of planet hits edge of star
        edge = np.where(z_ndx == (1.0 - p))[0]
        # notused6 = np.setdiff1d(np.arange(len(z_ndx)), edge)
        notused6 = np.delete(np.arange(len(z_ndx)), edge)

        if edge.size > 0:
            term1 = (2.0 / (3.0 * np.pi)) * np.arccos(1.0 - 2.0 * p)
            term2 = (4.0 / (9.0 * np.pi)) * np.sqrt(p * (1.0 - p)) * (3.0 + 2.0 * p - 8.0 * p ** 2)
            term3 = (2.0 / 3.0) if p > 0.5 else 0.0
            lambdad[ndxuse[edge]] = term1 - term2 - term3
            if notused6.size == 0:
                pass
            else:
                ndxuse = ndxuse[notused6]
                z_ndx = z[ndxuse]
        # Case 10: center of planet hits center of star
        origin = np.where(z_ndx == 0.0)[0]
        # notused7 = np.setdiff1d(np.arange(len(z_ndx)), origin)
        notused7 = np.delete(np.arange(len(z_ndx)), origin)
        
        if origin.size > 0:
            lambdad[ndxuse[origin]] = - (2.0 / 3.0) * (1.0 - p ** 2) ** 1.5

            if notused7.size == 0:
                pass
            else:
                ndxuse = ndxuse[notused7]
                z_ndx = z[ndxuse]

        # Prepare arguments for cel_bulirsch_vec
        onembpr2 = (1 - z_ndx - p) * (1 + z_ndx + p)
        onembmr2 = (p - z_ndx + 1) * (1 - p + z_ndx)
        fourbr = 4 * z_ndx * p
        fourbrinv = 1.0 / fourbr
        k2 = onembpr2 * fourbrinv + 1.0
        k2inv = 1.0 / k2
        kc2 = onembpr2 / onembmr2
        kc = np.sqrt(np.clip(kc2, 0, None))

        bmrdbpr = (z_ndx - p) / (z_ndx + p)
        mu = 3 * bmrdbpr / onembmr2
        p_bulirsch = bmrdbpr ** 2 * onembpr2 / onembmr2

        # Run Bulirsch-style elliptic integral vector routine
        Piofk, Eofk, Em1mKdm = cel_bulirsch_vec(
            k2=k2inv,
            kc=kc.copy(),
            p=p_bulirsch.copy(),
            a1=1 + mu,
            a2=np.ones_like(k2),
            a3=np.ones_like(k2),
            b1=p_bulirsch + mu,
            b2=kc2,
            b3=np.zeros_like(k2),
            f1=None,
            f2=None,
            f3=None
        )

        # Final expression for lambdad
        lambdad[ndxuse] = (
            2 * np.sqrt(onembmr2) *
            (onembpr2 * Piofk - (4 - 7 * p ** 2 - z_ndx ** 2) * Eofk)
            / (9.0 * np.pi)
        )

    # Case 2, 8 – Ingress/Egress with limb darkening
    inegress = notused5
    if inegress.size > 0:
        ndxuse = notusedyet[inegress]
        z_ndx = z[ndxuse]

        # Geometric terms
        onembpr2 = (1 - z_ndx - p) * (1 + z_ndx + p)
        onembmr2 = (p - z_ndx + 1) * (1 - p + z_ndx)
        fourbr = 4 * z_ndx * p
        fourbrinv = 1.0 / fourbr

        k2 = onembpr2 * fourbrinv + 1
        kc2 = -onembpr2 * fourbrinv
        kc = np.sqrt(np.clip(kc2, 0, None))  # clamp to prevent NaN

        # Arguments for cel_bulirsch_vec
        a1 = np.zeros_like(k2)         # 0.0
        a2 = np.ones_like(k2)          # 1.0
        a3 = np.ones_like(k2)          # 1.0
        b1 = 3 * kc2 * (z_ndx - p) * (z_ndx + p)
        b2 = kc2
        b3 = np.zeros_like(k2)
        p_bulirsch = (z_ndx - p) ** 2 * kc2

        # Call vectorized Bulirsch integrator
        Piofk, Eofk, Em1mKdm = cel_bulirsch_vec(
            k2=k2,
            kc=kc.copy(),
            p=p_bulirsch.copy(),
            a1=a1,
            a2=a2,
            a3=a3,
            b1=b1,
            b2=b2,
            b3=b3,
            f1=None,
            f2=None,
            f3=None
        )
        # Final lambdad for ingress/egress + limb darkening
        lambdad[ndxuse] = (
            onembmr2 * (
                Piofk +
                (-3 + 6 * p ** 2 + 2 * z_ndx * p) * Em1mKdm -
                fourbr * Eofk
            ) / (9 * np.pi * np.sqrt(z_ndx * p))
        )
    # === Final Light Curve Computation ===
    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    z_mask = p > z  # for condition (p > z)
    if p0 > 0:
        # Limb-darkened flux
        muo1 = 1.0 - (
            (1.0 - u1 - 2.0 * u2) * lambdae +
            (u1 + 2.0 * u2) * (lambdad + (2.0 / 3.0) * z_mask) +
            u2 * etad
        ) / omega

        # Uniform disk
        mu0 = 1.0 - lambdae

        # Optional limb darkening coefficient output
        if return_coeffs:  # mimic `arg_present(d)` behavior
            d = np.array([
                1.0 - lambdae,
                (2.0 / 3.0) * (lambdae - z_mask) - lambdad,
                lambdae / 2.0 - etad
            ])
            return muo1, mu0, d
        else:
            return muo1, mu0
    else:
        # Negative p0 — treat as anti-transit (e.g., for symmetry or edge cases)
        muo1 = 1.0 + (
            (1.0 - u1 - 2.0 * u2) * lambdae +
            (u1 + 2.0 * u2) * (lambdad + (2.0 / 3.0) * z_mask) +
            u2 * etad
        ) / omega

        mu0 = 1.0 + lambdae
        if return_coeffs:
            d = np.array([
                1.0 + lambdae,
                (2.0 / 3.0) * (z_mask - lambdae) + lambdad,
                etad - lambdae / 2.0
            ])
            return muo1, mu0, d
        else:
            return muo1, mu0