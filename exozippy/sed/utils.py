import numpy as np
from ..exozippy_keplereq import *
from ..utils import *
import requests
import re
import os
import exozippy
from scipy.io import readsav
from numba import njit

def filepath(filename, root_dir, subdir):
    return os.path.join(root_dir, *subdir, filename)

def file_lines(filename):
    with open(filename) as f:
        return len(f.readlines())



# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def ninterpolate(data, point):
    """
    Perform N-dimensional linear interpolation at a single point.

    Parameters
    ----------
    data : ndarray
        The N-dimensional data array to interpolate.
    point : array-like
        A sequence of length N giving the coordinates at which to interpolate.

    Returns
    -------
    float
        Interpolated value.
        
    ; MODIFICATION HISTORY:
    ;
    ;    Mon Jul 21 12:33:30 2003, J.D. Smith <jdsmith@as.arizona.edu>
    ;		Written.
    ;==========================================================================
    ; Copyright (C) 2003, J.D. Smith
    """
    point = np.asarray(point)
    ndim = point.size

    if data.ndim != ndim:
        raise ValueError("Point must specify 1 coordinate for each dimension of data")

    if ndim == 1:
        # 1D special case: np.interp
        x = np.arange(data.shape[0])
        return np.interp(point[0], x, data)

    base = np.floor(point).astype(int)
    frac = point - base
    result = 0.0

    for i in range(2 ** ndim):
        # Get corner offset in binary
        offset = [(i >> k) & 1 for k in range(ndim)]
        idx = tuple(base[k] + offset[k] for k in range(ndim))

        # Check bounds
        if any(j < 0 or j >= data.shape[k] for k, j in enumerate(idx)):
            continue  # skip out-of-bounds

        weight = 1.0
        for k in range(ndim):
            weight *= (1 - frac[k]) if offset[k] == 0 else frac[k]

        result += weight * data[idx]

    return result


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def mistmultised(teff, logg, feh, av, distance, lstar, errscale, sedfile,
                 redo=False, psname=None, debug=False, atmospheres=None,
                 wavelength=None, logname=None, xyrange=None, blend0=None):

    nstars = len(teff)
    if not all(len(x) == nstars for x in [logg, feh, av, distance, lstar]):
        raise ValueError("All stellar parameter arrays must have the same length as teff")

    # Load SED + filter info
    sed_data = read_sed_file(sedfile, nstars, logname=logname)
    sedbands = sed_data['sedbands']
    mags = sed_data['mag']
    errs = sed_data['errmag']
    blend = sed_data['blend']

    nbands = len(sedbands)

    # Load filter mapping
    filterfile = filepath('filternames2.txt', exozippy.MODULE_PATH, ['EXOZIPPy','exozippy','sed', 'mist'])
    keivanname, mistname, claretname, svoname = np.loadtxt(
        filterfile, dtype=str, comments="#", unpack=True
    )

    # Load grid
    mistgridfile = filepath('mist.sed.grid.idl', exozippy.MODULE_PATH, ['EXOZIPPy','exozippy','sed', 'mist'])
    grid = readsav(mistgridfile, python_dict=True)
    teffgrid, logggrid, fehgrid, avgrid = grid['teffgrid'], grid['logggrid'], grid['fehgrid'], grid['avgrid']

    # Load BC tables for each band
    bcarrays = []
    filterprops = []
    for band in sedbands:
        found = False
        candidates = [band]
        if band in keivanname:
            candidates.append(mistname[np.where(keivanname == band)[0][0]])
        if band in svoname:
            candidates.append(mistname[np.where(svoname == band)[0][0]])

        for name in candidates:
            bcarraysfilepath = filepath(f"{name}.idl", exozippy.MODULE_PATH, ['EXOZIPPy','exozippy','sed', 'mist'])
            if os.path.exists(bcarraysfilepath):
                data = readsav(bcarraysfilepath, python_dict=True)
                bcarrays.append(data['bcarray'])
                filterprops.append(data['filterproperties'])
                found = True
                break

        if not found:
            raise FileNotFoundError(f"{band} not supported, please remove it from {sedfile}")

    bcarrays = np.stack(bcarrays, axis=-1)  # shape: (nteff, nlogg, nfeh, nav, nbands)

    # Distance modulus
    mu = 5.0 * np.log10(distance) - 5.0

    # Interpolate BCs for each star and band
    bcs = np.zeros((nbands, nstars))
    for j in range(nstars):
        teff_ndx = get_grid_point(teffgrid, teff[j])
        logg_ndx = get_grid_point(logggrid, logg[j])
        feh_ndx = get_grid_point(fehgrid, feh[j])
        av_ndx = get_grid_point(avgrid, av[j])

        for i in range(nbands):
            bcs[i, j] = ninterpolate(
                bcarrays[..., i],
                [teff_ndx, logg_ndx, feh_ndx, av_ndx]
            )

    if blend0 is not None:
        blend0[:] = blend.copy()

    # Compute model magnitudes and fluxes
    modelmag = -2.5 * np.log10(lstar @ np.ones((1, nbands))) + 4.74 - bcs + np.outer(np.ones(nbands), mu)
    modelflux = 10 ** (-0.4 * modelmag)

    if nstars == 1:
        blendmag = modelmag[:, 0]
        blendflux = modelflux[:, 0]
        magresiduals = mags - blendmag
    else:
        blendfluxpos = np.sum(modelflux * (blend > 0), axis=1)
        blendfluxneg = np.sum(modelflux * (blend < 0), axis=1)
        blendfluxneg[blendfluxneg == 0.0] = 1.0
        blendmag = -2.5 * np.log10(blendfluxpos / blendfluxneg)
        magresiduals = mags - blendmag
        blendflux = np.sum(modelflux * blend, axis=1)

    # Chi2 likelihood
    sedchi2 = np.sum(((magresiduals / (errs * errscale[0])) ** 2))

    return sedchi2, blendmag, modelflux, magresiduals


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def get_grid_point(grid, value):
    """
    Linearly interpolate or extrapolate the index corresponding to a value in a 1D grid.

    Parameters:
    grid (np.ndarray): 1D sorted array of grid points.
    value (float): The value to locate within the grid.

    Returns:
    float: The fractional index position of the value.
    """
    grid = np.asarray(grid)
    ngrid = grid.size

    # Find index of the closest grid point
    match = np.argmin(np.abs(grid - value))

    if match == ngrid - 1:
        # Extrapolate beyond last grid point
        ndx = match + (value - grid[-1]) / (grid[-1] - grid[-2])
    elif match == 0:
        # Extrapolate before first grid point
        ndx = match + (value - grid[0]) / (grid[1] - grid[0])
    else:
        # Interpolate between two nearest grid points
        if value > grid[match]:
            ndx = match + (value - grid[match]) / (grid[match + 1] - grid[match])
        else:
            ndx = match + (value - grid[match]) / (grid[match] - grid[match - 1])

    return ndx


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def read_coeffs(filename, delimiter=','):
    """
    Reads Gaia EDR3 zero point coefficient files.
    Returns:
        j, k : index arrays for basis functions
        g    : g_mag bins
        q_jk : coefficient matrix of shape (len(g), m)
        n, m : dimensions
    """
    with open(filename, 'r') as f:
        j_line = f.readline()
        k_line = f.readline()

    j = np.array(j_line.strip().split(delimiter)[1:], dtype=int)
    k = np.array(k_line.strip().split(delimiter)[1:], dtype=int)

    data = np.genfromtxt(filename, delimiter=delimiter, skip_header=2)

    g = data[:, 0]
    q_jk = data[:, 1:]

    if q_jk.shape[1] != len(j):
        raise ValueError("Mismatch in number of coefficients")

    n, m = q_jk.shape
    return j, k, g, q_jk, n, m

# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def get_zpt(phot_g_mean_mag, nu_eff_used_in_astrometry, pseudocolor,
            ecl_lat, astrometric_params_solved, exofast_path='./'):
    """
    Gaia EDR3 parallax zero-point correction based on Lindegren+ 2021 prescription.
    """
    # Determine file based on solution type
    if astrometric_params_solved == 31:
        color = nu_eff_used_in_astrometry
        filename = os.path.join(exofast_path, 'sed', 'z5_200720.txt')
        j, k, g, q_jk, n, m = read_coeffs(filename, delimiter=' ')
    elif astrometric_params_solved == 95:
        color = pseudocolor
        filename = os.path.join(exofast_path, 'sed', 'z6_200720.txt')
        j, k, g, q_jk, n, m = read_coeffs(filename, delimiter=',')
    else:
        raise ValueError("Unknown astrometric_params_solved value (expected 31 or 95)")

    sinbeta = np.sin(np.radians(ecl_lat))

    # Color basis functions
    c = np.array([
        1.0,
        np.clip(color - 1.48, -0.24, 0.24),
        min(0.24, max(0.0, 1.48 - color))**3,
        min(0.0, color - 1.24),
        max(0.0, color - 1.72)
    ])

    # Latitude basis functions
    b = np.array([
        1.0,
        sinbeta,
        sinbeta**2 - 1.0 / 3.0
    ])

    # Determine g-mag bin
    if phot_g_mean_mag <= 6.0:
        ig = 0
    elif phot_g_mean_mag > 20.0:
        ig = len(g) - 2
    else:
        ig = np.where(phot_g_mean_mag >= g)[0][-1]

    h = np.clip((phot_g_mean_mag - g[ig]) / (g[ig + 1] - g[ig]), 0.0, 1.0)

    # Interpolate and compute zpt
    zpt = 0.0
    for i in range(m):
        coeff = (1.0 - h) * q_jk[ig, i] + h * q_jk[ig + 1, i]
        zpt += coeff * c[j[i]] * b[k[i]]

    return zpt / 1e3  # Return in arcseconds


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def read_sed_file(
    sedfile,
    nstars,
    sedbands=None,
    mag=None,
    errmag=None,
    blend=None,
    flux=None,
    errflux=None,
    filter_curves=None,
    weff=None,
    widtheff=None,
    zero_point=None,
    download_new=False,
    filter_curve_sum=None,
    logname=None,
):


    nlines = file_lines(sedfile)

    sedbands = np.empty(nlines, dtype=object)
    mag = np.empty(nlines)
    errmag = np.full(nlines, 99.0)
    flux = np.empty(nlines)
    errflux = np.full(nlines, 99.0)
    blend = np.zeros((nlines, nstars), dtype=int)
    filter_curves = np.empty((nlines, 24000))
    weff = np.empty(nlines)
    widtheff = np.empty(nlines)
    zero_point = np.empty(nlines)
    filter_curve_sum = np.empty(nlines)

    with open(sedfile, 'r') as f:
        lines = f.readlines()

    # Load filter name mapping
    root_dir = exozippy.MODULE_PATH
    filter_file = filepath('filternames2.txt', root_dir, ['EXOZIPPy','exozippy','sed', 'mist'])
    keivanname, mistname, claretname, svoname = np.loadtxt(
        filter_file,
        dtype=str,
        comments='#',
        unpack=True,
    )

    for i, line in enumerate(lines):
        line = line.strip()
        line = line.split('#')[0].strip()  # Remove comments
        if not line:
            continue

        entries = line.split()
        if len(entries) < 3:
            printandlog(f'Line {i+1} in SED file not a legal line: {lines[i]}', logname)
            continue

        sedbands[i] = entries[0]
        mag[i] = float(entries[1])
        errmag[i] = float(entries[2])

        # Attempt to load filter curve
        idlfile = filepath(sedbands[i] + '.idl', root_dir, ['EXOZIPPy','exozippy','sed', 'filtercurves'])

        if not os.path.isfile(idlfile):
            match = np.where(keivanname == sedbands[i])[0]
            if match.size == 1:
                if svoname[match[0]] == 'Unsupported':
                    printandlog(f'{sedbands[i]} is unsupported; try using the SVO name', logname)
                    continue
                idlfile = filepath(svoname[match[0]] + '.idl', root_dir,  ['EXOZIPPy','exozippy','sed', 'filtercurves'])
            else:
                match = np.where(mistname == sedbands[i])[0]
                if match.size == 1:
                    if svoname[match[0]] == 'Unsupported':
                        printandlog(f'{sedbands[i]} is unsupported; try using the SVO name', logname)
                        continue
                    idlfile = filepath(svoname[match[0]] + '.idl', root_dir,  ['EXOZIPPy','exozippy','sed', 'filtercurves'])

        if not os.path.isfile(idlfile) and download_new:
            getfilter(sedbands[i])  # needs implementation

        if not os.path.isfile(idlfile):
            printandlog(f'band="{sedbands[i]}" in SED file not recognized; skipping', logname)
            errmag[i] = 99.0
            continue

        filter = (readsav(idlfile, python_dict=True))['filter']
        filter_curves[i, :] = filter['transmission'][0]
        weff[i] = filter['weff'][0]
        widtheff[i] = filter['widtheff'][0]
        zero_point[i] = filter['zero_point'][0]
        filter_curve_sum[i] = np.sum(filter_curves[i, :])

        flux[i] = zero_point[i] * 10 ** (-0.4 * mag[i])
        errflux[i] = flux[i] * np.log(10) / 2.5 * errmag[i]

        if len(entries) == 5:
            if '-' in entries[4]:
                pos_part, neg_part = entries[4].split('-')
                posndx = np.array([int(x) for x in pos_part.split(',')])
                good = posndx[(posndx < nstars) & (posndx >= 0)]
                bad = np.setdiff1d(posndx, good)
                if bad.size > 0:
                    printandlog(f'WARNING: STARNDX ({bad}) does not correspond to a star', logname)
                if good.size == 0:
                    continue
                blend[i, good] = 1

                negndx = np.array([int(x) for x in neg_part.split(',')])
                good = negndx[(negndx < nstars) & (negndx >= 0)]
                bad = np.setdiff1d(negndx, good)
                if bad.size > 0:
                    printandlog(f'WARNING: STARNDX ({bad}) does not correspond to a star', logname)
                if good.size == 0:
                    continue
                blend[i, good] = -1
            else:
                starndx = np.array([int(x) for x in entries[4].split(',')])
                good = starndx[(starndx < nstars) & (starndx >= 0)]
                bad = np.setdiff1d(starndx, good)
                if bad.size > 0:
                    printandlog(f'WARNING: STARNDX ({bad}) does not correspond to a star', logname)
                if good.size == 0:
                    continue
                blend[i, good] = 1
        else:
            blend[i, :] = 1  # Assume blended by default

    good = np.where(errmag < 1.0)[0]
    if good.size > 1:
        sedbands = sedbands[good]
        mag = mag[good]
        errmag = errmag[good]
        flux = flux[good]
        errflux = errflux[good]
        blend = blend[good, :]
        filter_curves = filter_curves[good, :]
        weff = weff[good]
        widtheff = widtheff[good]
        zero_point = zero_point[good]
        filter_curve_sum = filter_curve_sum[good]
    else:
        printandlog("Bands must have errors less than 1 mag; no good bands", logname)
        raise ValueError("No valid SED bands")

    return {
        'sedbands': sedbands,
        'mag': mag,
        'errmag': errmag,
        'flux': flux,
        'errflux': errflux,
        'blend': blend,
        'filter_curves': filter_curves,
        'weff': weff,
        'widtheff': widtheff,
        'zero_point': zero_point,
        'filter_curve_sum': filter_curve_sum
    }
