import numpy as np
from ..exozippy_keplereq import *
from ..utils import *
import requests
import re
import os
import exozippy
from scipy.io import readsav
from numba import njit
import functools
import os
import pathlib
def filepath(filename, root_dir, subdir):
    return os.path.join(root_dir, *subdir, filename)

def file_lines(filename):
    with open(filename) as f:
        return len(f.readlines())



# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
# def ninterpolate(data, point):
#     """
#     Perform N-dimensional linear interpolation at a single point.

#     Parameters
#     ----------
#     data : ndarray
#         The N-dimensional data array to interpolate.
#     point : array-like
#         A sequence of length N giving the coordinates at which to interpolate.

#     Returns
#     -------
#     float
#         Interpolated value.
        
#     ; MODIFICATION HISTORY:
#     ;
#     ;    Mon Jul 21 12:33:30 2003, J.D. Smith <jdsmith@as.arizona.edu>
#     ;		Written.
#     ;==========================================================================
#     ; Copyright (C) 2003, J.D. Smith
#     """
#     point = np.asarray(point)
#     ndim = point.size

#     if data.ndim != ndim:
#         raise ValueError("Point must specify 1 coordinate for each dimension of data")

#     if ndim == 1:
#         # 1D special case: np.interp
#         x = np.arange(data.shape[0])
#         return np.interp(point[0], x, data)

#     base = np.floor(point).astype(int)
#     frac = point - base
#     result = 0.0

#     for i in range(2 ** ndim):
#         # Get corner offset in binary
#         offset = [(i >> k) & 1 for k in range(ndim)]
#         idx = tuple(base[k] + offset[k] for k in range(ndim))

#         # Check bounds
#         if any(j < 0 or j >= data.shape[k] for k, j in enumerate(idx)):
#             continue  # skip out-of-bounds

#         weight = 1.0
#         for k in range(ndim):
#             weight *= (1 - frac[k]) if offset[k] == 0 else frac[k]

#         result += weight * data[idx]

#     return result

from scipy.ndimage import map_coordinates
# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
import numpy as np


def ninterpolate(data, point):
    """
    Multilinear interpolation of an n‑dimensional array.

    Parameters
    ----------
    data : ndarray
        n‑D array of values.
    point : 1‑D sequence (length n)
        Real‑valued coordinates at which to interpolate.

    Returns
    -------
    float
        Interpolated value at `point`.
    """
    data = np.asarray(data, dtype=float)
    point = np.asarray(point, dtype=float)

    # -- dimension checks -------------------------------------------------
    n = point.size
    if n != data.ndim:
        raise ValueError("`point` must supply one coordinate for each dimension")

    # -- trivial 1‑D case -------------------------------------------------
    if n == 1:
        x = point[0]
        x0 = int(np.floor(x))
        x1 = x0 + 1
        f = x - x0                        # fractional distance
        x0 = np.clip(x0, 0, data.shape[0] - 1)
        x1 = np.clip(x1, 0, data.shape[0] - 1)
        return (1.0 - f) * data[x0] + f * data[x1]

    # -- general n‑D multilinear case ------------------------------------
    base = np.floor(point).astype(int)    # “lower‑left” corner of the cell
    f = point - base                      # fractional part in each dimension

    value = 0.0
    two_n = 1 << n                        # 2**n corner combinations

    for corner in range(two_n):
        idx = base.copy()
        weight = 1.0
        for dim in range(n):
            if corner & (1 << dim):       # high corner along this axis
                idx[dim] += 1
                weight *= f[dim]
            else:                         # low corner along this axis
                weight *= (1.0 - f[dim])

            # stay inside array bounds
            idx[dim] = np.clip(idx[dim], 0, data.shape[dim] - 1)

        value += weight * data[tuple(idx)]

    return value


@functools.lru_cache(maxsize=8)
def _load_mist_grid(grid_path: str):
    """Read and cache mist.sed.grid.idl (grid definitions for the BC cubes)."""
    g = readsav(grid_path, python_dict=True)
    return g['teffgrid'], g['logggrid'], g['fehgrid'], g['avgrid']


@functools.lru_cache(maxsize=32)
def _load_bc_cube(bc_path: str):
    """Read and cache a single band’s 4‑D bolometric‑correction cube."""
    s = readsav(bc_path, python_dict=True)
    return s['bcarray'], s['filterproperties']

def mistmultised(teff, logg, feh, av, distance, lstar, errscale, sedfile,
                  *,
                  redo=False,
                  psname=None, debug=False, atmospheres=None,
                  wavelength=None, logname=None, xyrange=None,
                  blend0=None):
    """
    Parameters
    ----------
    teff, logg, feh, av, distance, lstar : array‑like, shape (nstars,)
    errscale   : scalar or (≥1,) array — identical to the IDL behaviour
    sedfile    : str  — path to the observed SED definition file

    Returns
    -------
    sedchi2      : float
    blendmag     : (nbands,) ndarray
    modelflux    : (nbands, nstars) ndarray
    magresiduals : (nbands,) ndarray
    """

    # ---------- 1. Input validation ---------------------------------------
    teff   = np.atleast_1d(teff).astype(float)
    nstars = teff.size

    def _check(name, arr):
        a = np.atleast_1d(arr).astype(float)
        if a.size != nstars:
            raise ValueError(f"{name} must have same length as teff")
        return a

    logg     = _check('logg',     logg)
    feh      = _check('feh',      feh)
    av       = _check('av',       av)
    distance = _check('distance', distance)
    lstar    = _check('lstar',    lstar)

    err0 = float(np.atleast_1d(errscale)[0])

    # ---------- 2. Read observed SED file ---------------------------------
    sed_data   = read_sed_file(sedfile, nstars, logname=logname)
    sedbands   = sed_data['sedbands']
    mags       = sed_data['mag']
    errs       = sed_data['errmag']
    blend_spec = sed_data['blend']
    nbands     = len(sedbands)
    
    # ---------- 3. Load / cache the MIST grid & BC cubes ------------------
    root = pathlib.Path(exozippy.MODULE_PATH) / 'EXOZIPPy' / 'exozippy' / 'sed' / 'mist'
    gridfile = root / 'mist.sed.grid.idl'
    teffgrid, logggrid, fehgrid, avgrid = _load_mist_grid(str(gridfile))

    # Filter mapping table
    kname, mname, cname, svoname = np.loadtxt(
        root / 'filternames2.txt', dtype=str, comments="#", unpack=True
    )

    bc_cubes, filterprops = [], []
    for band in sedbands:
        # Replicates the fallback order in the IDL code
        candidates = [band]
        if band in kname:
            candidates.append(mname[np.where(kname == band)[0][0]])
        if band in svoname:
            candidates.append(mname[np.where(svoname == band)[0][0]])

        for cand in candidates:
            bc_path = root / f"{cand}.idl"
            if bc_path.exists():
                bc, props = _load_bc_cube(str(bc_path))     
                bc = np.transpose(bc, (3, 2, 1, 0))
                bc_cubes.append(bc)
                filterprops.append(props)
                break
        else:
            raise FileNotFoundError(f"{band} not supported – remove it from {sedfile}")

    bcarrays = np.stack(bc_cubes, axis=-1)  # shape: (nteff, nlogg, nfeh, nav, nbands)
    # ---------- 4. Expand blend matrix to (nbands, nstars) ----------------
    blend = np.zeros((nbands, nstars), dtype=int)
    for i, token in enumerate(blend_spec):
        if token == 'ALL':
            blend[i, :] = 1
            continue
        if '-' in token:                      # differential magnitudes
            pos, neg = (np.fromstring(t, sep=',', dtype=int)
                        for t in token.split('-'))
            blend[i, np.clip(pos, 0, nstars-1)] = +1
            blend[i, np.clip(neg, 0, nstars-1)] = -1
        else:                                 # unblended magnitudes
            idx = np.fromstring(token, sep=',', dtype=int)
            blend[i, np.clip(idx, 0, nstars-1)] = 1

    if blend0 is not None:
        blend0[:] = blend.copy()

    # ---------- 5. Interpolate bolometric corrections ---------------------
    bcs = np.empty((nbands, nstars))
    for j in range(nstars):
        coord = [get_grid_point(g, v) for g, v in
                 ((teffgrid, teff[j]),
                  (logggrid, logg[j]),
                  (fehgrid,  feh[j]),
                  (avgrid,   av[j]))]
        for i in range(nbands):
            bcs[i, j] = ninterpolate(bcarrays[..., i], coord)
    # ---------- 6. Model magnitudes / fluxes ------------------------------
    mu         = 5.0 * np.log10(distance) - 5.0         # (nstars,)
    logL_term  = -2.5 * np.log10(lstar)                 # (nstars,)
    modelmag   = (logL_term[None, :] + 4.74             # (nbands, nstars)
                  - bcs + mu[None, :])

    modelflux  = 10.0 ** (-0.4 * modelmag)              # (nbands, nstars)

    if nstars == 1:
        blendmag       = modelmag[:, 0]                 # (nbands,)
        blendflux      = modelflux[:, 0]
        magresiduals   = mags - blendmag
    else:
        pos_flux  = (modelflux * (blend > 0)).sum(axis=1)
        neg_flux  = (modelflux * (blend < 0)).sum(axis=1)
        neg_flux[neg_flux == 0.0] = 1.0
        blendmag       = -2.5 * np.log10(pos_flux / neg_flux)
        magresiduals   = mags - blendmag
        blendflux      = (modelflux * blend).sum(axis=1)

    # ---------- 7. χ² likelihood ------------------------------------------
    sedchi2 = np.sum((magresiduals / (errs * err0)) ** 2)
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
