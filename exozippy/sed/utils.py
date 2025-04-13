
import numpy as np
from .exozippy_keplereq import *
import requests
import re
import os

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
