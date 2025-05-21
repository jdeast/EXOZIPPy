import numpy as np
import os
import re

def readrv(filename):
    """
    Reads a radial velocity (RV) file and returns a dictionary with structured data.

    Parameters
    ----------
    filename : str
        Path to the RV file.

    Returns
    -------
    rv_data : dict
        Dictionary containing:
        - bjd, rv, err: arrays of data
        - bi, bierr: duplicated error arrays (with NaNs allowed)
        - label: file label from filename
        - residuals, rm: zero arrays for later use
        - planet: int, if filename includes SB2_<planet>
    """
    label = os.path.splitext(os.path.basename(filename))[0]

    # Initialize planet index
    planet = -1

    with open(filename, 'r') as f:
        first_line = f.readline()
        match = re.search(r'SB2_(\d+)', first_line)
        if match:
            planet = int(match.group(1))

    # Read columns: bjd, rv, err
    bjd, rv, err = np.loadtxt(filename, unpack=True, comments='#', usecols=(0, 1, 2))

    # Allow NaNs in bi, bierr for further handling
    bi = err + np.nan
    bierr = err + np.nan

    residuals = np.zeros_like(bjd)
    rm = np.zeros_like(bjd)

    rv_data = {
        'bjd': bjd,
        'rv': rv,
        'err': err,
        'bi': bi,
        'bierr': bierr,
        'label': label,
        'residuals': residuals,
        'rm': rm,
        'planet': planet
    }

    return rv_data
