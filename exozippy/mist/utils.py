import numpy as np
import os
from scipy.io import readsav


def readeep(mstar, feh, vvcrit=0.0, alpha=0.0, rstar=None, teff=None, age=None, v1_2=False):
    """
    Load MIST evolutionary track data for given parameters.

    Parameters
    ----------
    mstar : float
        Stellar mass (Msun).
    feh : float
        Metallicity [Fe/H].
    vvcrit : float, optional
        Fraction of critical velocity (default: 0.0).
    alpha : float, optional
        Alpha-element enhancement (default: 0.0).
    rstar, teff, age : optional
        Not used in this function, passed for interface compatibility.
    v1_2 : bool, optional
        If True, use MIST v1.2 instead of v1.0.

    Returns
    -------
    dict or -1
        Evolutionary track dictionary or -1 if input is invalid.
    """

    allowed_feh = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5,
                            -0.25, 0.0, 0.25, 0.5])
    if feh not in allowed_feh:
        print("[Fe/H] must be one of:", allowed_feh)
        return -1
    sign = 'm' if feh < 0 else 'p'
    fehstr = f"{sign}{abs(feh):.2f}"

    allowed_alpha = np.array([0.0])
    if alpha not in allowed_alpha:
        print("[alpha/Fe] must be one of:", allowed_alpha)
        return -1
    sign = 'm' if alpha < 0 else 'p'
    alphastr = f"{sign}{abs(alpha):.1f}"

    allowed_vvcrit = np.array([0.0, 0.4])
    if vvcrit not in allowed_vvcrit:
        print("v/vcrit must be one of:", allowed_vvcrit)
        return -1
    vvcritstr = f"{vvcrit:.1f}"

    allowed_mass = np.array([
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
        0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06,
        1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24,
        1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42,
        1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58, 1.60,
        1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78,
        1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96,
        1.98, 2.00, 2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14,
        2.16, 2.18, 2.20, 2.22, 2.24, 2.26, 2.28, 2.30, 2.32,
        2.34, 2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48, 2.50,
        2.52, 2.54, 2.56, 2.58, 2.60, 2.62, 2.64, 2.66, 2.68,
        2.70, 2.72, 2.74, 2.76, 2.78, 2.80, 3.00, 3.20, 3.40,
        3.60, 3.80, 4.00, 4.20, 4.40, 4.60, 4.80, 5.00, 5.20,
        5.40, 5.60, 5.80, 6.00, 6.20, 6.40, 6.60, 6.80, 7.00,
        7.20, 7.40, 7.60, 7.80, 8.00, 9.00, 10.00, 11.00,
        12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00,
        19.00, 20.00, 22.00, 24.00, 26.00, 28.00, 30.00,
        32.00, 34.00, 36.00, 38.00, 40.00, 45.00, 50.00,
        55.00, 60.00, 65.00, 70.00, 75.00, 80.00, 85.00,
        90.00, 95.00, 100.00, 105.00, 110.00, 115.00, 120.00,
        125.00, 130.00, 135.00, 140.00, 145.00, 150.00, 175.00,
        200.00, 225.00, 250.00, 275.00, 300.00
    ])

    if mstar not in allowed_mass:
        print("mstar must be one of:", allowed_mass)
        return -1

    mstarstr = f"{int(round(mstar * 100)):05d}"

    base_dir = os.getenv("EXOFAST_PATH")
    if v1_2:
        subdir = os.path.join("mist", f"MIST_v1.2_tracks", f"feh_{fehstr}_afe_{alphastr}_vvcrit{vvcritstr}")
    else:
        subdir = os.path.join("mist", f"MIST_v1.0_tracks", f"feh_{fehstr}_afe_{alphastr}_vvcrit{vvcritstr}", "eeps")

    eepfile = os.path.join(base_dir, subdir, f"{mstarstr}M.track.eep.idl")

    try:
        track = readsav(eepfile, python_dict=True)
        return track
    except FileNotFoundError:
        print(f"EEP file not found: {eepfile}")
        return -1
