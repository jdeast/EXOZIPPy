"""The extinction laws: the shipped table and CCM89 + O'Donnell 94.

ONE READER OF models/extinction_law.ascii, because there were three and
they had already started to differ in convention.  make_bc.py applies it
to spectra and sed/plot.py draws it.  It is no longer run BACKWARDS: the
monochromatic inversions of a measured band extinction into an A_V prior
(av_from_band_extinction, av_from_colour_excess) were removed with review
2.9.16(b) -- a published clump extinction is a prior on the red clump's
integrated band extinction (components/redclump/redclump.md), which the
forward model computes.

THE CONVENTION THAT MATTERS, measured 2026-09-16.  A band's extinction must
be evaluated at the band's EFFECTIVE WAVELENGTH, not by integrating k(lam)
across the passband.  For Roman's W149 (0.90-2.07 micron) the two differ by
17%: effective-wavelength gives A_Z087/A_W149 = 1.922 and the integral
gives 2.250.  The DC2018 simulation's own per-event extinctions have a ratio
of 1.924 (median 1.932 over 293 events), so the effective-wavelength
convention is the one the data are on -- and an earlier version of this
analysis reported the integral, concluded our law disagreed with the
simulation's by 17%, and was wrong.

WHAT THE TABLE IS: byte-identical to $EXOFAST_PATH/sed/extinction_law.ascii
(EXOFASTv2, from Keivan Stassun, uncited there), spanning 0.0103 to 4870
micron -- a dust-opacity model's range rather than an extinction curve's,
normalized at V by every caller.  Its shape matches Cardelli, Clayton &
Mathis (1989) at R_V = 3.1 to about 2% in the optical but runs
SYSTEMATICALLY ~6% HIGH in J, H and Ks, which looks like a different NIR
power law rather than scatter.  Treat "R_V = 3.1" as approximate in the NIR.

DO NOT derive R_V from this table as k_V/(k_B - k_V).  That gives 3.47 and
it is not the law's nominal R_V: R_V is defined on a REDDENED STAR's B-V,
which depends on the star's spectrum, not on the curve alone.
"""

from __future__ import annotations

import numpy as np

from .bc_grid import DEFAULT_MODEL_ROOT

V_BAND_MICRON = 0.55

_LAW_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_law(model_root=None):
    """(wavelength_micron, k) from models/extinction_law.ascii, cached."""
    root = DEFAULT_MODEL_ROOT if model_root is None else model_root
    path = str(root / "extinction_law.ascii")
    if path not in _LAW_CACHE:
        arr = np.genfromtxt(path)
        _LAW_CACHE[path] = (arr[:, 0], arr[:, 1])
    return _LAW_CACHE[path]


def a_lambda_over_av(wavelength_micron, model_root=None):
    """A(lambda)/A_V at one or more wavelengths, from the shipped law."""
    wl, k = load_law(model_root)
    kv = np.interp(V_BAND_MICRON, wl, k)
    return np.interp(np.asarray(wavelength_micron, float), wl, k) / kv


# ---------------------------------------------------------------------------
# The R_V family the SED's fitted-R_V mode reddens with (sed.md, "The
# spectral mode"): Cardelli, Clayton & Mathis (1989), with the optical
# coefficients of O'Donnell (1994).  A_lambda / A_V = a(x) + b(x) / R_V,
# x = 1/lambda in inverse microns, so the law is LINEAR in 1/R_V and the
# SED evaluates it as two fixed arrays and one division -- differentiable in
# R_V and A_V everywhere.  It is normalized at x = 1.82 (0.5495 micron),
# where a = 1 and b = 0 exactly.
#
# It is NOT the shipped table above at R_V = 3.1: the two differ by up to
# ~0.1 mag in a band at A_V = 8 (the table runs ~6% high in J, H, Ks), so a
# fitted-R_V run at R_V = 3.1 does not reproduce a table-mode run exactly.
#
# Range.  CCM is defined on 0.3 <= x <= 10.  Redward of 3.3 micron its IR
# power law (x^1.61) is EXTENDED, not held constant: constant would
# overstate A_lambda in WISE W2-W4 and Spitzer IRAC, where the extinction is
# already < 6% of A_V (the 9.7 micron silicate feature is not modeled).
# Blueward of 0.1 micron (x > 10) x is clamped at 10; the model spectra
# carry essentially no flux there for the stars the grid covers.
# ---------------------------------------------------------------------------

CCM_X_RANGE = (0.0, 10.0)


def ccm89_ab(wavelength_micron):
    """CCM89 + O'Donnell 94 coefficients (a, b), A_lam/A_V = a + b / R_V."""
    x = np.clip(1.0 / np.asarray(wavelength_micron, float), *CCM_X_RANGE)
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61
    op = (x >= 1.1) & (x < 3.3)
    y = x[op] - 1.82
    a[op] = np.polyval(
        [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0], y
    )
    b[op] = np.polyval(
        [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0], y
    )
    uv = (x >= 3.3) & (x < 8.0)
    xu = x[uv]
    hi = xu >= 5.9
    fa = np.where(
        hi, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0
    )
    fb = np.where(hi, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.0)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb
    fu = x >= 8.0
    xf = x[fu] - 8.0
    a[fu] = -1.073 - 0.628 * xf + 0.137 * xf**2 - 0.070 * xf**3
    b[fu] = 13.670 + 4.257 * xf - 0.420 * xf**2 + 0.374 * xf**3
    return a, b


def a_lambda_over_av_ccm(wavelength_micron, rv):
    """A(lambda)/A_V of the CCM89 + O'Donnell 94 law at R_V = ``rv``."""
    a, b = ccm89_ab(wavelength_micron)
    return a + b / float(rv)
