"""The extinction laws: the shipped table, CCM89 + O'Donnell 94, and the
SED's fitted-R_V law built on it (CCM89 + O'Donnell 94 joined to a near-IR
power law of free exponent, review 8.9.2; at the end of this module).

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
# The R_V family the SED's fitted-R_V mode builds its law on (sed.md, "The
# spectral mode"; the near-IR power law below replaces its infrared
# segment): Cardelli, Clayton & Mathis (1989), with the optical
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


def _odonnell_optical_ab(x):
    """O'Donnell (1994)'s optical polynomial (a, b) at x = 1/lambda,
    inverse micron (CCM's 1.1 <= x < 3.3 segment)."""
    y = np.asarray(x, float) - 1.82
    a = np.polyval(
        [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1.0], y
    )
    b = np.polyval(
        [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0.0], y
    )
    return a, b


def ccm89_ab(wavelength_micron):
    """CCM89 + O'Donnell 94 coefficients (a, b), A_lam/A_V = a + b / R_V."""
    x = np.clip(1.0 / np.asarray(wavelength_micron, float), *CCM_X_RANGE)
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61
    op = (x >= 1.1) & (x < 3.3)
    a[op], b[op] = _odonnell_optical_ab(x[op])
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


# ---------------------------------------------------------------------------
# THE SED's LAW: CCM89 + O'Donnell 94 blueward of the join, a free near-IR
# power law redward of it (review 8.9.2, ruled 2026-11-09; sed.md, "The
# near-infrared slope").
#
#   x >= 1.1 / micron   A_lam/A_V = a(x) + b(x) / R_V                 (CCM/O'D)
#   x <  1.1 / micron   A_lam/A_V = [a(1.1) + b(1.1) / R_V] (x/1.1)**alpha
#
# The join is CCM's OWN segment boundary, x = 1.1 / micron (0.909 micron),
# and the law is CONTINUOUS there by construction (in value; not in slope,
# like any piecewise law): the power law starts from the optical
# polynomial's value AT the join, a(1.1) and b(1.1) evaluated on the
# optical side.  CCM89 itself is not continuous there -- its infrared
# segment (a = 0.574 x**1.61, b = -0.527 x**1.61) and its optical
# polynomial disagree at x = 1.1 by 0.05% of A_lam/A_V at R_V = 3.1 and
# 0.34% at R_V = 1.5 -- so alpha = CCM_NIR_ALPHA = 1.61 reproduces CCM's
# infrared segment times that one constant, (a(1.1) + b(1.1)/R_V) /
# (0.574 + -0.527/R_V) / 1.1**1.61, and the optical exactly.  The ruling
# asked for both continuity and an exact alpha = 1.61 identity; they cannot
# both hold, and continuity was kept (the optical side is the unchanged,
# fitted part; the near-IR is the free part that starts from it).
# alpha is the sight line's sed.nir_alpha; R_V stays the optical knob.  The
# two separate cleanly: alpha leaves A_I/E(V-I) unchanged and carries
# E(I-J)/E(V-I) and A_V/A_Ks.  Why a second parameter at all: toward the
# bulge one CCM R_V cannot match the optical and near-IR red-clump ratios at
# once (Nataf et al. 2016 find >= 2 degrees of freedom), and the measured
# bulge near-IR slopes are steeper than CCM's 1.61 (Nishiyama et al. 2009,
# ~2.0).  Nataf et al. (2013, sec. 10) tested exactly this family
# (O'Donnell with a modified near-IR exponent).
#
# THE FORM THE GRAPH NEEDS.  The bin centers are fixed, so
#   k(R_V, alpha) = (a + b / R_V) * exp(alpha * L)
# with, per bin, (a, b) = CCM/O'D's own coefficients and L = 0 blueward of
# the join, and (a, b) = the optical polynomial's (a(1.1), b(1.1)) and
# L = ln(x / 1.1) < 0 redward.
# No mask, no branch, no pt.where: one product, smooth in R_V and alpha
# everywhere, and still LINEAR in 1/R_V at fixed alpha -- so the dust
# profile's near/far R_V mix (1/R_eff = f/R_near + (1-f)/R_far) remains an
# exact sum of two layers that share the sight line's alpha.
#
# Redward of 3.3 micron (x < 0.3) the power law is extended as CCM's was.
# Every Galactic-center study finds the law GRAYER past ~3 micron, so WISE
# W2-W4 and Spitzer IRAC are underextincted here; the SED says so when such
# a filter is fitted in this mode (sed.py, SED._warn_mid_ir_law).
#
# SEAM: NATAF'S ROMAN-BAND BULGE CURVE.  Nataf's group has announced a mean
# bulge extinction curve in the Roman filters (a table, "late 2026").  When
# it is published it REPLACES this family's role as the bulge law (or, if
# the family reproduces it to the table's stated precision, re-centers the
# sed.nir_alpha prior on it).  The swap point is this function: SpectralBC
# consumes only the per-bin (a, b, L) it returns, so a tabulated curve
# enters as another basis builder of the same contract (and its own law
# name in spectral_bc.LAWS), with no change to the graph.  Reconcile its
# convention first: a table in filter bands is effective-wavelength or
# spectrum-specific, while this model integrates the reddened spectrum
# through each passband (conventions.md C29).
# ---------------------------------------------------------------------------

#: CCM's own optical/infrared segment boundary, in inverse microns: the join.
NIR_JOIN_X = 1.1
#: The join as a wavelength, micron (1 / 1.1 = 0.909).
NIR_JOIN_MICRON = 1.0 / NIR_JOIN_X
#: CCM89's infrared exponent (Rieke & Lebofsky 1985): the identity alpha.
CCM_NIR_ALPHA = 1.61


def ccm89_nir_basis(wavelength_micron):
    """Per-wavelength (a, b, L) of the CCM89 + near-IR power-law family:
    A_lam/A_V = (a + b / R_V) exp(alpha L).  Blueward of the join (a, b)
    are CCM/O'Donnell's and L = 0; redward they are the join's and
    L = ln(x / 1.1)."""
    wl = np.asarray(wavelength_micron, float)
    if np.any(~np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError(
            "ccm89_nir_basis: wavelengths must be finite and positive "
            f"(micron), got min {np.min(wl)!r}."
        )
    a, b = ccm89_ab(wl)
    x = 1.0 / wl
    ir = x < NIR_JOIN_X
    # The optical side's value AT the join (not ccm89_ab(1/1.1), which
    # round-off can land on the infrared segment).
    a_j, b_j = _odonnell_optical_ab(NIR_JOIN_X)
    a = np.where(ir, a_j, a)
    b = np.where(ir, b_j, b)
    L = np.where(ir, np.log(np.where(ir, x, NIR_JOIN_X) / NIR_JOIN_X), 0.0)
    return a, b, L


def a_lambda_over_av_nir(wavelength_micron, rv, alpha):
    """A(lambda)/A_V of the SED's law (CCM89 + O'Donnell 94 blueward of
    0.909 micron, a lambda**-alpha power law redward) at R_V = ``rv``."""
    a, b, L = ccm89_nir_basis(wavelength_micron)
    return (a + b / float(rv)) * np.exp(float(alpha) * L)
