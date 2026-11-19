"""Bolometric corrections computed at runtime from the spectrum.

The SED's fitted-R_V mode (``sed: {fit_rv_dust: true}``; sed.md, "The spectral
mode").  Where the default path interpolates tabulated BCs in (Teff, logg,
[Fe/H], A_V) at one extinction law, this interpolates the SPECTRUM in
(Teff, logg, [Fe/H]), reddens it with the star's A_V and R_V, and
integrates every band, at every evaluation:

    log10 F_b   trilinear in (teff, logg, feh) on the R = 3000 runtime grid
                (models/NextGen/runtime_spectra.py; log flux, because the
                linear flux interpolates 2-3x worse for M dwarfs)
    A_b         A_V (a_b + b_b / R_V) exp(alpha L_b): CCM89 + O'Donnell 94
                blueward of 0.909 micron and a lambda**-alpha near-IR power
                law redward, at each bin center (sed/extinction.py,
                ccm89_nir_basis; alpha is the sight line's sed.nir_alpha)
    flux_f      sum_b F_b exp(-A_b / 1.086) Sint_fb, with Sint_fb the
                integral over bin b of the filter's weighted profile
    BC_f        2.5 log10[ L0 / (4 pi (10 pc)^2 sigma Teff^4)
                           * flux_f / (ZP_f int S_f dlam) ]

-- the same formula, filter profiles, DetectorType weighting and Vega
zeropoints as the shipped tables (models/NextGen/bolometric_correction.py,
whose ``_filter_set_data`` this reads), so the two paths differ only in
how the spectrum is sampled and reddened.  Against the R = 20000 band
integrals the R = 3000 sampling is within 0.43 mmag for all 93 shipped
filters at A_V 0-20.

``law="shipped"`` reddens with the tables' own law instead (no R_V, no
alpha): that is the validation mode, in which this reproduces the tables at
their grid nodes.  No config key reaches it; the SED always uses
``"ccm89_nir"``.
"""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt
from astropy import constants as const
from astropy import units as u

from exozippy.models.NextGen.bolometric_correction import (
    L0,
    _filter_set_data,
)
from exozippy.models.NextGen.runtime_spectra import bin_integrals

from .bc_grid import RegularGridInterpolator
from .extinction import a_lambda_over_av, ccm89_nir_basis

LAWS = ("ccm89_nir", "shipped")
LN10 = float(np.log(10.0))
# 2.5 log10 of the BC's constant factor L0 / (4 pi (10 pc)^2 sigma), so
# BC = BC_CONST + 2.5 log10(flux / zero_flux) - 10 log10(Teff).
BC_CONST = 2.5 * float(
    np.log10(L0 / (100 * u.pc.to(u.m) ** 2 * 4 * np.pi * const.sigma_sb.value))
)


class SpectralBC:
    """Runtime BCs for an ordered list of filters on a ``RuntimeGrid``.

    Parameters
    ----------
    grid : models.NextGen.runtime_spectra.RuntimeGrid
    filters : sequence of str
        Filter names in any convention the alias table knows; the output
        columns are in this order.
    law : {"ccm89_nir", "shipped"}
    weighting : str
        Band-average weighting, as for the tables ("detector").
    """

    def __init__(self, grid, filters, law="ccm89_nir", weighting="detector"):
        if law not in LAWS:
            raise ValueError(f"law must be one of {LAWS}, not {law!r}.")
        self.law = law
        self.filters = list(filters)
        d = _filter_set_data(tuple(self.filters), weighting)
        self.filters_mist = list(d["filters_MIST"])
        w = np.asarray(d["filter_wavelengths"][0], float)
        S = np.asarray(d["filter_profiles"], float)
        self.zero_flux = np.asarray(d["filter_zero_flux"], float)

        sint = bin_integrals(w, S, grid.edges)  # (n_filter, n_bin)
        used = np.flatnonzero((sint > 0).any(axis=0))
        if used.size == 0:
            raise ValueError(
                f"No runtime-grid bin overlaps the filters {self.filters}."
            )
        self.bins = used
        self.M = np.ascontiguousarray(sint[:, used])  # (n_filter, n_used)
        centre_um = np.sqrt(grid.edges[used] * grid.edges[used + 1]) * 1e-4
        self.centre_um = centre_um
        if law == "ccm89_nir":
            self.ext_a, self.ext_b, self.ext_L = ccm89_nir_basis(centre_um)
        else:
            self.ext_a = np.asarray(a_lambda_over_av(centre_um), float)
            self.ext_b = np.zeros_like(self.ext_a)
            self.ext_L = np.zeros_like(self.ext_a)
        self.axes = [grid.axes["teff"], grid.axes["logg"], grid.axes["feh"]]
        # (n_teff, n_logg, n_feh, n_used): only the bins the filters touch
        self.logflux = np.ascontiguousarray(grid.logflux[..., used])
        self._interp = None
        self._np_interp = None

    # ------------------------------------------------------------ pytensor

    def bc_node(self, teff, logg, feh, av, rv, alpha=None):
        """(n_star, n_filter) BCs; teff..rv (n_star,) tensors, ``alpha`` the
        sight line's near-IR exponent, a SCALAR (one law per sight line).
        ``rv`` and ``alpha`` are ignored under ``law="shipped"``."""
        if self._interp is None:
            self._interp = RegularGridInterpolator(
                self.axes,
                pytensor.shared(self.logflux, name="sed_runtime_logflux"),
            )
        lg = self._interp.evaluate(pt.stack([teff, logg, feh], axis=-1))
        k = pt.as_tensor_variable(self.ext_a)[None, :]
        if self.law == "ccm89_nir":
            if alpha is None:
                raise ValueError("law='ccm89_nir' needs alpha.")
            alpha = pt.as_tensor_variable(alpha)
            if alpha.ndim != 0:
                raise ValueError(
                    f"SpectralBC.bc_node: alpha must be a scalar (one law "
                    f"per sight line), got ndim {alpha.ndim}."
                )
            # (a + b / R_V) exp(alpha L): L = 0 blueward of the join, so
            # one smooth product, no branch (extinction.py).
            k = (
                k + pt.as_tensor_variable(self.ext_b)[None, :] / rv[:, None]
            ) * pt.exp(alpha * pt.as_tensor_variable(self.ext_L))[None, :]
        flux_bins = pt.exp(LN10 * lg - av[:, None] * k / 1.086)
        flux = pt.dot(flux_bins, pt.as_tensor_variable(self.M.T))
        return (
            BC_CONST
            + 2.5
            * pt.log10(flux / pt.as_tensor_variable(self.zero_flux)[None, :])
            - 10.0 * pt.log10(teff)[:, None]
        )

    # --------------------------------------------------------------- numpy

    def bc_np(self, teff, logg, feh, av, rv=None, alpha=None, clip=True):
        """The same BCs for one star, as numpy (start-value solvers).
        Coordinates are clipped to the grid when ``clip`` (a start only
        needs to be close), as the table path's numpy twin does."""
        from scipy.interpolate import RegularGridInterpolator as _RGI

        if self._np_interp is None:
            self._np_interp = _RGI(self.axes, self.logflux)
        x = [float(teff), float(logg), float(feh)]
        if clip:
            x = [float(np.clip(v, a[0], a[-1])) for v, a in zip(x, self.axes)]
        lg = np.asarray(self._np_interp([x])[0], float)
        k = self.ext_a
        if self.law == "ccm89_nir":
            if rv is None or alpha is None:
                raise ValueError("law='ccm89_nir' needs rv and alpha.")
            k = (k + self.ext_b / float(rv)) * np.exp(
                float(alpha) * self.ext_L
            )
        flux = self.M @ np.exp(LN10 * lg - float(av) * k / 1.086)
        return (
            BC_CONST
            + 2.5 * np.log10(flux / self.zero_flux)
            - 10.0 * np.log10(x[0])
        )
