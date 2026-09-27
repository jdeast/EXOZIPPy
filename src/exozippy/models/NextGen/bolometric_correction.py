"""
Bolometric correction of one model spectrum through a set of filters.

``BolometricCorrection`` is carried over from ``Code for Models/
creating_BC_Table.ipynb`` with the physics unchanged:

    BC_X = 2.5 log10[ L0 / (4 pi (10 pc)^2 sigma Teff^4)
                      * int(F_lam e^-tau S dlam) / (ZP_X int(S dlam)) ]

i.e. M_bol from sigma*Teff^4 and the IAU 2015 zero point L0 = 3.0128e28 W,
a band-averaged flux density, and the filter's SVO Vega F_lambda
zeropoint (specified value when SVO quotes one, else calculated).
Extinction: tau(lam) = ext(lam)/ext(0.55 um) * Av / 1.086, with ext from
models/extinction_law.ascii (R_V = 3.1).

Two changes from the notebook DO change BC values; both were measured
against the tables that shipped before this pipeline:

* The extinction law is tabulated in MICRONS. The notebook evaluated it
  at the filter wavelengths in Angstrom (i.e. far off the end of the
  table, by extrapolation), which put 2MASS_J wrong by 1.8 mag at Av = 6.
  The wavelengths are now converted, as the notebook's own comment says.
* The band average is weighted per filter by its SVO ``DetectorType``
  (``weighting="detector"``, the default): energy-weighted, int(F S dlam),
  for an energy counter (DetectorType 0) and photon-weighted,
  int(F lam S dlam), for a photon counter (DetectorType 1). That is the
  convention SVO computes the Vega zeropoints with, so the ratio to the
  zeropoint is only consistent this way. The notebook energy-weighted
  every filter (``weighting="energy"`` reproduces it exactly), which is
  right for Gaia, TESS, WISE, Bessell and Cousins and off by ~0.01 mag for
  2MASS and NIRC2. The tables that shipped before this pipeline were
  photon-weighted for every filter (``weighting="photon"``), which against
  the MIST BC tables at [Fe/H] = 0, Av = 0 puts WISE_W3 off by 0.18 mag,
  Gaia_G by 0.07 - 0.16 mag and TESS by 0.08 - 0.10 mag; the energy
  weighting of those filters agrees with MIST to 0.01 mag (solar-type)
  to 0.04 mag (3500 K).

Changes from the notebook that do not change a BC value:

* filter names resolve through the package alias table
  (bc_grid.resolve_filter_name) -- the same lookup the notebook did by
  hand, plus the ambiguous-label map and the synthesized MIST name for
  filters with no alias row, so a column written here is found by name
  by bc_grid.build_bc_grid.
* ``av`` may be a 1-D array: the spectrum is read once and the BC is
  computed for every Av at once (``BC_by_av``, shape (n_av, n_filters)).
* the spectrum can be passed in (the pipeline reads it from its parquet
  cache) instead of being read from the raw file.
* the per-filter-set data (profiles, zeropoints, extinction curve) is
  built once per filter set and shared across instances, and every array
  is trimmed to the wavelength support of the filter set (plus one zero
  point either side). Outside that support every integrand is exactly
  zero, so the trapezoid integrals are unchanged up to floating-point
  summation order.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from scipy import interpolate

from exozippy.components.sed.bc_grid import (
    DEFAULT_MODEL_ROOT,
    _load_alias_table,
    resolve_filter_name,
)
from exozippy.filters.filter import Filter
from exozippy.models.NextGen.nextgen_spectra import (
    SPECTRA_RAW_PATH_DEFAULT,
    load_spectrum,
)

L0 = 3.0128e28  # Watts

V_BAND_MICRON = 0.55


def trapezoid(x, y):
    dx = x[..., 1:] - x[..., :-1]
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)


def _read_extinction_law() -> pd.DataFrame:
    # read in extinction values
    return pd.read_csv(
        DEFAULT_MODEL_ROOT / "extinction_law.ascii",
        names=["wavelength", "extinction"],
        delimiter=" ",
        index_col=False,
        skipinitialspace=True,
    )


FLUX_WEIGHTINGS = ("detector", "photon", "energy")


def _detector_type(filt: Filter) -> int:
    """SVO DetectorType: 0 = energy counter, 1 = photon counter.

    Filter does not keep this VOTable field, so it is read from the
    VOTable (.xml) saved next to the filter's .filter file.
    """
    from astropy.io.votable import parse

    value = getattr(filt, "DetectorType", None)
    if value is None:
        xml = Path(filt.filterDirectory) / f"{filt.filterName}.xml"
        try:
            value = parse(xml).get_field_by_id_or_name("DetectorType").value
        except Exception as e:
            raise ValueError(
                f"Cannot read the SVO DetectorType of {filt.filterID} from "
                f"{xml} ({e}); pass weighting='photon' or 'energy' explicitly."
            ) from e
    value = int(value)
    if value not in (0, 1):
        raise ValueError(
            f"Unknown SVO DetectorType {value} for {filt.filterID}."
        )
    return value


@lru_cache(maxsize=32)
def _filter_set_data(filters: tuple[str, ...], weighting: str) -> dict:
    """What BolometricCorrection needs that depends only on the filters."""
    if weighting not in FLUX_WEIGHTINGS:
        raise ValueError(
            f"weighting must be one of {FLUX_WEIGHTINGS}, not {weighting!r}."
        )
    alias_df = _load_alias_table()

    # get all relavant filter information and save it for easy access later
    # first retrieve all SVO/MIST filter names
    filters_SVO = []
    filters_MIST = []
    for name in filters:
        SVO_name = resolve_filter_name(name, alias_df, alias="SVO")
        MIST_name = resolve_filter_name(name, alias_df, alias="MIST")
        filters_SVO.append(SVO_name)
        filters_MIST.append(MIST_name)

    # create Filter objects
    filter_objs = {}
    for filter in filters_SVO:
        filter_objs[filter] = Filter(filter)
    if len(filter_objs) != len(filters):
        raise ValueError(
            f"Filter list {list(filters)} names the same SVO filter twice."
        )

    # create arrays
    filter_wavelengths = np.array(
        [filter_objs[filter].ProcessedFilterCurve[0] for filter in filter_objs]
    )
    filter_profiles = np.array(
        [filter_objs[filter].ProcessedFilterCurve[1] for filter in filter_objs]
    )
    filter_zero_pts = np.array(
        [
            getattr(filter_objs[filter], "Zp_Spec_Fl_Vega", None)
            if getattr(filter_objs[filter], "Zp_Spec_Fl_Vega", None)
            else getattr(filter_objs[filter], "Zp_Calc_Fl_Vega", None)
            for filter in filter_objs
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(filter_zero_pts)):
        bad = [f for f, zp in zip(filters_SVO, filter_zero_pts) if not zp > 0]
        raise ValueError(f"No Vega F_lambda zeropoint available for {bad}.")

    # Trim to the wavelength support of the filter set, keeping one zero
    # point either side of every non-zero stretch so each trapezoid
    # segment with a non-zero endpoint survives (see module docstring).
    nonzero = np.any(filter_profiles > 0, axis=0)
    wave_mask = nonzero.copy()
    wave_mask[:-1] |= nonzero[1:]
    wave_mask[1:] |= nonzero[:-1]
    filter_wavelengths = filter_wavelengths[:, wave_mask]
    filter_profiles = filter_profiles[:, wave_mask]

    # Photon weighting folds lambda into the profile, so it enters BOTH the
    # zeropoint normalization and the flux integral below.
    if weighting == "detector":
        filter_weightings = [
            "energy" if _detector_type(filter_objs[filter]) == 0 else "photon"
            for filter in filter_objs
        ]
    else:
        filter_weightings = [weighting] * len(filter_objs)
    photon = np.array([w == "photon" for w in filter_weightings])
    filter_profiles = np.where(
        photon[:, None], filter_profiles * filter_wavelengths, filter_profiles
    )

    # calculate the normalization constant for each filter
    filter_zero_flux = filter_zero_pts * trapezoid(
        filter_wavelengths, filter_profiles
    )  # normalize the filter transmission curve

    # interpolate extinction function onto same wavelength scale, but in microns
    extinction_df = _read_extinction_law()
    extinction_func = interpolate.interp1d(
        extinction_df["wavelength"],
        extinction_df["extinction"],
        fill_value="extrapolate",
    )
    V_band_extinction = extinction_func(V_BAND_MICRON)
    # the extinction law is tabulated in microns; the filter grid is Angstrom
    extinction_modeled = extinction_func(
        filter_wavelengths[0] * u.Angstrom.to(u.micron)
    )

    return {
        "filters_SVO": filters_SVO,
        "filters_MIST": filters_MIST,
        "filter_objs": filter_objs,
        "filter_wavelengths": filter_wavelengths,
        "filter_profiles": filter_profiles,
        "filter_zero_pts": filter_zero_pts,
        "filter_weightings": filter_weightings,
        "filter_zero_flux": filter_zero_flux,
        "wave_mask": wave_mask,
        "V_band_extinction": V_band_extinction,
        "extinction_modeled": extinction_modeled,
    }


class BolometricCorrection:
    """
    Bolometric corrections for one (teff, logg, feh) spectrum through
    ``filters`` at one or more Av values.

    Parameters
    ----------
    filters : sequence of str
        Filter names in any convention the alias table knows (SVO id,
        MIST column name, VOID label, ...).
    star_dict : dict
        'teff', 'logg', 'feh' (a grid node of the spectra) and 'av'
        (a scalar or a 1-D array).
    spectrum : (alpha, flux) tuple, optional
        F_lambda on nextgen_spectra.WAVELENGTH_PTS and the [alpha/Fe] it
        was computed at. Read from the raw spectrum in ``spectra_dir``
        when not given.
    spectra_dir : Path, optional
        Directory of raw BT-NextGen spectra.
    weighting : {"detector", "photon", "energy"}
        Band-average weighting (see the module docstring); "detector"
        picks energy or photon per filter from its SVO DetectorType.

    Attributes
    ----------
    BC : np.ndarray
        np.squeeze of the (n_av, n_filters) result, as in the notebook.
    BC_by_av : np.ndarray, shape (n_av, n_filters) (or (n_filters,) for
        a scalar av)
    alpha : float
        [alpha/Fe] of the spectrum actually used.
    filters_SVO, filters_MIST : list of str
    filter_weightings : list of str
        "energy" or "photon", per filter, as actually applied.
    """

    def __init__(
        self,
        filters: Sequence[str],
        star_dict: dict,
        spectrum: tuple[float, np.ndarray] | None = None,
        spectra_dir: Path | str = SPECTRA_RAW_PATH_DEFAULT,
        weighting: str = "detector",
    ):

        # filter information
        self.filters = list(filters)
        self.weighting = weighting
        self.nfilters = len(self.filters)
        self._load_filter_data()

        # star information
        self.star_dict = star_dict
        self.teff = self.star_dict.get("teff")
        self.logg = self.star_dict.get("logg")
        self.feh = self.star_dict.get("feh")
        self.av = np.asarray(self.star_dict.get("av"), dtype=float)

        # load spectrum for star
        self._load_spectra_data(spectrum, spectra_dir)

        ##### calculate bolometric correction #####
        # first calculate optical depth at each wavelength, normalized to V-band
        self._normalize_optical_depth()
        self._flux_within_filter()
        self._calc_BC()

    def _load_filter_data(self):
        for key, value in _filter_set_data(
            tuple(self.filters), self.weighting
        ).items():
            setattr(self, key, value)

    def _load_spectra_data(self, spectrum, spectra_dir):
        if spectrum is None:
            spectrum = load_spectrum(
                self.teff, self.logg, self.feh, spectra_dir
            )
        self.alpha, spectrum_full = spectrum
        self.spectrum = np.asarray(spectrum_full)[self.wave_mask]

    ######## functions used in calculating flux within a filter ########

    def _normalize_optical_depth(self):
        # calculate the optical depth at each wavelength normalized to V-band extinction
        # av gets two trailing axes so an (n_av,) array broadcasts to
        # (n_av, 1, n_wave), i.e. against the (n_filters, n_wave) profiles
        av = np.reshape(self.av, self.av.shape + (1, 1))
        optical_depth = (
            (self.extinction_modeled / self.V_band_extinction) * av / 1.086
        )
        self.optical_depth = optical_depth

    def _flux_within_filter(self):
        x = self.filter_wavelengths
        y = self.spectrum * np.exp(-self.optical_depth) * self.filter_profiles
        flux_in_filter = trapezoid(x, y)
        self.flux_in_filter = flux_in_filter  # units of erg/s/cm^2

    def _calc_BC(self):
        term1 = L0 / (
            100
            * u.pc.to(u.m) ** 2
            * 4
            * np.pi
            * const.sigma_sb.value
            * self.teff**4
        )
        term2 = self.flux_in_filter / self.filter_zero_flux
        BC = 2.5 * np.log10(term1 * term2)
        self.BC_by_av = np.reshape(BC, self.av.shape + (self.nfilters,))
        self.BC = np.squeeze(BC)
