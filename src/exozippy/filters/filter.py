import logging
import os
import pathlib

# pickling and querying
import pickle
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.io.votable import parse
from astropy.units import Quantity
from astroquery.query import BaseQuery
from scipy import interpolate

logger = logging.getLogger(__name__)

# Where the profiles that SHIP with EXOZIPPy live: this package directory.
# Read-only by policy -- see _writable_filter_root.
DEFAULT_FILTER_DIR = pathlib.Path(__file__).parent

# Subdirectory of the machine-level cache that downloaded filter profiles
# live in, alongside utilities/zenodo.py's "downloads".
_CACHE_SUBDIR = "filters"


def filter_cache_root() -> Path | None:
    """Where downloaded filter profiles are cached, or None if unavailable.

    The same machine-level root the large Zenodo assets use
    (``$XDG_CACHE_HOME/exozippy``, relocatable or switchable off with
    ``EXOZIPPY_CACHE_DIR``), in a ``filters`` subdirectory of its own.
    Purely a path computation: nothing is created here.
    """
    from ..utilities.zenodo import shared_cache_root

    root = shared_cache_root()
    return None if root is None else root / _CACHE_SUBDIR


def _writable_filter_root() -> Path:
    """The directory a freshly downloaded profile should be written to.

    The cache first. With the cache switched off or unusable, the shipped
    package directory -- which is the pre-2026-08 behaviour, and the right
    answer for a dev checkout, where that directory is writable and is where
    the shipped profiles came from. Only if THAT is read-only too (a
    site-packages install with no cache) does it fall back to a temporary
    directory, so the fit still runs; it just re-downloads next time.
    """
    cache = filter_cache_root()
    if cache is not None:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            if os.access(cache, os.W_OK | os.X_OK):
                return cache
        except OSError as e:
            logger.warning("Filter cache %s is unusable (%s).", cache, e)

    if os.access(DEFAULT_FILTER_DIR, os.W_OK | os.X_OK):
        return DEFAULT_FILTER_DIR

    fallback = Path(tempfile.gettempdir()) / "exozippy-filters"
    logger.warning(
        "Neither the filter cache nor %s is writable; caching downloaded "
        "filter profiles in %s for this run only. Set EXOZIPPY_CACHE_DIR to "
        "a writable path to keep them.",
        DEFAULT_FILTER_DIR,
        fallback,
    )
    return fallback


def construct_wave_grid(
    R: int | float,
    wave_min: float,
    wave_max: float,
    # Quantity["length"] is astropy's physical-type annotation and is valid at
    # runtime (it evaluates to Annotated[Quantity, PhysicalType('length')]).
    # Linters read the string subscript as a forward reference and report
    # `length` as an undefined name; it is not. Do not "fix" this by removing
    # the annotation.
    input_unit: Quantity["length"] = u.micron,  # noqa: F821
    output_unit: Quantity["length"] = u.Angstrom,  # noqa: F821
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs grid of wavelengths for a specified wavelength range at a specific spectral resolution

    Parameters
    ----------
    R : int | float
        spectral resolution of wavelength grid
    wave_min : float
        minimum wavelength
    wave_max : float
        maximum wavelength
    input_unit : astropy.units.Quantity["length"], optional
        wavelength unit that matches the inputs ``wave_min`` and ``wave_max``. By default, set to ``u.micron``
    output_unit : astropy.units.Quantity["length"], optional
        wavelength unit that matches the second output array ``wave_array_output``. By default, set to ``u.Angstrom``

    Returns
    -------
        wave_array         : np.ndarray, shape
            wavelength array in units of ``input_unit``
        wave_array_output  : np.ndarray, shape
            wavelength array in units of ``output_unit``
        dwave_array        : np.ndarray, shape
            wavelength spacing (dlambda) array in units of ``input_unit``
    """

    init_dlambda = wave_min / R  # initial delta lambda
    wave_range = wave_max - wave_min  # full wavelength range
    max_array_size = int(
        np.ceil(wave_range / init_dlambda)
    )  # max array length

    # initialize arrays
    wave_array = np.zeros(max_array_size)
    dwave_array = np.zeros(max_array_size)

    i = 0
    wave_array[i] = wave_min
    dwave_array[i] = wave_min / R
    next_wave = wave_min

    while next_wave < wave_max:
        # iterate forward wavelength
        next_wave += dwave_array[i]

        wave_array[i + 1] = next_wave
        dwave_array[i + 1] = next_wave / R

        # iterate forward index
        i += 1

    # unit conversion
    CONVERT = input_unit.to(output_unit)
    wave_array_output = CONVERT * wave_array

    return (
        np.trim_zeros(wave_array, "b"),
        np.trim_zeros(wave_array_output, "b"),
        np.trim_zeros(dwave_array, "b"),
    )


class Filter(BaseQuery):
    SVO_BASE_URL = "https://svo2.cab.inta-csic.es/theory/fps/"
    # Alias of the module-level constant, kept for callers that reach for it
    # on the class. It is the SHIPPED (read-only) root; a download goes to
    # _writable_filter_root() instead.
    DEFAULT_FILTER_DIR = globals()["DEFAULT_FILTER_DIR"]

    VOTABLE_FIELD_NAMES = [
        "FilterProfileService",
        "Facility",
        "Instrument",
        "ProfileReference",
        "CalibrationReference",
        "PhotSystem",
        "MagSys",
        "WavelengthEff",
        "WavelengthRef",
        "WavelengthMin",
        "WavelengthMax",
        "FWHM",
        "WidthEff",
        "Fsun",
    ]

    # define common wavelength grid for all spectra
    # everything will use the same grid of wavelengths
    RESOLUTION = 20000  # Wavelength resolution
    _LAMBDA_MIN = 0.03  # microns
    _LAMBDA_MAX = 30  # microns
    (
        _LAMBDA_GRID_COMMON_MICRONS,
        WAVELENGTH_PTS,
        _DLAMBDA_GRID_COMMON_MICRONS,
    ) = construct_wave_grid(RESOLUTION, _LAMBDA_MIN, _LAMBDA_MAX)

    def __init__(self, filterID, **kwargs):
        """
        Parameters
        ----------
        filterID : string
            Used to create a HTTP query string i.e. send to SVO FPS to get data.
            String will take form of 'FACILTIY/INSTRUMENT.FILTER
            All filter options are available at: https://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse
            Examples: '2MASS/2MASS.Ks' or 'Keck/NIRC2.Brgamma'
        """

        self.filterID = filterID
        self.facility = filterID.split("/")[0]
        self.filterName = filterID.split("/")[-1]
        self.filterDirectory = None
        self._session = requests.Session()
        self._check_if_filter_saved(**kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        for field in state:
            setattr(self, field, state[field])

    def __str__(self):
        # f-string, not concatenation: filterDirectory is a pathlib.Path, so
        # `str + Path` raised TypeError -- the one debugging affordance this
        # class offers crashed whenever it was used (review 1.9.6).
        return (
            f"{self.filterName} filter data available in "
            f"{self.filterDirectory}"
        )

    def _check_if_filter_saved(self, filterDir=None, overwrite=False):
        """Load this filter from the first place it is found, else fetch it.

        Read order is the shipped package directory first (20 profiles ship
        with EXOZIPPy) and the machine-level cache second; a fetch is written
        to the cache. Writing into the package directory -- what this did
        until 2026-08 -- is a PermissionError on a read-only site-packages
        install and source-tree litter in a dev checkout (review 2.9.3), and
        the unconditional makedirs meant even a purely READ path created an
        empty facility directory there.

        ``filterDir`` overrides both roots with one directory, for a caller
        that wants a private filter store.
        """
        filename_filter = self.filterName + ".filter"

        if filterDir is not None:
            read_roots = [Path(filterDir)]
        else:
            read_roots = [
                root
                for root in (DEFAULT_FILTER_DIR, filter_cache_root())
                if root is not None
            ]

        if not overwrite:
            for root in read_roots:
                if (root / self.facility / filename_filter).exists():
                    self.filterDirectory = root / self.facility
                    self._read_filter_file()
                    return

        # Only now, on the path that really has something to write. Choosing
        # the write root eagerly would create the cache directory even for a
        # filter that ships with the package.
        write_root = (
            Path(filterDir)
            if filterDir is not None
            else _writable_filter_root()
        )
        self.filterDirectory = write_root / self.facility
        os.makedirs(self.filterDirectory, exist_ok=True)
        self._download_filter()
        self._set_attrs()
        self._create_filter_file()

    def _download_filter(self):
        """Get and save all filter data in response a query sent to SVO FPS.

        Parameters
        ----------
        filterDir : String
            Directory where filter VOTables will be saved/checked for

        Returns
        -------
        Dictionary of filter properties, including full filter transmission profile
        """

        filename_VOTable = self.filterName + ".xml"

        # set URL
        url = self.SVO_BASE_URL + "fps.php?ID=" + self.filterID
        # response = self._request('GET', url, save=True, savedir=filterDir)
        response = self._download_file(
            url,
            local_filepath=self.filterDirectory / filename_VOTable,
            verbose=False,
        )

        return

    def _set_zeropoint_values(self):

        ### access zeropoint information ###
        CALIBRATION_TABLE_DROP_ROWS = [0, 3, 4, 5]
        CALIBRATION_TABLE_DROP_COLUMNS = [0]
        CALIBRATION_TABLE_INDEX = {1: "ZeroPoint_Fl", 2: "ZeroPoint_Fv"}
        CALIBRATION_TABLE_HEADER = {1: "Specified", 2: "Calculated", 3: "Unit"}
        MISSING_VALUE = "--"

        url = self.SVO_BASE_URL + "index.php?id=" + self.filterID

        # read tables from html url provided
        extracted_tables = pd.read_html(url)
        df_calibration_vega = extracted_tables[-3]
        df_calibration_ab = extracted_tables[-2]
        df_calibration_st = extracted_tables[-1]

        # make the tables nice and easily parsable
        df_vega = (
            df_calibration_vega.drop(CALIBRATION_TABLE_DROP_ROWS)
            .drop(columns=CALIBRATION_TABLE_DROP_COLUMNS)
            .rename(index=CALIBRATION_TABLE_INDEX)
            .rename(columns=CALIBRATION_TABLE_HEADER)
        )
        df_ab = (
            df_calibration_ab.drop(CALIBRATION_TABLE_DROP_ROWS)
            .drop(columns=CALIBRATION_TABLE_DROP_COLUMNS)
            .rename(index=CALIBRATION_TABLE_INDEX)
            .rename(columns=CALIBRATION_TABLE_HEADER)
        )
        df_st = (
            df_calibration_st.drop(CALIBRATION_TABLE_DROP_ROWS)
            .drop(columns=CALIBRATION_TABLE_DROP_COLUMNS)
            .rename(index=CALIBRATION_TABLE_INDEX)
            .rename(columns=CALIBRATION_TABLE_HEADER)
        )

        # initialize dictionary and lists
        df_dict = {
            "Vega": {"df": df_vega, "suffix": "_Vega"},
            "AB": {"df": df_ab, "suffix": "_AB"},
            "ST": {"df": df_st, "suffix": "_ST"},
        }
        rows = ["ZeroPoint_Fl", "ZeroPoint_Fv"]
        columns = ["Specified", "Calculated"]
        attr_names = ["Zp_Spec_Fl", "Zp_Calc_Fl", "Zp_Spec_Fv", "Zp_Calc_Fv"]

        # loops through lists and set zeropoint attributes
        for df_sys in df_dict:
            count = 0
            for row in rows:
                for column in columns:
                    field = attr_names[count] + df_dict[df_sys]["suffix"]
                    if (
                        MISSING_VALUE
                        not in df_dict[df_sys]["df"].loc[row][column]
                    ):
                        value = float(df_dict[df_sys]["df"].loc[row][column])
                        setattr(self, field, value)
                    else:
                        setattr(self, field, None)
                    count += 1

        return

    def _process_raw_filter(self, raw):

        # pad the filter profile to force it to go zero on either side
        # first calculate lower and upper wavelength spacing for the padding
        wave_filter_profile = raw[0]
        wave_dlambda = np.diff(wave_filter_profile)
        wave_lower_dlambda = wave_dlambda[0]
        wave_upper_dlambda = wave_dlambda[-1]

        N_pad = 10
        wave_pad_lower = np.linspace(
            min(wave_filter_profile) - N_pad * wave_lower_dlambda,
            min(wave_filter_profile),
            N_pad,
            endpoint=False,
        )
        wave_pad_upper = np.linspace(
            max(wave_filter_profile) + wave_upper_dlambda,
            max(wave_filter_profile) + (N_pad + 1) * wave_upper_dlambda,
            N_pad,
        )
        wave_filter_profile_padded = np.concatenate(
            (wave_pad_lower, wave_filter_profile, wave_pad_upper), axis=None
        )

        # pad filter transmission curve with zeros
        trans_filter_profile = raw[1]
        trans_filter_profile_padded = np.pad(
            trans_filter_profile, (N_pad, N_pad), "constant"
        )

        filter_func = interpolate.interp1d(
            wave_filter_profile_padded,
            trans_filter_profile_padded,
            fill_value="extrapolate",
        )  # in angstroms
        filter_interpolated = filter_func(self.WAVELENGTH_PTS)
        filter_interpolated[filter_interpolated < 0] = 0

        return (
            np.array(
                [wave_filter_profile_padded, trans_filter_profile_padded]
            ),
            np.array([self.WAVELENGTH_PTS, filter_interpolated]),
        )

    def _set_attrs(self):

        filename_VOTable = self.filterName + ".xml"
        votable = parse(self.filterDirectory / filename_VOTable)

        for field in self.VOTABLE_FIELD_NAMES:
            try:
                value = votable.get_field_by_id_or_name(field).value
                setattr(self, field, value)
            except:
                setattr(self, field, None)

        self._set_zeropoint_values()

        self.TransmissionUnit = votable.get_field_by_id_or_name(
            "Transmission"
        ).unit
        self.RawFilterCurve = np.array(
            [list(i) for i in votable.get_first_table().array.data]
        ).T
        self.PaddedFilterCurve, self.ProcessedFilterCurve = (
            self._process_raw_filter(self.RawFilterCurve)
        )

        return

    def _create_filter_file(self):

        filename_filter = self.filterName + ".filter"
        state = self.__getstate__()
        with open(self.filterDirectory / filename_filter, "wb") as file:
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def _read_filter_file(self):

        filename_filter = self.filterName + ".filter"
        directory = self.filterDirectory
        with open(directory / filename_filter, "rb") as file:
            state = pickle.load(file)

        self.__setstate__(state)
        # The pickle carries the filterDirectory of the machine that WROTE
        # it -- the shipped profiles name a path on a developer's laptop,
        # under a package layout that no longer exists. Where the file
        # actually is is what this read just proved, so restore it; otherwise
        # __str__ (and any later write) points somewhere that does not exist.
        self.filterDirectory = directory

        return
