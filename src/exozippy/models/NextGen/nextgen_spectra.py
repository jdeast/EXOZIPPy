"""
BT-NextGen (AGSS2009) spectra: filenames, grid points and the raw-spectrum
reader used by the bolometric-correction pipeline
(generate_NextGen_BC_Tables.py).

Carried over from ``Code for Models/Classes/Spectra.py`` (``NextGenSpectra``)
with two deliberate changes:

* the common wavelength grid is ``exozippy.filters.filter.Filter``'s
  (R = 20000, 0.03 - 30 micron), so a spectrum and a filter's
  ``ProcessedFilterCurve`` are sampled on the SAME points and can be
  multiplied element-wise -- which is what ``BolometricCorrection`` does.
  (The original built the identical grid through ``coronagraph``, which
  exozippy does not depend on.)
* no per-spectrum pickle cache: the old ``.spectra`` pickles hold
  pytensor objects that no longer unpickle under current pytensor. The
  pipeline caches resampled spectra as parquet instead (step 1 of
  generate_NextGen_BC_Tables.py).

Raw spectra are the SVO ``.BT-NextGen.7.dat.txt`` ASCII files (wavelength
in Angstrom, F_lambda in erg/s/cm^2/A), from
https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-nextgen-agss2009
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

from exozippy.filters.filter import Filter

# replace with your path to the directory containing the raw BT-NextGen spectra
SPECTRA_RAW_PATH_DEFAULT = Path("/Volumes/Data/Spectra/BT-NextGen_AGSS2009/")

# Alpha-abundance fallback order when a grid point has no alpha=0
# spectrum. Order matters: the first alpha with a spectrum on disk wins.
ALPHA_GRID_PTS = np.array([0, 0.2, -0.2, 0.4, 0.6])

# Define all available grid points for BT-NextGen AGSS 2009 Spectra
# only points part of a fully filled portion of this grid are included here
# (the shipped models/NextGen/BCs/NextGen.grid.yaml carries the same axes)
LOGG_GRID_PTS = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
)

FEH_GRID_PTS = np.array(
    [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.5]
)

TEFF_GRID_PTS = np.array(
    [2600., 2700., 2800., 2900., 3000., 3100., 3200., 3300.,
     3400., 3500., 3600., 3700., 3800., 3900., 4000., 4100.,
     4200., 4300., 4400., 4500., 4600., 4700., 4800., 4900.,
     5000., 5100., 5200., 5300., 5400., 5500., 5600., 5700.,
     5800., 5900., 6000., 6100., 6200., 6300., 6400., 6500.,
     6600., 6700., 6800., 6900., 7000., 7200., 7400., 7600.,
     7800., 8000., 8200., 8400., 8600., 8800., 9000., 9200.,
     9400., 9600., 9800., 10000.]
)  # fmt: skip

# common wavelength grid (Angstrom) shared with every Filter's
# ProcessedFilterCurve
WAVELENGTH_PTS = Filter.WAVELENGTH_PTS


def get_NextGen2009_filename(model_dict: dict) -> str:
    """
    Raw spectrum filename for one grid point.

    model_dict takes four values: 'teff', 'logg', 'feh' (metallicity) and
    'alpha' (alpha abundance), e.g.
        {'teff': 5800, 'logg': 4.5, 'feh': 0.0, 'alpha': 0.0}
        -> 'lte058-4.5-0.0a+0.0.BT-NextGen.7.dat.txt'
    """
    teff = int(model_dict["teff"])
    logg = model_dict["logg"]
    feh = model_dict["feh"]
    alpha = model_dict["alpha"]

    filename = "lte"

    # teff part of file name
    base_teff_str_length = 5
    leading_zeros = "0" * (base_teff_str_length - len(str(teff)))
    teff_file_str = (leading_zeros + str(teff))[:3]
    filename += teff_file_str

    # logg part of file name
    if logg < 0:
        filename += "+"
    else:
        filename += "-"
    if logg == 0:
        filename += "0.0"
    else:
        filename += str(abs(logg))

    # feh part of file name
    if feh > 0:
        filename += "+"
    else:
        filename += "-"
    if feh == 0:
        filename += "0.0"
    else:
        filename += str(abs(feh))

    filename += "a"

    # alpha part of file name
    if alpha < 0:
        filename += "-"
    else:
        filename += "+"
    if alpha == 0:
        filename += "0.0"
    else:
        filename += str(abs(alpha))

    filename += ".BT-NextGen.7.dat.txt"

    return filename


def process_spectrum(filepath: Path | str) -> np.ndarray:
    """
    Read one raw spectrum and resample it onto WAVELENGTH_PTS.

    Returns F_lambda (erg/s/cm^2/A) on WAVELENGTH_PTS.
    """
    spectra_df = pd.read_csv(
        filepath, sep=r"\s+", comment="#", names=["wavelength", "flux"]
    )

    # down sample full spectrum onto new wavelength grid
    spectra_func = interpolate.interp1d(
        spectra_df["wavelength"], spectra_df["flux"], fill_value="extrapolate"
    )  # in angstroms
    spectra_interpolated = spectra_func(WAVELENGTH_PTS)

    return spectra_interpolated


def find_spectrum_file(
    teff: float,
    logg: float,
    feh: float,
    spectra_dir: Path | str = SPECTRA_RAW_PATH_DEFAULT,
) -> tuple[float, Path]:
    """
    (alpha, path) of the raw spectrum for a (teff, logg, feh) grid point,
    trying the alphas in ALPHA_GRID_PTS order.

    Raises FileNotFoundError when no alpha has a spectrum on disk.
    """
    spectra_dir = Path(spectra_dir)
    for alpha in ALPHA_GRID_PTS:
        pt_dict = {"teff": teff, "logg": logg, "feh": feh, "alpha": alpha}
        path = spectra_dir / get_NextGen2009_filename(pt_dict)
        if path.is_file():
            return float(alpha), path
    raise FileNotFoundError(
        f"No BT-NextGen spectrum in {spectra_dir} for teff={teff}, "
        f"logg={logg}, feh={feh} at any alpha in {list(ALPHA_GRID_PTS)}."
    )


def load_spectrum(
    teff: float,
    logg: float,
    feh: float,
    spectra_dir: Path | str = SPECTRA_RAW_PATH_DEFAULT,
) -> tuple[float, np.ndarray]:
    """(alpha, F_lambda on WAVELENGTH_PTS) for a (teff, logg, feh) node."""
    alpha, path = find_spectrum_file(teff, logg, feh, spectra_dir)
    return alpha, process_spectrum(path)
