"""
Build the shipped NextGen bolometric-correction tables from the
full-resolution BT-NextGen (AGSS2009) spectra.

Two steps, mirroring models/MIST/generate_MIST_EEP_Tables.py (see
README.md in this directory for the workflow):

1. Resample every raw spectrum on the (teff, logg, feh) grid of
   NextGen.grid.yaml onto the common R = 20000 wavelength grid and cache
   them as one parquet file per feh (``feh{+X.X}.spectra.parquet``, ~0.7 GB
   each) in SPECTRA_PROCESSED_PATH. Reading the ~13 MB ASCII spectra is
   the slow part (~1 s each, ~7000 of them), so it is done once; step 2
   can then be re-run for new filters in minutes. Resumable: a feh whose
   parquet already exists is skipped.

2. Compute BC_X(teff, logg, feh, Av) with BolometricCorrection for every
   filter set in FILTER_SETS and write one table per facility,
   ``models/NextGen/BCs/{FACILITY}.bc.parquet`` (bc_grid.write_bc_table).
   Existing columns of a table that are not being regenerated are kept.
"""

from __future__ import annotations

import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

from exozippy.components.sed.bc_grid import (
    DEFAULT_MODEL_ROOT,
    bc_table_path,
    facility_from_svo_name,
    write_bc_table,
)
from exozippy.filters.filter import Filter
from exozippy.models.NextGen.bolometric_correction import (
    L0,
    BolometricCorrection,
)
from exozippy.models.NextGen.nextgen_spectra import (
    SPECTRA_RAW_PATH_DEFAULT,
    WAVELENGTH_PTS,
    find_spectrum_file,
    process_spectrum,
)

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# -------------------------------------------------------------------
# Defining Default Paths / Settings
# -------------------------------------------------------------------

MODEL = "NextGen"

# replace with your path to where the resampled spectra should be cached
SPECTRA_PROCESSED_PATH_DEFAULT = Path(
    "/Volumes/Data/Spectra/BT-NextGen_AGSS2009_processed/"
)

# Fixed R_V (the extinction law in models/extinction_law.ascii)
RV = 3.10

# Magnitude system of the zeropoints BolometricCorrection uses
MAG_SYSTEM = "Vega"

# Band-average weighting (see bolometric_correction.py's docstring):
# "detector" = per filter from its SVO DetectorType
FLUX_WEIGHTING = "detector"

# Filters to compute BCs for, grouped by facility (the facility must be the
# SVO id's prefix -- it names the output table). Defaults reproduce every
# column of the tables that shipped before this pipeline existed.
FILTER_SETS: Dict[str, List[str]] = {
    "2MASS": ["2MASS/2MASS.J", "2MASS/2MASS.H", "2MASS/2MASS.Ks"],
    "GAIA": ["GAIA/GAIA2r.G", "GAIA/GAIA2r.Gbp", "GAIA/GAIA2r.Grp"],
    "WISE": ["WISE/WISE.W1", "WISE/WISE.W2", "WISE/WISE.W3", "WISE/WISE.W4"],
    "Generic": [
        "Generic/Cousins.I",
        "Generic/Cousins.R",
        "Generic/Bessell.U",
        "Generic/Bessell.B",
        "Generic/Bessell.V",
        "Generic/Bessell.R",
        "Generic/Bessell.I",
    ],
    "Keck": ["Keck/NIRC2.Kp", "Keck/NIRC2.J"],
    "TESS": ["TESS/TESS.Red"],
}


def read_grid_yaml(model_root: Path | str = DEFAULT_MODEL_ROOT) -> dict:
    """The (teff, logg, feh, av) axes the tables are built on."""
    path = Path(model_root) / MODEL / "BCs" / f"{MODEL}.grid.yaml"
    with open(path, "r") as f:
        grid = yaml.safe_load(f)["grid"]
    return {k: np.array(v, dtype=float) for k, v in grid.items()}


# -------------------------------------------------------------------
# Step 1: Resample Raw Spectra into Per-feh Parquet Files
# -------------------------------------------------------------------


def _processed_filename(feh: float) -> str:
    return f"feh{feh:+.1f}.spectra.parquet"


def process_raw_spectra_for_feh(
    feh: float,
    teff_pts: Sequence[float],
    logg_pts: Sequence[float],
    raw_path: Path = SPECTRA_RAW_PATH_DEFAULT,
    processed_path: Path = SPECTRA_PROCESSED_PATH_DEFAULT,
    n_workers: int = 4,
    save: bool = True,
) -> pd.DataFrame:
    """
    Resample the raw spectrum of every (teff, logg) node at one feh onto
    nextgen_spectra.WAVELENGTH_PTS, taking the first alpha in
    ALPHA_GRID_PTS order that has a spectrum on disk.

    Returns (and optionally saves) a DataFrame with one row per node:
    teff, logg, feh, alpha, flux (F_lambda in erg/s/cm^2/A, an array on
    WAVELENGTH_PTS).
    """
    nodes = []
    missing = []
    for teff in teff_pts:
        for logg in logg_pts:
            try:
                alpha, path = find_spectrum_file(teff, logg, feh, raw_path)
            except FileNotFoundError:
                missing.append((teff, logg))
                continue
            nodes.append((float(teff), float(logg), alpha, path))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} (teff, logg) nodes at feh={feh} have no raw "
            f"spectrum in {raw_path} at any alpha: {missing}"
        )

    paths = [path for *_, path in nodes]
    desc = f"feh{feh:+.1f}"
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            fluxes = list(
                tqdm(
                    pool.map(process_spectrum, paths, chunksize=4),
                    total=len(paths),
                    desc=desc,
                    leave=False,
                )
            )
    else:
        fluxes = [
            process_spectrum(p) for p in tqdm(paths, desc=desc, leave=False)
        ]

    df = pd.DataFrame(
        {
            "teff": [n[0] for n in nodes],
            "logg": [n[1] for n in nodes],
            "feh": float(feh),
            "alpha": [n[2] for n in nodes],
            "flux": fluxes,
        }
    )
    df.attrs["meta"] = {
        "model": "BT-NextGen (AGSS2009)",
        "raw_path": str(raw_path),
        "flux_unit": "erg/s/cm^2/A",
        "wavelength_unit": "Angstrom",
        "wavelength_resolution": Filter.RESOLUTION,
        "wavelength_min_micron": Filter._LAMBDA_MIN,
        "wavelength_max_micron": Filter._LAMBDA_MAX,
        "n_wavelength": len(WAVELENGTH_PTS),
    }

    if save:
        processed_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            processed_path / _processed_filename(feh), compression="snappy"
        )

    return df


def load_processed_spectra(
    feh: float, processed_path: Path = SPECTRA_PROCESSED_PATH_DEFAULT
) -> pd.DataFrame:
    """Read one step-1 parquet, checking its wavelength grid is current."""
    path = processed_path / _processed_filename(feh)
    df = pd.read_parquet(path, engine="pyarrow")
    n_wave = df.attrs.get("meta", {}).get("n_wavelength")
    if n_wave != len(WAVELENGTH_PTS):
        raise ValueError(
            f"{path} was resampled onto {n_wave} wavelengths, but the "
            f"current grid has {len(WAVELENGTH_PTS)}; re-run step 1 for it."
        )
    return df


# -------------------------------------------------------------------
# Step 2: Compute and Write the BC Tables
# -------------------------------------------------------------------


def _check_filter_sets(filter_sets: Dict[str, List[str]]) -> None:
    for facility, filters in filter_sets.items():
        wrong = [f for f in filters if facility_from_svo_name(f) != facility]
        if wrong:
            raise ValueError(
                f"Filter set '{facility}' contains {wrong}, whose SVO "
                f"facility prefix is not '{facility}'. Group filters by "
                f"their SVO prefix -- it names the table they are read from."
            )


def generate_bc_tables(
    filter_sets: Dict[str, List[str]] = FILTER_SETS,
    processed_path: Path = SPECTRA_PROCESSED_PATH_DEFAULT,
    model_root: Path = DEFAULT_MODEL_ROOT,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Compute BCs for every filter set on the NextGen.grid.yaml axes from
    the step-1 spectra and write one table per facility.

    feh is the outer loop so each (large) processed-spectra file is read
    once for all filter sets.

    Returns {facility: DataFrame of the newly computed columns}.
    """
    _check_filter_sets(filter_sets)
    grid = read_grid_yaml(model_root)
    av_pts = grid["av"]

    records: Dict[str, list] = {fac: [] for fac in filter_sets}
    bc_objs: Dict[str, BolometricCorrection] = {}

    for feh in tqdm(grid["feh"], desc="feh"):
        df_spec = load_processed_spectra(feh, processed_path)
        df_spec = df_spec.set_index(["teff", "logg"])
        for teff in grid["teff"]:
            for logg in grid["logg"]:
                row = df_spec.loc[(teff, logg)]
                star_dict = {
                    "teff": teff,
                    "logg": logg,
                    "feh": feh,
                    "av": av_pts,
                }
                spectrum = (float(row["alpha"]), np.asarray(row["flux"]))
                for fac, filters in filter_sets.items():
                    bc_obj = BolometricCorrection(
                        filters,
                        star_dict,
                        spectrum=spectrum,
                        weighting=FLUX_WEIGHTING,
                    )
                    bc_objs[fac] = bc_obj
                    for av, bcs in zip(av_pts, bc_obj.BC_by_av):
                        records[fac].append(
                            (teff, logg, feh, bc_obj.alpha, av, RV, *bcs)
                        )

    generated = datetime.date.today().isoformat()
    out = {}
    for fac, recs in records.items():
        bc_obj = bc_objs[fac]
        cols = bc_obj.filters_MIST
        df = pd.DataFrame(
            recs, columns=["teff", "logg", "feh", "alpha", "Av", "Rv"] + cols
        )
        filter_meta = {
            col: {
                "svo_id": svo_id,
                "zeropoint_Fl_Vega": float(zp),
                "flux_weighting": weighting,
                "spectra": "BT-NextGen (AGSS2009), full resolution "
                f"resampled to R={Filter.RESOLUTION}",
                "generator": "models/NextGen/generate_NextGen_BC_Tables.py",
                "generated": generated,
            }
            for col, svo_id, zp, weighting in zip(
                cols,
                bc_obj.filters_SVO,
                bc_obj.filter_zero_pts,
                bc_obj.filter_weightings,
            )
        }
        if save:
            path = bc_table_path(model_root, MODEL, fac)
            write_bc_table(
                df,
                path,
                filter_meta,
                table_meta={
                    "model": MODEL,
                    "facility": fac,
                    "mag_system": MAG_SYSTEM,
                    "Rv": RV,
                    "L0_W": L0,
                },
            )
            print(f"Wrote {path}")
        out[fac] = df
    return out


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

SPECTRA_RAW_PATH = SPECTRA_RAW_PATH_DEFAULT
SPECTRA_PROCESSED_PATH = SPECTRA_PROCESSED_PATH_DEFAULT


def __main_step1_process_raw_spectra__():

    grid = read_grid_yaml()
    for feh in grid["feh"]:
        # first check if the parquet file for this feh already exists
        parquet_file = SPECTRA_PROCESSED_PATH / _processed_filename(feh)
        if parquet_file.exists():
            print(f"Parquet file already exists for feh={feh:+.1f}")
            continue
        print(f"Processing feh={feh:+.1f}")
        process_raw_spectra_for_feh(
            feh,
            grid["teff"],
            grid["logg"],
            raw_path=SPECTRA_RAW_PATH,
            processed_path=SPECTRA_PROCESSED_PATH,
            save=True,
        )


def __main_step2_generate_bc_tables__():

    generate_bc_tables(
        filter_sets=FILTER_SETS,
        processed_path=SPECTRA_PROCESSED_PATH,
        save=True,
    )


# depending on what you want to run, you can comment out either step
if __name__ == "__main__":
    __main_step1_process_raw_spectra__()
    __main_step2_generate_bc_tables__()
