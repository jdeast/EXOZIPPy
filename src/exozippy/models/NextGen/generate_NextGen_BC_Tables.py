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
   can then be re-run for new filters in minutes. Incremental
   (process_missing_spectra): only the (teff, logg, feh) nodes the
   parquets do not hold yet are resampled.

2. Compute BC_X(teff, logg, feh, Av) with BolometricCorrection for every
   filter set in FILTER_SETS and write one table per facility,
   ``models/NextGen/BCs/{FACILITY}.bc.parquet`` (bc_grid.write_bc_table).
   Incremental on every axis and filter: only the (node, filter) cells
   the tables do not hold yet are computed (plan_bc_work), so extending
   NextGen.grid.yaml (e.g. to larger Av) or adding a filter to
   FILTER_SETS costs only the new cells.
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
    BC_PARAM_COLS,
    DEFAULT_MODEL_ROOT,
    GRID_KEY_COLS,
    _KEY_DECIMALS,
    _load_alias_table,
    bc_nodes_to_compute,
    bc_table_filter_columns,
    bc_table_path,
    facility_from_svo_name,
    read_bc_meta,
    resolve_filter_name,
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
    # The full Roman WFI imaging set (master shipped these as a text table
    # built by make_bc from the downsampled spectra).
    "Roman": [
        "Roman/WFI.F062",
        "Roman/WFI.F087",
        "Roman/WFI.F106",
        "Roman/WFI.F129",
        "Roman/WFI.F146",
        "Roman/WFI.F158",
        "Roman/WFI.F184",
        "Roman/WFI.F213",
    ],
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


def _node_key(teff, logg) -> tuple:
    """A (teff, logg) node, rounded like the BC table keys."""
    return (
        round(float(teff), _KEY_DECIMALS),
        round(float(logg), _KEY_DECIMALS),
    )


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
    only_nodes: Sequence[tuple] | None = None,
) -> pd.DataFrame:
    """
    Resample the raw spectrum of every (teff, logg) node at one feh onto
    nextgen_spectra.WAVELENGTH_PTS, taking the first alpha in
    ALPHA_GRID_PTS order that has a spectrum on disk.

    `only_nodes`, a collection of (teff, logg) pairs, restricts the work
    to those nodes (process_missing_spectra passes the ones a feh's
    parquet does not hold yet). With `save`, the result is written as that
    feh's parquet -- so a partial run should go through
    process_missing_spectra, which merges instead.

    Returns (and optionally saves) a DataFrame with one row per node:
    teff, logg, feh, alpha, flux (F_lambda in erg/s/cm^2/A, an array on
    WAVELENGTH_PTS).
    """
    only = (
        None
        if only_nodes is None
        else {_node_key(t, g) for t, g in only_nodes}
    )
    nodes = []
    missing = []
    for teff in teff_pts:
        for logg in logg_pts:
            if only is not None and _node_key(teff, logg) not in only:
                continue
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


def process_missing_spectra(
    grid: dict | None = None,
    raw_path: Path = SPECTRA_RAW_PATH_DEFAULT,
    processed_path: Path = SPECTRA_PROCESSED_PATH_DEFAULT,
    n_workers: int = 4,
) -> Dict[float, int]:
    """
    Step 1, incrementally: bring every feh parquet up to the (teff, logg,
    feh) axes of `grid` (default: NextGen.grid.yaml), resampling ONLY the
    nodes it does not hold yet.

    Only the teff/logg columns of an existing parquet are read to find
    what it holds; the flux column is read (and the file rewritten) only
    when there is something to add. The Av axis needs no spectra, so an
    extension along Av alone makes this a no-op.

    Returns {feh: number of nodes processed}.
    """
    grid = read_grid_yaml() if grid is None else grid
    done = {}
    for feh in grid["feh"]:
        path = processed_path / _processed_filename(feh)
        want = {_node_key(t, g) for t in grid["teff"] for g in grid["logg"]}
        if path.exists():
            have = pd.read_parquet(path, columns=["teff", "logg"])
            want -= {_node_key(t, g) for t, g in zip(have.teff, have.logg)}
        if not want:
            print(f"feh={feh:+.1f}: every node already processed")
            done[float(feh)] = 0
            continue
        print(f"feh={feh:+.1f}: processing {len(want)} missing node(s)")
        df_new = process_raw_spectra_for_feh(
            feh,
            grid["teff"],
            grid["logg"],
            raw_path=raw_path,
            processed_path=processed_path,
            n_workers=n_workers,
            save=False,
            only_nodes=want,
        )
        if path.exists():
            df_old = load_processed_spectra(feh, processed_path)
            meta = df_old.attrs.get("meta", {})
            df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.attrs["meta"] = meta
        df_new = df_new.sort_values(["teff", "logg"]).reset_index(drop=True)
        processed_path.mkdir(parents=True, exist_ok=True)
        df_new.to_parquet(path, compression="snappy")
        done[float(feh)] = len(want)
    return done


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


def _index_spectra(df_spec: pd.DataFrame) -> pd.DataFrame:
    """Index a step-1 frame by its (teff, logg) node, rounded like the BC
    table keys. Rounds the index only -- the flux column is ~0.7 GB."""
    df_spec = df_spec.set_index(["teff", "logg"])
    df_spec.index = pd.MultiIndex.from_arrays(
        [
            np.round(
                df_spec.index.get_level_values(n).astype(float),
                _KEY_DECIMALS,
            )
            for n in ("teff", "logg")
        ],
        names=["teff", "logg"],
    )
    return df_spec


# Provenance a column must carry for its existing values to count as
# already computed by THIS pipeline. A column written by make_bc (the
# downsampled Zenodo spectra) or converted from the legacy text tables is
# replaced in full rather than extended, so no column ends up mixing two
# pipelines.
GENERATOR = "models/NextGen/generate_NextGen_BC_Tables.py"
SPECTRA_TAG = (
    "BT-NextGen (AGSS2009), full resolution resampled to "
    f"R={Filter.RESOLUTION}"
)


def _reusable(filter_meta: dict) -> bool:
    return (
        filter_meta.get("generator") == GENERATOR
        and filter_meta.get("spectra") == SPECTRA_TAG
    )


def plan_bc_work(
    filter_sets: Dict[str, List[str]] = FILTER_SETS,
    model_root: Path = DEFAULT_MODEL_ROOT,
    overwrite: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    What generate_bc_tables would compute, without computing it.

    Returns {facility: DataFrame} with one row per (teff, logg, feh, Av)
    node of NextGen.grid.yaml at which at least one of the facility's
    filters is still missing, and one boolean column per filter (MIST
    column name; True = compute that cell). A facility with nothing to do
    is left out. `overwrite=True` plans every cell.
    """
    _check_filter_sets(filter_sets)
    grid = read_grid_yaml(model_root)
    alias_df = _load_alias_table()
    reusable = (lambda meta: False) if overwrite else _reusable

    plan = {}
    for fac, filters in filter_sets.items():
        cols = [
            resolve_filter_name(f, alias_df, alias="MIST") for f in filters
        ]
        todo = bc_nodes_to_compute(
            bc_table_path(model_root, MODEL, fac), grid, cols, reusable
        )
        if len(todo):
            plan[fac] = todo
    return plan


def _summarize_plan(plan: Dict[str, pd.DataFrame], n_nodes: int) -> None:
    if not plan:
        print("Every requested BC column is already computed; nothing to do.")
        return
    for fac, todo in plan.items():
        cols = [c for c in todo.columns if c not in GRID_KEY_COLS]
        counts = ", ".join(f"{c}: {int(todo[c].sum())}" for c in cols)
        print(f"{fac}: cells to compute (of {n_nodes} grid nodes) -- {counts}")


def generate_bc_tables(
    filter_sets: Dict[str, List[str]] = FILTER_SETS,
    processed_path: Path = SPECTRA_PROCESSED_PATH_DEFAULT,
    model_root: Path = DEFAULT_MODEL_ROOT,
    save: bool = True,
    overwrite: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Compute BCs for every filter set on the NextGen.grid.yaml axes from
    the step-1 spectra and write one table per facility -- computing ONLY
    the (node, filter) cells the tables do not already hold.

    Incremental on every axis: add Av (or teff, logg, feh) values to
    NextGen.grid.yaml, or filters to a facility's list, and re-run; the
    existing cells are read from the tables (plan_bc_work) and skipped.
    A [Fe/H] with nothing to compute is never loaded, and at each node
    BolometricCorrection runs only for the filters and Av values missing
    there. Existing values are never rewritten: new cells are merged in
    cell by cell (write_bc_table), and each table is checkpointed after
    every [Fe/H], so an interrupted run resumes where it stopped.

    A column this pipeline did not write (make_bc, legacy conversion) is
    recomputed in full; `overwrite=True` recomputes everything.

    feh is the outer loop so each (large) processed-spectra file is read
    once for all filter sets.

    Returns {facility: DataFrame of the newly computed cells} (NaN where a
    cell was not computed).
    """
    grid = read_grid_yaml(model_root)
    plan = plan_bc_work(filter_sets, model_root, overwrite=overwrite)
    _summarize_plan(plan, int(np.prod([len(v) for v in grid.values()])))

    alias_df = _load_alias_table()
    svo_of = {
        fac: {resolve_filter_name(f, alias_df, alias="MIST"): f for f in fs}
        for fac, fs in filter_sets.items()
    }
    generated = datetime.date.today().isoformat()
    table_meta = {
        "mag_system": MAG_SYSTEM,
        "Rv": RV,
        "L0_W": L0,
        "model": MODEL,
    }

    # Existing columns this run recomputes from scratch (written by another
    # pipeline, or everything under overwrite): their old values are
    # dropped on the facility's first checkpoint, so an interrupted run
    # never leaves old-pipeline cells under this pipeline's metadata.
    reusable = (lambda meta: False) if overwrite else _reusable
    to_replace = {}
    for fac in plan:
        path = bc_table_path(model_root, MODEL, fac)
        if path.is_file():
            fmeta = read_bc_meta(path).get("filters", {})
            to_replace[fac] = [
                c
                for c in bc_table_filter_columns(path)
                if c in svo_of[fac] and not reusable(fmeta.get(c, {}))
            ]

    out: Dict[str, list] = {fac: [] for fac in plan}
    fehs = sorted({f for todo in plan.values() for f in todo["feh"].unique()})
    for feh in tqdm(fehs, desc="feh"):
        df_spec = _index_spectra(load_processed_spectra(feh, processed_path))
        for fac, todo in plan.items():
            cols = [c for c in todo.columns if c not in GRID_KEY_COLS]
            sub = todo[todo["feh"] == feh]
            recs, filter_meta = [], {}
            for (teff, logg), node in sub.groupby(["teff", "logg"]):
                need = node[cols]
                ncols = [c for c in cols if need[c].any()]
                try:
                    row = df_spec.loc[(teff, logg)]
                except KeyError:
                    raise FileNotFoundError(
                        f"No step-1 spectrum for teff={teff}, logg={logg}, "
                        f"feh={feh} in {processed_path}; re-run step 1 "
                        f"(it processes only the missing nodes)."
                    ) from None
                av = node["Av"].values
                bc_obj = BolometricCorrection(
                    [svo_of[fac][c] for c in ncols],
                    {"teff": teff, "logg": logg, "feh": feh, "av": av},
                    spectrum=(float(row["alpha"]), np.asarray(row["flux"])),
                    weighting=FLUX_WEIGHTING,
                )
                if list(bc_obj.filters_MIST) != ncols:
                    raise RuntimeError(
                        f"BolometricCorrection named the columns "
                        f"{bc_obj.filters_MIST}, expected {ncols}."
                    )
                # Keep only the cells that were missing; NaN leaves the
                # table's existing value in place.
                vals = np.where(need[ncols].values, bc_obj.BC_by_av, np.nan)
                for a, v in zip(av, vals):
                    recs.append(
                        {
                            "teff": teff,
                            "logg": logg,
                            "feh": feh,
                            "alpha": bc_obj.alpha,
                            "Av": a,
                            "Rv": RV,
                            **dict(zip(ncols, v)),
                        }
                    )
                for col, svo_id, zp, weighting in zip(
                    ncols,
                    bc_obj.filters_SVO,
                    bc_obj.filter_zero_pts,
                    bc_obj.filter_weightings,
                ):
                    filter_meta[col] = {
                        "svo_id": svo_id,
                        "zeropoint_Fl_Vega": float(zp),
                        "flux_weighting": weighting,
                        "spectra": SPECTRA_TAG,
                        "generator": GENERATOR,
                        "generated": generated,
                    }
            if not recs:
                continue
            df = pd.DataFrame(recs)
            df = df[BC_PARAM_COLS + [c for c in cols if c in filter_meta]]
            out[fac].append(df)
            if save:
                # Checkpoint: this [Fe/H]'s cells are on disk before the
                # next [Fe/H] is started.
                path = bc_table_path(model_root, MODEL, fac)
                write_bc_table(
                    df,
                    path,
                    filter_meta,
                    table_meta={"facility": fac, **table_meta},
                    allow_new_nodes=True,
                    replace_columns=to_replace.pop(fac, ()),
                )
        del df_spec

    if save:
        for fac in out:
            print(f"Wrote {bc_table_path(model_root, MODEL, fac)}")
    return {
        fac: pd.concat(frames, ignore_index=True)
        for fac, frames in out.items()
        if frames
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

SPECTRA_RAW_PATH = SPECTRA_RAW_PATH_DEFAULT
SPECTRA_PROCESSED_PATH = SPECTRA_PROCESSED_PATH_DEFAULT


def __main_step1_process_raw_spectra__():

    # Only the (teff, logg, feh) nodes not yet in the processed parquets.
    process_missing_spectra(
        raw_path=SPECTRA_RAW_PATH,
        processed_path=SPECTRA_PROCESSED_PATH,
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
