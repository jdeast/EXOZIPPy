"""
Convert the pre-parquet NextGen BC text tables into the parquet format
bc_grid.py now loads.

The old layout was one whitespace-delimited file per (facility, feh):
    models/NextGen/BCs/{FACILITY}/feh{+/-X.X}_afe{+/-Y.Y}.{FACILITY}
with a 5-line `#` header and columns
    lgTef  logg  Fe_H  a_Fe  Av  Rv  <filter1> <filter2> ...
Each facility directory becomes models/NextGen/BCs/{FACILITY}.bc.parquet,
with every column's filter_meta marking it as converted from a legacy
table. The values are copied unchanged (to the text tables' 4 decimals).

This is a one-off migration aid; generate_NextGen_BC_Tables.py
regenerates the tables from the full-resolution spectra.

    poetry run python -m exozippy.models.NextGen.convert_legacy_BC_tables
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from exozippy.components.sed.bc_grid import (
    DEFAULT_MODEL_ROOT,
    _load_alias_table,
    bc_table_path,
    resolve_filter_name,
    write_bc_table,
)
from exozippy.models.NextGen.generate_NextGen_BC_Tables import FILTER_SETS

MODEL = "NextGen"

# compile pattern for bolometric correction tables
_FEH_FILENAME_RE = re.compile(
    r"feh(?P<feh>[+-]\d+\.\d+)_afe(?P<alpha>[+-]\d+\.\d+)\.(?P<facility>\w+)"
)


def _read_single_bc_file(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse a single BC file. Returns (DataFrame, filter_column_names).

    The header format is 4 or 5 `#` lines, the fourth/fifth of which contains the
    column names; pandas' comment="#" skips them, so we reconstruct
    the column names manually.
    """
    # Pull number of filters in file
    # Pull the header line that starts with "# lgTef"
    header_line = None
    with open(path, "r") as f:
        line_numfilters = -1
        for l, line in enumerate(f):
            stripped = line.lstrip("#").strip()
            if stripped.startswith("filters"):
                line_numfilters = l + 1
            if l == line_numfilters:
                numfilters = int(stripped.split()[0])
            if stripped.startswith("lgTef"):
                header_line = stripped
                break
    if header_line is None:
        raise ValueError(f"No column header line found in {path}")

    col_names = header_line.split()
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="c",
        comment="#",
        header=None,
        names=col_names,
    )

    # make changes to the columns' name
    df.insert(0, "teff", round(10 ** df["lgTef"]))
    df.rename(columns={"Fe_H": "feh", "a_Fe": "alpha"}, inplace=True)
    df.drop(columns=["lgTef"], inplace=True)
    filter_cols = col_names[
        -numfilters:
    ]  # after teff, logg, feh, alpha, Av, Rv

    return df, filter_cols


def _svo_id_by_column() -> dict:
    """MIST column name -> SVO id, for every filter in FILTER_SETS."""
    alias_df = _load_alias_table()
    return {
        resolve_filter_name(svo, alias_df, alias="MIST"): svo
        for filters in FILTER_SETS.values()
        for svo in filters
    }


def convert_facility(
    facility_dir: Path, model_root: Path = DEFAULT_MODEL_ROOT
) -> Path:
    """Convert every feh*_afe*.{FACILITY} file in one directory to parquet."""
    facility = facility_dir.name
    files = sorted(
        p
        for p in facility_dir.glob(f"feh*_afe*.{facility}")
        if _FEH_FILENAME_RE.match(p.name)
    )
    if not files:
        raise FileNotFoundError(f"No legacy BC files in {facility_dir}")

    frames = []
    filter_cols = None
    for p in files:
        df, cols = _read_single_bc_file(p)
        if filter_cols is None:
            filter_cols = cols
        elif cols != filter_cols:
            raise ValueError(
                f"{p} has filter columns {cols}, expected {filter_cols}."
            )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    svo_by_col = _svo_id_by_column()
    filter_meta = {
        col: {
            "svo_id": svo_by_col.get(col),
            "generator": "legacy text table (converted by "
            "models/NextGen/convert_legacy_BC_tables.py)",
            "source_files": f"{facility}/feh*_afe*.{facility}",
        }
        for col in filter_cols
    }
    path = bc_table_path(model_root, MODEL, facility)
    if path.exists():
        path.unlink()  # a conversion replaces, never merges
    write_bc_table(
        df,
        path,
        filter_meta,
        table_meta={
            "model": MODEL,
            "facility": facility,
            "mag_system": "Vega",
        },
    )
    return path


def convert_all(model_root: Path = DEFAULT_MODEL_ROOT) -> List[Path]:
    bc_dir = Path(model_root) / MODEL / "BCs"
    written = []
    for facility_dir in sorted(p for p in bc_dir.iterdir() if p.is_dir()):
        if not any(facility_dir.glob(f"feh*_afe*.{facility_dir.name}")):
            continue
        written.append(convert_facility(facility_dir, model_root))
        print(f"Wrote {written[-1]}")
    return written


if __name__ == "__main__":
    convert_all()
