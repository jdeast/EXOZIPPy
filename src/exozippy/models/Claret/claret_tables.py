"""
The Claret limb-darkening tables on disk: where they live, what they are
called, and the ONE reader.

The tables are built by ``generate_claret_tables.py`` (in this directory)
from Claret's VizieR catalogues and are destined for a Zenodo record, from
which they will be fetched on first use exactly like the NextGen BC tables
(``models/NextGen/bc_tables.py``). Until that record exists the only way to
get them is to run the generator, and a missing table says so.

Every spelling shared by the generator and the readers lives here, so the
file a reader asks for is by construction the file the generator wrote:
the dataset name, the directory, the file names, the per-law coefficient
columns and the box every table is trimmed to.

The design is in docs/claret_ld_prior_plan.md; the consumer is
``components/band/claret.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT, read_bc_meta

# The subset EXOFASTv2 uses: ATLAS models, least-squares fit, 2 km/s.
DATASET = "claret_ATLAS_L_vt2"

CLARET_ROOT = DEFAULT_MODEL_ROOT / "Claret"
DEFAULT_TABLE_DIR = CLARET_ROOT / "ATLAS_L_vt2"
DEFAULT_SOURCE_DIR = CLARET_ROOT / "sources"

# Coefficient columns per limb-darkening law, in the order a table stores
# them: I(mu)/I(1) = 1 - a(1-mu) - b(1-mu)^2 (quadratic) or 1 - u(1-mu)
# (linear). The law names are band.ld_law's.
LAW_COLUMNS = {"quadratic": ("a", "b"), "linear": ("u",)}
LAW_SUFFIX = {"quadratic": "quad", "linear": "linear"}

# The box complete in every table (docs/claret_ld_prior_plan.md, A2).
# Every [Fe/H] <= -2.5 row has missing dwarf cells, and below logg 2.0 the
# complete part shrinks to Teff <= 8000 K. A star with the tie on is held
# inside it (Band injects these as bounds on the star).
BOX = {"teff": (3500.0, 10000.0), "logg": (2.0, 5.0), "feh": (-2.0, 1.0)}

AXIS_ORDER = ("teff", "logg", "feh")

# Which (Claret band, law) tables exist. Claret & Bloemen (2011) tabulate
# both laws for their 23 bands; Claret (2017) is used for TESS's quadratic
# coefficients only (EXOFASTv2 has no linear TESS table either). The
# generator asserts it builds exactly this set, so asking for anything else
# is "there is no such table", not "the table is missing".
_CB11_BANDS = (
    "U", "B", "V", "R", "I", "J", "H", "K",
    "Sloanu", "Sloang", "Sloanr", "Sloani", "Sloanz",
    "Kepler", "CoRoT", "Spit36", "Spit45", "Spit58", "Spit80",
    "Stromu", "Stromb", "Stromv", "Stromy",
)  # fmt: skip
TABLE_BANDS = {
    "quadratic": _CB11_BANDS + ("TESS",),
    "linear": _CB11_BANDS,
}


def table_filename(band: str, law: str) -> str:
    """The file name of one (Claret band, law) table."""
    return f"{DATASET}.{band}.{LAW_SUFFIX[law]}.parquet"


def table_path(band: str, law: str, table_dir: Path | str | None = None) -> Path:
    """Where one table lives: ``table_dir``, or DEFAULT_TABLE_DIR when None.

    The default is read at CALL time (not bound as a default argument at
    import), so the test suite can point every reader at synthetic tables
    by patching this module's DEFAULT_TABLE_DIR.
    """
    root = DEFAULT_TABLE_DIR if table_dir is None else table_dir
    return Path(root) / table_filename(band, law)


@dataclass(frozen=True)
class ClaretTable:
    """One band's coefficients on a complete regular (teff, logg, feh) grid.

    ``values`` has shape (n_teff, n_logg, n_feh, n_coefficients), the
    coefficients in ``columns`` order -- the layout
    ``bc_grid.RegularGridInterpolator`` interpolates directly.
    """

    band: str
    law: str
    columns: Tuple[str, ...]
    axes: Dict[str, np.ndarray]
    values: np.ndarray
    meta: Dict
    path: Path


def load_table(band: str, law: str, table_dir: Path | str | None = None) -> ClaretTable:
    """Read one table and reshape it onto its grid.

    Raises ``RuntimeError`` (deliberately not ``FileNotFoundError``, the
    rule the BC-table fetch follows: callers must not be able to read a
    missing table as "no table for this filter") when the file is absent,
    and ``ValueError`` when what is on disk is not a complete grid of the
    expected columns -- a table the generator would never have written.
    """
    if law not in LAW_COLUMNS:
        raise ValueError(
            f"Claret tables exist for the laws {sorted(LAW_COLUMNS)}, not {law!r}."
        )
    if band not in TABLE_BANDS[law]:
        raise ValueError(
            f"There is no {law} Claret table for {band!r}; {law} tables exist "
            f"for {', '.join(TABLE_BANDS[law])}."
        )
    path = table_path(band, law, table_dir)
    if not path.is_file():
        raise RuntimeError(
            f"Claret limb-darkening table {path} is missing. The tables are "
            f"not yet published on Zenodo; build them with\n"
            f"    python -m exozippy.models.Claret.generate_claret_tables "
            f"--exofast <EXOFASTv2 checkout>\n"
            f"(see docs/claret_ld_prior_plan.md, Phase 1)."
        )
    df = pd.read_parquet(path)
    meta = read_bc_meta(path)
    columns = LAW_COLUMNS[law]
    expected = set(AXIS_ORDER) | set(columns)
    if set(df.columns) != expected:
        raise ValueError(
            f"{path}: columns {sorted(df.columns)}, expected {sorted(expected)}."
        )
    if meta.get("band") != band or meta.get("law") != law:
        raise ValueError(
            f"{path}: its metadata says band {meta.get('band')!r}, law "
            f"{meta.get('law')!r}, but it was read as {band!r} ({law})."
        )

    axes = {ax: np.unique(df[ax].to_numpy(dtype=float)) for ax in AXIS_ORDER}
    shape = tuple(len(axes[ax]) for ax in AXIS_ORDER)
    if len(df) != int(np.prod(shape)):
        raise ValueError(
            f"{path}: {len(df)} rows do not fill the {shape} grid of its own "
            f"axes; the table is not a complete regular grid."
        )
    ordered = df.sort_values(list(AXIS_ORDER), ignore_index=True)
    values = np.stack(
        [ordered[c].to_numpy(dtype=float).reshape(shape) for c in columns],
        axis=-1,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: the table holds non-finite coefficients.")
    for ax in AXIS_ORDER:
        lo, hi = BOX[ax]
        if axes[ax][0] != lo or axes[ax][-1] != hi:
            raise ValueError(
                f"{path}: its {ax} axis spans [{axes[ax][0]}, {axes[ax][-1]}], "
                f"not the box [{lo}, {hi}] this code bounds the star to."
            )
    return ClaretTable(
        band=band,
        law=law,
        columns=columns,
        axes=axes,
        values=values,
        meta=meta,
        path=path,
    )
