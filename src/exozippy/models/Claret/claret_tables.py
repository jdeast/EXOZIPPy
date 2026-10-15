"""
The Claret limb-darkening tables on disk: where they live, what they are
called, and the ONE reader.

The tables are built by ``generate_claret_tables.py`` (in this directory)
from Claret's VizieR catalogs and published on Zenodo; a fit fetches the
tables it reads on first use, exactly like the NextGen BC tables
(``models/NextGen/bc_tables.py``), through the same
``utilities/zenodo.fetch_assets`` core and machine cache. The pins (record
id, concept record, every file's size and md5) live in
``utilities/zenodo_assets.py`` as record ``claret_ld_tables``. A present
table that differs from its pin RAISES rather than being overwritten or
used: it can only be a local rebuild, and a fit scored on it could not be
reproduced from the record. ``exozippy-fetch-claret-tables`` pre-fetches the
whole set for a machine that will run offline.

Every spelling shared by the generator and the readers lives here, so the
file a reader asks for is by construction the file the generator wrote:
the dataset name, the directory, the file names, the per-law coefficient
columns and the box every table is trimmed to.

The rulings are in components/sed/sed.md ("The Claret limb-darkening
tie"); the consumer is ``components/band/claret.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT, read_bc_meta
from exozippy.utilities import zenodo, zenodo_assets

# The subset EXOFASTv2 uses: ATLAS models, least-squares fit, 2 km/s.
DATASET = "claret_ATLAS_L_vt2"

CLARET_ROOT = DEFAULT_MODEL_ROOT / "Claret"
# Where a fit reads (and fetches) the published tables.
DEFAULT_TABLE_DIR = CLARET_ROOT / "ATLAS_L_vt2"
# Where the generator writes a rebuild and its raw VizieR downloads -- NOT
# the fetch destination, so a local build can never sit where a fit would
# read it and fail the published pins.
DEFAULT_BUILD_DIR = CLARET_ROOT / "build"
DEFAULT_SOURCE_DIR = CLARET_ROOT / "sources"

# The published record (utilities/zenodo_assets.py, "claret_ld_tables").
_RECORD = zenodo_assets.record("claret_ld_tables")
ZENODO_RECORD = _RECORD.record_id
_TABLE_FILES = {
    name: {"size": pin.size, "md5": pin.md5}
    for name, pin in _RECORD.files.items()
    if name.endswith(".parquet")
}
FETCH_COMMAND = "exozippy-fetch-claret-tables"

# (path, size, mtime_ns, inode) of tables already md5-verified this process.
_verified: set[tuple[str, int, int, int]] = set()

# Coefficient columns per limb-darkening law, in the order a table stores
# them: I(mu)/I(1) = 1 - a(1-mu) - b(1-mu)^2 (quadratic) or 1 - u(1-mu)
# (linear). The law names are band.ld_law's.
LAW_COLUMNS = {"quadratic": ("a", "b"), "linear": ("u",)}
LAW_SUFFIX = {"quadratic": "quad", "linear": "linear"}

# The box complete in every table (the generator's README reports it).
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


def table_path(
    band: str, law: str, table_dir: Path | str | None = None
) -> Path:
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


def load_table(
    band: str, law: str, table_dir: Path | str | None = None
) -> ClaretTable:
    """Read one table and reshape it onto its grid.

    With ``table_dir`` None the table is the PUBLISHED one (``ensure_tables``:
    fetched if absent, md5-checked if present). Raises ``RuntimeError``
    (deliberately not ``FileNotFoundError``, the rule the BC-table fetch
    follows: callers must not be able to read a missing table as "no table
    for this filter") when it is absent and cannot be had, and ``ValueError`` when what is on disk is not a complete grid of the
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
    if table_dir is None:
        # The published set: fetched on first use, checked against its pin.
        path = ensure_tables([table_filename(band, law)])[0]
    else:
        # An explicit directory is used as given and never fetched into.
        path = table_path(band, law, table_dir)
        if not path.is_file():
            raise RuntimeError(
                f"Claret limb-darkening table {path} is missing from the "
                f"explicitly given table directory {table_dir}."
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


# ---------------------------------------------------------------------------
# The published set: lazy fetch and pin check (models/NextGen/bc_tables.py's
# pattern).
# ---------------------------------------------------------------------------
def published_tables() -> List[str]:
    """Filenames of every published table, sorted."""
    return sorted(_TABLE_FILES)


def record_url() -> str:
    """Human-facing URL of the pinned record."""
    return f"https://zenodo.org/records/{ZENODO_RECORD}"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_present(path: Path, name: str) -> None:
    """Raise unless the table at ``path`` is byte-identical to its pin."""
    meta = _TABLE_FILES[name]
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
    if key in _verified:
        return
    if st.st_size != meta["size"] or _md5(path) != meta["md5"]:
        raise RuntimeError(
            f"{path} is not the published {name} (pinned size {meta['size']}, "
            f"md5 {meta['md5']}; record {record_url()}). It was most likely "
            f"rebuilt locally. It is not overwritten; move it aside to fetch "
            f"the published table, or publish the rebuild as a new version "
            f"of the record and update the pins in utilities/zenodo_assets.py."
        )
    _verified.add(key)


def ensure_tables(
    filenames: Sequence[str] | None = None, dest_dir: Path | str | None = None
) -> List[Path]:
    """Make sure the named published tables are on disk; return their paths.

    ``filenames`` None means every published table; a name that is not
    published raises KeyError. ``dest_dir`` defaults to DEFAULT_TABLE_DIR,
    read at CALL time. A present table that differs from its pin raises
    (``_check_present``); an absent one that cannot be fetched raises
    RuntimeError naming the file, the record and how to pre-fetch.
    """
    dest_dir = DEFAULT_TABLE_DIR if dest_dir is None else Path(dest_dir)
    names = published_tables() if filenames is None else list(filenames)
    unknown = [n for n in names if n not in _TABLE_FILES]
    if unknown:
        raise KeyError(
            f"{unknown} are not published Claret tables; published: "
            f"{published_tables()}"
        )

    missing = []
    for name in names:
        path = dest_dir / name
        if path.is_file():
            _check_present(path, name)
        else:
            missing.append(name)

    if missing:
        assets = {
            name: {**_TABLE_FILES[name], "url": f"{record_url()}/files/{name}"}
            for name in missing
        }
        try:
            zenodo.fetch_assets(assets, dest_dir)
        except (RuntimeError, urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                f"Could not fetch the Claret limb-darkening table(s) "
                f"{missing} into {dest_dir} from Zenodo record "
                f"{record_url()}: {e}. Pre-fetch them on a machine with "
                f"network access with `{FETCH_COMMAND}`; the machine cache "
                f"it fills then serves every checkout on this machine."
            ) from e
        for name in missing:
            # fetch_assets verified size+md5 before the file became visible.
            path = dest_dir / name
            st = path.stat()
            _verified.add((str(path), st.st_size, st.st_mtime_ns, st.st_ino))

    return [dest_dir / n for n in names]


def fetch_all() -> List[Path]:
    """Fetch every published table into DEFAULT_TABLE_DIR.

    The owner hook for the ``claret_ld_tables`` record
    (``zenodo_assets.OWNERS``): the pre-fetch that ``zenodo
    .fetch_every_record`` runs before the test suite and in CI. The
    record's MANIFEST.txt and README.md are not fetched; no reader opens
    them.
    """
    return ensure_tables(None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=FETCH_COMMAND,
        description=(
            "Download (or link from the machine cache) every published "
            "Claret limb-darkening table, so fits can run offline afterwards."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"Destination directory (default: {DEFAULT_TABLE_DIR}).",
    )
    args = parser.parse_args(argv)
    paths = ensure_tables(None, dest_dir=args.dest)
    print(f"{len(paths)} Claret tables present in {paths[0].parent}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
