"""
Bolometric Correction (BC) grid loader and pytensor interpolator.

Given a set of filter names and a model name ("NextGen" in v1),
it loads the matching per-facility BC tables from the `{MODEL}/BCs/`
tree and builds a pytensor-compatible RegularGridInterpolator over
(teff, logg, feh, Av) returning a vector of BC values, one per
requested filter.

File layout assumed (the NextGen tree):
    {model_root}/
        {model}/                     e.g. "NextGen"
            BCs/
                {model}.grid.yaml
                {FACILITY}.bc.parquet    e.g. "2MASS.bc.parquet"

Each table is a long-format parquet file (one row per grid node),
written by write_bc_table below, with columns:
    teff  logg  feh  alpha  Av  Rv  <filter1> <filter2> ...
Filter columns are named by their MIST BC-column name (see
resolve_filter_name). `alpha` is PROVENANCE -- the [alpha/Fe] of the
spectrum a row was computed from (the generator falls back through
alternate alphas when a node has no alpha = 0 spectrum) -- not a grid
axis. df.attrs["meta"] carries the table-level metadata and, per filter
column, its SVO id and how it was computed (see write_bc_table).

The tables are produced by models/NextGen/generate_NextGen_BC_Tables.py
(full-resolution spectra) or on demand by make_bc.py (the downsampled
Zenodo spectra); models/NextGen/README.md describes the workflow.

Grid assumptions in v1:
  * single Rv slice (Rv = 3.10) across all tables
  * (teff, logg, feh, Av) axes are identical across facilities
    (checked in build_bc_grid)
  * the grid assembled from these tables has axes (teff, logg, feh, av)

These assumptions hold for the NextGen tree as it currently ships.
When MIST is added, this loader grows a `model` dispatch so each
family can parse its own layout; the interpolator interface stays
the same.
"""

from __future__ import annotations

import itertools
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import pytensor.tensor as pt

# -------------------------------------------------------------------
# Filter name plumbing
# -------------------------------------------------------------------

# Filename alias table. Path relative to this file's location inside
# the installed package. The original file lives at:
#   Code for Models/Classes/filternames.txt
# but during packaging we expect a copy at components/sed/filternames.txt.
# Fall back to None if not found; the loader then assumes the user
# passes BC-column names directly.
try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

source_code_dir = current_dir.parent.parent  # source code two directories up
DEFAULT_FILTER_ROOT = source_code_dir / "filters"

_FILTERNAMES_ = "filternames.txt"

# Bare filter labels that name more than one row of the alias table, mapped
# to the MIST name of the row they resolve to.  The MIST column is unique
# per row (verified by tests/test_filter_aliases.py), so one MIST name picks
# exactly one row and the choice applies to every alias column at once.
#
# "I" and "R" appear twice in the Claret column -- Bessell and Cousins -- and
# once more in the Keivan column (Bessell).  ``resolve_filter_name`` scans the
# table row by row and takes the first hit, so before this map a bare "I"
# resolved to Bessell purely because Bessell_I is typed above Cousins_I in
# filternames.txt: an accident of file order, not a decision.  The project
# convention is that a bare "I"/"R" means the COUSINS band -- that is what
# the ground-based surveys these labels come from (OGLE, KMTNet, MOA, MEarth)
# actually observe in, and what the other shipped configs spell out as
# "Generic/Cousins.I".  Spell the filter out ("Bessell.I", "Cousins.I", or
# any full SVO id) whenever you mean something else; the map is only ever
# consulted for the bare label.
AMBIGUOUS_FILTER_ALIASES = {
    "I": "Cousins_I",
    "R": "Cousins_R",
}


def _load_alias_table(path: Path = DEFAULT_FILTER_ROOT) -> pd.DataFrame | None:
    """Load the VOID<->MIST<->SVO name alias table, if present.

    Cached (review 6.9.2): six call sites re-read and re-stripped the same
    small file, several of them inside per-filter loops, and Plot reads it
    at class-definition time.

    The cache key carries the file's mtime and size as well as its path, so
    a test (or a user) that REWRITES a filternames.txt in place still sees
    the new contents -- keying on the path alone is how a cache turns a
    correct test into a stale one.

    The returned frame is SHARED. Every caller in the tree only reads it
    (resolve_filter_name does lookups); do not mutate it in place.
    """
    path = Path(path)
    _FILTERNAMES_TXT = path / _FILTERNAMES_
    if not _FILTERNAMES_TXT.exists():
        return None
    stat = _FILTERNAMES_TXT.stat()
    return _load_alias_table_cached(
        _FILTERNAMES_TXT, stat.st_mtime_ns, stat.st_size
    )


@lru_cache(maxsize=8)
def _load_alias_table_cached(
    _FILTERNAMES_TXT: Path, _mtime_ns: int, _size: int
) -> pd.DataFrame:
    df = pd.read_csv(
        _FILTERNAMES_TXT, sep="\t", comment="#", skipinitialspace=True
    )
    # Columns are hand-aligned with literal spaces for readability, which
    # leaves stray leading/trailing whitespace in individual cells (e.g.
    # "TESS/TESS.Red     "); strip it so downstream lookups/comparisons
    # match cleanly.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def synthesize_mist_name(svo_id: str) -> str:
    """Derive a MIST-style BC-column name from an SVO filter ID when the
    alias table has no entry for it, e.g. "Keck/NIRC2.Kp" -> "NIRC2_Kp".

    Used consistently by both the BC-table generator (make_bc.py, which
    writes columns under this name) and the BC-grid loader (build_bc_grid
    below, which looks columns up by this name) so an arbitrary SVO
    filter with no alias-table row round-trips correctly.
    """
    return svo_id.split("/")[-1].replace(".", "_")


def resolve_filter_name(
    user_name: str,
    alias_df: pd.DataFrame | None,
    alias: Literal["MIST", "SVO"],
) -> str:
    """
    Translate a user-facing filter label (e.g. "2MASS.J", "Gaia.G")
    into the corresponding MIST/SVO filter label.

    Examples:
        "2MASS.J" --> "2MASS_J" | "2MASS/2MASS.J"
        "Gaia.G" --> "Gaia_G_DR2Rev" | "GAIA/GAIA2r.G"

    If the alias table is missing or doesn't know the name: for alias
    'SVO', assume the user has already provided the SVO ID and return it
    unchanged; for alias 'MIST', synthesize a column name (see
    synthesize_mist_name) if the input looks like an SVO ID (has a "/"),
    else assume it's already a bare column name and return it unchanged.

    A label listed in AMBIGUOUS_FILTER_ALIASES (a bare "I" or "R", which
    name two rows each) is resolved through that map instead of by table
    order -- see the comment on the map.
    """

    def _mist_fallback():
        return (
            synthesize_mist_name(user_name) if "/" in user_name else user_name
        )

    if alias_df is None:
        return _mist_fallback() if alias == "MIST" else user_name

    # Ambiguous bare labels: pick the row by name, never by file order.
    disambiguated = AMBIGUOUS_FILTER_ALIASES.get(user_name)
    if disambiguated is not None:
        try:
            row = alias_df[alias_df["MIST"] == disambiguated]
            if len(row):
                return str(row[alias].values[0])
        except Exception:
            pass

    try:
        rename = alias_df[alias_df.eq(user_name).any(axis=1)][alias].values[0]
        if rename in ("Unsupported", None) or pd.isna(rename):
            # No alias column — fall back so the caller gets a clear
            # KeyError later instead of a silent mismatch.
            return _mist_fallback() if alias == "MIST" else user_name
        return str(rename)
    except Exception:
        return _mist_fallback() if alias == "MIST" else user_name


def facility_from_svo_name(svo_name: str) -> str:
    """
    Infer the facility subdirectory from an SVO column name.

    Examples:
        "2MASS/2MASS.J"    -> "2MASS"
        "GAIA/GAIA2r.G"    -> "GAIA"
        "WISE/WISE.W1"     -> "WISE"

    The mapping is a simple prefix lookup; extend as more facilities
    are added to the tree.
    """
    prefix = svo_name.split("/")[0]
    return prefix


# -------------------------------------------------------------------
# BC table I/O (parquet)
# -------------------------------------------------------------------

# Default root; callers should override via the SED config when not
# running out of the project directory.
DEFAULT_MODEL_ROOT = source_code_dir / "models"

# Stellar-parameter columns every BC table carries; every other column
# is a filter column. GRID_KEY_COLS are the ones that locate a row on the
# (teff, logg, feh, Av) grid -- alpha and Rv ride along as provenance.
BC_PARAM_COLS = ["teff", "logg", "feh", "alpha", "Av", "Rv"]
GRID_KEY_COLS = ["teff", "logg", "feh", "Av"]

# Bumped from the text tables' "version 1" header field.
BC_TABLE_VERSION = 2

_BC_TABLE_SUFFIX = ".bc.parquet"


def bc_table_path(model_root: Path | str, model: str, facility: str) -> Path:
    """Path to the BC table for one (model, facility)."""
    return Path(model_root) / model / "BCs" / f"{facility}{_BC_TABLE_SUFFIX}"


def bc_filter_columns(df: pd.DataFrame) -> List[str]:
    """The filter (BC) columns of a BC table, in file order."""
    return [c for c in df.columns if c not in BC_PARAM_COLS]


def read_bc_table(
    path: Path | str, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read a BC table (see write_bc_table for the format).

    `columns` restricts the read to those columns -- the grid-key
    columns alone are a cheap read, which is what peek_grid_axes uses.
    df.attrs["meta"] holds the table metadata either way.
    """
    return pd.read_parquet(
        path,
        engine="pyarrow",
        columns=None if columns is None else list(columns),
    )


def find_bc_table(model_root: Path | str, model: str, facility: str) -> Path:
    """
    Locate the BC table for one facility, raising the same two errors the
    SED (and build_bc_grid's auto-generation) key on:

      FileNotFoundError    -- no BC tree at all for `model`
      NotImplementedError  -- the model exists but `facility` has no table
    """
    model_dir = Path(model_root) / model / "BCs"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Bolometric corrections not calculated for ``{model}`` model. Specify a different model."
        )
    path = bc_table_path(model_root, model, facility)
    if not path.is_file():
        raise NotImplementedError(
            f"Bolometric corrections not calculated for ``{facility}``. Specify a different filter set.\n Future implementation will automate this step."
        )
    return path


def write_bc_table(
    df_new: pd.DataFrame,
    path: Path | str,
    filter_meta: Dict[str, Dict],
    table_meta: Dict | None = None,
) -> pd.DataFrame:
    """
    Write (or merge into) a BC table.

    Parameters
    ----------
    df_new : DataFrame
        Columns BC_PARAM_COLS plus one column per filter (MIST BC-column
        names), one row per (teff, logg, feh, Av) grid node.
    path : Path
        Destination, normally bc_table_path(model_root, model, facility).
    filter_meta : dict
        {filter_column: {...}} provenance for every filter column in
        df_new -- at least "svo_id"; the generators also record the
        zeropoint, the flux weighting and the spectra used, so a table
        whose columns come from different pipelines says so per column.
    table_meta : dict, optional
        Table-level metadata (model, facility, mag_system, ...).

    If `path` already exists, its filter columns that df_new does NOT
    carry are kept unchanged (joined on GRID_KEY_COLS), together with
    their filter_meta; columns df_new does carry are replaced. A grid
    mismatch between the two raises rather than writing NaNs.

    Returns the DataFrame written.
    """
    path = Path(path)
    new_cols = bc_filter_columns(df_new)
    missing_meta = [c for c in new_cols if c not in filter_meta]
    if missing_meta:
        raise ValueError(f"No filter_meta given for columns {missing_meta}.")

    meta = {"version": BC_TABLE_VERSION, **(table_meta or {})}
    filters_meta = {c: filter_meta[c] for c in new_cols}

    df_out = df_new[BC_PARAM_COLS + new_cols]
    if path.exists():
        # Merge into the existing table WITHOUT touching its other
        # columns (they may come from a different pipeline, e.g. the
        # full-resolution generator vs make_bc's downsampled spectra).
        df_old = read_bc_table(path)
        old_meta = df_old.attrs.get("meta", {})
        keep_old_cols = [
            c for c in bc_filter_columns(df_old) if c not in new_cols
        ]
        if keep_old_cols:
            df_out = df_out.merge(
                df_old[GRID_KEY_COLS + keep_old_cols],
                on=GRID_KEY_COLS,
                how="left",
            )
            if df_out[keep_old_cols].isna().any().any():
                raise ValueError(
                    f"Grid-axis mismatch while merging new BC "
                    f"columns into existing {path}."
                )
            old_filters_meta = old_meta.get("filters", {})
            filters_meta = {
                **{c: old_filters_meta.get(c, {}) for c in keep_old_cols},
                **filters_meta,
            }
            df_out = df_out[BC_PARAM_COLS + keep_old_cols + new_cols]
        meta = {**old_meta, **meta}

    df_out = df_out.sort_values(GRID_KEY_COLS).reset_index(drop=True)
    meta["filters"] = filters_meta
    df_out.attrs = {"meta": meta}

    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(path, compression="snappy", index=False)
    return df_out


def peek_grid_axes(
    model: str = "NextGen",
    model_root: Path | str = DEFAULT_MODEL_ROOT,
) -> Dict[str, np.ndarray]:
    """
    Cheap axis-metadata reader for the BC grid.

    Used by the SED component to coordinate star-parameter bounds
    with the grid extent BEFORE the full grid is loaded (i.e. during
    SED.__init__, when star.build_parameters hasn't run yet).

    Assumes the grid's axes are identical across facilities
    (build_bc_grid checks this), so only the grid-key columns of one
    table are read.

    Parameters
    ----------
    model : str
        BC model name (selects the first-level subdirectory of model_root).
    model_root : Path
        Root directory holding the {model}/BCs/{FACILITY}.bc.parquet tree.

    Returns
    -------
    dict with keys:
        teff_pts  : np.ndarray, shape (n_teff,)
        logg_pts  : np.ndarray, shape (n_logg,)
        feh_pts   : np.ndarray, shape (n_feh,)
        av_pts    : np.ndarray, shape (n_av,)
    """
    model_root = Path(model_root)
    model_dir = model_root / model / "BCs"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"BC model directory not found: {model_dir}")

    # We don't care which facility; axes are identical across.
    tables = sorted(model_dir.glob(f"*{_BC_TABLE_SUFFIX}"))
    if not tables:
        raise FileNotFoundError(
            f"No *{_BC_TABLE_SUFFIX} BC tables found in {model_dir}"
        )

    df = read_bc_table(tables[0], columns=GRID_KEY_COLS)
    return _grid_axes(df)


def _grid_axes(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "teff_pts": np.sort(df["teff"].unique()).astype(float),
        "logg_pts": np.sort(df["logg"].unique()).astype(float),
        "feh_pts": np.sort(df["feh"].unique()).astype(float),
        "av_pts": np.sort(df["Av"].unique()).astype(float),
    }


# -------------------------------------------------------------------
# Grid assembly
# -------------------------------------------------------------------


def build_bc_grid(
    user_filter_names: Sequence[str],
    model: str = "NextGen",
    model_root: Path | str = DEFAULT_MODEL_ROOT,
) -> Dict:
    """
    Assemble a 4D BC grid for a specific set of filters.

    Parameters
    ----------
    user_filter_names : sequence of str
        Filter labels as they appear in the .sed file (VOID-style,
        e.g. "2MASS.J", "Gaia.G", "WISE.W1").
    model : str
        BC model name; selects the first-level subdirectory of model_root.
    model_root : Path
        Root directory holding the {model}/BCs/{FACILITY}.bc.parquet tree.

    Returns
    -------
    dict with keys:
        teff_pts   : np.ndarray, shape (n_teff,)
        logg_pts   : np.ndarray, shape (n_logg,)
        feh_pts    : np.ndarray, shape (n_feh,)
        av_pts     : np.ndarray, shape (n_av,)
        bc_values  : np.ndarray,
            shape (n_teff, n_logg, n_feh, n_av, n_filters)
        filter_order : list[str]
            MIST BC column names, in the same order as the requested
            user_filter_names.
    """
    model_root = Path(model_root)
    alias_df = _load_alias_table()

    # 1. Resolve user names -> MIST column names and group by facility.
    mist_names = [
        resolve_filter_name(n, alias_df, alias="MIST")
        for n in user_filter_names
    ]
    svo_names = [
        resolve_filter_name(n, alias_df, alias="SVO")
        for n in user_filter_names
    ]
    facilities = [facility_from_svo_name(s) for s in svo_names]
    by_facility: Dict[str, List[Tuple[int, str]]] = {}
    for idx, (fac, mist) in enumerate(zip(facilities, mist_names)):
        by_facility.setdefault(fac, []).append((idx, mist))

    # 2. For each facility, read its table, keeping only the requested
    # columns. Missing facilities/columns trigger one-time
    # auto-generation from the model spectra (make_bc.py).
    per_facility_frames: Dict[str, pd.DataFrame] = {}
    for fac, items in by_facility.items():
        fac_svo = [svo_names[idx] for idx, _ in items]
        wanted_cols = [mist for _, mist in items]

        try:
            path = find_bc_table(model_root, model, fac)
        except (FileNotFoundError, NotImplementedError):
            from .make_bc import generate_missing_facility

            if not generate_missing_facility(fac, fac_svo, model, model_root):
                raise
            path = find_bc_table(model_root, model, fac)

        df = read_bc_table(path)
        missing = set(wanted_cols) - set(bc_filter_columns(df))
        if missing:
            # Facility exists but lacks some requested columns; generate
            # the missing ones (make_bc merges into the existing table
            # without touching the existing columns).
            from .make_bc import generate_missing_facility

            miss_svo = [
                svo_names[idx] for idx, mist in items if mist in missing
            ]
            if generate_missing_facility(fac, miss_svo, model, model_root):
                df = read_bc_table(path)
                missing = set(wanted_cols) - set(bc_filter_columns(df))
        if missing:
            raise NotImplementedError(
                f"Bolometric corrections unavailable for ``{sorted(missing)}`` "
                f"and auto-generation failed; see the log above, or run "
                f"scripts/make_bc_tables.py manually."
            )

        # Dedupe: the same MIST column may be requested by more than one
        # .sed row (e.g. two independent V-band measurements, or the same
        # filter used for both a blend row and a differential row).
        # df[keep] with a repeated name would return a 2-D slice for that
        # column, breaking the by-name lookup in step 4 below.
        unique_wanted_cols = list(dict.fromkeys(wanted_cols))
        per_facility_frames[fac] = df[GRID_KEY_COLS + unique_wanted_cols]

    # 3. Adopt the first facility's (teff, logg, feh, Av) axes as the
    # canonical grid, and require every other facility to be on exactly
    # that grid. Step 4 places rows into the canonical axes by
    # searchsorted, so a facility on a different grid would otherwise be
    # silently mis-binned (its row lands in the next canonical cell up)
    # rather than rejected -- step 5's NaN check only catches an
    # UNDER-populated grid, not a mis-binned one.
    canonical_fac = next(iter(per_facility_frames))
    axes = _grid_axes(per_facility_frames[canonical_fac])
    for fac, df in per_facility_frames.items():
        fac_axes = _grid_axes(df)
        for key, pts in axes.items():
            if not np.array_equal(fac_axes[key], pts):
                raise ValueError(
                    f"BC table for facility '{fac}' is on a different "
                    f"{key[:-4]} grid than '{canonical_fac}': "
                    f"{fac_axes[key]} vs {pts}. Regenerate it on the "
                    f"{model}.grid.yaml axes."
                )
    teff_pts = axes["teff_pts"]
    logg_pts = axes["logg_pts"]
    feh_pts = axes["feh_pts"]
    av_pts = axes["av_pts"]

    n_teff, n_logg, n_feh, n_av = (
        len(teff_pts),
        len(logg_pts),
        len(feh_pts),
        len(av_pts),
    )
    n_filters = len(user_filter_names)

    # 4. Allocate the full (teff, logg, feh, Av, filters) array and
    # fill it. Using searchsorted gives O(N) indexing per row.
    bc_values = np.full(
        (n_teff, n_logg, n_feh, n_av, n_filters), np.nan, dtype=float
    )

    for fac, df in per_facility_frames.items():
        items = by_facility[fac]  # [(global_idx, mist_name), ...]
        t_idx = np.searchsorted(teff_pts, df["teff"].values)
        g_idx = np.searchsorted(logg_pts, df["logg"].values)
        f_idx = np.searchsorted(feh_pts, df["feh"].values)
        a_idx = np.searchsorted(av_pts, df["Av"].values)
        for filter_idx, mist_name in items:
            bc_values[t_idx, g_idx, f_idx, a_idx, filter_idx] = df[
                mist_name
            ].values

    # 5. Sanity check for gaps. If there are NaNs, the grid is ragged
    # and the interpolator will propagate them; better to raise now.
    if np.any(np.isnan(bc_values)):
        n_bad = int(np.isnan(bc_values).sum())
        raise ValueError(
            f"BC grid has {n_bad} missing entries after assembly. "
            "The (teff, logg, feh, Av) grid is not fully populated "
            "for every requested filter."
        )

    return {
        "teff_pts": teff_pts,
        "logg_pts": logg_pts,
        "feh_pts": feh_pts,
        "av_pts": av_pts,
        "bc_values": bc_values,
        "filter_order": mist_names,
    }


# -------------------------------------------------------------------
# Slicing BC Grid depending on user-specified bounds
# -------------------------------------------------------------------


def _range_indices(pts, lo, hi):
    """
    Return indices of all grid points needed to cover [lo, hi], including
    the bracketing points outside the bounds when lo/hi fall between grid pts.
    """
    pts = np.asarray(pts)
    n = len(pts)

    if lo is None:
        i_lo = 0
    else:
        i_lo = int(np.searchsorted(pts, lo, side="left"))
        # If lo lands exactly on a grid point, i_lo is already correct.
        # If lo falls between pts[i_lo-1] and pts[i_lo], we need pts[i_lo-1]
        # to bracket lo from below.
        if i_lo > 0 and pts[i_lo] > lo:
            i_lo -= 1

    if hi is None:
        i_hi = n - 1
    else:
        i_hi = int(np.searchsorted(pts, hi, side="right")) - 1
        # Symmetric: if hi falls between pts[i_hi] and pts[i_hi+1],
        # we need pts[i_hi+1] to bracket hi from above.
        if i_hi < n - 1 and pts[i_hi] < hi:
            i_hi += 1

    return np.arange(i_lo, i_hi + 1)


def _create_AXES(grid_yaml):
    grid = grid_yaml.get("grid")
    AXES = {}
    for i, axis in enumerate(grid):
        AXES[axis] = (np.array(grid[axis]), i)
    return AXES


def slice_bc(grid_dict, bc_values, **bounds):
    """
    Slice bc_values along any combination of its four grid axes.

    Parameters
    ----------
    grid_dict : dictionary with keys (``model``, ``grid``) and within ``grid``, names and values of axes
        Example yaml file that can be used to create grid_dict:
            - ``NextGen.grid.yaml``  in components.sed
            - ``MISTv1.2.grid.yaml`` in components.sed
    bc_values : np.ndarray, shape (len(grid_dict.get("grid")[axis]), ... , nfilters)
        Example:
            # len(teff)=60, len(logg)=11, len(feh)=11, len(av)=13, nfilters=9
            bc_values.shape = (60, 11, 11, 13, 9)
    **bounds : keyword arguments of the form
        param=value          # nearest single point
        param=(lo, hi)       # inclusive range [lo, hi]
        param=(None, hi)     # open lower bound  (≤ hi)
        param=(lo, None)     # open upper bound  (≥ lo)

    Returns
    -------
    sliced : np.ndarray
        Sub-array with the same number of dimensions (singleton axes are
        kept so the caller always knows which axis is which).
    selected : dict
        Maps each constrained parameter name to the grid points that were
        selected, for easy inspection.

    Examples
    --------
    sliced, info = slice_bc(bc_values, av=(None, 0.27), logg=(4.35, 4.9))
    sliced, info = slice_bc(bc_values, teff=(5000, 6000), feh=(-1.0, 0.0))
    sliced, info = slice_bc(bc_values, teff=5800)          # nearest point
    """
    idx = [slice(None)] * (bc_values.ndim - 1)  # one entry per grid axis
    selected = {}

    AXES = _create_AXES(grid_dict)

    for param, bound in bounds.items():
        if param not in AXES:
            raise ValueError(
                f"Unknown parameter {param!r}. Choose from {list(AXES)}."
            )
        pts, axis = AXES[param]

        # ---- single value: find nearest grid points --------------------
        if not isinstance(bound, (tuple, list)):
            nearest_idx = int(np.argmin(np.abs(pts - bound)))
            idx[axis] = np.array([nearest_idx])  # keep axis with length 1
            selected[param] = pts[nearest_idx : nearest_idx + 1]
            continue

        # ---- (lo, hi) range ----------------------------------------------
        lo, hi = bound
        chosen = _range_indices(pts, lo, hi)

        if chosen.size == 0:
            raise ValueError(
                f"No {param!r} grid points found in range "
                f"[{lo}, {hi}]. Grid spans {pts[0]} – {pts[-1]}."
            )
        idx[axis] = chosen
        selected[param] = pts[chosen]

    # np.ix_ lets us index multiple axes simultaneously with fancy indexing.
    # Build the full cross-product index, keeping the filter axis intact.
    grid_idx = np.ix_(
        *[
            (
                idx[ax]
                if isinstance(idx[ax], np.ndarray)
                else np.arange(bc_values.shape[ax])
            )
            for ax in range(bc_values.ndim - 1)
        ]
    )
    # Append a full slice for the filter axis
    full_idx = grid_idx + (slice(None),)

    return bc_values[full_idx], selected


# -------------------------------------------------------------------
# Pytensor interpolator (reused from Code for Models/Classes/Grid.py)
# -------------------------------------------------------------------


class RegularGridInterpolator:
    """
    Linear N-D interpolation on a regular grid, pytensor-compatible.
    Spacing may be uneven in any dimension, as long as the grid is filled.

    The values array may carry trailing "output" axes
    (e.g. n_filters) that ride along with the interpolation.

    Parameters
    ----------
    points : sequence of 1-D arrays with shapes ``(m1,), ... (mn,)``
        Grid points along each interpolated dimension.
    values : array, shape (m1, ..., m_ndim, ..., nout)
        Tabulated values; the first ndim axes must match `points`.
    fill_value : float, optional
        Value used when coords are outside the grid. None leaves
        extrapolation to the caller.

        No caller in the tree passes it, and that is a DECISION, not an
        oversight (review 5.9.2): the SED builds this with fill_value=None
        deliberately, so an off-grid draw is linearly extrapolated off the
        edge cell rather than replaced. The restoring force is the measured
        soft barrier on the one reachably off-grid axis, star.loggsed --
        see components/sed/sed.md, which spells out why a wall (-inf, NaN,
        or a constant) would be strictly worse: NUTS has nothing to follow
        across one. The parameter stays as the seam for a grid whose
        off-grid answer really is a constant; do not delete it, and do not
        wire the SED to it without reading that ruling first.
    """

    def __init__(self, points, values, fill_value=None):
        self.ndim = len(points)
        self.points = [pt.as_tensor_variable(p) for p in points]
        self.values = pt.as_tensor_variable(values)
        self.fill_value = fill_value

    def regular_grid_interp(self, coords):
        """
        Perform a linear interpolation in N-dimensions on a regular grid.
        Works within a PyMC model where coords is a stacked tensor of random variables
        that may have shape=1 or shape=N.

        Args:
            coords: A tensor of shape (ntest, ndim) or (ndim,)
                Example:
                    coords = pt.stack([teff_coord, logg_coord, feh_coord, Av_coord], axis=-1)
        """
        coords = pt.atleast_2d(coords)  # (N, ndim)
        n_points = coords.shape[0]

        indices = []
        norm_distances = []
        out_of_bounds = pt.zeros((n_points,), dtype=bool)

        for n, grid in enumerate(self.points):
            grid = pt.as_tensor_variable(grid)
            x = coords[:, n]
            i = pt.extra_ops.searchsorted(grid, x) - 1
            oob = pt.or_(pt.lt(i, 0), pt.ge(i, grid.shape[0] - 1))
            out_of_bounds = pt.or_(out_of_bounds, oob)
            i = pt.clip(i, 0, grid.shape[0] - 2)
            norm_dist = (x - grid[i]) / (grid[i + 1] - grid[i])
            indices.append(i)
            norm_distances.append(norm_dist)

        values = pt.as_tensor_variable(self.values)
        if values.ndim > self.ndim:
            nout = self.values.shape[self.ndim]
            result = pt.zeros((n_points, nout))
        else:
            result = pt.zeros((n_points,))

        for edge_indices in itertools.product(*((i, i + 1) for i in indices)):
            weight = pt.ones((n_points,))
            for ei, idx, yi in zip(edge_indices, indices, norm_distances):
                w = pt.where(pt.eq(ei, idx), 1.0 - yi, yi)
                weight = weight * w

            corner_vals = values[edge_indices]
            if values.ndim > self.ndim:
                result = result + corner_vals * weight[:, None]
            else:
                result = result + corner_vals * weight

        if self.fill_value is not None:
            if values.ndim > self.ndim:
                oob_bc = out_of_bounds[:, None]
            else:
                oob_bc = out_of_bounds
            result = pt.switch(oob_bc, self.fill_value, result)

        return result

    def evaluate(self, coords):
        """
        Interpolate the data

        Args:
            coords: A tensor defining the coordinates where the interpolation
                should be evaluated. This must have the shape
                ``(ntest, ndim)`` or ``(ndim,)``.
                Example:
                    coords = pt.stack([teff_coord, logg_coord, feh_coord, Av_coord], axis=-1)
        """
        return self.regular_grid_interp(coords)
