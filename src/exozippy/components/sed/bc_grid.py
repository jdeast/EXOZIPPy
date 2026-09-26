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

Tables grow incrementally. A NaN cell means "not computed yet", so a
table may be extended along any axis (new Av values, say) for some of
its filters before the others; write_bc_table merges cell by cell and
bc_nodes_to_compute tells a generator which (node, filter) cells are
still missing, so nothing already on disk is recomputed. Readers only
ever use the complete part of the columns they ask for (complete_axes).

Reads are selective: build_bc_grid reads only the requested filter
columns, and only the rows inside the fit's parameter bounds (plus the
bracketing grid point on each side), pushing both down to parquet.

Grid assumptions in v1:
  * single Rv slice (Rv = 3.10) across all tables
  * facilities may differ in EXTENT along an axis, but inside the span
    they share their (teff, logg, feh, Av) points agree (checked in
    build_bc_grid and peek_grid_axes)
  * the grid assembled from these tables has axes (teff, logg, feh, av)

These assumptions hold for the NextGen tree as it currently ships.
When MIST is added, this loader grows a `model` dispatch so each
family can parse its own layout; the interpolator interface stays
the same.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytensor.tensor as pt

logger = logging.getLogger(__name__)

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

# Axis name as it appears in {model}.grid.yaml and in the `bounds` /
# `axes` dicts below -> the table column holding it.
AXIS_COLUMNS = {"teff": "teff", "logg": "logg", "feh": "feh", "av": "Av"}

# Grid keys are rounded to this many decimals before any node is matched
# (merge, coverage, pushdown), so a node written from a yaml literal and
# the same node parsed from a legacy text table are the same node.
_KEY_DECIMALS = 6

# Row order on disk. Av is outermost because it is the axis a fit most
# often truncates (an Av upper limit) and the axis the grid is extended
# along: with each Av in its own run of row groups, a pushdown filter on
# Av skips the row groups past the limit instead of decoding them. The
# loader places rows by value, so nothing depends on this order.
BC_SORT_COLS = ["Av", "feh", "logg", "teff"]
_ROW_GROUP_SIZE = 8192

# Bumped from the text tables' "version 1" header field.
BC_TABLE_VERSION = 2

_BC_TABLE_SUFFIX = ".bc.parquet"


def bc_table_path(model_root: Path | str, model: str, facility: str) -> Path:
    """Path to the BC table for one (model, facility)."""
    return Path(model_root) / model / "BCs" / f"{facility}{_BC_TABLE_SUFFIX}"


def bc_filter_columns(df: pd.DataFrame) -> List[str]:
    """The filter (BC) columns of a BC table, in file order."""
    return [c for c in df.columns if c not in BC_PARAM_COLS]


def bc_table_filter_columns(path: Path | str) -> List[str]:
    """The filter columns of a BC table on disk, read from its schema only
    (no data is read)."""
    return [n for n in pq.read_schema(path).names if n not in BC_PARAM_COLS]


def read_bc_meta(path: Path | str) -> Dict:
    """df.attrs["meta"] of a BC table on disk, read from its schema only.

    pandas stores df.attrs as JSON under the PANDAS_ATTRS key of the
    parquet schema metadata; reading the footer is enough to get it.
    """
    raw = (pq.read_schema(path).metadata or {}).get(b"PANDAS_ATTRS")
    if not raw:
        return {}
    return json.loads(raw).get("meta", {})


def _canonical_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Round the grid-key columns (see _KEY_DECIMALS); returns a copy."""
    df = df.copy()
    keys = [c for c in GRID_KEY_COLS if c in df.columns]
    df[keys] = df[keys].astype(float).round(_KEY_DECIMALS)
    return df


def _pushdown_filters(where: Dict[str, Tuple[float | None, float | None]]):
    """{axis: (lo, hi)} -> a pyarrow filter list (inclusive, with a
    rounding tolerance so a bound that IS a grid point keeps that point)."""
    tol = 0.5 * 10.0**-_KEY_DECIMALS
    filters = []
    for axis, (lo, hi) in (where or {}).items():
        col = AXIS_COLUMNS.get(axis, axis)
        if lo is not None:
            filters.append((col, ">=", float(lo) - tol))
        if hi is not None:
            filters.append((col, "<=", float(hi) + tol))
    return filters or None


def read_bc_table(
    path: Path | str,
    columns: Sequence[str] | None = None,
    where: Dict[str, Tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Read a BC table (see write_bc_table for the format).

    `columns` restricts the read to those columns: parquet is columnar, so
    the other filter columns are never read from disk. The grid-key
    columns alone are a cheap read, which is what peek_grid_axes uses.

    `where` = {axis: (lo, hi)} (axis names as in AXIS_COLUMNS, inclusive,
    None for an open end) is pushed down to the parquet reader: row groups
    whose statistics fall outside the range are skipped and the remaining
    rows are filtered before they reach pandas.

    df.attrs["meta"] holds the table metadata either way.
    """
    return pd.read_parquet(
        path,
        engine="pyarrow",
        columns=None if columns is None else list(columns),
        filters=_pushdown_filters(where),
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
    allow_new_nodes: bool = False,
    replace_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Write (or merge into) a BC table.

    Parameters
    ----------
    df_new : DataFrame
        Columns BC_PARAM_COLS plus one column per filter (MIST BC-column
        names), one row per (teff, logg, feh, Av) grid node. A NaN cell
        means "not computed here" and never overwrites an existing value,
        so a generator can hand over only the (node, filter) cells it
        actually computed.
    path : Path
        Destination, normally bc_table_path(model_root, model, facility).
    filter_meta : dict
        {filter_column: {...}} provenance for every filter column in
        df_new -- at least "svo_id"; the generators also record the
        zeropoint, the flux weighting and the spectra used, so a table
        whose columns come from different pipelines says so per column.
    table_meta : dict, optional
        Table-level metadata (model, facility, mag_system, ...).
    allow_new_nodes : bool
        Whether df_new may carry grid nodes the existing table does not
        (e.g. new Av values after NextGen.grid.yaml was extended). The
        generators, which build their nodes from the grid yaml, pass True.
        Off by default so a table computed on an accidentally different
        grid raises instead of turning the file ragged.
    replace_columns : sequence of str
        Columns of df_new whose EXISTING values are discarded before the
        merge (and whose metadata is replaced), for a column being
        recomputed from scratch by a different pipeline. A generator that
        checkpoints part-way through such a column must pass it on its
        first write, or the unfinished part would keep the old pipeline's
        values under the new pipeline's metadata.

    If `path` already exists the merge is cell-wise: every existing value
    is kept unless df_new carries a non-NaN value for that (node, filter)
    cell. Filter columns df_new does not carry are untouched, together
    with their filter_meta. A column that df_new REPLACES in full takes
    df_new's filter_meta; a column it only extends (new nodes, or cells
    that were missing) keeps its old metadata, updated by df_new's.

    A node that some filter column has no value at is stored as NaN; the
    loader only ever reads the complete part of a column (complete_axes),
    and the generators treat NaN as "still to compute".

    Returns the DataFrame written.
    """
    path = Path(path)
    new_cols = bc_filter_columns(df_new)
    missing_meta = [c for c in new_cols if c not in filter_meta]
    if missing_meta:
        raise ValueError(f"No filter_meta given for columns {missing_meta}.")

    meta = {"version": BC_TABLE_VERSION, **(table_meta or {})}
    filters_meta = {c: filter_meta[c] for c in new_cols}

    new = _canonical_keys(df_new[BC_PARAM_COLS + new_cols])
    if new.duplicated(GRID_KEY_COLS).any():
        raise ValueError("df_new has more than one row for a grid node.")
    new = new.set_index(GRID_KEY_COLS)

    if path.exists():
        # Merge into the existing table WITHOUT touching the cells df_new
        # does not compute (they may come from a different pipeline, e.g.
        # the full-resolution generator vs make_bc's downsampled spectra).
        df_old = _canonical_keys(read_bc_table(path))
        old_meta = df_old.attrs.get("meta", {})
        old_filters_meta = old_meta.get("filters", {})
        old = df_old.set_index(GRID_KEY_COLS)
        for c in replace_columns:
            if c in old.columns:
                old[c] = np.nan
                old_filters_meta.pop(c, None)

        added = new.index.difference(old.index)
        if len(added) and not allow_new_nodes:
            raise ValueError(
                f"Grid-axis mismatch while merging new BC columns into "
                f"existing {path}: {len(added)} node(s) of the new columns "
                f"are not on the existing grid, e.g. "
                f"{dict(zip(GRID_KEY_COLS, added[0]))}. Pass "
                f"allow_new_nodes=True if the grid is being extended."
            )

        out = old.reindex(old.index.union(new.index))
        for c in ["alpha", "Rv"] + new_cols:
            incoming = new[c].reindex(out.index)
            if c in out.columns:
                if c in new_cols:
                    covered = incoming.reindex(old.index).notna()
                    replaced = (covered | old[c].isna()).all()
                    if not replaced:
                        filters_meta[c] = {
                            **old_filters_meta.get(c, {}),
                            **filters_meta[c],
                        }
                out[c] = incoming.combine_first(out[c])
            else:
                out[c] = incoming

        old_cols = bc_filter_columns(df_old)
        keep_old_cols = [c for c in old_cols if c not in new_cols]
        filters_meta = {
            **{c: old_filters_meta.get(c, {}) for c in keep_old_cols},
            **filters_meta,
        }
        order = old_cols + [c for c in new_cols if c not in old_cols]
        df_out = out.reset_index()[BC_PARAM_COLS + order]
        meta = {**old_meta, **meta}
    else:
        df_out = new.reset_index()[BC_PARAM_COLS + new_cols]

    df_out = df_out.sort_values(BC_SORT_COLS).reset_index(drop=True)
    meta["filters"] = {c: filters_meta[c] for c in bc_filter_columns(df_out)}
    df_out.attrs = {"meta": meta}

    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(
        path,
        compression="snappy",
        index=False,
        row_group_size=_ROW_GROUP_SIZE,
    )
    return df_out


# -------------------------------------------------------------------
# Coverage: which part of the grid a set of filter columns fills
# -------------------------------------------------------------------


def _axes_from_keys(keys: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        f"{axis}_pts": np.sort(keys[col].unique()).astype(float)
        for axis, col in AXIS_COLUMNS.items()
    }


def complete_axes(
    df: pd.DataFrame, columns: Sequence[str]
) -> Dict[str, np.ndarray]:
    """
    The largest box of grid nodes on which EVERY one of `columns` has a
    value -- the part of a BC table a regular-grid interpolator can use.

    A table is ragged when it has been extended for some filters but not
    yet for others (new Av values computed for 2MASS_J only, say) or when
    a generator run was interrupted. The nodes where all `columns` are
    non-NaN are gathered per axis; if their cartesian product is not all
    present, edge slabs are trimmed, most-incomplete first, until it is.
    Only EDGE slabs are ever dropped, so the result is always a contiguous
    run of each axis's points, and never below two points (an axis
    collapsed to one point cannot be interpolated along). A hole that no
    such trimming removes raises, because no box around it is honest.
    Trimming that does happen is warned about: it means some filter's
    column has gaps that its generator should fill.

    Returns {teff_pts, logg_pts, feh_pts, av_pts}.
    """
    df = _canonical_keys(df)
    columns = list(columns)
    present = (
        df[columns].notna().all(axis=1)
        if columns
        else pd.Series(True, index=df.index)
    )
    keys = df.loc[present, GRID_KEY_COLS]
    if keys.empty:
        raise ValueError(f"BC columns {columns} hold no values at all.")

    axes = _axes_from_keys(keys)
    names = list(axes)
    pts = [axes[n] for n in names]
    cube = np.zeros(tuple(len(p) for p in pts), dtype=bool)
    idx = tuple(
        np.searchsorted(p, keys[col].values)
        for p, col in zip(pts, AXIS_COLUMNS.values())
    )
    cube[idx] = True

    lo = [0] * len(pts)
    hi = [len(p) for p in pts]
    while True:
        sub = cube[tuple(slice(a, b) for a, b in zip(lo, hi))]
        if sub.all():
            break
        best = None
        for ax in range(sub.ndim):
            if sub.shape[ax] <= 2:
                continue
            for end, pos in ((0, 0), (1, sub.shape[ax] - 1)):
                frac = 1.0 - np.take(sub, pos, axis=ax).mean()
                if frac > 0 and (best is None or frac > best[0]):
                    best = (frac, ax, end)
        if best is None:
            holes = np.argwhere(~sub)[:3]
            example = [
                {
                    n: float(pts[k][lo[k] + i])
                    for k, (n, i) in enumerate(zip(names, h))
                }
                for h in holes
            ]
            raise ValueError(
                f"BC columns {columns} have {int((~sub).sum())} missing "
                f"node(s) in the interior of their grid (no edge slab can "
                f"be dropped around them), e.g. {example}; "
                f"regenerate them (generate_NextGen_BC_Tables.py step 2 "
                f"computes only the missing nodes)."
            )
        _, ax, end = best
        if end == 0:
            lo[ax] += 1
        else:
            hi[ax] -= 1

    box = {n: p[a:b] for n, p, a, b in zip(names, pts, lo, hi)}
    trimmed = {
        n[:-4]: (p[0], p[-1])
        for n, p in axes.items()
        if len(box[n]) < len(p)
    }
    if trimmed:
        logger.warning(
            f"BC columns {columns} are only partly computed; using the "
            f"complete part of the grid, which drops the edge(s) of "
            f"{trimmed} -> "
            f"{ {n[:-4]: (box[n][0], box[n][-1]) for n in box} }. Re-run "
            f"the table generator to fill the gaps (it computes only the "
            f"missing nodes)."
        )
    return box


def _merge_axes(
    a: Dict[str, np.ndarray],
    b: Dict[str, np.ndarray],
    a_name: str,
    b_name: str,
    model: str,
) -> Dict[str, np.ndarray]:
    """
    Intersect two complete boxes. They may have different EXTENTS (one
    facility extended to Av = 15, another still at 6), but inside the
    span they share, their points must agree -- otherwise the two tables
    are on different grids and placing one's rows on the other's axes
    would silently mis-bin them.
    """
    out = {}
    for key in a:
        pa, pb = a[key], b[key]
        lo, hi = max(pa[0], pb[0]), min(pa[-1], pb[-1])
        ina = pa[(pa >= lo) & (pa <= hi)]
        inb = pb[(pb >= lo) & (pb <= hi)]
        if lo > hi or not np.array_equal(ina, inb):
            raise ValueError(
                f"BC table for facility '{b_name}' is on a different "
                f"{key[:-4]} grid than '{a_name}': {pb} vs {pa}. Regenerate "
                f"it on the {model}.grid.yaml axes."
            )
        out[key] = ina
    return out


def _filters_by_facility(
    user_filter_names: Sequence[str],
) -> Tuple[List[str], List[str], Dict[str, List[Tuple[int, str]]]]:
    """Resolve filter labels -> (MIST names, SVO names, {facility:
    [(index, MIST name), ...]})."""
    alias_df = _load_alias_table()
    mist_names = [
        resolve_filter_name(n, alias_df, alias="MIST")
        for n in user_filter_names
    ]
    svo_names = [
        resolve_filter_name(n, alias_df, alias="SVO")
        for n in user_filter_names
    ]
    by_facility: Dict[str, List[Tuple[int, str]]] = {}
    for idx, (svo, mist) in enumerate(zip(svo_names, mist_names)):
        by_facility.setdefault(facility_from_svo_name(svo), []).append(
            (idx, mist)
        )
    return mist_names, svo_names, by_facility


def peek_grid_axes(
    model: str = "NextGen",
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    filters: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Cheap axis-metadata reader for the BC grid.

    Used by the SED component to coordinate star-parameter bounds
    with the grid extent BEFORE the full grid is loaded (i.e. during
    SED.__init__, when star.build_parameters hasn't run yet).

    Returns the part of the grid the BCs actually COVER: the intersection
    of the complete boxes (complete_axes) of the requested filter columns,
    so a fit is never bounded onto nodes that one of its filters has not
    been computed at yet. Only the grid-key columns and the requested
    filter columns are read.

    Parameters
    ----------
    model : str
        BC model name (selects the first-level subdirectory of model_root).
    model_root : Path
        Root directory holding the {model}/BCs/{FACILITY}.bc.parquet tree.
    filters : sequence of str, optional
        Filter labels (any spelling resolve_filter_name accepts). Filters
        whose table or column does not exist yet are ignored (they will be
        generated on the grid yaml's axes). When None, or when none of
        them exists yet, every filter column of every table counts.

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

    tables = sorted(model_dir.glob(f"*{_BC_TABLE_SUFFIX}"))
    if not tables:
        raise FileNotFoundError(
            f"No *{_BC_TABLE_SUFFIX} BC tables found in {model_dir}"
        )

    wanted: Dict[Path, List[str]] = {}
    if filters:
        _, _, by_facility = _filters_by_facility(filters)
        for fac, items in by_facility.items():
            path = bc_table_path(model_root, model, fac)
            if not path.is_file():
                continue
            have = set(bc_table_filter_columns(path))
            cols = list(dict.fromkeys(m for _, m in items if m in have))
            if cols:
                wanted[path] = cols
    if not wanted:
        wanted = {p: bc_table_filter_columns(p) for p in tables}

    axes = None
    first = None
    for path, cols in wanted.items():
        df = read_bc_table(path, columns=GRID_KEY_COLS + cols)
        box = complete_axes(df, cols)
        if axes is None:
            axes, first = box, path.name
        else:
            axes = _merge_axes(axes, box, first, path.name, model)
    return axes


def _grid_axes(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return _axes_from_keys(_canonical_keys(df))


# -------------------------------------------------------------------
# Grid assembly
# -------------------------------------------------------------------


def _clean_bounds(bounds) -> Dict[str, Tuple[float | None, float | None]]:
    """Normalize {axis: (lo, hi)}: non-finite / None ends become open."""
    out = {}
    for axis, (lo, hi) in (bounds or {}).items():
        if axis not in AXIS_COLUMNS:
            raise ValueError(
                f"Unknown BC grid axis {axis!r}. Choose from "
                f"{list(AXIS_COLUMNS)}."
            )

        def _f(x):
            if x is None:
                return None
            x = float(x)
            return x if np.isfinite(x) else None

        out[axis] = (_f(lo), _f(hi))
    return out


# The star Parameter each axis is sampled (or derived) as, for messages.
_AXIS_PARAMS = {"teff": "teffsed", "logg": "loggsed", "feh": "feh", "av": "av"}


def _bracket(pts: np.ndarray, lo, hi) -> Tuple[float, float]:
    """The grid points that bracket [lo, hi] (see _range_indices)."""
    idx = _range_indices(pts, lo, hi)
    if idx.size == 0:
        raise ValueError(
            f"No grid points in range [{lo}, {hi}]; grid spans "
            f"{pts[0]} - {pts[-1]}."
        )
    return float(pts[idx[0]]), float(pts[idx[-1]])


def build_bc_grid(
    user_filter_names: Sequence[str],
    model: str = "NextGen",
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    bounds: Dict[str, Tuple[float | None, float | None]] | None = None,
) -> Dict:
    """
    Assemble a 4D BC grid for a specific set of filters, reading only what
    the fit needs.

    Parameters
    ----------
    user_filter_names : sequence of str
        Filter labels as they appear in the .sed file (VOID-style,
        e.g. "2MASS.J", "Gaia.G", "WISE.W1").
    model : str
        BC model name; selects the first-level subdirectory of model_root.
    model_root : Path
        Root directory holding the {model}/BCs/{FACILITY}.bc.parquet tree.
    bounds : dict, optional
        {axis: (lo, hi)} with axis in "teff", "logg", "feh", "av" (None or
        a non-finite value for an open end) -- the range the sampler can
        reach. Only the grid points needed to cover it are read: the points
        inside it plus the one bracketing it on each side, so an Av upper
        limit of 0.09 reads Av = 0, 0.05, 0.1. Unbounded axes are read in
        full.

    What is read: per facility, the grid-key columns (to find the
    bracketing points), then ONLY the requested filter columns, with the
    bounds pushed down to the parquet reader. Every other filter column
    and every row outside the bounds stays on disk.

    The grid is the complete box (complete_axes) of the requested columns
    inside the bounds, intersected across facilities; it must still cover
    the bracketed bounds on every bounded axis, or this raises naming the
    filters that have not been computed that far.

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
    bounds = _clean_bounds(bounds)

    # 1. Resolve user names -> MIST column names and group by facility.
    mist_names, svo_names, by_facility = _filters_by_facility(
        user_filter_names
    )

    # 2. For each facility, read its table, keeping only the requested
    # columns and rows. Missing facilities/columns trigger one-time
    # auto-generation from the model spectra (make_bc.py).
    per_facility_frames: Dict[str, pd.DataFrame] = {}
    per_facility_axes: Dict[str, Dict[str, np.ndarray]] = {}
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

        missing = set(wanted_cols) - set(bc_table_filter_columns(path))
        if missing:
            # Facility exists but lacks some requested columns; generate
            # the missing ones (make_bc merges into the existing table
            # without touching the existing columns).
            from .make_bc import generate_missing_facility

            miss_svo = [
                svo_names[idx] for idx, mist in items if mist in missing
            ]
            if generate_missing_facility(fac, miss_svo, model, model_root):
                missing = set(wanted_cols) - set(
                    bc_table_filter_columns(path)
                )
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

        # Bracket the bounds on this table's own points (a cheap read of
        # the key columns), then read only those rows of only these columns.
        where = {}
        if bounds:
            keys = _canonical_keys(read_bc_table(path, columns=GRID_KEY_COLS))
            for axis, (lo, hi) in bounds.items():
                pts = np.sort(keys[AXIS_COLUMNS[axis]].unique())
                where[axis] = _bracket(pts, lo, hi)
        df = _canonical_keys(
            read_bc_table(
                path, columns=GRID_KEY_COLS + unique_wanted_cols, where=where
            )
        )
        box = complete_axes(df, unique_wanted_cols)
        for axis, (lo, hi) in where.items():
            pts = box[f"{axis}_pts"]
            if pts[0] > lo or pts[-1] < hi:
                raise ValueError(
                    f"BC columns {unique_wanted_cols} of facility '{fac}' "
                    f"cover {axis} = {pts[0]} - {pts[-1]}, but the fit needs "
                    f"{lo} - {hi}. Add the missing {axis} values to "
                    f"{model}.grid.yaml and re-run the table generator (it "
                    f"computes only the missing nodes), or tighten the "
                    f"star.{_AXIS_PARAMS[axis]} bounds."
                )
        per_facility_frames[fac] = df
        per_facility_axes[fac] = box

    # 3. Intersect the facilities' complete boxes. Extents may differ
    # (one facility extended further than another); points inside the
    # shared span may not -- step 4 places rows by searchsorted, so a
    # facility on a different grid would otherwise be silently mis-binned.
    canonical_fac = next(iter(per_facility_axes))
    axes = per_facility_axes[canonical_fac]
    for fac, box in per_facility_axes.items():
        axes = _merge_axes(axes, box, canonical_fac, fac, model)
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
        inside = np.ones(len(df), dtype=bool)
        for key, col in zip(axes, AXIS_COLUMNS.values()):
            inside &= np.isin(df[col].values, axes[key])
        df = df[inside]
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
# Incremental generation: which (node, filter) cells still need computing
# -------------------------------------------------------------------


def grid_nodes(axes: Dict[str, Sequence[float]]) -> pd.DataFrame:
    """Every node of a (teff, logg, feh, av) grid, as GRID_KEY_COLS rows.

    `axes` is keyed like {model}.grid.yaml's `grid:` block ("teff",
    "logg", "feh", "av"); "*_pts" keys (peek_grid_axes) are accepted too.
    """
    pts = []
    for axis in AXIS_COLUMNS:
        vals = axes[axis] if axis in axes else axes[f"{axis}_pts"]
        pts.append(np.asarray(vals, dtype=float))
    mesh = np.meshgrid(*pts, indexing="ij")
    df = pd.DataFrame(
        {col: m.ravel() for col, m in zip(AXIS_COLUMNS.values(), mesh)}
    )
    return _canonical_keys(df)[GRID_KEY_COLS]


def bc_nodes_to_compute(
    path: Path | str,
    axes: Dict[str, Sequence[float]],
    columns: Sequence[str],
    reusable: Callable[[Dict], bool] | None = None,
) -> pd.DataFrame:
    """
    The work list for bringing `columns` of one BC table up to the grid
    `axes`: the target nodes at which at least one column still needs
    computing, with one boolean column per filter (True = compute it).

    A (node, filter) cell needs computing when the table does not exist,
    the column does not exist, the node is not in the table, or the cell
    is NaN. `reusable(filter_meta)` can additionally reject a whole
    existing column (e.g. one written by a different pipeline, which a
    full-resolution run should replace rather than extend); by default
    every existing value is kept.

    Only the key columns and the existing requested columns are read.
    """
    columns = list(dict.fromkeys(columns))
    target = grid_nodes(axes)
    need = pd.DataFrame(True, index=target.index, columns=columns)

    path = Path(path)
    if path.is_file() and columns:
        have = [c for c in columns if c in bc_table_filter_columns(path)]
        if reusable is not None:
            fmeta = read_bc_meta(path).get("filters", {})
            have = [c for c in have if reusable(fmeta.get(c, {}))]
        if have:
            where = {
                axis: (float(target[col].min()), float(target[col].max()))
                for axis, col in AXIS_COLUMNS.items()
            }
            old = _canonical_keys(
                read_bc_table(path, columns=GRID_KEY_COLS + have, where=where)
            )
            merged = target.merge(old, how="left", on=GRID_KEY_COLS)
            for c in have:
                need[c] = merged[c].isna().values

    out = pd.concat([target, need], axis=1)
    return out[need.any(axis=1).values].reset_index(drop=True)


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
            # len(teff)=60, len(logg)=11, len(feh)=11, len(av)=20, nfilters=9
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
