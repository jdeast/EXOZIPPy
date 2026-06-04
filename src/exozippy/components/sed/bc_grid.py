"""
Bolometric Correction (BC) grid loader and pytensor interpolator.

This module replaces the spectra grid / filter integration machinery
from `Code for Models/Classes/Grid.py` + `Spectra.py` + `Filter.py`
with a direct interpolation over a precomputed BC grid. Given a set
of filter names and a model name ("NextGen" in v1), it loads the
matching per-feh BC files from the `BCs/{MODEL}/{FACILITY}/` tree
and builds a pytensor-compatible RegularGridInterpolator over
(lgTeff, logg, feh, Av) returning a vector of BC values, one per
requested filter.

File layout assumed (the NextGen tree):
    {bc_root}/
        {model}/                     e.g. "NextGen"
            {facility}/              e.g. "2MASS", "GAIA", "WISE"
                feh{+/-X.X}_afe{+/-Y.Y}.{FACILITY}

Each file is whitespace-delimited with 5 header lines (`#` prefixed)
and columns:
    lgTef  logg  Fe_H  a_Fe  Av  Rv  <filter1> <filter2> ...

Grid assumptions in v1:
  * single alpha/Fe slice (afe = 0.0) across all files
  * single Rv slice (Rv = 3.10) across all files
  * feh varies across files via filename parsing
  * (lgTeff, logg, Av) grids are identical across feh files
  * dataframe created using these files will have axes (teff, logg, feh, av)

These assumptions hold for the NextGen tree as it currently ships.
When MIST is added, this loader grows a `model` dispatch so each
family can parse its own layout; the interpolator interface stays
the same.
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Literal
import warnings

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

_FILTERNAMES_ = "filternames.txt"


def _load_alias_table(path: Path = current_dir) -> pd.DataFrame | None:
    """Load the VOID↔MIST↔SVO name alias table, if present."""
    _FILTERNAMES_TXT = path / "filters" / _FILTERNAMES_
    if not _FILTERNAMES_TXT.exists():
        return None
    return pd.read_csv(
        _FILTERNAMES_TXT, sep="\t", comment="#", skipinitialspace=True
    )


def resolve_filter_name(user_name: str, alias_df: pd.DataFrame | None, 
                        alias: Literal["MIST", "SVO"]) -> str:
    """
    Translate a user-facing filter label (e.g. "2MASS.J", "Gaia.G")
    into the corresponding MIST/SVO filter label.

    Examples: 
        "2MASS.J" --> "2MASS_J" | "2MASS/2MASS.J"
        "Gaia.G" --> "Gaia_G_DR2Rev" | "GAIA/GAIA2r.G"

    If the alias table is missing or doesn't know the name, assume the
    user has already provided the MIST/SVO column name and return it.
    """
    if alias_df is None:
        return user_name
    try:
        rename = alias_df[alias_df.eq(user_name).any(axis=1)][alias].values[0]
        if rename in ("Unsupported", None) or pd.isna(rename):
            # No alias column — fall back to the user string so the
            # caller gets a clear KeyError later instead of a silent
            # mismatch.
            return user_name
        return str(rename)
    except Exception:
        return user_name


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
# BC file parsing
# -------------------------------------------------------------------

# Default root; callers should override via the SED config when not
# running out of the project directory.
DEFAULT_BC_ROOT = Path(__file__).parent / "models"

# compile pattern for bolometric correction tables
_FEH_FILENAME_RE = re.compile(r"feh(?P<feh>[+-]\d+\.\d+)_afe(?P<alpha>[+-]\d+\.\d+)\.(?P<facility>\w+)")


def _parse_feh_from_filename(name: str) -> float:
    m = _FEH_FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse feh from BC filename: {name}")
    return float(m.group("feh"))


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
                numfilters = int(stripped[0])
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
    df.insert(0, 'teff', round(10**df['lgTef']))
    df.rename(columns={'Fe_H': 'feh', 'a_Fe': 'alpha'}, inplace=True)
    df.drop(columns=['lgTef'], inplace=True)
    filter_cols = col_names[-numfilters:]  # after teff, logg, feh, alpha, Av, Rv

    return df, filter_cols


def _collect_facility_files(
    bc_root: Path, model: str, facility: str
) -> List[Path]:
    subdir = bc_root / model
    if not subdir.is_dir():
        raise FileNotFoundError(
            f"Bolometric corrections not calculated for ``{model}`` model. Specify a different model.")
    subdir = subdir / "BCs" / facility
    if not subdir.is_dir():
        raise NotImplementedError(
            f"Bolometric corrections not calculated for ``{facility}``. Specify a different filter set.\n Future implementation will automate this step.")
    
    return sorted(subdir.glob(f"feh*_afe+0.0.{facility}"))


def peek_grid_axes(
    model: str = "NextGen",
    bc_root: Path | str = DEFAULT_BC_ROOT,
) -> Dict[str, np.ndarray]:
    """
    Cheap axis-metadata reader for the BC grid.

    Used by the SED component to coordinate star-parameter bounds
    with the grid extent BEFORE the full grid is loaded (i.e. during
    SED.__init__, when star.build_parameters hasn't run yet).

    Assumes the grid's (teff, logg, av) axes are identical across
    facilities and across feh files, so we only need to open one file
    per model. The feh axis is derived from filenames under the
    chosen facility directory.

    Parameters
    ----------
    model : str
        BC model name (selects the first-level subdirectory of bc_root).
    bc_root : Path
        Root directory holding the {model}/{facility}/feh*_afe*.{FAC}
        tree.

    Returns
    -------
    dict with keys:
        teff_pts  : np.ndarray, shape (n_teff,)
        logg_pts  : np.ndarray, shape (n_logg,)
        feh_pts   : np.ndarray, shape (n_feh,)
        av_pts    : np.ndarray, shape (n_av,)
    """
    bc_root = Path(bc_root)
    model_dir = bc_root / model / "BCs"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"BC model directory not found: {model_dir}"
        )

    # Pick the first facility subdir that actually has feh*_afe*.<FAC>
    # files. We don't care which facility; axes are identical across.
    facility_dirs = [p for p in sorted(model_dir.iterdir()) if p.is_dir()]
    if not facility_dirs:
        raise FileNotFoundError(
            f"No facility subdirectories under {model_dir}"
        )

    chosen_fac_dir = None
    feh_files: List[Path] = []
    for fac_dir in facility_dirs:
        candidates = sorted(fac_dir.glob(f"feh*_afe+0.0.{fac_dir.name}"))
        if candidates:
            chosen_fac_dir = fac_dir
            feh_files = candidates
            break
    if chosen_fac_dir is None:
        raise FileNotFoundError(
            f"No feh*_afe+0.0.<FAC> files found under any facility in "
            f"{model_dir}"
        )

    feh_pts = np.array(
        sorted(_parse_feh_from_filename(p.name) for p in feh_files),
        dtype=float,
    )

    df, _ = _read_single_bc_file(feh_files[0])
    teff_pts = np.sort(df["teff"].unique()).astype(float)
    logg_pts = np.sort(df["logg"].unique()).astype(float)
    av_pts = np.sort(df["Av"].unique()).astype(float)

    return {
        "teff_pts": teff_pts,
        "logg_pts": logg_pts,
        "feh_pts": feh_pts,
        "av_pts": av_pts,
    }

# -------------------------------------------------------------------
# Grid assembly
# -------------------------------------------------------------------


def build_bc_grid(
    user_filter_names: Sequence[str],
    model: str = "NextGen",
    bc_root: Path | str = DEFAULT_BC_ROOT,
) -> Dict:
    """
    Assemble a 4D BC grid for a specific set of filters.

    Parameters
    ----------
    user_filter_names : sequence of str
        Filter labels as they appear in the .sed file (VOID-style,
        e.g. "2MASS.J", "Gaia.G", "WISE.W1").
    model : str
        BC model name; selects the first-level subdirectory of bc_root.
    bc_root : Path
        Root directory holding the {model}/{facility}/feh*_afe*.{FACILITY}
        tree.

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
    bc_root = Path(bc_root)
    alias_df = _load_alias_table()

    # 1. Resolve user names -> MIST column names and group by facility.
    mist_names = [resolve_filter_name(n, alias_df, alias='MIST') for n in user_filter_names]
    svo_names = [resolve_filter_name(n, alias_df, alias='SVO') for n in user_filter_names]
    facilities = [facility_from_svo_name(s) for s in svo_names]
    by_facility: Dict[str, List[Tuple[int, str]]] = {}
    for idx, (fac, mist) in enumerate(zip(facilities, mist_names)):
        by_facility.setdefault(fac, []).append((idx, mist))

    # 2. For each facility, load all feh files, keeping only the
    # requested columns. We stash them per feh so we can later stack
    # into one monolithic grid.
    per_facility_frames: Dict[str, Dict[float, pd.DataFrame]] = {}
    for fac, items in by_facility.items():
        feh_files = _collect_facility_files(bc_root, model, fac)
        if not feh_files:
            file_dir = bc_root / model / "BCs" / fac
            raise FileNotFoundError(
                f"No BC files for facility '{fac}' under "
                f"{file_dir}"
            )
        wanted_cols = [mist for _, mist in items]
        frames: Dict[float, pd.DataFrame] = {}
        for p in feh_files:
            feh = _parse_feh_from_filename(p.name)
            df, file_filters = _read_single_bc_file(p)
            missing = set(wanted_cols) - set(file_filters)
            if missing:
                warnings.warn(
                    f"Bolometric corrections not calculated for "
                    f"``{sorted(missing)}``.\n Removing ``{sorted(missing)}`` from fit. "
                    f"Future implementation will automate filter calculations.", 
                    UserWarning)
                wanted_cols = set(wanted_cols) - missing
            keep = ["teff", "logg", "feh", "Av"] + wanted_cols
            frames[feh] = df[keep].copy()
        per_facility_frames[fac] = frames

    # 3. Cross-check: all facilities must share the same (teff, logg,
    # feh, Av) grid. Use the first facility as the canonical grid.
    canonical_fac = next(iter(per_facility_frames))
    canonical_frames = per_facility_frames[canonical_fac]
    feh_pts = np.array(sorted(canonical_frames.keys()), dtype=float)

    any_frame = next(iter(canonical_frames.values()))
    teff_pts = np.sort(any_frame["teff"].unique()).astype(float)
    logg_pts = np.sort(any_frame["logg"].unique()).astype(float)
    av_pts = np.sort(any_frame["Av"].unique()).astype(float)

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

    for fac, frames in per_facility_frames.items():
        items = by_facility[fac]  # [(global_idx, mist_name), ...]
        for feh_val, df in frames.items():
            f_idx = int(np.searchsorted(feh_pts, feh_val))
            t_idx = np.searchsorted(teff_pts, df["teff"].values)
            g_idx = np.searchsorted(logg_pts, df["logg"].values)
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
        i_lo = int(np.searchsorted(pts, lo, side='left'))
        # If lo lands exactly on a grid point, i_lo is already correct.
        # If lo falls between pts[i_lo-1] and pts[i_lo], we need pts[i_lo-1]
        # to bracket lo from below.
        if i_lo > 0 and pts[i_lo] > lo:
            i_lo -= 1

    if hi is None:
        i_hi = n - 1
    else:
        i_hi = int(np.searchsorted(pts, hi, side='right')) - 1
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
    idx = [slice(None)] * (bc_values.ndim - 1)   # one entry per grid axis
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
            idx[axis] = np.array([nearest_idx])   # keep axis with length 1
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
    grid_idx = np.ix_(*[
        idx[ax] if isinstance(idx[ax], np.ndarray) else np.arange(bc_values.shape[ax])
        for ax in range(bc_values.ndim - 1)
    ])
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
        coords = pt.atleast_2d(coords)                        # (N, ndim)
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
