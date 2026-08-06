"""
For each missing (initfeh, mass) grid point at alpha and vvcrit, determine
whether it could be filled in by interpolating existing adjacent grid points.

Search priority (per spec): alpha -> initfeh -> mass. A dimension is only
used if BOTH its immediate neighbors already exist as real grid points
(bracketing is required; we don't extrapolate from one side). The first
dimension (in priority order) with both neighbors present wins; if none of
the three dimensions have both neighbors present, the point is marked as not
interpolatable.

Neighbor values for initfeh and mass are taken from the *global* grids (the
union of all initfeh / mass values that appear anywhere in the MISTv2.5
processed tracks), since the true MIST sampling grid is irregular (finer
mass sampling near the MS turnoff, irregular initfeh spacing at low
metallicity) and per-feh/per-alpha subsets are exactly what's incomplete.
Neighbor values for alpha are the fixed step of 0.2 (the only sampled alpha
values are -0.2, 0.0, 0.2, 0.4, 0.6).

Usage:
    poetry run python src/exozippy/models/MIST/MISTv2.5/find_interpolatable_missing_grid_points.py
"""
import bisect
import pandas as pd
from exozippy.models.MIST.generate_MIST_EEP_Tables import _generate_alpha_vvcrit_filename_parts
import numpy as np

from pathlib import Path

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

# define where things are/should be saved by default

EEP_PROCESSED_TRACKS_PATH_DEFAULT = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_GRID_PATH_DEFAULT = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
MISSING_GRID_POINTS_PATH_DEFAULT = current_dir / "EEPs" / "MissingGridPoints" 


# -------------------------------------------------------------------
# Find Missing Grid Points and Generate CSV Containing Points
# -------------------------------------------------------------------

def _find_missing_grid_points(alpha, vvcrit=0.0,
                              grid_path=EEP_GRID_PATH_DEFAULT,
                              save=True,
                              save_path=MISSING_GRID_POINTS_PATH_DEFAULT):

    filename_alpha_part, filename_vvcrit_part = _generate_alpha_vvcrit_filename_parts(alpha, vvcrit)
    filename = f"{filename_alpha_part}_{filename_vvcrit_part}.grid.parquet"

    alpha_vvcrit_df = pd.read_parquet(grid_path / filename, columns=["mass", "initfeh", "EEP"], engine="pyarrow")

    mass_vals = np.sort(alpha_vvcrit_df["mass"].unique())
    feh_vals = np.sort(alpha_vvcrit_df["initfeh"].unique())

    expected_index = pd.MultiIndex.from_product(
        [mass_vals, feh_vals],
        names=["mass", "initfeh"],
    )

    present_index = pd.MultiIndex.from_arrays(
        [alpha_vvcrit_df["mass"], alpha_vvcrit_df["initfeh"]],
        names=expected_index.names,
    )

    missing_index = expected_index.difference(present_index)
    missing_noEEP_df = missing_index.to_frame(index=False)
    print(f"Alpha {alpha} Vvcrit {vvcrit}: Missing combinations: {len(missing_noEEP_df)}")

    # sort by initfeh and then by mass
    missing_noEEP_df = missing_noEEP_df.sort_values(by=["initfeh", "mass"])
    
    if save:
        missing_points_filename = f"missing_grid_points_{filename_alpha_part}_{filename_vvcrit_part}.csv"
        missing_noEEP_df[["initfeh", "mass"]].to_csv(save_path / missing_points_filename, index=False)

    return missing_noEEP_df


# -------------------------------------------------------------------
# Generate File Determining Which Missing Points are Interpolable
# -------------------------------------------------------------------

FEH_CODES = [
    "m400", "m350", "m300", "m275", "m250", "m225", "m200", "m175",
    "m150", "m125", "m100", "m075", "m050", "m025", "p000", "p025", "p050",
]
ALPHA_CODES = {"-0.2": "m2", 
               "0.0": "p0", 
               "0.2": "p2",
               "0.4": "p4",
               "0.6": "p6"}

ALPHA_VALS = [float(k) for k in ALPHA_CODES.keys()]

# find ALPHA_CODES corresponding to ALPHA_VALS surrounding an alpha value
def _get_adjacent_alpha_codes(alpha_val):
    # Find the indices of the two ALPHA_VALS surrounding the given alpha_val
    idx = bisect.bisect_left(ALPHA_VALS, alpha_val)
    
    # Get the adjacent alpha codes
    low_code, low_key = None, None
    code, code_key = None, None
    high_code, high_key = None, None
    if idx > 0:
        low_code = ALPHA_CODES[str(ALPHA_VALS[idx - 1])]
        low_key = str(ALPHA_VALS[idx - 1])
    if idx < len(ALPHA_VALS):
        code = ALPHA_CODES[str(ALPHA_VALS[idx])]
        code_key = str(ALPHA_VALS[idx])
    if idx + 1 < len(ALPHA_VALS):
        high_code = ALPHA_CODES[str(ALPHA_VALS[idx + 1])]
        high_key = str(ALPHA_VALS[idx + 1])
    return [low_code, code, high_code], [low_key, code_key, high_key]


def _feh_code_to_value(code):
    sign = -1.0 if code[0] == "m" else 1.0
    return sign * int(code[1:]) * 0.01


FEH_VALUES = sorted(_feh_code_to_value(c) for c in FEH_CODES)


def load_existing_points(alpha, vvcrit=0.0, processed_path=EEP_PROCESSED_TRACKS_PATH_DEFAULT):
    """Return {alpha_str: set of (initfeh, mass)} for alpha in adjacent ALPHA_CODES at
    a specific vvcrit value, and the global sorted list of unique mass values seen anywhere."""

    _, alpha_adj_keys = _get_adjacent_alpha_codes(alpha)

    ALPHA_ADJ_CODES_DICT = {k: ALPHA_CODES[k] for k in alpha_adj_keys if k in ALPHA_CODES}

    points = {a: set() for a in ALPHA_ADJ_CODES_DICT}
    all_masses = set()

    for feh_code in FEH_CODES:
        for alpha_str, alpha_code in ALPHA_ADJ_CODES_DICT.items():
            fname = processed_path / f"feh_{feh_code}_afe_{alpha_code}_vvcrit{vvcrit}.parquet"
            try:
                df = pd.read_parquet(fname, columns=["initfeh", "mass"])
                pairs = set(zip(df["initfeh"].round(6), df["mass"].round(6)))
                points[alpha_str].update(pairs)
                all_masses.update(df["mass"].round(6).unique().tolist())
            except FileNotFoundError:
                continue

    return points, sorted(all_masses)


def neighbors_in_grid(value, grid):
    """Return (prev, next) immediate neighbors of `value` in sorted `grid`,
    or None for either side if `value` is at/near a grid boundary."""
    idx = bisect.bisect_left(grid, value)
    prev_val = grid[idx - 1] if idx - 1 >= 0 else None
    # if value itself is in the grid at idx, next neighbor is idx+1
    if idx < len(grid) and abs(grid[idx] - value) < 1e-9:
        next_val = grid[idx + 1] if idx + 1 < len(grid) else None
    else:
        next_val = grid[idx] if idx < len(grid) else None
    return prev_val, next_val


def find_interpolatable_points(alpha, missing_df, points, mass_grid):
    results = []
    for _, row in missing_df.iterrows():
        initfeh, mass = round(row["initfeh"], 6), round(row["mass"], 6)

        result = {
            "initfeh": initfeh, "mass": mass,
            "interpolatable": False, "method": "none",
            "neighbor_1": None, "neighbor_2": None,
        }

        # --- 1. alpha neighbors: at same (initfeh, mass) ---
        # first check if alpha_val is on the edge 
        # because it cant be interpolated in alpha in that case
        alpha_adj_codes, alpha_adj_keys = _get_adjacent_alpha_codes(alpha)

        if None in alpha_adj_codes:
            # alpha_val is on the edge, cannot interpolate in alpha
            pass

        else:
            alpha_low_key = alpha_adj_keys[0]
            alpha_high_key = alpha_adj_keys[2]
            lo_exists = (initfeh, mass) in points[alpha_low_key]
            hi_exists = (initfeh, mass) in points[alpha_high_key]
            if lo_exists and hi_exists:
                result.update(
                    interpolatable=True, method="alpha",
                    neighbor_1=f"alpha={alpha_low_key}", neighbor_2=f"alpha={alpha_high_key}",
                )
                results.append(result)
                continue

        # --- 2. initfeh neighbors: prev/next initfeh at same mass, alpha=0 ---
        feh_prev, feh_next = neighbors_in_grid(initfeh, FEH_VALUES)
        if feh_prev is not None and feh_next is not None:
            lo_exists = (feh_prev, mass) in points[alpha_adj_keys[1]]
            hi_exists = (feh_next, mass) in points[alpha_adj_keys[1]]
            if lo_exists and hi_exists:
                result.update(
                    interpolatable=True, method="initfeh",
                    neighbor_1=f"initfeh={feh_prev}", neighbor_2=f"initfeh={feh_next}",
                )
                results.append(result)
                continue

        # --- 3. mass neighbors: prev/next mass at same initfeh, alpha=0 ---
        mass_prev, mass_next = neighbors_in_grid(mass, mass_grid)
        if mass_prev is not None and mass_next is not None:
            lo_exists = (initfeh, mass_prev) in points[alpha_adj_keys[1]]
            hi_exists = (initfeh, mass_next) in points[alpha_adj_keys[1]]
            if lo_exists and hi_exists:
                result.update(
                    interpolatable=True, method="mass",
                    neighbor_1=f"mass={mass_prev}", neighbor_2=f"mass={mass_next}",
                )
                results.append(result)
                continue

        results.append(result)

    return pd.DataFrame(results)


# -------------------------------------------------------------------
# Generate Files for Missing Grid Points
# -------------------------------------------------------------------

EEP_PROCESSED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_GRID_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
MISSING_GRID_POINTS_PATH = current_dir / "EEPs" / "MissingGridPoints" 


def main():

    alpha_grid_values = [-0.2, 0.0, 0.2, 0.4, 0.6]
    vvcrit_values = [0.0]

    for alpha in alpha_grid_values:
        for vvcrit in vvcrit_values:
            print(f"Processing alpha {alpha} and vvcrit {vvcrit}")

            # first see if we have already generated the missing grid points
            filename_alpha_part, filename_vvcrit_part = _generate_alpha_vvcrit_filename_parts(alpha, vvcrit)
            missing_filename = f"missing_grid_points_{filename_alpha_part}_{filename_vvcrit_part}.csv"

            if (MISSING_GRID_POINTS_PATH / missing_filename).exists():
                missing_df = pd.read_csv(MISSING_GRID_POINTS_PATH / missing_filename)
            else:
                missing_df = _find_missing_grid_points(alpha, vvcrit=vvcrit,
                                                        grid_path=EEP_GRID_PATH,
                                                        save_path=MISSING_GRID_POINTS_PATH)

            missing_interp_filename = f"missing_grid_points_{filename_alpha_part}_{filename_vvcrit_part}_interpolatable.csv"

            if (MISSING_GRID_POINTS_PATH / missing_interp_filename).exists():
                out_df = pd.read_csv(MISSING_GRID_POINTS_PATH / missing_interp_filename)
            else:
                points, mass_grid = load_existing_points(alpha, vvcrit=vvcrit)
                out_df = find_interpolatable_points(alpha, missing_df, points, mass_grid)
                out_df.to_csv(MISSING_GRID_POINTS_PATH / missing_interp_filename, index=False)

                n_total = len(out_df)
                n_interp = out_df["interpolatable"].sum()
                print(f"Total missing points: {n_total}")
                print(f"Interpolatable: {n_interp}")
                print(out_df["method"].value_counts())
                print(f"\nWrote results to {MISSING_GRID_POINTS_PATH / missing_interp_filename}")


if __name__ == "__main__":
    main()
