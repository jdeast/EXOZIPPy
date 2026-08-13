"""
Generate synthetic EEP tracks for missing (initfeh, mass) grid points that
were flagged as interpolatable by find_interpolatable_missing_grid_points.py.

Because MIST EEP tracks are indexed by Equivalent Evolutionary Phase, row N
means "the same evolutionary phase" across every track in the grid -- so a
missing track can be synthesized by linearly interpolating, row-by-row, the
two bracketing tracks along whichever dimension (alpha, initfeh, or mass)
found both neighbors present.

Per-row treatment:
  - feh_mist, radius_mist, teff_mist, delta_nu, nu_max, age_mist: linear
    blend of the two neighbor tracks, weighted by where the target value
    falls between them.
  - here_be_dragons: elementwise max of the two neighbors' flags -- if
    either neighbor is past its valid range at a given EEP, the
    interpolated point should be flagged too.
  - dEEP_dage: recomputed via np.gradient on the interpolated age/EEP,
    not blended directly (it's a derived quantity).
  - mass / initfeh: set to the exact target grid coordinates (not blended).

Interpolated tracks are saved to a dedicated `interpolated_tracks/` folder
(mirroring processed_tracks' per-feh/alpha/vvcrit parquet naming) rather than
merged into the real processed_tracks/grid files, and are tagged with
provenance columns (interpolated, interp_method, interp_neighbor_1/2) so
they're always distinguishable from genuine MESA-derived tracks.

Usage:
    poetry run python src/exozippy/models/MIST/MISTv2.5/generate_interpolated_missing_tracks.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from exozippy.models.MIST.generate_MIST_EEP_Tables import _generate_alpha_vvcrit_filename_parts

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

EEP_PROCESSED_TRACKS_PATH_DEFAULT = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_INTERPOLATED_TRACKS_PATH_DEFAULT = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/interpolated_tracks/")
MISSING_GRID_POINTS_PATH_DEFAULT = current_dir / "MISTv2.5" / "EEPs" / "MissingGridPoints"

FEH_CODES = [
    "m400", "m350", "m300", "m275", "m250", "m225", "m200", "m175",
    "m150", "m125", "m100", "m075", "m050", "m025", "p000", "p025", "p050",
]
ALPHA_CODES = {"-0.2": "m2", "0.0": "p0", "0.2": "p2", "0.4": "p4", "0.6": "p6"}

TARGET_LENGTH = 807
BLENDED_COLUMNS = ["feh_mist", "radius_mist", "teff_mist", "delta_nu", "nu_max", "age_mist"]


def _feh_value_to_code(value):
    sign = "m" if value < 0 else "p"
    return f"{sign}{round(abs(value) * 100):03d}"


def _parse_neighbor_value(neighbor_str):
    # e.g. "alpha=-0.2" / "initfeh=-2.75" / "mass=0.1" -> -0.2 / -2.75 / 0.1
    return float(neighbor_str.split("=")[1])


def _load_single_track(feh_code, alpha_code, vvcrit, mass, processed_path):
    fname = processed_path / f"feh_{feh_code}_afe_{alpha_code}_vvcrit{vvcrit:0.1f}.parquet"
    df = pd.read_parquet(fname)
    track = df[np.isclose(df["mass"], mass)].sort_values("EEP").reset_index(drop=True)
    if len(track) != TARGET_LENGTH:
        raise ValueError(
            f"Expected {TARGET_LENGTH} rows for mass={mass} in {fname.name}, got {len(track)}"
        )
    return track


def _generate_interpolated_track(row, alpha, vvcrit, processed_path):
    """Build one synthetic 807-row track for a single interpolatable missing point."""
    initfeh, mass, method = row["initfeh"], row["mass"], row["method"]
    lo_val, hi_val = sorted([_parse_neighbor_value(row["neighbor_1"]),
                              _parse_neighbor_value(row["neighbor_2"])])

    if method == "alpha":
        feh_code = _feh_value_to_code(initfeh)
        target = alpha
        lo_track = _load_single_track(feh_code, ALPHA_CODES[str(lo_val)], vvcrit, mass, processed_path)
        hi_track = _load_single_track(feh_code, ALPHA_CODES[str(hi_val)], vvcrit, mass, processed_path)
    elif method == "initfeh":
        alpha_code = ALPHA_CODES[str(alpha)]
        target = initfeh
        lo_track = _load_single_track(_feh_value_to_code(lo_val), alpha_code, vvcrit, mass, processed_path)
        hi_track = _load_single_track(_feh_value_to_code(hi_val), alpha_code, vvcrit, mass, processed_path)
    elif method == "mass":
        feh_code = _feh_value_to_code(initfeh)
        alpha_code = ALPHA_CODES[str(alpha)]
        target = mass
        lo_track = _load_single_track(feh_code, alpha_code, vvcrit, lo_val, processed_path)
        hi_track = _load_single_track(feh_code, alpha_code, vvcrit, hi_val, processed_path)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    if not np.array_equal(lo_track["EEP"].values, hi_track["EEP"].values):
        raise ValueError(f"EEP index mismatch between neighbor tracks for initfeh={initfeh}, mass={mass}")

    weight_hi = (target - lo_val) / (hi_val - lo_val)

    new_track = pd.DataFrame()
    new_track["mass"] = np.full(len(lo_track), mass)
    new_track["EEP"] = lo_track["EEP"].values
    new_track["initfeh"] = np.full(len(lo_track), initfeh)

    for col in BLENDED_COLUMNS:
        new_track[col] = (1 - weight_hi) * lo_track[col].values + weight_hi * hi_track[col].values

    new_track["here_be_dragons"] = np.maximum(lo_track["here_be_dragons"].values, hi_track["here_be_dragons"].values)
    new_track["dEEP_dage"] = np.gradient(new_track["EEP"].values, new_track["age_mist"].values)

    new_track["interpolated"] = True
    new_track["interp_method"] = method
    new_track["interp_neighbor_1"] = row["neighbor_1"]
    new_track["interp_neighbor_2"] = row["neighbor_2"]

    return new_track


def generate_missing_tracks_for_alpha_vvcrit(alpha, vvcrit=0.0,
                                              missing_points_path=MISSING_GRID_POINTS_PATH_DEFAULT,
                                              processed_path=EEP_PROCESSED_TRACKS_PATH_DEFAULT,
                                              output_path=EEP_INTERPOLATED_TRACKS_PATH_DEFAULT,
                                              save=True):
    filename_alpha_part, filename_vvcrit_part = _generate_alpha_vvcrit_filename_parts(alpha, vvcrit)
    interp_filename = f"missing_grid_points_{filename_alpha_part}_{filename_vvcrit_part}_interpolatable.csv"
    interp_df = pd.read_csv(missing_points_path / interp_filename)

    interpolatable_rows = interp_df[interp_df["interpolatable"]]
    if len(interpolatable_rows) == 0:
        print(f"No interpolatable points found for alpha={alpha}, vvcrit={vvcrit}.")
        return {}

    tracks_by_file = {}
    for _, row in interpolatable_rows.iterrows():
        track = _generate_interpolated_track(row, alpha, vvcrit, processed_path)
        feh_code = _feh_value_to_code(row["initfeh"])
        out_fname = f"feh_{feh_code}_{filename_alpha_part}_{filename_vvcrit_part}.parquet"
        tracks_by_file.setdefault(out_fname, []).append(track)
        print(f"Generated interpolated track: initfeh={row['initfeh']}, mass={row['mass']} "
              f"(method={row['method']}, neighbors={row['neighbor_1']}/{row['neighbor_2']})")

    if save:
        output_path.mkdir(parents=True, exist_ok=True)
        for out_fname, tracks in tracks_by_file.items():
            combined = pd.concat(tracks, ignore_index=True)
            combined.to_parquet(output_path / out_fname, compression="snappy")
            print(f"Saved {len(tracks)} track(s) -> {output_path / out_fname}")

    return tracks_by_file


# -------------------------------------------------------------------
# Generate Missing Tracks for Grid Points
# -------------------------------------------------------------------

MISSING_GRID_POINTS_PATH = current_dir / "MISTv2.5" / "EEPs" / "MissingGridPoints"
EEP_PROCESSED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_INTERPOLATED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/interpolated_tracks/")

if __name__ == "__main__":
    generate_missing_tracks_for_alpha_vvcrit(alpha=0.0, vvcrit=0.0,
                                             missing_points_path=MISSING_GRID_POINTS_PATH,
                                             processed_path=EEP_PROCESSED_TRACKS_PATH,
                                             output_path=EEP_INTERPOLATED_TRACKS_PATH)
