"""
Merge synthetic tracks from generate_interpolated_missing_tracks.py into the
real MIST EEP data: both the per-feh processed_tracks/*.parquet source files
(the source of truth) and the alpha/vvcrit grid.parquet file built from them
(what the model actually reads).

Existing (real) rows get provenance columns added (interpolated=False,
interp_method=None, ...) so the merged files still let you tell real vs.
synthetic data apart. A one-time .bak copy of every processed_tracks file
this touches is made before its first write (skipped if a .bak already
exists, so reruns don't clobber the original backup). The merge is
idempotent -- an (initfeh, mass) track already present in a processed_tracks
file is skipped rather than duplicated. grid.parquet is not patched
directly; it's fully regenerated from the (now-merged) processed_tracks
files via generate_EEP_grid_for_alpha_vvcrit(), which is what naturally
keeps it consistent on any future regeneration too.

Usage:
    poetry run python src/exozippy/models/MIST/MISTv2.5/merge_interpolated_missing_tracks.py
"""
import shutil
from pathlib import Path

import pandas as pd

from exozippy.models.MIST.generate_MIST_EEP_Tables import (
    _generate_alpha_vvcrit_filename_parts,
    generate_EEP_grid_for_alpha_vvcrit,
)

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

EEP_PROCESSED_TRACKS_PATH_DEFAULT = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_INTERPOLATED_TRACKS_PATH_DEFAULT = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/interpolated_tracks/")
EEP_GRID_PATH_DEFAULT = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/grids/")

BASE_COLUMNS = [
    "mass", "EEP", "initfeh", "feh_mist", "radius_mist",
    "teff_mist", "delta_nu", "nu_max", "age_mist", "dEEP_dage", "here_be_dragons",
]
PROVENANCE_COLUMNS = ["interpolated", "interp_method", "interp_neighbor_1", "interp_neighbor_2"]
ALL_COLUMNS = BASE_COLUMNS + PROVENANCE_COLUMNS


def _backup(filepath):
    backup_path = filepath.with_suffix(filepath.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(filepath, backup_path)
        print(f"  backed up {filepath.name} -> {backup_path.name}")
    return backup_path


def _ensure_provenance_columns(df):
    df = df.copy()
    if "interpolated" not in df.columns:
        df["interpolated"] = False
        df["interp_method"] = None
        df["interp_neighbor_1"] = None
        df["interp_neighbor_2"] = None
    return df


def merge_interpolated_tracks_for_alpha_vvcrit(alpha, vvcrit=0.0,
                                                interpolated_path=EEP_INTERPOLATED_TRACKS_PATH_DEFAULT,
                                                processed_path=EEP_PROCESSED_TRACKS_PATH_DEFAULT,
                                                grid_path=EEP_GRID_PATH_DEFAULT):
    filename_alpha_part, filename_vvcrit_part = _generate_alpha_vvcrit_filename_parts(alpha, vvcrit)
    suffix = f"_{filename_alpha_part}_{filename_vvcrit_part}.parquet"

    interpolated_files = sorted(interpolated_path.glob(f"feh_*{suffix}"))
    if not interpolated_files:
        print(f"No interpolated tracks found for alpha={alpha}, vvcrit={vvcrit} in {interpolated_path}")
        return

    any_merged = False

    for interp_file in interpolated_files:
        processed_file = processed_path / interp_file.name
        if not processed_file.exists():
            raise FileNotFoundError(
                f"No matching processed_tracks file for {interp_file.name} at {processed_file}"
            )

        interp_df = pd.read_parquet(interp_file)
        processed_df = _ensure_provenance_columns(pd.read_parquet(processed_file))

        existing_pairs = set(zip(processed_df["initfeh"].round(6), processed_df["mass"].round(6)))

        to_add = []
        for (initfeh, mass), track in interp_df.groupby(["initfeh", "mass"]):
            key = (round(initfeh, 6), round(mass, 6))
            if key in existing_pairs:
                print(f"  {interp_file.name}: initfeh={initfeh}, mass={mass} already present, skipping")
                continue
            to_add.append(track)

        if not to_add:
            print(f"{interp_file.name}: nothing new to merge")
            continue

        print(f"{interp_file.name}: merging {len(to_add)} track(s) into {processed_file.name}")
        _backup(processed_file)

        merged_df = pd.concat([processed_df[ALL_COLUMNS]] + [t[ALL_COLUMNS] for t in to_add], ignore_index=True)
        merged_df.to_parquet(processed_file, compression="snappy")
        any_merged = True

    if not any_merged:
        print("Nothing new merged into processed_tracks; leaving grid.parquet untouched.")
        return

    grid_filename = f"{filename_alpha_part}_{filename_vvcrit_part}.grid.parquet"
    grid_file = grid_path / grid_filename
    if grid_file.exists():
        _backup(grid_file)

    print(f"Regenerating {grid_filename} from processed_tracks...")
    generate_EEP_grid_for_alpha_vvcrit(alpha, vvcrit, processed_path=processed_path,
                                        grid_path=grid_path, save=True)
    print(f"Done -> {grid_file}")

# -------------------------------------------------------------------
# Merge Missing Tracks for Grid Points
# -------------------------------------------------------------------

EEP_PROCESSED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_INTERPOLATED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/interpolated_tracks/")
EEP_GRID_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/grids/")

if __name__ == "__main__":
    merge_interpolated_tracks_for_alpha_vvcrit(alpha=0.0, vvcrit=0.0, 
                                               interpolated_path=EEP_INTERPOLATED_TRACKS_PATH,
                                               processed_path=EEP_PROCESSED_TRACKS_PATH,
                                               grid_path=EEP_GRID_PATH)
