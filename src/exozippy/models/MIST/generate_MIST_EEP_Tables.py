import re
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from exozippy.models.MIST.parse_MIST_EEP_filenames import (
    _generate_MIST_EEP_url,
    _parse_initfeh_alpha_vvcrit_from_name,
)

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable

# -------------------------------------------------------------------
# Defining Default Paths to EEP Tracks
# -------------------------------------------------------------------

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

# replace with your path to the directory containing the MIST EEP tracks/parquet files/grids
EEP_RAW_TRACKS_PATH_DEFAULT = Path(
    "/Volumes/Data/EEP_Tracks/MISTv2.5/raw_tracks/"
)
EEP_PROCESSED_TRACKS_PATH_DEFAULT = Path(
    "/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/"
)
EEP_GRID_PATH_DEFAULT = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/grids/")

# -------------------------------------------------------------------
# Read MIST EEP Track Files
# -------------------------------------------------------------------


def read_mist_eep(filepath):
    """
    Read a MIST EEP track file into a pandas DataFrame.

    Returns a DataFrame of the track data plus a `meta` attribute dict with
    header quantities (initial_mass, [Fe/H], Yinit, Zinit, [a/Fe], v/vcrit,
    N_pts, N_EEP, N_col, phase, type, EEP_indices, mist_version, mesa_revision).
    """
    meta = {}
    col_numbers_line = None
    col_names_line = None
    data_start = None

    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        stripped = line.lstrip("# ").strip()

        if "MIST version number" in line:
            meta["mist_version"] = float(line.split("=")[1].strip())
        elif "MESA revision number" in line:
            meta["mesa_revision"] = int(line.split("=")[1].strip())
        elif "Yinit" in line and "Zinit" in line:
            # Next line has the values
            vals = lines[i + 1].lstrip("#").split()
            meta["Yinit"] = float(vals[0])
            meta["Zinit"] = float(vals[1])
            meta["FeH"] = float(vals[2])
            meta["aFe"] = float(vals[3])
            meta["v_vcrit"] = float(vals[4])
        elif "initial_mass" in line and "N_pts" in line:
            vals = lines[i + 1].lstrip("#").split()
            meta["initial_mass"] = float(vals[0])
            meta["N_pts"] = int(vals[1])
            meta["N_EEP"] = int(vals[2])
            meta["N_col"] = int(vals[3])
            meta["phase"] = vals[4]
            meta["type"] = vals[5]
        elif line.startswith("# EEPs:"):
            meta["EEP_indices"] = list(map(int, line.split(":")[1].split()))
        elif re.match(r"^#\s+\d+\s+\d+", line):
            # The column-number index line (e.g. "#   1   2   3 ...")
            col_numbers_line = i
        elif (
            col_numbers_line is not None
            and col_names_line is None
            and line.startswith("#")
        ):
            # The line immediately after column numbers is column names
            col_names_line = i
            columns = line.lstrip("#").split()
        elif col_names_line is not None and not line.startswith("#"):
            data_start = i
            break

    df = pd.read_csv(
        filepath,
        skiprows=data_start,
        names=columns,
        sep=r"\s+",
        engine="python",
    )

    df.attrs["meta"] = meta
    return df


def _add_or_grab_solar_values_to_MIST_yaml(
    EEP_path=EEP_RAW_TRACKS_PATH_DEFAULT,
    version: Literal["1.2", "2.5"] = "2.5",
):

    # locate the YAML file corresponding to the MIST EEP grid
    if "models" in str(current_dir):
        # assume we're running this within the "models" directory
        yaml_filename = (
            current_dir
            / f"MISTv{version}"
            / "EEPs"
            / f"MISTv{version}.grid.yaml"
        )
    else:
        yaml_filename = (
            current_dir.parent.parent
            / "models"
            / "MIST"
            / f"MISTv{version}"
            / "EEPs"
            / f"MISTv{version}.grid.yaml"
        )

    if not yaml_filename.is_file():
        raise FileNotFoundError(f"YAML file not found at {yaml_filename}")

    # check if the YAML file already has the key "log_surf_fe_over_x_solar"
    with open(yaml_filename, "r") as file:
        grid_data = yaml.safe_load(file) or {}

    if "log_surf_fe_over_x_solar" in grid_data:
        return grid_data[
            "log_surf_fe_over_x_solar"
        ]  # key already exists, no need to append

    # if the key does not exist, we need to compute the solar value and append it to the YAML file
    solar_folder, _ = _generate_MIST_EEP_url(
        initfeh=0.0, alpha=0.0, vvcrit=0.0, version=version
    )
    solar_path = EEP_path / f"MISTv{version}" / "raw_tracks" / solar_folder

    # first check if tracks are in a nested "eeps" folder within the solar_path
    nested_folder = solar_path / "eeps"
    if nested_folder.is_dir():
        solar_path = nested_folder

    # grab any track from the folder
    filename = next(solar_path.glob("*.track.eep"))

    df_solar = read_mist_eep(filename)
    log_surf_fe_over_x_solar = np.log10(
        df_solar["surface_fe56"].values[0]
    ) - np.log10(df_solar["surface_h1"].values[0])

    # otherwise, we will just append the new key-value pair to the YAML file
    grid_data = {}
    grid_data["log_surf_fe_over_x_solar"] = float(log_surf_fe_over_x_solar)

    with open(yaml_filename, "a") as file:
        yaml_text = yaml.safe_dump(grid_data)
        file.write(yaml_text)

    return grid_data["log_surf_fe_over_x_solar"]


# -------------------------------------------------------------------
# Create New DataFrame for a Single Set of MIST EEP Tracks
# -------------------------------------------------------------------


def _pad_or_trim_df(df, target_length=807):

    # EEP 808 = low mass stars enter the thermally pulsating asymptotic giant branch (TPAGB)
    # mass loss becomes significant
    # invalidating the assumption that the current mass is equal to the initial mass

    nrows = len(df)
    if nrows == target_length:
        return df
    elif nrows > target_length:
        return df.iloc[:target_length]
    else:
        n_missing = target_length - nrows
        # make a copy of the final row to pad the DataFrame
        final_row = df.iloc[[-1]].copy()
        padding = pd.DataFrame(
            np.nan, index=range(n_missing), columns=df.columns
        )
        padding.iloc[:] = final_row.values

        padded_df = pd.concat([df, padding], ignore_index=True)
        padded_df.attrs = df.attrs.copy()
        return padded_df


def _add_mass_col(df, new_df):

    nrows = len(df)
    new_df["mass"] = np.full(nrows, df.attrs["meta"]["initial_mass"])
    return


def _add_EEP_col(df, new_df):

    nrows = len(df)
    new_df["EEP"] = np.arange(nrows) + 1
    return


def _add_initfeh_col(df, new_df):

    nrows = len(df)
    new_df["initfeh"] = np.full(nrows, df.attrs["meta"]["FeH"])
    return


def _add_feh_col(df, new_df):

    # grab solar value from the MIST YAML file
    log_surf_fe_over_x_solar = _add_or_grab_solar_values_to_MIST_yaml()

    # find values of df['surface_h1'] > 1e-10
    # avoid division by zero or taking log of very small numbers
    valid_mask = df["surface_h1"] > 1e-10
    log_surf_fe_over_x_star = np.log10(
        df["surface_fe56"][valid_mask]
    ) - np.log10(df["surface_h1"][valid_mask])

    feh = log_surf_fe_over_x_star - log_surf_fe_over_x_solar
    new_df["feh_mist"] = np.full(
        len(df), 30.0
    )  # make all values an arbitrary large number
    new_df.loc[valid_mask, "feh_mist"] = feh
    return


def _add_radius_col(df, new_df):

    radius = 10 ** df["log_R"]
    new_df["radius_mist"] = radius
    return


def _add_teff_col(df, new_df):

    teff = 10 ** df["log_Teff"]
    new_df["teff_mist"] = teff
    return


def _add_astroseismic_cols(df, new_df):

    new_df["delta_nu"] = df["delta_nu"].values
    new_df["nu_max"] = df["nu_max"].values
    return


def _add_age_and_here_be_dragons_col(df, new_df):

    age = df["star_age"].values
    here_be_dragons = np.zeros(len(age))

    # Find values that appear more than once
    vals, counts = np.unique(age, return_counts=True)
    repeated_vals = vals[counts > 1]

    # Find if any feh values are equal to 30 (indicating invalid or placeholder values)
    feh_values = new_df["feh_mist"].values
    invalid_feh_mask = feh_values == 30.0

    # if no repeated values and no invalid feh values, just return the age array
    if repeated_vals.size == 0 and not np.any(invalid_feh_mask):
        new_df["age_mist"] = age
        new_df["here_be_dragons"] = here_be_dragons
        return

    # Get indices for all occurrences of those repeated ages
    duplicate_indices = np.isin(age, repeated_vals)

    # ensure unique ages by adding a small increment to duplicates
    # add small increments to duplicates to ensure unique ages
    add_time = 1  # yr
    age[duplicate_indices] += add_time * np.arange(np.sum(duplicate_indices))

    # flag all indices from the first invalid feh value onward, since this indicates
    # problematic ages that need adjustment -- not just the ones with duplicate ages
    # only want to calculate first invalid feh index if an invalid value exists
    if np.any(invalid_feh_mask):
        first_invalid_feh_index = np.argmax(invalid_feh_mask)
        problematic_indices = np.arange(len(age)) >= first_invalid_feh_index

    else:
        # only want to include *added* duplicated ages, not the original age that was duplicated
        problematic_indices = age > repeated_vals

    # mark the problematic indices in the "here_be_dragons" column
    here_be_dragons[problematic_indices] = 1 + np.arange(
        np.sum(problematic_indices)
    )

    new_df["age_mist"] = age
    new_df["here_be_dragons"] = here_be_dragons
    return


def _add_deep_dage_col(new_df):

    new_df["dEEP_dage"] = np.gradient(new_df["EEP"], new_df["age_mist"])
    return


def generate_new_df(df, target_length=807):

    # Define column headers and initialize df
    columns = [
        "mass",
        "EEP",
        "initfeh",
        "feh_mist",
        "radius_mist",
        "teff_mist",
        "delta_nu",
        "nu_max",
        "age_mist",
        "dEEP_dage",
        "here_be_dragons",
    ]

    new_df = pd.DataFrame(columns=columns)

    # pad or trim the original DataFrame to the target length
    pad_or_trim_df = _pad_or_trim_df(df, target_length=target_length)

    # fill in all the new columns
    _add_mass_col(pad_or_trim_df, new_df)
    _add_EEP_col(pad_or_trim_df, new_df)
    _add_initfeh_col(pad_or_trim_df, new_df)
    _add_feh_col(pad_or_trim_df, new_df)
    _add_radius_col(pad_or_trim_df, new_df)
    _add_teff_col(pad_or_trim_df, new_df)
    _add_astroseismic_cols(pad_or_trim_df, new_df)
    _add_age_and_here_be_dragons_col(pad_or_trim_df, new_df)
    _add_deep_dage_col(new_df)

    return new_df


def process_eep_directory(
    eep_dir,
    pattern="*.track.eep",
    show_progress=True,
    save=False,
    output_path=EEP_PROCESSED_TRACKS_PATH_DEFAULT,
):
    """
    Read every MIST EEP track file in `eep_dir` matching `pattern`, transform each with
    `generate_new_df`, and concatenate into one DataFrame. Optionally save the resulting
    DataFrame to `output_path` as a parquet file.
    """
    filepaths = sorted(
        p for p in eep_dir.glob(pattern) if not p.name.startswith("._")
    )

    if not filepaths:
        warnings.warn(f"No files matching '{pattern}' found in {eep_dir}")
        return pd.DataFrame()

    iterator = (
        tqdm(filepaths, desc=eep_dir.name, leave=False)
        if show_progress
        else filepaths
    )

    frames = [
        generate_new_df(read_mist_eep(filepath)) for filepath in iterator
    ]

    df = pd.concat(frames, ignore_index=True, copy=False)

    if save:
        filename = f"{eep_dir.name}.parquet"
        df.to_parquet(output_path / filename, compression="snappy")

    return df


# -------------------------------------------------------------------
# Generate Grid for Given Alpha and Vvcrit Values
# -------------------------------------------------------------------


def _generate_alpha_vvcrit_filename_parts(
    alpha: float, vvcrit: float, version: Literal["1.2", "2.5"] = "2.5"
):

    # alpha part
    filename_alpha_part = "afe_"
    filename_alpha_part += "m" if alpha < 0 else "p"
    filename_alpha_part += (
        f"{abs(alpha) * 10:.0f}" if version == "2.5" else f"{abs(alpha):0.1f}"
    )

    # vvcrit part
    filename_vvcrit_part = "vvcrit"
    filename_vvcrit_part += f"{vvcrit:0.1f}"

    return filename_alpha_part, filename_vvcrit_part


def generate_EEP_grid_for_alpha_vvcrit(
    alpha,
    vvcrit,
    processed_pattern="*.parquet",
    processed_path=EEP_PROCESSED_TRACKS_PATH_DEFAULT,
    grid_path=EEP_GRID_PATH_DEFAULT,
    version: Literal["1.2", "2.5"] = "2.5",
    save=True,
):

    filepaths = sorted(processed_path.glob(processed_pattern))
    alpha_vvcrit_files = []

    for file in filepaths:
        _, alpha_val, vvcrit_val = _parse_initfeh_alpha_vvcrit_from_name(
            file.name
        )
        alpha_val = np.round(alpha_val, decimals=1)
        if vvcrit_val == vvcrit and alpha_val == alpha:
            alpha_vvcrit_files.append(file)

    frames = [
        pd.read_parquet(file, engine="pyarrow") for file in alpha_vvcrit_files
    ]

    if len(frames) == 0:
        print(
            f"No files found in {processed_path.name} for alpha={alpha}, vvcrit={vvcrit}"
        )
        return

    alpha_vvcrit_df = pd.concat(frames, ignore_index=True)

    filename_alpha_part, filename_vvcrit_part = (
        _generate_alpha_vvcrit_filename_parts(alpha, vvcrit, version)
    )
    filename = f"{filename_alpha_part}_{filename_vvcrit_part}.grid.parquet"

    if save:
        alpha_vvcrit_df.to_parquet(grid_path / filename, compression="snappy")

    return alpha_vvcrit_df


# -------------------------------------------------------------------
# Generate New DataFrames for Each MIST EEP Track
# -------------------------------------------------------------------

EEP_RAW_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/raw_tracks/")
EEP_PROCESSED_PATH = Path(
    "/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/"
)
EEP_GRID_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/grids/")


def __main_step1_process_raw_eep_tracks__():

    # want to iterate through all subdirectories in the raw EEP path and process each directory
    for folderpath in EEP_RAW_PATH.iterdir():
        # first check if the parquet file for this directory already exists
        parquet_file = EEP_PROCESSED_PATH / f"{folderpath.name}.parquet"
        if parquet_file.exists():
            print(
                f"Parquet file already exists for directory: {folderpath.name}"
            )
            continue
        if folderpath.is_dir():
            print(f"Processing directory: {folderpath.name}")
            process_eep_directory(
                folderpath, save=True, output_path=EEP_PROCESSED_PATH
            )


def __main_step2_generate_eep_grids__():

    alpha_grid_values = [-0.2, 0.0, 0.2, 0.4, 0.6]
    vvcrit_values = [0.0, 0.4]

    for alpha in alpha_grid_values:
        for vvcrit in vvcrit_values:
            df = generate_EEP_grid_for_alpha_vvcrit(
                alpha,
                vvcrit,
                processed_path=EEP_PROCESSED_PATH,
                grid_path=EEP_GRID_PATH,
                save=True,
            )


# depending on what you want to run, you can comment out either step
if __name__ == "__main__":
    __main_step1_process_raw_eep_tracks__()
    __main_step2_generate_eep_grids__()
