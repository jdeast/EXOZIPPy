import re
import yaml
from pathlib import Path

import pandas as pd
import numpy as np
import astropy.units as u

from typing import Literal

from .parse_MIST_EEP_filenames import _generate_MIST_EEP_url, _parse_initfeh_alpha_vvcrit_from_filename

# -------------------------------------------------------------------
# MIST File I/O
# -------------------------------------------------------------------

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

# replace with your path to the directory containing the MIST EEP tracks
EEP_PATH = Path("/Volumes/Data/EEP_Tracks/")

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

    with open(filepath) as f:
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
        elif col_numbers_line is not None and col_names_line is None and line.startswith("#"):
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


def _add_or_grab_solar_values_to_MIST_yaml(EEP_path = EEP_PATH, 
                                   version: Literal["1.2", "2.5"] = "2.5"):

    # locate the YAML file corresponding to the MIST EEP grid
    if "models" in str(current_dir):
        # assume we're running this within the "models" directory
        yaml_filename = current_dir / f"MISTv{version}" / "EEPs" / f"MISTv{version}.grid.yaml"
    else:
        yaml_filename = current_dir.parent.parent / "models" / f"MISTv{version}" / "EEPs" / f"MISTv{version}.grid.yaml"

    if not yaml_filename.is_file():
        raise FileNotFoundError(f"YAML file not found at {yaml_filename}")

    # check if the YAML file already has the key "log_surf_fe_over_x_solar"
    with open(yaml_filename, "r") as file:
        grid_data = yaml.safe_load(file) or {}

    if "log_surf_fe_over_x_solar" in grid_data:
        return grid_data["log_surf_fe_over_x_solar"] # key already exists, no need to append

    # if the key does not exist, we need to compute the solar value and append it to the YAML file
    solar_folder, _ = _generate_MIST_EEP_url(initfeh=0.0, alpha=0.0, vvcrit=0.0, version=version)
    solar_path = EEP_path / f"MISTv{version}" / solar_folder

    # first check if tracks are in a nested "eeps" folder within the solar_path
    nested_folder = solar_path / "eeps"
    if nested_folder.is_dir():
        solar_path = nested_folder

    # grab any track from the folder
    filename = next(solar_path.glob("*.eep"))

    df_solar = read_mist_eep(filename)
    log_surf_fe_over_x_solar = np.log10(df_solar['surface_fe56'].values[0]) - np.log10(df_solar['surface_h1'].values[0])

    # otherwise, we will just append the new key-value pair to the YAML file
    grid_data = {}
    grid_data["log_surf_fe_over_x_solar"] = float(log_surf_fe_over_x_solar)

    with open(yaml_filename, "a") as file:
        yaml_text = yaml.safe_dump(grid_data)
        file.write(yaml_text)

    return grid_data["log_surf_fe_over_x_solar"]


# -------------------------------------------------------------------
# Create New DataFrame from MIST EEP Track
# -------------------------------------------------------------------

def _pad_or_trim_df(df, target_length=808):

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
        padding = pd.DataFrame(np.nan, index=range(n_missing), columns=df.columns)
        padding.iloc[:] = final_row.values

        padded_df = pd.concat([df, padding], ignore_index=True)
        padded_df.attrs = df.attrs.copy()
        return padded_df

def _add_logM_col(df, new_df):
    
    nrows = len(df)
    new_df["logM"] = np.full(nrows, np.log10(df.attrs["meta"]["initial_mass"]))
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
    log_surf_fe_over_x_star= np.log10(df['surface_fe56']) - np.log10(df['surface_h1'])
    feh = log_surf_fe_over_x_star - log_surf_fe_over_x_solar
    new_df["feh_mist"] = feh
    return

def _add_radius_col(df, new_df):

    radius = 10**df["log_R"]
    new_df["radius_mist"] = radius
    return

def _add_teff_col(df, new_df):

    teff = 10**df["log_Teff"]
    new_df["teff_mist"] = teff
    return

def _add_astroseismic_cols(df, new_df):

    new_df["delta_nu"] = df["delta_nu"].values
    new_df["nu_max"] = df["nu_max"].values
    return

def _add_age_and_turn_back_col(df, new_df):

    age = df["star_age"].values
    turn_back = np.zeros(len(age))

    # Find values that appear more than once
    vals, counts = np.unique(age, return_counts=True)
    repeated_vals = vals[counts > 1]

    # if no repeated values, just return the age array
    if repeated_vals.size == 0:
        new_df["age_mist"] = age
        new_df["turn_back"] = turn_back
        return
    
    # Get indices for all occurrences of those repeated values
    duplicate_bools = np.isin(age, repeated_vals)

    # ensure unique ages by adding a small increment to duplicates
    # add small increments to duplicates to ensure unique ages
    add_time = 1 * u.hr.to(u.yr)
    age[duplicate_bools] += add_time * np.arange(np.sum(duplicate_bools))
    turn_back[duplicate_bools] = 1 + np.arange(np.sum(duplicate_bools))

    new_df["age_mist"] = age
    new_df["turn_back"] = turn_back
    return

def _add_deep_dage_col(new_df):
    
    new_df["dEEP_dage"] = np.gradient(new_df["EEP"], new_df["age_mist"])
    return


def generate_new_df(df, target_length=808):

    # Define column headers and initialize df
    columns = ["logM", "EEP", "initfeh", 
               "feh_mist", "radius_mist", 
               "teff_mist", "delta_nu", "nu_max",
               "age_mist","dEEP_dage", "turn_back",
               ]
    
    new_df = pd.DataFrame(columns=columns)

    # pad or trim the original DataFrame to the target length
    pad_or_trim_df = _pad_or_trim_df(df, target_length=target_length)

    # fill in all the new columns
    _add_logM_col(pad_or_trim_df, new_df)
    _add_EEP_col(pad_or_trim_df, new_df)
    _add_initfeh_col(pad_or_trim_df, new_df)
    _add_feh_col(pad_or_trim_df, new_df)
    _add_radius_col(pad_or_trim_df, new_df)
    _add_teff_col(pad_or_trim_df, new_df)
    _add_astroseismic_cols(pad_or_trim_df, new_df)
    _add_age_and_turn_back_col(pad_or_trim_df, new_df)
    _add_deep_dage_col(new_df)

    return new_df