# Workflow for Downloading and Processing MIST EEP Tracks

## 1. **Run `download_MIST_EEPs.py`**

You can visit [MIST.SCIENCE](https://mist.science/model_grids.html#eeps) to determine which evolutionary tracks you would like to download.  Currently, there are two options: MISTv1.2 and MISTv2.5.  MISTv2.5 has an added dimension of alpha enhancement, and is the default assumed in `download_MIST_EEPs.py`.  You can specify which values of $[\text{Fe}/\text{H}]_\text{initial}$, $[\alpha/\text{Fe}]$, and $v/v_\text{crit}$ you would like to download by changing the values in the arrays on Lines 37-42, which by default is set to:

```
initfeh_vals = [ -4.0,  -3.5,  -3.0, -2.75,  
        -2.5, -2.25,  -2.0, -1.75,  
        -1.5, -1.25, -1.0, -0.75,  
        -0.5, -0.25,   0.0,  0.25,   0.5]
alpha_vals = [-0.2, 0.0, 0.2, 0.4, 0.6]
vvcrit_vals = [0.0, 0.4]
```

You also should set the path for where you want these tracks saved on Line 33:

```
EEP_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/raw_tracks/") # change path to where you want to store the EEP tracks
```

> [!WARNING]
> Downloading the full raw grid of tracks for MISTv2.5 is ~186 GB and can take multiple hours to fully download.

## 2. **Run `generate_MIST_EEP_Tables.py`**

This file will do two things: 

1. It will process the raw EEP tracks into ``.parquet`` files[^1], consolidating all the different mass tracks for one combination of $[\text{Fe}/\text{H}]_\text{initial}$, $[\alpha/\text{Fe}]$, and $v/v_\text{crit}$ into a single file.

2. It will further consolidate all of the different $[\text{Fe}/\text{H}]_\text{initial}$ tracks for one combination of $[\alpha/\text{Fe}]$ and $v/v_\text{crit}$ into a single larger grid, now with the file extension: ``.grid.parquet``.  For now, we will only further process the $[\alpha/\text{Fe}]=0.0$ and $v/v_\text{crit}=0.0$ grid.

By default, Step 1 will process all the raw EEP tracks contained in folders within `EEP_RAW_PATH`, set on Line 376, and save the processed `.parquet` files within `EEP_PROCESSED_PATH`, set on Line 377:

```
EEP_RAW_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/raw_tracks/")
EEP_PROCESSED_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
```
**Note:** `EEP_RAW_PATH` will likely be the same location as you set for `EEP_PATH` in Step 1.

> [!WARNING]
> This initial processing of the raw EEP tracks into the `.parquet` files can take multiple hours to run.

The processing of these `.parquet` files into `grid.parquet` files is **much** faster (on the order of seconds to minutes), and you can set which combination of $[\alpha/\text{Fe}]$ and $v/v_\text{crit}$ values you want grids processed for by changing the values set on Lines 396-397:

```
alpha_grid_values = [-0.2, 0.0, 0.2, 0.4, 0.6]
vvcrit_values = [0.0]
```

You also need to set the path where you would like to save the `grid.parquet` files on Line 378:

```
EEP_GRID_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/grids/")
```

The amount of space required for:

- All processed `.parquet` files: ~1.5 GB
- A single `grid.parquet` file: ~130 MB

## 3. **Run `find_interpolatable_missing_grid_points.py`**

Unfortunately, at this time, a full grid  of values (albeit, it can be unevenly spaced) is required for `exozippy` models that involve the interpolation of values over a grid of variables.  These next few steps are meant to identify and mitigate any holes within the grid that we can.

You must again set the paths where you saved your `.parquet` and `.grid.parquet` files, and the path to where you want to save the `.csv` files tracking the missing grid points (Lines 234-236):

```
EEP_PROCESSED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_GRID_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/grids/")
MISSING_GRID_POINTS_PATH = current_dir / "MISTv2.5" / "EEPs" / "MissingGridPoints" 
```

You also can set which values you want to run this process for (Lines 241-242):

```
alpha_grid_values = [-0.2, 0.0, 0.2, 0.4, 0.6]
vvcrit_values = [0.0]
```

## 4. **Run `generate_interpolated_missing_tracks.py`**

This file will generate any missing tracks that are able to be interpolated along one of our grid axes.  You will need to set your paths again, along with the path where you want to save the interpolated tracks (Lines 164-166):

```
MISSING_GRID_POINTS_PATH = current_dir / "MISTv2.5" / "EEPs" / "MissingGridPoints"
EEP_PROCESSED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_INTERPOLATED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/interpolated_tracks/")
```

You can also change which $[\alpha/\text{Fe}]$ and $v/v_\text{crit}$ values you want to generate these tracks for (Line 169).  By default, we only generate the missing tracks for $[\alpha/\text{Fe}]=0.0$ and $v/v_\text{crit}=0.0$.

```
generate_missing_tracks_for_alpha_vvcrit(alpha=0.0, vvcrit=0.0, ...)
```

## 5. **Run `merge_interpolated_missing_tracks.py`**

This file will merge the tracks that we interpolated in Step 4.  You will need to set your paths again (Lines 129-131):


```
EEP_PROCESSED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/processed_tracks/")
EEP_INTERPOLATED_TRACKS_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/interpolated_tracks/")
EEP_GRID_PATH = Path("/Volumes/Data/EEP_Tracks/MISTv2.5/grids/")
```

You can also change which values of $[\alpha/\text{Fe}]$ and $v/v_\text{crit}$ tracks you want merged into your grid (Line 134).  By default, we only generate the merge tracks for $[\alpha/\text{Fe}]=0.0$ and $v/v_\text{crit}=0.0$.

```
merge_interpolated_tracks_for_alpha_vvcrit(alpha=0.0, vvcrit=0.0, ...)
```

[^1]: You can read more about these files and the Python package ``pyarrow`` that manages them [here](https://arrow.apache.org/docs/python/index.html). 