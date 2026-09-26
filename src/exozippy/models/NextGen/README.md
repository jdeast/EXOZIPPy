# Workflow for Building the NextGen Bolometric-Correction Tables

The SED component reads bolometric corrections BC_X(Teff, log g, [Fe/H], A_V) from one parquet table per facility:

```
models/NextGen/BCs/
    NextGen.grid.yaml          # the (teff, logg, feh, av) axes
    2MASS.bc.parquet
    GAIA.bc.parquet
    Generic.bc.parquet
    Keck.bc.parquet
    TESS.bc.parquet
    WISE.bc.parquet
```

Each table is long-format, one row per grid node, with columns `teff logg feh alpha Av Rv <filter columns>`. Filter columns are named by their MIST BC-column name (e.g. `2MASS_J`, `Gaia_G_DR2Rev`). `alpha` records the $[\alpha/\text{Fe}]$ of the spectrum a row was computed from; it is provenance, not a grid axis. `df.attrs["meta"]` (preserved by `pd.read_parquet`) holds the table metadata and, per filter column, its SVO id, zeropoint, flux weighting and generator. `components/sed/bc_grid.py` owns the format (`read_bc_table`, `write_bc_table`, `bc_table_path`).

These tables are built from the full-resolution BT-NextGen (AGSS2009) spectra[^1] in two steps, mirroring the MIST EEP workflow in `models/MIST/README.md`.

## 0. Get the raw spectra

Download the BT-NextGen (AGSS2009) spectra from SVO[^1] as ASCII (`.BT-NextGen.7.dat.txt`, ~13 MB each). The default location is set in `nextgen_spectra.py`:

```
SPECTRA_RAW_PATH_DEFAULT = Path("/Volumes/Data/Spectra/BT-NextGen_AGSS2009/")
```

Only the rectangular grid in `NextGen.grid.yaml` is needed (60 Teff x 11 log g x 11 [Fe/H] = 7260 spectra). Where a node has no $[\alpha/\text{Fe}]=0$ spectrum, the next alpha in `ALPHA_GRID_PTS = [0, 0.2, -0.2, 0.4, 0.6]` is used and recorded in the `alpha` column.

## 1. Resample the raw spectra (`generate_NextGen_BC_Tables.py`, step 1)

Every raw spectrum is interpolated onto the common R = 20000 wavelength grid (0.03 - 30 micron) that `exozippy.filters.filter.Filter` resamples every filter profile onto, and the spectra for each [Fe/H] are saved as one parquet file:

```
SPECTRA_PROCESSED_PATH_DEFAULT = Path("/Volumes/Data/Spectra/BT-NextGen_AGSS2009_processed/")
    feh-4.0.spectra.parquet ... feh+0.5.spectra.parquet
```

> [!WARNING]
> Reading the raw ASCII spectra is the slow part (~1 s each, so ~2 hours single-process; `n_workers` in `process_raw_spectra_for_feh` parallelizes it). Each processed file is ~0.7 GB (~8 GB total). Step 1 is resumable: an [Fe/H] whose parquet already exists is skipped.

## 2. Compute the BC tables (`generate_NextGen_BC_Tables.py`, step 2)

`generate_bc_tables` runs `BolometricCorrection` (`bolometric_correction.py`) at every grid node for every filter set in `FILTER_SETS`, with all 13 A_V values per spectrum at once, and writes `models/NextGen/BCs/{FACILITY}.bc.parquet`. The keys of `FILTER_SETS` are facilities and must equal the SVO id prefix of their filters (that prefix is how the loader finds a filter's table). The defaults reproduce every column that shipped before this pipeline existed. To add filters, add them to `FILTER_SETS` (or pass your own dict) and re-run step 2 only; this takes minutes. Columns of an existing table that are not being regenerated are kept unchanged.

Run both steps with

```
poetry run python -m exozippy.models.NextGen.generate_NextGen_BC_Tables
```

(comment out either step in `__main__` to run just one).

### What is computed

$$\text{BC}_X = 2.5\log_{10}\left[\frac{L_0}{4\pi(10\,\text{pc})^2\,\sigma T_\text{eff}^4}\cdot\frac{\int F_\lambda\, e^{-\tau_\lambda}\, S_\lambda\, w_\lambda\, d\lambda}{\text{ZP}_X \int S_\lambda\, w_\lambda\, d\lambda}\right]$$

with $L_0 = 3.0128\times10^{28}$ W (IAU 2015), the SVO Vega $F_\lambda$ zeropoint $\text{ZP}_X$ (specified value when SVO quotes one, else calculated), and $\tau_\lambda = \frac{A_\lambda/A_V}{1.086}\,A_V$ from `models/extinction_law.ascii` (tabulated in microns, $R_V = 3.1$). The weight $w_\lambda$ follows the filter's SVO `DetectorType`: $w_\lambda = 1$ for an energy counter, $w_\lambda = \lambda$ for a photon counter. That is the convention SVO computes its zeropoints with. The module docstring of `bolometric_correction.py` records how this compares with the notebook the class came from, with the previously shipped tables, and with the MIST BC tables.

## Legacy text tables

Before this pipeline, the tables shipped as one text file per facility and [Fe/H] (`BCs/{FACILITY}/feh{+/-X.X}_afe+0.0.{FACILITY}`). `convert_legacy_BC_tables.py` converts those into the parquet format unchanged, so the package keeps working until step 2 has been run. Those text tables were photon-weighted for every filter, which is 0.07 - 0.18 mag off for the energy-counter bands (Gaia, TESS, WISE W3); regenerate them with this pipeline.

## Filters with no table: `components/sed/make_bc.py`

A fit that asks for a filter with no column triggers `make_bc.py`, which synthesizes the column from the downsampled (R = 150) spectra on Zenodo and merges it into the facility's table. It is the fallback for users without the full-resolution spectra; its columns say so in their metadata.

[^1]: https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-nextgen-agss2009 (Allard et al. 2011, 2012; Asplund et al. 2009).
