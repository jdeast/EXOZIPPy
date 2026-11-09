# Workflow for Building the NextGen Bolometric-Correction Tables

The SED component reads bolometric corrections BC_X(Teff, log g, [Fe/H], A_V) from one parquet table per facility:

```
models/NextGen/BCs/
    NextGen.grid.yaml          # the (teff, logg, feh, av) axes (tracked)
    2MASS.bc.parquet           # fetched from Zenodo on first use
    GAIA.bc.parquet
    ...                        # one per facility; bc_tables.py lists them
```

The tables are NOT tracked in git or shipped in the wheel: they are published on Zenodo and pinned (record id, size, md5) in `utilities/zenodo_assets.py` (record `nextgen_bc_tables`); `bc_tables.py` fetches the ones a fit reads on first use (`exozippy-fetch-bc-tables` pre-fetches all of them). After regenerating or extending tables here, publish them as a NEW VERSION of the Zenodo record and update the pins in `utilities/zenodo_assets.py` -- until then a fit refuses the locally changed table rather than overwriting it or silently using it (`components/sed/sed.md`).

Each table is long-format, one row per grid node, with columns `teff logg feh alpha Av Rv <filter columns>`. Filter columns are named by their MIST BC-column name (e.g. `2MASS_J`, `Gaia_G_DR2Rev`). `alpha` records the $[\alpha/\text{Fe}]$ of the spectrum a row was computed from; it is provenance, not a grid axis. `df.attrs["meta"]` (preserved by `pd.read_parquet`) holds the table metadata and, per filter column, its SVO id, zeropoint, flux weighting and generator. `components/sed/bc_grid.py` owns the format (`read_bc_table`, `write_bc_table`, `bc_table_path`).

These tables are built from the full-resolution BT-NextGen (AGSS2009) spectra[^1] in two steps, mirroring the MIST EEP workflow in `models/MIST/README.md`.

## 0. Get the raw spectra

`fetch_raw_spectra.py` downloads the BT-NextGen (AGSS2009) spectra from SVO[^1] as the ASCII files the pipeline reads (`.BT-NextGen.7.dat.txt`, ~12.5 MB each): it reads SVO's SSAP index of the model (cached as `svo_index.votable.xml` beside the spectra), maps every (teff, logg, feh) node of the grid onto a file id -- trying the alphas in `ALPHA_GRID_PTS` order, exactly as `find_spectrum_file` reads them back -- and skips what is already on disk. The index is also the pipeline's AUTHORITY on which nodes exist (see Padding below).

```bash
python -m exozippy.models.NextGen.fetch_raw_spectra --logg 5.5   # one plane
python -m exozippy.models.NextGen.fetch_raw_spectra              # the grid
```

The default location is set in `nextgen_spectra.py`:

```
SPECTRA_RAW_PATH_DEFAULT = Path("/Volumes/Data/Spectra/BT-NextGen_AGSS2009/")
```

Only the rectangular grid in `NextGen.grid.yaml` is needed (60 Teff x 12 log g x 11 [Fe/H] = 7920 nodes, of which the logg 5.5 plane above 5000 K -- 385 nodes -- has no spectrum and is padded, and four interior nodes have empty SVO files, see 1b). Where a node has no $[\alpha/\text{Fe}]=0$ spectrum, the next alpha in `ALPHA_GRID_PTS = [0, 0.2, -0.2, 0.4, 0.6]` is used and recorded in the `alpha` column.

## 1. Resample the raw spectra (`generate_NextGen_BC_Tables.py`, step 1)

Every raw spectrum is interpolated onto the common R = 20000 wavelength grid (0.03 - 30 micron) that `exozippy.filters.filter.Filter` resamples every filter profile onto, and the spectra for each [Fe/H] are saved as one parquet file:

```
SPECTRA_PROCESSED_PATH_DEFAULT = Path("/Volumes/Data/Spectra/BT-NextGen_AGSS2009_processed/")
    feh-4.0.spectra.parquet ... feh+0.5.spectra.parquet
```

> [!WARNING]
> Reading the raw ASCII spectra is the slow part (~1 s each, so ~2 hours single-process; `n_workers` in `process_raw_spectra_for_feh` parallelizes it). Each processed file is ~0.7 GB (~8 GB total). Step 1 is incremental (`process_missing_spectra`): for each [Fe/H] it reads only the `teff`/`logg` columns of the existing parquet and resamples only the (teff, log g) nodes it does not hold yet, so adding Teff, log g or [Fe/H] values to `NextGen.grid.yaml` costs only the new spectra. Extending A_V needs no new spectra, so step 1 is then a no-op.

**The step-1 product is published**, so steps 0 and 1 can be skipped: Zenodo [10.5281/zenodo.24033825](https://doi.org/10.5281/zenodo.24033825) holds the 11 processed parquets (flux cast to float32; 3.3 GB) and `wavelength.parquet`. Download them into one directory and pass it to step 2 as `processed_path`; `load_processed_spectra` reads them as they are. float32 moves a flux by at most 6e-8 relative wherever the spectrum carries light, and BCs recomputed from these files match the published tables to 8e-10 mag (`scripts/build_nextgen_spectra_bundle.job`, which also builds the bundle). The record is pinned in `utilities/zenodo_assets.py` as `nextgen_spectra_r20000` and listed there in `ON_DEMAND`: the test suite's prewarm and CI's cache step fetch every other pinned record, but not this one. A fit downloads it only to synthesize a column the published tables lack ("Filters with no table" below); `python -c "from exozippy.models.NextGen import bc_synthesis; bc_synthesis.fetch_all()"` pre-fetches it by hand.

## 1b. Padding: the grid nodes no spectrum exists for (`pad_missing_nodes`)

BT-NextGen's raw grid is ragged: the logg 5.5 plane exists only to 5000 K and the 6.0 plane to 4000 K. The loader (`bc_grid.complete_axes`) reads only the largest COMPLETE box of a table, so a half plane would be trimmed away -- or cost the hot end of the Teff axis -- and the ceiling would stay at logg 5.0, where an M dwarf below ~0.2 Msun sits above it (logg 5.16 at 0.11 Msun) and the grid barrier pushes its mass. So after step 1 and BEFORE step 2, `pad_missing_nodes` copies, for every grid node SVO lists no spectrum for (`unavailable_grid_nodes`, from the index -- never from the local raw directory, which is empty on most machines), the BC rows of the nearest lower-gravity plane that has one, at every Av and for every filter column, and records each padded node in the table's `meta["padding"]` (`alpha` is copied with the row: it is the provenance of the spectrum actually used). No star lives at logg 5.5 and 6000 K; the padded cells exist to keep the box rectangular and are never interpolated near a real star. Step 2 then plans nothing for those nodes. `process_missing_spectra(skip_unavailable=True)` is step 1's half of the same rule: it leaves the unavailable nodes out instead of raising. The 2026-10-20 extension to logg 5.5 (`scripts/build_nextgen_logg55.job`) padded 385 nodes per facility (35 Teff values from 5100 to 10000 K x 11 [Fe/H]).

A column ADDED later (a new filter on an existing facility, or a new facility) is padded by step 2 itself, because `pad_missing_nodes` skips a node it padded before and cannot fill a column that did not exist then. `generate_bc_tables` computes such a node's cells from the spectrum of the nearest lower-gravity node at the same (teff, feh), which gives the same numbers as copying that node's rows (a BC depends on its node only through the spectrum), and records the node in `meta["padding"]` once, with the columns padded there. Four INTERIOR nodes are also absent from SVO's index -- (4400 K, 3.0, -3.0), (4700, 5.0, -2.0), (7000, 0.0, -1.0), (7800, 2.0, +0.3); their SVO file ids (2922, 3417, 6508, 7077) return empty files -- although the published tables hold values for them, from the original spectra. A new column takes those nodes from the spectrum linearly interpolated between the two Teff neighbours at the same (logg, feh), recorded as `from_teff`; the row's `alpha` keeps the published value, which labels its other columns. It does this ONLY for nodes SVO's index lists no spectrum for; a node SVO lists but step 1 has not processed still raises. The 2026-11 HST/VISTA/DECam/Euclid VIS/MOA/Spitzer/UKIDSS columns (#495) were the first to need it. Tests: `tests/test_nextgen_fetch_pad.py`.

## 2. Compute the BC tables (`generate_NextGen_BC_Tables.py`, step 2)

`generate_bc_tables` runs `BolometricCorrection` (`bolometric_correction.py`) for every filter set in `FILTER_SETS` and writes `models/NextGen/BCs/{FACILITY}.bc.parquet`. The keys of `FILTER_SETS` are facilities and must equal the SVO id prefix of their filters (that prefix is how the loader finds a filter's table). `FILTER_SETS` lists every shipped column. Every column is Vega-referenced (`MAG_SYSTEM`, recorded as the table's `mag_system`), including the filters whose catalogs are natively AB (SLOAN, PAN-STARRS, GALEX, Euclid, and Roman -- whose photometry will be AB, per an email from Alison Duck, 2026-10-01; Kepler is recorded with no native system until it is resolved). The SED converts an AB row onto Vega itself (review 1.9.1) with the column's m_AB - m_Vega from `BCs/NextGen.magsys.csv`, which `write_magsys_table` writes beside the tables (and `generate_bc_tables` rewrites with them): per column, the filter's native system (`NATIVE_SYSTEM`) and the offset computed from the same profile, flux weighting and Vega zeropoint as the column's BCs. Not `filters/filter_magsys.txt`'s `mag(Vega/AB)`, which is MIST's and differs from ours by up to 0.03 mag. Adding a filter to `FILTER_SETS` means adding its `NATIVE_SYSTEM` entry too.

Step 2 is **incremental on every axis and every filter**: it first works out which (grid node, filter) cells the tables do not hold yet (`plan_bc_work`, which you can call on its own to see what a run would do), prints that plan, and computes only those cells. So:

- **To add filters**, add them to their facility's list in `FILTER_SETS` (or pass your own dict) and re-run step 2. Only the new columns are computed; the facility's other columns are not touched.
- **To extend an axis** (e.g. A_V up to 15), add the values to `NextGen.grid.yaml` and re-run step 2 (and step 1 first if the new values are Teff, log g or [Fe/H]). Only the new nodes are computed. An [Fe/H] with nothing to compute is never loaded, and at each node `BolometricCorrection` is called with only the filters and A_V values still missing there.
- **Re-running a finished table** computes nothing.

New cells are merged into the table cell by cell (`bc_grid.write_bc_table`), and each table is checkpointed after every [Fe/H], so an interrupted run resumes where it stopped. A column counts as already computed only if its metadata says this pipeline wrote it with the same spectra (`GENERATOR`, `SPECTRA_TAG`). A column written by `make_bc.py` or converted from the legacy text tables is recomputed in full, so no column mixes two pipelines. `overwrite=True` recomputes everything.

A table may therefore be ragged for a while, e.g. 2MASS extended to A_V = 15 and GAIA not yet. A missing cell is stored as NaN. The fit only uses the part of the grid that **every filter it asks for** covers: the SED's A_V bound is set from that coverage (`bc_grid.peek_grid_axes(filters=...)`), so a fit using only 2MASS can reach A_V = 15 while one that also uses Gaia stops at 6.

Run both steps with

```
poetry run python -m exozippy.models.NextGen.generate_NextGen_BC_Tables
```

(comment out either step in `__main__` to run just one).

### What is computed

$$\text{BC}_X = 2.5\log_{10}\left[\frac{L_0}{4\pi(10\,\text{pc})^2\,\sigma T_\text{eff}^4}\cdot\frac{\int F_\lambda\, e^{-\tau_\lambda}\, S_\lambda\, w_\lambda\, d\lambda}{\text{ZP}_X \int S_\lambda\, w_\lambda\, d\lambda}\right]$$

with $L_0 = 3.0128\times10^{28}$ W (IAU 2015), the SVO Vega $F_\lambda$ zeropoint $\text{ZP}_X$ (specified value when SVO quotes one, else calculated), and $\tau_\lambda = \frac{A_\lambda/A_V}{1.086}\,A_V$ from `models/extinction_law.ascii` (tabulated in microns, $R_V = 3.1$). The weight $w_\lambda$ follows the filter's SVO `DetectorType`: $w_\lambda = 1$ for an energy counter, $w_\lambda = \lambda$ for a photon counter. That is the convention SVO computes its zeropoints with. The module docstring of `bolometric_correction.py` records how this compares with the notebook the class came from, with the previously shipped tables, and with the MIST BC tables.

## The runtime spectral grid (`runtime_spectra.py`)

The SED's fitted-R_V mode (`sed: {fit_rv_dust: true}`; `components/sed/sed.md`, "The spectral mode") does not read the BC tables' values. It interpolates the SPECTRUM in (Teff, logg, [Fe/H]), reddens it at each star's own A_V and R_V, and integrates the bands at every evaluation (`components/sed/spectral_bc.py`). This is the grid it interpolates: every node of `NextGen.grid.yaml`, resampled from the step-1 spectra onto 20722 log-uniform bins at R = 3000 over 0.03-30 micron. Each bin holds the flux-conserving mean F_lambda (the trapezoid integral of the R = 20000 spectrum between the bin edges, over the bin width), stored as log10 in float32: `runtime/NextGen.R3000.logflux.npy` (60 x 12 x 11 x 20722, 656 MB, memory-mapped) and `runtime/NextGen.R3000.meta.json` (axes, edges, per-node alpha, padding). Published on Zenodo (DOI 10.5281/zenodo.24060411, record `nextgen_runtime_spectra` in `utilities/zenodo_assets.py`) and fetched into the default model root on first use; an explicit `model_root:` is used as given. Rebuild: `qsub scripts/build_nextgen_runtime_grid.job` (~10 min from the step-1 spectra).

**Why R = 3000 and log flux** (the 2026-11-07 feasibility study, `scripts/runtime_sed_feasibility.py`). A band flux from bins of resolution R -- the bin-mean spectrum times the bin integral of the filter's weighted profile, reddened at the bin centre -- against the R = 20000 integral, worst case over 28 nodes and A_V 0-20, all 93 shipped filters: 13.8 mmag at R = 300 (23 filters over 1 mmag), 1.4 at R = 1000 (4 narrow bands over), 0.43 at R = 3000 (none). Off-node, interpolating the LINEAR flux is 2-3x worse than the BC tables for M dwarfs (held-out nodes, Teff runs), while the log flux matches them, so the grid stores log10 F and the SED interpolates it trilinearly. Grid spacing, not resolution, is then the accuracy floor, the same for both paths.

**Validation against the shipped tables** (`scripts/validate_runtime_grid.py`, 2026-11-08). With the tables' own extinction law (`SpectralBC(..., law="shipped")`), the runtime BCs reproduce every table column at the tables' nodes: median over filters of the worst error 0.02 mmag, worst 0.69 mmag (GALEX NUV), none over 1 mmag, 66 nodes x A_V 0, 1, 8, 20. The exceptions are the four interior holes (1b): there the tables hold values from the original spectra and the runtime grid an interpolation along Teff. Three agree with that interpolation to the usual 5-50 mmag of curvature; the fourth, (7000 K, 0.0, -1.0), is an OUTLIER in the tables -- an alpha = 0.2 spectrum between alpha = 0.4 neighbours, its J BC 0.18 mag off the line through them -- so there the runtime grid is the smoother of the two.

**Padding** is step 2's (1b): the 385 logg 5.5 nodes above 5000 K from the nearest lower-gravity spectrum, the four interior holes along Teff (`_pad_spectrum`); all 389 are listed in the metadata.

## Legacy text tables

Before this pipeline, the tables shipped as one text file per facility and [Fe/H] (`BCs/{FACILITY}/feh{+/-X.X}_afe+0.0.{FACILITY}`), photon-weighted for every filter (0.07 - 0.18 mag off for the energy-counter bands: Gaia, TESS, WISE W3). They were converted to parquet once, superseded by step 2, and deleted together with their converter (`convert_legacy_BC_tables.py`) when the tables moved to Zenodo; both remain in git history.

## Filters with no table

A fit that needs a filter with no published column gets one SYNTHESIZED from the published step-1 spectra above, by step 2 itself -- when the filter is a `band:` block's and some component reads that band's SED flux (a stated microlensing zeropoint, a transit `seddeblend`, an astrometric flux fraction; `Component.sed_flux_bands`). A band filter nothing reads simply stays out of the grid. `bc_synthesis.synthesize_column` downloads the 3.3 GB of processed spectra on first use, runs `generate_bc_tables` for that one filter into a scratch root -- padding the nodes no spectrum exists for exactly as above, with which nodes those are read from the bundle (a complete step-1 product) rather than from SVO's index -- and merges the column, with this pipeline's `GENERATOR`/`SPECTRA_TAG` metadata and its padding record, into a per-user cache table, `<machine cache>/bc_synthesized/NextGen/BCs/{FACILITY}.bc.parquet`, beside a `NextGen.magsys.csv` of its own. The package's published tables are never written. The band must state the filter's native magnitude system (`mag_system: Vega` or `AB`), since a synthesized filter has no `NATIVE_SYSTEM` entry; the offset is computed as `write_magsys_table` computes it (`native_systems=` carries the stated system through). A file lock per (facility, filter) keeps two fits from computing one column at once. The loader (`bc_grid.locate_bc_column`) reads published and cached columns alike, published first, and treats a cached column as missing once it is no longer current (another pipeline, or nodes the grid yaml has gained). The rules and their reasons are in `components/sed/sed.md`, "Which band filters need a BC column".

What is still refused, by `bc_grid.build_bc_grid` with a message naming the request path: a filter listed in a `.sed` file with no column, and any filter when synthesis does not apply (an explicit `model_root:`, or the machine cache switched off). The R = 150 path is not coming back: `components/sed/make_bc.py` once synthesized columns from the downsampled spectra on Zenodo, ~2 percent wrong, and that hook was removed on 2026-09-28. `make_bc.py` remains as a manual development tool (`scripts/make_bc_tables.py`); its columns say so in their metadata, and it never writes into a column another pipeline produced unless called with `overwrite=True` (`--overwrite`). A filter many fits will use is still best requested, generated here and published with the tables.

## Reading the tables during a fit

`bc_grid.build_bc_grid` reads only what the fit needs. For each facility it reads the grid-key columns (to find the grid points), then **only the requested filter columns**, and **only the rows inside the star's `teffsed`, `feh` and `av` bounds** plus the grid point bracketing each bound. Both restrictions are passed to the parquet reader, so the other columns and rows are never read into memory. For example, an A_V upper limit of 0.09 reads A_V = 0, 0.05 and 0.1. `loggsed` is derived and only soft-bounded, so its axis is always read in full. On disk the rows are sorted with A_V outermost, so rows past an A_V limit mostly sit in row groups the reader skips entirely.

[^1]: https://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-nextgen-agss2009 (Allard et al. 2011, 2012; Asplund et al. 2009).
