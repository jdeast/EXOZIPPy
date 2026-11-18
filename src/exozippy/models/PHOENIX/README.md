# The PHOENIX limb-darkening and bolometric-correction tables

The band component's limb-darkening tie (`band.star_constrains_ld`, see
`components/sed/sed.md`, "The limb-darkening tie") can read two table
sources. The default is the Claret tables (ATLAS, least squares; Teff
3500-10000 K, logg 2.0-5.0). `ld_source: phoenix` reads the tables in this
directory, which EXOZIPPy generates itself from the PHOENIX-ACES specific
intensities of Husser et al. (2013), through the SAME filter profiles,
detector weighting and extinction law as the NextGen bolometric
corrections. They reach 2300 K and logg 6, cover every filter the BC
tables ship inside 500-26000 A (WISE and Johnson M are outside), and carry
A_V as a fourth axis.

An SED file saying `model: PHOENIX` reads the bolometric-correction
tables built from the same stream of the same archive (section "The
bolometric-correction tables" below; `components/sed/sed.md`, "Choosing
the atmospheres").

```
models/PHOENIX/
    specint.py                       # the archive: names, fetch, read
    generate_PHOENIX_LD_Tables.py    # the generator (streams the archive)
    ld_tables.py                     # fetch of the published LD tables
    bc_tables.py                     # fetch of the published BC tables
    BCs/
        PHOENIX.grid.yaml            # the BC axes (= the LD axes) + coverage (tracked)
        PHOENIX.magsys.csv           # per-column magnitude-system record (tracked)
        plot.py                      # the SED figure (NextGen spectra; tracked)
        {FACILITY}.bc.parquet        # one column per filter, NextGen format
    LDs/
        PHOENIX.grid.yaml            # the (teff, logg, feh, av) axes (tracked)
        linear/{FACILITY}.ld.parquet     # one column per filter: u
        quadratic/{FACILITY}.ld.parquet  # two per filter: {name}__u1, {name}__u2
        intensities/feh{+X.X}.intensities.parquet  # the band-integrated I(mu)
```

The tables and the intensities are NOT tracked (git-ignored, like the BC
tables). The tables are published on Zenodo (record `phoenix_ld_tables`
in `utilities/zenodo_assets.py`, DOI 10.5281/zenodo.23522329, cite as
`Eastman:2026ld`) and fetched on first use by `ld_tables.py`, per
facility and law, through `components/band/ldgrid.py`; the record's flat
file names `{FACILITY}.{law}.ld.parquet` land as `LDs/{law}/{FACILITY}.ld.parquet`.
`exozippy-fetch-ld-tables` pre-fetches the whole set for offline use. The
band intensities (2.7 GB) are not published; they live on the machine that
generated them. Their format is the BC tables' (`bc_grid.write_bc_table`; columns
`teff logg feh alpha Av Rv` plus the coefficient columns; `alpha` is 0;
`df.attrs["meta"]` holds the method, the intensities' provenance, and
`padding`, the list of interior holes filled), so the incremental merge,
`complete_axes` and the publish path all transfer.

## Why specific intensities, and why not the NextGen spectra

A bolometric correction needs the disk-integrated flux F(lambda), which
is what the SVO BT-NextGen spectra are. Limb darkening needs the emergent
specific intensity I(lambda, mu) -- an extra dimension the flux grid does
not have. The PHOENIX-ACES archive serves one FITS per node: I on 25500
wavelengths (500-26000 A at 1 A) by 78 angles, 16 MB each, about 7000
nodes (teff 2300-12000 K, logg 0-6, [Fe/H] -4 to +1, alpha = 0). The
generator streams each file into `$TMPDIR`, integrates it and deletes
it; the raw archive (~110 GB) is never rehosted. What is kept is the
band-integrated intensities, about 0.6 MB per node (float32), from which
any other law or fit recipe can be refitted without the archive.

The PHOENIX-ACES grid is not the BT-NextGen grid the default BC tables
use (different code versions and abundance tables). This was the v1
mismatch the 2026-08-04 plan accepted; since review 8.9.12 the same
stream also writes PHOENIX BC tables, so `model: PHOENIX` with
`ld_source: phoenix` describes a star with one atmosphere.

## What is computed

For a node and a filter set (one facility's filters), with the profiles
`S_w(lambda)` already weighted per SVO `DetectorType` (energy or photon
counter, as for the BCs) and `tau(lambda; A_V)` from
`models/extinction_law.ascii` at R_V = 3.1:

    I_band(mu; A_V) = int I(lambda, mu) exp(-tau) S_w dlambda,

normalized to `I_band(1; A_V) = 1`. A_V tilts the weighting inside the
band -- extinction is stronger at the blue edge, so the band's effective
wavelength moves redward, where limb darkening is weaker -- and for
microlensing sight lines (A_V > 5) and the Roman wide bands the effect
reaches the 0.05 width of the tie, which is why A_V is an axis (the
Claret tables have none).

Each profile is then fitted:

* **Quasi-spherical remap** (Claret, Hauschildt & Witte 2012). PHOENIX
  models are spherical: rays near the geometric edge traverse only
  tenuous upper atmosphere, so I(mu) collapses over the last few percent
  of the disk. The limb a transit or finite-source code assumes is at the
  TOP of that cliff. Per profile the limb is placed at the mu of steepest
  descent (max dI/dmu, searched at mu < 0.5) and mu is remapped so it sits
  at mu' = 0: `mu' = sqrt((mu^2 - mu_limb^2) / (1 - mu_limb^2))`.
* **Flux-conserving fit.** The law conserves the remapped profile's flux
  `F = int_0^1 I(mu') mu' dmu'` exactly. For the linear law that fixes
  `u = 3 - 6F` outright; for the quadratic law `u1 = (3 - 6F) - u2/2` and
  `u2` minimizes the mu'-weighted least-squares residual. The mu weight
  sends the limb's contribution to zero, so the fit is insensitive to any
  residual cliff, and the conserved flux is the observable (blocked flux
  in a transit, magnified flux in a finite-source event).

`tests/test_phoenix_ld.py` pins both: a profile that IS a quadratic law
comes back exactly, and a profile with a cliff comes back to the
cliff-free coefficients after the remap.

## Validation against Claret (2026-10-20)

`scripts/compare_ld_sources.py`, 30 (band, node) pairs over TESS, V,
Kepler, Cousins I, 2MASS J and Sloan g at five nodes inside both boxes,
A_V = 0, PHOENIX minus Claret (ATLAS, least squares):

| quantity | mean +/- sd |
|---|---|
| u1 + u2 (the limb intensity) | -0.012 +/- 0.025 |
| linear u | -0.025 +/- 0.043 |
| u1 alone | +0.080 +/- 0.055 |
| u2 alone | -0.091 +/- 0.041 |

The limb intensity and the linear coefficient agree inside the tie's 0.05
width; the split between u1 and u2 differs by ~0.08 with opposite signs,
which is the known difference between a flux-conserving fit and a
least-squares one on the same profile (the two laws' fluxes agree, their
shapes near the limb do not). The largest departures are at the 3600 K
node, the edge of the ATLAS tables. The first pass of the generator
fitted every node of a [Fe/H] plane on one node's mu grid and this same
comparison read +0.20 in u1 + u2, which is how the per-node mu was found.

The tables' complete box is Teff 2300-9400 K, logg 2.0-6.0, [Fe/H] -4 to
+1, A_V 0-20 (60 x 9 x 9 x 20 nodes): the hot end above 9400 K and the
planes below logg 2.0 are the archive's incomplete edges, which
`complete_axes` trims; 284 interior holes per table were filled along
Teff. A_V matters at the level expected: Bessell V at the Sun moves from
(0.545, 0.163) at A_V = 0 to (0.514, 0.169) at A_V = 10.

## Holes

The archive is not a rectangle: hot stars are absent at the lowest
gravities and a few interior nodes are missing (5000 K at several
metallicities). An interior hole is filled by linear interpolation ALONG
TEFF of the band-integrated intensities of its two nearest available
neighbours at the same (logg, feh), then fitted like any other node, and
recorded in `meta["padding"]`. Edge planes with no such neighbours stay
absent; the loader's complete box (`bc_grid.complete_axes`) trims them.

## The bolometric-correction tables

`--products bc` (the default writes both) computes, per node and
filter, NextGen's BC (`models/NextGen/bolometric_correction.py`) with two
inputs replaced:

    BC_X = 2.5 log10[ L0 / (4 pi (10 pc)^2 F_bol)
                      * int(F_lam e^-tau S dlam) / (ZP_X int(S dlam)) ]

* the band flux is 2 pi int I_band(mu) mu dmu of the unnormalized band
  intensities above (the intensities are per centimetre of wavelength;
  1e-8 makes them per Angstrom, NextGen's unit), so the limb darkening
  and the flux of a star come from one atmosphere;
* F_bol is the model's OWN bolometric flux, not sigma Teff^4. A PHOENIX
  flux is given at the model's outer radius, where it is smaller than
  sigma Teff^4 by (R_Teff / R_out)^2 -- 0.985 at (5800 K, 4.5), 0.93 at
  (4500, 2.0), 0.78 at (4000, 0.5) over 500 A - 5.5 um -- and only the
  ratio of band to bolometric flux at one radius is physical. F_bol is
  read from the archive's flux spectrum of the node (HiResFITS, 6 MB,
  500 A - 5.5 um, plus a Rayleigh-Jeans tail of 0.1-3 percent;
  `specint.bolometric_flux`), which equals the intensities' flux to 1e-4.

With F_bol in the denominator, the 2.6 um end of the intensities limits
which filters get a column (99 percent of the weighted bandpass inside
500-26000 A: GALEX through 2MASS Ks; not WISE, Spitzer IRAC or Johnson
M), not their accuracy. A node needs both archives; an interior hole is
filled by interpolating the BCs along Teff (`meta["padding"]`). The
unnormalized band fluxes and F_bol are added to the intensities
intermediate (`band_flux`, `fbol`, `fbol_tail_fraction`); an intermediate
written before them has neither, and its nodes are streamed again for
`bc`.

A subset check (2026-11-18; 2MASS, Gaia DR3, Bessell V, Cousins I and
TESS at [Fe/H] = 0), PHOENIX minus the published NextGen tables:
0.005-0.04 mag at (5800 K, 4.0-4.5); 0.06-0.11 mag at (4500, 2.0);
0.29 mag at (4000, 0.5); 0.02-0.14 mag at (3000, 4.5), largest in Gaia
BP. With sigma Teff^4 in place of F_bol the giants would agree with
NextGen to 0.02-0.04 mag -- the NextGen tables divide by sigma Teff^4,
and their spectra integrate to 0.77 sigma Teff^4 at (4000, 0.5) -- while
MIST's BCs (ATLAS, normalized to sigma Teff^4) agree with the F_bol
version: at (4000, 0.5) V -0.894 / -0.990 / -1.290 and J 1.601 / 1.606 /
1.315 for MIST / PHOENIX / NextGen.

The tables are published as Zenodo record `phoenix_bc_tables`
(`utilities/zenodo_assets.py`, PENDING until it exists) and fetched per
facility on first use by `bc_tables.py`; `exozippy-fetch-phoenix-bc-tables`
pre-fetches them. `PHOENIX.magsys.csv` is written by the same run and
depends only on the filter profiles, so it is tracked and identical, row
for row, to NextGen's for every shared column.

## Running it

```bash
# ~7000 nodes streamed; a compute node with network, 32 workers, ~1 h
python -m exozippy.models.PHOENIX.generate_PHOENIX_LD_Tables --workers 32
qsub scripts/build_phoenix_ld.job        # the same on Hydra
# the BC tables only (also streams each node's 6 MB flux spectrum)
python -m exozippy.models.PHOENIX.generate_PHOENIX_LD_Tables --workers 32 --products bc
qsub scripts/build_phoenix_bc.job        # the same on Hydra
```

Resumable and incremental: a rerun computes only the nodes the tables
lack, and the intensities of a [Fe/H] already streamed are read back from
`LDs/intensities/` instead.

## Reading the tables during a fit

`components/band/ldgrid.py` is the band side: `ld_table_lookup` maps a
band's `filter:` label to its facility and column(s) through the filter
alias table, `load_ld_table` reads one filter's complete box into an
`LDTable` (axes `teff, logg, feh, av`), and `interpolate` /
`interpolate_node` are the numpy and PyTensor evaluations the start seed
and the model share. The band reads the star's A_V when it has one (an
`sed:` block in the topology); without one the tables are read at
A_V = 0 and no A_V bound is placed. Cite as `Husser:2013` for the
atmospheres and `Claret:2012` for the method.
