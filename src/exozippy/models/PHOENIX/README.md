# The PHOENIX limb-darkening tables

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

```
models/PHOENIX/
    specint.py                       # the archive: names, fetch, read
    generate_PHOENIX_LD_Tables.py    # the generator (streams the archive)
    LDs/
        PHOENIX.grid.yaml            # the (teff, logg, feh, av) axes (tracked)
        linear/{FACILITY}.ld.parquet     # one column per filter: u
        quadratic/{FACILITY}.ld.parquet  # two per filter: {name}__u1, {name}__u2
        intensities/feh{+X.X}.intensities.parquet  # the band-integrated I(mu)
```

The tables and the intensities are NOT tracked (git-ignored, like the BC
tables); the tables are to be published on Zenodo and fetched on first
use. Their format is the BC tables' (`bc_grid.write_bc_table`; columns
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

The PHOENIX-ACES grid is not the BT-NextGen grid the BC tables use
(different code versions and abundance tables). This is the v1 mismatch
the 2026-08-04 plan accepted; the BC tables stay NextGen until a
flux-plus-intensity product from one atmosphere set exists.

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

## Running it

```bash
# ~7000 nodes streamed; a compute node with network, 32 workers, ~1 h
python -m exozippy.models.PHOENIX.generate_PHOENIX_LD_Tables --workers 32
qsub scripts/build_phoenix_ld.job        # the same on Hydra
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
