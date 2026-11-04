# HD 114762: Hipparcos + Gaia DR4 epoch astrometry, with the frame tie

HD 114762 (HIP 64426, Gaia DR3 3937211745905473024; G = 7.15, 39 pc) is a
metal-poor F9 star whose 84-day companion HD 114762 b (Latham et al. 1989)
was long counted among the first brown-dwarf/planet candidates (m sin i
~ 11 M_J); the Gaia DR3 astrometric two-body solution (P = 83.74 d,
e = 0.32, a0 = 1.80 mas, i = 18.9 deg) makes it a low-mass star on a
nearly face-on orbit.  It is one of the twelve sources of the Gaia DR4
pre-release AND a Hipparcos star, so it is the shipped demonstration of an
epoch-level Hipparcos + Gaia fit of one star: the 1991 Hipparcos abscissae
and the 2014.6-2019.9 DR4 abscissae share one position, proper motion,
parallax and photocenter orbit, and the 25-year gap between them is the
lever arm the Hipparcos-Gaia Catalog of Accelerations (Brandt 2018, 2021)
uses -- here from the epochs.

```
poetry run exozippy hd114762.yaml
```

## Data (both written by `exozippy-getdata`)

- `HIP64426.Hipparcos.astrom`: the Hipparcos 1997 Intermediate Astrometric
  Data (ESA 1997; cached page `HIP64426.iad.txt`), reconstructed relative
  to the J1991.25 catalog position by
  `exozippy-getdata "HIP 64426" --hipparcos-iad --hip-merge-consortia`:
  36 FAST/NDAC records -> 20 abscissae, each correlated FAST/NDAC pair of
  an orbit combined into one at its minimum variance (IA10).  The model's
  along-scan parallax factors match the IAD's IA5 to 3.2e-4.  The file's
  abscissae are measured from that position, which the config states as
  `reference_point:` (the dataset's tangent point).
- `Gaia3937211745905473024.DR4prerelease.epoch_astrometry.xml`: this
  source's rows of the June 2026 Gaia DR4 pre-release (ESA/Gaia/DPAC;
  the same file `examples/gaia4` ships for all twelve sources), extracted
  by `exozippy-getdata "HIP 64426" --gaia-epoch-astrometry
  --gaia-prerelease` (binary2 VOTable cells, so the values round-trip
  exactly): 90 field-of-view transits, 558 CCD observations AGIS used
  along scan, read with `data_format: gaia_epoch`.  After 2026-12-02
  `--gaia-epoch-astrometry` alone fetches the public DR4 product.

## Model

- One star, priors from Gaia DR3 (FLAME mass and radius, GSP-Spec Teff and
  [M/H]); no MIST.
- One dark companion on a Keplerian orbit, started at the Gaia DR3
  two-body solution (Thiele-Innes constants converted to Campbell
  elements; the conventions are the same, `examples/gaia4/README.md`).
  The companion is luminous in reality, so the fitted mass describes the
  photocenter orbit.  Without RVs, (Omega, omega) and (Omega + 180,
  omega + 180) are degenerate and Omega is restricted to [0, 180] deg.
- `fithipframe: true` on the Hipparcos dataset: the Hipparcos frame minus
  the Gaia frame at the star -- a position offset at J1991.25 (`frame_dE`,
  `frame_dN`) and a proper-motion offset (`frame_dpmE`, `frame_dpmN`) --
  with Gaussian priors of 0.3 mas and 0.28 mas/yr per coordinate from the
  Hipparcos-Gaia cross-calibration of Brandt (2018, 2021) and the EDR3
  bright-frame alignment (Lindegren et al. 2021b); the reasoning is in
  `src/exozippy/components/astrometryinstrument/frametie.py`.

## Status

WIP like `examples/gaia4` and `examples/GaiaBH3`: the system prepares and
builds (tests/test_examples_prepare.py), and a maximum-a-posteriori fit
from the DR3 start (below) behaves as expected, but no posterior has been
sampled yet.

## What it would take to make this a science example

- RVs.  The CfA (Latham et al. 1989) and later velocities (e.g. Kiefer et
  al. 2019, SOPHIE) would break the node degeneracy and give the
  companion's mass function; they were not retrievable from this machine
  (CDS/VizieR unreachable), so none are shipped.
- The companion's light (an M dwarf in an 84-d orbit around an F9 star):
  give it a star block and an SED so the photocenter flux fraction is
  predicted (`band:` + `companion_star_ndx:`).
- The reported wide companion HD 114762 B (a few arcsec, i.e. ~100 AU or
  more) would add a second, long-period orbit -- the case the 25-year
  lever arm exists for.
