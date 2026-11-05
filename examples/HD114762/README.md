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
- One dark companion on a Keplerian orbit (`chen: false`: its radius is
  pinned, so the mass-radius relation would act on the mass), started at
  the Gaia DR3
  two-body solution (Thiele-Innes constants converted to Campbell
  elements; the conventions are the same, `examples/gaia4/README.md`).
  The companion is luminous in reality, so the fitted mass describes the
  photocenter orbit.  Without RVs, (Omega, omega) and (Omega + 180,
  omega + 180) are degenerate and Omega is restricted to [0, 180] deg.
- `fithipframe` (on by default: Hipparcos epochs fit with Gaia): the Hipparcos frame minus
  the Gaia frame at the star -- a position offset at J1991.25 (`frame_dE`,
  `frame_dN`) and a proper-motion offset (`frame_dpmE`, `frame_dpmN`) --
  with Gaussian priors of 0.3 mas and 0.28 mas/yr per coordinate from the
  Hipparcos-Gaia cross-calibration of Brandt (2018, 2021) and the EDR3
  bright-frame alignment (Lindegren et al. 2021b); the reasoning is in
  `src/exozippy/components/astrometryinstrument/frametie.py`.

## Status

`hd114762.yaml` was sampled once (2026-11-04; the config's numpyro
settings, 4 chains x (1000 tune + 3000 draws)).  It did NOT converge to the
code's thresholds: 250 divergences after tuning, max Rhat 1.014
(`planet.bs`), min ESS 312 (`orbit.tc_sampled`).  The reason is physical:
without RVs the orbit is nearly face-on (cos i = 0.966 +/- 0.027, against
its bound) and e and omega are then barely constrained (secosw, sesinw =
0.00 +/- 0.41, -0.03 +/- 0.38).  The well-measured quantities:

| quantity | posterior (mean +/- sd) |
|---|---|
| companion mass (dark) [M_J] | 242 +/- 16 |
| period [d] | 83.891 +/- 0.032 |
| pm_ra*, pm_dec [mas/yr] (J2017.5) | -582.469 +/- 0.020, -0.548 +/- 0.016 |
| distance [pc] | 39.544 +/- 0.054 |
| frame_dE, frame_dN [mas] | 0.09 +/- 0.29, 0.02 +/- 0.29 |
| frame_dpmE, frame_dpmN [mas/yr] | -0.01 +/- 0.27, -0.13 +/- 0.27 |

The frame terms stay at their priors: the Hipparcos abscissae are 2-5 mas,
and an 84-day orbit is covered many times by DR4 alone, so this companion
gains little from the 25-year lever arm.  The astrometric orbit also
disagrees with the RV one: a0 sin i ~ 0.6 mas from the astrometry (Gaia
DR3: a0 = 1.80 mas, i = 18.9 deg) against a_* sin i = 0.121 +/- 0.012 mas
from the RV orbit (K = 612.5 m/s; Kiefer et al. 2019, Table 1) -- a
factor of 5 that RVs in the fit would have to resolve.

### The wide companion B: `hd114762_accel.yaml`

HD 114762 B is a 0.088 Msun ultracool subdwarf 3.3 arcsec (128 AU
projected) from A (Bowler et al. 2009).  Its pull on A is at most
G M_B / r_proj^2 = 0.0054 mas/yr^2 on the sky: over the 26 years between
Hipparcos and DR4 that bends A's track by up to a t^2/2 = 1.8 mas, against
Hipparcos position errors of ~1 mas -- while over DR4's 5.3 years it is
0.02 mas.  `hd114762_accel.yaml` adds B as an outer `constant_acceleration`
orbit (orbit.B.accel_ra, accel_dec).  Maximum a posteriori, 1-sigma from
the Hessian (2026-11-04):

| | Hipparcos + DR4 | DR4 alone |
|---|---|---|
| accel_ra [mas/yr^2] | 0.0028 +/- 0.0033 | -0.046 +/- 0.039 |
| accel_dec [mas/yr^2] | 0.0007 +/- 0.0028 | 0.007 +/- 0.033 |

The joint fit constrains the acceleration 12x better than DR4 alone, to
the level of B's expected pull; DR4 alone is 7x too weak to see it.  It is
NOT a detection (0.9 sigma): B's acceleration is at most ~1.7 sigma of
what this star's Hipparcos positions can measure, and the true one is
smaller unless B is at its projected separation.  A star with ~10x the
acceleration (a brown dwarf at 20 AU, a stellar companion at 50 AU) would
be a clean demonstration; none is among the twelve pre-release sources,
of which HD 114762 is the only Hipparcos star (checked by cone match
against the Hipparcos catalog; HD 183633 is not a Hipparcos star).

## What it would take to make this a science example

- RVs.  The Lick velocities (Kane et al. 2011, the orbit Kiefer et al.
  2019 adopt) would break the node and face-on degeneracies, settle the
  factor-of-5 a sin i tension above and let the sampler converge; they
  were not retrievable from this machine (CDS/VizieR unreachable), so none
  are shipped.
- The companion's light (an M dwarf in an 84-d orbit around an F9 star):
  give it a star block and an SED so the photocenter flux fraction is
  predicted (`band:` + `companion_star_ndx:`).
- B as a Keplerian (rather than constant-acceleration) orbit, with its
  measured separation and position angle from the AO imaging as rel
  astrometry, would turn the acceleration bound into a mass constraint.
