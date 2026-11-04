# Gaia-4b: real Gaia DR4 epoch astrometry + radial velocities

Gaia-4b (Stefansson et al. 2025, AJ 169, 107) is an 11.8 M_J planet on a
571-day, e = 0.34 orbit around a 0.64 Msun K/M dwarf at 73 pc -- one of
the first planets found by Gaia astrometry, confirmed with HPF, NEID and
FIES radial velocities.  This example fits the REAL Gaia DR4 epoch
astrometry jointly with those RVs.

```
poetry run exozippy gaia4.yaml          # one planet (the default)
poetry run exozippy gaia4_accel.yaml    # + an outer body: constant acceleration
poetry run exozippy gaia4_jerk.yaml     # + an outer body: constant jerk
poetry run exozippy gaia4_outer.yaml    # + an outer planet on a Keplerian orbit
```

## Data

- `GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml`: the Gaia DR4 pre-release
  of June 2026 (ESA/Gaia/DPAC,
  https://www.cosmos.esa.int/web/gaia/dr4-prerelease), shipped UNMODIFIED
  as DPAC distributes it (sha256 `f81f4dc1...016f3ce3`; the zip is
  `gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip`).  It holds the
  CCD-level along-scan astrometry of 12 illustrative sources, Gaia-4
  among them (`source_id: 1457486023639239296`); the draft DR4 data model
  is `Gaia DR4_RC3`.  It is read directly with `data_format: gaia_epoch`
  (src/exozippy/components/astrometryinstrument), which keeps the 824 CCD
  observations AGIS used along scan (from 115 field-of-view transits,
  2014.9-2019.9), measures them from the file's reference point (ra0, dec0)
  and converts the barycentric TCB times to BJD_TDB.
- `Gaia4.{HPF,NEID,FIES}.rv`: the 23 published RVs (10 HPF, 8 NEID, 5
  FIES; Stefansson et al. 2025, Table 5, machine-readable table
  `ajada9e1t5_mrt.txt`), BJD_TDB and m/s, one offset per instrument.

The reader and the sky conventions are validated on Gaia BH3
(examples/GaiaBH3), whose orbit DPAC published from the same data.  This
example is the template for fitting a long-term trend: an unresolved outer
companion's reflex as a Taylor orbit (`constant_acceleration`,
`constant_jerk`) read by BOTH the RVs and the Gaia epochs.

## Model

One star with spectroscopic/isochrone priors from Stefansson et al.
(2025, Table 1: M = 0.644 +0.025/-0.023 Msun, symmetrized to 0.024;
R = 0.624 +/- 0.015 Rsun; Teff = 4034 +/- 77 K; [Fe/H] = 0.05 +/- 0.13),
no MIST; one dark companion; a Keplerian orbit; three RV instruments with
their own offsets and jitters; one Gaia dataset with an along-scan jitter
(the AGIS source excess noise is 0.12 mas against a median per-CCD error
of 0.09 mas, so the jitter is not optional).  ra/dec/pm are at J2017.5
(TCB), the DR4 reference epoch (`epoch: 2457936.87477064` BJD_TDB), and
the position starts at the file's reference point; the other starts are
the published Gaia+RV solution.

The CCD observations of one transit are treated as independent, as DPAC's
own `gaiasupdate` and BH3 notebook treat them.

## Comparison with the literature

Medians and 68% intervals; omega is omega_* (the star's), Omega the
ascending node where the star recedes (EXOFASTv2 convention; Gaia's
Thiele-Innes constants are the coefficients of the same projection).

| parameter | this example (DR4 epochs + RVs) | DR4 astrometry only | RVs only | Stefansson+2025 Gaia+RV (adopted) | Stefansson+2025 RV only | Gaia DR3 two-body |
|---|---|---|---|---|---|---|
| P (d) | 575.8 +/- 1.3 | 578.6 +/- 3.5 | 526 +96/-29 | 571.3 +1.4/-1.3 | 513 +21/-16 | 564 +/- 11 |
| e | 0.329 +/- 0.030 | 0.41 +/- 0.07 | 0.28 +0.13/-0.05 | 0.338 +0.026/-0.023 | 0.263 +0.037/-0.033 | 0.51 +/- 0.20 |
| omega (deg) | 203.5 +/- 7 | 213.8 +/- 9 | 181 +11/-8 | 180.3 +5.5/-4.8 | 181.8 +6.8/-6.2 | 185 +/- 16 |
| Omega (deg) | 173.3 +/- 2.4 | 174.5 +/- 2.6 | -- | 158.6 +/- 5.1 | -- | 159.4 +/- 5 |
| i (deg) | 120.7 +/- 2.2 | 120.4 +/- 2.7 | -- | 116.9 +4.2/-4.4 | -- | 115.4 +/- 4.6 |
| K (m/s) | 316 +11/-10 | -- | 359 +54/-23 | 368 +22/-18 | 355 +17/-14 | -- |
| M_b (M_J) | 10.7 +/- 0.4 | 10.9 +0.55/-0.50 | m sin i 10.5 +1.6/-0.7 | 11.8 +0.73/-0.66 | m sin i 10.06 +0.51/-0.46 | 13.2 +/- 1.9 |
| parallax (mas) | 13.626 +/- 0.007 | 13.624 +/- 0.008 | -- | 13.628 +/- 0.021 | -- | 13.643 +/- 0.025 |

(The DR4-only inclination is from its cos i = -0.506 +/- 0.040; the
astrometry-only (Omega, omega) is the branch the RVs select of the
(Omega + 180, omega + 180) pair astrometry alone cannot distinguish.)

What the comparison shows:

- The DR4 epochs and the RVs are in tension within one Keplerian: the
  astrometry alone wants P = 578.6 +/- 3.5 d and omega = 214 +/- 9 deg, the
  RVs alone ~513-526 d and ~181 deg.  The DR3 solution (P +/- 11 d) was too
  loose to show this; the DR4 epochs constrain P eight times better.  The
  joint fit compromises (K drops from ~360 to 316 m/s, and the RV jitters
  inflate to 22-54 m/s), and its RV residuals are structured.
- The RV jitter priors differ from Stefansson et al.'s: EXOZIPPy samples
  the signed jitter VARIANCE uniformly, which with only 5 FIES points lets
  the FIES jitter run large (their 2.5 m/s is a posterior median under an
  unstated "broad uninformative" prior; its 84th percentile is 15.5 m/s,
  so neither measures it).


## Looking for another body

A real disagreement between the RV orbit and the astrometric orbit needs
an astrophysical cause, and the natural one is another body.  Two clues
pointed that way: an RV trend lets the RVs fit at the astrometric period
(slope -0.20 +/- 0.10 m/s/day, omega = 203 +/- 20 deg, jitters back to
19-26 m/s), and Gaia's pm_dec differs between DR3 (18.101 +/- 0.017
mas/yr) and this fit (17.966 +/- 0.004) by ~8 sigma.  We looked for it
three ways, each a variant config here.  None found it -- but each is a
working template for the same search on another star.

| | `gaia4.yaml` | `gaia4_accel.yaml` | `gaia4_jerk.yaml` | `gaia4_outer.yaml` |
|---|---|---|---|---|
| outer body | -- | constant 3-D acceleration | constant 3-D jerk | Keplerian, P = 650-1e5 d |
| sky acceleration (mas/yr^2) | -- | ra -0.003 +/- 0.005, dec +0.002 +/- 0.006 | ra -0.001 +/- 0.006, dec +0.003 +/- 0.006 | -- |
| sky jerk (mas/yr^3) | -- | -- | ra -0.008 +/- 0.013, dec -0.009 +/- 0.013 | -- |
| gammadot (m/s/day, at J2017.5) | -- | -0.12 +/- 0.08 | -6.7 +/- 2.6 | -- |
| gammaddot (m/s/day^2) | -- | -- | 0.0027 +/- 0.0011 | -- |
| K_c (m/s) | -- | -- | -- | 3 +8/-1.5 (P_c pushed to the long end) |
| P_b (d) | 575.8 +/- 1.3 | 575.6 +/- 1.4 | 573.5 +/- 1.4 | 576.1 +/- 1.1 |
| omega_b (deg) | 203.5 +/- 7 | 206 +/- 6 | 212 +/- 7 | 204 +/- 7 |
| RV jitter HPF / NEID / FIES (m/s) | 35 / 22 / 54 | 35 / 22 / 32 | 22 / 23 / 19 | 35 / 21 / 57 |

What each says:

- **Constant acceleration**: DR4 sees no sky acceleration.  If the DR3 ->
  DR4 pm_dec shift were acceleration, accel_dec would be ~ -0.09
  mas/yr^2, excluded at ~15 sigma.  The sky limit (< ~0.012 mas/yr^2 at
  2 sigma, ~0.011 m/s/day at 73 pc) also caps the companion at M_c/r^2 <
  0.023 M_J/AU^2 (under a Jupiter mass at 6.6 AU, 10 M_J at 21 AU) unless
  it lies within ~6 deg of the line of sight.
- **Constant jerk**: a jerk lets the acceleration pass through zero at
  J2017.5 while the DR3 window still sees a different mean pm; that needs
  ~0.09 mas/yr^3, excluded at ~7 sigma.  The RV curvature is 2.5 sigma and
  brings the RVs into line, but it is not physical: the RVs (2023.2-2024.6)
  start more than three years after the last DR4 epoch (2014.9-2019.9),
  so the shared Taylor epoch (the astrometric one, J2017.5) extrapolates
  the RV curvature 6.5 years back, to a line-of-sight acceleration ~600x
  the sky limit at the same instant.  The lesson for the template: a
  Taylor orbit is a local expansion, valid only where every dataset that
  reads it overlaps in time; set `epoch:` deliberately when they do not.
- **Outer Keplerian**: the companion collapses to K_c ~ 3 m/s with its
  period pushed to the long end, and b and the jitters do not move.  This
  run did not converge (PTDE, 8 temperatures, no ladder round trips; the
  sampler recommends `n_temps: 15`), so it is a template, not a limit.

So the tension remains: one Keplerian fits the astrometry, and the RVs
keep ~35 m/s of HPF jitter against 14-23 m/s errors with structure that
none of these models absorbs.  Stefansson et al.'s own RV-only period
(513 +21/-16 d against their joint 571 d) shows the same disagreement.
The RVs cover 2023.2-2024.6, less than one orbit of b, and do not overlap
the DR4 epochs, which limits every test above; RVs contemporaneous with
a later Gaia baseline would settle it.

## References

- Stefansson et al. 2025, AJ 169, 107 (arXiv:2410.05654): the planet, the
  RVs, the stellar parameters and the Gaia DR3 + RV joint solution.
- Gaia Collaboration, DR4 pre-release (June 2026); `gaiasupdate`
  (doi:10.5281/zenodo.20924676) and the Gaia BH3 notebook
  (github.com/esa/gaia-bhthree) for the data model and the reference fits.
- Gaia DR3 non-single-star catalogue (Gaia Collaboration, Arenou et al.
  2023; Halbwachs et al. 2023): the two-body orbit solution of Gaia-4.
