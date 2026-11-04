# Gaia BH3: real Gaia DR4 epoch astrometry + Gaia RVS radial velocities

Gaia BH3 (Gaia Collaboration, Panuzzo et al. 2024, A&A 686, L2) is a
0.76 Msun, [Fe/H] = -2.56 halo giant in an 11.6-year, e = 0.73 orbit
around a dark ~33 Msun black hole at 590 pc -- the most massive stellar
black hole known in the Milky Way, found in pre-release Gaia DR4
astrometry.  DPAC published both its orbit and the CCD-level epoch
astrometry it was derived from, so this example is THE validation case of
the Gaia epoch-astrometry reader (`data_format: gaia_epoch`) and of
EXOZIPPy's sky conventions: same data, an independently published answer.

```
poetry run exozippy gaiabh3.yaml          # DR4 epochs + Gaia RVS RVs (the paper's data)
poetry run exozippy gaiabh3_ground.yaml   # ... plus the four ground-based RVs
```

## Data

- `GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml`: the Gaia DR4
  pre-release of June 2026 (ESA/Gaia/DPAC,
  https://www.cosmos.esa.int/web/gaia/dr4-prerelease), shipped UNMODIFIED
  as DPAC distributes it in
  `gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip` (sha256
  `f81f4dc11064b72d99f536e3d34365694b839629b424d8247a4b5501016f3ce3`;
  draft data model `Gaia DR4_RC3`).  It is a VOTable holding the
  CCD-level along-scan astrometry of 12 illustrative sources; Gaia BH3 is
  `source_id: 4318465066420528000`.  It is read directly with
  `data_format: gaia_epoch` (src/exozippy/components/astrometryinstrument),
  which keeps the 558 CCD observations AGIS used along scan (from 77
  field-of-view transits, 2014.8-2019.9; median per-CCD error 0.085 mas),
  measures them from the file's own reference point (ra0, dec0) -- which
  is then the tangent point -- and converts the barycentric TCB times
  (`obs_time_tcb + obs_time_bary_corr`, ns from 2010-01-01 TCB) to
  BJD_TDB.  The format fixes the layout and the time system, so the config
  carries no `columns:` or `time_*:` key for it.
- `GaiaBH3.GaiaRVS.rv`: the 17 Gaia RVS epoch radial velocities of the
  paper's epoch-data appendix (pre-DR4 pipeline; the paper notes they "may
  slightly differ from the final DR4 values").  Times are BJD in **TCB**,
  so the config says `time_scale: tcb` and the reader converts them to
  BJD_TDB (a -19.5 s shift).  These are the RVs of the paper's combined
  solution, so `gaiabh3.yaml` fits the same data the paper did.
- `GaiaBH3.ground.rv`: the four ground-based RVs of the paper's
  ground-based RV table (UVES 2020-11-05, HERMES 2023-07-17 and
  2023-09-07, SOPHIE 2023-09-04), which the paper did not fit.  Times are
  BJD in **UTC** (`time_scale: utc`).  `gaiabh3_ground.yaml` adds them as a
  single instrument with one offset (the instrument zero points differ by
  far less than the orbit's tens of km/s); they extend the RV baseline
  four years past the Gaia window.

## Validation against DPAC

`tests/test_gaia_epoch_astrometry.py` checks the reader and the model
against DPAC's own numbers on this file:

- the AGIS-used along-scan selection keeps Gaia BH3's 558 CCD
  observations, the count DPAC's BH3 notebook (github.com/esa/gaia-bhthree,
  branch `gaia-dr4-prerelease`) prints after its filter;
- the model's along-scan parallax factors -- the Gaia spacecraft
  ephemeris (src/exozippy/ephemerides/gaia.eph) at each epoch's reception
  instant, from the BJD_TDB times -- agree with DPAC's
  `parallax_factor_al` to <= 1.6e-5 on all twelve sources in the file
  (1.3e-5 on BH3, which the fit also logs at load);
- the five-parameter (single-star) solution of BH3 formed from what a
  PREPARED system holds (BJD_TDB times, abscissae, errors, scan angles and
  the model's own parallax factors) reproduces the notebook's fit to the
  digits it prints, values and errors both (largest difference 0.0024
  sigma) -- an external number, so the test carries no constant of its
  own.

The along-scan model is the one DPAC states for these data,

    w = (dra* + pmra t) sin psi + (ddec + pmdec t) cos psi + plx f_plx
        + (B X + G Y) sin psi + (A X + F Y) cos psi,

and Gaia's Thiele-Innes constants A, B, F, G are term for term the
coefficients of EXOZIPPy's own photocenter projection
(`orbit.physics.thiele_innes_xy`; omega = omega_*, the star's argument of
periastron, and bigomega the ascending node where the star recedes,
src/exozippy/skyframe.md).  The orbit comparison below then tests what the
unit tests cannot: that the full model -- the sky-frame handedness, the
node and periastron conventions and the RV-vs-sky sign -- recovers the
published orbit from the same data.

## Model

One star with the paper's spectroscopic priors (Table 3: M = 0.76 +/-
0.05 Msun, R = 4.936 +/- 0.016 Rsun, Teff = 5212 +/- 80 K, [Fe/H] = -2.56
+/- 0.11), no MIST; one dark companion (the `planet` component, mass in
jupiterMass); one Keplerian orbit; one Gaia dataset with an along-scan
jitter; one RV instrument (two for `gaiabh3_ground.yaml`) with its own
offset and jitter.  ra/dec/pm are the barycenter's at J2017.5 (TCB), the
DR4 reference epoch (`epoch: 2457936.87477064` BJD_TDB); the starts are
the paper's combined solution (Table 2).  The CCD observations of one
transit are treated as independent, as DPAC's own `gaiasupdate` and BH3
notebook treat them.

**Convergence needs the direct-angle sampling of PR #561.**  The node is
measured to ~0.13 deg.  Sampled through its direction latents
(xbigomega, ybigomega -- the angle is their arctan2, with a structural
isotropic prior), an angle that precise is a narrow ray from the origin
(~930:1 here) that narrows to a funnel at the origin, and no diagonal
metric can straighten it: NUTS ran at a ~0.001 step size and maximum tree
depth on every draw, and `ybigomega` reached Rhat 2.2 even at
`target_accept: 0.99`.  With #561 the code notices the angle is measured,
samples bigomega directly (with a warning naming the override,
`fitbigomega: false`; the prior is unchanged), and the default settings
here (numpyro, 4 chains x 1000 tune + 3000 draws, `target_accept: 0.95`)
converge: 0 divergences in 12000 draws, max Rhat 1.002, min ESS 2881.

## Comparison with the literature

Medians and 68% intervals (the paper's are its +/- 1 sigma). omega is
omega_*, the star's; Omega the ascending node where the star recedes
(the agreement below is itself the check that these are the paper's
conventions too). The paper's Tp is in JD TCB, ours BJD_TDB (they differ
by ~20 s). a0 is the photocenter's (here: the star's) semimajor axis in
mas and a1 the star's in AU, which the paper gives for its combined
solution only; ours are derived from the fit as a1 = a M_BH/(M_* + M_BH)
and a0 = a1 plx. The mass function is the astrometric one, fM = a1^3/P^2
(AU, yr) = M_BH^3/(M_* + M_BH)^2. M_BH for the paper is its Eq. 5 (from
the combined solution and M_* = 0.76 +/- 0.05).

| parameter | this example (`gaiabh3`: DR4 epochs + RVS RVs) | `gaiabh3_ground` (+ 4 ground RVs) | Panuzzo+2024 astrometric solution | Panuzzo+2024 combined solution (astrometry + RVS) |
|---|---|---|---|---|
| P (d) | 4126 +93/-87 | 4344 +68/-66 | 4194.7 +/- 112.3 | 4253.1 +/- 98.5 |
| e | 0.7251 +/- 0.0047 | 0.7357 +/- 0.0032 | 0.7262 +/- 0.0056 | 0.7291 +/- 0.0048 |
| omega (deg) | 78.10 +/- 0.17 | 78.09 +/- 0.17 | 77.77 +/- 0.66 | 77.34 +/- 0.76 |
| Omega (deg) | 136.270 +/- 0.137 | 136.519 +/- 0.112 | 136.200 +/- 0.147 | 136.236 +/- 0.128 |
| i (deg) | 110.61 +/- 0.11 | 110.50 +/- 0.10 | 110.659 +/- 0.107 | 110.580 +/- 0.095 |
| Tp - 2458000 (d) | 178.5 +/- 1.1 | 177.6 +/- 1.0 | 177.28 +/- 0.98 | 177.39 +/- 0.88 |
| a0 (mas) | 26.80 +0.47/-0.44 | 27.88 +/- 0.34 | 27.07 +/- 0.56 | 27.39 +/- 0.49 |
| a1 (AU) | 16.01 +0.28/-0.26 | 16.55 +0.23/-0.22 | -- | 16.17 +/- 0.27 |
| fM (Msun) | 32.17 +0.57/-0.55 | 32.09 +/- 0.55 | 32.03 +/- 0.64 | 31.23 +/- 0.81 |
| M_* (Msun) | 0.757 +/- 0.050 (prior) | 0.756 +/- 0.051 (prior) | 0.76 +/- 0.05 (Table 3) | 0.76 +/- 0.05 (Table 3) |
| M_BH (Msun) | 33.64 +/- 0.57 | 33.55 +0.57/-0.55 | -- | 32.70 +/- 0.82 (Eq. 5) |
| parallax (mas) | 1.6742 +/- 0.0096 | 1.6841 +/- 0.0090 | 1.6747 +/- 0.0094 | 1.6933 +/- 0.0164 |
| pm_ra* (mas/yr) | -28.395 +0.068/-0.065 | -28.260 +/- 0.051 | -28.372 +/- 0.077 | -28.317 +/- 0.067 |
| pm_dec (mas/yr) | -155.094 +0.104/-0.109 | -155.331 +0.075/-0.078 | -155.150 +/- 0.129 | -155.221 +/- 0.111 |
| gamma (km/s) | -357.07 +0.43/-0.42 | -357.51 +0.38/-0.40 (RVS) | -- | -357.31 +/- 0.44 |

The Panuzzo et al. columns are their Table 2 (the Campbell orbital
elements and the barycenter's astrometric parameters), the stellar mass
their Table 3 and M_BH their Eq. 5.  Both runs used the shipped configs and settings (numpyro, 4 chains x 1000
tune + 3000 draws) with the direct-angle sampling of PR #561:
`gaiabh3` 0 divergences, max Rhat 1.002, min ESS 2881; `gaiabh3_ground`
17 divergences in 12000 draws, max Rhat 1.001, min ESS 2703.

What the comparison shows:

- On the paper's own data (`gaiabh3`), every orbital element and the
  barycenter's astrometry agree with the paper to about 1 sigma or
  better, on the same branch: Omega and i within 0.4 sigma of both of
  its solutions, e within 0.6, omega within 1.0, P 0.5 and 1.0 sigma
  short of its astrometric and combined values.  A mirrored sky frame
  would reverse the sense of motion on the sky (i -> 180 - i: 69.4 deg,
  not 110.6); a wrong RV-vs-sky sign, or the companion's omega in place
  of the star's, would move omega (and with it Omega) by 180 deg.  None of that happens, at sub-degree
  precision.
- The parallax is the paper's ASTROMETRIC one to 0.05 sigma (1.6742 +/-
  0.0096 against 1.6747 +/- 0.0094); the paper's combined solution moves
  it to 1.6933 +/- 0.0164, 1.0 sigma from both.  The fit's distance is
  correspondingly 597 pc.
- Our omega interval (+/- 0.17 deg) is about four times narrower than
  either of the paper's (+/- 0.66 and +/- 0.76), while every other
  element's interval is within ~20% of at least one of the paper's two.
  This is not yet understood.
- M_BH = 33.6 +/- 0.6 Msun against the paper's 32.70 +/- 0.82 (0.9
  sigma).  It follows from the mass function fM = a1^3/P^2: against the
  combined solution our a1 is 1% smaller and our P 3% shorter, a net 3%
  larger fM.  Our fM matches the paper's astrometric-solution fM (32.03
  +/- 0.64); its M_BH is from the combined solution's.  The stellar mass
  is the paper's prior and is returned unchanged.
- The four ground-based RVs (2020-2023, past the end of the Gaia window)
  lengthen the period to 4344 +/- 68 d and tighten e.  a1 grows with P
  (a0 = 27.9 mas), so fM = a1^3/P^2 and M_BH barely move (33.5 Msun).
  Omega moves by 0.25 deg and the proper motions by ~1.7 sigma of this
  example's own intervals; every element stays within 1.7 sigma of the
  paper's combined solution, which did not use these RVs.

## References

- Gaia Collaboration, Panuzzo, P., Mazeh, T., Arenou, F., et al. 2024,
  A&A 686, L2 (arXiv:2404.10486): the discovery, the orbit (Table 2), the
  stellar parameters (Table 3) and the epoch RVs.
- Gaia Collaboration, DR4 pre-release (June 2026); `gaiasupdate`
  (doi:10.5281/zenodo.20924676) and the Gaia BH3 notebook
  (github.com/esa/gaia-bhthree, branch `gaia-dr4-prerelease`) for the
  data model and the reference fits.
