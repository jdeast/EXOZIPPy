# HD 80606 b: the high-eccentricity validation

HD 80606 b is a ~4 M_J planet on a 111.4-day orbit with e ~ 0.93 (Naef et
al. 2001), transiting a V = 9 G dwarf (Moutou et al. 2009). It is
EXOZIPPy's high-eccentricity validation: three fits of the same system, each
with real data, and none of them pins cos i.

| config | data | star | what it exercises |
|---|---|---|---|
| `hd80606.yaml` | TESS S21 transit + all public RVs | MIST + SED | best practice: sqrt(e)cos/sin(omega), cos i, linear mass |
| `hd80606_rvonly.yaml` | all public RVs | MIST + SED | the `fitmsini` default: (m sin i, cos i) sampled, cos i free |
| `hd80606_transitonly.yaml` | TESS S21 transit | MIST + SED | the transit-only defaults: V_c/V_e + transit chord, log_q + Chen |

`hd80606.params.yaml` (shared by the first two) and
`hd80606_transitonly.params.yaml` hold published start values: the star from
Rosenthal et al. (2021), the orbit and planet from Pearson et al. (2022). The
only priors are the [Fe/H] (Rosenthal et al. 2021), the Gaia DR3 parallax, the
Schlegel et al. (1998) extinction ceiling and, in the transit-only file, the
period (Pearson et al. 2022): a single transit cannot measure a period, and
with the period and the stellar density known the duration measures V_c/V_e
(the photo-eccentric effect).

## The data

`prepare_data.py` rebuilds every data file from its public source (MAST and
VizieR); it is the exact recipe for everything below.

**TESS** (`n20200213.TESS.TESS.HD80606.S21.0120.SPOC.dat`). TESS observed
HD 80606 in Sectors 21 and 47 at 2-min cadence. Sector 21 caught the transit
of 2020 Feb 14; Sector 47 covers neither a transit nor a secondary eclipse,
so it is not used. The file is the SPOC PDCSAP light curve (lightkurve's
default quality mask, NaNs removed) cut to +/- 1 day of the Pearson et al.
(2022) transit time and normalized to its out-of-transit median: 1405
cadences. Its fourth column, time minus that transit time, is a detrend
column, so a linear baseline is fit with the transit.

- PDCSAP is deblended: HD 80607, 20.5" away, puts ~38% of the aperture's flux
  in (CROWDSAP = 0.623), and the radius ratio depends on SPOC's correction for
  it.
- The 2-min scatter is ~2400 ppm, ~4x the pipeline errors (the 1-hour CDPP is
  ~470 ppm); the fitted `jitter_variance` absorbs it.
- The secondary eclipse (~5.7 days before the transit) falls in Sector 21 too,
  but is left out: its expected TESS depth (reflected light at r ~ 0.034 AU,
  well under 100 ppm) is far below what one ~2-hour eclipse at this scatter
  can detect (~300 ppm), and fitting it would add two unconstrained band
  parameters -- and, through `fitreflect`/`fitthermal`, turn off the
  transit-only defaults the third config exists to test.

**RVs** (m/s; columns BJD, RV, error):

| file | instrument | source | N | used |
|---|---|---|---|---|
| `HD80606b.ELODIE-2001.rv` | ELODIE | Naef et al. (2001), VizieR J/A+A/375/L27 | 55 | 55 |
| `HD80606b.ELODIE-2009.rv` | ELODIE | Moutou et al. (2009), VizieR J/A+A/498/L5 | 19 | 19 |
| `HD80606b.SOPHIE.rv` | SOPHIE | Hebrard et al. (2010), VizieR J/A+A/516/A95 | 105 | 48 |
| `HD80606b.HIRES-pre.rv` | HIRES (before the 2004 upgrade) | Rosenthal et al. (2021), VizieR J/ApJS/255/8 | 39 | 39 |
| `HD80606b.HIRES-post.rv` | HIRES (after it) | Rosenthal et al. (2021) | 73 | 63 |

- The two ELODIE sets get separate offsets: fit separately their zero points
  differ by ~16 m/s (~3.6 sigma).
- Hebrard et al.'s SOPHIE set is a superset of Moutou et al.'s SOPHIE points,
  so only the former is used.
- The California Legacy Survey reduction (Rosenthal et al. 2021) of the HIRES
  data supersedes Butler et al. (2017); Naef et al.'s six 1999-2000 HIRES
  points are older than it and not used.
- The `.mask` files exclude every RV taken within T14/2 + 1 hour of a
  transit (57 SOPHIE points from the 2009 and 2010 transits, 10 HIRES ones,
  among them the Winn et al. 2009 sequence): the Rossiter-McLaughlin anomaly
  there is not modeled. The points stay in the files, so an `rm:` fit can
  use them.
- None of the catalogs states whether its BJD is UTC- or TDB-based; they are
  read as BJD_TDB. The ~66 s difference is up to ~2 m/s near periastron,
  inside the fitted jitters.

**SED** (`hd80606.sed.yaml`): Gaia DR3, 2MASS and WISE, from
`scripts/mkticsed.py "HD 80606"`. HD 80607 is resolved in all three.

## Start logps

`model.compile_logp()(model.initial_point())` after `prepare()` and
`build_model()`, i.e. before the seed polish, on radish (Linux x86_64):

| config | start logp | free RVs | mass coordinate | e / inclination coordinates |
|---|---|---|---|---|
| `hd80606.yaml` | -132978.06645524 | 26 | linear | sqrt(e)cos/sin(omega), cos i |
| `hd80606_rvonly.yaml` | -125231.13057992 | 21 | msini | sqrt(e)cos/sin(omega), cos i |
| `hd80606_transitonly.yaml` | -7861.49272575 | 25 | log_q (+ Chen) | V_c/V_e, chord |

Almost all of the first two is the RVs (-125060 and -125014 nats): each
instrument's `gamma` starts at the MEAN of its RVs, and with e ~ 0.93 and
observations clustered near periastron and the transits that mean is 50-220
m/s from the systemic velocity (SOPHIE: 4113 vs ~3905 m/s). See below.

## Check fits (numpyro, 4 chains, 2026-10-08)

**`hd80606_rvonly.yaml`** -- converged at 4 x (1000 + 1000), seed 12345: 0
divergences, max Rhat 1.004, min ESS 671 (msini; the cos i / mass tail ESS is
~390, the expected banana). The tree depth averages 9.6-10, though, at step
sizes of 0.004-0.006: the RV-only high-e geometry is expensive.

**`hd80606.yaml`** -- does NOT converge as shipped, and the cause is the start,
not the model:

- The seed polish starts from the mean-of-RVs gammas (logp -132978) and
  L-BFGS walks 2500 iterations to a WRONG basin at logp +6760: a pre-main-
  sequence star (0.10 M_sun, EEP ~2), a non-transiting orbit (i = 20 deg) and
  p = 1.6. The startup report lists the moved seeds.
- From there, seed 12345 (4 x 1000 + 1000): chain 1 stays in that basin for
  550 draws; chain 3 settles into a grazing, star-sized "planet" (p ~ 2.6,
  b ~ 3.6), 13 nats below the real mode; 295 divergences in one chain, max
  Rhat 1.8. Seed 777 at 4 x (2000 + 1000): 0 divergences, but one chain again
  in the grazing mode, Rhat 1.55 on cos i and T_C.
- With only the five gammas seeded at their RV-fit values (a diagnostic; they
  are not published, so the params file does not carry them) the start logp
  is -7050.7, the polish reaches +7553.7 in 64 s, and the same 4 x (1000 +
  1000), seed 12345, converges: 0 divergences, max Rhat 1.004, min ESS 809
  (star.eep; every orbit and planet parameter > 1000), tree depth 8.4-9.0.
  That run's numbers are in the table below.

**`hd80606_transitonly.yaml`** -- does not converge at 4 x (1000 + 1000):
every chain saturates the tree depth (10) at step sizes of 1e-4 to 6e-4, max
Rhat 2.2. V_c/V_e is recovered (1.80 +0.05/-0.07, against 1.795 from Pearson
et al. 2022), but e and omega slide along the V_c/V_e ridge (e 0.66 +0.11/-0.10,
consistent only with e >~ 0.53, which is all a single duration can say), and
the Chen mass (1-90 M_J per chain) and the MIST age do not mix.

## Against the literature

Median and 68% interval. EXOZIPPy columns: the gamma-seeded best-practice run
and the RV-only run above. Pearson et al. (2022) combine the RVs with many
transits, this TESS one among them; Rosenthal et al. (2021) fit the HIRES RVs
alone. Their values are as the NASA Exoplanet Archive lists them.

| parameter | Pearson+2022 | best practice | Rosenthal+2021 | RV-only |
|---|---|---|---|---|
| P (d) | 111.436765 +/- 0.000074 | 111.43634 +/- 0.00031 | 111.43639 +/- 0.00032 | 111.43637 +/- 0.00031 |
| T_C (BJD_TDB) | 2458888.0747 +/- 0.0020 | 2458888.0755 +0.0028/-0.0029 | -- | 2458888.27 +/- 0.14 |
| e | 0.93183 +/- 0.00014 | 0.93103 +0.00042/-0.00040 | 0.93043 +/- 0.00069 | 0.93023 +/- 0.00073 |
| omega (deg) | -58.887 +/- 0.043 | -58.70 +/- 0.13 | -58.95 +/- 0.25 | -58.99 +/- 0.25 |
| K (m/s) | 469.22 +/- 0.61 | 470.1 +/- 1.4 | 465.5 +/- 2.8 | 466.4 +/- 3.1 |
| i (deg) | 89.24 +/- 0.01 | 89.215 +/- 0.051 | -- | free (cos i ~ U) |
| R_P/R_* | 0.1009 +/- 0.015 | 0.1013 +0.0030/-0.0023 | -- | -- |
| M_P (M_J) | 4.1641 +/- 0.0047 | 4.157 +0.160/-0.150 | 4.16 +/- 0.13 (m sin i) | 4.20 +0.18/-0.17 (m sin i) |
| R_P (R_J) | 1.032 +/- 0.015 | 1.047 +0.047/-0.042 | -- | -- |

The best-practice e is 1.9 sigma (of its own interval) below Pearson et al.'s,
whose much smaller error comes from their many more transits.
