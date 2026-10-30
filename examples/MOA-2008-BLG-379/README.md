# MOA-2008-BLG-379: an image-constrained planetary microlensing fit

A Jupiter-mass planet around an M dwarf, measured by combining the 2008
light curve with high-resolution imaging taken years after the event,
when the lens had moved far enough from the source to be seen on its own.
This follows Bennett et al. (2024, AJ 168, 15; arXiv:2311.00627, open
access, CC BY 4.0), who call the method "image-constrained modeling".

```bash
cd examples/MOA-2008-BLG-379 && poetry run exozippy MOA-2008-BLG-379.yaml
```

## What it demonstrates

Everything enters as an observation of the physical system, fit at once:

| constraint | how EXOZIPPy takes it |
|---|---|
| OGLE I, OGLE V and MOA-red light curves | `mulensinstrument` (2L1S, finite source, VBM) |
| lens minus source position, Keck NIRC2 Ks, 2018 May 26 and Aug 6 | `astrometryinstrument`, `mode: lensrel` (`keck_lensrel.dat`) |
| lens Ks, lens I, source I, lens+source I | SED rows (`MOA-2008-BLG-379.sed.yaml`), `photType` naming the star(s) |
| the OGLE light curve's source flux is the source's I magnitude | `mulensinstrument.OGLE_I.zeropoint` prior (the f_source -> SED tie) |
| source extinction from the red clump | prior on `star.Source.av` |
| lens extinction: the part of the dust in front of the lens | a hard link `star.Lens.av` = f(D_L, D_S) x `star.Source.av` |
| Galactic density, kinematics and mass function | `galacticmodel`, on both stars |
| stellar physics | MIST tracks (`evolutionarymodel`) on both stars |

Nothing about the lens is assumed: its mass, distance, proper motion,
age, metallicity and extinction (through the link) are free and are
constrained only through the physics -- the lens's Ks and I magnitudes
through MIST and its distance, `theta_E` through the Keck separation and
the light curve's `t_E`, and so on.  Only the sky position is pinned.

The `lensrel` observable is the light curve's own trajectory scaled by
`theta_E`, evaluated at the Keck epochs and seen from the Earth (C9, C30
in `src/exozippy/components/mulensing/conventions.md`; `mulensing.md`,
"The lens's position is derived from the source's"), so the heliocentric
versus geocentric proper-motion conversion Bennett et al. do by hand
(their mu_rel,G = mu_rel,H - v_earth pi_rel / AU) is not an input here: it
is what the model computes.

## Why this event

It was chosen as the first public image-constrained example over
OGLE-2012-BLG-0563 (Bennett et al. 2024b, arXiv:2412.03651) and the
other events with published resolved-lens follow-up, by three criteria in
order (the full evaluation is in the pull request that added it):

1. **Public, redistributable light curves.**  All three light curves are
   the NASA Exoplanet Archive's contributed data set for this event
   (survey data only: OGLE and MOA; no follow-up network data).  The
   OGLE-2012-BLG-0563 analysis rests on an unpublished MOA re-reduction
   and is not on the archive.
2. **Cleanly resolved.**  In 2018 the pair is 57.5 mas apart, 5.8 NIRC2
   narrow-camera pixels and about one PSF FWHM (59-72 mas), measured at
   two epochs.  OGLE-2012-BLG-0563's pair is 24.6 mas, 0.6 of a Hubble
   UVIS pixel, and only partially resolved.
3. **A clear published analysis**, with every constraint listed with its
   value and uncertainty, four degenerate solutions tabulated, and an
   appendix that re-models exactly the photometry the archive holds.

## Data provenance and licensing

- `OGLE_I.dat`, `OGLE_V.dat`, `MOA_R.dat`: the NASA Exoplanet Archive's
  microlensing time series for MOA-2008-BLG-379
  (`https://exoplanetarchive.ipac.caltech.edu/data/ExoData/0302/0302515/data/`,
  `UID_0302515_PLC_00{1,2,3}.tbl`, contributed with Suzuki et al. 2014,
  ApJ 780, 123).  `prepare_data.py` copies the three columns verbatim and
  keeps the archive header as comments.  The archive's data are public;
  it asks that use be acknowledged ("This research has made use of the
  NASA Exoplanet Archive, which is operated by the California Institute of
  Technology, under contract with the National Aeronautics and Space
  Administration under the Exoplanet Exploration Program") and that the
  contributing paper be cited (Suzuki et al. 2014).
  These are the 2013 reductions used in the discovery paper.  Bennett et
  al. (2024) fit a 2018 MOA re-reduction (detrended for differential
  refraction, Bond et al. 2017) that is not public; their appendix table
  re-models the 2013 photometry and gets the same solution within errors
  (close, no parallax: t_E = 40.5 +/- 6.0 d on the 2013 data, 46.6 +/- 5.9
  on the 2018 data).
- `keck_lensrel.dat`: the two per-epoch rows of Bennett et al.'s Keck
  NIRC2 table (not their weighted sum), star 1 (the lens) minus star 2
  (the source).  The images themselves are in the Keck Observatory
  Archive (DOI 10.26135/KOA3).
- `MOA-2008-BLG-379.sed.yaml`: the Keck and Hubble magnitudes (Hubble GO
  program data in MAST, DOI 10.17909/edn8-2564), as the paper quotes them.

## Conventions

- **Times** are HJD (UTC) in the files, converted to BJD_TDB by
  `time_frame: hjd`, `time_scale: utc`; the paper's `t_0` (HJD') maps to
  BJD_TDB by +62.8 s.
- **alpha and u_0** were MEASURED, by the C21 procedure: the light curves
  were scored at the paper's (u_0 < 0, s > 1) and (u_0 < 0, s < 1)
  solutions under the eight combinations `(+/-u_0) x (alpha, -alpha,
  180 +/- alpha)` and both lens origins (center of mass, host), with a
  static binary and a uniform finite source:

  | origin | combination | chi2 (1245 points), wide / close |
  |---|---|---|
  | center of mass | `(-u_0, alpha)` or its mirror `(u_0, -alpha)` | **1249.3 / 1250.6** |
  | center of mass | any other | 24,250 - 46,256 |
  | host | best | 7,767 / 11,646 |

  So the paper's origin is the lens center of mass, as ours (C12), and its
  `(u_0, alpha)` is our `(-u_0, alpha)` up to the C23 mirror, which only
  the weak parallax distinguishes.  The four seeds therefore span both
  mirror branches whichever of the two it is, and the comparison below is
  made on physical quantities, which the mirror does not change.
- **pi_E** is not seeded: it is derived from the masses, distances and
  proper motions, which are.

## Gaps: what the paper uses that this fit does not

None of these is patched over with a hand-written prior; each is a code
change.

1. **The extinction law.**  The SED and BC grid apply one fixed law
   (`models/extinction_law.ascii`, close to Cardelli R_V = 3.1, and ~6%
   high in the NIR; `components/sed/extinction.py`).  Toward this field
   the bulge law is different: the paper's red-clump values are A_I =
   1.800, A_V = 3.315, A_K = 0.182, i.e. A_I/A_V = 0.543 and A_K/A_V =
   0.055, where ours gives 0.599 and 0.125.  No single `av` reproduces
   both: anchored on A_I (as here, because the source flux and t_E are
   I-band), our law puts A_K = 0.375 on the source, 0.19 mag more than the
   paper, and ~0.16 mag more on the lens, which makes the lens's measured
   Ks look intrinsically brighter and pulls its mass UP.  The fix is the
   runtime extinction law of `components/sed/sed.md` ("A runtime
   extinction law: the plan, and why it is blocked"), with R_V (or a
   bulge law such as Nataf et al. 2016) as a parameter.  This is also why
   the source's V and Ks magnitudes are left out: they would turn the law
   error into a colour error and push it onto the source's Teff.
2. **A one-sided magnitude limit.**  The lens is detected in F555W at only
   ~1 sigma, so the paper applies V_L >= 26.49 +/- 0.68 as a half-Gaussian
   (no penalty when fainter).  An SED row is a two-sided Gaussian, so the
   limit is dropped.  Needs a `limit:` (upper/lower) option on SED rows.
3. **Correlated photometry.**  The lens and source Ks magnitudes share the
   0.05 mag calibration of the combined image, and the lens, source and
   combined F814W magnitudes come from one PSF fit with strongly
   anti-correlated lens/source fluxes.  SED rows are independent, so --
   like the paper, which used the same independent constraints -- this
   fit keeps the combined F814W row at the paper's inflated 0.101 mag and
   uses only the lens's Ks.  Needs a row covariance in the `.sed` file.
4. **A dust model.**  The lens's extinction follows the paper's dust
   layer (scale height 0.10 kpc), written as a params-file hard link on
   `star.Lens.av`.  It works, but it is a physical model living in a
   params file, with the scale height fixed (the paper's +/- 0.02 kpc is
   not propagated).  A 3-D dust component (or a `galacticmodel` option)
   reading both distances would be its proper home.
5. **The planet-hosting prior.**  The paper weights the posterior by a
   host-mass prior proportional to M_host (they report < 0.2 sigma
   effect).  EXOZIPPy has no host-mass occurrence prior; not adopted.
6. **The Hubble positions.**  The 2013 Hubble positions of the two stars
   were fit WITH the Keck proper motion imposed, so they are not an
   independent epoch and are not used.  Only Keck enters `lensrel`.
7. **The 2018 MOA re-reduction** (detrended, recalibrated) is not public;
   the archive's 2013 reduction is used (above).

This example needs #522: before it, a microlensing source star on MIST
tracks had its `logmass` pinned at the seed.

## Comparison with Bennett et al. (2024)

**No converged fit exists yet** (cost: next section).  The table compares
the paper with this model evaluated AT the paper's solution (seed 0:
their u_0 < 0, s > 1, with M_host = 0.434, D_L = 3.44 kpc, D_S = 7.77 kpc
and the Keck mu_rel,H), and with a 150-step, 2-rung PTDE run from the
four seeds.  That run did not move the chains: D_L stayed at 3440 +/- 8
pc, and the mode report says the 24 chains never mixed.  So the third
column is the start, not a posterior; do not quote it.

| quantity | Bennett+2024 | this model at their solution | 150-step run (unconverged) |
|---|---|---|---|
| theta_E (mas) | 0.754 +/- 0.040 | 0.759 | 0.758 |
| mu_rel,H E, N (mas/yr) | 5.430 +/- 0.063, 1.926 +/- 0.038 | 5.430, 1.926 (seeded) | 5.426, 1.922 |
| mu_rel,G (mas/yr) | 5.042 +/- 0.149 | 5.169 | 5.169 |
| t_E (d) | 55.8 +/- 5.5 | 53.6 | 53.6 |
| pi_E N, E | 0.080 +/- 0.026, 0.207 +/- 0.037 | 0.083, 0.197 | 0.083, 0.196 |
| rho (= t_*/t_E) | 4.0e-4 (0.0221/55.8) | **6.1e-4** | **6.0e-4** |
| M_host (Msun) | 0.434 +/- 0.065 | 0.434 (seeded) | 0.434 |
| m_p (M_Jup) | 2.44 +/- 0.49 | 2.41 | 2.42 |
| D_L (kpc) | 3.44 +/- 0.53 | 3.44 (seeded) | 3.44 |
| D_S (kpc) | 7.77 +/- 1.27 | 7.77 (seeded) | 7.74 |
| a_perp (AU), close / wide | 2.70 +/- 0.42 (both) | 2.42 / 2.84 | -- |
| s, close / wide | 0.929 / 1.086 | seeded | 0.95 - 1.07 (unmixed) |
| q (1e-3) | 5.37 +/- 0.41 | 5.31 | 5.31 |
| Keck lensrel, logp (E + N terms) | -- | -4.44 | -- |

What agrees: the Keck separation, `theta_E`, `t_E` and `pi_E` follow
from the physical parameters exactly as the paper's own relations give
them.  `mu_rel,G` differs by 0.8 sigma: the paper's 5.042 is an MCMC
average over D_L, while their own conversion
(mu_G = mu_H - v_earth pi_rel, v_earth = (4.152, -0.548) AU/yr) at
D_L = 3.44, D_S = 7.77 gives (4.759, 2.015), |mu_G| = 5.168, which is what
this model computes.  Not a convention difference.

**What does not agree, and is the finding of this example: rho.**  Here
`rho = theta_*/theta_E` is not a free light-curve parameter but follows
from the source's radius and distance, and the source is constrained
physically: its Hubble I magnitude, the red-clump extinction and MIST
make it a ~5100 K, 0.76 Rsun bulge dwarf, theta_* = 0.46 uas, so rho =
6.0e-4.  The light curve measures t_*/t_E = 4.0e-4 (theta_* = 0.30 uas).
Scored on the archive light curves at the paper's solution with a
uniform source (the C21 scan code), rho = 4.0e-4 gives chi2 1249.3 and
6.0e-4 gives 1431.3 (+182).  The paper never imposes a theta_* from the
source colour on the light curve (theta_E comes from the Keck mu_rel and
t_E alone), so its analysis cannot see this; a joint physical model has
to absorb it somewhere -- a hotter (smaller) source, which their
V_S - I_S = 2.11 +/- 0.16 with E(V-I) = 1.515, i.e. (V-I)_0 = 0.60,
actually favours (~6000 K), or a different t_E / source flux.  This fit
cannot use that colour (gap 1: under the shipped extinction law the same
V - I reads as (V-I)_0 = 0.91, a COOLER source), so the extinction-law
gap and the rho tension are the same problem.  It is the first thing a
converged run must be read for.

## Cost, and what a full run needs

One model evaluation is ~0.45 s single-core on an idle core (2L1S finite
source through VBM, 1252 epochs; ~2 s under this box's load), and the
model has 46 free parameters.  The 2-rung, 48-chain run took ~11 s per
step on 8 cores.  The shipped sampler block (8 rungs, `2 x n_params`
chains, 20000 draws) is ~740 evaluations per step: roughly 2.5 days on
64 cores, which is a cluster job (`examples/run_example.job`).  Two
start-up costs are worth knowing: the DE seed polish ran at ~2 sweeps a
minute and accepted nothing in its first 38 sweeps (it was stopped;
the short run used `seed_polish: "off"`), and the serial start and
start-ensemble plots took ~35 minutes before the first step.
