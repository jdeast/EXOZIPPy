# OGLE-2016-BLG-1045 -- a finite-source point lens (FSPL)

This is the example that exercises finite-source effects on a SINGLE lens.
Nothing else shipped covers that: `ob161003` fits a source radius on a
BINARY lens, and `DC2018_128` only toggles `finite_source` off. Here the
finite source IS the measurement -- the source transits the lens
(`u_0 = 1.31e-2` against `rho = 3.19e-2`, so `u_0 < rho`), which is what
makes `theta_E`, and through it the lens mass, measurable at all.

The directory name is correct and is **not** a misnamed KMT event. This is
OGLE-2016-BLG-1045 at 17:36:51.19 -34:32:39.7; its KMT designation is
KMT-2016-BLG-0848 (field SSO13N0507), and KMT-2016-BLG-1045 is a different
event 2.3 degrees away whose cross-ID is OB161756.

## Data

The Shin+18 reductions themselves, provided by J. Yee to JDE, committed here
unmodified. One format for all five files -- three whitespace columns,
`HJD-2450000 / magnitude / magnitude error` -- so there is no conversion
script in this directory and the example reads what the authors actually
fitted.

| file | N | role |
|------|---|------|
| `I_OGLE_I.nom`     | 683 | the only BASELINE, 6 years of it |
| `I_KMTC_I.pys`     | 243 | survey coverage |
| `I_KMTS_I.pys`     | 185 | survey coverage |
| `I_KMTA_I.pys`     | 146 | survey coverage |
| `I_Auckland_R.pys` |  91 | the PEAK |

1348 epochs in total, which is exactly the `N_data` Shin+18 quote for their
ground-based fit.

They are not interchangeable. **Auckland is what resolves the finite
source**: 47 of its 91 epochs fall within 0.2 d of `t_0`, where all three
KMT sites have ZERO (KMTC has 2 within 0.5 d, the others none). KMT's
survey cadence missed the peak. **OGLE is the only baseline**: without it
the source/blend split is unconstrained, because everything else sits on or
near the peak.

`I_OGLE_I.nom` is OGLE's `.nom` variant, whose errors already carry the
Skowron et al. 2016 rescaling; the factor below is applied on top, exactly
as the reductions specify.

GROUND-BASED ONLY. Shin+18's headline result is a Spitzer space-based
parallax, and `I_Spitzer_L.dat` (24 epochs) exists, but it is not included
here and no parallax is fit. Adding it is the natural extension and needs
an `observer_location` with its ephemeris. A `color/` set of DoPHOT data
(for the source colour) also exists and is likewise not used.

## Published solution

Shin et al. 2018, ApJ 863, 23,
https://ui.adsabs.harvard.edu/abs/2018ApJ...863...23S/abstract -- Table 2,
the (-, +) "Actual" column. All six of their cases agree within the quoted
errors, so this is the solution to the precision that matters.

| parameter | value |
|-----------|-------|
| t_0 (HJD')| 7559.201 +/- 0.001 |
| u_0       | -1.308e-2 +0.033/-0.042 |
| t_E       | 11.981 +0.064/-0.098 d |
| rho       | 3.186e-2 +0.033/-0.026 |
| theta_E   | 0.244 +/- 0.015 mas (their Eq. 7) |
| theta_*   | 7.80 +/- 0.47 uas (their Eq. A1) |
| M_L       | 0.08 +/- 0.01 M_sun |
| D_L       | 5.02 +/- 0.14 kpc |

`theta_*/theta_E = 0.0320` reproduces the published `rho`, which is the
consistency check worth keeping.

## Limb darkening and error rescaling

Both ship with the reductions and are used as given.

| dataset | Gamma (Claret+2000) | error factor |
|---------|---------------------|--------------|
| OGLE    | 0.5103 | 0.913 |
| KMTC    | 0.5103 | 1.116 |
| KMTS    | 0.5103 | 1.501 |
| KMTA    | 0.5103 | 1.446 |
| Auckland| 0.6583 | 2.370 |

The rescaling is `error_new = A * sqrt(error_old**2 + B**2)` with `B = 0`
throughout, so it is a pure multiplicative factor and maps one-to-one onto
`err_scale`. Applying it brings chi2/N to 1.00-1.04 at OGLE and all three
KMT sites, which is the check that these are the right factors for these
files.

**Limb darkening is converted, not copied.** Shin+18 quote `Gamma`;
`band.u1` is the ordinary linear coefficient `u` of
`I(mu)/I(0) = 1 - u(1 - mu)`, and `Gamma = 2u/(3 - u)`, so
`u = 3*Gamma/(2 + Gamma)`:

    Gamma_I = 0.5103  ->  u = 0.60985
    Gamma_R = 0.6583  ->  u = 0.74292

Taking the Gammas as `u` verbatim would put the I-band coefficient 16% low.
`ld_law: linear` is mandatory on a finite-source fit (mulensing.md).

One I band serves OGLE and all three KMT sites: Shin+18 give them the same
coefficient, and a band carries the bandpass, not the site -- per-site
zeropoints are absorbed by `f_source`/`f_blend`.

## Conventions

- `t_0`, `u_0`, `t_E` and `rho` are the standard MulensModel/Skowron
  quantities, which are EXOZIPPy's verbatim (conventions.md C18). No alpha
  or q to shift or invert -- this is a single lens.
- `u_0`'s SIGN is not measurable from these data. With one lens and no
  parallax fit, the likelihood is exactly symmetric under `u_0 -> -u_0`,
  which is why Shin+18 carry (-, +) and (+, +) as a degenerate pair whose
  columns differ only in that sign. Seeded at the published negative value.

## Blending, and a bound that is NOT a problem

The fitted blending is small and consistent: `q_source = f_source/f_total`
comes out 1.022-1.030 at OGLE and all three KMT sites -- about 3% blending
-- nowhere near `q_source`'s `[0, 2]` bound.

That is worth recording, because an earlier version of this example was
built on re-derived three-column flux files instead of these reductions and
measured `q_source = 2.161 +/- 0.006` at KMTC, appearing to need the bound
widened. It did not. That number was an artifact of reconstructing a total
flux from difference imaging with a solved reference; the bound was fine
and the data were not. Using the reductions as distributed makes it vanish.

Auckland is the exception, and honestly so: its 91 epochs never reach
baseline, so its blending is unconstrained by its own data (free solution
`q_source = 4.3 +/- 1.5`, consistent with the other sites at 2.2 sigma and
with much else besides). It is seeded at no blending, which the sampler is
free to leave.

## Two caveats

- **Only one band's limb darkening is used.** EXOZIPPy warns at startup:
  "Multiple bands for finite-source instruments; using first band's u1."
  Auckland's R-band `u1` is declared and then ignored, and its light curve
  is modelled with the I-band coefficient. Four of the five datasets are I,
  so that is the lesser error, but it is a departure from Shin+18, who use
  both.
- **The source radius is not separately identifiable.** `rho` is derived as
  `theta_*/theta_E`, and EXOZIPPy warns that `theta_E` and `D_S` absorb any
  rescaling of the radius. It is seeded (13.42 R_sun, which is
  `theta_* = 7.80 uas` at `D_S = 8 kpc`) and left free, with the transit's
  upper limit on `rho` as the real constraint.

`t_E` and `rho` are deliberately NOT seeded directly: both are derived from
the physical chain, and seeding them alongside the physical leaves
over-determines the relaxation engine at equal rank -- the failure
`ob170114`'s params file documents, and one measured here.

## Running it

    cd examples/ob161045 && poetry run exozippy ob161045.yaml

Start logp +21063.
