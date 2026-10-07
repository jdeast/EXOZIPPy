# OGLE-2016-BLG-1045 -- a finite-source point lens (FSPL)

**STATUS: runnable, but it is a finite-source DEMONSTRATION, not a
reproduction of the published result.** The light curves are DIA difference
fluxes and no reference flux is available for them, so each one has been
offset to ZERO BLENDING (see "Difference flux" below). The trajectory
(`t_0`, `u_0`, `t_E`, `rho`) remains meaningful; `theta_E`, `theta_*` and
the lens mass will NOT reproduce Shin et al. 2018, because the blending is
imposed rather than fitted. Start logp is -2057, with the data term at
-1956.

This is intended to be the example that exercises finite-source effects on a
SINGLE lens. Nothing shipped covers that today: `ob161003` fits a source
radius on a BINARY lens, and `DC2018_128` only toggles `finite_source` off.
Here the finite source IS the measurement -- the source transits the lens
(`u_0 = 1.31e-2` against `rho = 3.19e-2`, so `u_0 < rho`), which is what
makes `theta_E`, and through it the lens mass, measurable at all.

The directory name is correct and is **not** a misnamed KMT event. The data
are OGLE-2016-BLG-1045 at 17:36:51.19 -34:32:39.7, confirmed against both
the OGLE EWS page and Shin+18 Sec 2.1.1. KMT-2016-BLG-1045 is a different
event 2.3 degrees away (17:47:54.91 -32:52:02.28, cross-ID OB161756); the
two surveys number independently.

## Published solution

Shin et al. 2018, ApJ 863, 23, "OGLE-2016-BLG-1045: A Test of Cheap
Space-based Microlens Parallaxes",
https://ui.adsabs.harvard.edu/abs/2018ApJ...863...23S/abstract

Table 2, the (-, +) "Actual" column. All six of their cases agree to within
the quoted errors, so this is the solution to the precision that matters:

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

GROUND-BASED ONLY. The paper's headline result is a Spitzer space-based
parallax (`pi_E = 0.355`), but no Spitzer photometry ships with this data
set, so no parallax is fit here. That matches MMEXOFAST's own
`examples/use_case_03_ob1045.py`, likewise titled "Analyze the ground-based
data".

## Difference flux: why the data had to be offset

The four light curves are DIA difference fluxes -- the baseline sits at
zero, with the source's own flux subtracted away by the reference image.
EXOZIPPy's microlensing flux model cannot represent that
(`components/mulensing/defaults.yaml`, `physics.calc_f_source/f_blend`):

    f_total  = 10**log_f_total            (strictly positive)
    f_source = f_total * q_source
    f_blend  = f_total * (1 - q_source)   with q_source bounded [0, 2]

Solving the linear flux problem at the published trajectory gives, for
KMTC, `f_source = 1.784` and `f_blend = -1.774` in the units shipped here.
So `f_total = f_source + f_blend = 0.0099`, and `q_source = f_source/f_total
= 181` -- against an upper bound of 2. The bound is not arbitrary:
`q_source <= 2` says the blend can be at most as negative as the total
flux, which is true of a TOTAL-flux light curve and false of a difference
one.

This was structural, not a bad starting guess: pinning every physical
parameter at the published values still left `mulensinstrument.model` at
-458114.

**`q_source`'s bound is correct and must not be raised.** It encodes "the
blend is at most as negative as the total flux", which is true of the
total-flux curves it was written for. Widening it to admit 181 would
loosen the prior on every other microlensing fit to accommodate one file,
and 181 is meaningless for what the parameter represents.

### The reference flux, which would fix this exactly

The right fix is to add the DIA reference flux back per site. It is
recoverable **exactly** from a NATIVE KMT pySIS file, whose five columns
are `HJD' dflux dflux_err mag mag_err`: `mag` and `dflux` together solve

    mag = zp - 2.5*log10(ref - dflux)

for `ref`. Verified on MulensModel's native KB180003 set -- all three
sites return `ref = 1584.9` at `zp = 28.000` with a residual rms of
2.8e-5 mag.

**These files cannot supply it.** They carry three columns and no `mag`
column, so that information is absent from them -- whether it was ever
present and removed, or never written, is not known here. Nor can the
constant be borrowed: it is event-specific (KB180003's reference sits at
exactly mag 20.000, while OB161045 needs roughly 17,500).

### What was done instead, and what it costs

Each curve is offset to ZERO BLENDING: the linear flux problem is solved at
the published trajectory and `-f_blend` is added, putting the baseline at
the source flux. The result is exact to rounding (`f_blend ~ 1e-9`,
`q_source = 1.00000`).

- The blending is **imposed, not fitted**. Any blend flux this example
  reports is an artifact of the offset.
- Shin+18 derive `theta_*` from a CMD analysis that depends on the source
  flux, so `theta_E = 0.244 mas` and `M_L = 0.08 M_sun` will NOT be
  reproduced.
- `convert_data.py` prints a warning naming both the cause and the fix
  every time it runs, and repeats it in each generated file's header.

To remove the approximation: obtain the native five-column KMT pySIS files
for OB161045 (and Auckland's reference flux), set `REFERENCE_FLUX` in
`convert_data.py`, and the offset step turns itself off. The `log_f_total`
and `q_source` seeds in the params file must then be recomputed, since they
carry the same assumption.

### A note for native support

A `data_format:` that reads difference flux natively -- converting to total
flux on read, with an optional reference flux or magnitude to tie it to the
SED constraint -- is worth having, and now exists as `data_format: dia`.

How common that five-column layout actually is, though, is NOT established
here. The only specimen in reach is MulensModel's KB180003, and this
repository's other KMT files (`kb180087_obj3/L_data/*.pys` and
`KMT-2019-BLG-1806/*.pys`) are three-column MAGNITUDE files, not difference
flux at all. KMT photometry therefore arrives in at least three shapes, and
`dia` should be read as a format that is SUPPORTED when supplied, not one
that can be assumed.

It was deliberately not built around the files in this directory, which are
three-column and so specify nothing about the native layout; KB180003 is
the only specimen in reach and is what the tests use.

## Data

`n20*.OB161045.txt` are copied verbatim from MMEXOFAST
(`source/mmexofast/data/OB161045/`) and kept as the provenance record. Its
`00README.txt` reads: "KMT data from KMT website. Spitzer and Auckland
provided by Spitzer Microlensing Team and microFUN (authorized by Jennifer
C. Yee)" -- carried forward here because this repository now redistributes
it.

`convert_data.py` generates the `*.dat` files the config reads. It fixes
two things, both established by measurement rather than assumed:

- **The KMT flux sign, which looks native rather than like a defect.**
  These files report a difference flux that grows MORE NEGATIVE as the star
  brightens. One independent set points the same way -- MulensModel's
  KB180003 (`data/photometry_files/KB180003/*.pysis`), whose brightest
  epoch at mag 11.579 carries `dflux = -3,699,974`, and both sets share a
  baseline `dflux` near -337. That is corroboration from a single event,
  not a survey of KMT's output, so it is the likely reading rather than a
  settled one. The flip does not depend on it: at sign +1 the fitted source
  flux is NEGATIVE, which is unphysical either way. Auckland's microFUN
  photometry is already positive.
- **The scale.** Raw DIA counts put `f_source` at ~1.8e4, against
  `f_source`'s default bound of [0, 1000] -- a start 150,000 nats inside
  the wall. The files are divided by a round power of ten (1e4 for KMT, 1e2
  for Auckland). A microlensing flux unit is arbitrary with no `sed:` block
  to tie it to a magnitude, so this is free; `DC2018_128` ships normalized
  flux for the same reason.

## Conventions, settled

- `t_0`, `u_0`, `t_E` and `rho` are the standard MulensModel/Skowron
  quantities, which are EXOZIPPy's verbatim (conventions.md C18). No alpha
  or q here to shift or invert -- this is a single lens.
- `u_0`'s SIGN is not measurable from these data. With one lens and no
  parallax fit, the likelihood is exactly symmetric under `u_0 -> -u_0`,
  which is why Shin+18 carry (-, +) and (+, +) as a degenerate pair whose
  columns differ only in that sign. Seeded at the published negative value.
- **Limb darkening is converted, not copied.** Shin+18 Table 1 quotes
  `Gamma`; `band.u1` is the ordinary linear coefficient `u` of
  `I(mu)/I(0) = 1 - u(1 - mu)`. They are related by `Gamma = 2u/(3 - u)`:

      Gamma_I = 0.5103  ->  u = 0.60985
      Gamma_R = 0.6583  ->  u = 0.74292

  Taking the Gammas as `u` verbatim would put the I-band coefficient 16%
  low. The R value is the one Shin+18 modified for Auckland's Wratten 12
  filter, `(Gamma_R + Gamma_V)/2`.
- `ld_law: linear` is mandatory on a finite-source fit; Band's own default
  is quadratic, which on a band only microlensing reads leaves one
  combination of the sampled Kipping pair constrained by nothing but its
  prior (mulensing.md).

## Two further caveats

- **Only one band's limb darkening is used.** EXOZIPPy warns at startup:
  "Multiple bands for finite-source instruments; using first band's u1."
  So Auckland's R-band `u1` is declared and then ignored, and its light
  curve is modelled with the I-band coefficient. Three of the four data
  sets are I, so that is the lesser error, but it is a departure from
  Shin+18, who use both.
- **The source radius is not separately identifiable.** `rho` is derived as
  `theta_*/theta_E`, and EXOZIPPy warns that `theta_E` and `D_S` absorb any
  rescaling of the radius. It is seeded (13.42 R_sun, which is `theta_* =
  7.80 uas` at `D_S = 8 kpc`) and left free, with the transit's upper limit
  on `rho` as the real constraint. An `sed:` block or an explicit radius
  prior would break the degeneracy properly.

`t_E` and `rho` are deliberately NOT seeded directly: both are derived from
the physical chain, and seeding them alongside the physical leaves
over-determines the relaxation engine at equal rank -- the failure
`ob170114`'s params file documents, and one measured here (with them
seeded, the built start put `u_0` at -0.0248 against the published -0.01308
and moved between builds).
