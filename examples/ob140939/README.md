https://ui.adsabs.harvard.edu/abs/2015ApJ...802...76Y/abstract

This is a PSPL event seen from two distinct locations (Earth and Spitzer). It is fit with a Pytensor native implementation of the Paczyński curve; the config nevertheless uses PTDE, because it seeds all four of Yee et al.'s degenerate solutions and only PTDE spreads its chains across seeds and lets them cross between basins. It demonstrates the use of a Non-earth based ephemeris, using the real Spitzer ephemeris files from Horizons. The two separate sight lines directly and strongly constrains the lens mass.

Note the Spitzer data has no baseline observations, which makes its peak magnification uncertain.

## The data are Yee et al.'s published fluxes

Both light curves are read as fluxes (`data_format: flux`), exactly as published, with no conversion to magnitudes. The OGLE fluxes are on a scale where 1 flux unit is I = 22 (calibrated OGLE-IV I): the baseline of 437.3 units is I = 15.40, matching Yee et al.'s baseline of 11.0 flux units on their I = 18 scale and Gaia DR3 RP = 15.62 for the source. The Spitzer fluxes' zero point is not established here.

Nothing in this fit reads a zero point, because there is no SED and therefore no zeropoint tie. A fit that adds one states the calibration in the params file, `mulensinstrument.OGLE4.zeropoint: {mu: 22.0, sigma: <how much you trust it>}`, and leaves the Spitzer light curve untied unless its zero point is known. (Earlier versions shipped `.dat` magnitude files converted with an arbitrary zero point of 25, 3.00 mag fainter than calibrated, with errors rounded to 0.001 mag.)

## The source proper motion is in the bulge frame

Yee et al.'s source proper motion (their Eq. 8) is OGLE relative astrometry, measured against the field's bulge stars, so the prior sits on `star.Source.pm_ra_sgra`/`pm_dec_sgra`. Conventions rule C31 and "Details behind the parameter files" below give the evidence, and the absolute Gaia DR3 alternative.

## Two analyses: Yee et al.'s inputs, and today's

- **`ob140939.yaml` is the validation.** It reproduces Yee et al. (2015) using only the inputs they used: the OGLE and Spitzer light curves and their bulge-relative source proper motion. It recovers their preferred solution, Δu₀,−,−, at about 90% of the posterior (89–94% across independent repeats). Their Δχ² = 8 and 17 solutions are rejected, and the lens comes out at ~0.23 M☉ and ~3.3 kpc (theirs: 0.23 ± 0.07 M☉, 3.1 ± 0.4 kpc).
- **`ob140939_today.yaml` is the best-practice analysis with everything available today:**
  - **Proper motion:** the source's absolute proper motion from Gaia DR3 (4118632779506798848) instead of the OGLE relative one.
  - **Source SED:** Gaia G and 2MASS JHKs, fetched with `mkticsed` (TIC 169097367) and vetted for crowding. Gaia BP/RP are dropped for their excess factor and WISE for its 6″ beam. The SED is tied to the OGLE light curve through the calibrated zero point (22.0 ± 0.2; the width covers a 0.16 mag SED-vs-OGLE I-band systematic).
  - **Extinction:** an A_V prior from the VVV red-clump map (Surot et al. 2020) instead of the Schlegel cap.
  - **Colour:** Yee et al.'s independently measured instrumental (I − [3.6]) source colour ties the Spitzer source flux to OGLE's.

  It gives Δu₀,−,− at 86%, which is the same preference within the repeat scatter, and a lens at 0.22 ± 0.04 M☉ and 3.5 ± 0.4 kpc. The source is a reddened bulge K giant (Teff ~4300 K, ~19 R☉, ~9 kpc).

## Details behind the parameter files

These notes used to sit inline in `ob140939.params.yaml` and
`ob140939_today.params.yaml`.

**The source proper motion.** Yee et al. measure it from 4 years of
OGLE-IV astrometry of the baseline object: `mu_S,hel(N,E) = (-0.64, -5.31)
+/- 0.45 mas/yr` (their Eq. 8). This is the datum behind their
degeneracy-breaking argument: with `mu_S` measured, the disk-rotation term
in the galactic kinematic prior predicts the lens-source relative motion
direction to ~8.5 deg, which excludes the `u0,-,+` mode (direction ~130
deg against the expected ~54). Without it the source pm marginalizes over
the ~2.6 mas/yr bulge dispersion and the prior separates the
ecliptic-degenerate modes by only ~1 nat. The value is OGLE RELATIVE
astrometry -- the source's motion relative to the mean of the field
(bulge) stars; "hel" in their notation means not-geocentric, not absolute
-- so it goes on the Sgr-A*-relative `pm_*_sgra` parameters, never on the
absolute (ICRS) `star.Source.pm_ra`/`pm_dec` (conventions.md C31). Their
own Eq. 9 reproduces their 53.9 deg only for a bulge-relative `mu_S`.
Absolute, it is `(pm_ra, pm_dec) ~ (-8.43, -6.25)`. The width is their
0.45, deliberately: this example reproduces Yee et al. as a validation and
does not add the ~0.7 mas/yr zero-point term C31 recommends in quadrature
(field mean vs Sgr A*, measured 0.7 here); a science fit that trusts the
frame no more than it should would use ~0.83.

The more precise absolute alternative -- used INSTEAD of those two
entries, as `ob140939_today` does -- is Gaia DR3 4118632779506798848
(0.32" from the event, G = 16.79, RUWE 0.99, consistent with the baseline
I ~ 15.4). Its offset from the median of the surrounding field giants,
`(E, N) ~ (-4.9, -1.1) mas/yr`, matches Yee's relative value within ~1
sigma, which is the evidence that theirs is field-relative. It measures
the absolute frame directly, so neither the Sgr A* conversion nor its
field-mean zero point enters.

**Error scaling.** `err_scale` starts near its posterior so that the
whitening probe measures posterior widths instead of a huge start-point
gradient (a start thousands of nats low once measured scales ~1000x too
small and ~90% of draws diverged). The Spitzer file is Yee's raw published
photometry (flux errors ~0.09%, ~1 mmag); her fits imply working errors
~17x larger: the paper never states a renormalization, but every fit lands
at chi2/dof ~ 1 (Spitzer-only: 28.0/26, Table 3) and the Fig. 1 Spitzer
residuals scatter ~+/-20 mmag. `err_scale` absorbs that factor; do not
"fix" the data file.

**`ob140939_today` only.**

- *Spitzer/OGLE color.* Yee et al. Sec. 5.3 measure the source's
  instrumental color `(I - [3.6])_S = -1.216 +/- 0.044` from field-star
  color-color regressions (I-[3.6] vs V-I, and I-H), independent of the
  light-curve model. Both flux files share one instrumental scale
  (verified: fitted without this prior, the Spitzer/OGLE source-flux ratio
  is 0.39 +/- 0.06), so `F_S,Spitzer / F_S,OGLE = 10**(-0.4*1.216) =
  0.3266`, fractional width `0.4 ln(10) 0.044 = 0.0405`, i.e. sigma =
  0.0405 x ~171 = 7.0 flux units. It does not separate the `u0,-,+/-`
  ecliptic pair (they predict nearly the same color, -1.215 and -1.179
  in Yee's Table 1); it pins the poorly constrained Spitzer source flux
  and disfavors the `u0,+,+/-` pair further.
- *Zeropoint tie.* The band uses the grid's Cousins I (OGLE-IV I is
  calibrated onto, and close to, Cousins I). Sigma 0.2 covers the I-band
  calibration and bulge extinction-law systematics: the SED (G + JHKs)
  predicts the baseline object 0.16 mag fainter in I than OGLE measures,
  and with sigma 0.03 the fit absorbed that by inventing a +14% blend
  (Yee: blending ~0) and shifting `u_0` and `t_E`. With 0.2 the zero point
  absorbs it instead (fitted 22.13 +/- 0.15). Spitzer is left untied (no
  absolute [3.6] zero point; no IRAC BC column); the color prior above
  still ties its source flux to OGLE's.
- *Extinction.* `mkticsed`'s `star.Source.appks` prior is dropped (it is
  the input to the Mann M-dwarf relations, which this topology does not
  have; the Ks datum is already an SED row), and so is its Schlegel
  `A_V < 3.32` upper limit (a full-column limit, unreliable at b = 3.6 deg
  toward the bulge). `A_V` instead comes from the VVV red-clump extinction
  map of Surot et al. (2020; VizieR J/A+A/644/A140): `E(J-Ks) = 0.44 +/-
  0.01` at 2.9" (nine nearest 0.43-0.45), inverted through the SED's own
  law (`components/sed/extinction.py`, `av_from_colour_excess` at 2MASS
  J/Ks effective wavelengths 1.235/2.159 um): `A_V = 0.44 / (0.3048 -
  0.1248) = 2.44` (+/- 0.06 from the map alone). Sigma 0.49 (20%) covers
  the extinction-law uncertainty toward the bulge: the NIR slope there is
  steeper than this table's (Nishiyama et al. 2009), which moves the A_V
  implied by a fixed `E(J-Ks)` by tens of percent (and upward).
- *Metallicity and mass.* `[Fe/H]` gets the broad bulge distribution
  (Ness et al. 2013, Zoccali et al. 2017: mean ~ -0.1 to 0, dispersion
  ~0.4), there being no spectrum; the source is a bulge giant, old-bulge
  RGB mass ~0.8-1.25 Msun, and the mass reaches the SED only through
  `loggsed`.
