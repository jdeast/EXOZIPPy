https://ui.adsabs.harvard.edu/abs/2015ApJ...802...76Y/abstract

This is a PSPL event seen from two distinct locations (Earth and Spitzer). It is fit with a Pytensor native implementation of the Paczyński curve; the config nevertheless uses PTDE, because it seeds all four of Yee et al.'s degenerate solutions and only PTDE spreads its chains across seeds and lets them cross between basins. It demonstrates the use of a Non-earth based ephemeris, using the real Spitzer ephemeris files from Horizons. The two separate sight lines directly and strongly constrains the lens mass.

Note the Spitzer data has no baseline observations, which makes its peak magnification uncertain.

## The data are Yee et al.'s published fluxes

Both light curves are read as fluxes (`data_format: flux`), exactly as published, with no conversion to magnitudes. The OGLE fluxes are on a scale where 1 flux unit is I = 22 (calibrated OGLE-IV I): the baseline of 437.3 units is I = 15.40, matching Yee et al.'s baseline of 11.0 flux units on their I = 18 scale and Gaia DR3 RP = 15.62 for the source. The Spitzer fluxes' zero point is not established here.

Nothing in this fit reads a zero point, because there is no SED and therefore no zeropoint tie. A fit that adds one states the calibration in the params file, `mulensinstrument.OGLE4.zeropoint: {mu: 22.0, sigma: <how much you trust it>}`, and leaves the Spitzer light curve untied unless its zero point is known. (Earlier versions shipped `.dat` magnitude files converted with an arbitrary zero point of 25, 3.00 mag fainter than calibrated, with errors rounded to 0.001 mag.)

## The source proper motion is in the bulge frame

Yee et al.'s source proper motion (their Eq. 8) is OGLE relative astrometry, measured against the field's bulge stars, so the prior sits on `star.Source.pm_ra_sgra`/`pm_dec_sgra`. The comments in `ob140939.params.yaml` and conventions rule C31 give the evidence, and the absolute Gaia DR3 alternative.

## Two analyses: Yee et al.'s inputs, and today's

- **`ob140939.yaml` is the validation.** It reproduces Yee et al. (2015) using only the inputs they used: the OGLE and Spitzer light curves and their bulge-relative source proper motion. It recovers their preferred solution, Δu₀,−,−, at about 90% of the posterior (89–94% across independent repeats). Their Δχ² = 8 and 17 solutions are rejected, and the lens comes out at ~0.23 M☉ and ~3.3 kpc (theirs: 0.23 ± 0.07 M☉, 3.1 ± 0.4 kpc).
- **`ob140939_today.yaml` is the best-practice analysis with everything available today:**
  - **Proper motion:** the source's absolute proper motion from Gaia DR3 (4118632779506798848) instead of the OGLE relative one.
  - **Source SED:** Gaia G and 2MASS JHKs, fetched with `mkticsed` (TIC 169097367) and vetted for crowding. Gaia BP/RP are dropped for their excess factor and WISE for its 6″ beam. The SED is tied to the OGLE light curve through the calibrated zero point (22.0 ± 0.2; the width covers a 0.16 mag SED-vs-OGLE I-band systematic).
  - **Extinction:** an A_V prior from the VVV red-clump map (Surot et al. 2020) instead of the Schlegel cap.
  - **Colour:** Yee et al.'s independently measured instrumental (I − [3.6]) source colour ties the Spitzer source flux to OGLE's.

  It gives Δu₀,−,− at 86%, which is the same preference within the repeat scatter, and a lens at 0.22 ± 0.04 M☉ and 3.5 ± 0.4 kpc. The source is a reddened bulge K giant (Teff ~4300 K, ~19 R☉, ~9 kpc).

