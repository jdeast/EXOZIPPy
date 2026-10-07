https://ui.adsabs.harvard.edu/abs/2015ApJ...802...76Y/abstract

This is a PSPL event seen from two distinct locations (Earth and Spitzer). It is fit with a Pytensor native implementation of the Paczyński curve; the config nevertheless uses PTDE, because it seeds all four of Yee et al.'s degenerate solutions and only PTDE spreads its chains across seeds and lets them cross between basins. It demonstrates the use of a Non-earth based ephemeris, using the real Spitzer ephemeris files from Horizons. The two separate sight lines directly and strongly constrains the lens mass.

Note the Spitzer data has no baseline observations, which makes its peak magnification uncertain.

## The data are Yee et al.'s published fluxes

Both light curves are read as fluxes (`data_format: flux`), exactly as published, with no conversion to magnitudes. The OGLE fluxes are on a scale where 1 flux unit is I = 22 (calibrated OGLE-IV I): the baseline of 437.3 units is I = 15.40, matching Yee et al.'s baseline of 11.0 flux units on their I = 18 scale and Gaia DR3 RP = 15.62 for the source. The Spitzer fluxes' zero point is not established here.

Nothing in this fit reads a zero point, because there is no SED and therefore no zeropoint tie. A fit that adds one states the calibration in the params file, `mulensinstrument.OGLE4.zeropoint: {mu: 22.0, sigma: <how much you trust it>}`, and leaves the Spitzer light curve untied unless its zero point is known. (Earlier versions shipped `.dat` magnitude files converted with an arbitrary zero point of 25, 3.00 mag fainter than calibrated, with errors rounded to 0.001 mag.)

## The source proper motion is in the bulge frame

Yee et al.'s source proper motion (their Eq. 8) is OGLE relative astrometry, measured against the field's bulge stars, so the prior sits on `star.Source.pm_ra_sgra`/`pm_dec_sgra`. The comments in `ob140939.params.yaml` and conventions rule C31 give the evidence, and the absolute Gaia DR3 alternative.

