https://ui.adsabs.harvard.edu/abs/2015ApJ...802...76Y/abstract

This is a PSPL event seen from two distinct locations (Earth and Spitzer). It is fit with a Pytensor native implementation of the Paczyński curve; the config nevertheless uses PTDE, because it seeds all four of Yee et al.'s degenerate solutions and only PTDE spreads its chains across seeds and lets them cross between basins. It demonstrates the use of a Non-earth based ephemeris, using the real Spitzer ephemeris files from Horizons. The two separate sight lines directly and strongly constrains the lens mass.

Note the Spitzer data has no baseline observations, which makes its peak magnification uncertain.

## The magnitude zero points of the data files are arbitrary

The `.dat` files were converted from Yee et al.'s flux files (not shipped) with an arbitrary zero point of 25.0. Those OGLE fluxes are on a scale where 1 flux unit is I = 22, so the shipped `n20100310.I.OGLE.OB140939.dat` is **3.00 mag fainter than calibrated OGLE-IV I**: its baseline reads 18.40, while the calibrated baseline is 22 − 2.5 log₁₀(437.3) = 15.40. That matches Yee et al.'s baseline of 11.0 flux units on their I = 18 scale, and Gaia DR3 RP = 15.62 for the source. The Spitzer file went through the same zp 25; its calibrated zero point is not established here.

This is harmless as shipped, because each light curve's source and blend fluxes are free. It matters as soon as a light curve is tied to an SED or given a stated zeropoint (`zero_point`, `magsys`): correct the OGLE file by −3.00 mag first.

## The source proper motion is in the bulge frame

Yee et al.'s source proper motion (their Eq. 8) is OGLE relative astrometry, measured against the field's bulge stars, so the prior sits on `star.Source.pm_ra_sgra`/`pm_dec_sgra`. The comments in `ob140939.params.yaml` and conventions rule C31 give the evidence, and the absolute Gaia DR3 alternative.

