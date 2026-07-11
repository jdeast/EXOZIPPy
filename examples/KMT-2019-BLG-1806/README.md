This is a 2L1S system with real-world data, set up by Aini do Rio Apa Vincenzi. It should be similar to DC2018_128, but with a published result and real data.

See https://arxiv.org/abs/2102.01806

The `sed:` block (with an empty `filters:` list) exercises the pure
f_source constraint mode: the SED-predicted source I magnitude is tied to
each light curve's baseline source flux through a per-lightcurve zeropoint
with a 0 +/- 0.2 mag prior. Because all three light curves are I band,
the absolute calibration is set by the zeropoint prior, not the data --
the light curves only pin the site-to-site relative zeropoints (to ~0.006
mag; the three zeropoint posteriors are ~fully correlated). The source's
radius/teff/distance posteriors are therefore anchored by the stacked
prior (0.2/sqrt(3) ~ 0.12 mag, which assumes the three sites' calibration
errors are independent). Adding real catalog photometry of the baseline
object to the `.sed` file would break this degeneracy.
