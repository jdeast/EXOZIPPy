This is a 2L1S system with real-world data, set up by Aini do Rio Apa Vincenzi. It should be similar to DC2018_128, but with a published result and real data.

Published as KMT-2019-BLG-1806/OGLE-2019-BLG-1250 in Zang et al. 2023, AJ
165, 103 (arXiv:2210.12344). (This README used to cite arXiv:2102.01806,
which is KMT-2019-BLG-0797.)

The `sed:` block (with an empty `filters:` list) exercises the pure
f_source constraint mode: the SED-predicted source I magnitude is tied to
each light curve's baseline source flux through a per-lightcurve zeropoint.
The prior is the published calibration: Zang et al. calibrate the KMT I
magnitudes to the standard (Vega) I band with the OGLE-III catalog, so the
files' flux 10**(-0.4 m) has zeropoint 0, with sigma = 0.07 mag, their
I_S uncertainty (I_S = 21.35 +/- 0.07). Cross-check: at the shipped
start the three files give m_S = 21.28 / 21.28 / 21.31 (C/S/A). Because
all three light curves are I band, the absolute calibration is set by the
zeropoint prior, not the data -- the light curves only pin the site-to-site
relative zeropoints (the three zeropoint posteriors are ~fully
correlated), so the stacked prior is 0.07/sqrt(3) ~ 0.04 mag if the three
sites' calibration errors were independent; they share one OGLE-III
calibration, so read it as ~0.07. Adding real catalog photometry of the
baseline object to the `.sed` file would break this degeneracy.
