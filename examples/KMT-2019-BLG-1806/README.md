This is a 2L1S system with real-world data, set up by Aini do Rio Apa Vincenzi. It should be similar to DC2018_128, but with a published result and real data.

Published as KMT-2019-BLG-1806/OGLE-2019-BLG-1250 in Zang et al. 2023, AJ
165, 103 (arXiv:2210.12344). (This README used to cite arXiv:2102.01806,
which is KMT-2019-BLG-0797.)

Starting values: Zang et al. 2023 Table 4, the "Outer" u_0 > 0 solution
(t_0, u_0, t_E, s, log q, alpha and pi_E; rho is only an upper limit there
and is not seeded), with alpha converted as the params file shows.  Until
2026-10-14 the event starts were MMEXOFAST solution 0 of
`KMT-2019-BLG-1806-r_exozippy_init.json`, our own static (no-parallax) fit;
its t_E = 118.2 d is a no-parallax timescale, and the physical model here
always carries the parallax its masses and distances imply, so once the
start delivered that t_E exactly the data preferred 122.3 d by 82.6 nats.
Start logp: Zang +45,541.9, MMEXOFAST +45,420.9.

Coordinates: RA, Dec (J2000) = 18:02:09.01, -29:24:53.60 (270.53754,
-29.41489 deg; l, b = +1.41, -3.35), Zang et al. 2023 Table 1, matching
the KMTNet and OGLE (OGLE-2019-BLG-1250) alert pages. (Until 2026-10 the
params file carried 17:46:29.58, -24:16:20.17 -- OGLE-2017-BLG-1806, a
different event in the same table.) They set the line of sight for the
annual-parallax projection of Earth's orbit and the galactic-model priors.

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
