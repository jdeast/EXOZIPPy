OGLE-2016-BLG-1003: a binary-source, binary-lens (2S2L / NSNL) event, published in
Jung et al. 2017, ApJ 841, 75:
https://ui.adsabs.harvard.edu/abs/2017ApJ...841...75J/abstract

Data downloaded from the paper is dbf1.txt. convert_data.py generates the *.dat
EXOZIPPy input files.

ob161003.params.yaml is seeded with the 2S2L "standard" solution from Table 1 of
the paper.  This example exercises EXOZIPPy's NSNL pathway:

- Two source stars (star.2 = SourceA, star.3 = SourceB) share one binary lens
  (star.0 = Lens, star.1 = LensB; q ~ 1.19, so the secondary is stellar).
- Per-source trajectory parameters (t_0, u_0, rho, and the derived t_E, theta_E,
  pi_E, mu_rel chains) are vectors with one element per source.  In the params
  file they are addressed by the source star's instance name
  (lens.SourceA.t_0, lens.SourceB.t_0) or by slot index (lens.0.t_0, lens.1.t_0).
- Each instrument fits a chromatic source flux ratio q_flux = f_s,B/f_s,A in
  addition to the usual f_source (total) and f_blend.
- theta_E/t_E/rho/pi_E are referenced to the TOTAL lens mass
  (lens.mlens_total), matching the community convention for binary lenses.
- Source radii (star.SourceA.radius, star.SourceB.radius) float freely
  (bounded, no sigma fix): rho is DERIVED from radius via
  calc_rho(radius, distance, theta_E), so the finite-source lightcurve shape
  constrains radius directly through rho rather than radius being clamped at
  the value back-solved from a seeded rho. Lens radii stay fixed (no
  finite-source coupling; unconstrained without SED data).

Conventions: Jung et al. report alpha = 48.243 deg; MulensModel (EXOZIPPy's
magnification backend) uses alpha_MM = 180 - alpha_paper (see the comment in
ob161003.params.yaml, verified by a chi-squared scan over the discrete
convention choices).

Run with:

    cd examples/ob161003 && poetry run exozippy ob161003.yaml
