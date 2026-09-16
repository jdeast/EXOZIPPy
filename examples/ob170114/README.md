# OGLE-2017-BLG-0114: a planetary binary lens with source orbital motion (xallarap)

> **WORK IN PROGRESS -- the start is verified at the LIGHT-CURVE level
> and matches MulensModel to the third decimal, but no production fit has
> run.**  What is verified at the start (2026-09, on the 351 OGLE epochs
> in this directory): the built magnification curve reproduces
> MulensModel's full 2L1S + parallax + xallarap model at the published
> values to `max|dA| = 0.0011` (0.033% peak-relative, rms 0.0004), and it
> FITS this photometry as well as MulensModel does -- chi2 = 2061.8
> against MulensModel's 2063.0 with the file's own errors, i.e.
> chi2/N = 1.498 vs 1.499 at the paper's error renormalization
> k_I = 1.98.  The derived quantities land on the printed ones:
> `t_E = 172.999` (173.0), `theta_E = 0.86000` (0.86),
> `|pi_E| = 0.209805` (0.209805), `xi_a = 0.19999` (0.200).
>
> Two earlier revisions were wrong, both recorded in full in the params
> file because each is a trap a reader can walk into:
>
> 1. The first verified only the shift TRACK and the derived quantities,
>    never the likelihood.  The C25 mapping and the xallarap projection
>    each carried a sign error that cancelled in the track comparison and
>    inverted the shift the light curve applied, so the "verified" start
>    fit this photometry WORSE than switching xallarap off (chi2 781,739
>    against 548,162 off; review 2.6.13).
> 2. The second fixed the signs but seeded `M_h = 0.50`, the paper's
>    printed median.  `theta_E = kappa M_tot |pi_E|` makes that
>    inconsistent with the printed `theta_E = 0.86` by 1.5%, which dragged
>    the derived `t_E` to 176.011 and `xi_a` to 0.19698 and cost +53.9
>    chi2 (2113 rather than 2062) -- entirely parametric, not a code
>    error.  See the note at `star.L1.mass`.
>
> The photometry here is the EWS quick-look file, not the re-calibrated
> set the paper fit (see Data).

The shipped worked example of `source_orbital_motion: keplerian`
(conventions.md **C25**; review 8.6.9): a planetary microlensing event
whose light-curve asymmetry is modeled as the SOURCE orbiting a dark
companion, entering the trajectory at exactly the slot the parallax
occupies -- with **no new sampled parameters**.  The orbit's period,
eccentricity, orientation, and the companion's mass (through the
barycentric scale `a_1/(D_S theta_E)`) are the coordinates.

* Mroz, M. J., et al. 2026 (arXiv:2606.20555) -- the analysis this example
  reproduces (2L1S/2L2S + parallax + xallarap).  Their "ET" mode samples
  the source's physical parameters directly, which is EXOZIPPy's native
  architecture; their "Std" mode uses the Zhai et al. (2024) `xi_*`
  elements, which map onto this config through C25's verified closed form.
* Poleski et al. 2021 (arXiv:2104.02079) -- the discovery analysis
  (wide-orbit planet candidate; superseded by the above).

```
cd examples/ob170114 && poetry run exozippy ob170114.yaml
```

Binary lens + finite source means the VBMicrolensing Op path (no
gradient), so the sampler is `ptde_async`.

## Data

| file | what it is |
|---|---|
| `phot.dat` | OGLE-IV EWS I-band photometry (BLG615.24, star 109070), <https://ogle.astrouw.edu.pl/ogle4/ews/2017/blg-0114.html> -- `HJD_UTC I[mag] err seeing sky`, read with explicit `columns:` so seeing/sky stay out of the detrend matrix |
| `params.dat` | OGLE EWS metadata (coordinates, the survey's own PSPL fit) |

Two data caveats, both material:

* **The paper fit re-calibrated photometry** (standard Cousins I /
  Johnson V, 2018 reference images), whose baseline differs from this EWS
  quick-look file by ~0.07 mag.  The affine per-instrument flux model
  absorbs a pure rescaling, but not reduction systematics, so this example
  reproduces the caustic geometry and the xallarap signal, not the paper's
  error budget.  The published fluxes (`f_S1 = 398`, `f_B = 230` in
  MulensModel's 22-mag system) are recorded in the params file next to the
  file-native seeds actually used.
* **The occasional OGLE V-band points the paper used are not public** (not
  in the EWS download; no data link in the paper).  The config carries a
  commented-out `OGLE_V` instrument block to enable when they are.

## Model

    star.0   = L1      lens host     ~0.50 Msun at ~3.3 kpc
    star.1   = Source  bulge red giant at 8.54 kpc (assumed, red clump)
    star.2   = SComp   the source's DARK companion -- exists only to move
                       the source; its mass is what the xallarap
                       amplitude measures
    planet.0 = b       lens planet, q = 0.0219 (~11.5 M_J)

Orbit `S` (`Source` + `SComp`) drives the source's barycentric sky track;
the lens block's `source_orbital_motion: keplerian` projects it onto the
trajectory (C25).  Everything is coupled through the graph: `xi_a` is not
a parameter but the derived `a_1/(D_S theta_E)`, so the same masses that
set the orbit set the amplitude -- at the seeds, `xi_a` derives to 0.2004
against the published 0.200, which is the closure check.

**The Std solution's dark companion is ~10 Msun.**  At an assumed 1-Msun
source, `a_1 = 1.47 au` with `P = 221.5 d` requires `M_SComp ~ 10.4 Msun`
(the paper's own `K_RV,S1 ~ 44 km/s`, Table B.2) -- a black-hole candidate
if taken at face value.  Interrogating whether that is physical is why
Mroz+26 built their ET parameterization, and this config asks the same
question natively: `star.SComp.mass` is an ordinary sampled star mass
under the IMF prior, so the fit itself weighs a 10-Msun dark companion
against the light curve's alternatives (the 2L2S solutions, xallarap
periods near 1 yr, etc. -- see their Table B.1 for the degenerate
solution set this example seeds only ONE of).

## Starting values

Seeded from Mroz+26 Table B.1 "Std: 2L1S" (the BIC-preferred model), with
four traps recorded in full in the params file:

1. **The `xi_*` -> EXOZIPPy mapping** (C25, verified against
   MulensModel's implementation at the light-curve level -- the code the
   paper used): `bigomega = phi_pi + xi_Omega + 180`, `i = xi_i`,
   `omega_* = xi_omega`, `nu(t0_par) = xi_u - xi_omega -> tp`.  (Until
   review 2.6.13 this read `phi_pi - xi_Omega` / `180 - xi_i`, tuned
   against a sign error in the shift projection; the pair cancelled in
   the track-level check and inverted the applied shift.)
2. **`t_E` and `pi_E` are not seeded** -- they are derived from the four
   proper-motion leaves, and at this event's large `pi_rel` the
   helio->geo term is comparable to `mu_rel` itself, so all four pm
   leaves are pinned instead.  The printed values then emerge to well
   under 0.1% (`t_E = 172.999`, `|pi_E| = 0.209805`,
   `phi_pi = 37.244` deg against 37.252) -- close, but measure rather
   than assume "exactly".  Seeding them alongside would over-determine
   the engine (measured: `phi_pi` rotated 35 deg, corrupting the
   xallarap track through C25's `bigomega = phi_pi + xi_Omega + 180`).
3. **`orbit.S.tc`** is seeded at the conjunction one period forward of
   the tp-implied one: the tc hard window and the tp->tc solver disagreed
   about the mod-P wrap for this omega quadrant (filed for review).
4. **The mass is NOT the printed median** -- `theta_E = kappa M_tot
   |pi_E|` leaves no distance freedom, and the paper's printed
   (`M_h`, `|pi_E|`, `theta_E`) triple does not satisfy it to 2 s.f.
   The seed is the `M_h` that closes it.  Full derivation at
   `star.L1.mass` in the params file.
