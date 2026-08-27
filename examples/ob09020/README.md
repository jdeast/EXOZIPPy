# OGLE-2009-BLG-020: binary lens + the spectroscopic orbit of its lens

> **WORK IN PROGRESS -- the model is now the full joint one**
> (`orbital_motion: keplerian`: the RV-constrained orbit drives
> `s(t)/alpha(t)` with no new free parameters), **but the acceptance FIT of
> review item 8.6.8 -- recovering Yee+2016's joint solution -- has not run
> yet.**  What IS verified, at the start point (2026-08-27): the derived
> geometry reproduces all four published orbital-motion observables in the
> published `u_0 > 0` labeling (`s_0`, `alpha_0`, `sign(gamma_par)`,
> `sign(gamma_perp)`); the Yee (Omega, i) -> EXOZIPPy frame mapping is
> measured (see the params file); and the start logp is finite (+19,024).
> The per-instrument flux/`err_scale` seeds still date from the static
> geometry and are refreshed by the acceptance fit.  Every sign statement
> below uses the corrected identity `gamma_perp = -dalpha/dt` (Skowron
> Appendix A.4; their Section 3.3.1 in so many symbols:
> `alpha(t) = alpha_0 - gamma_perp (t - t_0,par)`).

A binary-lens microlensing light curve fit jointly with radial velocities of
the **lens primary** -- the first (and still cleanest) case where a
microlensing mass/distance/orbit prediction was tested against Doppler
measurements.

* Skowron et al. 2011, ApJ 738, 87 (arXiv:1101.3312) -- the light-curve
  solution and the prediction.  Its Appendix A is also the convention
  EXOZIPPy follows verbatim (`components/mulensing/conventions.md`, C17), so
  its parameters transfer with no transformation.  **The arXiv v1 posting
  and the IOP version of record print different Table 1 values; this
  example references the IOP version** -- see "Which Table 1" below.
* Yee et al. 2016, ApJ 821, 121 (arXiv:1506.01441) -- the Keck/HIRES and
  Magellan/MIKE velocities that tested it, and the joint solution.

### Which Table 1

Skowron+2011 exists in two numerically different versions: the arXiv v1
posting (1101.3312, 2011 January) and the refereed IOP version of record
(DOI 10.1088/0004-637X/738/1/87, 2011 September).  The refereed version
evidently re-ran the chains: in the "with priors" column, `t_E` moved from
76.02 to 76.9 d, `u_0` from +0.06193 to +0.0613, `s_0` from 0.4315 to
0.4294, `pi_E_N` from -0.025 +/- 0.075 to -0.022 +/- 0.086, `gamma_z` from
+1.7 +/- 0.6 to +1.5 +/- 1.0, several other error bars changed, and a
`theta_E` row (2.95 mas with priors) exists only in the IOP table.  The
"2 par. motion" column moved similarly (`t_E` 78.57 -> 78.3, `u_0`
+0.06010 -> +0.0603, `s_0` 0.4261 -> 0.4268).

**This example references the IOP version of record everywhere** -- it is
the refereed, later version, and it is the one a citation of ApJ 738, 87
resolves to.  Two side effects worth knowing:

* Any pre-2026-08-27 record in this repo (or a reader checking against the
  freely downloadable arXiv PDF) will see ~1% "discrepancies" on `t_E`,
  `u_0` and `s_0`.  They are version skew, not misreads -- both value sets
  are faithfully printed in their respective versions.
* Within the IOP with-priors column, the derived block is self-consistent
  with `t_eff/t_E` (`4.708/0.0613 = 76.80 ~ 76.9`) but its `u_0` is 1% from
  the fit-parameter product `(u_0/w) * 10**(log w) = 0.06193`; in arXiv v1
  the derived block matched the product instead.  These are per-parameter
  MCMC medians, not one self-consistent point -- do not chase the third
  decimal across rows.

The lens is a 0.89 + 0.24 Msun binary at 0.75 kpc in a 276.6 d, e = 0.27
orbit; the source is a bulge K giant.  Because the lens is nearby disk,
`theta_E = 2.95 mas` -- several times the sub-mas value typical of a bulge
lens, and worth knowing before anything here looks wrong.

```
cd examples/ob09020 && poetry run exozippy ob09020.yaml
```

Binary lens + finite source means the VBMicrolensing Op path, which has no
gradient, so the sampler is `ptde_async`.

## Data

Every file is used **as downloaded**; the time system, column layout and
units are declared in `ob09020.yaml` rather than baked into a converted copy.

| file | what it is |
|---|---|
| `phot.dat` | raw OGLE-III EWS photometry, <https://ogle.astrouw.edu.pl/ogle3/ews/2009/blg-020.html> -- columns `HJD_UTC I[mag] err seeing[pix, 0.26"/pix] sky` |
| `params.dat` | OGLE EWS metadata (coordinates, the survey's own PSPL fit) |
| `*_OB09020*.pho` | muFUN follow-up photometry, <https://cgi.astronomy.osu.edu/microfun/Data/2009/OB09020/> -- see below |
| `OGLE-2009-BLG-020L.HIRES.rv` | Keck/HIRES RVs, Yee+2016 Table 5; `BJD_TDB - 2450000`, km/s |
| `OGLE-2009-BLG-020L.MIKE.rv` | Magellan/MIKE RVs, Yee+2016 Table 5; same units |
| `OGLE-2009-BLG-020L.HIRES.lags` | Yee+2016 Table 6 -- intermediate data, not fit (see below) |

* `phot.dat` uses `columns: {time: 0, mag: 1, err: 2}`.  This is not
  decoration: naming the layout also switches OFF the reader's "every column
  past the error is a detrend column" default, so **seeing and sky are read
  past and never fitted** (verified: `total_detrend_cols == 0`, no
  `detrend_coeffs` in the manifest).  To detrend against seeing after all,
  add `detrend: [3]` -- with `columns:` present it must be listed explicitly.
* `phot.dat` also sets `time_scale: utc`, `time_frame: hjd`, converting
  HJD_UTC -> BJD_TDB at load (verified: +63.6 s median, the expected UTC->TDB
  offset for 2001-2009).  That conversion refuses placeholder coordinates, so
  `ob09020.params.yaml` supplies `star.ra` / `star.dec`.
* The `.rv` files use `time_offset: 2450000.0` and `unit: km/s`.  This is the
  first shipped example with a non-default RV unit; the path is covered by
  `tests/test_rv_unit_and_k_seed.py`, and is confirmed here by
  `rvinstrument.gamma` starting at -41.05 km/s (the HIRES data mean).

### What the "lags" are

Not a light-travel-time correction -- there is no such lag between the
photometry and the spectroscopy.  A "lag" is the raw cross-correlation
velocity shift of a spectrum against a template.  The lens and the source are
blended in the Keck spectra, so Yee+2016 measure *two* lags per epoch and
take the difference (their Eq. 6, `RV_lens = lag_source - lag_lens`), using
the source -- which has no orbital motion -- as the velocity reference that
removes the instrumental zeropoint.  Table 5's RVs are the product of that
subtraction.  So `*.lags` is provenance; the fit consumes only the `.rv`
files.

### The muFUN follow-up photometry

The eight `.pho` files are muFUN quick-look photometry, included here by
**explicit permission of Jennifer Yee**, designate of the muFUN team leader
Andrew Gould -- which the archive's conditions of use require before these
data may be published in any form.  Do not redistribute them from here
without checking that the permission travels.

They are what OGLE is missing.  Skowron+2011 put the caustic entry at HJD'
4917.3, the peak at 4917.6 and the exit at 4917.75; OGLE has seven points in
HJD' 4917-4918 and samples only the exit.  These 1540 points cluster in HJD'
4916-4920 and bracket all three, Bronberg alone contributing 839 points
across the whole crossing.

| site | file | N | HJD' coverage |
|---|---|---|---|
| Bronberg 0.36 m | `Bron_OB09020U.pho` | 839 | 4916.51-4919.55 |
| Campo Catino Austral 0.40 m | `CAO_OB09020U.pho` | 279 | 4917.73-4918.92 |
| CTIO/SMARTS 1.3 m, I | `CT13_OB09020I.pho` | 23 | 4917.74-4985.78 |
| CTIO/SMARTS 1.3 m, V | `CT13_OB09020V.pho` | 5 | 4917.74-4985.78 |
| Farm Cove 0.36 m | `FCO_OB09020U.pho` | 67 | 4917.14-4919.13 |
| Kumeu 0.36 m | `Kumeu_OB09020U.pho` | 120 | 4916.05-4918.24 |
| Possum | `Pos_OB09020U.pho` | 72 | 4916.08-4917.21 |
| Vintage Lane | `VLO_OB09020U.pho` | 134 | 4917.02-4920.18 |
| Craigie | `Craigie_OB09020U.pho` | 20 | 4919.24-4919.28 (OFF) |

Read with `columns: {time: 0, mag: 1, err: 2}`, `time_offset: 2450000.0`,
`time_scale: utc`, `time_frame: hjd`.  The on-disk layout is `HJDmid Lens Err
FWHM Sky Typ Ref1 Err ...`, where `Lens` is the target and the trailing pairs
are comparison stars (a different number per file), so naming the three roles
both selects the right columns and keeps FWHM/Sky out of the detrend matrix.
CAO's on-disk row 0 is its one failed measurement (`mag = 0.000`, `Typ = 14`
where every good row is `Typ = 11`) and is dropped with `mask: [0]` -- a row
index in on-disk order, which is why the reader applies masks before sorting.

**Craigie is deliberately switched off.**  Its 20 points span 50 minutes at
HJD' 4919.24, long after the caustic, over which the magnification is flat --
so the model enters only as `f_source*A0 + f_blend` with `A0` constant and
`(f_source, f_blend)` has an exactly degenerate direction.  Measured: the
linear flux fit returns `f_total < 0`.  Two free parameters and one flat
direction for ~zero information, on a gradient-free sampler.  Uncomment in
`ob09020.yaml` if you want it.

**Reduction caveat.**  The archive is explicit that these are "uncalibrated,
preliminary, and incomplete on-the-fly reductions" in raw instrumental units,
not scaled to OGLE.  Skowron+2011 fit the *re-reduced* versions.  So this
example gets the caustic geometry, not the paper's error budget, and anyone
using it scientifically should talk to muFUN first.

## Model

    star.0 = L1      lens primary   ~0.85-0.89 Msun -- the star the RVs are of
    star.1 = L2      lens companion ~0.23-0.24 Msun -- unseen
    star.2 = Source  K giant in the bulge, I_s = 16.43, (V-I)_s = 1.93

The two halves are coupled through the masses: `lens.q` is derived from
`star.L1.mass` / `star.L2.mass`, and the `orbit` block derives
`m_total` / `a` / `K` from the same two mass nodes, so the RV mass function
and the light curve's mass ratio are automatically consistent.
`rvinstrument`'s `star_ndx: 0` makes the RV model the sum of `orbit.K` over
every orbit L1 belongs to.

With `orbital_motion: keplerian` on the lens block, orbit `L` also drives
the binary geometry: `s(t) = |delta(t)|` and `alpha(t) = phi_pi - PA(axis)`
per epoch (conventions.md C24), in Einstein units `a/(D_L theta_E)`.  No
`i180` hand-holding remains: the sky rotation sense of the binary axis in
the light curve measures `sign(cos i)`, and the axis orientation measures
`bigomega` -- both sampled over their full ranges
(`Orbit._lens_keplerian_orbits`).  NOTE the frame trap the params file
documents in full: Yee+2016 quote `(Omega_node, i)` in the Skowron
Appendix B frame (first axis = binary axis, third axis TOWARD the
observer), so their printed values do NOT drop into EXOZIPPy's sky frame
-- `i_EXOZIPPy = 180 - i_Yee`, and `bigomega` was pinned numerically
against the four published orbital-motion observables.

All the unfiltered follow-up shares one band (`N`, standing in as Cousins R);
the sites differ in zeropoint, which `f_source`/`f_blend` already absorb per
instrument, not in bandpass.  With `finite_source` on and three bands present
the model warns "Multiple bands for finite-source instruments; using first
band's u1", so the limb darkening applied is the I-band coefficient
throughout.  That is architectural, not physical: limb darkening lives
*inside* the magnification, and `build_likelihood` computes one magnification
curve per source over the concatenated times of every instrument, so a
per-band `u1` would need a per-band magnification.  Measured cost on this
example's 839 caustic-crossing Bronberg epochs: 6.5 mmag max between I and R
and 17 mmag between I and V, against a 26 mmag median error -- cheap here
(the V band is five points), not cheap for an event with dense simultaneous
V and I coverage.  Scoped in `notes/orbital_motion_and_nbody.txt` section 4a.

**Orbital motion of the lens binary: `orbital_motion: keplerian` is ON**
-- the RV-constrained orbit drives `s(t)`/`alpha(t)` with no new free
parameters (review 8.6.8 5a/5b), which is Skowron+2011's proposed
over-constraint test on the event they proposed it for.  The start state
is verified (banner above); the acceptance FIT -- recover (or refute)
Yee's joint solution -- is the next step.  The `linear` mode remains the
default recommendation for events without RVs.

The linear mode's acceptance measurement, re-made 2026-08-27 with the
shipped machinery (fluxes fit linearly per instrument, all 2837 points, raw
errors): at the IOP Table 1's **"parallax + 2 par. motion" column** -- the
one fitted with EXACTLY this model, so its printed values are the referee
(`u_0 = +0.0603`, `t_E = 78.3`, `s_0 = 0.4268`, `alpha = 189.06`,
`pi_E_N = -0.11`, `gamma_par = +0.12`, `gamma_perp = +2.78`):

| model | chi2 | chi2/N |
|---|---|---|
| static `s`, `alpha` | 72,485 | 25.55 |
| linear, `dalpha/dt = -2.78 rad/yr` (`gamma_perp = +2.78`, as printed) | **16,308** | **5.75** |
| linear, `dalpha/dt = +2.78 rad/yr` (wrong sign) | 306,660 | 108.09 |
| `gamma_par` only | 78,066 | 27.52 |

A factor 4.4 from the printed rates in the printed labeling, with the wrong
rotation sense 19x worse -- the sign is decisive, and it agrees with
`gamma_perp = -dalpha/dt` (Skowron A.4; C24) with no transformation.  Two
traps this measurement stepped in so the next reader does not have to:

* **Do not mix columns.**  The "full orbit (with priors)" column's
  parameters are a KEPLERIAN solution; through the strong
  `pi_E_N`-`gamma_perp` degeneracy (their Section 4.2) its
  `pi_E_N = -0.022` pairs with the full-orbit trajectory, and the LINEAR
  model at that column's values gains almost nothing over static
  (static 30,932 vs 29,376 with the printed rates -- against the factor
  4.4 the model earns at its own column).  Each column is self-consistent
  only as a set.
* Earlier versions of this README carried tables measured first at
  second-hand parameter values and then at the arXiv v1 Table 1
  (static 81,885 -> 15,388, factor 5.3); the arXiv-vs-IOP version skew
  ("Which Table 1" above) is ~1% on the trajectory parameters and moves
  the chi2 baselines by ~10% here.  The table above, at the IOP values,
  supersedes both.

The design -- source orbital motion, real `gamma_dot`/`gamma_ddot`, and the
eventual N-body backend -- is in `notes/orbital_motion_and_nbody.txt`.

## Starting values

Seeded at the published solutions; `ob09020.params.yaml` carries the
derivation next to each value.  Three things are traps for the next reader
and are recorded there in full:

1. **Skowron's Table 1 fit block is not in EXOZIPPy's parameters.**  It
   tabulates `t_eff = u_0 t_E`, `t_* = rho t_E`, `log q` and `log w`, with
   the impact parameter given as `u_0/w` in units of the central-caustic
   width -- but its "derived" block prints the conversions itself
   (IOP: `s_0 = 0.4294`, `q = 0.273`, `t_E = 76.9`, `u_0 = +0.0613`,
   `theta_E = 2.95 mas`).  Cross-check with `t_eff/u_0 = 4.708/0.0613 =
   76.80 ~ 76.9`, NOT with the fit-parameter product ("Which Table 1"
   above).
2. **The blending is not what the EWS page says.**  EWS reports `fbl = 0.987`
   from a PSPL fit to survey data alone; the real source fraction is
   `q_source ~ 0.335` (Skowron: `I_s = 16.43`, `I_b = 15.68`).  Fitting
   `f_source`/`f_blend` linearly to `phot.dat` at the published geometry
   independently returns 0.336 -- which is also what pins the convention
   branch, `(alpha = 189.08, u_0 > 0)`.
3. **Yee+2016's tabulated `omega_peri` is the other end of the apsidal
   line.**  At `omega = 151.6 deg` the RV model is anti-correlated with the
   data (chi2 = 2087 for 14 points); at `omega_* = 331.6 deg` it is chi2 =
   3.2.  (This is just the relative-orbit convention -- omega of the
   companion about the primary, standard for a visual binary -- against
   EXOZIPPy's `omega_*`, the primary about the barycenter.)
4. **`u_0` is seeded POSITIVE, as published.**  Table 1's own note reads
   "All parameters represent positive u_0 solutions ... which is slightly
   preferred", and Appendix A is EXOZIPPy's convention verbatim (C17), so
   the printed `(u_0 = +0.0613, alpha = +189.08)` transfers with no
   transformation.  The mirror `(u_0, alpha, pi_E_N, gamma_perp) ->
   -(...)` (their Eq. 16) is a distinct, slightly disfavored solution once
   parallax and orbital motion are believed -- a candidate second seed for
   the mode reporter, not an equivalent labeling.  (An earlier version
   seeded the mirror on three claimed lines of evidence; all three rested
   on one inverted sign identity and are retracted, 3.6.4.)
5. **`orbit.L.bigomega` and `sign(cos i)` are now MEASURED parameters** --
   the keplerian lens motion consumes the orbit, so the axis's sky
   orientation (bigomega) and rotation sense (`sign(cos i)`) reach the
   light curve; the orbit component samples both over their full ranges
   (`Orbit._lens_keplerian_orbits`).  The seeds carry the frame-mapped
   Yee branch (`i = 50.58 = 180 - 129.42`, `bigomega = 337.48`; the
   mapping table lives in the params file).  The retired `i180: true`
   hand-holding and the inert commented-out `bigomega` are exactly what
   review 8.6.8 5e said this mode would remove.

Every light curve also carries its own `(log_f_total, q_source)` and
`err_scale`, and these are fit rather than guessed: with the geometry held
fixed, `f_source` and `f_blend` are **linear**, so they are solved exactly
per instrument against the seed model, and `err_scale` is the `sqrt(chi2/N)`
that solve leaves behind.  This is not cosmetic for the follow-up -- the
`.pho` files are raw instrumental magnitudes with an arbitrary per-site
zeropoint, so `log_f_total` has no meaningful default (it spans -6.03 to
-7.36 across these sites), and CAO and CT13 have no baseline coverage from
which it could be inferred.  Left at the defaults, CT13_I and CT13_V started
at chi2/N of 384 and 1055.  The physical cross-check: `q_source` lands
between 0.17 and 0.37 at every site against Skowron's calibrated 0.334, the
spread being the expected consequence of unfiltered detectors with different
responses.

Start logp is +19,024 in the full keplerian-mode model at the IOP-referenced
seeds (2026-08-27; the same model started at +3,276 at the arXiv v1 values
-- the ~1% version skew on `t_E`/`u_0`/`rho` is worth ~16k nats through
this caustic-crossing dataset, which is the concrete argument for pinning
WHICH version a params file references).  For the record of the seed
lineage: the static model started at +41,458 at the arXiv published-branch
seeds, and +47,309 at the retracted mirror seeds.  The keplerian start is
lower than the static one NOT because the model is worse but because the
published point-values do not close as a set: the per-instrument flux and
`err_scale` seeds still date from the static geometry (measured at the
arXiv-seed start: a linear-flux refit leaves the caustic window healthy --
Bronberg chi2/N = 4.3 on raw errors -- while the OGLE wings sit at
chi2/N ~ 375, carrying the mismatch between Skowron's with-priors
trajectory values and the joint solution's unprinted ones), and Table 1's
own medians are mutually inconsistent at the point-estimate level (e.g.
its printed |pi_E| = 0.151 against the theta_E/(kappa M) = 0.30 implied by
its own priors' masses -- marginal medians, as the params file notes).
Resolving that tension is precisely what the acceptance FIT is for; the
per-instrument chi2/N table and the flux/`err_scale` seeds are refreshed
from its posterior.
