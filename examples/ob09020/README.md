# OGLE-2009-BLG-020: binary lens + the spectroscopic orbit of its lens

> **WORK IN PROGRESS -- do not treat this as a validated reproduction**
> until lens orbital motion is modelled and the acceptance criteria of
> review item 8.6.8 are met.  The 3.6.4 convention retraction is APPLIED:
> the seeds are the published `u_0 > 0` solution, read off Skowron+2011
> Table 1's "parallax + full orbit (with priors)" column directly
> (2026-08-27), and every sign statement below uses the corrected identity
> `gamma_perp = -dalpha/dt` (Skowron Appendix A.4; also their Section 3.3.1
> in so many symbols: `alpha(t) = alpha_0 - gamma_perp (t - t_0,par)`).
> What is verified here: the data files and their read options, the
> per-instrument flux/`err_scale` seeds, the `omega_* = 331.6` result, and
> the measured size of the missing-orbital-motion effect.

A binary-lens microlensing light curve fit jointly with radial velocities of
the **lens primary** -- the first (and still cleanest) case where a
microlensing mass/distance/orbit prediction was tested against Doppler
measurements.

* Skowron et al. 2011, ApJ 738, 87 (arXiv:1101.3312) -- the light-curve
  solution and the prediction.  Its Appendix A is also the convention
  EXOZIPPy follows verbatim (`components/mulensing/conventions.md`, C17), so
  its parameters transfer with no transformation.
* Yee et al. 2016, ApJ 821, 121 (arXiv:1506.01441) -- the Keck/HIRES and
  Magellan/MIKE velocities that tested it, and the joint solution.

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

`i180: true` on the orbit selects Yee's `i > 90 deg` branch: `i` and
`180 - i` are a reflection through the sky plane, and RVs alone cannot tell
them apart.

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

**Orbital motion of the lens binary: the machinery is in
(`orbital_motion: linear`, conventions.md C24), not yet enabled here** --
this example's endgame is the `keplerian` mode, where the RV-constrained
orbit drives `s(t)`/`alpha(t)` with no new free parameters (review 8.6.8
5a/5b), and it flips on when that lands.

The linear mode's acceptance measurement, re-made 2026-08-27 with the
shipped machinery (fluxes fit linearly per instrument, all 2837 points, raw
errors): at Skowron Table 1's **"parallax + 2 par. motion" column** -- the
one fitted with EXACTLY this model, so its printed values are the referee
(`u_0 = +0.06010`, `t_E = 78.57`, `s_0 = 0.4261`, `alpha = 189.07`,
`pi_E_N = -0.11`, `gamma_par = +0.12`, `gamma_perp = +2.78`):

| model | chi2 | chi2/N |
|---|---|---|
| static `s`, `alpha` | 81,885 | 28.86 |
| linear, `dalpha/dt = -2.78 rad/yr` (`gamma_perp = +2.78`, as printed) | **15,388** | **5.42** |
| linear, `dalpha/dt = +2.78 rad/yr` (wrong sign) | 331,126 | 116.72 |
| `gamma_par` only | 87,957 | 31.00 |

A factor 5.3 from the printed rates in the printed labeling, with the wrong
rotation sense 21x worse -- the sign is decisive, and it agrees with
`gamma_perp = -dalpha/dt` (Skowron A.4; C24) with no transformation.  Two
traps this measurement stepped in so the next reader does not have to:

* **Do not mix columns.**  The "full orbit (with priors)" column's
  parameters are a KEPLERIAN solution; through the strong
  `pi_E_N`-`gamma_perp` degeneracy (their Section 4.2) its
  `pi_E_N = -0.025` pairs with the full-orbit trajectory, and evaluating
  the LINEAR model at that column's values fits WORSE than static
  (81k/69k vs 29k over these points).  Each column is self-consistent
  only as a set.
* An earlier version of this README carried a factor-3.5 table
  (160,258 -> 45,661) measured at second-hand parameter values ~1% off
  the printed ones; at the correctly-read values its static baseline is
  not reproducible and that table is superseded by the one above.

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
   (`s_0 = 0.4315`, `q = 0.272`, `t_E = 76.02`, `u_0 = +0.06193`), and the
   cross-check is exact: `(u_0/w) * 10**(log w) = 0.42942 * 0.14421 =
   0.06193`, identical to the printed derived `u_0`.
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
   the printed `(u_0 = +0.06193, alpha = +189.08)` transfers with no
   transformation.  The mirror `(u_0, alpha, pi_E_N, gamma_perp) ->
   -(...)` (their Eq. 16) is a distinct, slightly disfavored solution once
   parallax and orbital motion are believed -- a candidate second seed for
   the mode reporter, not an equivalent labeling.  (An earlier version
   seeded the mirror on three claimed lines of evidence; all three rested
   on one inverted sign identity and are retracted, 3.6.4.)
5. **`orbit.L.bigomega` is inert here and is commented out.**  With no
   astrometry the orbit component never declares it -- verified, there is no
   `bigomega` among the sampled variables -- because RVs carry no information
   about the node's position angle.  Yee measure it only because they model
   the lens orbital motion this fit is missing.  Likewise `i180: true`
   selects Yee's `i > 90 deg` branch **by hand**: the RV mass function sees
   only `sin i`, and what would measure the sign is the sky rotation sense,
   i.e. orbital motion again.

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

Start logp is +41,458 at the published-branch seeds above (it was +47,309
at the retracted mirror seeds; the drop is expected -- the per-instrument
flux and `err_scale` seeds were fit at the old geometry, and the model is
still missing the orbital motion that dominates the residuals either way).
The per-instrument start chi2/N table that used to sit here was measured at
the mirror seeds and is removed rather than left to mislead; it is
re-measured as part of the orbital-motion acceptance (review 8.6.8), along
with refreshed flux/`err_scale` seeds.  The strongest single pre-revert
confirmation stands unchanged in kind: Bronberg's 839 points spanning the
entire caustic crossing sat at chi2/N ~ 1 from literature values with no
fitting, which is what validates the data handling and the parameter
transfer.
