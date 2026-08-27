# OGLE-2007-BLG-224: terrestrial parallax

> ## WORK IN PROGRESS -- UNVERIFIED
>
> **This example has never been run to completion.** The config builds, the
> data load, and the model evaluates at its start point (see the measured
> numbers below), but **no fit has been sampled**, no posterior exists, and
> nothing here has been checked against a converged run.
>
> Known open items:
>
> * The sampler path (`ptde_async`, forced by `finite_source`) is untested on
>   this event. It is the same path `examples/ob09020` uses.
> * **`pi_E` -- the quantity this example exists to demonstrate -- is not
>   recoverable from the shipped data.** The two light curves that cover the
>   peak are 130 km apart; the intercontinental legs Gould+2009 actually used
>   are not public. See "What these data can, and cannot, recover" below.
>   What is demonstrated here is the *setup*, not the measurement.
> * Start values are seeded from the published solution and from a
>   least-squares fit done outside EXOZIPPy; they have not been confirmed by
>   the fit itself.
> * The muFUN photometry is the archive's own "uncalibrated, preliminary,
>   and incomplete on-the-fly" reduction, not the re-reduced data the paper
>   fit.
>
> Treat every number below as a start-point measurement, not a result.

The point of this example is one config key, repeated seven times:

```yaml
observer_location: "-70.70167,-29.00333,2282"   # lon, lat, height -- lon FIRST
```

Everywhere else in `examples/`, a ground-based light curve says
`observer_location: "earth"` and is modelled from the geocenter. Here every
instrument declares the actual spot on the globe it was observed from, so the
parallax the fit sees contains the Earth's **rotation** as well as its orbit.
That is the whole of terrestrial parallax as a modelling problem: no new
parameter, no new physics module, just an honest observer position.

* **Reference paper:** Gould et al. 2009, ApJ 698, L147 --
  ["The Extreme Microlensing Event OGLE-2007-BLG-224: Terrestrial Parallax
  Observation of a Thick-Disk Brown
  Dwarf"](https://ui.adsabs.harvard.edu/abs/2009ApJ...698L.147G/abstract)
  ([arXiv:0904.0249](https://arxiv.org/abs/0904.0249)). Its result:
  `M = 0.056 +/- 0.004 Msun` at `D_L = 525 +/- 40 pc`, moving at
  `113 +/- 21 km/s` counter to Galactic rotation -- a thick-disk brown dwarf,
  weighed by parallax over a baseline of a few thousand kilometres.

```
cd examples/ob07224 && poetry run exozippy ob07224.yaml
```

## Why this event and not another

Terrestrial parallax is normally hopeless. Two observers separated by a
baseline `D` see the source-lens trajectory displaced by `D / r_tilde_E` in
units of the Einstein radius, and `r_tilde_E = AU / pi_E` is typically several
AU -- tens of thousands of Earth diameters. You need three things at once, and
OGLE-2007-BLG-224 is the first event that had all three:

1. **A tiny `r_tilde_E`.** `pi_E = 1.97` here, the signature of a *nearby*
   lens, so `r_tilde_E = 0.51 AU ~ 12,000 Earth radii` rather than millions.
2. **Extreme magnification, so the light curve is steep.** `A_max ~ 2350`
   (8.4 magnitudes). The magnification only responds to a trajectory offset
   while `u ~ rho`, i.e. within one source-crossing time
   `t_* = rho * t_E = 8.5 minutes` of the peak. Away from the peak the same
   offset is invisible.
3. **Simultaneous coverage from different continents during those minutes.**

At the published solution, this model predicts the following differences from
the OGLE (Las Campanas) light curve during the peak:

| site | separation from Las Campanas | max &#124;dmag&#124; | peak time offset |
|---|---:|---:|---:|
| CTIO / muFUN SMARTS, Chile | 130 km | 5.6 mmag | +0.5 s |
| Mt Lemmon / muFUN LOAO, Arizona | 7,500 km | 122 mmag | -18.8 s |
| Bronberg, South Africa | 8,600 km | 239 mmag | -44.0 s |
| Roque de los Muchachos, La Palma | 7,900 km | 231 mmag | -60.4 s |
| geocenter (`observer_location: "earth"`) | 6,400 km | 143 mmag | -26.8 s |

The last row is the reason the key matters: modelling these light curves from
the geocenter is not a small approximation here, it is a 0.14 mag error in the
middle of the peak. The Letter's own statement -- "the event passed both South
Africa and the Canaries about 1 minute earlier than Chile" -- is the -44 s and
-60 s rows.

## The mechanics

Everything below is in `ob07224.yaml`; this is the reasoning behind it.

### Per-site observer positions

`observer_location` takes a geodetic string `"lon_deg,lat_deg[,height_m]"`,
**longitude first** (`exozippy/ephemeris.py::_parse_geodetic`, matching
astropy's `EarthLocation.from_geodetic`). It also accepts an astropy site
name, but this example spells the numbers out: four of the six telescopes
here are not in astropy's site registry at all, and `EarthLocation.of_site()`
otherwise wants a network fetch of the site table for an example that should
run offline. muFUN telescope coordinates are the collaboration's own, from
<https://cgi.astronomy.osu.edu/microfun/microfun.html>; Las Campanas is
astropy's `lco`.

Downstream this is completely generic: `MulensInstrument.load_data` asks
`exozippy.ephemeris.get_observer_position` for each file's absolute
barycentric position and `_abs_to_delta` converts it to the Skowron+2011
geocentric deviation, exactly as it does for Spitzer in `examples/ob140939`.
A ground site is a spacecraft whose orbit happens to be an Earth radius wide.
Nothing in `lens.py` knows the difference. On MulensModel's side the model is
built with `parallax(topocentric=False)` -- MulensModel's own topocentric term
is *not* used, because EXOZIPPy has already put the observer in the right
place.

Heights are nominal to ~100 m. That is `1.6e-5` of an Earth radius, four
orders of magnitude below the effect being fitted.

### Per-site time systems

Every file here is `HJD_UTC`, so each carries

```yaml
time_scale: "utc"
time_frame: "hjd"
time_location: [-70.70167, -29.00333, 2282]
```

`time_location` is the same site again, in the same lon-first order, and it is
**not** decoration at this precision. Omitting it costs up to 21 ms of
geocenter-versus-observatory Romer delay in the `HJD -> BJD_TDB` conversion
(`components/instrument.py::_to_bjd_tdb`). Against the 60 s intercontinental
signal that is 0.03%; against the 0.5 s Chile-to-Chile signal it is 4%. There
is a residual ambiguity nobody can remove -- whether each archive's HJD was
computed from the geocenter or from the telescope -- and it has exactly this
size. It is a good reason to trust an intercontinental baseline and not a
130 km one.

The conversion needs the target direction, so `star.ra` / `star.dec` are
mandatory in the params file (it refuses to run against `defaults.yaml`
placeholders).

### Finite source, and what it forces

`finite_source: True` is not optional here: the lens transited the source
(`rho = 8.5e-4`, impact parameter `~theta_*/3`), so the peak *is* the source.
Two consequences:

* The event goes on the MulensModel Op path
  (`finite_source_LD_Yoo04` inside `+/- 5*rho*t_E` of `t_0`, point source
  outside). That path has no gradient, so the sampler is `ptde_async` and
  NUTS is unavailable -- `Lens.sampler_requirements()` will say so if you try.
* Limb darkening lives *inside* the magnification, and one magnification curve
  is computed per source over every instrument's concatenated times, so a
  per-band `u1` would need a per-band magnification. With more than one band
  present the model warns "Multiple bands for finite-source instruments; using
  first band's u1" and uses the **lowest-indexed** band any instrument
  references. That is why `band:` lists `I` first: the peak, the only place
  8.4 mag of magnification makes limb darkening matter, is covered by OGLE `I`
  and SMARTS `I`.

`t0_par: 2454233.667` anchors the Skowron+2011 geocentric frame. For
terrestrial parallax the choice matters far less than for annual parallax --
the entire signal is one night wide -- but it must sit at the peak.

## Data

Every file is used **as downloaded**; the time system, column layout and units
are declared in `ob07224.yaml` rather than baked into a converted copy.

| file | what it is |
|---|---|
| `phot.dat` | raw OGLE-III EWS photometry, <https://ogle.astrouw.edu.pl/ogle3/ews/2007/blg-224.html> -- columns `HJD_UTC I[mag] err seeing[pix] sky` |
| `params.dat` | OGLE EWS metadata: the coordinates, and the survey's own (finite-source-free) PSPL fit |
| `*OB07224*.pho` | muFUN quick-look photometry, <https://cgi.astronomy.osu.edu/microfun/Data/2007/OB07224/> |

Both `columns: {time: 0, mag: 1, err: 2}` blocks are load-bearing, not
decoration: naming the layout also switches OFF the reader's "every column
past the error is a detrend column" default, so OGLE's seeing and sky and the
`.pho` files' FWHM, Sky and comparison-star columns are read past and never
fitted (verified: `total_detrend_cols == 0`). To detrend against seeing after
all, add `detrend: [3]` -- with `columns:` present it must be listed
explicitly.

The `.pho` layout is `HJDmid Lens Err FWHM Sky Typ Ref1 Err ...`, where
`HJDmid = HJD_UTC - 2450000` at mid-exposure, `Lens` is the target's **raw
instrumental** magnitude (uncalibrated, arbitrary zeropoint per site -- which
is fine, `f_source`/`f_blend` are per instrument) and the trailing pairs are
comparison stars, a different number per file. Every row in all six files is
`Typ = 11`, so unlike `examples/ob09020` nothing needs masking.

| site | file | N | HJD' coverage | relative to peak (4233.668) |
|---|---|---:|---|---|
| OGLE 1.3 m, Las Campanas | `phot.dat` | 1079 | 2125.68-2409.53 (8 yr) | 57 points on the peak night, to I = 10.47 |
| CTIO/SMARTS 1.3 m, I | `COB07224I.pho` | 98 | 4231.81-4240.93 | 47 points across the peak, to I = 11.06 |
| CTIO/SMARTS 1.3 m, V | `COB07224V.pho` | 14 | 4231.82-4239.80 | 8 points across the peak |
| Auckland Obs 0.40 m, R | `AOB07224R.pho` | 58 | 4233.04-4233.27 | ends ~10 h before |
| Farm Cove 0.36 m, unfiltered | `FOB07224U.pho` | 48 | 4232.98-4233.25 | ends ~10 h before |
| Vintage Lane 0.40 m, unfiltered | `VOB07224U.pho` | 39 | 4233.09-4233.93 | resumes ~6 h after |
| Mt Lemmon LOAO 1.0 m, I | `LOB07224I.pho` | 16 | 4233.94-4234.99 | resumes ~6.6 h after |

All three unfiltered/R light curves share one band (`R`, standing in as
Cousins R): the sites differ in zeropoint, which `f_source`/`f_blend` already
absorb per instrument, not in bandpass. None of them covers the peak, so the
choice costs nothing here.

**Reduction caveat.** The muFUN archive is explicit that these are
"uncalibrated, preliminary, and incomplete on-the-fly reductions". Gould+2009
fit re-reduced photometry. So this example gets the geometry, not the paper's
error budget, and anyone using it scientifically should talk to muFUN first.
The muFUN archive's conditions of use require permission before these data are
published in any form; it was given for this repository by Jennifer Yee. Do
not redistribute them from here without checking that the permission travels.

## What these data can, and cannot, recover

**They reproduce the geometry.** Refit `t_0, u_0, t_E, rho` and the I-band
`u1` with `pi_E` held at the published value, `f_source`/`f_blend` solved
linearly per instrument (they are linear, so this is exact) and `err_scale`
profiled out per instrument:

| | this fit, archival data | Gould+2009 |
|---|---|---|
| `t_E` | 7.10 d | 6.91 +/- 0.13 d |
| `rho` | 8.03e-4 | (8.50 +/- 0.16)e-4 |
| OGLE `q_source` | 0.4806 | 0.4761, from the published `I_s = 18.91` against the EWS baseline `I_bl = 18.104` |
| `mu_rel,hel` | 42.3 mas/yr at 60.4 deg S of W | 43 mas/yr, and `v_tilde_hel` at 61 deg S of W |

Three percent on `t_E`, six on `rho`, one on the blending, from raw archival
photometry with no re-reduction. The last row is not fitted at all -- it is
what the seeded proper motions imply once EXOZIPPy converts them to the
geocentric frame, and neither of the Letter's two numbers was used to build
them.

**They do not recover `pi_E`, and this is not a bug in the setup.** Profile
the same likelihood over `|pi_E|` along the published position angle,
re-optimizing `t_0, u_0, t_E, rho, u1` at every point:

| `pi_E` | 0.0 | 0.5 | 1.0 | 1.5 | **1.97** | 2.5 | 3.0 | 4.0 | 6.0 | 10.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| d(-2 ln L) | -2.1 | -1.6 | -1.0 | -0.5 | **0** | +0.6 | +1.2 | +1.7 | +4.0 | +13.8 |

The curve is monotonic and almost flat: `pi_E = 0` is *preferred* over the
published 1.97, by two units of `-2 ln L`, and everything from 0 to 4 lies
within four. There is no measurement here.

Freeze the nuisance parameters at the `pi_E = 1.97` optimum instead of
profiling them and the same scan looks like an overwhelming detection --
`pi_E = 0` then costs **+2,013** rather than gaining 2.1. That apparent
signal is the parallax masquerading as a shift of `t_0`, and a free `t_0`
eats all of it: **a single site's topocentric offset is degenerate with
`t_0`; only the difference between two sites is parallax.**

The reason is in the table at the top. The two data sets that cover the peak
are OGLE at Las Campanas and muFUN SMARTS at CTIO -- 130 km apart, predicting
a 5.6 mmag / 0.5 s difference, below the photometry. The three legs
Gould+2009 actually used were the Bronberg 0.35 m in South Africa (754
unfiltered images ending 10 minutes before peak), the RoboNet Liverpool
Telescope on La Palma (8 R images bracketing the peak) and Chile. Neither of
the first two is in any public archive. `ob07224.yaml` carries commented-out
`mulensinstrument` blocks for both, coordinates and all, so they can be
dropped in the day they are.

What the setup *is* demonstrably doing: switching every `observer_location`
in this config to `"earth"` and re-evaluating at the same seed costs **9,403
nats** of start logp (+23,072 -> +13,669). Almost all of that is the ~27 s
peak-time shift a free `t_0` would absorb, so read it as "the observer
positions are being used", not as a parallax detection.

## Starting values

`ob07224.params.yaml` carries the derivation next to every value. Four
things worth knowing before reading it:

1. **`t_0` and `u_0` are not in the Letter**, which prints only derived
   quantities. They are measured here as described above. The OGLE EWS PSPL
   fit independently reports `Tmax = 2454233.667`, agreeing to 0.7 s.
2. **`pi_E` is given as a magnitude and a direction**, not as components:
   `|pi_E| = 1.97` at "52 deg south of west". That is position angle
   `270 - 52 = 218 deg` measured North through East, and
   `phi_pi = atan2(pi_E_E, pi_E_N)` is exactly that angle (`conventions.md`,
   C8), so `pi_E_N = -1.5524`, `pi_E_E = -1.2129`. The quadrant is confirmed
   independently: removing the Earth's 23 km/s almost-due-east motion from
   the Letter's `v_tilde_geo = 127 km/s` reproduces its `v_tilde_hel =
   112 km/s at 61 deg south of west` only with this sign.
3. **The stellar proper motions have to be seeded by hand**, and this is the
   one trap in the file. `t_E` and both `pi_E` components are *derived* --
   `t_E = theta_E / mu_rel,geo` and `pi_E = (pi_rel/theta_E) * mu_hat_rel,geo`
   -- from four leaves (`pm_ra`/`pm_dec` of lens and source). Two seeded
   constraints cannot pin four leaves, so seeding `t_E` and `pi_E_*` alone
   leaves the engine free anywhere on a two-dimensional family. Measured
   before the `star.*.pm_*` entries existed: the model started at
   `t_E = 6.14 d` and `pi_E` at position angle **224.4 deg** -- `|pi_E|` and
   `theta_E` right, the timescale and the *direction* wrong. That direction
   is precisely what terrestrial parallax measures.
4. **The stellar chain closes.** `pi_rel = 1000/525 - 1000/8000 = 1.780 mas`;
   `theta_E = sqrt(kappa M pi_rel) = 0.9009 mas` (published 0.91);
   `|pi_E| = pi_rel/theta_E = 1.976` (published 1.97);
   `t_E = theta_E / 47.62 mas/yr = 6.910 d` (published 6.91 +/- 0.13). Those
   are the values the *model* reports at the start point, not the seeds.

`star.Source.radius` is left free, exactly as in `examples/ob09020` and for
the same reason: `theta_* = 0.77 +/- 0.03 uas` is an *angle*, and turning it
into a radius (1.31 Rsun at 8 kpc) would feed the assumed source distance
back into the fit as though it had been measured.

Every light curve also carries its own `(log_f_total, q_source)` and
`err_scale`, fit rather than guessed: with the geometry held fixed
`f_source` and `f_blend` are linear, so they are solved exactly per
instrument, and `err_scale` is the `sqrt(chi2/N)` that solve leaves behind.
This is not cosmetic for the `.pho` files -- they are raw instrumental
magnitudes with an arbitrary per-site zeropoint, so `log_f_total` has no
meaningful default, and `SMARTS_V` has fourteen points and no baseline
coverage from which one could be inferred.

Start logp is **+23,072** over 12 (vector-valued) free RVs and 1352
photometric epochs. Per-instrument chi2/N at the start, on raw errors before
`err_scale`:

| OGLE | SMARTS_I | SMARTS_V | Auckland | FarmCove | VintageLane | LOAO |
|---|---|---|---|---|---|---|
| 6.56 | 7.96 | 7.93 | **0.97** | 1.69 | 1.63 | **0.83** |

The two ~1.0 columns -- Auckland (0.40 m, R) and LOAO (1.0 m, I), both on
the wings -- match literature values with nothing fitted but their own two
fluxes. OGLE's 6.56
is concentrated entirely at the peak: out of event (957 of its 1079 points)
it measures 2.11, which is why `err_scale` is seeded at 1.45 rather than its
all-points 2.56. The residual is tens of mmag of structure across the peak
night against quoted errors of 1-5 mmag -- quick-look photometry of an
`I = 10.5` star, which is precisely the systematic Gould+2009 re-reduced away
and this example does not.
