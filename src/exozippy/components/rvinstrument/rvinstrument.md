# The RV instrument

`src/exozippy/components/rvinstrument/`: radial velocities, as barycentric-corrected
velocities (the default) or as raw measured redshifts whose barycentric correction the
model computes itself (`data_format: raw_redshift`).

Why an instrument should publish raw `c z` at all, the metadata a release must carry and
the file layout EXOZIPPy reads are in `docs/raw_redshift_data.md`.

Read this before changing what an RV file means, the RV model's time argument, the
instrument offset `gamma`, or the barycentric-correction module `redshift.py`. Related:
`src/exozippy/components/instrument.md` (the shared reader, detrending, GP and robust
likelihoods every RV file goes through), `src/exozippy/components/orbit/orbit.md` (the
orbit state every RV model projects), `src/exozippy/skyframe.md` (the sky frame the
reflex is written in), `src/exozippy/components/timesystem.py` (the `time_*` keys).

## Corrected velocities (`data_format: rv`, the default)

A file's columns are time (converted to BJD_TDB by the shared reader), RV and its error
in `unit:` (default m/s), then optional detrend columns. The model is
`gamma_i + sum over every orbit containing the star of its reflex` (Keplerian `K`
phase terms and Taylor `VZ(t) - VZ(rv_epoch)` columns, `RVInstrument._rv_model`), plus a
file's Rossiter-McLaughlin anomaly where `rm:` asks for one. `data_format` absent is
byte-identical to the file before the key existed.

**The star's own light-travel time** (`light_travel_time:`, per file, default ON; JDE
2026-11-04: "we should do it right and include the ltt term and recompute the logp").
The times are BJD_TDB ARRIVAL times, and the reflex light left the star at its EMISSION
time `t_emit = t - Z_star(t_emit)/c`, `Z_star` its own line-of-sight displacement. So
every orbit column of `_rv_model` -- Keplerian and Taylor -- is evaluated at its own
`t_emit` (`_orbit_emission_times` -> `_emission_time`, the raw-redshift mode's
machinery, through `components/ltt.py`; per row, by the row's file), in the likelihood
and on the plotted grid alike.

**Which displacement retards which orbit.** Each orbit's reflex is retarded by the
star's displacement in THAT orbit and in the orbits nested inside it (an orbit whose
bodies are all bodies of this one -- a Jacobi inner planet), the subsystem whose light
travel a transit of that orbit sees. Not by the star's motion in an orbit OUTSIDE it:
that is the light-time effect of an outer companion on the inner orbit (`ltt.py`,
"MULTIPLE COMPANIONS", item 2), 2.5 s for a Jupiter at 5 AU and HOURS for a stellar
companion, which no other consumer models -- transits time the inner orbit against its
own barycenter, and every period is the observed one. Retarding the RVs alone by it
put the RV and transit ephemerides of KELT-4 b 8.3 h apart (the star is in a
hierarchical triple): -2471 nats at the start, a model no joint fit could reconcile.
Sibling orbits (both around the star alone) are not nested in each other; a Taylor
orbit (an unresolved outer companion) is retarded by its own displacement only. The
raw-redshift mode uses the same per-orbit times, and the star's WHOLE displacement for
its direction (the wobble in `rho_hat`). Modeling the outer light-time effect is one
change for every timing consumer at once, not for RVs alone. The
Rossiter-McLaughlin anomaly is not: it is an occultation seam with its own retardation
(`rm.compute_rm_rv`), and it reads the arrival times. The term is `~K^2/c`: 0.06 mm/s
for kelt4, 3 cm/s at K = 3 km/s, 3 m/s at K = 30 km/s, tens of m/s on a close binary's
secondary (`tests/test_rv_light_travel.py`, against an independent fixed-point
retardation). The sign: ltt.py's `z` is companion-minus-primary toward the observer, so
the star's own toward-observer displacement is `-(M_c/M) z` on an orbit's primary side
and `+(M_p/M) z` on its companion side.

With the key on, `Orbit._ltt_reporting_mask` counts the file as a retarding consumer of
every Keplerian orbit of its star, exactly as a transit file is: the Kepler epoch is the
target frame's (`tp_target` from `tc_target`), and `tc` is the OBSERVED conjunction --
the time a transit would be seen, `(M_p - M_c)/M a sin i / c` before `tc_target`. That
is what a published transit time and every other component's `tc` mean, so a joint
transit + RV fit has one `tc`. In an RV-only fit it is a change of meaning: a seeded
`tc` is now read as the observed conjunction, which moves the model's phase by that
shift (21 s for kelt4 b, 0.07 m/s) -- the larger part of every RV-only example's
start-logp move (the per-example table is in #568). A `tc` from an older,
unretarded fit is closer to the target-frame one (an RV-only fit's conjunction is the
star's, which is late by only `(M_c/M) a sin i / c`); shift it by the seconds above, or
set `light_travel_time: false` to keep the old model.

When it cannot apply: an orbit component without physical `a`/masses/`sini` (its
bodies did not resolve) warns once (`_keplerian_ltt_supported`, the decision rm.py and
transit.py make through `ltt.orbit_supports_ltt`) and its Keplerian columns are
evaluated at the arrival time; a Taylor orbit retards by its own line-of-sight
displacement (its acceleration and jerk terms) and needs no `a`. `light_travel_time:
false` on every file is the arrival-time model node for node (no row is retarded, and
`_rv_model` reads the time vector itself), so a fit that wants the old numbers sets it.

## Raw redshifts (`data_format: raw_redshift`)

The barycentric correction "in all its glory", inside the model: a file carries what the
spectrograph measures, and the prediction is a function of the FITTED star and of every
orbit. It follows Wright & Eastman (2014, PASP 126, 838; WE14) -- the formalism behind
barycorrpy (Kanodia & Wright 2018) -- run as a forward model: instead of correcting the
data with a fixed catalog star, the model predicts the redshift the instrument saw from
`star.ra`/`dec`/`pm_ra`/`pm_dec`/`parallax`/`rv` and the orbits, so the multiplicative
cross terms (the planet's reflex times the barycentric velocity among them) and a
consistent proper motion are not steps anyone can apply inconsistently: they are the
model.

### The spelling

```yaml
rvinstrument:
  - name: "HPF"
    file: "Gaia4like.HPF.rv"
    data_format: raw_redshift
    observer_location: "-104.0147,30.6814,2026"   # or an astropy site name, or 'gaia'
    time_frame: jd        # the observatory's own clock
    time_scale: utc       # a flux-weighted UTC exposure midpoint
    epoch: 2457389.0      # BJD_TDB of star.ra/dec/pm (J2016.0); not with abs astrometry
```

- **The data column is `c * z_meas`** in `unit:` (default m/s), with `z_meas =
  lambda_obs / lambda_template - 1` the redshift measured against the template BEFORE any
  barycentric correction. It is a redshift times c, not a velocity: a pipeline that
  reports an uncorrected "RV" from `c ln(1 + z)` or a relativistic formula differs from
  `c z` by `c z^2 / 2` (1.5 m/s at 30 km/s), so convert to `c z` first. Errors are in the
  same unit. Detrend columns, `gp:`, `likelihood:` and the jitter work as for any file,
  in this `c z` space.
- **The time column** is the flux-weighted exposure midpoint. `time_frame:` must be SAID
  (its default, `bjd`, is a corrected file's frame and almost never a raw spectrum's):
  `jd` for the observatory's own clock (with `time_scale: utc` typically), or `bjd` for
  barycentric tags (a spacecraft archive's), whose reception instant is then solved from
  the observer's own position and the start direction (`ephemeris.reception_instant`;
  an arcsec of direction moves it by 2.4 ms). `hjd` raises. The shared reader delivers
  the observer's JD_TDB at reception (`TimeSystem._to_observer_tdb`, reached through
  `Instrument._read_data(..., observer_time=)`), never a BJD_TDB: the Roemer delay is a
  function of the fitted star and lives in the graph.
- **The observatory is `observer_location:`**, required, in the vocabulary of
  `ephemeris.get_observer_position`: an astropy site name, a `'lon_deg,lat_deg,height_m'`
  string, or a spacecraft `.eph` file (`'gaia'`). `time_location:` is refused (a second
  spelling of the same place), and so are `'earth'`/`'moon'` (a body's center is no
  spectrograph's site; the Earth's rotation alone is 0.46 km/s).
- **`epoch:`** is the BJD_TDB reference epoch of `star.ra`/`dec`/`pm_ra`/`pm_dec` (e.g.
  2457389.0, Gaia DR3's J2016.0). One star, one set of astrometric parameters, one epoch:
  with absolute astrometry in the system the astrometry's reference epoch is used and a
  raw file's `epoch:` that disagrees raises; without it every raw file must name it.
- **The exposure**: optional columns named in `columns:` (`exptime`, `t_rms`,
  `t_skew3`, in seconds of the file's own time scale), an optional `exposure_meter:`
  file, and `ninterp:` for the nodes ("The exposure average", below).
- **`star.rv` must carry a prior** (next section). `observer_location` and `epoch` on a
  corrected file raise. Corrected and raw files may be mixed in one system; the raw rows
  are written into the likelihood's model with `pt.set_subtensor` on their concrete rows.

### The model

Precomputed at stage 1, parameter-independent (`redshift.observer_terms`):

- the observer's barycentric ICRS position `r_o` and velocity `beta_o = v_o / c` at the
  reception instant (`ephemeris.get_observer_posvel`): for a ground site the Earth's
  barycentric velocity plus the site's GCRS velocity (rotation, with precession,
  nutation and polar motion from astropy/erfa and the IERS tables), combined by the
  relativistic addition `(v_site + v_earth)/(1 + v_site . v_earth / c^2)` WE14 and
  barycorrpy use; for a spacecraft the analytic derivative of the `.eph` position spline;
- the solar-system potential at the observer, `1 + z_grav = 1 / (1 + sum GM/(r c^2))` over
  the Sun, the planets and the Moon (`redshift.GRAVITY_BODIES`; barycorrpy keeps the Sun,
  Earth, Jupiter and Saturn), and for the Sun, Jupiter and Saturn the body-to-observer
  unit vectors, the observer's velocity relative to each, and the `-2GM/(c^2 r)` factors
  of the Shapiro term, taken as the exact time derivative of the Shapiro delay (WE14's
  eq. 27 plus its `d ln r / dt` piece; the Earth's term is dropped).

In the graph, functions of the fitted parameters (`redshift.star_geometry`,
`redshift.doppler_ratio`, `RVInstrument._raw_rv_model`):

- **the star's 3-D motion**: `d0 = 1/parallax`, `mu = pm_ra e_hat + pm_dec n_hat`, and
  `X(T) = d0 r0_hat + (d0 mu + rv r0_hat) T` (AU, `T` Julian years from `epoch`) -- the
  constant space velocity whose tangent-plane linearization is the astrometry's own track
  (`AstrometryInstrument._star_track`), read from the SAME `star.*` parameters. `pm` and
  the position are APPARENT, per year of barycentric arrival time, as astrometry measures
  them; the star's TRUE transverse velocity is then `d0 mu (1 + rv/c)`, because a receding
  star's emission times are compressed by `1 + rv/c` against their arrival times. That
  factor is WE14's/barycorrpy's `z_lighttravel` (4 cm/s for Barnard's star 25 yr from its
  epoch), here part of the velocity rather than a separate term. The perspective
  (secular) acceleration is not a term either: it is `rho_hat . beta_*` changing as
  `rho_hat` turns along `X(T)`.
- **the direction** `rho_hat = (X - r_o)/|X - r_o|`, observer to star (parallax included),
  with `X` taken at the barycentric arrival time `t_B` (one fixed-point step from the
  reception instant; the first pass is off by up to 8.3 min, ~8e-10 rad of Barnard's
  star's motion, +/-0.07 mm/s), and moved by the star's own orbital displacement --
  along the line of sight always, across it where the sky orientation is modeled
  (`redshift.displace_star`; the astrometric wobble in `rho_hat`, 0.5 mm/s for a binary,
  0.03 mm/s for the Gaia-4-like planet);
- **the Roemer delay** to the barycentric arrival time,
  `t_B = t_o + (|X| - |X - r_o|)/c` (with its curvature term; written as
  `(2 X.r_o - r_o.r_o)/(|X| + |X - r_o|)` so it does not cancel). The Shapiro DELAY
  (< 0.1 ms) is not modeled;
- **the star's own light-travel time**: the reflex light leaves the star at its EMISSION
  time `t_emit = t_B - Z_star(t_emit)/c`, with `Z_star` the star's line-of-sight reflex
  displacement in each orbit's own subsystem -- that orbit and the orbits nested inside
  it ("Which displacement retards which orbit", above) -- (`-(M_c/M) z` on an orbit's
  primary side, `+(M_p/M) z` on its companion side in `ltt.py`'s toward-observer `z`;
  a Taylor orbit's own `Z`). `RVInstrument._emission_time` builds it through
  `components/ltt.py`'s one implementation (`line_of_sight_kinematics` per orbit, the
  summed kinematics through `solve_delay`), and `Orbit._ltt_reporting_mask` counts a raw
  file as a retarding consumer of every orbit of its star, so `tc` stays the observed
  conjunction and `tc_target` the target-frame one, as with transits. The effect is
  `~K^2/c`: 0.43 mm/s peak (0.20 mm/s RMS) for the Gaia-4-like example, 3 cm/s for a
  K = 3 km/s, 3-day brown dwarf or M-dwarf companion, 3 m/s at K = 30 km/s and 53 m/s on
  the secondary of a 1.0 + 0.3 Msun 3-day binary (`tests/test_raw_redshift_orbits.py`).
  `light_travel_time: false` on a raw file turns it off. The corrected-velocity path
  retards the reflex the same way (above, "Corrected velocities");
- **the star's total velocity** `beta_* = beta_sys + v_reflex / c`, with every orbit's
  reflex evaluated at `t_emit` through the same `_rv_model` columns a corrected file reads
  (Keplerian and Taylor, `orbit.md`): the line-of-sight part along `r0_hat` (the orbit's
  Z axis), and -- where the system models the orbit's sky orientation (`bigomega`, i.e.
  astrometry is present) -- the transverse part from `Orbit.state_vectors` scaled to the
  star's own reflex (`M_c/M` on the primary side, `M_p/M` on the companion side;
  `RVInstrument._reflex_sky`). Without `bigomega` the transverse reflex is unknown and
  left out: it enters only through `gamma_*`'s `beta_sys . beta_reflex` term, at most
  `|V_perp| K / c` (3 cm/s for the Gaia-4-like example, a periodic signal at the orbit's
  period);
- **the companions' gravitational redshift at the star**, `phi_* = sum_o G M_other,o /
  (r_o c^2)` with `r_o` each orbit's separation at `t_emit`
  (`RVInstrument._companion_potential`): 1.7-3.5 cm/s for the Gaia-4-like planet, 19-39
  cm/s for a K = 3 km/s, 571-day binary (PEXO's numbers). A gamma/K/omega refit absorbs
  most of it, but only by biasing K and omega by ~its amplitude over K, so it is modeled.
  For a star deep in a hierarchy the separation to an outer group is that orbit's
  relative separation (the inner offset is a fraction a_in/a_out); a Taylor orbit has
  none and adds nothing. The star's OWN surface potential is constant and in the zero
  point;
- **the prediction** (WE14 eq. 28 turned around):

      1 + z_meas = (1 + z0_i) gamma_* (1 + rho_hat . beta_*) (1 + phi_*)
                   / [gamma_o (1 + rho_hat . beta_o)] * (1 + z_grav) * (1 + z_Shapiro)

  and the likelihood's model for the row is `c z_meas`.

Why `t_B` and not the emission time: every other component evaluates the orbit at BJD_TDB
arrival times, so the orbit's `period` and `tc` are OBSERVED-frame quantities; evaluating
the raw rows there keeps one orbit for every consumer. The Doppler shift measures the
star's true line-of-sight velocity, so `K` here is the true velocity semi-amplitude, and
the period it pairs with is the observed one -- the two differ from the rest-frame pair
by `1 + rv/c` (~7e-5 at 20 km/s), which only Kepler's third law notices (a 1e-4-level
mass, common to every EXOZIPPy and EXOFASTv2 fit; open item below).

### The exposure average

A spectrum is an exposure, not an instant: its measured redshift is the PHOTON-WEIGHTED
MEAN of the instantaneous one, and the correction is not linear in time, so the model at
the flux-weighted midpoint is itself an error (Tronsgaard, Buchhave, Wright, Eastman &
Blackman 2019, MNRAS 489, 2395; bib key `Tronsgaard:2019`). Its leading term is the
curvature of the Earth's rotation, `<v> - v(<t>) = v'' Var(t) / 2`, with `v''` up to
`omega_E^2 * 465 m/s * cos(lat) cos(dec) = 2.5e-6 m/s^2` six hours from the meridian. For
a uniform exposure of length T (Var = T^2/12) that is **1.33 m/s (T / 1 h)^2 cos(lat)
cos(dec) sin(HA)** -- 0.33 m/s for 30 min and 1.3 m/s for 60 min at worst; Tronsgaard et
al. give `1.32 m/s cos(lat) cos(dec) sin(HA) (T / 1 h)^2`. The third moment adds
`v''' mu_3 / 6`: 9 mm/s for a 1-hour exposure whose flux ramps linearly to zero, 1 mm/s
at 30 min. An orbit's own curvature enters the same way (a K = 30 km/s, 2-day binary:
meters per second).

So the model's prediction for a raw row is the photon-weighted mean of the FULL
instantaneous model -- the barycentric terms, the reflex of every orbit at its own
retarded emission time, the astrometric wobble and the companion potential -- over
quadrature nodes in the exposure. In `RVInstrument._raw_rv_model` the nodes are the rows
of the geometry, `_rv_model` runs on their emission times, and an `inc_subtensor` segment
sum by a constant index forms each row's mean. Per exposure, the most complete
description it has wins (`RVInstrument._exposure_nodes`; the rules are in `redshift.py`):

1. **`exposure_meter: <file>`**, the full record. Long format, one row per meter bin:
   `t_mid` (the exposure's midpoint EXACTLY as the data file's time column writes it --
   that is how a bin finds its exposure: within 1e-6 d, and no match or two matches
   raise; bins of a masked exposure are ignored), `t` (the bin center, in the data file's
   time system), `dt` (the bin width, s) and `counts`. Each bin is a uniform box of
   photons. The model uses the `ninterp`-node Gauss rule OF THAT PHOTON DISTRIBUTION
   (Golub-Welsch on the discrete measure, `redshift.weighted_exposure_rule`): positive
   weights, nodes inside the record, and the photon-weighted mean of every polynomial of
   degree `2 ninterp - 1` reproduced exactly, so a 1000-bin record costs 3 model
   evaluations, not 1000. A meter whose photon-weighted mean time differs from the tag by
   more than 1 s warns (the model uses the meter). Box bins are an assumption: a flux
   gradient inside a bin moves the mean time by ~`dt^2 f' / (12 f)`. For a linear ramp to
   zero over a 10-min exposure that is 1 s (3 cm/s) in 60-s bins and 0.03 mm/s in 2-s
   bins.
2. **`t_rms` (and `t_skew3`) columns**: the flux-weighted RMS of the photon arrival times
   about the flux-weighted midpoint (s) and their third central moment (s^3; needs
   `t_rms`). The 2-node rule with exactly those moments (`redshift.moment_rule`) is
   exact for every cubic in time, which covers the curvature and the skew terms. What it
   cannot know is the 4th moment: 0.43 mm/s at 60 min for a uniform exposure and 0.26
   mm/s for the ramp, scaling as T^4.
3. **`exptime` column** (s): uniform flux over `[t - exptime/2, t + exptime/2]`,
   `ninterp`-node Gauss-Legendre. (Its `t_rms` is `exptime / sqrt(12)`; the column adds
   the shape.)
4. **Nothing**: one node at the tag, the model at the midpoint, wrong by the Tronsgaard
   term above. That is what corrected-RV practice does, and the reason
   `docs/raw_redshift_data.md` asks a release for the meter record.

**`ninterp:`** (default 3, `redshift.DEFAULT_EXPOSURE_NODES`) is read by rules 1 and 3.
Gaussian rules converge far faster than the transit component's midpoint grid, which is
built for light curves with kinks at contact; the redshift is smooth. Against a
4001-point Simpson reference, a 60-min exposure 5 h from the meridian is off by 1.2 m/s
with 1 node, 0.47 mm/s with 2 and 1.3e-4 mm/s with 3. With a K = 100 km/s, P = 1 d binary
on top, every n >= 3 sits at the float64 time-resolution floor (~0.1 mm/s for that K: a
JD carries 40 us). The cost is `ninterp` times the raw rows' geometry and Kepler solves.

The exposure columns are not detrend columns (they are split off before the detrend
block), and a corrected file refuses them (`columns:` names an unknown role). The plots
read the tag: `self.time` is the start star's BJD_TDB of the midpoint. The seed
correction is photon-averaged like the model.

Tests (`tests/test_raw_redshift.py`): the rules' moment exactness; the averaged
correction against barycorrpy integrated densely
(`tests/fixtures/raw_redshift/barycorrpy_exposure.txt`: two sites x two stars x hour
angles -5..+5 h x 10/30/60 min, uniform and ramp exposures), where the averaging shift,
up to 1.13 m/s, agrees to 0.0012 mm/s with Gauss-Legendre, 0.034 mm/s with a 2-s meter
and 0.43/0.26 mm/s with the moment rules; convergence with a fast binary; and the
example's likelihood against `make_data.py`'s 9-point-per-step integration, for all
three kinds of input.

### The instrument zero point, and what the data constrain

Each raw file keeps its own `gamma`, and in this mode `gamma_i` is that instrument's
systemic velocity as its zero point sees it: `1 + gamma_i/c = (1 + z0_i)(1 + rv/c)`.
`z0_i` -- the template's own redshift, the star's gravitational redshift (~0.6 km/s for
the Sun) and its convective blueshift -- is a constant MULTIPLICATIVE factor on the
frequency ratio, so it lives in redshift space; parameterizing it through `gamma_i`
keeps `gamma`'s meaning, units and seed (the mean start-corrected velocity) and keeps
the sampler off the ridge below.

The data constrain the PRODUCT `(1 + z0_i)(1 + rv/c)` per instrument, i.e. `gamma_i`, to
the usual `sigma/sqrt(N)`, and nothing more: every other appearance of `rv` -- the
transverse-velocity stretch, `gamma_*`'s `rv v_reflex / c^2`, the star's motion over the
baseline -- moves the model by at most `~dv K / c` for an error `dv` (1 mm/s per km/s at
K = 330 m/s). So `star.rv` is set by its PRIOR, which is required (a flat `rv` over +/-c
would wander to relativistic speeds, where those terms are nonsense); a Gaia DR3
`radial_velocity` is the natural one, and ~1 km/s is ample. Differences between
instruments' `gamma` are measured exactly as in a corrected fit; the absolute zero point
of any one instrument is not, and is not claimed.

### Validation

- **Against barycorrpy** (`tests/test_raw_redshift.py`, fixture
  `tests/fixtures/raw_redshift/barycorrpy_zb.txt`, regenerated by the script beside it in
  a throwaway barycorrpy environment -- barycorrpy is not a dependency): four sites
  (McDonald, Kitt Peak, La Palma, Paranal) x four stars (Barnard's star at J2016 and at
  J2000 -- 25 yr of its 10 arcsec/yr between epoch and data --, tau Ceti, and a Gaia-4-like
  star at ra 209.5063, dec 31.6955, pm (-75.53, 17.97) mas/yr, parallax 13.63 mas) x 40
  epochs 2015-2026: `c z_B` agrees to **0.12 mm/s worst with DE430** on both sides at
  every epoch with the star above the horizon (0.28 mm/s for Barnard's star at J2000), and
  to **0.27 cm/s worst with the builtin (erfa) ephemeris**, the default, which the
  committed test uses. Before three refinements this model has and barycorrpy does not
  the DE430 agreement was 0.017 cm/s; the differences now are, each confirmed against
  PINT and TEMPO2 by an independent comparison: `X(T)` at `t_B` (barycorrpy propagates
  the star to the observer's TDB; Barnard's star against PINT goes from 0.063 mm/s with
  barycorrpy to 0.015 mm/s here, so up to 0.09 mm/s of the difference is barycorrpy's);
  the Moon and the remaining planets' potentials (a constant 0.065 mm/s); and the exact
  Shapiro derivative with the observer velocity relative to each body and no Earth term
  (barycorrpy's Earth term divides by `1 + R_hat . rho`, which vanishes toward the
  nadir: the worst differences, up to 0.48 mm/s, are all at epochs with the star below
  the horizon). And the inverse: barycorrpy, given the redshifts this model PREDICTS for
  a companion-free star, recovers a constant velocity to < 1 cm/s peak-to-peak.
- **Against the TEMPO comparison set published with WE14** (informational: everything
  here is below barycorrpy's own ~0.2 mm/s precision, and nothing in this mode waits on
  it) (tau Ceti from the CTIO
  1.5-m, 6097 epochs 2000-2015, TEMPO2 in `-tempo1` prediction mode with DE405; WE14
  sec. 5.1 and its online table, as redistributed in barycorrpy's repository as
  `barycorrpy/data/New_tempo_check.txt`): this model and barycorrpy agree with each
  other to 0.002 mm/s and differ from TEMPO identically, by 0.33 mm/s RMS (mean -0.10
  mm/s), with DE405 or DE430 alike. That residual is WE14's sidereal-day signature
  (0.37 mm/s amplitude), and it is **polar motion**: recomputing the observatory velocity
  without polar motion removes the sidereal term entirely (amplitude 0.37 -> 0.013 mm/s,
  RMS 0.33 -> 0.15 mm/s), so TEMPO's `-tempo1` prediction mode leaves polar motion out,
  and this model and barycorrpy keep it (it is real: ~0.3 mm/s at CTIO). What remains is
  white (point-to-point scatter equals the RMS, 0.14 mm/s), plus six epochs within a day
  of the 2005, 2008 and 2012 leap seconds (up to 2.5 mm/s, TEMPO's side). Ruled out, each
  measured against the residual: the TDB/TCB scale `L_B = 1.55e-8` on the velocity (the
  fitted coefficient is -1.2e-9; applying `1 +/- L_B` makes the RMS worse, 0.42/0.45
  mm/s); IAU 1976/1980 precession-nutation with GST94 in place of IAU 2006/2000A (0.01
  mm/s); UT1 = UTC (6.5 mm/s, far worse); the Moon, Venus, Mars, Mercury, Uranus and
  Neptune potentials (a constant 0.064 mm/s, which moves the mean -0.099 -> -0.035 mm/s
  and leaves the RMS unchanged; they are added since, see `GRAVITY_BODIES`); `v^2/c^2`,
  annual, semiannual, lunar and Jupiter-period terms (none reduces the RMS by more than
  0.003 mm/s). The comparison script is `scripts/compare_tempo_zb.py`; the table itself
  is not committed (it is redistributed under barycorrpy's GPL-3.0, this repository is
  BSD-3), so the script takes its path.
- **Against PINT, the regression reference** (`tests/test_raw_redshift.py`, fixture
  `tests/fixtures/raw_redshift/pint_zb.txt`, regenerated by the script beside it, which
  records the PINT version, kernel, clock and IERS settings; PINT is not a dependency):
  four sites x three stars (tau Ceti, Gaia-4-like, Barnard's) x six epochs 2023-2025 on
  DE432s on both sides. Per (site, star) the two differ by the observatory clock
  constant below plus at most 0.032 mm/s; the test pins 0.05 mm/s and the constant's
  range. Marked `network` (it needs the 10 MB DE432s kernel) and skipped, never run on
  the builtin ephemeris, without it.
- **Against native TEMPO2, the independent cross-check** (informational, by hand; the
  method, install recipe and full table are in `docs/timing_code_crosscheck.md`), five
  sites x four stars, 640 epochs over 2023-2025 and 1835 through one night, DE440, per
  (site, star) constant removed:

  | comparison | RMS | max |
  |---|---|---|
  | this model - PINT | 0.012-0.015 mm/s | 0.050 mm/s |
  | this model - native TEMPO2 | 0.016-0.018 mm/s | 0.078 mm/s |
  | PINT - native TEMPO2 | 0.013-0.015 mm/s | 0.060 mm/s |

  The per-site constant between this model and the TT-based codes is the observatory
  clock term `(GM_E/r + v_rot^2/2 - W0)/c` (the geoid potential W0): -0.21 mm/s at Keck,
  -0.087 at Kitt Peak, -0.092 at McDonald, -0.113 at La Palma, -0.116 at Las Campanas.
  Before the Moon and the remaining planets were added to `redshift.GRAVITY_BODIES` the
  measured constant was this plus 0.066 mm/s, and the sum matched to <= 0.006 mm/s at
  every site. The clock term is a constant per site, which each instrument's `z0_i`
  absorbs, so it is not modeled. The float64 floor of one JD time stamp (~40 us) is
  0.0004 mm/s RMS.
- **TEMPO1** (historical): 0.21 mm/s RMS, 0.63 max, all of it polar motion, which TEMPO1
  lacks (with polar motion off here: 0.03 mm/s RMS). The same effect as the 0.33 mm/s
  against the WE14 TEMPO set above.
- **Against PEXO** (Feng et al. 2019; an independent comparison run by a separate
  agent, scripts not in this repository). For single stars at ground sites this model
  matches PEXO to <= 0.015 mm/s once each term is attributed. The attributed terms:

  | term | size | status here |
  |---|---|---|
  | target-system Roemer delay (the star's own light travel) | 0.44 mm/s max for the Gaia-4-like planet; 36 mm/s for a K = 3 km/s, P = 571 d binary (0.27 / 22 mm/s left after a K/omega refit) | modeled (`_emission_time`) |
  | `X` at `t_B` rather than at the reception instant | up to +/-0.07 mm/s, Barnard's star | modeled |
  | the star's orbital displacement in `rho_hat` | 0.5 mm/s binary, 0.03 mm/s planet | modeled |
  | the companion's gravitational redshift at the star | 20-40 mm/s planet, 190-390 mm/s binary (absorbed by a refit only by biasing K, omega) | modeled (`_companion_potential`) |
  | the observatory clock / potential model: the geoid potential W0 against GM_E/r + v_rot^2/2, Earth's J2 (+0.02 to +0.055 mm/s), the Moon (0.042), Venus (0.008, variable), Mars, Mercury, Uranus, Neptune (0.057-0.084 mm/s in all) | a CONSTANT per site: McDonald -0.155, Kitt Peak -0.150, La Palma -0.176, Paranal -0.199 mm/s, shared with barycorrpy | the bodies are now modeled. The clock/geoid/J2 constant is not modeled: it is informational, below barycorrpy's ~0.2 mm/s, and absorbed by every zero point. This model keeps the site-height term, which PEXO drops |
  | tropospheric Doppler | 0.1-1 mm/s at 20-60 deg elevation | out of scope: it needs per-epoch pressure and humidity |

  The WE14 TEMPO residual's mean (-0.099 mm/s at CTIO, -0.035 mm/s with the Moon and
  planets added, above) has the sign and size of the same per-site constant.
  Informational.

  **A trap when comparing with PEXO:** PEXO's orbit runs on BJD_TCB, so its period is
  longer by `1 + L_B` (L_B = 1.55e-8) as well as its periastron time being shifted. This
  model's orbit, like every EXOZIPPy orbit, is in BJD_TDB.
- **Multi-orbit composition** (`tests/test_raw_redshift_orbits.py`): two planets around one
  star with the outer one on a Jacobi orbit (primary group `[A, b]`), and a binary
  observed in its secondary's light, each against an independent numpy composition of
  the star's velocity from the orbits' elements, with its own retardation solve: < 0.1
  mm/s and < 1 mm/s.
- **Recovery** on the simulated example (`examples/raw_redshift_sim`): the likelihood's
  graph reproduces the numpy forward model that made the data to < 1 mm/s at the truth;
  a MAP fit recovers K (328.1 against 330 m/s) and the three zero points (within 1.8 m/s), P within its loose 1.3-orbit constraint (577.7 against 571 d), with every spectrum exposure-averaged; C and JAX gradients are finite and
  agree in every coordinate (ra, dec, pm, distance, rv, the orbit).

### What a corrected-RV fit misses: a Gaia-4-like system

`test_gaia4_like_cross_term_and_pm_consistency_magnitudes`, on the example's 75 epochs
(K = 330 m/s, three ground sites, barycentric velocities -22.4 to +23.2 km/s):

| conventional step | difference from the full model | left after refitting offsets, K and a slope |
|---|---|---|
| correction applied ADDITIVELY, `c z_meas + c z_B` -- the planet cross term `K v_B / c` | 1.1 cm/s RMS, 1.7 cm/s peak | 1.0 cm/s RMS (not absorbable) |
| correction toward the catalog position frozen at its epoch (pm not applied, data 7-9 yr later) | 14 cm/s RMS | 2.9 cm/s RMS |
| exact multiplicative correction with the same, known star, then a Keplerian | the companion's gravitational redshift (1.7-3.5 cm/s) + the star's own light travel (0.4 mm/s) | 2.6 mm/s RMS |

The additive step also multiplies the systemic velocity into the barycentric one
(`rv v_B / c`, 1.4 m/s RMS at the example's assumed rv = -20 km/s), but that part is
common to every planet-free fit and mostly absorbed; the planet's share is what the
table states. The last row is the bound for a fixed, known star: a careful barycorrpy
correction plus a Keplerian misses only the target-system terms -- the companion's
gravitational redshift (most of which a refit hides in gamma and K, biasing them) and
the star's light travel across its orbit. Beyond those, what the mode adds is that the
star is FITTED -- the same `ra`/`dec`/`pm`/`parallax` the astrometry reads, one model and many
perspectives -- and that no step can be applied inconsistently.

### Plots, seeds and prose

- `self.time` is each spectrum's BJD_TDB at the START star (minutes from the input
  clock), used by the plots, the GP's ordering and the seeds; the likelihood forms its own
  `t_B` from the fitted star.
- The seeds (Lomb-Scargle, K, the conjunction-epoch information, gamma's hint) read
  `self.rv_seed`: the redshifts corrected with the start star, `c[(1 + z_B)(1 + z_meas) -
  1]`. For a corrected file `rv_seed` IS `self.rv` (the same array), so nothing moves.
- At a point, the plots draw a raw file's data as `gamma + reflex + the likelihood's own
  residual` (`_plotted_rv`, from the compiled `_rv_bary_data_node = model - (gamma +
  reflex)`), so every corrected-mode panel and residual applies unchanged; without a
  point (the data-only preview) a raw file is drawn as measured.
- The prose (`rvinstrument.raw_redshift`, section `orbits`) states the in-model correction
  and cites `Wright:2014`; where some spectrum is exposure-averaged, a second sentence
  (`rvinstrument.exposure_average`) says so and cites `Tronsgaard:2019`.

### Not modeled, with their size

The tropospheric Doppler term (0.1-1 mm/s at 20-60 deg elevation; it needs pressure
data); Earth's J2 and the geoid clock offset (a per-site constant of -0.09 to -0.21 mm/s,
above, absorbed by each zero point); the Shapiro delay on `t_B` (< 0.1 ms); a Taylor orbit's transverse reflex and transverse displacement (its sky terms are angular and need a parallax; < 1e-3 of a
resolved reflex); the TDB-vs-SI velocity scale (L_B = 1.55e-8: 0.5 mm/s at 30 km/s, as in
WE14; the TEMPO comparison rules it out as a term this model misses).

### Open decisions (for JDE)

1. `c z` (m/s, honoring `unit:`) rather than `z` as the data column.
2. `gamma_i` as the product `(1 + z0_i)(1 + rv/c)` rather than a separate `z0_i`
   parameter, and a REQUIRED `star.rv` prior (raise) rather than a warning.
3. The orbit evaluated at the in-graph `t_B` (observed frame, consistent with every other
   component) with `K` the true velocity amplitude; the `1 + rv/c` between observed and
   rest-frame period in Kepler's law is not applied anywhere in EXOZIPPy.
4. The transverse reflex is included only where `bigomega` exists (astrometry in the
   system); otherwise up to `|V_perp| K / c` (3 cm/s here) is left out.
5. The astrometry's own track is still the linear tangent-plane model; moving it onto
   `redshift.star_geometry`'s 3-D motion would add the astrometric perspective terms
   (mas-level for Barnard's star over a decade) and is not part of this mode.
7. Settled (JDE 2026-11-04): the corrected-velocity path retards the reflex too,
   default on ("Corrected velocities").
6. Default ephemeris `builtin` (0.27 cm/s against DE430); `time_ephemeris: de440` is
   one key away.

Tests: `tests/test_raw_redshift.py`.
