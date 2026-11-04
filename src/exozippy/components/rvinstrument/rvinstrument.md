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
byte-identical to the file before the key existed: the start logp of `kelt4_rvonly`,
`hd80606_rvonly`, `toi3919`, `GaiaBH1`, `wasp18` and `hat3` is unchanged to the last bit.

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
  the Sun, Earth, Jupiter and Saturn (barycorrpy's set), and the body-to-observer unit
  vectors and `-2GM/(c^2 r)` factors of WE14's Shapiro term (eq. 27).

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
  displacement summed over EVERY orbit it is a body of (`-(M_c/M) z` on an orbit's
  primary side, `+(M_p/M) z` on its companion side in `ltt.py`'s toward-observer `z`;
  a Taylor orbit's own `Z`). `RVInstrument._emission_time` builds it through
  `components/ltt.py`'s one implementation (`line_of_sight_kinematics` per orbit, the
  summed kinematics through `solve_delay`), and `Orbit._ltt_reporting_mask` counts a raw
  file as a retarding consumer of every orbit of its star, so `tc` stays the observed
  conjunction and `tc_target` the target-frame one, as with transits. The effect is
  `~K^2/c`: 0.43 mm/s peak (0.20 mm/s RMS) for the Gaia-4-like example, 3 cm/s for a
  K = 3 km/s, 3-day brown dwarf or M-dwarf companion, 3 m/s at K = 30 km/s and 53 m/s on
  the secondary of a 1.0 + 0.3 Msun 3-day binary (`tests/test_raw_redshift_orbits.py`).
  `light_travel_time: false` on a raw file turns it off. The CORRECTED-velocity path does
  not retard the reflex (it evaluates the orbit at the BJD_TDB tag), which is the same
  omission; it is left as it is, byte-identical, and recorded as an open item;
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
  epochs 2015-2026: `c z_B` agrees to **0.017 cm/s worst with DE430** on both sides (0.017
  is Barnard's at J2000; 0.004 at J2016, 0.0002 for the others), and to **0.27 cm/s worst
  with the builtin (erfa) ephemeris**, the default, which the committed test uses. And
  the inverse: barycorrpy, given the redshifts this model PREDICTS for a companion-free
  star, recovers a constant velocity to < 1 cm/s peak-to-peak.
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
  and leaves the RMS unchanged -- absorbed by any zero point, so not added); `v^2/c^2`,
  annual, semiannual, lunar and Jupiter-period terms (none reduces the RMS by more than
  0.003 mm/s). The comparison script is `scripts/compare_tempo_zb.py`; the table itself
  is not committed (it is redistributed under barycorrpy's GPL-3.0, this repository is
  BSD-3), so the script takes its path.
- **Against PEXO** (Feng et al. 2019; an independent comparison run by a separate
  agent, scripts not in this repository). For single stars at ground sites this model
  matches PEXO to <= 0.015 mm/s once each term is attributed. The attributed terms:

  | term | size | status here |
  |---|---|---|
  | target-system Roemer delay (the star's own light travel) | 0.44 mm/s max for the Gaia-4-like planet; 36 mm/s for a K = 3 km/s, P = 571 d binary (0.27 / 22 mm/s left after a K/omega refit) | modeled (`_emission_time`) |
  | `X` at `t_B` rather than at the reception instant | up to +/-0.07 mm/s, Barnard's star | modeled |
  | the star's orbital displacement in `rho_hat` | 0.5 mm/s binary, 0.03 mm/s planet | modeled |
  | the companion's gravitational redshift at the star | 20-40 mm/s planet, 190-390 mm/s binary (absorbed by a refit only by biasing K, omega) | modeled (`_companion_potential`) |
  | the observatory clock / potential model: the geoid potential W0 against GM_E/r + v_rot^2/2, Earth's J2 (+0.02 to +0.055 mm/s), the Moon (0.042), Venus (0.008, variable), Mars, Mercury, Uranus, Neptune (0.057-0.084 mm/s in all) | a CONSTANT per site: McDonald -0.155, Kitt Peak -0.150, La Palma -0.176, Paranal -0.199 mm/s, shared with barycorrpy | not modeled: informational, below barycorrpy's ~0.2 mm/s and absorbed by every zero point. This model keeps the site-height term, which PEXO drops |
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
  a MAP fit recovers K, P and the three zero points; C and JAX gradients are finite and
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
  and cites `Wright:2014`.

### Not modeled, with their size

The tropospheric Doppler term (0.1-1 mm/s at 20-60 deg elevation; it needs pressure
data); Earth's J2 and the geoid clock offset (a per-site constant < 0.2 mm/s, above); the
Shapiro delay on `t_B` (< 0.1 ms); the Moon's and the other planets' potentials (<
0.1 mm/s); a Taylor orbit's transverse reflex and transverse displacement (its sky terms are angular and need a parallax; < 1e-3 of a
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
7. The corrected-velocity path (`data_format: rv`) also evaluates the reflex at the
   BJD_TDB tag, without the star's own light-travel time (`~K^2/c`); left unchanged and
   byte-identical here, for a separate decision.
6. Default ephemeris `builtin` (0.27 cm/s against DE430); `time_ephemeris: de440` is
   one key away.

Tests: `tests/test_raw_redshift.py`.
