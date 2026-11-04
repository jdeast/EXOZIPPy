# Publishing raw redshifts: a data standard for precise RVs

This note is for instrument teams and authors releasing precise radial velocities
(G-CLEF and the next generation of extreme-precision RV instruments in particular).
It argues for releasing the **raw measured redshift** `c * z_meas` of every spectrum,
with a small fixed set of metadata, **alongside or instead of** barycentric-corrected
velocities. It then gives a file layout EXOZIPPy reads directly, and describes what
EXOZIPPy does with it (`rvinstrument` `data_format: raw_redshift`;
`src/exozippy/components/rvinstrument/rvinstrument.md`).

## Why publish `c z_meas` rather than corrected RVs

A barycentric-corrected velocity is not a measurement. It is a measurement combined
with a model of the observer, the star and the solar system. That model includes:

- the star's coordinates, their epoch, its proper motion, parallax and systemic RV;
- the ephemeris;
- the Earth orientation (precession, nutation, UT1, polar motion);
- the observatory coordinates;
- the time of the exposure's photon-weighted midpoint;
- the choice between applying the correction multiplicatively (Wright & Eastman 2014,
  eq. 28) and additively.

A reader of the corrected values usually cannot recover which star model was used, or
whether the correction was multiplicative. Every one of these choices leaves a
time-variable signal in the corrected velocities. That signal correlates with the
barycentric velocity (annual, and sidereal-daily for a ground site), and no later fit can
remove it without knowing the choices. A raw `c z_meas` with its metadata lets anyone
redo the correction. In EXOZIPPy the correction is part of the fit itself, using the same
star parameters the astrometry constrains.

How large the errors are. Each number below is the RMS over three years of epochs at
McDonald (2023-2026, star coordinates at J2016.0), from EXOZIPPy's model
(`redshift.py`, which agrees with barycorrpy to 0.017 cm/s). "After a refit" removes an
offset, a linear trend and a refit Keplerian amplitude, which is what a corrected-RV fit
could absorb on its own.

### Applying the correction additively, `c z_meas + c z_B`, instead of multiplicatively

| systemic rv | K = 10 m/s | K = 330 m/s | K = 3 km/s | K = 30 km/s |
|---|---|---|---|---|
| 0 km/s | 0.64 m/s | 0.64 m/s | 0.64 m/s | 1.22 m/s |
| 30 km/s | 1.67 m/s | 1.67 m/s | 1.69 m/s | 2.15 m/s |
| 100 km/s | 5.3 m/s | 5.3 m/s | 5.4 m/s | 5.7 m/s |
| 300 km/s | 16.0 m/s | 16.0 m/s | 16.1 m/s | 16.3 m/s |
| after a refit, rv = 30 km/s | 1.64 m/s | 1.64 m/s | 1.66 m/s | 2.09 m/s |
| the companion's own share, `K v_B / c` | 0.04 cm/s | 1.2 cm/s | 11 cm/s | 1.1 m/s |

The cross term `z_B * z_meas` has three parts:

- the barycentric velocity times itself, `v_B^2 / c` (present even at rv = 0);
- the systemic rv times `v_B` (`rv v_B / c`, which grows linearly with rv);
- the companion's reflex times `v_B`, which nobody accounts for.

The first two survive any refit at the m/s level for every star. The companion term
exceeds 1 cm/s at K ~ 300 m/s, 10 cm/s at K ~ 3 km/s (brown dwarfs, M-dwarf binaries)
and 1 m/s at K ~ 30 km/s (stellar binaries).

### The star's own light-travel time across its orbit

A corrected-RV fit evaluates the reflex at the barycentric arrival time; the light left
the star earlier or later by its own line-of-sight displacement over c, which is worth
`~K^2 / c` in velocity: 0.4 mm/s for K = 330 m/s, 3 cm/s for K = 3 km/s (a brown dwarf
or M-dwarf companion on a 3-day orbit), 3 m/s for K = 30 km/s, and 53 m/s on the
secondary of a 1.0 + 0.3 Msun 3-day binary. EXOZIPPy's raw-redshift mode retards the
reflex; it exceeds 1 cm/s at K ~ 1.7 km/s, 10 cm/s at K ~ 5.5 km/s and 1 m/s at
K ~ 17 km/s.

### Coordinate, epoch and proper-motion errors in the correction

| error in the star model | Gaia-4-like (78 mas/yr, 73 pc) | tau Ceti (1.9"/yr, 3.7 pc) | Barnard's star (10.4"/yr, 1.8 pc) |
|---|---|---|---|
| pm not applied (J2016 position, 2023-26 data) | 6.4 cm/s | 1.3 m/s | 5.3 m/s |
| J2016 coordinates taken to be J2000 | 12 cm/s | 2.2 m/s | 8.1 m/s |
| position wrong by 1 arcsec | 9.5 cm/s | 9.5 cm/s | 10 cm/s |
| pm wrong by 1 mas/yr | 0.08 cm/s | 0.09 cm/s | 0.09 cm/s |
| parallax ignored (direction from the barycenter) | 0.04 cm/s | 1.1 cm/s | 2.0 cm/s |
| systemic rv wrong by 100 km/s | < 0.01 cm/s | 0.04 cm/s | 0.6 cm/s |
| perspective acceleration `d mu^2` (per year) | 1.0 cm/s | 31 cm/s | 4.5 m/s |

Refitting an offset and a slope barely helps the first rows: after a refit they are
4.2 m/s for Barnard's star and 1.1 m/s for tau Ceti. The perspective acceleration is a
slope that a corrected-RV fit absorbs only when it fits one, and then it is degenerate
with a real companion's acceleration.

### Spacecraft observers

Treating Gaia as if it observed from the geocenter changes `c z_B` by 165 m/s RMS (330
m/s peak). Its L2 orbit and the 1.5 million km offset from the Earth must come from its
own ephemeris.

### Totals

| threshold | what exceeds it |
|---|---|
| 1 m/s | the additive correction for any star; a pm error or wrong epoch for any star within ~10 pc; Barnard-class perspective acceleration |
| 10 cm/s | a 1-arcsec position error; the companion cross term for K >~ 3 km/s |
| 1 cm/s | the companion cross term at K >~ 300 m/s; a frozen pm at 70 pc; the star's light-travel time at K >~ 1.7 km/s |

## What a release must carry, per spectrum

1. **The flux-weighted (photon-weighted) exposure midpoint**, and the time scale and
   frame it is in. Usually UTC at the observatory: say "JD_UTC" or "MJD_UTC", never a
   bare "JD". The exposure-meter weighting matters at the 1 cm/s level (WE14 sec. 6).
2. **The observatory.** Give geodetic longitude (East positive), latitude and height above
   the WGS84 ellipsoid to ~10 m, or ITRF XYZ. For a spacecraft, give the ephemeris it used
   (a NAIF id, or the Horizons query).
3. **The exposure time.** Needed to judge smearing, and for any later re-weighting.
4. **`c * z_meas` and its uncertainty**, in m/s, defined as `c (lambda_obs / lambda_ref -
   1)`. This is NOT the output of a log or relativistic "velocity" formula, which differ
   by `c z^2 / 2` (1.5 m/s at 30 km/s). If the pipeline works in `ln(lambda)`, convert
   before publishing.
5. **The template / zero-point definition**: what `lambda_ref` is (a stellar template, its
   own rest frame, a line list, a laser frequency comb with an absolute calibration).
   State whether the zero point is meant to be absolute or only relative within the
   instrument.
6. **Every correction already applied to `z_meas`, and every one that is not**:
   instrumental drift (simultaneous reference, comb), detector charge-transfer
   corrections, telluric masking, the instrument's own zero-point offsets between
   epochs, and in particular **whether any part of a barycentric correction has
   already been applied** (some pipelines remove the Earth rotation or a coarse BERV
   before cross-correlating). A raw file must have none applied.
7. Optionally, the pipeline's own BERV and the star model it used (coordinates, epoch,
   pm, parallax, rv). This lets a reader audit the correction, not just redo it.

## A file layout EXOZIPPy reads directly

One whitespace-delimited text file per instrument and observatory. `#` header lines
carry the metadata as `key: value` pairs. The data columns follow, in this order: time,
`c z`, uncertainty, then any per-spectrum extras.

```
# instrument: G-CLEF
# observatory: Las Campanas, lon -70.6915 lat -29.0146 height 2380 m (WGS84)
# time: JD_UTC, flux-weighted exposure midpoint (exposure meter)
# cz: c*(lambda_obs/lambda_ref - 1) [m/s]; NO barycentric correction applied
# zero_point: template = co-added observed spectrum of this star (relative)
# corrections_applied: simultaneous LFC drift; CTI
# corrections_not_applied: barycentric (none of it), telluric
# columns: JD_UTC  cz[m/s]  cz_err[m/s]  exptime[s]
2460401.732514 -24103.112 0.31 900
...
```

EXOZIPPy reads the first three columns as written:

```yaml
rvinstrument:
  - name: "GCLEF"
    file: "star.GCLEF.cz"
    data_format: raw_redshift
    observer_location: "-70.6915,-29.0146,2380"
    time_frame: jd
    time_scale: utc
    epoch: 2457389.0          # the epoch of the star.ra/dec/pm you give
    columns: {time: 0, rv: 1, err: 2}
```

The `#` lines are ignored by the reader, which is the point of putting the metadata
there: the file stays the document of record. If a release interleaves several
observatories, split it per observatory, since the observer is a per-file key.

## What EXOZIPPy does with it

`data_format: raw_redshift` predicts each spectrum's `c z_meas` as a function of the
fitted parameters. The full description is in
`src/exozippy/components/rvinstrument/rvinstrument.md`; in outline:

- The fitted star's direction and 3-D space motion (`star.ra/dec/pm_ra/pm_dec/parallax`
  at their epoch, and `star.rv`) are the same parameters any astrometry in the fit reads.
- The Roemer delay to each spectrum's barycentric arrival time is computed from that
  fitted star.
- The star's own light-travel time across every one of its orbits is included.
- Every orbit's reflex (Keplerian and Taylor) enters through the multiplicative
  relativistic Doppler relation of WE14, together with the observatory's barycentric
  velocity (Earth orbit + rotation with full Earth orientation, or a spacecraft
  ephemeris) and the solar-system gravitational and Shapiro terms.
- Each instrument keeps one zero point, which absorbs the template, the star's
  gravitational redshift and convective blueshift. The star's absolute rv comes from a
  prior, such as a Gaia radial velocity.

The cross terms, the proper-motion and perspective terms and the coordinate epoch are
therefore never separate steps that can be skipped or applied inconsistently. The
correction is validated against barycorrpy (Kanodia & Wright 2018) to 0.017 cm/s, and
against the TEMPO comparison set published with WE14 to the precision of that set.
