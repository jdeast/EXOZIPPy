# raw_redshift_sim: raw measured redshifts, corrected inside the model

**SIMULATED DATA, NOT A MEASUREMENT.** No raw redshifts of Gaia-4 exist in this
repository. `make_data.py` generates `Gaia4like.{HPF,NEID,FIES}.rv` from the same
forward model the fit uses, so the mode has a worked example and
`tests/test_raw_redshift.py` a recovery case.

The star has Gaia-4's position, proper motion and parallax at Gaia DR3's J2016.0 (ra
209.5063, dec 31.6955, pm (-75.53, 17.97) mas/yr, parallax 13.63 mas). Its systemic
radial velocity (-20 km/s), the three instrument zero points (c z0 = +120, -340, +55
m/s) and the planet (P = 571 d, e = 0.35, omega_* = 210 deg, K = 330 m/s, tc =
2460300) are ASSUMED, chosen to resemble Gaia-4 b. 25 spectra per site at 10 m/s, on
random nights 2023-2025 with the star above 35 deg, from McDonald (HPF), Kitt Peak (NEID)
and Roque de los Muchachos (FIES): barycentric velocities from -22.4 to +23.2 km/s.

`make_data.py`'s `tc` is the TARGET-frame conjunction. Because a raw file retards the
orbit, the fit's `orbit.b.tc` is the OBSERVED (virtual transit) time and its
`tc_target` the target-frame one, as for a transit fit, so the fitted `tc` sits the
planet's light travel (hundreds of seconds) from the value in `make_data.py`. The reflex
is evaluated at the star's emission time (its own light-travel time across
the orbit, 0.4 mm/s here). Each file holds the flux-weighted exposure midpoint in JD_UTC, `c * z_meas` in m/s (the
redshift against the template before any barycentric correction) and its error; the
config reads them with `data_format: raw_redshift`, `time_frame: jd`, `time_scale: utc`,
the site as `observer_location` and `epoch: 2457389.0` for the star's astrometric
parameters. `star.rv` carries a prior (as a Gaia DR3 radial velocity would supply): the
redshifts constrain only each instrument's `(1 + z0_i)(1 + rv/c)`, which the fit reports
as that instrument's `gamma`.

The model, its validation against barycorrpy and what a corrected-RV fit would miss on
this system are in `src/exozippy/components/rvinstrument/rvinstrument.md`; why to publish
raw `c z`, and what a release should carry, in `docs/raw_redshift_data.md`.

    python make_data.py                 # regenerate the data (deterministic seed)
    exozippy raw_redshift_sim.yaml      # fit
