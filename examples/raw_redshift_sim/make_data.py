"""Simulate the raw-redshift example: a Gaia-4-like star seen from three sites.

SIMULATED DATA, NOT A MEASUREMENT.  No raw redshifts of Gaia-4 exist in this
repository; these files are generated from the forward model below so that
the raw-redshift mode (rvinstrument data_format: raw_redshift) has a worked
example and tests/test_raw_redshift.py a recovery case.  The star's
position, proper motion and parallax are Gaia-4's (ra 209.5063, dec 31.6955,
pm (-75.53, 17.97) mas/yr, parallax 13.63 mas); its systemic radial
velocity, the instrument zero points and the planet's elements are ASSUMED,
chosen to resemble Gaia-4 b (K ~ 330 m/s on a ~570-day eccentric orbit).

Each spectrum's c*z_meas is the forward model of
src/exozippy/components/rvinstrument/redshift.py at the truth (the same
model the fit uses), plus Gaussian noise:

    1 + z_meas = (1 + z0_i) gamma_*(1 + rho.beta_*) (1 + z_grav)(1 + z_Shapiro)
                 / [gamma_o (1 + rho.beta_o)]

with the observatory state from astropy (builtin ephemeris) at the
flux-weighted exposure midpoint (JD_UTC, column 1), and the reflex at the
star's EMISSION time (its own light-travel time across the orbit,
t_emit = t_B - Z_star(t_emit)/c).  Epochs are random
nights with the star above 35 deg and the Sun below -12 deg.

    python make_data.py          # rewrites Gaia4like.{HPF,NEID,FIES}.rv
"""

import numpy as np

#: The injected truth.  ra/dec/pm/parallax at the Gaia DR3 epoch J2016.0.
TRUTH = {
    "ra": 209.5063,
    "dec": 31.6955,
    "pm_ra": -75.53,
    "pm_dec": 17.97,
    "parallax": 13.63,
    "rv": -20000.0,  # ASSUMED systemic velocity, m/s
    "epoch": 2457389.0,
    "period": 571.0,
    "ecc": 0.35,
    "omega": 210.0,  # omega_*, deg
    "K": 330.0,  # m/s
    "tc": 2460300.0,
}
#: The three ground sites (East longitude, latitude, height m): McDonald
#: (HPF), Kitt Peak (NEID), Roque de los Muchachos (FIES).
SITES = {
    "HPF": "-104.0147,30.6814,2026",
    "NEID": "-111.6003,31.9583,2091",
    "FIES": "-17.8850,28.7577,2382",
}
#: ASSUMED instrument zero points, c*z0 in m/s (template, gravitational
#: redshift and convective blueshift of the star -- constants).
ZERO_POINTS = {"HPF": 120.0, "NEID": -340.0, "FIES": 55.0}
N_PER_SITE = 25
ERR = 10.0  # m/s
SEED = 7
SPAN = (2459950.0, 2460700.0)  # JD_UTC


def gamma_truth(name):
    """The fit's gamma for instrument ``name``: c[(1 + z0)(1 + rv/c) - 1]."""
    c = 299792458.0
    return c * ((1.0 + ZERO_POINTS[name] / c) * (1.0 + TRUTH["rv"] / c) - 1.0)


def keplerian_rv(t, tc=None):
    """The star's line-of-sight reflex (m/s) at BJD_TDB ``t``, in
    EXOZIPPy's convention: K [cos(f + omega_*) + e cos omega_*], with the
    conjunction tc at f = 90 deg - omega_*."""
    P, e, K = TRUTH["period"], TRUTH["ecc"], TRUTH["K"]
    tc = TRUTH["tc"] if tc is None else tc
    w = np.radians(TRUTH["omega"])
    f0 = 0.5 * np.pi - w
    E0 = 2.0 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(0.5 * f0))
    tp = tc - (E0 - e * np.sin(E0)) * P / (2.0 * np.pi)
    M = 2.0 * np.pi * (np.asarray(t) - tp) / P
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    f = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(0.5 * E), np.sqrt(1 - e) * np.cos(0.5 * E)
    )
    return K * (np.cos(f + w) + e * np.cos(w))


def star_los_position(t, tc=None):
    """The star's reflex displacement AWAY from the observer (m) at
    target-frame time ``t``: K sqrt(1 - e^2)/n (r/a) sin(f + omega_*),
    whose time derivative is ``keplerian_rv``."""
    P, e, K = TRUTH["period"], TRUTH["ecc"], TRUTH["K"]
    tc = TRUTH["tc"] if tc is None else tc
    w = np.radians(TRUTH["omega"])
    f0 = 0.5 * np.pi - w
    E0 = 2.0 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(0.5 * f0))
    tp = tc - (E0 - e * np.sin(E0)) * P / (2.0 * np.pi)
    M = 2.0 * np.pi * (np.asarray(t) - tp) / P
    E = M.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    f = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(0.5 * E), np.sqrt(1 - e) * np.cos(0.5 * E)
    )
    n = 2.0 * np.pi / (P * 86400.0)
    return K * np.sqrt(1 - e**2) / n * (1 - e * np.cos(E)) * np.sin(f + w)


def emission_time(t_bary, tc=None):
    """The star's emission time for light reaching the barycenter at
    ``t_bary``: t_emit = t_bary - Z(t_emit)/c, by fixed point."""
    t = np.array(t_bary, dtype=float)
    for _ in range(8):
        t = t_bary - star_los_position(t, tc) / 299792458.0 / 86400.0
    return t


def night_times(site, n, rng):
    """``n`` sorted JD_UTC with the star above 35 deg, the Sun below -12."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
    from astropy.time import Time

    lon, lat, h = (float(x) for x in site.split(","))
    loc = EarthLocation.from_geodetic(lon, lat, h)
    target = SkyCoord(TRUTH["ra"] * u.deg, TRUTH["dec"] * u.deg)
    out = []
    while len(out) < n:
        t = Time(rng.uniform(*SPAN), format="jd", scale="utc")
        frame = AltAz(obstime=t, location=loc)
        if (
            target.transform_to(frame).alt.deg > 35.0
            and get_sun(t).transform_to(frame).alt.deg < -12.0
        ):
            out.append(t.jd)
    return np.sort(np.array(out))


def predict(name, jd_utc, reflex=True, star=None, tc=None):
    """The noise-free c*z_meas (m/s) at JD_UTC ``jd_utc`` for site
    ``name``; ``star`` overrides the truth's (ra, dec, pm_ra, pm_dec,
    parallax, rv) in internal units; ``tc`` the TARGET-frame conjunction
    (default the truth's -- see the README on tc vs tc_target)."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    from exozippy.components.rvinstrument.redshift import (
        observer_terms,
        predicted_cz,
    )

    site = SITES[name]
    lon, lat, h = (float(x) for x in site.split(","))
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h * u.m)
    t_tdb = Time(jd_utc, format="jd", scale="utc", location=loc).tdb.jd
    obs = observer_terms(t_tdb, site, "builtin")
    if star is None:
        star = (
            np.radians(TRUTH["ra"]),
            np.radians(TRUTH["dec"]),
            TRUTH["pm_ra"],
            TRUTH["pm_dec"],
            TRUTH["parallax"],
            TRUTH["rv"],
        )
    z0 = ZERO_POINTS[name] / 299792458.0
    cz, t_bary = predicted_cz(
        t_tdb,
        obs,
        star,
        TRUTH["epoch"],
        z0=z0,
        v_reflex=(
            (lambda tb: (keplerian_rv(emission_time(tb, tc), tc), None, None))
            if reflex
            else None
        ),
    )
    return cz, t_bary, t_tdb, obs


def main():
    rng = np.random.default_rng(SEED)
    for name, site in SITES.items():
        jd = night_times(site, N_PER_SITE, rng)
        cz, _, _, _ = predict(name, jd)
        cz = cz + rng.normal(0.0, ERR, cz.size)
        np.savetxt(
            f"Gaia4like.{name}.rv",
            np.column_stack([jd, cz, np.full(cz.size, ERR)]),
            fmt="%.8f %.4f %.2f",
            header=(
                "SIMULATED raw redshifts -- NOT REAL DATA (make_data.py).\n"
                "JD_UTC (flux-weighted exposure midpoint)  c*z_meas [m/s]  "
                "err [m/s]"
            ),
        )


if __name__ == "__main__":
    main()
