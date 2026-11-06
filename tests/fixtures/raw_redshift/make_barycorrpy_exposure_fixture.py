"""Regenerate barycorrpy_exposure.txt: barycorrpy's correction AVERAGED over
whole exposures, for the photon-weighted test in tests/test_raw_redshift.py.

Run in the same THROWAWAY barycorrpy environment as
make_barycorrpy_fixture.py (barycorrpy is not an EXOZIPPy dependency):

    /some/scratch/bcvenv/bin/python make_barycorrpy_exposure_fixture.py

The measured redshift of an exposure is the photon-weighted MEAN of the
instantaneous one (Tronsgaard et al. 2019, MNRAS 489, 2395), so for a star
with no companions

    1 + z_meas = (1 + z_true) < 1 / (1 + z_B(t)) >_photons

with barycorrpy's z_B (convention 1 + z_true = (1 + z_B)(1 + z_meas)).
Each case integrates barycorrpy's c*z_B on a dense grid (N_DENSE-point
Simpson rule) over one exposure and records the EFFECTIVE correction
c*z_B,eff = c (1 / <1/(1 + z_B)> - 1) beside barycorrpy's value at the
flux-weighted midpoint, for two photon distributions:

* ``uniform``: constant flux over [start, start + T];
* ``ramp``:    flux falling linearly to zero, f ~ (start + T - t) -- a
               strongly skewed exposure (clouds rolling in); its
               flux-weighted midpoint is start + T/3.

Cases: two sites x two stars x hour angles -5..+5 h x T = 10, 30, 60 min.
barycorrpy does not care that some of these are below the horizon.
"""

import numpy as np
from astropy.time import Time
from make_barycorrpy_fixture import SITES, STARS

C = 299792458.0
N_DENSE = 1001
HOUR_ANGLES = (-5.0, -3.0, 0.0, 3.0, 5.0)
EXPOSURES_S = (600.0, 1800.0, 3600.0)
SITE_IDX = (0, 3)  # McDonald, Paranal
STAR_IDX = (1, 2)  # Gaia4like, tauCet
JD0 = 2460400.5
SIDEREAL = 0.9972695663


def _time_at_hour_angle(ra_deg, lon, lat, alt, ha_h):
    from astropy.coordinates import EarthLocation

    loc = EarthLocation.from_geodetic(lon, lat, alt)
    t = JD0
    for _ in range(4):
        lst = (
            Time(t, format="jd", scale="utc", location=loc)
            .sidereal_time("apparent")
            .hour
        )
        dh = ((ha_h - (lst - ra_deg / 15.0)) + 12.0) % 24.0 - 12.0
        t = t + dh / 24.0 * SIDEREAL
    return t


if __name__ == "__main__":
    from barycorrpy import get_BC_vel

    lines = [
        "# barycorrpy get_BC_vel(zmeas=0, ephemeris='de430') averaged over "
        "exposures (make_barycorrpy_exposure_fixture.py)",
        "# STAR/SITE indices refer to barycorrpy_zb.txt's lists",
        "# columns: site_index star_index shape JD_UTC_start T[s] "
        "JD_UTC_fluxmid c*z_B(fluxmid)[m/s] c*z_B,eff[m/s]",
    ]
    x = np.linspace(0.0, 1.0, N_DENSE)
    simpson = np.ones(N_DENSE)
    simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
    for a in SITE_IDX:
        _, lon, lat, alt = SITES[a]
        for b in STAR_IDX:
            _, ra, dec, pmra, pmdec, px, rv, ep = STARS[b]

            def zb(jd):
                v, _, _ = get_BC_vel(
                    Time(jd, format="jd", scale="utc"),
                    ra=ra,
                    dec=dec,
                    epoch=ep,
                    pmra=pmra,
                    pmdec=pmdec,
                    px=px,
                    rv=rv,
                    lat=lat,
                    longi=lon,
                    alt=alt,
                    zmeas=0.0,
                    ephemeris="de430",
                    leap_update=False,
                )
                return np.asarray(v, float) / C

            for ha in HOUR_ANGLES:
                mid0 = _time_at_hour_angle(ra, lon, lat, alt, ha)
                for T in EXPOSURES_S:
                    start = mid0 - 0.5 * T / 86400.0
                    jd = start + x * T / 86400.0
                    z = zb(jd)
                    for shape, flux, fmid in (
                        ("uniform", np.ones(N_DENSE), 0.5),
                        ("ramp", 1.0 - x, 1.0 / 3.0),
                    ):
                        w = simpson * flux
                        w = w / w.sum()
                        eff = 1.0 / np.sum(w / (1.0 + z)) - 1.0
                        tmid = start + fmid * T / 86400.0
                        lines.append(
                            f"{a} {b} {shape} {start:.10f} {T:.1f} "
                            f"{tmid:.10f} {C * zb([tmid])[0]:.6f} "
                            f"{C * eff:.6f}"
                        )
    with open("barycorrpy_exposure.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
