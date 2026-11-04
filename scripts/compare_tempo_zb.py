"""Compare redshift.barycentric_redshift with the TEMPO set of Wright & Eastman 2014.

WE14 (PASP 126, 838, sec. 5.1 and its online table) compared their barycentric
correction with TEMPO2 (``-tempo1`` prediction mode, DE405) for tau Ceti from the
CTIO 1.5-m.  barycorrpy redistributes that table as
``barycorrpy/data/New_tempo_check.txt`` (columns JD_UTC, TEMPO c*z_B [m/s],
barycorrpy c*z_B [m/s]); it is GPL-3.0 there, so it is not committed here.

    python scripts/compare_tempo_zb.py New_tempo_check.txt de430
    python scripts/compare_tempo_zb.py New_tempo_check.txt /path/to/de405.bsp

Prints this model minus TEMPO, the sidereal-day amplitude, and the same with the
observatory velocity recomputed WITHOUT polar motion (TEMPO's -tempo1 mode leaves it
out; see src/exozippy/components/rvinstrument/rvinstrument.md, "Validation").
"""

import argparse

import numpy as np


def compare(path, ephemeris):
    import astropy.units as u
    import erfa
    from astropy.coordinates import Angle, EarthLocation
    from astropy.time import Time
    from astropy.utils import iers

    from exozippy.components.rvinstrument.redshift import (
        barycentric_redshift,
        observer_terms,
    )
    from exozippy.constants import C_MPS

    d = np.loadtxt(path, delimiter=",", skiprows=1)
    jd, tempo = d[:, 0], d[:, 1]
    xyz = np.array([1814985.3, -5213916.8, -3187738.1])  # WE14 optical.dat
    loc = EarthLocation.from_geocentric(*xyz, unit=u.m)
    site = (
        f"{loc.lon.deg:.12f},{loc.lat.deg:.12f},{loc.height.to(u.m).value:.6f}"
    )
    # WE14's 10700.par: tau Ceti at the Hipparcos epoch, rv 0.
    ra = Angle("01:44:05.1275 hours").rad
    dec = Angle("-15:56:22.4006 deg").rad
    star = (ra, dec, -1721.05, 854.16, 273.96, 0.0)
    epoch = 48348.5625 + 2400000.5

    t_tdb = Time(jd, format="jd", scale="utc", location=loc).tdb.jd
    obs = observer_terms(t_tdb, site, ephemeris)
    resid = C_MPS * barycentric_redshift(t_tdb, obs, star, epoch) - tempo

    # The site's GCRS velocity with and without polar motion (IAU 2006/2000A).
    t = Time(jd, format="jd", scale="utc")
    tab = iers.earth_orientation_table.get()
    xp, yp = (a.to(u.rad).value for a in tab.pm_xy(t))
    dut1 = tab.ut1_utc(t).to(u.s).value

    def gcrs(dt, pm):
        tt = t.tt
        X, Y = (xp, yp) if pm else (0 * xp, 0 * yp)
        R = erfa.c2t06a(
            tt.jd1, tt.jd2 + dt, t.jd1, t.jd2 + dt + dut1 / 86400.0, X, Y
        )
        return np.einsum("nji,j->ni", R, xyz)

    h = 1.0 / 86400.0

    def vel(pm):
        return (gcrs(h, pm) - gcrs(-h, pm)) / (2 * h * 86400.0)

    n_hat = np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    no_pm = resid + (vel(False) - vel(True)) @ n_hat

    leaps = np.array([2453736.5, 2454832.5, 2456109.5])
    ok = np.min(np.abs(jd[:, None] - leaps[None, :]), axis=1) >= 1.0
    tt = jd - jd[0]
    A = np.column_stack(
        [
            np.ones_like(tt),
            np.sin(2 * np.pi * tt / 0.99726957),
            np.cos(2 * np.pi * tt / 0.99726957),
        ]
    )[ok]
    for label, r in (
        ("this model - TEMPO", resid),
        ("  without polar motion", no_pm),
    ):
        y = r[ok]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        print(
            f"{label:24s} mean {y.mean() * 1e3:7.3f}  rms {y.std() * 1e3:6.3f}  "
            f"sidereal amp {np.hypot(*c[1:]) * 1e3:6.3f} mm/s  "
            f"(excluding {np.count_nonzero(~ok)} leap-second epochs)"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare EXOZIPPy's in-model barycentric correction with the "
            "TEMPO set of Wright & Eastman (2014)."
        )
    )
    parser.add_argument(
        "table", help="New_tempo_check.txt (barycorrpy/data/ in its repo)"
    )
    parser.add_argument(
        "ephemeris",
        nargs="?",
        default="de430",
        help="astropy ephemeris name or a .bsp path (default de430)",
    )
    args = parser.parse_args()
    compare(args.table, args.ephemeris)


if __name__ == "__main__":
    main()
