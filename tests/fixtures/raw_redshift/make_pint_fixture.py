"""Regenerate pint_zb.txt: PINT's barycentric correction, the regression
reference for rvinstrument data_format: raw_redshift.

PINT (pint-pulsar; Luo et al. 2021) is NOT an EXOZIPPy dependency.  Run this
in a throwaway environment that has it:

    conda create -p <scratch>/tempo-env -c conda-forge pint-pulsar=1.1.8
    <scratch>/tempo-env/bin/python make_pint_fixture.py <path/to/de432s.bsp>

The committed table was made with pint-pulsar 1.1.8, astropy 7 with its
bundled IERS-A/B tables (auto_download off), TT(TAI) with no BIPM
correction, no observatory clock file, PLANET_SHAPIRO Y and the JPL kernel
DE432s (https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
de432s.bsp, 10 MB).  DE432s and not DE440: the test must use the SAME kernel
on the EXOZIPPy side, and DE432s is the one astropy fetches by name in
seconds (DE440 is 114 MB; it differs from DE432s by up to 0.04 mm/s in this
correction).

The method is Wright & Eastman (2014) Appendix A's (docs/
timing_code_crosscheck.md): each (site, star, UTC instant) is a "pulsar" --
the star re-referenced to that day with erfa.pmsafe, F0 = 1 Hz, F1 = 0, DM =
0 -- so the topocentric spin frequency is nu_obs/nu_SSB.  It is read off
PINT's absolute phase at 121 nodes over +/-2 h (a degree-10 polynomial's
derivative at the center), multiplied by 1 + L_B (TDB -> TCB seconds), and
turned into barycorrpy's c*z_B with the star's secular-acceleration factor
(c + rv0)/(c + rv(t_B)), rv(t) from first principles.  The table holds that
c*z_B, which tests/test_raw_redshift.py compares with
redshift.barycentric_redshift.
"""

import io
import sys

import erfa
import numpy as np

C = 299792458.0
L_B = 1.550519768e-8
AU = 149597870700.0
PC = 648000.0 / np.pi * AU
YR = 365.25 * 86400.0
J2016 = 2457389.0
FREQ = 545000000.0
MIN_ALT_DEG = 20.0

# (name, ra, dec, pmra* [mas/yr], pmdec, parallax [mas], rv [m/s], epoch)
STARS = [
    (
        "tauCet",
        26.00930287,
        -15.93379865,
        -1729.726,
        855.493,
        273.8097,
        -16597.0,
        J2016,
    ),
    ("Gaia4like", 209.5063, 31.6955, -75.53, 17.97, 13.63, -20000.0, J2016),
    (
        "Barnard",
        269.44850252,
        4.73942005,
        -801.551,
        10362.394,
        546.9759,
        -110110.0,
        J2016,
    ),
]
# (name, East lon, geodetic lat, height m)
SITES = [
    ("Keck", -155.47833, 19.82833, 4160.0),
    ("McDonald", -104.0147, 30.6814, 2026.0),
    ("LaPalma", -17.8850, 28.7577, 2382.0),
    ("LasCampanas", -70.6833, -29.0486, 2516.0),
]
N_EPOCHS = 6
SEED = 20261104


def site_itrf(site):
    _, lon, lat, h = site
    return np.asarray(erfa.gd2gc(1, np.radians(lon), np.radians(lat), h))


def propagate(star, jd_tdb):
    """erfa.pmsafe rectilinear propagation of the catalog star to jd_tdb."""
    _, ra, dec, pmra, pmdec, px, rv, ep = star
    d = np.radians(dec)
    mas = np.radians(1.0 / 3.6e6)
    out = erfa.pmsafe(
        np.radians(ra),
        d,
        pmra * mas / np.cos(d),
        pmdec * mas,
        px / 1000.0,
        rv / 1000.0,
        ep,
        0.0,
        jd_tdb,
        0.0,
    )
    ra2, dec2, pmr2, pmd2, px2, rv2 = out[:6]
    return (
        np.degrees(ra2) % 360.0,
        np.degrees(dec2),
        pmr2 * np.cos(dec2) / mas,
        pmd2 / mas,
        px2 * 1000.0,
        rv2 * 1000.0,
    )


def altitude_deg(site, ra_deg, dec_deg, jd_utc):
    _, lon, lat, _ = site
    gmst = erfa.gmst06(jd_utc, 0.0, jd_utc, 0.0)
    ha = gmst + np.radians(lon) - np.radians(ra_deg)
    la, de = np.radians(lat), np.radians(dec_deg)
    s = np.sin(la) * np.sin(de) + np.cos(la) * np.cos(de) * np.cos(ha)
    return np.degrees(np.arcsin(s))


def sexa(x, hours):
    v = x / 15.0 if hours else x
    sgn = "-" if v < 0 else ("" if hours else "+")
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    s = ((v - d) * 60 - m) * 60
    if s >= 59.9999999995:
        s, m = 0.0, m + 1
    if m == 60:
        m, d = 0, d + 1
    return "%s%02d:%02d:%013.10f" % (sgn, d, m, s)


def par_text(name, star, mjd_ref):
    ra, dec, pmra, pmdec, px, _ = propagate(star, mjd_ref + 2400000.5)
    lines = [
        ("PSRJ", name),
        ("RAJ", sexa(ra, True)),
        ("DECJ", sexa(dec, False)),
        ("PMRA", "%.9f" % pmra),
        ("PMDEC", "%.9f" % pmdec),
        ("PX", "%.9f" % px),
        ("POSEPOCH", "%.6f" % mjd_ref),
        ("F0", "1.0"),
        ("F1", "0.0"),
        ("PEPOCH", "%.6f" % mjd_ref),
        ("DM", "0.0"),
        ("EPHEM", "DE432S"),
        ("CLK", "TT(TAI)"),
        ("UNITS", "TDB"),
        ("PLANET_SHAPIRO", "Y"),
        ("NE_SW", "0.0"),
    ]
    return "\n".join("%-16s %s" % kv for kv in lines) + "\n"


def pint_freq(model, obs, mjd_utc, nnode=121, half_span_min=120.0, deg=10):
    from astropy.time import Time
    from pint.toa import get_TOAs_array

    x = np.linspace(-1.0, 1.0, nnode) * half_span_min
    day = np.floor(mjd_utc)
    tt = Time(
        np.full(nnode, day),
        (mjd_utc - day) + x / 1440.0,
        format="mjd",
        scale="utc",
    )
    toas = get_TOAs_array(
        tt,
        obs=obs,
        freqs=FREQ * np.ones(nnode),
        ephem="DE432S",
        planets=True,
        include_bipm=False,
        include_gps=False,
        errors=1.0,
    )
    ph = model.phase(toas, abs_phase=False)
    pi = np.asarray(ph.int, dtype=np.longdouble)
    pf = np.asarray(ph.frac, dtype=np.longdouble)
    y = np.asarray((pi - pi[nnode // 2]) + pf, dtype=float) - 60.0 * x
    return 1.0 + np.polynomial.polynomial.polyfit(x, y, deg)[1] / 60.0


def rv_first_principles(star, jd_tdb):
    """The true radial velocity of the light reaching the SSB at jd_tdb."""
    _, ra, dec, pmra, pmdec, px, rv, ep = star
    a, d = np.radians(ra), np.radians(dec)
    r0 = np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    e = np.array([-np.sin(a), np.cos(a), 0.0])
    n = np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])
    mas = np.radians(1 / 3.6e6)
    d0 = PC * 1000.0 / px
    V = d0 * (pmra * e + pmdec * n) * mas / YR * (1 + rv / C) + rv * r0
    ta = (jd_tdb - ep) * 86400.0
    te = ta / (1 + rv / C)
    for _ in range(6):
        te = ta - (np.linalg.norm(d0 * r0 + te * V) - d0) / C
    X = d0 * r0 + te * V
    return X @ V / np.linalg.norm(X)


def main(kernel):
    import pint.logging
    from astropy.utils import iers
    from pint.models import get_model
    from pint.observatory.topo_obs import TopoObs
    from pint.solar_system_ephemerides import load_kernel

    iers.conf.auto_download = False
    pint.logging.setup(level="WARNING")
    load_kernel("de432s", path=kernel)
    rng = np.random.default_rng(SEED)
    lines = [
        "# PINT %s c*z_B [m/s] (make_pint_fixture.py; DE432s, TT(TAI), "
        "PLANET_SHAPIRO Y)" % __import__("pint").__version__,
        "# STAR/SITE lists are make_pint_fixture.py's",
        "# columns: site_index star_index MJD_UTC c*z_B[m/s]",
    ]
    for a, site in enumerate(SITES):
        obs = "zfix" + site[0].lower()
        TopoObs(
            obs,
            itrf_xyz=list(site_itrf(site)),
            clock_file=[],
            apply_gps2utc=False,
            overwrite=True,
            origin="fixture site",
        )
        for b, star in enumerate(STARS):
            days = np.sort(
                rng.choice(np.arange(60218, 60936), N_EPOCHS, replace=False)
            )
            for mjd in days:
                grid = mjd + np.arange(0, 1, 1 / 1440.0)
                ok = grid[
                    altitude_deg(site, star[1], star[2], grid + 2400000.5)
                    > MIN_ALT_DEG
                ]
                if not ok.size:
                    continue
                t = float(rng.choice(ok) + rng.uniform(0, 1 / 1440.0))
                model = get_model(
                    io.StringIO(par_text("J%d" % (1000 + b), star, mjd + 0.5))
                )
                K = pint_freq(model, obs, t) * (1.0 + L_B)
                jd = t + 2400000.5 + 69.184 / 86400.0
                pvb = erfa.epv00(jd, 0.0)[1]["p"]
                ra, dec = np.radians(star[1]), np.radians(star[2])
                r0 = np.array(
                    [
                        np.cos(dec) * np.cos(ra),
                        np.cos(dec) * np.sin(ra),
                        np.sin(dec),
                    ]
                )
                tb = jd + (pvb @ r0) * AU / C / 86400.0
                S = (C + star[6]) / (C + rv_first_principles(star, tb))
                lines.append("%d %d %.10f %.6f" % (a, b, t, C * (K * S - 1)))
            print(site[0], star[0], flush=True)
    with open("pint_zb.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
