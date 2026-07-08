"""
Simulate Gaia along-scan epoch astrometry for Gaia BH1.

The underlying Gaia epoch astrometry of BH1 is not public (it will be in
DR4), so this script generates a realistic stand-in:

  1. Scan times, scan angles, and along-scan parallax factors come from the
     REAL Gaia scanning law, queried from the Gaia Observation Forecast Tool
     (GOST, https://gaia.esac.esa.int/gost/) for BH1's position.  The
     reduced per-FoV-transit forecast is cached in gost_bh1_transits.csv;
     rerun with --refresh-gost to re-query.
  2. The photocenter orbit uses the joint astrometry+RV solution of
     El-Badry et al. (2023), MNRAS 518, 1057 (their Table 1).
  3. Gaussian noise of AL_ERR_MAS per FoV transit (approximately the
     DR3-era per-transit AL precision at G = 13.8).

The output (GaiaBH1.GaiaDR4sim.astrom) contains one row per predicted FoV
transit through the nominal DR4 window:

    BJD_TDB  w_AL[mas]  err[mas]  scan_PA[deg]

where w_AL is the along-scan coordinate of the photocenter relative to the
catalog position (ra_ref, dec_ref at epoch J2016.0 = JD 2457389.0 TCB),
including proper motion, parallax, and orbital motion:

    w_AL = dE * sin(psi) + dN * cos(psi)

As a cross-check, the along-scan parallax factors computed from the Gaia
spacecraft ephemeris (exozippy.ephemeris, JPL Horizons) are compared with
the values reported by GOST; the script aborts if they disagree.
"""

import argparse
import os
import sys
import re

import numpy as np

RAD2MAS = 180.0 / np.pi * 3600e3
DEG2RAD = np.pi / 180.0

# --- Gaia BH1 (El-Badry et al. 2023, Table 1: astrometry + RVs) -----------
RA0 = 262.17120816          # deg (J2016.0)
DEC0 = -0.58109202          # deg
PLX = 2.09                  # mas
PMRA = -7.70                # mas/yr
PMDEC = -25.85              # mas/yr
PORB = 185.59               # d
ECC = 0.451
INC = np.radians(126.6)
OMEGA_STAR = np.radians(12.8)      # argument of periastron (omega_*)
BIGOMEGA = np.radians(97.8)        # PA of ascending node (E of N)
TP = 2457389.0 - 1.1               # periastron time (JD)
A0_MAS = 2.67                      # photocenter semimajor axis
EPOCH = 2457389.0                  # J2016.0 (Gaia DR3 reference epoch)

AL_ERR_MAS = 0.2                   # per-FoV-transit AL uncertainty
DR4_END = 2458868.0                # ~2020-01-20 (66-month DR4 window)
SEED = 20230101

GOST_CSV = "gost_bh1_transits.csv"
OUTFILE = "GaiaBH1.GaiaDR4sim.astrom"


def query_gost():
    """Query GOST for BH1 and reduce per-CCD events to per-FoV transits."""
    import urllib.request
    import urllib.parse
    import http.cookiejar

    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", "Mozilla/5.0")]
    op.open("https://gaia.esac.esa.int/gost/", timeout=60).read()
    params = {
        "ra": f"{RA0}", "dec": f"{DEC0}",
        "service": "1", "inputmode": "single", "srcname": "GaiaBH1",
        "fromdate": "2014-07-25T10:30:00", "todate": "2025-01-15T00:00:00",
        "format": "csv",
    }
    url = ("https://gaia.esac.esa.int/gost/GostServlet?"
           + urllib.parse.urlencode(params))
    xml = op.open(url, timeout=300).read().decode()

    events = []
    for m in re.finditer(
            r"<scanAngle>([-\d.eE+]+)</scanAngle>"
            r"<parallaxFactorAl>([-\d.eE+]+)</parallaxFactorAl>"
            r".*?<eventTcbBarycentricJulianDateAtBarycentre>([-\d.eE+]+)"
            r"</eventTcbBarycentricJulianDateAtBarycentre>", xml, re.S):
        scan, pfal, bjd = map(float, m.groups())
        events.append((bjd, scan, pfal))
    events.sort()

    # Reduce CCD-level events to FoV transits (events within 60 s)
    transits = []
    group = [events[0]]
    for ev in events[1:]:
        if ev[0] - group[-1][0] < 60.0 / 86400.0:
            group.append(ev)
        else:
            transits.append(np.mean(group, axis=0))
            group = [ev]
    transits.append(np.mean(group, axis=0))
    transits = np.array(transits)

    np.savetxt(GOST_CSV, transits,
               header=("GOST forecast for Gaia BH1 (ra=262.17120816, "
                       "dec=-0.58109202), one row per FoV transit\n"
                       "BJD_TDB(barycentre) scanAngle[rad] "
                       "parallaxFactorAlongScan"),
               fmt="%.8f %.10f %.10f")
    return transits


def kepler_E(M, e):
    E = np.mod(M, 2 * np.pi)
    for _ in range(100):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def photocenter_orbit(t):
    """(dE, dN) of the photocenter in mas (Thiele-Innes, EXOFASTv2

    conventions: omega = omega_*, bigomega = PA of ascending node E of N,
    ascending node = where the photocenter recedes from the observer).
    """
    M = 2 * np.pi * (t - TP) / PORB
    E = kepler_E(M, ECC)
    cosf = (np.cos(E) - ECC) / (1 - ECC * np.cos(E))
    sinf = (np.sqrt(1 - ECC**2) * np.sin(E)) / (1 - ECC * np.cos(E))
    r = A0_MAS * (1 - ECC**2) / (1 + ECC * cosf)
    coswf = np.cos(OMEGA_STAR) * cosf - np.sin(OMEGA_STAR) * sinf
    sinwf = np.sin(OMEGA_STAR) * cosf + np.cos(OMEGA_STAR) * sinf
    dN = r * (np.cos(BIGOMEGA) * coswf - np.sin(BIGOMEGA) * sinwf * np.cos(INC))
    dE = r * (np.sin(BIGOMEGA) * coswf + np.cos(BIGOMEGA) * sinwf * np.cos(INC))
    return dE, dN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-gost", action="store_true",
                        help="re-query GOST instead of using the cached CSV")
    args = parser.parse_args()

    if args.refresh_gost or not os.path.exists(GOST_CSV):
        print("Querying GOST (this can take a minute)...")
        transits = query_gost()
    else:
        transits = np.loadtxt(GOST_CSV)
    print(f"{len(transits)} FoV transits from GOST")

    # DR4-like window
    keep = transits[:, 0] < DR4_END
    t, psi, pfal_gost = transits[keep].T
    print(f"{len(t)} transits within the DR4 window")

    # Parallax factors from the Gaia spacecraft ephemeris; validated
    # against GOST's own along-scan parallax factor below.
    from exozippy.ephemeris import get_observer_position
    xyz = get_observer_position(t, "gaia")
    ra_r, dec_r = RA0 * DEG2RAD, DEC0 * DEG2RAD
    P_E = xyz[:, 0] * np.sin(ra_r) - xyz[:, 1] * np.cos(ra_r)
    P_N = (xyz[:, 0] * np.cos(ra_r) * np.sin(dec_r)
           + xyz[:, 1] * np.sin(ra_r) * np.sin(dec_r)
           - xyz[:, 2] * np.cos(dec_r))
    pfal_ours = P_E * np.sin(psi) + P_N * np.cos(psi)
    max_diff = np.max(np.abs(pfal_ours - pfal_gost))
    print(f"AL parallax factor: max |ours - GOST| = {max_diff:.5f}")
    if max_diff > 0.01:
        raise SystemExit("Parallax factor mismatch with GOST -- "
                         "check ephemeris and scan-angle conventions!")

    # Full along-scan model + noise
    dE_orb, dN_orb = photocenter_orbit(t)
    dt_yr = (t - EPOCH) / 365.25
    dE = PMRA * dt_yr + PLX * P_E + dE_orb
    dN = PMDEC * dt_yr + PLX * P_N + dN_orb
    rng = np.random.default_rng(SEED)
    w = dE * np.sin(psi) + dN * np.cos(psi) + rng.normal(0, AL_ERR_MAS, len(t))
    err = np.full(len(t), AL_ERR_MAS)

    np.savetxt(OUTFILE, np.column_stack([t, w, err, psi / DEG2RAD]),
               header=("Simulated Gaia epoch astrometry for Gaia BH1 "
                       "(see simulate_gaia_astrometry.py)\n"
                       "Scan times/angles: real Gaia scanning law via GOST; "
                       "orbit: El-Badry et al. (2023) joint solution\n"
                       "BJD_TDB  w_AL[mas]  err[mas]  scan_PA[deg]"),
               fmt="%.8f %.6f %.3f %.6f")
    print(f"wrote {OUTFILE}")

    # convenience: the time of conjunction implied by TP (for the params file)
    f_c = np.pi / 2 - OMEGA_STAR
    E_c = 2 * np.arctan2(np.sqrt(1 - ECC) * (1 - np.sin(OMEGA_STAR)),
                         np.sqrt(1 + ECC) * np.cos(OMEGA_STAR))
    M_c = E_c - ECC * np.sin(E_c)
    tc = TP + M_c * PORB / (2 * np.pi)
    print(f"tc (time of conjunction) = {tc:.4f}")


if __name__ == "__main__":
    main()
