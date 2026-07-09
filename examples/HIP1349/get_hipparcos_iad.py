"""
Download and reconstruct Hipparcos epoch astrometry for HIP 1349.

HIP 1349 (HD 1273) has an orbital (O-type) solution in the Hipparcos
Double and Multiple Systems Annex (DMSA/O): a 411-day photocenter orbit
with a0 = 19.9 mas.  This is REAL 1-D along-scan epoch astrometry, the
same data structure as Gaia epoch astrometry.

The 1997-consortium Intermediate Astrometric Data (IAD) are fetched from
the ESA Hipparcos catalogue search tool
(https://hipparcos-tools.cosmos.esa.int/cgi-bin/HIPcatalogueSearch.pl?hipiId=1349)
and cached in HIP1349.iad.txt.  Each abscissa record contains the partial
derivatives of the abscissa with respect to the five astrometric
parameters plus the abscissa residual with respect to the adopted DMSA/O
solution:

  IA3 = da/dalpha* (= sin psi), IA4 = da/ddelta (= cos psi),
  IA5 = da/dparallax (along-scan parallax factor),
  IA6 = da/dmu_alpha* (= dt * sin psi), IA7 = da/dmu_delta (= dt * cos psi),
  IA8 = abscissa residual [mas], IA9 = abscissa error [mas],
  IA10 = FAST/NDAC correlation.

The residuals are relative to the 5-parameter reference solution in the
IAD header (they still contain the full orbital signal: chi2/N = 33 raw,
1.1 after subtracting the DMSA/O orbit).  The full along-scan coordinate
relative to the catalog reference position (J1991.25) is reconstructed by
adding the 5-parameter model back to the residuals:

  w = IA8 + plx*IA5 + (pm_ra*dt)*IA3 + (pm_dec*dt)*IA4

The DMSA/O orbital elements (ESA 1997, I/239/hip_dm_o):
  P = 411.449 d, T_p = JD 2448245.6103, a0 = 19.94 mas, e = 0.5671,
  omega = 4.68 deg, i = 80.48 deg, Omega = 352.57 deg
are used only to validate conventions: subtracting our Thiele-Innes
projection of that orbit from the raw residuals must yield chi2/N ~ 1
(it does: 1.11), confirming that the DMSA/O (omega, Omega, i) convention
matches the EXOFASTv2 convention implemented in EXOZIPPy.  The epoch
consistency (IA6/IA3 vs IA7/IA4) and our Earth-ephemeris parallax factors
vs the consortium's IA5 are also validated.

Caveats (this is a demo, not a publication): the FAST and NDAC abscissae
of the same orbit are positively correlated (IA10) but treated as
independent here, and the companion is assumed dark (if it contributes
light in Hp, the fitted companion mass describes the photocenter orbit,
not the true mass).

Output: HIP1349.Hipparcos.astrom with columns
  BJD_TDB  w[mas]  err[mas]  scan_PA[deg]
"""

import os
import re
import urllib.request

import numpy as np

HIP = 1349
IAD_CACHE = "HIP1349.iad.txt"
OUTFILE = "HIP1349.Hipparcos.astrom"

# Catalog reference solution (IAD header, J1991.25)
RA0 = 4.22329974        # deg
DEC0 = -52.65159230     # deg
PLX = 43.45             # mas
PMRA = 314.94           # mas/yr
PMDEC = 182.50          # mas/yr

# DMSA/O photocenter orbit (ESA 1997)
PORB = 411.449          # d
TP = 2448245.6103       # JD
A0 = 19.94              # mas
ECC = 0.5671
OMEGA = np.radians(4.68)
INC = np.radians(80.48)
BIGOMEGA = np.radians(352.57)

EPOCH_JD = 2448349.0625  # J1991.25
RAD2MAS = 180.0 / np.pi * 3600e3


def fetch_iad():
    url = ("https://hipparcos-tools.cosmos.esa.int/cgi-bin/"
           f"HIPcatalogueSearch.pl?hipiId={HIP}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    body = urllib.request.urlopen(req, timeout=60).read().decode(errors="replace")
    with open(IAD_CACHE, "w") as f:
        f.write(body)
    return body


def parse_iad(body):
    rows = []
    for line in body.splitlines():
        m = re.match(r"^\s*(\d+)\|([FN])\|"
                     r"([-\d. ]+)\|([-\d. ]+)\|([-\d. ]+)\|([-\d. ]+)\|"
                     r"([-\d. ]+)\|([-\d. ]+)\|([-\d. ]+)\|", line)
        if m:
            orbit, cons = int(m.group(1)), m.group(2)
            vals = [float(v) for v in m.groups()[2:9]]
            rows.append((orbit, cons, *vals))
    return rows


def kepler_E(M, e):
    E = np.mod(M, 2 * np.pi)
    for _ in range(100):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def orbit_EN(t):
    """Photocenter (dE, dN) in mas from the DMSA/O elements."""
    M = 2 * np.pi * (t - TP) / PORB
    E = kepler_E(M, ECC)
    cosf = (np.cos(E) - ECC) / (1 - ECC * np.cos(E))
    sinf = (np.sqrt(1 - ECC**2) * np.sin(E)) / (1 - ECC * np.cos(E))
    r = A0 * (1 - ECC**2) / (1 + ECC * cosf)
    coswf = np.cos(OMEGA) * cosf - np.sin(OMEGA) * sinf
    sinwf = np.sin(OMEGA) * cosf + np.cos(OMEGA) * sinf
    dN = r * (np.cos(BIGOMEGA) * coswf - np.sin(BIGOMEGA) * sinwf * np.cos(INC))
    dE = r * (np.sin(BIGOMEGA) * coswf + np.cos(BIGOMEGA) * sinwf * np.cos(INC))
    return dE, dN


def main():
    if os.path.exists(IAD_CACHE):
        body = open(IAD_CACHE).read()
    else:
        print("Fetching IAD from ESA...")
        body = fetch_iad()
    rows = parse_iad(body)
    print(f"{len(rows)} abscissa records (FAST + NDAC)")

    orbitno = np.array([r[0] for r in rows])
    sinpsi = np.array([r[2] for r in rows])   # IA3
    cospsi = np.array([r[3] for r in rows])   # IA4
    pfal = np.array([r[4] for r in rows])     # IA5
    dmu_a = np.array([r[5] for r in rows])    # IA6
    dmu_d = np.array([r[6] for r in rows])    # IA7
    resid = np.array([r[7] for r in rows])    # IA8 [mas]
    err = np.array([r[8] for r in rows])      # IA9 [mas]

    # Epoch from the proper-motion partials; use the better-conditioned ratio
    dt_a = np.where(np.abs(sinpsi) > 1e-3, dmu_a / sinpsi, np.nan)
    dt_d = np.where(np.abs(cospsi) > 1e-3, dmu_d / cospsi, np.nan)
    dt_yr = np.where(np.abs(sinpsi) > np.abs(cospsi), dt_a, dt_d)
    both = ~np.isnan(dt_a) & ~np.isnan(dt_d)
    max_incon = np.nanmax(np.abs(dt_a[both] - dt_d[both]))
    print(f"epoch consistency IA6/IA3 vs IA7/IA4: max {max_incon:.4f} yr")
    assert max_incon < 0.02, "inconsistent epochs -- check IAD parsing"

    t = EPOCH_JD + dt_yr * 365.25

    # Validate our parallax factors against the consortium's IA5
    # (Hipparcos is within ~3e-4 AU of the geocenter: 'earth' is fine)
    from exozippy.ephemeris import get_observer_position
    xyz = get_observer_position(t, "earth")
    ra_r, dec_r = np.radians(RA0), np.radians(DEC0)
    P_E = xyz[:, 0] * np.sin(ra_r) - xyz[:, 1] * np.cos(ra_r)
    P_N = (xyz[:, 0] * np.cos(ra_r) * np.sin(dec_r)
           + xyz[:, 1] * np.sin(ra_r) * np.sin(dec_r)
           - xyz[:, 2] * np.cos(dec_r))
    pfal_ours = P_E * sinpsi + P_N * cospsi
    max_diff = np.max(np.abs(pfal_ours - pfal))
    print(f"AL parallax factor: max |ours - IAD| = {max_diff:.4f}")
    if max_diff > 0.02:
        raise SystemExit("Parallax factor mismatch with the IAD -- "
                         "check conventions!")

    # Convention check: the raw residuals (wrt the 5-parameter reference
    # solution) must be explained by the published DMSA/O orbit projected
    # along scan with OUR Thiele-Innes conventions.
    dE_orb, dN_orb = orbit_EN(t)
    orb_al = dE_orb * sinpsi + dN_orb * cospsi
    chi2_raw = np.mean((resid / err) ** 2)
    chi2_orb = np.mean(((resid - orb_al) / err) ** 2)
    print(f"chi2/N of residuals: {chi2_raw:.2f} raw, "
          f"{chi2_orb:.2f} after subtracting the DMSA/O orbit")
    if chi2_orb > 3.0:
        raise SystemExit("DMSA/O orbit does not explain the residuals -- "
                         "check conventions!")

    # Reconstruct the full abscissa relative to the J1991.25 catalog
    # position (5-parameter model + residual; the orbit stays in the data)
    w = (resid
         + PLX * pfal
         + (PMRA * dt_yr) * sinpsi + (PMDEC * dt_yr) * cospsi)

    psi_deg = np.degrees(np.arctan2(sinpsi, cospsi))
    np.savetxt(OUTFILE, np.column_stack([t, w, err, psi_deg]),
               header=("Hipparcos epoch astrometry of HIP 1349, "
                       "reconstructed from the 1997 IAD "
                       "(see get_hipparcos_iad.py)\n"
                       "BJD_TDB  w[mas]  err[mas]  scan_PA[deg]"),
               fmt="%.6f %.4f %.3f %.6f")
    print(f"wrote {OUTFILE}")

    # convenience values for the params file
    E_c = 2 * np.arctan2(np.sqrt(1 - ECC) * (1 - np.sin(OMEGA)),
                         np.sqrt(1 + ECC) * np.cos(OMEGA))
    M_c = E_c - ECC * np.sin(E_c)
    print(f"tc = {TP + M_c * PORB / (2 * np.pi):.4f}")
    print(f"secosw = {np.sqrt(ECC) * np.cos(OMEGA):.5f}, "
          f"sesinw = {np.sqrt(ECC) * np.sin(OMEGA):.5f}, "
          f"cosi = {np.cos(INC):.5f}, bigomega = {np.degrees(BIGOMEGA)}")


if __name__ == "__main__":
    main()
