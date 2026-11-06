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
-- the file `exozippy-getdata "HIP 1349" --hipparcos-iad` writes, which
this script now calls for the download and reconstruction.
"""

import os

import numpy as np

from exozippy.utilities.getastrometry import (
    check_hipparcos_parallax_factors,
    fetch_hipparcos_iad,
    parse_hipparcos_iad,
    reconstruct_hipparcos_abscissae,
    write_hipparcos_astrom,
)

HIP = 1349
IAD_CACHE = "HIP1349.iad.txt"
OUTFILE = "HIP1349.Hipparcos.astrom"

# DMSA/O photocenter orbit (ESA 1997)
PORB = 411.449  # d
TP = 2448245.6103  # JD
A0 = 19.94  # mas
ECC = 0.5671
OMEGA = np.radians(4.68)
INC = np.radians(80.48)
BIGOMEGA = np.radians(352.57)


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
    dN = r * (
        np.cos(BIGOMEGA) * coswf - np.sin(BIGOMEGA) * sinwf * np.cos(INC)
    )
    dE = r * (
        np.sin(BIGOMEGA) * coswf + np.cos(BIGOMEGA) * sinwf * np.cos(INC)
    )
    return dE, dN


def main():
    # The download, the reconstruction and the epoch / parallax-factor
    # checks are exozippy-getdata --hipparcos-iad's
    # (exozippy.utilities.getastrometry); this script adds the DMSA/O
    # convention check that is specific to this star.
    if os.path.exists(IAD_CACHE):
        body = open(IAD_CACHE).read()
    else:
        print("Fetching IAD from ESA...")
        body = fetch_hipparcos_iad(HIP)
        with open(IAD_CACHE, "w") as f:
            f.write(body)
    header, rec = parse_hipparcos_iad(body)
    print(f"{len(rec['orbit'])} abscissa records (FAST + NDAC)")
    ab = reconstruct_hipparcos_abscissae(header, rec)
    diff = check_hipparcos_parallax_factors(ab)
    print(f"AL parallax factor: max |ours - IAD| = {diff:.4f}")

    # Convention check: the raw residuals (wrt the 5-parameter reference
    # solution) must be explained by the published DMSA/O orbit projected
    # along scan with OUR Thiele-Innes conventions.
    t = ab["time"]
    sinpsi, cospsi, resid, err = rec["IA3"], rec["IA4"], rec["IA8"], rec["IA9"]
    dE_orb, dN_orb = orbit_EN(t)
    orb_al = dE_orb * sinpsi + dN_orb * cospsi
    chi2_raw = np.mean((resid / err) ** 2)
    chi2_orb = np.mean(((resid - orb_al) / err) ** 2)
    print(
        f"chi2/N of residuals: {chi2_raw:.2f} raw, "
        f"{chi2_orb:.2f} after subtracting the DMSA/O orbit"
    )
    if chi2_orb > 3.0:
        raise SystemExit(
            "DMSA/O orbit does not explain the residuals -- check conventions!"
        )

    write_hipparcos_astrom(OUTFILE, ab)
    print(f"wrote {OUTFILE}")

    # convenience values for the params file
    E_c = 2 * np.arctan2(
        np.sqrt(1 - ECC) * (1 - np.sin(OMEGA)),
        np.sqrt(1 + ECC) * np.cos(OMEGA),
    )
    M_c = E_c - ECC * np.sin(E_c)
    print(f"tc = {TP + M_c * PORB / (2 * np.pi):.4f}")
    print(
        f"secosw = {np.sqrt(ECC) * np.cos(OMEGA):.5f}, "
        f"sesinw = {np.sqrt(ECC) * np.sin(OMEGA):.5f}, "
        f"cosi = {np.cos(INC):.5f}, bigomega = {np.degrees(BIGOMEGA)}"
    )


if __name__ == "__main__":
    main()
