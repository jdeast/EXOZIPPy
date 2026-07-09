"""
Simulate ground-based astrometry of a nearby G+M binary.

A fully self-consistent synthetic system exercising the 'abs' (2-D absolute)
and 'rel' (separation / position angle) astrometry modes, including a
luminous companion (nonzero photocenter flux fraction) in the absolute data.

System (see simbinary.params.yaml):
  primary   0.80 Msun at 20 pc (parallax 50 mas), pm = (+120, -80) mas/yr
  secondary 0.25 Msun M dwarf, fluxfrac beta = 0.02 in the imaging band
  P = 1095.75 d (3 yr), e = 0.35, omega_* = 55 deg, Omega = 210 deg,
  i = 62 deg, Tp = 2457500

  a_rel   = ((M1+M2) * (P/yr)^2)^(1/3) = 2.115 AU = 105.7 mas
  a_phot  = a_rel * (M2/Mtot - beta)   =            23.0 mas

Outputs (whitespace files read by the astrometryinstrument component):
  SimBinary.Ground.abs.astrom : BJD  ra[deg]  dec[deg]  err_ra[mas]  err_dec[mas]
  SimBinary.Ground.rel.astrom : BJD  sep[mas]  err_sep[mas]  pa[deg]  err_pa[deg]

Note: with no RVs, (Omega, omega) -> (Omega+180, omega+180) is degenerate;
the params file starts the fit at the injected mode.
"""

import numpy as np

RAD2MAS = 180.0 / np.pi * 3600e3

# --- truth ----------------------------------------------------------------
RA0, DEC0 = 150.0, 35.0        # deg, at EPOCH
PLX = 50.0                     # mas
PMRA, PMDEC = 120.0, -80.0     # mas/yr
M1, M2 = 0.80, 0.25            # Msun
BETA = 0.02                    # companion flux fraction (imaging band)
PORB = 1095.75                 # d
ECC = 0.35
OMEGA_STAR = np.radians(55.0)
BIGOMEGA = np.radians(210.0)
INC = np.radians(62.0)
TP = 2457500.0
EPOCH = 2457800.0

ABS_ERR = 1.0                  # mas
SEP_ERR = 0.5                  # mas
PA_ERR = 0.3                   # deg
SEED = 4242

MTOT = M1 + M2
A_AU = (MTOT * (PORB / 365.25) ** 2) ** (1.0 / 3.0)
A_REL_MAS = A_AU * PLX
A_PHOT_MAS = A_REL_MAS * (M2 / MTOT - BETA)


def kepler_E(M, e):
    E = np.mod(M, 2 * np.pi)
    for _ in range(100):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def sky_pos(t, w, a_mas):
    M = 2 * np.pi * (t - TP) / PORB
    E = kepler_E(M, ECC)
    cosf = (np.cos(E) - ECC) / (1 - ECC * np.cos(E))
    sinf = (np.sqrt(1 - ECC**2) * np.sin(E)) / (1 - ECC * np.cos(E))
    r = a_mas * (1 - ECC**2) / (1 + ECC * cosf)
    coswf = np.cos(w) * cosf - np.sin(w) * sinf
    sinwf = np.sin(w) * cosf + np.cos(w) * sinf
    dN = r * (np.cos(BIGOMEGA) * coswf - np.sin(BIGOMEGA) * sinwf * np.cos(INC))
    dE = r * (np.sin(BIGOMEGA) * coswf + np.cos(BIGOMEGA) * sinwf * np.cos(INC))
    return dE, dN


def main():
    import sys
    sys.path.insert(0, ".")
    from exozippy.ephemeris import get_observer_position

    rng = np.random.default_rng(SEED)
    ra_r, dec_r = np.radians(RA0), np.radians(DEC0)

    # --- absolute astrometry (photocenter, ~monthly for 6 yr) -------------
    t_abs = np.sort(rng.uniform(2456700.0, 2458900.0, 70))
    xyz = get_observer_position(t_abs, "earth")
    P_E = xyz[:, 0] * np.sin(ra_r) - xyz[:, 1] * np.cos(ra_r)
    P_N = (xyz[:, 0] * np.cos(ra_r) * np.sin(dec_r)
           + xyz[:, 1] * np.sin(ra_r) * np.sin(dec_r)
           - xyz[:, 2] * np.cos(dec_r))
    dE_o, dN_o = sky_pos(t_abs, OMEGA_STAR, A_PHOT_MAS)
    dt_yr = (t_abs - EPOCH) / 365.25
    dE = PMRA * dt_yr + PLX * P_E + dE_o + rng.normal(0, ABS_ERR, len(t_abs))
    dN = PMDEC * dt_yr + PLX * P_N + dN_o + rng.normal(0, ABS_ERR, len(t_abs))
    ra_obs = RA0 + dE / RAD2MAS / np.cos(dec_r) * 180 / np.pi
    dec_obs = DEC0 + dN / RAD2MAS * 180 / np.pi
    np.savetxt("SimBinary.Ground.abs.astrom",
               np.column_stack([t_abs, ra_obs, dec_obs,
                                np.full_like(t_abs, ABS_ERR),
                                np.full_like(t_abs, ABS_ERR)]),
               header=("Simulated 2-D absolute astrometry "
                       "(see simulate_astrometry.py)\n"
                       "BJD_TDB  ra[deg]  dec[deg]  err_ra*[mas]  err_dec[mas]"),
               fmt="%.6f %.12f %.12f %.3f %.3f")

    # --- relative astrometry (companion wrt primary, ~quarterly) ----------
    t_rel = np.sort(rng.uniform(2456700.0, 2458900.0, 25))
    dE_r, dN_r = sky_pos(t_rel, OMEGA_STAR + np.pi, A_REL_MAS)
    sep = np.hypot(dE_r, dN_r) + rng.normal(0, SEP_ERR, len(t_rel))
    pa = np.degrees(np.arctan2(dE_r, dN_r)) % 360.0 \
        + rng.normal(0, PA_ERR, len(t_rel))
    np.savetxt("SimBinary.Ground.rel.astrom",
               np.column_stack([t_rel, sep, np.full_like(t_rel, SEP_ERR),
                                pa, np.full_like(t_rel, PA_ERR)]),
               header=("Simulated relative astrometry "
                       "(see simulate_astrometry.py)\n"
                       "BJD_TDB  sep[mas]  err_sep[mas]  pa[deg]  err_pa[deg]"),
               fmt="%.6f %.4f %.3f %.5f %.3f")

    print(f"a_rel = {A_REL_MAS:.2f} mas, a_phot = {A_PHOT_MAS:.2f} mas")
    # time of conjunction for the params file
    E_c = 2 * np.arctan2(np.sqrt(1 - ECC) * (1 - np.sin(OMEGA_STAR)),
                         np.sqrt(1 + ECC) * np.cos(OMEGA_STAR))
    M_c = E_c - ECC * np.sin(E_c)
    print(f"tc = {TP + M_c * PORB / (2 * np.pi):.4f}")
    print(f"secosw = {np.sqrt(ECC) * np.cos(OMEGA_STAR):.5f}, "
          f"sesinw = {np.sqrt(ECC) * np.sin(OMEGA_STAR):.5f}, "
          f"cosi = {np.cos(INC):.5f}")
    print(f"M2 = {M2 * 1047.5655:.1f} MJup")


if __name__ == "__main__":
    main()
