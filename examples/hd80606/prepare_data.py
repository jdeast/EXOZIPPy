"""Rebuild the HD 80606 data files in this directory from their public sources.

    python prepare_data.py

Needs network access (MAST through lightkurve, VizieR through astroquery).
The README in this directory says what each file is, where it came from and
why each cut was made; this script is the exact recipe.

TESS
    The SPOC 2-min PDCSAP light curve of TIC 457134360 in Sector 21, the only
    TESS sector to cover a transit of HD 80606 b (Sector 47 covers neither a
    transit nor a secondary eclipse).  Quality-flagged and NaN cadences are
    dropped (lightkurve's default bitmask), the flux is normalized to its
    median, only the cadences within +/- WINDOW days of the transit are
    kept, and those are renormalized to their out-of-transit median.  A
    fourth column, time minus the transit time, is the detrend column the
    fit uses to remove a linear baseline.

RVs (all converted to m/s)
    ELODIE  Naef et al. (2001), VizieR J/A+A/375/L27/table3, Instr = 1
    ELODIE  Moutou et al. (2009), VizieR J/A+A/498/L5/rv, Inst = ELODIE (later
            epochs; its zero point differs from the 2001 set, so its own file)
    SOPHIE  Hebrard et al. (2010), VizieR J/A+A/516/A95/sophie (a superset of
            Moutou et al. 2009's SOPHIE points)
    HIRES   Rosenthal et al. (2021), VizieR J/ApJS/255/8/table6, CPS = 80606,
            split at the 2004 detector upgrade (Inst k = pre, j = post)
    Each file keeps every published point; where any fall in a transit, a
    companion ``.mask`` flag file (one 0/1 per row) excludes the points taken during a transit, where the
    Rossiter-McLaughlin effect (not modeled here) dominates.
"""

import warnings

import numpy as np

# Pearson et al. (2022): the transit ephemeris and duration used for the cuts.
PERIOD = 111.436765
TC = 2458888.07466
T14 = 11.98 / 24.0
# Half-width of the TESS window around the transit, in days.
WINDOW = 1.0
# Padding added to T14/2 before an RV is called in-transit, in days.
RV_PAD = 1.0 / 24.0


def tess():
    import lightkurve as lk

    search = lk.search_lightcurve(
        "TIC 457134360", mission="TESS", author="SPOC", exptime=120, sector=21
    )
    if len(search) != 1:
        raise RuntimeError(
            f"expected one SPOC S21 product, found {len(search)}"
        )
    lc = search.download().remove_nans().normalize()
    t = np.asarray(lc.time.value, float) + 2457000.0
    f = np.asarray(lc.flux.value, float)
    e = np.asarray(lc.flux_err.value, float)
    keep = np.abs(t - TC) < WINDOW
    t, f, e = t[keep], f[keep], e[keep]
    norm = np.median(f[np.abs(t - TC) > 0.5 * T14])
    f, e = f / norm, e / norm
    out = "n20200213.TESS.TESS.HD80606.S21.0120.SPOC.dat"
    np.savetxt(out, np.column_stack([t, f, e, t - TC]), fmt="%.8f")
    print(f"{out}: {len(t)} cadences")


def in_transit(t):
    phase = ((t - TC + 0.5 * PERIOD) % PERIOD) - 0.5 * PERIOD
    return np.abs(phase) < 0.5 * T14 + RV_PAD


def write_rv(name, t, rv, err):
    order = np.argsort(t)
    t, rv, err = t[order], rv[order], err[order]
    np.savetxt(
        f"{name}.rv", np.column_stack([t, rv, err]), fmt="%.6f %.2f %.2f"
    )
    flags = in_transit(t)
    if flags.any():
        np.savetxt(f"{name}.mask", flags.astype(int), fmt="%d")
    print(f"{name}.rv: {len(t)} points, {flags.sum()} in transit (masked)")


def rvs():
    from astroquery.vizier import Vizier

    viz = Vizier(row_limit=-1, columns=["**"])

    naef = viz.get_catalogs("J/A+A/375/L27/table3")[0]
    s = naef[naef["Instr"] == 1]
    write_rv(
        "HD80606b.ELODIE-2001",
        np.asarray(s["BJD"], float),
        1e3 * np.asarray(s["RVel"], float),
        1e3 * np.asarray(s["e_RVel"], float),
    )

    moutou = viz.get_catalogs("J/A+A/498/L5/rv")[0]
    s = moutou[moutou["Inst"] == "ELODIE"]
    write_rv(
        "HD80606b.ELODIE-2009",
        np.asarray(s["JD"], float),
        1e3 * np.asarray(s["RV"], float),
        1e3 * np.asarray(s["e_RV"], float),
    )

    s = viz.get_catalogs("J/A+A/516/A95/sophie")[0]
    write_rv(
        "HD80606b.SOPHIE",
        np.asarray(s["BJD"], float),
        1e3 * np.asarray(s["RV"], float),
        1e3 * np.asarray(s["e_RV"], float),
    )

    cls = viz.query_constraints(catalog="J/ApJS/255/8/table6", CPS="80606")[0]
    for inst, label in (("k", "HIRES-pre"), ("j", "HIRES-post")):
        s = cls[cls["Inst"] == inst]
        write_rv(
            f"HD80606b.{label}",
            np.asarray(s["BJD"], float),
            np.asarray(s["RVel"], float),
            np.asarray(s["e_RVel"], float),
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tess()
    rvs()
