"""Is the simulation's released source PHOTOMETRY consistent with its own
quoted source RADIUS and DISTANCE?  IT IS -- ONCE THE MAGNITUDE SYSTEM IS
RIGHT, WHICH IS THE POINT OF THIS SCRIPT.

supercomputer_queue.txt 7.7.3 closed the GEOMETRIC leg and left the
PHOTOMETRIC one untested.  The first version of this script tested it in the
wrong system and "found" the answer key 1.8-2.4 mag inconsistent with
itself.  It is not.  THE CHALLENGE'S MAGNITUDES ARE AB AND OUR BC GRID IS
VEGA (its header says so, and the shipped SVO filter files give
MagSys=Vega with ZeroPoint 2294.04 Jy for F087 and 1396.74 Jy for F146,
i.e. AB - Vega = +0.499 and +1.037).

The proof is the challenge's own colour-surface-brightness relation, run
backwards on a star whose size is not in question -- the Sun.  Treated as
Vega it over-predicts the solar angular diameter by 2.815x; treated as AB it
lands within 1.10x across 4000-6000 K, against the relation's own 0.034 dex
scatter.  A 2.8x error in angular size is the same 2.4 mag the "photometric
gap" had, so the gap was the system, not the simulation.

With the conversion applied, the six events' released magnitudes reproduce
their own quoted radii to 0.81-1.28x (median 0.93), and the two events whose
dereddened colour fell off the blue end of the BC grid move onto it.

WHAT THAT MEANS FOR THE FITS, and it is why this matters beyond scoring:
these configs anchor their photometry with `mulensinstrument.zeropoint`
N(22.0, 0.02), which is the simulation's AB zeropoint, while the SED predicts
Vega magnitudes.  Nothing converts between them -- `magsys` exists in the SED
component for its own `filters:` list and these runs have `filters: []`, so
the only photometric anchor in the fit carries no system.  The source
therefore looks 1.04 mag too faint in W149 and 0.54 mag too red, which is the
right sign and roughly the right size for the R_source deficit every event
shows.  The decisive test is a rerun with the zeropoint priors at
22.0 - offset (20.963 W149, 21.501 Z087).

Run with --magsys vega to reproduce the original, wrong, comparison.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dc18_common as C  # noqa: E402

BC = os.path.normpath(
    os.path.join(
        HERE,
        "..",
        "..",
        "src",
        "exozippy",
        "models",
        "NextGen",
        "BCs",
        "Roman",
        "feh+0.0_afe+0.0.Roman",
    )
)
TSUN = 5772.0
MBOL_SUN = 4.74
EVENTS = ("008", "062", "128", "152", "194", "223")
# our fitted source radius / the quoted one, from the completed sweep
FIT_RATIO = {
    "008": 0.738,
    "062": 0.639,
    "128": 0.439,
    "152": 0.374,
    "194": 0.364,
    "223": 0.672,
}


def bc_grid(logg_target=4.5):
    header = next(ln for ln in open(BC) if "lgTef" in ln)
    cols = header.lstrip("#").split()
    tab = np.genfromtxt(BC, comments="#", skip_header=1)
    i146 = 6 + cols[6:].index("WFI_F146")
    i087 = 6 + cols[6:].index("WFI_F087")
    av0 = np.isclose(tab[:, 4], 0.0)
    loggs = np.unique(tab[av0, 1])
    lg = loggs[np.argmin(abs(loggs - logg_target))]
    sel = av0 & np.isclose(tab[:, 1], lg)
    order = np.argsort(tab[sel, 0])
    teff = 10.0 ** tab[sel, 0][order]
    return teff, tab[sel, i146][order], tab[sel, i087][order], lg


def ab_minus_vega(filter_root, names=("Roman/WFI.F087", "Roman/WFI.F146")):
    """AB - Vega per filter, from the shipped SVO zeropoints.

    Read rather than hardcoded: the offset is a property of the filter file
    the rest of the pipeline uses, and a table that drifts from it would be
    exactly the kind of silent mismatch this script exists to catch.
    """
    import re

    out = {}
    for nm in names:
        path = os.path.join(
            filter_root, nm.split("/")[0], nm.split("/")[1] + ".xml"
        )
        text = open(path).read()
        zp = float(
            re.search(r'name="ZeroPoint"[^/]*value="([^"]+)"', text).group(1)
        )
        sysname = re.search(r'name="MagSys"[^/]*value="([^"]+)"', text).group(
            1
        )
        if sysname.strip().lower() != "vega":
            raise SystemExit(f"{nm}: expected a Vega zeropoint, got {sysname}")
        out[nm.split(".")[-1]] = 2.5 * np.log10(3631.0 / zp)
    return out


def event_info_row(data_dir, number):
    path = os.path.join(data_dir, "event_info.txt")
    for ln in open(path):
        f = ln.split()
        if len(f) > 8 and int(f[1]) == number:
            return float(f[5]), float(f[7])  # A_W149, A_Z087
    raise SystemExit(f"event {number} not in event_info.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--magsys",
        default="ab",
        choices=("ab", "vega"),
        help="the system the challenge's magnitudes are in (default ab; "
        "'vega' reproduces the original, wrong, comparison)",
    )
    args = ap.parse_args()
    d = C.data_dir_or_raise(None)
    froot = os.path.normpath(
        os.path.join(HERE, "..", "..", "src", "exozippy", "filters")
    )
    off = (
        ab_minus_vega(froot)
        if args.magsys == "ab"
        else {"F087": 0.0, "F146": 0.0}
    )
    print(
        f"magnitudes treated as {args.magsys.upper()}; applying "
        f"AB-Vega = {off['F087']:+.4f} (Z087), {off['F146']:+.4f} (W149)"
    )
    teff, bc146, bc087, lg = bc_grid()
    # colour on the grid: M_087 - M_146 = BC_146 - BC_087
    colour = bc146 - bc087
    print(
        f"BC grid: NextGen Roman, feh 0.0, Av 0, logg {lg:.1f}, "
        f"Teff {teff.min():.0f}-{teff.max():.0f} K"
    )
    print(
        f"grid colour (Z087-W149)_0 spans {colour.min():+.3f} to {colour.max():+.3f}\n"
    )
    print(
        f"{'ev':>4}{'Rs':>7}{'Ds/kpc':>8}{'W149s':>8}{'F087s':>8}"
        f"{'A_W':>6}{'A_Z':>6}{'(Z-W)0':>8}{'Teff':>7}"
        f"{'W_pred':>8}{'gap':>7}{'R_phot/Rs':>11}{'R_fit/Rs':>10}"
    )
    out = {}
    for ev in EVENTS:
        n = int(ev)
        row, _ = C.load_master_row(d, n)
        a_w, a_z = event_info_row(d, n)
        Rs, Ds = float(row["Rs"]), float(row["Ds"]) * 1000.0
        w = float(row["W149s"]) - off["F146"]
        z = float(row["F087s"]) - off["F087"]
        col0 = (z - a_z) - (w - a_w)
        if not (colour.min() <= col0 <= colour.max()):
            print(f"{ev:>4}  dereddened colour {col0:+.3f} is off the grid")
            continue
        # colour DECREASES with Teff, and np.interp needs an ascending x
        ordc = np.argsort(colour)
        t_src = float(np.interp(col0, colour[ordc], teff[ordc]))
        b146 = float(np.interp(t_src, teff, bc146))
        lum = Rs**2 * (t_src / TSUN) ** 4
        m_bol = MBOL_SUN - 2.5 * np.log10(lum)
        w_pred = (m_bol - b146) + 5.0 * np.log10(Ds / 10.0) + a_w
        gap = w - w_pred
        r_ratio = 10.0 ** (-0.2 * gap)
        out[ev] = (gap, r_ratio)
        print(
            f"{ev:>4}{Rs:7.3f}{Ds / 1000:8.2f}{w:8.3f}{z:8.3f}{a_w:6.2f}"
            f"{a_z:6.2f}{col0:8.3f}{t_src:7.0f}{w_pred:8.3f}{gap:+7.3f}"
            f"{r_ratio:11.3f}{FIT_RATIO[ev]:10.3f}"
        )
    if out:
        g = np.array([v[0] for v in out.values()])
        r = np.array([v[1] for v in out.values()])
        f = np.array([FIT_RATIO[k] for k in out])
        print(
            f"\nmedian gap {np.median(g):+.2f} mag -> the sim's own photometry "
            f"implies {np.median(r):.3f}x its own quoted radius"
        )
        print(f"our fits give a median {np.median(f):.3f}x")
        print(
            f"correlation of R_phot/Rs with R_fit/Rs over {len(r)} events: "
            f"{np.corrcoef(r, f)[0, 1]:+.3f}"
        )


if __name__ == "__main__":
    main()
