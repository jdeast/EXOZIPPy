"""Is the simulation's released source PHOTOMETRY consistent with its own
quoted source RADIUS and DISTANCE?

supercomputer_queue.txt 7.7.3 closed the GEOMETRIC leg -- 4.6503*Rs/Ds agrees
with rhos*thetaE, and tE*murel/365.25 agrees with thetaE, on all six events --
and left the PHOTOMETRIC leg untested.  That is the only route left by which
the answer key itself could be at fault for our low R_source.

The test, per event, with no fit involved:
  1. Deredden the released source colour with the clump's own (A_W149,
     A_Z087) from event_info.txt.
  2. Read the intrinsic colour off OUR BC grid (the one the fits use) to get
     a source Teff at dwarf gravity.
  3. Predict W149 from the quoted radius and distance: L = R^2 (T/Tsun)^4,
     M_bol = 4.74 - 2.5 log10 L, M_146 = M_bol - BC_146, add the distance
     modulus and A_W149 back.
  4. Compare with the released W149s.  The gap, as a radius: the sim's own
     photometry says the source is 10**(-0.2*gap) times the radius the sim
     also quotes.

If that ratio tracks the R_source deficit our fits report, the deficit is the
answer key's, not ours.
"""

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


def event_info_row(data_dir, number):
    path = os.path.join(data_dir, "event_info.txt")
    for ln in open(path):
        f = ln.split()
        if len(f) > 8 and int(f[1]) == number:
            return float(f[5]), float(f[7])  # A_W149, A_Z087
    raise SystemExit(f"event {number} not in event_info.txt")


def main():
    d = C.data_dir_or_raise(None)
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
        w, z = float(row["W149s"]), float(row["F087s"])
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
