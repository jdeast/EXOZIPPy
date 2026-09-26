"""The DC2018 source's photometric chain, pushed through OUR BC grid with
the answer key's own numbers -- four measurements that together locate the
R_source deficit (2026-09-23).

  lens-law     The key gives each LENS its Teff, logg, R, D and four
               magnitudes (J, F087, H, W149).  Nothing is free, so the
               residual per band between the key's magnitude and our
               unreddened prediction IS the simulation's band extinction.
               Read off A_J : A_F087 : A_H : A_W149 and compare with what our
               CCM89 R_V = 3.1 grid gives for one av.  On 194 the sim's law
               is GREYER in the NIR: A_W149/A_F087 = 0.51 against our 0.45,
               so for the same colour excess the sim extincts W149 by 1.27x
               ours -- 0.65 mag on the 194 source.

  styp-teff    The key's `styp`/`ltyp` is a spectral-type code (5.x G, 6.x K,
               7.x M, with the decimal as subclass), NOT a population and NOT
               a Teff.  Calibrate it against Teff on the 293 lenses, which
               carry both, then read each SOURCE's Teff off the calibration.
               194's source (styp 6.0) is a 5257 K K dwarf, and the
               class-median lens radius at that code is 0.822 Rsun -- the
               key's R_s exactly, so the key assigns R by type.

  source-solve With the lens-derived band ratios, solve a source's (R,
               extinction scale) from its four magnitudes at the key's own
               distance, for a grid of Teff.  194 comes out at R = 0.77-0.88
               for ANY Teff 3900-5400 K: the key is self-consistent, and the
               four bands cannot pick the Teff -- only the extinction moves.

  line         Our model's TWO-band degeneracy line (F087, W149 only, as the
               fits have): for each Teff, the av that reproduces the observed
               colour, the theta_star that reproduces the W149 flux, Torres's
               main-sequence radius, and hence the distance.  The truth
               (5257 K, R 0.822, D 9980) is NOT on this line -- our chain puts
               5257 K at R 0.70, D 12.1 kpc with theta_star 0.71x truth (the
               grey shortfall) -- and the fits sit at 3300-3700 K, where D is
               3.3-4.8 kpc and R 0.29-0.40, which is what every 194 run
               reports.  What selects that Teff is the priors, measured
               separately by dc18_ds_profile.py.

All magnitudes in the key are AB (dc18_photometric_leg.py); our grid is
Vega; the offsets are the shipped SVO ones for Roman and the SVO zeropoints
for 2MASS (1594 and 1024 Jy).  BC(Av) - BC(0) is the band extinction, so the
grid's Av axis carries the law.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dc18_common as C  # noqa: E402

ROOT = HERE.parent.parent / "src" / "exozippy" / "models" / "NextGen" / "BCs"
AB_MINUS_VEGA = {
    "WFI_F087": 0.4986,
    "WFI_F146": 1.0373,
    "2MASS_J": 2.5 * np.log10(3631 / 1594.0),
    "2MASS_H": 2.5 * np.log10(3631 / 1024.0),
}
KEYCOL = {
    "WFI_F087": "F087",
    "WFI_F146": "W149",
    "2MASS_J": "J",
    "2MASS_H": "H",
}
BANDS = ("2MASS_J", "WFI_F087", "2MASS_H", "WFI_F146")

# Torres+2010 Table 4, as components/torres/physics.py carries them.
_A = (1.5689, 1.3787, 0.4243, 1.139, -0.14250, 0.01969, 0.10100)
_B = (2.4427, 0.6679, 0.1771, 0.705, -0.21415, 0.02306, 0.04173)


def torres(teff, logg, feh):
    x = np.log10(teff) - 4.1
    t = (1.0, x, x * x, x**3, logg**2, logg**3, feh)
    return 10 ** sum(a * b for a, b in zip(_A, t)), 10 ** sum(
        a * b for a, b in zip(_B, t)
    )


def ms_torres(teff, feh):
    """Self-consistent (M, R, logg) fixed point of the two relations."""
    g = 4.6
    for _ in range(50):
        m, r = torres(teff, g, feh)
        g = np.log10(m) - 2 * np.log10(r) + 4.438
    return m, r, g


def load_grid(feh="+0.0"):
    """Per band: (Av axis, one (logTeff, logg) interpolator per Av)."""
    out = {}
    for fam in ("Roman", "2MASS"):
        p = ROOT / fam / f"feh{feh}_afe+0.0.{fam}"
        hdr = next(ln for ln in open(p) if "lgTef" in ln).lstrip("#").split()
        tab = np.genfromtxt(p, comments="#", skip_header=1)
        avs = np.unique(tab[:, 4])
        for b in hdr[6:]:
            if b not in BANDS:
                continue
            j = hdr.index(b)
            fns = []
            for av in avs:
                s = np.isclose(tab[:, 4], av)
                fns.append(
                    LinearNDInterpolator(tab[s][:, [0, 1]], tab[s][:, j])
                )
            out[b] = (avs, fns)
    return out


def bc(grid, band, teff, logg, av):
    avs, fns = grid[band]
    vals = np.array([float(f(np.log10(teff), logg)) for f in fns])
    return float(np.interp(av, avs, vals))


def appmag(grid, band, teff, logg, r, d_pc, av, ab=True):
    """Apparent magnitude: m = Mbol - BC + 5 log10(D) - 5, Mbol from R, T."""
    mbol = 4.74 - 2.5 * np.log10(r**2 * (teff / 5772.0) ** 4)
    m = mbol - bc(grid, band, teff, logg, av) + 5 * np.log10(d_pc) - 5
    return m + (AB_MINUS_VEGA[band] if ab else 0.0)


def master(data_dir):
    d = Path(data_dir)
    cols = np.genfromtxt(
        d / "Answers" / "wfirstColumnNumbers.txt",
        dtype=None,
        encoding="utf-8",
        usecols=[0, 1],
        skip_header=2,
        names=["i", "name"],
    )
    names = [
        f"col{i}" if nm == "|" else nm for i, nm in enumerate(cols["name"])
    ]
    return pd.read_csv(
        d / "Answers" / "master_file.txt",
        names=names,
        usecols=range(len(names)),
        sep=r"\s+",
        skiprows=1,
    )


def event_info(data_dir, ev):
    for ln in open(Path(data_dir) / "event_info.txt"):
        f = ln.split()
        if len(f) > 8 and int(f[1]) == ev:
            return (
                float(f[4]),
                float(f[5]),
                float(f[6]),
                float(f[7]),
                float(f[8]),
            )
    return None


def cmd_lens_law(args, data_dir, grid):
    for ev in args.events:
        row, _ = C.load_master_row(data_dir, ev)
        t, g, r, dl = (
            float(row.lTeff),
            min(float(row.Llogg), 5.0),
            float(row.Rl),
            float(row.Dl) * 1000,
        )
        print(
            f"\n=== event {ev}  LENS Teff={t:.0f} logg={g:.2f} R={r} D={dl:.0f} pc"
        )
        key = {b: float(row[KEYCOL[b] + "l"]) for b in BANDS}
        a_impl = {b: key[b] - appmag(grid, b, t, g, r, dl, 0.0) for b in BANDS}
        print(
            "  implied A_band = key mag (AB) - our unreddened AB prediction:"
        )
        print("    " + "  ".join(f"{b}={a_impl[b]:6.3f}" for b in BANDS))
        print(
            f"  sim's ratios  A_W149/A_F087 = {a_impl['WFI_F146'] / a_impl['WFI_F087']:.3f}"
            f"   A_W149/E(F087-W149) = "
            f"{a_impl['WFI_F146'] / (a_impl['WFI_F087'] - a_impl['WFI_F146']):.3f}"
        )
        avs = np.linspace(0, 20, 401)
        print(
            "  our grid's av that reproduces each band's A (one law -> one number):"
        )
        for b in BANDS:
            a = np.array(
                [bc(grid, b, t, g, 0.0) - bc(grid, b, t, g, av) for av in avs]
            )
            print(f"    {b:9s} av = {np.interp(a_impl[b], a, avs):6.2f}")

        def chi2(av):
            return sum(
                (key[b] - appmag(grid, b, t, g, r, dl, av)) ** 2 for b in BANDS
            )

        c = np.array([chi2(av) for av in avs])
        avb = avs[np.argmin(c)]
        a146 = bc(grid, "WFI_F146", t, g, 0.0) - bc(
            grid, "WFI_F146", t, g, avb
        )
        a087 = bc(grid, "WFI_F087", t, g, 0.0) - bc(
            grid, "WFI_F087", t, g, avb
        )
        print(
            f"  best single av in OUR law: {avb:.2f}  (our A_W149/A_F087 there "
            f"{a146 / a087:.3f}, A_W149/E = {a146 / (a087 - a146):.3f});  "
            "residuals key - model: "
            + "  ".join(
                f"{b}={key[b] - appmag(grid, b, t, g, r, dl, avb):+.3f}"
                for b in BANDS
            )
        )
        info = event_info(data_dir, ev)
        if info:
            print(
                f"  event_info.txt: clump at {info[0]} kpc, A_W149 = {info[1]} +/- "
                f"{info[2]}, A_Z087 = {info[3]} +/- {info[4]}"
            )


def cmd_styp_teff(args, data_dir, grid):
    df = master(data_dir)
    ms = df[df.lcl == 5]
    print(
        "styp/ltyp code vs Teff, from the key's MAIN-SEQUENCE lenses (lcl = 5):"
    )
    print(
        f"{'code bin':>10}{'n':>5}{'Teff med':>10}{'16-84%':>14}{'R med':>8}"
    )
    for lo in np.arange(4.0, 7.9, 0.25):
        g = ms[(ms.ltyp >= lo) & (ms.ltyp < lo + 0.25)]
        if len(g) < 2:
            continue
        q = np.percentile(g.lTeff, [16, 50, 84])
        print(
            f"{lo:5.2f}-{lo + 0.25:4.2f}{len(g):5d}{q[1]:10.0f}"
            f"{q[0]:7.0f}-{q[2]:<6.0f}{g.Rl.median():8.3f}"
        )
    print(
        f"\n{'event':>6}{'styp':>6}{'scl':>5}{'key R_s':>8}{'key D_s':>8}"
        f"{'Teff(code)':>11}{'16-84%':>12}{'lens R at code':>15}"
    )
    for ev in args.events:
        row, _ = C.load_master_row(data_dir, ev)
        g = ms[np.abs(ms.ltyp - float(row.styp)) <= 0.13]
        lo, hi = np.percentile(g.lTeff, [16, 84])
        print(
            f"{ev:>6}{float(row.styp):6.2f}{int(row.scl):5d}{float(row.Rs):8.3f}"
            f"{float(row.Ds) * 1000:8.0f}{np.median(g.lTeff):11.0f}"
            f"{lo:6.0f}-{hi:<5.0f}{g.Rl.median():15.3f}"
        )


def cmd_source_solve(args, data_dir, grid):
    for ev in args.events:
        row, _ = C.load_master_row(data_dir, ev)
        t, g, r, dl = (
            float(row.lTeff),
            min(float(row.Llogg), 5.0),
            float(row.Rl),
            float(row.Dl) * 1000,
        )
        keyl = {b: float(row[KEYCOL[b] + "l"]) for b in BANDS}
        a_lens = {
            b: keyl[b] - appmag(grid, b, t, g, r, dl, 0.0) for b in BANDS
        }
        key = {b: float(row[KEYCOL[b] + "s"]) for b in BANDS}
        ds = float(row.Ds) * 1000
        print(
            f"\n=== event {ev}  SOURCE at the key's D = {ds:.0f} pc; band ratios "
            f"from its lens (f = 1 is the lens's own extinction)"
        )
        print(
            f"{'Teff':>6}{'R best':>8}{'f best':>8}{'A_W149':>8}{'rms':>7}"
            "   residuals J F087 H W149"
        )
        for teff in (3100, 3500, 3900, 4200, 4500, 4800, 5100, 5400):
            lg = 4.6
            m0 = {b: appmag(grid, b, teff, lg, 1.0, ds, 0.0) for b in BANDS}
            best = None
            for f in np.linspace(0.6, 1.6, 201):
                off = np.mean([key[b] - m0[b] - f * a_lens[b] for b in BANDS])
                rr = 10 ** (-off / 5)
                res = np.array(
                    [
                        key[b] - (m0[b] - 5 * np.log10(rr) + f * a_lens[b])
                        for b in BANDS
                    ]
                )
                rms = float(np.sqrt(np.mean(res**2)))
                if best is None or rms < best[0]:
                    best = (rms, f, rr, res)
            rms, f, rr, res = best
            print(
                f"{teff:6d}{rr:8.3f}{f:8.3f}{f * a_lens['WFI_F146']:8.3f}{rms:7.3f}   "
                + " ".join(f"{x:+.3f}" for x in res)
            )
        print(f"  key R_s = {row.Rs}")


def cmd_line(args, data_dir, grid):
    for ev in args.events:
        row, _ = C.load_master_row(data_dir, ev)
        obs = {
            "WFI_F087": float(row.F087s) - AB_MINUS_VEGA["WFI_F087"],
            "WFI_F146": float(row.W149s) - AB_MINUS_VEGA["WFI_F146"],
        }
        target = obs["WFI_F087"] - obs["WFI_F146"]
        theta_true = float(row.Rs) / (float(row.Ds) * 1000)
        print(
            f"\n=== event {ev}  our TWO-band line (Vega): observed F087-W149 = "
            f"{target:.3f}, W149 = {obs['WFI_F146']:.3f}; truth R {row.Rs}, "
            f"D {float(row.Ds) * 1000:.0f}"
        )
        print(
            f"{'Teff':>6}{'feh':>5}{'logg':>6}{'av':>7}{'A_W149':>8}{'R_Torres':>9}"
            f"{'D (pc)':>8}{'D/truth':>8}{'th*/truth':>10}{'M_Torres':>9}"
        )
        avs = np.linspace(0, 20, 801)
        for feh in args.feh:
            for teff in (
                2800,
                3090,
                3300,
                3600,
                4000,
                4500,
                5000,
                5257,
                5500,
                6000,
            ):
                m, r, g = ms_torres(teff, feh)
                g = min(g, 5.0)
                col = np.array(
                    [
                        bc(grid, "WFI_F146", teff, g, av)
                        - bc(grid, "WFI_F087", teff, g, av)
                        for av in avs
                    ]
                )
                if not (col.min() <= target <= col.max()):
                    print(
                        f"{teff:6d}{feh:5.1f}   colour out of the grid's reach"
                    )
                    continue
                av = float(np.interp(target, col, avs))
                bcw = bc(grid, "WFI_F146", teff, g, av)
                # m = Mbol - BC + 5 log10(D) - 5, Mbol = 4.74 - 5 log10(R) - 10 log10(T/5772)
                log_r_over_d = (
                    4.74
                    - 10 * np.log10(teff / 5772.0)
                    - bcw
                    - obs["WFI_F146"]
                    - 5
                ) / 5.0
                theta = 10**log_r_over_d
                d_pc = r / theta
                aw = bc(grid, "WFI_F146", teff, g, 0.0) - bcw
                print(
                    f"{teff:6d}{feh:5.1f}{g:6.2f}{av:7.2f}{aw:8.2f}{r:9.3f}"
                    f"{d_pc:8.0f}{d_pc / (float(row.Ds) * 1000):8.2f}"
                    f"{theta / theta_true:10.2f}{m:9.3f}"
                )
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command", choices=("lens-law", "styp-teff", "source-solve", "line")
    )
    ap.add_argument(
        "--events", type=int, nargs="+", default=[8, 62, 128, 152, 194, 223]
    )
    ap.add_argument("--feh", type=float, nargs="+", default=[0.0, -2.0])
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    data_dir = C.data_dir_or_raise(args.data_dir)
    grid = load_grid()
    {
        "lens-law": cmd_lens_law,
        "styp-teff": cmd_styp_teff,
        "source-solve": cmd_source_solve,
        "line": cmd_line,
    }[args.command](args, data_dir, grid)


if __name__ == "__main__":
    main()
