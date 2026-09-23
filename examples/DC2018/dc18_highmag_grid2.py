"""(s, q) grid with per-node re-minimization -- the corrected high_mag landscape.

The first attempt gridded (s, q, alpha) with t_0/u_0/t_E FROZEN at the
point-lens fit, and it could not see the true basin at all: the truth ranked
35th-144th among local minima.  Two defects, both in the measurement:

1. It scored "chi2 at truth" using the DC2018 truth ALPHA, which this project
   established over all 44 events is unmappable to our convention (see
   dc18_common.ALPHA_IS_UNMAPPABLE).  Best-fit alphas came out 288/330/316/
   112/356 deg against truth 320/259.5/71.25/29.33/299.33 with no consistent
   offset, so that metric carried no information.
2. Freezing the nuisance parameters costs THOUSANDS of chi2 on these events.
   Event 163's point-lens fit is off by 2% in u_0 and 1.7% in t_E, and that
   alone accounts for ~11565 chi2 over its 38568 points.  For a sharply peaked
   high-magnification light curve the "nuisances" dominate.

Measured afterwards: at the truth (s, q) with alpha scanned and the true
nuisances, chi2 beats the entire frozen grid's global minimum by 896/357/
11565/1720 on events 69/258/163/193.  The basin is deep and findable; the
search just has to re-minimize.

So this grids (log s, log q) only and minimizes (t_0, u_0, t_E, alpha) at every
node from several alpha starts, which is what KMTNet-style grid searches and
Bozza's RTModel actually do.  Every local minimum is retained: the two clean
central-caustic events (69, 258) put their best solution on the s <-> 1/s
MIRROR of the truth, to better than 1.5%, so a search reporting one best fit
would hand back the wrong topology.

Parallelism is over NODES, not events: the nodes are independent, so 5 events
x 2911 nodes spread over all available cores runs in a few hours of wall clock
instead of ~33 h per event.
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402
from dc18_highmag_landscape import (  # noqa: E402
    _chi2_linear_flux,
    _fit_pspl,
    _load,
)

EVENTS = (69, 258, 163, 193, 62)
_CACHE = {}


def _event_data(event, band):
    """Light curve + point-lens fit + rho, cached per worker process."""
    if event not in _CACHE:
        import MulensModel as mm

        t, flux, ferr, mm = _load(event, band)
        truth, _ = dc.load_truth(dc.data_dir_or_raise(None), event)
        pspl, chi2_pspl = _fit_pspl(t, flux, ferr, mm, truth)
        _CACHE[event] = (
            t,
            flux,
            ferr,
            pspl,
            chi2_pspl,
            float(truth["rho"]),
            truth,
        )
    return _CACHE[event]


def _node(task):
    """Minimize (t_0, u_0, t_E, alpha) at one (s, q) node."""
    event, i, j, log_s, log_q, band, n_alpha_starts = task
    try:
        import MulensModel as mm
        from scipy.optimize import minimize

        t, flux, ferr, pspl, chi2_pspl, rho, truth = _event_data(event, band)
        s, q = 10.0**log_s, 10.0**log_q
        fs_lo, fs_hi = pspl["t_0"] - 2.0, pspl["t_0"] + 2.0

        def nll(p):
            t0, u0, tE, a = p
            if tE <= 0:
                return 1e30
            par = {
                "t_0": t0,
                "u_0": u0,
                "t_E": tE,
                "s": s,
                "q": q,
                "alpha": float(a) % 360.0,
                "rho": rho,
            }
            try:
                model = mm.Model(par)
                model.set_magnification_methods([fs_lo, "VBBL", fs_hi])
                return _chi2_linear_flux(
                    flux, ferr, model.get_magnification(t)
                )
            except Exception:
                return 1e30

        best, best_a = np.inf, np.nan
        for a0 in np.linspace(0.0, 360.0, n_alpha_starts, endpoint=False):
            p0 = [pspl["t_0"], pspl["u_0"], pspl["t_E"], a0]
            r = minimize(
                nll,
                p0,
                method="Nelder-Mead",
                options={"maxiter": 600, "xatol": 1e-6, "fatol": 1e-2},
            )
            if r.fun < best:
                best, best_a = float(r.fun), float(r.x[3] % 360.0)
        return (event, i, j, best, best_a)
    except BaseException:
        return (event, i, j, np.inf, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=",".join(str(e) for e in EVENTS))
    ap.add_argument("--band", default="W149")
    ap.add_argument("--n-s", type=int, default=71)
    ap.add_argument("--n-q", type=int, default=41)
    ap.add_argument("--alpha-starts", type=int, default=4)
    ap.add_argument(
        "--ncpu", type=int, default=int(os.environ.get("NSLOTS", 0)) or 8
    )
    ap.add_argument("--out", default="dc18_highmag_grid2.npz")
    args = ap.parse_args()

    events = [int(e) for e in args.events.split(",") if e.strip()]
    # CONTINUOUS through s = 1: events 163 and 193 are resonant, where the
    # central-caustic size formula diverges and s <-> 1/s is invalid.
    log_s = np.linspace(-0.35, 0.35, args.n_s)
    log_q = np.linspace(-5.5, -1.5, args.n_q)

    tasks = [
        (ev, i, j, ls, lq, args.band, args.alpha_starts)
        for ev in events
        for i, ls in enumerate(log_s)
        for j, lq in enumerate(log_q)
    ]
    print(
        f"{len(events)} events x {args.n_s}x{args.n_q} = {len(tasks)} node "
        f"minimizations on {args.ncpu} cores "
        f"({args.alpha_starts} alpha starts each)",
        flush=True,
    )

    chi2 = {ev: np.full((args.n_s, args.n_q), np.inf) for ev in events}
    alpha = {ev: np.full((args.n_s, args.n_q), np.nan) for ev in events}
    done = 0
    with mp.Pool(args.ncpu) as pool:
        for ev, i, j, c, a in pool.imap_unordered(_node, tasks, chunksize=4):
            chi2[ev][i, j] = c
            alpha[ev][i, j] = a
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} nodes", flush=True)

    payload = {"log_s": log_s, "log_q": log_q}
    d = dc.data_dir_or_raise(None)
    for ev in events:
        c = chi2[ev]
        truth, _ = dc.load_truth(d, ev)
        bi, bj = np.unravel_index(np.nanargmin(c), c.shape)
        s_b, q_b = 10.0 ** log_s[bi], 10.0 ** log_q[bj]
        s_t, q_t = float(truth["s"]), float(truth["q"])
        # s and q are convention-independent; alpha is NOT (see the module
        # docstring), so it is recorded but never scored against truth.
        d_direct = abs(np.log10(s_b / s_t))
        d_mirror = abs(np.log10(s_b * s_t))
        print(f"\n===== event {ev} =====")
        print(
            f"    best: s={s_b:.4f} q={q_b:.3g} alpha={alpha[ev][bi, bj]:.1f}"
            f"  chi2={c[bi, bj]:.1f}"
        )
        print(f"    truth: s={s_t:.4f} q={q_t:.3g}")
        print(
            f"    |dlog10 s| direct={d_direct:.4f}  mirror={d_mirror:.4f}"
            f"  -> {'MIRROR' if d_mirror < d_direct else 'direct'}"
        )
        print(f"    |dlog10 q| = {abs(np.log10(q_b / q_t)):.4f}")
        payload[f"ev{ev}_chi2"] = c
        payload[f"ev{ev}_alpha"] = alpha[ev]
        payload[f"ev{ev}_truth"] = np.array([s_t, q_t])
    np.savez_compressed(args.out, **payload)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
