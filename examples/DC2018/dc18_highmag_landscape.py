"""Map the chi2 landscape in (s, q, alpha) for the high-magnification events.

MMEXOFAST classifies 6 of the 44 DC2018 planetary events as ``high_mag`` and
then generates NO binary-lens seeds for them -- an explicit "not implemented"
branch -- so they return success with an empty solution list.  Before writing a
solver we measure whether one can work: how many local chi2 minima exist, how
deep the true one is, and what grid spacing lands inside its basin.

Design follows the physics of these events rather than the close/wide ones:

* The ``s`` grid is CONTINUOUS THROUGH s = 1, not two mirrored central-caustic
  branches.  Two of these five events (163, 193) are RESONANT -- inside
  1 - (3/4)q^(1/3) < s < 1 + (3/2)q^(1/3) (Erdl & Schneider 1993; Dominik
  1999), which is still +/-2.5-5% wide even at q = 4e-5 because it scales only
  as q^(1/3).  There the central caustic merges with the planetary ones, the
  4q/(s-1/s)^2 size formula DIVERGES, and the s <-> 1/s degeneracy is broken.
  A grid built as two mirrored branches would straddle exactly the region
  these events live in.
* Every local minimum is retained, not just the best.  The degenerate partner
  of a solution is generally NEAR, not AT, s' = 1/s (the offset degeneracy;
  Zhang, Gaudi & Bloom 2022), so a search that emits only reflection/reciprocal
  pairs misses it.  This is also what KMTNet grid searches and Bozza's RTModel
  do in practice.

t_0/u_0/t_E come from OUR OWN point-lens fit, not from the truth table: the
question is whether the true (s, q, alpha) is findable from a realistic
starting point.  They are held fixed across the grid and only the linear flux
parameters are re-solved at each node, which is the cheap first pass -- a real
search re-minimizes them per node, so the basins measured here are an
UNDERESTIMATE of what a full search would find.  That direction is deliberate:
if the true minimum is findable under this handicap it is certainly findable
without it.
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402

# The five events worth mapping.  227 is excluded: its truth u_0 = 0.543 gives
# A_max ~ 2, it is an ordinary close-topology planetary-caustic event, and it
# reached the high_mag branch only because the PSPL stage estimated u_0 < 0.05.
# That is a classifier/PSPL bug, not a case for this solver.
EVENTS = (69, 258, 163, 193, 62)


def _chi2_linear_flux(data_flux, data_err, magnification):
    """Best chi2 over the linear (f_source, f_blend), solved exactly.

    F = f_s * A + f_b is linear in both, so the optimum is a 2x2 normal-equation
    solve rather than anything iterative -- which is what makes a grid of this
    size affordable.
    """
    w = 1.0 / data_err**2
    a11 = np.sum(w * magnification * magnification)
    a12 = np.sum(w * magnification)
    a22 = np.sum(w)
    b1 = np.sum(w * magnification * data_flux)
    b2 = np.sum(w * data_flux)
    det = a11 * a22 - a12 * a12
    if not np.isfinite(det) or abs(det) < 1e-300:
        return np.inf
    fs = (a22 * b1 - a12 * b2) / det
    fb = (a11 * b2 - a12 * b1) / det
    resid = data_flux - (fs * magnification + fb)
    return float(np.sum(w * resid * resid))


def _load(event, band):
    import MulensModel as mm

    d = dc.data_dir_or_raise(None)
    files = dc.light_curve_files(d, event, (band,))
    path = list(files.values())[0]
    arr = np.genfromtxt(path, comments="#")
    # The DC2018 files are ALREADY IN FLUX -- column 1 of event 69's W149
    # curve runs ~0.34 to ~48, a baseline near 0.4 with a high-mag peak, not
    # magnitudes.  The first version applied 10**(-0.4*m) on top of that and
    # produced a point-lens chi2 of 3.8e8.  Flux is also what makes the
    # per-node (f_source, f_blend) solve exact, and matches the flux
    # likelihood EXOZIPPy uses.
    t, flux, ferr = arr[:, 0], arr[:, 1], arr[:, 2]
    good = np.isfinite(t) & np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
    return t[good], flux[good], ferr[good], mm


def _fit_pspl(t, flux, ferr, mm, truth):
    """Point-lens fit for (t_0, u_0, t_E), started near the truth's peak.

    Deliberately OUR fit and not the truth values: the landscape question is
    whether the binary solution is reachable from a realistic point-lens start.
    """
    from scipy.optimize import minimize

    def nll(p):
        t0, u0, teff = p
        if u0 <= 0 or teff <= 0:
            return 1e30
        tE = teff / u0
        model = mm.Model({"t_0": t0, "u_0": u0, "t_E": tE})
        try:
            A = model.get_magnification(t)
        except Exception:
            return 1e30
        return _chi2_linear_flux(flux, ferr, A)

    # t_eff = u_0 * t_E is the well-constrained combination for a high-mag
    # event; fitting it instead of t_E avoids the u_0-t_E ridge.
    best, best_chi2 = None, np.inf
    for u0_try in (0.3, 0.1, 0.03, 0.01, 0.003):
        p0 = [truth["t_0"], u0_try, u0_try * truth["t_E"]]
        r = minimize(
            nll,
            p0,
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-3},
        )
        if r.fun < best_chi2:
            best, best_chi2 = r.x, r.fun
    t0, u0, teff = best
    return {
        "t_0": float(t0),
        "u_0": float(u0),
        "t_E": float(teff / u0),
    }, float(best_chi2)


def _one_event(args):
    event, n_s, n_q, n_alpha, band = args
    out = {"event": event}
    try:
        import MulensModel as mm

        truth, _ = dc.load_truth(dc.data_dir_or_raise(None), event)
        t, flux, ferr, mm = _load(event, band)
        pspl, chi2_pspl = _fit_pspl(t, flux, ferr, mm, truth)
        out["pspl"] = pspl
        out["chi2_pspl"] = chi2_pspl
        out["truth"] = {
            k: truth[k] for k in ("s", "q", "alpha", "u_0", "t_E", "rho")
        }
        out["n_data"] = int(t.size)

        # CONTINUOUS through s = 1 -- see the module docstring.
        log_s = np.linspace(-0.35, 0.35, n_s)
        log_q = np.linspace(-5.5, -1.5, n_q)
        alpha = np.linspace(0.0, 360.0, n_alpha, endpoint=False)
        chi2 = np.full((n_s, n_q, n_alpha), np.inf)
        # rho is held at TRUTH, and that is the one idealization here:
        # t_0/u_0/t_E come from our own point-lens fit (realistic), but a
        # solver would not know rho.  Finite-source smoothing BROADENS caustic
        # features, so supplying the right rho is the favourable case -- if
        # the true minimum is not findable even here, no solver will find it.
        # It cannot simply be dropped: event 258 crosses a caustic of width
        # 2.6e-3 at u_0 = 2.1e-3, where a point source does not reproduce the
        # data even at the true parameters.
        rho = float(truth["rho"])
        fs_lo, fs_hi = pspl["t_0"] - 2.0, pspl["t_0"] + 2.0
        n_fail, first_err = [0], [None]

        for i, ls in enumerate(log_s):
            for j, lq in enumerate(log_q):
                for k, al in enumerate(alpha):
                    params = dict(pspl)
                    params.update(
                        {
                            "s": 10.0**ls,
                            "q": 10.0**lq,
                            "alpha": float(al),
                            "rho": rho,
                        }
                    )
                    try:
                        model = mm.Model(params)
                        # VBBL is a FINITE-SOURCE method and raises outright
                        # ("impossible to use finite source method for a
                        # point source") unless rho is present -- which is
                        # how the first pilot returned inf at all 1188 of its
                        # nodes, silently, through this except.  Confining it
                        # to a window around the peak is also what makes the
                        # grid affordable: elsewhere the point-source
                        # magnification is right to far better than the
                        # photometry.
                        model.set_magnification_methods([fs_lo, "VBBL", fs_hi])
                        A = model.get_magnification(t)
                        chi2[i, j, k] = _chi2_linear_flux(flux, ferr, A)
                    except Exception as exc:
                        n_fail[0] += 1
                        if first_err[0] is None:
                            first_err[0] = f"{type(exc).__name__}: {exc}"

        out["log_s"] = log_s.tolist()
        out["log_q"] = log_q.tolist()
        out["alpha"] = alpha.tolist()
        out["chi2"] = chi2
        out["rho"] = rho
        out["n_fail"] = n_fail[0]
        out["first_err"] = first_err[0]
        out["status"] = "ok"
    except BaseException as exc:
        out["status"] = f"failed: {type(exc).__name__}: {exc}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=",".join(str(e) for e in EVENTS))
    ap.add_argument("--band", default="W149")
    ap.add_argument("--n-s", type=int, default=71)
    ap.add_argument("--n-q", type=int, default=41)
    ap.add_argument("--n-alpha", type=int, default=36)
    ap.add_argument("--ncpu", type=int, default=5)
    ap.add_argument("--out", default="dc18_highmag_landscape.npz")
    args = ap.parse_args()

    events = [int(e) for e in args.events.split(",") if e.strip()]
    jobs = [(e, args.n_s, args.n_q, args.n_alpha, args.band) for e in events]
    print(
        f"{len(events)} events, grid {args.n_s} x {args.n_q} x "
        f"{args.n_alpha} = {args.n_s * args.n_q * args.n_alpha} nodes each",
        flush=True,
    )

    with mp.Pool(min(args.ncpu, len(jobs))) as pool:
        rows = pool.map(_one_event, jobs)

    payload = {}
    for r in rows:
        ev = r["event"]
        print(f"\n===== event {ev} =====", flush=True)
        if r["status"] != "ok":
            print("   ", r["status"])
            continue
        if r.get("n_fail"):
            print(
                f"    !! {r['n_fail']} nodes FAILED; first: {r['first_err']}"
            )
        chi2 = r["chi2"]
        best = np.unravel_index(np.nanargmin(chi2), chi2.shape)
        ls, lq, al = (
            r["log_s"][best[0]],
            r["log_q"][best[1]],
            r["alpha"][best[2]],
        )
        tr = r["truth"]
        print(
            f"    PSPL:  t_0={r['pspl']['t_0']:.4f} u_0={r['pspl']['u_0']:.5f} "
            f"t_E={r['pspl']['t_E']:.3f}  chi2={r['chi2_pspl']:.1f}"
        )
        print(
            f"    truth: s={tr['s']:.4f} q={tr['q']:.3g} alpha={tr['alpha']:.2f} "
            f"(u_0={tr['u_0']:.5f} t_E={tr['t_E']:.2f})"
        )
        print(
            f"    grid best: s={10**ls:.4f} q={10**lq:.3g} alpha={al:.1f}  "
            f"chi2={chi2[best]:.1f}  (dchi2 vs PSPL = "
            f"{chi2[best] - r['chi2_pspl']:+.1f})"
        )
        print(
            f"    |log10(s_best/s_true)| = {abs(ls - np.log10(tr['s'])):.4f}   "
            f"|log10(q_best/q_true)| = {abs(lq - np.log10(tr['q'])):.4f}"
        )
        for key in ("chi2", "log_s", "log_q", "alpha"):
            payload[f"ev{ev}_{key}"] = np.asarray(r[key])
        payload[f"ev{ev}_pspl"] = np.array(
            [
                r["pspl"]["t_0"],
                r["pspl"]["u_0"],
                r["pspl"]["t_E"],
                r["chi2_pspl"],
            ]
        )
        payload[f"ev{ev}_truth"] = np.array(
            [tr["s"], tr["q"], tr["alpha"], tr["u_0"], tr["t_E"]]
        )

    np.savez_compressed(args.out, **payload)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
