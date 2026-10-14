"""(s, q) grid with per-node re-minimization -- the corrected high_mag landscape.

The first attempt gridded (s, q, alpha) with t_0/u_0/t_E FROZEN at the
point-lens fit, and it could not see the true basin at all: the truth ranked
35th-144th among local minima.  Two defects, both in the measurement:

1. It scored "chi2 at truth" using the DC2018 truth ALPHA AS WRITTEN in the
   key, which is not in our convention: it maps by the per-event node-line
   rule dc18_common.key_alpha_to_exozippy (conventions.md C22, found
   2026-10-02; until then the key was wrongly recorded as unmappable).
   Best-fit alphas came out 288/330/316/112/356 deg against raw key values
   320/259.5/71.25/29.33/299.33 with no consistent offset, so that metric
   carried no information.
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
import time
import traceback
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
# Set from --blind in main(); module-level so the worker processes inherit it.
_BLIND = [True]
_FREE_RHO = [True]
# Blind PRIOR range for rho, not a peek.  The default 1e-5..1e-1 spans two
# orders of magnitude above anything real (the DC2018 static truths run
# 5e-4..1.7e-3), and VBBL's finite-source contour integration gets much more
# expensive as rho grows -- measured, the free-rho grid ran at 70 nodes/h
# against the fixed-rho 237, a 3.4x slowdown that is mostly NM wandering into
# large rho.  --rho-lo/--rho-hi narrow it without consulting the key.
_RHO_LO = [1e-5]
_RHO_HI = [1e-1]
# How rho is handled at each node.  "simplex" puts log10 rho in the NM vector
# (measured: 70 nodes/h against 237 with rho fixed -- a 3.4x tax).  "separate"
# is the standard microlensing treatment and the default: rho is LARGELY
# INDEPENDENT of s and q (it sets the smoothing of the caustic crossing, not
# its location or width), so it does not belong in a simplex shared with
# parameters it barely couples to.  The 4-D NM runs with rho held, then a
# short 1-D Brent search moves rho at that optimum, then the 4-D NM is
# re-run once from there.  One line search is ~15 evaluations against the
# NM's ~600, so the tax is a few percent rather than 240%.
_RHO_MODE = ["separate"]
# One-shot latch per WORKER PROCESS (the pool is forked, so this is at most
# one traceback per core, not one per node).  See the catch in _node.
_REPORTED_FAIL = []


def _blind_pspl(event):
    """(t_0, u_0, t_E) from the project's own PEAK FINDER, not the answer key.

    dc18_highmag_landscape._fit_pspl starts its point-lens fit at
    `truth["t_0"]` and `truth["t_E"]`.  That is fine for the landscape
    question it was written for -- "is the binary basin reachable from a
    realistic point-lens start" -- but it is a PEEK, and this grid is now
    being proposed as the BLIND default.  dc18_seed.py produces exactly
    these three numbers with no access to the key, and is what the blind
    pipeline already seeds every fit from.
    """
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(
        here, "events", "%03d" % event, "DC2018_%03d_seed.json" % event
    )
    if not os.path.exists(f):
        raise SystemExit(
            "no peak-finder seed for event %d (%s).  Run dc18_seed.py %d -- "
            "the blind grid must not fall back to the answer key."
            % (event, f, event)
        )
    p = json.load(open(f))["fits"][0]["parameters"]
    return {
        "t_0": float(p["t_0"]),
        "u_0": float(p["u_0"]),
        "t_E": float(p["t_E"]),
    }


def _event_data(event, band):
    """Light curve + point-lens fit + rho, cached per worker process."""
    if event not in _CACHE:
        import MulensModel as mm

        t, flux, ferr, mm = _load(event, band)
        truth, _ = dc.load_truth(dc.data_dir_or_raise(None), event)
        if _BLIND[0]:
            pspl, chi2_pspl = _blind_pspl(event), np.nan
        else:
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
    (
        event,
        i,
        j,
        log_s,
        log_q,
        band,
        n_alpha_starts,
        max_points,
        keep_window,
        optimizer,
    ) = task
    try:
        import MulensModel as mm
        from scipy.optimize import minimize

        t, flux, ferr, pspl, chi2_pspl, rho, truth = _event_data(event, band)
        # Blind starting rho.  In PEEK mode this is the key's value; blind it
        # is the geometric centre of the prior range, which the separate
        # 1-D search then moves.
        rho_box = [
            rho
            if not _FREE_RHO[0]
            else float(np.sqrt(_RHO_LO[0] * _RHO_HI[0]))
        ]
        if max_points and t.size > max_points:
            near = np.abs(t - pspl["t_0"]) <= keep_window
            far = np.where(~near)[0]
            n_far = max(0, max_points - int(near.sum()))
            if n_far < far.size:
                keep = np.zeros(t.size, bool)
                keep[near] = True
                keep[far[np.linspace(0, far.size - 1, n_far).astype(int)]] = (
                    True
                )
                t, flux, ferr = t[keep], flux[keep], ferr[keep]
        s, q = 10.0**log_s, 10.0**log_q
        fs_lo, fs_hi = pspl["t_0"] - 2.0, pspl["t_0"] + 2.0

        # The NM/DE vector carries log10 rho ONLY in simplex mode.  Gating
        # the start vector on _FREE_RHO alone instead (as this did until
        # 2026-10-05) hands nll a 5-vector it unpacks into 4 names, and the
        # ValueError is eaten by the blanket `except BaseException` at the
        # bottom of _node -- so every node returns inf and the whole run
        # "succeeds" in 24 s with NO FINISHED NODES (job 50664420).
        simplex_rho = _FREE_RHO[0] and _RHO_MODE[0] == "simplex"

        def nll(p):
            if simplex_rho:
                t0, u0, tE, a, lrho = p
                rho_use = 10.0**lrho
                if not (_RHO_LO[0] < rho_use < _RHO_HI[0]):
                    return 1e30
            else:
                t0, u0, tE, a = p
                rho_use = rho_box[0]
            if tE <= 0:
                return 1e30
            par = {
                "t_0": t0,
                "u_0": u0,
                "t_E": tE,
                "s": s,
                "q": q,
                "alpha": float(a) % 360.0,
                "rho": rho_use,
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
        best_p = (np.nan, np.nan, np.nan)
        if optimizer == "de":
            from scipy.optimize import differential_evolution

            t0, u0, tE = pspl["t_0"], pspl["u_0"], pspl["t_E"]
            bounds = [
                (t0 - 5.0, t0 + 5.0),
                (-3.0, 3.0),
                (max(0.1, 0.2 * tE), 5.0 * tE),
                (0.0, 360.0),
            ] + ([(-5.0, -1.0)] if simplex_rho else [])
            r = differential_evolution(
                nll, bounds, maxiter=150, tol=1e-6, polish=True, seed=0
            )
            starts = [r]
        else:
            method = "Nelder-Mead" if optimizer == "nm" else "Powell"
            starts = [
                minimize(
                    nll,
                    (
                        [pspl["t_0"], pspl["u_0"], pspl["t_E"], a0, -3.0]
                        if simplex_rho
                        else [pspl["t_0"], pspl["u_0"], pspl["t_E"], a0]
                    ),
                    method=method,
                    options={"maxiter": 600, "xatol": 1e-6, "fatol": 1e-2}
                    if optimizer == "nm"
                    else {"maxiter": 600},
                )
                for a0 in np.linspace(
                    0.0, 360.0, n_alpha_starts, endpoint=False
                )
            ]
        if _FREE_RHO[0] and _RHO_MODE[0] == "separate" and starts:
            # rho OPTIMIZED SEPARATELY at the best 4-D optimum, then the 4-D
            # fit re-run once at the new rho.  Brent on log10 rho inside the
            # blind prior range.
            from scipy.optimize import minimize_scalar

            r0 = min(starts, key=lambda r: r.fun)

            def nll_rho(lr):
                rho_box[0] = 10.0**lr
                return nll(list(r0.x[:4]))

            try:
                rr = minimize_scalar(
                    nll_rho,
                    bounds=(np.log10(_RHO_LO[0]), np.log10(_RHO_HI[0])),
                    method="bounded",
                    options={"xatol": 1e-3, "maxiter": 40},
                )
                rho_box[0] = 10.0**rr.x
                starts = starts + [
                    minimize(
                        nll,
                        list(r0.x[:4]),
                        method="Nelder-Mead",
                        options={
                            "maxiter": 600,
                            "xatol": 1e-6,
                            "fatol": 1e-2,
                        },
                    )
                ]
            except Exception:
                pass
        for r in starts:
            if r.fun < best:
                best, best_a = float(r.fun), float(r.x[3] % 360.0)
                # KEEP the whole fitted vector, not just alpha.  A node is
                # only useful as a SEED if it carries t_0/u_0/t_E too --
                # without them a later seeder has to re-minimize every node
                # it wants to use, which is the same work twice.
                best_p = (float(r.x[0]), float(r.x[1]), float(r.x[2]))
        if not np.isfinite(best):
            return (event, i, j, np.inf, np.nan, np.nan, np.nan, np.nan)
        return (event, i, j, best, best_a) + best_p
    except BaseException:
        # A node is ALLOWED to fail -- VBBL throws on pathological geometry
        # and one inf among thousands is normal, which is why this is a
        # blanket catch.  But a catch this wide also hides a programming
        # error in EVERY node equally well: job 50664420 lost 8 h of queue
        # time to a start-vector/unpack mismatch that presented only as
        # "NO FINISHED NODES".  So the FIRST failure in each worker prints
        # its traceback -- once, to stderr, so a real defect is visible in
        # the log on the first checkpoint instead of at the summary.
        if not _REPORTED_FAIL:
            _REPORTED_FAIL.append(1)
            traceback.print_exc()
            sys.stderr.flush()
        return (event, i, j, np.inf, np.nan, np.nan, np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=",".join(str(e) for e in EVENTS))
    ap.add_argument("--band", default="W149")
    # Grid EXTENT.  The defaults are the original high-magnification study's
    # box, kept so earlier runs reproduce.  They are too narrow for a broad
    # sweep on BLIND grounds alone: log_s +/-0.35 is s in 0.45-2.2, which
    # excludes the wide and close systems a survey routinely detects, and
    # log_q -5.5..-1.5 excludes stellar mass ratios entirely.  (Checked
    # against the DC2018 key after the fact: the narrow box misses the truth
    # of 6 of the 30 static events in s and 2 in q.)
    ap.add_argument("--log-s-lo", type=float, default=-0.35)
    ap.add_argument("--log-s-hi", type=float, default=0.35)
    ap.add_argument("--log-q-lo", type=float, default=-5.5)
    ap.add_argument("--log-q-hi", type=float, default=-1.5)
    ap.add_argument("--n-s", type=int, default=71)
    ap.add_argument("--n-q", type=int, default=41)
    ap.add_argument("--alpha-starts", type=int, default=4)
    ap.add_argument(
        "--max-points",
        type=int,
        default=0,
        help=(
            "thin the light curve to about this many points for the SEARCH "
            "(0 = use all).  Every evaluation currently costs a full ~39,000 "
            "point VBBL magnification, and basin RANKING does not need that "
            "-- characterization does.  Points within --keep-window days of "
            "the point-lens t_0 are always kept, because that is where the "
            "anomaly lives and thinning it would destroy the signal being "
            "searched for."
        ),
    )
    ap.add_argument("--keep-window", type=float, default=5.0)
    ap.add_argument("--rho-lo", type=float, default=1e-5)
    ap.add_argument("--rho-hi", type=float, default=1e-1)
    ap.add_argument(
        "--rho-mode", choices=("separate", "simplex"), default="separate"
    )
    ap.add_argument(
        "--peek",
        action="store_true",
        help=(
            "reproduce the ORIGINAL diagnostic: take the point-lens start "
            "from truth[t_0]/truth[t_E] and FIX rho at the answer key's "
            "value.  That is what this script did when it was a landscape "
            "study, and it is not blind -- rho controls the finite-source "
            "smoothing of the very caustic features being matched.  Default "
            "is blind: start from dc18_seed.py's peak finder and OPTIMIZE "
            "log10 rho as a fifth parameter."
        ),
    )
    ap.add_argument(
        "--optimizer",
        choices=("nm", "powell", "de"),
        default="nm",
        help=(
            "per-node optimizer over (t_0,u_0,t_E,alpha).  2.4.14 names "
            "'Nelder-Mead restarts, CMA-ES, or Powell' as the candidates to "
            "replace the DE seed polish, and that comparison has never been "
            "run at FIXED dimensionality -- the polish works on ~39 "
            "parameters, this works on 4, so a 4-D win proves nothing about "
            "it.  'de' is scipy's differential_evolution, a well-tuned DE; "
            "it is NOT our seed polish (0.003-0.004 acceptance, 400-step "
            "cap), so this tests the ALGORITHM, not that implementation."
        ),
    )
    ap.add_argument(
        "--ncpu", type=int, default=int(os.environ.get("NSLOTS", 0)) or 8
    )
    ap.add_argument("--out", default="dc18_highmag_grid2.npz")
    ap.add_argument(
        "--maxtime",
        type=float,
        default=None,
        help=(
            "seconds; stop accepting results after this and write what is "
            "finished.  Size it UNDER the scheduler's walltime so the .npz "
            "always lands -- run 49432738 did 3000 of 11644 nodes in 24 h, "
            "hit the wall, and lost ALL of it because the only write is at "
            "the end.  Unfinished nodes stay inf/nan and are skipped by the "
            "summary, so a partial grid is still a usable grid."
        ),
    )
    args = ap.parse_args()
    _BLIND[0] = not args.peek
    _FREE_RHO[0] = not args.peek
    _RHO_LO[0], _RHO_HI[0] = args.rho_lo, args.rho_hi
    _RHO_MODE[0] = args.rho_mode
    print(
        "mode: %s"
        % (
            "PEEK (truth t_0/t_E start, rho fixed at the key) -- NOT BLIND"
            if args.peek
            else "BLIND (peak-finder start, log10 rho optimized)"
        ),
        flush=True,
    )

    events = [int(e) for e in args.events.split(",") if e.strip()]
    # CONTINUOUS through s = 1: events 163 and 193 are resonant, where the
    # central-caustic size formula diverges and s <-> 1/s is invalid.
    log_s = np.linspace(args.log_s_lo, args.log_s_hi, args.n_s)
    log_q = np.linspace(args.log_q_lo, args.log_q_hi, args.n_q)

    tasks = [
        (
            ev,
            i,
            j,
            ls,
            lq,
            args.band,
            args.alpha_starts,
            args.max_points,
            args.keep_window,
            args.optimizer,
        )
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
    # t_0/u_0/t_E of the best alpha start at each node, so the grid is
    # directly seedable (see dc18_grid_seed.py).
    tuE = {ev: np.full((args.n_s, args.n_q, 3), np.nan) for ev in events}
    done = 0
    t_start = time.time()
    stopped_early = False
    with mp.Pool(args.ncpu) as pool:
        it = pool.imap_unordered(_node, tasks, chunksize=4)
        for ev, i, j, c, a, t0, u0, tE in it:
            chi2[ev][i, j] = c
            alpha[ev][i, j] = a
            tuE[ev][i, j] = (t0, u0, tE)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} nodes", flush=True)
            if args.maxtime and (time.time() - t_start) > args.maxtime:
                stopped_early = True
                print(
                    f"  MAXTIME {args.maxtime:.0f}s reached at {done}/"
                    f"{len(tasks)} nodes -- terminating the pool and writing "
                    f"the partial grid.",
                    flush=True,
                )
                pool.terminate()
                break

    payload = {"log_s": log_s, "log_q": log_q}
    d = dc.data_dir_or_raise(None)
    for ev in events:
        c = chi2[ev]
        truth, _ = dc.load_truth(d, ev)
        n_fin = int(np.isfinite(c).sum())
        if n_fin == 0:
            # A maxtime stop can leave a whole event untouched; nanargmin on
            # an all-inf array returns index 0 without raising, which would
            # report a confident best-fit at the grid corner.
            print(f"\n===== event {ev} =====")
            print("    NO FINISHED NODES -- skipped (maxtime stop)")
            payload[f"ev{ev}_chi2"] = c
            payload[f"ev{ev}_alpha"] = alpha[ev]
            payload[f"ev{ev}_t0u0tE"] = tuE[ev]
            continue
        bi, bj = np.unravel_index(np.nanargmin(c), c.shape)
        s_b, q_b = 10.0 ** log_s[bi], 10.0 ** log_q[bj]
        s_t, q_t = float(truth["s"]), float(truth["q"])
        # s and q are convention-independent; alpha is NOT (see the module
        # docstring), so it is recorded but never scored against truth.
        d_direct = abs(np.log10(s_b / s_t))
        d_mirror = abs(np.log10(s_b * s_t))
        print(f"\n===== event {ev} =====")
        if n_fin < c.size:
            print(f"    PARTIAL: {n_fin}/{c.size} nodes finished")
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
        payload[f"ev{ev}_t0u0tE"] = tuE[ev]
        payload[f"ev{ev}_truth"] = np.array([s_t, q_t])
    payload["n_done"] = np.array([done])
    payload["n_tasks"] = np.array([len(tasks)])
    payload["complete"] = np.array([not stopped_early])
    np.savez_compressed(args.out, **payload)
    print(
        f"\nwrote {args.out} ({done}/{len(tasks)} nodes"
        f"{', PARTIAL' if stopped_early else ''})",
        flush=True,
    )


if __name__ == "__main__":
    main()
