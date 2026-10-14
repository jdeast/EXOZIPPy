"""(s, q) grid on DC2018 events -- a thin caller of the package mode finder.

THE SEARCH ITSELF MOVED.  Every decision about how a node is minimized, how
basins are picked out of a finished grid and how a grid becomes a fit's
start population now lives in
`exozippy.components.mulensing.gridsearch`, because RMDC26 needs the same
search and a second copy would drift from this one.  Read that module's
docstring for the why; this file is the DC2018 half -- the answer key, the
per-event loop, the mirror scoring and the `ev<N>_*` npz layout that
`dc18_grid_seed.py` consumes.

WHAT THE ORIGINAL ESTABLISHED, kept here because it is what justifies the
search's shape and is not re-derivable from the package module:

  1. Scoring "chi2 at truth" with the DC2018 key's ALPHA AS WRITTEN carries
     no information: it is not in our convention and maps by the per-event
     node-line rule `dc18_common.key_alpha_to_exozippy` (conventions.md
     C22, found 2026-10-02).  Best-fit alphas came out 288/330/316/112/356
     against raw key values 320/259.5/71.25/29.33/299.33, no consistent
     offset.  s and q ARE convention-independent and are scored; alpha is
     recorded and never scored.
  2. Freezing the nuisances costs THOUSANDS of chi2 on these events, which
     is why the package re-minimizes at every node.  Event 163's point-lens
     fit is off 2% in u_0 and 1.7% in t_E -- ~11,565 chi2 over 38,568
     points on its own.
  3. The two clean central-caustic events (69, 258) put their best solution
     on the s <-> 1/s MIRROR of the truth to better than 1.5%, so a search
     reporting one best fit hands back the wrong topology.

REPRODUCING EARLIER RUNS.  Two package defaults are deliberately overridden
below to keep this script's numbers comparable with the grids already on
disk: the VBBL window is a FIXED +/-2 days (the package scales it with t_E,
which is the better default and a different number here), and the grid box
defaults stay the original narrow one.  The narrow box is too narrow on
blind grounds -- log_s +/-0.35 excludes the wide and close systems a survey
routinely detects, and checked against the key afterwards it misses the
truth of 6 of the 30 static events in s and 2 in q -- so pass --log-s-lo/hi
for anything new.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402
from dc18_highmag_landscape import _fit_pspl, _load  # noqa: E402

from exozippy.components.mulensing.gridsearch import (  # noqa: E402
    GridSpec,
    grid_search,
)

EVENTS = (69, 258, 163, 193, 62)

# The original's fixed finite-source window, in days.  See the docstring.
DC18_VBBL_HALF_WINDOW_DAYS = 2.0


def _blind_pspl(event):
    """(t_0, u_0, t_E) from the project's own PEAK FINDER, not the key.

    `dc18_highmag_landscape._fit_pspl` starts at `truth["t_0"]` and
    `truth["t_E"]`, which is fine for the landscape question it was written
    for -- "is the binary basin reachable from a realistic point-lens
    start" -- but it is a PEEK, and this grid is the blind default.
    `dc18_seed.py` produces exactly these three numbers with no access to
    the key, and is what the blind pipeline already seeds every fit from.
    """
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
    return {k: float(p[k]) for k in ("t_0", "u_0", "t_E")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=",".join(str(e) for e in EVENTS))
    ap.add_argument("--band", default="W149")
    ap.add_argument("--log-s-lo", type=float, default=-0.35)
    ap.add_argument("--log-s-hi", type=float, default=0.35)
    ap.add_argument("--log-q-lo", type=float, default=-5.5)
    ap.add_argument("--log-q-hi", type=float, default=-1.5)
    ap.add_argument("--n-s", type=int, default=71)
    ap.add_argument("--n-q", type=int, default=41)
    ap.add_argument("--alpha-starts", type=int, default=4)
    ap.add_argument(
        "--trim-t-E",
        type=float,
        default=None,
        help=(
            "keep only |t - t_0| <= this many t_E (the package default is "
            "10; None here, because the DC2018 curves are one season "
            "already and trimming them changes nothing but the comparison "
            "with the grids on disk)"
        ),
    )
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
            "value.  Not blind -- rho controls the finite-source smoothing "
            "of the very caustic features being matched.  The default is "
            "blind: dc18_seed.py's peak finder, with log10 rho optimized."
        ),
    )
    ap.add_argument(
        "--optimizer", choices=("nm", "powell", "de"), default="nm"
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
            "seconds PER EVENT; stop and keep what finished.  Size it under "
            "the scheduler's walltime so the .npz always lands -- run "
            "49432738 did 3000 of 11644 nodes in 24 h, hit the wall and "
            "lost ALL of it, because the only write is at the end."
        ),
    )
    args = ap.parse_args()

    blind = not args.peek
    print(
        "mode: %s"
        % (
            "BLIND (peak-finder start, log10 rho optimized)"
            if blind
            else "PEEK (truth t_0/t_E start, rho fixed at the key) -- "
            "NOT BLIND"
        ),
        flush=True,
    )

    events = [int(e) for e in args.events.split(",") if e.strip()]
    d = dc.data_dir_or_raise(None)
    payload = {}
    for ev in events:
        t, flux, ferr, _mm = _load(ev, args.band)
        truth, _ = dc.load_truth(d, ev)
        if blind:
            pspl = _blind_pspl(ev)
            rho_fixed = None
        else:
            pspl, _chi2 = _fit_pspl(t, flux, ferr, _mm, truth)
            rho_fixed = float(truth["rho"])

        spec = GridSpec(
            log_s_lo=args.log_s_lo,
            log_s_hi=args.log_s_hi,
            n_s=args.n_s,
            log_q_lo=args.log_q_lo,
            log_q_hi=args.log_q_hi,
            n_q=args.n_q,
            n_alpha_starts=args.alpha_starts,
            trim_t_E=args.trim_t_E,
            rho_mode=args.rho_mode,
            rho_lo=args.rho_lo,
            rho_hi=args.rho_hi,
            rho_fixed=rho_fixed,
            # The original's FIXED window, not the package's t_E-scaled
            # one: see the module docstring.
            vbbl_half_window_t_E=None,
            vbbl_half_window_min=DC18_VBBL_HALF_WINDOW_DAYS,
            optimizer=args.optimizer,
        )
        # One curve: these files are ALREADY IN FLUX (column 1 of event 69's
        # W149 curve runs ~0.34 to ~48 -- a baseline near 0.4 with a
        # high-mag peak, not magnitudes).  Applying 10**(-0.4 m) on top
        # gave a point-lens chi2 of 3.8e8.
        curves = [(t, flux, 1.0 / ferr**2)]
        res = grid_search(
            curves,
            pspl,
            spec,
            ncpu=args.ncpu,
            maxtime=args.maxtime,
            progress_every=500,
        )

        # The npz layout dc18_grid_seed.py reads.  Kept exactly: there are
        # finished grids on disk in this shape, and a seeder that cannot
        # read them is a seeder that throws away the geometry the grid was
        # run to find.
        payload["log_s"] = res.log_s
        payload["log_q"] = res.log_q
        payload["ev%d_chi2" % ev] = res.chi2
        payload["ev%d_alpha" % ev] = res.alpha
        payload["ev%d_t0u0tE" % ev] = np.stack(
            [res.t_0, res.u_0, res.t_E], axis=-1
        )

        print("\n===== event %d =====" % ev)
        n_fin = int(np.isfinite(res.chi2).sum())
        if n_fin == 0:
            # A maxtime stop can leave a whole event untouched, and
            # nanargmin on an all-inf array returns index 0 WITHOUT
            # raising -- which would report a confident best fit at the
            # grid corner.
            print("    NO FINISHED NODES -- skipped (maxtime stop)")
            continue
        if n_fin < res.chi2.size:
            print("    PARTIAL: %d/%d nodes finished" % (n_fin, res.chi2.size))
        best = res.local_minima(k=1)[0]
        s_t, q_t = float(truth["s"]), float(truth["q"])
        d_direct = abs(np.log10(best["s"] / s_t))
        d_mirror = abs(np.log10(best["s"] * s_t))
        print(
            "    best: s=%.4f q=%.3g alpha=%.1f  chi2=%.1f"
            % (best["s"], best["q"], best["alpha"], best["chi2"])
        )
        print("    truth: s=%.4f q=%.3g" % (s_t, q_t))
        print(
            "    |dlog10 s| direct=%.4f  mirror=%.4f  -> %s"
            % (
                d_direct,
                d_mirror,
                "MIRROR" if d_mirror < d_direct else "direct",
            )
        )
        print("    |dlog10 q| = %.4f" % abs(np.log10(best["q"] / q_t)))
        payload["ev%d_truth" % ev] = np.array([s_t, q_t])

    np.savez_compressed(args.out, **payload)
    print("\nwrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
