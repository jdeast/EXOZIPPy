"""chi2 at the EXACT truth (s, q) against a point-lens reference, per event.

Produces the measurement behind events_undetectable.txt: a planet the data
cannot distinguish from a point lens is not a test of any search, so it is
removed from the grid-seeded sweep rather than counted as a failure.

The sweep grid's q step is 0.714 dex -- a factor of five -- so chi2 at the
nearest NODE to the truth is not chi2 at the truth, and cannot be used to
say whether the truth's planet is detectable.  This re-minimizes at the
exact truth (s, q) and at q = 1e-8 (the point-lens limit) at the same s,
both with dc18_highmag_grid2's own node minimizer (rho optimized
separately), and reports dchi2_truth = chi2(point lens) - chi2(truth).

rho/sqrt(q) is NOT a substitute for this (tried 2026-10-11, wrong both
ways): it says the caustic is resolvable IF the source crosses it, not that
the trajectory comes near it.

    python dc18_truth_detect.py --events events.txt --out truth_detect.tsv
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402
import dc18_highmag_grid2 as g  # noqa: E402

# Point-lens limit: small enough that the planetary perturbation is far
# below the photometric noise on every DC2018 event.
_LOG_Q_POINT_LENS = -8.0


def _one(task):
    ev, data_dir = task
    truth, _ = dc.load_truth(data_dir, ev)
    log_s, log_q = np.log10(truth["s"]), np.log10(truth["q"])
    # _node's task tuple: (event, i, j, log_s, log_q, band, n_alpha_starts,
    # max_points, keep_window, optimizer); index 3 of its result is chi2.
    chi2_truth = g._node((ev, 0, 0, log_s, log_q, "W149", 12, 0, 5.0, "nm"))[3]
    chi2_pl = g._node(
        (ev, 0, 0, log_s, _LOG_Q_POINT_LENS, "W149", 12, 0, 5.0, "nm")
    )[3]
    return (
        ev,
        chi2_pl - chi2_truth,
        truth["rho"] / np.sqrt(truth["q"]),
        truth["q"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--events", default="events.txt", help="one event per line"
    )
    ap.add_argument("--out", default="truth_detect.tsv")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="2018DataChallenge tree (default $DC18_DATA)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="dchi2 below which an event is flagged (JDE "
        "2026-10-12: 10 now, 5 once the pipeline works)",
    )
    ap.add_argument(
        "--ncpu",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")),
    )
    args = ap.parse_args()

    g._FREE_RHO[0] = True
    g._RHO_LO[0], g._RHO_HI[0] = 1e-4, 1e-2
    g._RHO_MODE[0] = "separate"

    events = [int(x) for x in Path(args.events).read_text().split()]
    data_dir = dc.data_dir_or_raise(args.data_dir)
    # dc18_highmag_grid2._node takes no data dir: it falls back to
    # dc18_common.DEFAULT_DATA_DIR, which is read from $DC18_DATA at IMPORT
    # time, so neither the flag nor a later os.environ reaches the workers
    # (they then score every node inf and report dchi2 = nan).  Set the
    # module constant before the pool forks.
    dc.DEFAULT_DATA_DIR = str(data_dir)
    tasks = [(ev, data_dir) for ev in events]

    # imap_unordered + flush: every event's result is on disk the moment it
    # exists, so a walltime kill loses only the events still running.
    with open(args.out, "w") as fh, mp.Pool(args.ncpu) as pool:
        fh.write("event\tdchi2_truth\trho_over_sqrtq\tq_true\n")
        fh.flush()
        for ev, dchi2, r, q in pool.imap_unordered(_one, tasks):
            fh.write("%03d\t%.3f\t%.5f\t%.4e\n" % (ev, dchi2, r, q))
            fh.flush()
            if not np.isfinite(dchi2):
                # _node swallows exceptions and scores the node inf.
                flag = "   <- FAILED (non-finite chi2; see worker traceback)"
            elif dchi2 < args.threshold:
                flag = "   <- NOT DETECTABLE"
            else:
                flag = ""
            print(
                "%03d  dchi2=%10.1f  rho/sqrt(q)=%7.3f%s"
                % (ev, dchi2, r, flag),
                flush=True,
            )


if __name__ == "__main__":
    main()
