"""Stage 1: the blind (s, q) grid -- reuse the shipped one, or compute it.

    python 01_grid.py                      # reuse grid/sqgrid30_069.npz
    python 01_grid.py --compute --ncpu 64  # from scratch (sbatch it)

Writes $DC069_RUNS/grid/DC2018_069_grid.npz (the package GridResult layout
whichever path made it) and DC2018_069_chi2.png, the chi2 surface with the
local minima and their s <-> 1/s mirrors marked.

REUSE.  The shipped npz is the 30-event sweep's grid for this event
(dc18_rerun/grid30/static30.sbatch): W149 only, the MMEXOFAST flux copy of
the data, 21 x 8 nodes over log s in [-0.7, 0.7] and log q in [-6, -1], 12
alpha starts per node, rho optimized in [1e-4, 1e-2] outside the simplex
and a fixed +/-2 d finite-source window.  It cost 42.7 core-h (16 cores x
2.67 h, job 52010555_6).

COMPUTE.  The same box and node count through the PACKAGE path
(gridsearch.curves_from_system -> pspl_start -> grid_search), on both bands
of the full data release, read exactly the way the fit reads them (a
prepared 2L1S System).  Two differences from the shipped grid, both the
package's better defaults: both bands rather than W149, and the
finite-source window scales with t_E.  Its chi2 values are therefore not
the shipped grid's numbers; the basins are what matters.  Budget ~45 core-h
on the 168-node box (about 45 min at 64 cores; the per-node cost grows with
the epoch count, so both bands cost more than W149 alone); the package
default 141 x 41 box is ~35x that.  It is blind: the PSPL start is the
built-in peak finder's and nothing reads the answer key.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind as B  # noqa: E402

B.dc18_path()

# The 30-event sweep's box (static30.sbatch): validated, see that file.
BOX = dict(
    log_s_lo=-0.7, log_s_hi=0.7, n_s=21,
    log_q_lo=-6.0, log_q_hi=-1.0, n_q=8,
    n_alpha_starts=12, rho_mode="separate", rho_lo=1e-4, rho_hi=1e-2,
)  # fmt: skip


def compute(ncpu, maxtime, data_dir):
    import dc18_arch_config as A
    import yaml

    from exozippy.components.mulensing.gridsearch import (
        GridSpec,
        curves_from_system,
        grid_search,
        pspl_start,
    )
    from exozippy.system import System

    # A seedless 2L1S config, only to read the data the way the fit does.
    out = B.run_dir("grid", "cfg")
    ev = A.Event(data_dir, B.EVENT)
    run = dict(n_temps="auto", tune=1, draws=1, maxtime=60, cores=1,
               t_max=200.0, hogg=False)  # fmt: skip
    cfg_path = A.build(ev, "2L1S", str(out), run)
    cfg = yaml.safe_load(open(cfg_path))
    params = yaml.safe_load(open(cfg["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    os.chdir(out)
    system = System(cfg, user_params=params)
    system.prepare()
    curves = curves_from_system(system)
    pspl = pspl_start(curves)
    print("PSPL start (peak finder): %s" % pspl, flush=True)
    return grid_search(
        curves, pspl, GridSpec(**BOX), ncpu=ncpu, maxtime=maxtime,
        progress_every=20,
    )  # fmt: skip


def plot(grid, cands, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = grid.chi2 - np.nanmin(grid.chi2[np.isfinite(grid.chi2)])
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=110)
    ds, dq = grid.d_log_s, grid.d_log_q
    im = ax.imshow(
        np.log10(1.0 + d.T),
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        extent=[grid.log_s[0] - ds / 2, grid.log_s[-1] + ds / 2,
                grid.log_q[0] - dq / 2, grid.log_q[-1] + dq / 2],
    )  # fmt: skip
    fig.colorbar(im, ax=ax, label=r"log$_{10}$(1 + $\Delta\chi^2$)")
    for c in cands:
        mk = "o" if c["origin"].startswith("minimum") else "s"
        ax.plot(c["log_s"], c["log_q"], mk, mfc="none", mec="r", ms=9)
        ax.annotate(
            c["mode"],
            (c["log_s"], c["log_q"]),
            color="r",
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0.0, color="w", lw=0.6, ls=":")
    ax.set_xlabel("log s")
    ax.set_ylabel("log q")
    ax.set_title("%s blind (s, q) grid: candidate modes" % B.NAME, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    print("wrote %s" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--grid", default=str(B.SHIPPED_GRID),
                    help="an ev<N>_* npz (dc18_highmag_grid2 layout) or a "
                    "GridResult npz to reuse")  # fmt: skip
    ap.add_argument("--ncpu", type=int, default=None)
    ap.add_argument("--maxtime", type=float, default=None,
                    help="seconds; keep under the job wall (a partial grid "
                    "is a uniform subsample and still usable)")  # fmt: skip
    ap.add_argument("--data-dir", default=B.DEFAULT_DATA)
    ap.add_argument("--png", default=None)
    a = ap.parse_args(argv)

    from exozippy.components.mulensing.gridsearch import GridResult

    if a.compute:
        grid = compute(a.ncpu, a.maxtime, a.data_dir)
    elif "ev%d_chi2" % B.EVENT in np.load(a.grid):
        from dc18_grid_seed import load_event_grid

        grid = load_event_grid(a.grid, B.EVENT)
    else:
        grid = GridResult.from_npz(a.grid)

    out = B.run_dir("grid")
    out.mkdir(parents=True, exist_ok=True)
    npz = out / ("%s_grid.npz" % B.NAME)
    grid.to_npz(str(npz))
    print("grid %d/%d nodes -> %s" % (grid.n_done, grid.n_tasks, npz))
    cands = B.candidate_modes(grid)
    plot(grid, cands, a.png or str(out / ("%s_chi2.png" % B.NAME)))


if __name__ == "__main__":
    main()
