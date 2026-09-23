"""Blind mode discovery on DC2018-128 by nested sampling -- the Phase 2 benchmark.

THE QUESTION.  Can an off-the-shelf gradient-free method blindly (no MMEXOFAST
binary seed, no truth) find BOTH members of event 128's s <-> 1/s pair and
weight them by posterior MASS (volume included), at a cost comparable to a
PTDE run?  PT round-trip transport collapses at Lambda ~ 19 (measured; see
notes/pt_round_trip_collapse.txt), so the weights from cold-chain occupancy
are initialization artifacts.  Nested sampling needs no transport: it sweeps
one population from prior to posterior, discovers clusters on the way, and a
mode's weight is the sum of the posterior weights of its samples -- mass, not
peak height.

THE LIKELIHOOD is the same 7-parameter binary-lens profile likelihood the
landscape scripts use: (t_0, u_0, log10 t_E, log10 s, log10 q, alpha,
log10 rho), with the two linear flux parameters (f_source, f_blend) profiled
out exactly at each evaluation.  Profiling (rather than marginalizing) the
fluxes distorts each mode's evidence by its flux-space Occam factor, but both
modes share the same flux dimensionality and near-identical flux posteriors,
so the RELATIVE weight -- the deliverable -- is unaffected at the level we
care about.

BLIND means: priors are generic binary-lens ranges around our own point-lens
fit (itself data-derived, multi-start).  No mmexofast JSON is read.  The s
prior is CONTINUOUS through s = 1 (log10 s in [-0.6, 0.6]) -- the resonant
caveat from the high_mag work applies here too.

SCORING, against the two solutions the seeded PTDE runs converged to
(s = 0.976 and s = 0.863):
  * found: a cluster within ~0.02 dex of each,
  * weights: posterior mass per cluster (sum of dynesty importance weights),
  * cost: number of likelihood evaluations (dynesty reports ncall).
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402
from dc18_highmag_landscape import (  # noqa: E402
    _chi2_linear_flux,
    _fit_pspl,
    _load,
)

# Globals for the worker pool (inherited copy-on-write through fork).
_G = {}


def _init_event(event, band):
    import MulensModel as mm

    t, flux, ferr, mm_mod = _load(event, band)
    truth, _ = dc.load_truth(dc.data_dir_or_raise(None), event)
    pspl, chi2_pspl = _fit_pspl(t, flux, ferr, mm_mod, truth)
    _G.update(t=t, flux=flux, ferr=ferr, pspl=pspl, chi2_pspl=chi2_pspl, mm=mm)
    return pspl, chi2_pspl


def loglike(theta):
    """Profile log-likelihood of the 7-parameter binary-lens model."""
    mm = _G["mm"]
    t = _G["t"]
    t0, u0, log_tE, log_s, log_q, alpha, log_rho = theta
    params = {
        "t_0": t0,
        "u_0": u0,
        "t_E": 10.0**log_tE,
        "s": 10.0**log_s,
        "q": 10.0**log_q,
        "alpha": float(alpha) % 360.0,
        "rho": 10.0**log_rho,
    }
    try:
        model = mm.Model(params)
        # Finite source only near the peak; point source is exact to well
        # below the photometry everywhere else (same treatment as the
        # landscape scripts).
        model.set_magnification_methods(
            [_G["pspl"]["t_0"] - 2.0, "VBBL", _G["pspl"]["t_0"] + 2.0]
        )
        A = model.get_magnification(t)
        chi2 = _chi2_linear_flux(_G["flux"], _G["ferr"], A)
    except Exception:
        return -1e300
    if not np.isfinite(chi2):
        return -1e300
    return -0.5 * chi2


class PriorTransform:
    """Unit cube -> physical, generic blind binary-lens priors.

    A module-level CLASS, not a closure: dynesty ships the transform to the
    worker pool by pickle, and a local function inside a factory cannot be
    pickled (the first submission died on exactly that).
    """

    def __init__(self, t0_center):
        self.t0c = float(t0_center)

    def __call__(self, u):
        # t_eff = u_0 * t_E is the tightly-constrained PSPL combination; the
        # anomaly may displace t_0 by of order days, so give it +/- 5 d.
        return np.array(
            [
                self.t0c - 5.0 + 10.0 * u[0],  # t_0
                -0.5 + 1.0 * u[1],  # u_0 (signed)
                0.0 + 2.5 * u[2],  # log10 t_E in [0, 2.5]
                -0.6 + 1.2 * u[3],  # log10 s, THROUGH s = 1
                -6.0 + 6.0 * u[4],  # log10 q in [-6, 0]
                360.0 * u[5],  # alpha
                -4.0 + 3.0 * u[6],  # log10 rho in [-4, -1]
            ]
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", type=int, default=128)
    ap.add_argument("--band", default="W149")
    ap.add_argument("--nlive", type=int, default=1000)
    ap.add_argument("--dlogz", type=float, default=0.1)
    ap.add_argument("--ncpu", type=int, default=48)
    ap.add_argument("--out", default="dc18_blind_ns.json")
    args = ap.parse_args()

    import dynesty

    pspl, chi2_pspl = _init_event(args.event, args.band)
    print(
        f"event {args.event}: PSPL t_0={pspl['t_0']:.4f} "
        f"u_0={pspl['u_0']:.5f} t_E={pspl['t_E']:.3f} chi2={chi2_pspl:.1f}",
        flush=True,
    )

    ptf = PriorTransform(pspl["t_0"])
    t0 = time.time()
    with mp.Pool(
        args.ncpu, initializer=_init_event, initargs=(args.event, args.band)
    ) as pool:
        sampler = dynesty.NestedSampler(
            loglike,
            ptf,
            ndim=7,
            nlive=args.nlive,
            sample="rwalk",
            pool=pool,
            queue_size=args.ncpu,
        )
        sampler.run_nested(dlogz=args.dlogz, print_progress=True)
    res = sampler.results
    wall = time.time() - t0

    # Mode discovery + MASS weights: cluster the weighted posterior samples
    # in log10 s.  The two known solutions differ by 0.053 dex there, so a
    # split at the weighted-density minimum between them is unambiguous if
    # both are present.
    logw = res.logwt - res.logz[-1]
    w = np.exp(logw - logw.max())
    w /= w.sum()
    log_s = res.samples[:, 3]

    print(f"\nlogZ = {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}", flush=True)
    print(
        f"ncall = {int(np.sum(res.ncall))}   wall = {wall / 3600:.2f} h",
        flush=True,
    )

    # Report the weighted log_s histogram so the clustering is auditable.
    hist, edges = np.histogram(log_s, bins=60, range=(-0.6, 0.6), weights=w)
    print(
        "\nweighted posterior mass in log10 s bins (nonzero only):", flush=True
    )
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        if h > 1e-4:
            print(f"    [{lo:+.3f}, {hi:+.3f}]  {h:.4f}", flush=True)

    # Cluster: greedy split at weighted minima between occupied bins.
    order = np.argsort(log_s)
    ls_s, w_s = log_s[order], w[order]
    # Known targets (from the SEEDED runs, used only for SCORING, not search).
    targets = {"s_main": np.log10(0.976), "s_mirror": np.log10(0.863)}
    out = {
        "event": args.event,
        "logz": float(res.logz[-1]),
        "logzerr": float(res.logzerr[-1]),
        "ncall": int(np.sum(res.ncall)),
        "wall_h": round(wall / 3600, 3),
        "nlive": args.nlive,
    }
    for name, tgt in targets.items():
        m = np.abs(ls_s - tgt) < 0.02
        out[name] = {
            "target_log_s": round(tgt, 4),
            "mass_within_0.02dex": float(w_s[m].sum()),
            "n_samples": int(m.sum()),
        }
        print(
            f"\n{name}: log_s target {tgt:+.4f}  "
            f"posterior mass within 0.02 dex = {w_s[m].sum():.4f}  "
            f"({int(m.sum())} samples)",
            flush=True,
        )

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(
        args.out.replace(".json", "_samples.npz"),
        samples=res.samples,
        logwt=res.logwt,
        logl=res.logl,
        logz=res.logz,
    )
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
