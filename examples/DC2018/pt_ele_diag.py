"""Measure tau_V, the energy autocorrelation time of the within-rung kernel.

This is the diagnostic every other PT tuning decision keys on, and we have
never measured it.  DEO's round-trip theory assumes Efficient Local
Exploration: V(X) and V(X') independent across an inter-swap interval (Syed
et al. 2022, assumption A2).  Our samplers do ONE Differential Evolution
proposal per chain between swap sweeps, accepted 20-45% of the time, so V
barely changes -- which is the failure Bittner, Nussbaumer & Janke (PRL 101,
130603, 2008) named, and at the extreme their "complete trapping ... the
replicas do not move from low to high temperatures at all".  Our zero round
trips at 48 rungs is that.

Their prescription is n_local(beta) = tau_can(beta), the energy
autocorrelation time, with the CPU-optimal choice around tau_can/64 to
tau_can/8.  So tau_V per rung is what sets the exploration budget, and it is
also what tells us how biased our Lambda estimate is: per-pair acceptances
measured over correlated energies are not the stationary rejection rates in
Syed's formula.

Measured with swaps DISABLED (swap_interval enormous), so each rung is a pure
DE population sampler at its own temperature and the autocorrelation is the
kernel's own, uncontaminated by exchange.
"""

import argparse
import sys

import numpy as np


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _iact(x, max_lag=None):
    """Integrated autocorrelation time by the initial-positive-sequence rule.

    Sum 1 + 2*sum_k rho_k, truncated at the first non-positive rho -- the
    standard estimator (Geyer 1992); a fixed max_lag would bias tau low
    exactly when it is large, which is the case of interest.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 8 or not np.all(np.isfinite(x)) or x.std() == 0:
        return float("nan")
    max_lag = max_lag or min(n // 4, 5000)
    var = np.dot(x, x) / n
    tau = 1.0
    for k in range(1, max_lag):
        rho = np.dot(x[:-k], x[k:]) / (n - k) / var
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return float(tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=27)
    ap.add_argument("--n-temps", type=int, default=8)
    ap.add_argument("--n-chains", type=int, default=54)
    ap.add_argument("--t-max", type=float, default=8500.0)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--cores", type=int, default=48)
    args = ap.parse_args()

    import pymc as pm

    from exozippy.samplers.ptde_async import ptde_async_sample

    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0, shape=args.dim)

    print(
        f"d={args.dim} n_temps={args.n_temps} n_chains={args.n_chains} "
        f"T_max={args.t_max} draws={args.draws}\n"
        "swaps effectively DISABLED so each rung is a pure DE sampler",
        flush=True,
    )
    idata = ptde_async_sample(
        model,
        _MinimalSystem(),
        draws=args.draws,
        tune=args.tune,
        n_temps=args.n_temps,
        T_max=args.t_max,
        n_chains=args.n_chains,
        cores=args.cores,
        seed=5,
        swap_interval=10**12,
        log_interval=10**9,
        store_hot_chains=True,
        min_ess=None,
        max_rhat=None,
    )

    # T=1 group: lp is the energy proxy V(X) up to sign/constants.
    lp = (
        np.asarray(idata.sample_stats["lp"])
        if "lp" in idata.sample_stats
        else None
    )
    if lp is None:
        print("no lp in sample_stats; cannot measure tau_V", flush=True)
        return 1
    taus = [_iact(lp[c]) for c in range(lp.shape[0])]
    taus = [t for t in taus if np.isfinite(t)]
    print("\n=== tau_V at T=1 (in DE proposals per chain) ===", flush=True)
    print(
        f"    chains={len(taus)}  median={np.median(taus):.1f}  "
        f"min={np.min(taus):.1f}  max={np.max(taus):.1f}",
        flush=True,
    )
    tv = float(np.median(taus))
    print(
        f"\nOur samplers do ONE proposal between swap sweeps.  ELE wants of "
        f"order tau_V = {tv:.0f}.\n"
        f"    Bittner et al. CPU-optimal band tau_V/64..tau_V/8 = "
        f"{tv / 64:.1f}..{tv / 8:.1f} proposals per swap\n"
        f"    full-decorrelation setting  = {tv:.0f} proposals per swap\n"
        f"    we currently use            = 1",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
