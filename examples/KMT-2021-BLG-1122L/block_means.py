"""Has the slow drift in the Lens/Source proper motions finished?  Mean and
sd of the T=1 population in 8 consecutive blocks of the trace for the
drifting parameters and what they feed.

    python block_means.py <run dir> <prefix>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARS = [("star.pm_ra", 0), ("star.pm_ra", 3), ("star.pm_dec", 0),
        ("star.pm_dec", 3), ("mulensevent.mu_rel_mag", 0),
        ("mulensevent.mu_rel_geo_mag", 0), ("mulensevent.theta_E", 0),
        ("mulensevent.pi_E_E", 0), ("mulensevent.pi_E_N", 0),
        ("star.distance", 0), ("star.distance", 3), ("star.mass", 0),
        ("source.u_0", 0), ("mulensevent.t_E", 0), ("source.rho", 0)]


def main(run_dir, prefix):
    import arviz as az
    os.chdir(HERE)
    idata = az.from_netcdf(os.path.join(run_dir, f"{prefix}_trace.nc"))
    post = idata.posterior
    lp = idata.sample_stats["lp"].values
    n = lp.shape[1]
    edges = np.linspace(0, n, 9).astype(int)
    labels = [f"{edges[i]}-{edges[i+1]}" for i in range(8)]
    print(f"{'variable':28s} " + " ".join(f"{l:>12s}" for l in labels))
    rows = [("lp", lp)]
    for v, k in VARS:
        a = post[v].values
        rows.append((f"{v}[{k}]", a[:, :, k] if a.ndim == 3 else a))
    for name, a in rows:
        means = [a[:, edges[i]:edges[i+1]].mean() for i in range(8)]
        sds = [a[:, edges[i]:edges[i+1]].std() for i in range(8)]
        print(f"{name:28s} " + " ".join(f"{m:12.5g}" for m in means))
        print(f"{'  sd':28s} " + " ".join(f"{s:12.3g}" for s in sds))
    # within-chain sd vs chain-mean sd of pm_ra in the last quarter
    a = post["star.pm_ra"].values[:, 3 * n // 4:, 0]
    print(f"\nLens pm_ra, last quarter: pooled sd {a.std():.3f}, "
          f"chain-mean sd {a.mean(axis=1).std():.3f}, "
          f"mean within-chain sd {a.std(axis=1).mean():.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
