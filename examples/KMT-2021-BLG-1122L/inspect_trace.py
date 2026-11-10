"""Why did the KMT-2021-BLG-1122L sync fit's max Rhat stall at 1.012-1.016
while min ESS sat at 4000-8000?  Per-parameter split-Rhat and ESS on the
T=1 chains, per-chain means of the worst parameters (a mode split shows up
as a cluster of chain means, a slow nuisance parameter as a smooth spread),
first-half vs second-half chain means (drift), and the key parameters
against Han+2023 Table 2.

    python inspect_trace.py <run dir> <prefix>
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

HAN = {  # Han+2023 Table 2 / Sect. 5 (alpha mapped: alpha_EXZ = -alpha_Han)
    "t_E": (14.74, 0.93), "u_0": (0.126, 0.008), "rho": (2.50e-3, 0.21e-3),
    "s_2": (1.386, 0.027), "q_2": (0.526, 0.039), "s_3": (1.601, 0.043),
    "q_3": (0.241, 0.027), "alpha": (-2.292, 0.013), "psi": (1.383, 0.010),
    "theta_E": (0.50, None), "D_L": (6.57, None), "M_1": (0.47, None),
    "M_2": (0.24, None), "M_3": (0.11, None),
}


def main(run_dir, prefix):
    import arviz as az

    from exozippy.samplers import convergence as cv

    os.chdir(HERE)
    idata = az.from_netcdf(os.path.join(run_dir, f"{prefix}_trace.nc"))
    post = idata.posterior
    groups = idata.groups
    groups = list(groups() if callable(groups) else groups)
    print("groups:", groups)
    print("posterior dims:", dict(post.sizes))
    lp = None
    if "sample_stats" in groups and "lp" in idata.sample_stats:
        lp = idata.sample_stats["lp"].values
    elif "lp" in post:
        lp = post["lp"].values
    if lp is not None:
        mask, reliable = cv.good_chain_mask(lp)
        print(f"good chains: {mask.sum()}/{len(mask)} (reliable={reliable})")
        cm = np.nanmax(lp, axis=1)
        print("chain max lp: min/median/max = "
              f"{cm.min():.1f} / {np.median(cm):.1f} / {cm.max():.1f}")
        print("chain median lp (sorted, lowest 8):",
              np.round(np.sort(np.nanmedian(lp, axis=1))[:8], 1))

    # scalar-ize every variable (flatten extra dims into separate names)
    flat = {}
    for v in post.data_vars:
        arr = post[v].values
        if arr.ndim == 2:
            flat[v] = arr
        else:
            a = arr.reshape(arr.shape[0], arr.shape[1], -1)
            for k in range(a.shape[2]):
                flat[f"{v}[{k}]"] = a[:, :, k]
    flat = {k: a for k, a in flat.items()
            if np.isfinite(a).all() and np.nanstd(a) > 0}
    n_ch, n_dr = next(iter(flat.values())).shape
    print(f"{len(flat)} scalar variables, {n_ch} chains x {n_dr} draws")

    def stats(sub, label):
        ds = az.convert_to_dataset({k: a for k, a in sub.items()})
        rh = az.rhat(ds, method="rank")
        es = az.ess(ds, method="bulk")
        rows = sorted(((float(rh[k].values), float(es[k].values), k)
                       for k in sub), reverse=True)
        print(f"\n== {label}: worst 12 by split-Rhat ==")
        print(f"{'variable':42s} {'Rhat':>7s} {'ESS':>8s}")
        for r, e, k in rows[:12]:
            print(f"{k:42s} {r:7.4f} {e:8.0f}")
        print(f"max Rhat {rows[0][0]:.4f} ({rows[0][2]}); "
              f"min ESS {min(x[1] for x in rows):.0f} "
              f"({min(rows, key=lambda x: x[1])[2]})")
        return rows

    half = n_dr // 2
    tail = {k: a[:, half:] for k, a in flat.items()}
    rows = stats(tail, "last half (what the live check used)")
    stats(flat, "full trace")

    # per-chain anatomy of the three worst
    for r, e, k in rows[:3]:
        a = tail[k]
        m = a.mean(axis=1)
        s = a.std(axis=1)
        pooled = a.std()
        z = (m - m.mean()) / (pooled / np.sqrt(a.shape[1]) + 1e-300)
        print(f"\n-- {k}: Rhat {r:.4f} --")
        print(f"pooled sd {pooled:.4g}; chain-mean sd {m.std():.4g}; "
              f"mean within-chain sd {s.mean():.4g}")
        order = np.argsort(m)
        print("chain means sorted (lowest 6 / highest 6):")
        print("  ", np.round(m[order[:6]], 5), "chains", order[:6])
        print("  ", np.round(m[order[-6:]], 5), "chains", order[-6:])
        # gap test: largest gap between consecutive sorted chain means vs
        # the typical gap -> a cluster of outlier chains
        gaps = np.diff(m[order])
        i = int(np.argmax(gaps))
        print(f"largest gap in chain means {gaps[i]:.4g} at rank {i+1} "
              f"(median gap {np.median(gaps):.4g}) -> "
              f"{i+1} chains below, {n_ch-i-1} above")
        # drift: first vs second half of the FULL trace, per chain
        d = flat[k][:, half:].mean(axis=1) - flat[k][:, :half].mean(axis=1)
        print(f"drift (2nd half - 1st half) over chains: mean {d.mean():.4g}, "
              f"sd {d.std():.4g}, in pooled-sd units {d.mean()/pooled:+.3f}")
        if lp is not None:
            med = np.nanmedian(lp[:, half:], axis=1)
            cc = np.corrcoef(m, med)[0, 1]
            print(f"corr(chain mean, chain median lp) = {cc:+.3f}")

    # key params vs Han
    print("\n== key parameters (median, 16-84) vs Han+2023 ==")
    names = sorted(flat)
    for key in ("t_E", "u_0", "rho", "log_s", "s", "q", "log_q", "alpha",
                "theta_E", "distance", "mass", "logmass", "pi_E", "mu_rel",
                "t_0", "xalpha", "yalpha"):
        for n in names:
            base = n.split(".")[-1]
            if base == key or base.startswith(key + "[") or base.startswith(key + "_"):
                a = flat[n][:, half:].ravel()
                q16, q50, q84 = np.percentile(a, [16, 50, 84])
                print(f"{n:42s} {q50:12.5g} +{q84-q50:.3g} -{q50-q16:.3g}")
    print("\nHan+2023:", HAN)


if __name__ == "__main__":
    main(*sys.argv[1:3])
