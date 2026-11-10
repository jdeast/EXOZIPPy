"""Detailed comparison of the KMT-2021-BLG-1122L sync posterior with
Han+2023 Table 2: z-scores, width ratios, and the diagnostics that tell a
genuinely narrow posterior from an ensemble that never diffused out of its
start -- distance moved from the start, within-chain vs pooled spread,
stationarity of the width across the post-burn-in trace, autocorrelation,
and the posterior width against the Laplace (curvature) width at the
optimum from the hot-search ledger stamped on the trace.

    python compare_han.py <run dir> <prefix> <burnin draws>
"""

import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

DEG = 180.0 / np.pi
# Han+2023 Table 2 (value, sigma) in OUR coordinates: alpha_EXZ = -alpha_Han
# (deg), alpha_3 = alpha_2 - psi, psi in deg.  t_0 and u_0 are in Han's
# origin (the M1-M2 effective position), ours in the primary's, so their
# rows are a different-origin comparison and are flagged.
HAN = {
    "t_0 (HJD')": (9370.226, 0.061),
    "u_0": (0.126, 0.008),
    "t_E [d]": (14.74, 0.93),
    "rho [1e-3]": (2.50, 0.21),
    "s_2": (1.386, 0.027),
    "q_2": (0.526, 0.039),
    "alpha_2 [deg]": (-2.292 * DEG, 0.013 * DEG),
    "s_3": (1.601, 0.043),
    "q_3": (0.241, 0.027),
    "psi [deg]": (1.383 * DEG, 0.010 * DEG),
    # Sect. 5 physical values; the README quotes no errors for them
    "theta_E [mas]": (0.50, None),
    "mu_rel [mas/yr]": (12.4, None),
    "D_L [kpc]": (6.57, None),
    "M_1 [Msun]": (0.47, None),
    "M_2 [Msun]": (0.24, None),
    "M_3 [Msun]": (0.11, None),
}


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def main(run_dir, prefix, burnin):
    import arviz as az

    from exozippy.outputs.ledger import hot_search_from_json
    from exozippy.trace_meta import HOT_SEARCH_ATTR

    os.chdir(HERE)
    burnin = int(burnin)
    idata = az.from_netcdf(os.path.join(run_dir, f"{prefix}_trace.nc"))
    idata.load()
    idata.close()
    post = idata.posterior
    n_dr = post.sizes["draw"]
    print(f"trace: {post.sizes['chain']} chains x {n_dr} draws; "
          f"using draws {burnin}..{n_dr} ({n_dr - burnin} per chain)")

    def v(name, k=None):
        a = post[name].values
        a = a if k is None else a[:, :, k]
        return a[:, burnin:]

    # our quantities in Han's units
    ours = {
        "t_0 (HJD')": v("source.t_0", 0) - 2450000.0,
        "u_0": v("source.u_0", 0),
        "t_E [d]": v("mulensevent.t_E", 0),
        "rho [1e-3]": v("source.rho", 0) * 1e3,
        "s_2": v("lens.s", 1),
        "q_2": v("lens.q", 1),
        "alpha_2 [deg]": v("lens.alpha", 1),
        "s_3": v("lens.s", 2),
        "q_3": v("lens.q", 2),
        "psi [deg]": wrap(v("lens.alpha", 1) - v("lens.alpha", 2)),
        "theta_E [mas]": v("mulensevent.theta_E", 0),
        "mu_rel [mas/yr]": v("mulensevent.mu_rel_mag", 0),
        "D_L [kpc]": v("star.distance", 0) / 1e3,
        "M_1 [Msun]": v("star.mass", 0),
        "M_2 [Msun]": v("star.mass", 1),
        "M_3 [Msun]": v("star.mass", 2),
    }
    # the start (the convention-scan optimum = Han's solution mapped)
    up = yaml.safe_load(open("KMT-2021-BLG-1122L.params.scanopt.yaml"))

    def start(key):
        e = up.get(key)
        if isinstance(e, dict):
            return e.get("initval")
        return e

    s2 = start("lens.LensB.s"); a2 = start("lens.LensB.alpha")
    ls3 = start("lens.LensC.log_s")
    xa3, ya3 = start("lens.LensC.xalpha"), start("lens.LensC.yalpha")
    a3 = np.degrees(np.arctan2(ya3, xa3)) if xa3 is not None else None
    starts = {
        "t_0 (HJD')": (start("source.Source.t_0") or np.nan) - 2450000.0,
        "u_0": start("source.Source.u_0"),
        "s_2": s2, "alpha_2 [deg]": a2,
        "s_3": 10 ** ls3 if ls3 is not None else None,
        "psi [deg]": wrap(a2 - a3) if (a2 is not None and a3 is not None) else None,
        "q_2": start("lens.LensB.q"), "q_3": start("lens.LensC.q"),
        "t_E [d]": start("mulensevent.t_E") or start("mulensevent.0.t_E"),
        "rho [1e-3]": (start("source.Source.rho") or np.nan) * 1e3,
    }

    print("\n== posterior vs Han+2023 (post-burn-in) ==")
    hdr = (f"{'parameter':16s} {'ours med':>10s} {'-1s':>8s} {'+1s':>8s} "
           f"{'sd':>8s} | {'Han':>9s} {'sig_H':>7s} | {'z_comb':>6s} "
           f"{'z_Han':>6s} {'sigH/sd':>7s} | {'start':>9s} {'(med-st)/sd':>11s}")
    print(hdr)
    for name, a in ours.items():
        a = a.ravel()
        q16, q50, q84 = np.percentile(a, [16, 50, 84])
        sd = 0.5 * (q84 - q16)
        hv, hs = HAN[name]
        if hs is not None:
            zc = (q50 - hv) / np.hypot(sd, hs)
            zh = (q50 - hv) / hs
            ratio = hs / sd
            zs, zhs, rs = f"{zc:6.2f}", f"{zh:6.2f}", f"{ratio:7.2f}"
            hss = f"{hs:7.4g}"
        else:
            zs = f"{(q50 - hv) / sd:6.2f}"
            zhs, rs, hss = "   n/a", "    n/a", "    n/a"
        st = starts.get(name)
        sts = f"{st:9.4g}" if st is not None and np.isfinite(st) else "      n/a"
        mv = (f"{(q50 - st) / sd:11.2f}"
              if st is not None and np.isfinite(st) else "        n/a")
        flag = "  <- different origin" if name.startswith(("t_0", "u_0")) else ""
        print(f"{name:16s} {q50:10.5g} {q16 - q50:8.3g} {q84 - q50:+8.3g} "
              f"{sd:8.3g} | {hv:9.5g} {hss} | {zs} {zhs} {rs} | {sts} {mv}{flag}")

    # ---- diffusion diagnostics on the light-curve parameters -------------
    print("\n== diffusion diagnostics (post-burn-in) ==")
    print(f"{'parameter':16s} {'pooled sd':>10s} {'within/pooled':>13s} "
          f"{'sd q1':>8s} {'sd q2':>8s} {'sd q3':>8s} {'sd q4':>8s} "
          f"{'ESS':>7s} {'tau':>6s} {'chain-mean sd/sd':>16s}")
    lc = ["t_0 (HJD')", "u_0", "t_E [d]", "rho [1e-3]", "s_2", "q_2",
          "alpha_2 [deg]", "s_3", "q_3", "psi [deg]", "theta_E [mas]",
          "D_L [kpc]"]
    n = n_dr - burnin
    edges = np.linspace(0, n, 5).astype(int)
    for name in lc:
        a = ours[name]
        pooled = a.std()
        within = a.std(axis=1).mean()
        qs = [a[:, edges[i]:edges[i + 1]].std() for i in range(4)]
        ess = float(az.ess(az.convert_to_dataset({"x": a}), method="bulk")["x"].values)
        tau = a.size / ess
        cm = a.mean(axis=1).std() / pooled
        print(f"{name:16s} {pooled:10.4g} {within / pooled:13.3f} "
              + " ".join(f"{q:8.3g}" for q in qs)
              + f" {ess:7.0f} {tau:6.1f} {cm:16.3f}")

    # ---- posterior width vs the Laplace width at the optimum --------------
    blob = idata.attrs.get(HOT_SEARCH_ATTR)
    if blob is None:
        print("\n(no hot-search stamp on the trace yet; Laplace comparison skipped)")
        return
    ledger, status = hot_search_from_json(blob)
    rec = min(ledger, key=lambda r: r.delta_lp)  # the surviving optimum
    print(f"\n== posterior sd vs Laplace (1-D curvature, conditional) sigma at "
          f"the optimum (ledger seed {rec.seed_index}, lp {rec.lp_max:.1f}) ==")
    print("a marginal sd must be >= the conditional sigma; a ratio < 1 means the "
          "ensemble is narrower than the basin's own curvature allows (stuck)")
    pairs = [
        ("source.t_0", 0, "t_0 (HJD')", 1.0), ("source.u_0", 0, "u_0", 1.0),
        ("lens.log_s", 1, "log s_2", None), ("lens.log_s", 2, "log s_3", None),
        ("lens.xalpha", 1, "xalpha_2", None), ("lens.yalpha", 1, "yalpha_2", None),
        ("lens.xalpha", 2, "xalpha_3", None), ("lens.yalpha", 2, "yalpha_3", None),
        ("star.logmass", 0, "logM_1", None), ("star.logmass", 1, "logM_2", None),
        ("star.logmass", 2, "logM_3", None), ("star.distance", 0, "D_L [pc]", None),
    ]
    print(f"{'parameter':12s} {'posterior sd':>13s} {'Laplace sig':>12s} {'ratio':>7s}")
    for key, k, label, _ in pairs:
        if key not in rec.phys_sigma:
            continue
        sig = np.asarray(rec.phys_sigma[key]).ravel()
        idx = list(np.asarray(rec.sampled_idx[key]).ravel()) if key in rec.sampled_idx else list(range(sig.size))
        if k not in idx:
            continue
        lap = float(sig[idx.index(k)])
        a = post[key].values[:, burnin:, k]
        sd = a.std()
        print(f"{label:12s} {sd:13.4g} {lap:12.4g} {sd / lap:7.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
