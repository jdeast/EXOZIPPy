"""The marginal posterior of log10(pi_rel) from a run's trace, tabulated at
the profile grid so marginal and profile can be read side by side.

Why: a profile (dc18_ds_profile.py, pin log_pi_rel and re-optimize the
rest) is the likelihood-plus-prior MAXIMUM along the coordinate; the
posterior marginal is the INTEGRAL over everything else.  Their difference
along the grid is the marginalization-volume term, and its sign says
whether the posterior peak is where the model's maximum is or where the
nuisance volume is largest.  On 194 (sweep2) that difference was -4, -7,
-11 nats at -2.2, -1.6, -1.2 and named the lens-Teff volume; with the
mamajek tie in place (sweep4) the same numbers on 047 say whether any
volume term is left, or whether the 2x lens mass is the profile's own.

Also printed: the marginal of pi_rel's two ingredients, D_l and D_s, and
their correlation, since pi_rel = 1/D_l - 1/D_s is the DIFFERENCE of two
distances a bulge-bulge event barely separates.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--grid",
        default="-2.8,-2.6,-2.4,-2.2,-2.0,-1.9,-1.8,-1.7,-1.6,-1.4,-1.2,-1.0",
    )
    ap.add_argument("--ref", type=float, default=-2.8)
    ap.add_argument(
        "--truth", type=float, default=None, help="log10 pi_rel truth"
    )
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    import arviz as az

    pre = os.path.join(os.path.abspath(args.run_dir), f"DC2018_{args.event}")
    idata = az.from_netcdf(pre + "_trace.nc")
    post = idata.posterior
    lp = idata.sample_stats.lp.values
    x = post["mulensevent.log_pi_rel"].values.reshape(-1)
    dist = post["star.distance"].values
    dl = dist[..., 0].reshape(-1)
    ds = dist[..., 1].reshape(-1)
    lm = post["star.logmass"].values[..., 0].reshape(-1)
    lte = post["mulensevent.log_theta_E"].values.reshape(-1)
    print(
        f"{args.event}: {x.size} draws; log_pi_rel median {np.median(x):.3f} "
        f"[{np.percentile(x, 16):.3f}, {np.percentile(x, 84):.3f}]; "
        f"D_l {np.median(dl):.0f} +/- {np.std(dl):.0f}, D_s {np.median(ds):.0f} "
        f"+/- {np.std(ds):.0f}, corr(D_l, D_s) = {np.corrcoef(dl, ds)[0, 1]:.3f}; "
        f"best-lp draw log_pi_rel {x[np.argmax(lp.reshape(-1))]:.3f}"
    )
    if args.truth is not None:
        frac = np.mean(x > args.truth)
        print(f"truth {args.truth:.3f}: fraction of draws above it {frac:.4f}")

    # kernel density in log_pi_rel (Gaussian, Silverman), evaluated at the grid
    h = 1.06 * np.std(x) * x.size ** (-0.2)
    grid = np.array([float(s) for s in args.grid.split(",")])

    def logdens(v):
        return np.log(
            np.mean(np.exp(-0.5 * ((v - x) / h) ** 2))
            / (h * np.sqrt(2 * np.pi))
            + 1e-300
        )

    ref = logdens(args.ref)
    print(
        f"KDE bandwidth {h:.3f} dex; log marginal density relative to {args.ref}:"
    )
    print(
        f"{'log_pi_rel':>11}{'ln p_marg':>11}{'<logmass_l>':>13}{'<log_thE>':>11}{'<D_l>':>8}{'<D_s>':>8}{'n':>7}"
    )
    for g in grid:
        sel = np.abs(x - g) < 0.1
        n = int(sel.sum())
        print(
            f"{g:11.2f}{logdens(g) - ref:11.2f}"
            + (
                f"{np.mean(lm[sel]):13.3f}{np.mean(lte[sel]):11.3f}{np.mean(dl[sel]):8.0f}{np.mean(ds[sel]):8.0f}{n:7d}"
                if n
                else f"{'--':>13}{'--':>11}{'--':>8}{'--':>8}{n:7d}"
            )
        )
    np.savez(
        pre + "_pirel_marginal.npz",
        grid=grid,
        log_marginal=np.array([logdens(g) - ref for g in grid]),
        bandwidth=h,
        ref=args.ref,
    )


if __name__ == "__main__":
    main()
