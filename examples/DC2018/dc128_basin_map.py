#!/usr/bin/env python3
"""Map the (s, q) and (s, alpha) chi2 surface for DC2018 event 128.

Follow-up to dc128_basin_scan.py, which established that the truth basin
(s = 0.993) beats the basin EXOZIPPy landed in (s = 0.863) by ~8400 in
chi2.  The question this answers is why a sampler seeded in the truth
basin left it: where the barrier between the two basins sits, and where a
plain downhill refine from each MMEXOFAST seed actually ends up.
"""

import json

import numpy as np
from dc128_basin_scan import (
    BANDS,
    EXOZIPPY,
    MMX0,
    MMX1,
    PARNAMES,
    TRUTH,
    chi2_of,
    load_datasets,
)
from scipy.optimize import minimize

BEST_ALPHA = 308.0  # from the dc128_basin_scan alpha scan


def grid2d(base, datasets, xname, xs, yname, ys):
    out = np.empty((len(ys), len(xs)))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            out[i, j] = chi2_of(dict(base, **{xname: xv, yname: yv}), datasets)
        print(
            f"  {yname}={yv:.6g}  min chi2 = {np.nanmin(out[i]):.1f} "
            f"at {xname}={xs[np.nanargmin(out[i])]:.5g}",
            flush=True,
        )
    return out


def refine(start, datasets, label):
    def unpack(x):
        return dict(
            t_0=x[0],
            u_0=x[1],
            t_E=x[2],
            rho=10 ** x[3],
            s=10 ** x[4],
            q=10 ** x[5],
            alpha=x[6],
        )

    def cost(x):
        if not (
            1e-7 < 10 ** x[3] < 0.2
            and 1e-4 < 10 ** x[4] < 10
            and 1e-8 < 10 ** x[5] < 1
            and 0.5 < x[2] < 500
        ):
            return 1e12
        return chi2_of(unpack(x), datasets)

    x0 = np.array(
        [
            start["t_0"],
            start["u_0"],
            start["t_E"],
            np.log10(max(start["rho"], 1e-6)),
            np.log10(start["s"]),
            np.log10(start["q"]),
            start["alpha"],
        ]
    )
    print(
        f"\n--- downhill refine from {label} (start chi2 {cost(x0):.1f}) ---",
        flush=True,
    )
    res = minimize(
        cost,
        x0,
        method="Nelder-Mead",
        options=dict(maxfev=8000, xatol=1e-9, fatol=1e-3, adaptive=True),
    )
    p = unpack(res.x)
    print(f"    chi2 = {res.fun:.2f}  (nfev={res.nfev})")
    for k in PARNAMES:
        print(f"      {k:6s} {p[k]:.6g}")
    return p, float(res.fun)


def main():
    datasets = load_datasets()
    n = sum(len(d.time) for d in datasets)
    base = dict(TRUTH, alpha=BEST_ALPHA)
    out = {}

    print("=" * 70)
    print("A. downhill refine from each MMEXOFAST seed -- where does it go?")
    print("=" * 70)
    for label, seed in [
        ("mmx0 (s=0.859)", MMX0),
        ("mmx1 (s=0.982)", MMX1),
        ("exozippy (s=0.863)", EXOZIPPY),
    ]:
        p, c = refine(seed, datasets, label)
        out[label] = dict(params=p, chi2=c)

    print("\n" + "=" * 70)
    print("B. (s, q) surface at truth t_0,u_0,t_E,rho and alpha=308")
    print("=" * 70)
    s_grid = np.linspace(0.80, 1.12, 65)
    q_grid = np.logspace(-4.2, -2.4, 37)
    z_sq = grid2d(base, datasets, "s", s_grid, "q", q_grid)
    np.savez("dc128_grid_sq.npz", s=s_grid, q=q_grid, chi2=z_sq)
    i, j = np.unravel_index(np.nanargmin(z_sq), z_sq.shape)
    print(
        f"  grid minimum chi2 = {z_sq[i, j]:.1f} at s={s_grid[j]:.5g}, "
        f"q={q_grid[i]:.5g}"
    )

    print("\n" + "=" * 70)
    print("C. (s, alpha) surface at truth t_0,u_0,t_E,rho,q")
    print("=" * 70)
    a_grid = np.arange(280.0, 340.0, 1.0)
    z_sa = grid2d(base, datasets, "s", s_grid, "alpha", a_grid)
    np.savez("dc128_grid_salpha.npz", s=s_grid, alpha=a_grid, chi2=z_sa)
    i, j = np.unravel_index(np.nanargmin(z_sa), z_sa.shape)
    print(
        f"  grid minimum chi2 = {z_sa[i, j]:.1f} at s={s_grid[j]:.5g}, "
        f"alpha={a_grid[i]:.5g}"
    )

    print("\n" + "=" * 70)
    print("D. chi2 profile along s (each s minimized over q and alpha)")
    print("=" * 70)
    prof = np.nanmin(z_sq, axis=0)
    prof_a = np.nanmin(z_sa, axis=0)
    print(f"  {'s':>8s} {'min over q':>14s} {'min over alpha':>16s}")
    for k, sv in enumerate(s_grid):
        print(f"  {sv:8.4f} {prof[k]:14.1f} {prof_a[k]:16.1f}")

    json.dump(
        {k: dict(chi2=v["chi2"], params=v["params"]) for k, v in out.items()}
        | {"n_points": n},
        open("dc128_basin_map.json", "w"),
        indent=2,
    )
    print(
        "\nwrote dc128_basin_map.json, dc128_grid_sq.npz, "
        "dc128_grid_salpha.npz"
    )


if __name__ == "__main__":
    main()
