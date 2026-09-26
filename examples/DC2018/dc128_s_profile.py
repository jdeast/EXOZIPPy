#!/usr/bin/env python3
"""Profile likelihood in s: how high is the barrier between the two basins?

The 2D slices in dc128_basin_map.py show an enormous ridge between the
s = 0.85 and s = 0.98 minima, but a slice is an UPPER BOUND on a barrier --
it holds t_0, u_0, t_E, rho and (in one case) alpha at truth while the real
crossing would re-optimize all of them.  The height matters a lot for what
to conclude:

  barrier >> T_max * few nats   only the multi-seed start can populate the
                                good basin, tempering can never bridge it,
                                and a seed that arrives under-polished is
                                unrecoverable -- the run is decided before
                                sampling begins
  barrier ~ T_max * few nats    the ladder could cross it, and the right
                                fix is more rungs / a taller T_max

So this measures the profile properly: s is FIXED on a grid, everything else
(t_0, u_0, t_E, rho, q, alpha, plus the linear fluxes) is re-optimized at
each s.  The peak of that curve is the true barrier along the s coordinate.
"""

import json

import numpy as np
from dc128_basin_scan import PARNAMES, chi2_of, load_datasets
from scipy.optimize import minimize

# Walk outward from each minimum so every fit starts from its neighbour's
# solution -- a cold start at the ridge top finds nothing.
S_GRID = np.round(np.arange(0.84, 1.005, 0.005), 4)
LEFT_START = dict(
    t_0=2458554.8868815,
    u_0=0.135498,
    t_E=18.7107,
    rho=3.84e-06,
    q=0.0010694,
    alpha=308.662,
)
RIGHT_START = dict(
    t_0=2458554.8868815,
    u_0=0.142493,
    t_E=18.1978,
    rho=0.00715733,
    q=0.00107938,
    alpha=307.986,
)


def fit_at_s(s, start, datasets):
    """Minimize chi2 over everything except s, which is held at `s`."""

    def unpack(x):
        return dict(
            t_0=x[0],
            u_0=x[1],
            t_E=x[2],
            rho=10 ** x[3],
            s=s,
            q=10 ** x[4],
            alpha=x[5],
        )

    def cost(x):
        if not (
            1e-7 < 10 ** x[3] < 0.2
            and 1e-8 < 10 ** x[4] < 1
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
            np.log10(start["q"]),
            start["alpha"],
        ]
    )
    res = minimize(
        cost,
        x0,
        method="Nelder-Mead",
        options=dict(maxfev=3000, xatol=1e-9, fatol=1e-3, adaptive=True),
    )
    return unpack(res.x), float(res.fun)


def sweep(order, start, datasets, label):
    out = {}
    cur = dict(start)
    for s in order:
        best, c = fit_at_s(float(s), cur, datasets)
        out[float(s)] = (best, c)
        cur = best  # warm start the next s
        print(
            f"  [{label}] s={s:.4f}  chi2={c:12.1f}   q={best['q']:.5g} "
            f"rho={best['rho']:.4g} alpha={best['alpha']:8.3f}",
            flush=True,
        )
    return out


def main():
    datasets = load_datasets()
    n = sum(len(d.time) for d in datasets)
    print(
        f"profile likelihood in s, {n} points, all other parameters "
        f"re-optimized at each s\n"
    )

    print("sweeping RIGHTWARD from the s=0.85 basin:")
    left = sweep(S_GRID, LEFT_START, datasets, "L->R")
    print("\nsweeping LEFTWARD from the s=0.98 basin:")
    right = sweep(S_GRID[::-1], RIGHT_START, datasets, "R->L")

    # The profile is the better of the two sweeps at each s: a warm-started
    # walk can be dragged along by its own history past a real minimum.
    prof = {s: min(left[s][1], right[s][1]) for s in left}
    smin = min(prof, key=prof.get)
    cmin = prof[smin]
    # Barrier: the highest point on the profile BETWEEN the two minima.
    lo, hi = 0.86, 0.975
    between = {s: c for s, c in prof.items() if lo <= s <= hi}
    speak = max(between, key=between.get)

    print("\n" + "=" * 66)
    print("PROFILE LIKELIHOOD IN s   (chi2, everything else re-optimized)")
    print("=" * 66)
    print(
        f"  {'s':>8s} {'L->R':>12s} {'R->L':>12s} {'profile':>12s} "
        f"{'dchi2':>10s} {'nats':>9s}"
    )
    for s in sorted(prof):
        print(
            f"  {s:8.4f} {left[s][1]:12.1f} {right[s][1]:12.1f} "
            f"{prof[s]:12.1f} {prof[s] - cmin:10.1f} "
            f"{(prof[s] - cmin) / 2:9.1f}"
        )

    barrier_nats = (between[speak] - cmin) / 2
    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    print(f"  global minimum        s = {smin:.4f}   chi2 = {cmin:.1f}")
    print(
        f"  ridge top between     s = {speak:.4f}   chi2 = "
        f"{between[speak]:.1f}"
    )
    print(
        f"  BARRIER = {between[speak] - cmin:.1f} chi2 = "
        f"{barrier_nats:.1f} nats"
    )
    for T in (1, 10, 50, 200):
        print(
            f"     at T={T:4d} the ridge costs {barrier_nats / T:10.1f} "
            f"nats -> crossing probability ~ e^-{barrier_nats / T:.0f}"
        )
    print("\n  A ladder topping out at T_max=200 can only bridge a barrier of")
    print("  order a few hundred nats.  If the number above is far larger,")
    print("  the multi-seed start is the ONLY route into the good basin and")
    print("  seed quality decides the run before sampling starts.")

    json.dump(
        {
            "profile": {str(s): prof[s] for s in prof},
            "s_min": smin,
            "chi2_min": cmin,
            "s_peak": speak,
            "chi2_peak": between[speak],
            "barrier_nats": barrier_nats,
        },
        open("dc128_s_profile.json", "w"),
        indent=2,
    )
    print("\nwrote dc128_s_profile.json")


if __name__ == "__main__":
    main()
