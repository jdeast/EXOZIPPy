#!/usr/bin/env python3
"""What did the ladder actually SEARCH, and how fine was the search?

The aim is a quotable completeness claim of the form

    "within these bounds, any solution whose basin is wider than X (in units
     of the posterior's own width) would have contained at least one point
     we visited"

which is a covering-radius statement, not a likelihood-threshold one.  Two
numbers make it up:

  BREADTH     the hypervolume the search actually entered, per parameter,
              reported separately for the T=1 posterior and the hot rungs
              (`store_hot_chains`).  The hot rungs are the search; the cold
              chain only reports the winner.

  RESOLUTION  the covering radius of the visited set inside that volume:
              draw random probe points, measure the distance from each to
              the nearest visited draw, and quote a high quantile.  A basin
              of radius r > that quantile could not have hidden from us
              (with the stated probability); one of radius < it could.

Distances are computed in WHITENED raw coordinates, where one unit is one
measured posterior sigma by construction -- so "width" in the claim means
width relative to the peak we found, which is the scientifically useful
comparison.

DIMENSIONALITY IS THE WHOLE DIFFICULTY, and it has to be stated honestly:
covering a 27-dimensional box is hopeless, and the covering radius there
will be many sigma no matter how long we run.  But microlensing
degeneracies do not live in 27 dimensions -- they live in the light-curve
observables (close/wide in s, the u_0 sign, alpha reflections, the parallax
families).  So the report is made in a NAMED SUBSPACE, and the number is
only meaningful for alternative solutions that differ within it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Where 2L1S alternative solutions actually differ.  Everything else in the
# model (fluxes, error scaling, the galactic sector) follows from these.
DEGENERACY_SUBSPACE = [
    "lens.log_s",
    "lens.xalpha",
    "lens.yalpha",
    "lens.u_0",
    "lens.t_0",
    "planet.log_q",
]


def _stack(group, names):
    """(n_draws, n_par) array of the named raw variables, plus those found."""
    cols, found = [], []
    for n in names:
        if n in group:
            v = np.asarray(group[n].values, dtype=float)
            cols.append(v.reshape(-1) if v.ndim <= 2 else v.reshape(-1))
            found.append(n)
    if not cols:
        return None, []
    m = min(c.size for c in cols)
    return np.column_stack([c[:m] for c in cols]), found


def covering_radius(points, lo, hi, n_probe=20000, quantile=0.95, rng=None):
    """Monte-Carlo covering radius of `points` inside the box [lo, hi].

    Returns the `quantile` of "distance from a uniform probe point to the
    nearest visited point".  A basin whose radius exceeds this contains a
    visited point with at least `quantile` probability; below it, the search
    had gaps it could have hidden in.
    """
    rng = rng or np.random.default_rng(0)
    d = points.shape[1]
    probes = rng.uniform(lo, hi, size=(n_probe, d))
    # Chunked to keep the pairwise distance matrix bounded.
    best = np.full(n_probe, np.inf)
    step = max(1, int(2e7 // max(points.shape[0], 1)))
    for a in range(0, n_probe, step):
        b = min(a + step, n_probe)
        dist = np.sqrt(
            ((probes[a:b, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        )
        best[a:b] = dist.min(axis=1)
    return float(np.quantile(best, quantile)), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="path to *_trace.nc")
    ap.add_argument("--thin", type=int, default=25)
    ap.add_argument("--n-probe", type=int, default=20000)
    args = ap.parse_args()

    import arviz as az

    idata = az.from_netcdf(args.trace)
    print(f"trace: {args.trace}")
    groups = idata.groups
    print(f"groups: {list(groups() if callable(groups) else groups)}\n")

    raw_cold = {
        k: v
        for k, v in idata.posterior.data_vars.items()
        if k.endswith("_raw")
    }
    hot = getattr(idata, "posterior_hot", None)

    print("=" * 74)
    print("BREADTH -- range of each degeneracy coordinate actually entered")
    print("(whitened raw units: 1.0 = one measured posterior sigma)")
    print("=" * 74)
    print(
        f"{'coordinate':<18s}{'T=1 min':>10s}{'T=1 max':>10s}"
        f"{'hot min':>10s}{'hot max':>10s}{'hot/cold':>10s}"
    )
    rows = {}
    for name in DEGENERACY_SUBSPACE:
        key = name.replace(".", ".") + "_raw"
        cold_v = hot_v = None
        for k, v in raw_cold.items():
            if k.startswith(name.split(".")[0]) and name.split(".")[-1] in k:
                cold_v = np.asarray(v.values, float).ravel()
        if hot is not None:
            for k in hot.data_vars:
                if (
                    k.startswith(name.split(".")[0])
                    and name.split(".")[-1] in k
                ):
                    hot_v = np.asarray(hot[k].values, float).ravel()
        if cold_v is None:
            continue
        cw = cold_v.max() - cold_v.min()
        hw = (hot_v.max() - hot_v.min()) if hot_v is not None else np.nan
        rows[name] = (cold_v, hot_v)
        print(
            f"{name:<18s}{cold_v.min():10.3f}{cold_v.max():10.3f}"
            f"{(hot_v.min() if hot_v is not None else np.nan):10.3f}"
            f"{(hot_v.max() if hot_v is not None else np.nan):10.3f}"
            f"{(hw / cw if cw else np.nan):10.2f}"
        )

    if not rows:
        print("no degeneracy coordinates found in this trace; nothing to do.")
        return

    print("\n" + "=" * 74)
    print("RESOLUTION -- covering radius of the visited set")
    print("=" * 74)
    out = {}
    for tier in ("T=1 only", "T=1 + hot rungs"):
        cols = []
        for name, (cold_v, hot_v) in rows.items():
            v = (
                cold_v
                if tier == "T=1 only" or hot_v is None
                else np.concatenate([cold_v, hot_v])
            )
            cols.append(v)
        m = min(c.size for c in cols)
        pts = np.column_stack([c[:m] for c in cols])[:: args.thin]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        r95, dists = covering_radius(
            pts, lo, hi, n_probe=args.n_probe, quantile=0.95
        )
        r50 = float(np.quantile(dists, 0.50))
        vol_side = float(np.mean(hi - lo))
        print(
            f"\n  {tier}:  {pts.shape[0]} visited points in "
            f"{pts.shape[1]}-D, mean box side {vol_side:.2f} sigma"
        )
        print(f"    median gap to nearest visited point : {r50:.3f} sigma")
        print(f"    95th-percentile gap (covering radius): {r95:.3f} sigma")
        print(
            f"    => any basin of radius > {r95:.2f} sigma inside the "
            f"searched box\n       would have contained a visited point "
            f"(95% of the time)."
        )
        out[tier] = dict(
            n_points=int(pts.shape[0]),
            dim=int(pts.shape[1]),
            box_side_sigma=vol_side,
            covering_r95=r95,
            median_gap=r50,
            lo=lo.tolist(),
            hi=hi.tolist(),
        )

    print("\n" + "=" * 74)
    print("CAVEATS, which belong next to the number")
    print("=" * 74)
    print(
        f"  - The claim is confined to the {len(rows)}-D subspace "
        f"{list(rows)};\n    a solution differing only in the other "
        f"coordinates is NOT covered by it."
    )
    print(
        "  - The box is what the search entered, not the prior support: a\n"
        "    basin outside these bounds was never a candidate at all."
    )
    print(
        "  - Covering radius grows as the box^(1/d) / n^(1/d), so quoting\n"
        "    it for the full 27-D model would be honest but useless."
    )
    json.dump(
        out,
        open(
            Path(args.trace).with_suffix("").name + "_completeness.json", "w"
        ),
        indent=2,
    )


if __name__ == "__main__":
    main()
