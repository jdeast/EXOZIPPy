"""Turn a finished (s,q) grid into the SEED of a blind fit.

WHY THIS EXISTS.  The blind pipeline does no binary mode finding at all: every
peak-finder seed carries s = 1.0, q = 0.001 verbatim, so topology is set at
initialization by an initialization containing no topology information (notes
2.4.14).  dc18_highmag_grid2.py finds the truth's basin as its GLOBAL minimum
on events the sampler cannot converge on -- 069 and 163, both ABSTAINED --
and on the 194 control, every time on the direct branch rather than the
s <-> 1/s mirror.  This is the wire between the two.

ONE SEED OR K?  Deliberately both, because which works best is open (JDE
2026-10-04).  `--n-seeds 1` writes the global minimum as scalar initvals.
`--n-seeds K` writes the K best DISTINCT local minima as a joint multi-seed
start set, which `mkparam`'s P4 path already consumes, with
`overdisperse: false` because these are K separate MODES rather than K draws
from one posterior that still need scattering (samplers.md).

DISTINCT matters: adjacent nodes of one basin are not K modes.  Candidates
are suppressed within `--min-sep` nodes of an already-accepted one, so K
seeds are K basins.

The grid records t_0/u_0/t_E per node, so no re-minimization is needed.  A
grid written before that (no `ev<N>_t0u0tE` key) is refused rather than
silently seeded from the point-lens start, which would throw away exactly the
geometry the grid was run to find.
"""

import argparse
import json

import numpy as np

# What the DC2018 model actually SAMPLES, verified by building the 194 system
# with these keys and reading the resolved initvals back (element 1 is the
# companion; element 0 is the pinned primary at log_s 0 / alpha nan):
#     lens.Companion.log_s    -> [0.0,  0.2099973]
#     planet.Companion.log_q  -> [-4.0, -3.0]
#     lens.Companion.alpha    -> [nan,  51.3]
#
# NOT s/q: the model samples log_s and log_q.  NOT t_E: it is DERIVED from
# theta_E/mu_rel and is not in the trace at all, so seeding it is review
# 2.3.17's trap -- the injected value is silently overridden and the seed
# that was meant to be tested never existed.  u_0 IS accepted (the existing
# params file sets it) even though u0te is what is sampled, because the
# relaxation engine propagates it.
NAMES = {
    "t_0": "source.Source.t_0",
    "u_0": "source.Source.u_0",
    "log_s": "lens.Companion.log_s",
    "log_q": "planet.Companion.log_q",
    "alpha": "lens.Companion.alpha",
}
SEEDED = ("t_0", "u_0", "log_s", "log_q", "alpha")


def local_minima(c, k, min_sep):
    """The k best nodes, no two within `min_sep` nodes of each other."""
    out = []
    flat = np.argsort(c, axis=None)
    for idx in flat:
        if not np.isfinite(c.flat[idx]):
            break
        i, j = np.unravel_index(idx, c.shape)
        if any(abs(i - a) < min_sep and abs(j - b) < min_sep for a, b in out):
            continue
        out.append((i, j))
        if len(out) == k:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, default=1)
    # MIN-SEP IS IN NODES, SO IT DOES NOT TRANSFER BETWEEN GRID
    # RESOLUTIONS, and the default is therefore derived from the grid's own
    # spacing rather than fixed.  The suppression radius that matters is a
    # distance in log s -- "near enough to be the same basin" -- and 0.03
    # dex is what the original 71-node grid's default of 3 nodes meant.  On
    # the 21-node sweep grid (0.07 dex per node) a literal 3 means 0.21 dex,
    # which swallows genuinely distinct basins: measured over the 30-event
    # sweep, the fixed default finds the truth among 5 seeds on 25/30 where
    # the scaled one finds it on 27/30, losing events 069 and 103 outright.
    # Floor of 2 because 1 suppresses nothing and spends seeds on adjacent
    # nodes of one basin.
    ap.add_argument(
        "--min-sep",
        type=int,
        default=None,
        help=(
            "suppress candidates within this many NODES of an accepted one; "
            "default scales with the grid so the radius is ~0.03 dex in "
            "log s (3 on a 0.01 dex grid, 2 on a 0.07 dex one)"
        ),
    )
    ap.add_argument("--out", default=None, help="params YAML fragment")
    a = ap.parse_args()

    z = np.load(a.npz)
    ev = a.event
    key = "ev%d_t0u0tE" % ev
    if key not in z:
        raise SystemExit(
            "%s has no '%s': it was written before the grid recorded "
            "t_0/u_0/t_E per node.  Re-run dc18_highmag_grid2.py rather "
            "than seeding s and q onto a point-lens t_0/u_0/t_E, which "
            "discards the geometry the grid was run to find." % (a.npz, key)
        )
    c = z["ev%d_chi2" % ev]
    alpha = z["ev%d_alpha" % ev]
    tuE = z[key]
    ls, lq = z["log_s"], z["log_q"]

    if a.min_sep is None:
        d_log_s = float(ls[1] - ls[0]) if len(ls) > 1 else 0.01
        a.min_sep = max(2, int(round(0.03 / d_log_s)))
        print("min-sep %d nodes (%.3f dex per node)" % (a.min_sep, d_log_s))
    picks = local_minima(c, a.n_seeds, a.min_sep)
    if not picks:
        raise SystemExit("no finite nodes in %s for event %d" % (a.npz, ev))

    seeds = []
    for i, j in picks:
        t0, u0, tE = tuE[i, j]
        seeds.append(
            {
                "t_0": float(t0),
                "u_0": float(u0),
                "t_E": float(tE),
                "log_s": float(ls[i]),
                "log_q": float(lq[j]),
                "alpha": float(alpha[i, j]),
                "chi2": float(c[i, j]),
            }
        )

    best = seeds[0]["chi2"]
    print("event %d: %d seed(s) from %s" % (ev, len(seeds), a.npz))
    for n, s in enumerate(seeds):
        print(
            "  %d: s=%.4f q=%.3g alpha=%6.1f  t_0=%.4f u_0=%+.4f t_E=%.3f"
            "  chi2=%.1f (+%.1f)"
            % (
                n,
                10.0 ** s["log_s"],
                10.0 ** s["log_q"],
                s["alpha"],
                s["t_0"],
                s["u_0"],
                s["t_E"],
                s["chi2"],
                s["chi2"] - best,
            )
        )

    if a.out:
        lines = [
            "# Generated by dc18_grid_seed.py from %s" % a.npz,
            "# event %d, %d seed(s), min-sep %d nodes."
            % (ev, len(seeds), a.min_sep),
            "#",
            "# These are K separate MODES off a (s,q) grid, not K draws from",
            "# one posterior, so they do NOT want scattering.",
            "overdisperse: false" if len(seeds) > 1 else "overdisperse: true",
            "",
        ]
        names = NAMES
        for k in SEEDED:
            vals = [s[k] for s in seeds]
            v = vals[0] if len(vals) == 1 else vals
            lines.append("%s:" % names[k])
            lines.append("  initval: %r" % (v,))
        open(a.out, "w").write("\n".join(lines) + "\n")
        print("\nwrote %s" % a.out)
    else:
        print("\n(no --out given; JSON to stdout)")
        print(json.dumps(seeds, indent=1))


if __name__ == "__main__":
    main()
