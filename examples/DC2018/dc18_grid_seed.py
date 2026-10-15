"""Turn a finished DC2018 (s, q) grid into the SEED of a blind fit.

THE SEEDING LOGIC MOVED.  Which basins count as distinct, which leaves may
be seeded at all, and what `overdisperse:` must say are decided in
`exozippy.components.mulensing.gridsearch` -- RMDC26 needs the same
answers, and a second copy drifts.  This file is the DC2018 half: reading
the `ev<N>_*` npz layout and naming the parameters the way a DC2018 config
spells them.

WHAT CHANGED WHEN IT MOVED, and it matters for any fit seeded by the older
version of this script: it wrote `overdisperse: false`, reasoning that K
basins are K modes and so "do NOT want scattering".  samplers.md (8.3.3)
defines the key the other way round -- `false` declares the seeds to be
ALREADY-DISPERSED posterior draws, so every chain starts exactly on one of
K points with NO jitter and the start population's affine hull is the hull
of those K points.  Five basins against a 34-parameter model spans three
directions where n_params + 2 = 36 are needed to span the space at all.
`seed_fragment` writes `true`.

ONE SEED OR K?  Deliberately both, because which works best is open (JDE
2026-10-04).  `--n-seeds 1` writes the global minimum as scalar initvals;
`--n-seeds K` writes the K best DISTINCT local minima as a joint multi-seed
start set, which `mkparam`'s P4 path already consumes.

A grid written before it recorded t_0/u_0/t_E per node (no `ev<N>_t0u0tE`
key) is REFUSED rather than silently seeded from the point-lens start,
which would throw away exactly the geometry the grid was run to find.
"""

import argparse
import json

import numpy as np

from exozippy.components.mulensing.gridsearch import (
    GridResult,
    seed_fragment,
)

# What a DC2018 model actually SAMPLES, verified by building the 194 system
# with these keys and reading the resolved initvals back (element 1 is the
# companion; element 0 is the pinned primary at log_s 0 / alpha nan):
#     lens.Companion.log_s    -> [0.0,  0.2099973]
#     planet.Companion.log_q  -> [-4.0, -3.0]
#     lens.Companion.alpha    -> [nan,  51.3]
# NOT s/q: the model samples log_s and log_q.  t_E and rho are absent on
# purpose -- gridsearch.SEED_LEAVES explains why.
NAMES = {
    "t_0": "source.Source.t_0",
    "u_0": "source.Source.u_0",
    "log_s": "lens.Companion.log_s",
    "log_q": "planet.Companion.log_q",
    "alpha": "lens.Companion.alpha",
}


def load_event_grid(npz_path, event):
    """One event's grid out of the multi-event DC2018 npz."""
    z = np.load(npz_path)
    key = "ev%d_t0u0tE" % event
    if key not in z:
        raise SystemExit(
            "%s has no '%s': it was written before the grid recorded "
            "t_0/u_0/t_E per node.  Re-run dc18_highmag_grid2.py rather "
            "than seeding s and q onto a point-lens t_0/u_0/t_E, which "
            "discards the geometry the grid was run to find." % (npz_path, key)
        )
    chi2 = z["ev%d_chi2" % event]
    tuE = z[key]
    return GridResult(
        log_s=z["log_s"],
        log_q=z["log_q"],
        chi2=chi2,
        alpha=z["ev%d_alpha" % event],
        t_0=tuE[..., 0],
        u_0=tuE[..., 1],
        t_E=tuE[..., 2],
        # This layout predates the grid recording rho per node, so there is
        # none to report.  nan rather than a plausible-looking number: the
        # fragment then says nothing about rho instead of asserting one.
        rho=np.full(chi2.shape, np.nan),
        n_done=int(np.isfinite(chi2).sum()),
        n_tasks=chi2.size,
        complete=bool(np.isfinite(chi2).all()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument(
        "--min-sep",
        type=int,
        default=None,
        help=(
            "suppress candidates within this many NODES of an accepted one "
            "on BOTH axes; default is gridsearch's, which is a distance in "
            "dex (0.03 in log s, 0.2 in log q) converted at this grid's own "
            "spacing -- a node count does not transfer between resolutions"
        ),
    )
    ap.add_argument("--out", default=None, help="params YAML fragment")
    a = ap.parse_args()

    res = load_event_grid(a.npz, a.event)
    seeds = res.local_minima(k=a.n_seeds, min_sep=a.min_sep)

    sep = a.min_sep if a.min_sep is not None else res.default_min_sep()
    print(
        "event %d: %d seed(s) from %s (min-sep %s nodes, %.4f dex per log_s "
        "node)" % (a.event, len(seeds), a.npz, sep, res.d_log_s)
    )
    for n, s in enumerate(seeds):
        print(
            "  %d: s=%.4f q=%.3g alpha=%6.1f  t_0=%.4f u_0=%+.4f t_E=%.3f"
            "  chi2=%.1f (+%.1f)"
            % (
                n,
                s["s"],
                s["q"],
                s["alpha"],
                s["t_0"],
                s["u_0"],
                s["t_E"],
                s["chi2"],
                s["dchi2"],
            )
        )

    if a.out:
        text = seed_fragment(
            seeds,
            NAMES,
            header=[
                "# Generated by dc18_grid_seed.py from %s" % a.npz,
                "# event %d." % a.event,
            ],
        )
        with open(a.out, "w") as fh:
            fh.write(text)
        print("\nwrote %s" % a.out)
    else:
        print("\n(no --out given; JSON to stdout)")
        print(json.dumps(seeds, indent=1))


if __name__ == "__main__":
    main()
