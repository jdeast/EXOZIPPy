"""Score a microlensing fit MODE-AWARE, because a pull cannot.

WHY THE PULL FAILS HERE, measured on DC2018-128's d=27 observable arm:

    sampler                n_eff     max|pull|
    dynesty rwalk  500     4,803        25.5
    dynesty rwalk 1350    33,752        27.6
    dynesty rwalk 2700    80,138        43.2
    dynesty rslice 500    16,770        51.1
    PTDE (control)      converged       39.4

The score is MONOTONIC IN n_eff.  The best-sampled run scores worst and
converges toward PTDE's, because resolving the posterior better makes it
NARROWER, and narrow-in-the-wrong-mode is exactly what a pull punishes.  So
the pull ranks samplers by how vague they are.  PTDE's log_s sits at
-0.01051 +/- 0.00019 -- cleanly inside one mode, truth (-0.00299) outside
it, 39 sigma.  dynesty rwalk/500's sits at -0.03230 +/- 0.03261, 170x
broader, straddling both modes, so truth falls inside and the pull is 0.9.
Neither found truth's mode; one was just less willing to commit.

This is 7.13.1's lesson in a new costume: there the t_E marginal was skewed
and the pull was the wrong statistic (the PIT quantile was right).  Here the
posterior is MULTIMODAL and the pull is wrong again.

WHAT THIS SCORES INSTEAD, which is what a sweep actually needs to know:
  1. IS TRUTH'S MODE PRESENT at all?  Nearest-draw distance in whitened
     units -- if no draw comes near truth, the fit never visited it and no
     amount of pull arithmetic redeems that.
  2. WHAT WEIGHT does the region around truth carry?  Posterior mass within
     a tolerance of truth.  A fit that finds the right mode at 2% weight is
     a different outcome from one that finds it at 98%, and both differ from
     one that never finds it.
  3. WITHIN that region, is it unbiased?  A pull computed on the
     truth-local subset only, which is the question the global pull was
     trying and failing to ask.
Run: python dc18_mode_aware_score.py <trace.nc> [more.nc ...]
"""

import sys

import numpy as np
import xarray as xr

# (trace name post-#246, trace name pre-#246, truth, tolerance in the
# parameter's own units).  Tolerances are ~1% of the plausible range, chosen
# to be wide enough that "near truth" means the right MODE rather than the
# right draw.
SPEC = [
    ("source.t_0", "lens.t_0", 2458554.8868815, 0.05),
    ("source.u_0", "lens.u_0", 0.141832, 0.005),
    ("lens.Companion.log_s", "lens.log_s", float(np.log10(0.993145)), 0.010),
    (
        "planet.Companion.log_q",
        "planet.log_q",
        float(np.log10(0.0012118)),
        0.030,
    ),
    (
        "source.log_rho",
        "lens.log_rho",
        float(np.log10(0.006066783367838937)),
        0.060,
    ),
]


def pick(ds, new, old):
    for n in (new, old):
        if n in ds.data_vars:
            return n, np.asarray(ds[n]).ravel()
    return None, None


def score(path):
    ds = xr.open_dataset(path, group="posterior")
    cols, truths, tols, names = [], [], [], []
    for new, old, truth, tol in SPEC:
        n, v = pick(ds, new, old)
        if n is None:
            continue
        v = v.astype(float)
        cols.append(v)
        truths.append(truth)
        tols.append(tol)
        names.append(n)
    if not cols:
        print("%-46s no comparable parameters" % path)
        return
    k = min(len(c) for c in cols)
    X = np.vstack([c[:k] for c in cols])
    t = np.array(truths)[:, None]
    tol = np.array(tols)[:, None]

    good = np.isfinite(X).all(axis=0)
    X, n_draw = X[:, good], int(good.sum())

    # 1. nearest approach, in units of each parameter's tolerance
    d = np.abs(X - t) / tol
    worst_per_draw = d.max(axis=0)  # a draw is "at truth" only if
    near = worst_per_draw <= 1.0  # EVERY parameter is within tol
    nearest = float(worst_per_draw.min())

    # 2. weight of the truth-local region
    weight = float(near.mean())

    # 3. pull WITHIN that region (None when it was never visited)
    local = {}
    if near.sum() >= 20:
        for i, nm in enumerate(names):
            v = X[i, near]
            sd = float(np.std(v))
            local[nm] = (float(np.median(v)) - truths[i]) / sd if sd else None

    print(
        "%-46s draws=%-8d nearest=%6.2f tol  weight=%7.3f%%  %s"
        % (
            path.split("/")[-2] if "/" in path else path,
            n_draw,
            nearest,
            100 * weight,
            "FOUND truth's mode" if weight > 0 else "NEVER visited truth",
        ),
        flush=True,
    )
    if local:
        s = "  ".join(
            "%s %+.2f" % (nm.split(".")[-1], p)
            for nm, p in local.items()
            if p is not None
        )
        print("      in-mode pulls: %s" % s, flush=True)
    ds.close()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            score(p)
        except Exception as e:  # noqa: BLE001
            print("%-46s FAILED %s: %s" % (p, type(e).__name__, str(e)[:70]))
