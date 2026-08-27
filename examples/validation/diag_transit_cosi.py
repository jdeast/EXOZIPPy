"""Why does planet.b mix at ESS ~8 in the transit inject-recover leg?

Reproduces realization seed=1000, snr=30, n_epochs=100 (the cell that showed
planet.b at ESS 8.0 / Rhat 1.43 while orbit.period beside it sat at ESS 3414)
and reports what the CHAINS are actually doing, instead of inferring a cause
from summary statistics.

Two hypotheses have already been eliminated by measurement:
  * "cos i sign degeneracy" -- bounding cos i >= 0 changed nothing
    (ESS 7.8/Rhat 1.45 before, ESS 8.0/Rhat 1.43 after), and under the
    shipped fitchord parameterization the sign is not sampled at all.
  * "not enough draws" -- 1500/1500 and 2000/2000 give the same ESS.

What this prints, per chain: mean/sd/min/max of the sampled coordinate
(orbit.cosi_raw) and of the derived b, ar and t14, plus the divergence count
and the step size.  Chains sitting in disjoint ranges means a multimodal or
disconnected posterior; chains agreeing on the range but with tiny ESS means
a badly conditioned single mode (a ridge); many divergences means geometry
the sampler cannot follow.  Those three want different fixes, and the
summary statistics alone cannot tell them apart.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject_recover as ir  # noqa: E402


def main():
    seed = int(os.environ.get("DIAG_SEED", "1000"))
    snr = float(os.environ.get("DIAG_SNR", "30"))
    nep = int(os.environ.get("DIAG_NEP", "100"))

    rng = np.random.default_rng(seed)
    wd = Path(tempfile.mkdtemp(prefix="diag_transit_"))
    os.chdir(wd)
    config, params, sampler, truth, checks = ir.make_transit(rng, snr, nep, wd)
    print(
        "truth: " + "  ".join("%s=%.6g" % (k, v) for k, v in truth.items()),
        flush=True,
    )

    d = np.loadtxt(wd / "synth.trn")
    depth = 1.0 - d[:, 1].min()
    print(
        "data: n=%d depth=%.5f sigma=%.3e in-transit=%d"
        % (len(d), depth, d[0, 2], int((d[:, 1] < 1 - 2 * d[0, 2]).sum())),
        flush=True,
    )

    import arviz as az
    import pymc as pm

    from exozippy.system import System

    system = System(dict(config), user_params=dict(params))
    system.prepare()
    model = system.build_model()
    print("free RVs: %s" % sorted(v.name for v in model.free_RVs), flush=True)

    with model:
        idata = pm.sample(
            draws=sampler["draws"],
            tune=sampler["tune"],
            chains=sampler["chains"],
            cores=sampler["cores"],
            target_accept=sampler["target_accept"],
            progressbar=False,
            random_seed=seed,
        )

    post = idata.posterior
    ss = idata.sample_stats
    print(
        "\ndivergences per chain: %s"
        % np.asarray(ss["diverging"]).sum(axis=1).tolist(),
        flush=True,
    )
    if "step_size" in ss:
        print(
            "step size per chain: %s"
            % [float(x) for x in np.asarray(ss["step_size"])[:, -1]],
            flush=True,
        )

    print(
        "\n=== per-chain ranges (the discriminating measurement) ===",
        flush=True,
    )
    print(
        "%-22s %6s %12s %12s %12s %12s"
        % ("variable", "chain", "mean", "sd", "min", "max")
    )
    for name in sorted(post.data_vars):
        arr = np.asarray(post[name])
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)[:, :, 0]
        if not np.isfinite(arr).all():
            continue
        spread = arr.mean(axis=1).max() - arr.mean(axis=1).min()
        within = float(np.median(arr.std(axis=1)))
        # Only print the ones that are actually misbehaving, plus a couple of
        # well-behaved references, or the output is unreadable.
        try:
            ess = float(az.ess(post[name]).to_array().min())
        except Exception:  # noqa: BLE001
            ess = float("nan")
        if ess > 500 and name not in ("orbit.cosi", "orbit.period"):
            continue
        print(
            "  %-20s ESS=%.0f  between-chain mean spread=%.4g  "
            "typical within-chain sd=%.4g  ratio=%.2f"
            % (
                name,
                ess,
                spread,
                within,
                spread / within if within else float("nan"),
            ),
            flush=True,
        )
        for c in range(arr.shape[0]):
            print(
                "  %-20s %6d %12.6g %12.4g %12.6g %12.6g"
                % (
                    "",
                    c,
                    arr[c].mean(),
                    arr[c].std(),
                    arr[c].min(),
                    arr[c].max(),
                ),
                flush=True,
            )

    print(
        "\n=== correlations with the worst-mixing sampled coordinate ===",
        flush=True,
    )
    flat = {}
    for name in post.data_vars:
        a = np.asarray(post[name])
        if a.ndim > 2:
            a = a.reshape(a.shape[0], a.shape[1], -1)[:, :, 0]
        flat[name] = a.reshape(-1)
    target = "orbit.cosi" if "orbit.cosi" in flat else None
    if target:
        base = flat[target]
        rows = []
        for name, v in flat.items():
            if name == target or v.shape != base.shape:
                continue
            if v.std() == 0 or base.std() == 0:
                continue
            rows.append((abs(np.corrcoef(base, v)[0, 1]), name))
        for r, name in sorted(rows, reverse=True)[:8]:
            print("  |corr(%s, %-24s)| = %.3f" % (target, name, r), flush=True)


if __name__ == "__main__":
    main()
