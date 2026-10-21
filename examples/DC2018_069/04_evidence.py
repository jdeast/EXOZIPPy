"""Stage 4: per run, offline from its trace -- normalized lnZ and the flags.

    python 04_evidence.py m0 m1 ... [1L1S_fs] [--burn-frac 0.5]

For each run directory under $DC069_RUNS/modes/ it rebuilds the model the
run sampled (same config, same params file), restores the run's whitening
(never re-measured: raw draws decode only under the whitening they were
sampled with), drops the first `--burn-frac` of the draws, and writes
$DC069_RUNS/evidence/<run>.json with

  * lnZ +/- err, NORMALIZED: estimate_mode_evidences(log_norm=
    System.evidence_log_norm()).  Every draw belongs to THIS run's mode; if
    identify_modes splits the run into sub-modes, each gets its own bridge
    and the run's lnZ is their logsumexp (blind.lnz_from_submodes).
  * re2 and the bridge status (ok / partial / refused),
  * stuck fraction, lp split-Rhat, drift out of the seeded basin,
  * lnL_max (the observed-data log likelihood, max over the top-lp draws),
  * median and 68% interval of s, q, alpha, t_0, u_0, t_E, rho (user units),
  * the (log s, log q, alpha, sign u_0) basin features 05_report dedupes on.

Light work: run it under `srun -p test -c 16 --mem=64G`.  The wrap-up of a
run computes per-mode evidence only for a MULTIMODAL posterior (and then
over that run's own modes only), which is why this stage exists.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind as B  # noqa: E402

# WORKAROUND (gap for the driver): outputs/evidence.py sizes its logp pool
# with mp.cpu_count(), which inside a Slurm allocation is the NODE's core
# count (112 on sapphire/test nodes), not the grant -- gridsearch's
# default_grid_cores documents the same trap.  Use the affinity mask.
multiprocessing.cpu_count = lambda: len(os.sched_getaffinity(0))

# Trace variable -> (label, element).  The companion is the LAST element
# of `lens` (element 0 is the primary).  A finished trace is in USER units
# (run.py's post-save step converts it and stamps it, trace_meta.py), so
# no conversion happens here: alpha comes back in degrees, t_0 in BJD.
VARS = {
    "s": ("lens.s", -1),
    "log_s": ("lens.log_s", -1),
    "alpha": ("lens.alpha", -1),
    "log_q": ("planet.log_q", -1),
    "q": ("planet.q", -1),
    "t_0": ("source.t_0", 0),
    "u_0": ("source.u_0", 0),
    "t_E": ("mulensevent.t_E", 0),
    "rho": ("source.rho", 0),
}


def draws(post, key):
    """(chain, draw) user-unit draws of one element, or None."""
    label, el = VARS[key]
    if label not in post.data_vars:
        return None
    a = np.asarray(post[label].values, dtype=float)
    if a.ndim == 3:
        a = a[..., el if el >= 0 else a.shape[2] + el]
    return a


def lnl_max(model, post, lp, top=32):
    """Observed-data log likelihood, maximized over the `top` best-lp draws.

    Gaussian DC2018 fits put the whole light-curve likelihood in observed
    RVs (pm.Normal), so this is chi2-equivalent: -2 lnL_max is the best
    chi2 plus a constant from the error bars.
    """
    import pytensor

    if not model.observed_RVs:
        # e.g. a hogg-mixture fit, whose likelihood is a Potential.
        return float("nan")
    f = pytensor.function(
        model.value_vars,
        model.logp(vars=model.observed_RVs, sum=True),
        on_unused_input="ignore",
    )
    ip = model.initial_point()
    names = [v.name for v in model.value_vars]
    flat = np.argsort(lp, axis=None)[::-1][:top]
    best = -np.inf
    for k in flat:
        c, d = np.unravel_index(k, lp.shape)
        vals = [
            np.asarray(post[n].values[c, d], dtype=float).reshape(
                np.shape(ip[n])
            )
            for n in names
        ]
        best = max(best, float(f(*vals)))
    return best


def one_run(name, burn_frac, max_draws):
    import arviz as az
    import yaml

    from exozippy.outputs.evidence import estimate_mode_evidences
    from exozippy.outputs.modes import identify_modes
    from exozippy.system import System
    from exozippy.whitening import restore_whitening_for_trace

    d = B.run_dir("modes", name)
    # The config is DC2018_069_<arch>.yaml: a stem with no dot.  The params
    # file, the SED file and the run's restart file (.params.2.yaml) all
    # carry one.
    cfg_path = next(
        p for p in sorted(d.glob(B.NAME + "_*.yaml")) if "." not in p.stem
    )
    arch = cfg_path.stem.split("_", 2)[-1]
    cfg = yaml.safe_load(open(cfg_path))
    prefix = cfg["prefix"]
    params = yaml.safe_load(open(cfg["parameter_file"]))
    sampler = cfg.get("sampler", {})
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    os.chdir(d)
    t0 = time.time()
    system = System(cfg, user_params=params)
    system.prepare()
    model = system.build_model()
    restore_whitening_for_trace(
        system, prefix + "_whitening.json", prefix + "_trace.nc"
    )
    log_norm = float(system.evidence_log_norm())

    full = az.from_netcdf(prefix + "_trace.nc")
    n_draw = full.posterior.sizes["draw"]
    start = int(burn_frac * n_draw)
    # Only the T=1 groups (the hot rungs are not this run's posterior);
    # arviz 1.x InferenceData is an xarray DataTree.
    import xarray as xr

    idata = xr.DataTree.from_dict(
        {
            "posterior": full["posterior"]
            .to_dataset()
            .isel(draw=slice(start, None)),
            "sample_stats": full["sample_stats"]
            .to_dataset()
            .isel(draw=slice(start, None)),
        }
    )
    post = idata.posterior
    lp = np.asarray(idata.sample_stats["lp"].values, dtype=float)

    rep = identify_modes(idata, attach=True)
    ev = estimate_mode_evidences(
        model, idata, rep, max_posterior_draws=max_draws, log_norm=log_norm
    )
    sub = [
        {"mode": int(e.mode), "lnZ": float(e.lnZ),
         "lnZ_err": float(e.lnZ_err), "re2": float(e.re2),
         "refused": bool(e.refused), "reason": e.reason,
         "state": e.state, "n_post": int(e.n_post),
         "weight_occupancy": float(rep.weights[e.mode])}
        for e in ev
    ]  # fmt: skip
    tot = B.lnz_from_submodes(sub)

    out = {
        "mode": name,
        "arch": arch,
        "config": str(cfg_path),
        "n_chain": int(post.sizes["chain"]),
        "n_draw_used": int(post.sizes["draw"]),
        "burn_frac": burn_frac,
        "draws_requested": sampler.get("draws"),
        "hit_maxtime": bool(n_draw < int(sampler.get("draws") or 0)),
        "n_temps": sampler.get("n_temps"),
        "log_norm": log_norm,
        "lnZ": tot["lnZ"],
        "lnZ_err": tot["lnZ_err"],
        "re2": tot["re2"],
        "bridge_status": tot["status"],
        "lnZ_indicative": tot["lnZ_indicative"],
        "submodes": sub,
        "lp_max": float(lp.max()),
        "lp_rhat": B.split_rhat(lp),
        "stuck_frac": B.stuck_fraction(lp),
        "lnL_max": lnl_max(model, post, lp),
    }
    summ = {}
    vals = {k: draws(post, k) for k in VARS}
    if vals["s"] is None and vals["log_s"] is not None:
        vals["s"] = 10.0 ** vals["log_s"]
    if vals["q"] is None and vals["log_q"] is not None:
        vals["q"] = 10.0 ** vals["log_q"]
    for k in B.PARAMS:
        v = vals.get(k)
        if v is None:
            continue
        v = np.ravel(v)
        if k == "alpha":
            # Circular: summarize around the circular mean.
            a0 = B.circ_mean_deg(v)
            v = a0 + B.wrap180(v - a0)
        lo, med, hi = np.percentile(v, [15.865, 50.0, 84.135])
        summ[k] = [float(med), float(lo), float(hi)]
    out["summary"] = summ
    if arch == B.ARCH_2L1S:
        out["features"] = B.basin_features(
            np.log10(vals["s"]), np.log10(vals["q"]), vals["alpha"],
            vals["u_0"],
        )  # fmt: skip
        cpath = B.run_dir("modes", "candidates.json")
        cands = json.load(open(cpath)) if cpath.exists() else {}
        seed = next(
            (c for c in cands.get("candidates", []) if c["mode"] == name),
            None,
        )
        if seed is None:
            # A run 02_seeds.py did not write (e.g. an older single fit):
            # no seeded basin to drift from.
            out["origin"] = "not a stage-2 mode (no seed recorded)"
            out["drift"] = ""
        else:
            out["origin"] = seed["origin"]
            out["seed"] = {
                k: seed[k]
                for k in ("log_s", "log_q", "alpha", "u_0", "t_0", "dchi2")
            }
            out["drift"] = B.drift(
                seed,
                out["features"]["mean"][0],
                out["features"]["mean"][1],
                out["features"]["u0_sign"],
                cands["d_log_s"],
                cands["d_log_q"],
            )
    else:
        out["origin"] = "architecture baseline (no companion)"
        out["drift"] = ""
    out["flags"] = B.reliability_flags(out)
    out["evidence_wall_s"] = time.time() - t0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run names under modes/")
    ap.add_argument("--burn-frac", type=float, default=0.5)
    ap.add_argument("--max-draws", type=int, default=4000)
    a = ap.parse_args(argv)
    outdir = B.run_dir("evidence")
    outdir.mkdir(parents=True, exist_ok=True)
    for name in a.runs:
        res = one_run(name, a.burn_frac, a.max_draws)
        B.dump_json(res, outdir / ("%s.json" % name))
        print(
            "%s: lnZ = %.2f +/- %.2f (log_norm %+.2f, %s, re2 %.3g)  "
            "lnL_max %.2f  flags: %s"
            % (name, res["lnZ"], res["lnZ_err"], res["log_norm"],
               res["bridge_status"], res["re2"], res["lnL_max"],
               "; ".join(res["flags"]) or "none"),
            flush=True,
        )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
