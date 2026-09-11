"""One dynesty (nlive, sample, bound) combination, scored against TRUTH.

WHY.  The first dynesty run on DC2018-128's d=27 observable arm terminated
in 1.6 h and landed up to 40 SIGMA from truth with absurdly small errors --
a collapsed live-point set, not efficient exploration (review 8.2.3).  The
wrapper hardcoded `sample="rwalk"`, which dynesty recommends only for
10 <= ndim <= 20, and nlive=500 is thin for multi-ellipsoid bounding at
d=27 (the usual guidance is ~50*ndim).  So the open question is whether
that collapse indicts the BACKEND or the DEFAULTS.

THE SCORE IS THE TRUTH PULL, not the wall clock.  A fast wrong answer is
worse than a slow right one, and the previous round's mistake was reading
speed as success.  Only the LIGHT-CURVE OBSERVABLES count: t_0, u_0, log_s,
log_q and log_rho are what the data constrain.  theta_E/pi_rel/mass need
theta_star, which this arm has no SED to supply, so they inherit 8.6.7's
mu_rel error whatever the sampler does and are reported but EXCLUDED.
"""

import argparse
import io
import json
import logging
import os
import time

import numpy as np
import yaml

logging.disable(logging.WARNING)

# dc128_truth_forward.json.  Names are the POST-#246 trace names; an earlier
# check keyed source.rho and mulensevent.t_E and got "not in trace" for
# both, which is a naming slip rather than missing data.
TRUTH = {
    "source.t_0": 2458554.8868815,
    "source.u_0": 0.141832,
    "lens.log_s": float(np.log10(0.993145)),
    "planet.log_q": float(np.log10(0.0012118)),
    "source.log_rho": float(np.log10(0.006066783367838937)),
}
CONTEXT = {  # reported, never scored -- needs an SED (8.6.7)
    "mulensevent.log_theta_E": float(np.log10(0.0904642645231668)),
    "mulensevent.log_pi_rel": float(np.log10(0.0022107638807915136)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/DC2018_128_tightpriors.yaml")
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--bound", default=None)
    ap.add_argument("--walks", type=int, default=None)
    ap.add_argument(
        "--ncpu", type=int, default=int(os.environ.get("NSLOTS", 60))
    )
    ap.add_argument("--dlogz", type=float, default=0.5)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from exozippy.samplers.nested import nested_sample
    from exozippy.system import System

    cfg = yaml.safe_load(io.open(args.config, encoding="utf-8"))
    cfgdir = os.path.dirname(os.path.abspath(args.config))
    pf = os.path.join(cfgdir, cfg.get("parameter_file", ""))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    cwd = os.getcwd()
    os.chdir(cfgdir)
    system = System(cfg, yaml.safe_load(io.open(pf, encoding="utf-8")))
    system.prepare()
    model = system.build_model()
    os.chdir(cwd)

    t0 = time.time()
    idata = nested_sample(
        model,
        system,
        backend="dynesty",
        nlive=args.nlive,
        sample=args.sample,
        bound=args.bound,
        walks=args.walks,
        dlogz=args.dlogz,
        cores=args.ncpu,
        seed=11,
    )
    wall = time.time() - t0

    a = idata.posterior.attrs
    post = idata.posterior
    rec, pulls = {}, []
    for name, truth in list(TRUTH.items()) + list(CONTEXT.items()):
        scored = name in TRUTH
        if name not in post.data_vars:
            rec[name] = {
                "truth": truth,
                "status": "not in trace",
                "scored": scored,
            }
            continue
        v = np.asarray(post[name]).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        med, sd = float(np.median(v)), float(np.std(v))
        pull = (med - truth) / sd if sd > 0 else float("nan")
        rec[name] = {
            "truth": truth,
            "median": med,
            "sd": sd,
            "pull": pull,
            "scored": scored,
        }
        if scored and np.isfinite(pull):
            pulls.append(abs(pull))

    out = {
        "tag": args.tag,
        "nlive": args.nlive,
        "sample": args.sample or "auto-by-ndim",
        "bound": args.bound or "multi",
        "walks": args.walks,
        "wall_s": wall,
        "logz": float(a["nested_logz"]),
        "logzerr": float(a["nested_logzerr"]),
        "ncall": int(a["nested_ncall"]),
        "n_eff": int(a["nested_n_eff"]),
        "max_abs_pull": float(max(pulls)) if pulls else None,
        "median_abs_pull": float(np.median(pulls)) if pulls else None,
        "n_scored": len(pulls),
        "params": rec,
    }
    print(json.dumps(out, indent=2), flush=True)
    dest = args.out or "dynsweep_%s.json" % args.tag
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote %s" % dest, flush=True)


if __name__ == "__main__":
    main()
