"""Full-model (d=27) nested sampling pilot on DC2018 event 128.

Drives exozippy.samplers.nested (the `method: nested` machinery) directly on
the event-128 build, so the pilot and the production path are ONE
implementation.  See that module's docstring for the bridge and the u-space
decomposition; this script only adds the event-specific scoring: posterior
mass near the two known solutions (s = 0.976 and its s <-> 1/s mirror at
0.863), which the seeded PT runs and the d=7 profile-likelihood benchmark
established.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

RUN = Path(
    "/home/jeastman/python/EXOZIPPy/examples/DC2018/events_ladderfix/128"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="dynesty")
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--walks", type=int, default=None)
    ap.add_argument("--dlogz", type=float, default=0.5)
    ap.add_argument("--ncpu", type=int, default=60)
    ap.add_argument("--out", default="dc18_fullns_pilot.json")
    ap.add_argument("--config", default="DC2018_128.yaml")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    out_path = Path(args.out).resolve()

    os.chdir(RUN)
    from exozippy.samplers.nested import nested_sample
    from exozippy.system import System

    config = yaml.safe_load(open(args.config))
    pf = config.get("parameter_file", "DC2018_128.params.yaml")
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    params = yaml.safe_load(open(pf))
    system = System(config, params)
    system.prepare()
    model = system.build_model()

    idata = nested_sample(
        model,
        system,
        backend=args.backend,
        nlive=args.nlive,
        walks=args.walks,
        dlogz=args.dlogz,
        cores=args.ncpu,
        seed=11,
        checkpoint_dir=args.ckpt,
    )

    a = idata.posterior.attrs
    log_s = np.asarray(idata.posterior["lens.log_s"]).ravel()
    lp = np.asarray(idata.sample_stats["lp"]).ravel()
    best = int(np.argmax(lp))
    out = {
        "backend": args.backend,
        "logz": float(a["nested_logz"]),
        "logzerr": float(a["nested_logzerr"]),
        "ncall": int(a["nested_ncall"]),
        "n_eff": int(a["nested_n_eff"]),
        "best_lp": float(lp[best]),
        "best_log_s": float(log_s[best]),
        "mass_main": float(np.mean(np.abs(log_s - (-0.0104)) < 0.02)),
        "mass_mirror": float(np.mean(np.abs(log_s - (-0.0640)) < 0.02)),
    }
    print(json.dumps(out, indent=2), flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
