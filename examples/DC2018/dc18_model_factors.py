"""Print what a built model's logp is actually made of: how many free RVs,
observed RVs, potentials and deterministics, and whether the named factors
sum to model.logp.  A decomposition that cannot account for the total is
not a decomposition (dc18_logp_terms.py's first two runs both failed here).
"""

import argparse
import os

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    from exozippy.system import System

    run_dir = os.path.abspath(args.run_dir)
    pre = os.path.join(run_dir, f"DC2018_{args.event}")
    os.chdir(run_dir)
    config = yaml.safe_load(open(pre + ".yaml"))
    user_params = yaml.safe_load(open(config["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()

    print(f"free_RVs      : {len(model.free_RVs)}")
    print(f"observed_RVs  : {len(model.observed_RVs)}")
    print(f"potentials    : {len(model.potentials)}")
    print(f"deterministics: {len(model.deterministics)}")
    print("\npotential names:")
    for p in model.potentials:
        print(f"  {p.name}")
    print("\nobserved names:")
    for p in model.observed_RVs:
        print(f"  {p.name}")

    factors = model.logp(sum=False)
    print(f"\nmodel.logp(sum=False) returns {len(factors)} factors")
    ip = model.initial_point()
    import pytensor

    fn = pytensor.function(
        model.value_vars,
        [f.sum() for f in factors] + [model.logp(sum=True)],
        on_unused_input="ignore",
    )
    vals = [np.asarray(ip[v.name], dtype=float) for v in model.value_vars]
    out = [float(np.sum(x)) for x in fn(*vals)]
    print(f"sum of factors = {sum(out[:-1]):.4f}   total = {out[-1]:.4f}")
    print(f"difference     = {sum(out[:-1]) - out[-1]:+.6f}")
    print("\nfactor values at the initial point:")
    for f, v in zip(factors, out[:-1]):
        print(f"  {str(f):<60.60} {v:16.4f}")


if __name__ == "__main__":
    main()
