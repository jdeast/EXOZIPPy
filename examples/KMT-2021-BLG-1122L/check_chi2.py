"""chi2 of the shipped start, per light curve, from the likelihood node
itself (observed, mu, sigma), so the data logp is not misread through an
assumed normalization."""

import importlib.util
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import pytensor

    from exozippy.system import System

    spec = importlib.util.spec_from_file_location(
        "mmf",
        os.path.join(HERE, "..", "..", "scripts", "make_mulens_fixtures.py"),
    )
    mmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mmf)
    os.chdir(HERE)
    cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    prm = yaml.safe_load(open("KMT-2021-BLG-1122L.params.yaml"))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    system = System(cfg, user_params=prm)
    system.prepare()
    model = system.build_model()
    start = mmf.raw_start(system, model)
    rv = model["mulensinstrument.model"]
    mu, sigma = rv.owner.op.dist_params(rv.owner)
    obs = model.rvs_to_values[rv]
    f = pytensor.function(
        list(model.value_vars),
        model.replace_rvs_by_values([mu, sigma, obs]),
        on_unused_input="ignore",
    )
    mu_v, sig_v, obs_v = [
        np.asarray(x)
        for x in f(*[np.asarray(start[v.name]) for v in model.value_vars])
    ]
    r2 = ((obs_v - mu_v) / sig_v) ** 2
    norm = -np.log(sig_v) - 0.5 * np.log(2 * np.pi)
    print(
        f"N {r2.size}  chi2 {r2.sum():.1f}  (per point {r2.mean():.2f})  normalization {norm.sum():.1f}  logp {(-0.5 * r2 + norm).sum():.1f}"
    )
    bounds = np.cumsum([0, 1346, 676, 412])
    for i, lab in enumerate(("KMTC14", "KMTS14", "KMTA14")):
        s = slice(bounds[i], bounds[i + 1])
        print(
            f"   {lab}: n {bounds[i + 1] - bounds[i]}  chi2 {r2[s].sum():.1f}  per point {r2[s].mean():.2f}  median sigma {np.median(sig_v[s]):.3e}  obs median {np.median(obs_v[s]):.3e}"
        )
    parts, total, ok, _ = mmf.decompose(system, model, start)
    print(
        "decompose data term:",
        {
            k: round(float(v), 1)
            for k, v in parts.items()
            if "mulensinstrument" in k and "model" in k
        },
        "total",
        round(float(total), 1),
    )


if __name__ == "__main__":
    main()
