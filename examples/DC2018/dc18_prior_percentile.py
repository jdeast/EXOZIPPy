"""Review 7.7.3 (1): IS THE PRIOR BEHAVING?  For each sweep event, where
does the simulation truth sit in OUR PRIOR's marginal (a percentile per
physical parameter: mu_rel, pi_rel, D_lens, D_source, M_lens), and how far
did the posterior pull from the truth (sigma pull from the fit's
results.csv)?  A correctly applied prior shrinks in proportion to how far
out the truth is; a wrongly applied one (sign error, missing Jacobian,
double count) pulls every event the same way.

THE PRIOR ALONE is sampled by building the event's model exactly as the
fit did (same config, params, seeds) and keeping only the prior terms --
every free parameter's own prior plus the galactic-model and stellar
potentials -- while dropping every OBSERVED term (the light curves) and
every data-derived constraint (the SED source-flux tie, the microlensing
model's own potentials that encode the measured geometry).  Which terms
are kept is decided by NAME and printed, so a reader can check the split;
the physical observables (mulensevent.mu_rel_mag, mulensevent.pi_rel) are
compiled from the same value vector the sampler walks, so the percentile
is read in the coordinates the fit reports.

The sampler is emcee on the compiled prior logp (the model's VBM Op never
enters the kept terms, so a step costs microseconds).

    python dc18_prior_percentile.py --run-dir sweep4/047 --event 47
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dc18_common as C  # noqa: E402
import dc18_truth_table as T  # noqa: E402

# (label, truth key in dc18_truth_table.truth_for, model deterministic /
# parameter name, results.csv name)
PHYS = [
    (
        "mu_rel",
        "murel",
        "mulensevent.mu_rel_mag",
        "mulensevent",
        "mu_rel_mag",
        None,
    ),
    ("pi_rel", "pi_rel", "mulensevent.pi_rel", "mulensevent", "pi_rel", None),
    ("D_lens", "Dl", "star.Lens.distance", "star", "distance", "Lens"),
    ("D_source", "Ds", "star.Source.distance", "star", "distance", "Source"),
    ("M_lens", "Ml", "star.Lens.mass", "star", "mass", "Lens"),
]

# Terms whose name carries one of these are DATA: dropped.  ONLY the light
# curves.  Everything else in the model is prior or measure: the galactic
# event-rate prior and the fitpirel / fitthetae Jacobians live under
# `mulensevent.*` (the fit samples log_pi_rel and log_theta_E, so the lens
# distance and mass are DERIVED and their prior is exactly those terms), the
# trajectory and geometry coordinates' uniform priors under `source.*` and
# `lens.*`, the SED floor priors under `sed.*`.  A first version dropped all
# of those as "data-derived" and sampled a different measure: the lens mass
# marginal came out at 0.7-1.4 Msun and the IMF-only control ran to the
# bounds (distances of 10 pc).  The mulensinstrument terms carry the data
# (the hogg RVs), the flux nuisances and the SED source-flux tie.
DATA_MARKERS = ("mulensinstrument",)


KEEP_REGEX = None  # set from --keep: keep ONLY the terms matching it


def keep_term(name):
    if KEEP_REGEX is not None:
        return KEEP_REGEX.search(name) is not None
    return not any(m in name for m in DATA_MARKERS)


def read_results_all(path):
    """{parname: (value, up_err, low_err)} for the 'all' mode of a
    *_results.csv -- every row, not the mapped subset dc18_common keeps."""
    import csv

    out = {}
    with open(path, newline="") as fh:
        first = fh.readline()
        hdr = [c.strip() for c in first.lstrip("#").split(",") if c.strip()]
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith("#")]
    for r in rows:
        d = dict(zip(hdr, [c.strip() for c in r]))
        if d.get("mode", "all") != "all":
            continue
        try:
            out[d["parname"]] = tuple(
                float(d[k]) for k in ("value", "up_err", "low_err")
            )
        except (KeyError, ValueError):
            continue
    return out


def build(run_dir, config_json=None, params_json=None):
    import pytensor

    from exozippy.system import System

    pre = os.path.join(
        run_dir,
        [
            f
            for f in os.listdir(run_dir)
            if f.endswith(".yaml")
            and not f.endswith(".sed.yaml")
            and ".params" not in f
        ][0][:-5],
    )
    os.chdir(run_dir)
    config = yaml.safe_load(open(pre + ".yaml"))
    config.pop("sampler", None)
    # The sweep4 configs predate the MMEXOFAST removal (2026-10-01); the
    # seeding keys are rejected at build and are irrelevant to the prior.
    for ev in config.get("mulensevent", []):
        for k in ("mmexofast", "mmexofast_options"):
            ev.pop(k, None)
    user_params = yaml.safe_load(open(config["parameter_file"]))
    if config_json:
        for k, v in config_json.items():
            print(f"   control: config[{k!r}] <- {v!r}")
            config[k] = v
    if params_json:
        for k, v in params_json.items():
            print(f"   control: params[{k!r}] <- {v!r}")
            user_params[k] = v
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    vv = list(model.value_vars)
    kept, dropped = [], []
    for rv in list(model.free_RVs) + list(model.potentials):
        (kept if keep_term(rv.name) else dropped).append(rv)
    for rv in model.observed_RVs:
        dropped.append(rv)
    logp = model.logp(vars=kept, sum=True)
    f = pytensor.function(vv, logp, on_unused_input="ignore")
    # Each observable is read through its component's Parameter.value
    # tensor (vector over elements; the element is looked up by name), the
    # same tensor the likelihood consumes, so the percentile is in the
    # fit's own coordinates.
    outs = []
    for _, _, _, comp, param, elem in PHYS:
        c = getattr(system, comp)
        idx = 0 if elem is None else list(c.names).index(elem)
        outs.append(getattr(c, param).value[idx])
    g = pytensor.function(
        vv, model.replace_rvs_by_values(outs), on_unused_input="ignore"
    )
    return system, model, vv, f, g, kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--walkers", type=int, default=0, help="default 4 x dims")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--keep",
        default=None,
        help="regex: keep ONLY the terms whose name matches (default: every "
        "term that is not data or data-derived, see DATA_MARKERS)",
    )
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument(
        "--config-json",
        default=None,
        help="JSON object merged over the config's top-level keys (a control "
        "that swaps a component block, e.g. mann with an observed Ks)",
    )
    ap.add_argument(
        "--params-json",
        default=None,
        help="JSON object merged into the params file (a control that pins "
        "a parameter, e.g. star.Lens.feh with sigma 0)",
    )
    a = ap.parse_args()
    global KEEP_REGEX
    if a.keep:
        import re

        KEEP_REGEX = re.compile(a.keep)
    import importlib.util

    import emcee

    spec = importlib.util.spec_from_file_location(
        "mmf",
        os.path.join(HERE, "..", "..", "scripts", "make_mulens_fixtures.py"),
    )
    mmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mmf)
    data_dir = C.data_dir_or_raise(a.data_dir)
    truth, cls = T.truth_for(a.event, data_dir)
    run_dir = os.path.abspath(a.run_dir)
    csvs = [f for f in os.listdir(run_dir) if f.endswith("_results.csv")]
    post = read_results_all(os.path.join(run_dir, csvs[0])) if csvs else {}

    system, model, vv, f, g, kept, dropped = build(
        run_dir,
        config_json=json.loads(a.config_json) if a.config_json else None,
        params_json=json.loads(a.params_json) if a.params_json else None,
    )
    print(
        f"event {a.event} ({cls}): prior terms KEPT ({len(kept)}): "
        + ", ".join(sorted(r.name for r in kept))
    )
    print(
        f"   DROPPED ({len(dropped)}): "
        + ", ".join(sorted(r.name for r in dropped))
    )
    start = mmf.raw_start(system, model)
    x0 = np.concatenate(
        [
            np.atleast_1d(np.asarray(start[v.name], dtype=float)).ravel()
            for v in vv
        ]
    )
    shapes = [np.shape(np.asarray(start[v.name])) for v in vv]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]

    def unpack(x):
        out, k = [], 0
        for s, n in zip(shapes, sizes):
            out.append(
                np.asarray(x[k : k + n]).reshape(s) if s else float(x[k])
            )
            k += n
        return out

    def lnprob(x):
        v = float(f(*unpack(x)))
        return v if np.isfinite(v) else -np.inf

    ndim = x0.size
    nw = a.walkers or 4 * ndim
    rng = np.random.default_rng(a.seed)
    p0 = x0 + 0.01 * rng.standard_normal((nw, ndim))
    lp0 = lnprob(x0)
    print(
        f"   dims {ndim}, walkers {nw}, steps {a.steps}; prior logp at the fit's start {lp0:.2f}"
    )
    sampler = emcee.EnsembleSampler(nw, ndim, lnprob)
    sampler.run_mcmc(p0, a.steps, progress=False)
    chain = sampler.get_chain(discard=a.burn, flat=True)
    try:
        tau = sampler.get_autocorr_time(discard=a.burn, tol=0)
        print(
            f"   autocorr time: max {np.max(tau):.0f} steps over {a.steps - a.burn} kept (x {nw} walkers)"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"   autocorr time unavailable: {exc}")
    vals = np.array(
        [
            np.array([float(np.ravel(y)[0]) for y in g(*unpack(x))])
            for x in chain[:: max(1, chain.shape[0] // 20000)]
        ]
    )
    print(f"   {vals.shape[0]} prior draws")
    rows = []
    print(
        f"   {'parameter':10s} {'truth':>10s} {'prior pctile':>12s} {'prior 16/50/84':>30s} {'fit median':>11s} {'pull (sigma)':>12s}"
    )
    for (label, tkey, name, _, _, _), col in zip(PHYS, vals.T):
        tv = float(truth[tkey])
        pct = 100.0 * float(np.mean(col < tv))
        q16, q50, q84 = np.percentile(col, [16, 50, 84])
        fit = post.get(name)
        pull = None
        if fit is not None and fit[0] is not None:
            pull = C.sigma_pull(tv, fit[0], fit[1], fit[2])
        fit_s = (
            f"{fit[0]:11.4g}" if fit and fit[0] is not None else f"{'--':>11s}"
        )
        pull_s = f"{pull:12.2f}" if pull is not None else f"{'--':>12s}"
        print(
            f"   {label:10s} {tv:10.4g} {pct:11.1f}% {q16:9.4g} {q50:9.4g} {q84:9.4g}  {fit_s} {pull_s}"
        )
        rows.append(
            {
                "param": label,
                "truth": tv,
                "prior_percentile": pct,
                "prior_q16": float(q16),
                "prior_q50": float(q50),
                "prior_q84": float(q84),
                "fit": None if not fit else fit[0],
                "fit_up": None if not fit else fit[1],
                "fit_lo": None if not fit else fit[2],
                "pull": pull,
            }
        )
    out = os.path.join(run_dir, f"prior_percentile_{a.event:03d}{a.tag}.json")
    np.savez(
        os.path.join(run_dir, f"prior_percentile_{a.event:03d}{a.tag}.npz"),
        names=np.array([p[0] for p in PHYS]),
        draws=vals,
    )
    json.dump(
        {
            "event": a.event,
            "class": cls,
            "kept": sorted(r.name for r in kept),
            "dropped": sorted(r.name for r in dropped),
            "rows": rows,
        },
        open(out, "w"),
        indent=1,
    )
    print(f"   wrote {out}")


if __name__ == "__main__":
    main()
