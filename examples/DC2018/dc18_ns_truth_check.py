"""Did dynesty get the right answer, or just a fast one?

nsdiag (job 15426023) terminated in 1.59 h with n_eff = 7007 where
ultranest has spent 5.4 days on the same config and not finished.  That is
a 55-180x difference, which is large enough to deserve suspicion rather
than celebration: a fast wrong answer is worse than a slow right one.
dynesty's best_lp also came in 100 nats BELOW ultranest's Lmax (86,876.6 vs
86,976.4), which is the expected shape for a sampler that is not an
optimizer but needs checking against truth rather than against a rival.

WHAT SHOULD AND SHOULD NOT BE RECOVERED.  This is the OBSERVABLE arm and
carries no SED, so:
  * the LIGHT-CURVE observables -- t_0, u_0, t_E, rho, s, q -- are what the
    data actually constrain and must come out right.  This is the test.
  * theta_E, pi_rel and the lens mass/distance need theta_star, which only
    an SED supplies, so they inherit the mu_rel problem no matter which
    sampler runs (review 8.6.7).  Their being wrong here says nothing about
    dynesty.
Resumes from the existing checkpoint, so it costs minutes, not hours.
"""

import argparse
import io
import json
import logging
import os

import numpy as np
import yaml

logging.disable(logging.WARNING)

# dc128_truth_forward.json.  Keyed by the TRACE variable name; a tuple gives
# (truth, transform) where the trace stores a log coordinate.
TRUTH = {
    "source.t_0": 2458554.8868815,
    "source.u_0": 0.141832,
    "mulensevent.t_E": 18.234393055542185,
    "source.rho": 0.006066783367838937,
    "lens.log_s": np.log10(0.993145),
    "planet.log_q": np.log10(0.0012117999999999996),
    "mulensevent.theta_E": 0.0904642645231668,
    "mulensevent.pi_rel": 0.0022107638807915136,
}
LC_OBSERVABLES = {
    "source.t_0",
    "source.u_0",
    "mulensevent.t_E",
    "source.rho",
    "lens.log_s",
    "planet.log_q",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/DC2018_128_tightpriors.yaml")
    ap.add_argument("--ckpt", default="dc18_ns_dynesty_ckpt")
    ap.add_argument("--backend", default="dynesty")
    ap.add_argument("--nlive", type=int, default=500)
    ap.add_argument("--ncpu", type=int, default=60)
    ap.add_argument("--out", default="dc18_ns_truth_check.json")
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

    idata = nested_sample(
        model,
        system,
        backend=args.backend,
        nlive=args.nlive,
        cores=args.ncpu,
        seed=11,
        checkpoint_dir=args.ckpt,
    )
    # Keep the trace (see the note in dc18_dynesty_sweep.py): a summary
    # cannot be rescored when the metric turns out to be wrong.
    try:
        idata.to_netcdf(os.path.abspath("ns_truth_check_trace.nc"))
        print("saved trace -> ns_truth_check_trace.nc", flush=True)
    except Exception as e:  # noqa: BLE001
        print("WARNING: trace not saved (%s)" % type(e).__name__, flush=True)

    post = idata.posterior
    print(
        "posterior vars: %d   draws: %s"
        % (len(post.data_vars), dict(post.sizes)),
        flush=True,
    )

    rows, out = [], {}
    print(
        "\n%-24s %12s %12s %12s %9s  %s"
        % ("parameter", "truth", "median", "sd", "pull", "role"),
        flush=True,
    )
    for name, truth in TRUTH.items():
        if name not in post.data_vars:
            print(
                "%-24s %12.6g %12s  (not in trace)" % (name, truth, "--"),
                flush=True,
            )
            continue
        a = np.asarray(post[name]).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        med, sd = float(np.median(a)), float(np.std(a))
        pull = (med - truth) / sd if sd > 0 else float("nan")
        role = "LC observable" if name in LC_OBSERVABLES else "needs SED"
        print(
            "%-24s %12.6g %12.6g %12.4g %9.2f  %s"
            % (name, truth, med, sd, pull, role),
            flush=True,
        )
        out[name] = {
            "truth": truth,
            "median": med,
            "sd": sd,
            "pull": pull,
            "role": role,
        }
        if name in LC_OBSERVABLES:
            rows.append(abs(pull))

    if rows:
        print(
            "\nLIGHT-CURVE OBSERVABLES: %d compared, max |pull| = %.2f, "
            "median |pull| = %.2f"
            % (len(rows), max(rows), float(np.median(rows))),
            flush=True,
        )
        print(
            "A max |pull| within ~3 says dynesty found the right answer and "
            "the 1.6 h is real.\nLarge pulls on the LC observables would "
            "mean the speed came at the cost of correctness.",
            flush=True,
        )
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
