"""Was the truth DETECTABLE?  The (b)-vs-(c) discriminator.

dc18_evaluate.py cannot separate TRUTH_UNRECOVERABLE from
TRUTH_NOT_RECOVERED on a fit's own output: both look like "truth's mode is
not in the posterior".  The difference is whether the data SUPPORTED truth,
and that needs the truth solution's own likelihood.

  delta = best lp reachable from TRUTH  -  best lp the fit found
    delta >~ 0        the truth solution is competitive and the fit missed
                      it  -> (c) TRUTH_NOT_RECOVERED, a SEARCH failure
    delta << 0        the data genuinely disfavour truth -> (b)
                      TRUTH_UNRECOVERABLE, and not detecting it is correct

A CORRECTION TO THE ORIGINAL DESIGN.  This was specified as "one logp
evaluation at truth, not a second fit".  That is wrong and would misclassify
systematically: injecting truth's GEOMETRY leaves the fluxes, error scales
and Hogg mixture parameters at the other solution's values, so the logp comes
out biased LOW and every event would look disfavoured -- i.e. everything
would fall into (b), which is exactly the class a broken discriminator would
flatter us by choosing.  The nuisances have to be refitted at the truth
geometry, so this runs a SHORT conditional fit.

Both numbers are reported: the raw injected logp AND the short-fit best, so
the size of that bias is visible rather than assumed.  Only sampled
observables are injected -- review 2.3.17 measured that injected SAMPLED
parameters round-trip (49 of 52 exactly) while DERIVED ones are silently
overridden by up to 1000x.
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as C  # noqa: E402
import run_event as R  # noqa: E402

logging.disable(logging.WARNING)

# truth key -> (post-#246 params key, log10?).  These are the SAMPLED
# observables; nothing derived is injected.
INJECT = [
    ("t_0", "source.Source.t_0", False),
    ("u_0", "source.Source.u_0", False),
    ("t_E", "mulensevent.t_E", False),
    ("rho", "source.Source.rho", False),
    ("s", "lens.Companion.s", False),
    ("q", "lens.Companion.q", False),
]


def best_lp_of(trace):
    """Max finite lp in an existing trace's sample_stats."""
    import xarray as xr

    try:
        ss = xr.open_dataset(trace, group="sample_stats")
    except Exception:  # noqa: BLE001
        return None
    if "lp" not in ss.data_vars:
        ss.close()
        return None
    a = np.asarray(ss["lp"]).ravel()
    ss.close()
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--tune", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = C.data_dir_or_raise(os.environ.get("DC18_DATA"))
    truth, cls = C.load_truth(str(data_dir), args.event)
    files = C.light_curve_files(str(data_dir), args.event)
    ra, dec = C.event_coords(str(data_dir), args.event)
    ev = "%03d" % args.event
    base = Path("events") / ev
    print(
        "event %d (class %s)  bands %s" % (args.event, cls, list(files)),
        flush=True,
    )

    class _A:
        finite_source = True
        fix_u1 = True

    prefix = base / "detect" / ("DC2018_%s" % ev)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cfg = R.build_config(
        "DC2018_%s" % ev,
        files,
        prefix,
        base / ("DC2018_%s_mmexofast.json" % ev),
        _A(),
    )
    cfg["mulensevent"][0]["mmexofast"] = False  # TRUTH is the seed here
    params = R.build_user_params(
        ra, dec, fix_u1=True, bands_for_u1=list(files)
    )
    injected = {}
    for tkey, pkey, is_log in INJECT:
        if tkey not in truth:
            continue
        v = float(truth[tkey])
        params[pkey] = {"initval": float(np.log10(v)) if is_log else v}
        injected[pkey] = params[pkey]["initval"]
    print(
        "injected %d sampled observables: %s"
        % (len(injected), {k: round(v, 6) for k, v in injected.items()}),
        flush=True,
    )

    from exozippy.system import System

    # 1. the RAW injected logp, reported so the nuisance bias is visible
    c0 = copy.deepcopy(cfg)
    for k in ("run", "prefix", "sampler"):
        c0.pop(k, None)
    s0 = System(c0, copy.deepcopy(params))
    s0.prepare()
    m0 = s0.build_model()
    raw = float(m0.compile_logp()(m0.initial_point()))
    print(
        "raw logp at the injected truth (nuisances NOT refitted): %.3f" % raw,
        flush=True,
    )

    # 2. the short conditional fit, which is the number that counts
    from exozippy.run import run_fit

    c1 = copy.deepcopy(cfg)
    c1["sampler"] = {
        "method": "ptde",
        "tune": args.tune,
        "draws": args.draws,
        "cores": int(os.environ.get("NSLOTS", 32)),
    }
    try:
        out = run_fit(c1, copy.deepcopy(params))
        idata = out[0] if isinstance(out, tuple) else out
        lp = np.asarray(idata.sample_stats["lp"]).ravel()
        lp = lp[np.isfinite(lp)]
        truth_best = float(lp.max()) if lp.size else float("nan")
    except Exception as e:  # noqa: BLE001
        print(
            "the truth-seeded short fit FAILED (%s: %s)"
            % (type(e).__name__, str(e)[:140]),
            flush=True,
        )
        truth_best = float("nan")
    print(
        "best lp reachable from truth (short fit): %.3f" % truth_best,
        flush=True,
    )
    print("nuisance-refit gain: %+.1f nats" % (truth_best - raw), flush=True)

    # 3. what the production fit found
    fr = base / "fitresults" / ("DC2018_%s_trace.nc" % ev)
    fit_best = best_lp_of(str(fr))
    print(
        "best lp the production fit found: %s"
        % ("%.3f" % fit_best if fit_best is not None else "no trace"),
        flush=True,
    )

    delta = (
        (truth_best - fit_best)
        if (fit_best is not None and np.isfinite(truth_best))
        else None
    )
    print("\n=== DISCRIMINATOR ===", flush=True)
    if delta is None:
        print(
            "UNDETERMINED: need both the truth-seeded fit and the "
            "production trace",
            flush=True,
        )
    else:
        print("delta = truth - found = %+.1f nats" % delta, flush=True)
        if delta > -20:
            print(
                ">>> (c) TRUTH_NOT_RECOVERED: truth is competitive and the "
                "search missed it.",
                flush=True,
            )
        else:
            print(
                ">>> (b) TRUTH_UNRECOVERABLE: the data disfavour truth by "
                "%.1f nats; not detecting it is correct." % (-delta),
                flush=True,
            )
    json.dump(
        {
            "event": args.event,
            "class": cls,
            "raw_injected_logp": raw,
            "truth_best_lp": truth_best,
            "fit_best_lp": fit_best,
            "delta_logp_truth_minus_best": delta,
            "injected": injected,
        },
        open(args.out or "detect_%s.json" % ev, "w"),
        indent=1,
    )
    print("\nwrote %s" % (args.out or "detect_%s.json" % ev), flush=True)


if __name__ == "__main__":
    main()
