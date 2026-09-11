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
# (truth key, params key to INJECT, trace vars to READ BACK, log10?).
# The params key and the trace variable disagree for the log coordinates --
# `lens.Companion.s` is seeded but the trace stores `lens.log_s` -- so a
# single name cannot serve both, and deriving the trace name from the params
# key silently found only t_0 and u_0 (2 of 6 observables) on the first
# version of the basin test.  t_E is derived and appears in no trace.
INJECT = [
    ("t_0", "source.Source.t_0", ("source.t_0",), False),
    ("u_0", "source.Source.u_0", ("source.u_0",), False),
    ("t_E", "mulensevent.t_E", (), False),
    ("rho", "source.Source.rho", ("source.log_rho", "source.rho"), None),
    ("s", "lens.Companion.s", ("lens.log_s",), True),
    ("q", "lens.Companion.q", ("planet.log_q", "lens.log_q"), True),
]


def _n_free(trace):
    """Free scalar parameter count, from the trace's *_raw variables."""
    import xarray as xr

    try:
        ds = xr.open_dataset(trace, group="posterior")
    except Exception:  # noqa: BLE001
        return None
    n = 0
    for v in ds.data_vars:
        if v.endswith("_raw"):
            shp = ds[v].shape
            n += int(np.prod(shp[2:])) if len(shp) > 2 else 1
    ds.close()
    return n or None


def _n_draws(trace):
    """Total draws in a trace, for the max-lp sample-size correction."""
    import xarray as xr

    try:
        ss = xr.open_dataset(trace, group="sample_stats")
    except Exception:  # noqa: BLE001
        return None
    n = int(np.asarray(ss["lp"]).size) if "lp" in ss.data_vars else None
    ss.close()
    return n


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

    # BORROW run_event's own parser rather than stubbing an args object.
    # build_config reads eleven attributes off it, and hand-stubbed classes
    # crashed both these scripts with AttributeError on args.sampler -- a
    # bug that recurs whenever that parser gains an option.
    a = R.build_parser().parse_args([str(args.event)])
    a.finite_source = True
    a.fix_u1 = True
    a.sampler = "ptde"
    a.tune = args.tune
    a.draws = args.draws

    prefix = base / "detect" / ("DC2018_%s" % ev)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cfg = R.build_config(
        "DC2018_%s" % ev,
        files,
        prefix,
        base / ("DC2018_%s_mmexofast.json" % ev),
        a,
    )
    cfg["mulensevent"][0]["mmexofast"] = False  # TRUTH is the seed here
    params = R.build_user_params(
        ra, dec, fix_u1=True, bands_for_u1=list(files)
    )
    injected = {}
    for tkey, pkey, _tvars, _is_log in INJECT:
        if tkey not in truth:
            continue
        # ALWAYS inject the LINEAR value: the params key is the linear
        # spelling (lens.Companion.s), and the engine derives its own log
        # coordinate.  The log flag below is for reading the TRACE back.
        params[pkey] = {"initval": float(truth[tkey])}
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
    # READ THE TRACE FROM DISK, do not trust a return value.  run_fit is
    # SIDE-EFFECTING: it writes {prefix}_trace.nc (run.py:543) and returns
    # None, so unpacking its result gave
    # "AttributeError: 'NoneType' object has no attribute 'sample_stats'"
    # on event 223 -- while the fit itself had succeeded and left a complete
    # 19.7 MB trace on disk.  The result was never lost, only unreachable.
    truth_trace = str(prefix) + "_trace.nc"
    try:
        run_fit(c1, copy.deepcopy(params))
    except Exception as e:  # noqa: BLE001
        print(
            "the truth-seeded short fit raised (%s: %s) -- checking for a "
            "trace anyway, since run_fit writes before it returns"
            % (type(e).__name__, str(e)[:120]),
            flush=True,
        )
    truth_best = best_lp_of(truth_trace)
    if truth_best is None:
        print("no usable lp in %s" % truth_trace, flush=True)
        truth_best = float("nan")
    print(
        "best lp reachable from truth (short fit): %.3f" % truth_best,
        flush=True,
    )
    print("nuisance-refit gain: %+.1f nats" % (truth_best - raw), flush=True)

    # 3. THE BASIN TEST.  Delta max-lp alone CANNOT separate "the data
    # disfavour truth" from "truth is a real basin the search never
    # visited" -- both give a modestly negative delta.  On event 223 the
    # delta said (b) UNRECOVERABLE while the truth-seeded posterior sat
    # ENTIRELY on truth's side of the +/-u_0 degeneracy (u_0 in
    # [-0.0170, -0.0135] against a truth of -0.0128) and the production
    # posterior held ZERO draws with u_0 < 0.  Truth was a real basin, just
    # not the deepest, and was owed a second mode.
    #
    # JDE 2026-09-11: "i'd like to avoid the scenario where the truth is a
    # real basin, just not the deepest one.  that scenario should be
    # reported as two modes."
    #
    # The test: did the truth-seeded fit STAY, or drift to the global
    # solution?  Staying means truth is a local optimum the production fit
    # owed us.  Drifting means truth is only a point on a slope and the
    # disfavour is real.  Distance is in the truth-seeded fit's OWN
    # posterior sd, because the question is whether it remained.
    print(
        "\n=== BASIN TEST: did the truth-seeded fit stay at truth? ===",
        flush=True,
    )
    basin = {}
    import xarray as xr

    try:
        ds = xr.open_dataset(truth_trace, group="posterior")
        for tkey, _pkey, tvars, is_log in INJECT:
            if tkey not in truth or not tvars:
                continue
            var = next((v for v in tvars if v in ds.data_vars), None)
            if var is None:
                print(
                    "  %-6s (not in the trace: %s)" % (tkey, list(tvars)),
                    flush=True,
                )
                continue
            # a `log_`-prefixed trace var is compared in log space
            is_log = var.split(".")[-1].startswith("log_")
            a = np.asarray(ds[var]).ravel()
            a = a[np.isfinite(a)]
            if a.size == 0:
                continue
            med, sd = float(np.median(a)), float(np.std(a))
            t = float(truth[tkey])
            if is_log:
                if t <= 0:
                    continue
                t = float(np.log10(t))
            z = (med - t) / sd if sd > 0 else float("nan")
            basin[tkey] = {
                "truth": t,
                "median": med,
                "sd": sd,
                "z": z,
                "stayed": bool(abs(z) < 5.0),
            }
            print(
                "  %-6s truth %-14.6g settled %-14.6g sd %-10.3g "
                "z=%+7.2f  %s"
                % (
                    tkey,
                    t,
                    med,
                    sd,
                    z,
                    "stayed" if abs(z) < 5.0 else "DRIFTED",
                ),
                flush=True,
            )
        ds.close()
    except Exception as e:  # noqa: BLE001
        print(
            "  basin test unavailable (%s: %s)"
            % (type(e).__name__, str(e)[:100]),
            flush=True,
        )

    stayed = [k for k, v in basin.items() if v["stayed"]]
    is_basin = bool(basin) and len(stayed) >= max(1, len(basin) - 1)
    print(
        "  -> truth %s a local basin (%d of %d parameters stayed)"
        % ("IS" if is_basin else "is NOT", len(stayed), len(basin)),
        flush=True,
    )

    # 4. what the production fit found
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
    verdict = None
    if delta is None:
        print(
            "UNDETERMINED: need both the truth-seeded fit and the "
            "production trace",
            flush=True,
        )
    else:
        # DELTA CHI2 IS THE PRIMARY NUMBER, per JDE 2026-09-11: chi2 is the
        # field standard.  delta chi2 = 2 * delta lnL, and sigma = sqrt of
        # that ONLY for one degree of freedom.  With the seven trajectory
        # parameters it is the chi2_7 tail (49.6 -> 5.6 sigma, not 7.0), and
        # for two basins of the SAME model with the SAME parameter count it
        # is not a nested test at all, so no single sigma is calibrated.
        dchi2 = 2.0 * delta
        print("delta lnL  = %+.1f nats" % delta, flush=True)
        print(
            "delta chi2 = %+.1f  (sqrt = %.1f sigma, valid for k=1 ONLY; "
            "with k=7 use the chi2_7 tail)" % (dchi2, np.sqrt(abs(dchi2))),
            flush=True,
        )

        # SAMPLE-SIZE CORRECTION.  max-lp grows with draw count: for lp
        # scatter sd ~ sqrt(d/2), E[max] ~ sd * sqrt(2 ln N).  The
        # truth-seeded fit is deliberately SHORT, so comparing its max
        # against a much longer production run favours the latter for free.
        # Measured on event 223: 41,600 vs 2,500,000 draws is ~3 of 24.8
        # nats.
        n_t, n_f = _n_draws(truth_trace), _n_draws(str(fr))
        if n_t and n_f and n_t > 1 and n_f > 1:
            # sd of lp for a fit at its optimum is ~sqrt(d/2) with d the
            # FREE PARAMETER COUNT -- using len(basin) (how many
            # observables happened to be readable) gave sqrt(1) and
            # understated the bias 4x.
            d_free = _n_free(truth_trace) or 27
            sd = np.sqrt(d_free / 2.0)
            bias = sd * (np.sqrt(2 * np.log(n_f)) - np.sqrt(2 * np.log(n_t)))
            print(
                "draws %d vs %d -> max-lp bias favours the production fit by "
                "~%.1f nats; corrected delta = %+.1f"
                % (n_t, n_f, bias, delta + bias),
                flush=True,
            )
            delta += bias

        if is_basin:
            verdict = "BASIN_OMITTED"
            print(
                ">>> TRUTH IS A REAL BASIN THAT THE FIT NEVER VISITED -- it "
                "should have been REPORTED AS A MODE.\n"
                "    Whether that is (h) DEGENERATE_COUNTERPART or (c) "
                "TRUTH_NOT_RECOVERED depends on whether the reported mode is "
                "a known degenerate image of truth; the MODE REPORT decides "
                "that, not this delta.\n"
                "    NOT (b): a modest disfavour does not license "
                "'undetectable' when the basin demonstrably exists.",
                flush=True,
            )
        elif delta > 0:
            verdict = "TRUTH_NOT_RECOVERED"
            print(
                ">>> (c) TRUTH_NOT_RECOVERED: truth is competitive and the "
                "search missed it.",
                flush=True,
            )
        else:
            verdict = "TRUTH_UNRECOVERABLE"
            print(
                ">>> (b) TRUTH_UNRECOVERABLE: truth is NOT a local basin and "
                "the data disfavour it by %.1f nats (chi2 %.1f); not "
                "detecting it is correct." % (-delta, -2 * delta),
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
            "delta_chi2": (2.0 * delta) if delta is not None else None,
            "basin_test": basin,
            "truth_is_a_basin": is_basin,
            "verdict": verdict,
            "injected": injected,
        },
        open(args.out or "detect_%s.json" % ev, "w"),
        indent=1,
    )
    print("\nwrote %s" % (args.out or "detect_%s.json" % ev), flush=True)


if __name__ == "__main__":
    main()
