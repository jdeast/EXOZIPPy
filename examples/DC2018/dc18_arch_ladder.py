"""The architecture ladder: flat -> PSPL -> FSPL -> 2L1S -> 2L1S+FS.

WHY THIS REPLACES THE (b)-vs-(c) DISCRIMINATOR, per JDE 2026-09-11.  The
previous discriminator compared the best lnL reachable from TRUTH against
the best the fit found, and had two defects JDE named:

  * "is it in a basin" is FRAGILE.  It asks whether a short PTDE fit stayed
    put, and that fit's optimizer is measured broken -- review 2.4.14 has
    the polish plateauing at ~0.9 nats/sweep and drifting into a region
    ~1000x more expensive to evaluate.  So "stayed" partly measures the
    optimizer rather than the likelihood surface.
  * it needs truth INJECTED, which walked straight into review 2.3.17: with
    `star_constrains_rho` at its default, source.rho is DERIVED, so an
    injected rho is silently overridden and the rung it was meant to test
    never existed.

The ladder has neither problem:

  NESTED, so the statistics are legitimate.  Each rung adds parameters, so
  delta chi2 with the right delta-dof is a calibrated likelihood-ratio test.
  Comparing two BASINS of one model -- what the old discriminator did -- is
  not nested at all, which is why no single sigma could be quoted there.

  TRUTH-INDEPENDENT.  "Does flat -> PSPL clear threshold" (is there an
  event) and "does PSPL -> 2L1S clear threshold" (is there an anomaly) are
  answerable without the truth table.  That makes the ladder usable in the
  BLIND pipeline, not only in evaluation -- it is the same machinery the
  architecture escalation needs.

  DETECTABILITY FALLS OUT.  If truth is 2L1S but PSPL -> 2L1S is not
  significant, truth is genuinely undetectable: class (b), PASS, with no
  reference to truth's own likelihood.

WHAT IT DOES NOT ANSWER: whether a reported mode is a degenerate IMAGE of
truth (class (h)).  That is a within-architecture question and still needs
the mode report and per-mode evidence.

THE STATISTIC IS THE DATA LOG-LIKELIHOOD, NOT logp.  Rungs differ in their
priors and parameter counts, so the posterior includes different prior
normalisations and its differences are not a likelihood ratio.  Only the
`mulensinstrument.model.hogg.*` terms are summed.

delta chi2 = 2 * delta lnL.  sigma is the chi2_{delta-dof} tail, NOT
sqrt(delta chi2) -- that shortcut is the k=1 case only.
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

# (name, binary lens?, finite source?)
RUNGS = [
    ("PSPL", False, False),
    ("FSPL", False, True),
    ("2L1S", True, False),
    ("2L1S+FS", True, True),
]


def flat_via_model(event, files, mmx, outdir, tune, draws):
    """The NO-EVENT rung, evaluated through the SAME likelihood as the rest.

    An analytic constant-flux chi2 is NOT comparable to the fitted rungs and
    the first version of this file got that wrong: the instruments carry a
    HOGG OUTLIER MIXTURE, whose log-likelihood is not -chi2/2, so a delta
    between an analytic Gaussian chi2 (6,414,123 on event 223) and a Hogg
    lnL is comparing two different statistics.  Every rung must use one
    likelihood or none of the deltas mean anything.

    So "flat" is a PSPL fit with the magnification pinned out: u_0 forced
    far from the lens (A = 1.00005 at u_0 = 10), with t_0 and t_E pinned too
    because they are unidentifiable once there is no event to time.  The
    flux and noise nuisances still float, exactly as in every other rung, so
    this is properly NESTED inside PSPL and the delta-dof is the three
    pinned trajectory parameters.
    """
    cfg, trace = make_config(
        event, files, mmx, False, False, outdir, tune, draws, tag="FLAT"
    )
    cfg["mulensevent"][0]["mmexofast"] = False  # nothing to seed
    extra = {
        "source.Source.u_0": {"initval": 10.0, "sigma": 0},
        "source.Source.t_0": {"initval": 2459000.0, "sigma": 0},
        "mulensevent.t_E": {"initval": 20.0, "sigma": 0},
    }
    return cfg, trace, extra


def data_lnL(trace, model=None):
    """Max DATA log-likelihood over the draws.

    Reads the Hogg per-instrument likelihood terms from the trace's
    log_likelihood group when present; otherwise falls back to lp, which
    INCLUDES priors and is flagged so a reader does not mistake it for a
    likelihood.
    """
    import xarray as xr

    try:
        ll = xr.open_dataset(trace, group="log_likelihood")
        tot = None
        for v in ll.data_vars:
            a = np.asarray(ll[v])
            a = a.reshape(a.shape[0], a.shape[1], -1).sum(axis=2)
            tot = a if tot is None else tot + a
        ll.close()
        if tot is not None:
            t = tot[np.isfinite(tot)]
            if t.size:
                return float(t.max()), "log_likelihood group"
    except Exception:  # noqa: BLE001
        pass
    try:
        ss = xr.open_dataset(trace, group="sample_stats")
        a = np.asarray(ss["lp"]).ravel()
        ss.close()
        a = a[np.isfinite(a)]
        if a.size:
            return float(a.max()), "lp (INCLUDES PRIORS -- not a pure lnL)"
    except Exception:  # noqa: BLE001
        pass
    return None, "unavailable"


def n_free(trace):
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


def make_config(event, files, mmx, binary, fs, outdir, tune, draws, tag=None):
    a = R.build_parser().parse_args([str(event)])
    a.finite_source = bool(fs)
    a.fix_u1 = True
    a.sampler = "ptde"
    a.tune = tune
    a.draws = draws
    # FLAT is also a binary=False / fs=False config, so without an explicit
    # tag it would share PSPL's prefix and the two would overwrite each
    # other's trace.
    tag = tag or "%s%s" % ("2L1S" if binary else "1L1S", "_FS" if fs else "")
    prefix = Path(outdir) / tag / ("DC2018_%03d" % event)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cfg = R.build_config("DC2018_%03d" % event, files, prefix, mmx, a)
    if not binary:
        # A point lens is ONE lens body and no planet.  The 8.6.17 split
        # makes this a config edit rather than a code path.
        cfg.pop("planet", None)
        cfg["lens"] = [{"body": "star.Lens"}]
    if fs:
        # rho must be SAMPLED for this rung to add a parameter at all.  With
        # star_constrains_rho at its default (True) rho is DERIVED from
        # theta_star/theta_E, the rung adds no freedom, and an injected rho
        # is silently overridden (review 2.3.17).
        cfg["source"][0]["star_constrains_rho"] = False
    cfg["sampler"] = {
        "method": "ptde",
        "tune": tune,
        "draws": draws,
        "cores": int(os.environ.get("NSLOTS", 32)),
    }
    return cfg, str(prefix) + "_trace.nc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--tune", type=int, default=400)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--rungs", default="FLAT,PSPL,FSPL,2L1S,2L1S+FS")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = str(C.data_dir_or_raise(os.environ.get("DC18_DATA")))
    truth, cls = C.load_truth(data_dir, args.event)
    files = C.light_curve_files(data_dir, args.event)
    ra, dec = C.event_coords(data_dir, args.event)
    ev = "%03d" % args.event
    base = Path("events") / ev
    mmx = base / ("DC2018_%s_mmexofast.json" % ev)
    outdir = base / "ladder"
    print(
        "event %d (class %s)  bands %s" % (args.event, cls, list(files)),
        flush=True,
    )
    print("truth: %s" % {k: round(v, 6) for k, v in truth.items()}, flush=True)
    print(
        "seeds: %s (truth-independent -- push_seed_hints drops whatever a "
        "rung's topology does not want)" % mmx,
        flush=True,
    )

    results = []
    from exozippy.run import run_fit

    asked = args.rungs.split(",")
    want = [("FLAT", None, None)] if "FLAT" in asked else []
    want += [r for r in RUNGS if r[0] in asked]
    params = R.build_user_params(
        ra, dec, fix_u1=True, bands_for_u1=list(files)
    )
    for name, binary, fs in want:
        if name == "FLAT":
            cfg, trace, extra = flat_via_model(
                args.event, files, mmx, outdir, args.tune, args.draws
            )
            rung_params = dict(params)
            rung_params.update(extra)
        else:
            cfg, trace = make_config(
                args.event,
                files,
                mmx,
                binary,
                fs,
                outdir,
                args.tune,
                args.draws,
            )
            rung_params = params
        print(
            "\n=== rung %s (binary=%s finite_source=%s) ==="
            % (name, binary, fs),
            flush=True,
        )
        try:
            run_fit(copy.deepcopy(cfg), copy.deepcopy(rung_params))
        except Exception as e:  # noqa: BLE001
            print(
                "  raised (%s: %s) -- checking for a trace anyway, since "
                "run_fit writes before it returns"
                % (type(e).__name__, str(e)[:110]),
                flush=True,
            )
        lnL, src = data_lnL(trace)
        k = n_free(trace)
        if lnL is None:
            print("  NO RESULT (%s)" % src, flush=True)
            results.append({"name": name, "lnL": None, "k": k, "note": src})
            continue
        print(
            "  max lnL = %.1f  (from %s)   k = %s" % (lnL, src, k), flush=True
        )
        results.append(
            {
                "name": name,
                "lnL": lnL,
                "chi2": -2.0 * lnL,
                "k": k,
                "trace": trace,
                "note": src,
            }
        )

    print("\n" + "=" * 70, flush=True)
    print(
        "LADDER: each step is NESTED in the one below, so delta chi2 with "
        "delta-dof is a\nlegitimate likelihood-ratio test.",
        flush=True,
    )
    print("=" * 70, flush=True)
    print(
        "%-9s %14s %6s %14s %8s %10s"
        % ("rung", "lnL", "k", "delta chi2", "d.dof", "sigma"),
        flush=True,
    )
    prev = None
    try:
        from scipy import stats
    except Exception:  # noqa: BLE001
        stats = None
    for r in results:
        if r.get("lnL") is None:
            print("%-9s %14s %6s" % (r["name"], "--", r.get("k")), flush=True)
            continue
        if prev is None:
            print(
                "%-9s %14.1f %6s %14s %8s %10s"
                % (r["name"], r["lnL"], r.get("k"), "-", "-", "-"),
                flush=True,
            )
        else:
            dchi2 = 2.0 * (r["lnL"] - prev["lnL"])
            ddof = (r.get("k") or 0) - (prev.get("k") or 0)
            sig = "-"
            if stats is not None and ddof > 0 and dchi2 > 0:
                p = stats.chi2.sf(dchi2, ddof)
                sig = "%.1f" % stats.norm.isf(max(p, 1e-300) / 2)
            print(
                "%-9s %14.1f %6s %14.1f %8s %10s"
                % (r["name"], r["lnL"], r.get("k"), dchi2, ddof, sig),
                flush=True,
            )
        prev = r
    print(
        "\nsigma is the chi2_{d.dof} tail.  sqrt(delta chi2) is the k=1 "
        "shortcut ONLY.\nA rung that does not clear its predecessor is not "
        "supported by the data --\nwhich is the truth-independent "
        "detectability statement class (b) needs.",
        flush=True,
    )
    json.dump(
        {"event": args.event, "class": cls, "truth": truth, "rungs": results},
        open(args.out or "ladder_%s.json" % ev, "w"),
        indent=1,
        default=str,
    )
    print("\nwrote %s" % (args.out or "ladder_%s.json" % ev), flush=True)


if __name__ == "__main__":
    main()
