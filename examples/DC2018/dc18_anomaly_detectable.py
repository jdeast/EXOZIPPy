"""Is the anomaly UNDETECTABLE, or did MMEXOFAST merely fail to find it?

DC2018 event 8 died in 28 s with NoAnomalyFoundError: AnomalyFinder found no
grid window holding "MORE THAN THREE consecutive points at >= 2 sigma", so
no binary-lens fit could be seeded.  Its own message says that is a
DETECTION threshold rather than a sparse-data condition.  But event 8 is
labelled `omcassan` in the master file -- it IS a 2L1S planet event -- so
"undetectable" and "MMEXOFAST's grid search missed it" are different claims
and the pipeline currently cannot tell them apart.

THE TEST, per JDE: fit 2L1S SEEDED AT TRUTH and see whether it keeps the
binary solution, then compare against a PSPL fit of the same data.

  Delta = best 2L1S logp - best PSPL logp
    Delta LARGE and positive, with q staying away from zero
        -> the anomaly IS in the data and MMEXOFAST's detector missed it.
           That is a seeding defect, not a physical limit, and the event
           should be recoverable.
    Delta ~ 0, or q collapsing toward zero
        -> the binary adds nothing; the anomaly really is undetectable and
           PSPL is the correct answer.  The PSPL rung is then required
           rather than optional.

This also PROTOTYPES the PSPL rung, which the post-8.6.17 component split
makes straightforward: a point-source single-lens fit is just one `lens:`
body, one `source:`, and no `planet:` block at all.
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


def pspl_from(config):
    """A PSPL config from a 2L1S one: drop the companion and the planet."""
    c = copy.deepcopy(config)
    c.pop("planet", None)
    c["lens"] = [{"body": "star.Lens"}]
    for ev in c.get("mulensevent", []):
        ev["finite_source"] = False
        ev.pop("mmexofast", None)  # nothing to seed a binary with
    return c


def fit(config, params, label, draws, tune):
    from exozippy.run import run_fit

    cfg = copy.deepcopy(config)
    cfg["sampler"] = {
        "method": "ptde",
        "tune": tune,
        "draws": draws,
        "cores": int(os.environ.get("NSLOTS", 32)),
    }
    print("\n=== %s ===" % label, flush=True)
    try:
        out = run_fit(cfg, copy.deepcopy(params))
    except Exception as e:  # noqa: BLE001
        print(
            "%s FAILED: %s: %s" % (label, type(e).__name__, str(e)[:160]),
            flush=True,
        )
        return None
    idata = out[0] if isinstance(out, tuple) else out
    try:
        lp = np.asarray(idata.sample_stats["lp"]).ravel()
        best = float(np.nanmax(lp[np.isfinite(lp)]))
    except Exception:  # noqa: BLE001
        best = float("nan")
    q = None
    for n in ("lens.Companion.q", "lens.q", "planet.Companion.q"):
        if n in idata.posterior.data_vars:
            a = np.asarray(idata.posterior[n]).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                q = (
                    float(np.median(a)),
                    float(np.percentile(a, 2.5)),
                    float(np.percentile(a, 97.5)),
                )
            break
    print("%s: best lp = %.3f   q = %s" % (label, best, q), flush=True)
    return {"best_lp": best, "q": q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, default=8)
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = C.data_dir_or_raise(os.environ.get("DC18_DATA"))
    truth, cls = C.load_truth(str(data_dir), args.event)
    print("event %d  class=%s" % (args.event, cls), flush=True)
    print("truth: %s" % {k: round(v, 6) for k, v in truth.items()}, flush=True)

    files = C.light_curve_files(str(data_dir), args.event)
    if not files:
        print(
            "no light curves for event %d; aborting rather than testing "
            "the wrong data" % args.event,
            flush=True,
        )
        return
    print("bands: %s" % list(files), flush=True)

    ev = "%03d" % args.event
    base = Path("events") / ev
    prefix = base / "detect" / ("DC2018_%s" % ev)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    class _A:
        finite_source = True
        fix_u1 = True

    cfg2 = R.build_config(
        "DC2018_%s" % ev,
        files,
        prefix,
        base / ("DC2018_%s_mmexofast.json" % ev),
        _A(),
    )
    # SEED AT TRUTH.  Post-#246 homes: the track is per-source, the geometry
    # per-companion, the timescale event-level.
    ra, dec = C.event_coords(str(data_dir), args.event)
    params = R.build_user_params(
        ra, dec, fix_u1=True, bands_for_u1=list(files)
    )
    params.update(
        {
            "source.Source.t_0": {"initval": float(truth["t_0"])},
            "source.Source.u_0": {"initval": float(truth["u_0"])},
            "mulensevent.t_E": {"initval": float(truth["t_E"])},
            "source.Source.rho": {"initval": float(truth["rho"])},
            "lens.Companion.s": {"initval": float(truth["s"])},
            "lens.Companion.q": {"initval": float(truth["q"])},
        }
    )
    cfg2["mulensevent"][0]["mmexofast"] = False  # truth IS the seed here

    r2 = fit(cfg2, params, "2L1S seeded at truth", args.draws, args.tune)
    rp = fit(pspl_from(cfg2), params, "PSPL", args.draws, args.tune)

    print("\n=== VERDICT ===", flush=True)
    if r2 and rp and np.isfinite(r2["best_lp"]) and np.isfinite(rp["best_lp"]):
        d = r2["best_lp"] - rp["best_lp"]
        print("2L1S - PSPL = %+.1f nats" % d, flush=True)
        if d > 20 and r2["q"] and r2["q"][1] > 1e-7:
            print(
                ">>> THE ANOMALY IS DETECTABLE.  MMEXOFAST's grid search "
                "missed it -- a SEEDING defect, not a physical limit.",
                flush=True,
            )
        elif d < 5:
            print(
                ">>> THE ANOMALY IS NOT DETECTABLE.  PSPL is the right "
                "answer and the PSPL rung is REQUIRED, not optional.",
                flush=True,
            )
        else:
            print(
                ">>> MARGINAL (%.1f nats).  Neither claim is safe; this is "
                "the regime a PSPL-vs-2L1S evidence comparison exists "
                "for." % d,
                flush=True,
            )
    else:
        print("one or both fits failed; no verdict", flush=True)
    json.dump(
        {
            "event": args.event,
            "class": cls,
            "truth": truth,
            "twol1s": r2,
            "pspl": rp,
        },
        open(args.out or "anomaly_detect_%s.json" % ev, "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
