#!/usr/bin/env python3
"""Would MMEXOFAST have handed us the right architecture, unprompted?

Every DC2018 fit this pipeline has run was configured as 2L1S by
`run_event.py`'s build_config -- two stars, one planet companion,
finite_source on.  On the 43 `omcassan` events that happens to be correct,
so "43/43 recovered" would report a decision WE made rather than one the
pipeline derived.  This measures the derived one.

`fit_type="binary_lens"` is an ENTRY POINT, not an assertion: the workflow
runs the point-lens stages, finds and classifies anomalies, and
`fit_binary_lens_models` declines binary outright when it fails to improve
chi2 by 3.  So MMEXOFAST already decides; nothing recorded the decision.

For each event this records what it decided -- the anomaly classification,
whether a binary model was accepted or skipped, and the fitted q -- so the
verdict can be compared against the master file's class AFTER the fact.
Reading the truth is legitimate here precisely because the fit does not:
this is scoring the classifier, not seeding it.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402

# Per-event artifact directory.  A list so the worker processes inherit the
# value set in main() through fork, without threading it through pool.map --
# the whole point is to run this script twice (MMEXOFAST main vs PR9) and diff
# the results, which needs the two runs' artifacts kept apart.
POSTHOC_DIR = ["posthoc"]


def one_event(event):
    """Run the MMEXOFAST stage and record the architecture it chose."""
    import logging

    from astropy.coordinates import SkyCoord

    from exozippy.components.mulensing import mmexofast_support

    out = {"event": event}
    try:
        data_dir = dc.data_dir_or_raise(None)
        files = dc.light_curve_files(data_dir, event, ("W149", "Z087"))
        ra, dec = dc.event_coords(data_dir, event)
        coords = SkyCoord(ra, dec, unit="deg").to_string(style="hmsdms")
        jpath = Path(POSTHOC_DIR[0]) / f"ev{event:03d}_mmexofast.json"
        jpath.parent.mkdir(parents=True, exist_ok=True)

        # Capture MMEXOFAST's own log: the architecture verdict is announced
        # there ("Binary model does not improve chi2 enough, skipping") and
        # nowhere in the returned JSON.
        buf = jpath.with_suffix(".log")
        h = logging.FileHandler(buf, mode="w")
        h.setLevel(logging.DEBUG)
        logging.getLogger("mmexofast").addHandler(h)
        logging.getLogger("mmexofast").setLevel(logging.DEBUG)

        # try/finally, NOT a bare sequence: run_or_load RAISES on any event
        # whose anomaly search fails, and with the removeHandler call placed
        # after it the handler stayed attached to the GLOBAL mmexofast logger.
        # The next event that worker picked up then wrote into the FAILED
        # event's log file, which reads exactly like the pipeline having run
        # twice on one event.  That fiction cost a long investigation: event
        # 227's log acquired event 258's PSPL fit (t_0 = 2458377.97,
        # u_0 = 0.00194 -- identical to 258's own independent fit) and its
        # high_mag classification, and I read the pair as a wrong-epoch
        # failure inside MMEXOFAST.  227 simply crashed; it was never
        # classified at all.
        try:
            mmexofast_support.run_or_load(
                jpath,
                list(files.values()),
                coords=coords,
                fit_type="binary_lens",
                renormalize_errors=True,
                no_parallax=True,
                options={
                    "limb_darkening_coeffs_gamma": {
                        "W149": 0.0,
                        "Z087": 0.0,
                    },
                    "pool": 1,
                    "stop_before": "fit_binary_lens:fit_binary_lens_models",
                },
            )
        finally:
            logging.getLogger("mmexofast").removeHandler(h)
            h.close()

        d = json.load(open(jpath))
        fits = d.get("fits", [])
        out["n_solutions"] = len(fits)
        out["q"] = [f["parameters"].get("q") for f in fits]
        out["s"] = [f["parameters"].get("s") for f in fits]
        out["rho"] = [f["parameters"].get("rho") for f in fits]
        text = buf.read_text(errors="replace")
        out["declined_binary"] = "does not improve chi2 enough" in text
        # The classifier's verdict and the seed topologies it then generates,
        # scraped from the captured log because neither reaches the JSON.
        # Both are UNQUOTED there ("anomaly_type = wide", "Estimated binary
        # params (WidePlanet)"); the first version of this looked for quoted
        # lowercase tags and so silently captured NOTHING on all 44 events.
        # These two fields are the whole point of the comparison against
        # MMEXOFAST PR9, which renames the verdicts (close/wide/high_mag ->
        # bump/dip/caustic_crossing/high_mag) and so may change which
        # topologies get seeded.
        out["anomaly_type"] = sorted(
            set(re.findall(r"anomaly_type\s*=\s*(\w+)", text))
        )
        out["seed_topologies"] = sorted(
            set(re.findall(r"Estimated binary params \((\w+)\)", text))
        )
        out["status"] = "ok"
    except BaseException as exc:
        out["status"] = f"failed: {type(exc).__name__}: {exc}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=None)
    ap.add_argument(
        "--ncpu", type=int, default=int(os.environ.get("NSLOTS", 0)) or 8
    )
    ap.add_argument("--out", default="dc18_architecture_posthoc.json")
    ap.add_argument(
        "--posthoc-dir",
        default="posthoc",
        help="Directory for the per-event MMEXOFAST JSON and captured log "
        "(default posthoc). Use a distinct one per MMEXOFAST version.",
    )
    args = ap.parse_args()

    if args.events:
        events = [int(x) for x in args.events.split(",") if x.strip()]
    else:
        events = [int(x) for x in open("events.txt") if x.strip()]
    print(f"{len(events)} events on {args.ncpu} workers", flush=True)

    POSTHOC_DIR[0] = args.posthoc_dir

    with mp.Pool(args.ncpu) as pool:
        rows = pool.map(one_event, events)

    data_dir = dc.data_dir_or_raise(None)
    print(
        f"\n{'ev':>4s} {'truth class':>12s} {'nsol':>5s} {'declined':>9s} "
        f"{'q (first)':>12s} {'status':>10s}"
    )
    for r in sorted(rows, key=lambda x: x["event"]):
        try:
            _, cls = dc.load_master_row(data_dir, r["event"])
        except Exception:
            cls = "?"
        r["truth_class"] = cls
        q = (r.get("q") or [None])[0]
        print(
            f"{r['event']:>4d} {cls:>12s} {r.get('n_solutions', 0):>5} "
            f"{str(r.get('declined_binary', '')):>9s} "
            f"{(f'{q:.3g}' if q else '-'):>12s} "
            f"{r['status'][:10]:>10s}"
        )
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
