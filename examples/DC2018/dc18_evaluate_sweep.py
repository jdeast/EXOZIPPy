"""Run dc18_evaluate over every event with output on disk.

Usage:  python dc18_evaluate_sweep.py [event ...]
"""

import sys
from pathlib import Path

import dc18_common as C
import dc18_evaluate as E


def paths(ev):
    d = Path("events") / ("%03d" % ev)
    alt = Path("events") / str(ev)
    base = d if d.exists() else alt
    fr = base / "fitresults"
    tag = "DC2018_%03d" % ev
    logs = sorted(
        Path(".").glob("dc2018.*.log"), key=lambda p: p.stat().st_mtime
    )
    log = None
    for p in reversed(logs):
        try:
            if ("event %d " % ev) in p.read_text(errors="ignore")[:4000]:
                log = str(p)
                break
        except OSError:
            pass
    return {
        "trace": str(fr / (tag + "_trace.nc")),
        "modes_txt": str(fr / (tag + "_modes.txt")),
        "log": log,
        "detect_json": "anomaly_detect_%03d.json" % ev,
    }


def main(argv):
    data = C.data_dir_or_raise(None)
    evs = [int(a) for a in argv] or [
        int(x) for x in Path("events.txt").read_text().split()
    ]
    results = []
    for ev in evs:
        p = paths(ev)
        if not (Path(p["trace"]).exists() or p["log"]):
            continue  # never run; not a failure, just absent
        try:
            truth, _cls = C.load_truth(str(data), ev)
        except Exception as e:  # noqa: BLE001
            print("event %d: no truth (%s)" % (ev, type(e).__name__))
            continue
        results.append(E.classify(ev, truth, **p))
    if not results:
        print("no events with output on disk yet")
        return
    E.summarise(results)


if __name__ == "__main__":
    main(sys.argv[1:])
