"""Validate a submission tree with microlens-submit's OWN code.

    python 04_validate.py <submission dir> [--out validation.json]
                          [--listing evidence/manifest_resolved.json]

microlens-submit is read from the unpacked wheel ($RMDC26_MSUB_PKG,
default /n/holylabs/jjohnson_lab/Lab/rmdc26_report/msub_check/pkg), never
installed.  Its models need pydantic and psutil, which the exozippy env
lacks: put them on PYTHONPATH (03_submission.py points it at
$RMDC26_REPORT/deps, filled by `pip install --target`).  With them the
check is the package's own `load()` + `Submission.run_validation()` (the
team/tier/event-id/probability-sum/alias checks and every active
solution's checks) plus `Solution.run_validation()` on every solution,
inactive ones included.  Without them it falls back to the pure
validators in validate_parameters.py and says so ("mode").

On top, the checks the package does not make:

- physical_parameters keys are in validate_parameters.PHYSICAL_PARAMETERS,
  and validate_physical_parameters() passes on them;
- t0 > 2.4e6 (a full BJD) and a 1S2L alpha in [0, 2 pi);
- the alpha and t0 ROUND TRIP: each solution's radians, back to degrees,
  against the campaign's results.csv (degrees) for the same fit, with no
  offset (an error beyond 10 degrees; 180 is the C21 trap);
- every notes_path exists, and the event directory is named by event_id.

Messages are split into errors and warnings by the package's own wording
("Warning", "Recommendation", "very large/small", "Unusual" are advisory).
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

PKG = os.environ.get(
    "RMDC26_MSUB_PKG",
    "/n/holylabs/jjohnson_lab/Lab/rmdc26_report/msub_check/pkg",
)
ADVISORY = ("warning", "recommendation", "very large", "very small",
            "unusual", "consider")  # fmt: skip


def is_advisory(msg):
    m = msg.lower()
    return any(k in m for k in ADVISORY)


def results_rows(path):
    """{parname: [(mode, value, up, low)]} from a fit's results.csv.

    Two schemas exist: multi-mode "parname, mode, weight, weight_err,
    value, up_err, low_err" and single-mode "parname, value, up_err,
    low_err" (a fit that settles into one mode).  Each row is read by its
    length; a single-mode row's mode is "all".  Reading only the 7-column
    form dropped every single-mode fit (029, 040 on 2026-10-25), and its
    round trip then passed with nothing compared."""
    out = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            if len(row) >= 7:
                mode, vals = row[1], row[4:7]
            elif len(row) == 4:
                mode, vals = "all", row[1:4]
            else:
                continue
            if mode.startswith("rejected"):
                continue
            try:
                out[row[0]].append((mode,) + tuple(float(x) for x in vals))
            except ValueError:
                continue
    return out


def roundtrip(sol, rows):
    """(alpha_ddeg, t0_dt): the smallest difference between this solution
    and any non-rejected mode row of results.csv, alpha rad -> deg with no
    offset (None when the csv lacks the row)."""
    p = sol["parameters"]
    da = dt = None
    alpha_rows = [r for k, v in rows.items()
                  if k.startswith("lens.") and k.endswith(".alpha")
                  and "xalpha" not in k and "yalpha" not in k for r in v]  # fmt: skip
    if "alpha" in p and alpha_rows:
        deg = math.degrees(p["alpha"])
        da = min(abs((deg - r[1] + 180.0) % 360.0 - 180.0) for r in alpha_rows)
    if rows.get("source.t_0"):
        dt = min(abs(p["t0"] - r[1]) for r in rows["source.t_0"])
    return da, dt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--listing", default=None,
                    help="manifest_resolved.json (for the round trip); "
                    "default <root>/../evidence/manifest_resolved.json")  # fmt: skip
    a = ap.parse_args(argv)
    root = Path(a.root)
    sys.path.insert(0, PKG)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "msub_vp", Path(PKG) / "microlens_submit" / "validate_parameters.py"
    )
    vp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp)
    errors, warnings = [], []
    version = None
    try:
        import microlens_submit as ms
        from microlens_submit.utils import load

        version = getattr(ms, "__version__", None)
        mode = "package models (load + run_validation)"
    except ImportError as exc:
        ms = None
        mode = "pure validators only (models not importable: %s)" % exc

    def add(where, msgs):
        for m in msgs:
            (warnings if is_advisory(m) else errors).append(
                "%s: %s" % (where, m)
            )

    files = sorted(
        glob.glob(str(root / "events" / "*" / "solutions" / "*.json"))
    )
    sols = {f: json.loads(Path(f).read_text()) for f in files}
    if ms is not None:
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            sub = load(str(root))
        # the models WARN (not raise) on malformed fields at construction
        for c in caught:
            errors.append("model construction: %s" % c.message)
        add("submission", sub.run_validation())
        for eid, ev in sub.events.items():
            for sid, s in ev.solutions.items():
                if not s.is_active:  # active ones are in run_validation
                    add(sid, s.run_validation())
    else:
        for f, j in sols.items():
            m = vp.check_solution_completeness(
                j["model_type"], j["parameters"], j["higher_order_effects"],
                j["bands"], j["t_ref"])  # fmt: skip
            m += vp.validate_parameter_uncertainties(
                j["parameters"], j["parameter_uncertainties"])  # fmt: skip
            m += vp.validate_solution_consistency(
                j["model_type"], j["parameters"],
                relative_probability=j["relative_probability"])  # fmt: skip
            add(j["solution_id"], m)
        probs = collections.defaultdict(float)
        for f, j in sols.items():
            if j["is_active"]:
                probs[f.split("/")[-3]] += j["relative_probability"]
        for k, v in probs.items():
            if abs(v - 1) > 1e-6:
                errors.append("%s: active probabilities sum to %.6f" % (k, v))

    sj = json.loads((root / "submission.json").read_text())
    if sj.get("team_name") == "TEAM_NAME_TBD":
        warnings.append("submission: Warning: team_name is the placeholder "
                        "TEAM_NAME_TBD; the PI must supply it")  # fmt: skip
    listing = (
        Path(a.listing)
        if a.listing
        else root.parent / "evidence" / "manifest_resolved.json"
    )
    traces = {}
    if listing.exists():
        for r in json.loads(listing.read_text())["submitted"]:
            if r.get("trace"):
                traces["RMDC26_%06d" % int(r["event"])] = r["trace"]
    n_rt = 0
    for f, j in sols.items():
        sid = j["solution_id"]
        eid = f.split("/")[-3]
        if not sid.startswith(eid):
            errors.append("%s: in directory %s" % (sid, eid))
        ev_json = json.loads(
            (root / "events" / eid / "event.json").read_text()
        )
        if ev_json.get("event_id") != eid:
            errors.append(
                "%s: event.json says %s" % (eid, ev_json.get("event_id"))
            )
        if not (root / j["notes_path"]).exists():
            errors.append("%s: notes_path %s missing" % (sid, j["notes_path"]))
        p = j["parameters"]
        if not p.get("t0", 0) > 2.4e6:
            errors.append("%s: t0 %r is not a full BJD" % (sid, p.get("t0")))
        if j["model_type"] == "1S2L" and not 0 <= p["alpha"] < 2 * math.pi:
            errors.append("%s: alpha %r outside [0, 2 pi)" % (sid, p["alpha"]))
        pp = j.get("physical_parameters") or {}
        extra = set(pp) - set(vp.PHYSICAL_PARAMETERS)
        if extra:
            errors.append("%s: physical keys not in the schema: %s"
                          % (sid, sorted(extra)))  # fmt: skip
        add(sid + " (physical)", vp.validate_physical_parameters(pp))
        if eid in traces:
            res = traces[eid].replace("_trace.nc", "_results.csv")
            if os.path.exists(res):
                da, dt = roundtrip(j, results_rows(res))
                n_rt += 1
                if dt is None or (da is None and "alpha" in p):
                    errors.append("%s: round trip compared nothing: %s has "
                                  "no %s row" % (sid, res, "source.t_0"
                                  if dt is None else "lens.*.alpha"))  # fmt: skip
                if da is not None and da > 10.0:
                    errors.append("%s: alpha %.3f deg back from radians is "
                                  "%.1f deg from every results.csv alpha"
                                  % (sid, math.degrees(p["alpha"]), da))  # fmt: skip
                if dt is not None and dt > 1.0:
                    errors.append("%s: t0 %.5f is %.3f d from results.csv"
                                  % (sid, p["t0"], dt))  # fmt: skip
    counts = collections.Counter(
        w.split(": ", 1)[1].split("(")[0][:100] for w in warnings
    )
    out = {"version": version, "mode": mode, "n_solutions": len(sols),
           "n_roundtrip": n_rt, "errors": errors, "warnings": warnings,
           "warning_counts": dict(counts)}  # fmt: skip
    txt = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print("validation [%s]: %d solutions, %d errors, %d warnings, %d alpha/"
          "t0 round trips" % (mode, len(sols), len(errors), len(warnings),
                              n_rt))  # fmt: skip
    for e in errors:
        print("  ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
