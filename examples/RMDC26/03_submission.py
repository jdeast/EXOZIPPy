"""Stage 3: the microlens-submit tree for the campaign's MANIFEST.

    python 03_submission.py [--no-validate]

Reads what `01_evidence.py --manifest` wrote under $RMDC26_REPORT:

  evidence/manifest_resolved.json   the submitted events (tier lists read
                                    at stage-1 time), the fit each one uses
                                    (out_long/ preferred) and the problems
  evidence/<NNN>/<fit>.json         the per-fit records

and writes, under $RMDC26_REPORT:

  submission/submission.json        team TEAM_NAME_TBD (placeholder), tier
                                    "beginner"
  submission/events/<id>/event.json
  submission/events/<id>/notes.md   the campaign's generated note VERBATIM
                                    + one appended section (all solutions
                                    of the event point here)
  submission/events/<id>/solutions/<solution_id>.json
  report/submission_summary.{txt,json,csv}
  report/validation.json            04_validate.py (microlens-submit's own
                                    models and validators, from the wheel)

The submission/ tree is rebuilt from scratch on every run, so an event that
leaves the tier lists leaves the submission.  Per event:

- probabilities: 02_report.build_event (normalized lnZ, valid draws only,
  +/-2 nat floor, equal shares inside indistinguishable groups, refused
  bridges INDICATIVE only), normalized over the event's own solutions;
- architecture: rmdc26.architecture_verdict (the PI-approved blind-grid
  dchi2_np rule for 1S2L, the outlier audit for 1S1L);
- physical_parameters: FULL tier only, per draw within each mode;
- the extinction caveat only above max Rhat 1.3.

Light work: run it under `srun -p test` (run_submission.sbatch does).
"""

import argparse
import csv
import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rmdc26 as R  # noqa: E402


def _load_report():
    spec = importlib.util.spec_from_file_location(
        "report02", HERE / "02_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPORT = _load_report()

#: The challenge requires hardware_info; the fits' own compute is in each
#: solution's compute_info.
HARDWARE_INFO = {
    "cluster": "Harvard FASRC Cannon (Slurm)",
    "notes": "SED-tied fits sampled with exozippy ptde_async on 64-core "
    "Cannon nodes; per-solution wall and CPU hours are in compute_info",
}
REPO_URL = "https://github.com/jdeast/EXOZIPPy"


def cpu_details():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def model_type_check(sols, arch):
    """Problems when the model type disagrees with the fit's own parameters
    (s and q present <=> 1S2L)."""
    out = []
    for s in sols:
        has = {"s", "q"} <= set(s["summary"])
        if has != (arch == R.ARCH_BINARY):
            out.append("%s: model_type %s but s/q %s in the posterior"
                       % (s["solution_id"], arch,
                          "present" if has else "absent"))  # fmt: skip
    return out


def build_submitted_event(row, rec, dchi2_table, campaign_note):
    """One submitted event: 02_report's probabilities + tier, architecture,
    physical parameters, flags and the notes text."""
    ev = int(row["event"])
    rec = dict(rec, fit=R.ev3(ev))  # plain solution ids (no fit suffix)
    e = REPORT.build_event(ev, [rec])
    sols = e["solutions"]
    tier = row["tier"]
    arch = rec["arch"]
    verdict = R.parse_verdict_line(row.get("verdict"))
    problems = model_type_check(sols, arch)

    # architecture: dchi2 for the top solution's q; ANOMALY from every mode
    top = max(sols, key=lambda s: s["p_robust"]) if sols else None
    anomaly = sorted({f for s in sols
                      for f, ks in (s.get("outlier_kinds") or {}).items()
                      if "ANOMALY" in ks})  # fmt: skip
    d = dchi2_table.get(R.ev3(ev)) if arch == R.ARCH_BINARY else None
    av = R.architecture_verdict(
        arch, d, q=(top["summary"].get("q") if top else None),
        anomaly_files=anomaly,
    )  # fmt: skip
    extra = R.ARCH_EVENT_NOTES.get(ev)
    if extra:
        av["statement"] += " " + extra
    if arch == R.ARCH_PSPL and tier == R.TIER_FULL:
        av["statement"] += (
            " Its FULL status rests on t_E, rho and Rhat only (no grid, "
            "no s or q)."
        )
    e["architecture"] = av

    max_rhat = verdict.get("rhat", float("nan"))
    if not np.isfinite(max_rhat):
        max_rhat = float(rec.get("max_rhat", np.nan))
    e["max_rhat"] = max_rhat
    e["extinction_caveat"] = R.extinction_caveat(max_rhat)

    flags = []
    if av["flag"]:
        flags.append(av["flag"])
    if e["basis"] != "accepted":
        flags.append("INDICATIVE_EVIDENCE")
    n_bad = sum(bool(s["unreliable"]) for s in sols)
    if n_bad:
        flags.append("UNRELIABLE(%d/%d)" % (n_bad, len(sols)))
    if e["extinction_caveat"]:
        flags.append("EXTINCTION_CAVEAT")
    e["flags"] = flags

    for s in sols:
        s["basis_indicative"] = e["basis"] != "accepted"
        s["physical_submit"] = None
        if tier == R.TIER_FULL:
            if s.get("physical"):
                s["physical_submit"] = R.submit_physical(s["physical"])
            else:
                problems.append("%s: FULL tier but no physical parameters "
                                "in the record" % s["solution_id"])  # fmt: skip
    e.update(tier=tier, arch=arch, verdict=verdict,
             verdict_line=row.get("verdict"), fit=row.get("fit"),
             fit_notes=row.get("notes") or [], problems=problems,
             tier_reason=R.tier_reason(tier, verdict))  # fmt: skip
    e["notes_text"] = R.event_notes_markdown(campaign_note, e)
    return e


def write_event(e, root):
    eid = e["event_id"]
    eroot = Path(root) / "events" / eid
    R.dump_json(R.event_json(e["event"]), eroot / "event.json")
    rel = "events/%s/notes.md" % eid
    (Path(root) / rel).write_text(e["notes_text"])
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for s in e["solutions"]:
        sj = R.solution_json(s, rel)
        sj["creation_timestamp"] = stamp
        R.dump_json(sj, eroot / "solutions" / (s["solution_id"] + ".json"))


SUMMARY_COLS = (
    "event", "tier", "arch", "solution_id", "lnZ", "lnZ_err", "bridge_state",
    "relative_probability", "is_active", "group", "event_flags",
    "dchi2_np", "max_rhat", "t0", "u0", "tE", "s", "q", "alpha",
    "unreliable", "warnings",
)  # fmt: skip


def summary_rows(events):
    for e in events:
        for s in e["solutions"]:
            par, _ = R.submit_parameters(s)
            yield {
                "event": R.ev3(e["event"]), "tier": e["tier"],
                "arch": e["arch"], "solution_id": s["solution_id"],
                "lnZ": "%.2f" % s["lnZ"], "lnZ_err": "%.2f" % s["lnZ_err"],
                "bridge_state": s["bridge_state"],
                "relative_probability": "%.4g" % s["relative_probability"],
                "is_active": s["is_active"], "group": s["group"],
                "event_flags": " ".join(e["flags"]),
                "dchi2_np": e["architecture"]["dchi2_np"],
                "max_rhat": "%.3f" % e["max_rhat"],
                "t0": par.get("t0"), "u0": par.get("u0"), "tE": par.get("tE"),
                "s": par.get("s"), "q": par.get("q"),
                "alpha": par.get("alpha"),
                "unreliable": "; ".join(s["unreliable"]),
                "warnings": "; ".join(s["warnings"]),
            }  # fmt: skip


def master_lpcheck(listing):
    """{fit: lpcheck.py result} of THIS tree (master) on each submitted
    fit, from $RMDC26_REPORT/lpcheck_master/ (run_submission.sbatch)."""
    out = {}
    for r in listing["submitted"]:
        p = R.OUT / "lpcheck_master" / ("%s.json" % r.get("fit"))
        if r.get("fit") and p.exists():
            out[r["fit"]] = json.loads(p.read_text())
    return out


def summary_text(events, listing, problems, validation, lpm=None, recs=None):
    L = ["RMDC26 submission summary (%s)" % dt.datetime.now().isoformat(
        timespec="minutes"), ""]  # fmt: skip
    L.append("manifest: %s" % listing["manifest"])
    L.append("team_name: %s (PLACEHOLDER: the PI supplies it); tier: %s"
             % (R.TEAM_NAME_PLACEHOLDER, R.CHALLENGE_TIER))  # fmt: skip
    for t in (R.TIER_FULL, R.TIER_SCORED):
        evs = [e for e in events if e["tier"] == t]
        L.append("%-6s %2d events, %3d solutions (%d active): %s"
                 % (t, len(evs), sum(len(e["solutions"]) for e in evs),
                    sum(s["is_active"] for e in evs for s in e["solutions"]),
                    " ".join(R.ev3(e["event"]) for e in evs)))  # fmt: skip
    unc = [e for e in events if e["architecture"]["flag"] == R.ARCH_UNCERTAIN]
    L.append("")
    L.append("FOR PI REVIEW -- ARCHITECTURE_UNCERTAIN (%d):" % len(unc))
    for e in unc:
        L.append("  %s [%s]: %s" % (R.ev3(e["event"]), e["tier"],
                                     e["architecture"]["statement"]))  # fmt: skip
    if not unc:
        L.append("  none")
    ind = [e for e in events if e["basis"] != "accepted"]
    ref = [(e, s) for e in events for s in e["solutions"]
           if s["bridge_refused"]]  # fmt: skip
    L.append("")
    L.append("Evidence: %d of %d solutions have a REFUSED bridge (indicative "
             "only); events with no accepted mode (INDICATIVE): %s"
             % (len(ref), sum(len(e["solutions"]) for e in events),
                " ".join(R.ev3(e["event"]) for e in ind) or "none"))  # fmt: skip
    for e, s in ref:
        L.append("  %s: %s" % (s["solution_id"], s["bridge_state"]))
    L.append("")
    L.append("%-4s %-6s %-5s %4s %10s %8s  %-40s %s" % (
        "ev", "tier", "arch", "nsol", "dchi2_np", "maxRhat", "flags",
        "architecture"))  # fmt: skip
    for e in events:
        d = e["architecture"]["dchi2_np"]
        L.append("%-4s %-6s %-5s %4d %10s %8.3f  %-40s %s" % (
            R.ev3(e["event"]), e["tier"], e["arch"], len(e["solutions"]),
            "-" if d is None else "%.1f" % d, e["max_rhat"],
            " ".join(e["flags"]) or "-",
            e["architecture"]["statement"][:90]))  # fmt: skip
    L.append("")
    recs = recs or {}
    trees = sorted({(r.get("report_tree"), r.get("report_tree_commit"))
                    for r in recs.values()})  # fmt: skip
    tcs = sorted({str(r.get("trace_git_commit")) for r in recs.values()})
    L.append("lp reproduction (50 stored draws per fit, tolerance %g nat):"
             % R.LPCHECK_TOL)  # fmt: skip
    L.append("  evidence tree(s): %s; trace commit(s): %s" % (
        ", ".join("%s @ %s" % t for t in trees), ", ".join(tcs)))  # fmt: skip
    ev_lp = [(e, recs[e["event"]]["lpcheck"]) for e in events
             if e["event"] in recs]  # fmt: skip
    L.append("  evidence tree: max |dlp| over fits %.3g; FAIL: %s" % (
        max([x["dlp_max_abs"] for _e, x in ev_lp] or [float("nan")]),
        " ".join(R.ev3(e["event"]) for e, x in ev_lp
                 if x["verdict"] != "PASS") or "none"))  # fmt: skip
    lpm = lpm or {}
    if lpm:
        v = sorted(lpm.items())
        L.append("  master (this tree, %s): %d fits checked, max |dlp| %.3g, "
                 "median %.3g; FAIL (>= %g): %s" % (
                     next(iter(lpm.values())).get("tree"), len(v),
                     max(x["dlp_max_abs"] for _k, x in v),
                     float(np.median([x["dlp_max_abs"] for _k, x in v])),
                     R.LPCHECK_TOL,
                     " ".join("%s(%.3g)" % (k[:3], x["dlp_max_abs"])
                              for k, x in v if x["verdict"] != "PASS")
                     or "none"))  # fmt: skip
    else:
        L.append("  master: not run")
    L.append("")
    L.append("Problems (%d):" % len(problems))
    L.extend("  " + p for p in problems) if problems else L.append("  none")
    L.append("")
    if validation is None:
        L.append("Validation: NOT RUN")
    else:
        L.append("Validation (microlens-submit %s, %s): %d errors, %d "
                 "warnings over %d solutions"
                 % (validation.get("version"), validation.get("mode"),
                    len(validation["errors"]), len(validation["warnings"]),
                    validation["n_solutions"]))  # fmt: skip
        for m in validation["errors"]:
            L.append("  ERROR " + m)
        for m, n in sorted(validation.get("warning_counts", {}).items(),
                           key=lambda kv: -kv[1]):  # fmt: skip
            L.append("  warn x%d %s" % (n, m))
    return "\n".join(L) + "\n"


def run_validation(root, out):
    """04_validate.py in its own process: pydantic and psutil come from the
    report's deps directory and must not leak into this one."""
    env = dict(os.environ)
    deps = os.environ.get("RMDC26_MSUB_DEPS", str(R.OUT / "deps"))
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in (deps, env.get("PYTHONPATH")) if p]
    )
    r = subprocess.run(
        [sys.executable, str(HERE / "04_validate.py"), str(root),
         "--out", str(out)], env=env, capture_output=True, text=True,
    )  # fmt: skip
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return json.loads(Path(out).read_text()) if Path(out).exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args(argv)
    listing = json.loads((R.OUT / "evidence" / "manifest_resolved.json")
                         .read_text())  # fmt: skip
    problems = list(listing["problems"])
    dchi2 = json.loads(Path(R.GRID_DCHI2_FILE).read_text())
    sub = R.OUT / "submission"
    rep = R.OUT / "report"
    rep.mkdir(parents=True, exist_ok=True)
    if (sub / "events").exists():
        shutil.rmtree(sub / "events")
    events, recs = [], {}
    for row in listing["submitted"]:
        ev = int(row["event"])
        if row.get("unusable"):
            continue
        p = R.OUT / "evidence" / R.ev3(ev) / (row["fit"] + ".json")
        if not p.exists():
            problems.append("%s: no evidence record %s" % (R.ev3(ev), p))
            continue
        rec = json.loads(p.read_text())
        if "modes" not in rec:
            problems.append("%s: record has no modes (%s)" % (
                R.ev3(ev), rec.get("skipped") or rec.get("error", "")[:200]))  # fmt: skip
            continue
        tr = Path(rec["trace"])
        st = tr.stat()
        if rec.get("trace_identity") != [float(st.st_mtime), int(st.st_size)]:
            problems.append("%s: record is for an older trace: re-run stage 1"
                            % R.ev3(ev))  # fmt: skip
            continue
        lc = rec.get("lpcheck") or {}
        if not lc.get("dlp_max_abs", np.inf) < R.LPCHECK_TOL:
            problems.append("%s: lp NOT reproduced (max |dlp| %s)"
                            % (R.ev3(ev), lc.get("dlp_max_abs")))  # fmt: skip
        note_p = R.CAMPAIGN_NOTES_DIR / ("%s.md" % R.ev3(ev))
        note = note_p.read_text() if note_p.exists() else None
        if note is None:
            problems.append("%s: no campaign note %s (our section only)"
                            % (R.ev3(ev), note_p))  # fmt: skip
        recs[ev] = rec
        e = build_submitted_event(row, rec, dchi2, note)
        e["fit_paths"] = {"trace": rec["trace"], "config": rec.get("config")}
        problems.extend("%s: %s" % (R.ev3(ev), x) for x in e["problems"])
        write_event(e, sub)
        events.append(e)
    hw = dict(HARDWARE_INFO)
    cd = cpu_details()
    if cd:
        hw["cpu_details_reporting_node"] = cd
    R.dump_json({"team_name": R.TEAM_NAME_PLACEHOLDER,
                 "tier": R.CHALLENGE_TIER, "hardware_info": hw,
                 "repo_url": REPO_URL}, sub / "submission.json")  # fmt: skip
    validation = None
    if not a.no_validate:
        validation = run_validation(sub, rep / "validation.json")
    with open(rep / "submission_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for row in summary_rows(events):
            w.writerow(row)
    R.dump_json(
        {"events": [{k: e[k] for k in ("event", "tier", "arch", "flags",
                     "architecture", "basis", "max_rhat", "fit",
                     "fit_notes")} | {"n_solutions": len(e["solutions"])}
                    for e in events],
         "problems": problems}, rep / "submission_summary.json")  # fmt: skip
    txt = summary_text(events, listing, problems, validation,
                       master_lpcheck(listing), recs)  # fmt: skip
    (rep / "submission_summary.txt").write_text(txt)
    print(txt)
    bad = validation is None and not a.no_validate
    return 1 if (bad or (validation and validation["errors"])) else 0


if __name__ == "__main__":
    sys.exit(main())
