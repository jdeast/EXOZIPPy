"""Stage 2: per-event probabilities, the microlens-submit files, the tables.

    python 02_report.py [94 173 ...] [--events-file list.txt]

Reads $RMDC26_REPORT/evidence/<ev>/*.json (01_evidence.py) -- every event
present when no list is given -- and writes, under $RMDC26_REPORT:

  submission/events/<event_id>/event.json
  submission/events/<event_id>/solutions/<solution_id>.json   one per mode
  submission/events/<event_id>/solutions/<solution_id>.md     its notes
  submission/submission.json            team/tier left blank for the PI
  report/<ev>_report.txt                the human-readable event summary
  report/campaign_summary.{txt,csv}     one line per solution

Every identified mode of every fit is a solution.  The probability rule is
rmdc26.robust_probabilities: a softmax of the normalized lnZ over ALL modes
of ALL architectures of the event, then equal shares inside each group the
+/-2 nat systematic floor cannot order.  Only modes with an ACCEPTED bridge
(and a reproduced lp) enter it; if an event has none, the refused
estimates are used and every number is labelled INDICATIVE.  No CPU-heavy
work: run it under `srun -p test -c 1`.
"""

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rmdc26 as R  # noqa: E402


def load_event(ev):
    recs = []
    for p in sorted((R.OUT / "evidence" / R.ev3(ev)).glob("*.json")):
        recs.append(json.loads(p.read_text()))
    return recs


#: Fit-level fields every one of its solutions inherits for the flags.
FIT_FIELDS = (
    "lpcheck", "equilibrated", "converged", "max_rhat", "worst_rhat_var",
    "min_ess", "stuck_frac", "lp_rhat", "invalid_frac", "out_scale_rhat",
    "t0_par", "bands", "finite_source", "arch", "fit", "hit_maxtime",
)  # fmt: skip


def build_event(ev, recs):
    """The event's solutions with probabilities, flags and activity."""
    sols, skipped, errors = [], [], []
    for r in recs:
        if "error" in r:
            errors.append("%s: %s" % (r["fit"], r["error"]))
            continue
        if "skipped" in r:
            skipped.append(
                "%s (%s): skipped -- %s" % (r["fit"], r["arch"], r["skipped"])
            )
            continue
        for m in r["modes"]:
            s = dict(m)
            for k in FIT_FIELDS:
                s[k] = r.get(k)
            s["solution_id"] = R.solution_id(ev, r["arch"], m["mode"])
            if r["fit"] != R.ev3(ev):
                # a second fit of one architecture must not collide
                s["solution_id"] += "_" + r["fit"].split("_", 1)[-1]
            hours = r.get("fit_wall_h", float("nan"))
            cores = (r.get("config_sampler") or {}).get("cores")
            s["compute_info"] = {
                "wall_time_hours": None if not np.isfinite(hours) else round(hours, 2),
                "cpu_hours": None if not (np.isfinite(hours) and cores)
                else round(hours * cores, 1),
            }  # fmt: skip
            s["unreliable"], s["warnings"] = R.reliability_flags(s)
            lp_ok = (s.get("lpcheck") or {}).get(
                "dlp_max_abs", 1.0
            ) < R.LPCHECK_TOL
            s["accepted"] = bool(not s["bridge_refused"] and lp_ok
                                 and np.isfinite(s["lnZ"]))  # fmt: skip
            sols.append(s)
    basis = "accepted"
    if sols and not any(s["accepted"] for s in sols):
        basis = "INDICATIVE (no accepted bridge)"
    lnz = np.array(
        [
            s["lnZ"] if (s["accepted"] or basis != "accepted") else np.nan
            for s in sols
        ],
        dtype=float,
    )
    # A refused bridge still reports its estimate in lnZ (status lnZ_bridge
    # + log_norm); evidence.py sets lnZ itself for every mode it attempted.
    err = np.array([s["lnZ_err"] for s in sols], dtype=float)
    pr = R.robust_probabilities(lnz, err) if sols else {"groups": []}
    eligible = np.array([np.isfinite(v) for v in lnz], bool)
    act, rel = (
        R.active_and_renormalized(pr["p_robust"], eligible)
        if sols
        else ([], [])
    )
    arch_prob = {}
    for i, s in enumerate(sols):
        s["p_raw"] = float(pr["p_raw"][i])
        s["p_robust"] = float(pr["p_robust"][i])
        s["sigma_total"] = float(pr["sigma"][i])
        s["is_active"] = bool(act[i])
        s["relative_probability"] = float(rel[i])
        g = pr["group_of"][i]
        s["group"] = "G%d" % (g + 1) if g >= 0 else "none"
        s["group_members"] = (
            [sols[j]["solution_id"] for j in pr["groups"][g]] if g >= 0 else []
        )
        arch_prob[s["arch"]] = arch_prob.get(s["arch"], 0.0) + s["p_robust"]
    missing = list(skipped) + errors
    for arch in (R.ARCH_PSPL, R.ARCH_BINARY):
        if arch not in {s["arch"] for s in sols}:
            missing.append(
                "%s: NO usable fit of this architecture for this event, so the "
                "probabilities above are CONDITIONAL on the architectures "
                "fitted" % arch
            )
    return {
        "event": int(ev),
        "event_id": R.event_id(ev),
        "basis": basis,
        "solutions": sols,
        "arch_prob": arch_prob,
        "groups": [[sols[j]["solution_id"] for j in g] for g in pr["groups"]],
        "missing": missing,
    }


def write_submission(event, root):
    eroot = Path(root) / "events" / event["event_id"]
    R.dump_json(R.event_json(event["event"]), eroot / "event.json")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for s in event["solutions"]:
        rel = "events/%s/solutions/%s.md" % (
            event["event_id"],
            s["solution_id"],
        )
        sj = R.solution_json(s, rel)
        sj["creation_timestamp"] = stamp
        R.dump_json(sj, eroot / "solutions" / (s["solution_id"] + ".json"))
        (Path(root) / rel).write_text(R.notes_markdown(s, event))


def _pm(s, key, fmt="%.4g"):
    v = s["summary"].get(key)
    if not v:
        return "-"
    return (fmt + " -" + fmt + " +" + fmt) % tuple(v)


def event_text(event):
    L = ["%s  (probability basis: %s)" % (event["event_id"], event["basis"])]
    L.append("=" * len(L[0]))
    L.append(
        "architecture probability: "
        + ", ".join(
            "%s %.4g" % kv for kv in sorted(event["arch_prob"].items())
        )
    )
    for m in event["missing"]:
        L.append("  note: " + m)
    L.append("")
    L.append(
        "%-24s %5s %14s %6s %6s %9s %9s %4s %6s %12s"
        % ("solution", "arch", "lnZ", "err", "re2", "p_raw", "p_rel",
           "grp", "active", "logL")
    )  # fmt: skip
    order = sorted(event["solutions"], key=lambda s: -s["p_robust"])
    for s in order:
        L.append(
            "%-24s %5s %14.2f %6.2f %6.3f %9.3g %9.3g %4s %6s %12.2f"
            % (s["solution_id"][14:], s["arch"], s["lnZ"], s["lnZ_err"],
               s["re2"], s["p_raw"], s["relative_probability"], s["group"],
               "yes" if s["is_active"] else "no", s["log_likelihood"])
        )  # fmt: skip
    L.append("")
    for s in order:
        L.append("%s  [%s, %d draws, bridge %s]" % (
            s["solution_id"], s["arch"], s["n_draws"], s["bridge_state"]))  # fmt: skip
        for k in ("t_0", "u_0", "t_E", "rho", "s", "q", "alpha_rad",
                  "pi_E_N", "pi_E_E"):  # fmt: skip
            if k in s["summary"]:
                fmt = "%.5f" if k == "t_0" else "%.4g"
                L.append("    %-9s %s" % (k, _pm(s, k, fmt)))
        L.append("    err_scale " + ", ".join(
            "%s %.3g" % kv for kv in sorted(s["err_scale"].items())))  # fmt: skip
        L.append("    UNRELIABLE: " + ("; ".join(s["unreliable"]) or "no"))
        L.append("    warnings:   " + ("; ".join(s["warnings"]) or "none"))
    return "\n".join(L) + "\n"


CSV_COLS = (
    "event", "solution_id", "arch", "mode", "n_draws", "lnZ", "lnZ_err",
    "re2", "bridge_state", "p_raw", "relative_probability", "group",
    "is_active", "arch_prob_1S1L", "arch_prob_1S2L", "log_likelihood",
    "n_data_points", "t_0", "u_0", "t_E", "s", "q", "alpha_rad", "rho",
    "unreliable", "warnings",
)  # fmt: skip


def csv_rows(event):
    for s in event["solutions"]:
        row = {k: s.get(k) for k in CSV_COLS}
        row["event"] = event["event"]
        for a in (R.ARCH_PSPL, R.ARCH_BINARY):
            row["arch_prob_" + a] = event["arch_prob"].get(a, 0.0)
        for k in ("t_0", "u_0", "t_E", "s", "q", "alpha_rad", "rho"):
            row[k] = s["summary"][k][0] if k in s["summary"] else ""
        row["unreliable"] = "; ".join(s["unreliable"])
        row["warnings"] = "; ".join(s["warnings"])
        yield row


def campaign_text(events):
    L = ["RMDC26 campaign summary (%d events)" % len(events), ""]
    L.append(
        "%-4s %-9s %-9s %-26s %8s %s"
        % ("ev", "P(1S1L)", "P(1S2L)", "top solution", "p_rel", "flags")
    )
    for e in events:
        sols = sorted(e["solutions"], key=lambda s: -s["p_robust"])
        if not sols:
            L.append(
                "%-4s no solutions: %s"
                % (R.ev3(e["event"]), "; ".join(e["missing"]))
            )
            continue
        top = sols[0]
        L.append(
            "%-4s %-9.3g %-9.3g %-26s %8.3g %s"
            % (R.ev3(e["event"]), e["arch_prob"].get(R.ARCH_PSPL, 0.0),
               e["arch_prob"].get(R.ARCH_BINARY, 0.0), top["solution_id"][14:],
               top["relative_probability"],
               ("UNRELIABLE " if top["unreliable"] else "")
               + "%d warn" % len(top["warnings"]))
        )  # fmt: skip
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("events", nargs="*", type=int)
    ap.add_argument("--events-file", default=None)
    a = ap.parse_args(argv)
    evs = list(a.events)
    if a.events_file:
        for line in Path(a.events_file).read_text().splitlines():
            tok = line.split("#")[0].split()
            if tok:
                evs.append(int(tok[0]))
    if not evs:
        evs = sorted(int(p.name) for p in (R.OUT / "evidence").iterdir()
                     if p.is_dir() and p.name.isdigit())  # fmt: skip
    sub = R.OUT / "submission"
    rep = R.OUT / "report"
    rep.mkdir(parents=True, exist_ok=True)
    events = []
    for ev in evs:
        recs = load_event(ev)
        if not recs:
            print("%s: no evidence records" % R.ev3(ev))
            continue
        e = build_event(ev, recs)
        events.append(e)
        write_submission(e, sub)
        (rep / ("%s_report.txt" % R.ev3(ev))).write_text(event_text(e))
        R.dump_json(e, rep / ("%s_report.json" % R.ev3(ev)))
    R.dump_json(
        {"team_name": "", "tier": "", "hardware_info": None,
         "repo_url": "https://github.com/jdeast/EXOZIPPy"},
        sub / "submission.json",
    )  # fmt: skip
    with open(rep / "campaign_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for e in events:
            for row in csv_rows(e):
                w.writerow(row)
    txt = campaign_text(events)
    (rep / "campaign_summary.txt").write_text(txt)
    print(txt)
    for e in events:
        print(event_text(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
