"""Classify a DC2018 fit into the sweep's outcome taxonomy.

The taxonomy, the PASS/FAIL rules and the soft/hard split are specified in
dc18_evaluate.md -- read that first; this module only implements it.

Tier 1 is PASS/TOTAL.  Tier 2 is the per-class breakdown with failures split
HARD vs SOFT, plus the mechanical remedy where one exists (ABSTAINED wants a
longer run; INFRASTRUCTURE_FAILURE wants a fix-and-rerun).

DESIGN NOTE on why this is not just a pull table: review 7.15.1 measured
max|pull| to be MONOTONIC IN n_eff on DC2018-128 -- the best-sampled run
scored worst -- because a better-resolved posterior is narrower and
narrow-in-the-wrong-mode is what a pull punishes.  So every judgement here
is mode-aware, and the per-parameter pull appears only INSIDE an identified
mode, which is the question the global pull was failing to ask.
"""

import json
import os
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- classes
TRUTH_RECOVERED = "TRUTH_RECOVERED"
TRUTH_UNRECOVERABLE = "TRUTH_UNRECOVERABLE"
TRUTH_NOT_RECOVERED = "TRUTH_NOT_RECOVERED"
WRONG_ARCHITECTURE = "WRONG_ARCHITECTURE"
ABSTAINED = "ABSTAINED"
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
DEGENERATE_COUNTERPART = "DEGENERATE_COUNTERPART"

# Only (a) and (b) and (d) can PASS at all; (c) is always a hard FAIL and
# (e)/(f)/(h) are always SOFT fails.  See dc18_evaluate.md.
CAN_PASS = {TRUTH_RECOVERED, TRUTH_UNRECOVERABLE, WRONG_ARCHITECTURE}
SOFT_FAIL = {ABSTAINED, INFRASTRUCTURE_FAILURE, DEGENERATE_COUNTERPART}
REMEDY = {
    ABSTAINED: "longer run (automatic)",
    INFRASTRUCTURE_FAILURE: "fix the code, rerun",
    WRONG_ARCHITECTURE: "architecture escalation (not built)",
    TRUTH_NOT_RECOVERED: "investigate: the data supported it",
}

# Observables whose truth the master file gives, with a per-parameter
# tolerance that means "the right MODE" rather than "the right draw".
# Tolerances are in the parameter's own units and are ~1% of a plausible
# range; log coordinates are compared in log space.
OBSERVABLES = [
    # (truth key, candidate trace names, tolerance, log10?)
    ("t_0", ("source.t_0", "lens.t_0"), 0.05, False),
    ("u_0", ("source.u_0", "lens.u_0"), 0.005, False),
    ("t_E", ("mulensevent.t_E", "lens.t_E"), 0.30, False),
    ("rho", ("source.log_rho", "lens.log_rho"), 0.060, True),
    ("s", ("lens.Companion.log_s", "lens.log_s"), 0.010, True),
    ("q", ("planet.Companion.log_q", "planet.log_q"), 0.030, True),
]


def _pick(ds, names):
    for n in names:
        if n in ds.data_vars:
            return n, np.asarray(ds[n]).ravel().astype(float)
    return None, None


def _log_scan(path):
    """What the job log says about how the run ended."""
    if not path or not Path(path).exists():
        return {}
    txt = Path(path).read_text(errors="ignore")
    return {
        "no_anomaly": "NoAnomalyFoundError" in txt,
        "abstained": bool(
            re.search(
                r"rejected as numerically invalid, exceeding|"
                r"CONVERGENCE NOT REACHED|NOT EQUILIBRATED",
                txt,
            )
        ),
        "traceback": "Traceback (most recent call last)" in txt,
        "wrapup_ok": "Wrap-up complete" in txt,
    }


def mode_geometry(trace, truth):
    """Mode-aware geometry: is truth's mode present, and how close?

    Returns nearest joint approach in tolerance units, the posterior weight
    of the truth-local region, and the in-mode pulls.  `nearest` uses the
    WORST parameter per draw, so a draw only counts as "at truth" when every
    observable is simultaneously inside tolerance -- being right about t_0
    while wrong about q is not finding the mode.
    """
    import xarray as xr

    ds = xr.open_dataset(trace, group="posterior")
    cols, tvals, tols, names = [], [], [], []
    for key, cands, tol, is_log in OBSERVABLES:
        if key not in truth:
            continue
        n, v = _pick(ds, cands)
        if n is None:
            continue
        t = float(truth[key])
        if is_log:
            if t <= 0:
                continue
            t = float(np.log10(t))
        cols.append(v)
        tvals.append(t)
        tols.append(tol)
        names.append(key)
    if not cols:
        ds.close()
        return None
    k = min(len(c) for c in cols)
    X = np.vstack([c[:k] for c in cols])
    ok = np.isfinite(X).all(axis=0)
    X = X[:, ok]
    t = np.array(tvals)[:, None]
    tol = np.array(tols)[:, None]
    d = np.abs(X - t) / tol
    worst = d.max(axis=0)
    near = worst <= 1.0
    out = {
        "params": names,
        "nearest_tol": float(worst.min()) if worst.size else float("inf"),
        "truth_mode_weight": float(near.mean()) if worst.size else 0.0,
        "n_draw": int(X.shape[1]),
        "in_mode_pulls": {},
        "global_medians": {},
    }
    for i, nm in enumerate(names):
        v = X[i]
        out["global_medians"][nm] = float(np.median(v))
        if near.sum() >= 20:
            w = v[near]
            sd = float(np.std(w))
            if sd > 0:
                out["in_mode_pulls"][nm] = (
                    float(np.median(w)) - tvals[i]
                ) / sd
    ds.close()
    return out


def informative(trace, prior_widths=None):
    """Is the posterior narrower than its prior, or is it just the prior?

    An interval that contains truth only because it contains everything is a
    non-detection in a measurement's clothing -- UNINFORMATIVE folds into
    TRUTH_UNRECOVERABLE (JDE 2026-09-11).  Measured as the posterior sd
    against the prior's, per observable, and summarised by the SHARPEST
    parameter: if nothing was constrained, nothing was learned.
    """
    import xarray as xr

    ds = xr.open_dataset(trace, group="posterior")
    ratios = {}
    for key, cands, _tol, _log in OBSERVABLES:
        n, v = _pick(ds, cands)
        if n is None:
            continue
        v = v[np.isfinite(v)]
        if v.size < 10:
            continue
        pw = (prior_widths or {}).get(key)
        sd = float(np.std(v))
        ratios[key] = (sd / pw) if pw else None
    ds.close()
    known = [r for r in ratios.values() if r is not None]
    return {
        "sd_over_prior": ratios,
        # no prior widths supplied -> cannot judge; say so rather than guess
        "informative": (min(known) < 0.1) if known else None,
    }


def classify(
    event,
    truth,
    trace=None,
    log=None,
    detect_json=None,
    modes_txt=None,
    prior_widths=None,
):
    """One event -> {class, verdict, soft, detail}.  See dc18_evaluate.md."""
    sig = _log_scan(log)
    det = {}
    if detect_json and Path(detect_json).exists():
        det = json.load(open(detect_json))

    # (f) our code broke -- checked FIRST so plumbing is never scored as
    # physics.  A traceback with no usable posterior is ours.
    if sig.get("traceback") and not (trace and Path(trace).exists()):
        if sig.get("no_anomaly"):
            # (d): the event needs an architecture we never fit.  Until the
            # PSPL rung exists this is "failed to consider it at all".
            return {
                "event": event,
                "class": WRONG_ARCHITECTURE,
                "verdict": "FAIL",
                "soft": False,
                "detail": "no detectable anomaly for the 2L1S seeder, and no "
                "PSPL rung exists to fall back to",
                "remedy": REMEDY[WRONG_ARCHITECTURE],
            }
        return {
            "event": event,
            "class": INFRASTRUCTURE_FAILURE,
            "verdict": "FAIL",
            "soft": True,
            "detail": "crashed before producing a posterior",
            "remedy": REMEDY[INFRASTRUCTURE_FAILURE],
        }

    if not (trace and Path(trace).exists()):
        return {
            "event": event,
            "class": INFRASTRUCTURE_FAILURE,
            "verdict": "FAIL",
            "soft": True,
            "detail": "no trace on disk",
            "remedy": REMEDY[INFRASTRUCTURE_FAILURE],
        }

    geo = mode_geometry(trace, truth)
    inf = informative(trace, prior_widths)
    if geo is None:
        return {
            "event": event,
            "class": INFRASTRUCTURE_FAILURE,
            "verdict": "FAIL",
            "soft": True,
            "detail": "no comparable observables in the trace",
            "remedy": REMEDY[INFRASTRUCTURE_FAILURE],
        }

    # (e) the pipeline declined.  A soft FAIL: better than a confident wrong
    # answer, still not a result.  Checked AFTER geometry so the detail line
    # can say how close it got.
    if sig.get("abstained") and not sig.get("wrapup_ok"):
        return {
            "event": event,
            "class": ABSTAINED,
            "verdict": "FAIL",
            "soft": True,
            "detail": "declined to report (non-convergence / invalid draws); "
            "nearest approach %.2f tol" % geo["nearest_tol"],
            "remedy": REMEDY[ABSTAINED],
            "geometry": geo,
        }

    found = geo["truth_mode_weight"] > 0.0

    # UNINFORMATIVE folds into (b).
    if inf["informative"] is False:
        return {
            "event": event,
            "class": TRUTH_UNRECOVERABLE,
            "verdict": "PASS",
            "soft": False,
            "detail": "posterior is prior-width; returned an uninformative "
            "prior rather than a claim",
            "geometry": geo,
            "informativeness": inf,
        }

    if found:
        # (a) or (h): did we report the truth mode, or a degenerate image?
        wt = geo["truth_mode_weight"]
        degenerate = wt < 0.5 and _has_degenerate_partner(modes_txt)
        if degenerate:
            return {
                "event": event,
                "class": DEGENERATE_COUNTERPART,
                "verdict": "FAIL",
                "soft": True,
                "detail": "truth's mode found at weight %.3f but a degenerate "
                "counterpart is reported as the answer" % wt,
                "geometry": geo,
            }
        return {
            "event": event,
            "class": TRUTH_RECOVERED,
            "verdict": "PASS",
            "soft": False,
            "detail": "truth's mode found at weight %.3f; mode-weight check "
            "UNVERIFIED (per-mode evidence unavailable, 1.11.3)" % wt,
            "geometry": geo,
            "weight_check": "UNVERIFIED",
        }

    # truth's mode absent: (b) or (c), and only a detectability test tells
    # them apart.  Refuse to guess.
    delta = det.get("delta_logp_truth_minus_best")
    if delta is None:
        return {
            "event": event,
            "class": TRUTH_NOT_RECOVERED,
            "verdict": "FAIL",
            "soft": False,
            "detail": "truth's mode not in the posterior (nearest %.2f tol) "
            "and NO detectability test was run, so (b) vs (c) is "
            "UNDETERMINED -- scored as (c), the pessimistic read"
            % geo["nearest_tol"],
            "geometry": geo,
            "undetermined": True,
        }
    if delta < -20:
        return {
            "event": event,
            "class": TRUTH_UNRECOVERABLE,
            "verdict": "PASS",
            "soft": False,
            "detail": "truth is disfavoured by %.1f nats; not detecting it is "
            "correct" % (-delta),
            "geometry": geo,
        }
    return {
        "event": event,
        "class": TRUTH_NOT_RECOVERED,
        "verdict": "FAIL",
        "soft": False,
        "detail": "truth within %.1f nats of the best solution but its mode "
        "was never visited: a SEARCH failure" % abs(delta),
        "geometry": geo,
        "remedy": REMEDY[TRUTH_NOT_RECOVERED],
    }


def _has_degenerate_partner(modes_txt):
    """Does the mode report name more than one mode?

    Deliberately crude: a real close/wide or +/-u_0 test needs the mode
    parameters, and the honest answer today is that the mode report's
    weights are unreliable (1.11.3 -- error bars understated 8-23x).  So
    this only asks whether a partner was reported at all.
    """
    if not modes_txt or not Path(modes_txt).exists():
        return False
    t = Path(modes_txt).read_text(errors="ignore")
    m = re.search(r"(\d+)\s+mode\(s\)", t)
    return bool(m and int(m.group(1)) > 1)


def summarise(results):
    """Tier 1 then tier 2, as specified."""
    n = len(results)
    npass = sum(1 for r in results if r["verdict"] == "PASS")
    print("=" * 68)
    print(
        "TIER 1:  %d / %d PASS  (%.0f%%)"
        % (npass, n, 100.0 * npass / n if n else 0.0)
    )
    print("=" * 68)
    hard = [r for r in results if r["verdict"] == "FAIL" and not r["soft"]]
    soft = [r for r in results if r["verdict"] == "FAIL" and r["soft"]]
    print(
        "         %d pass, %d HARD fail, %d soft fail"
        % (npass, len(hard), len(soft))
    )
    print()
    print("TIER 2 -- by class")
    order = [
        TRUTH_RECOVERED,
        TRUTH_UNRECOVERABLE,
        TRUTH_NOT_RECOVERED,
        WRONG_ARCHITECTURE,
        DEGENERATE_COUNTERPART,
        ABSTAINED,
        INFRASTRUCTURE_FAILURE,
    ]
    for cls in order:
        rs = [r for r in results if r["class"] == cls]
        if not rs:
            continue
        p = sum(1 for r in rs if r["verdict"] == "PASS")
        tag = "soft" if cls in SOFT_FAIL else ("hard" if p == 0 else "")
        print(
            "  %-24s %2d  (%d pass / %d fail) %s"
            % (cls, len(rs), p, len(rs) - p, tag)
        )
        for r in rs:
            print("       event %-5s %s" % (r["event"], r["detail"]))
        rem = {r.get("remedy") for r in rs if r.get("remedy")}
        for x in sorted(x for x in rem if x):
            print("       -> remedy: %s" % x)
    und = [r for r in results if r.get("undetermined")]
    if und:
        print()
        print(
            "  NOTE %d event(s) scored pessimistically as "
            "TRUTH_NOT_RECOVERED because no detectability test was run; "
            "they may belong in TRUTH_UNRECOVERABLE." % len(und)
        )
    unver = [r for r in results if r.get("weight_check") == "UNVERIFIED"]
    if unver:
        print(
            "  NOTE %d recovered event(s) could not have their MODE WEIGHTS "
            "verified (per-mode evidence unavailable, review 1.11.3), so "
            "(a)-PASS is provisional." % len(unver)
        )
