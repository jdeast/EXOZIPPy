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

AND WHY IT IS NOT A TOLERANCE TABLE EITHER (JDE 2026-09-22).  Until this
revision the "right mode" test asked for one draw inside FIXED tolerances
(t_0 0.05 d, u_0 0.005, log q 0.030 ...) in every observable at once.  On a
peak that fell in a season gap the honest posterior is 3.6 d wide in t_0
and 0.5 in u_0, so no draw qualified and a posterior that CONTAINED the
truth at 0.1-2 sigma scored the same class, TRUTH_NOT_RECOVERED, as a
wrong-topology fit 1,356 tolerances away (experiment A vs the blind fit on
DC2018-226, review 2.4.14).  The rule is now the one the docstring above
already implied:
  a. DISPERSION GATE.  Is the posterior's own uncertainty believable?  Per
     mode, the drop from the best lp to the median lp is chi2_k / 2 for a
     k-dimensional near-Gaussian mode, so its z-score against k/2 says
     whether the draws are under-dispersed (stuck) or over-dispersed
     (unconverged, hot-contaminated) -- internal to the fit, no truth
     needed.  A gate failure is the PRIMARY failure (UNRELIABLE_POSTERIOR)
     and nothing downstream is judged.
  b. MODE-AWARE PULL.  The truth's pull against the width of the mode it
     is nearest, judged as a joint chi2 p-value, not per-parameter boxes.
     A 30-day-wide t_0 that contains the truth passes exactly as a
     0.05-day one does.
  c. WEIGHTS ON EVIDENCE.  The truth inside a minor mode is a clean PASS
     when the reported weights are trustworthy (evidence, or validated
     occupancy): the data+prior topology is being reported honestly and
     the truth basin is identified within its uncertainties.  It is
     WRONG_BASIN when the weights favour another mode although the
     evidence -- or, with untrusted weights, the best lp -- lies in the
     truth's basin.
  d. INFORMATION, not a threshold.  Every core observable reports its
     posterior width over its prior width; ~1 means the prior came back
     and the sweep says so (TRUTH_UNRECOVERABLE, a PASS with an
     explanation); anything smaller is sculpting and is worth having.
  e. COVERAGE.  Across the sweep the pulls themselves are the dispersion
     test: honest posteriors give standard-normal pulls, and the fraction
     inside 1/2/3 sigma is reported against 68/95/99.7.
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
DEGENERATE_COUNTERPART = (
    "DEGENERATE_COUNTERPART"  # retired 2026-09-22, see .md
)
UNRELIABLE_POSTERIOR = "UNRELIABLE_POSTERIOR"
WRONG_BASIN = "WRONG_BASIN"

# Only (a) and (b) and (d) can PASS at all; (c) and (i) are always hard
# FAILs and (e)/(f)/(g) are always SOFT fails.  See dc18_evaluate.md.
CAN_PASS = {TRUTH_RECOVERED, TRUTH_UNRECOVERABLE, WRONG_ARCHITECTURE}
SOFT_FAIL = {ABSTAINED, INFRASTRUCTURE_FAILURE, UNRELIABLE_POSTERIOR}
REMEDY = {
    ABSTAINED: "longer run (automatic)",
    INFRASTRUCTURE_FAILURE: "fix the code, rerun",
    UNRELIABLE_POSTERIOR: "longer run / ladder tuning; do not read the pulls",
    WRONG_ARCHITECTURE: "architecture escalation (not built)",
    TRUTH_NOT_RECOVERED: "investigate: the data supported it",
    WRONG_BASIN: "investigate the mode weights (evidence vs occupancy)",
}

# Observables whose truth the master file gives.  The third field is the
# (component, parameter) whose resolved bounds give the PRIOR width for the
# information ratio; None for a derived quantity with no bounds of its own.
# Log coordinates are compared in log space.
# Compared in ABSOLUTE VALUE.  (u_0, alpha) -> -(u_0, alpha) is EXACT for a
# static binary with no parallax (conventions.md C23, Skowron Eq. A12), so the
# truth table's trajectory-side sign is not one the fits carry and the mirror
# pair is ONE physical solution, not two modes.  dc18_common.py has done this
# since the comparison table existed (truth["u_0"] = abs(truth["u_0"])); the
# mode-aware evaluator did not, and scored the sign as a pull -- on event 004
# that put the truth in an 8% mode at 16.4 sigma when it is in the 90.6%
# favourite at 1.04.  Taking abs of the DRAWS too is what merges the mirror.
ABS_COMPARED = {"u_0"}

OBSERVABLES = [
    # (truth key, candidate trace names, prior (component, param), log10?)
    ("t_0", ("source.t_0", "lens.t_0"), ("source", "t_0"), False),
    ("u_0", ("source.u_0", "lens.u_0"), ("source", "u_0"), False),
    ("t_E", ("mulensevent.t_E", "lens.t_E"), None, False),
    ("rho", ("source.log_rho", "lens.log_rho"), ("source", "log_rho"), True),
    ("s", ("lens.Companion.log_s", "lens.log_s"), ("lens", "log_s"), True),
    (
        "q",
        ("planet.Companion.log_q", "planet.log_q"),
        ("planet", "log_q"),
        True,
    ),
]
# The truth is "in" a mode when its joint pull against that mode's width
# is not rejected at 3 sigma (chi2 survival over the compared observables).
PULL_P_MIN = 0.0027
# The dispersion gate: |z| of the mode's lp drop against its chi2_k/2
# expectation.  5 is loose on purpose -- real posteriors are not Gaussian
# and k counts every sampled dimension, constrained or not -- so the gate
# only catches the gross cases (chains frozen in one basin's floor, or a
# mode padded with hot-rung draws).
DISPERSION_Z_MAX = 5.0
# A posterior whose width is this fraction of its prior's has returned the
# prior; the ratio itself is always reported.
PRIOR_RECOVERED_RATIO = 0.9


def _pick(ds, names):
    """The trace column for the first candidate name that exists, as one
    value per (chain, draw).

    A per-element parameter (lens.log_s, lens.xalpha, ... on a 2-body lens)
    carries an extra element dimension, and one element is the PINNED
    primary (log_s[0] constant at 0).  Ravelling the whole array, as this
    did until 2026-09-21, interleaved the pinned zeros with the free
    element and then truncated to the other columns' length, so every
    per-draw comparison against `s` was misaligned and half its entries
    were the pin: expA/226 scored a "nearest 11.5 tol" with a global median
    log_s of exactly 0.0.  Select the element that actually varies (the
    free companion); if none varies, take the last element.
    """
    for n in names:
        if n in ds.data_vars:
            v = ds[n]
            extra = [d for d in v.dims if d not in ("chain", "draw")]
            if extra:
                dim = extra[0]
                stds = [
                    float(np.nanstd(np.asarray(v.isel({dim: i}))))
                    for i in range(v.sizes[dim])
                ]
                i = int(np.argmax(stds)) if max(stds) > 0 else v.sizes[dim] - 1
                v = v.isel({dim: i})
            return n, np.asarray(v).ravel().astype(float)
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


def _robust_sd(x):
    mad = float(np.median(np.abs(x - np.median(x))))
    sd = 1.4826 * mad
    return sd if sd > 0 else float(np.std(x))


def _parse_modes_txt(modes_txt):
    """Weight provenance and per-mode lnZ from the run's own mode report.

    identify_modes re-run on the trace gives labels and occupancy, but the
    evidence attempt (bridge sampling) needs the model and happened only in
    the run, so its verdict is read back from the text it wrote."""
    out = {"provenance": None, "lnZ": {}}
    if not modes_txt or not Path(modes_txt).exists():
        return out
    t = Path(modes_txt).read_text(errors="ignore")
    m = re.search(r"^weight provenance:\s*(.*)$", t, re.M)
    if m:
        out["provenance"] = m.group(1).strip()
    # lnZ is reported as "lnZ=92499.73+/-0.41"; a bare [-+0-9.eE]+ swallows
    # the leading "+" of the "+/-" and float() then raises on "92499.73+".
    for mm in re.finditer(
        r"mode (\d+):\s*lnZ\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", t
    ):
        out["lnZ"][int(mm.group(1)) - 1] = float(mm.group(2))
    return out


def _weights_trusted(provenance):
    """Evidence weights, or occupancy the report itself validated."""
    if not provenance:
        return False
    p = provenance.lower()
    return (
        p.startswith("evidence")
        or p.startswith("unimodal")
        or (
            p.startswith("occupancy")
            and "validated" in p
            and "unreliable" not in p
        )
    )


def _n_sampled_dims(post):
    n = 0
    for v in post.data_vars:
        if not str(v).endswith("_raw"):
            continue
        shape = [
            post[v].sizes[d]
            for d in post[v].dims
            if d not in ("chain", "draw")
        ]
        n += int(np.prod(shape)) if shape else 1
    return n


def mode_geometry(trace, truth, modes_txt=None, idata=None):
    """Mode-aware geometry: which mode holds the truth, how well, and whether
    that mode's own dispersion is believable.

    Every mode gets the truth's pull against ITS width (robust sd of its
    draws) per observable, the joint chi2 p-value of those pulls, and an lp
    dispersion z-score.  The truth's mode is the one with the highest joint
    p-value.  Nothing here is an absolute tolerance.
    """
    import arviz as az
    from scipy.stats import chi2 as chi2_dist

    from exozippy.outputs.modes import identify_modes

    if idata is None:
        idata = az.from_netcdf(trace)
    post = idata.posterior
    cols, tvals, names = [], [], []
    for key, cands, _prior, is_log in OBSERVABLES:
        if key not in truth:
            continue
        n, v = _pick(post, cands)
        if n is None:
            continue
        t = float(truth[key])
        if key in ABS_COMPARED:
            t = abs(t)
            v = np.abs(v)
        if is_log:
            if t <= 0:
                continue
            t = float(np.log10(t))
        cols.append(v)
        tvals.append(t)
        names.append(key)
    if not cols:
        return None
    X = np.vstack(cols)
    tvals = np.array(tvals)

    report = identify_modes(idata, attach=False)
    labels = report.labels.ravel()
    lp = None
    if hasattr(idata, "sample_stats") and "lp" in idata.sample_stats:
        lp = np.asarray(idata.sample_stats["lp"]).ravel()
    k = _n_sampled_dims(post)
    txt = _parse_modes_txt(modes_txt)
    provenance = txt["provenance"] or report.provenance

    modes = []
    for m in report.modes:
        sel = labels == m.index
        sel &= np.isfinite(X).all(axis=0)
        if sel.sum() < 20:
            continue
        pulls, sds, meds = {}, {}, {}
        for i, nm in enumerate(names):
            w = X[i, sel]
            sd = _robust_sd(w)
            med = float(np.median(w))
            meds[nm] = med
            sds[nm] = sd
            pulls[nm] = (tvals[i] - med) / sd if sd > 0 else np.inf
        finite = [p for p in pulls.values() if np.isfinite(p)]
        chi2 = float(np.sum(np.square(finite))) if finite else np.inf
        p_joint = float(chi2_dist.sf(chi2, len(finite))) if finite else 0.0
        disp = None
        if lp is not None and k > 0:
            lpm = lp[sel]
            lpm = lpm[np.isfinite(lpm)]
            if lpm.size >= 20:
                drop = float(lpm.max() - np.median(lpm))
                expect = 0.5 * k
                disp = {
                    "lp_drop": drop,
                    "expected": expect,
                    "z": (drop - expect) / np.sqrt(expect),
                }
        modes.append(
            {
                "index": int(m.index),
                "weight": float(m.weight),
                "occ_weight": float(m.occ_weight),
                "n_draws": int(sel.sum()),
                "lp_max": float(m.lp_max),
                "delta_lp_max": float(m.delta_lp_max),
                "lnZ": txt["lnZ"].get(int(m.index)),
                "medians": meds,
                "sds": sds,
                "pulls": pulls,
                "worst_pull": max(abs(p) for p in pulls.values())
                if pulls
                else np.inf,
                "p_joint": p_joint,
                "dispersion": disp,
            }
        )
    if not modes:
        return None
    truth_mode = max(modes, key=lambda d: d["p_joint"])
    favourite = min(
        modes, key=lambda d: d["index"]
    )  # index 0 = highest weight
    return {
        "params": names,
        "k_sampled": k,
        "n_modes": len(modes),
        "modes": modes,
        "truth_mode": truth_mode["index"],
        "truth_p": truth_mode["p_joint"],
        "truth_worst_pull": truth_mode["worst_pull"],
        "truth_pulls": truth_mode["pulls"],
        "truth_mode_weight": truth_mode["weight"],
        "favourite": favourite["index"],
        "favourite_weight": favourite["weight"],
        "provenance": provenance,
        "weights_trusted": _weights_trusted(provenance),
        "weights_reliable_by_mixing": bool(report.weights_reliable),
        "n_chains_no_switch": int(report.n_chains_no_switch),
        "invalid_frac": float(report.invalid_frac),
        # kept for readers of the old field: the nearest mode's worst pull
        "nearest_tol": truth_mode["worst_pull"],
    }


def prior_widths_from_config(config_path):
    """Prior sd per observable from the fit's own resolved bounds.

    Builds the System through prepare() only (I/O + registration + the
    relaxation engine; no PyMC graph), then reads each observable's
    lower/upper and returns the sd of a uniform over that box.  A parameter
    with no bounds (derived, or Gaussian-priored) is left out and its ratio
    reported as None.
    """
    import yaml

    from exozippy.system import System

    cfg = yaml.safe_load(open(config_path))
    pf = cfg.get("parameter_file")
    user = (yaml.safe_load(open(pf)) or {}) if pf and Path(pf).exists() else {}
    system = System(cfg, user)
    system.prepare()
    cm = system.config_manager
    widths = {}
    for key, _cands, prior, _log in OBSERVABLES:
        if prior is None:
            continue
        comp, param = prior
        try:
            n = getattr(system, comp).n_elements
            r = cm.resolve(comp, param, shape=(n,))
            lo = np.asarray(r.get("lower"), dtype=float).ravel()
            hi = np.asarray(r.get("upper"), dtype=float).ravel()
            # the free element on a pinned pair is the one with a real box
            span = hi - lo
            span = span[np.isfinite(span) & (span > 0)]
            if span.size:
                widths[key] = float(span.max() / np.sqrt(12.0))
        except Exception as exc:  # noqa: BLE001
            widths[key] = None
            widths[key + "_error"] = f"{type(exc).__name__}: {exc}"
    return widths


def informative(trace, prior_widths=None, idata=None):
    """Posterior width over prior width, per observable, whole posterior.

    ~1 means the prior came back (nothing learned about that parameter);
    the smaller the ratio the more the data sculpted it.  The event-level
    verdict `prior_recovered` is True when every core observable with a
    known prior width is at or above PRIOR_RECOVERED_RATIO -- that fit is a
    non-detection wearing a measurement's clothes and folds into
    TRUTH_UNRECOVERABLE (JDE 2026-09-11, 2026-09-22).  The ratios are
    always reported; the label is the only threshold.
    """
    import arviz as az

    if idata is None:
        idata = az.from_netcdf(trace)
    post = idata.posterior
    ratios = {}
    for key, cands, _prior, _log in OBSERVABLES:
        n, v = _pick(post, cands)
        if n is None:
            continue
        v = v[np.isfinite(v)]
        if v.size < 10:
            continue
        pw = (prior_widths or {}).get(key)
        ratios[key] = (_robust_sd(v) / pw) if pw else None
    known = [r for r in ratios.values() if r is not None]
    return {
        "sd_over_prior": ratios,
        "prior_recovered": (min(known) >= PRIOR_RECOVERED_RATIO)
        if known
        else None,
        "informative": (min(known) < PRIOR_RECOVERED_RATIO) if known else None,
    }


def _result(event, cls, verdict, soft, detail, **extra):
    out = {
        "event": event,
        "class": cls,
        "verdict": verdict,
        "soft": soft,
        "detail": detail,
    }
    if cls in REMEDY and verdict == "FAIL":
        out["remedy"] = REMEDY[cls]
    out.update(extra)
    return out


def _fmt_pulls(pulls):
    return ", ".join("%s %+.1f" % (k, v) for k, v in pulls.items())


def classify(
    event,
    truth,
    trace=None,
    log=None,
    detect_json=None,
    modes_txt=None,
    prior_widths=None,
    config=None,
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
            return _result(
                event,
                WRONG_ARCHITECTURE,
                "FAIL",
                False,
                "no detectable anomaly for the 2L1S seeder, and no PSPL "
                "rung exists to fall back to",
            )
        return _result(
            event,
            INFRASTRUCTURE_FAILURE,
            "FAIL",
            True,
            "crashed before producing a posterior",
        )
    if not (trace and Path(trace).exists()):
        return _result(
            event, INFRASTRUCTURE_FAILURE, "FAIL", True, "no trace on disk"
        )

    import arviz as az

    idata = az.from_netcdf(trace)
    geo = mode_geometry(trace, truth, modes_txt=modes_txt, idata=idata)
    if geo is None:
        return _result(
            event,
            INFRASTRUCTURE_FAILURE,
            "FAIL",
            True,
            "no comparable observables in the trace",
        )
    if prior_widths is None and config and Path(config).exists():
        try:
            prior_widths = prior_widths_from_config(config)
        except Exception as exc:  # noqa: BLE001
            prior_widths = {"_error": f"{type(exc).__name__}: {exc}"}
    inf = informative(trace, prior_widths, idata=idata)
    common = {"geometry": geo, "informativeness": inf}

    # (e) the pipeline declined.  A soft FAIL: better than a confident wrong
    # answer, still not a result.
    if sig.get("abstained") and not sig.get("wrapup_ok"):
        return _result(
            event,
            ABSTAINED,
            "FAIL",
            True,
            "declined to report (non-convergence / invalid draws); truth's "
            "nearest mode has worst pull %.1f sigma" % geo["truth_worst_pull"],
            **common,
        )

    # (b) the prior came back: an accurate statement that nothing was
    # measured.  Checked before the gate -- a prior-width posterior has no
    # dispersion to judge.
    if inf["prior_recovered"]:
        return _result(
            event,
            TRUTH_UNRECOVERABLE,
            "PASS",
            False,
            "posterior is prior-width in every core observable (sd/prior: %s); "
            "returned the prior rather than a claim"
            % ", ".join(
                "%s %.2f" % (k, v)
                for k, v in inf["sd_over_prior"].items()
                if v is not None
            ),
            **common,
        )

    # (g) the dispersion gate, on the truth's mode and on the favourite.
    tm = next(m for m in geo["modes"] if m["index"] == geo["truth_mode"])
    fav = next(m for m in geo["modes"] if m["index"] == geo["favourite"])
    for which, m in (("truth's mode", tm), ("favourite mode", fav)):
        d = m["dispersion"]
        if d is not None and abs(d["z"]) > DISPERSION_Z_MAX:
            kind = "under" if d["z"] < 0 else "over"
            return _result(
                event,
                UNRELIABLE_POSTERIOR,
                "FAIL",
                True,
                "%s (mode %d) is %s-dispersed: lp drop %.1f vs chi2_%d/2 = %.1f "
                "expected (z = %+.1f); its pulls (%s) are not to be read"
                % (
                    which,
                    m["index"] + 1,
                    kind,
                    d["lp_drop"],
                    geo["k_sampled"],
                    d["expected"],
                    d["z"],
                    _fmt_pulls(m["pulls"]),
                ),
                **common,
            )
    if geo["invalid_frac"] > 0.1:
        return _result(
            event,
            UNRELIABLE_POSTERIOR,
            "FAIL",
            True,
            "%.0f%% of draws numerically invalid"
            % (100 * geo["invalid_frac"]),
            **common,
        )

    found = geo["truth_p"] >= PULL_P_MIN
    if found:
        wt = geo["truth_mode_weight"]
        where = "mode %d (weight %.3f, %s)" % (
            geo["truth_mode"] + 1,
            wt,
            geo["provenance"] or "provenance unknown",
        )
        pulls = "pulls %s (joint p = %.2g)" % (
            _fmt_pulls(tm["pulls"]),
            geo["truth_p"],
        )
        if geo["truth_mode"] == geo["favourite"]:
            return _result(
                event,
                TRUTH_RECOVERED,
                "PASS",
                False,
                "truth in the favourite %s; %s" % (where, pulls),
                weight_check="favourite"
                if geo["weights_trusted"]
                else "UNVERIFIED (truth mode is also the favourite)",
                **common,
            )
        # truth in a minor mode: honest topology or wrong basin?
        if geo["weights_trusted"]:
            return _result(
                event,
                TRUTH_RECOVERED,
                "PASS",
                False,
                "truth in minor %s while the %s weights favour mode %d at %.3f: "
                "the data+prior prefer the other basin and say so; %s"
                % (
                    where,
                    geo["provenance"].split(" ")[0],
                    geo["favourite"] + 1,
                    geo["favourite_weight"],
                    pulls,
                ),
                weight_check="trusted, favours another mode",
                **common,
            )
        # untrusted weights: the best lp is the only basin arbiter we have
        if tm["delta_lp_max"] <= 0.0 and fav["delta_lp_max"] > 0.0:
            return _result(
                event,
                WRONG_BASIN,
                "FAIL",
                False,
                "truth in minor %s and the best lp of the whole run is in it "
                "(favourite mode %d is %.1f nats below), yet the reported weights "
                "favour the other basin and are not trustworthy; %s"
                % (where, geo["favourite"] + 1, fav["delta_lp_max"], pulls),
                weight_check="occupancy contradicts lp",
                **common,
            )
        return _result(
            event,
            TRUTH_RECOVERED,
            "PASS",
            False,
            "truth in minor %s; favourite mode %d holds the best lp (truth's "
            "mode %.1f nats below) so the occupancy ordering is consistent, but "
            "the weights themselves are UNVERIFIED; %s"
            % (where, geo["favourite"] + 1, tm["delta_lp_max"], pulls),
            weight_check="UNVERIFIED",
            **common,
        )

    # truth in no mode: (b) or (c), and only a detectability test tells
    # them apart.  Refuse to guess.
    near = "nearest mode %d: worst pull %.1f sigma (%s)" % (
        geo["truth_mode"] + 1,
        geo["truth_worst_pull"],
        _fmt_pulls(tm["pulls"]),
    )
    delta = det.get("delta_logp_truth_minus_best")
    if delta is None:
        return _result(
            event,
            TRUTH_NOT_RECOVERED,
            "FAIL",
            False,
            "truth in no mode (%s) and NO detectability test was run, so (b) vs "
            "(c) is UNDETERMINED -- scored as (c), the pessimistic read"
            % near,
            undetermined=True,
            **common,
        )
    if delta < -20:
        return _result(
            event,
            TRUTH_UNRECOVERABLE,
            "PASS",
            False,
            "truth is disfavoured by %.1f nats; not detecting it is correct (%s)"
            % (-delta, near),
            **common,
        )
    return _result(
        event,
        TRUTH_NOT_RECOVERED,
        "FAIL",
        False,
        "truth within %.1f nats of the best solution but its mode was never "
        "visited: a SEARCH failure (%s)" % (abs(delta), near),
        **common,
    )


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
        WRONG_BASIN,
        WRONG_ARCHITECTURE,
        UNRELIABLE_POSTERIOR,
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
    unver = [
        r
        for r in results
        if str(r.get("weight_check", "")).startswith("UNVERIFIED")
    ]
    if unver:
        print(
            "  NOTE %d recovered event(s) could not have their MODE WEIGHTS "
            "verified (per-mode evidence unavailable, review 1.11.3), so "
            "(a)-PASS is provisional." % len(unver)
        )
    # COVERAGE: the sweep-level dispersion test.  Pulls of the truth against
    # the truth's mode, over gated events, excluding parameters whose
    # posterior is prior-width (those always contain the truth).
    pulls = []
    for r in results:
        geo, inf = r.get("geometry"), r.get("informativeness") or {}
        if not geo or r["class"] in (
            UNRELIABLE_POSTERIOR,
            ABSTAINED,
            INFRASTRUCTURE_FAILURE,
        ):
            continue
        ratios = inf.get("sd_over_prior") or {}
        for k, v in geo["truth_pulls"].items():
            rr = ratios.get(k)
            if rr is not None and rr >= PRIOR_RECOVERED_RATIO:
                continue
            if np.isfinite(v):
                pulls.append(abs(v))
    if pulls:
        a = np.array(pulls)
        print()
        print(
            "COVERAGE of the truth by the truth's mode, %d informative "
            "(event, observable) pulls:" % a.size
        )
        for s_, want in ((1, 68.3), (2, 95.4), (3, 99.7)):
            print(
                "    within %d sigma: %5.1f%%  (honest posteriors: %.1f%%)"
                % (s_, 100.0 * np.mean(a <= s_), want)
            )
    info = [
        (r["event"], k, v)
        for r in results
        for k, v in (
            (r.get("informativeness") or {}).get("sd_over_prior") or {}
        ).items()
        if v is not None
    ]
    if info:
        print()
        print(
            "INFORMATION (posterior sd / prior sd; ~1 = the prior came back):"
        )
        for ev, k, v in info:
            print(
                "    event %-5s %-4s %.3f%s"
                % (
                    ev,
                    k,
                    v,
                    "  <- prior recovered"
                    if v >= PRIOR_RECOVERED_RATIO
                    else "",
                )
            )
