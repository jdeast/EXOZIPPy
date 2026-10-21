"""Shared paths and the PURE logic of the blind 2L1S pipeline (stages 2, 4, 5).

Everything here that decides something -- which grid basins become modes,
when two runs are the same basin, how lnZ becomes a relative probability,
what the report says -- is a plain function of numbers, so
`tests/test_dc2018_069_blind.py` exercises it without a sampler, a trace or
the challenge data.  The stage scripts (01_grid.py ... 05_report.py) only do
I/O around these functions.

This file is also the specification the blind-pipeline DRIVER has to meet:
anything the driver does differently from these functions is a change of
procedure, not a refactor.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

EVENT = 69
NAME = "DC2018_%03d" % EVENT
HERE = Path(__file__).resolve().parent
DC18_DIR = HERE.parent / "DC2018"

#: Where every run artifact goes.  NOT the repo and NOT $HOME (95 G quota):
#: traces are ~1 GB each.  Override with $DC069_RUNS.
RUNS = Path(
    os.environ.get("DC069_RUNS", "/n/holylabs/jjohnson_lab/Lab/dc069_runs")
)

#: The full data release (lc/ulwdc1_NNN_{W149,Z087}.txt, magnitudes).
#: dc18_common reads either layout; this is the one dc18_arch_config's
#: canonical priors were written against.  Override with $DC18_DATA.
DEFAULT_DATA = os.environ.get(
    "DC18_DATA", "/n/holylabs/jjohnson_lab/Lab/dc18_full/data-challenge-1"
)

#: The shipped grid (168 nodes, W149 only, 8 KB): the one the 30-event
#: sweep computed, reused here so the example does not have to pay for it.
SHIPPED_GRID = HERE / "grid" / "sqgrid30_069.npz"

# --- stage 2: grid basins -> candidate modes -------------------------------

#: Candidate modes are grid basins within this many chi2 of the grid's best
#: node.  The grid is a static, W149-only, lattice-resolution chi2 (its
#: values are comparable to each other, not to a fit's), so the cut is
#: generous.  On 069 it keeps the close/wide pair (s = 0.85 / 1.18, dchi2 0
#: and 18) and the second pair (s = 0.53 / 1.91, dchi2 273 and 281); the
#: next basin is at dchi2 992 (s = 3.6) and the rest are beyond 1000.
DCHI2_MAX = 500.0
#: At most this many modes are fitted (each is one 64-core run).
MAX_MODES = 5

# --- stage 4/5 thresholds ---------------------------------------------------

#: Flag a run when more than this fraction of its chains sit > STUCK_DLP
#: below the best chain's mean lp over the second half (launch1015
#: RESTART_PROCEDURE.md: the flag that caught 7/9 detectable failures).
STUCK_FRAC_MAX = 0.1
STUCK_DLP = 20.0
#: Split-Rhat on lp.  Mild false alarms are known (RESTART_PROCEDURE.md),
#: so it is a flag, never an exclusion.
LP_RHAT_MAX = 1.1
#: Bridge relative MSE above this is flagged (evidence.py refuses at its
#: own DEFAULT_RE2_MAX; this is the stricter "worth a look" level).
RE2_FLAG = 0.1
#: Two runs are ONE basin when sign(u_0) agrees and the Mahalanobis
#: distance between their (log s, log q, alpha) posteriors is below this.
DEDUPE_SIGMA = 3.0

ARCH_2L1S = "2L1S"
ARCH_BASELINE = "1L1S_fs"


def dc18_path():
    """Put examples/DC2018 on sys.path (dc18_common, dc18_arch_config,
    dc18_grid_seed live there)."""
    p = str(DC18_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def run_dir(*parts):
    return RUNS.joinpath(*parts)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


def _node_index(grid, cand):
    i = int(np.argmin(np.abs(grid.log_s - cand["log_s"])))
    j = int(np.argmin(np.abs(grid.log_q - cand["log_q"])))
    return i, j


def candidate_modes(grid, dchi2_max=DCHI2_MAX, max_modes=MAX_MODES):
    """Grid basins -> a PRE-DEDUPED list of candidate modes, best first.

    The grid's distinct local minima within `dchi2_max` of the best node,
    plus the s <-> 1/s mirror of each (`GridResult.mirror_of`), because on
    069 the grid's best node IS the mirror of the truth (gridsearch.py
    docstring) -- the close/wide pair must both be fitted, whichever the
    lattice happens to prefer.  A candidate within the grid's own basin
    suppression radius (`default_min_sep`, both axes) of an already
    accepted one is the same basin and is dropped, so a mirror that is
    itself a local minimum is not fitted twice.  Basins are taken best
    first, each immediately followed by its mirror, until `max_modes`; the
    result is then ordered by grid chi2 and named m0, m1, ...  Each entry carries
    `origin` ("minimum k" or "mirror of k").
    """
    sep_s, sep_q = grid.default_min_sep()
    mins = [
        m
        for m in grid.local_minima(k=4 * max_modes)
        if m["dchi2"] <= dchi2_max
    ]
    pool = []
    for k, m in enumerate(mins):
        pool.append(dict(m, origin="minimum %d" % k))
        mir = grid.mirror_of(m)
        if mir is not None and mir["dchi2"] <= dchi2_max:
            pool.append(dict(mir, origin="mirror of minimum %d" % k))
    # NOT re-sorted by chi2 before the cap: each basin is followed by its
    # mirror, so a mirror is never crowded out of `max_modes` by shallow
    # lattice neighbours that happen to score a little better.
    out, taken = [], []
    for c in pool:
        i, j = _node_index(grid, c)
        if any(abs(i - a) < sep_s and abs(j - b) < sep_q for a, b in taken):
            continue
        taken.append((i, j))
        out.append(c)
        if len(out) == max_modes:
            break
    out.sort(key=lambda c: c["chi2"])
    for k, c in enumerate(out):
        c["mode"] = "m%d" % k
    return out


# ---------------------------------------------------------------------------
# Stage 4: per-run numbers that are pure functions of arrays
# ---------------------------------------------------------------------------


def split_rhat(m):
    """Split-Rhat of an (n_chain, n_draw) array."""
    m = np.asarray(m, dtype=float)
    h = m.shape[1] // 2
    m = np.concatenate([m[:, :h], m[:, h : 2 * h]], axis=0)
    n = m.shape[1]
    w = m.var(axis=1, ddof=1).mean()
    b = n * m.mean(axis=1).var(ddof=1)
    if not np.isfinite(w) or w <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def stuck_fraction(lp, dlp=STUCK_DLP):
    """Fraction of chains whose mean lp is > dlp below the best chain's."""
    cm = np.asarray(lp, dtype=float).mean(axis=1)
    return float((cm < cm.max() - dlp).mean())


def wrap180(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def circ_mean_deg(a):
    a = np.radians(np.asarray(a, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def basin_features(log_s, log_q, alpha_deg, u_0):
    """Mean and covariance of (log s, log q, alpha) plus the u_0 sign.

    alpha is unwrapped around its circular mean first, so a basin that
    straddles +/-180 has a sensible covariance.
    """
    log_s = np.ravel(log_s)
    log_q = np.ravel(log_q)
    alpha_deg = np.ravel(alpha_deg)
    a0 = circ_mean_deg(alpha_deg)
    a = a0 + wrap180(alpha_deg - a0)
    X = np.column_stack([log_s, log_q, a])
    pos = float((np.ravel(u_0) > 0).mean())
    return {
        "mean": [float(np.mean(log_s)), float(np.mean(log_q)), a0],
        "cov": np.cov(X, rowvar=False).tolist(),
        "u0_sign": 1 if pos >= 0.5 else -1,
        "u0_sign_purity": max(pos, 1.0 - pos),
    }


def lnz_from_submodes(results):
    """One run's lnZ from its identify_modes sub-modes.

    `results` is a list of dicts with lnZ, lnZ_err, re2, refused.  The run
    is ONE mode of the event, so its evidence is the SUM of its sub-modes'
    (logsumexp), each from its own bridge; the error adds in quadrature with
    the sub-modes' weights.  A refused sub-mode contributes nothing and the
    status says so ("partial") -- its mass is then missing from lnZ.
    """
    ok = [r for r in results if not r["refused"] and np.isfinite(r["lnZ"])]
    if not ok:
        # Nothing accepted: lnZ is nan and the run gets no weight.  The
        # refused estimates are kept as `lnZ_indicative` so a reader can
        # see what the bridge said -- it is never used for a weight.
        fin = [r["lnZ"] for r in results if np.isfinite(r["lnZ"])]
        ind = float("nan")
        if fin:
            top = max(fin)
            ind = top + math.log(sum(math.exp(v - top) for v in fin))
        return {
            "lnZ": float("nan"),
            "lnZ_err": float("inf"),
            "re2": float("inf"),
            "status": "refused",
            "lnZ_indicative": ind,
        }
    lz = np.array([r["lnZ"] for r in ok])
    err = np.array([r["lnZ_err"] for r in ok])
    top = lz.max()
    tot = top + math.log(np.exp(lz - top).sum())
    w = np.exp(lz - tot)
    return {
        "lnZ": float(tot),
        "lnZ_err": float(np.sqrt(np.sum((w * err) ** 2))),
        "re2": float(max(r["re2"] for r in ok)),
        "status": "ok" if len(ok) == len(results) else "partial",
        "lnZ_indicative": float(tot),
    }


def drift(seed, med_log_s, med_log_q, u0_sign, d_log_s, d_log_q):
    """Did the run leave the basin it was seeded in?  (reason or "")

    A seed is a lattice node, up to half a node from its basin's minimum,
    so "left" means more than ONE node in log s or log q, or crossing
    s = 1 (a close <-> wide topology change), or flipping the sign of u_0.
    """
    why = []
    if np.sign(med_log_s) != np.sign(seed["log_s"]) and seed["log_s"] != 0:
        why.append(
            "crossed s=1 (seed log s %+.3f, run %+.3f)"
            % (seed["log_s"], med_log_s)
        )
    elif abs(med_log_s - seed["log_s"]) > d_log_s:
        why.append(
            "log s moved %+.3f (> one node, %.3f)"
            % (med_log_s - seed["log_s"], d_log_s)
        )
    if abs(med_log_q - seed["log_q"]) > d_log_q:
        why.append(
            "log q moved %+.2f (> one node, %.2f)"
            % (med_log_q - seed["log_q"], d_log_q)
        )
    if np.sign(seed["u_0"]) != u0_sign:
        why.append("u_0 sign flipped")
    return "; ".join(why)


def reliability_flags(run):
    """The flags a reader must see next to a run's lnZ (list of str)."""
    f = []
    if run.get("stuck_frac", 0.0) > STUCK_FRAC_MAX:
        f.append("stuck %.2f > %.1f" % (run["stuck_frac"], STUCK_FRAC_MAX))
    if run.get("lp_rhat", 1.0) > LP_RHAT_MAX:
        f.append("lp Rhat %.2f" % run["lp_rhat"])
    st = run.get("bridge_status", "ok")
    if st != "ok":
        f.append("bridge %s" % st)
    re2 = run.get("re2", 0.0)
    if np.isfinite(re2) and re2 > RE2_FLAG:
        f.append("re2 %.2g" % re2)
    if run.get("drift"):
        f.append("drift: %s" % run["drift"])
    if run.get("hit_maxtime"):
        f.append("hit maxtime")
    return f


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------


def mahalanobis(fa, fb):
    """Distance between two basins' (log s, log q, alpha) means, in units
    of their combined spread (covariances summed)."""
    d = np.array(fa["mean"], dtype=float) - np.array(fb["mean"], dtype=float)
    d[2] = wrap180(d[2])
    c = np.array(fa["cov"], dtype=float) + np.array(fb["cov"], dtype=float)
    try:
        return float(np.sqrt(d @ np.linalg.solve(c, d)))
    except np.linalg.LinAlgError:
        return float(np.sqrt(d @ np.linalg.pinv(c) @ d))


def same_basin(fa, fb, n_sigma=DEDUPE_SIGMA):
    return fa["u0_sign"] == fb["u0_sign"] and mahalanobis(fa, fb) < n_sigma


def bridge_re2(r):
    """The re2 to rank a run by.  An accepted run's own; for a refused run,
    the worst finite re2 of its sub-mode bridges (a refused bridge still
    measured how badly its proposal fit), else inf (did not converge)."""
    re2 = r.get("re2", float("inf"))
    if np.isfinite(re2):
        return float(re2)
    vals = [m["re2"] for m in r.get("submodes", []) if np.isfinite(m["re2"])]
    return float(max(vals)) if vals else float("inf")


def dedupe(runs, n_sigma=DEDUPE_SIGMA):
    """Collapse runs that converged to the same basin.

    Greedy in order of re2 (lowest first; a refused bridge counts as inf):
    a run that is `same_basin` as an already kept one is marked
    `duplicate_of` that run and contributes NO lnZ -- adding two runs'
    evidences for one basin would double-count it.  Returns (kept, dupes).
    """

    def key(r):
        re2 = bridge_re2(r)
        return (not np.isfinite(re2), re2 if np.isfinite(re2) else 0.0)

    kept, dupes = [], []
    for r in sorted(runs, key=key):
        hit = next(
            (
                k
                for k in kept
                if same_basin(r["features"], k["features"], n_sigma)
            ),
            None,
        )
        if hit is None:
            kept.append(r)
        else:
            dupes.append(
                dict(
                    r,
                    duplicate_of=hit["mode"],
                    mahalanobis=mahalanobis(r["features"], hit["features"]),
                )
            )
    return kept, dupes


def softmax(lnz, err):
    """Relative probabilities and linearized errors; nan lnZ gets 0."""
    lnz = np.asarray(lnz, dtype=float)
    err = np.asarray(err, dtype=float)
    ok = np.isfinite(lnz)
    w = np.zeros_like(lnz)
    dw = np.zeros_like(lnz)
    if not ok.any():
        return w, dw
    z = lnz[ok]
    e = np.where(np.isfinite(err[ok]), err[ok], 0.0)
    p = np.exp(z - z.max())
    p /= p.sum()
    d = np.empty_like(p)
    for k in range(p.size):
        jac = p[k] * ((np.arange(p.size) == k) - p)
        d[k] = np.sqrt(np.sum((jac * e) ** 2))
    w[ok], dw[ok] = p, d
    return w, dw


def build_report(runs, baseline=None, event=EVENT):
    """The blind mode report (a JSON-able dict) from stage-4 run records.

    `runs` are 2L1S run records (04_evidence.py output); `baseline` an
    optional 1L1S_fs record.  Runs are deduped; weights are a softmax over
    the unique 2L1S modes ("weight_2L1S") and over those plus the baseline
    architecture ("weight_all").  Runs whose bridge was refused have no lnZ
    and get weight 0, listed with their flags.  Nothing here reads truth.
    """
    kept, dupes = dedupe(runs)
    kept.sort(key=lambda r: -r["lnZ"] if np.isfinite(r["lnZ"]) else np.inf)
    lz = [r["lnZ"] for r in kept]
    er = [r["lnZ_err"] for r in kept]
    w2, dw2 = softmax(lz, er)
    if baseline is not None:
        wa, dwa = softmax(lz + [baseline["lnZ"]], er + [baseline["lnZ_err"]])
    modes = []
    for k, r in enumerate(kept):
        row = {key: r[key] for key in r if key not in ("features",)}
        row["weight_2L1S"] = float(w2[k])
        row["weight_2L1S_err"] = float(dw2[k])
        if baseline is not None:
            row["weight_all"] = float(wa[k])
            row["weight_all_err"] = float(dwa[k])
        row["flags"] = reliability_flags(r)
        row["bridge_re2"] = bridge_re2(r)
        row["bridge_states"] = [m["state"] for m in r.get("submodes", [])]
        modes.append(row)
    # When NO bridge was accepted there are no weights.  The refused
    # estimates still say something, so they get their own softmax --
    # labelled INDICATIVE everywhere, never merged into the weights above.
    ind = [r.get("lnZ_indicative", float("nan")) for r in kept]
    iw, _ = softmax(ind, [0.0] * len(ind))
    for row, w in zip(modes, iw):
        row["weight_2L1S_indicative"] = float(w)
    n_ok = sum(1 for r in kept if np.isfinite(r["lnZ"]))
    out = {
        "event": int(event),
        "name": "DC2018_%03d" % event,
        "procedure": "blind: grid -> modes -> one PTDE run per mode -> "
        "dedupe -> normalized per-mode lnZ (bridge) -> softmax",
        "dedupe_rule": "same sign(u_0) and Mahalanobis(log s, log q, alpha)"
        " < %.1f; keep lower re2" % DEDUPE_SIGMA,
        "weights_status": (
            "%d of %d modes accepted by the bridge" % (n_ok, len(kept))
            if n_ok
            else "NO mode accepted by the bridge: there are no evidence "
            "weights; weight_2L1S_indicative is from the REFUSED estimates"
        ),
        "modes": modes,
        "duplicates": [
            {
                "mode": d["mode"],
                "duplicate_of": d["duplicate_of"],
                "mahalanobis": d["mahalanobis"],
                "lnZ": d["lnZ"],
                "re2": d.get("re2"),
            }
            for d in dupes
        ],
    }
    if baseline is not None:
        b = {key: baseline[key] for key in baseline if key != "features"}
        b["weight_all"] = float(wa[-1])
        b["weight_all_err"] = float(dwa[-1])
        b["flags"] = reliability_flags(baseline)
        best = max(
            (r["lnZ"] for r in kept if np.isfinite(r["lnZ"])),
            default=float("nan"),
        )
        b["dlnZ_best_2L1S_minus_baseline"] = float(best - baseline["lnZ"])
        bi = max((v for v in ind if np.isfinite(v)), default=float("nan"))
        b["dlnZ_best_2L1S_minus_baseline_indicative"] = float(
            bi - baseline.get("lnZ_indicative", float("nan"))
        )
        out["baseline"] = b
    return out


PARAMS = ("s", "q", "alpha", "t_0", "u_0", "t_E", "rho")


def _fmt(v, n=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    return "%.*g" % (n, v)


def _summ(row, p):
    s = row.get("summary", {}).get(p)
    if not s:
        return "nan"
    med, lo, hi = s
    if p == "t_0":
        return "%.4f +%.4f/-%.4f" % (med, hi - med, med - lo)
    return "%s +%s/-%s" % (_fmt(med), _fmt(hi - med, 2), _fmt(med - lo, 2))


def report_text(rep, truth=None):
    """Human-readable report.  `truth` (dict of PARAMS) is printed ONLY in
    a separate, labelled validation section at the end."""
    L = []
    L.append("%s blind 2L1S mode report" % rep["name"])
    L.append("=" * len(L[-1]))
    L.append(rep["procedure"])
    L.append("dedupe: " + rep["dedupe_rule"])
    L.append(
        "lnZ is the NORMALIZED log evidence (System.evidence_log_norm "
        "added to the bridge lnZ), so it compares across runs and "
        "architectures."
    )
    L.append("")
    L.append(
        "%-4s %-22s %13s %7s %10s %10s %12s"
        % ("mode", "origin", "lnZ", "err", "w(2L1S)", "w(all)", "lnL_max")
    )
    for r in rep["modes"]:
        L.append(
            "%-4s %-22s %13.2f %7.2f %10.4g %10s %12.2f"
            % (
                r["mode"],
                r.get("origin", "")[:22],
                r["lnZ"],
                r["lnZ_err"],
                r["weight_2L1S"],
                _fmt(r.get("weight_all"), 4),
                r.get("lnL_max", np.nan),
            )
        )
        ind = r.get("lnZ_indicative", float("nan"))
        if not np.isfinite(r["lnZ"]) and np.isfinite(ind):
            L.append("     bridge refused (estimate below, NOT a weight)")
    b = rep.get("baseline")
    if b:
        L.append(
            "%-4s %-22s %13.2f %7.2f %10s %10.4g %12.2f"
            % (
                "base",
                ARCH_BASELINE + " (no companion)",
                b["lnZ"],
                b["lnZ_err"],
                "-",
                b["weight_all"],
                b.get("lnL_max", np.nan),
            )
        )
        L.append(
            "  best 2L1S minus %s: dlnZ = %.1f (accepted); %.1f "
            "(INDICATIVE, refused estimates)"
            % (
                ARCH_BASELINE,
                b["dlnZ_best_2L1S_minus_baseline"],
                b.get("dlnZ_best_2L1S_minus_baseline_indicative", np.nan),
            )
        )
    L.append("")
    L.append("Weights: " + rep.get("weights_status", ""))
    L.append("Bridge per mode (re2 limit 0.25), and the INDICATIVE")
    L.append("softmax of the refused estimates (never a weight):")
    for r in rep["modes"]:
        L.append(
            "  %-4s %-40s re2 %-8s lnZ_ind %13.2f  w_ind %.3g"
            % (
                r["mode"],
                ",".join(r.get("bridge_states", [])),
                _fmt(r.get("bridge_re2"), 3),
                r.get("lnZ_indicative", np.nan),
                r.get("weight_2L1S_indicative", np.nan),
            )
        )
    if b:
        L.append(
            "  %-4s %-40s re2 %-8s lnZ_ind %13.2f"
            % (
                "base",
                ",".join(m["state"] for m in b.get("submodes", [])),
                _fmt(bridge_re2(b), 3),
                b.get("lnZ_indicative", np.nan),
            )
        )
    L.append("")
    L.append("Parameters per mode (median, 68% interval):")
    for r in rep["modes"] + ([dict(b, mode="base")] if b else []):
        L.append("  %s:" % r["mode"])
        for p in PARAMS:
            if p in r.get("summary", {}):
                L.append("    %-6s %s" % (p, _summ(r, p)))
    L.append("")
    L.append("Reliability flags:")
    for r in rep["modes"] + ([dict(b, mode="base")] if b else []):
        L.append("  %s: %s" % (r["mode"], "; ".join(r["flags"]) or "none"))
    if rep["duplicates"]:
        L.append("")
        L.append("Duplicates (same basin; lnZ NOT added):")
        for d in rep["duplicates"]:
            L.append(
                "  %s == %s (Mahalanobis %.2f)"
                % (d["mode"], d["duplicate_of"], d["mahalanobis"])
            )
    if truth:
        L.append("")
        L.append("-" * 72)
        L.append("VALIDATION ONLY -- answer key, NOT used by the procedure")
        L.append("-" * 72)
        L.append(
            "  truth: s=%.4f (mirror 1/s=%.4f) q=%.3g log q=%.3f"
            % (
                truth["s"],
                1.0 / truth["s"],
                truth["q"],
                math.log10(truth["q"]),
            )
        )
        for r in rep["modes"]:
            s = r.get("summary", {})
            if "s" not in s:
                continue
            ls = math.log10(s["s"][0])
            lt = math.log10(truth["s"])
            side = "truth" if abs(ls - lt) < abs(ls + lt) else "mirror"
            L.append(
                "  %s: s=%.4f (%s side), dlog s=%+.3f vs %s; "
                "dlog q=%+.3f; weight %.3g"
                % (
                    r["mode"],
                    s["s"][0],
                    side,
                    ls - (lt if side == "truth" else -lt),
                    side,
                    math.log10(s["q"][0]) - math.log10(truth["q"]),
                    r["weight_2L1S"],
                )
            )
    return "\n".join(L) + "\n"


def dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    Path(path).write_text(json.dumps(obj, indent=1, default=conv) + "\n")
