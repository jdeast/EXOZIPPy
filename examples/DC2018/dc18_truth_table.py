"""Everything comparable, sampled or derived, against DC2018 truth.

JDE's criterion, verbatim: "the truth is in one of our modes, within the
stated uncertainty, and we're reasonably well mixed (so we can trust the
uncertainty)."  Three conditions; each fails in a way the other two cannot
see, so each is reported separately.

WHERE THE NUMBERS COME FROM, and this is the one thing to understand before
reading a table it prints.  `<prefix>_results.csv` is the PRIMARY source,
not the trace.  t_E, theta_E, pi_rel and mu_rel are PURE PHYSICS
EXPRESSIONS with no sampled element of their own, so they get no
pm.Deterministic and never enter the .nc file; what fills them in is
`System.distribute_posterior`, at REPORT time, straight into the CSV and
the LaTeX table (review 2.6.14 -- whose first version claimed the opposite
and was wrong).  Reading the trace instead throws away the event's
best-measured observable: on event 223 it discarded a t_E of
9.536 +0.038/-0.037 against a truth of 9.55012 and scored it "MISSING".

THE TRACE IS THE FALLBACK, for the case that made this necessary: wrap-up
dies.  v5, v6 and v7 were all OOM-killed after the trace was written and
before the CSV was, so their derived quantities existed nowhere and had to
be rebuilt by hand.  When the CSV is absent this reconstructs what it can
and SAYS SO in the provenance column, because a reconstruction is not the
pipeline's own answer.

MODES COME FROM THE PIPELINE, not from a symmetry rule here.  The CSV
carries one row per parameter per mode with its weight, which is
`outputs.modes.identify_modes`' own clustering -- so the "is truth in one of
our modes" question is asked of the modes the fit actually reported.  The
+/-u_0 and close/wide symmetry folding is kept only for the CONVERGENCE
numbers, where mode-hopping otherwise inflates Rhat, and both the raw and
the folded value are printed so a disagreement between them is visible
rather than assumed away: a symmetry that is only approximate shows up
exactly there.
"""

import argparse
import csv
import glob
import io
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import arviz as az  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc18_common as C  # noqa: E402

KAPPA_STAR = 4.6503  # theta_star[mas] = this * R[Rsun] / D[pc]
DAYS_PER_YEAR = 365.25
# master_file.txt is in kpc and solar masses; the model reports parsecs and
# Jupiter masses.  Getting either wrong manufactures a confident
# three-order-of-magnitude pull that reads like a physics failure.
PC_PER_KPC = 1000.0
MJUP_PER_MSUN = 1.0 / 9.5458e-4

# (label, truth key, results.csv name, is this a core light-curve
# observable?).  CORE means the data measure it directly: those are the rows
# that have to hit for a fit to count.  The rest lean on the galactic prior
# or the SED and can miss for reasons that are not the fit's fault -- which
# is the whole substance of 8.6.7, so the two are never pooled.
ROWS = [
    ("t_0", "t0_bjd", "source.t_0", True),
    ("|u_0|", "u0_abs", "source.u_0", True),
    ("t_E", "tE", "mulensevent.t_E", True),
    ("rho", "rhos", "source.rho", True),
    ("s", "s", "lens.Companion.s", True),
    # THE CLOSE/WIDE COUNTERPART, scored explicitly.  s and 1/s are the two
    # branches of a symmetry with near-identical likelihood, so a fit that
    # lands on the counterpart has the geometry right and the branch wrong
    # -- two different claims that the bare `s` row cannot separate.  This
    # row scores against whichever of (s_truth, 1/s_truth) the fit is
    # closer to and says which, so "wrong branch" stops reading as "wrong
    # geometry".
    ("s or 1/s", "s_branch", "lens.Companion.s", True),
    ("q", "q", "lens.Companion.q", True),
    ("theta_E", "thE", "mulensevent.theta_E", False),
    ("mu_rel", "murel", "mulensevent.mu_rel_mag", False),
    ("pi_rel", "pi_rel", "mulensevent.pi_rel", False),
    ("M_lens", "Ml", "star.Lens.mass", False),
    ("M_planet", "Mp", "planet.mass", False),
    ("D_lens", "Dl", "star.Lens.distance", False),
    ("D_source", "Ds", "star.Source.distance", False),
    ("R_source", "Rs", "star.Source.radius", False),
]
# alpha is deliberately absent: dc18_common's header documents the
# origin/handedness mismatch between the challenge's convention and ours,
# and a pull computed across that would be a fabricated number.

# Mixing gates.  "strict" is what a published fit should clear; "explore"
# is for iterating -- JDE, 2026-09-14: "our rhat<1.01, ess>1000 might be too
# aggressive, at least at this exploratory phase.  it would speed
# development dramatically if we could iterate in much less time."  The
# explore tier is calibrated so a 5,000-draw run can pass it, which is the
# point: 50,000 draws is 5-20 h and 5,000 is under two.
TIERS = {
    "strict": {"rhat": 1.01, "ess": 1000.0},
    "default": {"rhat": 1.05, "ess": 400.0},
    "explore": {"rhat": 1.20, "ess": 100.0},
}


def truth_for(event, data_dir):
    row, cls = C.load_master_row(data_dir, event)
    t, _ = C.load_truth(data_dir, event)
    g = lambda k: float(row[k])
    return {
        "t0_bjd": t["t_0"],
        "u0_abs": abs(g("u0")),
        "tE": g("tE"),
        "rhos": g("rhos"),
        "s": g("s"),
        "q": g("q"),
        "s_branch": g("s"),  # replaced per-row by the nearer branch
        "thE": g("thE"),
        "murel": g("murel"),
        "pi_rel": g("piE") * g("thE"),  # not a column; piE * thE by definition
        "Ml": g("Ml"),
        "Mp": g("Mp") * MJUP_PER_MSUN,
        "Dl": g("Dl") * PC_PER_KPC,
        "Ds": g("Ds") * PC_PER_KPC,
        "Rs": g("Rs"),
    }, cls


def read_results_all_modes(path):
    """{param: {mode: (weight, value, up_err, low_err)}} -- every mode.

    dc18_common.read_results_csv drops everything but the combined 'all'
    row and keeps only the names in its own map, which is right for the
    per-event comparison table and wrong here: "is truth in ONE OF our
    modes" is a question about the individual modes.
    """
    with io.open(path, newline="", encoding="utf-8") as f:
        first = f.readline()
        hdr = [c.strip() for c in first.lstrip("#").split(",") if c.strip()]
        known = {
            "parname",
            "mode",
            "weight",
            "weight_err",
            "value",
            "up_err",
            "low_err",
        }
        if not (hdr and set(hdr) <= known and "parname" in hdr):
            hdr = (
                [
                    "parname",
                    "mode",
                    "weight",
                    "weight_err",
                    "value",
                    "up_err",
                    "low_err",
                ]
                if "mode" in first
                else ["parname", "value", "up_err", "low_err"]
            )
        out = {}
        for r in csv.DictReader(f, fieldnames=hdr):
            nm = (r.get("parname") or "").strip()
            if not nm or nm.startswith("#"):
                continue
            mode = (r.get("mode") or "all").strip()

            def _f(k):
                try:
                    return float(r.get(k))
                except (TypeError, ValueError):
                    return None

            out.setdefault(nm, {})[mode] = (
                _f("weight"),
                _f("value"),
                _f("up_err"),
                _f("low_err"),
            )
    return out


def _first(post, *names):
    for n in names:
        if n in post.data_vars:
            return np.asarray(post[n])
    return None


def reconstruct_from_trace(post):
    """Per-draw derived quantities, for when wrap-up never wrote the CSV.

    Returns {results.csv-name: array} so the caller can treat it exactly
    like a CSV row, plus a provenance string per entry.
    """
    d, prov = {}, {}

    def put(k, v, how):
        if v is not None:
            d[k], prov[k] = np.asarray(v, dtype=float), how

    put(
        "source.t_0",
        _first(post, "source.t_0", "source.Source.t_0", "lens.t_0"),
        "trace (sampled)",
    )
    u0 = _first(post, "source.u_0", "source.Source.u_0", "lens.u_0")
    if u0 is not None:
        put("source.u_0", np.abs(u0), "trace (|sampled|)")

    lthE = _first(post, "mulensevent.log_theta_E", "lens.log_theta_E")
    thE = _first(post, "mulensevent.theta_E", "lens.theta_E")
    thE = (
        thE if thE is not None else (10.0**lthE if lthE is not None else None)
    )
    put("mulensevent.theta_E", thE, "REBUILT 10^log_theta_E")

    lpir = _first(post, "mulensevent.log_pi_rel", "lens.log_pi_rel")
    pir = _first(post, "mulensevent.pi_rel", "lens.pi_rel")
    pir = (
        pir if pir is not None else (10.0**lpir if lpir is not None else None)
    )
    put("mulensevent.pi_rel", pir, "REBUILT 10^log_pi_rel")

    mra = _first(post, "mulensevent.mu_ra_rel", "lens.mu_ra_rel")
    mdec = _first(post, "mulensevent.mu_dec_rel", "lens.mu_dec_rel")
    mu = _first(post, "mulensevent.mu_rel_mag", "mulensevent.mu_rel")
    if mu is None and mra is not None and mdec is not None:
        mu = np.hypot(mra, mdec)
        put("mulensevent.mu_rel_mag", mu, "REBUILT hypot(mu_ra, mu_dec)")
    else:
        put("mulensevent.mu_rel_mag", mu, "trace")

    tE = _first(post, "mulensevent.t_E", "lens.t_E")
    if tE is None and thE is not None and mu is not None:
        put(
            "mulensevent.t_E",
            DAYS_PER_YEAR * thE / mu,
            "REBUILT 365.25*theta_E/mu_rel",
        )
    else:
        put("mulensevent.t_E", tE, "trace")

    ls = _first(post, "lens.log_s", "lens.Companion.log_s")
    if ls is not None:
        ls = ls[..., -1] if ls.ndim == 3 else ls
        d["_log_s"], prov["_log_s"] = ls, "trace"
        put("lens.Companion.s", 10.0**ls, "REBUILT 10^log_s")
    lq = _first(post, "planet.log_q", "lens.log_q", "lens.Companion.log_q")
    if lq is not None:
        lq = lq[..., -1] if lq.ndim == 3 else lq
        d["_log_q"], prov["_log_q"] = lq, "trace"
        put("lens.Companion.q", 10.0**lq, "REBUILT 10^log_q")

    rad = _first(post, "star.radius")
    dist = _first(post, "star.distance")
    rho = _first(post, "source.rho", "source.Source.rho")
    lrho = _first(post, "source.log_rho", "lens.log_rho")
    if rho is not None:
        put("source.rho", rho, "trace")
    elif lrho is not None:
        put("source.rho", 10.0**lrho, "REBUILT 10^log_rho")
    elif rad is not None and dist is not None and thE is not None:
        # star.0 = Lens, star.1 = Source, fixed by the `star:` block order.
        put(
            "source.rho",
            KAPPA_STAR * rad[..., 1] / dist[..., 1] / thE,
            "REBUILT theta_star/theta_E",
        )
    if dist is not None:
        put("star.Lens.distance", dist[..., 0], "trace")
        put("star.Source.distance", dist[..., 1], "trace")
    if rad is not None:
        put("star.Source.radius", rad[..., 1], "trace")

    lm = _first(post, "star.logmass")
    ml = (
        10.0 ** lm[..., 0]
        if lm is not None
        else _first(post, "mulensevent.mlens_total")
    )
    put("star.Lens.mass", ml, "REBUILT 10^star.logmass[Lens]")
    mp = _first(post, "planet.mass", "planet.Companion.mass")
    if mp is not None:
        put("planet.mass", mp, "trace (Mjup)")
    elif ml is not None and "lens.Companion.q" in d:
        put(
            "planet.mass",
            d["lens.Companion.q"] * ml * MJUP_PER_MSUN,
            "REBUILT q*M_lens -> Mjup",
        )
    return d, prov


def convergence(post, tier):
    """Rhat/ESS on the RAW and the FOLDED coordinates, side by side.

    Folding matters because +/-u_0 and s <-> 1/s are symmetries: a chain
    that visits both branches is mixing WELL and raw Rhat scores it as
    catastrophic disagreement (223 changed mode 549 times and was punished
    for it).  But JDE is right that they are not always PERFECT
    degeneracies, and folding throws that information away -- so both are
    printed.  Where raw is bad and folded is fine, the spread is the
    symmetry; where BOTH are bad, it is not, and the fit has a real mixing
    problem inside a single branch.
    """
    import xarray as xr

    names = {
        "t_0": ("source.t_0", "source.Source.t_0"),
        "u_0": ("source.u_0", "source.Source.u_0", "lens.u_0"),
        "log_s": ("lens.log_s", "lens.Companion.log_s"),
        "log_q": ("planet.log_q", "lens.log_q"),
    }
    out = {}
    for key, spellings in names.items():
        v = _first(post, *spellings)
        if v is None:
            continue
        if v.ndim == 3:
            v = v[..., -1]
        if v.ndim != 2 or not np.all(np.isfinite(v)):
            continue
        row = {}
        for kind, arr in (
            ("raw", v),
            ("folded", np.abs(v) if key in ("u_0", "log_s") else v),
        ):
            try:
                da = xr.DataArray(arr, dims=("chain", "draw"))
                row[kind] = (
                    float(np.max(az.rhat(da).values)),
                    float(np.min(az.ess(da).values)),
                )
            except Exception:  # noqa: BLE001
                pass
        if row:
            out[key] = row
    gate = TIERS[tier]
    bad = [
        k
        for k, r in out.items()
        if "folded" in r
        and (r["folded"][0] > gate["rhat"] or r["folded"][1] < gate["ess"])
    ]
    return out, bad, gate


def report(prefix, event, data_dir, tier="default"):
    """`prefix` is the run prefix, e.g. sweep/128/DC2018_128."""
    truth, cls = truth_for(event, data_dir)
    csv_path = prefix + "_results.csv"
    trace_path = prefix + "_trace.nc"

    have_csv = os.path.exists(csv_path)
    table = read_results_all_modes(csv_path) if have_csv else {}
    post = (
        az.from_netcdf(trace_path).posterior
        if os.path.exists(trace_path)
        else None
    )
    rebuilt, rprov = (
        reconstruct_from_trace(post)
        if (post is not None and not have_csv)
        else ({}, {})
    )

    print("\n" + "=" * 104)
    print("EVENT %s   class=%s   %s" % (event, cls, os.path.relpath(prefix)))
    print(
        "  source: %s"
        % (
            "results.csv (the pipeline's own per-mode summaries)"
            if have_csv
            else "TRACE RECONSTRUCTION -- results.csv absent, so wrap-up never "
            "finished; these are not the pipeline's numbers"
        )
    )
    if post is not None:
        print(
            "  chains=%d draws=%d" % (post.sizes["chain"], post.sizes["draw"])
        )

    if post is not None:
        conv, bad, gate = convergence(post, tier)
        print(
            "  mixing, tier '%s' (Rhat < %.2f, ESS > %.0f):"
            % (tier, gate["rhat"], gate["ess"])
        )
        print(
            "    %-8s %20s %22s" % ("", "RAW", "FOLDED (symmetry-invariant)")
        )
        for k, r in sorted(conv.items()):
            raw = "Rhat %5.2f ESS %8.0f" % r["raw"] if "raw" in r else "--"
            fld = (
                "Rhat %5.2f ESS %8.0f" % r["folded"] if "folded" in r else "--"
            )
            flag = "  <<<" if k in bad else ""
            note = ""
            if (
                "raw" in r
                and "folded" in r
                and r["raw"][0] > gate["rhat"]
                and r["folded"][0] <= gate["rhat"]
            ):
                note = "   (the spread IS the symmetry)"
            print("    %-8s %20s %22s%s%s" % (k, raw, fld, flag, note))
        mixed = not bad
    else:
        mixed, bad = False, ["no trace"]
        print("  mixing: NO TRACE -- cannot check")

    # REJECTED SEEDS ARE NOT MODES.  The writer emits a row per rejected
    # seed ("rejected-seedN", weight 0) alongside the real modes, and those
    # carry a sliver's worth of draws -- one reported t_0 uncertainty of
    # 0.00014 d, i.e. 12 seconds, which turns any offset at all into a
    # five-figure pull.  Scoring them answers no question anyone asked, so
    # they are dropped unless they are all there is.
    def _is_real(m, weights):
        if m == "all" or "reject" in m.lower():
            return False
        w = weights.get(m)
        return w is None or w > 0.0

    weights = {}
    for v in table.values():
        for m, cell in v.items():
            if cell[0] is not None:
                weights.setdefault(m, cell[0])
    modes = sorted(
        m
        for m in {m for v in table.values() for m in v}
        if _is_real(m, weights)
    )
    dropped = sorted(
        m
        for m in {m for v in table.values() for m in v}
        if m != "all" and not _is_real(m, weights)
    )
    if dropped:
        print(
            "  not scored (rejected seeds / zero weight): %s"
            % ", ".join(dropped)
        )
    modes = modes or ["all"]
    if not have_csv:
        modes = ["all"]
    best = {"mode": None, "core_hit": -1, "core_n": 0, "hit": 0, "n": 0}
    for m in modes:
        w = None
        for v in table.values():
            if m in v and v[m][0] is not None:
                w = v[m][0]
                break
        print(
            "\n  MODE %s%s" % (m, "" if w is None else "  (weight %.3f)" % w)
        )
        print(
            "    %-9s %14s %14s %20s %7s  %s"
            % (
                "quantity",
                "truth",
                "value",
                "stated 1-sigma",
                "pull",
                "provenance",
            )
        )
        core_hit = core_n = hit = n = 0
        for label, tkey, name, is_core in ROWS:
            tv = truth[tkey]
            branch = ""
            if have_csv:
                cell = table.get(name, {}).get(m) or table.get(name, {}).get(
                    "all"
                )
                if cell is None or cell[1] is None:
                    print(
                        "    %-9s %14.6g %14s %20s %7s  %s"
                        % (label, tv, "--", "not reported", "--", "MISSING")
                    )
                    continue
                _, val, up, low = cell
                up = up if up is not None else 0.0
                low = low if low is not None else up
                prov = "results.csv"
            else:
                arr = rebuilt.get(name)
                if arr is None:
                    print(
                        "    %-9s %14.6g %14s %20s %7s  %s"
                        % (label, tv, "--", "not in trace", "--", "MISSING")
                    )
                    continue
                a = np.asarray(arr).ravel()
                a = a[np.isfinite(a)]
                if a.size == 0:
                    continue
                val = float(np.median(a))
                lo, hi = np.percentile(a, [15.865, 84.135])
                up, low = float(hi - val), float(val - lo)
                prov = rprov.get(name, "trace")
            if label == "|u_0|":
                val = abs(val)
            if label == "s or 1/s":
                alt = 1.0 / tv if tv else tv
                if abs(np.log(max(val, 1e-30) / alt)) < abs(
                    np.log(max(val, 1e-30) / tv)
                ):
                    tv, branch = alt, " (wide<->close)"
            sig = 0.5 * (up + low) or max(up, low)
            pull = (val - tv) / sig if sig > 0 else float("nan")
            in1 = (val - low) <= tv <= (val + up)
            in2 = (val - 2 * low) <= tv <= (val + 2 * up)
            mark = "1sig" if in1 else ("2sig" if in2 else "    ")
            n += 1
            hit += int(in1)
            if is_core:
                core_n += 1
                core_hit += int(in1)
            print(
                "    %-9s %14.6g %14.6g  +%-9.4g -%-8.4g %7.2f %s %s%s"
                % (label, tv, val, up, low, pull, mark, prov, branch)
            )
        print(
            "    -> core %d/%d, all %d/%d inside the stated 1 sigma"
            % (core_hit, core_n, hit, n)
        )
        if core_hit > best["core_hit"]:
            best = {
                "mode": m,
                "core_hit": core_hit,
                "core_n": core_n,
                "hit": hit,
                "n": n,
            }

    core_ok = best["core_n"] > 0 and best["core_hit"] == best["core_n"]
    winner = bool(core_ok and mixed and have_csv)
    print(
        "\n  VERDICT: best mode %s -- core %d/%d, all %d/%d at 1 sigma; "
        "mixing %s"
        % (
            best["mode"],
            best["core_hit"],
            best["core_n"],
            best["hit"],
            best["n"],
            "OK" if mixed else "FAILED on " + ",".join(bad),
        )
    )
    print("  CLEAR WINNER: %s" % winner)
    return {
        "event": event,
        "class": cls,
        "prefix": prefix,
        "tier": tier,
        "have_csv": have_csv,
        "mixed": mixed,
        "unmixed_on": bad,
        "best_mode": best["mode"],
        "core_hit": best["core_hit"],
        "core_n": best["core_n"],
        "hit": best["hit"],
        "n": best["n"],
        "clear_winner": winner,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "prefixes",
        nargs="+",
        help="run prefixes, e.g. sweep/128/DC2018_128 (globs fine)",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tier", default="default", choices=sorted(TIERS))
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    d = C.data_dir_or_raise(a.data_dir)
    out = []
    for pat in a.prefixes:
        hits = sorted(glob.glob(pat)) or [pat]
        for h in hits:
            p = (
                h[: -len("_results.csv")]
                if h.endswith("_results.csv")
                else (h[: -len("_trace.nc")] if h.endswith("_trace.nc") else h)
            )
            ev = None
            for part in os.path.normpath(p).split(os.sep):
                if part.isdigit():
                    ev = int(part)
            if ev is None:
                digits = "".join(c for c in os.path.basename(p) if c.isdigit())
                ev = int(digits[:3]) if digits else None
            if ev is None:
                print("skip %s: cannot infer the event number" % p)
                continue
            try:
                out.append(report(p, ev, d, tier=a.tier))
            except Exception as e:  # noqa: BLE001
                print("\n%s FAILED: %s: %s" % (p, type(e).__name__, e))
    if out:
        print("\n" + "=" * 104)
        print(
            "%-6s %-9s %-7s %-8s %-8s %s"
            % ("event", "mode", "core", "all", "mixed", "CLEAR WINNER")
        )
        for r in out:
            print(
                "%-6s %-9s %-7s %-8s %-8s %s"
                % (
                    r["event"],
                    r["best_mode"],
                    "%d/%d" % (r["core_hit"], r["core_n"]),
                    "%d/%d" % (r["hit"], r["n"]),
                    "yes" if r["mixed"] else "no",
                    "YES" if r["clear_winner"] else "no",
                )
            )
    if a.json_out:
        json.dump(out, io.open(a.json_out, "w", encoding="utf-8"), indent=1)
        print("\nwrote %s" % a.json_out)


if __name__ == "__main__":
    main()
