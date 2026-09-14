"""Everything comparable, sampled or derived, against DC2018 truth.

WHAT THIS ANSWERS, in JDE's words: "the truth is in one of our modes,
within the stated uncertainty, and we're reasonably well mixed (so we can
trust the uncertainty)."  Three conditions, and all three have to be
checked separately, because each fails in a way the other two cannot see.

WHY IT RECONSTRUCTS RATHER THAN READS.  Review 2.6.14: t_E is in NO trace
in ANY coordinate arm, and rho vanishes the moment `star_constrains_rho:
True` makes it derived.  The published name set changes with the
parameterization, so a truth table built by looking up trace variables
silently drops the event's best-measured observable.  Everything here is
therefore rebuilt per draw from whatever IS stored, and DERIVED is printed
next to each row so nobody mistakes a reconstruction for a sampled column.

WHY IT FOLDS THE SYMMETRIES BEFORE MEASURING CONVERGENCE.  The +/-u_0 and
close/wide (s <-> 1/s) degeneracies are SYMMETRIES: near-identical
likelihood, genuinely different parameters.  A chain that visits both is
mixing WELL, and yet Rhat on the raw u_0 reads that as catastrophic
disagreement -- 223's production run changed mode 549 times and Rhat
punished it for it.  So modes are labelled by (sign u_0, sign log_s) and
convergence is measured on the FOLDED coordinates (|u_0|, |log_s|), where
mode-hopping no longer contaminates the number.  Rhat on the folded
coordinates answers the question actually being asked: within a mode, do
the chains agree?

WHAT IT CANNOT DO, stated because the table looks more authoritative than
it is: it cannot detect a mode NO chain visited.  223's posterior spans
u_0 = +0.0082 to +0.0122, every chain on the positive side; the counterpart
was never sampled, so there is no spread for any diagnostic to find.  A
"truth not in any mode" verdict here means "not in any mode WE FOUND".
"""

import argparse
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

# theta_star[mas] = KAPPA_STAR * R[Rsun] / D[pc]:  (Rsun/pc) in radians
# times mas-per-radian.
KAPPA_STAR = 4.6503
DAYS_PER_YEAR = 365.25
# master_file.txt's distances are in kpc and its planet mass in solar masses;
# the model reports parsecs and Jupiter masses.  Getting either of these
# wrong produces a confident three-order-of-magnitude "pull" that looks like
# a physics failure, which is exactly what the first run of this script did.
PC_PER_KPC = 1000.0
MJUP_PER_MSUN = 1.0 / 9.5458e-4

# (label, truth column in master_file.txt, how to get it from the trace)
# The third element is a key into derive()'s output.  Truth columns whose
# CONVENTION does not map onto ours are deliberately absent: alpha is the
# documented example (dc18_common's header explains the origin/handedness
# mismatch), and quoting a pull for it would be a fabricated number.
ROWS = [
    ("t_0",      "t0_bjd", "t_0"),
    ("|u_0|",    "u0_abs", "u_0_abs"),
    ("t_E",      "tE",     "t_E"),
    ("rho",      "rhos",   "rho"),
    ("s",        "s",      "s"),
    # |log s| is the close/wide-INVARIANT comparison.  s and 1/s are the same
    # physical solution seen from the two branches of a symmetry, so a fit
    # that lands on the counterpart is not wrong about the geometry -- it is
    # wrong about which branch, and only this row separates the two claims.
    ("|log s|",  "abs_log_s", "log_s_abs"),
    ("q",        "q",      "q"),
    ("theta_E",  "thE",    "theta_E"),
    ("mu_rel",   "murel",  "mu_rel"),
    ("pi_E",     "piE",    "pi_E"),
    ("pi_rel",   "pi_rel", "pi_rel"),
    ("M_lens",   "Ml",     "M_lens"),
    ("M_planet", "Mp",     "M_planet"),
    ("D_lens",   "Dl",     "D_lens"),
    ("D_source", "Ds",     "D_source"),
    ("R_source", "Rs",     "R_source"),
]


def truth_for(event, data_dir):
    row, cls = C.load_master_row(data_dir, event)
    t, _ = C.load_truth(data_dir, event)
    g = lambda k: float(row[k])
    out = {
        "t0_bjd": t["t_0"],
        "u0_abs": abs(g("u0")),
        "tE": g("tE"), "rhos": g("rhos"), "s": g("s"), "q": g("q"),
        "thE": g("thE"), "murel": g("murel"), "piE": g("piE"),
        # pi_rel is not a master_file column; it is piE * thE by definition.
        "pi_rel": g("piE") * g("thE"),
        "Ml": g("Ml"),
        "Mp": g("Mp") * MJUP_PER_MSUN,
        "Dl": g("Dl") * PC_PER_KPC,
        "Ds": g("Ds") * PC_PER_KPC,
        "Rs": g("Rs"),
        "abs_log_s": abs(float(np.log10(g("s")))),
    }
    return out, cls


def _first(post, *names):
    for n in names:
        if n in post.data_vars:
            return np.asarray(post[n])
    return None


def derive(post):
    """Per-draw values for every comparable quantity, however it is stored.

    Each block tries the sampled spelling first and falls back to
    reconstruction, so the SAME table comes out of a severed run, a relinked
    run and an observable-swap run -- which is the whole point, because
    those three do not publish the same variables (2.6.14).
    """
    d, prov = {}, {}

    def put(key, val, how):
        if val is not None:
            d[key], prov[key] = np.asarray(val, dtype=float), how

    put("t_0", _first(post, "source.t_0", "source.Source.t_0", "lens.t_0"),
        "sampled")
    u0 = _first(post, "source.u_0", "source.Source.u_0", "lens.u_0")
    if u0 is not None:
        d["u_0"], prov["u_0"] = u0, "sampled"
        d["u_0_abs"], prov["u_0_abs"] = np.abs(u0), "folded"

    lthE = _first(post, "mulensevent.log_theta_E", "lens.log_theta_E")
    thE = _first(post, "mulensevent.theta_E", "lens.theta_E")
    if thE is None and lthE is not None:
        thE = 10.0 ** lthE
        put("theta_E", thE, "DERIVED 10^log_theta_E")
    else:
        put("theta_E", thE, "sampled")

    lpir = _first(post, "mulensevent.log_pi_rel", "lens.log_pi_rel")
    pir = _first(post, "mulensevent.pi_rel", "lens.pi_rel")
    if pir is None and lpir is not None:
        pir = 10.0 ** lpir
        put("pi_rel", pir, "DERIVED 10^log_pi_rel")
    else:
        put("pi_rel", pir, "sampled")

    mra = _first(post, "mulensevent.mu_ra_rel", "lens.mu_ra_rel")
    mdec = _first(post, "mulensevent.mu_dec_rel", "lens.mu_dec_rel")
    mu = _first(post, "mulensevent.mu_rel", "lens.mu_rel")
    if mu is None and mra is not None and mdec is not None:
        mu = np.hypot(mra, mdec)
        put("mu_rel", mu, "DERIVED hypot(mu_ra_rel, mu_dec_rel)")
    else:
        put("mu_rel", mu, "sampled")

    tE = _first(post, "mulensevent.t_E", "lens.t_E")
    if tE is None and thE is not None and mu is not None:
        put("t_E", DAYS_PER_YEAR * thE / mu, "DERIVED 365.25*theta_E/mu_rel")
    else:
        put("t_E", tE, "sampled")

    if thE is not None and pir is not None:
        put("pi_E", pir / thE, "DERIVED pi_rel/theta_E")

    ls = _first(post, "lens.log_s", "lens.Companion.log_s")
    if ls is not None:
        ls = ls[..., -1] if ls.ndim == 3 else ls   # per-companion vector
        d["log_s"], prov["log_s"] = ls, "sampled"
        d["s"], prov["s"] = 10.0 ** ls, "DERIVED 10^log_s"
        # |log_s| is the close/wide-invariant coordinate: s and 1/s fold onto
        # the same value, so convergence measured here is not punished for
        # a chain that legitimately visits both.
        d["log_s_abs"], prov["log_s_abs"] = np.abs(ls), "folded"

    lq = _first(post, "planet.log_q", "lens.log_q", "lens.Companion.log_q")
    if lq is not None:
        lq = lq[..., -1] if lq.ndim == 3 else lq
        d["log_q"], prov["log_q"] = lq, "sampled"
        d["q"], prov["q"] = 10.0 ** lq, "DERIVED 10^log_q"

    rho = _first(post, "source.rho", "source.Source.rho")
    lrho = _first(post, "source.log_rho", "lens.log_rho")
    rad = _first(post, "star.radius")
    dist = _first(post, "star.distance")
    if rho is not None:
        put("rho", rho, "sampled")
    elif lrho is not None:
        put("rho", 10.0 ** lrho, "DERIVED 10^log_rho")
    elif rad is not None and dist is not None and thE is not None:
        # theta_star / theta_E.  Index 1 is the Source (star.0 = Lens,
        # star.1 = Source, fixed by the `star:` block's order).
        thstar = KAPPA_STAR * rad[..., 1] / dist[..., 1]
        put("rho", thstar / thE, "DERIVED theta_star/theta_E")

    if dist is not None:
        put("D_lens", dist[..., 0], "sampled/derived (star.distance[Lens])")
        put("D_source", dist[..., 1], "sampled/derived (star.distance[Source])")
    if rad is not None:
        put("R_source", rad[..., 1], "sampled/derived (star.radius[Source])")

    lm = _first(post, "star.logmass")
    ml = None
    if lm is not None:
        ml = 10.0 ** lm[..., 0]
        put("M_lens", ml, "DERIVED 10^star.logmass[Lens]")
    else:
        mt = _first(post, "mulensevent.mlens_total")
        if mt is not None:
            ml = mt
            put("M_lens", mt, "sampled (mlens_total)")

    mp = _first(post, "planet.mass", "planet.Companion.mass")
    if mp is not None:
        put("M_planet", mp, "sampled/derived (planet.mass)")
    elif ml is not None and "q" in d:
        put("M_planet", d["q"] * ml * MJUP_PER_MSUN,
            "DERIVED q * M_lens (-> Mjup)")
    return d, prov


def label_modes(d):
    """(sign u_0, sign log_s) -> the 2L1S discrete degeneracies.

    These are the two SYMMETRIES (conventions.md): both branches fit the
    data almost equally well and correspond to genuinely different geometry.
    Anything finer -- a second q basin, say -- shows up as a wide log_q
    inside one label rather than as its own mode, and the per-mode log_q
    interval in the table is what exposes it.
    """
    if "u_0" not in d or "log_s" not in d:
        return None, {}
    su = np.where(d["u_0"] >= 0, "+", "-")
    ss = np.where(d["log_s"] >= 0, "wide", "close")
    lab = np.char.add(np.char.add(su, "u0/"), ss)
    counts = {k: int((lab == k).sum()) for k in np.unique(lab)}
    return lab, counts


def folded_convergence(d):
    """Rhat/ESS on the mode-INVARIANT coordinates only.

    az.rhat wants named (chain, draw) dims, so the raw arrays are wrapped
    rather than passed through -- handing it a bare ndarray silently returns
    nothing usable, which is how the first version of this printed an empty
    table and called it convergence.
    """
    import xarray as xr
    out = {}
    for k in ("t_0", "u_0_abs", "log_s_abs", "log_q", "t_E", "theta_E", "rho"):
        v = d.get(k)
        if v is None or np.ndim(v) != 2 or not np.all(np.isfinite(v)):
            continue
        da = xr.DataArray(v, dims=("chain", "draw"))
        try:
            out[k] = (float(np.max(az.rhat(da).values)),
                      float(np.min(az.ess(da).values)))
        except Exception:  # noqa: BLE001
            continue
    return out


def summarize(vals, truth):
    v = np.asarray(vals).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    med = float(np.median(v))
    lo, hi = np.percentile(v, [15.865, 84.135])
    c025, c975 = np.percentile(v, [2.5, 97.5])
    inside68 = bool(lo <= truth <= hi)
    inside95 = bool(c025 <= truth <= c975)
    sd = float(0.5 * (hi - lo)) or float(np.std(v))
    pull = (med - truth) / sd if sd > 0 else float("nan")
    return dict(median=med, lo=float(lo), hi=float(hi), c025=float(c025),
                c975=float(c975), inside68=inside68, inside95=inside95,
                pull=float(pull))


def report(trace, event, data_dir, ess_min=200.0, rhat_max=1.05):
    post = az.from_netcdf(trace).posterior
    truth, cls = truth_for(event, data_dir)
    d, prov = derive(post)
    lab, counts = label_modes(d)
    conv = folded_convergence(d)

    print("\n" + "=" * 100)
    print("EVENT %s   class=%s   %s" % (event, cls, os.path.relpath(trace)))
    print("  chains=%d draws=%d" % (post.sizes["chain"], post.sizes["draw"]))
    if counts:
        tot = sum(counts.values())
        print("  modes found: " + ", ".join(
            "%s %.1f%%" % (k, 100.0 * n / tot)
            for k, n in sorted(counts.items(), key=lambda kv: -kv[1])))
    else:
        print("  modes found: (u_0 or log_s missing -- cannot label)")

    print("  FOLDED convergence (mode-invariant coordinates):")
    bad = []
    for k, (r, e) in sorted(conv.items()):
        flag = ""
        if r > rhat_max or e < ess_min:
            flag = "  <<< not mixed"
            bad.append(k)
        print("    %-12s Rhat %5.2f  ESS %9.0f%s" % (k, r, e, flag))
    mixed = not bad

    # THE VERDICT IS PER-MODE AND SPLIT BY ROLE.  "truth is in one of our
    # modes" cannot mean "some row happened to land", which is what an
    # any()-over-all-quantities test degenerates into once the table has
    # fifteen rows: with that many, one hit is close to guaranteed.  So the
    # CORE set below -- the light-curve observables, the things the data
    # actually measure -- has to hit, and the physical quantities are
    # counted separately because they lean on the galactic prior and the SED
    # and can fail for reasons that are not the fit's fault.
    CORE = {"t_0", "|u_0|", "t_E", "|log s|", "q", "rho"}
    modes = sorted(counts, key=lambda k: -counts[k]) if counts else [None]
    best = {"mode": None, "core_hit": -1, "core_n": 0, "all_hit": 0, "all_n": 0}
    for m in modes:
        sel = slice(None) if m is None else (lab == m)
        head = "ALL DRAWS" if m is None else "MODE %s (%.1f%%)" % (
            m, 100.0 * counts[m] / sum(counts.values()))
        print("\n  %s" % head)
        print("    %-10s %14s %14s %26s %8s  %s"
              % ("quantity", "truth", "median", "68% interval", "pull", "provenance"))
        core_hit = core_n = all_hit = all_n = 0
        for name, tkey, dkey in ROWS:
            if dkey not in d or tkey not in truth:
                print("    %-10s %14s %14s %26s %8s  %s"
                      % (name, "%.6g" % truth.get(tkey, float("nan")),
                         "--", "not in trace", "--", "MISSING"))
                continue
            arr = d[dkey] if m is None else d[dkey][sel]
            s = summarize(arr, truth[tkey])
            if s is None:
                continue
            mark = "OK " if s["inside68"] else ("95%" if s["inside95"] else "   ")
            all_n += 1
            all_hit += int(s["inside95"])
            if name in CORE:
                core_n += 1
                core_hit += int(s["inside95"])
            print("    %-10s %14.6g %14.6g   [%10.5g,%10.5g] %8.2f %s %s"
                  % (name, truth[tkey], s["median"], s["lo"], s["hi"],
                     s["pull"], mark, prov.get(dkey, "")))

        print("    -> core %d/%d, all %d/%d inside 95%%"
              % (core_hit, core_n, all_hit, all_n))
        if core_hit > best["core_hit"]:
            best = {"mode": m, "core_hit": core_hit, "core_n": core_n,
                    "all_hit": all_hit, "all_n": all_n}

    core_ok = best["core_n"] > 0 and best["core_hit"] == best["core_n"]
    winner = bool(core_ok and mixed)
    print("\n  VERDICT: best mode %s -- core %d/%d, all %d/%d at 95%%; "
          "folded mixing %s"
          % (best["mode"], best["core_hit"], best["core_n"],
             best["all_hit"], best["all_n"],
             "OK" if mixed else "FAILED on " + ",".join(bad)))
    print("  CLEAR WINNER: %s%s" % (
        winner,
        "" if winner else
        "   (needs every core observable inside 95% in one mode AND "
        "folded mixing OK)"))
    return {"event": event, "class": cls, "modes": counts,
            "mixed": mixed, "unmixed_on": bad, "best_mode": best["mode"],
            "core_hit": best["core_hit"], "core_n": best["core_n"],
            "all_hit": best["all_hit"], "all_n": best["all_n"],
            "clear_winner": winner}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    d = C.data_dir_or_raise(a.data_dir)
    out = []
    for pat in a.traces:
        for t in sorted(glob.glob(pat)):
            ev = None
            for part in os.path.normpath(t).split(os.sep):
                if part.isdigit():
                    ev = int(part)
            if ev is None:
                base = os.path.basename(t)
                digits = "".join(c for c in base if c.isdigit())
                ev = int(digits[:3]) if digits else None
            if ev is None:
                print("skip %s: cannot infer the event number" % t)
                continue
            try:
                out.append(report(t, ev, d))
            except Exception as e:  # noqa: BLE001
                print("\n%s FAILED: %s: %s" % (t, type(e).__name__, e))
    if a.json_out:
        json.dump(out, io.open(a.json_out, "w", encoding="utf-8"), indent=1)
        print("\nwrote %s" % a.json_out)


if __name__ == "__main__":
    main()
