"""Does the severed+SED model PREFER the wrong solution? (8.6.7)

8.6.7 deliberately does not claim that severed-v3's wrong solution beats the
right one: its lp (max 87,786) carries SED/torres/mann terms the capped
mulens-only baseline (87,759) never evaluates, so the two are incomparable.
The honest test is the SAME model at two points.

WHY THE FIRST VERSION OF THIS SCRIPT FAILED (job 15408109, recorded under
8.6.7).  It injected SIX truth values into a 38-parameter model and left the
other 32 at the as-shipped start.  That is not "the correct solution", it is
an inconsistent mixture of two, and it returned logp = -3.86e6 -- a number
about nothing.  The injection did not even take cleanly (log_rho wanted
-2.217, the relaxation engine reconciled it to -2.051), and the check that
reported the failure was itself reading prior draws through 2.4.15's trap 1.

WHAT THIS VERSION DOES INSTEAD.  Both points are COMPLETE 38-parameter
vectors, built the same way through the same engine so any reconciliation
applies equally to both:

  POINT A  the run's own argmax-lp draw, every physical value read from the
           trace's stored posterior Deterministics (the only reliable route
           -- 2.4.15).
  POINT B  point A with the LENS/SOURCE PHYSICS overwritten by truth
           (dc128_truth_forward.json) and everything else left at A, so the
           two differ ONLY in the disputed block and the light-curve
           nuisances are held fixed.

This is a START-ONLY comparison and says so.  A polish of both points was
considered and DROPPED rather than wired badly: polish_seed_starts takes a
RAW-space logp plus whitening scales, not a model, and mis-wiring exactly
that kind of interface is what produced this file's previous wrong answer.
(2.4.14 also measured the gradient-free polish plateauing at ~0.9
nats/sweep, so it would not have reached either basin's optimum anyway.)

What replaces it is better evidence: the PER-TERM logp breakdown.  If the
model prefers the wrong solution, this says WHICH terms pay for it -- the
microlensing likelihood, the SED photometry, torres/mann, or the priors --
and that is the actual question behind 8.6.7, not the scalar total.
Because both points are COMPLETE and built through the same engine, a large
total gap is meaningful even without a polish; a small one is not.
"""

import io
import json
import logging
import os

import numpy as np
import yaml

logging.disable(logging.WARNING)
os.chdir("/home/jeastman/python/EXOZIPPy/examples/DC2018/configs")

import xarray as xr  # noqa: E402

from exozippy.system import System  # noqa: E402

CFG = "DC2018_128_severed_v3.yaml"
TRACE = "fitresults_severed_v3/DC2018_128_trace.nc"

# Instance order per component, as the components build their vectors.
INSTANCES = {"star": ["Lens", "Source"], "lens": ["Lens"], "planet":
             ["Companion"], "mulensinstrument": ["Roman_W149", "Roman_Z087"],
             "sed": ["sed"], "mann": ["Lens"], "band": ["W149", "Z087"]}

# dc128_truth_forward.json.  The DISPUTED block only -- what the star swap
# got wrong -- so point B differs from A in exactly this and nothing else.
TRUTH = {
    "lens.Lens.log_rho": float(np.log10(0.0060668)),
    "lens.Lens.log_theta_E": float(np.log10(0.0904643)),
    "lens.Lens.log_pi_rel": float(np.log10(0.0022108)),
    "star.Lens.logmass": float(np.log10(0.454)),
    "star.Lens.distance": 7999.0,
    "star.Source.radius": 0.961,
    "star.Source.distance": 8140.0,
}


def point_from_trace():
    """The run's argmax-lp draw as {param_key: physical value}."""
    ds = xr.open_dataset(TRACE, group="posterior")
    lp = np.asarray(ds["lp"]) if "lp" in ds.data_vars else None
    if lp is None:
        raise SystemExit("trace has no lp; cannot locate the argmax draw")
    c, d = np.unravel_index(int(np.nanargmax(lp)), lp.shape)
    print("argmax lp draw: chain %d draw %d  lp = %.1f"
          % (c, d, float(lp[c, d])), flush=True)
    out = {}
    for v in ds.data_vars:
        if v.endswith("_raw") or v in ("lp",):
            continue
        parts = v.split(".", 1)
        if len(parts) != 2:
            continue
        comp, pname = parts
        names = INSTANCES.get(comp)
        if names is None:
            continue
        da = ds[v].isel(chain=c, draw=d)
        arr = np.atleast_1d(np.asarray(da, dtype=float))
        for i, val in enumerate(arr):
            if i >= len(names) or not np.isfinite(val):
                continue
            out["%s.%s.%s" % (comp, names[i], pname)] = float(val)
    print("read %d physical values from the trace" % len(out), flush=True)
    return out


def build(overrides, label):
    cfg = yaml.safe_load(io.open(CFG, encoding="utf-8"))
    params = yaml.safe_load(io.open(cfg["parameter_file"], encoding="utf-8"))
    for k, v in overrides.items():
        base = params.get(k)
        # PRESERVE an existing sigma:0 pin -- overwriting it would change the
        # MODEL, not just the start, and the two points must share a model.
        if isinstance(base, dict) and base.get("sigma") == 0:
            continue
        params[k] = {"initval": v}
    s = System(cfg, user_params=params)
    s.prepare()
    m = s.build_model()
    ip = m.initial_point()
    lp = float(m.compile_logp()(ip))
    print("%-10s start logp = %+14.3f   (%d free RVs)"
          % (label, lp, len(m.free_RVs)), flush=True)
    return s, m, ip, lp


def term_logps(m, ip, label):
    """Per-term logp at `ip`, so the difference can be attributed.

    model.logp(sum=False) returns one node per factor (each observed
    likelihood, each prior, each Potential), which is exactly the
    decomposition 8.6.7 needs: severed-v3's total is not comparable to the
    mulens-only baseline BECAUSE of the extra terms, so the extra terms are
    the thing to look at.
    """
    import pytensor
    try:
        terms = m.logp(sum=False)
        names = [getattr(t, "name", None) or "term%d" % i
                 for i, t in enumerate(terms)]
        fn = pytensor.function(m.value_vars, terms, on_unused_input="ignore")
        vals = fn(*[ip[v.name] for v in m.value_vars])
        out = {n: float(np.sum(np.asarray(v))) for n, v in zip(names, vals)}
        print("%-10s %d logp terms, total %+.3f"
              % (label, len(out), sum(out.values())), flush=True)
        return out
    except Exception as e:  # noqa: BLE001
        print("%-10s PER-TERM BREAKDOWN UNAVAILABLE (%s: %s)"
              % (label, type(e).__name__, e), flush=True)
        return None


A = point_from_trace()
B = dict(A)
B.update(TRUTH)
print("\npoint B overrides point A in %d of %d values:" % (len(TRUTH), len(A)),
      flush=True)
for k, v in TRUTH.items():
    print("   %-26s A=%-14.6g -> B=%-14.6g" % (k, A.get(k, float("nan")), v),
          flush=True)

print("\n=== SAME MODEL, TWO COMPLETE POINTS ===", flush=True)
_, mA, ipA, lpA = build(A, "POINT A")
_, mB, ipB, lpB = build(B, "POINT B")
tA = term_logps(mA, ipA, "POINT A")
tB = term_logps(mB, ipB, "POINT B")

print("\n=== VERDICT ===", flush=True)
print("A = run's own argmax   : logp %+14.3f" % lpA, flush=True)
print("B = A + truth physics  : logp %+14.3f" % lpB, flush=True)
gap = lpB - lpA
print("B - A                  : %+14.3f nats" % gap, flush=True)

if tA and tB:
    print("\n=== WHERE THE DIFFERENCE LIVES (B - A, per logp term) ===",
          flush=True)
    keys = sorted(set(tA) | set(tB),
                  key=lambda k: -abs(tB.get(k, 0.0) - tA.get(k, 0.0)))
    print("%-44s %14s %14s %14s" % ("term", "A", "B", "B - A"), flush=True)
    for k in keys:
        va, vb = tA.get(k, float("nan")), tB.get(k, float("nan"))
        if abs(vb - va) < 0.05:
            continue
        print("%-44s %14.3f %14.3f %+14.3f" % (k[:44], va, vb, vb - va),
              flush=True)
    print("\nRead the sign per term: a term that is WORSE at truth (negative"
          "\nB - A) is a term whose model disagrees with the truth, which is"
          "\nthe misspecification 8.6.7 is looking for.  If the SED/torres/"
          "\nmann terms carry it, the stellar chain is the culprit; if the"
          "\nmicrolensing likelihood carries it, the light curve itself"
          "\nprefers the wrong geometry and that is a much bigger claim.",
          flush=True)
print("""
B - A NEGATIVE and LARGE -> this model genuinely prefers the wrong solution.
     MISSPECIFICATION, not a sampler failure; 8.6.7's "NOT CLAIMED" becomes
     a claim.  The per-term table then says which terms bought it.
B - A POSITIVE           -> the correct solution scores better and the
     sampler never reached it: a SEARCH failure, and the star-swap basin is
     a trap rather than the optimum.
|B - A| SMALL            -> INCONCLUSIVE.  Neither point is a basin optimum
     and this script does not polish (see the module docstring), so a gap of
     order tens of nats on a total near 87,786 decides nothing.
""", flush=True)
json.dump({"lp_A": lpA, "lp_B": lpB, "terms_A": tA, "terms_B": tB,
           "truth_overrides": TRUTH},
          open("../dc128_severed_startlogp.json", "w"), indent=1)
