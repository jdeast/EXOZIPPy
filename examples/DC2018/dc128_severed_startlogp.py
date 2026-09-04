"""Does the severed+SED model actually PREFER the wrong solution? (8.6.7)

8.6.7 deliberately does NOT claim that severed-v3's wrong solution beats the
right one on posterior probability: its lp (max 87,786) carries SED, torres
and mann terms the capped mulens-only baseline (87,759) never evaluates, so
the two numbers are not comparable.  The honest test is the SAME model
evaluated at the correct solution and compared with itself.  That is a
start-logp evaluation, not a fit.

This is split out of dc128_severed_v3_followup.py on purpose: there it sat
AFTER the hot-chain section, whose guard correctly refused to continue (the
rebuilt model's raw->physical map disagrees with the run's by 3.65 in log10
rho -- review 2.4.15), so this half never ran at all.  It has no dependency
on the trace and should not share a process with something that can bail.

TRUTH is dc128_truth_forward.json, not the mulens-only arm's recovered rho:
rho = 0.0060668, theta_E = 0.0904643 mas, M_lens = 0.454 Msun,
pi_rel = 0.0022108 mas, source 0.961 Rsun at 8.14 kpc, lens at 7.999 kpc.
An earlier version of the follow-up injected rho = 0.00713, which is the
CAPPED ARM'S POSTERIOR (8.6.3), not the truth.

READ THE SIGN CAREFULLY.  A start point is not a basin optimum, so a small
negative number is not evidence; a large one is.  Reported both ways.
"""

import io
import json
import logging
import os

import numpy as np
import yaml

logging.disable(logging.WARNING)
os.chdir("/home/jeastman/python/EXOZIPPy/examples/DC2018/configs")

from exozippy.system import System  # noqa: E402

CFG = "DC2018_128_severed_v3.yaml"
RUN_MAX_LP = 87786.2  # severed-v3's own posterior max

TRUTH = {
    "lens.Lens.log_rho": float(np.log10(0.0060668)),
    "lens.Lens.log_theta_E": float(np.log10(0.0904643)),
    "star.Source.radius": 0.961,
    "star.Source.distance": 8140.0,
    "star.Lens.distance": 7999.0,
    "star.Lens.logmass": float(np.log10(0.454)),
}


def build(extra, label):
    cfg = yaml.safe_load(io.open(CFG, encoding="utf-8"))
    params = yaml.safe_load(io.open(cfg["parameter_file"], encoding="utf-8"))
    for k, v in extra.items():
        params[k] = {"initval": v}
    s = System(cfg, user_params=params)
    s.prepare()
    m = s.build_model()
    ip = m.initial_point()
    n = sum(int(np.asarray(ip[v.name]).size) for v in m.value_vars)
    lp = float(m.compile_logp()(ip))
    print("%-22s %d free RVs / %d scalar elements   start logp = %.3f"
          % (label, len(m.free_RVs), n, lp), flush=True)
    return lp, s


print("=== SAME MODEL, TWO START POINTS ===", flush=True)
lp_as_shipped, _ = build({}, "as-shipped start:")
lp_truth, s_truth = build(TRUTH, "truth-injected:")

# Did the injection actually take?  Setting a DERIVED quantity does not
# round-trip (the transit leg of 7.13.1 is the worked example), so this is
# checked rather than assumed.
print("\n=== DID THE INJECTION TAKE? ===", flush=True)
for name, want in TRUTH.items():
    comp, inst, pname = name.split(".", 2)
    p = getattr(getattr(s_truth, comp, None), pname, None)
    if p is None:
        print("  %-26s NO SUCH PARAMETER" % name, flush=True)
        continue
    got = np.atleast_1d(np.asarray(p.value.eval(), dtype=float))
    hit = float(np.nanmin(np.abs(got - want)))
    print("  %-26s wanted %-12.6g got %-28s |min diff| %.3g"
          % (name, want, np.array2string(got, precision=5), hit), flush=True)

print("\n=== VERDICT ===", flush=True)
print("severed-v3 posterior max lp        : %.1f" % RUN_MAX_LP, flush=True)
print("this model at the TRUTH start      : %.1f" % lp_truth, flush=True)
print("this model at its as-shipped start : %.1f" % lp_as_shipped, flush=True)
print("truth - run max                    : %+.1f nats"
      % (lp_truth - RUN_MAX_LP), flush=True)
print("""
NEGATIVE and LARGE  -> the model genuinely prefers the wrong solution.
                       That is MISSPECIFICATION, not a sampler failure, and
                       8.6.7's 'NOT CLAIMED' can be upgraded to a claim.
POSITIVE            -> the correct solution scores better and the sampler
                       never found it: a search failure, and the star-swap
                       basin is a trap rather than the true optimum.
NEGATIVE and SMALL  -> inconclusive; a start is not an optimum.  Polish both
                       points before concluding anything.
""", flush=True)
json.dump({"lp_truth": lp_truth, "lp_as_shipped": lp_as_shipped,
           "run_max_lp": RUN_MAX_LP, "truth": TRUTH},
          open("../dc128_severed_startlogp.json", "w"), indent=1)
