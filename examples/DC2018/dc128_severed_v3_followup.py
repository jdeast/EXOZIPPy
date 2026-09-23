"""severed-v3 follow-ups: same-model lp comparison + the hot-chain landscape.

Two questions, one model build.

(1) DOES THE WRONG SOLUTION ACTUALLY WIN?  The run's own lp (max 87,786) is
    NOT comparable to the capped mulens-only baseline's (87,759) because
    severed-v3 carries SED/torres/mann likelihood terms the baseline never
    evaluates.  The honest test is the SAME model evaluated at the correct
    solution and compared with itself.  So: take the severed+SED model, set
    the light-curve and stellar quantities to the known-good values
    (rho = 0.0060668, theta_E = 0.0904643 mas, a 0.961 Rsun source at
    8.14 kpc -- dc128_truth_forward.json, NOT the 0.00713 an earlier
    version of this file carried), and
    read off logp.

(2) DOES THE CORRECT SOLUTION EXIST IN THIS POSTERIOR AT ALL?  The trace
    kept its hot chains (`posterior_hot`, 1748 x 2474 draws) but stores only
    RAW coordinates.  Rather than reverse-engineer the whitened logit by
    hand -- an earlier attempt at hand-rolled extraction produced confident
    nonsense -- this uses the MODEL'S OWN transform
    (model.rvs_to_transforms[...].backward) to map raw -> physical.  That is
    the authoritative route and it is the reason this waits on a model
    build.

NOTE the model rebuild is now known FAITHFUL: 29 free RVs is 38 scalar
elements, matching the run's n_params=38.  The earlier "it loses the
SED/mann parameters" claim was a category error and is retracted.
"""

import io
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

cfg = yaml.safe_load(io.open(CFG, encoding="utf-8"))
system = System(cfg, user_params=None)
system.prepare()
model = system.build_model()
ip = model.initial_point()
n_elem = sum(int(np.asarray(ip[v.name]).size) for v in model.value_vars)
print(
    "model: %d free RVs / %d scalar elements (run logged n_params=38)"
    % (len(model.free_RVs), n_elem),
    flush=True,
)
logp_fn = model.compile_logp()
print("start logp = %.3f" % float(logp_fn(ip)), flush=True)

# ---------------------------------------------------------------- (2) hot
print(
    "\n=== HOT-CHAIN LANDSCAPE: does a LOW-theta_E mode exist here? ===",
    flush=True,
)
hot = xr.open_dataset(TRACE, group="posterior_hot")
cold = xr.open_dataset(TRACE, group="posterior")

# WHY THE QUESTION CHANGED.  The first version of this script asked whether
# the correct RHO exists in the hot chains.  It does not need to: the cold
# posterior's rho is 0.00715 [0.00673, 0.00752] against a truth of
# 0.0060668 -- 18% high, so truth is just OUTSIDE the 95% interval, but
# against the 8x error the TIED run made this is a different regime.  What is wrong is theta_E -- 0.597 mas [0.387, 0.969]
# against a truth of 0.0906, a factor of 6.6 -- and it is wrong in exactly
# the way severing predicts, since severing is what removed
# rho = theta_star/theta_E and left theta_E to be set by
# sqrt(kappa * M_lens * pi_rel) alone.  (Check: 8.144 * 0.899 * 0.0486 gives
# 0.597 mas, so the posterior IS self-consistent -- it is on the wrong branch
# of the mass-distance degeneracy, not incoherent.)  So the useful hot-chain
# question is whether the LOW-theta_E branch was ever visited.

# WHY NOT THE TRANSFORM.  `model.rvs_to_transforms[rv]` is None for these
# RVs -- the free RV IS `lens.log_rho_raw` and the physical `lens.log_rho`
# is a Deterministic downstream of it, so there is no transform object to
# call (that AttributeError is what killed job 15405143).  And the affine
# fit the cold group supports (p = 7.6575*r - 8.3142, max resid 3.5e-4) is
# only valid over the cold chain's raw range 0.7985..0.8110; the hot chains
# range far outside it, and extrapolating a local affine fit is precisely
# the hand-rolled-logit mistake this file already made once.
#
# The authoritative route: the hot group stores ALL 29 raw value vars -- the
# model's complete free-RV set -- so the model's own graph can simply be
# evaluated on them.
import pytensor  # noqa: E402

WANT = ["lens.log_rho", "lens.log_theta_E"]
dets = {d.name: d for d in model.deterministics}
missing = [w for w in WANT if w not in dets]
if missing:
    raise SystemExit("model has no deterministic named %r" % missing)

vnames = [v.name for v in model.value_vars]
absent = [n for n in vnames if n not in hot.data_vars]
if absent:
    raise SystemExit("hot group is missing value vars: %r" % absent)
print("hot group carries all %d value vars" % len(vnames), flush=True)

fn = pytensor.function(
    model.value_vars, [dets[w] for w in WANT], on_unused_input="ignore"
)


def evaluate(ds, stride):
    """[n, 2] of (log_rho, log_theta_E) over every `stride`-th draw."""
    cols = []
    for n in vnames:
        a = np.asarray(ds[n].isel(draw=slice(0, None, stride)))
        cols.append(a.reshape(a.shape[0] * a.shape[1], -1))
    out = np.empty((cols[0].shape[0], len(WANT)))
    for i in range(out.shape[0]):
        r = fn(*[c[i] for c in cols])
        out[i] = [float(np.atleast_1d(x)[0]) for x in r]
    return out


# VERIFY ON THE COLD GROUP FIRST, where raw and physical coexist.  If the
# graph does not reproduce the stored physical values there is no reason to
# trust it on the hot draws, and the script says so instead of guessing.
chk = evaluate(cold, 2000)
ref = np.column_stack(
    [np.asarray(cold[w].isel(draw=slice(0, None, 2000))).ravel() for w in WANT]
)
err = float(np.nanmax(np.abs(chk - ref)))
print(
    "graph verified on %d cold draws: max|err| = %.3e" % (len(chk), err),
    flush=True,
)
if err > 1e-6:
    raise SystemExit(
        "graph does not reproduce the cold group -- refusing to "
        "apply it to the hot draws."
    )

STRIDE = 20  # 1748 x 2474 draws is 4.3M; every 20th is ~216k
h = evaluate(hot, STRIDE)
rho_hot = 10 ** h[:, 0]
tE_hot = 10 ** h[:, 1]
lph = (
    np.asarray(hot["lp"].isel(draw=slice(0, None, STRIDE))).ravel()
    if "lp" in hot.data_vars
    else np.full(len(h), np.nan)
)
print(
    "hot draws evaluated: %d   rho %.3g..%.3g   theta_E %.3g..%.3g"
    % (
        len(h),
        np.nanmin(rho_hot),
        np.nanmax(rho_hot),
        np.nanmin(tE_hot),
        np.nanmax(tE_hot),
    ),
    flush=True,
)

print(
    "\n%-30s %10s %9s %12s"
    % ("theta_E window [mas]", "draws", "frac", "best lp")
)
for lo, hi, lab in (
    (0.00, 0.15, "TRUTH-LIKE  <0.15"),
    (0.15, 0.30, "0.15 - 0.30"),
    (0.30, 1.00, "COLD BASIN  0.3 - 1.0"),
    (1.00, 1e9, ">1.0"),
):
    m = (tE_hot >= lo) & (tE_hot < hi)
    best = float(np.nanmax(lph[m])) if m.any() else float("nan")
    print(
        "%-30s %10d %8.4f%% %12.1f"
        % (lab, int(m.sum()), 100 * float(m.mean()), best),
        flush=True,
    )
print("cold chain lp: max 87786.2, median 87772.5", flush=True)
print("truth: theta_E = 0.0904643 mas, rho = 0.0060668", flush=True)

# ------------------------------------------------- (1) same-model lp check
print("\n=== SAME-MODEL lp AT THE CORRECT SOLUTION ===", flush=True)
print(
    "Injecting the known-good values as initvals and re-building, so the",
    flush=True,
)
print("comparison is this model against itself.", flush=True)
good = {
    "lens.Lens.log_rho": {"initval": float(np.log10(0.0060668))},
    "lens.Lens.log_theta_E": {"initval": float(np.log10(0.0904643))},
    "star.Source.radius": {"initval": 0.961},
    "star.Source.distance": {"initval": 8140.0},
    "star.Lens.distance": {"initval": 7999.0},
}
base = yaml.safe_load(io.open(cfg["parameter_file"], encoding="utf-8"))
base.update(good)
cfg2 = yaml.safe_load(io.open(CFG, encoding="utf-8"))
s2 = System(cfg2, user_params=base)
s2.prepare()
m2 = s2.build_model()
ip2 = m2.initial_point()
lp_good = float(m2.compile_logp()(ip2))
print("logp at the correct-solution start : %.3f" % lp_good, flush=True)
print("severed-v3's own posterior max lp  : 87786.2", flush=True)
print(
    "difference (correct - found)       : %+.1f nats" % (lp_good - 87786.2),
    flush=True,
)
print(
    "\nA NEGATIVE difference means this model genuinely prefers the wrong",
    flush=True,
)
print(
    "solution -- misspecification, not a sampler failure.  A POSITIVE one",
    flush=True,
)
print(
    "means the correct solution is better and the sampler never found it.",
    flush=True,
)
print(
    "Caveat: a start point is not a basin optimum, so a small negative",
    flush=True,
)
print("number is not conclusive; a large one is.", flush=True)
