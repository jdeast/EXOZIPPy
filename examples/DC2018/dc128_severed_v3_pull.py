"""severed-v3 rho vs rho_pred pull, extracted TRACE-DIRECT (8.6.7 / sweep gate).

Job 15377879 sampled all 50,000 draws successfully (max_rhat 1.023, min_ess
2445, Lambda 12.75) and then was KILLED in wrap-up -- execd h_rss limit,
exit 137, maxvmem 771.6 GB, during the hot-chain candidate polish.  So there
is no results.csv and no modes.txt, but the 4.2 GB trace was written before
the kill and the measurement is recoverable.

This follows the established trace-direct forensic pattern rather than
re-running the reporter: review 2.10.3 records that the stale-trace hash
blocks even modes:{force:true}, so going through the normal output path is
not available.

THE PULL IS THE PRODUCT.  rho is what the light curve measures; rho_pred is
what the stellar chain (radius, distance, theta_E) predicts for it.  With the
rho tie SEVERED (star_constrains_rho: False) the two are independent
statements about the same quantity, so their disagreement is the test of
whether the SED-constrained stellar model and the light curve agree.  A large
pull means the stellar side is pulling rho somewhere the light curve does not
want it -- which is the whole reason the tie was severed.
"""

import logging
import os

import numpy as np
import yaml

logging.disable(logging.WARNING)
os.chdir("/home/jeastman/python/EXOZIPPy/examples/DC2018/configs")

import arviz as az  # noqa: E402
import pytensor  # noqa: E402

from exozippy.system import System  # noqa: E402

CFG = "DC2018_128_severed_v3.yaml"
TRACE = "fitresults_severed_v3/DC2018_128_trace.nc"

config = yaml.safe_load(open(CFG))
user = yaml.safe_load(open(config["parameter_file"]))
for k in ("run", "prefix", "parameter_file", "sampler"):
    config.pop(k, None)
s = System(config, user_params=user)
s.prepare()
model = s.build_model()
print("model built: %d free RVs" % len(model.free_RVs), flush=True)

idata = az.from_netcdf(TRACE)
post = idata.posterior
lp = np.asarray(idata.sample_stats["lp"]).ravel()
names_v = [v.name for v in model.value_vars]
raws = [
    np.asarray(post[n]).reshape(-1, *np.asarray(post[n]).shape[2:])
    for n in names_v
]

# Same validity screen the mode reporter would have applied: finite lp, not a
# runaway in lp, and raw-space robust z < 50.  Reproduced here because the
# reporter never got to run.
med = np.median(lp[np.isfinite(lp)])
valid = np.isfinite(lp) & (lp > med - 5000)
Z = np.zeros(lp.size)
for r in raws:
    rr = r.reshape(lp.size, -1)
    mu, sd = np.median(rr, axis=0), np.std(rr, axis=0) + 1e-12
    Z = np.maximum(Z, np.abs((rr - mu) / sd).max(axis=1))
valid &= Z < 50
print(
    "draws %d, valid %d (%.2f%%)  -- invalid frac %.2f%%"
    % (lp.size, valid.sum(), 100 * valid.mean(), 100 * (1 - valid.mean())),
    flush=True,
)


def draws_of(param, n=4000):
    (node,) = model.replace_rvs_by_values([param.value])
    fn = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    idx = np.nonzero(valid)[0]
    sel = idx[np.linspace(0, idx.size - 1, min(n, idx.size)).astype(int)]
    ndims = [v.ndim for v in model.value_vars]

    def args(k):
        out = []
        for r, nd in zip(raws, ndims):
            a = np.asarray(r[k], dtype=float)
            if nd == 1 and a.ndim == 0:
                a = a.reshape(1)
            out.append(a)
        return out

    return np.array([np.atleast_1d(fn(*args(k)))[0] for k in sel])


def stat(v, name):
    lo, m, hi = np.percentile(v, [15.865, 50, 84.135])
    print("  %-12s %.6g  +%.3g -%.3g" % (name, m, hi - m, m - lo), flush=True)
    return m, 0.5 * (hi - lo)


print("\n=== the product ===", flush=True)
rho = draws_of(s.lens.rho)
m1, s1 = stat(rho, "rho")
try:
    rho_pred = draws_of(s.lens.rho_pred)
    m2, s2 = stat(rho_pred, "rho_pred")
    pull = (np.log(m2) - np.log(m1)) / np.sqrt((s1 / m1) ** 2 + (s2 / m2) ** 2)
    print("\n  log-pull (rho_pred vs rho): %+.2f sigma" % pull, flush=True)
except AttributeError:
    print("  rho_pred: not present on this parameterization", flush=True)

print("\n=== context ===", flush=True)
for label, param in (
    ("theta_E", getattr(s.lens, "theta_E", None)),
    ("mu_rel_mag", getattr(s.lens, "mu_rel_mag", None)),
    ("t_E", getattr(s.lens, "t_E", None)),
):
    if param is not None:
        stat(draws_of(param, n=1500), label)
print(
    "\n128 truth for reference: rho 6.07e-3, theta_E 0.0906 mas,"
    " mu_rel ~1.8 mas/yr; the light curve's own blind d=7 NS value"
    " was rho = 7.3e-3 +/- 2.5%.",
    flush=True,
)
