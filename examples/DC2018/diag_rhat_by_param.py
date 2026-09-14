"""Which parameters carry the residual Rhat?

WHY.  diag_convergence.py answers WHETHER a run equilibrated and whether its
chains split across modes.  Three of the five DC2018 traces came back
"SLOW/ok" -- stationary, chains in agreement -- and yet report Rhat ~1.5.
That number has to come from SOME parameter, and which one decides whether
the run is usable:

  * nuisances only (fluxes, blends, noise scales) -> the geometry is
    resolved and the fit is reportable; the remedy is more draws, not a
    different sampler.
  * geometry (t_0, u_0, s, q, rho, t_E) -> the physics is not resolved and
    NOTHING downstream (detectability, ladder, truth comparison) means
    anything yet.

So this splits the Rhat table by ROLE rather than reporting a single worst
number, which is what a bare Rhat summary hides.
"""

import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import arviz as az  # noqa: E402

# Anything matching one of these is a nuisance: an instrument-level flux,
# blend, zeropoint, detrend coefficient or noise knob.  Everything else is
# treated as physics, so a NEW physics parameter shows up as physics rather
# than being silently excluded.
NUISANCE_HINTS = (
    "instrument", "flux", "blend", "zero", "detrend", "jitter",
    "errscale", "sigma", "outlier", "hogg", "gp_", "noise",
)


def role(name):
    low = name.lower()
    return "nuisance" if any(h in low for h in NUISANCE_HINTS) else "physics"


def report(path):
    idata = az.from_netcdf(path)
    post = idata.posterior
    rows = []
    for name in post.data_vars:
        v = np.asarray(post[name])
        if v.ndim < 2 or not np.all(np.isfinite(v)):
            continue
        if np.std(v) == 0:
            continue  # pinned
        try:
            r = float(az.rhat(post[name]).values.max())
            e = float(az.ess(post[name]).values.min())
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(r):
            rows.append((r, e, name, role(name)))
    if not rows:
        print("  (no usable variables)")
        return
    rows.sort(reverse=True)

    for label in ("physics", "nuisance"):
        sub = [r for r in rows if r[3] == label]
        if not sub:
            continue
        worst = [r for r in sub if r[0] > 1.05]
        print(
            "  %-8s n=%3d  max Rhat %5.2f  n>1.05: %d/%d"
            % (label, len(sub), sub[0][0], len(worst), len(sub))
        )
        for r, e, name, _ in sub[:6]:
            flag = "  <<<" if r > 1.05 else ""
            print("      %-42s Rhat %5.2f  ESS %8.0f%s" % (name, r, e, flag))


for path in sys.argv[1:]:
    for p in sorted(glob.glob(path)):
        print("\n=== %s" % os.path.relpath(p))
        try:
            report(p)
        except Exception as e:  # noqa: BLE001
            print("  FAILED: %s: %s" % (type(e).__name__, e))
