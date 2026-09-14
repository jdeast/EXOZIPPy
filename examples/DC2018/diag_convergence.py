"""Why do these fits not converge?  Classify, do not guess.

Convergence is the prerequisite for everything else: the architecture
ladder's rungs violated monotonicity (a nested model fitting WORSE than its
parent, which is impossible at an optimum), the basin test's widths are
ambiguous between "unconstrained" and "unconverged", and delta chi2 only
means something between two CONVERGED fits.

So the question is not "is Rhat bad" but WHICH failure it is, because the
remedies are different and mutually exclusive:

  TRANSIENT   the T=1 lp is still trending -- the chains never reached a
              stationary distribution.  Rhat/ESS describe a burn-in
              (review 2.4.13).  Remedy: longer run, a tune phase whose
              draws are discarded, or a better start.  NOT a ladder change.
  MULTIMODAL  equilibrated, but chains sit in different modes and do not
              exchange.  Rhat is then measuring mode separation, which is
              real structure rather than a sampler defect.  Remedy: better
              tempering transport, or report per-mode.
  SLOW        equilibrated, single mode, just not enough effective samples.
              Remedy: more draws.  The cheap case.
  STUCK       some chains never reached the good-likelihood region at all.

The equilibration test is convergence.check_equilibration, merged in PR
#249: it compares the final quarter's mean lp against the third quarter's,
with the noise estimated from the LAG-1 DIFFERENCE so a trending chain
cannot hide behind its own trend.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, "/home/jeastman/python/EXOZIPPy/src")
from exozippy.samplers.convergence import check_equilibration  # noqa: E402


def classify(path):
    lab = Path(path).parent.name if "fitresults" in path else path
    try:
        ss = xr.open_dataset(path, group="sample_stats")
    except Exception as e:  # noqa: BLE001
        return (lab, "unreadable", str(e)[:40], None, None, None, None)
    if "lp" not in ss.data_vars:
        ss.close()
        return (lab, "no lp", "", None, None, None, None)
    lp = np.asarray(ss["lp"], dtype=float)
    ss.close()
    if lp.ndim == 1:
        lp = lp[None, :]
    nch, ndr = lp.shape

    eq = check_equilibration(lp)
    equilibrated = eq["equilibrated"] if eq else None
    drift_sig = eq["lp_drift_sigma"] if eq else float("nan")

    # MODE SEPARATION MUST BE MEASURED IN PARAMETER SPACE, NOT lp.
    # The microlensing degeneracies are SYMMETRIES -- +/-u_0 and close/wide
    # s <-> 1/s -- so their modes have near-identical likelihood by
    # construction.  Event 223's two modes differ by 6.5 nats out of
    # 105,000; an lp-based test called that "SLOW/ok, just needs more
    # draws" for a fit that demonstrably has two modes and is missing a
    # third.  lp cannot see a mirror.
    q4 = lp[:, 3 * ndr // 4:]
    per = np.nanmean(q4, axis=1)
    within = float(np.nanmedian(np.nanstd(q4, axis=1)))
    sep = 0.0
    sep_var = "-"
    try:
        post = xr.open_dataset(path, group="posterior")
        # the parameters the known degeneracies act on
        for v in ("source.u_0", "lens.u_0", "lens.log_s",
                  "lens.Companion.log_s", "planet.log_q"):
            if v not in post.data_vars:
                continue
            a = np.asarray(post[v], dtype=float)
            if a.ndim < 2:
                continue
            a = a.reshape(a.shape[0], a.shape[1], -1)[:, :, 0]
            b = a[:, 3 * a.shape[1] // 4:]
            w = float(np.nanmedian(np.nanstd(b, axis=1)))
            x = float(np.nanstd(np.nanmean(b, axis=1)))
            r = x / w if w > 0 else 0.0
            if r > sep:
                sep, sep_var = r, v
        post.close()
    except Exception:  # noqa: BLE001
        pass

    # stuck chains: far below the best chain in lp
    best = float(np.nanmax(per))
    stuck = int(np.sum(per < best - 10 * max(within, 1e-9)))

    if equilibrated is False:
        kind = "TRANSIENT"
    elif stuck > 0:
        kind = "STUCK(%d)" % stuck
    elif sep > 3:
        kind = "MULTIMODAL"
    else:
        kind = "SLOW/ok"
    return (lab, kind, sep_var, nch, ndr, drift_sig, sep)


def main(paths):
    print("%-38s %-12s %5s %8s %10s %8s"
          % ("run", "diagnosis", "chn", "draws", "lp drift", "mode sep"))
    print("-" * 88)
    for p in paths:
        lab, kind, note, nch, ndr, drift, sep = classify(p)
        if nch is None:
            print("%-38s %-12s %s" % (lab[:38], kind, note))
            continue
        print("%-38s %-12s %5d %8d %10.1f %8.1f  %s"
              % (lab[:38], kind, nch, ndr, drift, sep, note))
    print("\nlp drift = (mean lp of the last quarter - the third quarter) in")
    print("units of the LAG-1 noise.  >3 means not stationary (2.4.13).")
    print("mode sep = spread of per-chain means / within-chain scatter, in")
    print("           PARAMETER space (the column names the worst one).  lp")
    print("           cannot be used: +/-u_0 and close/wide are SYMMETRIES,")
    print("           so their modes have near-equal likelihood -- 223's two")
    print("           differ by 6.5 nats out of 105,000.")


if __name__ == "__main__":
    main(sys.argv[1:])
