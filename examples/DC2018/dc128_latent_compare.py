"""ESS-weighted latent comparison across DC2018-128's healthy arms (8.6.7(3)).

The last loose end of the 128 decomposition.  Conclusion 3 of the A/B verdict
says the residual scatter among the HEALTHY arms -- logmass -0.13/-0.59/-0.41,
D_l 1.8-3.8 kpc -- is "~2 sigma with imperfect latent mixing".  That is a
statement about two things at once (a disagreement AND the mixing that makes
it hard to measure), which is exactly what an ESS-weighted comparison
separates: an arm whose latent block barely moved should not get the same
vote as one that mixed well.

WHAT "MODE-CONDITIONED" TURNS OUT TO MEAN HERE: nothing.  Every healthy arm
reports a UNIMODAL posterior (identify_modes: 1 mode, weight 1.000), so there
is no mode to condition on and the conditioning collapses to the whole trace.
That is worth stating rather than silently skipping -- the item asked for a
mode-conditioned comparison because it was written before the arms were known
to be unimodal.

WHAT IS COMPARED: the SAMPLED latent block, i.e. what is actually in the
posterior group -- star.logmass, star.distance, star.pm_ra, star.pm_dec.  The
derived lens quantities (theta_E, mu_rel, rho, t_E) are Parameters computed
through the model graph, not trace variables, so comparing them would need a
model rebuild per arm; out of scope here and noted rather than faked.

THE STATISTIC.  For each arm and each latent element:
  mean, sd, ESS (bulk), and MCSE = sd / sqrt(ESS).
Across arms, the question "do these agree?" is then a chi2 against the
inverse-MCSE-weighted mean:
  chi2 = sum_i (x_i - xbar)^2 / mcse_i^2,   dof = n_arms - 1
using MCSE and NOT sd, deliberately: sd is the posterior WIDTH (how well the
data constrain the parameter), while MCSE is the uncertainty on the arm's
estimate of the centre (how well the sampler resolved it).  Two arms whose
posteriors overlap heavily can still disagree about the centre if both mixed
well, and that disagreement is the thing worth knowing.  Both are reported so
the reader can see which regime each element is in.

Run:  python3 dc128_latent_compare.py [--out dc128_latent_compare.json]
"""

import argparse
import json
import os

import numpy as np

ARMS = [
    ("cap-only", "configs/fitresults_mulens_hoggcap/DC2018_128_trace.nc"),
    ("murel", "configs/fitresults_mulens_hoggcap_murel/DC2018_128_trace.nc"),
    ("u0te", "configs/fitresults_mulens_hoggcap_u0te/DC2018_128_trace.nc"),
    (
        "observable",
        "configs/fitresults_mulens_hoggcap_observable/DC2018_128_trace.nc",
    ),
]
# The under-mixed / stuck arms, carried for contrast only and EXCLUDED from
# the chi2 -- including them would be measuring the sampler, not the physics.
CONTRAST = [
    (
        "pirel (under-mixed)",
        "configs/fitresults_mulens_hoggcap_pirel/DC2018_128_trace.nc",
    ),
    (
        "thetae (STUCK)",
        "configs/fitresults_mulens_hoggcap_thetae/DC2018_128_trace.nc",
    ),
]

# Element 0 is the lens, element 1 the source (star order in the config).
LATENT = [
    ("star.logmass", 0, "log10 Msun (lens)"),
    ("star.distance", 0, "pc (lens, D_l)"),
    ("star.distance", 1, "pc (source, D_s)"),
    ("star.pm_ra", 0, "mas/yr (lens)"),
    ("star.pm_dec", 0, "mas/yr (lens)"),
]


def summarize(path, label):
    """Per-element mean/sd/ESS/MCSE for one arm, read one variable at a time."""
    import arviz as az
    import xarray as xr

    if not os.path.exists(path):
        print("  %-22s MISSING: %s" % (label, path), flush=True)
        return None
    ds = xr.open_dataset(path, group="posterior")
    out = {}
    for name, idx, unit in LATENT:
        if name not in ds.data_vars:
            continue
        da = ds[name]
        # Trailing element dim, when present.
        extra = [d for d in da.dims if d not in ("chain", "draw")]
        if extra:
            if da.sizes[extra[0]] <= idx:
                continue
            da = da.isel({extra[0]: idx})
        arr = np.asarray(da)  # (chain, draw)
        if not np.isfinite(arr).all():
            arr = np.where(np.isfinite(arr), arr, np.nan)
        mean = float(np.nanmean(arr))
        sd = float(np.nanstd(arr, ddof=1))
        try:
            ess = float(az.ess(da).values)
        except Exception:  # noqa: BLE001
            ess = float("nan")
        mcse = sd / np.sqrt(ess) if ess and np.isfinite(ess) else float("nan")
        key = "%s[%d]" % (name, idx)
        out[key] = {
            "unit": unit,
            "mean": mean,
            "sd": sd,
            "ess": ess,
            "mcse": mcse,
            "n_chain": int(arr.shape[0]),
            "n_draw": int(arr.shape[1]),
        }
        print(
            "  %-22s %-22s mean=%12.5g sd=%10.4g ess=%9.1f mcse=%10.4g"
            % (label, key, mean, sd, ess, mcse),
            flush=True,
        )
    ds.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dc128_latent_compare.json")
    args = ap.parse_args()

    result = {"arms": {}, "contrast": {}, "consistency": {}}
    print("=== HEALTHY ARMS (enter the chi2) ===", flush=True)
    for label, path in ARMS:
        s = summarize(path, label)
        if s:
            result["arms"][label] = s
    print("\n=== CONTRAST ARMS (excluded from the chi2) ===", flush=True)
    for label, path in CONTRAST:
        s = summarize(path, label)
        if s:
            result["contrast"][label] = s

    print("\n=== CROSS-ARM CONSISTENCY (healthy arms only) ===", flush=True)
    print(
        "%-22s %6s %14s %14s %8s %6s  %s"
        % ("element", "n", "ESS-wtd mean", "spread", "chi2", "dof", "reading"),
        flush=True,
    )
    for name, idx, unit in LATENT:
        key = "%s[%d]" % (name, idx)
        xs, ws, sds = [], [], []
        for label in result["arms"]:
            e = result["arms"][label].get(key)
            if not e or not np.isfinite(e["mcse"]) or e["mcse"] <= 0:
                continue
            xs.append(e["mean"])
            ws.append(1.0 / e["mcse"] ** 2)
            sds.append(e["sd"])
        if len(xs) < 2:
            continue
        xs = np.array(xs)
        ws = np.array(ws)
        xbar = float((xs * ws).sum() / ws.sum())
        chi2 = float((ws * (xs - xbar) ** 2).sum())
        dof = len(xs) - 1
        spread = float(xs.max() - xs.min())
        typ_sd = float(np.median(sds))
        # Is the disagreement large compared with the POSTERIOR width, or only
        # compared with the sampler's resolution of the centre?
        reading = (
            "consistent"
            if chi2 < 2 * dof
            else "MARGINAL"
            if chi2 < 9 * dof
            else "INCONSISTENT"
        )
        reading += " (spread = %.2f posterior sd)" % (
            spread / typ_sd if typ_sd else float("nan")
        )
        print(
            "%-22s %6d %14.6g %14.4g %8.1f %6d  %s"
            % (key, len(xs), xbar, spread, chi2, dof, reading),
            flush=True,
        )
        result["consistency"][key] = {
            "unit": unit,
            "n_arms": len(xs),
            "ess_weighted_mean": xbar,
            "spread": spread,
            "chi2": chi2,
            "dof": dof,
            "spread_in_posterior_sd": (spread / typ_sd if typ_sd else None),
            "reading": reading,
        }

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("\nwrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
