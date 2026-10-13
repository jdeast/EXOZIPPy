"""Review 7.6.4 step 2: which lens geometry do these data prefer?

Builds the shipped model (I + V difference flux, A_V prior, companions tied
to the primary) three ways and Powell-optimizes every remaining coordinate:

  published : Han+2023 Table 2 geometry PINNED (s2 1.386, q2 0.526, alpha2
              48.68; s3 1.601, q3 0.241, alpha3 127.92 in EXOZIPPy's
              convention), everything else free;
  async     : the first acceptance fit's geometry PINNED (s2 2.009, q2
              0.414, alpha2 52.7; s3 2.09, q3 0.211, alpha3 167.5), seeded at
              its t_0 / u_0 / t_E / rho, everything else free;
  free      : nothing pinned, from the published seed.

Pins are params-file hard links (sigma 0) on the sampled coordinates log_s,
xalpha, yalpha and on the companion logmasses relative to the primary (q is
derived from the masses).  Each optimum is decomposed with the raw priors
collapsed against their logit-Jacobian potentials, so the comparison is
light-curve data vs priors.

    python profile_geometry.py <published|async|free> [maxfev]
"""

import copy
import importlib.util
import math
import os
import sys
import time

import numpy as np
import yaml
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
D2R = math.pi / 180.0
# The angle pair cannot be pinned in a params file (its prior is structural;
# the loader refuses a sigma), so the pinned variants seed it and HOLD these
# sampled coordinates fixed in the optimizer instead.
FROZEN = {"lens.xalpha_raw", "lens.yalpha_raw"}


def geometry_pins(s2, q2, a2, s3, q3, a3):
    return {
        "lens.LensB.log_s": {"initval": math.log10(s2), "sigma": 0},
        "lens.LensB.xalpha": {"initval": math.cos(a2 * D2R)},
        "lens.LensB.yalpha": {"initval": math.sin(a2 * D2R)},
        "lens.LensC.log_s": {"initval": math.log10(s3), "sigma": 0},
        "lens.LensC.xalpha": {"initval": math.cos(a3 * D2R)},
        "lens.LensC.yalpha": {"initval": math.sin(a3 * D2R)},
        "star.LensB.logmass": {
            "initval": f"star.Lens.logmass + {math.log10(q2):.6f}",
            "sigma": 0,
        },
        "star.LensC.logmass": {
            "initval": f"star.Lens.logmass + {math.log10(q3):.6f}",
            "sigma": 0,
        },
    }


def variant_params(prm, which):
    p = copy.deepcopy(prm)
    for k in (
        "lens.LensB.s",
        "lens.LensB.alpha",
        "star.LensB.mass",
        "star.LensC.mass",
    ):
        if which != "free":
            p.pop(k, None)
    if which == "published":
        p.update(geometry_pins(1.386, 0.526, 48.68, 1.601, 0.241, 127.92))
    elif which == "async":
        p.update(geometry_pins(2.009, 0.414, 52.7, 2.09, 0.211, 167.5))
        p["source.Source.t_0"] = {"initval": 2459370.09}
        p["source.Source.u_0"] = {"initval": -0.436}
        # t_E 6.55 d at the published theta_E 0.50 mas needs mu_rel 27.9
        # mas/yr: move the lens pm 27.9 east of the source's.
        p["star.Lens.pm_ra"] = {"initval": -3.0 + 27.9}
        p["star.Lens.pm_dec"] = {"initval": -5.4}
        # rho 0.0235 at theta_E 0.50 mas, D_S 8680 pc: R = rho*theta_E*D_S
        p["star.Source.radius"] = {
            "initval": 0.0235 * 0.50e-3 * 8680 * 215.032
        }
    return p


def main(which, maxfev, params=None, tag=None, out_npy=None):
    """`params` (a full params dict) overrides the named variant; `which`
    then only selects whether the angle coordinates are held (anything but
    \"free\").  `tag` labels the RESULT lines."""
    import pytensor
    import pytensor.tensor as pt

    from exozippy.system import System

    spec = importlib.util.spec_from_file_location(
        "mmf",
        os.path.join(HERE, "..", "..", "scripts", "make_mulens_fixtures.py"),
    )
    mmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mmf)
    os.chdir(HERE)
    cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    prm = yaml.safe_load(open(cfg["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    # PROFILE_LIKELIHOOD=hogg: every light curve gets the outlier mixture
    # (instrument.md "Robust observation likelihoods"), as a diagnostic of
    # whether a basin preference rides on a handful of points.
    like = os.environ.get("PROFILE_LIKELIHOOD")
    if like:
        for inst in cfg["mulensinstrument"]:
            inst["likelihood"] = like
        print(
            f"likelihood: {like} on all {len(cfg['mulensinstrument'])} light curves"
        )
    system = System(
        cfg,
        user_params=params
        if params is not None
        else variant_params(prm, which),
    )
    system.prepare()
    model = system.build_model()
    vv = list(model.value_vars)
    vnames = [v.name for v in vv]
    start = mmf.raw_start(system, model)
    shapes = [np.shape(np.asarray(start[n])) for n in vnames]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    offs = np.cumsum([0] + sizes)
    split = lambda x: [
        np.asarray(x[offs[i] : offs[i + 1]], dtype=float).reshape(shapes[i])
        for i in range(len(vv))
    ]
    f_lp = pytensor.function(
        vv, model.logp(sum=True), on_unused_input="ignore"
    )
    lp_at = lambda x: (lambda v: v if np.isfinite(v) else -np.inf)(
        float(f_lp(*split(x)))
    )
    dets = {d.name: d for d in model.deterministics}
    want = [
        n
        for n in (
            "lens.s",
            "lens.q",
            "lens.alpha",
            "mulensevent.t_E",
            "mulensevent.theta_E",
            "source.rho",
            "source.t_0",
            "source.u_0",
            "star.distance",
            "star.mass",
            "star.radius",
            "star.teff",
            "star.av",
            "mulensinstrument.f_source",
            "mulensinstrument.zeropoint",
            "mulensinstrument.err_scale",
            "mulensinstrument.q_source",
        )
        if n in dets
    ]
    g = pytensor.function(
        vv,
        model.replace_rvs_by_values([dets[n] for n in want]),
        on_unused_input="ignore",
    )

    def report(label, x):
        parts, total, ok, _ = mmf.decompose(
            system, model, dict(zip(vnames, split(x)))
        )
        col = {}
        for k, v in parts.items():
            v = float(v)
            if k.startswith("RV:") and k.endswith("_raw"):
                k = "CoV:" + k[3:-4]
            elif k.startswith("POT:logit_uniform_prior."):
                k = "CoV:" + k[len("POT:logit_uniform_prior.") :]
            col[k] = col.get(k, 0.0) + v
        data = sum(
            v
            for k, v in col.items()
            if k.startswith("RV:") and "mulensinstrument" in k
        )
        cov = sum(v for k, v in col.items() if k.startswith("CoV:"))
        print(
            f"\nRESULT {tag or which} {label}: total {total:.2f} = data {data:.2f} + coordinate priors {cov:.2f} + other {total - data - cov:.2f}",
            flush=True,
        )
        for k, v in sorted(col.items(), key=lambda kv: -abs(kv[1]))[:14]:
            print(f"   {k:55s} {v:12.2f}")
        for n, v in zip(want, g(*split(x))):
            print(
                f"   {n:30s} {np.array2string(np.atleast_1d(np.asarray(v, dtype=float)), precision=4, max_line_width=150)}"
            )

    x0 = np.concatenate(
        [
            np.atleast_1d(np.asarray(start[n], dtype=float)).ravel()
            for n in vnames
        ]
    )
    print(f"{which}: {x0.size} free elements")
    report("start", x0)
    lp0 = lp_at(x0)
    t0 = time.time()
    n = [0]

    elem = [nm for nm, k in zip(vnames, sizes) for _ in range(k)]
    vary = np.array([not (which != "free" and nm in FROZEN) for nm in elem])
    print(
        f"optimizing {int(vary.sum())} of {vary.size} elements; held: {sorted(set(e for e, v in zip(elem, vary) if not v))}"
    )

    def fun(xv):
        x = x0.copy()
        x[vary] = xv
        n[0] += 1
        v = lp_at(x)
        if n[0] % 500 == 0:
            print(
                f"   ... {n[0]} evaluations, {time.time() - t0:.0f} s",
                flush=True,
            )
        return 100.0 - (v - lp0) if np.isfinite(v) else 1e7

    res = minimize(
        fun,
        x0[vary],
        method="Powell",
        options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4},
    )
    xbest = x0.copy()
    xbest[vary] = res.x
    res.x = xbest
    print(
        f"\n{which}: {n[0]} evaluations in {time.time() - t0:.0f} s, lp {lp0:.2f} -> {lp_at(res.x):.2f}"
    )
    report("optimum", res.x)
    np.save(
        out_npy or os.path.join(HERE, f"profile_geometry_{which}.npy"), res.x
    )
    inst = system.mulensinstrument
    if getattr(inst, "_hogg_logodds", None):
        # Per-point outlier probability at the optimum, and WHERE the flagged
        # points sit: inside the anomaly window they are model misfit the
        # mixture is forgiving (the DC2018-128 failure); scattered through
        # the baseline they are bad data.
        tt = np.asarray(inst.time, dtype=float)
        print(
            "\nOUTLIERS (p > 0.5) at the optimum: total / within 3 d of HJD 2459371.0 / outside"
        )
        for i, node in inst._hogg_logodds.items():
            f = pytensor.function(
                vv,
                model.replace_rvs_by_values([pt.sigmoid(node)])[0],
                on_unused_input="ignore",
            )
            prob = np.asarray(f(*split(res.x)), dtype=float).ravel()
            t_i = tt[inst.rows(i)]
            flag = prob > 0.5
            near = np.abs(t_i - 2459371.0) < 3.0
            print(
                f"   {inst.names[i]:8s} n {t_i.size:5d}  flagged {int(flag.sum()):4d}  in-window {int((flag & near).sum()):4d} of {int(near.sum())}  outside {int((flag & ~near).sum()):4d}"
            )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
