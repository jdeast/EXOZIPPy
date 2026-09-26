"""The galacticmodel kinematic prior ALONE, maximized over the velocity
coordinates at fixed distances, evaluated on the MODEL's own graph.

Why: dc18_ds_profile.py reported galacticmodel.kinematic_prior 12 nats
worse with the 194 source at 9980 pc than at 4809 (Teff pinned to the key's
5257 K, everything else optimized), while a numpy re-optimization of the
same prior (scratch kin_profile.py) preferred the FAR source by 3 nats.
Either the numpy transcription differs from the model, or Powell -- a
coordinate search -- never left the velocity basin it started in (the prior
is three-branch multimodal in proper motion).  This settles it on the
model's own potential, with nothing else in the objective:

  * compile the kinematic_prior Potential as a function of the value
    vector (its graph has no light-curve Op, so this is cheap);
  * take a posterior draw, pin the source distance to each requested
    value (root-solving its raw, as the profiler does) with pi_rel and
    mu_rel held at the draw's values, and maximize the potential over
    pm_ra/pm_dec of the anchor star and rv of both stars from many starts
    (Nelder-Mead; the function costs microseconds);
  * print, per distance, the maximum, the maximizing velocities, and the
    value at the profiler's own reported proper motions if given.
"""

import argparse
import os
import re
import sys

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--distances",
        default="3000,4000,4809,5500,6500,7500,8500,9980,11000,12000",
    )
    ap.add_argument(
        "--vary",
        default=r"^(star\.pm_ra_raw|star\.pm_dec_raw|star\.rv_raw)",
        help="regex on value-vector element names the maximization moves",
    )
    ap.add_argument("--starts", type=int, default=24)
    args = ap.parse_args()

    import arviz as az
    import pytensor
    from scipy.optimize import brentq, minimize

    from exozippy.system import System
    from exozippy.whitening import restore_whitening_for_trace

    run_dir = os.path.abspath(args.run_dir)
    pre = os.path.join(run_dir, f"DC2018_{args.event}")
    os.chdir(run_dir)
    config = yaml.safe_load(open(pre + ".yaml"))
    user_params = yaml.safe_load(open(config["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    print(
        "whitening:",
        restore_whitening_for_trace(
            system, pre + "_whitening.json", pre + "_trace.nc"
        ),
    )

    idata = az.from_netcdf(pre + "_trace.nc")
    post, lp = idata.posterior, idata.sample_stats.lp.values
    vv = list(model.value_vars)
    vnames = [v.name for v in vv]
    ip = model.initial_point()
    shapes = [np.asarray(ip[n]).shape for n in vnames]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    offsets = np.cumsum([0] + sizes)
    elem = []
    for n, s, k in zip(vnames, shapes, sizes):
        elem.extend(
            [n] if (k == 1 and s == ()) else [f"{n}[{i}]" for i in range(k)]
        )

    def split(x):
        return [
            np.asarray(x[offsets[i] : offsets[i + 1]], dtype=float).reshape(
                shapes[i]
            )
            for i in range(len(vv))
        ]

    pot = next(
        p for p in model.potentials if p.name.endswith("kinematic_prior")
    )
    f_kin = pytensor.function(
        vv,
        model.replace_rvs_by_values([pot])[0].sum(),
        on_unused_input="ignore",
    )
    want = [
        "star.distance",
        "star.pm_ra",
        "star.pm_dec",
        "star.rv",
        "mulensevent.mu_ra_rel",
        "mulensevent.mu_dec_rel",
        "mulensevent.log_pi_rel",
    ]
    dets = [d for d in model.deterministics if d.name in want]
    f_det = pytensor.function(
        vv, model.replace_rvs_by_values(dets), on_unused_input="ignore"
    )

    def det(x):
        return {
            d.name: np.atleast_1d(np.asarray(o, dtype=float))
            for d, o in zip(dets, f_det(*split(x)))
        }

    ib = np.unravel_index(np.argmax(lp), lp.shape)
    x0 = np.concatenate(
        [
            np.asarray(post[n].values[ib], dtype=float).reshape(-1)
            for n in vnames
        ]
    )
    d0 = det(x0)
    print(
        f"start: best draw chain {ib[0]} draw {ib[1]}; distances {d0['star.distance']}, pm_ra {d0['star.pm_ra']}, "
        f"pm_dec {d0['star.pm_dec']}, rv {d0['star.rv']}, mu_rel ({d0['mulensevent.mu_ra_rel'][0]:.3f}, "
        f"{d0['mulensevent.mu_dec_rel'][0]:.3f}), log_pi_rel {d0['mulensevent.log_pi_rel'][0]:.3f}"
    )
    print(f"kinematic_prior at the start draw: {float(f_kin(*split(x0))):.3f}")

    # the knob that moves star.distance[1]
    dgrad = pytensor.function(
        vv,
        pytensor.grad(
            model.replace_rvs_by_values([model["star.distance"]])[0][1],
            vv,
            disconnected_inputs="ignore",
        ),
        on_unused_input="ignore",
    )
    g = np.concatenate(
        [np.asarray(a, dtype=float).reshape(-1) for a in dgrad(*split(x0))]
    )
    knob = int(np.argmax(np.abs(g)))
    assert np.sum(np.abs(g) > 1e-9 * np.abs(g).max()) == 1, (
        "distance pin is not a single element"
    )
    print(f"star.distance[1] knob: {elem[knob]}")
    vary = np.array([bool(re.search(args.vary, n)) for n in elem])
    print(
        f"maximizing over {int(vary.sum())} elements: {[n for n, v in zip(elem, vary) if v]}\n"
    )

    def pin(x, tgt):
        x = x.copy()

        def resid(v):
            y = x.copy()
            y[knob] = v
            return float(det(y)["star.distance"][1]) - tgt

        lo, hi, step = x[knob] - 0.5, x[knob] + 0.5, 0.5
        while resid(lo) * resid(hi) > 0:
            step *= 1.6
            lo, hi = x[knob] - step, x[knob] + step
        x[knob] = brentq(resid, lo, hi, xtol=1e-10)
        return x

    rng = np.random.default_rng(1)
    print(
        f"{'D_s':>6}{'D_l':>7}{'max kin':>10}{'at draw vel':>12}   pm_s (ra,dec)   pm_l (ra,dec)   rv_s  rv_l (km/s)"
    )
    for tgt in [float(s) for s in args.distances.split(",")]:
        xp = pin(x0, tgt)
        base = float(f_kin(*split(xp)))

        def fun(z):
            x = xp.copy()
            x[vary] = z
            v = f_kin(*split(x))
            return 1e6 if not np.isfinite(v) else -float(v)

        best = None
        for s in range(args.starts):
            z0 = xp[vary] + (
                0.0 if s == 0 else rng.normal(0, 3.0, size=int(vary.sum()))
            )
            r = minimize(
                fun,
                z0,
                method="Nelder-Mead",
                options=dict(
                    xatol=1e-4, fatol=1e-5, maxiter=40000, maxfev=80000
                ),
            )
            if best is None or r.fun < best.fun:
                best = r
        x = xp.copy()
        x[vary] = best.x
        d = det(x)
        print(
            f"{tgt:6.0f}{d['star.distance'][0]:7.0f}{-best.fun:10.2f}{base:12.2f}   "
            f"({d['star.pm_ra'][1]:+.2f},{d['star.pm_dec'][1]:+.2f})   ({d['star.pm_ra'][0]:+.2f},{d['star.pm_dec'][0]:+.2f})"
            f"   {d['star.rv'][1] / 1e3:5.0f} {d['star.rv'][0] / 1e3:5.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
