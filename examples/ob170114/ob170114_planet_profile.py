"""Review 8.6.9 (1): did the acceptance fit MISS the published planet basin,
or does our likelihood prefer no planet?

The fit (fitresults_auto) left the published 2L1S + parallax + xallarap
seed (lp 5939.4; the DE polish moved it +0.0) for a planet-less basin 61
nats higher (lp max 6000.2, q at the log-uniform floor, s/alpha free).
Three optimizations from the SEED, all Powell on the model's value vector:

  pinned : planet.log_q, lens.log_s, lens.xalpha, lens.yalpha held at the
           seed (the published close planet, q 0.0219 / s 0.337 / alpha
           330.4), everything else free (err_scale included) -- the
           planet basin's own optimum;
  free   : everything free -- where a local optimizer goes from the seed;
  trace  : the stored posterior's best draw, re-evaluated and decomposed.

Each optimum is decomposed into data (observed) and prior terms with
scripts/make_mulens_fixtures.decompose, so "the planet-less basin wins by
X nats" is read off as data vs prior.
"""

import importlib.util
import os
import sys
import time

import numpy as np
import yaml
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
PINNED = (
    "planet.log_q_raw",
    "lens.log_s_raw",
    "lens.xalpha_raw",
    "lens.yalpha_raw",
)


def main(maxfev=30000):
    import arviz as az
    import pytensor

    from exozippy.system import System

    spec = importlib.util.spec_from_file_location(
        "mmf",
        os.path.join(HERE, "..", "..", "scripts", "make_mulens_fixtures.py"),
    )
    mmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mmf)
    os.chdir(HERE)
    config = yaml.safe_load(open("ob170114.yaml"))
    user_params = yaml.safe_load(open(config["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    # The trace's raw coordinates are defined by the whitening state the FIT
    # persisted; a fresh build has its own, and a trace draw read into it
    # is garbage (first run: the stored lp-6000 draw re-evaluated to
    # -370,535 with t_E 0.6 d).  Restore it before reading anything raw.
    from exozippy.whitening import restore_whitening_for_trace

    print(
        "whitening:",
        restore_whitening_for_trace(
            system,
            os.path.join(
                HERE, "fitresults_auto", "OGLE-2017-BLG-0114_whitening.json"
            ),
            os.path.join(
                HERE, "fitresults_auto", "OGLE-2017-BLG-0114_trace.nc"
            ),
        ),
    )
    vv = list(model.value_vars)
    vnames = [v.name for v in vv]
    start = mmf.raw_start(system, model)
    shapes = [np.shape(np.asarray(start[n])) for n in vnames]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    offsets = np.cumsum([0] + sizes)
    elem = []
    for n, s, k in zip(vnames, shapes, sizes):
        elem.extend(
            [n] if (k == 1 and s == ()) else [f"{n}[{i}]" for i in range(k)]
        )

    def pack(d):
        return np.concatenate(
            [np.asarray(d[n], dtype=float).reshape(-1) for n in vnames]
        )

    def split(x):
        return [
            np.asarray(x[offsets[i] : offsets[i + 1]], dtype=float).reshape(
                shapes[i]
            )
            for i in range(len(vv))
        ]

    def as_dict(x):
        return dict(zip(vnames, split(x)))

    f_lp = pytensor.function(
        vv, model.logp(sum=True), on_unused_input="ignore"
    )

    def lp_at(x):
        v = float(f_lp(*split(x)))
        return v if np.isfinite(v) else -np.inf

    def report(label, x):
        parts, total, ok, _ = mmf.decompose(system, model, as_dict(x))
        # Collapse each sampled coordinate's raw prior with its logit-Jacobian
        # potential (RV:<x>_raw + POT:logit_uniform_prior.<x> -> CoV:<x>): the
        # pair cancels to a few nats and is huge term by term, so a split
        # that leaves them apart calls the Jacobian "prior" (first run:
        # +47,000 nats of "prior" that were 47,000 of cancelling Jacobian).
        col = {}
        for k, v in parts.items():
            v = float(v)
            if k.startswith("RV:") and k.endswith("_raw"):
                col["CoV:" + k[3:-4]] = col.get("CoV:" + k[3:-4], 0.0) + v
            elif k.startswith("POT:logit_uniform_prior."):
                col["CoV:" + k[len("POT:logit_uniform_prior.") :]] = (
                    col.get("CoV:" + k[len("POT:logit_uniform_prior.") :], 0.0)
                    + v
                )
            else:
                col[k] = col.get(k, 0.0) + v
        data = sum(
            v
            for k, v in col.items()
            if k.startswith("RV:") and "mulensinstrument" in k
        )
        cov = sum(v for k, v in col.items() if k.startswith("CoV:"))
        other = total - data - cov
        big = sorted(col.items(), key=lambda kv: -abs(kv[1]))[:14]
        print(f"\nTERMS {label}: " + "; ".join(f"{k} {v:.1f}" for k, v in big))
        print(
            f"\nRESULT {label}: total lp {total:.2f} = data {data:.2f} + coordinate priors {cov:.2f} + other potentials {other:.2f}"
        )
        dets = {d.name: d for d in model.deterministics}
        want = [
            "planet.q",
            "lens.s",
            "lens.alpha",
            "mulensevent.t_E",
            "source.u_0",
            "source.rho",
            "mulensinstrument.err_scale",
            "star.mass",
            "orbit.period",
            "orbit.ecc",
            "mulensevent.theta_E",
        ]
        g = pytensor.function(
            vv,
            model.replace_rvs_by_values([dets[n] for n in want if n in dets]),
            on_unused_input="ignore",
        )
        vals = dict(zip([n for n in want if n in dets], g(*split(x))))
        for n, v in vals.items():
            print(
                f"   {n:28s} {np.array2string(np.atleast_1d(np.asarray(v, dtype=float)), precision=5, max_line_width=120)}"
            )
        return total, data, other

    x0 = pack(start)
    print(
        f"value vector: {len(elem)} elements; pinned: {[e for e in elem if any(e.startswith(p) for p in PINNED)]}"
    )
    report("seed (published)", x0)

    def optimize(x_start, vary, label):
        lp0 = lp_at(x_start)
        n = [0]
        t0 = time.time()

        def fun(xf):
            x = x_start.copy()
            x[vary] = xf
            n[0] += 1
            v = lp_at(x)
            return 100.0 - (v - lp0) if np.isfinite(v) else 1e6

        res = minimize(
            fun,
            x_start[vary],
            method="Powell",
            options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4},
        )
        x = x_start.copy()
        x[vary] = res.x
        print(
            f"\n{label}: {n[0]} evaluations in {time.time() - t0:.0f} s, lp {lp0:.2f} -> {lp_at(x):.2f}",
            flush=True,
        )
        return x

    pin_mask = np.array([any(e.startswith(p) for p in PINNED) for e in elem])
    x_pinned = optimize(x0, ~pin_mask, "PINNED planet, everything else free")
    report("pinned optimum", x_pinned)
    x_free = optimize(x0, np.ones(len(elem), bool), "FREE, from the seed")
    report("free optimum", x_free)

    idata = az.from_netcdf(
        os.path.join(HERE, "fitresults_auto", "OGLE-2017-BLG-0114_trace.nc")
    )
    lp = idata.sample_stats.lp.values
    ib = np.unravel_index(np.argmax(lp), lp.shape)
    post = idata.posterior
    missing = [n for n in vnames if n not in post.data_vars]
    if missing:
        print(f"trace lacks value vars {missing}; skipping the trace point")
        return
    xb = np.concatenate(
        [
            np.asarray(post[n].values[ib[0], ib[1]], dtype=float).reshape(-1)
            for n in vnames
        ]
    )
    print(
        f"\ntrace best draw chain {ib[0]} draw {ib[1]}: stored lp {lp[ib]:.2f}, recomputed {lp_at(xb):.2f}"
    )
    report("trace best draw", xb)
    x_trace = optimize(
        xb, np.ones(len(elem), bool), "FREE, from the trace's best draw"
    )
    report("trace-basin optimum", x_trace)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30000)
