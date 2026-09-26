"""Profile the model logp along the source distance, term by term.

Why: on DC2018 194 the source lands at ~5 kpc against a truth of 9.98, and
R_source follows it down (R = theta_star * D).  The earlier dissection
compared INDIVIDUAL posterior draws (dc18_logp_terms.py), and a draw's
kinematic term carries several nats of random velocity scatter -- two stars,
three Gaussian components each -- so "the kinematic prior costs 24.5 nats at
the true distance" and "it is not a distance penalty" were both read off
noise.  A profile removes the noise: pin D_source on a grid, re-OPTIMIZE
every other parameter, and split the optimum's logp into named terms.  The
slope of each term across the grid is the pull, and the slope of the total
is how strong it is.

Mechanics (all generic -- nothing here knows what a source is):

  * the model is rebuilt from the run's own config and its whitening is
    RESTORED from <prefix>_whitening.json (whitening.md: a raw draw is a
    coordinate in whitened space and only decodes under the whitening it
    was sampled with).  The start draw's stored lp must be reproduced
    before anything is reported.
  * a pin is "<deterministic>[<index>]=<value>".  The controlling value
    variable is found by differentiating the deterministic with respect to
    every value variable (deterministics have gradients even when the
    likelihood does not); the pin is honoured EXACTLY by removing that one
    element from the optimization vector and root-solving it to the target
    (brentq).  If more than one element controls the deterministic the
    script says so and stops -- a penalty pin would be silently soft.
  * THE OPTIMIZER IS GRADIENT-FREE, BECAUSE THE LIKELIHOOD IS.  The
    microlensing magnification Op (VBMDirectMagOp) has no gradient -- that is
    why the production sampler is PTDE -- so L-BFGS on the total logp is
    impossible (the first version of this script died on exactly that).
    Powell's method is used instead, over the value-vector elements whose
    names match --vary (default: the star, event, SED and Mann parameters);
    everything else -- the light-curve shape, fluxes and noise parameters --
    stays at the start draw.  Those are pinned by ~40,000 photometric points
    and are nearly orthogonal to the source distance, so freezing them costs
    well under a nat; the FULL logp, light curve included, is still what is
    optimized, so anything that does couple (t_E through theta_E/mu_rel,
    rho through theta_star) is handled, not assumed away.
  * each grid point is optimized from several posterior draws: the nearest
    ones in the pinned quantity with the highest lp, plus the run's best-lp
    draw.  The best optimum is reported; the spread across restarts is
    printed so a multimodal profile is visible rather than hidden.

Output per grid point: total logp, the three groups (data likelihood /
physics priors and barriers / whitening bookkeeping), every physics term
that moves by more than 0.05 nats across the grid, and the physical values
of the deterministics named in --report at the optimum.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import yaml


def _parse_pin(spec):
    lhs, rhs = spec.split("=")
    lhs = lhs.strip()
    idx = None
    if lhs.endswith("]"):
        lhs, i = lhs[:-1].split("[")
        idx = int(i)
    return lhs.strip(), idx, float(rhs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--pin",
        required=True,
        help="deterministic to pin, e.g. 'star.distance[1]=9980'; the grid "
        "(if given) replaces this pin's value",
    )
    ap.add_argument(
        "--grid",
        default=None,
        help="comma-separated values for the pinned quantity",
    )
    ap.add_argument(
        "--extra-pin",
        action="append",
        default=[],
        help="additional fixed pins, same syntax (repeatable)",
    )
    ap.add_argument(
        "--vary",
        default=r"^(star|mulensevent|mann|sed)\.",
        help="regex on value-vector element names to optimize; the rest stay "
        "at the start draw",
    )
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--neighbours", type=int, default=300)
    ap.add_argument("--maxfev", type=int, default=20000)
    ap.add_argument(
        "--report",
        default="star.distance,star.radius,star.teff,star.av,star.feh,"
        "star.logmass,star.pm_ra,star.pm_dec,star.rv,mulensevent.log_theta_E,"
        "mulensevent.log_pi_rel,mulensevent.mu_ra_rel,mulensevent.mu_dec_rel,"
        "mulensevent.t_E,source.rho,mulensinstrument.q_source",
        help="comma-separated deterministic names to print at each optimum",
    )
    args = ap.parse_args()

    import arviz as az
    import pytensor
    from scipy.optimize import brentq, minimize

    from exozippy.system import System
    from exozippy.whitening import restore_whitening_for_trace

    t0 = time.time()
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
    state = restore_whitening_for_trace(
        system, pre + "_whitening.json", pre + "_trace.nc"
    )
    print(f"whitening: {state}")
    print(f"model built in {time.time() - t0:.0f} s", flush=True)

    idata = az.from_netcdf(pre + "_trace.nc")
    post = idata.posterior
    lp = idata.sample_stats.lp.values
    nchain, ndraw = lp.shape

    # --- value vector bookkeeping ---------------------------------------
    vv = list(model.value_vars)
    vnames = [v.name for v in vv]
    missing = [n for n in vnames if n not in post.data_vars]
    if missing:
        print(f"value vars absent from the trace: {missing}")
        sys.exit(1)
    ip = model.initial_point()
    shapes = [np.asarray(ip[n]).shape for n in vnames]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    offsets = np.cumsum([0] + sizes)
    ntot = int(offsets[-1])
    elem_names = []
    for n, s, k in zip(vnames, shapes, sizes):
        if k == 1 and s == ():
            elem_names.append(n)
        else:
            elem_names.extend(f"{n}[{i}]" for i in range(k))

    def split(x):
        return [
            np.asarray(x[offsets[i] : offsets[i + 1]], dtype=float).reshape(
                shapes[i]
            )
            for i in range(len(vv))
        ]

    def draw_vector(c, d):
        return np.concatenate(
            [
                np.asarray(post[n].values[c, d], dtype=float).reshape(-1)
                for n in vnames
            ]
        )

    # --- compiled functions ----------------------------------------------
    names, terms = [], []
    for rv in model.free_RVs:
        names.append(f"prior   {rv.name}")
        terms.append(model.logp(vars=[rv], sum=True))
    for rv in model.observed_RVs:
        names.append(f"obs     {rv.name}")
        terms.append(model.logp(vars=[rv], sum=True))
    for pot in model.potentials:
        names.append(f"potent  {pot.name}")
        terms.append(model.replace_rvs_by_values([pot])[0].sum())
    total = model.logp(sum=True)
    f_terms = pytensor.function(vv, terms + [total], on_unused_input="ignore")
    f_lp = pytensor.function(vv, total, on_unused_input="ignore")

    det_names = [d.name for d in model.deterministics]
    det_vals = model.replace_rvs_by_values(list(model.deterministics))
    f_dets = pytensor.function(vv, det_vals, on_unused_input="ignore")

    def dets_at(x):
        out = f_dets(*split(x))
        return {
            n: np.atleast_1d(np.asarray(o, dtype=float))
            for n, o in zip(det_names, out)
        }

    def lp_at(x):
        return float(f_lp(*split(x)))

    def terms_at(x):
        return np.array([float(np.sum(o)) for o in f_terms(*split(x))])

    # --- pins ------------------------------------------------------------
    pins = [_parse_pin(args.pin)] + [_parse_pin(s) for s in args.extra_pin]
    for nm, idx, _ in pins:
        if nm not in det_names:
            cands = [d for d in det_names if nm.split(".")[-1] in d]
            print(
                f"pin target {nm!r} is not a deterministic; candidates: {cands}"
            )
            sys.exit(1)

    pin_grad_fns = []
    for nm, idx, _ in pins:
        d = model.replace_rvs_by_values([model[nm]])[0]
        scalar = d if idx is None else d[idx]
        g = pytensor.grad(scalar, vv, disconnected_inputs="ignore")
        pin_grad_fns.append(pytensor.function(vv, g, on_unused_input="ignore"))

    def pinned_value(x, nm, idx):
        v = dets_at(x)[nm]
        return float(v[0] if idx is None else v[idx])

    ib = np.unravel_index(np.argmax(lp), lp.shape)
    x_best = draw_vector(*ib)
    t1 = time.time()
    rec = lp_at(x_best)
    dt_eval = time.time() - t1
    print(
        f"best draw chain {ib[0]} draw {ib[1]}: stored lp {lp[ib]:.2f}, "
        f"recomputed {rec:.2f}, diff {lp[ib] - rec:+.2f}   "
        f"(one logp evaluation: {dt_eval * 1e3:.0f} ms)"
    )
    if abs(lp[ib] - rec) >= 1.0:
        print("REFUSING: recomputed lp does not reproduce the stored lp.")
        sys.exit(2)

    knobs = []
    for (nm, idx, _), gfn in zip(pins, pin_grad_fns):
        g = np.concatenate(
            [
                np.asarray(a, dtype=float).reshape(-1)
                for a in gfn(*split(x_best))
            ]
        )
        nz = np.flatnonzero(np.abs(g) > 1e-12 * max(1.0, np.abs(g).max()))
        label = nm if idx is None else f"{nm}[{idx}]"
        print(
            f"pin {label}: controlled by "
            + ", ".join(f"{elem_names[i]} (d/dx={g[i]:.3g})" for i in nz)
        )
        if len(nz) != 1:
            print(
                "REFUSING: a pin must map to exactly ONE value-vector element."
            )
            sys.exit(3)
        knobs.append(int(nz[0]))

    vary = np.array([bool(re.search(args.vary, n)) for n in elem_names])
    vary[knobs] = False
    frozen = [n for n, v in zip(elem_names, vary) if not v]
    print(
        f"\noptimizing {int(vary.sum())} of {ntot} value-vector elements; "
        f"frozen at the start draw: {frozen}\n",
        flush=True,
    )

    def apply_pins(x, targets):
        """Root-solve each knob so its deterministic hits its target."""
        x = x.copy()
        for _round in range(6):
            worst = 0.0
            for (nm, idx, _), k, tgt in zip(pins, knobs, targets):

                def resid(val):
                    y = x.copy()
                    y[k] = val
                    return pinned_value(y, nm, idx) - tgt

                x0 = x[k]
                if abs(resid(x0)) <= 1e-6 * max(1.0, abs(tgt)):
                    continue
                step = 0.5
                lo, hi = x0 - step, x0 + step
                for _ in range(60):
                    if resid(lo) * resid(hi) < 0:
                        break
                    step *= 1.6
                    lo, hi = x0 - step, x0 + step
                else:
                    raise RuntimeError(f"could not bracket pin {nm} -> {tgt}")
                x[k] = brentq(resid, lo, hi, xtol=1e-10, maxiter=200)
                worst = max(worst, abs(resid(x[k])))
            if worst <= 1e-6 * max(1.0, abs(tgt)):
                break
        return x

    def optimize(x_start, targets):
        x_start = apply_pins(x_start, targets)
        lp0 = lp_at(x_start)
        n_eval = [0]

        def fun(xf):
            x = x_start.copy()
            x[vary] = xf
            n_eval[0] += 1
            val = f_lp(*split(x))
            if not np.isfinite(val):
                return 1e6
            # Powell's ftol is RELATIVE to |f|, so put the objective on a
            # known scale: 100 at the start, improvements bring it down.
            # ftol = 1e-4 then stops at ~0.01 nats.
            return 100.0 - (float(val) - lp0)

        res = minimize(
            fun,
            x_start[vary],
            method="Powell",
            options={"maxfev": args.maxfev, "xtol": 1e-3, "ftol": 1e-4},
        )
        x = x_start.copy()
        x[vary] = res.x
        res.fun = res.fun - 100.0
        for (nm, idx, _), tgt in zip(pins, targets):
            got = pinned_value(x, nm, idx)
            if abs(got - tgt) > 1e-4 * max(1.0, abs(tgt)):
                print(f"   WARNING pin {nm} drifted: {got} vs {tgt}")
        return x, lp0 - res.fun, n_eval[0], lp0

    # --- grid -------------------------------------------------------------
    nm0, idx0, v0 = pins[0]
    grid = (
        [v0] if args.grid is None else [float(s) for s in args.grid.split(",")]
    )
    fixed_targets = [p[2] for p in pins[1:]]

    pinned_post = (
        post[nm0].values if idx0 is None else post[nm0].values[..., idx0]
    )
    flat_lp = lp.reshape(-1)
    flat_pin = pinned_post.reshape(-1)

    results = []
    for tgt in grid:
        t1 = time.time()
        order = np.argsort(np.abs(flat_pin - tgt))[: args.neighbours]
        cand = order[np.argsort(-flat_lp[order])][: args.restarts]
        starts = [np.unravel_index(int(i), lp.shape) for i in cand] + [ib]
        best = None
        for c, d in starts:
            try:
                x, val, nev, lp0 = optimize(
                    draw_vector(c, d), [tgt] + fixed_targets
                )
            except RuntimeError as e:
                print(f"   start chain {c} draw {d}: {e}")
                continue
            print(
                f"   {nm0}={tgt:g}: start chain {c:3d} draw {d:4d} "
                f"(pin there {flat_pin[c * ndraw + d]:.0f}, lp {lp[c, d]:.2f}, "
                f"after pinning {lp0:.2f}) -> optimum {val:.2f} "
                f"after {nev} evaluations",
                flush=True,
            )
            if best is None or val > best[1]:
                best = (x, val)
        if best is None:
            print(f"   no optimum for {nm0}={tgt:g}; skipped")
            continue
        x, val = best
        results.append((tgt, x, val, terms_at(x), dets_at(x)))
        # keep the optimum's value vector: a later evaluation of any term at
        # exactly this point must not depend on re-finding it
        tag = re.sub(r"[^A-Za-z0-9]+", "_", args.pin.split("=")[0])
        np.savez(
            f"{pre}_dsprofile_{tag}_{tgt:g}.npz",
            x=x,
            elem_names=np.array(elem_names),
            logp=val,
            pins=np.array(
                [
                    f"{p[0]}[{p[1]}]={t_}"
                    for p, t_ in zip(pins, [tgt] + fixed_targets)
                ]
            ),
        )
        print(f"   grid point done in {time.time() - t1:.0f} s\n", flush=True)

    # --- report -----------------------------------------------------------
    groups = {
        "whitening bookkeeping": [],
        "data likelihood": [],
        "physics priors and barriers": [],
    }
    for i, nm in enumerate(names):
        body = nm.split(None, 1)[1]
        if body.endswith("_raw") or body.startswith("logit_uniform_prior."):
            groups["whitening bookkeeping"].append(i)
        elif ".model." in body:
            groups["data likelihood"].append(i)
        else:
            groups["physics priors and barriers"].append(i)

    label0 = nm0 if idx0 is None else f"{nm0}[{idx0}]"
    print(
        f"\nPROFILE of the optimized logp along {label0}  "
        f"(event {args.event}, {run_dir})"
    )
    if fixed_targets:
        print(
            "   with fixed pins: "
            + ", ".join(f"{p[0]}[{p[1]}]={p[2]}" for p in pins[1:])
        )
    ref = results[0]
    print(f"\n{'':<44}" + "".join(f"{r[0]:>12g}" for r in results))
    print(f"{'TOTAL logp':<44}" + "".join(f"{r[2]:12.2f}" for r in results))
    print(
        f"{'  minus reference (first grid point)':<44}"
        + "".join(f"{r[2] - ref[2]:+12.2f}" for r in results)
    )
    # The first column is the ABSOLUTE value at the first grid point, the
    # rest are differences against it -- so a single-point run (e.g. a
    # truth configuration) can be set against another run's numbers.
    print(
        "\nGROUPS (absolute at the first grid point, then differences "
        "against it):"
    )
    for g, idx in groups.items():
        base = float(np.sum(ref[3][idx]))
        print(
            f"{g:<44}{base:12.2f}"
            + "".join(
                f"{float(np.sum(r[3][idx])) - base:+12.2f}"
                for r in results[1:]
            )
        )
    print(
        "\nEVERY physics prior / barrier term that moves by > 0.05 nats "
        "across the grid, or is more than 0.5 nats from zero:"
    )
    idx = groups["physics priors and barriers"]
    idx.sort(
        key=lambda i: -max(np.ptp([r[3][i] for r in results]), abs(ref[3][i]))
    )
    for i in idx:
        if np.ptp([r[3][i] for r in results]) < 0.05 and abs(ref[3][i]) < 0.5:
            continue
        print(
            f"  {names[i]:<42}{ref[3][i]:12.2f}"
            + "".join(f"{r[3][i] - ref[3][i]:+12.2f}" for r in results[1:])
        )
    print("\nDATA terms (absolute, then differences):")
    for i in groups["data likelihood"]:
        print(
            f"  {names[i]:<42}{ref[3][i]:12.2f}"
            + "".join(f"{r[3][i] - ref[3][i]:+12.2f}" for r in results[1:])
        )

    print("\nPHYSICAL VALUES at each optimum:")
    for nm in args.report.split(","):
        nm = nm.strip()
        if nm not in det_names:
            print(f"  {nm:<42} (not a deterministic here)")
            continue
        n_el = len(ref[4][nm])
        for j in range(n_el):
            lab = nm if n_el == 1 else f"{nm}[{j}]"
            print(
                f"  {lab:<42}"
                + "".join(f"{r[4][nm][j]:12.5g}" for r in results)
            )
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
