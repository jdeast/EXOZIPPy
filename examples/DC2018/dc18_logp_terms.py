"""Split a model's total logp into its named terms at chosen posterior draws.

Why: on DC2018 062 the basin that took the run over is 7 nats WORSE in the
photometric likelihood and ~110 nats BETTER in everything else
(dc18_hogg_modes.py).  "Everything else" is priors + the SED + Mann/Torres +
transform Jacobians, and which of those is doing it matters -- a kinematics
prior that outvotes the light curve by 100 nats is a different bug from an
SED that does.

Prints, for every free RV, observed RV and Potential in the model, its logp
at each requested draw and the difference against a reference draw, sorted
by |difference|.  No sampling; one compiled function, evaluated per point.
"""

import argparse
import os
import sys

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--points",
        required=True,
        help="comma-separated chain:draw pairs; the first is the reference",
    )
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    import arviz as az
    import pytensor

    from exozippy.system import System

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

    # A raw draw is a coordinate in the WHITENED space, so it only decodes
    # under the whitening it was sampled with (whitening.md: "a recomputed
    # lp" is named as one of the things that break otherwise -- on 128 a
    # single dropped barrier entry moved a pinned draw's logp by 575 nats).
    # Restore, never re-measure: this is the decode path.
    from exozippy.whitening import restore_whitening_for_trace

    state = restore_whitening_for_trace(
        system, pre + "_whitening.json", pre + "_trace.nc"
    )
    print(f"whitening: {state}")

    idata = az.from_netcdf(pre + "_trace.nc")
    post = idata.posterior
    lp = idata.sample_stats.lp.values

    # --- named logp terms ------------------------------------------------
    names, terms = [], []
    for rv in model.free_RVs:
        names.append(f"prior   {rv.name}")
        terms.append(model.logp(vars=[rv], sum=True))
    for rv in model.observed_RVs:
        names.append(f"obs     {rv.name}")
        terms.append(model.logp(vars=[rv], sum=True))
    # A Potential's graph is written in terms of the model's RANDOM
    # variables.  Compiling it directly against model.value_vars does not
    # fail -- it silently evaluates the RVs as constants, so every point
    # gets the same number and the whole likelihood looks like it does not
    # vary (that is exactly what the first working run of this script
    # reported: 88 potentials with under 0.2 nats of spread between points,
    # and a 14,602 nat "residual").  replace_rvs_by_values puts them in
    # value space, where the compiled function can see the inputs.
    for pot in model.potentials:
        names.append(f"potent  {pot.name}")
        terms.append(model.replace_rvs_by_values([pot])[0].sum())
    # the Jacobians of the value transforms, as one lump: total - sum(parts)
    total = model.logp(sum=True)
    fn = pytensor.function(
        model.value_vars, terms + [total], on_unused_input="ignore"
    )

    pts = []
    for spec in args.points.split(","):
        c, d = (int(x) for x in spec.split(":"))
        pts.append((c, d))

    vv = [v.name for v in model.value_vars]
    missing = [n for n in vv if n not in post.data_vars]
    if missing:
        print(f"value vars absent from the trace: {missing}")
        sys.exit(1)

    # the trace stores a length-1 vector as a scalar and vice versa; the
    # compiled function is strict about ndim, so reshape to what the model
    # says each value variable is.
    ip = model.initial_point()
    shapes = {n: np.asarray(ip[n]).shape for n in vv}

    cols = {}
    bad = False
    for c, d in pts:
        vals = [
            np.asarray(post[n].values[c, d], dtype=float).reshape(shapes[n])
            for n in vv
        ]
        out = fn(*vals)
        cols[(c, d)] = np.array([float(np.sum(x)) for x in out])
        gap = lp[c, d] - cols[(c, d)][-1]
        flag = "" if abs(gap) < 1.0 else "   <<< RECOMPUTE DOES NOT MATCH"
        print(
            f"point chain {c:3d} draw {d:6d}: stored lp = {lp[c, d]:12.2f}, "
            f"recomputed = {cols[(c, d)][-1]:12.2f}, "
            f"diff = {gap:+8.2f}{flag}"
        )
        if abs(gap) >= 1.0:
            bad = True

    if bad:
        print(
            "\nREFUSING TO DECOMPOSE: the recomputed total does not reproduce"
            "\nthe stored lp, so the per-term numbers would be meaningless."
        )
        sys.exit(2)

    ref = pts[0]
    print(
        f"\nPER-TERM logp, and the difference against the reference "
        f"(chain {ref[0]} draw {ref[1]}).\n"
        f"A POSITIVE difference means that term PREFERS the other point.\n"
    )
    head = f"{'term':<44}" + "".join(f"{'c%d d%d' % p:>16}" for p in pts[1:])
    print(f"{'':<44}{'reference':>16}" + head[44:])
    order = np.argsort(
        -np.max(
            [np.abs(cols[p][:-1] - cols[ref][:-1]) for p in pts[1:]], axis=0
        )
    )
    shown = 0
    for i in order:
        deltas = [cols[p][i] - cols[ref][i] for p in pts[1:]]
        if max(abs(x) for x in deltas) < 0.05:
            continue
        if shown >= args.top:
            break
        shown += 1
        print(
            f"{names[i]:<44}{cols[ref][i]:16.2f}"
            + "".join(f"{x:+16.2f}" for x in deltas)
        )
    # GROUPED, because the per-term table is dominated by bookkeeping: each
    # sampled element contributes a `prior X_raw` (the whitened N(0,1)) and
    # a `logit_uniform_prior.X` potential that CANCELS it -- that pair is
    # the change of variables, not physics, and the two halves individually
    # reach thousands of nats while their sum is a rounding error.
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
    print("\nGROUPED (the pairs that cancel are separated out):")
    print(
        f"{'group':<44}{'reference':>16}"
        + "".join(f"{'c%d d%d' % q:>16}" for q in pts[1:])
    )
    for g, idx in groups.items():
        base = float(np.sum(cols[ref][idx]))
        print(
            f"{g:<44}{base:16.2f}"
            + "".join(
                f"{float(np.sum(cols[q][idx])) - base:+16.2f}" for q in pts[1:]
            )
        )
    print("\nEVERY physics prior / barrier term, largest difference first:")
    idx = groups["physics priors and barriers"]
    idx.sort(
        key=lambda i: -max(abs(cols[q][i] - cols[ref][i]) for q in pts[1:])
    )
    for i in idx:
        print(
            f"  {names[i]:<42}{cols[ref][i]:16.2f}"
            + "".join(f"{cols[q][i] - cols[ref][i]:+16.2f}" for q in pts[1:])
        )
    print("\nEVERY data-likelihood term:")
    for i in groups["data likelihood"]:
        print(
            f"  {names[i]:<42}{cols[ref][i]:16.2f}"
            + "".join(f"{cols[q][i] - cols[ref][i]:+16.2f}" for q in pts[1:])
        )

    part_ref = float(np.sum(cols[ref][:-1]))
    print(f"\n{'sum of named terms':<44}{part_ref:16.2f}", end="")
    for p in pts[1:]:
        print(f"{float(np.sum(cols[p][:-1])) - part_ref:+16.2f}", end="")
    print(f"\n{'model.logp total':<44}{cols[ref][-1]:16.2f}", end="")
    for p in pts[1:]:
        print(f"{cols[p][-1] - cols[ref][-1]:+16.2f}", end="")
    print(
        f"\n{'residual (unaccounted -- must be ~0)':<44}"
        f"{cols[ref][-1] - part_ref:16.2f}",
        end="",
    )
    for p in pts[1:]:
        print(
            f"{(cols[p][-1] - float(np.sum(cols[p][:-1]))) - (cols[ref][-1] - part_ref):+16.2f}",
            end="",
        )
    print()


if __name__ == "__main__":
    main()
