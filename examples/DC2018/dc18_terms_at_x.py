"""Term-by-term logp of a run's model at saved profile optima (the npz files
dc18_ds_profile.py writes), differenced against the first.

Why: dc18_ds_profile.py prints the group totals and the DATA terms at each
grid point, but a run with one grid point per invocation (two pinned
quantities, as the 062 geometry profile needed) never gets its prior terms
listed side by side.  This reloads the saved optimum vectors into the
model's value space (whitening restored, as the profiler did) and runs the
acceptance recorder's decomposition on each, so "which priors carry the 66
nats" is a table rather than a guess.
"""

import argparse
import importlib.util
import os

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "npz", nargs="+", help="profile optima; the first is the reference"
    )
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    from exozippy.system import System
    from exozippy.whitening import restore_whitening_for_trace

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "mmf",
        os.path.join(here, "..", "..", "scripts", "make_mulens_fixtures.py"),
    )
    mmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mmf)

    npz_paths = [
        os.path.abspath(f) for f in args.npz
    ]  # before the chdir below
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

    vv = list(model.value_vars)
    ip = model.initial_point()
    shapes = [np.asarray(ip[v.name]).shape for v in vv]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    offsets = np.cumsum([0] + sizes)

    def as_start(x):
        return {
            v.name: np.asarray(
                x[offsets[i] : offsets[i + 1]], dtype=float
            ).reshape(shapes[i])
            for i, v in enumerate(vv)
        }

    import pytensor

    det_names = [d.name for d in model.deterministics]
    f_dets = pytensor.function(
        vv,
        model.replace_rvs_by_values(list(model.deterministics)),
        on_unused_input="ignore",
    )

    def alpha_radius(start):
        """hypot(xalpha, yalpha) per lens element, or None without a lens angle.

        The pair's N(0, 1) priors are flat in alpha and Gaussian in this
        radius, which nothing else reads; an optimizer leaves it anywhere,
        so point comparisons are made at r = 1 (add sum (r^2 - 1)/2).
        """
        if "lens.xalpha" not in det_names:
            return None
        out = dict(zip(det_names, f_dets(*[start[v.name] for v in vv])))
        return np.hypot(
            np.atleast_1d(out["lens.xalpha"]),
            np.atleast_1d(out["lens.yalpha"]),
        )

    results = []
    for f in npz_paths:
        z = np.load(f, allow_pickle=True)
        start = as_start(z["x"])
        parts, total, ok, summed = mmf.decompose(system, model, start)
        label = ", ".join(str(p) for p in z["pins"])
        rr = alpha_radius(start)
        corr = float(np.sum((rr**2 - 1.0) / 2.0)) if rr is not None else 0.0
        print(
            f"{os.path.basename(f)}: logp {total:.3f} (saved {float(z['logp']):.3f}; "
            f"decomposition {'reconciles' if ok else 'DOES NOT reconcile: ' + str(summed)})  "
            f"pins: {label}"
            + (
                f"  alpha radius r = {' '.join(f'{v:.2f}' for v in rr)}; "
                f"total at r = 1: {total + corr:.3f}"
                if rr is not None
                else ""
            )
        )
        results.append((label, parts, total, corr))

    # Collapse each sampled element's cancelling pair -- `RV:<x>_raw` (the
    # prior on the raw coordinate) and `POT:logit_uniform_prior.<x>` (its
    # change of variables) -- into one term, as the 2026-09-22 note
    # requires: each half reaches thousands of nats and their sum is the
    # physics.  Everything else is left alone.
    def collapse(parts):
        out = dict(parts)
        for k in list(parts):
            if k.startswith("RV:") and k.endswith("_raw"):
                base = k[3:-4]
                mate = f"POT:logit_uniform_prior.{base}"
                if mate in out:
                    out[f"CoV:{base}"] = out.pop(k) + out.pop(mate)
        return out

    import json

    with open(
        os.path.join(run_dir, f"DC2018_{args.event}_terms_at_x.json"), "w"
    ) as fh:
        json.dump(
            [
                {
                    "pins": l,
                    "total": tot,
                    "total_at_unit_alpha_radius": tot + c,
                    "terms": pr,
                }
                for l, pr, tot, c in results
            ],
            fh,
            indent=1,
        )
    results = [(l, collapse(pr), tot, c) for l, pr, tot, c in results]
    ref_label, ref, ref_total, ref_corr = results[0]
    for label, parts, total, corr in results[1:]:
        print(
            f"\n=== {label}  minus  {ref_label}:  total {total - ref_total:+.2f}"
            f"   | at unit alpha radius: "
            f"{(total + corr) - (ref_total + ref_corr):+.2f}"
        )
        rows = sorted(
            (
                (parts.get(k, 0.0) - ref.get(k, 0.0), k)
                for k in set(parts) | set(ref)
            ),
            key=lambda r: -abs(r[0]),
        )
        shown = 0
        for d, k in rows:
            if abs(d) < 0.05 and shown >= 10:
                break
            print(
                f"  {d:+10.2f}  {k}   ({ref.get(k, float('nan')):.2f} -> {parts.get(k, float('nan')):.2f})"
            )
            shown += 1
            if shown >= args.top:
                break


if __name__ == "__main__":
    main()
