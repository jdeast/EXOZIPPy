"""The PEAK-TO-VALLEY barrier between two modes, which is not what the
mode report's "delta vs best seed" measures.

JDE, 2026-09-23: "we don't want some arbitrary difference between the
polished seed and the starting point, we need the peak to valley barrier
height between two legit modes."  Exactly right, and the distribution I
quoted (median 104 nats over 279 candidates) was peak-to-PEAK -- the lp
difference between two optima.  That number says nothing about what a
tempered chain has to climb: two modes 1 nat apart can be separated by a
1000-nat valley, and the valley is what sets the crossing probability at
temperature T (a barrier B is crossable at the top rung when B/T_max ~ 1).

This walks a straight line between two draws IN THE MODEL'S RAW
(whitened) COORDINATES and evaluates the model's own logp along it, so
the profile is the thing the sampler actually sees.

WHAT IT MEASURES AND WHAT IT DOES NOT.  A straight line is one path, not
the best one, so the valley it finds is an UPPER BOUND on the barrier:
the minimum-energy path between the same two modes can only be lower.
For sizing T_max that is the conservative direction (it can overstate how
hot you need to be) and it can be very loose when the two modes differ in
a discrete-like way -- s vs 1/s, say, where the straight line passes
through geometries neither mode resembles.  Read it as "no worse than
this", and use a string/NEB method if a tight number is ever needed.

The whitening must be restored before any of this means anything
(whitening.md; a raw draw only decodes under the whitening it was sampled
with), and the script refuses to report if the endpoints' recomputed lp
does not reproduce the trace's own.
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
        help="two chain:draw pairs, the two modes' representatives",
    )
    ap.add_argument("--steps", type=int, default=41)
    args = ap.parse_args()

    import arviz as az
    import pytensor

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
        f"whitening: {restore_whitening_for_trace(system, pre + '_whitening.json', pre + '_trace.nc')}"
    )

    idata = az.from_netcdf(pre + "_trace.nc")
    post, lp = idata.posterior, idata.sample_stats.lp.values
    vv = [v.name for v in model.value_vars]
    ip = model.initial_point()
    shapes = {n: np.asarray(ip[n]).shape for n in vv}
    fn = pytensor.function(
        model.value_vars, model.logp(sum=True), on_unused_input="ignore"
    )

    ends = []
    for spec in args.points.split(","):
        c, d = (int(x) for x in spec.split(":"))
        vals = [
            np.asarray(post[n].values[c, d], dtype=float).reshape(shapes[n])
            for n in vv
        ]
        got, want = float(fn(*vals)), float(lp[c, d])
        print(
            f"endpoint chain {c:3d} draw {d:6d}: stored lp {want:12.2f}, "
            f"recomputed {got:12.2f}, diff {want - got:+.2f}"
        )
        if abs(want - got) > 1.0:
            sys.exit("recomputed lp does not reproduce the trace's own")
        ends.append(vals)

    print(
        f"\nlogp along the straight line between them ({args.steps} points):"
    )
    ts = np.linspace(0.0, 1.0, args.steps)
    prof = []
    for t in ts:
        vals = [(1 - t) * a + t * b for a, b in zip(*ends)]
        prof.append(float(fn(*vals)))
    prof = np.array(prof)
    lo = int(np.argmin(prof))
    for i in range(0, args.steps, max(1, args.steps // 20)):
        bar = "#" * int(
            60 * (prof[i] - prof.min()) / max(float(np.ptp(prof)), 1e-9)
        )
        print(f"  t={ts[i]:4.2f}  lp={prof[i]:12.2f}  {bar}")
    print(f"\n  valley at t={ts[lo]:.2f}, lp={prof[lo]:.2f}")
    print(f"  BARRIER from endpoint A: {prof[0] - prof[lo]:10.1f} nats")
    print(f"  BARRIER from endpoint B: {prof[-1] - prof[lo]:10.1f} nats")
    print(
        f"  peak-to-peak (what the mode report calls delta): "
        f"{abs(prof[0] - prof[-1]):.1f} nats"
    )
    print(
        "\n  upper bounds: a straight line is one path, and the "
        "minimum-energy\n  path between the same modes can only be lower."
    )
    print(
        f"  T_max would need to be ~{(prof[0] - prof[lo]) / 1.0:.0f} to "
        f"flatten the larger of these to ~1 nat at the top rung."
    )


if __name__ == "__main__":
    main()
