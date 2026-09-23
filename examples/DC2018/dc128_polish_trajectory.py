#!/usr/bin/env python3
"""How far does the DE seed polish actually get, and can it be stopped well?

Two questions in one run, both about examples/DC2018 event 128:

Q1  Would a LONG polish do what --mmx-emcee does?  The 150- and 1200-sweep
    endpoints alone cannot answer that: they say seed 1 closed its shortfall
    from ~5960 to ~3820 nats, which extrapolates anywhere between "a few
    thousand more sweeps" and "never" depending on the assumed law.  This
    logs the whole trajectory so the curve decides.  The number that matters
    is not seed 1's lp but whether it OVERTAKES seed 0 (~129354 once
    converged), because the ranking is what the sampler acts on.

Q2  Is there an honest stopping rule?  e8487d6 showed a best-lp improvement
    window cannot work here -- the plateaus are exactly flat, so no
    threshold separates "converged" from "has not jumped yet".  But a T=1
    DE-MC population carries a signal that does not depend on improvement
    history at all: at equilibrium in D dimensions the population's lp
    spread is set by the local curvature (best - median ~ D/2 nats), while
    a population still descending a valley is strung out along it with the
    best member far ahead of the pack.  This logs best - median per sweep so
    that criterion can be checked against the ground truth of the curve.

Run one seed per job (they are independent); the CSV is flushed every sweep
so a partial run is still usable.

    python dc128_polish_trajectory.py <seed_index> [n_sweeps]
"""

import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from exozippy.samplers.ptde import _pick_two
from exozippy.system import System

HERE = Path(__file__).resolve().parent
EVENT_DIR = HERE / "events" / "128"


def main():
    seed_index = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    out_csv = HERE / f"dc128_polish_traj_seed{seed_index}.csv"

    with open(EVENT_DIR / "DC2018_128.yaml") as f:
        config = yaml.safe_load(f)
    config.pop("parameter_file", None)
    config["sampler"]["recompute_trace"] = False
    with open(EVENT_DIR / "DC2018_128.params.yaml") as f:
        user_params = yaml.safe_load(f)

    os.chdir(tempfile.mkdtemp(prefix="dc128traj_"))
    system = System(config, user_params)
    system.prepare()
    model = system.build_model()
    logp_fn = model.compile_logp()

    raw_starts, seed_indices = system.get_raw_starts(model)
    print(f"seeds available: {seed_indices}", flush=True)
    center = raw_starts[seed_index]
    keys = list(center.keys())
    n_params = sum(np.asarray(v).size for v in center.values())

    # Replicates polish.polish_raw_starts + ptde.polish_seed_starts exactly:
    # unit scales in raw space, rng(0), pop = min(2*n_params, 64).
    pop_size = int(max(8, min(2 * n_params, 64)))
    gamma = 2.38 / np.sqrt(2 * max(n_params, 1))
    scales = {k: np.ones(np.shape(center[k]), dtype=float) for k in keys}
    rng = np.random.default_rng(0)
    print(
        f"n_params={n_params}  pop_size={pop_size}  gamma={gamma:.4f}",
        flush=True,
    )

    pop = [{k: np.array(v, dtype=float) for k, v in center.items()}]
    for _ in range(pop_size - 1):
        pop.append(
            {
                k: center[k]
                + scales[k] * rng.standard_normal(np.shape(center[k]))
                for k in keys
            }
        )
    lps = np.array([float(logp_fn(p)) for p in pop])
    for i in np.nonzero(~np.isfinite(lps))[0]:
        pop[i] = {k: np.array(v, dtype=float) for k, v in center.items()}
        lps[i] = lps[0]
    best_lp = float(np.nanmax(lps))
    lp0 = float(lps[0])
    print(f"seed {seed_index}: lp0={lp0:.1f}", flush=True)

    def spread():
        """RMS population spread in raw units, averaged over parameters."""
        tot = 0.0
        for k in keys:
            arr = np.array([p[k] for p in pop], dtype=float)
            tot += float(np.mean(np.std(arr, axis=0) ** 2))
        return float(np.sqrt(tot / len(keys)))

    t0 = time.time()
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "sweep",
                "best_lp",
                "median_lp",
                "best_minus_median",
                "accept",
                "pop_spread",
                "elapsed_s",
            ]
        )
        for sweep in range(1, n_sweeps + 1):
            n_acc = 0
            for i in range(pop_size):
                j1, j2 = _pick_two(rng, pop_size, i)
                prop = {
                    k: pop[i][k]
                    + gamma * (pop[j1][k] - pop[j2][k])
                    + 1e-4
                    * scales[k]
                    * rng.standard_normal(np.shape(pop[i][k]))
                    for k in keys
                }
                lp = float(logp_fn(prop))
                if np.isfinite(lp) and np.log(rng.random()) < lp - lps[i]:
                    pop[i], lps[i] = prop, lp
                    n_acc += 1
                    best_lp = max(best_lp, lp)
            med = float(np.nanmedian(lps))
            w.writerow(
                [
                    sweep,
                    f"{best_lp:.4f}",
                    f"{med:.4f}",
                    f"{best_lp - med:.4f}",
                    f"{n_acc / pop_size:.4f}",
                    f"{spread():.6g}",
                    f"{time.time() - t0:.1f}",
                ]
            )
            fh.flush()
            if sweep % 25 == 0:
                print(
                    f"  sweep {sweep:6d}  best={best_lp:12.1f}  "
                    f"best-med={best_lp - med:9.1f}  "
                    f"acc={n_acc / pop_size:.3f}  spread={spread():.4g}  "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )

    print(f"\nwrote {out_csv}")
    print(
        f"seed {seed_index}: lp {lp0:.1f} -> {best_lp:.1f} "
        f"(dlp=+{best_lp - lp0:.1f}) over {n_sweeps} sweeps"
    )


if __name__ == "__main__":
    main()
