"""Is the s <-> 1/s mirror present in the hot draws, hidden by HOT_LP_MARGIN?

The ladder run's hot-chain discovery found 2 clusters and neither is the
mirror.  One suspected cause is the near-viable cut: HOT_LP_MARGIN = 50 nats,
justified in ledger.py by mass ("a genuinely real mode more than 50 nats down
carries e^-50 of the mass and loses nothing by being ignored").  True for
WEIGHTS -- wrong for the requirement to REPORT explored-and-rejected modes:
the mirror was previously measured ~77 nats down, i.e. excluded by design.

This decodes the stored hot draws with the run's own whitening (never by
hand -- raw draws only decode correctly under the whitening they were sampled
with) and answers, directly:
  1. do hot draws exist on the mirror side of log_s at all, and at what lp?
  2. what does discover_hot_modes find with margin_nats = 500?
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

RUN = Path(
    "/home/jeastman/python/EXOZIPPy/examples/DC2018/events_ladderfix/128"
)


def main():
    import os

    os.chdir(RUN)
    from exozippy import whitening
    from exozippy.outputs import ledger
    from exozippy.samplers import _common
    from exozippy.system import System

    config = yaml.safe_load(open("DC2018_128.yaml"))
    params = yaml.safe_load(open("DC2018_128.params.yaml"))
    system = System(config, params)
    system.prepare()
    model = system.build_model()
    # load_whitening(system, path): the model's shared variables are reached
    # through the system's parameters, so no model argument.
    ok = whitening.load_whitening(
        system, "fitresults/DC2018_128_whitening.json"
    )
    print(f"whitening restored: {ok}", flush=True)

    hot = xr.open_dataset(
        "fitresults/DC2018_128_trace.nc", group="posterior_hot"
    )
    lp = np.asarray(hot["lp"]).ravel()
    n = lp.size

    # compile_conversions returns a TUPLE:
    # (raw_to_phys, raw_to_phys_batched, raw_var_names, out_var_names).
    # The batched function takes one array per raw variable, positionally in
    # raw_var_names order, and returns free_RVs + deterministics in
    # out_var_names order.  The hot group stores exactly the raw variables.
    _, batched, raw_var_names, out_var_names = _common.compile_conversions(
        model
    )
    missing = [v for v in raw_var_names if v not in hot.data_vars]
    if missing:
        raise RuntimeError(f"hot group lacks raw vars: {missing}")
    raws = {
        v: np.asarray(hot[v]).reshape((n,) + np.asarray(hot[v]).shape[2:])
        for v in raw_var_names
    }
    i_logs = next(i for i, v in enumerate(out_var_names) if v == "lens.log_s")
    # Chunked: 3.1M rows through the full deterministic graph at once is
    # several GB; 100k-row chunks keep it flat.
    log_s = np.empty(n)
    step = 100_000
    for a in range(0, n, step):
        b = min(a + step, n)
        outs = batched(*[raws[v][a:b] for v in raw_var_names])
        chunk = np.asarray(outs[i_logs])
        log_s[a:b] = chunk.reshape(b - a, -1)[:, 0]

    best = np.nanmax(lp)
    print(f"hot draws: {n}   best untempered lp = {best:.2f}", flush=True)
    main_c, mirror_c = -0.0104, -0.0640
    for margin in (50, 100, 200, 500, 1000, 5000):
        m = lp >= best - margin
        near_main = m & (np.abs(log_s - main_c) < 0.02)
        near_mirror = m & (np.abs(log_s - mirror_c) < 0.02)
        print(
            f"  margin {margin:5d}: viable={int(m.sum()):8d}  "
            f"near s=0.976: {int(near_main.sum()):8d}  "
            f"near s=0.863: {int(near_mirror.sum()):8d}  "
            f"(best lp there: "
            f"{np.nanmax(lp[near_mirror]) if near_mirror.any() else float('nan'):.1f})",
            flush=True,
        )

    # VALIDATION of the horizon + clustering-units fix, against a trace
    # where the right answer is known: discovery with the DERIVED horizon
    # (10 x T_max = 85,000 nats) must return the main basin AND the s <-> 1/s
    # mirror -- log_s ~ -0.064, delta lp ~ 779 -- which the shipped code
    # missed twice over (50-nat margin excluded its draws; hot-spread
    # standardization broke the clustering even when the margin admitted
    # them).
    print("\n=== discover_hot_modes, DERIVED horizon margin ===", flush=True)
    status = {}
    entries = ledger.discover_hot_modes(
        system, model, hot, seed_ledger=[], status=status
    )
    print(
        f"  status: margin={status.get('margin_nats')} "
        f"t_max={status.get('t_max')} n_viable={status.get('n_viable')} "
        f"n_clusters={status.get('n_clusters')}",
        flush=True,
    )
    raw_to_phys, _, raw_names_1, out_names_1 = _common.compile_conversions(
        model
    )
    want = ("lens.log_s", "lens.u_0", "planet.log_q")
    found_main = found_mirror = False
    best_lp = max(e.lp_max for e in entries) if entries else float("nan")
    for e in entries:
        outs = raw_to_phys(*[e.raw_point[v] for v in raw_names_1])
        phys = dict(zip(out_names_1, outs))
        ls = float(np.asarray(phys["lens.log_s"]).ravel()[0])
        print(
            f"  entry {e.seed_index} [{e.source}]: lp_max={e.lp_max:.2f} "
            f"(delta={best_lp - e.lp_max:.1f})  "
            f"laplace_logw={e.laplace_logw:.2f}  log_s={ls:+.4f}",
            flush=True,
        )
        for k in want[1:]:
            print(f"      {k} = {np.asarray(phys[k]).ravel()}", flush=True)
        if abs(ls - (-0.0104)) < 0.02:
            found_main = True
        if abs(ls - (-0.0640)) < 0.02:
            found_mirror = True
            d = best_lp - e.lp_max
            assert 600 < d < 950, (
                f"mirror found but delta lp = {d:.1f}, expected ~779"
            )

    print(
        f"\nVALIDATION: main={found_main}  mirror={found_mirror}", flush=True
    )
    assert found_main, "main basin not rediscovered"
    assert found_mirror, "MIRROR NOT FOUND -- the fix does not validate"
    print("VALIDATION PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
