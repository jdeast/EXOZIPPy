"""Dump the bridge round-trip failure in full numbers.

Single-element probes say every element is linear-logit to ~3e-4; the
pairwise coupling probe found NO partner for err_scale[1]; yet verify() at
random u fails with rel err 2.1 attributed to err_scale[1].  Three stories
have now failed inspection (log-logit form, dynamic support, pairwise
coupling), so this prints the actual comparison, element by element, for the
exact u that fails: u, recovered raw, got = phys(raw(u)), want = lo + span*u,
absolute and relative error -- plus the raw magnitudes, since a tiny
recovered scale s would send raw(u) far outside the region where the probes
measured the transform.
"""

import os
import sys
from pathlib import Path

import numpy as np
import yaml

RUN = Path(
    "/home/jeastman/python/EXOZIPPy/examples/DC2018/events_ladderfix/128"
)


def main():
    os.chdir(RUN)
    from exozippy.samplers.nested import UnitCubeBridge
    from exozippy.system import System

    config = yaml.safe_load(open("DC2018_128.yaml"))
    params = yaml.safe_load(open("DC2018_128.params.yaml"))
    system = System(config, params)
    system.prepare()
    model = system.build_model()
    b = UnitCubeBridge(model)

    rng = np.random.default_rng(1)
    u = rng.uniform(0.02, 0.98, b.ndim)  # the first verify draw
    raw = b.raw_from_u(u)
    got = b._phys_at(raw)
    want = b.lower + b.span * u
    abs_err = np.abs(got - want)
    rel_err = abs_err / np.maximum(np.abs(want), 1e-9)

    order = np.argsort(-rel_err)
    print(
        f"{'element':38s} {'u':>7s} {'raw(u)':>10s} {'c':>8s} {'s':>10s} "
        f"{'got':>12s} {'want':>12s} {'abs':>10s} {'rel':>10s}",
        flush=True,
    )
    for k in order[:12]:
        print(
            f"{b.flat_names[k]:38s} {u[k]:7.4f} {raw[k]:10.2f} "
            f"{b.c[k]:8.3f} {b.s[k]:10.5f} {got[k]:12.5g} {want[k]:12.5g} "
            f"{abs_err[k]:10.3g} {rel_err[k]:10.3g}",
            flush=True,
        )
    print(
        "\nraw magnitude summary: |raw| median "
        f"{np.median(np.abs(raw)):.2f}, max {np.abs(raw).max():.2f} at "
        f"{b.flat_names[int(np.argmax(np.abs(raw)))]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
