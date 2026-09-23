"""Which functional form does each raw element's transform take?

The unit-cube bridge assumes physical = lower + span*sigmoid(c + s*raw) and
verifies by round-trip; on the event-128 build the verification refused
mulensinstrument.err_scale_raw[1] (rel err 2.1).  This prints the raw ->
physical map on a grid for the worst elements and tests two candidate forms:

    LINEAR-LOGIT: phys = lower + span * sigmoid(c + s*raw)
    LOG-LOGIT:    log(phys) = log(lower) + log(span_ratio) * sigmoid(c + s*raw)

If the second fits, the bridge grows a per-element form flag and the
prior_transform maps u -> log-uniform for those elements (which is also the
statement of what their PRIOR actually is).
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
    from exozippy.samplers._common import compile_conversions
    from exozippy.system import System

    config = yaml.safe_load(open("DC2018_128.yaml"))
    params = yaml.safe_load(open("DC2018_128.params.yaml"))
    system = System(config, params)
    system.prepare()
    model = system.build_model()
    raw_to_phys, _, raw_names, out_names = compile_conversions(model)
    ip = model.initial_point()

    def phys_dict(raw_flat, sizes):
        args, ofs = [], 0
        for v, n in zip(raw_names, sizes):
            args.append(
                np.asarray(raw_flat[ofs : ofs + n]).reshape(
                    np.asarray(ip[v]).shape
                )
            )
            ofs += n
        return dict(zip(out_names, raw_to_phys(*args)))

    sizes = [int(np.asarray(ip[v]).size) for v in raw_names]
    ndim = sum(sizes)
    grid = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])

    ofs = 0
    for v, n in zip(raw_names, sizes):
        base = v[:-4] if v.endswith("_raw") else v
        for j in range(n):
            vals = []
            for r in grid:
                z = np.zeros(ndim)
                z[ofs + j] = r
                d = phys_dict(z, sizes)
                # element map: which element of base moved vs origin
                p = np.asarray(d[base]).ravel()
                vals.append(p)
            vals = np.array(vals)  # (len(grid), n_phys)
            z0 = phys_dict(np.zeros(ndim), sizes)
            ref = np.asarray(z0[base]).ravel()
            moved = np.argmax(np.abs(vals - ref).max(axis=0))
            y = vals[:, moved]
            # test LINEAR-LOGIT: recover from r=0,1 then predict r=grid
            lo, hi = y.min(), y.max()

            def rms(form):
                # bounds from +/-40
                zlo = np.zeros(ndim)
                zlo[ofs + j] = -40
                zhi = np.zeros(ndim)
                zhi[ofs + j] = 40
                a = np.asarray(phys_dict(zlo, sizes)[base]).ravel()[moved]
                b = np.asarray(phys_dict(zhi, sizes)[base]).ravel()[moved]
                lo_, hi_ = min(a, b), max(a, b)
                if form == "log" and lo_ <= 0:
                    return np.inf
                if form == "lin":
                    t = (y - lo_) / (hi_ - lo_)
                else:
                    t = (np.log(y) - np.log(lo_)) / (np.log(hi_) - np.log(lo_))
                t = np.clip(t, 1e-12, 1 - 1e-12)
                L = np.log(t / (1 - t))
                # linear in r?
                A = np.vstack([grid, np.ones_like(grid)]).T
                coef, res, *_ = np.linalg.lstsq(A, L, rcond=None)
                pred = A @ coef
                return float(np.sqrt(np.mean((L - pred) ** 2)))

            r_lin, r_log = rms("lin"), rms("log")
            tag = "LIN" if r_lin < 1e-6 else ("LOG" if r_log < 1e-6 else "??")
            if tag != "LIN":
                print(
                    f"{v}[{j}] -> {base}[{moved}]: form={tag}  "
                    f"rms_lin={r_lin:.3g} rms_log={r_log:.3g}  "
                    f"y(grid)={np.array2string(y, precision=4)}",
                    flush=True,
                )
        ofs += n
    print("done (only non-LIN elements printed)", flush=True)


def coupling_probe():
    """Which OTHER raw element changes mulensinstrument.err_scale[1]?

    Single-element probes show every element is linear-logit to ~3e-4, yet
    the bridge round-trip at random u fails on err_scale[1] with rel err
    2.1 -- so its value must depend on at least one other raw element.
    Baseline: err_scale_raw[1] = 1, everything else 0.  Then kick each other
    element to 2 and see whether phys err_scale[1] moves.
    """
    import os

    os.chdir(RUN)
    from exozippy.samplers._common import compile_conversions
    from exozippy.system import System

    config = yaml.safe_load(open("DC2018_128.yaml"))
    params = yaml.safe_load(open("DC2018_128.params.yaml"))
    system = System(config, params)
    system.prepare()
    model = system.build_model()
    raw_to_phys, _, raw_names, out_names = compile_conversions(model)
    ip = model.initial_point()
    sizes = [int(np.asarray(ip[v]).size) for v in raw_names]
    ndim = sum(sizes)
    flat = []
    for v, n in zip(raw_names, sizes):
        flat += [f"{v}[{j}]" for j in range(n)]

    def es1(raw_flat):
        args, ofs = [], 0
        for v, n in zip(raw_names, sizes):
            args.append(
                np.asarray(raw_flat[ofs : ofs + n]).reshape(
                    np.asarray(ip[v]).shape
                )
            )
            ofs += n
        d = dict(zip(out_names, raw_to_phys(*args)))
        return float(np.asarray(d["mulensinstrument.err_scale"]).ravel()[1])

    k_es1 = flat.index("mulensinstrument.err_scale_raw[1]")
    base = np.zeros(ndim)
    base[k_es1] = 1.0
    ref = es1(base)
    print(
        f"\nbaseline err_scale[1] (own raw = 1, others 0): {ref:.6f}",
        flush=True,
    )
    for j in range(ndim):
        if j == k_es1:
            continue
        z = base.copy()
        z[j] = 2.0
        v = es1(z)
        if abs(v - ref) > 1e-9 * max(abs(ref), 1.0):
            print(
                f"  COUPLED: {flat[j]} = 2  ->  err_scale[1] = {v:.6f} "
                f"(moved {v - ref:+.6f})",
                flush=True,
            )
    print("coupling probe done", flush=True)


if __name__ == "__main__":
    main()
    coupling_probe()
    sys.exit(0)
