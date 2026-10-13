"""Build the shipped KMT-2021-BLG-1122L model and report what is sampled,
the start log-probability and its term split, and the instrument flux
state -- a smoke check after a config change."""

import importlib.util
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import pytensor

    from exozippy.system import System

    spec = importlib.util.spec_from_file_location(
        "mmf",
        os.path.join(HERE, "..", "..", "scripts", "make_mulens_fixtures.py"),
    )
    mmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mmf)
    os.chdir(HERE)
    cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    cfg["parameter_file"] = os.environ.get(
        "PARAMS", cfg["parameter_file"]
    )  # a params-file override for checks
    prm = yaml.safe_load(open(cfg["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    s = System(cfg, user_params=prm)
    s.prepare()
    m = s.build_model()
    print(
        "free RVs:",
        sum(int(np.prod(v.type.shape) or 1) for v in m.value_vars),
        "elements",
    )
    for v in m.value_vars:
        print(f"   {v.name:40s} {tuple(v.type.shape)}")
    st = mmf.raw_start(s, m)
    parts, total, ok, _ = mmf.decompose(s, m, st)
    data = sum(
        float(v)
        for k, v in parts.items()
        if k.startswith("RV:") and "mulensinstrument" in k
    )
    print(f"start total {total:.2f}, data {data:.2f}")
    for k, v in sorted(parts.items(), key=lambda kv: -abs(float(kv[1])))[:12]:
        print(f"   {k:55s} {float(v):12.2f}")
    dets = {d.name: d for d in m.deterministics}
    want = [
        n
        for n in (
            "star.distance",
            "star.radius",
            "star.teff",
            "star.av",
            "mulensevent.t_E",
            "source.rho",
            "mulensinstrument.f_source",
            "mulensinstrument.f_blend",
            "mulensinstrument.q_source",
            "mulensinstrument.zeropoint",
        )
        if n in dets
    ]
    g = pytensor.function(
        m.value_vars,
        m.replace_rvs_by_values([dets[n] for n in want]),
        on_unused_input="ignore",
    )
    for n, v in zip(want, g(*[np.asarray(st[x.name]) for x in m.value_vars])):
        print(
            f"   {n:32s} {np.array2string(np.atleast_1d(np.asarray(v, dtype=float)), precision=4, max_line_width=150)}"
        )
    print("instruments:", list(s.mulensinstrument.names))


if __name__ == "__main__":
    main()
