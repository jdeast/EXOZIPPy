"""Write a params file whose starts are the convention scan's optimum.

The winner of scan_conventions_opt.py (task 13, light-curve term 9424)
was optimized with its geometry pinned.  This rebuilds that exact model,
loads its optimum vector (scan_opt/x_13.npy), evaluates every parameter
the SHIPPED model samples at that point, and writes them as initvals into
a copy of the shipped params file (comments, priors, bounds and links
kept; only initvals change), so a fit starts its population inside the
basin instead of climbing ~10,000 nats into it.

    python seed_from_optimum.py <task> <out params file>
"""

import os
import sys

import numpy as np
import yaml
from ruamel.yaml import YAML

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scan_conventions_opt as S  # noqa: E402

# (component, parameter, elements) the shipped model samples; elements by name.
SAMPLED = [
    ("star", "mass", ["Lens", "LensB", "LensC"]),
    ("star", "distance", ["Lens", "Source"]),
    ("star", "pm_ra", ["Lens", "Source"]),
    ("star", "pm_dec", ["Lens", "Source"]),
    ("star", "rv", ["Lens", "Source"]),
    ("star", "radius", ["Source"]),
    ("star", "teff", ["Source"]),
    ("star", "av", ["Source"]),
    ("star", "radiussed", ["Source"]),
    ("star", "teffsed", ["Source"]),
    ("lens", "s", ["LensB"]),
    ("lens", "alpha", ["LensB"]),
    ("lens", "log_s", ["LensC"]),
    ("source", "t_0", ["Source"]),
    ("source", "u_0", ["Source"]),
    ("band", "u1", ["I", "V"]),
    ("mulensinstrument", "q_source", None),
    ("mulensinstrument", "err_scale", None),
    ("mulensinstrument", "log_f_total", None),
]


def main(task, out):
    import copy

    import profile_geometry as PG
    import pytensor

    from exozippy.system import System

    c = S.candidates()[task - 1]
    os.chdir(HERE)
    cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    prm = yaml.safe_load(open(cfg["parameter_file"]))
    p = copy.deepcopy(prm)
    for k in (
        "lens.LensB.s",
        "lens.LensB.alpha",
        "star.LensB.mass",
        "star.LensC.mass",
    ):
        p.pop(k, None)
    p.update(PG.geometry_pins(c["s2"], S.Q2, c["a2"], c["s3"], S.Q3, c["a3"]))
    p["source.Source.t_0"] = {"initval": float(c["t0"])}
    p["source.Source.u_0"] = {"initval": float(c["u0"])}
    # The optimum vector is RAW coordinates, meaningful only under the
    # whitening the scan's build measured at ITS start -- so rebuild with the
    # params exactly as they were then: q_source starts 0.0093 (I) / 0.0059
    # (V) and no err_scale starts (both seeded into the shipped file after
    # the scan; with them the vector decodes to lp 8247 instead of 9296).
    for inst, q in (
        ("KMTC14", 0.0093),
        ("KMTS14", 0.0093),
        ("KMTA14", 0.0093),
        ("KMTC14V", 0.0059),
        ("KMTS14V", 0.0059),
        ("KMTA14V", 0.0059),
    ):
        p[f"mulensinstrument.{inst}.q_source"] = {"initval": q}
        p.pop(f"mulensinstrument.{inst}.err_scale", None)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)
    system = System(cfg, user_params=p)
    system.prepare()
    model = system.build_model()
    vv = list(model.value_vars)
    x = np.load(os.path.join(HERE, "scan_opt", f"x_{task:02d}.npy"))
    shapes = [tuple(v.type.shape) for v in vv]
    sizes = [int(np.prod(s)) if s else 1 for s in shapes]
    offs = np.cumsum([0] + sizes)
    assert offs[-1] == x.size, (offs[-1], x.size)
    args = [
        x[offs[i] : offs[i + 1]].reshape(shapes[i]) for i in range(len(vv))
    ]
    lp = float(
        pytensor.function(vv, model.logp(sum=True), on_unused_input="ignore")(
            *args
        )
    )
    print(
        f"optimum lp re-evaluated: {lp:.2f} (scan reported 9295.65 for task 13)"
    )
    if abs(lp - 9295.65) > 1.0:
        raise SystemExit(
            "REFUSING: the rebuilt model does not reproduce the scan optimum lp, "
            "so the vector would decode to the wrong point"
        )

    seeds = {}
    for comp, par, elems in SAMPLED:
        C = getattr(system, comp)
        P = getattr(C, par)
        f = pytensor.function(
            vv,
            model.replace_rvs_by_values([P.value])[0],
            on_unused_input="ignore",
        )
        val = np.atleast_1d(np.asarray(f(*args), dtype=float))
        names = list(C.names)
        for e in elems if elems is not None else names:
            i = names.index(e)
            seeds[f"{comp}.{e}.{par}"] = float(
                np.asarray(P.from_internal(val[i], index=i))
            )
    # LensC's angle is sampled as (xalpha, yalpha): seed the unit vector.
    f = pytensor.function(
        vv,
        model.replace_rvs_by_values([system.lens.alpha.value])[0],
        on_unused_input="ignore",
    )
    a3 = float(np.atleast_1d(f(*args))[list(system.lens.names).index("LensC")])
    seeds["lens.LensC.xalpha"] = float(np.cos(a3))
    seeds["lens.LensC.yalpha"] = float(np.sin(a3))

    ry = YAML()
    ry.preserve_quotes = True
    doc = ry.load(open(os.path.join(HERE, "KMT-2021-BLG-1122L.params.yaml")))
    for k, v in seeds.items():
        cur = doc.get(k)
        if isinstance(cur, dict):
            if isinstance(cur.get("initval"), str):
                continue  # a link: never overwrite
            cur["initval"] = v
        elif cur is None:
            doc[k] = {"initval": v}
        else:
            doc[k] = {"initval": v}
        print(f"   {k:40s} {v:.8g}")
    doc.yaml_set_start_comment(
        "STARTS SEEDED FROM THE CONVENTION SCAN'S OPTIMUM (seed_from_optimum.py,\n"
        f"scan_conventions_opt.py task {task}, light-curve term 9424): every sampled\n"
        "parameter's initval is its value at that optimum; priors, bounds and\n"
        "links are the shipped file's.  Everything below is otherwise\n"
        "KMT-2021-BLG-1122L.params.yaml.\n"
    )
    ry.dump(doc, open(out, "w"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
