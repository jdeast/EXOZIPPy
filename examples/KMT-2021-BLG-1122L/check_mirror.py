"""Is the triple-lens light curve mirror-symmetric?  Without parallax,
(u_0, alpha_j) -> (-u_0, -alpha_j) for EVERY companion is an exact
reflection of the lens plane (conventions.md C23), so the model
magnification must be identical point for point.  The 2026-10-06
convention scan found the winner (180-a, -u_0, alpha_2 + psi, COM shift)
at chi2 ~ 800 while its exact mirror (a+180, +u_0, alpha_2 - psi, COM
shift) sat 9.5e6 higher -- impossible if the N >= 2 companion layout
respects the reflection.  This builds both, plus the same pair with the
third body removed (the binary path, validated against MulensModel), and
compares the model magnification on the data epochs.
"""

import copy
import importlib.util
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def build(cfg, prm):
    from exozippy.system import System

    system = System(copy.deepcopy(cfg), user_params=copy.deepcopy(prm))
    system.prepare()
    model = system.build_model()
    return system, model


def magnification(system, model, mmf):
    import pytensor

    start = mmf.raw_start(system, model)
    dets = {d.name: d for d in model.deterministics}
    names = [
        n
        for n in dets
        if "mulensinstrument" in n
        and (
            "magnification" in n
            or n.endswith(".model")
            or "model_mag" in n
            or "A_" in n
        )
    ]
    if not names:
        names = [n for n in dets if "mulensinstrument" in n][:6]
    f = pytensor.function(
        list(model.value_vars),
        model.replace_rvs_by_values([dets[n] for n in names]),
        on_unused_input="ignore",
    )
    vals = f(*[np.asarray(start[v.name]) for v in model.value_vars])
    parts, total, ok, _ = mmf.decompose(system, model, start)
    data = sum(
        v for k, v in parts.items() if "mulensinstrument" in k and "model" in k
    )
    return dict(
        zip(names, [np.atleast_1d(np.asarray(v)) for v in vals])
    ), float(data)


def main():
    spec = importlib.util.spec_from_file_location(
        "mmf",
        os.path.join(HERE, "..", "..", "scripts", "make_mulens_fixtures.py"),
    )
    mmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mmf)
    os.chdir(HERE)
    cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    prm = yaml.safe_load(open("KMT-2021-BLG-1122L.params.yaml"))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        cfg.pop(k, None)

    a2, a3, u0, t0 = (
        48.68,
        127.92,
        -0.4213,
        2459370.609,
    )  # the scan's winner (COM shift applied)

    def params(a2_, a3_, u0_, with_c=True):
        p = copy.deepcopy(prm)
        p["source.Source.t_0"] = {"initval": t0}
        p["source.Source.u_0"] = {"initval": u0_}
        p["lens.LensB.alpha"] = {"initval": a2_}
        if with_c:
            p.pop("lens.LensC.alpha", None)
            p.pop("lens.LensC.s", None)
            p["lens.LensC.log_s"] = {"initval": float(np.log10(1.601))}
            p["lens.LensC.xalpha"] = {
                "initval": float(np.cos(np.radians(a3_)))
            }
            p["lens.LensC.yalpha"] = {
                "initval": float(np.sin(np.radians(a3_)))
            }
        else:
            for k in list(p):
                if "LensC" in k:
                    p.pop(k)
        return p

    def config(with_c=True):
        c = copy.deepcopy(cfg)
        if not with_c:
            c["star"] = [s for s in c["star"] if s["name"] != "LensC"]
            c["lens"] = [b for b in c["lens"] if b["body"] != "star.LensC"]
        return c

    for with_c, label in (
        (True, "TRIPLE (LensB + LensC)"),
        (False, "BINARY control (LensC removed)"),
    ):
        print(f"\n=== {label}")
        out = {}
        for tag, (A2, A3, U0) in (
            ("winner", (a2, a3, u0)),
            ("mirror", (-a2, -a3, -u0)),
        ):
            try:
                s, m = build(config(with_c), params(A2, A3, U0, with_c))
                mags, dlogp = magnification(s, m, mmf)
                out[tag] = (mags, dlogp)
                print(
                    f"  {tag:7s} alpha2 {A2:+8.2f} alpha3 {A3:+8.2f} u0 {U0:+.4f}: data logp {dlogp:12.2f}; deterministics {list(mags)[:4]}"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  {tag:7s} FAILED {type(exc).__name__}: {str(exc)[:200]}"
                )
        if len(out) == 2:
            (mw, dw), (mm, dm) = out["winner"], out["mirror"]
            print(
                f"  data logp winner - mirror = {dw - dm:+.2f}  (chi2 difference {2 * (dm - dw):+.1f})"
            )
            for n in mw:
                if n in mm and mw[n].shape == mm[n].shape:
                    d = np.abs(mw[n] - mm[n])
                    print(
                        f"  {n}: max |diff| {d.max():.3e}, median {np.median(d):.3e}, n {d.size}"
                    )


if __name__ == "__main__":
    main()
