"""Start-state check for the triple-lens example: which mapping of Han et
al. 2023's (alpha, psi, u_0) onto EXOZIPPy's per-companion alpha_j, and
which origin, reproduces the published light curve?

conventions.md C21: a paper's alpha can differ from ours by +180 (measured
to the source's motion), by a reflection (opposite rotation sense), or
both; u_0 is never touched by those, but the C23 mirror (-u_0, -alpha) is
a separate exact degeneracy without parallax.  For a THIRD body the sense
of psi is a further unknown.  And the ORIGIN differs: Han+2023 centre
their coordinates "at the effective position of the M1-M2 pair" (An & Han
2002) -- the primary displaced toward M2 by q2 / ((1 + q2) s2) = 0.249
theta_E (their Fig. 4 inset puts M1 at x ~ -0.2) -- while EXOZIPPy's
origin is the lens CENTRE OF MASS, (q2 s2 e2 + q3 s3 e3) / (1 + q2 + q3)
from the primary.  The translation D = COM - effective is ~0.27 theta_E,
twice u_0, so (t_0, u_0) must be re-referenced: with the source at
(-tau, -u_0) in the trajectory frame, moving the origin by D gives
t_0' = t_0 - D.tau_hat t_E and u_0' = u_0 + D.beta_hat.

So evaluate all 32 combinations

    alpha_2  in {a + 180, 180 - a, a, -a}      (a = 131.31 deg, Han's alpha)
    u_0      in {+0.126, -0.126}
    alpha_3  in {alpha_2 - psi, alpha_2 + psi} (psi = 79.24 deg)
    origin   in {shift off, shift on}

on the shipped data at the published (s_2, s_3, masses, rho), each built
from the params file with only those entries changed, and report the data
log-likelihood (every instrument's err_scale at 1, so differences are
-chi2/2).  The winner goes into the params file; the table into the README.
"""

import copy
import importlib.util
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
A_HAN = np.degrees(2.292)
PSI = np.degrees(1.383)
Q2, Q3, S2, S3, TE, T0, U0 = (
    0.526,
    0.241,
    1.386,
    1.601,
    14.74,
    2459370.226,
    0.126,
)
EFF = Q2 / ((1.0 + Q2) * S2)


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
    base_cfg = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    base_prm = yaml.safe_load(open("KMT-2021-BLG-1122L.params.yaml"))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        base_cfg.pop(k, None)

    rules = {
        "a+180": A_HAN + 180.0,
        "180-a": 180.0 - A_HAN,
        "a": A_HAN,
        "-a": -A_HAN,
    }
    combos = []
    for rule, a2 in rules.items():
        for u0 in (U0, -U0):
            for psi_sign, tag in ((-1, "a2-psi"), (+1, "a2+psi")):
                a3 = a2 + psi_sign * PSI
                e2 = np.array(
                    [np.cos(np.radians(a2)), -np.sin(np.radians(a2))]
                )
                e3 = np.array(
                    [np.cos(np.radians(a3)), -np.sin(np.radians(a3))]
                )
                com = (Q2 * S2 * e2 + Q3 * S3 * e3) / (1.0 + Q2 + Q3)
                D = com - EFF * e2
                for shift, stag in ((False, "noshift"), (True, "COMshift")):
                    t0 = T0 - (D[0] * TE if shift else 0.0)
                    u0s = u0 + (D[1] if shift else 0.0)
                    combos.append(
                        (
                            f"{rule:>6s} u0={u0:+.3f} {tag} {stag}",
                            a2,
                            a3,
                            t0,
                            u0s,
                            shift,
                        )
                    )

    rows = []
    for label, a2, a3, t0, u0s, shift in combos:
        prm = copy.deepcopy(base_prm)
        prm["source.Source.t_0"] = {"initval": float(t0)}
        prm["source.Source.u_0"] = {"initval": float(u0s)}
        prm["lens.LensB.alpha"] = {"initval": float(a2)}
        # slot 1 has no alpha/s relations: seed its sampled coordinates
        prm.pop("lens.LensC.alpha", None)
        prm.pop("lens.LensC.s", None)
        prm["lens.LensC.log_s"] = {"initval": float(np.log10(S3))}
        prm["lens.LensC.xalpha"] = {"initval": float(np.cos(np.radians(a3)))}
        prm["lens.LensC.yalpha"] = {"initval": float(np.sin(np.radians(a3)))}
        try:
            system = System(copy.deepcopy(base_cfg), user_params=prm)
            system.prepare()
            model = system.build_model()
            start = mmf.raw_start(system, model)
            parts, total, ok, _ = mmf.decompose(system, model, start)
            data = {
                k: v
                for k, v in parts.items()
                if "mulensinstrument" in k and "model" in k
            }
            dsum = float(sum(data.values()))
            dets = {d.name: d for d in model.deterministics}
            want = [
                n
                for n in (
                    "lens.s",
                    "lens.alpha",
                    "lens.q",
                    "mulensevent.t_E",
                    "mulensevent.theta_E",
                    "source.rho",
                )
                if n in dets
            ]
            f = pytensor.function(
                list(model.value_vars),
                model.replace_rvs_by_values([dets[n] for n in want]),
                on_unused_input="ignore",
            )
            vals = f(*[np.asarray(start[v.name]) for v in model.value_vars])
            rows.append(
                {
                    "label": label,
                    "alpha2": float(a2),
                    "alpha3": float(a3),
                    "t0": float(t0),
                    "u0": float(u0s),
                    "shift": shift,
                    "data_logp": dsum,
                    "total": float(total),
                    "terms": data,
                }
            )
            print(
                f"{label:46s} t0 {t0:.3f} u0 {u0s:+.4f}  data logp {dsum:12.2f}  total {total:12.2f}",
                flush=True,
            )
            if len(rows) == 1:
                print(
                    "   start:",
                    "  ".join(
                        f"{n}={np.round(np.atleast_1d(v), 4).tolist()}"
                        for n, v in zip(want, vals)
                    ),
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "label": label,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            print(
                f"{label:46s} FAILED {type(exc).__name__}: {str(exc)[:160]}",
                flush=True,
            )
    rows_ok = [r for r in rows if "data_logp" in r]
    if rows_ok:
        best = max(rows_ok, key=lambda r: r["data_logp"])
        print(
            f"\nBEST by data logp: {best['label']}  alpha2 {best['alpha2']:.2f} alpha3 {best['alpha3']:.2f} t0 {best['t0']:.3f} u0 {best['u0']:+.4f}  data logp {best['data_logp']:.2f}"
        )
        for r in sorted(rows_ok, key=lambda r: -r["data_logp"]):
            print(
                f"  {r['label']:46s} delta chi2 = {2 * (best['data_logp'] - r['data_logp']):12.1f}"
            )
    with open("check_start.json", "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
