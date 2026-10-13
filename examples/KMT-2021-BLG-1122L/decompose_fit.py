"""Where does the KMT-2021-BLG-1122L async fit's unphysical posterior come
from?  Decompose the log-probability, term by term, at (a) the fit's best
draw, (b) the median-lp draw, and (c) the shipped start, and print the
physical quantities that tie distance to the data (source radius/teff/av,
the SED-predicted source magnitude vs the light curves' source flux, the
zeropoints).  Raw priors are collapsed against their logit-Jacobian
potentials (CoV:<x>), so the split is data / coordinate priors / other.

    python decompose_fit.py <run dir> <prefix>
"""

import importlib.util
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main(run_dir, prefix):
    import arviz as az
    import pytensor

    from exozippy.system import System
    from exozippy.whitening import restore_whitening_for_trace

    spec = importlib.util.spec_from_file_location(
        "mmf",
        os.path.join(HERE, "..", "..", "scripts", "make_mulens_fixtures.py"),
    )
    mmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mmf)
    os.chdir(HERE)
    config = yaml.safe_load(
        open(os.path.join(run_dir, f"{prefix}.cores64.yaml"))
    )
    user_params = yaml.safe_load(open(config["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    start = mmf.raw_start(
        system, model
    )  # BEFORE restoring: the shipped start in this build's coordinates
    start_phys = None
    vv = list(model.value_vars)
    vnames = [v.name for v in vv]
    dets = {d.name: d for d in model.deterministics}
    want = [
        n
        for n in (
            "star.distance",
            "star.mass",
            "star.radius",
            "star.teff",
            "star.av",
            "star.logg",
            "mulensevent.theta_E",
            "mulensevent.t_E",
            "mulensevent.pi_rel",
            "mulensevent.mu_rel_mag",
            "source.rho",
            "lens.s",
            "lens.q",
            "lens.alpha",
            "mulensinstrument.f_source",
            "mulensinstrument.f_blend",
            "mulensinstrument.zeropoint",
            "mulensinstrument.err_scale",
            "mulensinstrument.q_source",
        )
        if n in dets
    ]
    g0 = pytensor.function(
        vv,
        model.replace_rvs_by_values([dets[n] for n in want]),
        on_unused_input="ignore",
    )
    start_phys = g0(*[np.asarray(start[n]) for n in vnames])
    st = restore_whitening_for_trace(
        system,
        os.path.join(run_dir, f"{prefix}_whitening.json"),
        os.path.join(run_dir, f"{prefix}_trace.nc"),
    )
    print("whitening:", st)
    g = pytensor.function(
        vv,
        model.replace_rvs_by_values([dets[n] for n in want]),
        on_unused_input="ignore",
    )
    idata = az.from_netcdf(os.path.join(run_dir, f"{prefix}_trace.nc"))
    lp = idata.sample_stats.lp.values
    post = idata.posterior
    ib = np.unravel_index(np.argmax(lp), lp.shape)
    flat = lp.ravel()
    im = np.unravel_index(int(np.argsort(flat)[flat.size // 2]), lp.shape)

    def point(c, d):
        return {
            n: np.asarray(post[n].values[c, d], dtype=float) for n in vnames
        }

    def decomp(label, pt, phys=None):
        parts, total, ok, _ = mmf.decompose(system, model, pt)
        col = {}
        for k, v in parts.items():
            v = float(v)
            if k.startswith("RV:") and k.endswith("_raw"):
                key = "CoV:" + k[3:-4]
            elif k.startswith("POT:logit_uniform_prior."):
                key = "CoV:" + k[len("POT:logit_uniform_prior.") :]
            else:
                key = k
            col[key] = col.get(key, 0.0) + v
        data = sum(
            v
            for k, v in col.items()
            if k.startswith("RV:") and "mulensinstrument" in k
        )
        cov = sum(v for k, v in col.items() if k.startswith("CoV:"))
        print(
            f"\n===== {label}: total {total:.2f} = data {data:.2f} + coordinate priors {cov:.2f} + other {total - data - cov:.2f}"
        )
        for k, v in sorted(col.items(), key=lambda kv: -abs(kv[1]))[:22]:
            print(f"   {k:55s} {v:12.2f}")
        vals = phys if phys is not None else g(*[pt[n] for n in vnames])
        for n, v in zip(want, vals):
            print(
                f"   {n:32s} {np.array2string(np.atleast_1d(np.asarray(v, dtype=float)), precision=4, max_line_width=140)}"
            )
        return col

    print(
        f"best draw chain {ib[0]} draw {ib[1]} stored lp {lp[ib]:.2f}; median-lp draw stored lp {lp[im]:.2f}"
    )
    a = decomp("BEST DRAW", point(*ib))
    decomp("MEDIAN-LP DRAW", point(*im))
    # the shipped start, evaluated with the RESTORED whitening would be wrong;
    # report its physical state from before the restore and skip its terms.
    print(
        "\n===== SHIPPED START (physical state only; terms need the build's own whitening)"
    )
    for n, v in zip(want, start_phys):
        print(
            f"   {n:32s} {np.array2string(np.atleast_1d(np.asarray(v, dtype=float)), precision=4, max_line_width=140)}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:3])
