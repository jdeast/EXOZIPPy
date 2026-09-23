#!/usr/bin/env python3
"""Does EXOZIPPy's PRIOR disfavour the true lens/source geometry?

The MulensModel basin comparison (dc128_basin_scan.py) is light-curve only:
it says the data prefer the truth basin by ~4800 in chi2.  But EXOZIPPy
samples logp = light curve + galactic kinematic/density prior + IMF, and
the truth here is an awkward configuration for a prior -- lens at 7999 pc
against a source at 8143 pc, i.e. pi_rel = 0.0022 mas, a lens sitting
essentially on top of its source.  If the galactic prior charges more than
~2400 nats for that, then preferring the s = 0.86 basin would be a PRIOR
problem, not a sampling one, and no amount of polishing or laddering would
fix it.

This evaluates every potential in the built model at two pinned points --
the answer key's geometry, and the 2026-08-10 posterior median -- and
prints the term-by-term difference.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytensor
import yaml

from exozippy.system import System

HERE = Path(__file__).resolve().parent
EVENT_DIR = HERE / "events" / "128"

RA, DEC = 267.595, -28.982

# Answer key (see dc128_truth_forward.py) and the old posterior median.
SOLUTIONS = {
    "truth": dict(
        d_lens=7999.0,
        d_source=8143.0,
        logmass=np.log10(0.454),
        r_source=0.961,
        lmu_l=-6.48044,
        lmu_b=-1.37793,
        smu_l=-7.73446,
        smu_b=-2.6885,
        t_0=2458554.8868815,
        u_0=0.141832,
        s=0.993145,
        q=0.0012118,
        alpha=308.0,
    ),
    "oldfit": dict(
        d_lens=1470.0,
        d_source=3290.0,
        logmass=-0.097,
        r_source=1.0,
        # The old fit reports equatorial pm directly; converted to galactic
        # below via the same helper, so both points go through one path.
        pm_eq=dict(lens=(-8.8, -46.2), source=(18.0, -30.5)),
        t_0=2458554.9038,
        u_0=0.13878,
        s=0.86321,
        q=0.000921,
        alpha=-52.314,
    ),
}


def galactic_pm_to_equatorial(mu_l, mu_b, ra, dec):
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs",
        distance=1.0 * u.kpc,
        pm_ra_cosdec=0 * u.mas / u.yr,
        pm_dec=0 * u.mas / u.yr,
    )
    g = c.galactic
    gg = SkyCoord(
        l=g.l,
        b=g.b,
        distance=1.0 * u.kpc,
        frame="galactic",
        pm_l_cosb=mu_l * u.mas / u.yr,
        pm_b=mu_b * u.mas / u.yr,
    )
    e = gg.icrs
    return (
        float(e.pm_ra_cosdec / (u.mas / u.yr)),
        float(e.pm_dec / (u.mas / u.yr)),
    )


def build_params(sol):
    pin = lambda v: {"initval": float(v), "sigma": 0}  # noqa: E731
    if "pm_eq" in sol:
        lmu, smu = sol["pm_eq"]["lens"], sol["pm_eq"]["source"]
    else:
        lmu = galactic_pm_to_equatorial(sol["lmu_l"], sol["lmu_b"], RA, DEC)
        smu = galactic_pm_to_equatorial(sol["smu_l"], sol["smu_b"], RA, DEC)
    a = np.radians(sol["alpha"])
    return {
        "star.Lens.ra": pin(RA),
        "star.Lens.dec": pin(DEC),
        "star.Lens.teff": {"sigma": 0.0},
        "star.Lens.feh": {"sigma": 0.0},
        "star.Lens.radius": {"sigma": 0.0},
        "planet.Companion.radius": {"sigma": 0},
        "star.Lens.distance": pin(sol["d_lens"]),
        "star.Source.distance": pin(sol["d_source"]),
        "star.Lens.logmass": pin(sol["logmass"]),
        "star.Source.radius": pin(sol["r_source"]),
        "star.Lens.pm_ra": pin(lmu[0]),
        "star.Lens.pm_dec": pin(lmu[1]),
        "star.Source.pm_ra": pin(smu[0]),
        "star.Source.pm_dec": pin(smu[1]),
        "lens.Lens.t_0": pin(sol["t_0"]),
        "lens.Lens.u_0": pin(sol["u_0"]),
        "lens.Lens.log_s": pin(np.log10(sol["s"])),
        "planet.Companion.log_q": pin(np.log10(sol["q"])),
        "lens.Lens.xalpha": pin(np.cos(a)),
        "lens.Lens.yalpha": pin(np.sin(a)),
    }


def evaluate(name, sol, config):
    system = System(config, build_params(sol))
    system.prepare()
    model = system.build_model()
    ip = model.initial_point()
    args = [ip[vv.name] for vv in model.value_vars]

    terms = {}
    for v in list(model.potentials) + list(model.observed_RVs):
        fn = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values([v])[0],
            on_unused_input="ignore",
        )
        terms[v.name] = float(np.sum(fn(*args)))
    total = float(np.asarray(model.compile_logp()(ip)))

    derived = {}
    for p in system.get_all_parameters():
        val = getattr(p, "value", None)
        if isinstance(val, pytensor.graph.basic.Variable):
            fn = pytensor.function(
                model.value_vars,
                model.replace_rvs_by_values([val])[0],
                on_unused_input="ignore",
            )
            derived[p.label] = float(
                np.atleast_1d(np.asarray(fn(*args), float)).ravel()[0]
            )
    print(f"  {name}: total logp = {total:.3f}", flush=True)
    return terms, total, derived


def main():
    with open(EVENT_DIR / "DC2018_128.yaml") as f:
        config = yaml.safe_load(f)
    config.pop("parameter_file", None)
    for blk in config.get("lens", []):
        blk["mmexofast"] = False
    config["sampler"]["recompute_trace"] = False
    os.chdir(tempfile.mkdtemp(prefix="dc128prior_"))

    out = {}
    for name, sol in SOLUTIONS.items():
        out[name] = evaluate(name, sol, config)

    (t_terms, t_total, t_der) = out["truth"]
    (o_terms, o_total, o_der) = out["oldfit"]

    print("\n" + "=" * 78)
    print("logp TERM BY TERM   (positive delta favours TRUTH)")
    print("=" * 78)
    keys = sorted(set(t_terms) | set(o_terms))
    lc = pr = 0.0
    for k in keys:
        a, b = t_terms.get(k, 0.0), o_terms.get(k, 0.0)
        d = a - b
        if abs(d) < 1e-6:
            continue
        is_lc = "hogg" in k or "model." in k
        (lc, pr) = (lc + d, pr) if is_lc else (lc, pr + d)
        print(f"  {k:<62s} {a:14.2f} {b:14.2f} {d:+12.2f}")
    print("-" * 78)
    print(f"  {'LIGHT CURVE subtotal':<62s} {'':14s} {'':14s} {lc:+12.2f}")
    print(f"  {'PRIOR subtotal':<62s} {'':14s} {'':14s} {pr:+12.2f}")
    print(
        f"  {'TOTAL':<62s} {t_total:14.2f} {o_total:14.2f} "
        f"{t_total - o_total:+12.2f}"
    )

    print("\n" + "=" * 78)
    print("derived quantities at each point")
    print("=" * 78)
    for k in sorted(set(t_der) | set(o_der)):
        print(
            f"  {k:<44s} {t_der.get(k, float('nan')):14.6g} "
            f"{o_der.get(k, float('nan')):14.6g}"
        )

    print(
        "\nNOTE: fluxes/err_scale/out_* are at their SEEDED values in both "
        "points, not\nre-optimized, so the light-curve subtotal is "
        "indicative; the PRIOR subtotal is\nthe number this script exists "
        "to measure and it is unaffected by that."
    )


if __name__ == "__main__":
    main()
