#!/usr/bin/env python3
"""Does EXOZIPPy's derived chain reproduce the answer key at truth?

Pins every INPUT the answer key gives directly -- both distances, the lens
mass, the source radius, both proper motions, t_0, u_0, s, q, alpha -- and
reads back the quantities EXOZIPPy DERIVES from them: pi_rel, theta_E,
mu_rel, t_E, rho, pi_E.  Those are exactly the links that produced last
run's mu_rel = 31 mas/yr and D_lens = 1.5 kpc, so a forward check against
the truth table tests the physics without involving the sampler at all.

t_E and pi_E are geocentric (Skowron+2011 t0_par frame) while the answer
key's are the simulation's own frame, so a percent-level difference there
is expected; a factor-of-several difference is not.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
import pytensor
import yaml
from astropy.coordinates import SkyCoord

from exozippy.system import System

HERE = Path(__file__).resolve().parent
EVENT_DIR = HERE / "events" / "128"

# --- answer key, event 128 (master_file.txt row 128) ----------------------
T = dict(
    Ds=8.143,
    Dl=7.999,
    Ml=0.454,
    Rs=0.961,
    smu_l=-7.73446,
    smu_b=-2.6885,
    lmu_l=-6.48044,
    lmu_b=-1.37793,
    u0=0.141832,
    alpha_dc=348.357,
    t0=2458554.8868815,
    tE=18.2374,
    thE=0.0904762,
    piE=0.0244348,
    rhos=0.00606537,
    murel=1.81388,
    q=0.0012118,
    s=0.993145,
    ra=267.595,
    dec=-28.982,
)
PI_REL_TRUTH = 1000.0 / T["Dl"] / 1000.0 - 1000.0 / T["Ds"] / 1000.0  # mas


def galactic_pm_to_equatorial(mu_l, mu_b, ra, dec):
    """(mu_l*, mu_b) [mas/yr] -> (mu_ra*, mu_dec) at this sky position."""
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


def build_params(alpha_deg, u0_sign=+1.0):
    lmu = galactic_pm_to_equatorial(T["lmu_l"], T["lmu_b"], T["ra"], T["dec"])
    smu = galactic_pm_to_equatorial(T["smu_l"], T["smu_b"], T["ra"], T["dec"])
    a = np.radians(alpha_deg)
    pin = lambda v: {"initval": float(v), "sigma": 0}  # noqa: E731
    return {
        "star.Lens.ra": pin(T["ra"]),
        "star.Lens.dec": pin(T["dec"]),
        "star.Lens.teff": {"sigma": 0.0},
        "star.Lens.feh": {"sigma": 0.0},
        "star.Lens.radius": {"sigma": 0.0},
        "planet.Companion.radius": {"sigma": 0},
        # --- the answer key's inputs ---
        "star.Lens.distance": pin(T["Dl"] * 1000.0),
        "star.Source.distance": pin(T["Ds"] * 1000.0),
        "star.Lens.logmass": pin(np.log10(T["Ml"])),
        "star.Source.radius": pin(T["Rs"]),
        "star.Lens.pm_ra": pin(lmu[0]),
        "star.Lens.pm_dec": pin(lmu[1]),
        "star.Source.pm_ra": pin(smu[0]),
        "star.Source.pm_dec": pin(smu[1]),
        "lens.Lens.t_0": pin(T["t0"]),
        "lens.Lens.u_0": pin(u0_sign * T["u0"]),
        "lens.Lens.log_s": pin(np.log10(T["s"])),
        "planet.Companion.log_q": pin(np.log10(T["q"])),
        "lens.Lens.xalpha": pin(np.cos(a)),
        "lens.Lens.yalpha": pin(np.sin(a)),
    }


def main():
    alpha_deg = float(sys.argv[1]) if len(sys.argv) > 1 else -50.4
    u0_sign = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    print(f"pinning alpha = {alpha_deg} deg, u_0 sign = {u0_sign:+.0f}\n")

    with open(EVENT_DIR / "DC2018_128.yaml") as f:
        config = yaml.safe_load(f)
    config.pop("parameter_file", None)
    for blk in config.get("lens", []):
        blk["mmexofast"] = False  # pin truth, do not let seeds win
    config["sampler"]["recompute_trace"] = False

    # Every path in the dumped config is absolute, so run from scratch space
    # rather than inside the live fit's directory.
    os.chdir(tempfile.mkdtemp(prefix="dc128fwd_"))
    system = System(config, build_params(alpha_deg, u0_sign))
    system.prepare()
    model = system.build_model()

    params = system.get_all_parameters()
    labels, nodes = [], []
    for p in params:
        v = getattr(p, "value", None)
        if isinstance(v, pytensor.graph.basic.Variable):
            labels.append(p.label)
            nodes.append(v)
    exprs = model.replace_rvs_by_values(nodes)
    fn = pytensor.function(model.value_vars, exprs, on_unused_input="ignore")
    ip = model.initial_point()
    vals = fn(*[ip[vv.name] for vv in model.value_vars])
    got = {
        lab: np.atleast_1d(np.asarray(v, float)).ravel()[0]
        for lab, v in zip(labels, vals)
    }

    print("=" * 72)
    print("DERIVED vs the answer key   (inputs pinned, nothing fitted)")
    print("=" * 72)
    checks = [
        ("lens.pi_rel", "pi_rel", PI_REL_TRUTH, "mas"),
        ("lens.theta_E", "theta_E", T["thE"], "mas"),
        ("lens.mu_rel_mag", "mu_rel (helio)", T["murel"], "mas/yr"),
        ("lens.t_E", "t_E", T["tE"], "d"),
        ("lens.rho", "rho", T["rhos"], ""),
    ]
    for key, name, truth, unit in checks:
        cand = [
            k
            for k in got
            if k.endswith(key.split(".")[-1]) and k.startswith("lens")
        ]
        v = got.get(key, got.get(cand[0]) if cand else np.nan)
        ratio = v / truth if truth else np.nan
        print(
            f"  {name:16s} exozippy = {v:14.6g}   truth = {truth:12.6g} "
            f"{unit:7s} ratio = {ratio:8.4f}"
        )
    pn = got.get("lens.pi_E_N", np.nan)
    pe = got.get("lens.pi_E_E", np.nan)
    print(
        f"  {'|pi_E|':16s} exozippy = {np.hypot(pn, pe):14.6g}   "
        f"truth = {T['piE']:12.6g} {'':7s} "
        f"ratio = {np.hypot(pn, pe) / T['piE']:8.4f}"
    )
    print(f"      pi_E_N = {pn:.6g}   pi_E_E = {pe:.6g}")

    print("\n" + "=" * 72)
    print("ALL derived values at the pinned truth point")
    print("=" * 72)
    for lab in sorted(got):
        print(f"  {lab:<42s} {got[lab]:.8g}")

    total = float(np.asarray(model.compile_logp()(ip)))
    print(f"\nTOTAL logp at the pinned truth point = {total:.4f}")

    obs = [v for v in model.observed_RVs if "mulens" in v.name]
    pot = [v for v in model.potentials if "mulens" in v.name]
    print(f"observed mulens RVs: {[v.name for v in obs]}")
    print(f"mulens potentials:   {[v.name for v in pot]}")
    for v in pot:
        f2 = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values([v])[0],
            on_unused_input="ignore",
        )
        print(
            f"  {v.name} = "
            f"{float(np.sum(f2(*[ip[vv.name] for vv in model.value_vars]))):.3f}"
        )

    json.dump(
        {k: float(v) for k, v in got.items()} | {"logp": total},
        open(HERE / "dc128_truth_forward.json", "w"),
        indent=2,
    )
    print(f"\nwrote {HERE / 'dc128_truth_forward.json'}")


if __name__ == "__main__":
    main()
