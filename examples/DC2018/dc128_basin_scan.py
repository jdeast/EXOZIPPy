#!/usr/bin/env python3
"""Which 2L1S basin do the event-128 light curves actually prefer?

Independent of EXOZIPPy: this uses MulensModel directly, so a disagreement
localizes to the light-curve model rather than to our parameterization,
priors, or sampler.  Source fluxes are fit linearly per dataset at every
evaluation, so only the nonlinear parameters are being compared.

Candidates:
  truth       the challenge answer key (alpha is scanned -- the master
              file's alpha convention is not MulensModel's)
  mmx0/mmx1   MMEXOFAST's two seed solutions (found with no_parallax)
  exozippy    the 2026-08-10 EXOZIPPy posterior median

Two parallax settings, because EXOZIPPy derives pi_E from the lens/source
distances rather than sampling it, and its reported |pi_E| = 0.239 is ten
times the truth's 0.0244.  If a basin only wins under a forced large
parallax, that is the mechanism, not the light curve.
"""

import argparse
import json
from pathlib import Path

import MulensModel as mm
import numpy as np
from scipy.optimize import minimize

DATA = Path.home() / "python/MMEXOFAST/data/2018DataChallenge"
EVENT = 128
BANDS = ("W149", "Z087")
PARNAMES = ["t_0", "u_0", "t_E", "rho", "s", "q", "alpha"]
COORDS = "17h50m22.8s -28d58m55.2s"
MAG_WINDOW = (2458535.660763293, 2458573.9869036176)

# MMEXOFAST's error renormalization factors, so chi2/N is interpretable.
ERRFAC = {"W149": 1.1572828624601683, "Z087": 1.1250691305325033}

TRUTH = dict(
    t_0=2458554.8868815,
    u_0=0.141832,
    t_E=18.2374,
    rho=0.00606537,
    s=0.993145,
    q=0.0012118,
    alpha=348.357,
)
MMX0 = dict(
    t_0=2458554.8238334553,
    u_0=0.13120606277363886,
    t_E=19.1630701622593,
    s=0.8590716149893207,
    alpha=-50.36627971968676,
    rho=1.212872626449141e-06,
    q=0.0009259770843883544,
)
MMX1 = dict(
    t_0=2458554.8238334553,
    u_0=0.13120606277363886,
    t_E=19.1630701622593,
    s=0.9821672860486524,
    alpha=-50.474548801060585,
    rho=0.008904392591931045,
    q=0.0002547434221159895,
)
EXOZIPPY = dict(
    t_0=2458554.9038,
    u_0=0.13878,
    t_E=18.354,
    rho=0.00091,
    s=0.86321,
    q=0.000921,
    alpha=-52.314,
)
EXOZIPPY_PIE = dict(pi_E_N=-0.121, pi_E_E=-0.206)

# Truth pi_E: |pi_E| = pi_rel/theta_E = 0.0244348, directed along mu_rel.
# mu_rel(galactic) = lens - source = (1.25402, 1.31057) mas/yr in (l, b).
TRUTH_MU_REL_LB = (-6.48044 - (-7.73446), -1.37793 - (-2.6885))
TRUTH_PIE_MAG = 0.0244348


def truth_pi_E():
    """(pi_E_N, pi_E_E) from the answer key's galactic-frame mu_rel."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(COORDS, frame="icrs", unit=(u.hourangle, u.deg))
    g = c.galactic
    mu_l, mu_b = TRUTH_MU_REL_LB
    # Rotate the (l, b) proper-motion vector into (ra, dec) by finite
    # difference of the frame transformation at this sky position.
    eps = 1e-5
    p0 = np.array([c.ra.deg, c.dec.deg])
    dl = SkyCoord(
        l=g.l.deg + eps / np.cos(g.b.rad),
        b=g.b.deg,
        frame="galactic",
        unit="deg",
    ).icrs
    db = SkyCoord(
        l=g.l.deg, b=g.b.deg + eps, frame="galactic", unit="deg"
    ).icrs
    e_l = (
        np.array([(dl.ra.deg - p0[0]) * np.cos(c.dec.rad), dl.dec.deg - p0[1]])
        / eps
    )
    e_b = (
        np.array([(db.ra.deg - p0[0]) * np.cos(c.dec.rad), db.dec.deg - p0[1]])
        / eps
    )
    v = mu_l * e_l + mu_b * e_b  # (E, N) components
    v = v / np.hypot(*v)
    return float(TRUTH_PIE_MAG * v[1]), float(TRUTH_PIE_MAG * v[0])


def load_datasets(renorm=True):
    sets = []
    for b in BANDS:
        f = DATA / f"n20180816.{b}.WFIRST18.{EVENT}.txt"
        t, flux, err = np.loadtxt(f, unpack=True)
        if renorm:
            err = err * ERRFAC[b]
        sets.append(
            mm.MulensData(
                data_list=[t, flux, err], phot_fmt="flux", bandpass=b
            )
        )
    return sets


def make_event(params, datasets, pi_E=None):
    p = {k: float(params[k]) for k in PARNAMES}
    if pi_E is not None:
        p["pi_E_N"], p["pi_E_E"] = float(pi_E[0]), float(pi_E[1])
        p["t_0_par"] = float(params["t_0"])
        model = mm.Model(p, coords=COORDS)
    else:
        model = mm.Model(p)
    model.set_magnification_methods([MAG_WINDOW[0], "VBBL", MAG_WINDOW[1]])
    return mm.Event(datasets=datasets, model=model)


def chi2_of(params, datasets, pi_E=None):
    try:
        return float(make_event(params, datasets, pi_E).get_chi2())
    except Exception:  # geometry VBBL rejects
        return np.inf


def chi2_per_dataset(params, datasets, pi_E=None):
    ev = make_event(params, datasets, pi_E)
    ev.get_chi2()
    return [float(np.nansum(c)) for c in ev.get_chi2_per_point()]


def scan_alpha(base, datasets, pi_E=None, n=720):
    grid = np.linspace(0.0, 360.0, n, endpoint=False)
    curve = np.array(
        [chi2_of(dict(base, alpha=float(a)), datasets, pi_E) for a in grid]
    )
    return grid, curve


def refine(start, datasets, label, pi_E=None, free_pi_E=False):
    """Nelder-Mead on the nonlinear parameters (s, q, rho sampled in log)."""
    npar = 7 + (2 if free_pi_E else 0)

    def unpack(x):
        p = dict(
            t_0=x[0],
            u_0=x[1],
            t_E=x[2],
            rho=10 ** x[3],
            s=10 ** x[4],
            q=10 ** x[5],
            alpha=x[6],
        )
        pe = (x[7], x[8]) if free_pi_E else pi_E
        return p, pe

    def cost(x):
        if not (
            1e-7 < 10 ** x[3] < 0.2
            and 1e-4 < 10 ** x[4] < 10
            and 1e-8 < 10 ** x[5] < 1
            and 0.5 < x[2] < 500
        ):
            return 1e12
        p, pe = unpack(x)
        return chi2_of(p, datasets, pe)

    x0 = [
        start["t_0"],
        start["u_0"],
        start["t_E"],
        np.log10(start["rho"]),
        np.log10(start["s"]),
        np.log10(start["q"]),
        start["alpha"],
    ]
    if free_pi_E:
        x0 += list(pi_E if pi_E is not None else (0.0, 0.0))
    x0 = np.array(x0[:npar])
    c0 = cost(x0)
    print(f"\n--- refine: {label}  (start chi2 {c0:.2f}) ---", flush=True)
    res = minimize(
        cost,
        x0,
        method="Nelder-Mead",
        options=dict(maxfev=6000, xatol=1e-9, fatol=1e-3, adaptive=True),
    )
    p, pe = unpack(res.x)
    print(f"    chi2 = {res.fun:.2f}   (nfev={res.nfev})")
    for k in PARNAMES:
        print(f"      {k:6s} {p[k]:.6g}")
    if pe is not None:
        print(
            f"      pi_E   N={pe[0]:.4f} E={pe[1]:.4f} "
            f"|pi_E|={np.hypot(*pe):.4f}"
        )
    return p, pe, float(res.fun)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick", action="store_true", help="alpha scan only, coarser grid"
    )
    args = ap.parse_args()

    datasets = load_datasets()
    n_tot = sum(len(d.time) for d in datasets)
    t_pie = truth_pi_E()
    e_pie = (EXOZIPPY_PIE["pi_E_N"], EXOZIPPY_PIE["pi_E_E"])
    print(f"event {EVENT}: {n_tot} points; errors x MMEXOFAST errfacs")
    print(
        f"truth pi_E    = (N {t_pie[0]:+.5f}, E {t_pie[1]:+.5f})  "
        f"|pi_E|={np.hypot(*t_pie):.5f}"
    )
    print(
        f"exozippy pi_E = (N {e_pie[0]:+.5f}, E {e_pie[1]:+.5f})  "
        f"|pi_E|={np.hypot(*e_pie):.5f}\n"
    )

    print("=" * 70)
    print("1. chi2 of each candidate AS REPORTED (no parallax)")
    print("=" * 70)
    for label, p in [
        ("truth(raw alpha)", TRUTH),
        ("mmx0 s=0.859", MMX0),
        ("mmx1 s=0.982", MMX1),
        ("exozippy s=0.863", EXOZIPPY),
    ]:
        c = chi2_of(p, datasets)
        per = chi2_per_dataset(p, datasets) if np.isfinite(c) else []
        extra = (
            (
                "  ["
                + ", ".join(f"{b}={v:.0f}" for b, v in zip(BANDS, per))
                + "]"
            )
            if per
            else ""
        )
        print(
            f"  {label:18s} chi2 = {c:12.2f}  chi2/N = {c / n_tot:7.4f}{extra}"
        )

    print("\n" + "=" * 70)
    print("2. alpha scan at the truth t_0,u_0,t_E,rho,s,q (no parallax)")
    print("=" * 70)
    grid, curve = scan_alpha(TRUTH, datasets, n=360 if args.quick else 720)
    order = np.argsort(curve)
    for i in order[:6]:
        print(
            f"    alpha = {grid[i]:7.2f}  chi2 = {curve[i]:12.2f}"
            f"  chi2/N = {curve[i] / n_tot:7.4f}"
        )
    best_alpha = float(grid[order[0]])
    off = (best_alpha - TRUTH["alpha"] + 180) % 360 - 180
    print(
        f"\n  master-file alpha {TRUTH['alpha']:.3f} -> best {best_alpha:.3f}"
        f"  (offset {off:+.3f} deg)"
    )
    np.savetxt(
        "dc128_alpha_scan.txt",
        np.column_stack([grid, curve]),
        header="alpha_deg chi2",
    )
    if args.quick:
        return

    truth_best = dict(TRUTH, alpha=best_alpha)

    print("\n" + "=" * 70)
    print("3. each basin refined to its own optimum")
    print("=" * 70)
    out = {}
    out["truth_nopar"] = refine(truth_best, datasets, "truth basin, pi_E=0")
    out["exo_nopar"] = refine(EXOZIPPY, datasets, "exozippy basin, pi_E=0")
    out["truth_truthpie"] = refine(
        truth_best, datasets, "truth basin, pi_E fixed at truth", pi_E=t_pie
    )
    out["exo_exopie"] = refine(
        EXOZIPPY,
        datasets,
        "exozippy basin, pi_E fixed at exozippy's",
        pi_E=e_pie,
    )
    out["truth_freepie"] = refine(
        truth_best,
        datasets,
        "truth basin, pi_E FREE",
        pi_E=t_pie,
        free_pi_E=True,
    )
    out["exo_freepie"] = refine(
        EXOZIPPY,
        datasets,
        "exozippy basin, pi_E FREE",
        pi_E=e_pie,
        free_pi_E=True,
    )

    print("\n" + "=" * 70)
    print("VERDICT   (chi2 over %d points)" % n_tot)
    print("=" * 70)
    rows = [
        ("truth basin", "pi_E=0", out["truth_nopar"][2]),
        ("exozippy basin", "pi_E=0", out["exo_nopar"][2]),
        ("truth basin", "pi_E=truth", out["truth_truthpie"][2]),
        ("exozippy basin", "pi_E=exozippy", out["exo_exopie"][2]),
        ("truth basin", "pi_E free", out["truth_freepie"][2]),
        ("exozippy basin", "pi_E free", out["exo_freepie"][2]),
    ]
    best = min(r[2] for r in rows)
    for name, mode, c in rows:
        print(
            f"  {name:16s} {mode:16s} chi2 = {c:12.2f}   "
            f"dchi2 = {c - best:+10.2f}"
        )
    d = out["exo_freepie"][2] - out["truth_freepie"][2]
    print(
        f"\n  with pi_E free: exozippy - truth = {d:+.2f} chi2 "
        f"({d / 2:+.1f} nats)"
    )
    print(
        "  positive -> the data prefer the TRUTH basin and EXOZIPPy is "
        "stuck in a worse one (sampling failure)."
    )
    print("  negative -> the data really do prefer EXOZIPPy's basin.")

    json.dump(
        {
            k: dict(
                params=v[0],
                pi_E=(list(v[1]) if v[1] is not None else None),
                chi2=v[2],
            )
            for k, v in out.items()
        }
        | {"n_points": n_tot, "best_alpha_at_truth": best_alpha},
        open("dc128_basin_scan.json", "w"),
        indent=2,
    )
    print("\nwrote dc128_basin_scan.json, dc128_alpha_scan.txt")


if __name__ == "__main__":
    main()
