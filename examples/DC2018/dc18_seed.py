"""Our own microlensing seed finder: a PSPL peak fit, no MMEXOFAST.

WHY THIS EXISTS.  Review 8.4.9 established that EXOZIPPy HAS NO PEAK
FINDER -- searched the tree, every branch and the full history.
`push_seed_hints` is the only source of t_0/u_0/t_E hints, and it reads
MMEXOFAST's JSON, so `mmexofast: false` leaves DC2018-128 starting 1,445
days from its own peak.  That makes MMEXOFAST a hard dependency of every
microlensing fit, which is exactly what JDE asked to remove: "seeded with
our own peak finder and generic (defaults.yaml) values for logs and logq".

WHY IT LIVES HERE AND NOT IN src/.  8.4.9 is filed and belongs to whoever
takes it -- a real peak finder is a component-level feature with its own
tests and its own interface.  Duplicating that work is explicitly not our
job.  But the seed file is just DATA: mmexofast_support.py documents the
JSON contract (`fits` list of `parameters` dicts, plus `errfacs`,
`mag_methods`, `coords`, `excluded_points`), and nothing checks who wrote
it.  So this emits that same shape and the fit consumes it unchanged --
no src edit, no fork of 8.4.9, and the sweep stops depending on MMEXOFAST
today.

THE METHOD, and why it is this and not something cleverer.  PSPL flux is
LINEAR in (f_source, f_blend) once the magnification is known:

    flux(t) = f_s * A(t; t_0, u_0, t_E) + f_b

so for any trial (t_0, u_0, t_E) the two flux parameters have a closed-form
weighted least-squares solution and drop out of the search entirely.  That
turns a 5-parameter fit into a 3-parameter one, which a coarse grid plus a
Nelder-Mead refinement solves in seconds on 38,000 points.  No gradients,
no VBBL, no binary-lens magnification -- deliberately, because the seed's
job is to find the EVENT, not the anomaly.  s, q and alpha are then left at
defaults.yaml's generic values (log_s = 0, i.e. s = 1) and the sampler is
asked to find the planet itself.

WHAT IT DELIBERATELY DOES NOT DO.  It does not mask outliers
(`excluded_points` is emitted empty), because the sweep runs the Hogg
mixture likelihood on every light curve and a hard mask would double-count
that job.  It does not rescale errors (`errfacs` = 1.0) for the same
reason: err_scale is a fitted parameter.  Both keys are still written, so
the consumer sees the shape it expects rather than a missing key.
"""

import argparse
import io
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc18_common as C  # noqa: E402


def pspl_mag(t, t_0, u_0, t_E):
    """Paczynski magnification.  u_0 here is |u_0|; the sign is a symmetry."""
    tau = (t - t_0) / t_E
    u2 = tau * tau + u_0 * u_0
    return (u2 + 2.0) / np.sqrt(u2 * (u2 + 4.0))


def _chi2_linear_flux(mag, flux, ivar):
    """chi2 after solving f_s, f_b analytically.  Returns (chi2, f_s, f_b).

    The 2x2 normal equations for flux = f_s * mag + f_b.  A singular matrix
    means the trial magnification is constant over the data (t_E far outside
    the baseline, or u_0 so large there is no event), which is a legitimate
    trial to reject rather than an error -- hence the inf, not an exception.
    """
    s11 = np.sum(ivar * mag * mag)
    s10 = np.sum(ivar * mag)
    s00 = np.sum(ivar)
    b1 = np.sum(ivar * mag * flux)
    b0 = np.sum(ivar * flux)
    det = s11 * s00 - s10 * s10
    if not np.isfinite(det) or abs(det) < 1e-30:
        return np.inf, 0.0, 0.0
    f_s = (b1 * s00 - b0 * s10) / det
    f_b = (b0 * s11 - b1 * s10) / det
    resid = flux - (f_s * mag + f_b)
    return float(np.sum(ivar * resid * resid)), float(f_s), float(f_b)


def _chi2(theta, curves):
    """Summed chi2 over bands.  Each band gets its OWN f_s, f_b (different
    filters, different source colour) but they SHARE the geometry."""
    t_0, u_0, t_E = theta
    if u_0 <= 0 or t_E <= 0:
        return np.inf
    total = 0.0
    for t, flux, ivar in curves:
        c, _, _ = _chi2_linear_flux(pspl_mag(t, t_0, u_0, t_E), flux, ivar)
        total += c
    return total


def find_seed(curves, verbose=True):
    """Coarse grid over (t_0, u_0, t_E), then Nelder-Mead.

    The grid exists because the chi2 surface in t_0 is a forest of narrow
    minima -- one per candidate peak -- and a local optimizer started at the
    wrong one stays there.  t_0 candidates come from the flux excess itself
    rather than a uniform scan, so the grid tracks the data's own structure.
    """
    t_all = np.concatenate([c[0] for c in curves])
    tmin, tmax = float(t_all.min()), float(t_all.max())

    # t_0 candidates: the highest-excess epochs of the densest band, thinned
    # so two candidates are never within a day of each other (they would
    # descend into the same minimum and waste grid points).
    t, flux, ivar = max(curves, key=lambda c: len(c[0]))
    base = np.median(flux)
    scat = 1.4826 * np.median(np.abs(flux - base)) or 1.0
    excess = (flux - base) / scat
    order = np.argsort(excess)[::-1]
    cands, seen = [], []
    for i in order[:5000]:
        ti = float(t[i])
        if all(abs(ti - s) > 1.0 for s in seen):
            cands.append(ti)
            seen.append(ti)
        if len(cands) >= 12:
            break
    if not cands:
        cands = [0.5 * (tmin + tmax)]

    best = (np.inf, None)
    for t0 in cands:
        for u0 in (0.005, 0.02, 0.05, 0.15, 0.4, 0.8, 1.2):
            for tE in (3.0, 8.0, 20.0, 50.0, 120.0):
                c = _chi2((t0, u0, tE), curves)
                if c < best[0]:
                    best = (c, (t0, u0, tE))
    if verbose:
        print(
            "  grid best: t_0=%.4f u_0=%.4f t_E=%.3f  chi2=%.1f"
            % (best[1] + (best[0],)),
            flush=True,
        )

    # Refine in log(u_0), log(t_E) so the optimizer cannot step negative.
    t0g, u0g, tEg = best[1]

    def wrapped(p):
        return _chi2((p[0], np.exp(p[1]), np.exp(p[2])), curves)

    res = minimize(
        wrapped,
        [t0g, np.log(u0g), np.log(tEg)],
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-3},
    )
    t_0, u_0, t_E = res.x[0], float(np.exp(res.x[1])), float(np.exp(res.x[2]))
    chi2 = float(res.fun)
    if verbose:
        print(
            "  refined  : t_0=%.4f u_0=%.4f t_E=%.3f  chi2=%.1f (%s)"
            % (
                t_0,
                u_0,
                t_E,
                chi2,
                "converged" if res.success else "NOT converged",
            ),
            flush=True,
        )
    return {
        "t_0": float(t_0),
        "u_0": u_0,
        "t_E": t_E,
        "chi2": chi2,
        "t_span": (tmin, tmax),
    }


# Generic companion seeds.  s = 1 is defaults.yaml's log_s initval; the two
# alpha values are the only genuinely distinct trajectory orientations a
# seed can offer without pretending to know the caustic (0 and 90 deg cover
# the along-axis and across-axis cases).  q = 1e-3 sits mid-log between the
# Roman detection floor and the brown-dwarf boundary.  These are STARTS, not
# priors: the sampler is expected to move off them.
GENERIC = [
    {"s": 1.0, "q": 1.0e-3, "alpha": 0.0, "rho": 1.0e-3},
    {"s": 1.0, "q": 1.0e-3, "alpha": 90.0, "rho": 1.0e-3},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("event", type=int)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--bands", default="W149,Z087")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = C.data_dir_or_raise(args.data_dir)
    bands = tuple(b.strip() for b in args.bands.split(","))
    files = C.light_curve_files(d, args.event, bands=bands)

    curves, names = [], []
    for b in bands:
        arr = np.loadtxt(files[b])
        t, f, e = arr[:, 0], arr[:, 1], arr[:, 2]
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0)
        curves.append((t[ok], f[ok], 1.0 / e[ok] ** 2))
        names.append(os.path.basename(files[b]))
        print("  %s: %d points" % (b, ok.sum()), flush=True)

    seed = find_seed(curves)
    t_0, u_0, t_E = seed["t_0"], seed["u_0"], seed["t_E"]

    ra, dec = C.event_coords(d, args.event)
    fits = [
        {"parameters": dict(t_0=t_0, u_0=u_0, t_E=t_E, **g)} for g in GENERIC
    ]

    # mag_methods is the finite-source window: [t_start, method, t_end].
    # +/- 2 t_E around the peak covers the caustic crossings for any s the
    # sampler can reach, clipped to the data so the window is never empty.
    lo = max(seed["t_span"][0], t_0 - 2.0 * t_E)
    hi = min(seed["t_span"][1], t_0 + 2.0 * t_E)

    out = {
        "fits": fits,
        "errfacs": {n: 1.0 for n in names},
        "mag_methods": [lo, "VBBL", hi],
        "coords": "%.6f %.6f" % (ra, dec),
        "jd_offset": 0.0,
        "excluded_points": {
            n: {"n_data": int(len(c[0])), "indices": [], "times": []}
            for n, c in zip(names, curves)
        },
        "_provenance": {
            "writer": "examples/DC2018/dc18_seed.py (PSPL peak fit, no MMEXOFAST)",
            "pspl_chi2": seed["chi2"],
            "note": "s/q/alpha are defaults.yaml generics, NOT fitted.",
        },
    }
    ev3 = "%03d" % args.event
    dest = args.out or "events/%s/DC2018_%s_seed.json" % (ev3, ev3)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    json.dump(out, io.open(dest, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % dest, flush=True)


if __name__ == "__main__":
    main()
