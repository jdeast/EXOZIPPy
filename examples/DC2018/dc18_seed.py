"""Our own microlensing seed finder: a PSPL peak fit, no MMEXOFAST.

WHY THIS EXISTS.  Review 8.4.9 established that EXOZIPPy HAS NO PEAK
FINDER -- searched the tree, every branch and the full history.
`push_seed_hints` is the only source of t_0/u_0/t_E hints, and it reads
MMEXOFAST's JSON, so `mmexofast: false` leaves DC2018-128 starting 1,445
days from its own peak.  That makes MMEXOFAST a hard dependency of every
microlensing fit, which is exactly what JDE asked to remove: "seeded with
our own peak finder and generic (defaults.yaml) values for logs and logq".

WHY IT LIVES HERE.  The seed file is just DATA: mmexofast_support.py
documents the JSON contract (`fits` list of `parameters` dicts, plus
`errfacs`, `mag_methods`, `coords`, `excluded_points`), and nothing checks
who wrote it.  So this emits that same shape and the fit consumes it
unchanged, and the sweep does not depend on MMEXOFAST.

THE METHOD IS THE COMPONENT'S.  This script used to carry its own copy of
the PSPL grid + Nelder-Mead fit, written before 8.4.9 had landed in src/.
Since 2026-09-21 it calls `peakfind.find_pspl_seed` -- the same function
`MulensInstrument` runs when a fit has no seed -- so the sweep's seeds and
a bare `exozippy` run's seeds cannot disagree, and the PRIMARY-FIRST pass
that fixes DC2018-226 (the strongest feature was the planetary anomaly,
not the event; see the peakfind module docstring, review 2.4.14) reaches
the sweep without a second implementation.  PSPL flux is LINEAR in
(f_source, f_blend) once the magnification is known, so the search is
3-dimensional and takes seconds on 38,000 points; s, q and alpha are left
at defaults.yaml's generic values (log_s = 0, i.e. s = 1) and the sampler
is asked to find the planet itself.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc18_common as C  # noqa: E402


def find_seed(curves, verbose=True):
    """The component's PSPL peak finder (peakfind.find_pspl_seed), with the
    primary-first pass on.  Returns its dict plus the data's time span."""
    from exozippy.components.mulensing import peakfind

    seed = peakfind.find_pspl_seed(curves)
    if seed is None:
        raise SystemExit("no fittable light curve: fewer than 4 usable epochs")
    t_all = np.concatenate([c[0] for c in curves])
    seed["t_span"] = (float(t_all.min()), float(t_all.max()))
    if verbose:
        an = seed.get("anomaly")
        if an:
            print(
                "  anomaly  : t_0=%.4f FWHM %.2f d masked (%.4f..%.4f); the "
                "seed below is the broader event behind it"
                % (an["t_0"], an["fwhm"], an["window"][0], an["window"][1]),
                flush=True,
            )
        print(
            "  seed     : t_0=%.4f u_0=%.4f t_E=%.3f  chi2=%.1f (%s)"
            % (
                seed["t_0"],
                seed["u_0"],
                seed["t_E"],
                seed["chi2"],
                "converged" if seed["converged"] else "NOT converged",
            ),
            flush=True,
        )
    return seed


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
    # sampler can reach, clipped to the data so the window is never empty;
    # a masked anomaly is a caustic feature by construction, so its window
    # is included too (226's sits at +3.8 t_E).  Only the mulensmodel
    # backend reads this list; the default vbm_direct always runs the
    # binary-lens solver.
    lo = max(seed["t_span"][0], t_0 - 2.0 * t_E)
    hi = min(seed["t_span"][1], t_0 + 2.0 * t_E)
    if seed.get("anomaly"):
        lo = min(lo, seed["anomaly"]["window"][0])
        hi = max(hi, seed["anomaly"]["window"][1])

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
            "writer": "examples/DC2018/dc18_seed.py (peakfind.find_pspl_seed, no MMEXOFAST)",
            "pspl_chi2": seed["chi2"],
            "anomaly": seed.get("anomaly"),
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
