"""Our own microlensing seed finder: a PSPL peak fit, as params-file starts.

WHY THIS EXISTS.  Review 8.4.9 established that EXOZIPPy HAD NO PEAK
FINDER: the only source of t_0/u_0/t_E start values was an MMEXOFAST JSON,
so without one DC2018-128 started 1,445 days from its own peak.  JDE asked
for "our own peak finder and generic (defaults.yaml) values for logs and
logq", and this was the sweep's.  The component now carries the same
search (``peakfind.find_pspl_seed``) and runs it by default whenever a
params file gives no trajectory start, so a bare `exozippy` run is seeded
the same way.

WHAT IT WRITES.  A params-file FRAGMENT, ``events/<NNN>/DC2018_<NNN>_seed.
params.yaml``, that ``dc18_sweep_config.py`` merges into each generated
params file: per-seed ``initval:`` lists, one entry per seed.  Until
2026-10-01 it wrote the same numbers in MMEXOFAST's JSON shape for the
config's ``mmexofast:`` key; that key is gone (JDE 2026-10-01: "that
should be refactored to use the param file input"), and the fragment is
exactly what ``convert_mmexofast_json.py`` produced from those JSONs, so
every committed sweep config starts where it did.

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
at generic values (log_s = 0, i.e. s = 1) and the sampler is asked to find
the planet itself.

WHAT IT DELIBERATELY DOES NOT DO.  It does not mask outliers, because the
sweep runs the Hogg mixture likelihood on every light curve and a hard mask
would double-count that job.  It does not rescale errors: each
instrument's err_scale starts at 1.0 and is a fitted parameter.
"""

import argparse
import io
import os
import sys

import numpy as np
import yaml

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


def seed_params(t_0, u_0, t_E, instruments, generic=GENERIC):
    """The params-file entries for one event's seeds: one per-seed list per
    parameter (one entry per GENERIC companion), and err_scale = 1.0 on
    every instrument.  Paths are the sweep config's spellings."""
    n = len(generic)
    out = {
        "source.Source.t_0": {"initval": [float(t_0)] * n},
        "source.Source.u_0": {"initval": [float(u_0)] * n},
        "mulensevent.0.t_E": {"initval": [float(t_E)] * n},
        "source.Source.rho": {"initval": [float(g["rho"]) for g in generic]},
        "lens.Companion.log_s": {
            "initval": [float(np.log10(g["s"])) for g in generic]
        },
        "lens.Companion.alpha": {
            "initval": [float(g["alpha"]) for g in generic]
        },
        "lens.Companion.q": {"initval": [float(g["q"]) for g in generic]},
    }
    for inst in instruments:
        out["mulensinstrument.%s.err_scale" % inst] = {"initval": 1.0}
    return out


def write_seed_params(dest, params, header):
    """Write the fragment: `header` lines as comments, then the entries."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with io.open(dest, "w", encoding="utf-8") as fh:
        for line in header:
            fh.write(("# " + line).rstrip() + "\n")
        fh.write("\n")
        yaml.safe_dump(params, fh, sort_keys=False, default_flow_style=None)


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

    params = seed_params(t_0, u_0, t_E, ["Roman_%s" % b for b in bands])
    an = seed.get("anomaly")
    header = [
        "Seeds for DC2018 event %03d, written by examples/DC2018/dc18_seed.py"
        % args.event,
        "(peakfind.find_pspl_seed, chi2 = %.1f).  s/q/alpha/rho are generic"
        % seed["chi2"],
        "starts, NOT fitted.  dc18_sweep_config.py merges these into the",
        "params file it writes.",
    ]
    if an:
        header.append(
            "Anomaly masked for the primary search: t_0 = %.4f, FWHM %.2f d."
            % (an["t_0"], an["fwhm"])
        )
    ev3 = "%03d" % args.event
    dest = args.out or "events/%s/DC2018_%s_seed.params.yaml" % (ev3, ev3)
    write_seed_params(dest, params, header)
    print("wrote %s" % dest, flush=True)


if __name__ == "__main__":
    main()
