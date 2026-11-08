"""Speed half of the runtime-spectral-SED feasibility study: PyTensor
logp + gradient time of the BC-table path against the runtime spectral
paths, as a function of the number of bands.

  table  the shipped path: bc_grid.RegularGridInterpolator over
         (teff, logg, feh, av) with the n_band BCs as a trailing axis --
         16 corners x n_band.
  specA  generic binned spectrum: the same interpolator over (teff, logg,
         feh) with N_lambda wavelengths trailing (8 corners x N_lambda),
         reddened by exp(-A_V (a + b/R_V)/1.086) on the grid and
         integrated by a fixed [n_band x N_lambda] matrix.
  specB  per-filter weighted bins: (teff, logg, feh) with n_band x n_bin
         trailing, reddened per bin centre and summed per band.

Grid sizes are the shipped ones (60 x 12 x 11 nodes, 20 A_V). Values are
random: only the arithmetic is timed. One star, one thread, C backend.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pytensor
import pytensor.tensor as pt

from exozippy.components.sed.bc_grid import RegularGridInterpolator

rng = np.random.default_rng(1)
TEFF = np.concatenate(
    [np.arange(2600, 7000, 100), np.arange(7000, 10001, 200)]
).astype(float)
LOGG = np.arange(0.0, 5.51, 0.5)
FEH = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.5])
AV = np.array(
    [
        0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.3,
        0.4,
        0.6,
        0.8,
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20.0,
    ]
)
NBANDS = [1, 2, 4, 8, 16, 32, 64]


def draws(n):
    return [
        (
            rng.uniform(3000, 9500),
            rng.uniform(0.6, 5.2),
            rng.uniform(-3.5, 0.4),
            rng.uniform(0.0, 15.0),
            rng.uniform(2.2, 4.0),
        )
        for _ in range(n)
    ]


def time_fn(fn, args_list):
    for a in args_list[:20]:
        fn(*a)
    t0 = time.perf_counter()
    for a in args_list:
        fn(*a)
    return (time.perf_counter() - t0) / len(args_list)


def build_table(nb):
    vals = pytensor.shared(
        rng.normal(size=(len(TEFF), len(LOGG), len(FEH), len(AV), nb))
    )
    interp = RegularGridInterpolator([TEFF, LOGG, FEH, AV], vals)
    t, g, z, a, rv = (pt.dscalar(n) for n in ("t", "g", "z", "a", "rv"))
    bc = interp.evaluate(pt.stack([t, g, z, a]))[0]
    w = pt.as_tensor_variable(rng.normal(size=nb))
    logp = pt.sum(w * bc)
    grads = pytensor.grad(logp, [t, g, z, a])
    return pytensor.function(
        [t, g, z, a, rv], [logp, *grads], on_unused_input="ignore"
    )


def build_specA(nb, nlam):
    vals = pytensor.shared(
        rng.uniform(0.5, 1.5, size=(len(TEFF), len(LOGG), len(FEH), nlam))
    )
    interp = RegularGridInterpolator([TEFF, LOGG, FEH], vals)
    ca = pt.as_tensor_variable(rng.uniform(0.1, 2.0, size=nlam))
    cb = pt.as_tensor_variable(rng.uniform(-0.5, 1.0, size=nlam))
    M = pt.as_tensor_variable(rng.uniform(0, 1, size=(nb, nlam)))
    t, g, z, a, rv = (pt.dscalar(n) for n in ("t", "g", "z", "a", "rv"))
    spec = interp.evaluate(pt.stack([t, g, z]))[0]
    red = spec * pt.exp(-a * (ca + cb / rv) / 1.086)
    flux = pt.dot(M, red)
    bc = 2.5 * pt.log10(flux) - 10 * pt.log10(t)
    w = pt.as_tensor_variable(rng.normal(size=nb))
    logp = pt.sum(w * bc)
    grads = pytensor.grad(logp, [t, g, z, a, rv])
    return pytensor.function([t, g, z, a, rv], [logp, *grads])


def build_specB(nb, nbin):
    vals = pytensor.shared(
        rng.uniform(0.5, 1.5, size=(len(TEFF), len(LOGG), len(FEH), nb * nbin))
    )
    interp = RegularGridInterpolator([TEFF, LOGG, FEH], vals)
    ca = pt.as_tensor_variable(rng.uniform(0.1, 2.0, size=nb * nbin))
    cb = pt.as_tensor_variable(rng.uniform(-0.5, 1.0, size=nb * nbin))
    t, g, z, a, rv = (pt.dscalar(n) for n in ("t", "g", "z", "a", "rv"))
    G = interp.evaluate(pt.stack([t, g, z]))[0]
    red = G * pt.exp(-a * (ca + cb / rv) / 1.086)
    flux = pt.sum(red.reshape((nb, nbin)), axis=1)
    bc = 2.5 * pt.log10(flux) - 10 * pt.log10(t)
    w = pt.as_tensor_variable(rng.normal(size=nb))
    logp = pt.sum(w * bc)
    grads = pytensor.grad(logp, [t, g, z, a, rv])
    return pytensor.function([t, g, z, a, rv], [logp, *grads])


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "out",
        nargs="?",
        default="runtime_sed_benchmark.json",
        help="JSON file the timings are written to",
    )
    ap.add_argument(
        "n_eval",
        nargs="?",
        type=int,
        default=400,
        help="evaluations timed per case",
    )
    opts = ap.parse_args()
    out_path = opts.out
    args = draws(opts.n_eval)
    rows = []
    for nb in NBANDS:
        cases = [("table", lambda: build_table(nb), None)]
        for nlam in (1000, 4000, 16000):
            cases.append(
                (
                    f"specA_N{nlam}",
                    lambda nlam=nlam: build_specA(nb, nlam),
                    nlam,
                )
            )
        for nbin in (5, 15):
            cases.append(
                (
                    f"specB_bin{nbin}",
                    lambda nbin=nbin: build_specB(nb, nbin),
                    nbin,
                )
            )
        for name, build, p in cases:
            t0 = time.perf_counter()
            fn = build()
            tc = time.perf_counter() - t0
            dt = time_fn(fn, args)
            rows.append(
                {
                    "nb": nb,
                    "path": name,
                    "us_per_eval": dt * 1e6,
                    "compile_s": tc,
                }
            )
            print(
                f"nb={nb:3d} {name:14s} {dt * 1e6:9.1f} us/eval (logp+grad)  compile {tc:5.1f} s",
                flush=True,
            )
            del fn
    json.dump(rows, open(out_path, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
