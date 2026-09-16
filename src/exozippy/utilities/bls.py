#!/usr/bin/env python3
"""Box Least Squares transit search over one or more EXOZIPPy light curves.

The standalone form of the search `Transit.load_data` runs automatically when
a fit has no start value for the orbital period (see
``src/exozippy/components/globalsearch.py``).  Use it to inspect what the
automatic seeding will do, to look at a system the automatic path refuses --
multi-planet systems, where a peak carries no statement about WHICH orbit it
belongs to -- or to get a params.yaml snippet to edit.

Usage:
    python -m exozippy.utilities.bls lightcurve.dat [more.dat ...]
    python -m exozippy.utilities.bls tess.dat --min-period 1 --max-period 20
"""

import argparse
import sys

import numpy as np

from ..components import globalsearch


def build_parser():
    p = argparse.ArgumentParser(
        prog="exozippy-bls",
        description=(
            "Box Least Squares transit search over EXOZIPPy light curve "
            "files (whitespace-delimited: time, flux, flux_err, ...)."
        ),
    )
    p.add_argument(
        "files",
        nargs="+",
        help="Light curve file(s); each is normalized by its own median flux.",
    )
    p.add_argument(
        "--min-period",
        type=float,
        default=None,
        help="Shortest trial period in days (default: astropy's choice).",
    )
    p.add_argument(
        "--max-period",
        type=float,
        default=None,
        help="Longest trial period in days (default: astropy's choice).",
    )
    p.add_argument(
        "--min-transits",
        type=int,
        default=2,
        help="Trial periods must fit at least this many transits (default 2).",
    )
    p.add_argument(
        "--orbit",
        default="b",
        help="Orbit name used in the printed params.yaml snippet.",
    )
    p.add_argument(
        "--planet",
        default="b",
        help="Planet name used in the printed params.yaml snippet.",
    )
    return p


def read_light_curves(paths):
    """Concatenate whitespace-delimited light curves, median-normalized."""
    times, fluxes, errs = [], [], []
    for path in paths:
        data = np.genfromtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 2:
            raise ValueError(
                f"{path}: expected at least time and flux columns, got "
                f"{data.shape[1]}."
            )
        t, f = data[:, 0], data[:, 1]
        e = data[:, 2] if data.shape[1] > 2 else np.full_like(f, np.nan)
        median = np.nanmedian(f)
        if not np.isfinite(median) or median == 0.0:
            median = 1.0
        times.append(t)
        fluxes.append(f / median)
        errs.append(e / median)
    err = np.concatenate(errs)
    return (
        np.concatenate(times),
        np.concatenate(fluxes),
        None if not np.isfinite(err).all() else err,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    time, flux, err = read_light_curves(args.files)
    print(
        f"{len(args.files)} file(s), {time.size} points, "
        f"{time.max() - time.min():.4g} d baseline."
    )
    signal = globalsearch.bls_search(
        time,
        flux,
        err,
        minimum_period=args.min_period,
        maximum_period=args.max_period,
        minimum_n_transit=args.min_transits,
        context="bls",
    )
    if signal is None:
        print("No convincing transit signal (see the warning above).")
        return 1

    print(f"BLS detection: {signal.summary()}")
    print("")
    print("params.yaml snippet (start values only -- they do not enter the")
    print("likelihood and cannot move the posterior):")
    print(f"  orbit.{args.orbit}.period: {signal.period:.8g}")
    print(f"  orbit.{args.orbit}.tc: {signal.epoch:.8f}")
    print(f"  planet.{args.planet}.p: {np.sqrt(signal.depth):.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
