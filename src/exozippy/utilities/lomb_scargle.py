#!/usr/bin/env python3
"""Lomb-Scargle periodogram search over one or more EXOZIPPy RV files.

The standalone form of the search `RVInstrument.load_data` runs automatically
when a fit has no start value for the orbital period (see
``src/exozippy/components/globalsearch.py``).  Use it to inspect what the
automatic seeding will do, to look at a system the automatic path refuses
(multi-planet systems, where a peak carries no statement about WHICH orbit it
belongs to), or to get a params.yaml snippet to edit.

Usage:
    python -m exozippy.utilities.lomb_scargle rvs.dat [more.rv ...]
    python -m exozippy.utilities.lomb_scargle rvs.dat --min-period 2
"""

import argparse
import sys

import numpy as np

from ..components import globalsearch


def build_parser():
    p = argparse.ArgumentParser(
        prog="exozippy-lomb-scargle",
        description=(
            "Lomb-Scargle periodogram over EXOZIPPy RV files "
            "(whitespace-delimited: time, rv, rv_err, ...).  Each file's own "
            "mean velocity is removed before the search."
        ),
    )
    p.add_argument("files", nargs="+", help="Radial-velocity file(s).")
    p.add_argument(
        "--min-period",
        type=float,
        default=0.5,
        help="Shortest trial period in days (default 0.5).",
    )
    p.add_argument(
        "--max-period",
        type=float,
        default=None,
        help="Longest trial period in days (default: the data baseline).",
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


def read_rvs(paths):
    """Concatenate whitespace-delimited RV files with per-file means removed.

    Returns ``(time, rv_residual, err, inst_map)``.
    """
    times, rvs, errs, imap = [], [], [], []
    for i, path in enumerate(paths):
        data = np.genfromtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 2:
            raise ValueError(
                f"{path}: expected at least time and rv columns, got "
                f"{data.shape[1]}."
            )
        t, v = data[:, 0], data[:, 1]
        e = data[:, 2] if data.shape[1] > 2 else np.full_like(v, np.nan)
        times.append(t)
        rvs.append(v - np.nanmean(v))
        errs.append(e)
        imap.append(np.full(t.size, i, dtype=int))
    err = np.concatenate(errs)
    return (
        np.concatenate(times),
        np.concatenate(rvs),
        None if not np.isfinite(err).all() else err,
        np.concatenate(imap),
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    time, rv, err, inst_map = read_rvs(args.files)
    print(
        f"{len(args.files)} file(s), {time.size} velocities, "
        f"{time.max() - time.min():.4g} d baseline."
    )
    signal = globalsearch.lombscargle_search(
        time,
        rv,
        err,
        inst_map=inst_map,
        minimum_period=args.min_period,
        maximum_period=args.max_period,
        context="lomb_scargle",
    )
    if signal is None:
        print("No convincing periodicity (see the warning above).")
        return 1

    print(f"Lomb-Scargle detection: {signal.summary()}")
    print("")
    print("params.yaml snippet (start values only -- they do not enter the")
    print("likelihood and cannot move the posterior):")
    print(f"  orbit.{args.orbit}.period: {signal.period:.8g}")
    print(f"  orbit.{args.orbit}.tc: {signal.epoch:.8f}")
    print(f"  planet.{args.planet}.K: {signal.amplitude:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
