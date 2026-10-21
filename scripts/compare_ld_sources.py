"""Compare the PHOENIX limb-darkening tables with the Claret tables where
both exist: the quadratic (u1, u2) and linear u at a set of (teff, logg,
feh) nodes, A_V = 0, for the bands both tabulate.  Numbers for the PR
and the doc; not a test (the real tables are not in the tree).

    python scripts/compare_ld_sources.py
"""

import numpy as np

from exozippy.components.band import claret as cl
from exozippy.components.band import ldgrid
from exozippy.models.Claret import claret_tables as ct

BANDS = [
    ("TESS", "TESS"),
    ("V", "V"),
    ("Kepler", "Kepler"),
    ("I", "I"),
    ("2MASS/2MASS.J", "J"),
    ("Sloang", "Sloang"),
]
NODES = [
    (5800, 4.5, 0.0),
    (4500, 4.5, 0.0),
    (3600, 4.8, 0.0),
    (6500, 4.2, -0.3),
    (5000, 3.0, -1.0),
]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model-root",
        default=None,
        help="model root holding models/PHOENIX/LDs (default: the package)",
    )
    args = parser.parse_args(argv)
    if args.model_root:
        ldgrid.DEFAULT_TABLE_ROOT = args.model_root
    print(
        f"{'band':14s} {'node':22s} {'PHOENIX u1,u2':>18s} {'Claret a,b':>18s} {'d(u1+u2)':>9s} {'PHX u':>7s} {'Clr u':>7s}"
    )
    diffs = []
    for label, cband in BANDS:
        try:
            pq = ldgrid.load_ld_table(label, "quadratic")
            pl = ldgrid.load_ld_table(label, "linear")
        except ValueError as exc:
            print(f"{label:14s} no PHOENIX table: {exc}")
            continue
        cq = ct.load_table(cband, "quadratic")
        try:
            clin = ct.load_table(cband, "linear")
        except Exception:
            clin = None
        for t, g, z in NODES:
            inside = all(
                lo <= v <= hi
                for v, (lo, hi) in zip(
                    (t, g, z), (ct.BOX["teff"], ct.BOX["logg"], ct.BOX["feh"])
                )
            )
            if not inside:
                continue
            p = ldgrid.interpolate(pq, t, g, z, 0.0)
            c = cl.interpolate(cq, t, g, z)
            pu = ldgrid.interpolate(pl, t, g, z, 0.0)[0]
            cu = (
                cl.interpolate(clin, t, g, z)[0]
                if clin is not None
                else np.nan
            )
            diffs.append(
                (
                    p[0] - c[0],
                    p[1] - c[1],
                    (p[0] + p[1]) - (c[0] + c[1]),
                    pu - cu,
                )
            )
            print(
                f"{label:14s} {str((t, g, z)):22s} {p[0]:8.3f} {p[1]:8.3f}  {c[0]:8.3f} {c[1]:8.3f}  {(p[0] + p[1]) - (c[0] + c[1]):+8.3f} {pu:7.3f} {cu:7.3f}"
            )
    d = np.array(diffs)
    print("\nPHOENIX - Claret over the nodes above: mean +/- sd")
    for k, name in enumerate(("u1", "u2", "u1+u2", "u_lin")):
        col = d[:, k][np.isfinite(d[:, k])]
        print(
            f"  {name:6s} {col.mean():+.3f} +/- {col.std():.3f}  (n={col.size})"
        )


if __name__ == "__main__":
    main()
