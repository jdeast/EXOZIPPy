"""
Generate bolometric-correction tables for arbitrary SVO filters.

Integrates the shipped model spectra through SVO filter profiles
(downloading/caching profiles and Vega zeropoints as needed) and writes
per-feh BC files in the layout components/sed/bc_grid.py loads. Existing
facility files gain new columns without their existing columns changing.

Examples:
    poetry run python scripts/make_bc_tables.py Generic/Cousins.I
    poetry run python scripts/make_bc_tables.py Generic/Bessell.V SLOAN/SDSS.g
    poetry run python scripts/make_bc_tables.py --model NextGen Kepler/Kepler.K
"""

import argparse
import logging

from exozippy.components.sed.make_bc import make_bc_tables


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "filters", nargs="+",
        help="SVO filter IDs, e.g. Generic/Cousins.I 2MASS/2MASS.J")
    parser.add_argument(
        "--model", default="NextGen",
        help="Spectral model whose grid to integrate (default: NextGen)")
    parser.add_argument(
        "--bc-root", default=None,
        help="Root of the models tree (default: the installed package's)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kwargs = {"model": args.model}
    if args.bc_root:
        kwargs["bc_root"] = args.bc_root
    written = make_bc_tables(args.filters, **kwargs)
    print(f"Wrote {len(written)} file(s):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
