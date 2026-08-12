#!/usr/bin/env python3
"""CLI wrapper for exozippy.mkparam.

Writes the next params.yaml from a finished fit's trace: start values
(``initval``) at the trace MAP, with any priors and bounds the previous
params file carried copied across unchanged.

Usage:
    poetry run python scripts/mkparam.py ob140939.yaml
    poetry run python scripts/mkparam.py ob140939.yaml --trace my.nc --output out.yaml
"""

import argparse
import logging

from exozippy.mkparam import write_param_file


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Seed the next params.yaml from the MAP of a previous trace."
        )
    )
    parser.add_argument(
        "config", help="System config YAML (e.g. ob140939.yaml)"
    )
    parser.add_argument(
        "--trace", help="Trace file (default: <prefix>_trace.nc)", default=None
    )
    parser.add_argument(
        "--output",
        help=(
            "Output params file. Default: the config's parameter_file with "
            "its version incremented (foo.params.yaml -> foo.params.2.yaml). "
            "The input file is never overwritten."
        ),
        default=None,
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help=(
            "Number of multi-seed start points to emit "
            "(default: the config's `mkparam: {n_seeds:}`, else 1)."
        ),
    )
    args = parser.parse_args()

    out = write_param_file(
        args.config,
        trace_path=args.trace,
        output_path=args.output,
        n_seeds=args.n_seeds,
    )
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
