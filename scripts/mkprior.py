#!/usr/bin/env python3
"""CLI wrapper for exozippy.mkprior.

Usage:
    poetry run python scripts/mkprior.py ob140939.yaml
    poetry run python scripts/mkprior.py ob140939.yaml --trace my.nc --output out.yaml
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

from exozippy.mkprior import backup_params, mkprior


def main():
    parser = argparse.ArgumentParser(
        description="Seed a params.yaml from the MAP of a previous trace."
    )
    parser.add_argument("config", help="System config YAML (e.g. ob140939.yaml)")
    parser.add_argument(
        "--trace", help="Trace file (default: <prefix>_trace.nc)", default=None
    )
    parser.add_argument(
        "--output",
        help="Output params.yaml (default: <runname>_params_N.yaml, N increments)",
        default=None,
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backing up the existing parameter file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not args.no_backup:
        backup = backup_params(config_path)
        if backup:
            print(f"Backed up params → {backup}")

    out = mkprior(config_path, trace_path=args.trace, output_path=args.output)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
