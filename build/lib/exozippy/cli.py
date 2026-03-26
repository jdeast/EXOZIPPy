from __future__ import annotations

import argparse
from pathlib import Path
import sys



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="exozippy")
    p.add_argument("fit_yaml", type=str, help="Top-level fit config YAML (e.g. kelt4.yaml)")
    p.add_argument("--outdir", type=str, default=None, help="Override output directory")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation (not implemented yet)")
    p.add_argument("--dry-run", action="store_true", help="Validate config/data/model wiring; do not sample")
    args = p.parse_args(argv)

    fit_path = Path(args.fit_yaml).expanduser().resolve()
    if not fit_path.exists():
        print(f"ERROR: file not found: {fit_path}", file=sys.stderr)
        return 2

    # argparse handles --help before we import anything heavy
    if args.fit_yaml is None:
        p.print_help()
        return 0

    # import here so help works without all its dependencies
    from exozippy.run import run_fit
    run_fit(
        fit_yaml_path=fit_path,
        outdir_override=args.outdir,
        seed=args.seed,
        make_plots=not args.no_plots,
        dry_run=args.dry_run,
    )
    return 0
