#!/usr/bin/env python3
"""Collect every DC2018 event's results into one table (one LC per row).

    poetry run python collect_results.py [--out-dir events] [--csv dc2018_summary.csv]

For each event directory under --out-dir this gathers:
  - status (ok / failed / running / missing pieces),
  - the event class from the challenge's answer key,
  - per parameter (t_0, u_0, t_E, rho, s, q, alpha): the EXOZIPPy posterior
    value, +/- errors, truth, and the sigma pull from comparison.csv,
  - convergence: overall max r_hat and min ess (bulk/tail) across all
    sampled/derived variables in <prefix>_summary.txt, plus per-parameter
    r_hat/ess for the lens variables that appear there (alpha is sampled as
    xalpha/yalpha, so its columns report the worse of the two; t_E/q/rho are
    pure expressions and have no chain diagnostics of their own),
  - the per-instrument err_scale posteriors.

Output: a CSV (default dc2018_summary.csv) and a compact stdout table.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc

# summary.txt variable(s) providing chain diagnostics for each parameter
SUMMARY_VARS = {
    "t_0": ["lens.t_0"],
    "u_0": ["lens.u_0"],
    "t_E": ["lens.t_E"],
    "rho": ["lens.rho"],
    "s": ["lens.s"],
    "q": ["lens.q"],
    "alpha": ["lens.xalpha", "lens.yalpha"],
}


def collect_event(event_dir):
    """One summary row (dict) for one event directory."""
    event = int(event_dir.name)
    row = {"event": event, "status": "missing"}

    status_file = event_dir / "status.txt"
    if status_file.exists():
        row["status"] = status_file.read_text().splitlines()[0].strip()

    fitresults = event_dir / "fitresults"
    prefix = fitresults / f"DC2018_{event:03d}"

    comparison = event_dir / "comparison.csv"
    if comparison.exists():
        with open(comparison, newline="") as f:
            for r in csv.DictReader(f):
                p = r["param"]
                row.setdefault("class", r.get("class"))
                if p in dc.PARAMS:
                    for src, dst in (
                        ("exozippy", p),
                        ("exo_err_hi", f"{p}_err_hi"),
                        ("exo_err_lo", f"{p}_err_lo"),
                        ("truth", f"{p}_truth"),
                        ("exo_pull", f"{p}_pull"),
                    ):
                        v = r.get(src)
                        row[dst] = float(v) if v not in (None, "") else None
                elif ".err_scale" in p:
                    v = r.get("exozippy")
                    # "comp.Inst.err_scale" (per-instrument) or
                    # "comp.err_scale" (single-element component)
                    parts = p.split(".")
                    inst = parts[1] if len(parts) == 3 else parts[0]
                    row[f"err_scale_{inst}"] = (
                        float(v) if v not in (None, "") else None
                    )

    summary = Path(str(prefix) + "_summary.txt")
    if summary.exists():
        overall, per_var = dc.read_summary_diagnostics(summary)
        row.update(overall)
        for p, names in SUMMARY_VARS.items():
            entries = [per_var[n] for n in names if n in per_var]
            if not entries:
                continue
            row[f"{p}_rhat"] = max(e[0] for e in entries)
            row[f"{p}_ess_bulk"] = min(e[1] for e in entries)

    n_seeds = Path(str(prefix) + "_mmexofast.json")
    if n_seeds.exists():
        try:
            row["n_mmxf_solutions"] = len(dc.read_mmexofast_solutions(n_seeds))
        except Exception:
            pass
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="events")
    ap.add_argument("--csv", default="dc2018_summary.csv")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    event_dirs = sorted(
        (d for d in out_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if not event_dirs:
        raise SystemExit(f"No event directories under {out_dir}/")

    rows = [collect_event(d) for d in event_dirs]

    # Stable, readable column order: identity, per-param blocks, diagnostics.
    keys = ["event", "class", "status"]
    for p in dc.PARAMS:
        keys += [
            p,
            f"{p}_err_hi",
            f"{p}_err_lo",
            f"{p}_truth",
            f"{p}_pull",
            f"{p}_rhat",
            f"{p}_ess_bulk",
        ]
    keys += ["rhat_max", "ess_bulk_min", "ess_tail_min", "n_mmxf_solutions"]
    extra = sorted({k for r in rows for k in r} - set(keys))
    keys += extra

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.csv} ({len(rows)} events)")

    # Compact stdout view
    def _n(row, key, fmt="{:.3g}"):
        v = row.get(key)
        return fmt.format(v) if isinstance(v, float) else "--"

    hdr = (
        f"{'event':>5} {'class':<8} {'status':<8} {'rhat':>6} {'ess':>7} "
        + " ".join(f"{p + '_pull':>11}" for p in dc.PARAMS)
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['event']:>5} {str(r.get('class', '--')):<8} "
            f"{r['status']:<8} {_n(r, 'rhat_max', '{:.2f}'):>6} "
            f"{_n(r, 'ess_bulk_min', '{:.0f}'):>7} "
            + " ".join(
                f"{_n(r, p + '_pull', '{:+.2f}'):>11}" for p in dc.PARAMS
            )
        )


if __name__ == "__main__":
    main()
