"""Start logp of a list of configs, term by term, written as JSON.

For TASK 10 (notes/supercomputer_queue.txt): the MMEXOFAST JSON seeds
move into the params files, and the JSON `sigmas` are dropped (JDE), so a
converted config need not start bit-identically.  This records the start
of each config on whatever exozippy is importable -- run it once with the
pre-#361 tree on PYTHONPATH against the unconverted configs ("old"), once
on master against the converted ones ("new") -- and dc18_start_logp.py
--compare prints the per-config table.

Reuses scripts/make_mulens_fixtures.record (the acceptance recorder) so
the decomposition is the one the fixtures use; --scripts names the
checkout whose recorder to import, which should be the same tree as the
exozippy on the path.
"""

import argparse
import importlib.util
import json
import os
import sys


def _record_fn(scripts_dir):
    spec = importlib.util.spec_from_file_location(
        "make_mulens_fixtures",
        os.path.join(scripts_dir, "make_mulens_fixtures.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.record


def _params_path(cfg_path):
    import yaml

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    pf = cfg.get("parameter_file")
    if pf is None:
        raise ValueError(f"{cfg_path}: no parameter_file")
    return (
        pf
        if os.path.isabs(pf)
        else os.path.join(os.path.dirname(cfg_path), pf)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument(
        "--scripts", help="<checkout>/scripts holding make_mulens_fixtures.py"
    )
    ap.add_argument("--out", help="directory for one JSON per config")
    ap.add_argument("--compare", nargs=2, metavar=("OLD_DIR", "NEW_DIR"))
    args = ap.parse_args()

    if args.compare:
        old_d, new_d = args.compare
        names = sorted(set(os.listdir(old_d)) | set(os.listdir(new_d)))
        print(
            f"{'config':58s}{'old logp':>14}{'new logp':>14}{'new-old':>10}  terms that moved > 0.01"
        )
        for n in names:
            if not n.endswith(".json"):
                continue
            o = (
                json.load(open(os.path.join(old_d, n)))
                if os.path.exists(os.path.join(old_d, n))
                else None
            )
            w = (
                json.load(open(os.path.join(new_d, n)))
                if os.path.exists(os.path.join(new_d, n))
                else None
            )
            label = n[:-5]
            if (
                o is None
                or w is None
                or "error" in (o or {})
                or "error" in (w or {})
            ):
                eo = (
                    (o or {}).get("error", "missing")
                    if (o is None or "error" in o)
                    else f"{o['total_logp']:.3f}"
                )
                ew = (
                    (w or {}).get("error", "missing")
                    if (w is None or "error" in w)
                    else f"{w['total_logp']:.3f}"
                )
                print(
                    f"{label:58s}  old: {str(eo)[:80]}\n{'':58s}  new: {str(ew)[:80]}"
                )
                continue
            moved = []
            for k in sorted(set(o["terms"]) | set(w["terms"])):
                a, b = o["terms"].get(k), w["terms"].get(k)
                if a is None or b is None:
                    moved.append(f"{k} ({'new' if a is None else 'gone'})")
                elif abs(a - b) > 0.01:
                    moved.append(f"{k} {b - a:+.2f}")
            print(
                f"{label:58s}{o['total_logp']:14.3f}{w['total_logp']:14.3f}{w['total_logp'] - o['total_logp']:+10.3f}  {'; '.join(moved)[:140]}"
            )
        return

    import exozippy

    print(f"exozippy from {exozippy.__file__}", flush=True)
    record = _record_fn(args.scripts)
    os.makedirs(args.out, exist_ok=True)
    for cfg in args.configs:
        cfg = os.path.abspath(cfg)
        label = (
            os.path.relpath(cfg, os.path.dirname(os.path.abspath(__file__)))
            .replace("/", "__")
            .replace(".yaml", "")
        )
        dest = os.path.join(args.out, label + ".json")
        try:
            r = record(cfg, _params_path(cfg))
            out = {
                "config": cfg,
                "total_logp": r["total_logp"],
                "terms": r["terms"],
                "n_terms": r["n_terms"],
            }
            print(
                f"{label:70s} {r['total_logp']:.3f}  ({r['n_terms']} terms)",
                flush=True,
            )
        except BaseException as exc:  # noqa: BLE001  SystemExit from the recorder included
            out = {
                "config": cfg,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
            print(f"{label:70s} FAILED {out['error'][:150]}", flush=True)
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
