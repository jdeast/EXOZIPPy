"""Record the pre-split acceptance fixtures for review 8.6.17 (stage 0).

Run from the repo root:

    python scripts/make_mulens_fixtures.py            # write tests/fixtures/mulens
    python scripts/make_mulens_fixtures.py --check    # regenerate and compare

The split collapses parameters that are stored per source but physically
singular, so start logp WILL move and byte-identity is not the acceptance
currency.  These fixtures are: for every shipped microlensing example, the
reconciled per-term logp decomposition at the start point.  During the
refactor each stage is accepted by explaining every delta term by term
against them.

--check regenerates and compares WITH A TOLERANCE.  Nothing here is
byte-comparable and the first version was wrong to try: these are float64
sums over thousands of epochs, so a different CPU, BLAS or PyTensor
compiledir reorders them and the last bits differ.  Measured against CI:
7.3e-12 and 2.0e-10 absolute on ~1e4 terms, i.e. 1.6e-16 and 7.6e-15
relative.

Note what --check can and cannot establish.  It reruns on ONE machine, so it
proves the pipeline is deterministic HERE; it says nothing about another
CPU.  Same-machine reproducibility is a strictly weaker claim than
cross-machine, and treating the first as evidence for the second is what put
exact equality in these fixtures to begin with.
"""

import argparse
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mulens_acceptance import decompose  # noqa: E402

from exozippy.system import System  # noqa: E402

# NOTHING is compared to the byte -- see the module docstring.  These match
# tests/test_mulens_acceptance.py; keep them in step.
TERM_RTOL = 1e-10
TERM_ATOL = 1e-6

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "tests", "fixtures", "mulens")


def microlensing_examples():
    """Every shipped example whose config declares a `lens:` block.

    The params file is taken from the config's own `parameter_file:` key,
    NOT from the `<name>.params.yaml` naming convention.  That convention is
    how System is usually driven but it is not a rule: examples/ogle0383
    ships `OGLE_0383LD.yaml` naming `OGLE_0383.paramsLD.yaml`, and a
    convention-based scan silently skips it.  A fixture set that quietly
    omits an example is the same failure as an empty one -- it just looks
    healthier.
    """
    found = []
    for cfg in sorted(
        glob.glob(os.path.join(ROOT, "examples", "*", "*.yaml"))
    ):
        if cfg.endswith(".params.yaml"):
            continue
        try:
            with open(cfg) as fh:
                doc = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if not isinstance(doc, dict) or "lens" not in doc:
            continue
        named = doc.get("parameter_file")
        par = (
            os.path.join(os.path.dirname(cfg), str(named))
            if named
            else cfg[:-5] + ".params.yaml"
        )
        if not os.path.exists(par):
            print(
                f"SKIP {os.path.relpath(cfg, ROOT)}: no params file at "
                f"{os.path.relpath(par, ROOT)}"
            )
            continue
        found.append((cfg, par))
    return found


def record(cfg_path, par_path):
    d = os.path.dirname(cfg_path)
    cwd = os.getcwd()
    try:
        os.chdir(d)
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        with open(par_path) as fh:
            par = yaml.safe_load(fh) or {}
        system = System(cfg, par)
        system.prepare()
        model = system.build_model()
        parts, total, reconciles, summed = decompose(system, model)
    finally:
        os.chdir(cwd)

    if not reconciles:
        raise SystemExit(
            f"REFUSING to write a fixture for {cfg_path}: the decomposition "
            f"does not reconcile (sum {summed!r} vs logp {total!r}). An "
            f"instrument that cannot be shown to add up produces confident "
            f"wrong attributions; fix it before recording anything."
        )
    return {
        "config": os.path.relpath(cfg_path, ROOT),
        "params": os.path.relpath(par_path, ROOT),
        "total_logp": total,
        "n_terms": len(parts),
        "terms": dict(sorted(parts.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    failures = []
    for cfg_path, par_path in microlensing_examples():
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        dest = os.path.join(OUT, name + ".json")
        try:
            data = record(cfg_path, par_path)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"{name:32s} FAILED {type(exc).__name__}: {exc}")
            failures.append(name)
            continue

        if args.check:
            if not os.path.exists(dest):
                print(f"{name:32s} MISSING fixture")
                failures.append(name)
                continue
            with open(dest) as fh:
                old = json.load(fh)
            same = old["terms"].keys() == data["terms"].keys()
            if same:
                for k, v in data["terms"].items():
                    ref = old["terms"][k]
                    ok = abs(v - ref) <= TERM_ATOL + TERM_RTOL * abs(ref)
                    if not ok:
                        same = False
                        print(f"{name:32s} term {k} {ref!r} -> {v!r}")
                        break
            else:
                print(f"{name:32s} TERM SET CHANGED")
            print(f"{name:32s} {'ok' if same else 'DIFFERS'}")
            if not same:
                failures.append(name)
        else:
            with open(dest, "w") as fh:
                json.dump(data, fh, indent=1, sort_keys=True)
                fh.write("\n")
            print(
                f"{name:32s} {data['n_terms']:3d} terms  logp {data['total_logp']!r}"
            )

    if failures:
        raise SystemExit("failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
