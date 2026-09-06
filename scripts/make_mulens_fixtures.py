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

A fixture stores THREE things: the reconciled per-term logp decomposition,
the total, and THE START POINT ITSELF.  Storing the start is what lets the
two questions be asked separately, and they are very different questions:

  1. Does the relaxation engine solve the same START on another machine?
     It contains a numerical solve (`sp.nsolve`, config.py:3839), so a small
     platform difference here is a solver-convergence difference and is
     harmless -- a chain forgets its start.
  2. Is the LIKELIHOOD FUNCTION the same on another machine?  If not, the
     posterior depends on the hardware, which is serious.

--check answers question 1, STRICTLY: it regenerates on THIS machine and
demands bit-identity of both the start and every term.  Same-machine
reproducibility is the right bar for a same-machine tool.

tests/test_mulens_acceptance.py answers question 2: it replays the STORED
start, so both machines evaluate identical parameter values and any residual
is the likelihood function alone.  It can therefore hold a tight tolerance
instead of absorbing the solver's drift.

The first version of these fixtures conflated the two -- it compared logp at
each machine's OWN solved start -- and macOS differed by 3.2e-05 nats
(7.1e-10 relative) where Linux differed by 7.3e-12 (machine epsilon).
Widening the tolerance would have hidden precisely the interesting part.

STAGE-1 STATE (8.6.17): the shipped example configs carry the pre-split
spellings until stage 3, so most examples cannot build.  An example whose
name has a CONVERTED config committed under tests/fixtures/mulens/configs/
is recorded/checked from that copy instead (currently ob08092 and ob140939,
whose fixtures were also label-translated to the post-split naming with
values untouched).  Use --only to limit a run to those, e.g.

    python scripts/make_mulens_fixtures.py --check --only ob08092 --only ob140939
"""

import argparse
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mulens_acceptance import (  # noqa: E402
    compare,
    compare_points,
    decompose,
    raw_start,
)

from exozippy.system import System  # noqa: E402

# --check is a SAME-MACHINE tool, so it demands bit-identity.  The
# cross-machine tolerances live in tests/test_mulens_acceptance.py, which
# asks a different question; see the module docstring.

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "tests", "fixtures", "mulens")
# Stage-1 converted config copies (see the module docstring); an example
# with a copy here is recorded from it instead of from examples/.
CONVERTED = os.path.join(OUT, "configs")


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

        # Stage-1 substitution: a converted copy, when committed, is the
        # buildable form of the same example (see the module docstring).
        name = os.path.splitext(os.path.basename(cfg))[0]
        conv = os.path.join(CONVERTED, name, os.path.basename(cfg))
        if os.path.exists(conv):
            with open(conv) as fh:
                conv_doc = yaml.safe_load(fh) or {}
            conv_par = os.path.join(
                os.path.dirname(conv),
                str(conv_doc.get("parameter_file", name + ".params.yaml")),
            )
            cfg, par = conv, conv_par
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
        start = raw_start(system, model)
        parts, total, reconciles, summed = decompose(system, model, start)
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
        # The point every term above was evaluated at.  Stored so another
        # machine can replay it instead of solving its own -- see the module
        # docstring on why that separates two questions.
        "start": dict(sorted(start.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "limit to the named example(s); repeatable.  At stage 1 the "
            "unconverted examples cannot build, so --check without --only "
            "reports them as failures (expected until stage 3)."
        ),
    )
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    failures = []
    for cfg_path, par_path in microlensing_examples():
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        if args.only and name not in args.only:
            continue
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

            # STRICT: this is a same-machine rerun, so anything but
            # bit-identity is a real nondeterminism worth chasing.
            moved_pt = compare_points(old.get("start", {}), data["start"])
            moved, appeared, vanished = compare(old["terms"], data["terms"])
            same = not (moved_pt or moved or appeared or vanished)
            if moved_pt:
                for k, (_, _, d) in list(moved_pt.items())[:3]:
                    print(f"{name:32s} START {k} moved by {d!r}")
            for k, (ref, cur, d) in list(moved.items())[:3]:
                print(f"{name:32s} term {k} {ref!r} -> {cur!r}")
            if appeared or vanished:
                print(
                    f"{name:32s} TERM SET CHANGED "
                    f"+{sorted(appeared)} -{sorted(vanished)}"
                )
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
