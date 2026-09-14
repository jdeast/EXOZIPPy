"""Does the model start where the user asked?  Report, or diagnose one miss.

THE CONTRACT: a value the user sets is either produced by the model or
explained (``ModelAuditor.check_user_starts``).  This script is the
measurement instrument for that contract, and the reason it is committed
rather than left as a scratch file is review item 3.14.22: any change to
which quantity absorbs a discrepancy has to be judged on its effect
across every shipped example, and rebuilding this harness from scratch is
what made the earlier attempts hard to evaluate.

Two modes:

    # every shipped example, one line per user value the model misses
    python scripts/start_value_report.py

    # one example, one parameter: ledger value, rank, who solved it,
    # element roles, bounds, and what the compiled graph produces
    python scripts/start_value_report.py --example ob09020 --param t_E

The second mode is what tells the failure classes apart.  A DERIVED
element (the roles it prints) cannot be set at all -- the value is an
expression, and only the sampled parameters under it can move it.  Failing
that: the ledger holding the user's value while the graph disagrees means
the DERIVATION cannot deliver it; the ledger holding something else means
the engine overwrote it, and ``solved by`` names the equation.

READ OFF THE COMPILED GRAPH.  Not ``Parameter.value`` (a draw from the
prior) and not ``Parameter.initval`` (the ledger, which is half of what is
under test) -- both look right while the model starts elsewhere.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pytensor
import yaml

# Import from the tree this script lives in, so it reports on the working
# copy rather than on whatever is installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from exozippy.diagnostics import ModelAuditor  # noqa: E402
from exozippy.system import System  # noqa: E402

EXAMPLES = os.path.join(_ROOT, "examples")

# Harness dirs, not single fits.
SKIP = {"validation", "DC2018"}


def _config_for(example_dir, name):
    """The top-level config, NOT a fragment.

    Deliberately prefers ``<dirname>.yaml``: picking the alphabetically
    first ``*.yaml`` grabs SED fragments like ``kelt4.sed.yaml``, whose
    ``name:`` entries are filter ids with dots and slashes that the
    instance-name validator rightly rejects.  That silently dropped three
    examples from the first sweep.
    """
    cands = [
        c
        for c in glob.glob(os.path.join(example_dir, "*.yaml"))
        if "params" not in os.path.basename(c)
        and "hpc" not in os.path.basename(c)
        and not os.path.basename(c).endswith(".2.yaml")
    ]
    preferred = os.path.join(example_dir, f"{name}.yaml")
    if preferred in cands:
        return preferred
    return sorted(cands)[0] if cands else None


def _build(example_dir, name):
    """Prepare + build one example, from inside its own directory."""
    cfg_path = _config_for(example_dir, name)
    if cfg_path is None:
        raise RuntimeError("no top-level config")
    cwd = os.getcwd()
    try:
        os.chdir(example_dir)
        with open(os.path.basename(cfg_path)) as fh:
            cfg = yaml.safe_load(fh)
        pf = cfg.get("parameter_file")
        params = {}
        if pf and os.path.exists(pf):
            with open(pf) as fh:
                params = yaml.safe_load(fh) or {}
        cfg["parameter_file"] = None
        # Never launch MMEXOFAST from a report: it wants a fitresults/ dir
        # and this is asking about the SEEDS, not running a fit.
        for e in cfg.get("mulensevent", []) or []:
            e["mmexofast"] = False

        system = System(cfg, params)
        system.prepare()
        model = system.build_model()
        start = system.get_raw_start(model)
        return system, model, start
    finally:
        os.chdir(cwd)


def sweep():
    # One counter per reason check_user_starts can return, so a reason added
    # there shows up in the summary line instead of being silently counted
    # and not printed.
    totals = {"examples": 0, "misses": 0}
    reasons = ("derived", "approximate", "overspecified")
    totals.update({r: 0 for r in reasons})
    for d in sorted(os.listdir(EXAMPLES)):
        path = os.path.join(EXAMPLES, d)
        if not os.path.isdir(path) or d in SKIP:
            continue
        if _config_for(path, d) is None:
            continue
        try:
            system, model, start = _build(path, d)
            misses = ModelAuditor(model, system, start).check_user_starts()
        except Exception as exc:
            print(
                f"=== {d} === BUILD FAILED: {type(exc).__name__}: "
                f"{str(exc)[:70]}"
            )
            sys.stdout.flush()
            continue
        totals["examples"] += 1
        totals["misses"] += len(misses)
        for f in misses:
            totals[f["reason"]] = totals.get(f["reason"], 0) + 1
        if not misses:
            print(f"=== {d} === clean")
        else:
            print(f"=== {d} ===")
            for f in misses:
                print(
                    "  %-34s %14.6g -> %-14.6g %+9.2f%%  %s"
                    % (
                        f["key"],
                        f["requested"],
                        f["produced"],
                        100 * f["rel"],
                        f["reason"],
                    )
                )
        sys.stdout.flush()

    print()
    print(
        "examples built %d | misses %d (%s)"
        % (
            totals["examples"],
            totals["misses"],
            ", ".join(
                "%s %d" % (r, totals.get(r, 0))
                for r in sorted(
                    k for k in totals if k not in ("examples", "misses")
                )
            ),
        )
    )
    return totals


def diagnose(name, needle):
    path = os.path.join(EXAMPLES, name)
    system, model, start = _build(path, name)
    cm = system.config_manager
    ledger = dict(getattr(cm, "_last_resolved", None) or {})
    prov = dict(getattr(cm, "_last_provenance", None) or {})
    solved = dict(getattr(cm, "_last_solved_by", None) or {})

    def _matches(path):
        """Index-insensitive: the ledger is keyed on the INDEX spelling
        (mulensevent.0.t_E) while a label is mulensevent.t_E, so a
        label-form needle would otherwise match nothing here."""
        if needle in path:
            return True
        parts = path.split(".")
        folded = ".".join(p for p in parts if not p.isdigit())
        return needle in folded

    print(f"=== LEDGER rows matching {needle!r} (INTERNAL units) ===")
    for k in sorted(ledger):
        if _matches(k):
            print(
                "  %-42s %16.8g  rank %-5s  %s"
                % (
                    k,
                    ledger[k],
                    prov.get(k, "-"),
                    solved.get(k, "(user / not solved)"),
                )
            )

    print()
    print("=== engine diagnostics ===")
    diags = getattr(cm, "diagnostics", None) or []
    for dg in diags:
        print("  [%s] %s" % (dg["severity"], dg["message"]))
        print("        paths: %s" % (dg["param_paths"],))
    if not diags:
        print("  (none)")

    targets = [p for p in system.get_all_parameters() if needle in p.label]
    print()
    print(f"=== PARAMETERS matching {needle!r} ===")
    for p in targets:
        n = int(np.prod(p.shape)) if p.shape != () else 1
        for i in range(n):
            print(
                "  %-34s idx %-3d sampled=%-5s derived=%-5s active=%-5s "
                "unit=%s"
                % (
                    p.get_display_label(i),
                    i,
                    p.element_is_sampled(i),
                    p.element_is_derived(i),
                    p.element_is_active(i),
                    p.unit,
                )
            )

    if targets:
        fn = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values([p.value for p in targets]),
            on_unused_input="ignore",
        )
        vals = fn(*[start[v.name] for v in model.value_vars])
        print()
        print("=== BUILT at the start point (USER units) ===")
        for p, v in zip(targets, vals):
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            conv = [
                float(p.from_internal(arr[i], index=i))
                for i in range(arr.size)
            ]
            print("  %-34s %s" % (p.label, np.array(conv)))

    print()
    print("start logp = %.8g" % float(model.compile_logp()(start)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--example", help="one example directory name (default: sweep all)"
    )
    ap.add_argument(
        "--param",
        default="",
        help="substring of a parameter label to diagnose; requires --example",
    )
    args = ap.parse_args()
    if args.example:
        diagnose(args.example, args.param)
    else:
        sweep()


if __name__ == "__main__":
    main()
