"""Harvest the magnification calls a PTDE-async fit rejected for exceeding
eval_timeout, as EXACT VBMicrolensing input vectors, and time each one
through VBM alone (review 2.6.12: the fixed kernel's slow tail).

The fit log prints, for every rejected call, the raw sampler coordinates
("raw params: {...}").  Those decode to physical values only under the
whitening the fit persisted, so this restores it from the run's
whitening JSON (load_whitening), rebuilds the model, finds the
VBMDirectMagOp apply node in the likelihood graph and compiles its first
input -- the p vector the Op hands VBM -- as a function of the raw
coordinates.  Each vector is then evaluated in a child process with
vbm_reproducer.run_case's layout (same epochs, same centre-of-mass
geometry, Multipoly) under a wall-clock budget, and everything is written
to a JSON beside the stress-test outputs.

    python harvest_slow_calls.py <fit log> <whitening json> <out json>
"""

import ast
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET_S = 900.0


def parse_log(path):
    """[(rung, chain, {raw name: [values]})] for every rejected call."""
    out = []
    lines = open(path, errors="replace").read().splitlines()
    for i, ln in enumerate(lines):
        m = re.search(
            r"exceeded eval_timeout=\S+ at rung (\d+) chain (\d+)", ln
        )
        if not m:
            continue
        for k in range(i + 1, min(i + 4, len(lines))):
            if "raw params:" in lines[k]:
                txt = lines[k].split("raw params:", 1)[1].strip()
                txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
                out.append(
                    (int(m.group(1)), int(m.group(2)), ast.literal_eval(txt))
                )
                break
    return out


def build(whitening_json):
    import pytensor
    from pytensor.graph.traversal import ancestors

    from exozippy.components.mulensing.op import VBMDirectMagOp
    from exozippy.system import System
    from exozippy.whitening import load_whitening

    os.chdir(HERE)
    config = yaml.safe_load(open("KMT-2021-BLG-1122L.yaml"))
    user_params = yaml.safe_load(open(config["parameter_file"]))
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    print("whitening:", load_whitening(system, whitening_json))
    vv = list(model.value_vars)
    nodes = [
        a.owner
        for a in ancestors(model.replace_rvs_by_values([model.logp(sum=True)]))
        if a.owner is not None and isinstance(a.owner.op, VBMDirectMagOp)
    ]
    if not nodes:
        raise RuntimeError(
            "no VBMDirectMagOp apply node in the likelihood graph"
        )
    seen, uniq = set(), []
    for n in nodes:
        if id(n) not in seen:
            seen.add(id(n))
            uniq.append(n)
    print(
        f"{len(uniq)} magnification Op node(s); compiling the p vector of the first"
    )
    f_p = pytensor.function(vv, uniq[0].inputs[0], on_unused_input="ignore")
    return system, model, vv, f_p


def main(log_path, whitening_json, out_json):
    calls = parse_log(log_path)
    print(f"{len(calls)} rejected calls parsed from {log_path}")
    system, model, vv, f_p = build(whitening_json)
    vnames = [v.name for v in vv]
    records = []
    for rung, chain, raw in calls:
        args = []
        for n in vnames:
            if n not in raw:
                raise KeyError(f"log record lacks {n}")
            args.append(np.asarray(raw[n], dtype=float))
        p = np.asarray(f_p(*args), dtype=float)
        records.append({"rung": rung, "chain": chain, "p": p.tolist()})
    json.dump(
        {
            "source": os.path.basename(log_path),
            "n": len(records),
            "calls": records,
        },
        open(out_json, "w"),
        indent=1,
    )
    print(f"wrote {out_json}")
    # time each through VBM alone, in a child with a budget
    spec = os.path.join(HERE, "vbm_reproducer.py")
    for rec in records:
        case = json.dumps(rec["p"])
        t0 = time.time()
        code = (
            "import importlib.util, json, sys; "
            f"spec = importlib.util.spec_from_file_location('vr', {spec!r}); "
            "m = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(m); "
            f"A = m.run_case('Multipoly', json.loads({case!r})); print(float(A.max()))"
        )
        try:
            r = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=BUDGET_S,
            )
            dt = time.time() - t0
            status = f"{dt:.1f} s" + (
                ""
                if r.returncode == 0
                else f" rc={r.returncode} {(r.stderr.strip().splitlines() or [''])[-1][:80]}"
            )
        except subprocess.TimeoutExpired:
            dt = BUDGET_S
            status = f"> {BUDGET_S:.0f} s (killed)"
        rec["vbm_alone_seconds"] = dt
        rec["status"] = status
        p = rec["p"]
        print(
            f"rung {rec['rung']:2d} chain {rec['chain']:2d}: {status:28s} s2 {p[6]:.3f} q2 {p[7]:.3g} a2 {p[8]:.1f} | s3 {p[9]:.3f} q3 {p[10]:.3g} a3 {p[11]:.1f} | rho {p[5]:.2e} u0 {p[1]:+.3f} tE {p[2]:.1f}",
            flush=True,
        )
    json.dump(
        {
            "source": os.path.basename(log_path),
            "n": len(records),
            "budget_s": BUDGET_S,
            "calls": records,
        },
        open(out_json, "w"),
        indent=1,
    )
    slow = [r for r in records if r["vbm_alone_seconds"] > 60]
    print(
        f"{len(slow)} of {len(records)} exceed 60 s through VBM alone; wrote {out_json}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:4])
