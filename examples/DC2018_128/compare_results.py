"""
Compare DC2018 event 128 results across three sources:
  1. Simulated truth  — DC18Answers (MMEXOFAST/2018DataChallenge)
  2. MMEXOFAST fit    — mmexofast.json in this directory
  3. EXOZIPPy fit     — fitresults/*_results.csv (most recently modified)

Output: human-readable table + CSV to stdout.

Usage:
    cd examples/DC2018_128
    poetry run python compare_results.py

Note on DC18Answers import:
    DC18_classes.py lives in MMEXOFAST/examples/ (not an installed package).
    It reads DATA_PATH from mmexofast.config at import time to build
    dir_ = DATA_PATH + "/../2018DataChallenge".  We redirect via sys.modules
    so dir_ resolves to the real data tree before the module-level code runs.
"""

import sys
import os
import csv
import json
import types
import numpy as np

# ── Monkey-patch mmexofast.config before DC18_classes runs its module-level code ──
_fake_cfg = types.ModuleType("mmexofast.config")
_fake_cfg.DATA_PATH = "/home/jeastman/python/MMEXOFAST/DC18Test"
sys.modules["mmexofast"] = types.ModuleType("mmexofast")
sys.modules["mmexofast.config"] = _fake_cfg

sys.path.insert(0, "/home/jeastman/python/MMEXOFAST/examples")
from DC18_classes import DC18Answers  # noqa: E402

EVENT = 128

# ── 1. Simulated truth via DC18Answers.get_model() ───────────────────────────
# Suppress the print inside get_model by redirecting stdout temporarily
import io, contextlib
answers = DC18Answers()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    model = answers.get_model(EVENT)
p = model.parameters
truth = {
    "t_0":   float(p.t_0),
    "u_0":   float(p.u_0),
    "t_E":   float(p.t_E),
    "rho":   float(p.rho),
    "s":     float(p.s),
    "q":     float(p.q),
    "alpha": float(p.alpha),
}

# ── 2. MMEXOFAST fit ──────────────────────────────────────────────────────────
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "mmexofast.json")) as f:
    mmx_data = json.load(f)

mmxf, mmxs = {}, {}
for sol_idx, sol in enumerate(mmx_data["fits"]):
    p = sol["parameters"]
    s = sol.get("sigmas", {})
    mmxf[sol_idx] = {k: p[k] for k in ("t_0", "u_0", "t_E", "rho", "s", "q", "alpha")}
    mmxs[sol_idx] = {
        "t_0":   s.get("t_0"),
        "u_0":   s.get("u_0"),
        "t_E":   s.get("t_E"),
        "rho":   p["rho"] * s["log_rho"] * np.log(10) if "log_rho" in s else None,
        "s":     p["s"]   * s["log_s"]   * np.log(10) if "log_s"   in s else None,
        "q":     p["q"]   * s["log_q"]   * np.log(10) if "log_q"   in s else None,
        "alpha": s.get("alpha"),
    }

# ── 3. EXOZIPPy posterior from results CSV ───────────────────────────────────
def _parse_csv(csv_path):
    """Return dict of parname → (value, up_err, low_err); err_* None if fixed."""
    result = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f, fieldnames=["parname", "value", "up_err", "low_err"])
        for row in reader:
            if row["parname"].startswith("#"):
                continue
            v   = float(row["value"])   if row["value"]   else None
            hi  = float(row["up_err"])  if row["up_err"]  else None
            lo  = float(row["low_err"]) if row["low_err"] else None
            result[row["parname"].strip()] = (v, hi, lo)
    return result

# CSV parname → comparison param name
CSV_MAPPING = {
    "lens.t_0":   "t_0",
    "lens.u_0":   "u_0",
    "lens.t_E":   "t_E",
    "lens.rho":   "rho",
    "lens.s":     "s",
    "lens.q":     "q",
    "lens.alpha": "alpha",
}

exo, exo_hi, exo_lo = {}, {}, {}
csv_file = None
csv_dir = os.path.join(here, "fitresults")
if os.path.isdir(csv_dir):
    candidates = [f for f in os.listdir(csv_dir) if f.endswith("_results.csv")]
    if candidates:
        candidates.sort(key=lambda f: os.path.getmtime(os.path.join(csv_dir, f)))
        csv_file = os.path.join(csv_dir, candidates[-1])

if csv_file:
    csv_data = _parse_csv(csv_file)
    for csv_key, param in CSV_MAPPING.items():
        if csv_key in csv_data:
            v, hi, lo = csv_data[csv_key]
            exo[param]    = v
            exo_hi[param] = hi
            exo_lo[param] = lo

# ── Sigma-pull helper ─────────────────────────────────────────────────────────
def _sigma_pull(truth_val, fit_val, err_hi, err_lo):
    """Return (truth - fit) / one-sided-sigma, using the error bar on side of discrepancy.

    err_hi is the +sigma (positive number); err_lo is the -sigma (negative number, or
    its absolute value — we take abs() to be safe).  Returns None if any value is missing.
    """
    if truth_val is None or fit_val is None:
        return None
    diff = truth_val - fit_val
    if diff >= 0:
        err = abs(err_hi) if err_hi is not None else None
    else:
        err = abs(err_lo) if err_lo is not None else None
    if not err:
        return None
    return diff / err

# ── 4. Print table ────────────────────────────────────────────────────────────
PARAMS = [
    ("t_0",   "JD",  "{:.4f}"),
    ("u_0",   "",    "{:.5f}"),
    ("t_E",   "d",   "{:.4f}"),
    ("rho",   "",    "{:.6f}"),
    ("s",     "",    "{:.6f}"),
    ("q",     "",    "{:.7f}"),
    ("alpha", "deg", "{:.3f}"),
]

n_mmx = len(mmxf)

def _fmt_sym(val, err, fstr):
    if val is None:
        return "—"
    s = fstr.format(val)
    if err is not None:
        s += " ± " + fstr.format(abs(err))
    return s

def _fmt_asym(val, hi, lo, fstr):
    if val is None:
        return "—"
    s = fstr.format(val)
    if hi is not None and lo is not None:
        s += f" +{fstr.format(abs(hi))}/{fstr.format(abs(lo))}"
    return s

def _fmt_sigma(sigma):
    if sigma is None:
        return "—"
    return f"{sigma:+.2f}σ"

val_w   = 32
sig_w   = 8
col_hdr = "Value" + " " * (val_w - 5) + "Δσ"
col_sep = "-" * val_w + " " + "-" * sig_w

hdrs = [f"{'Param':<8}", f"{'Unit':<5}", f"{'Truth (DC18)':<{val_w}}"]
for i in range(n_mmx):
    hdrs.append(f"{'MMEXOFAST sol '+str(i):<{val_w+1+sig_w}}")
hdrs.append(f"{'EXOZIPPy':<{val_w+1+sig_w}}")
header = "  ".join(hdrs)
print(header)
print("-" * len(header))

for pname, unit, fstr in PARAMS:
    tv = truth.get(pname)
    row_parts = [f"{pname:<8}", f"{unit:<5}", f"{_fmt_sym(tv, None, fstr):<{val_w}}"]

    for i in range(n_mmx):
        fv  = mmxf[i].get(pname)
        err = mmxs[i].get(pname)
        sig = _sigma_pull(tv, fv, err, err)  # symmetric error
        cell = f"{_fmt_sym(fv, err, fstr):<{val_w}} {_fmt_sigma(sig):<{sig_w}}"
        row_parts.append(cell)

    fv  = exo.get(pname)
    hi  = exo_hi.get(pname)
    lo  = exo_lo.get(pname)
    sig = _sigma_pull(tv, fv, hi, lo)
    cell = f"{_fmt_asym(fv, hi, lo, fstr):<{val_w}} {_fmt_sigma(sig):<{sig_w}}"
    row_parts.append(cell)

    print("  ".join(row_parts))

print()
if csv_file:
    print(f"EXOZIPPy source: {os.path.basename(csv_file)}")
else:
    print("No EXOZIPPy results CSV found — run the fit first.")
print("Note: alpha convention is unknown for DC18 truth values — no remapping applied.")
print("      EXOZIPPy uses the center of mass as the origin for alpha.")
print("      Direct comparison of alpha may be meaningless without convention alignment.")

# ── 5. CSV output ─────────────────────────────────────────────────────────────
print()
csv_header = ["param", "unit", "truth"]
for i in range(n_mmx):
    csv_header += [f"mmxf_sol{i}", f"mmxf_err_sol{i}", f"mmxf_sigma_sol{i}"]
csv_header += ["exozippy", "exozippy_err_hi", "exozippy_err_lo", "exozippy_sigma"]

writer = csv.DictWriter(sys.stdout, fieldnames=csv_header, lineterminator="\n")
writer.writeheader()
for pname, unit, fstr in PARAMS:
    tv  = truth.get(pname)
    row = {"param": pname, "unit": unit, "truth": tv if tv is not None else ""}
    for i in range(n_mmx):
        fv  = mmxf[i].get(pname)
        err = mmxs[i].get(pname)
        sig = _sigma_pull(tv, fv, err, err)
        row[f"mmxf_sol{i}"]       = fv  if fv  is not None else ""
        row[f"mmxf_err_sol{i}"]   = err if err is not None else ""
        row[f"mmxf_sigma_sol{i}"] = f"{sig:.4f}" if sig is not None else ""
    fv  = exo.get(pname)
    hi  = exo_hi.get(pname)
    lo  = exo_lo.get(pname)
    sig = _sigma_pull(tv, fv, hi, lo)
    row["exozippy"]        = fv  if fv  is not None else ""
    row["exozippy_err_hi"] = hi  if hi  is not None else ""
    row["exozippy_err_lo"] = lo  if lo  is not None else ""
    row["exozippy_sigma"]  = f"{sig:.4f}" if sig is not None else ""
    writer.writerow(row)
