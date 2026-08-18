"""Shared helpers for the DC2018 (2018 Roman Data Challenge) workflow.

Data layout (--data-dir / $DC18_DATA, default the MMEXOFAST source checkout):
    n20180816.{W149,Z087}.WFIRST18.<NNN>.txt   flux light curves (BJD flux err)
    event_info.txt                              per-event RA/Dec (degrees)
    Answers/master_file.txt                     simulation truth, one row per
                                                event, POSITIONAL lookup:
                                                event N = row N-1
    Answers/wfirstColumnNumbers.txt             column names for master_file

Truth parsing replicates MMEXOFAST's examples/DC18_classes.py (DC18Answers)
without importing it, so this workflow needs only the data tree, not an
MMEXOFAST source checkout on sys.path. The DC18 time origin is JD 2458234.0:
master-file t0 is relative to it, the light curves are full BJD.

Alpha conventions: the master file's alpha is NOT in EXOZIPPy's convention
(EXOZIPPy measures alpha from the binary axis with the center of mass at the
origin; the identity mapping holds between MMEXOFAST and EXOZIPPy, but not
between either and the challenge's truth table). No mapping between them
exists either -- see ALPHA_IS_UNMAPPABLE below for the measurement -- so
alpha is reported WITHOUT a truth value and without a pull. u_0 is compared
in absolute value for a related reason (the truth table's u_0 carries a
trajectory-side sign the fits do not, and with parallax negligible the sign
is degenerate with alpha's anyway).
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = os.environ.get(
    "DC18_DATA",
    os.path.expanduser("~/python/MMEXOFAST/data/2018DataChallenge"),
)

DC18_TIME_ORIGIN = 2458234.0

PARAMS = ["t_0", "u_0", "t_E", "rho", "s", "q", "alpha"]

# results.csv parname -> comparison param
RESULTS_CSV_MAP = {f"lens.{p}": p for p in PARAMS}


def data_dir_or_raise(data_dir=None):
    d = Path(data_dir or DEFAULT_DATA_DIR)
    if not (d / "event_info.txt").exists():
        raise FileNotFoundError(
            f"DC18 data dir '{d}' has no event_info.txt. Point --data-dir "
            f"(or $DC18_DATA) at a 2018DataChallenge tree (see "
            f"examples/DC2018/README.md)."
        )
    return d


def available_events(data_dir):
    """Sorted event numbers that have a W149 light curve on disk."""
    events = []
    for f in Path(data_dir).glob("n20180816.W149.WFIRST18.*.txt"):
        tag = f.name.split(".")[-2]
        if tag.isdigit():
            events.append(int(tag))
    return sorted(events)


def light_curve_files(data_dir, event, bands=("W149", "Z087")):
    """{band: absolute path} for this event, raising on missing files."""
    files = {}
    for band in bands:
        f = Path(data_dir) / f"n20180816.{band}.WFIRST18.{event:03d}.txt"
        if not f.exists():
            raise FileNotFoundError(
                f"No {band} light curve for event {event}: {f}"
            )
        files[band] = str(f.resolve())
    return files


def event_coords(data_dir, event):
    """(ra_deg, dec_deg) from event_info.txt."""
    info = np.genfromtxt(
        Path(data_dir) / "event_info.txt",
        dtype=None,
        encoding="utf-8",
        names=["file", "num", "ra", "dec"],
        usecols=range(4),
    )
    idx = np.where(info["num"] == event)[0]
    if idx.size == 0:
        raise ValueError(f"Event {event} not found in event_info.txt")
    return float(info["ra"][idx[0]]), float(info["dec"][idx[0]])


def load_truth(data_dir, event):
    """Simulation truth for one event, with t_0 in full BJD.

    Returns (params_dict, class_label): params has the PARAMS keys, class is
    the challenge's event class scraped from the master-file row ('cassan'
    for the 2L1S planet sample, 'cv' for cataclysmic variables, ...).
    """
    ans = Path(data_dir) / "Answers"
    cols = np.genfromtxt(
        ans / "wfirstColumnNumbers.txt",
        dtype=None,
        encoding="utf-8",
        usecols=[0, 1],
        skip_header=2,
        names=["index", "name"],
    )
    names = [
        f"col{i}" if nm == "|" else nm for i, nm in enumerate(cols["name"])
    ]
    master = ans / "master_file.txt"
    df = pd.read_csv(
        master,
        names=names,
        usecols=range(len(names)),
        sep=r"\s+",
        skiprows=1,
    )
    row = df.iloc[event - 1]
    with open(master) as f:
        line = f.readlines()[event]  # +1 for the header line
    class_label = line.split(" ")[-2].split("_")[0]
    truth = {
        "t_0": float(row["t0"]) + DC18_TIME_ORIGIN,
        "u_0": float(row["u0"]),
        "t_E": float(row["tE"]),
        "rho": float(row["rhos"]),
        "s": float(row["s"]),
        "q": float(row["q"]),
        "alpha": float(row["alpha"]),
    }
    return truth, class_label


# ---------------------------------------------------------------------------
# Fit-output readers
# ---------------------------------------------------------------------------


def read_results_csv(csv_path):
    """Parse an EXOZIPPy *_results.csv into {param: (value, up, low)}.

    Handles both the single-solution header (# parname, value, up_err,
    low_err) and the multimodal one (# parname, mode, weight, value, up_err,
    low_err), preferring the combined 'all' mode. Also returns the
    per-instrument err_scale rows as a second dict.
    """
    with open(csv_path, newline="") as f:
        first = f.readline()
        has_mode = "mode" in first
        fields = (
            ["parname", "mode", "weight", "value", "up_err", "low_err"]
            if has_mode
            else ["parname", "value", "up_err", "low_err"]
        )
        reader = csv.DictReader(f, fieldnames=fields)
        rows = []
        for r in reader:
            if r["parname"] is None or r["parname"].startswith("#"):
                continue
            rows.append(r)

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    params, err_scales = {}, {}
    for r in rows:
        name = r["parname"].strip()
        mode = (r.get("mode") or "all").strip() if has_mode else "all"
        if has_mode and mode != "all":
            continue
        entry = (_f(r["value"]), _f(r["up_err"]), _f(r["low_err"]))
        if name in RESULTS_CSV_MAP:
            params[RESULTS_CSV_MAP[name]] = entry
        elif ".err_scale" in name:
            err_scales[name] = entry
    return params, err_scales


def read_summary_diagnostics(summary_path):
    """Parse a *_summary.txt (arviz summary) for convergence diagnostics.

    Returns (overall, per_var): overall = {"rhat_max": float, "ess_bulk_min":
    float, "ess_tail_min": float}, per_var = {varname: (r_hat, ess_bulk,
    ess_tail)}. Column POSITIONS vary across arviz versions (hdi vs eti
    intervals, mcse before or after the ess columns), so the header row --
    the line naming r_hat/ess_bulk -- drives the mapping; data rows are
    name + one number per header column, and everything else (banners,
    repeated headers) is skipped by shape.
    """
    per_var = {}
    cols = None
    with open(summary_path) as f:
        for line in f:
            tok = line.split()
            if "r_hat" in tok and "ess_bulk" in tok:
                cols = tok
                continue
            if cols is None or len(tok) != len(cols) + 1:
                continue
            try:
                rec = dict(zip(cols, (float(x) for x in tok[1:])))
            except ValueError:
                continue
            per_var[tok[0]] = (
                rec["r_hat"],
                rec["ess_bulk"],
                rec.get("ess_tail", rec["ess_bulk"]),
            )
    if not per_var:
        return {}, {}
    overall = {
        "rhat_max": max(v[0] for v in per_var.values()),
        "ess_bulk_min": min(v[1] for v in per_var.values()),
        "ess_tail_min": min(v[2] for v in per_var.values()),
    }
    return overall, per_var


def read_mmexofast_solutions(json_path):
    """[{param: (value, sigma_or_None)}] per fit, log sigmas linearized."""
    with open(json_path) as f:
        data = json.load(f)
    jd_offset = float(data.get("jd_offset", 0.0) or 0.0)
    ln10 = np.log(10.0)
    sols = []
    for fit in data.get("fits", []):
        p, s = fit.get("parameters", {}), fit.get("sigmas", {})
        sol = {}
        for name in PARAMS:
            if name not in p:
                continue
            val = float(p[name])
            if name == "t_0":
                val -= jd_offset
            if name in ("rho", "s", "q"):
                sig = s.get(f"log_{name}")
                sig = abs(val) * ln10 * float(sig) if sig is not None else None
            else:
                sig = s.get(name)
                sig = abs(float(sig)) if sig is not None else None
            sol[name] = (val, sig)
        sols.append(sol)
    return sols


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


# The master file's alpha has NO GLOBAL MAPPING onto the fitted convention.
# Measured, not assumed (examples/DC2018/dc18_alpha_convention.py): for each
# of the 44 events, alpha was scanned in MulensModel's convention at the
# truth values of t_0, u_0, t_E, rho, s and q with the fluxes fit linearly,
# giving the alpha the light curve itself prefers.  Against that reference
# every candidate transformation scatters like noise --
#
#   hypothesis                              circular R (1.0 = it IS the rule)
#   fit - alpha_DC                                    0.09
#   fit + alpha_DC  (reflection)                      0.10
#   either, with the galactic->equatorial PA removed  0.11 - 0.19
#   either, with PA(mu_rel) removed (a sky position   0.03 - 0.19
#     angle, which is how BAGLE defines its alpha)
#
# -- and restricting to the twelve events where the anomaly pins alpha
# hardest does not help (R = 0.22 / 0.41, against ~0.29 expected from 12
# random angles).  So this is a property of the answer key, not of a weak
# constraint or of the wrong sign branch.
#
# The old sign/+-180 search is therefore DELETED rather than improved.  It
# could not fail visibly: it always returned its closest candidate, so an
# unmappable truth came back as a confident number, and the resulting pull
# was reported alongside real ones.  On event 128 it printed a 2034-sigma
# alpha pull while EXOZIPPy's fitted alpha (307.686) sat 0.3 deg from the
# light curve's own optimum (308.0) -- a fabricated failure on a parameter
# that was right.
#
# Note the two conventions really are only determined up to a reflection
# here: with pi_E ~ 0.02 these events have negligible parallax, and without
# it (u_0, alpha) -> (-u_0, -alpha) is an exact mirror symmetry of the
# light curve.  Event 128 shows it exactly -- (+0.1418, 308.15) and
# (-0.1418, 51.85) give identical chi2 to every digit.
ALPHA_IS_UNMAPPABLE = (
    "no global mapping from the master file's alpha convention exists "
    "(measured over all 44 events); reported for the record, not compared"
)


def sigma_pull(truth_val, fit_val, err_hi, err_lo):
    """(truth - fit) / one-sided sigma, or None when not computable."""
    if truth_val is None or fit_val is None:
        return None
    diff = truth_val - fit_val
    err = err_hi if diff >= 0 else err_lo
    if err is None or abs(err) == 0 or not np.isfinite(err):
        return None
    return diff / abs(err)


def compare_event(event, data_dir, results_csv, mmx_json=None, out_csv=None):
    """Build the per-event truth/MMEXOFAST/EXOZIPPy comparison table.

    Returns a list of row dicts (one per parameter) and writes them as CSV
    when out_csv is given. Convention handling: u_0 is compared as |u_0|;
    alpha carries no truth and no pull at all (see ALPHA_IS_UNMAPPABLE) --
    its fitted value is still reported so the row is not silently missing.
    """
    truth, class_label = load_truth(data_dir, event)
    exo, err_scales = read_results_csv(results_csv)
    mmx_sols = (
        read_mmexofast_solutions(mmx_json)
        if mmx_json and Path(mmx_json).exists()
        else []
    )

    truth = dict(truth)
    notes = {p: "" for p in PARAMS}
    truth["u_0"] = abs(truth["u_0"])
    # Keep the master file's raw alpha out of the truth column entirely: a
    # number there is a claim that it is comparable, and it is not.
    notes["alpha"] = ALPHA_IS_UNMAPPABLE
    truth["alpha"] = None

    rows = []
    for p in PARAMS:
        val, hi, lo = exo.get(p, (None, None, None))
        row = {
            "event": event,
            "class": class_label,
            "param": p,
            "truth": truth.get(p),
            "exozippy": val,
            "exo_err_hi": hi,
            "exo_err_lo": lo,
            "exo_pull": sigma_pull(truth.get(p), val, hi, lo),
        }
        for k, sol in enumerate(mmx_sols):
            v, sig = sol.get(p, (None, None))
            row[f"mmxf_sol{k}"] = v
            row[f"mmxf_err_sol{k}"] = sig
            row[f"mmxf_pull_sol{k}"] = sigma_pull(truth.get(p), v, sig, sig)
        row["note"] = notes[p]
        rows.append(row)

    for name, (val, hi, lo) in sorted(err_scales.items()):
        rows.append(
            {
                "event": event,
                "class": class_label,
                "param": name,
                "exozippy": val,
                "exo_err_hi": hi,
                "exo_err_lo": lo,
            }
        )

    if out_csv:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


def format_comparison(rows):
    """Human-readable table of a compare_event() result."""
    lines = []
    hdr = (
        f"{'param':<10} {'truth':>16} {'exozippy':>16} {'+err':>12} "
        f"{'-err':>12} {'pull':>8}  note"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def _n(x, fmt="{:.6g}"):
        return fmt.format(x) if x is not None else "--"

    for r in rows:
        if r["param"] not in PARAMS:
            continue
        pull = r.get("exo_pull")
        lines.append(
            f"{r['param']:<10} {_n(r.get('truth')):>16} "
            f"{_n(r.get('exozippy')):>16} {_n(r.get('exo_err_hi')):>12} "
            f"{_n(r.get('exo_err_lo')):>12} "
            f"{_n(pull, '{:+.2f}'):>8}  {r.get('note', '')}"
        )
    return "\n".join(lines)
