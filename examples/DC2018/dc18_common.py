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

Alpha and u_0 conventions (conventions.md C22, measured 2026-10-02; tables
in docs/alpha_conventions.md sec 4): the master file's alpha
maps onto EXOZIPPy's (= MulensModel's = MMEXOFAST's) by an EVENT-DEPENDENT
rule -- see key_alpha_to_exozippy below -- and u_0 by the identity, sign
included.  compare_event therefore reports a truth and a pull for both.  The
one thing these no-parallax events cannot identify is the exact mirror
(u_0, alpha) -> -(u_0, alpha) (C23): a fit in the mirror branch is scored
against the key's mirror image and the row SAYS so, rather than being
folded into |u_0| silently.
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


def nextgen_bc_slice(facility, feh=0.0):
    """The published NextGen BC table of one facility at one [Fe/H].

    The SAME table the SED fits read (bc_grid.find_bc_table, which fetches
    it from Zenodo on first use -- models/NextGen/bc_tables.py), not the
    per-[Fe/H] legacy text files these scripts used to parse: those were
    deleted when the tables moved to Zenodo, and they were the
    photon-weighted pre-#335 tables, so reading them had already drifted
    from what the model uses.  One row per (teff, logg, Av) node; teff is
    LINEAR (the text files carried log10 teff), filter columns are MIST
    names (WFI_F146, 2MASS_J, ...).
    """
    from exozippy.components.sed.bc_grid import (
        DEFAULT_MODEL_ROOT,
        find_bc_table,
        read_bc_table,
    )

    df = read_bc_table(find_bc_table(DEFAULT_MODEL_ROOT, "NextGen", facility))
    out = df[np.isclose(df["feh"], float(feh))].reset_index(drop=True)
    if out.empty:
        raise ValueError(
            f"the NextGen {facility} BC table has no [Fe/H] = {feh} rows"
        )
    return out


# results.csv parname -> comparison param.  No longer one component:
# the trajectory offsets and source size are per-SOURCE, the timescale
# is EVENT-level, and the geometry is per-COMPANION (named by the lens
# body, since a two-element lens vector reports per element).  Measured
# against a post-split fit's DC2018_128_results.csv.
COMPANION = "Companion"
RESULTS_CSV_MAP = {
    "source.t_0": "t_0",
    "source.u_0": "u_0",
    "source.rho": "rho",
    "mulensevent.t_E": "t_E",
    f"lens.{COMPANION}.s": "s",
    f"lens.{COMPANION}.q": "q",
    f"lens.{COMPANION}.alpha": "alpha",
}


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


def load_master_row(data_dir, event):
    """The raw master_file.txt row for one event, plus its class label.

    Exposed separately from load_truth because the row carries far more than
    the seven lensing parameters -- lens/source distances, masses, radii and
    galactic-frame proper motions -- which the physics-chain and
    alpha-convention checks (scripts/dc18_alpha_convention.py,
    examples/DC2018/dc128_truth_forward.py) need.  Returns (row, class).
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
    return row, line.split(" ")[-2].split("_")[0]


def load_truth(data_dir, event):
    """Simulation truth for one event, with t_0 in full BJD.

    Returns (params_dict, class_label): params has the PARAMS keys, class is
    the challenge's event class scraped from the master-file row ('cassan'
    for the 2L1S planet sample, 'cv' for cataclysmic variables, ...).

    NOTE the alpha returned here is the master file's OWN value (alpha_key),
    measured from the planet orbit's line of nodes; key_alpha_to_exozippy
    maps it onto the fitted convention.  compare_event applies the mapping;
    the scan scripts that need the raw key value read it from here.
    """
    row, class_label = load_master_row(data_dir, event)
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
    low_err) and the multimodal one (# parname, mode, weight, weight_err,
    value, up_err, low_err), preferring the combined 'all' mode. Also
    returns the per-instrument err_scale rows as a second dict.

    THE FIELD LIST IS READ FROM THE HEADER, not hardcoded, and that is the
    point.  The multimodal branch used to hardcode SIX names and omit
    `weight_err`, so every column after `weight` shifted by one: `value`
    picked up the (usually EMPTY) weight_err cell and became None, `up_err`
    picked up the value.  The comparison table then reported an EMPTY
    exozippy column for every multimodal event while exiting non-zero with
    no explanation -- and multimodal is the NORM for microlensing, because
    the +/-u_0 degeneracy is always there.  Measured on DC2018 events 152,
    194 and 223: all three fitted, all three wrote results.csv, all three
    compared to nothing.  Parsing the header keeps this fixed if the writer
    gains another column.
    """
    with open(csv_path, newline="") as f:
        first = f.readline()
        has_mode = "mode" in first
        hdr = [c.strip() for c in first.lstrip("#").split(",") if c.strip()]
        known = {
            "parname",
            "mode",
            "weight",
            "weight_err",
            "value",
            "up_err",
            "low_err",
        }
        fields = (
            hdr
            if hdr and set(hdr) <= known and "parname" in hdr
            else (
                [
                    "parname",
                    "mode",
                    "weight",
                    "weight_err",
                    "value",
                    "up_err",
                    "low_err",
                ]
                if "mode" in first
                else ["parname", "value", "up_err", "low_err"]
            )
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


# Orbital periods below this make the static 2L1S comparison of alpha
# unreliable: the simulator moves the lens, the key's alpha is the t_0
# geometry, and a static fit finds the anomaly-epoch compromise (C22).
SHORT_PERIOD_YR = 2.0


def key_alpha_to_exozippy(alpha_key, phase, inc):
    """The DC2018 answer key's alpha in EXOZIPPy's convention, in [0, 360).

        alpha_EXZ  = alpha_key + 180 - theta_axis           (u_0 kept)
        theta_axis = atan2(sin(phase) cos(inc), cos(phase))

    All angles in degrees; `phase` and `inc` are the master file's columns of
    those names.  The key measures the SOURCE's direction of motion (the +180
    of conventions.md C21) against the planet orbit's LINE OF NODES, not
    against the binary axis; theta_axis is the projected planet's angle from
    that line at t_0 -- the same projection that reproduces the key's s
    (event 4: 0.1086 a / r_E = 2.482 vs the key's 2.48124).  The node angle
    is not needed: alpha_key and theta_axis share the node line, so it
    cancels.  That theta_axis differs per event is why every GLOBAL offset
    tested before 2026-10-02 scattered (R <= 0.19) and the key was wrongly
    recorded as unmappable.

    MEASURED, not assumed (scripts/dc18_alpha_convention.py on 36 events,
    2026-10-02; the per-event table is docs/alpha_conventions.md sec 4):
    at the key's own t_0, signed u_0, t_E, rho, s and q, the light curve's
    preferred alpha matches this on all 19 events with chi2 contrast >= 1000
    to a median 0.10 deg, max 1.53 deg.  The only residuals above 1 deg are
    the shortest periods (event 40, P = 1.27 yr: -30.6 deg; 208, 1.51 yr:
    +4.2; 32, 3.28 yr: +2.2; 128, 1.27 yr: -1.5) -- the simulator's lens
    orbital motion, which a static fit cannot follow (SHORT_PERIOD_YR).

    TRUTH-SIDE ONLY.  This uses the truth table's own phase and inc to
    express the key's alpha in our convention; it is never applied to a
    posterior.  The key's alpha is NOT measurable from a static or linear-
    motion light curve, which constrains only the trajectory angle to the
    instantaneous binary axis (our alpha, = MulensModel's = MMEXOFAST's);
    the phase/inc term is invisible to such a fit, and is (weakly)
    constrained only on a keplerian lens orbit, where alpha(t) is
    derived from the orbit.  The > 1 deg short-period residuals are that
    axis rotating during the event: a static fit's alpha sits at an
    effective mean axis, while the key quotes the phase at a reference
    epoch.
    """
    ph = np.radians(np.asarray(phase, dtype=float))
    inc_r = np.radians(np.asarray(inc, dtype=float))
    theta_axis = np.degrees(np.arctan2(np.sin(ph) * np.cos(inc_r), np.cos(ph)))
    return (np.asarray(alpha_key, dtype=float) + 180.0 - theta_axis) % 360.0


def wrap180(x):
    """An angle difference in degrees, wrapped onto [-180, 180)."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def mirror_branch_truth(u_0_truth, alpha_truth, u_0_fit):
    """(u_0, alpha, mirrored) of the truth in the FIT's u_0 branch.

    (u_0, alpha) -> -(u_0, alpha) is an exact symmetry of a static binary
    without parallax (conventions.md C23, Skowron Eq. A12), and these
    events have |pi_E| ~ 0.02, so a fit in the opposite u_0 branch from the
    key is the same physical solution, not a miss -- and nothing in the
    data can say which branch is "right".  It is scored against the key's
    mirror image, and `mirrored` is returned so the caller REPORTS the tie
    instead of hiding it in an absolute value.  With no fitted u_0 the
    key's own branch is kept.
    """
    if u_0_fit is None or u_0_truth is None or u_0_fit * u_0_truth >= 0:
        return u_0_truth, alpha_truth, False
    alpha_m = None if alpha_truth is None else float((-alpha_truth) % 360.0)
    return -u_0_truth, alpha_m, True


MIRROR_NOTE = (
    "fit is in the key's no-parallax mirror branch, (u_0, alpha) -> "
    "-(u_0, alpha): an exact tie (C23), scored against the mirror image"
)


def sigma_pull(truth_val, fit_val, err_hi, err_lo, angle=False):
    """(truth - fit) / one-sided sigma, or None when not computable.

    angle=True wraps the difference onto [-180, 180) degrees first, so an
    alpha reported as -52 compares correctly against a truth of 309.
    """
    if truth_val is None or fit_val is None:
        return None
    diff = truth_val - fit_val
    if angle:
        diff = float(wrap180(diff))
    err = err_hi if diff >= 0 else err_lo
    if err is None or abs(err) == 0 or not np.isfinite(err):
        return None
    return diff / abs(err)


def compare_event(event, data_dir, results_csv, mmx_json=None, out_csv=None):
    """Build the per-event truth/MMEXOFAST/EXOZIPPy comparison table.

    Returns a list of row dicts (one per parameter) and writes them as CSV
    when out_csv is given. Convention handling (conventions.md C22): alpha's
    truth is the key's value mapped by key_alpha_to_exozippy and its pull is
    taken on the circle; u_0 is compared SIGNED.  Each solution (EXOZIPPy's
    and every MMEXOFAST one) whose u_0 sign is opposite the key's is scored
    against the key's exact no-parallax mirror (mirror_branch_truth) and the
    u_0 and alpha rows' notes say so.  An orbit shorter than SHORT_PERIOD_YR
    flags the alpha row: the static fit cannot match the key's t_0 geometry.
    """
    truth, class_label = load_truth(data_dir, event)
    master, _ = load_master_row(data_dir, event)
    exo, err_scales = read_results_csv(results_csv)
    mmx_sols = (
        read_mmexofast_solutions(mmx_json)
        if mmx_json and Path(mmx_json).exists()
        else []
    )

    truth = dict(truth)
    notes = {p: [] for p in PARAMS}
    if np.isfinite(truth["alpha"]):
        truth["alpha"] = float(
            key_alpha_to_exozippy(
                truth["alpha"], float(master["phase"]), float(master["inc"])
            )
        )
        period = float(master["period"])
        if period < SHORT_PERIOD_YR:
            notes["alpha"].append(
                f"P = {period:.2f} yr < {SHORT_PERIOD_YR:g} yr: lens orbital "
                "motion moves alpha over the event; a static fit cannot "
                "match the key's t_0 geometry (C22)"
            )
    else:
        truth["alpha"] = None

    def _branch(u_0_fit, who):
        u0, al, mirrored = mirror_branch_truth(
            truth["u_0"], truth["alpha"], u_0_fit
        )
        if mirrored:
            for p in ("u_0", "alpha"):
                notes[p].append(f"{who}: {MIRROR_NOTE}")
        return {"u_0": u0, "alpha": al}

    exo_truth = dict(truth, **_branch(exo.get("u_0", (None,))[0], "exozippy"))
    mmx_truth = [
        dict(truth, **_branch(sol.get("u_0", (None,))[0], f"mmxf_sol{k}"))
        for k, sol in enumerate(mmx_sols)
    ]

    rows = []
    for p in PARAMS:
        val, hi, lo = exo.get(p, (None, None, None))
        ang = p == "alpha"
        row = {
            "event": event,
            "class": class_label,
            "param": p,
            "truth": exo_truth.get(p),
            "exozippy": val,
            "exo_err_hi": hi,
            "exo_err_lo": lo,
            "exo_pull": sigma_pull(exo_truth.get(p), val, hi, lo, angle=ang),
        }
        for k, sol in enumerate(mmx_sols):
            v, sig = sol.get(p, (None, None))
            row[f"mmxf_sol{k}"] = v
            row[f"mmxf_err_sol{k}"] = sig
            row[f"mmxf_pull_sol{k}"] = sigma_pull(
                mmx_truth[k].get(p), v, sig, sig, angle=ang
            )
        row["note"] = "; ".join(notes[p])
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
