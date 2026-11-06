"""Shared helpers for the DC2018 (2018 Roman Data Challenge) workflow.

Two data layouts are read, told apart by what is on disk (data_layout):

  "full" -- the challenge's own release (data-challenge-1, all 293 events):
    lc/ulwdc1_<NNN>_{W149,Z087}.txt             MAGNITUDE light curves
                                                (BJD mag err), AB, so the
                                                fit's flux system is
                                                F = 10**(-0.4 m): zeropoint 0
  "mmexofast" -- the MMEXOFAST checkout's converted copy (~48 events):
    n20180816.{W149,Z087}.WFIRST18.<NNN>.txt   FLUX light curves
                                                (BJD flux err), F =
                                                10**(-0.4 (m - 22)): the same
                                                photometry on a zeropoint of
                                                22.0 (AB), errors propagated
                                                exactly as mag_to_flux does
  and in both (--data-dir / $DC18_DATA, default the MMEXOFAST checkout):
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


# Per layout: (mulensinstrument data_format, AB magnitude of unit flux in
# the fit's flux system).  The second number is the zeropoint a config
# states for the light curve (magsys: AB): the challenge's photometry is AB
# (dc18_photometric_leg.py), the full release's files are those magnitudes
# themselves (F = 10**(-0.4 m), zeropoint 0), and the MMEXOFAST copy put
# them on a zeropoint of 22.0.  Verified on event 194: flux 0.76435 against
# 22.29178 mag, and errors 0.0138862 against 0.019725 mag.
LAYOUT_FORMAT = {
    "full": ("magnitude", 0.0),
    "mmexofast": ("flux", 22.0),
}


def _full_lc(data_dir, event, band):
    return Path(data_dir) / "lc" / f"ulwdc1_{event:03d}_{band}.txt"


def _mmexofast_lc(data_dir, event, band):
    return Path(data_dir) / f"n20180816.{band}.WFIRST18.{event:03d}.txt"


def data_layout(data_dir):
    """'full' (lc/ulwdc1_NNN_<band>.txt) or 'mmexofast' (flat flux files).

    Decided by what is on disk: an lc/ directory holding ulwdc1 files is the
    full release.  Raises when neither is present, rather than guessing.
    """
    d = Path(data_dir)
    if (d / "lc").is_dir() and any((d / "lc").glob("ulwdc1_*_W149.txt")):
        return "full"
    if any(d.glob("n20180816.W149.WFIRST18.*.txt")):
        return "mmexofast"
    raise FileNotFoundError(
        f"DC18 data dir '{d}' holds neither lc/ulwdc1_NNN_W149.txt (the full "
        "release) nor n20180816.W149.WFIRST18.NNN.txt (the MMEXOFAST copy)."
    )


def available_events(data_dir):
    """Sorted event numbers that have a W149 light curve on disk."""
    events = []
    if data_layout(data_dir) == "full":
        for f in (Path(data_dir) / "lc").glob("ulwdc1_*_W149.txt"):
            tag = f.name.split("_")[1]
            if tag.isdigit():
                events.append(int(tag))
    else:
        for f in Path(data_dir).glob("n20180816.W149.WFIRST18.*.txt"):
            tag = f.name.split(".")[-2]
            if tag.isdigit():
                events.append(int(tag))
    return sorted(events)


def light_curve_files(data_dir, event, bands=("W149", "Z087")):
    """{band: absolute path} for this event, raising on missing files."""
    path = _full_lc if data_layout(data_dir) == "full" else _mmexofast_lc
    files = {}
    for band in bands:
        f = path(data_dir, int(event), band)
        if not f.exists():
            raise FileNotFoundError(
                f"No {band} light curve for event {event}: {f}"
            )
        files[band] = str(f.resolve())
    return files


def read_light_curve(path, data_format):
    """(t, flux, flux_err) of one file, in the flux system the FIT uses.

    A magnitude file goes through mulensinstrument.mag_to_flux, the one
    owner of that conversion, so anything a generator derives from the data
    (a median flux, a bound) is in the same system as the model's fluxes.
    Non-finite rows and non-positive errors are dropped.
    """
    d = np.loadtxt(path)
    t, y, e = d[:, 0], d[:, 1], d[:, 2]
    if data_format == "magnitude":
        from exozippy.components.mulensing.mulensinstrument import mag_to_flux

        y, e = mag_to_flux(y, e)
    elif data_format != "flux":
        raise ValueError(f"unknown data_format {data_format!r}")
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    return t[ok], y[ok], e[ok]


def event_class(data_dir, event):
    """The answer key's class label of one event ('dccv', 'ombin', ...)."""
    return load_master_row(data_dir, event)[1]


# The challenge's cataclysmic-variable class: not microlensing, skipped by
# every fit of the sample (93 of the 293 events).
CV_CLASS = "dccv"


def non_cv_events(data_dir):
    """The events on disk whose answer-key class is not a CV."""
    return [
        ev
        for ev in available_events(data_dir)
        if event_class(data_dir, ev) != CV_CLASS
    ]


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


# ---------------------------------------------------------------------------
# Physical quantities: bias and coverage against the answer key (7.16.5)
# ---------------------------------------------------------------------------
#
# The light-curve scoring above cannot see an error that lives DOWNSTREAM of
# the light curve -- the galactic prior, the SED tie, extinction, a stellar
# relation acting as a mass prior (2.6.37).  These rows score the physical
# quantities directly, from the posterior DRAWS, as the truth's percentile in
# the posterior (the probability integral transform, PIT): over many events a
# calibrated posterior gives PITs uniform on [0, 1], and a one-directional
# bias piles them at one end.
#
# WHAT THE KEY PROVIDES, and in which frame (master_file.txt +
# wfirstColumnNumbers.txt; checked on all 30 sweep events, 2026-10-29):
#   Ml       Msun     the HOST mass, not the total: Mp == q * Ml to 4e-6.
#   Dl, Ds   kpc      distances.
#   lmu_l, lmu_b, smu_l, smu_b   mas/yr   lens / source proper motions,
#            GALACTIC, heliocentric.  murel == |lmu - smu| to 1e-5 relative,
#            so the key's murel is the HELIOCENTRIC relative proper motion,
#            the same quantity as our mulensevent.mu_rel_mag (C5, C7; NOT
#            mu_rel_geo_mag).  The column label does not say whether mu_l
#            carries cos(b); it is taken as mu_l* (the magnitude check passes
#            either way, and at |b| < 2.1 deg on these events cos(b) > 0.999,
#            so the direction is unaffected below 0.1 deg).
#   Mp       Msun     the column header says Msun with an Earth-mass display
#            factor; the value is Msun (Mp == q * Ml), converted to Mjup here.
#   s, rE    -, AU    rE == thE * Dl exactly, so a_perp = s * rE is the
#            PROJECTED separation in AU at the key's t_0 (the key's `a` column
#            is the 3-D semimajor axis, NOT comparable: s * rE <= a always).
# The key does NOT give the direction of mu_rel directly; it is rebuilt from
# the four galactic components, rotated onto the local (E, N) axes at the
# event, as the position angle EAST OF NORTH of the LENS relative to the
# SOURCE (C2, C7).  Nothing is approximated silently: every one of these
# checks is re-run per event by physical_truth and a failure is a note on
# the row.
#
# Also found and NOT used here: the key's tE is not 365.25 * thE / murel (the
# heliocentric timescale) -- they differ by up to 1.8% on these events -- so
# the key's tE is in the simulator's own (observer-centric) frame, and the
# heliocentric murel above is the right partner for our mu_rel_mag.

# (key, unit, kind).  kind: "log" -> positive quantity, bias as log10 ratio;
# "angle" -> circular, degrees.
PHYSICAL = [
    ("M_lens", "Msun", "log"),
    ("D_L", "kpc", "log"),
    ("D_S", "kpc", "log"),
    ("mu_rel", "mas/yr", "log"),
    ("phi_mu", "deg E of N", "angle"),
    ("M_planet", "Mjup", "log"),
    ("a_perp", "AU", "log"),
]
PHYSICAL_KEYS = [k for k, _, _ in PHYSICAL]
PHYSICAL_KIND = {k: kind for k, _, kind in PHYSICAL}
PHYSICAL_UNIT = {k: u for k, u, _ in PHYSICAL}

MSUN_PER_MJUP = 9.5458e-4  # the IAU nominal ratio, as dc18_truth_table uses


def galactic_pm_to_equatorial(mu_l, mu_b, ra_deg, dec_deg):
    """(mu_l*, mu_b) [mas/yr] -> (mu_E, mu_N) = (mu_ra*, mu_dec) at (ra, dec).

    The same rotation as star.physics.sgra_pm_equatorial (Poleski 2013), with
    the same NGP constants, applied to an arbitrary galactic vector in numpy.
    Pinned against astropy in tests/test_dc18_physical.py.
    """
    from exozippy.components.star.physics import NGP_DEC_RAD, NGP_RA_RAD

    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    d_ra = ra - NGP_RA_RAD
    c1 = np.sin(NGP_DEC_RAD) * np.cos(dec) - np.cos(NGP_DEC_RAD) * np.sin(
        dec
    ) * np.cos(d_ra)
    c2 = np.cos(NGP_DEC_RAD) * np.sin(d_ra)
    cosb = np.hypot(c1, c2)
    mu_e = (c1 * mu_l - c2 * mu_b) / cosb
    mu_n = (c2 * mu_l + c1 * mu_b) / cosb
    return mu_e, mu_n


def physical_truth(data_dir, event):
    """({key: truth value}, {key: [notes]}) for the PHYSICAL quantities.

    Units are PHYSICAL_UNIT's.  The consistency checks described above the
    PHYSICAL list are re-run here and a failure is reported as a note, never
    patched over.
    """
    row, _ = load_master_row(data_dir, event)
    g = lambda k: float(row[k])  # noqa: E731
    notes = {k: [] for k in PHYSICAL_KEYS}
    dmu_l = g("lmu_l") - g("smu_l")
    dmu_b = g("lmu_b") - g("smu_b")
    mu_hyp = float(np.hypot(dmu_l, dmu_b))
    if abs(mu_hyp / g("murel") - 1.0) > 1e-3:
        msg = (
            f"key murel {g('murel'):.6g} != |lmu - smu| {mu_hyp:.6g}: the "
            "key's frame is not the heliocentric lens-minus-source motion"
        )
        notes["mu_rel"].append(msg)
        notes["phi_mu"].append(msg)
    mu_e, mu_n = galactic_pm_to_equatorial(dmu_l, dmu_b, g("ra"), g("dec"))
    mp_msun = g("Mp")
    if abs(mp_msun / (g("q") * g("Ml")) - 1.0) > 1e-4:
        notes["M_planet"].append(
            f"key Mp {mp_msun:.6g} != q * Ml {g('q') * g('Ml'):.6g}: Ml may "
            "not be the host mass"
        )
        notes["M_lens"].append("key Mp != q * Ml (see M_planet)")
    if abs(g("rE") / (g("thE") * g("Dl")) - 1.0) > 1e-3:
        notes["a_perp"].append(
            f"key rE {g('rE'):.6g} != thE * Dl {g('thE') * g('Dl'):.6g}"
        )
    truth = {
        "M_lens": g("Ml"),
        "D_L": g("Dl"),
        "D_S": g("Ds"),
        "mu_rel": g("murel"),
        "phi_mu": float(np.degrees(np.arctan2(mu_e, mu_n)) % 360.0),
        "M_planet": mp_msun / MSUN_PER_MJUP,
        "a_perp": g("s") * g("rE"),
    }
    return truth, notes


def read_wrapup_header(summary_path):
    """(burnin, chains_kept) from `<prefix>_summary.txt`'s first line, or None.

    The trace on disk is the FULL T=1 trace; the pipeline's wrap-up then
    drops a burn-in (samplers.convergence.find_burnin) and the chains that
    never reached the good-likelihood region (good_chain_mask), and every
    number in results.csv is computed on what is left.  The burn-in is read
    back from this header rather than recomputed, so the PIT is taken on the
    SAME draws the pipeline reported.
    """
    if summary_path is None or not Path(summary_path).exists():
        return None
    import re

    with open(summary_path) as f:
        first = f.readline()
    m = re.search(
        r"burn-in discarded:\s*(\d+)\s*draws.*chains kept:\s*(\d+)", first
    )
    return (int(m.group(1)), int(m.group(2))) if m else None


# Burn-in assumed when the wrap-up never wrote its header (it died after the
# trace was written): half the trace, the median of the 25 sweep30 headers.
FALLBACK_BURNIN_FRAC = 0.5


def physical_draws(trace_path, config_path=None, summary_path=None):
    """Pooled T=1 posterior draws of the PHYSICAL quantities, from a trace.

    Returns (draws, info): draws = {key: 1-D array, non-finite removed},
    info = {"burnin", "n_chains", "n_chains_total", "n_draws_kept",
    "provenance", "warnings"}.

    Every mode is POOLED, weighted by occupancy (the stored draws): the
    question is the whole posterior's calibration, not one mode's.  Which
    star element is the Lens and the Source, and which lens element is the
    companion, come from the run's own config when it is given (`star:` and
    `lens:` block order), else the sweep layout (Lens 0, Source 1, companion
    1).

    a_perp = s * theta_E * D_L, per draw: theta_E [mas] times D_L [kpc] is
    AU, and s is in Einstein radii of the TOTAL lens mass (C13), the same
    theta_E -- so this is the projected separation the key's s * rE is.
    """
    import xarray as xr

    from exozippy.samplers.convergence import good_chain_mask

    warns = []
    i_lens, i_src, i_comp = 0, 1, 1
    if config_path is not None and Path(config_path).exists():
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        names = [s.get("name") for s in cfg.get("star", []) or []]
        if "Lens" in names and "Source" in names:
            i_lens, i_src = names.index("Lens"), names.index("Source")
        bodies = [str(b.get("body", "")) for b in cfg.get("lens", []) or []]
        comp = [k for k, b in enumerate(bodies) if b.startswith("planet.")]
        if comp:
            i_comp = comp[0]

    post = xr.open_dataset(trace_path, group="posterior")
    stats = xr.open_dataset(trace_path, group="sample_stats")
    try:
        n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
        lp = stats["lp"].values if "lp" in stats else None
        hdr = read_wrapup_header(summary_path)
        if lp is not None:
            mask, _ = good_chain_mask(lp)
        else:
            mask = np.ones(n_chain, dtype=bool)
            warns.append("no sample_stats lp: every chain kept")
        if hdr is not None:
            burnin, kept = hdr
            prov = "trace, wrap-up's burn-in + chain filter"
            if kept == n_chain:
                mask = np.ones(n_chain, dtype=bool)
            elif int(mask.sum()) != kept:
                warns.append(
                    f"good_chain_mask keeps {int(mask.sum())} chains, the "
                    f"wrap-up kept {kept}: using good_chain_mask's"
                )
        else:
            burnin = int(FALLBACK_BURNIN_FRAC * n_draw)
            prov = (
                "trace, NO WRAP-UP HEADER: burn-in "
                f"{FALLBACK_BURNIN_FRAC:.0%} assumed, good_chain_mask"
            )
        idx = np.nonzero(mask)[0]

        def take(name, elem=None):
            if name not in post:
                return None
            a = post[name].isel(chain=idx, draw=slice(burnin, None)).values
            if a.ndim == 3:
                a = a[..., 0 if elem is None else elem]
            return np.asarray(a, dtype=float).ravel()

        def take_or_pow10(name, log_name, elem=None):
            # Older traces (e.g. d228e190's static30) store only the log10
            # coordinate of some derived quantities.
            a = take(name, elem)
            if a is None:
                a = take(log_name, elem)
                a = None if a is None else 10.0**a
            return a

        mass = take_or_pow10("star.mass", "star.logmass", i_lens)
        dist_l = take("star.distance", i_lens)
        dist_s = take("star.distance", i_src)
        mu_e = take("mulensevent.mu_ra_rel")
        mu_n = take("mulensevent.mu_dec_rel")
        mu = take("mulensevent.mu_rel_mag")
        if mu is None and mu_e is not None and mu_n is not None:
            mu = np.hypot(mu_e, mu_n)
        the = take_or_pow10("mulensevent.theta_E", "mulensevent.log_theta_E")
        s = take_or_pow10("lens.s", "lens.log_s", i_comp)
        mp = take("planet.mass")
        draws = {
            "M_lens": mass,
            "D_L": None if dist_l is None else dist_l / 1000.0,
            "D_S": None if dist_s is None else dist_s / 1000.0,
            "mu_rel": mu,
            "phi_mu": (
                None
                if mu_e is None or mu_n is None
                else np.degrees(np.arctan2(mu_e, mu_n)) % 360.0
            ),
            "M_planet": mp,
            "a_perp": (
                None
                if s is None or the is None or dist_l is None
                else s * the * dist_l / 1000.0
            ),
        }
        out = {}
        for k, v in draws.items():
            if v is None:
                warns.append(f"{k}: not in trace")
                continue
            ok = np.isfinite(v)
            if not ok.all():
                warns.append(
                    f"{k}: {int((~ok).sum())} of {ok.size} draws non-finite, "
                    "dropped"
                )
            out[k] = v[ok]
        info = {
            "burnin": int(burnin),
            "n_chains": int(len(idx)),
            "n_chains_total": int(n_chain),
            "n_draws_kept": int(max(n_draw - burnin, 0) * len(idx)),
            "provenance": prov,
            "warnings": warns,
            "s_companion": (
                None if s is None else float(np.median(s[np.isfinite(s)]))
            ),
        }
        return out, info
    finally:
        post.close()
        stats.close()


def score_draws(draws, truth, kind="log"):
    """Bias and coverage of one quantity: the truth's place in the draws.

    Returns {"pit", "median", "lo68", "hi68", "lo95", "hi95", "bias",
    "log10_ratio", "z", "in68", "in95", "n"}.

    pit is the fraction of draws below the truth (ties count half): 0.5 is
    the median, and near 0 (1) means the posterior sits ABOVE (BELOW) the
    truth.  in68 / in95 are the central-interval coverages, i.e. pit in
    [0.159, 0.841] / [0.025, 0.975].  z is (median - truth) over the
    half-interval on the truth's side, the same one-sided sigma as
    sigma_pull, with the opposite sign convention: z > 0 = posterior HIGH.

    kind="angle" (degrees): the draws are unwrapped about their circular
    mean first, so a posterior straddling 0/360 is one interval; pit, bias
    and z are taken on the circle.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or truth is None or not np.isfinite(truth):
        return None
    t = float(truth)
    if kind == "angle":
        r = np.radians(d)
        c = float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))
        d = c + wrap180(d - c)
        t = c + float(wrap180(t - c))
    pit = float((np.sum(d < t) + 0.5 * np.sum(d == t)) / d.size)
    q = np.percentile(d, [2.5, 15.865, 50.0, 84.135, 97.5])
    med = float(q[2])
    half = (q[3] - med) if t >= med else (med - q[1])
    out = {
        "pit": pit,
        "median": med,
        "lo68": float(q[1]),
        "hi68": float(q[3]),
        "lo95": float(q[0]),
        "hi95": float(q[4]),
        "bias": med - t,
        "log10_ratio": (
            float(np.log10(med / t))
            if kind == "log" and med > 0 and t > 0
            else None
        ),
        "z": float((med - t) / half) if half > 0 else None,
        "in68": bool(0.15865 <= pit <= 0.84135),
        "in95": bool(0.025 <= pit <= 0.975),
        "n": int(d.size),
    }
    if kind == "angle":
        out["median"] = med % 360.0
        out["lo68"], out["hi68"] = float(q[1]) % 360.0, float(q[3]) % 360.0
        out["lo95"], out["hi95"] = float(q[0]) % 360.0, float(q[4]) % 360.0
    return out


def physical_rows(
    event, data_dir, trace_path, config_path=None, summary_path=None
):
    """One row per PHYSICAL quantity for one event: truth, posterior, PIT.

    Returns (rows, info) with info from physical_draws.  A row whose truth or
    draws are missing is still returned, with pit None and a note, so a
    MISSING quantity is visible in a table rather than silently dropped.
    """
    truth, tnotes = physical_truth(data_dir, event)
    master, cls = load_master_row(data_dir, event)
    key_s = float(master["s"])
    draws, info = physical_draws(trace_path, config_path, summary_path)
    rows = []
    for k in PHYSICAL_KEYS:
        notes = list(tnotes[k])
        sc = (
            score_draws(draws[k], truth[k], PHYSICAL_KIND[k])
            if k in draws
            else None
        )
        if sc is None:
            notes.append("no posterior draws")
        if k == "a_perp" and info.get("s_companion") is not None:
            s_fit = info["s_companion"]
            if (s_fit - 1.0) * (key_s - 1.0) < 0:
                notes.append(
                    f"fit s {s_fit:.3g} is on the other side of 1 from the "
                    f"key's {key_s:.3g} (close/wide): a_perp scores the "
                    "light-curve branch, not the physics"
                )
        row = {
            "event": event,
            "class": cls,
            "param": k,
            "unit": PHYSICAL_UNIT[k],
            "truth": truth[k],
        }
        row.update(sc or {"pit": None})
        row["note"] = "; ".join(notes)
        rows.append(row)
    return rows, info


def ensemble_physical(rows):
    """Per-quantity ensemble bias and coverage over many events' rows.

    Returns {key: {"n", "mean_pit", "ks_d", "ks_p", "n_high", "sign_p",
    "cov68", "cov95", "median_log10_ratio", "median_bias", "hist"}}.

    n_high counts events whose posterior median is ABOVE the truth (pit <
    0.5); sign_p is the two-sided binomial probability of a split at least
    that lopsided from a fair coin, i.e. how one-directional the bias is.
    ks_p tests the PITs against U(0, 1).  hist is the 10-bin PIT histogram.
    """
    from scipy import stats as st

    out = {}
    for k in PHYSICAL_KEYS:
        rs = [r for r in rows if r["param"] == k and r.get("pit") is not None]
        if not rs:
            continue
        pit = np.array([r["pit"] for r in rs])
        n = pit.size
        n_high = int(np.sum(pit < 0.5))
        lr = [r["log10_ratio"] for r in rs if r.get("log10_ratio") is not None]
        ks = st.kstest(pit, "uniform")
        out[k] = {
            "n": n,
            "mean_pit": float(pit.mean()),
            "ks_d": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "n_high": n_high,
            "sign_p": float(st.binomtest(n_high, n, 0.5).pvalue),
            "cov68": float(np.mean([r["in68"] for r in rs])),
            "cov95": float(np.mean([r["in95"] for r in rs])),
            "median_log10_ratio": float(np.median(lr)) if lr else None,
            "median_bias": float(np.median([r["bias"] for r in rs])),
            "hist": np.histogram(pit, bins=10, range=(0.0, 1.0))[0].tolist(),
        }
    return out


def format_physical(rows):
    """Human-readable per-event table of physical_rows()."""
    hdr = (
        f"{'quantity':<9} {'unit':<11} {'truth':>11} {'median':>11} "
        f"{'68% interval':>25} {'PIT':>6} {'z':>7}  note"
    )
    lines = [hdr, "-" * len(hdr)]

    def _n(x, fmt="{:.4g}"):
        return fmt.format(x) if x is not None else "--"

    for r in rows:
        iv = (
            f"[{_n(r.get('lo68'))}, {_n(r.get('hi68'))}]"
            if r.get("pit") is not None
            else "--"
        )
        lines.append(
            f"{r['param']:<9} {r['unit']:<11} {_n(r['truth']):>11} "
            f"{_n(r.get('median')):>11} {iv:>25} "
            f"{_n(r.get('pit'), '{:.3f}'):>6} {_n(r.get('z'), '{:+.2f}'):>7}"
            f"  {r.get('note', '')}"
        )
    return "\n".join(lines)


def format_ensemble(ens, title="ENSEMBLE"):
    """Human-readable table of ensemble_physical()."""
    hdr = (
        f"{'quantity':<9} {'N':>3} {'meanPIT':>8} {'KS D':>6} {'KS p':>8} "
        f"{'hi/N':>6} {'sign p':>8} {'cov68':>6} {'cov95':>6} "
        f"{'med log10(fit/true)':>20}  PIT hist (10 bins, 0->1)"
    )
    lines = [title, hdr, "-" * len(hdr)]
    for k in PHYSICAL_KEYS:
        if k not in ens:
            continue
        e = ens[k]
        lr = e["median_log10_ratio"]
        lines.append(
            f"{k:<9} {e['n']:>3} {e['mean_pit']:>8.3f} {e['ks_d']:>6.3f} "
            f"{e['ks_p']:>8.2g} {e['n_high']:>3}/{e['n']:<2} "
            f"{e['sign_p']:>8.2g} {e['cov68']:>6.2f} {e['cov95']:>6.2f} "
            f"{('%+.3f' % lr) if lr is not None else '--':>20}  "
            f"{' '.join('%2d' % h for h in e['hist'])}"
        )
    lines.append(
        "  hi/N = events whose posterior median is ABOVE the truth; sign p = "
        "two-sided binomial p of that split; calibrated -> meanPIT 0.5, "
        "cov68 0.68, cov95 0.95, flat histogram"
    )
    return "\n".join(lines)


def compare_event(
    event,
    data_dir,
    results_csv,
    mmx_json=None,
    out_csv=None,
    trace_path="auto",
):
    """Build the per-event truth/MMEXOFAST/EXOZIPPy comparison table.

    Returns a list of row dicts (one per parameter) and writes them as CSV
    when out_csv is given. Convention handling (conventions.md C22): alpha's
    truth is the key's value mapped by key_alpha_to_exozippy and its pull is
    taken on the circle; u_0 is compared SIGNED.  Each solution (EXOZIPPy's
    and every MMEXOFAST one) whose u_0 sign is opposite the key's is scored
    against the key's exact no-parallax mirror (mirror_branch_truth) and the
    u_0 and alpha rows' notes say so.  An orbit shorter than SHORT_PERIOD_YR
    flags the alpha row: the static fit cannot match the key's t_0 geometry.

    PHYSICAL ROWS (7.16.5): when the run's trace is found -- trace_path
    "auto" looks for `<prefix>_trace.nc` beside results_csv; None skips --
    one row per PHYSICAL quantity is appended, param "phys.<key>", scored
    from the posterior draws by physical_rows: `exozippy` is the median,
    the errors the 68% interval, `exo_pull` the same (truth - fit) / sigma
    as the light-curve rows, and `pit` the truth's percentile in the
    posterior.  The draws are the pipeline's own (burn-in and chain filter
    from `<prefix>_summary.txt`), every mode pooled.
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

    prefix = str(results_csv)
    if prefix.endswith("_results.csv"):
        prefix = prefix[: -len("_results.csv")]
    if trace_path == "auto":
        trace_path = prefix + "_trace.nc"
    if trace_path is not None and Path(trace_path).exists():
        prows, info = physical_rows(
            event,
            data_dir,
            trace_path,
            config_path=prefix + ".yaml",
            summary_path=prefix + "_summary.txt",
        )
        for r in prows:
            med = r.get("median")
            note = r["note"]
            if info["warnings"]:
                note = "; ".join([note] + info["warnings"]).strip("; ")
            rows.append(
                {
                    "event": event,
                    "class": class_label,
                    "param": "phys." + r["param"],
                    "truth": r["truth"],
                    "exozippy": med,
                    "exo_err_hi": None if med is None else r["hi68"] - med,
                    "exo_err_lo": None if med is None else med - r["lo68"],
                    "exo_pull": None if r.get("z") is None else -r["z"],
                    "pit": r.get("pit"),
                    "note": note,
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
        phys = r["param"].startswith("phys.")
        if r["param"] not in PARAMS and not phys:
            continue
        pull = r.get("exo_pull")
        note = r.get("note", "")
        if phys and r.get("pit") is not None:
            note = f"PIT {r['pit']:.3f}" + (f"; {note}" if note else "")
        lines.append(
            f"{r['param']:<10} {_n(r.get('truth')):>16} "
            f"{_n(r.get('exozippy')):>16} {_n(r.get('exo_err_hi')):>12} "
            f"{_n(r.get('exo_err_lo')):>12} "
            f"{_n(pull, '{:+.2f}'):>8}  {note}"
        )
    return "\n".join(lines)
