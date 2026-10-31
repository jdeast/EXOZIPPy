"""Prior-predictive check of the microlensing prior (review 7.16.3).

Build a lens + source model with NO light curve for one bulge sight line,
sample the prior with event-rate weighting, and store the quantiles of
t_E, D_L, D_S, mu_rel and M_L -- for comparison with survey-measured,
efficiency-corrected t_E distributions (Mroz et al. 2017, 2019) and with
genulens (Koshimoto, Baba & Bennett 2021, ApJ 917, 78; K21) run for the
same sight lines.

A frame, sign, kinematics, IMF or density error shifts t_E, so this is one
end-to-end check of the whole prior that does not depend on genulens.

OGLE IS A CONSISTENCY CHECK, NOT AN INDEPENDENT CALIBRATION TARGET: K21
fit their IMF to these same OGLE-IV t_E distributions, so genulens
matching OGLE holds by construction, and a K21-like mass function here
matching OGLE checks our implementation of their model, not the IMF.  Do
not tune the prior to OGLE's t_E; that counts the data twice.

HOW THE PRIOR IS EVENT-RATE WEIGHTED (nothing here adds a term; this is
what the model already does, so the script samples exactly what a real
microlensing fit uses as its prior):

  * ``galacticmodel.kinematic_prior`` -- per STAR (lens and source alike,
    one ``pt.sum``): the thin-disk + thick-disk + bar NUMBER density
    (logsumexp mixture, each branch on genulens's absolute number-density
    scale), times each branch's Gaussian velocity distribution, times the
    ``d^2`` volume element, times the ``(K d)^2`` Jacobian from km/s to the
    sampled proper motions.  So each star's (D, mu, rv) is drawn from
    ``n(D) D^2 f(v)``.
  * the mass prior -- by default the present-day mass function of each
    star's population (the Chabrier IMF evolved to today, review 1.7.2),
    INSIDE the kinematic mixture above; with ``present_day: false`` the
    piecewise Chabrier IMF in log10 M on every star (``imf_prior``).  Only
    the lens mass reaches t_E.
  * ``mulensevent.event_rate_prior`` -- ``log(mu_rel,geo) + log(theta_E)``:
    the rate factor Gamma ~ theta_E * mu_rel (each lens sweeps a strip of
    angular width 2 theta_E at angular speed mu_rel).
  * ``mulensevent.source_behind_lens`` -- a soft log-sigmoid barrier on
    D_S - D_L (it is not a hard cut, so a small fraction of draws can sit
    with D_L > D_S; the script reports that fraction).

Together: p(D_L, D_S, v_L, v_S, M_L) ~ n_L(D_L) D_L^2 f_L(v_L) Phi(M_L)
x n_S(D_S) D_S^2 f_S(v_S) x theta_E mu_rel, the standard event-rate
weighting.  Differences from what a survey measures that are KNOWN and
are model choices, not bugs; Part B has to account for them, not
rediscover them:

  * sources are weighted by n(D_S) D_S^2 only -- no luminosity function or
    magnitude limit (genulens's ``gammaDs`` exponent is 2 here, versus its
    default 0.5 ~ an I < 21 source sample);
  * the lens mass function is the galacticmodel's present-day one by
    default (review 1.7.2: MIST turnoff, white dwarfs, neutron stars and
    black holes, as genulens REMNANT 1); ``sample --no-present-day``
    restores the IMF as the present-day mass function.  The genulens
    reference is its REMNANT 1 run for the same reason (REMNANT 0
    DISCARDS evolved lenses instead; see GENULENS_VARIANTS);
  * neither this prior nor the genulens references have binary lenses.

With no light curve there is no t0_par and no Earth velocity, so
``mu_rel_geo`` falls back to the heliocentric value (mulensevent warns).
That is the season average; a survey's geocentric t_E averages over
seasons the same way.

Subcommands::

  sample    sample the EXOZIPPy prior for one sight line (Slurm only)
  mroz      build the Mroz et al. 2017/2019 reference quantiles
  genulens  run genulens for the sight lines and store its quantiles
  compare   Part B: t_E vs Mroz (quantiles, KS), D_L/mu_rel/M_L vs
            genulens, and the lens-mass-swap decomposition
  table     print the stored quantiles side by side (no verdicts)

Run ``sample`` with PYTHONPATH pointing at the exozippy ``src``.  The
``mroz``/``genulens``/``table`` subcommands need only numpy.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REFS = HERE / "prior_predictive_tE_refs"

# Quantiles stored everywhere.  0.10/0.50/0.90 are the comparison the
# review item names; the rest are for the KS-style shape comparison.
QUANTILES = (0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95)

# Sight lines.  Galactic (l, b) in degrees.  `mroz` names the published
# efficiency-corrected sample used as the t_E reference for that line.
#
# * baade: Baade's window as review 7.16.3 specifies it.  It lies inside
#   OGLE-IV field BLG513 (events at (0.989, -3.869) are BLG513.12).
# * ob140939: OGLE-2014-BLG-0939, ra/dec from examples/ob140939's params
#   file (266.801041667, -21.3829722222) -> (l, b) = (6.6480, +3.6113).
#   The event itself is in Mroz 2019's Table 3 (BLG630.01, l = 6.64811,
#   b = 3.61142), so field BLG630 is its reference.
# * blg505, blg521: field centres.  With baade (BLG513) they make a
#   sequence in b at fixed l ~ 1.1 (b = -2.39, -3.74, -5.11); BLG505 is a
#   high-cadence field (501 events in Mroz 2017's Extended Data
#   Table 4), BLG521 a low-cadence one (69 events in Mroz 2019).
SIGHTLINES = {
    "baade": {"l": 1.0, "b": -3.9, "mroz": ("mroz2019", "BLG513")},
    "ob140939": {"l": 6.6480, "b": 3.6113, "mroz": ("mroz2019", "BLG630")},
    "blg505": {"l": 1.0870, "b": -2.3890, "mroz": ("mroz2017", "BLG505")},
    "blg521": {"l": 1.1905, "b": -5.1086, "mroz": ("mroz2019", "BLG521")},
}


# ----------------------------------------------------------------------
# statistics helpers (numpy only)
# ----------------------------------------------------------------------
def weighted_quantiles(x, w, qs=QUANTILES):
    """Quantiles of x under weights w (midpoint-CDF interpolation)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return {
        f"q{int(round(100 * q)):02d}": float(np.interp(q, cdf, x)) for q in qs
    }


def summarize(columns, weights=None):
    """{name: {quantiles..., mean, n}} for a dict of 1-D arrays."""
    out = {}
    for name, x in columns.items():
        x = np.asarray(x, float)
        w = np.ones_like(x) if weights is None else np.asarray(weights, float)
        d = weighted_quantiles(x, w)
        ok = np.isfinite(x) & (w > 0)
        d["mean"] = float(np.sum(w[ok] * x[ok]) / np.sum(w[ok]))
        d["n"] = int(ok.sum())
        if weights is not None:
            # Kish effective sample size: what the quantiles' noise scales as.
            d["n_eff"] = float(w[ok].sum() ** 2 / np.sum(w[ok] ** 2))
        out[name] = d
    return out


def binned_quantiles(log_centres, half_width, counts, qs=QUANTILES):
    """Quantiles of t_E from a histogram in log10 t_E (linear within bins)."""
    edges = np.concatenate(
        [log_centres - half_width, [log_centres[-1] + half_width]]
    )
    c = np.asarray(counts, float)
    cdf = np.concatenate([[0.0], np.cumsum(c)]) / c.sum()
    d = {
        f"q{int(round(100 * q)):02d}": float(10 ** np.interp(q, cdf, edges))
        for q in qs
    }
    d["mean"] = float(np.sum(c * 10**log_centres) / c.sum())
    return d


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")


# ----------------------------------------------------------------------
# mroz: the survey references
# ----------------------------------------------------------------------
def _read_mroz2017(name):
    rows = [
        line.split()
        for line in (REFS / name).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    hdr = [
        ln
        for ln in (REFS / name).read_text().splitlines()
        if ln.startswith("# bin")
    ][0].split()[1:]
    arr = np.array(rows, float)
    return hdr, arr


def _table3_events(path):
    """(subfield, tE_best, tE_med, weight) from Mroz 2019 Table 3.

    Split on whitespace, NOT on the ReadMe's byte columns: a few rows with
    t_E error bars wider than their F7.3/F6.3 fields overflow and shift
    every later column (e.g. BLG615.16.23888, tE_err2 = 128.578).  All
    5790 rows have exactly 31 tokens; token 10 is tE_best, 17 tE_med, 29
    the inverse detection efficiency.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) != 31:
            raise ValueError(
                f"unexpected Table 3 row ({len(p)} tokens): {line}"
            )
        out.append((p[1], float(p[10]), float(p[17]), float(p[29])))
    return out


def _table7(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        out[p[0]] = {
            "glon": float(p[1]),
            "glat": float(p[2]),
            "tE_mean": float(p[9]),
            "tE_mean_err": float(p[10]),
            "N_events": int(p[11]),
        }
    return out


def cmd_mroz(args):
    """Build prior_predictive_tE_refs/mroz_tE_reference.json.

    Mroz 2017 (Nature 548, 183): Extended Data Table 4 (counts per field
    per log t_E bin) / Extended Data Table 5 (detection efficiency, same
    bins) -> efficiency-corrected histogram per field, nine high-cadence
    fields, 2010-2015, I_s <= 22.  Transcribed in mroz2017_edt*.dat.

    Mroz 2019 (ApJS 244, 29): Table 3 (per-event best fit, low-cadence
    fields, 2010-2017) with its own inverse-efficiency weight column, and
    Table 7 (per-field efficiency-corrected mean t_E, I_s < 21), both from
    https://ftp.astrouw.edu.pl/ogle/ogle4/microlensing_maps/ .  Table 3 is
    1.5 MB, so only the events of the reference fields are copied into the
    repo (mroz2019_table3_<field>.dat); --mroz2019-dir points at a download.
    """
    ref = {
        "_note": (
            "Efficiency-corrected t_E quantiles (days). mroz2017: binned "
            "(Extended Data Tables 4/5, 0.14-dex bins, linear in log t_E "
            "within a bin). mroz2019: per-event tE_best weighted by Table 3's "
            "inverse-efficiency column. table7_*: the paper's own "
            "efficiency-corrected mean t_E, for checking the reconstruction."
        ),
        "mroz2017": {},
        "mroz2019": {},
    }
    hdr, counts = _read_mroz2017("mroz2017_edt4_counts.dat")
    _, eff = _read_mroz2017("mroz2017_edt5_eff.dat")
    logc = counts[:, 1]
    t7 = _table7(Path(args.mroz2019_dir) / "table7.dat")
    for j, field in enumerate(hdr[2:], start=2):
        corr = counts[:, j] / eff[:, j]
        d = binned_quantiles(logc, 0.07, corr)
        d["n_events"] = int(counts[:, j].sum())
        if field in t7:
            d["table7_tE_mean"] = t7[field]["tE_mean"]
            d["table7_tE_mean_err"] = t7[field]["tE_mean_err"]
            d["glon"], d["glat"] = t7[field]["glon"], t7[field]["glat"]
        ref["mroz2017"][field] = d

    events = _table3_events(Path(args.mroz2019_dir) / "table3.dat")
    fields = sorted({e[0][:6] for e in events})
    want = {
        v["mroz"][1] for v in SIGHTLINES.values() if v["mroz"][0] == "mroz2019"
    }
    for field in fields:
        ev = [e for e in events if e[0][:6] == field]
        te_best = np.array([e[1] for e in ev])
        te_med = np.array([e[2] for e in ev])
        w = np.array([e[3] for e in ev])
        d = summarize({"tE": te_best}, w)["tE"]
        d["mean_tE_med"] = float(np.sum(w * te_med) / w.sum())
        if field in t7:
            d["table7_tE_mean"] = t7[field]["tE_mean"]
            d["table7_tE_mean_err"] = t7[field]["tE_mean_err"]
            d["glon"], d["glat"] = t7[field]["glon"], t7[field]["glat"]
        ref["mroz2019"][field] = d
        if field in want:
            path = REFS / f"mroz2019_table3_{field}.dat"
            lines = [
                "# Mroz et al. 2019, ApJS 244, 29, Table 3 (electronic, "
                "ftp.astrouw.edu.pl/ogle/ogle4/microlensing_maps/table3.dat),",
                f"# events of field {field} only.  weight = inverse detection "
                "efficiency (Table 3's own column).",
                "# subfield tE_best tE_med weight",
            ] + [f"{e[0]} {e[1]:.3f} {e[2]:.3f} {e[3]:.2f}" for e in ev]
            path.write_text("\n".join(lines) + "\n")
            print(f"wrote {path}")
    _write_json(REFS / "mroz_tE_reference.json", ref)


# ----------------------------------------------------------------------
# genulens
# ----------------------------------------------------------------------
# Three configurations per sight line, none with binary lenses.
#
# * "remnants" -- THE comparison reference for M_L, D_L, D_S and mu_rel:
#   source weight n D^2 (gammaDs 2, as this prior has) and REMNANT 1, i.e.
#   every population's evolved stars become white dwarfs, neutron stars
#   and black holes (K21's present-day mass function; ~21% dark lenses
#   toward Baade's window).
# * "survey" -- the same with genulens's default source weighting
#   (gammaDs 0.5 ~ an I_s < 21 sample), the closest genulens gets to what
#   OGLE measured.  survey vs remnants differs ONLY in the source weight,
#   so it measures the one known model difference left (no source
#   luminosity function here).
# * "discard" -- REMNANT 0, which is NOT "no remnants": genulens then
#   DISCARDS every lens whose initial mass exceeds its population's dying
#   mass (event_sampler.cpp, `else { j--; continue; }`), so the dead mass
#   simply vanishes from the lens population.  Kept only to show the size
#   of that choice (Baade: M_L median 0.31 vs 0.46 Msun); never compare a
#   mass function against it.  Before 2026-10-30 this run was labelled
#   "matched" and was wrongly used as the mass reference.
GENULENS_VARIANTS = {
    "remnants": ["gammaDs", "2.0", "REMNANT", "1", "BINARY", "0"],
    "survey": ["REMNANT", "1", "BINARY", "0"],
    "discard": ["gammaDs", "2.0", "REMNANT", "0", "BINARY", "0"],
}
#: The variant every mass / geometry / kinematics comparison uses.
GENULENS_REFERENCE = "remnants"
# VERBOSITY 3 columns (genulens event_reporter.cpp print_event).
GENULENS_COLS = (
    "wtj M_L D_L D_S t_E theta_E pi_E pi_EN pi_EE mu_rel mu_Sl mu_Sb "
    "I_L K_L iS iL fREM"
).split()


def read_genulens(path):
    arr = np.loadtxt(path, comments="#", ndmin=2)
    if arr.shape[1] != len(GENULENS_COLS):
        raise ValueError(
            f"{path}: {arr.shape[1]} columns, expected VERBOSITY 3's "
            f"{len(GENULENS_COLS)}"
        )
    return {c: arr[:, i] for i, c in enumerate(GENULENS_COLS)}


def cmd_genulens(args):
    binary = Path(args.genulens).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.sightlines or list(SIGHTLINES)
    store = {}
    for name in names:
        sl = SIGHTLINES[name]
        store[name] = {"l": sl["l"], "b": sl["b"]}
        for variant, extra in GENULENS_VARIANTS.items():
            raw = out_dir / f"genulens_{name}_{variant}.out"
            cmd = [
                str(binary),
                "l",
                str(sl["l"]),
                "b",
                str(sl["b"]),
                "Nsimu",
                str(int(args.nsimu)),
                "seed",
                str(args.seed),
                "VERBOSITY",
                "3",
            ] + extra
            if not raw.exists() or args.force:
                t0 = time.time()
                # Written to a temporary name and renamed on success: an
                # existing raw file is SKIPPED on the next call, so a failed
                # run must not leave a (partial or empty) one behind.
                tmp = raw.with_suffix(".out.part")
                with open(tmp, "w") as fh:
                    # genulens reads input_files/ relative to its own
                    # checkout, so run it from there.
                    subprocess.run(
                        cmd, stdout=fh, check=True, cwd=binary.parent
                    )
                tmp.replace(raw)
                print(
                    f"{name}/{variant}: {time.time() - t0:.0f} s", flush=True
                )
            if args.run_only:
                continue
            d = read_genulens(raw)
            summary = [
                ln
                for ln in raw.read_text().splitlines()
                if ln.startswith("# avetE") or ln.startswith("# (n_BD")
            ]
            stats = summarize(
                {
                    "t_E": d["t_E"],
                    "D_L": d["D_L"],
                    "D_S": d["D_S"],
                    "mu_rel": d["mu_rel"],
                    "M_L": d["M_L"],
                    "theta_E": d["theta_E"],
                },
                d["wtj"],
            )
            # Rate-weighted lens types (fREM: 0 luminous, 1 WD, 2 NS, 3
            # BH) and the share of lenses above 1 Msun.
            wsum = d["wtj"].sum()
            stats["lens_type_fraction"] = {
                lab: float(d["wtj"][d["fREM"] == k].sum() / wsum)
                for k, lab in enumerate(("luminous", "WD", "NS", "BH"))
            }
            stats["frac_M_L_gt_1"] = float(
                d["wtj"][d["M_L"] > 1.0].sum() / wsum
            )
            stats["_cmd"] = " ".join(["genulens"] + cmd[1:])
            stats["_summary_lines"] = summary
            store[name][variant] = stats
            # A thinned, weight-RESAMPLED copy (equal weights) for the
            # Part B KS comparison, small enough to commit.
            rng = np.random.default_rng(args.seed)
            p = d["wtj"] / d["wtj"].sum()
            k = rng.choice(len(p), size=min(args.n_keep, len(p)), p=p)
            keep = np.column_stack(
                [
                    d[c][k]
                    for c in ("t_E", "D_L", "D_S", "mu_rel", "M_L", "fREM")
                ]
            )
            np.savetxt(
                REFS / f"genulens_{name}_{variant}_resampled.dat",
                keep,
                fmt="%.5g",
                header=(
                    f"genulens {stats['_cmd'][9:]}\n"
                    f"{args.n_keep} draws resampled by wtj (equal weight).\n"
                    "t_E[d] D_L[pc] D_S[pc] mu_rel_geo[mas/yr] M_L[Msun] "
                    "fREM[0 luminous, 1 WD, 2 NS, 3 BH]"
                ),
            )
    if args.run_only:
        return
    meta = {
        "_note": (
            "genulens (Koshimoto, Baba & Bennett 2021, ApJ 917, 78; "
            "github.com/nkoshimoto/genulens) wtj-weighted quantiles. "
            "mu_rel is GEOCENTRIC at genulens's default vEarthlb. "
            "Variants: remnants (gammaDs 2, REMNANT 1) is the reference; "
            "survey (gammaDs 0.5, REMNANT 1); discard (gammaDs 2, REMNANT "
            "0 -- DISCARDS evolved lenses, not a mass reference). K21 fit "
            "its IMF to the OGLE-IV t_E distributions (Mroz+2017/2019), so "
            "genulens agreeing with OGLE holds by construction."
        ),
        "_reference_variant": GENULENS_REFERENCE,
        "_genulens_commit": args.commit,
        "sightlines": store,
    }
    _write_json(REFS / "genulens_reference.json", meta)


# ----------------------------------------------------------------------
# sample: the EXOZIPPy prior
# ----------------------------------------------------------------------
def _radec(l_deg, b_deg):
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic").icrs
    return float(c.ra.deg), float(c.dec.deg)


def build_config(
    ra,
    dec,
    d_max_pc=20000.0,
    pm_max=50.0,
    rv_max_ms=1.0e6,
    lens_logmass_upper=None,
    present_day=None,
    model=None,
):
    """Lens + source + galactic model, no light curve.

    ``model`` selects the galactic model (``koshimoto21``, the genulens
    port, or ``simplified``); None leaves the component's default.

    ``present_day`` sets the galacticmodel's present-day mass function
    (review 1.7.2: MIST turnoff + WD/NS/BH remnants, per population; on by
    default); None leaves the model's default.

    The bounds tighten the defaults.yaml supports (1 mpc - 100 kpc,
    +/-1e5 mas/yr, +/-c) only to keep the logit coordinates well scaled.
    They cannot move the answer: the galactic densities are ~0 beyond 20
    kpc on a bulge sight line, and 50 mas/yr at 1 kpc is 240 km/s, i.e.
    > 5 sigma of the thick-disk dispersion and further out for the
    bulge; 1000 km/s on rv is > 7 sigma of any branch.  (A D_L below
    100 pc is excluded by the lower bound; its weight under n D^2 is
    negligible and the event-rate factor does not rescue it.)
    """
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        "mulensevent": [{"finite_source": False}],
        # anchor_idx 1: the line of sight is the SOURCE's, the microlensing
        # convention (examples/ob140939).
        "galacticmodel": [{"name": "gm", "anchor_idx": 1}],
        "logger_level": "WARNING",
    }
    if present_day is not None:
        config["galacticmodel"][0]["present_day"] = bool(present_day)
    if model is not None:
        config["galacticmodel"][0]["model"] = model
    params = {}
    for star, d0 in (("Lens", 6000.0), ("Source", 8500.0)):
        params[f"star.{star}.ra"] = {"initval": ra, "sigma": 0.0}
        params[f"star.{star}.dec"] = {"initval": dec, "sigma": 0.0}
        params[f"star.{star}.distance"] = {
            "initval": d0,
            "lower": 100.0,
            "upper": d_max_pc,
        }
        params[f"star.{star}.pm_ra"] = {"lower": -pm_max, "upper": pm_max}
        params[f"star.{star}.pm_dec"] = {"lower": -pm_max, "upper": pm_max}
        params[f"star.{star}.rv"] = {
            "initval": 0.0,
            "lower": -rv_max_ms,
            "upper": rv_max_ms,
        }
    params["star.Lens.logmass"] = {"initval": -0.5}
    if lens_logmass_upper is not None:
        # Localization switch (review 7.16.3): truncate the LENS mass
        # function, e.g. at the old bulge's main-sequence turnoff (~1
        # Msun, log 0.0) -- the IMF is an INITIAL mass function, and
        # the model applies it to lenses as if it were the present-day one.
        params["star.Lens.logmass"]["upper"] = float(lens_logmass_upper)
    return config, params


LENS_IDX, SOURCE_IDX = 0, 1  # config order in build_config


def sample_prior(
    sightline,
    *,
    draws,
    tune,
    chains,
    maxtime,
    seed=7163,
    target_accept=0.95,
    lens_logmass_upper=None,
    trace_path=None,
    present_day=None,
    model=None,
):
    """Sample the EXOZIPPy prior on one sight line.

    Returns ``(cols, idata, value_var_names, wall_seconds)``; ``cols`` maps
    t_E [d], theta_E [mas], pi_rel [mas], mu_rel / mu_rel_geo [mas/yr],
    D_L / D_S [pc] and M_L [Msun] to one array of draws each.
    """
    import pymc as pm
    import pytensor

    from exozippy.samplers.de_metropolis import _wallclock_callback
    from exozippy.system import System

    sl = SIGHTLINES[sightline]
    ra, dec = _radec(sl["l"], sl["b"])
    config, params = build_config(
        ra,
        dec,
        lens_logmass_upper=lens_logmass_upper,
        present_day=present_day,
        model=model,
    )
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()

    ev = system.mulensevent
    star = system.star
    assert int(ev.lens_bodies[0][1]) == LENS_IDX
    # What to record, as (name, Parameter, element).  Evaluated through the
    # model graph (most are derived, and a derived Parameter has no trace
    # copy unless something sampled feeds it -- see inject_recover.py).
    wanted = [
        ("t_E", ev.t_E, 0),
        ("theta_E", ev.theta_E, 0),
        ("pi_rel", ev.pi_rel, 0),
        ("mu_rel", ev.mu_rel_mag, 0),
        ("mu_rel_geo", ev.mu_rel_geo_mag, 0),
        ("D_L", star.distance, LENS_IDX),
        ("D_S", star.distance, SOURCE_IDX),
        ("M_L", star.mass, LENS_IDX),
    ]
    nodes = model.replace_rvs_by_values([p.value for _, p, _ in wanted])
    fn = pytensor.function(model.value_vars, nodes, on_unused_input="ignore")

    t0 = time.time()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            callback=_wallclock_callback(maxtime, "prior_predictive"),
            compute_convergence_checks=False,
        )
    wall = time.time() - t0
    if trace_path is not None:
        # Keep the trace before any post-processing can fail.
        idata.to_netcdf(str(trace_path))

    post = idata.posterior
    names_v = [vv.name for vv in model.value_vars]
    # exozippy samples raw coordinates with no PyMC transform, so every
    # value variable is in the posterior under its own name (the same
    # reading inject_recover.py does).
    missing = [n for n in names_v if n not in post]
    if missing:
        raise RuntimeError(f"value variables absent from the trace: {missing}")
    raws = [
        np.asarray(post[n]).reshape(-1, *np.asarray(post[n]).shape[2:])
        for n in names_v
    ]
    n_draws = raws[0].shape[0]
    cols = {name: np.empty(n_draws) for name, _, _ in wanted}
    for k in range(n_draws):
        vals = fn(*[r[k] for r in raws])
        for (name, p, i), v in zip(wanted, vals):
            cols[name][k] = float(
                p.from_internal(np.atleast_1d(v)[i], index=i)
            )
    return cols, idata, names_v, wall


def cmd_sample(args):
    sl = SIGHTLINES[args.sightline]
    ra, dec = _radec(sl["l"], sl["b"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / f"work_{args.sightline}"
    workdir.mkdir(exist_ok=True)
    os.chdir(workdir)
    tag = (
        ""
        if args.lens_logmass_upper is None
        else f"mlcap{args.lens_logmass_upper:+.2f}_"
    )
    if args.present_day is False:
        tag = "imf_" + tag
    if args.model is not None:
        tag = f"{args.model}_" + tag
    cols, idata, names_v, wall = sample_prior(
        args.sightline,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        maxtime=args.maxtime,
        seed=args.seed,
        target_accept=args.target_accept,
        lens_logmass_upper=args.lens_logmass_upper,
        trace_path=out_dir / f"exozippy_{tag}{args.sightline}_trace.nc",
        present_day=args.present_day,
        model=args.model,
    )
    post = idata.posterior

    behind = cols["D_L"] < cols["D_S"]
    stats = summarize(cols)
    import arviz as az

    sample_stats = idata.sample_stats
    out = {
        "label": tag.rstrip("_") or "baseline",
        "lens_logmass_upper": args.lens_logmass_upper,
        "sightline": args.sightline,
        "l": sl["l"],
        "b": sl["b"],
        "ra": ra,
        "dec": dec,
        "draws_per_chain": int(post.sizes["draw"]),
        "chains": int(post.sizes["chain"]),
        "wall_s": wall,
        "maxtime_s": args.maxtime,
        "divergences": int(np.asarray(sample_stats["diverging"]).sum()),
        "frac_DL_gt_DS": float(1 - behind.mean()),
        "stats": stats,
        "stats_DL_lt_DS": summarize({k: v[behind] for k, v in cols.items()}),
        # Over the SAMPLED raw coordinates (the value variables).  Passed
        # as (chain, draw, ...) arrays: arviz 1.x refuses multi-variable
        # selection on its DataTree.
        "ess_bulk_min": float(
            min(
                np.min(np.asarray(az.ess(np.asarray(post[n]))))
                for n in names_v
            )
        ),
        "rhat_max": float(
            max(
                np.max(np.asarray(az.rhat(np.asarray(post[n]))))
                for n in names_v
            )
        ),
    }
    if args.present_day is not False:
        # The galacticmodel's Deterministic: P(remnant) per PDMF star, in
        # star order (Lens first).  Its mean is the prior's rate-weighted
        # dark-lens fraction.
        p_rem = np.asarray(post["galacticmodel.p_remnant"])
        out["p_remnant_lens_mean"] = float(p_rem[..., LENS_IDX].mean())
    out["present_day"] = args.present_day is not False
    _write_json(out_dir / f"exozippy_{tag}{args.sightline}.json", out)
    np.savez_compressed(
        out_dir / f"exozippy_{tag}{args.sightline}_draws.npz", **cols
    )


# ----------------------------------------------------------------------
# compare: Part B of review 7.16.3
# ----------------------------------------------------------------------
def load_mroz_reference(sightline):
    """(kind, data) for a sight line's Mroz reference.

    kind "events": (t_E, weight) arrays, weight = inverse efficiency.
    kind "binned": (log_t_E bin edges, efficiency-corrected counts, raw
    counts) for Mroz 2017's Extended Data Tables 4/5.
    """
    paper, field = SIGHTLINES[sightline]["mroz"]
    if paper == "mroz2019":
        arr = np.loadtxt(REFS / f"mroz2019_table3_{field}.dat", usecols=(1, 3))
        return "events", (arr[:, 0], arr[:, 1])
    hdr, counts = _read_mroz2017("mroz2017_edt4_counts.dat")
    _, eff = _read_mroz2017("mroz2017_edt5_eff.dat")
    j = hdr.index(field)
    logc = counts[:, 1]
    edges = np.concatenate([logc - 0.07, [logc[-1] + 0.07]])
    return "binned", (edges, counts[:, j] / eff[:, j], counts[:, j])


def _q(x, w=None):
    w = np.ones_like(x) if w is None else w
    d = weighted_quantiles(x, w, (0.10, 0.50, 0.90))
    return np.array([d["q10"], d["q50"], d["q90"]])


def mroz_quantiles_with_errors(sightline, n_boot=2000, seed=0):
    """q10/q50/q90 of the reference and their bootstrap sigma: events
    resampled with replacement (weights travel with them), or, for the
    binned 2017 table, Poisson-resampled RAW counts divided by efficiency."""
    rng = np.random.default_rng(seed)
    kind, data = load_mroz_reference(sightline)
    if kind == "events":
        te, w = data
        q = _q(te, w)
        boots = []
        for _ in range(n_boot):
            k = rng.integers(0, len(te), len(te))
            boots.append(_q(te[k], w[k]))
    else:
        edges, corr, raw = data
        eff = np.where(raw > 0, raw / np.where(corr > 0, corr, 1), 1.0)

        def bq(c):
            cdf = np.concatenate([[0.0], np.cumsum(c)]) / c.sum()
            return np.array(
                [10 ** np.interp(p, cdf, edges) for p in (0.1, 0.5, 0.9)]
            )

        q = bq(corr)
        boots = [bq(rng.poisson(raw) / eff) for _ in range(n_boot)]
    boots = np.array(boots)
    return q, boots.std(axis=0), kind, data


def ks_against_mroz(te_model, kind, data):
    """KS distance between the model's t_E draws and the reference CDF.

    Events: the weighted ECDF, evaluated at every reference event and model
    draw.  Binned: at the 26 bin edges only (the reference has no finer
    resolution).  Returns (D, n_eff of the reference, asymptotic p) -- the
    p-value uses the reference's Kish n_eff, since the model sample is far
    larger and its own noise is negligible in comparison.
    """
    from scipy.stats import kstwo

    te_model = np.sort(np.asarray(te_model))
    if kind == "events":
        te, w = data
        o = np.argsort(te)
        te, w = te[o], w[o]
        grid = np.concatenate([te, te_model])
        f_ref = np.searchsorted(te, grid, side="right")
        cw = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
        f_ref = cw[f_ref]
        n_eff = w.sum() ** 2 / np.sum(w**2)
    else:
        edges, corr, raw = data
        grid = 10**edges
        f_ref = np.concatenate([[0.0], np.cumsum(corr)]) / corr.sum()
        # Model mass outside the tabulated range [0.1, 316] d is part of
        # the comparison: renormalize the model to the same range.
        lo, hi = grid[0], grid[-1]
        te_model = te_model[(te_model >= lo) & (te_model <= hi)]
        # Kish n_eff of a histogram of weighted events: var(c_i) =
        # N_i / eff_i^2 = c_i^2 / N_i.
        n_eff = corr.sum() ** 2 / np.sum(corr**2 / np.maximum(raw, 1))
    f_mod = np.searchsorted(te_model, grid, side="right") / te_model.size
    d = float(np.max(np.abs(f_mod - f_ref)))
    n = int(round(n_eff))
    return d, float(n_eff), float(kstwo.sf(d, n))


KAPPA = 8.144  # mas / Msun


def _read_resampled(sightline, variant):
    a = np.loadtxt(REFS / f"genulens_{sightline}_{variant}_resampled.dat")
    return {
        c: a[:, i]
        for i, c in enumerate(("t_E", "D_L", "D_S", "mu_rel", "M_L", "fREM"))
        if i < a.shape[1]
    }


def decompose(ez, gen, rng):
    """Localize a t_E difference.  t_E = sqrt(kappa M_L pi_rel) / mu_rel,
    and the rate weight theta_E mu_rel factorizes as sqrt(M_L) x
    sqrt(pi_rel) mu_rel, so when a model's mass function does not depend
    on position the rate-weighted lens mass is independent of geometry
    and kinematics and the mass function's share is isolated by swapping
    masses: t_E' = t_E sqrt(M'/M) with M' drawn from genulens's
    rate-weighted M_L (its reference variant, remnants ON).

    EXACT only for EXOZIPPy's ``present_day: false`` (one IMF everywhere).
    A present-day mass function differs by population (each has its own
    turnoff and remnants), in EXOZIPPy's default and in genulens alike,
    which couples the mass weakly to D_L; there the swap is an
    approximation.  What remains, t_E / sqrt(M_L), is geometry +
    kinematics, split further into pi_rel (density) and mu_rel
    (kinematics) marginals."""
    out = {}
    for name, d in (("exozippy", ez), ("genulens", gen)):
        pi_rel = 1000.0 / d["D_L"] - 1000.0 / d["D_S"]
        out[name] = {
            "t_E": _q(d["t_E"]),
            "M_L": _q(d["M_L"]),
            "tE_over_sqrtM": _q(d["t_E"] / np.sqrt(d["M_L"])),
            "pi_rel": _q(pi_rel),
            "mu_rel": _q(d["mu_rel"]),
            "D_L": _q(d["D_L"]),
            "D_S": _q(d["D_S"]),
        }
    m_swap = rng.choice(gen["M_L"], size=ez["M_L"].size)
    out["exozippy_with_genulens_masses"] = {
        "t_E": _q(ez["t_E"] * np.sqrt(m_swap / ez["M_L"]))
    }
    return {
        k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
        for k, v in out.items()
    }


def cmd_compare(args):
    ez_dir = Path(args.exozippy_dir)
    rng = np.random.default_rng(7163)
    result = {}
    lines = []
    for name in SIGHTLINES:
        paper, field = SIGHTLINES[name]["mroz"]
        q_ref, s_ref, kind, data = mroz_quantiles_with_errors(name)
        row = {
            "reference": f"{paper} {field}",
            "mroz_q10_q50_q90": q_ref.tolist(),
            "mroz_sigma": s_ref.tolist(),
            "models": {},
        }
        gen = {v: _read_resampled(name, v) for v in GENULENS_VARIANTS}
        models = {f"genulens_{v}": g["t_E"] for v, g in gen.items()}
        ez_runs = {}
        for f in sorted(ez_dir.glob(f"exozippy_*{name}_draws.npz")):
            label = (
                f.name[len("exozippy_") : -len(f"{name}_draws.npz")].rstrip(
                    "_"
                )
                or "baseline"
            )
            ez_runs[label] = dict(np.load(f))
            models[f"exozippy_{label}"] = ez_runs[label]["t_E"]
        for label, te in models.items():
            d, n_eff, p = ks_against_mroz(te, kind, data)
            row["models"][label] = {
                "q10_q50_q90": _q(np.asarray(te)).tolist(),
                "ks_D": d,
                "ks_n_eff_ref": n_eff,
                "ks_p": p,
            }
        if "baseline" in ez_runs:
            row[f"decomposition_vs_genulens_{GENULENS_REFERENCE}"] = decompose(
                ez_runs["baseline"], gen[GENULENS_REFERENCE], rng
            )
            # D_L / mu_rel / M_L against genulens (every variant; the
            # reference is GENULENS_REFERENCE), KS.
            from scipy.stats import ks_2samp

            row["vs_genulens"] = {
                v: {
                    c: {
                        "ks_D": float(
                            ks_2samp(ez_runs["baseline"][c], g[c]).statistic
                        ),
                        "q_exozippy": _q(ez_runs["baseline"][c]).tolist(),
                        "q_genulens": _q(g[c]).tolist(),
                    }
                    for c in ("D_L", "D_S", "mu_rel", "M_L", "t_E")
                }
                for v, g in gen.items()
            }
        result[name] = row
        lines.append(
            f"{name}: {paper} {field} t_E q10/q50/q90 = "
            + "/".join(f"{a:.1f}+-{b:.1f}" for a, b in zip(q_ref, s_ref))
        )
        for label, m in row["models"].items():
            lines.append(
                f"   {label:<24} "
                + "/".join(f"{x:.1f}" for x in m["q10_q50_q90"])
                + f"  KS D={m['ks_D']:.3f} p={m['ks_p']:.2g}"
            )
    _write_json(ez_dir / "comparison.json", result)
    print("\n".join(lines))


# ----------------------------------------------------------------------
# table
# ----------------------------------------------------------------------
def cmd_table(args):
    mroz = json.loads((REFS / "mroz_tE_reference.json").read_text())
    gpath = REFS / "genulens_reference.json"
    gen = json.loads(gpath.read_text())["sightlines"] if gpath.exists() else {}
    ez = {}
    for p in (
        sorted(Path(args.exozippy_dir).glob("exozippy_*.json"))
        if args.exozippy_dir
        else []
    ):
        d = json.loads(p.read_text())
        ez[d["sightline"]] = d
    fmt = "{:<9} {:<26} {:>7} {:>7} {:>7} {:>7}"
    print(fmt.format("sight", "source", "q10", "q50", "q90", "mean"))
    for name, sl in SIGHTLINES.items():
        paper, field = sl["mroz"]
        m = mroz[paper].get(field)
        if m:
            print(
                fmt.format(
                    name,
                    f"{paper} {field} t_E",
                    *(f"{m[k]:.1f}" for k in ("q10", "q50", "q90", "mean")),
                )
            )
        for variant, g in gen.get(name, {}).items():
            if isinstance(g, dict) and "t_E" in g:
                print(
                    fmt.format(
                        name,
                        f"genulens {variant} t_E",
                        *(
                            f"{g['t_E'][k]:.1f}"
                            for k in ("q10", "q50", "q90", "mean")
                        ),
                    )
                )
        if name in ez:
            e = ez[name]["stats"]["t_E"]
            print(
                fmt.format(
                    name,
                    f"exozippy {ez[name]['label']} t_E",
                    *(f"{e[k]:.1f}" for k in ("q10", "q50", "q90", "mean")),
                )
            )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="sample the EXOZIPPy prior (Slurm)")
    s.add_argument("--sightline", required=True, choices=sorted(SIGHTLINES))
    s.add_argument("--out-dir", required=True)
    s.add_argument("--draws", type=int, default=5000)
    s.add_argument("--tune", type=int, default=2000)
    s.add_argument("--chains", type=int, default=4)
    s.add_argument("--target-accept", type=float, default=0.95)
    s.add_argument("--seed", type=int, default=7163)
    s.add_argument(
        "--maxtime",
        type=float,
        required=True,
        help="wall-clock cap in s (graceful stop that keeps the draws)",
    )
    s.add_argument(
        "--lens-logmass-upper",
        type=float,
        default=None,
        help="localization switch: upper bound on the lens log10 mass",
    )
    s.add_argument(
        "--present-day",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="galacticmodel present_day (review 1.7.2's PDMF; the model's "
        "default is on, --no-present-day restores the IMF as the "
        "present-day mass function for comparison)",
    )
    s.add_argument(
        "--model",
        choices=["koshimoto21", "simplified"],
        default=None,
        help="galacticmodel model (default: the component's default)",
    )

    m = sub.add_parser("mroz", help="build the Mroz 2017/2019 references")
    m.add_argument("--mroz2019-dir", required=True)

    g = sub.add_parser("genulens", help="run genulens, store quantiles")
    g.add_argument("--genulens", required=True, help="path to the binary")
    g.add_argument("--out-dir", required=True, help="raw output directory")
    g.add_argument("--nsimu", type=float, default=2e6)
    g.add_argument("--seed", type=int, default=7163)
    g.add_argument("--n-keep", type=int, default=2000)
    g.add_argument("--commit", default="unknown")
    g.add_argument("--force", action="store_true")
    g.add_argument(
        "--run-only",
        action="store_true",
        help="only produce the raw outputs (parallel array tasks); a "
        "final call without it summarizes every sight line at once",
    )
    g.add_argument("--sightlines", nargs="*", choices=sorted(SIGHTLINES))

    c = sub.add_parser(
        "compare", help="Part B: t_E vs Mroz, KS, decomposition"
    )
    c.add_argument("--exozippy-dir", required=True)

    t = sub.add_parser("table", help="print stored quantiles side by side")
    t.add_argument("--exozippy-dir", default=None)

    args = ap.parse_args(argv)
    {
        "sample": cmd_sample,
        "mroz": cmd_mroz,
        "genulens": cmd_genulens,
        "compare": cmd_compare,
        "table": cmd_table,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
