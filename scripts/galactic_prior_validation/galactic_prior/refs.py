"""Reference distributions: OGLE-IV (Mroz et al.) and genulens (numpy only)."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import stats
from .sightlines import SIGHTLINES, parse_run

REFS = Path(__file__).resolve().parent.parent / "refs"
QUANTITIES = ("t_E", "D_L", "D_S", "mu_rel", "M_L")
UNITS = {
    "t_E": "d",
    "D_L": "kpc",
    "D_S": "kpc",
    "mu_rel": "mas/yr",
    "M_L": "Msun",
}
BH_MASS = 3.0  # Msun: the trough between luminous/WD/NS lenses and black holes


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")


# ----------------------------------------------------------------------
# OGLE-IV: Mroz et al. 2017 (Nature 548, 183), Mroz et al. 2019 (ApJS 244, 29)
# ----------------------------------------------------------------------
def read_mroz2017(name):
    text = (REFS / name).read_text().splitlines()
    hdr = [ln for ln in text if ln.startswith("# bin")][0].split()[1:]
    rows = [ln.split() for ln in text if ln.strip() and not ln.startswith("#")]
    return hdr, np.array(rows, float)


def table3_events(path):
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


def table7(path):
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


def load_ogle(line):
    """(kind, data) for a sight line's OGLE reference.

    kind "events": (t_E, weight) arrays, weight = inverse efficiency
    (Mroz 2019 Table 3).  kind "binned": (log10 t_E bin edges,
    efficiency-corrected counts, raw counts) (Mroz 2017 Extended Data
    Tables 4 / 5).
    """
    paper, field = SIGHTLINES[line]["ogle"]
    if paper == "mroz2019":
        arr = np.loadtxt(REFS / f"mroz2019_table3_{field}.dat", usecols=(1, 3))
        return "events", (arr[:, 0], arr[:, 1])
    hdr, counts = read_mroz2017("mroz2017_edt4_counts.dat")
    _, eff = read_mroz2017("mroz2017_edt5_eff.dat")
    j = hdr.index(field)
    logc = counts[:, 1]
    edges = np.concatenate([logc - 0.07, [logc[-1] + 0.07]])
    return "binned", (edges, counts[:, j] / eff[:, j], counts[:, j])


def ogle_quantiles(kind, data, qs, n_boot=2000, seed=0):
    """Quantiles of the OGLE reference and their bootstrap sigma.

    Events are resampled with replacement (weights travel with them); the
    binned 2017 table is Poisson-resampled in its RAW counts, then divided
    by the efficiency.
    """
    rng = np.random.default_rng(seed)
    if kind == "events":
        te, w = data
        q = stats.weighted_quantiles(te, w, qs)
        boots = []
        for _ in range(n_boot):
            k = rng.integers(0, te.size, te.size)
            boots.append(stats.weighted_quantiles(te[k], w[k], qs))
    else:
        edges, corr, raw = data
        eff = np.where(raw > 0, raw / np.where(corr > 0, corr, 1), 1.0)
        q = stats.binned_quantiles(edges, corr, qs)
        boots = [
            stats.binned_quantiles(edges, rng.poisson(raw) / eff, qs)
            for _ in range(n_boot)
        ]
    return q, np.array(boots).std(axis=0, ddof=1)


def ogle_cdf(kind, data, grid):
    """The reference's CDF on ``grid`` (t_E in days)."""
    if kind == "events":
        te, w = data
        return stats.weighted_ecdf(te, w, grid)
    edges, corr, _ = data
    cdf = np.concatenate([[0.0], np.cumsum(corr)]) / corr.sum()
    return np.interp(np.log10(grid), edges, cdf, left=0.0, right=1.0)


def ks_against_ogle(te_model, w_model, kind, data):
    """KS distance between the model's weighted t_E and the OGLE CDF.

    Events: the weighted ECDFs, evaluated at every reference event and model
    draw.  Binned: at the 26 bin edges only (the reference has no finer
    resolution), with the model restricted to the tabulated range
    [0.1, 316] d.  Returns (D, n_eff of the reference, asymptotic p): the
    p-value uses the reference's Kish n_eff, since the model sample is far
    larger.  For the binned table D is a lower bound and p an upper one.
    """
    from scipy.stats import kstwo

    te_model = np.asarray(te_model, float)
    w_model = np.asarray(w_model, float)
    if kind == "events":
        te, w = data
        grid = np.unique(np.concatenate([te, te_model]))
        f_ref = stats.weighted_ecdf(te, w, grid)
        n_eff = stats.kish_ess(w)
    else:
        edges, corr, raw = data
        grid = 10**edges
        f_ref = np.concatenate([[0.0], np.cumsum(corr)]) / corr.sum()
        inside = (te_model >= grid[0]) & (te_model <= grid[-1])
        te_model, w_model = te_model[inside], w_model[inside]
        # Kish n_eff of a histogram of weighted events: var(c_i) = c_i^2 / N_i.
        n_eff = corr.sum() ** 2 / np.sum(corr**2 / np.maximum(raw, 1))
    f_mod = stats.weighted_ecdf(te_model, w_model, grid)
    d = float(np.max(np.abs(f_mod - f_ref)))
    return d, float(n_eff), float(kstwo.sf(d, int(round(n_eff))))


def build_ogle_reference(mroz2019_dir):
    """mroz_tE_reference.json: per-field efficiency-corrected t_E quantiles.

    Mroz 2017: Extended Data Table 4 (counts per field per log t_E bin) /
    Extended Data Table 5 (detection efficiency, same bins), nine
    high-cadence fields, 2010-2015, I_s <= 22 (transcribed in
    mroz2017_edt*.dat).  Mroz 2019: Table 3 (per-event best fit, low-cadence
    fields, 2010-2017) with its own inverse-efficiency weight column, and
    Table 7 (per-field efficiency-corrected mean t_E), both from
    https://ftp.astrouw.edu.pl/ogle/ogle4/microlensing_maps/ .  Table 3 is
    1.5 MB, so only the events of the reference fields are copied into the
    repo (mroz2019_table3_<field>.dat).
    """
    qs = stats.QUANTILES
    keys = [stats.qkey(q) for q in qs]
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
    hdr, counts = read_mroz2017("mroz2017_edt4_counts.dat")
    _, eff = read_mroz2017("mroz2017_edt5_eff.dat")
    logc = counts[:, 1]
    edges = np.concatenate([logc - 0.07, [logc[-1] + 0.07]])
    t7 = table7(Path(mroz2019_dir) / "table7.dat")
    for j, field in enumerate(hdr[2:], start=2):
        corr = counts[:, j] / eff[:, j]
        d = dict(zip(keys, stats.binned_quantiles(edges, corr, qs).tolist()))
        d["mean"] = float(np.sum(corr * 10**logc) / corr.sum())
        d["n_events"] = int(counts[:, j].sum())
        if field in t7:
            d["table7_tE_mean"] = t7[field]["tE_mean"]
            d["table7_tE_mean_err"] = t7[field]["tE_mean_err"]
            d["glon"], d["glat"] = t7[field]["glon"], t7[field]["glat"]
        ref["mroz2017"][field] = d

    events = table3_events(Path(mroz2019_dir) / "table3.dat")
    want = {
        v["ogle"][1]
        for v in SIGHTLINES.values()
        if v["ogle"] and v["ogle"][0] == "mroz2019"
    }
    for field in sorted({e[0][:6] for e in events}):
        ev = [e for e in events if e[0][:6] == field]
        te_best = np.array([e[1] for e in ev])
        te_med = np.array([e[2] for e in ev])
        w = np.array([e[3] for e in ev])
        d = dict(zip(keys, stats.weighted_quantiles(te_best, w, qs).tolist()))
        d["mean"] = float(np.sum(w * te_best) / w.sum())
        d["n"] = int(te_best.size)
        d["n_eff"] = stats.kish_ess(w)
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
    write_json(REFS / "mroz_tE_reference.json", ref)


# ----------------------------------------------------------------------
# genulens (Koshimoto, Baba & Bennett 2021; github.com/nkoshimoto/genulens)
# ----------------------------------------------------------------------
# Every run: remnants on (REMNANT 1, K21's present-day mass function), the
# default survey source weighting (gammaDs 0.5, the port's default), and
# vEarthlb 0 0, which makes genulens's mu_rel and t_E heliocentric like the
# port's with no light curve.  Both variants are UNMODIFIED genulens:
#
# * plain:  genulens's defaults otherwise -- its line-of-sight grid ends at
#           Dmax = 16 kpc;
# * dmax20: the runtime option Dmax 20000, the port's distance support.
GENULENS_VARIANTS = {
    "plain": [],
    "dmax20": ["Dmax", "20000"],
}
GENULENS_DMAX = {"plain": 16000.0, "dmax20": 20000.0}
# VERBOSITY 3 columns (genulens event_reporter.cpp print_event).
GENULENS_COLS = (
    "wtj M_L D_L D_S t_E theta_E pi_E pi_EN pi_EE mu_rel mu_Sl mu_Sb "
    "I_L K_L iS iL fREM"
).split()
GENULENS_BINARY_EXTRA = 6
GENULENS_KEEP = ("wtj", "M_L", "D_L", "D_S", "t_E", "mu_rel", "fREM")


def genulens_command(binary, run, variant, nsimu, seed):
    line, binaries = parse_run(run)
    sl = SIGHTLINES[line]
    return [
        str(binary),
        "l",
        str(sl["l"]),
        "b",
        str(sl["b"]),
        "Nsimu",
        str(int(nsimu)),
        "seed",
        str(int(seed)),
        "VERBOSITY",
        "3",
        "REMNANT",
        "1",
        "BINARY",
        "1" if binaries else "0",
        "vEarthlb",
        "0",
        "0",
    ] + GENULENS_VARIANTS[variant]


def run_genulens(binary, run, variant, nsimu, seed, out_dir):
    """Run genulens once and keep the needed columns as an .npz.

    genulens reads input_files/ relative to its own checkout, so it runs
    from there.  The text output (~170 bytes per event) is converted and
    deleted; distances come out in pc and are stored in kpc.
    """
    binary = Path(binary).resolve()
    out = Path(out_dir) / f"genulens_{run}_{variant}.npz"
    _, binaries = parse_run(run)
    cmd = genulens_command(binary, run, variant, nsimu, seed)
    raw = out.with_suffix(".out.part")
    t0 = time.time()
    with open(raw, "w") as fh:
        subprocess.run(cmd, stdout=fh, check=True, cwd=binary.parent)
    arr = np.loadtxt(raw, comments="#", ndmin=2)
    # BINARY 1 appends six binary-lens columns; the first 17 are unchanged
    # and M_L is then the EFFECTIVE lens mass (the one in theta_E).
    expect = len(GENULENS_COLS) + (GENULENS_BINARY_EXTRA if binaries else 0)
    if arr.shape[1] != expect:
        raise ValueError(
            f"{raw}: {arr.shape[1]} columns, expected VERBOSITY 3's {expect}"
        )
    cols = {c: arr[:, GENULENS_COLS.index(c)] for c in GENULENS_KEEP}
    cols["D_L"] = cols["D_L"] / 1000.0
    cols["D_S"] = cols["D_S"] / 1000.0
    np.savez(
        out,
        cmd=np.array(" ".join(["genulens"] + cmd[1:])),
        **{k: v.astype(np.float64) for k, v in cols.items()},
    )
    raw.unlink()
    print(f"{run}/{variant}: {arr.shape[0]} events, {time.time() - t0:.0f} s")
    return out


def load_genulens(out_dir, run, variant):
    d = np.load(Path(out_dir) / f"genulens_{run}_{variant}.npz")
    return {k: d[k] for k in d.files}
