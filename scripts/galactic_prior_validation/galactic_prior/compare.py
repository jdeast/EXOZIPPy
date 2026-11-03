"""Port vs genulens and port vs OGLE (numpy/scipy only).

THE GENULENS COMPARISONS, per sight line (all quantiles q05 ... q95, the
black-hole share of the rate, KS distance):

* ``plain``     the port's prior as it is vs PLAIN genulens (unmodified,
                default options);
* ``explained`` the port's draws restricted to plain genulens's support --
                lens in front of the source's line-of-sight bin, source
                inside Dmax = 16 kpc (``genulens_support_mask``) -- vs
                plain genulens;
* ``matched``   the port restricted to the bin-edge support only vs
                genulens run with the runtime option Dmax 20000 (the port's
                own distance reach).

A z-score uses the combined Monte Carlo error sqrt(se_port^2 + se_gen^2),
both weighted-bootstrap errors.
"""

import json
from pathlib import Path

import numpy as np

from . import refs, stats
from .sightlines import (
    SIGHTLINES,
    genulens_bin_width,
    genulens_support_mask,
    parse_run,
)

QS = stats.QUANTILES
KEYS = [stats.qkey(q) for q in QS]
# The headline quantiles in printed tables.
SHOW = ("q05", "q10", "q50", "q90", "q95")


def load_port(port_dir, run):
    d = np.load(Path(port_dir) / f"port_{run}_is.npz")
    cols = {k: d[k] for k in d.files if k != "log_w"}
    lw = d["log_w"]
    w = np.exp(lw - lw[np.isfinite(lw)].max())
    w[~np.isfinite(lw)] = 0.0
    info = json.loads((Path(port_dir) / f"port_{run}_is.json").read_text())
    # Diagnostics of the stored weights, recomputed.
    info["final"] = {
        **info["final"],
        **stats.importance_diagnostics(d["log_w"]),
    }
    return cols, w, info


def summarize(cols, w, n_boot=100, seed=0):
    """Quantiles, bootstrap se, BH share and n_eff of a weighted sample."""
    keep = w > 0
    cols = {k: np.asarray(cols[k])[keep] for k in refs.QUANTITIES}
    w = np.asarray(w)[keep]
    se = stats.bootstrap_quantile_se(cols, w, QS, n_boot=n_boot, seed=seed)
    out = {}
    for k in refs.QUANTITIES:
        q = stats.weighted_quantiles(cols[k], w, QS)
        out[k] = {
            "q": dict(zip(KEYS, q.tolist())),
            "se": dict(zip(KEYS, se[k].tolist())),
        }
    f, fse = stats.fraction_with_se(
        cols["M_L"] > refs.BH_MASS, w, n_boot, seed
    )
    out["bh_fraction"] = f
    out["bh_fraction_se"] = fse
    out["n_eff"] = stats.kish_ess(w)
    return out


def versus(a, b):
    """Percent difference and z of summary a against summary b."""
    out = {}
    for k in refs.QUANTITIES:
        qa, qb = a[k]["q"], b[k]["q"]
        sa, sb = a[k]["se"], b[k]["se"]
        out[k] = {
            "pct": {key: 100.0 * (qa[key] / qb[key] - 1.0) for key in KEYS},
            "z": {
                key: (qa[key] - qb[key]) / np.hypot(sa[key], sb[key])
                for key in KEYS
            },
        }
    out["bh_fraction_z"] = (a["bh_fraction"] - b["bh_fraction"]) / np.hypot(
        a["bh_fraction_se"], b["bh_fraction_se"]
    )
    return out


def ks_all(pc, pw, g):
    return {
        k: stats.weighted_ks(pc[k], pw, g[k], g["wtj"])
        for k in refs.QUANTITIES
    }


def genulens_masks(cols, run):
    line, _ = parse_run(run)
    sl = SIGHTLINES[line]
    dd = genulens_bin_width(sl["l"], sl["b"], refs.GENULENS_DMAX["plain"])
    dd20 = genulens_bin_width(sl["l"], sl["b"], refs.GENULENS_DMAX["dmax20"])
    if dd != dd20:
        raise RuntimeError(f"{run}: genulens bin width differs between Dmax")
    d_l, d_s = cols["D_L"] * 1000.0, cols["D_S"] * 1000.0
    return dd, {
        "explained": genulens_support_mask(
            d_l, d_s, dd, refs.GENULENS_DMAX["plain"]
        ),
        "matched": genulens_support_mask(d_l, d_s, dd),
        "edge_only_16": d_s < refs.GENULENS_DMAX["plain"],
    }


def compare_run(port_dir, gen_dir, run):
    cols, w, info = load_port(port_dir, run)
    dd, masks = genulens_masks(cols, run)
    port = {"full": summarize(cols, w)}
    for name in ("explained", "matched"):
        port[name] = summarize(cols, w * masks[name])
    gen = {
        v: refs.load_genulens(gen_dir, run, v) for v in refs.GENULENS_VARIANTS
    }
    gsum = {v: summarize(g, g["wtj"]) for v, g in gen.items()}
    for v, g in gen.items():
        gsum[v]["cmd"] = str(g["cmd"])
        gsum[v]["n_events"] = int(g["wtj"].size)
        tot = g["wtj"].sum()
        gsum[v]["lens_type_fraction"] = {
            lab: float(g["wtj"][g["fREM"] == k].sum() / tot)
            for k, lab in enumerate(("luminous", "WD", "NS", "BH"))
        }
    res = {
        "run": run,
        "genulens_bin_width_pc": dd,
        "port_diagnostics": info["final"],
        "port_info": {k: info[k] for k in ("l", "b", "binaries", "wall_s")},
        "port_rate_share_removed": {
            # The share of the port's rate outside each genulens support:
            # what the discretization removes.
            name: float(1.0 - (w * m).sum() / w.sum())
            for name, m in masks.items()
        },
        "port": port,
        "genulens": gsum,
        "comparisons": {},
    }
    # The two genulens discretizations, measured on the port's own draws:
    # (e) bin edge: full -> matched; (f) Dmax 16 kpc: matched -> explained.
    res["support_effects_pct"] = {
        name: {
            qty: {
                key: 100.0
                * (port[b][qty]["q"][key] / port[a][qty]["q"][key] - 1)
                for key in KEYS
            }
            for qty in refs.QUANTITIES
        }
        for name, (a, b) in {
            "bin_edge": ("full", "matched"),
            "dmax16": ("matched", "explained"),
        }.items()
    }
    pairs = {
        "plain": ("full", "plain", w),
        "explained": ("explained", "plain", w * masks["explained"]),
        "matched": ("matched", "dmax20", w * masks["matched"]),
    }
    for name, (pk, gv, pw) in pairs.items():
        c = versus(port[pk], gsum[gv])
        c["ks"] = ks_all(cols, pw, gen[gv])
        c["port"], c["genulens"] = pk, gv
        res["comparisons"][name] = c
    # OGLE (t_E only), for lines in an OGLE field: the port as it is, and
    # plain genulens for reference.
    line, _ = parse_run(run)
    if SIGHTLINES[line]["ogle"] and not run.endswith("_bin"):
        kind, data = refs.load_ogle(line)
        q, se = refs.ogle_quantiles(kind, data, QS)
        o = {
            "reference": " ".join(SIGHTLINES[line]["ogle"]),
            "kind": kind,
            "q": dict(zip(KEYS, q.tolist())),
            "se": dict(zip(KEYS, se.tolist())),
        }
        for label, (x, ww) in {
            "port": (cols["t_E"], w),
            "genulens_plain": (gen["plain"]["t_E"], gen["plain"]["wtj"]),
        }.items():
            d, n_eff, p = refs.ks_against_ogle(x, ww, kind, data)
            o[label] = {"ks_D": d, "ks_n_eff_ref": n_eff, "ks_p": p}
        res["ogle"] = o
    return res


def _fmt_q(s, qty):
    return "/".join(f"{s[qty]['q'][k]:.3g}" for k in SHOW)


def print_tables(results):
    lines = []
    for name in ("plain", "explained", "matched"):
        lines.append(f"\n## port vs genulens: {name}  (% difference, z)")
        for run, r in results.items():
            c = r["comparisons"][name]
            for qty in refs.QUANTITIES:
                lines.append(
                    f"{run:<11} {qty:<6} "
                    + " ".join(
                        f"{c[qty]['pct'][k]:+5.1f}({c[qty]['z'][k]:+4.1f})"
                        for k in SHOW
                    )
                    + f"  KS {c['ks'][qty]:.4f}"
                )
            gv = c["genulens"]
            lines.append(
                f"{run:<11} BH>3Msun port {r['port'][c['port']]['bh_fraction']:.4f}"
                f"+-{r['port'][c['port']]['bh_fraction_se']:.4f} genulens "
                f"{r['genulens'][gv]['bh_fraction']:.4f}"
                f"+-{r['genulens'][gv]['bh_fraction_se']:.4f}"
            )
    lines.append(
        "\n## genulens discretizations on the port's draws (% shift): "
        "t_E q05/q50/q90/q95, D_S q95"
    )
    for run, r in results.items():
        e = r["support_effects_pct"]
        lines.append(
            f"{run:<11} "
            + "  ".join(
                f"{name} "
                + "/".join(
                    f"{e[name]['t_E'][k]:+.2f}"
                    for k in ("q05", "q50", "q90", "q95")
                )
                + f", {e[name]['D_S']['q95']:+.2f}"
                for name in ("bin_edge", "dmax16")
            )
        )
    lines.append("\n## importance-sampling diagnostics")
    for run, r in results.items():
        d = r["port_diagnostics"]
        lines.append(
            f"{run:<11} n {d['n_draws']}  ESS {d['kish_ess']:.0f}  max w "
            f"{100 * d['max_weight_fraction']:.3f}%  khat "
            f"{d['pareto_khat']:.2f}  NaN {d['n_nan_logp']}"
        )
    lines.append("\n## t_E vs OGLE (q05/q10/q50/q90/q95, KS)")
    for run, r in results.items():
        if "ogle" not in r:
            continue
        o = r["ogle"]
        lines.append(
            f"{run:<11} {o['reference']:<17} OGLE "
            + "/".join(f"{o['q'][k]:.3g}" for k in SHOW)
            + "  port "
            + _fmt_q(r["port"]["full"], "t_E")
            + f"  KS D {o['port']['ks_D']:.3f} p {o['port']['ks_p']:.2g}"
            + f"  (genulens D {o['genulens_plain']['ks_D']:.3f}"
            + f" p {o['genulens_plain']['ks_p']:.2g})"
        )
    print("\n".join(lines))


def main(port_dir, gen_dir, runs, store=False, commit="unknown", processes=6):
    import multiprocessing as mp

    have = []
    for run in runs:
        if (port_dir / f"port_{run}_is.npz").exists():
            have.append(run)
        else:
            print(f"{run}: no port draws, skipped")
    # One process per sight line: each holds ~2e6 port draws and 4e6
    # genulens events (~1 GB), so the default is a few at a time.
    with mp.get_context("fork").Pool(min(processes, len(have))) as pool:
        out = pool.starmap(
            compare_run, [(port_dir, gen_dir, run) for run in have]
        )
    results = dict(zip(have, out))
    refs.write_json(port_dir / "comparison.json", results)
    (port_dir / "comparison.md").write_text(markdown_tables(results))
    print_tables(results)
    if store:
        _store(results, commit)


def _store(results, commit):
    """The stored summaries the slow test and the doc read."""
    gen = {
        "_note": (
            "genulens 2.0.0a4 (Koshimoto, Baba & Bennett 2021, ApJ 917, 78; "
            "github.com/nkoshimoto/genulens), UNMODIFIED, REMNANT 1, default "
            "source weighting (gammaDs 0.5), vEarthlb 0 0 (heliocentric "
            "mu_rel and t_E). plain: default options (Dmax 16 kpc); dmax20: "
            "runtime option Dmax 20000. q: wtj-weighted quantiles; se: "
            "weighted (Poisson) bootstrap, 100 replicates. D in kpc."
        ),
        "_genulens_commit": commit,
        "runs": {run: r["genulens"] for run, r in results.items()},
    }
    refs.write_json(refs.REFS / "genulens_reference.json", gen)
    port = {
        "_note": (
            "EXOZIPPy galacticmodel (model: koshimoto21) prior with no light "
            "curve, evaluated by importance sampling. full: as it is; "
            "explained: restricted to plain genulens's support (lens in "
            "front of the source's line-of-sight bin, D_S < 16 kpc); "
            "matched: the bin-edge support only. se: weighted bootstrap."
        ),
        "runs": {
            run: {
                "port": r["port"],
                "diagnostics": r["port_diagnostics"],
                "genulens_bin_width_pc": r["genulens_bin_width_pc"],
                "rate_share_removed": r["port_rate_share_removed"],
                "comparisons": r["comparisons"],
                "support_effects_pct": r["support_effects_pct"],
                **({"ogle": r["ogle"]} if "ogle" in r else {}),
            }
            for run, r in results.items()
        },
    }
    refs.write_json(refs.REFS / "port_is_reference.json", port)


def markdown_tables(results):
    """The doc's tables, as Markdown (written next to comparison.json)."""
    out = []
    head = "| line | quantity | " + " | ".join(SHOW) + " | KS D |"
    sep = "|---|---|" + "---|" * (len(SHOW) + 1)
    for name, title in (
        ("plain", "Port vs plain genulens: port / genulens - 1 in % (z)"),
        (
            "explained",
            "Port restricted to plain genulens's support vs plain genulens",
        ),
        (
            "matched",
            "Port with the bin-edge support vs genulens Dmax 20 kpc",
        ),
    ):
        out += [f"### {title}", "", head, sep]
        for run, r in results.items():
            c = r["comparisons"][name]
            for qty in refs.QUANTITIES:
                cells = [
                    f"{c[qty]['pct'][k]:+.1f} ({c[qty]['z'][k]:+.1f})"
                    for k in SHOW
                ]
                out.append(
                    f"| {run} | {qty} | "
                    + " | ".join(cells)
                    + f" | {c['ks'][qty]:.4f} |"
                )
        out.append("")
    out += [
        "### Quantiles (port as it is | plain genulens)",
        "",
        "| line | quantity | port "
        + "/".join(SHOW)
        + " | genulens "
        + "/".join(SHOW)
        + " |",
        "|---|---|---|---|",
    ]
    for run, r in results.items():
        for qty in refs.QUANTITIES:
            out.append(
                f"| {run} | {qty} | {_fmt_q(r['port']['full'], qty)} | "
                f"{_fmt_q(r['genulens']['plain'], qty)} |"
            )
    out += [
        "",
        "### Black-hole share of the rate (M_L > 3 Msun) and diagnostics",
        "",
        "| line | port | genulens plain | genulens Dmax 20 | IS draws | "
        "Kish ESS | max weight | Pareto k |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run, r in results.items():
        p, g, d = r["port"]["full"], r["genulens"], r["port_diagnostics"]
        out.append(
            f"| {run} | {100 * p['bh_fraction']:.2f} +/- "
            f"{100 * p['bh_fraction_se']:.2f}% | "
            f"{100 * g['plain']['bh_fraction']:.2f} +/- "
            f"{100 * g['plain']['bh_fraction_se']:.2f}% | "
            f"{100 * g['dmax20']['bh_fraction']:.2f} +/- "
            f"{100 * g['dmax20']['bh_fraction_se']:.2f}% | "
            f"{d['n_draws']:.2g} | {d['kish_ess']:.3g} | "
            f"{100 * d['max_weight_fraction']:.3f}% | "
            f"{d['pareto_khat']:.2f} |"
        )
    out += [
        "",
        "### The two genulens discretizations, measured on the port's draws",
        "",
        "| line | dD (pc) | rate share removed by (e) | (e): t_E q05 / q50 / "
        "q95 shift | rate share beyond 16 kpc | (f): t_E q95 / D_S q95 "
        "shift |",
        "|---|---|---|---|---|---|",
    ]
    for run, r in results.items():
        e = r["support_effects_pct"]
        rm = r["port_rate_share_removed"]
        out.append(
            f"| {run} | {r['genulens_bin_width_pc']:.0f} | "
            f"{100 * rm['matched']:.2f}% | "
            + " / ".join(
                f"{e['bin_edge']['t_E'][k]:+.2f}%"
                for k in ("q05", "q50", "q95")
            )
            + f" | {100 * rm['edge_only_16']:.2f}% | "
            f"{e['dmax16']['t_E']['q95']:+.2f}% / "
            f"{e['dmax16']['D_S']['q95']:+.2f}% |"
        )
    out += [
        "",
        "### t_E against OGLE-IV (efficiency corrected)",
        "",
        "| line | OGLE field | OGLE q05/q10/q50/q90/q95 (d) | +/- q50 | "
        "port | KS D (p) | genulens plain KS D (p) |",
        "|---|---|---|---|---|---|---|",
    ]
    for run, r in results.items():
        if "ogle" not in r:
            continue
        o = r["ogle"]
        out.append(
            f"| {run} | {o['reference']} | "
            + "/".join(f"{o['q'][k]:.3g}" for k in SHOW)
            + f" | {o['se']['q50']:.2g} | {_fmt_q(r['port']['full'], 't_E')} | "
            f"{o['port']['ks_D']:.3f} ({o['port']['ks_p']:.2g}) | "
            f"{o['genulens_plain']['ks_D']:.3f} "
            f"({o['genulens_plain']['ks_p']:.2g}) |"
        )
    return "\n".join(out) + "\n"
