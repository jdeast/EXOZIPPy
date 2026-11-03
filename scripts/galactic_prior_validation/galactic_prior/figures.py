"""Figures for the galactic-prior validation (matplotlib; PDF + PNG).

Colors are the Okabe & Ito colorblind-safe set, and every series also has
its own line style or marker, so no comparison depends on color alone.
"""

from pathlib import Path

import numpy as np

from . import compare, refs, stats
from .sightlines import SIGHTLINES, parse_run

PORT, GEN, OGLE, MATCH = "#0072B2", "#D55E00", "#000000", "#009E73"
LABEL = {
    "t_E": r"$t_{\rm E}$ (d)",
    "D_L": r"$D_{\rm L}$ (kpc)",
    "D_S": r"$D_{\rm S}$ (kpc)",
    "mu_rel": r"$|\mu_{\rm rel}|$ (mas yr$^{-1}$)",
    "M_L": r"$M_{\rm L}$ ($M_\odot$)",
}
LOGX = {"t_E": True, "D_L": False, "D_S": False, "mu_rel": False, "M_L": True}
RANGE = {
    "t_E": (0.5, 400.0),
    "D_L": (0.0, 11.0),
    "D_S": (5.0, 20.0),
    "mu_rel": (0.0, 20.0),
    "M_L": (0.01, 20.0),
}


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200)
    print(f"wrote {out_dir / name}.pdf/.png")


def _grid(qty, n=400):
    lo, hi = RANGE[qty]
    return np.geomspace(lo, hi, n) if LOGX[qty] else np.linspace(lo, hi, n)


def cdf_figure(port_dir, gen_dir, out_dir, run):
    """CDFs of the five quantities, port vs plain genulens, with residuals.

    Bottom panels: F_port - F_genulens for the plain comparison (port as it
    is) and the matched one (port restricted to genulens's bin-edge support
    vs genulens at Dmax 20 kpc), with the +/-2 sigma band expected from the
    two samples' sizes alone, sqrt(F (1 - F) (1/n_port + 1/n_gen)) with
    Kish effective sizes.
    """
    import matplotlib.pyplot as plt

    cols, w, _ = compare.load_port(port_dir, run)
    _, masks = compare.genulens_masks(cols, run)
    gp = refs.load_genulens(gen_dir, run, "plain")
    g20 = refs.load_genulens(gen_dir, run, "dmax20")
    wm = w * masks["matched"]
    n_p, n_gp = stats.kish_ess(w), stats.kish_ess(gp["wtj"])
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(7.2, 2.7),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
        sharex="col",
    )
    for j, qty in enumerate(refs.QUANTITIES):
        x = _grid(qty)
        fp = stats.weighted_ecdf(cols[qty], w, x)
        fg = stats.weighted_ecdf(gp[qty], gp["wtj"], x)
        fm = stats.weighted_ecdf(cols[qty], wm, x)
        f20 = stats.weighted_ecdf(g20[qty], g20["wtj"], x)
        ax, rx = axes[0, j], axes[1, j]
        ax.plot(x, fp, color=PORT, ls="-", label="EXOZIPPy")
        ax.plot(x, fg, color=GEN, ls="--", label="genulens (plain)")
        rx.plot(x, 100 * (fp - fg), color=GEN, ls="-", lw=1.0, label="plain")
        rx.plot(
            x, 100 * (fm - f20), color=MATCH, ls=":", lw=1.2, label="matched"
        )
        band = 200 * np.sqrt(
            np.clip(fg * (1 - fg), 0, None) * (1 / n_p + 1 / n_gp)
        )
        rx.fill_between(x, -band, band, color="0.85", lw=0)
        rx.axhline(0, color="0.5", lw=0.5)
        if LOGX[qty]:
            ax.set_xscale("log")
        ax.set_ylim(0, 1)
        rx.set_ylim(-1.5, 1.5)
        rx.set_xlabel(LABEL[qty])
        if j == 0:
            ax.set_ylabel("CDF")
            rx.set_ylabel(r"$\Delta F$ (%)")
        else:
            ax.set_yticklabels([])
            rx.set_yticklabels([])
    axes[0, 0].legend(loc="upper left", frameon=False)
    axes[1, 3].legend(loc="lower right", frameon=False, ncol=2, fontsize=6)
    line, binaries = parse_run(run)
    sl = SIGHTLINES[line]
    fig.suptitle(
        f"{run}  (l, b) = ({sl['l']:.2f}, {sl['b']:.2f})"
        + ("  binary lenses" if binaries else "")
        + f"   EXOZIPPy ESS {n_p:.3g}, genulens n_eff {n_gp:.3g}",
        y=1.0,
    )
    _save(fig, out_dir, f"cdf_{run}")
    plt.close(fig)


def summary_figure(results, out_dir):
    """Fractional quantile differences per line, plain and matched."""
    import matplotlib.pyplot as plt

    runs = list(results)
    show = ("q05", "q10", "q50", "q90", "q95")
    markers = ("v", "<", "o", ">", "^")
    shades = ("#56B4E9", "#0072B2", "#000000", "#D55E00", "#CC79A7")
    fig, axes = plt.subplots(
        5, 2, figsize=(7.2, 7.0), sharex=True, sharey="row"
    )
    for col, name in enumerate(("plain", "matched")):
        for row, qty in enumerate(refs.QUANTITIES):
            ax = axes[row, col]
            for k, (key, mk, c) in enumerate(zip(show, markers, shades)):
                xs, ys, es = [], [], []
                for i, run in enumerate(runs):
                    r = results[run]
                    cmp_ = r["comparisons"][name]
                    p = r["port"][cmp_["port"]][qty]
                    g = r["genulens"][cmp_["genulens"]][qty]
                    xs.append(i + (k - 2) * 0.13)
                    ys.append(cmp_[qty]["pct"][key])
                    es.append(
                        100
                        * np.hypot(p["se"][key], g["se"][key])
                        / g["q"][key]
                    )
                ax.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    fmt=mk,
                    ms=3.0,
                    color=c,
                    lw=0.8,
                    capsize=0,
                    label=key,
                )
            ax.axhline(0, color="0.5", lw=0.5)
            ax.axhspan(-1, 1, color="0.92", lw=0, zorder=0)
            if col == 0:
                ax.set_ylabel(LABEL[qty])
            if row == 0:
                ax.set_title(
                    "EXOZIPPy vs plain genulens"
                    if name == "plain"
                    else "EXOZIPPy (bin-edge support) vs genulens Dmax 20 kpc"
                )
    for ax in axes[-1]:
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=60, ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.supylabel(
        "EXOZIPPy / genulens - 1 (%)   [grey band: +/-1%]", x=0.0, fontsize=8
    )
    _save(fig, out_dir, "summary_quantiles")
    plt.close(fig)


def ogle_figure(port_dir, gen_dir, results, out_dir):
    """t_E CDFs per OGLE field: port, plain genulens, OGLE with its error."""
    import matplotlib.pyplot as plt

    runs = [r for r in results if "ogle" in results[r]]
    n = len(runs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(7.2, 2.1 * nrow), sharex=True, sharey=True
    )
    fig.subplots_adjust(hspace=0.45)
    axes = np.atleast_1d(axes).ravel()
    x = np.geomspace(0.3, 400, 400)
    for ax, run in zip(axes, runs):
        line, _ = parse_run(run)
        cols, w, _ = compare.load_port(port_dir, run)
        g = refs.load_genulens(gen_dir, run, "plain")
        kind, data = refs.load_ogle(line)
        if kind == "events":
            te, ww = data
            o = np.argsort(te)
            f = np.cumsum(ww[o]) / ww.sum()
            n_eff = stats.kish_ess(ww)
            ax.step(te[o], f, where="post", color=OGLE, lw=0.9, label="OGLE")
            err = np.sqrt(np.clip(f * (1 - f), 0, None) / n_eff)
            ax.fill_between(
                te[o], f - err, f + err, step="post", color="0.8", lw=0
            )
        else:
            edges, corr, raw = data
            f = np.concatenate([[0.0], np.cumsum(corr)]) / corr.sum()
            n_eff = corr.sum() ** 2 / np.sum(corr**2 / np.maximum(raw, 1))
            err = np.sqrt(np.clip(f * (1 - f), 0, None) / n_eff)
            ax.errorbar(
                10**edges,
                f,
                yerr=err,
                fmt="o",
                ms=2.2,
                color=OGLE,
                lw=0.7,
                label="OGLE",
            )
        ax.plot(
            x,
            stats.weighted_ecdf(cols["t_E"], w, x),
            color=PORT,
            label="EXOZIPPy",
        )
        ax.plot(
            x,
            stats.weighted_ecdf(g["t_E"], g["wtj"], x),
            color=GEN,
            ls="--",
            label="genulens",
        )
        o = results[run]["ogle"]
        ax.set_xscale("log")
        ax.set_xlim(0.5, 300)
        ax.set_ylim(0, 1)
        # In the title, clear of the curves.
        ax.set_title(
            f"{o['reference'].split()[1]} ({o['reference'].split()[0][4:]})\n"
            f"KS D = {o['port']['ks_D']:.3f}, p = {o['port']['ks_p']:.2g}",
            fontsize=6.5,
        )
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[(nrow - 1) * ncol : n]:
        ax.set_xlabel(LABEL["t_E"])
    for i in range(0, n, ncol):
        axes[i].set_ylabel("CDF")
    axes[min(1, n - 1)].legend(loc="lower right", frameon=False)
    _save(fig, out_dir, "ogle_tE")
    plt.close(fig)


def main(port_dir, gen_dir, out_dir, runs):
    import json

    _style()
    results = json.loads((port_dir / "comparison.json").read_text())
    results = {r: results[r] for r in runs if r in results}
    for run in results:
        cdf_figure(port_dir, gen_dir, out_dir, run)
    summary_figure(results, out_dir)
    ogle_figure(port_dir, gen_dir, results, out_dir)
