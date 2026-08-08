"""Seeded-solution ledger: "considered and rejected" bookkeeping for
multimodal fits.

A posterior-suppressed mode (delta chi2 ~ 10 against the best solution)
carries ~zero T=1 occupancy, so it vanishes from the trace -- and with it
the distinction between "we considered this solution and rejected it at
delta lp = X" and "we never looked".  Systematics routinely make such modes
worth reporting anyway (published microlensing degenerate solutions are the
canonical case), and a referee wants their parameters, not just their
absence.

The ledger keeps them, without touching the sampling math:

- Every multi-seed start is already POLISHED to its own basin's optimum
  (polish.polish_raw_starts) before sampling, so its logp IS the basin's
  peak height.  build_seed_ledger measures, per seed, the local Gaussian
  widths with the same gradient-immune symmetric-curvature probe the
  whitening pass uses (whitening._probe_element; the model is whitened, so
  every bracket lands in a few evaluations) -- a Laplace approximation of
  each seeded mode: location, per-parameter sigmas, peak logp, and a
  relative Laplace log-weight lp_max + sum(log sigma_raw).
- After mode identification, match_ledger_to_modes assigns each seed to a
  surviving posterior mode (by distance in the raw feature space,
  normalized by the seed's own measured widths) or marks it REJECTED.
- Rejected seeds are reported: a section in <prefix>_modes.txt, rows in
  <prefix>_results.csv (mode column "rejected-seed<k>"), and a compact
  standalone LaTeX table <prefix>_rejected_modes.tex.

The Laplace numbers are approximations, labeled as such everywhere: good at
the "is this mode at the 1e-2 or the 1e-8 level" fidelity that reporting
needs, not for precision odds.  Values are quoted in each parameter's USER
units.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# A seed farther than this many of its own measured sigmas (Chebyshev over
# elements) from every surviving mode's center is "rejected".  Basins are
# separated by many sigma by construction (else they'd be one mode), so the
# threshold is not delicate; 10 keeps skew/nonlinearity from producing
# false rejections.
MATCH_SIGMA = 10.0

# Marginal-scale matching threshold (criterion (b) in match_ledger_to_modes):
# a mode's own median lies within ~2 of its marginal sigmas by construction,
# and the density-dip merge keeps distinct modes much farther apart.
_MATCH_MARGINAL_SIGMA = 3.0


@dataclass
class SeedRecord:
    seed_index: int  # original seed index (config order)
    lp_max: float  # logp at the (polished) seed point
    delta_lp: float  # lp_max(best seed) - lp_max  (>= 0)
    laplace_logw: float  # lp_max + sum(log sigma_raw): relative Laplace
    # log-weight (comparable ACROSS seeds only)
    raw_point: dict = field(default_factory=dict)  # raw coords (current)
    raw_scales: dict = field(default_factory=dict)  # curvature widths (raw)
    phys: dict = field(default_factory=dict)  # label -> user-unit values
    phys_sigma: dict = field(default_factory=dict)  # label -> user-unit widths
    sampled_idx: dict = field(default_factory=dict)  # label -> element indices
    matched_mode: Optional[int] = None  # 0-based surviving-mode index
    match_distance: float = float("nan")  # in seed sigmas
    source: str = "seed"  # "seed" | "hot-chain"


def _flat_names(key, size):
    """Element naming matching modes._feature_matrix: 'var' or 'var[j]'."""
    if size == 1:
        return [key]
    return [f"{key}[{j}]" for j in range(size)]


def build_seed_ledger(system, model, raw_starts, seed_indices, logp_fn=None):
    """Measure a Laplace record for every (polished) seed start.

    ``raw_starts``/``seed_indices`` are get_raw_starts' output in the
    CURRENT raw coordinates (i.e. after the whitening rescale).  Costs
    n_seeds x n_elements x O(15) logp evaluations -- the model is whitened,
    so the curvature brackets land immediately.

    Returns a list of SeedRecord ordered like ``raw_starts``.
    """
    from ..whitening import _probe_element

    if logp_fn is None:
        logp_fn = model.compile_logp()

    lookup = {p.label: p for p in system.get_all_parameters()}
    records = []
    for s, center in enumerate(raw_starts):
        lp0 = float(logp_fn(center))

        raw_scales = {}
        log_sigma_sum = 0.0
        for key, val in center.items():
            n = np.asarray(val).size
            sc = np.ones(n)
            for i in range(n):

                def eval_delta(step, key=key, i=i):
                    probe = {
                        k: np.array(v, dtype=float, copy=True)
                        for k, v in center.items()
                    }
                    probe[key].flat[i] += step
                    return lp0 - float(logp_fn(probe))

                scale, _method, _g = _probe_element(eval_delta)
                if scale is not None and np.isfinite(scale):
                    sc[i] = scale
            raw_scales[key] = sc.reshape(np.shape(val))
            log_sigma_sum += float(np.sum(np.log(sc)))

        # Physical (user-unit) center and half-widths through each
        # parameter's own frozen transform -- measured, not linearized.
        phys, phys_sigma, sampled_idx = {}, {}, {}
        for key, val in center.items():
            name = key[: -len("_raw")] if key.endswith("_raw") else key
            par = lookup.get(name)
            tf = getattr(par, "_raw_transform", None) if par else None
            if tf is None:
                continue
            c = np.asarray(val, dtype=float).reshape(-1)
            sc = np.asarray(raw_scales[key], dtype=float).reshape(-1)
            if c.size != len(tf["sampled_idx"]):
                continue
            p0 = np.asarray(par.phys_from_raw(c), dtype=float)
            p_hi = np.asarray(par.phys_from_raw(c + sc), dtype=float)
            p_lo = np.asarray(par.phys_from_raw(c - sc), dtype=float)
            # internal -> user units
            factors = np.asarray(
                par._get_conversion_factors(), dtype=float
            ).reshape(-1)
            if factors.size != p0.size:
                factors = np.ones(p0.size)
            phys[name] = p0 / factors
            phys_sigma[name] = 0.5 * np.abs(p_hi - p_lo) / factors
            sampled_idx[name] = list(tf["sampled_idx"])

        records.append(
            SeedRecord(
                seed_index=int(seed_indices[s]),
                lp_max=lp0,
                delta_lp=np.nan,  # filled below
                laplace_logw=lp0 + log_sigma_sum,
                raw_point={
                    k: np.array(v, dtype=float, copy=True)
                    for k, v in center.items()
                },
                raw_scales=raw_scales,
                phys=phys,
                phys_sigma=phys_sigma,
                sampled_idx=sampled_idx,
            )
        )

    best = max(r.lp_max for r in records) if records else np.nan
    for r in records:
        r.delta_lp = best - r.lp_max
    return records


def match_ledger_to_modes(ledger, mode_report, match_sigma=MATCH_SIGMA):
    """Assign each ledger record to a surviving posterior mode, or reject.

    Distance is Chebyshev over the mode-feature elements, each normalized
    by the seed's own measured width along that element.  A record whose
    nearest mode center is farther than ``match_sigma`` is left unmatched
    (matched_mode = None): considered and rejected by the posterior.
    """
    if mode_report is None or not getattr(mode_report, "modes", None):
        return ledger
    for rec in ledger:
        best_mode, best_d = None, np.inf
        # element name -> (value, scale) in the current raw coordinates
        elems = {}
        for key, val in rec.raw_point.items():
            flat_v = np.asarray(val, dtype=float).reshape(-1)
            flat_s = np.asarray(rec.raw_scales[key], dtype=float).reshape(-1)
            for j, nm in enumerate(_flat_names(key, flat_v.size)):
                elems[nm] = (flat_v[j], max(flat_s[j], 1e-300))
        for m in mode_report.modes:
            # Two complementary criteria, either suffices:
            # (a) within `match_sigma` of the mode center in the seed's own
            #     CONDITIONAL (curvature) widths -- catches a center right
            #     at the polished peak;
            # (b) within _MATCH_MARGINAL_SIGMA of the center in the mode's
            #     per-dim MARGINAL scale (ModeInfo.center_scale) -- the mode
            #     center is a posterior MEDIAN, which on correlated
            #     posteriors sits tens of conditional sigmas from the basin
            #     peak, so (a) alone falsely rejects surviving basins
            #     (observed on ob140939: all four seeds rejected, two of
            #     them sitting inside the two surviving modes). A mode
            #     always holds its own median within ~2 marginal sigmas,
            #     while the density-dip merge guarantees OTHER basins sit
            #     much farther, so 3 is not delicate.
            scales = getattr(m, "center_scale", {}) or {}
            ds_cond, ds_marg = [], []
            for nm, c in m.center.items():
                if nm not in elems:
                    continue
                delta = abs(elems[nm][0] - c)
                ds_cond.append(delta / elems[nm][1])
                ds_marg.append(delta / max(scales.get(nm, 0.0), 1e-300))
            if not ds_cond:
                continue
            # effective distance: fraction of whichever threshold is closer
            d = min(
                max(ds_cond) / match_sigma,
                max(ds_marg) / _MATCH_MARGINAL_SIGMA,
            )
            if d < best_d:
                best_mode, best_d = m.index, d
        if best_mode is not None and best_d <= 1.0:
            rec.matched_mode = best_mode
            rec.match_distance = best_d
        else:
            rec.matched_mode = None
            rec.match_distance = best_d
    return ledger


def rejected_records(ledger):
    return [r for r in ledger if r.matched_mode is None]


def ledger_to_text(ledger):
    """Human-readable ledger section (appended to <prefix>_modes.txt)."""
    lines = []
    lines.append("")
    lines.append("Seeded-solution ledger (Laplace approximations)")
    lines.append("-----------------------------------------------")
    lines.append(
        "Every seeded start was polished to its basin optimum before "
        "sampling; widths are"
    )
    lines.append(
        "symmetric-curvature (Laplace) estimates at that optimum, NOT "
        "posterior draws."
    )
    for r in sorted(ledger, key=lambda r: r.seed_index):
        lines.append("")
        status = (
            f"survived as mode {r.matched_mode + 1} "
            f"(match distance {r.match_distance:.2f} of threshold)"
            if r.matched_mode is not None
            else "REJECTED: no surviving posterior mode at this solution"
        )
        lines.append(f"seed {r.seed_index} ({r.source}): {status}")
        lines.append(
            f"  lp at optimum = {r.lp_max:.2f}  (delta vs best seed = "
            f"{r.delta_lp:.2f}; Laplace log-weight vs best is comparable "
            "at the ~1-nat level)"
        )
        if r.matched_mode is None:
            for name in sorted(r.phys):
                vals = np.asarray(r.phys[name]).reshape(-1)
                sigs = np.asarray(r.phys_sigma[name]).reshape(-1)
                idx = r.sampled_idx.get(name, range(vals.size))
                parts = [f"[{i}] {vals[i]:.6g} +/- {sigs[i]:.3g}" for i in idx]
                lines.append(f"    {name}: " + "; ".join(parts))
    return "\n".join(lines) + "\n"


def append_ledger_csv(ledger, csv_filename):
    """Append rejected-mode Laplace rows to the results CSV.

    Row shape matches build_csv_output: name, mode, weight, value, +err,
    -err -- with mode = 'rejected-seed<k>' and weight the Laplace weight
    relative to the best seed (an upper-bound-flavored estimate, labeled
    by the mode column itself).
    """
    rej = rejected_records(ledger)
    if not rej:
        return
    best_logw = max(r.laplace_logw for r in ledger)
    with open(csv_filename, "a", encoding="utf-8") as f:
        for r in rej:
            w = float(np.exp(r.laplace_logw - best_logw))
            for name in sorted(r.phys):
                vals = np.asarray(r.phys[name]).reshape(-1)
                sigs = np.asarray(r.phys_sigma[name]).reshape(-1)
                for i in r.sampled_idx.get(name, range(vals.size)):
                    f.write(
                        f"{name},rejected-seed{r.seed_index},{w:.3g},"
                        f"{vals[i]:.6g},{sigs[i]:.3g},{sigs[i]:.3g}\n"
                    )
    logger.info(
        f"Ledger: appended {len(rej)} rejected mode(s) to {csv_filename}"
    )


def write_rejected_latex(ledger, filename):
    """Standalone LaTeX table of the rejected solutions (Laplace)."""
    rej = rejected_records(ledger)
    if not rej:
        return False
    lines = []
    lines.append(r"% Considered-and-rejected solutions (Laplace")
    lines.append(r"% approximations at each seeded basin's polished optimum;")
    lines.append(r"% no posterior draws exist for these).")
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Solutions considered and rejected by the posterior. "
        r"Values are Laplace approximations at each seeded basin's "
        r"optimum; $\Delta \ln \mathcal{L}$ is measured against the best "
        r"seeded solution.}"
    )
    cols = "l" + "c" * len(rej)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\hline")
    header = ["Parameter"] + [f"seed {r.seed_index}" for r in rej]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\hline")
    dl = [r"$\Delta \ln \mathcal{L}$"] + [f"$-{r.delta_lp:.1f}$" for r in rej]
    lines.append(" & ".join(dl) + r" \\")
    names = sorted({n for r in rej for n in r.phys})
    for name in names:
        n_el = max(
            np.asarray(r.phys[name]).reshape(-1).size
            for r in rej
            if name in r.phys
        )
        el_lists = [r.sampled_idx.get(name, list(range(n_el))) for r in rej]
        for i in sorted({i for lst in el_lists for i in lst}):
            row = [f"{name}[{i}]".replace("_", r"\_")]
            for r in rej:
                vals = np.asarray(r.phys.get(name, []), dtype=float).reshape(
                    -1
                )
                sigs = np.asarray(
                    r.phys_sigma.get(name, []), dtype=float
                ).reshape(-1)
                if i < vals.size and i in r.sampled_idx.get(name, []):
                    row.append(f"${vals[i]:.6g} \\pm {sigs[i]:.3g}$")
                else:
                    row.append("--")
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Ledger: wrote rejected-mode table {filename}")
    return True


# ---------------------------------------------------------------------------
# Hot-chain mode discovery (PTDE store_hot_chains)
# ---------------------------------------------------------------------------

# A hot draw whose UNtempered lp is within this many nats of the best lp
# seen anywhere in the hot set is "near-viable": worth clustering as a
# candidate mode. Matches the spirit of modes.DEFAULT_LP_EXEMPT_MARGIN --
# a genuinely real mode more than 50 nats down carries e^-50 of the mass
# and loses nothing by being ignored.
HOT_LP_MARGIN = 50.0

# Fewer near-viable points than this cannot support a cluster.
HOT_MIN_POINTS = 25


def discover_hot_modes(
    system,
    model,
    hot,
    seed_ledger=None,
    margin_nats=HOT_LP_MARGIN,
    min_points=HOT_MIN_POINTS,
    max_modes=8,
    subsample=20000,
    seed=20260711,
    polish_steps=150,
):
    """Find posterior-suppressed modes in the thinned hot-rung draws.

    ``hot`` is the ``posterior_hot`` group written by
    ptde_async_sample(store_hot_chains=...): raw variables + UNtempered
    ``lp`` with dims (chain, draw). Hot draws are DETECTORS only -- a
    T-tempered mode is ~sqrt(T) too wide, so nothing here uses their
    spread. The pipeline is: filter to near-viable draws (lp within
    ``margin_nats`` of the best), cluster them (the same BIC k-means +
    density-dip merge the T=1 mode identification uses), take each
    cluster's best-lp draw as a candidate, POLISH it to its basin optimum
    (polish.polish_raw_starts), Laplace-characterize it
    (build_seed_ledger), and append it to ``seed_ledger`` unless an
    existing record already sits in the same basin.

    Appended records carry source='hot-chain' and seed indices continuing
    after the seeded ones. Returns the (possibly extended) ledger list.
    match_ledger_to_modes then classifies them against the surviving T=1
    modes like any other record -- a hot-chain record matching a surviving
    mode is simply confirmation; an unmatched one is a mode the T=1
    posterior never held.
    """
    from .modes import _dip_merge, _kmeans_bic

    ledger = list(seed_ledger) if seed_ledger else []

    raw_keys = [str(v) for v in hot.data_vars if str(v) != "lp"]
    if not raw_keys or "lp" not in hot.data_vars:
        return ledger
    lp = np.asarray(hot["lp"].values, dtype=float).reshape(-1)

    cols, names, shapes = [], [], {}
    for key in raw_keys:
        arr = np.asarray(hot[key].values, dtype=float)
        n_chain, n_draw = arr.shape[0], arr.shape[1]
        arr = arr.reshape(n_chain * n_draw, -1)
        shapes[key] = arr.shape[1]
        for j in range(arr.shape[1]):
            cols.append(arr[:, j])
        names.extend(_flat_names(key, arr.shape[1]))
    X = np.column_stack(cols)

    good = np.isfinite(lp) & np.all(np.isfinite(X), axis=1)
    if not good.any():
        return ledger
    viable = good & (lp >= np.nanmax(lp[good]) - margin_nats)
    n_viable = int(viable.sum())
    if n_viable < min_points:
        logger.info(
            f"Hot-chain discovery: only {n_viable} near-viable hot draws "
            f"(need {min_points}); nothing to cluster."
        )
        return ledger
    Xv, lpv = X[viable], lp[viable]

    rng = np.random.default_rng(seed)
    if Xv.shape[0] > subsample:
        keep = rng.choice(Xv.shape[0], size=subsample, replace=False)
        Xv, lpv = Xv[keep], lpv[keep]

    # Standardize for clustering (hot spreads are wide but finite).
    mu = np.median(Xv, axis=0)
    sig = np.maximum(np.std(Xv, axis=0), 1e-12)
    Z = (Xv - mu) / sig
    labels, centers = _kmeans_bic(Z, max_modes=max_modes, seed=seed)
    # iterate the density-dip merge to a fixed point, as identify_modes does
    while True:
        labels, centers, changed = _dip_merge(
            Z, labels, centers, merge_ratio=0.5
        )
        if not changed:
            break
    n_clusters = len(np.unique(labels[labels >= 0]))
    logger.info(
        f"Hot-chain discovery: {n_viable} near-viable draws -> "
        f"{n_clusters} cluster(s)."
    )

    from ..polish import polish_raw_starts

    next_index = max((r.seed_index for r in ledger), default=-1) + 1
    for c in np.unique(labels[labels >= 0]):
        sel = labels == c
        best = int(np.argmax(lpv[sel]))
        x_best = Xv[sel][best]
        # rebuild the raw dict for this candidate
        cand, ofs = {}, 0
        for key in raw_keys:
            n = shapes[key]
            cand[key] = np.array(x_best[ofs : ofs + n], dtype=float)
            ofs += n

        polished, _dlps, _method = polish_raw_starts(
            model, [cand], n_steps=polish_steps
        )
        recs = build_seed_ledger(system, model, polished, [next_index])
        rec = recs[0]
        rec.source = "hot-chain"

        # Dedup: an existing record already sitting in this basin (within
        # MATCH_SIGMA of the polished point, in the new record's own
        # widths) makes this a rediscovery, not a discovery.
        dup = False
        for other in ledger:
            ds = []
            for key in raw_keys:
                a = np.asarray(rec.raw_point[key], dtype=float).reshape(-1)
                b = np.asarray(
                    other.raw_point.get(key, np.full_like(a, np.nan)),
                    dtype=float,
                ).reshape(-1)
                s = np.asarray(rec.raw_scales[key], dtype=float).reshape(-1)
                if b.shape != a.shape or not np.all(np.isfinite(b)):
                    ds = []
                    break
                ds.extend(np.abs(a - b) / np.maximum(s, 1e-300))
            if ds and max(ds) <= MATCH_SIGMA:
                dup = True
                break
        if dup:
            continue
        ledger.append(rec)
        next_index += 1
        logger.info(
            f"Hot-chain discovery: new basin at lp={rec.lp_max:.2f} "
            f"recorded as ledger entry {rec.seed_index} (hot-chain)."
        )

    # delta_lp is relative to the best record across the WHOLE ledger.
    if ledger:
        best_lp = max(r.lp_max for r in ledger)
        for r in ledger:
            r.delta_lp = best_lp - r.lp_max
    return ledger
