"""Post-hoc identification, splitting, and reporting of posterior modes.

Works from an arviz InferenceData alone (no model or System required), so it
runs identically on a freshly sampled trace and on a trace loaded from disk.

Pipeline (see identify_modes):
  1. Build a feature matrix from the raw (unconstrained, sampled-space)
     variables in the posterior group -- the ``*_raw`` companions that
     Parameter.build_pymc stores next to every sampled parameter.  The raw
     space has no bound pileups and no unit issues, and circular parameters
     are already embedded (alpha is sampled as xalpha/yalpha), so Euclidean
     clustering is meaningful there.
  2. Reject invalid draws: non-finite features or lp, implausible |lp|, or
     raw values astronomically far from the bulk (robust z-score).  These are
     runaway/stuck chains pinned at parameter bounds, not posterior modes.
  3. Cluster the valid draws with k-means, selecting k by BIC.
  4. Merge over-split clusters with a density-dip test along the segment
     connecting each pair of cluster centers: if the empirical density does
     not dip between the two centers, they are one mode (this un-splits
     curved/banana-shaped posteriors that k-means fragments).
  5. Drop modes below ``min_weight``; order the survivors by weight.
  6. Compute per-chain occupancies and inter-mode transition counts, and
     derive a *provenance* label for the reported weights: draw-count
     occupancy equals posterior mass only if the sampler actually mixed
     between modes.  When chains do not mix, the weights are flagged
     UNRELIABLE (they reflect initialization, not the posterior).

The mode labels are attached to the InferenceData as an integer posterior
variable ``mode`` (chain, draw), -1 for invalid/unassigned draws, so a single
InferenceData object carries the full multimodal solution and every
downstream consumer (distribute_posterior, tables, plots) can filter on it.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Draws whose |lp| exceeds this are numerically broken, not a mode.  A real
# log-posterior scales with the number of data points; no realistic dataset
# reaches 1e12.
DEFAULT_LP_ABS_MAX = 1e12

# Robust z-score threshold in raw space.  Non-centered raw coordinates can
# legitimately sit hundreds of init-scales from the origin, but they cluster
# tightly there; draws thousands of *posterior widths* from the bulk are
# runaways pinned at bounds (observed values: 1e3..1e27 vs bulk ~1e2).
DEFAULT_Z_MAX = 50.0


def _idx_to_words(n):
    words = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
             '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
             '8': 'eight', '9': 'nine'}
    return "".join(words[char] for char in str(n))


def mode_suffix(k):
    """LaTeX-macro-safe suffix for mode ``k`` (0-based): 'modeone', ..."""
    return "mode" + _idx_to_words(k + 1)


@dataclass
class ModeInfo:
    index: int                    # 0-based, ordered by weight (descending)
    weight: float                 # fraction of valid assigned draws
    n_draws: int
    lp_med: float
    lp_max: float
    delta_lp_max: float           # lp_max(best mode) - lp_max(this mode)
    per_chain_weight: np.ndarray  # occupancy fraction of each chain's valid draws
    center: dict = field(default_factory=dict)  # feature var -> center (raw units)


@dataclass
class ModeReport:
    labels: np.ndarray            # (chain, draw) int; -1 = invalid/unassigned
    modes: List[ModeInfo]
    n_valid: int
    n_invalid: int
    n_unassigned: int             # valid draws in dropped minor clusters
    provenance: str
    weights_reliable: bool
    n_transitions: int            # inter-mode label changes along chains
    feature_vars: List[str]
    notes: List[str] = field(default_factory=list)

    @property
    def n_modes(self):
        return len(self.modes)

    @property
    def weights(self):
        return [m.weight for m in self.modes]

    def attach(self, idata):
        """Store labels as posterior variable ``mode`` on the InferenceData."""
        import xarray as xr
        post = idata.posterior
        da = xr.DataArray(
            self.labels.astype(np.int16), dims=("chain", "draw"),
            coords={"chain": post.chain, "draw": post.draw})
        da.attrs["n_modes"] = self.n_modes
        da.attrs["weights"] = [float(w) for w in self.weights]
        da.attrs["provenance"] = self.provenance
        da.attrs["n_invalid"] = int(self.n_invalid)
        post["mode"] = da
        return idata

    def to_text(self):
        lines = []
        lines.append("Posterior mode report")
        lines.append("=====================")
        n_total = self.labels.size
        lines.append(f"draws: {n_total} total, {self.n_valid} valid, "
                     f"{self.n_invalid} invalid (rejected), "
                     f"{self.n_unassigned} in minor/unassigned clusters")
        lines.append(f"modes found: {self.n_modes}")
        lines.append(f"weight provenance: {self.provenance}")
        lines.append(f"inter-mode transitions (all chains): {self.n_transitions}")
        for m in self.modes:
            lines.append("")
            lines.append(f"mode {m.index + 1}:")
            lines.append(f"  weight   = {m.weight:.4f}")
            lines.append(f"  n_draws  = {m.n_draws}")
            lines.append(f"  lp med/max = {m.lp_med:.2f} / {m.lp_max:.2f}"
                         f"  (delta lp_max vs best = {m.delta_lp_max:.2f})")
            occ = np.asarray(m.per_chain_weight)
            lines.append(f"  chains containing this mode: "
                         f"{int((occ > 0).sum())}/{occ.size}"
                         f"  (per-chain occupancy min/med/max = "
                         f"{occ.min():.2f}/{np.median(occ):.2f}/{occ.max():.2f})")
        if self.notes:
            lines.append("")
            lines.append("notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines) + "\n"


# ----------------------------
# internals
# ----------------------------

def _feature_matrix(post, feature_vars):
    """Stack the requested posterior variables into (n_samples, n_dims).

    Sample order is (chain, draw) row-major, matching both
    labels.reshape(chain, draw) and az.extract's stacked sample dim.
    """
    cols = []
    names = []
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    for v in feature_vars:
        arr = np.asarray(post[v].values, dtype=float)
        arr = arr.reshape(n_chain * n_draw, -1)
        for j in range(arr.shape[1]):
            cols.append(arr[:, j])
            names.append(v if arr.shape[1] == 1 else f"{v}[{j}]")
    X = np.column_stack(cols)
    return X, names


def _robust_center_scale(x):
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    return med, mad


def _kmeans_bic(X, max_modes, seed):
    """k-means over k = 1..max_modes; pick k by spherical-Gaussian BIC."""
    from scipy.cluster.vq import kmeans2

    n, d = X.shape
    rng = np.random.default_rng(seed)
    best = None  # (bic, labels, centers)
    for k in range(1, max_modes + 1):
        if k == 1:
            labels = np.zeros(n, dtype=int)
            centers = X.mean(axis=0)[None, :]
        else:
            try:
                centers, labels = kmeans2(
                    X, k, minit="++", seed=rng.integers(2**31), iter=30)
            except Exception:
                continue
            # kmeans2 can return empty clusters; drop them
            used = np.unique(labels)
            if used.size < k:
                centers = centers[used]
                labels = np.searchsorted(used, labels)
                k = used.size
        # hard-assignment spherical GMM log-likelihood
        ll = 0.0
        for j in range(k):
            m = labels == j
            nj = int(m.sum())
            if nj == 0:
                continue
            sig2 = float(np.mean((X[m] - centers[j]) ** 2)) + 1e-12
            ll += -0.5 * nj * d * (np.log(2 * np.pi * sig2) + 1.0)
            ll += nj * np.log(nj / n)
        n_par = k * (d + 1) + (k - 1)
        bic = -2.0 * ll + n_par * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, labels.copy(), np.atleast_2d(centers).copy())
    return best[1], best[2]


def _dip_merge(X, labels, centers, merge_ratio):
    """Merge cluster pairs with no density dip between their centers.

    Two genuinely distinct modes have a density valley along the segment
    connecting their centers; fragments that k-means carved out of one blob
    (or one banana) do not.  The test must look at ALL draws near the
    segment -- not just the two clusters' members, because in a fragmented
    blob the region between two centers is owned by other clusters and a
    members-only histogram dips artificially.

    Two merge criteria, both evaluated on the segment direction u
    (t = 0 at center i, t = 1 at center j):
      1. overlap: if the clusters' projected spreads cover the separation
         (sigma_i + sigma_j > ~half the separation) they are one blob;
      2. cylinder dip: histogram t for every draw within a perpendicular
         radius of the segment; merge when the valley between the peaks is
         at least merge_ratio times the smaller peak.
    """
    k = centers.shape[0]
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(k):
        for j in range(i + 1, k):
            u = centers[j] - centers[i]
            sep = np.linalg.norm(u)
            if sep == 0:
                parent[find(j)] = find(i)
                continue
            u = u / sep

            t_all = (X - centers[i]) @ u / sep    # 0 at c_i, 1 at c_j
            mi, mj = labels == i, labels == j

            # criterion 1: projected spreads overlap the separation
            sig_i = float(np.std(t_all[mi])) if mi.sum() > 1 else 0.0
            sig_j = float(np.std(t_all[mj])) if mj.sum() > 1 else 0.0
            if sig_i + sig_j > 0.6:
                parent[find(j)] = find(i)
                continue

            # criterion 2: density dip along the segment, all draws within
            # a cylinder whose radius comes from the members' perpendicular
            # scatter about the segment
            perp2 = ((X - centers[i]) ** 2).sum(axis=1) - (t_all * sep) ** 2
            r2 = np.median(perp2[mi | mj])
            in_cyl = perp2 <= 4.0 * max(r2, 1e-12)
            t = t_all[in_cyl]
            hist, edges = np.histogram(t, bins=50, range=(-0.5, 1.5))
            mids = 0.5 * (edges[:-1] + edges[1:])
            peak_i = hist[(mids >= -0.5) & (mids <= 0.3)].max(initial=0)
            peak_j = hist[(mids >= 0.7) & (mids <= 1.5)].max(initial=0)
            between = hist[(mids > 0.3) & (mids < 0.7)]
            valley = between.min(initial=0) if between.size else 0
            if valley >= merge_ratio * min(peak_i, peak_j):
                parent[find(j)] = find(i)

    # relabel by union-find roots
    roots = {}
    new_labels = np.empty_like(labels)
    for old in range(k):
        r = find(old)
        if r not in roots:
            roots[r] = len(roots)
    for old in range(k):
        new_labels[labels == old] = roots[find(old)]
    n_new = len(roots)
    new_centers = np.vstack([X[new_labels == c].mean(axis=0)
                             for c in range(n_new)])
    return new_labels, new_centers, n_new != k


def _count_transitions(labels_2d):
    """Inter-mode label changes along each chain, skipping unassigned draws."""
    n = 0
    for row in labels_2d:
        assigned = row[row >= 0]
        if assigned.size > 1:
            n += int((np.diff(assigned) != 0).sum())
    return n


# ----------------------------
# public entry point
# ----------------------------

def identify_modes(idata,
                   feature_vars: Optional[List[str]] = None,
                   min_weight: float = 0.005,
                   max_modes: int = 8,
                   z_max: float = DEFAULT_Z_MAX,
                   lp_abs_max: float = DEFAULT_LP_ABS_MAX,
                   merge_ratio: float = 0.5,
                   subsample: int = 20000,
                   seed: int = 20260711,
                   attach: bool = True) -> ModeReport:
    """Identify posterior modes in a trace and label every draw.

    Parameters
    ----------
    idata : arviz.InferenceData with a posterior group (and ideally
        sample_stats["lp"]).
    feature_vars : posterior variables to cluster on.  Default: every
        ``*_raw`` variable (the unconstrained sampled space); falls back to
        all float variables if no raw variables exist.
    min_weight : clusters below this fraction of valid draws are not
        reported as modes (their draws are labeled -1).
    max_modes : upper limit for the BIC scan.
    z_max : robust z-score threshold for the invalid-draw filter.
    lp_abs_max : |lp| above this marks a draw invalid (numerically broken).
    merge_ratio : density-dip merge threshold; higher merges more eagerly.
    subsample : cluster on at most this many draws (assignment of the rest
        is by nearest center); keeps k selection fast on huge traces.
    attach : write the labels into idata.posterior["mode"].

    Returns
    -------
    ModeReport
    """
    post = idata.posterior
    notes = []

    if feature_vars is None:
        feature_vars = sorted(v for v in post.data_vars
                              if str(v).endswith("_raw"))
        if not feature_vars:
            feature_vars = sorted(
                v for v in post.data_vars
                if np.issubdtype(post[v].dtype, np.floating))
            notes.append("no *_raw variables in trace; clustered on "
                         "physical variables instead")
    X, dim_names = _feature_matrix(post, feature_vars)
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    n_samples = n_chain * n_draw

    has_lp = hasattr(idata, "sample_stats") and "lp" in idata.sample_stats
    if has_lp:
        lp = np.asarray(idata.sample_stats["lp"].values,
                        dtype=float).reshape(n_samples)
    else:
        lp = np.full(n_samples, np.nan)
        notes.append("sample_stats['lp'] missing; lp-based diagnostics "
                     "and filters skipped")

    # ---- invalid-draw filter -------------------------------------------
    finite = np.isfinite(X).all(axis=1)
    if has_lp:
        lp_ok = np.isfinite(lp) & (np.abs(lp) <= lp_abs_max)
    else:
        lp_ok = np.ones(n_samples, dtype=bool)

    valid = finite & lp_ok
    if valid.any():
        med, mad = _robust_center_scale(X[valid])
        scale = np.where(mad > 0, mad, 1.0)
        z = np.abs((X - med) / scale)
        z_ok = np.nan_to_num(z, nan=np.inf).max(axis=1) <= z_max
        valid &= z_ok

    n_invalid = int((~valid).sum())
    if n_invalid:
        notes.append(f"{n_invalid} draws rejected as invalid "
                     f"(non-finite, |lp| > {lp_abs_max:g}, or raw-space "
                     f"robust z > {z_max:g}); these are runaway/stuck "
                     f"draws, not posterior modes")
    if not valid.any():
        raise ValueError("identify_modes: no valid draws in trace")

    # ---- standardize + cluster ------------------------------------------
    Xv = X[valid]
    med, mad = _robust_center_scale(Xv)
    keep = mad > 0
    if not keep.all():
        dropped = [n for n, k_ in zip(dim_names, keep) if not k_]
        notes.append(f"constant dimensions dropped from clustering: {dropped}")
    Xs = (Xv[:, keep] - med[keep]) / mad[keep]

    rng = np.random.default_rng(seed)
    if Xs.shape[0] > subsample:
        idx_fit = rng.choice(Xs.shape[0], subsample, replace=False)
    else:
        idx_fit = np.arange(Xs.shape[0])

    fit_labels, centers = _kmeans_bic(Xs[idx_fit], max_modes, seed)
    merged = True
    while merged and centers.shape[0] > 1:
        fit_labels, centers, merged = _dip_merge(
            Xs[idx_fit], fit_labels, centers, merge_ratio)

    # assign every valid draw to nearest surviving center
    d2 = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels_valid = np.argmin(d2, axis=1)

    # ---- weights, minor-cluster drop, ordering ---------------------------
    k = centers.shape[0]
    counts = np.bincount(labels_valid, minlength=k)
    weights = counts / counts.sum()
    major = np.where(weights >= min_weight)[0]
    n_unassigned = int(counts[weights < min_weight].sum())
    if n_unassigned:
        notes.append(f"{k - major.size} minor cluster(s) below "
                     f"min_weight={min_weight} ({n_unassigned} draws) "
                     f"left unassigned")
    order = major[np.argsort(weights[major])[::-1]]

    labels_full = np.full(n_samples, -1, dtype=int)
    valid_idx = np.flatnonzero(valid)
    for new, old in enumerate(order):
        labels_full[valid_idx[labels_valid == old]] = new
    labels_2d = labels_full.reshape(n_chain, n_draw)

    # ---- per-mode stats ---------------------------------------------------
    n_modes = order.size
    w_assigned = np.bincount(labels_full[labels_full >= 0], minlength=n_modes)
    w_assigned = w_assigned / w_assigned.sum()
    lp_2d = lp.reshape(n_chain, n_draw)

    lp_maxes = []
    modes = []
    for m in range(n_modes):
        sel = labels_full == m
        lp_m = lp[sel]
        lp_m = lp_m[np.isfinite(lp_m)]
        lp_maxes.append(lp_m.max() if lp_m.size else np.nan)
    best_lp = np.nanmax(lp_maxes) if n_modes else np.nan

    for m in range(n_modes):
        sel = labels_full == m
        lp_m = lp[sel]
        lp_m = lp_m[np.isfinite(lp_m)]
        per_chain = np.array([
            (row == m).sum() / max((row >= 0).sum(), 1)
            for row in labels_2d])
        center_raw = {}
        sel_v = labels_valid == order[m]
        for jn, name in enumerate(dim_names):
            center_raw[name] = float(np.median(Xv[sel_v, jn]))
        modes.append(ModeInfo(
            index=m,
            weight=float(w_assigned[m]),
            n_draws=int(sel.sum()),
            lp_med=float(np.median(lp_m)) if lp_m.size else np.nan,
            lp_max=float(lp_m.max()) if lp_m.size else np.nan,
            delta_lp_max=float(best_lp - lp_m.max()) if lp_m.size else np.nan,
            per_chain_weight=per_chain,
            center=center_raw,
        ))

    # ---- mixing diagnostics / weight provenance --------------------------
    n_transitions = _count_transitions(labels_2d)
    if n_modes <= 1:
        provenance = "unimodal"
        reliable = True
    else:
        chains_visiting_all = all(
            all((row == m).any() for m in range(n_modes))
            for row in labels_2d if (row >= 0).any())
        enough_transitions = n_transitions >= 10 * (n_modes - 1)
        reliable = chains_visiting_all and enough_transitions
        if reliable:
            provenance = (f"occupancy (validated: {n_transitions} inter-mode "
                          f"transitions; every chain visits every mode)")
        else:
            provenance = (
                "occupancy (UNRELIABLE: chains do not mix between modes -- "
                "weights reflect initialization, not posterior mass; use "
                "per-mode evidence weighting or a folded likelihood)")
            notes.append("relative mode weights are NOT trustworthy: "
                         f"{n_transitions} inter-mode transitions; "
                         "see provenance")

    report = ModeReport(
        labels=labels_2d,
        modes=modes,
        n_valid=int(valid.sum()),
        n_invalid=n_invalid,
        n_unassigned=n_unassigned,
        provenance=provenance,
        weights_reliable=reliable,
        n_transitions=n_transitions,
        feature_vars=list(feature_vars),
        notes=notes,
    )
    if attach:
        report.attach(idata)
    logger.info("identify_modes: %d mode(s), weights=%s, %s",
                report.n_modes,
                [f"{w:.3f}" for w in report.weights],
                provenance)
    return report
