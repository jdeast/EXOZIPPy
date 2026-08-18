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
  4b. Merge clusters connected by a populated, lp-flat path (the
     lp-barrier ridge test, ``_lp_ridge_merge``): a flat likelihood ridge
     -- an unconstrained degeneracy direction, e.g. m--cos i in an
     RV-only fit -- separates in density without separating in
     likelihood, and only an lp BARRIER makes two clusters two modes.
     Skipped when the trace has no lp.
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

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from .autocorr import iact

# Re-exported, not defined here: mode_suffix names both the per-mode LaTeX
# macros (emitted in components/parameter.py, referenced in outputs/latex.py)
# and the per-mode plot files (run.py), so it has to be spelled in a module
# components/ can import -- see the "Macro name pieces" note in texutils.py.
# Kept importable from here because `mode_suffix` reads as a modes concept
# at every call site.
from .texutils import mode_suffix  # noqa: F401

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

# lp exemption from the raw-z filter: a draw far outside the bulk in raw
# space but with lp this close to (or better than) the z-passing bulk's
# median is a CANDIDATE MODE, not a runaway, and is routed into clustering
# instead of being invalidated.  Every genuine runaway population observed
# to date sat >~1000 nats below the bulk (saturated-plateau states;
# lp-insane states are caught by DEFAULT_LP_ABS_MAX separately), while a
# real minority mode can sit ABOVE the bulk -- DC2018 event 128's true
# s~0.98 branch was found by 2/54 chains at +500 nats and was discarded as
# 'raw-z invalid', reporting the wrong branch as a clean unimodal fit.  A
# genuinely-real mode more than this many nats BELOW the dominant one
# carries e^-50 of its mass and loses nothing by staying flagged.
DEFAULT_LP_EXEMPT_MARGIN = 50.0

# With the runaway-lp cancellation bug fixed, ONLY a model or sampler bug
# should produce invalid draws.  Above this fraction of the trace, run.py
# refuses to emit final tables (evidence -- trace + mode report -- is written
# first); below it, invalid draws still complain loudly but do not block.
DEFAULT_MAX_INVALID_FRAC = 0.01

_INVALID_REASONS = ("nonfinite-raw", "nonfinite-lp", "lp-ceiling", "raw-z")

# Below this effective sample size the occupancy weights carry an error bar
# comparable to the weights themselves and the report says so in plain
# language.  ADVISORY ONLY: the weights are still reported, with their
# uncertainty, and nothing is substituted or suppressed -- the project warns
# rather than blocks.  N_eff = 30 puts sigma_w at ~0.09 for w = 0.5.
DEFAULT_MIN_WEIGHT_ESS = 30.0


# Outcome vocabulary for one mode-identification attempt, following the
# status-dict pattern of outputs/ledger.py's hot-chain search (HOT_*) and
# outputs/evidence.py's per-mode bridge estimates (EV_*).
#
# The distinction that matters is between the last two.  Both end with
# ``mode_report = None`` at the call site, and until 2026-08-12 that was the
# ONLY thing the caller could see -- so the numerical-validity gate below,
# which returns immediately on a None report, was bypassed in exactly the
# worst case: EVERY draw rejected.  A trace with 1.1% invalid draws refused
# to emit tables while a trace with 100% invalid draws emitted a full set
# that read as clean (review item 3.17).  MODE_NO_VALID_DRAWS is the signal
# that separates "the draws are unusable" from "we could not tell you
# anything, for some other reason"; it can only be set from
# NoValidDrawsError, which identify_modes raises from one place.
MODE_OK = "ok"
MODE_NO_VALID_DRAWS = "no-valid-draws"
MODE_FAILED = "failed"


class NoValidDrawsError(ValueError):
    """Every draw in the trace failed identify_modes' validity filter.

    A ValueError subclass, so any pre-existing ``except ValueError`` around
    identify_modes keeps behaving as before.

    It carries the counts because it is the only channel through which the
    all-invalid case can reach ``check_invalid_frac``: there is nothing to
    cluster, so no ModeReport exists to read ``n_invalid``/``invalid_frac``
    off, and a bare exception would leave the gate with nothing to gate on.
    """

    def __init__(self, n_invalid, n_draws, reason_counts=None, per_chain=None):
        self.n_invalid = int(n_invalid)
        self.n_draws = int(n_draws)
        self.invalid_frac = (
            self.n_invalid / self.n_draws if self.n_draws else 0.0
        )
        self.reason_counts = dict(reason_counts or {})
        self.per_chain_invalid = (
            list(per_chain) if per_chain is not None else []
        )
        # The leading clause is load-bearing: callers and tests match on
        # "no valid draws".
        super().__init__(
            f"identify_modes: no valid draws in trace -- all "
            f"{self.n_invalid} draws ({self.invalid_frac:.2%}) failed the "
            f"numerical-validity filter (reasons={self.reason_counts}, "
            f"per-chain invalid counts={self.per_chain_invalid})"
        )


def _invalid_reason_hint(reason_counts):
    """One actionable sentence about the dominant rejection reason."""
    if not reason_counts:
        return ""
    dominant = max(reason_counts, key=lambda r: reason_counts[r])
    if dominant == "nonfinite-raw":
        return (
            " The sampled parameter values themselves are non-finite, so "
            "the sampler never produced a usable point: check the model's "
            "logp at the start point and the sampler's step-size adaptation."
        )
    if dominant in ("nonfinite-lp", "lp-ceiling"):
        return (
            " The stored log-posterior is non-finite (or beyond "
            f"|lp| = {DEFAULT_LP_ABS_MAX:g}) for every draw. The parameter "
            "values may still look perfectly reasonable in the tables, "
            "which is exactly why this cannot be allowed to pass quietly: "
            "check the model's logp for NaN/inf-producing terms."
        )
    return ""


def check_invalid_frac(
    mode_report,
    max_invalid_frac=DEFAULT_MAX_INVALID_FRAC,
    force=False,
    trace_path=None,
    modes_path=None,
    status=None,
):
    """Raise if the trace's invalid-draw fraction exceeds the threshold.

    A trace this numerically broken must not silently emit final tables.
    Call this only after the trace and mode report have already been
    written to disk, so the raise preserves that evidence for forensics.
    ``force=True`` (config ``modes: {force: true}``) or a higher
    ``max_invalid_frac`` re-enables processing of a known-bad trace.

    ``mode_report`` may be None, which is ambiguous on its own and so is
    disambiguated by ``status`` (the dict build_mode_reports fills in):

    * ``state == MODE_NO_VALID_DRAWS`` -- identify_modes rejected EVERY
      draw and could not build a report.  That is a 100% invalid fraction,
      the most extreme form of exactly what this gate exists to catch, and
      it is gated on the same comparison and honours the same overrides as
      a partially-invalid report.  Passing it silently is review item 3.17.
    * anything else (including no status at all) -- mode identification
      produced no report for a reason that says nothing about the draws'
      numerical validity (an unclustered trace, a crash in the mode pass).
      There is nothing to gate on and this returns, unchanged.
    """
    if mode_report is not None:
        n_invalid = int(mode_report.n_invalid)
        frac = mode_report.invalid_frac
        reason_counts = dict(mode_report.invalid_reason_counts or {})
        per_chain = None
        all_invalid = False
    elif status and status.get("state") == MODE_NO_VALID_DRAWS:
        n_invalid = int(status.get("n_invalid", 0))
        frac = float(status.get("invalid_frac", 0.0))
        reason_counts = dict(status.get("reasons") or {})
        per_chain = status.get("per_chain_invalid")
        all_invalid = True
    else:
        return

    if not n_invalid or frac <= max_invalid_frac:
        return
    if force:
        logger.warning(
            "modes: %d draws (%.2f%%) are numerically invalid, above "
            "max_invalid_frac=%.2f%%, but `modes: {force: true}` was set -- "
            "emitting tables from a trace known to be broken. reasons=%s",
            n_invalid,
            100 * frac,
            100 * max_invalid_frac,
            reason_counts,
        )
        return

    where = ""
    if trace_path or modes_path:
        where = f" The trace ({trace_path}) and mode report ({modes_path}) have already been written."
    override = (
        " Override with config `modes: {force: true}` (or raise "
        "`modes.max_invalid_frac`) to re-process forensically."
    )
    if all_invalid:
        raise RuntimeError(
            f"identify_modes: ALL {n_invalid} draws ({frac:.2%}) were "
            "rejected as numerically invalid, so no posterior mode could be "
            "identified and NO summary of this trace is meaningful -- the "
            "tables would describe draws that were all rejected. "
            f"reasons={reason_counts}, per-chain invalid counts={per_chain}."
            + _invalid_reason_hint(reason_counts)
            + where
            + " Re-run `exozippy-modes <config>` to inspect the trace "
            "without raising." + override
        )
    raise RuntimeError(
        f"identify_modes: {n_invalid} draws "
        f"({frac:.2%}) rejected as numerically invalid, "
        f"exceeding max_invalid_frac={max_invalid_frac:.2%}. This indicates "
        "a model or sampler bug -- investigate before trusting any output."
        + where
        + override
    )


def mode_status_to_text(status):
    """Render a MODE_NO_VALID_DRAWS outcome for ``<prefix>_modes.txt``.

    Mirrors ``ledger.hot_status_to_text``: returns "" for anything it has
    nothing to say about, so callers can pass a status dict unconditionally.

    Only the all-invalid state renders.  MODE_OK writes a real report (that
    is ModeReport.to_text's job) and MODE_FAILED is deliberately left
    byte-identical to its pre-3.17 behaviour -- a crash in the mode pass is
    not evidence about the draws, and the file it would write here would be
    the only place claiming otherwise.
    """
    if not status or status.get("state") != MODE_NO_VALID_DRAWS:
        return ""
    n_invalid = int(status.get("n_invalid", 0))
    frac = float(status.get("invalid_frac", 0.0))
    lines = [
        "Posterior mode report",
        "=====================",
        "",
        "*** NO VALID DRAWS ***",
        "",
        f"All {n_invalid} draws ({frac:.2%}) in this trace were rejected as",
        "numerically invalid, so no mode could be identified and no mode",
        "report exists. Any table or plot generated from this trace",
        "describes draws that the validity filter rejected in full.",
        "",
        f"reasons: {status.get('reasons') or {}}",
        f"per-chain invalid counts: {status.get('per_chain_invalid')}",
    ]
    hint = _invalid_reason_hint(status.get("reasons") or {})
    if hint:
        lines += ["", hint.strip()]
    lines += [
        "",
        "This is a model or sampler bug, not a reporting problem. A live",
        "fit refuses to write final tables in this state; this file was",
        "written by a forensic re-processing run (exozippy-modes) or by a",
        "run with `modes: {force: true}` set.",
        "",
    ]
    return "\n".join(lines)


def _fmt_pm(value, err, fmt="{:.4f}"):
    """'0.7000 +/- 0.1400', or just the value when the error is unavailable."""
    text = fmt.format(value)
    if err is not None and np.isfinite(err):
        text += " +/- " + fmt.format(err)
    return text


@dataclass
class ModeInfo:
    index: int  # 0-based, ordered by weight (descending)
    weight: float  # fraction of valid assigned draws
    n_draws: int
    lp_med: float
    lp_max: float
    delta_lp_max: float  # lp_max(best mode) - lp_max(this mode)
    per_chain_weight: (
        np.ndarray
    )  # occupancy fraction of each chain's valid draws
    center: dict = field(
        default_factory=dict
    )  # feature var -> center (raw units)
    center_scale: dict = field(
        default_factory=dict
    )  # feature var -> robust per-dim MARGINAL scale of the mode's own
    # draws (raw units). The seed ledger's matching normalizes by this:
    # a seed's curvature widths are CONDITIONAL scales, and on correlated
    # posteriors the marginal median sits tens of conditional sigmas from
    # the basin peak, so normalizing by the seed widths alone falsely
    # rejects seeds whose basins plainly survived.
    # 1-sigma on `weight`, whatever its provenance: identify_modes fills it
    # from the mode indicator's effective sample size (occupancy weighting),
    # and outputs.evidence.apply_evidence_weighting overwrites it with the
    # lnZ-propagated value when evidence weights replace the occupancy ones.
    weight_err: float = float("nan")
    # The occupancy weight and its uncertainty are ALWAYS kept, even after
    # evidence weighting overwrites `weight`: reporting the two side by side
    # is what makes evidence weighting a usable cross-check on occupancy
    # (agreement means the weights can be trusted; disagreement localizes the
    # problem to ladder tuning or to the bridge proposal).
    occ_weight: float = float("nan")
    occ_weight_err: float = float("nan")
    weight_ess: float = float("nan")  # effective draws behind occ_weight
    weight_iact: float = float("nan")  # IACT of this mode's indicator series
    # Optional evidence-weighting fields (populated by
    # outputs.evidence.apply_evidence_weighting when modes: {weights: evidence}
    # is requested and every mode's bridge estimate is trustworthy).
    lnZ: float = float("nan")  # local log-evidence (bridge sampling)
    lnZ_err: float = float("nan")  # 1-sigma on lnZ


@dataclass
class ModeReport:
    labels: np.ndarray  # (chain, draw) int; -1 = invalid/unassigned
    modes: List[ModeInfo]
    n_valid: int
    n_invalid: int
    n_unassigned: int  # valid draws in dropped minor clusters
    provenance: str
    weights_reliable: bool
    n_transitions: int  # inter-mode label changes along chains
    feature_vars: List[str]
    notes: List[str] = field(default_factory=list)
    invalid_reason_counts: dict = field(default_factory=dict)
    invalid_per_chain: Optional[np.ndarray] = None
    # Mode-change bookkeeping (see transition_stats).  These are counts of
    # MODE changes in the stored draws, not the sampler's temperature-swap
    # statistics; ladder_round_trips below carries the latter when the
    # sampler recorded it, purely as context.
    transitions_per_chain: Optional[np.ndarray] = None
    n_round_trips: int = 0
    # Storage thinning of the trace these labels came from.  1 = stored every
    # sampler step.  thin_known=False means nothing recorded it and 1 was
    # assumed, which the report says out loud rather than quietly implying.
    thin_factor: int = 1
    thin_known: bool = True
    ladder_round_trips: Optional[int] = None
    ladder_swap_rounds: Optional[int] = None

    @property
    def n_modes(self):
        return len(self.modes)

    @property
    def weights(self):
        return [m.weight for m in self.modes]

    @property
    def weight_errs(self):
        return [m.weight_err for m in self.modes]

    @property
    def invalid_frac(self):
        n_total = self.labels.size
        return self.n_invalid / n_total if n_total else 0.0

    @property
    def n_chains_no_switch(self):
        """Chains holding assigned draws that never changed mode."""
        if self.transitions_per_chain is None:
            return 0
        has_draws = np.array(
            [bool((row >= 0).any()) for row in np.atleast_2d(self.labels)]
        )
        return int(
            ((np.asarray(self.transitions_per_chain) == 0) & has_draws).sum()
        )

    @property
    def n_chains_with_draws(self):
        return int(
            sum(bool((row >= 0).any()) for row in np.atleast_2d(self.labels))
        )

    def attach(self, idata):
        """Store labels as posterior variable ``mode`` on the InferenceData.

        Idempotent, and deliberately re-callable: ``apply_evidence_weighting``
        rewrites the weights and provenance on this report in place, so it
        calls back here to keep ``idata.posterior['mode'].attrs`` from holding
        the superseded occupancy values.
        """
        import xarray as xr

        post = idata.posterior
        da = xr.DataArray(
            self.labels.astype(np.int16),
            dims=("chain", "draw"),
            coords={"chain": post.chain, "draw": post.draw},
        )
        da.attrs["n_modes"] = self.n_modes
        da.attrs["weights"] = [float(w) for w in self.weights]
        da.attrs["weight_errs"] = [float(e) for e in self.weight_errs]
        da.attrs["occupancy_weights"] = [
            float(m.occ_weight) for m in self.modes
        ]
        da.attrs["provenance"] = self.provenance
        da.attrs["n_invalid"] = int(self.n_invalid)
        da.attrs["n_transitions"] = int(self.n_transitions)
        da.attrs["n_round_trips"] = int(self.n_round_trips)
        da.attrs["n_chains_no_switch"] = int(self.n_chains_no_switch)
        post["mode"] = da
        return idata

    def to_text(self):
        lines = []
        lines.append("Posterior mode report")
        lines.append("=====================")
        n_total = self.labels.size
        lines.append(
            f"draws: {n_total} total, {self.n_valid} valid, "
            f"{self.n_invalid} invalid (rejected), "
            f"{self.n_unassigned} in minor/unassigned clusters"
        )
        if self.n_invalid:
            lines.append("")
            lines.append(
                f"*** WARNING: {self.n_invalid} draws ({self.invalid_frac:.2%}) "
                "rejected as numerically invalid -- this indicates a model "
                "or sampler bug; investigate before trusting this report. "
                f"reasons={self.invalid_reason_counts} ***"
            )
            lines.append("")
        lines.append(f"modes found: {self.n_modes}")
        lines.append(f"weight provenance: {self.provenance}")
        lines.extend(self._mixing_lines())
        for m in self.modes:
            lines.append("")
            lines.append(f"mode {m.index + 1}:")
            lines.append(f"  weight   = {_fmt_pm(m.weight, m.weight_err)}")
            if np.isfinite(m.occ_weight) and np.isfinite(m.weight_ess):
                extra = ""
                if np.isfinite(m.lnZ):
                    extra = (
                        f"; evidence weight "
                        f"{_fmt_pm(m.weight, m.weight_err)} "
                        f"(lnZ = {_fmt_pm(m.lnZ, m.lnZ_err)})"
                    )
                lines.append(
                    f"    occupancy weight "
                    f"{_fmt_pm(m.occ_weight, m.occ_weight_err)} "
                    f"from N_eff = {m.weight_ess:.1f} independent draws "
                    f"(IACT {m.weight_iact:.1f}){extra}"
                )
            lines.append(f"  n_draws  = {m.n_draws}")
            lines.append(
                f"  lp med/max = {m.lp_med:.2f} / {m.lp_max:.2f}"
                f"  (delta lp_max vs best = {m.delta_lp_max:.2f})"
            )
            occ = np.asarray(m.per_chain_weight)
            lines.append(
                f"  chains containing this mode: "
                f"{int((occ > 0).sum())}/{occ.size}"
                f"  (per-chain occupancy min/med/max = "
                f"{occ.min():.2f}/{np.median(occ):.2f}/{occ.max():.2f})"
            )
        if self.notes:
            lines.append("")
            lines.append("notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines) + "\n"

    def _mixing_lines(self):
        """The mode-mixing block: how much the relative weights can be trusted.

        Deliberately spells out what it is counting.  'Swap' is ambiguous in a
        parallel-tempering run, so the mode-change counts and the ladder's own
        temperature round trips are labelled separately and never added.
        """
        lines = [
            f"inter-mode transitions (mode changes in the stored draws, "
            f"all chains): {self.n_transitions}"
            + (
                f"; round trips (k -> j -> k): {self.n_round_trips}"
                if self.n_modes > 1
                else ""
            )
        ]
        if self.transitions_per_chain is not None:
            per = np.asarray(self.transitions_per_chain).tolist()
            lines.append(f"  per chain: {per}")
            lines.append(
                f"  chains that never changed mode: "
                f"{self.n_chains_no_switch}/{self.n_chains_with_draws}"
            )
        if self.thin_factor > 1:
            lines.append(
                f"  NOTE: draws are stored thinned by {self.thin_factor}, so "
                f"the transition count above is a LOWER BOUND -- mode changes "
                f"inside a thinning block are invisible.  N_eff below is "
                f"measured on the stored series and is unaffected (thinning "
                f"divides both the draw count and the IACT), or conservative "
                f"once the thinning exceeds the correlation time."
            )
        elif not self.thin_known:
            lines.append(
                "  NOTE: the trace does not record its storage thinning; "
                "these counts assume every sampler step was stored.  If it "
                "was thinned, the transition count is a lower bound."
            )
        if self.ladder_round_trips is not None:
            lines.append(
                f"  for context, the sampler's own ladder statistic "
                f"(temperature round trips of a replica, T=1 -> T=max -> "
                f"T=1 -- NOT mode changes): {self.ladder_round_trips}"
                + (
                    f" over {self.ladder_swap_rounds} swap rounds"
                    if self.ladder_swap_rounds is not None
                    else ""
                )
            )
        return lines


# ----------------------------
# internals
# ----------------------------


def _int_attr(idata, name):
    """Integer posterior-group attribute, or None if the trace lacks it."""
    try:
        value = idata.posterior.attrs.get(name)
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trace_thinning(idata):
    """Storage thinning of the trace: ``(thin_factor, known)``.

    ``run.py`` stamps ``posterior.attrs['nthin']`` on every freshly sampled
    trace; older traces carry nothing, and the difference matters -- consecutive
    stored draws that are really ``t`` sampler steps apart make mode changes
    look more independent than they are, so an unknown thinning is reported as
    unknown rather than silently assumed to be 1.
    """
    value = _int_attr(idata, "nthin")
    if value is None or value < 1:
        return 1, False
    return value, True


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
                    X, k, minit="++", seed=rng.integers(2**31), iter=30
                )
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


# --- shared machinery of the two pairwise cluster-merge TESTS ---------------
#
# _dip_merge and _lp_ridge_merge ask different questions -- does the OCCUPANCY
# density dip between two centers, and is there an lp BARRIER between them --
# and those two tests stay separate functions.  Everything around the tests is
# the same in both: the same union-find, the same projection onto the
# center-to-center segment with the same cylinder around it, and the same
# relabel-and-recompute-centers epilogue.


def _assign_to_nearest_center(X, centers):
    """Index of the nearest center for every row of X, one center at a time.

    The obvious spelling,
    ``np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), 1)``,
    materializes an (N, k, d) intermediate to produce an (N,) answer.  On a
    real run -- 54 chains x 50k draws, k = 8 centers, d = 20 features -- that
    is 3.5 GB of temporary.  Looping over the (at most a handful of) centers
    holds one (N, d) temporary instead, no bigger than X itself: measured at
    a tenth of that scale, peak 346 MiB -> 50 MiB with the same wall time.

    Bit-identical to the broadcast, and both halves of that matter: each
    center's distance column is the same reduction over the same contiguous
    axis, and the strict ``<`` keeps the FIRST minimum on a tie, exactly as
    np.argmin does.
    """
    labels = np.zeros(X.shape[0], dtype=int)
    best_d2 = None
    for c in range(centers.shape[0]):
        d2_c = ((X - centers[c]) ** 2).sum(axis=1)
        if best_d2 is None:
            best_d2 = d2_c
            continue
        closer = d2_c < best_d2
        labels[closer] = c
        best_d2 = np.where(closer, d2_c, best_d2)
    return labels


def _make_union_find(k):
    """Union-find over k cluster indices; returns (find, union).

    Path halving, so repeated finds stay near-flat.  ``union(a, b)`` puts b's
    root under a's, matching what both merge loops wrote by hand.
    """
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(b)] = find(a)

    return find, union


def _segment_cylinder(X, labels, centers, i, j):
    """Project every draw onto the c_i -> c_j segment and select a cylinder.

    Returns ``(sep, t, in_cyl)``:

    * ``sep`` is the center separation.  When it is 0 the segment does not
      exist and ``t``/``in_cyl`` are None -- both callers merge such a pair
      unconditionally.
    * ``t`` is the projected coordinate of EVERY draw, 0 at center i and 1 at
      center j.  Every draw, not just the two clusters' members: in a
      fragmented blob the region between two centers is owned by OTHER
      clusters, and a members-only histogram dips artificially.
    * ``in_cyl`` selects the draws within twice the two clusters' median
      perpendicular offset from the segment.

    Both merge tests need exactly this, which is why they share it; what they
    do with ``t`` and ``in_cyl`` afterwards is the part that differs.
    """
    u = centers[j] - centers[i]
    sep = float(np.linalg.norm(u))
    if sep == 0:
        return 0.0, None, None
    u = u / sep
    t = (X - centers[i]) @ u / sep
    perp2 = ((X - centers[i]) ** 2).sum(axis=1) - (t * sep) ** 2
    r2 = np.median(perp2[(labels == i) | (labels == j)])
    return sep, t, perp2 <= 4.0 * max(r2, 1e-12)


def _union_relabel(X, labels, find, k):
    """Collapse merged clusters to contiguous labels and recompute centers.

    Returns ``(new_labels, new_centers, merged_any)``.  Roots are numbered in
    ascending order of their lowest old label, so an unmerged run keeps its
    labels exactly.
    """
    roots = {}
    new_labels = np.empty_like(labels)
    for old in range(k):
        r = find(old)
        if r not in roots:
            roots[r] = len(roots)
    for old in range(k):
        new_labels[labels == old] = roots[find(old)]
    n_new = len(roots)
    new_centers = np.vstack(
        [X[new_labels == c].mean(axis=0) for c in range(n_new)]
    )
    return new_labels, new_centers, n_new != k


def _dip_merge(X, labels, centers, merge_ratio):
    """Merge cluster pairs with no density dip between their centers.

    Two genuinely distinct modes have a density valley along the segment
    connecting their centers; fragments that k-means carved out of one blob
    (or one banana) do not.

    Two merge criteria, both evaluated on _segment_cylinder's projection
    (t = 0 at center i, t = 1 at center j):
      1. overlap: if the clusters' projected spreads cover the separation
         (sigma_i + sigma_j > ~half the separation) they are one blob;
      2. cylinder dip: histogram t for every draw in the cylinder; merge
         when the valley between the peaks is at least merge_ratio times
         the smaller peak.
    """
    k = centers.shape[0]
    find, union = _make_union_find(k)

    for i in range(k):
        for j in range(i + 1, k):
            sep, t_all, in_cyl = _segment_cylinder(X, labels, centers, i, j)
            if sep == 0:
                union(i, j)
                continue

            mi, mj = labels == i, labels == j

            # criterion 1: projected spreads overlap the separation
            sig_i = float(np.std(t_all[mi])) if mi.sum() > 1 else 0.0
            sig_j = float(np.std(t_all[mj])) if mj.sum() > 1 else 0.0
            if sig_i + sig_j > 0.6:
                union(i, j)
                continue

            # criterion 2: density dip along the segment
            t = t_all[in_cyl]
            hist, edges = np.histogram(t, bins=50, range=(-0.5, 1.5))
            mids = 0.5 * (edges[:-1] + edges[1:])
            peak_i = hist[(mids >= -0.5) & (mids <= 0.3)].max(initial=0)
            peak_j = hist[(mids >= 0.7) & (mids <= 1.5)].max(initial=0)
            between = hist[(mids > 0.3) & (mids < 0.7)]
            valley = between.min(initial=0) if between.size else 0
            if valley >= merge_ratio * min(peak_i, peak_j):
                union(i, j)

    return _union_relabel(X, labels, find, k)


def _lp_ridge_merge(X, lp, labels, centers, sigma_lp, k_sigma=3.0, n_bins=10):
    """Merge cluster pairs connected by a populated, lp-flat path.

    Complements ``_dip_merge``, which asks whether the OCCUPANCY density
    dips between two centers.  Occupancy is the wrong witness for a flat
    likelihood ridge whose far end is stretched out by the raw-space
    transform: on an RV-only fit (kelt4), the m--cos i degeneracy tail at
    cos i -> 1 sits ~1500 raw units from the bulk -- a huge density gap,
    so the dip test keeps it as a separate "mode" -- while the draws'
    max-lp is flat to ~6 nats along the whole path, i.e. there is NO
    likelihood barrier and it is one connected basin.

    Two genuinely distinct modes ARE separated by an lp barrier: the
    region between them is either empty of draws (chains rarely cross)
    or populated only by stragglers whose lp sits far below both peaks.
    So, for each cluster pair, take ``_segment_cylinder``'s projection --
    the same one ``_dip_merge`` tests -- and bin the interior:
    merge iff EVERY interior bin is populated (an empty bin is absence
    of evidence and never merges -- this is also what keeps a curved
    banana whose true path bows away from the straight segment safely
    unmerged) AND no bin's max-lp dips more than ``k_sigma * sigma_lp``
    below the lower of the two clusters' own lp peaks.

    ``sigma_lp`` is the draw-to-draw lp scatter (the caller measures it
    within the dominant cluster, floored at sqrt(n_dims/2), the chi2
    width that scatter approaches for a well-sampled Gaussian): max-lp
    per bin fluctuates by that much even along a perfectly flat ridge,
    so the threshold must scale with it.

    Returns ``(labels, centers, merged_any, merge_notes)``.
    """
    k = centers.shape[0]
    find, union = _make_union_find(k)

    finite = np.isfinite(lp)
    merge_notes = []
    for i in range(k):
        for j in range(i + 1, k):
            mi, mj = labels == i, labels == j
            if not ((mi & finite).any() and (mj & finite).any()):
                continue
            sep, t_all, in_cyl = _segment_cylinder(X, labels, centers, i, j)
            if sep == 0:
                union(i, j)
                continue
            # the same cylinder _dip_merge uses, restricted to draws whose
            # lp can actually be compared
            in_cyl = in_cyl & finite

            peak = min(lp[mi & finite].max(), lp[mj & finite].max())
            allowed_dip = k_sigma * sigma_lp

            edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_idx = np.digitize(t_all[in_cyl], edges) - 1
            lp_cyl = lp[in_cyl]
            connected = True
            worst = 0.0
            for b in range(n_bins):
                sel = bin_idx == b
                if not sel.any():
                    connected = False
                    break
                worst = max(worst, peak - lp_cyl[sel].max())
            if connected and worst <= allowed_dip:
                union(i, j)
                merge_notes.append(
                    f"clusters merged as one basin: the path between "
                    f"their centers is populated in every one of "
                    f"{n_bins} bins with max-lp within {worst:.1f} nats "
                    f"of the lower peak (threshold {allowed_dip:.1f} = "
                    f"{k_sigma:g} x sigma_lp {sigma_lp:.1f}) -- a flat "
                    f"likelihood ridge (e.g. an unconstrained "
                    f"degeneracy direction), not a separate mode"
                )

    new_labels, new_centers, merged_any = _union_relabel(X, labels, find, k)
    return new_labels, new_centers, merged_any, merge_notes


def transition_stats(labels_2d):
    """Mode-change bookkeeping for a (chain, draw) label array.

    Returns ``(n_transitions, per_chain, n_round_trips)``:

    * ``n_transitions`` -- total inter-mode label changes along the chains,
      skipping unassigned (-1) draws;
    * ``per_chain`` -- the same count for each chain, because a chain that
      never leaves one mode contributes **zero** information about the
      relative weights no matter how many draws it holds;
    * ``n_round_trips`` -- returns of the form k -> j -> k, counted on the
      run-length-compressed visit sequence.  This is the stricter quantity:
      a single one-way crossing tells you the two modes communicate, but only
      a round trip says the chain sampled the ratio of their masses.

    These are counts of MODE changes in the stored draws.  For a parallel
    tempering run they are not the sampler's temperature-swap or ladder
    round-trip statistics -- a T=1 replica can change mode precisely because
    it swapped with a hotter rung, and it is the resulting mode change in the
    stored T=1 draws that occupancy weighting rests on.
    """
    labels_2d = np.atleast_2d(np.asarray(labels_2d))
    per_chain = np.zeros(labels_2d.shape[0], dtype=int)
    round_trips = 0
    for c, row in enumerate(labels_2d):
        assigned = row[row >= 0]
        if assigned.size < 2:
            continue
        change_at = np.flatnonzero(np.diff(assigned) != 0)
        per_chain[c] = int(change_at.size)
        # run-length-compressed sequence of visited modes
        visits = assigned[np.concatenate(([0], change_at + 1))]
        if visits.size >= 3:
            round_trips += int((visits[:-2] == visits[2:]).sum())
    return int(per_chain.sum()), per_chain, round_trips


def mode_indicator_chains(labels_2d, mode):
    """Per-chain 0/1 indicator series for ``mode``, over assigned draws only.

    Unassigned (-1) draws are dropped rather than zeroed: they are rejected
    runaways, not evidence that the chain was somewhere else.
    """
    out = []
    for row in np.atleast_2d(np.asarray(labels_2d)):
        assigned = row[row >= 0]
        if assigned.size:
            out.append((assigned == mode).astype(float))
    return out


def weight_ess(labels_2d, mode):
    """Effective sample size behind ``mode``'s occupancy weight.

    Occupancy weighting is unbiased when the sampler mixes between modes, but
    its PRECISION is set by the number of independent mode transitions, not
    by the number of draws: a 50000-draw run that changed mode five times
    knows the weight to roughly 40%.  Treating the mode indicator as a time
    series and dividing its length by its IACT is exactly that statement.

    Returns ``(n_eff, tau)``.  ``outputs.autocorr.iact`` supplies both the
    within-chain autocorrelation and the between-chain scatter of the chain
    means, and the second term is what makes chains stuck in different modes
    come out at ``n_eff ~ n_chains``, rather than ``n_eff = n_draws`` with a
    spuriously tight weight.
    """
    segs = mode_indicator_chains(labels_2d, mode)
    n_total = sum(s.size for s in segs)
    if n_total == 0:
        return 0.0, 1.0
    tau = iact(segs)
    return n_total / tau, tau


def markov_indicator_iact(labels_2d, mode):
    """IACT of the mode indicator under a two-state Markov approximation.

    For a two-state chain with transition probabilities p01 (enter ``mode``)
    and p10 (leave it), the indicator's IACT is exactly ``2/(p01+p10) - 1``.
    Estimating p01/p10 from the observed one-step transition counts gives an
    independent cross-check on the time-series estimate in ``weight_ess``:
    the two should agree to within their sampling noise on a chain that is
    close to Markov in the mode label.  Returns NaN when either state is
    never entered (no transitions to estimate from).
    """
    n01 = n0 = n10 = n1 = 0
    for row in np.atleast_2d(np.asarray(labels_2d)):
        assigned = row[row >= 0]
        if assigned.size < 2:
            continue
        ind = assigned == mode
        a, b = ind[:-1], ind[1:]
        n0 += int((~a).sum())
        n1 += int(a.sum())
        n01 += int((~a & b).sum())
        n10 += int((a & ~b).sum())
    if n0 == 0 or n1 == 0:
        return float("nan")
    p01 = n01 / n0
    p10 = n10 / n1
    if p01 + p10 <= 0:
        return float("inf")
    return 2.0 / (p01 + p10) - 1.0


# ----------------------------
# public entry point
# ----------------------------


def identify_modes(
    idata,
    feature_vars: Optional[List[str]] = None,
    min_weight: float = 0.005,
    max_modes: int = 8,
    z_max: float = DEFAULT_Z_MAX,
    lp_abs_max: float = DEFAULT_LP_ABS_MAX,
    lp_exempt_margin: float = DEFAULT_LP_EXEMPT_MARGIN,
    merge_ratio: float = 0.5,
    subsample: int = 20000,
    seed: int = 20260711,
    attach: bool = True,
) -> ModeReport:
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
    lp_exempt_margin : a draw beyond z_max whose lp is within this many
        nats of the z-passing bulk's median (or better) is exempted from
        raw-z invalidation and clustered as a candidate mode.
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
        feature_vars = sorted(
            v for v in post.data_vars if str(v).endswith("_raw")
        )
        if not feature_vars:
            feature_vars = sorted(
                v
                for v in post.data_vars
                if np.issubdtype(post[v].dtype, np.floating)
            )
            notes.append(
                "no *_raw variables in trace; clustered on "
                "physical variables instead"
            )
    X, dim_names = _feature_matrix(post, feature_vars)
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    n_samples = n_chain * n_draw

    has_lp = hasattr(idata, "sample_stats") and "lp" in idata.sample_stats
    if has_lp:
        lp = np.asarray(idata.sample_stats["lp"].values, dtype=float).reshape(
            n_samples
        )
    else:
        lp = np.full(n_samples, np.nan)
        notes.append(
            "sample_stats['lp'] missing; lp-based diagnostics "
            "and filters skipped"
        )

    # ---- invalid-draw filter -------------------------------------------
    finite = np.isfinite(X).all(axis=1)
    if has_lp:
        lp_ok = np.isfinite(lp) & (np.abs(lp) <= lp_abs_max)
    else:
        lp_ok = np.ones(n_samples, dtype=bool)

    valid = finite & lp_ok

    # Reason a draw was rejected, in priority order (a draw failing an
    # earlier check is attributed to that check even if it would also fail
    # a later one).
    reasons = np.full(n_samples, "", dtype=object)
    reasons[~finite] = "nonfinite-raw"
    if has_lp:
        reasons[(reasons == "") & ~np.isfinite(lp)] = "nonfinite-lp"
        reasons[(reasons == "") & (np.abs(lp) > lp_abs_max)] = "lp-ceiling"

    if valid.any():
        med, mad = _robust_center_scale(X[valid])
        scale = np.where(mad > 0, mad, 1.0)
        z = np.abs((X - med) / scale)
        z_ok = np.nan_to_num(z, nan=np.inf).max(axis=1) <= z_max
        # lp exemption: a z-failing draw whose lp is within
        # lp_exempt_margin of the z-PASSING bulk's median (or better) is a
        # candidate mode, not a runaway -- route it into clustering.  Real
        # runaways (saturated-plateau states) sit >~1000 nats below the
        # bulk; a true minority mode can sit ABOVE it (see
        # DEFAULT_LP_EXEMPT_MARGIN).  Skipped when lp is unavailable.
        if has_lp and (valid & z_ok).any():
            lp_bulk_med = np.median(lp[valid & z_ok])
            # No NaN guard on the comparison: `valid` already requires
            # np.isfinite(lp), and a NaN lp compares False here anyway, so a
            # nonfinite-lp draw cannot be exempted either way.
            exempt = valid & ~z_ok & (lp >= lp_bulk_med - lp_exempt_margin)
            n_exempt = int(exempt.sum())
            if n_exempt:
                notes.append(
                    f"{n_exempt} draws beyond raw-z {z_max:g} kept as "
                    f"candidate modes: their lp is within "
                    f"{lp_exempt_margin:g} nats of the in-bulk median "
                    f"({lp_bulk_med:.2f}) or better -- a displaced basin, "
                    f"not a runaway."
                )
                z_ok |= exempt
        valid &= z_ok
        reasons[(reasons == "") & ~z_ok] = "raw-z"

    n_invalid = int((~valid).sum())
    invalid_reason_counts = {
        r: int((reasons[~valid] == r).sum())
        for r in _INVALID_REASONS
        if (reasons[~valid] == r).any()
    }
    invalid_per_chain = (~valid).reshape(n_chain, n_draw).sum(axis=1)
    if n_invalid:
        frac = n_invalid / n_samples
        notes.append(
            f"{n_invalid} draws ({frac:.2%}) rejected as invalid "
            f"(non-finite, |lp| > {lp_abs_max:g}, or raw-space "
            f"robust z > {z_max:g}); these are runaway/stuck "
            f"draws, not posterior modes. ONLY a model or sampler "
            f"bug should produce these -- investigate."
        )
        logger.warning(
            "identify_modes: %d/%d draws (%.2f%%) rejected as numerically "
            "invalid -- this should only happen due to a model or sampler "
            "bug. reasons=%s, per-chain invalid counts=%s",
            n_invalid,
            n_samples,
            100 * frac,
            invalid_reason_counts,
            invalid_per_chain.tolist(),
        )
    if not valid.any():
        # Carries the counts so the caller's validity gate has something to
        # gate on: no ModeReport can exist here, and a bare exception is
        # indistinguishable from any other mode-pass failure (review 3.17).
        raise NoValidDrawsError(
            n_invalid,
            n_samples,
            reason_counts=invalid_reason_counts,
            per_chain=invalid_per_chain.tolist(),
        )

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
            Xs[idx_fit], fit_labels, centers, merge_ratio
        )

    # lp-barrier ridge merge: the dip test above keeps any cluster the
    # occupancy density separates, but a flat likelihood ridge (an
    # unconstrained degeneracy direction, stretched out by the raw-space
    # transform) separates in DENSITY without separating in LIKELIHOOD.
    # Where lp exists, merge cluster pairs whose connecting path is
    # populated end to end with no lp barrier; alternate with the dip
    # merge until stable, since each merge moves centers.
    if has_lp and centers.shape[0] > 1:
        lp_fit = lp[valid][idx_fit]
        big = np.argmax(np.bincount(fit_labels, minlength=centers.shape[0]))
        lp_big = lp_fit[(fit_labels == big) & np.isfinite(lp_fit)]
        # lp scatter along a flat ridge: measured in the dominant cluster,
        # floored at the chi2 width sqrt(n_dims/2) it approaches for a
        # well-sampled Gaussian (degenerate fallback: the floor alone).
        sigma_lp = max(
            float(np.std(lp_big)) if lp_big.size > 1 else 0.0,
            float(np.sqrt(Xs.shape[1] / 2.0)),
        )
        merged = True
        while merged and centers.shape[0] > 1:
            fit_labels, centers, merged, ridge_notes = _lp_ridge_merge(
                Xs[idx_fit], lp_fit, fit_labels, centers, sigma_lp
            )
            notes.extend(ridge_notes)
            # a ridge merge moves centers; re-run the dip merge to its
            # own fixpoint before asking about ridges again
            dip = merged
            while dip and centers.shape[0] > 1:
                fit_labels, centers, dip = _dip_merge(
                    Xs[idx_fit], fit_labels, centers, merge_ratio
                )

    # assign every valid draw to nearest surviving center
    labels_valid = _assign_to_nearest_center(Xs, centers)

    # ---- weights, minor-cluster drop, ordering ---------------------------
    k = centers.shape[0]
    counts = np.bincount(labels_valid, minlength=k)
    weights = counts / counts.sum()
    major = np.where(weights >= min_weight)[0]
    n_unassigned = int(counts[weights < min_weight].sum())
    if n_unassigned:
        notes.append(
            f"{k - major.size} minor cluster(s) below "
            f"min_weight={min_weight} ({n_unassigned} draws) "
            f"left unassigned"
        )
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
        per_chain = np.array(
            [(row == m).sum() / max((row >= 0).sum(), 1) for row in labels_2d]
        )
        center_raw = {}
        center_scale_raw = {}
        sel_v = labels_valid == order[m]
        for jn, name in enumerate(dim_names):
            col = Xv[sel_v, jn]
            center_raw[name] = float(np.median(col))
            # 1.4826 * MAD: robust marginal sigma of this mode along jn
            center_scale_raw[name] = float(
                1.4826 * np.median(np.abs(col - np.median(col)))
            )
        # Occupancy weight uncertainty.  The weight is a mean of the mode
        # indicator, so its variance is w(1-w)/N_eff with N_eff = N/IACT --
        # governed by the number of independent mode transitions, NOT by the
        # number of draws.
        w_m = float(w_assigned[m])
        n_eff, tau_m = weight_ess(labels_2d, m)
        sigma_w = (
            float(np.sqrt(w_m * (1.0 - w_m) / max(n_eff, 1.0)))
            if n_modes > 1
            else 0.0
        )
        modes.append(
            ModeInfo(
                index=m,
                weight=w_m,
                n_draws=int(sel.sum()),
                lp_med=float(np.median(lp_m)) if lp_m.size else np.nan,
                lp_max=float(lp_m.max()) if lp_m.size else np.nan,
                delta_lp_max=(
                    float(best_lp - lp_m.max()) if lp_m.size else np.nan
                ),
                per_chain_weight=per_chain,
                center=center_raw,
                center_scale=center_scale_raw,
                weight_err=sigma_w,
                occ_weight=w_m,
                occ_weight_err=sigma_w,
                weight_ess=float(n_eff),
                weight_iact=float(tau_m),
            )
        )

    # ---- mixing diagnostics / weight provenance --------------------------
    n_transitions, transitions_per_chain, n_round_trips = transition_stats(
        labels_2d
    )
    thin_factor, thin_known = _trace_thinning(idata)
    n_no_switch = int(
        (
            (transitions_per_chain == 0)
            & np.array([bool((row >= 0).any()) for row in labels_2d])
        ).sum()
    )
    min_ess = min((m_.weight_ess for m_ in modes), default=float("nan"))
    if n_modes <= 1:
        provenance = "unimodal"
        reliable = True
    else:
        chains_visiting_all = all(
            all((row == m).any() for m in range(n_modes))
            for row in labels_2d
            if (row >= 0).any()
        )
        enough_transitions = n_transitions >= 10 * (n_modes - 1)
        reliable = chains_visiting_all and enough_transitions
        mixing = (
            f"{n_transitions} mode changes in the stored draws, "
            f"{n_round_trips} mode round trips, "
            f"{n_no_switch}/{int((transitions_per_chain >= 0).sum())} chains "
            f"never switched; N_eff for the weights >= {min_ess:.1f}"
        )
        if reliable:
            provenance = f"occupancy (validated: {mixing})"
        else:
            provenance = (
                f"occupancy (UNRELIABLE: chains do not mix between modes -- "
                f"weights reflect initialization, not posterior mass; use "
                f"per-mode evidence weighting or a folded likelihood) "
                f"[{mixing}]"
            )
            notes.append(
                "relative mode weights are NOT trustworthy: "
                f"{n_transitions} inter-mode transitions; "
                "see provenance"
            )
        # Advisory verdict, never a gate: the weights are always reported.
        if not np.isfinite(min_ess) or min_ess < DEFAULT_MIN_WEIGHT_ESS:
            worst = max(
                (m_ for m_ in modes),
                key=lambda m_: (
                    m_.occ_weight_err
                    if np.isfinite(m_.occ_weight_err)
                    else -1.0
                ),
            )
            notes.append(
                f"mode-weight precision: the mode-label series is worth only "
                f"N_eff ~ {min_ess:.1f} independent draws ({n_transitions} "
                f"mode changes, {n_round_trips} round trips, "
                f"{n_no_switch} chain(s) never switched), so the occupancy "
                f"weights carry ~{worst.occ_weight_err:.2f} 1-sigma "
                f"(mode {worst.index + 1}: "
                f"{_fmt_pm(worst.occ_weight, worst.occ_weight_err)}). "
                f"The weights above are still the best estimate available -- "
                f"treat their ORDERING as informative and their VALUES as "
                f"uncertain at that level. A longer run, a better-tuned "
                f"temperature ladder, or per-mode evidence weighting "
                f"(modes: {{weights: evidence}}) would tighten them."
            )

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
        invalid_reason_counts=invalid_reason_counts,
        invalid_per_chain=invalid_per_chain,
        transitions_per_chain=transitions_per_chain,
        n_round_trips=n_round_trips,
        thin_factor=thin_factor,
        thin_known=thin_known,
        ladder_round_trips=_int_attr(idata, "ptde_ladder_round_trips"),
        ladder_swap_rounds=_int_attr(idata, "ptde_swap_rounds"),
    )
    if attach:
        report.attach(idata)
    logger.info(
        "identify_modes: %d mode(s), weights=%s, %s",
        report.n_modes,
        [f"{m_.weight:.3f}+/-{m_.weight_err:.3f}" for m_ in report.modes],
        provenance,
    )
    return report
