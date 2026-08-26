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

import csv
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .latex import CSV_COLUMNS_MODE
from .texutils import latex_escape

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
        # ONE scratch copy of the whole center, reused for every probe.
        # _probe_element's bracket+bisect ladder calls eval_delta dozens of
        # times per element, and this used to rebuild the entire center dict
        # on every one of them -- for a perturbation of a single slot.  The
        # slot is restored after each evaluation instead, so the scratch is
        # always the unperturbed center on entry, exactly as a fresh copy
        # was.  (The restore is in a finally: a non-finite logp is a normal
        # outcome here -- _probe_element reads a wall inside +/-s as
        # past-target -- and an exception must not leave the scratch dirty
        # for the next element.)
        probe = {
            k: np.array(v, dtype=float, copy=True) for k, v in center.items()
        }
        for key, val in center.items():
            n = np.asarray(val).size
            sc = np.ones(n)
            for i in range(n):

                def eval_delta(step, key=key, i=i):
                    saved = probe[key].flat[i]
                    probe[key].flat[i] = saved + step
                    try:
                        return lp0 - float(logp_fn(probe))
                    finally:
                        probe[key].flat[i] = saved

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
            # internal -> user units.  Through Parameter.from_internal, and
            # NOT by dividing by _get_conversion_factors: that factor is the
            # internal -> user multiplier (config.py's get_conversion_factor
            # is the reciprocal), so the division this line used to do was
            # wrong by factor**2 for every converted parameter -- e.g. the
            # planet mass in examples/hd80606 was reported as 1.45e-06
            # jupiterMass for a start of 1.596, and every angle came out in
            # units of 1/3283 degree.  A SCALAR `unit:` normalizes to a
            # one-element list in Parameter.__post_init__, so the factor
            # vector is size 1 for every multi-element parameter and
            # broadcasts; a genuine per-element/element-count mismatch is
            # rejected inside from_internal.
            phys[name] = par.from_internal(p0)
            phys_sigma[name] = par.from_internal(0.5 * np.abs(p_hi - p_lo))
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


def _existing_row_width(csv_filename):
    """Width of the data rows already in ``csv_filename``, or None if empty.

    Comment lines ('#...') are skipped; the widest data row wins, so a
    file that is already ragged is reported as such rather than silently
    matching on its first row.
    """
    try:
        with open(csv_filename, newline="", encoding="utf-8") as f:
            widths = {
                len(row)
                for row in csv.reader(f)
                if row and not row[0].lstrip().startswith("#")
            }
    except FileNotFoundError:
        return None
    return max(widths) if widths else None


def append_ledger_csv(ledger, csv_filename):
    """Append rejected-mode Laplace rows to the results CSV.

    Row shape is latex.CSV_COLUMNS_MODE: name, mode, weight, weight_err,
    value, +err, -err -- with mode = 'rejected-seed<k>' and weight the
    Laplace weight relative to the best seed (an upper-bound-flavored
    estimate, labeled by the mode column itself).  weight_err is left blank:
    a Laplace ratio carries no sampling error bar of the kind the surviving
    modes' weights do.

    These rows only make sense in the mode-keyed layout -- the mode column
    IS their content -- so the caller must have written the file with
    build_csv_output(..., mode_columns=True).  A file whose rows are a
    different width raises instead of being made ragged: a mixed-width CSV
    breaks every consumer that assumes a rectangular table (pandas,
    csv.DictReader, any spreadsheet), and it breaks it silently.
    """
    rej = rejected_records(ledger)
    if not rej:
        return
    n_cols = len(CSV_COLUMNS_MODE)
    existing = _existing_row_width(csv_filename)
    if existing is not None and existing != n_cols:
        raise ValueError(
            f"{csv_filename} has {existing}-column rows but the ledger "
            f"writes {n_cols}-column rows ({', '.join(CSV_COLUMNS_MODE)}); "
            "write it with build_csv_output(..., mode_columns=True)."
        )
    best_logw = max(r.laplace_logw for r in ledger)
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        for r in rej:
            w = float(np.exp(r.laplace_logw - best_logw))
            for name in sorted(r.phys):
                vals = np.asarray(r.phys[name]).reshape(-1)
                sigs = np.asarray(r.phys_sigma[name]).reshape(-1)
                for i in r.sampled_idx.get(name, range(vals.size)):
                    writer.writerow(
                        [
                            name,
                            f"rejected-seed{r.seed_index}",
                            f"{w:.3g}",
                            "",
                            f"{vals[i]:.6g}",
                            f"{sigs[i]:.3g}",
                            f"{sigs[i]:.3g}",
                        ]
                    )
    logger.info(
        f"Ledger: appended {len(rej)} rejected mode(s) to {csv_filename}"
    )


def _delta_lp_cell(delta_lp):
    """The Delta ln P cell for one rejected seed.

    ``delta_lp`` is stored as best minus this seed, i.e. non-negative, and
    the table shows how far BELOW the best this seed sits -- so the printed
    value is its negation.  The best seed itself has delta_lp == 0, which
    formatted as "$-0.0$": a signed zero reads as a real (if tiny)
    difference.  Anything that rounds to zero at this precision is printed
    as an unsigned zero instead.
    """
    text = f"{-float(delta_lp):.1f}"
    if text == "-0.0":
        text = "0.0"
    return f"${text}$"


def write_rejected_latex(ledger, filename, hot_status=None):
    """Standalone LaTeX table of the rejected solutions (Laplace).

    ``hot_status`` (see hot_status_to_text) adds a RENDERED caption sentence
    -- not a % comment -- whenever the hot-chain suppressed-mode search did
    not run or did not complete.  This table is what goes into a paper, and
    a reader must not read its completeness as "these are all the
    alternatives that were considered" when the search for the others never
    happened.
    """
    rej = rejected_records(ledger)
    if not rej:
        return False
    caveat = ""
    state = (hot_status or {}).get("state")
    if state == HOT_NOT_SEARCHED:
        caveat = (
            r" No search for posterior-suppressed modes was performed, so "
            r"this list covers only the explicitly seeded solutions."
        )
    elif state == HOT_FAILED:
        caveat = (
            r" The search for additional posterior-suppressed modes did not "
            r"complete, so this list may be incomplete."
        )
    lines = []
    lines.append(r"% Considered-and-rejected solutions (Laplace")
    lines.append(r"% approximations at each seeded basin's polished optimum;")
    lines.append(r"% no posterior draws exist for these).")
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Solutions considered and rejected by the posterior. "
        r"Values are Laplace approximations at each seeded basin's "
        r"optimum; $\Delta \ln \mathcal{P}$ is the log-POSTERIOR "
        r"difference against the best seeded solution -- priors, "
        r"potentials and reparameterization Jacobians included, not a "
        r"likelihood ratio." + caveat + r"}"
    )
    cols = "l" + "c" * len(rej)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\hline")
    header = ["Parameter"] + [f"seed {r.seed_index}" for r in rej]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\hline")
    dl = [r"$\Delta \ln \mathcal{P}$"] + [
        _delta_lp_cell(r.delta_lp) for r in rej
    ]
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
            row = [latex_escape(f"{name}[{i}]")]
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

# The near-viable margin is derived from the LADDER, not fixed: a basin
# Delta nats below the best has hottest-rung occupancy e^(-Delta/T_max), so
# anything deeper than ~HOT_HORIZON_TMAX_FACTOR x T_max was not explorable by
# the run at all.  That depth is the run's SEARCH HORIZON, and it is reported
# as such -- absence of a basin beyond it means "not searchable at this
# T_max", never "considered and rejected".
#
# The old fixed HOT_LP_MARGIN = 50 was justified by MASS ("a mode 50 nats
# down carries e^-50 and loses nothing by being ignored") -- true for
# weights, wrong for the ledger's other job, REPORTING explored-and-rejected
# solutions.  Measured on DC2018 event 128 (48 rungs, T_max = 8500): the
# hot rungs held ~400k draws in the s <-> 1/s mirror basin, 779 nats below
# the main mode, and the 50-nat cut silently discarded every one of them.
HOT_HORIZON_TMAX_FACTOR = 10.0

# Fallback when the hot group carries no temperature coordinate (pre-2026-08
# traces): the old fixed margin, kept so old traces keep loading.
HOT_LP_MARGIN = 50.0

# Fewer near-viable points than this cannot support a cluster.
HOT_MIN_POINTS = 25

# The four distinguishable outcomes of the hot-chain suppressed-mode search.
#
# "no suppressed modes exist" and "the search never ran" and "the search
# crashed" all used to render identically in the final report -- nothing at
# all -- which manufactures exactly the false assurance this whole feature
# exists to prevent: a user reads "no other modes" and believes it was
# checked.  run.py owns the state machine (it is the only place that knows
# whether the group exists and whether the call raised); ledger.py owns the
# vocabulary and the rendering.
HOT_NOT_SEARCHED = "not-searched"
HOT_NONE_FOUND = "none-found"
HOT_FAILED = "failed"
HOT_FOUND = "found"


def run_hot_mode_discovery(system, model, idata, seed_ledger=None, **kwargs):
    """Run the hot-chain suppressed-mode search and CLASSIFY its outcome.

    Returns ``(seed_ledger, status)``.  The status dict always carries a
    ``state`` (one of the four HOT_* constants) so the final report can tell
    "no hot draws were kept, so nothing was searched for" from "we searched
    and found nothing" from "the search crashed".  Those three used to be
    indistinguishable in every output a user reads, which turns a silent
    failure into false assurance that a candidate mode was considered and
    rejected -- the exact thing this feature exists to provide.

    Non-fatal by contract: a wrap-up diagnostic must never take down a
    finished multi-day fit, so the catch stays broad -- but the exception's
    type and message go into the status rather than only into a log line.

    ``**kwargs`` reaches discover_hot_modes; ``cores=`` in particular is
    what keeps its polish off one core (see that function's docstring).
    """
    if not hasattr(idata, "posterior_hot"):
        return seed_ledger, {
            "state": HOT_NOT_SEARCHED,
            "detail": (
                "no posterior_hot group in the trace (sampler config "
                "`store_hot_chains`, which defaults on only for topologies "
                "that expect posterior-suppressed modes)"
            ),
        }
    status = {}
    try:
        seed_ledger = discover_hot_modes(
            system,
            model,
            idata.posterior_hot,
            seed_ledger,
            status=status,
            **kwargs,
        )
    except Exception as exc:
        status = {
            "state": HOT_FAILED,
            "detail": f"{type(exc).__name__}: {exc}",
        }
        logger.warning(
            "Hot-chain mode discovery failed; the final report will record "
            "that the suppressed-mode search did NOT complete",
            exc_info=True,
        )
    return seed_ledger, status


def hot_status_to_text(status):
    """Render the hot-chain search outcome for <prefix>_modes.txt.

    ``status`` is the dict discover_hot_modes fills in (plus a ``state`` set
    by the caller for the states discovery never reaches: it did not run, or
    it raised).  Returns "" for a falsy status so callers can pass it
    unconditionally.
    """
    if not status:
        return ""
    state = status.get("state", HOT_NOT_SEARCHED)
    detail = status.get("detail", "")
    lines = [
        "",
        "Hot-chain suppressed-mode search",
        "--------------------------------",
    ]
    if state == HOT_NOT_SEARCHED:
        lines.append(
            "NOT PERFORMED -- no hot-rung draws were available, so NO search "
            "for posterior-"
        )
        lines.append(
            "suppressed modes was made. The absence of extra solutions below "
            "is not evidence"
        )
        lines.append(
            "that none exist. Hot-rung retention defaults on only for "
            "topologies that expect"
        )
        lines.append(
            "suppressed modes (microlensing); force it with sampler config "
            "`store_hot_chains: true`."
        )
    elif state == HOT_FAILED:
        lines.append(
            "FAILED -- hot-rung draws were available but the search did not "
            "complete, so"
        )
        lines.append(
            "no conclusion can be drawn about posterior-suppressed modes. "
            "This is NOT the"
        )
        lines.append("same as having searched and found none.")
    elif state == HOT_NONE_FOUND:
        lines.append(
            "PERFORMED -- hot-rung draws were clustered and no candidate "
            "solution survived"
        )
        lines.append(
            "as a new basin. Within the limits of the Laplace/clustering "
            "approximations,"
        )
        lines.append("no additional mode was found.")
    elif state == HOT_FOUND:
        n = int(status.get("n_new", 0))
        lines.append(
            f"PERFORMED -- {n} candidate solution(s) found and recorded "
            f"below with source"
        )
        lines.append(
            "'hot-chain'. A candidate is a basin the T=1 posterior never "
            "held; whether it"
        )
        lines.append("survived is given by its own ledger entry.")
    else:  # unknown state -- never silently swallow it
        lines.append(f"UNKNOWN state '{state}'.")
    if state in (HOT_NONE_FOUND, HOT_FOUND) and "margin_nats" in status:
        # State the SEARCH HORIZON, so silence has a defined meaning: a basin
        # deeper than the horizon was not reachable at this T_max, and its
        # absence here is "not searched", never "considered and rejected".
        m = float(status["margin_nats"])
        if "t_max" in status:
            lines.append(
                f"search horizon: basins to Delta lp <= {m:g} nats "
                f"({HOT_HORIZON_TMAX_FACTOR:g} x T_max = "
                f"{float(status['t_max']):g}). Deeper basins have hottest-"
                f"rung occupancy below e^-{HOT_HORIZON_TMAX_FACTOR:g} and "
                "were not reachable by this run; their absence is 'not "
                "searchable at this T_max', not 'rejected'."
            )
        else:
            lines.append(
                f"search horizon: basins to Delta lp <= {m:g} nats (fixed "
                "fallback margin; the hot group carried no temperature "
                "coordinate)."
            )
    if state in (HOT_NONE_FOUND, HOT_FOUND):
        # A microlensing fit gets hot-rung retention without asking for it,
        # so say plainly that a search HAPPENED -- otherwise a reader who
        # never typed `store_hot_chains` reads this section as boilerplate.
        lines.append(
            "The search ran on the retained hot-rung draws (sampler config "
            "`store_hot_chains`,"
        )
        lines.append(
            "on by default for topologies that expect suppressed modes); it "
            "did not have to be"
        )
        lines.append("requested.")
    counts = [
        f"{k} = {status[k]}"
        for k in ("n_hot_draws", "n_viable", "n_clusters", "n_new")
        if k in status
    ]
    if counts:
        lines.append("  " + ", ".join(counts))
    if detail:
        lines.append(f"  {detail}")
    return "\n".join(lines) + "\n"


def discover_hot_modes(
    system,
    model,
    hot,
    seed_ledger=None,
    margin_nats=None,
    min_points=HOT_MIN_POINTS,
    max_modes=8,
    subsample=20000,
    seed=20260711,
    polish_steps=150,
    cores=None,
    status=None,
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

    ``status``, if given, is a dict this function fills with the outcome
    (``state`` in HOT_NONE_FOUND/HOT_FOUND, ``detail``, and the draw/cluster
    counts) so the final report can say WHICH of the four outcomes occurred
    instead of rendering "searched and found nothing" and "never searched"
    and "crashed" identically.  See hot_status_to_text.

    ``cores`` is the core grant for the polish, and callers must pass it:
    the DE engine is the branch a gradient-free model (every VBM-backed
    microlensing fit) takes, and with no grant it runs SERIAL on one core
    while the rest of the machine the fit just held sits idle -- measured
    on examples/ob09020 at 1 core of 36 for 38 minutes on a SINGLE
    candidate.  Left None (serial) rather than defaulted to a core count
    here because this function must not fork a pool a library caller never
    asked for; run.py hands over the same grant the sampler used, and
    polish.py logs the serial case so a third caller omitting it is
    visible in the log instead of silent (review 6.11.3).
    """
    from .modes import _dip_merge, _kmeans_bic

    if status is None:
        status = {}
    ledger = list(seed_ledger) if seed_ledger else []
    n_before = len(ledger)

    def _finish(state, detail=""):
        status["state"] = state
        status["detail"] = detail
        status["n_new"] = len(ledger) - n_before
        return ledger

    raw_keys = [str(v) for v in hot.data_vars if str(v) != "lp"]
    if not raw_keys or "lp" not in hot.data_vars:
        logger.warning(
            "Hot-chain discovery: the posterior_hot group carries no raw "
            "variables and/or no lp; no suppressed-mode search was made."
        )
        return _finish(
            HOT_FAILED, "the posterior_hot group has no raw variables or no lp"
        )
    lp = np.asarray(hot["lp"].values, dtype=float).reshape(-1)
    status["n_hot_draws"] = int(lp.size)
    if lp.size == 0:
        # An empty group is what a mis-sliced posterior_hot looks like (the
        # burn-in/stuck-chain trim used to index it by T=1 chains and draws
        # -- notes/code_review_20260808.txt 2.9.2).  Report it as a FAILED
        # search, never as "searched and found nothing".
        logger.warning(
            "Hot-chain discovery: the posterior_hot group holds no draws; "
            "no suppressed-mode search was performed. This group must reach "
            "discovery UNTRIMMED (samplers.convergence._TRIMMED_GROUPS)."
        )
        return _finish(
            HOT_FAILED,
            "the posterior_hot group reached discovery with 0 draws",
        )

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
        logger.warning(
            "Hot-chain discovery: every hot draw is non-finite in lp or in "
            "at least one variable; no suppressed-mode search was made."
        )
        return _finish(HOT_FAILED, "no finite hot draws to cluster")
    # Resolve the margin from the ladder unless the caller pinned one.
    if margin_nats is None:
        t_max = float("nan")
        try:
            if "temperature" in hot.coords or "temperature" in hot:
                t_max = float(np.nanmax(np.asarray(hot["temperature"])))
        except Exception:
            pass
        if np.isfinite(t_max) and t_max > 1.0:
            margin_nats = HOT_HORIZON_TMAX_FACTOR * t_max
            status["t_max"] = t_max
        else:
            margin_nats = HOT_LP_MARGIN
            logger.warning(
                "Hot-chain discovery: no temperature coordinate on the hot "
                f"group; falling back to the fixed {HOT_LP_MARGIN:g}-nat "
                "margin. Basins deeper than that will not be reported."
            )
    status["margin_nats"] = float(margin_nats)

    viable = good & (lp >= np.nanmax(lp[good]) - margin_nats)
    n_viable = int(viable.sum())
    status["n_viable"] = n_viable
    if n_viable < min_points:
        # Genuinely searched: the hot rungs simply never came near the best
        # solution, which IS the "nothing else is competitive" answer.
        logger.info(
            f"Hot-chain discovery: only {n_viable} near-viable hot draws "
            f"(need {min_points}); nothing to cluster."
        )
        return _finish(
            HOT_NONE_FOUND,
            f"only {n_viable} of {lp.size} hot draws came within "
            f"{margin_nats:g} nats of the best (need {min_points} to "
            f"cluster)",
        )
    # -----------------------------------------------------------------
    # LEVEL-SET LADDER.  One clustering pass at the full horizon margin
    # does not work, and the failure is measured, not hypothetical: on
    # DC2018 event 128 (horizon 85,000 nats, 2.9M viable draws) the space
    # BETWEEN the two basins fills with in-transit hot draws, the density
    # dip vanishes, and the merge step returns a single cluster -- the
    # mirror basin, held by ~400k stored draws, went unreported.  At
    # intermediate margins (1000-5000 nats) the same basin separates
    # cleanly, because the lp cut removes the bridge before it removes the
    # basin.  So basins are found the way sublevel sets find them: cluster
    # at an increasing ladder of margins, collect each level's cluster
    # candidates, and let the polish + dedup collapse the duplicates (the
    # main basin is rediscovered at every level; that is what dedup is
    # for).  A basin is reportable if it separates at ANY level up to the
    # horizon.
    #
    # Clustering is in WHITENED RAW UNITS, center-only.  The previous
    # per-column standardization divided by the viable set's own (hot)
    # spread, which crushes exactly the likelihood-informed directions
    # where a second basin shows up as a large multiple of the cold sigma,
    # and promotes diffuse prior-dominated directions to unit scale --
    # measured on the same event, it returned two clusters BOTH in the
    # main basin, split along the stellar/kinematic sector.
    rng = np.random.default_rng(seed)
    ladder = [
        m
        for m in (50.0, 250.0, 1000.0, 5000.0, 20000.0, margin_nats)
        if m <= margin_nats
    ]
    if ladder[-1] != margin_nats:
        ladder.append(margin_nats)
    status["margin_ladder"] = list(ladder)

    candidates = []  # (lp, packed raw row)
    per_level = []
    lp_best = np.nanmax(lp[good])
    prev_level = 0.0
    for level in ladder:
        # Cluster the SHELL -- draws new at this level -- not the cumulative
        # sublevel set.  Measured failure of the cumulative version on the
        # DC2018-128 trace: at margin 1000 the mirror basin is 12k of 1.26M
        # viable draws (~1%), and k-means simply never places a centroid on
        # a small remote cluster, so every level's candidates were the main
        # basin again.  In the 500-1000 shell the same basin is ~11% of the
        # points and separates reliably; a basin whose depth falls between
        # two levels ALWAYS dominates some shell, which is the whole logic
        # of a level-set filtration.
        v = good & (lp >= lp_best - level) & (lp < lp_best - prev_level)
        if prev_level == 0.0:
            # the innermost shell includes the best point itself
            v = good & (lp >= lp_best - level)
        prev_level = level
        Xv, lpv = X[v], lp[v]
        if Xv.shape[0] < min_points:
            per_level.append((level, int(Xv.shape[0]), 0))
            continue
        if Xv.shape[0] > subsample:
            keep = rng.choice(Xv.shape[0], size=subsample, replace=False)
            Xv, lpv = Xv[keep], lpv[keep]
        Z = Xv - np.median(Xv, axis=0)
        labels, centers = _kmeans_bic(Z, max_modes=max_modes, seed=seed)
        while True:
            labels, centers, changed = _dip_merge(
                Z, labels, centers, merge_ratio=0.5
            )
            if not changed:
                break
        ks = np.unique(labels[labels >= 0])
        per_level.append((level, int(Xv.shape[0]), int(len(ks))))
        for c in ks:
            sel = labels == c
            b_i = int(np.argmax(lpv[sel]))
            candidates.append((float(lpv[sel][b_i]), Xv[sel][b_i]))

    # Pre-dedup candidates by the MIDPOINT TEST -- topological, scale-free.
    # Position thresholds cannot work here: under unmeasured preliminary
    # whitening a toy's two genuine basins sat 2.2 "sigma" apart while the
    # real run's same-basin annulus duplicates sat hundreds out, so any
    # fixed or lp-implied radius misclassifies one of them.  The property
    # that actually distinguishes "same basin" is concavity along the chord:
    # within one concave basin, lp at the midpoint of two points is >= the
    # smaller endpoint lp (a concave function on a segment attains its
    # minimum at an endpoint); a VALLEY between two basins breaks that by
    # construction.  One logp evaluation per pair, using the model's own lp.
    candidates.sort(key=lambda t: -t[0])
    logp_fn = model.compile_logp()

    def _lp_of_row(row):
        cand, ofs = {}, 0
        for key in raw_keys:
            n = shapes[key]
            cand[key] = np.array(row[ofs : ofs + n], dtype=float)
            ofs += n
        try:
            return float(logp_fn(cand))
        except Exception:
            return -np.inf

    kept = []
    for lp_c, x_c in candidates:
        dup = False
        for lp_k, x_k in kept:
            lp_mid = _lp_of_row(0.5 * (x_c + x_k))
            if lp_mid >= min(lp_c, lp_k) - 1.0:
                dup = True  # no valley between them: same basin
                break
        if not dup:
            kept.append((lp_c, x_c))
    status["n_clusters"] = int(len(kept))
    logger.info(
        "Hot-chain discovery (level ladder): "
        + "; ".join(
            f"margin {lv:g}: {nv} viable -> {nc} cluster(s)"
            for lv, nv, nc in per_level
        )
        + f" => {len(kept)} distinct candidate(s) after pre-dedup"
    )

    from ..polish import polish_raw_starts

    next_index = max((r.seed_index for r in ledger), default=-1) + 1

    # Rebuild raw dicts for every kept candidate, then polish them as ONE
    # BATCH.  Polishing one candidate at a time gives each an infinite trust
    # radius (a single seed is unconstrained by design), and a candidate
    # from a suppressed basin defects to the dominant one during its own
    # polish -- the mirror, 779 nats below the main mode, walked out exactly
    # as the MMEXOFAST seeds once did, and dedup then erased the discovery.
    # As one batch, polish_seed_starts' multi-seed trust region cages each
    # candidate within half the distance to its nearest neighbour, which is
    # the same contract multi-seed sampling starts get.
    cands = []
    for _lp_c, x_best in kept:
        cand, ofs = {}, 0
        for key in raw_keys:
            n = shapes[key]
            cand[key] = np.array(x_best[ofs : ofs + n], dtype=float)
            ofs += n
        cands.append(cand)
    if cands:
        logger.info(
            f"Hot-chain discovery: polishing {len(cands)} candidate(s) to "
            f"their basin optima (at most {int(polish_steps)} steps each)."
        )
        polished_all, _dlps, _method = polish_raw_starts(
            model,
            cands,
            n_steps=polish_steps,
            # The SAME core grant run.py hands the pre-sampling polish, and
            # for the same reason -- this call omitted it and so ran the DE
            # engine serial on one core of the machine the sampler had just
            # been using in full (review 6.11.3).
            cores=cores,
        )
        recs_all = build_seed_ledger(
            system,
            model,
            polished_all,
            list(range(next_index, next_index + len(polished_all))),
        )
    else:
        recs_all = []

    for rec in recs_all:
        rec.source = "hot-chain"
        # Dedup: an existing record already sitting in this basin (within
        # MATCH_SIGMA of the polished point, in the new record's own widths)
        # makes this a rediscovery, not a discovery.
        dup = False
        for other in ledger:
            ds = []
            for key in raw_keys:
                a = np.asarray(rec.raw_point[key], dtype=float).reshape(-1)
                b = np.asarray(
                    other.raw_point.get(key, np.full_like(a, np.nan)),
                    dtype=float,
                ).reshape(-1)
                sc = np.asarray(rec.raw_scales[key], dtype=float).reshape(-1)
                if b.shape != a.shape or not np.all(np.isfinite(b)):
                    ds = []
                    break
                ds.extend(np.abs(a - b) / np.maximum(sc, 1e-300))
            if ds and max(ds) <= MATCH_SIGMA:
                dup = True
                break
        if dup:
            continue
        logger.info(
            f"Hot-chain discovery: new basin at lp={rec.lp_max:.2f} "
            f"recorded as ledger entry {rec.seed_index} (hot-chain)."
        )
        ledger.append(rec)
    n_new = len(ledger) - n_before
    if n_new:
        return _finish(
            HOT_FOUND,
            f"{len(kept)} hot cluster(s) -> {n_new} new basin(s) after "
            f"dedup against the existing ledger",
        )
    return _finish(
        HOT_NONE_FOUND,
        f"{len(kept)} hot cluster(s), all of them rediscoveries of basins "
        f"already in the ledger",
    )
