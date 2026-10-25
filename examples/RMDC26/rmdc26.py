"""Paths, thresholds and every DECISION of the RMDC26 evidence report.

The two stage scripts (01_evidence.py, 02_report.py) only do I/O around
these functions: which fits an event has, how per-mode lnZ becomes a
relative probability, which solutions are flagged or inactive, and what the
microlens-submit files contain.  Everything here is a plain function of
numbers and dicts, so ``tests/test_rmdc26_report.py`` drives it without a
sampler, a trace or the challenge data.

This is a port of examples/DC2018_069 (blind.py, stages 4-5) to the RMDC26
layout, where ONE sampling run per architecture carries several grid seeds
and identify_modes splits it into the event's modes.  So there is no
cross-run dedupe here: every identified mode of every fit is a solution.
"""

import json
import math
import os
import re
from pathlib import Path

import numpy as np

#: The campaign (READ ONLY: another agent's live run writes there).
CAMPAIGN = Path(
    os.environ.get("RMDC26_RUN", "/n/holylabs/jjohnson_lab/Lab/RMDC26_run")
)
#: Everything this report writes.  Never $HOME (95 G quota).
OUT = Path(
    os.environ.get(
        "RMDC26_REPORT", "/n/holylabs/jjohnson_lab/Lab/rmdc26_report"
    )
)

# --- the submission vocabulary ---------------------------------------------

#: microlens-submit numbers bands by INDEX ("0", "1", "2" -> F0_S, ...), so
#: the order is fixed once, here: F0 = F146, F1 = F087, F2 = F213.
BAND_INDEX = {"F146": 0, "F087": 1, "F213": 2}

ARCH_PSPL = "1S1L"
ARCH_BINARY = "1S2L"

# --- evidence and probabilities ---------------------------------------------

#: Systematic floor (nats) added in quadrature to every mode's bridge error.
#: The bridge error describes one run's draws, not run-to-run scatter: two
#: runs of ONE DC2018_069 basin differed by 2.2 nats against bridge errors of
#: 0.2 (examples/DC2018_069/README.md, "Why two runs of one basin
#: disagree").  2 nats is that measured scatter, rounded.
SYSTEMATIC_FLOOR = 2.0
#: Bridge sizing: estimate_mode_evidences caps the PROPOSAL draws at the
#: posterior cap, and at 4000 every DC2018_069 run was refused (re2 0.37-
#: 0.63); at 20000 re2 was 0.03-0.05.
MAX_POSTERIOR_DRAWS = 20000
#: A solution is is_active when its robust probability is at least this.
#: Under the +/-2 nat floor two modes are only distinguishable when they
#: differ by more than ~2.8 nats (the floors of both, in quadrature); 1e-3 is
#: dlnZ = 6.9 nats below the best, ~2.4 of those combined sigmas, so an
#: inactive solution is one the evidence disfavours beyond what the
#: systematic can explain, while everything plausibly competitive stays in
#: the export.  The active set is renormalized to sum to 1, because
#: microlens-submit's validator requires that of the active solutions.
ACTIVE_MIN_PROB = 1e-3

# --- reliability thresholds -------------------------------------------------

#: lp reproduction (lpcheck.py): the report's tree must evaluate the
#: sampler's logp at the stored draws to this many nats, or no lnZ is used.
LPCHECK_TOL = 0.01
#: Stuck chains: a chain whose mean lp over the second half sits more than
#: STUCK_DLP below the best chain's (dc18_rerun/launch1015/diag30.py, the
#: flag that caught 7/9 detectable failures).
STUCK_FRAC_MAX = 0.1
STUCK_DLP = 20.0
#: Split-Rhat on lp (a flag, never an exclusion).
LP_RHAT_MAX = 1.1
#: Bridge re2 above this is a warning (evidence.py refuses at 0.25).
RE2_FLAG = 0.1
#: More than this fraction of draws rejected by identify_modes.
INVALID_FRAC_MAX = 0.05
#: err_scale outside this band means the noise model, not the lens model, is
#: carrying the fit: the solution is UNRELIABLE whatever its lnZ.
ERR_SCALE_RANGE = (0.8, 1.25)
#: out_scale split-Rhat quoted next to an AT_CAP flag; above this it is
#: also called out on its own.
OUT_SCALE_RHAT_MAX = 1.1

#: A mode held by fewer chains than this is one or two chains parked in a
#: basin, not a population sampling it: its posterior width and its bridge
#: describe those chains only.
MIN_MODE_CHAINS = 3

#: POSSIBLE_UNLISTED_MODE: more than INVALID_FRAC_MAX of the draws are
#: raw-z rejected AND at least UNLISTED_COVERAGE of those rejected draws are
#: out of range ONLY in one or two GEOMETRIC parameters.  That is a coherent
#: displaced basin the mode pass threw away (173: 28% rejected, all in
#: source.t_0_raw, 10 of 36 chains, ~25,000 nats below the bulk), so the
#: mode list may be incomplete.  Diffuse, many-parameter rejection keeps
#: the generic invalid-fraction warning.
UNLISTED_COVERAGE = 0.8
UNLISTED_MAX_PARAMS = 2
#: Sampled (raw) parameters that set the light-curve geometry: the
#: trajectory, the lens, and the physical parameters t_E / pi_E / rho are
#: derived from.  Noise and flux parameters (err_scale, out_frac,
#: out_scale, q_source, log_f_total) are not modes of the event.
GEOMETRIC_RAW = (
    "source.t_0_raw", "source.u_0_raw", "lens.log_s_raw", "planet.log_q_raw",
    "lens.xalpha_raw", "lens.yalpha_raw", "star.logmass_raw",
    "star.distance_raw", "star.pm_ra_raw", "star.pm_dec_raw",
    "star.radius_raw",
)  # fmt: skip

#: Events every solution of which is UNRELIABLE whatever its lnZ, with the
#: reason (DC2026 campaign agent, 2026-10-21).
FORCED_UNRELIABLE = {
    16: "forced: err_scale F213 21.0 (the noise model carries the fit)",
    163: "forced: err_scale F087 3.2 (the noise model carries the fit)",
    173: "forced: err_scale F213 8.1; VARIABILITY + FORGIVEN + AT_CAP in "
    "all three bands; a third of the chains in a worse t_0 mode",
}

#: The outlier-audit kinds (exozippy.components.likelihood FLAG_*).
OUTLIER_KINDS = ("ANOMALY", "VARIABILITY", "FORGIVEN", "AT_CAP")

#: The one paragraph every solution's notes carry (PI rulings, 2026-10).
CONVENTIONS = (
    "Conventions (EXOZIPPy, src/exozippy/components/mulensing/"
    "conventions.md): t0 is BJD_TDB exactly as the challenge's `bjd` column "
    "gives it (no offset subtracted). u0 > 0 means the lens passes the "
    "source on its right (Skowron+2011 A.6). alpha is in RADIANS and is our "
    "fitted binary-axis angle with NO conversion: the angle from the "
    "primary->companion axis to the direction of LENS motion relative to "
    "the source, counter-clockwise (North through East) -- identical to "
    "MulensModel and to Skowron+2011 as written (C15, C18). q = m_companion "
    "/ m_primary (C14). piEN/piEE are the geocentric parallax components in "
    "the same frame (C8-C10), anchored at t_ref = t0_par, and are DERIVED "
    "from the physical lens/source model (masses, distances, proper "
    "motions), not free light-curve parameters. Fluxes are F = 10**(-0.4 m) "
    "on the challenge's own (AB) magnitudes, zero point 0, band 0 = F146, "
    "1 = F087, 2 = F213. Limb darkening is zero (u1 pinned at 0 in every "
    "band). Binary sources (2S1L) are out of scope."
)


def event_id(ev):
    """The challenge's event name, RMDC26_000094 for 94."""
    return "RMDC26_%06d" % int(ev)


def ev3(ev):
    return "%03d" % int(ev)


# ---------------------------------------------------------------------------
# Which fits an event has
# ---------------------------------------------------------------------------


def config_path(fit_dir):
    """A fit's system config: RMDC26_<ev>.yaml, the one stem with no dot."""
    d = Path(fit_dir)
    cands = [p for p in sorted(d.glob("RMDC26_*.yaml")) if "." not in p.stem]
    if len(cands) != 1:
        raise FileNotFoundError(f"{d}: expected one RMDC26_*.yaml: {cands}")
    return cands[0]


def architecture(config):
    """1S2L when the lens block has a companion body, else 1S1L."""
    lens = config.get("lens") or []
    return ARCH_BINARY if len(lens) > 1 or config.get("planet") else ARCH_PSPL


def ld_pinned_zero(params):
    """True when every band.<name>.u1 in the params file is pinned at 0
    (initval 0, sigma 0) -- the PI's zero-limb-darkening ruling."""
    keys = [k for k in params if re.fullmatch(r"band\.[^.]+\.u1", str(k))]
    if not keys:
        return False
    for k in keys:
        v = params[k] or {}
        if float(v.get("initval", np.nan)) != 0.0 or v.get("sigma") != 0:
            return False
    return True


def staleness_from_mtimes(trace, config, params, whitening):
    """Why a trace cannot belong to the files beside it, or "" (pure).

    All arguments are mtimes (seconds; None when the file is missing).  A
    config or params file written AFTER the trace means the configs were
    regenerated after this fit ran (2026-10-20 03:46: the switch to zero
    limb darkening), so the trace is the OLD model's.  A whitening file
    written after the trace means a new fit of this event has started and
    overwritten it: the trace's draws no longer decode.
    """
    why = []
    if trace is None:
        return "no trace"
    for name, t in (("config", config), ("params file", params)):
        if t is not None and t > trace:
            why.append(
                "%s is newer than the trace (regenerated after this fit)"
                % name
            )
    if whitening is None:
        why.append("no whitening file")
    elif whitening > trace + 60:
        why.append("whitening is newer than the trace (a re-fit is under way)")
    return "; ".join(why)


def trace_staleness(fit_dir):
    d = Path(fit_dir)

    def mt(paths):
        paths = list(paths)
        return paths[0].stat().st_mtime if paths else None

    cfg = config_path(d)
    import yaml

    pf = (yaml.safe_load(cfg.read_text()) or {}).get("parameter_file")
    return staleness_from_mtimes(
        mt((d / "fitresults").glob("*_trace.nc")),
        cfg.stat().st_mtime,
        (d / pf).stat().st_mtime if pf and (d / pf).exists() else None,
        mt((d / "fitresults").glob("*_whitening.json")),
    )


def candidate_fit_dirs(ev, root=CAMPAIGN):
    """events/<ev>/ and events/<ev>_*/ that hold a finished trace."""
    base = Path(root) / "events"
    out = []
    for d in [base / ev3(ev)] + sorted(base.glob(ev3(ev) + "_*")):
        if d.is_dir() and list((d / "fitresults").glob("*_trace.nc")):
            out.append(d)
    return out


_BURN_RE = re.compile(
    r"burn-in discarded: (\d+) draws \(\d+% of (\d+)\); chains kept: (\d+)"
)
_UNCONV_RE = re.compile(
    r"convergence NOT reached -- max Rhat=([0-9.eE+-]+) \(([^)]*)\), "
    r"min ESS=([0-9.eE+-]+) \(([^)]*)\)"
)


def parse_wrapup_text(summary_text, log_text):
    """The fit's own wrap-up verdicts, from <prefix>_summary.txt (the
    burn-in banner and the convergence WARNING) and <prefix>.log (the
    NOT EQUILIBRATED warning)."""
    out = {}
    m = _BURN_RE.search(summary_text or "")
    if m:
        out["burnin"] = int(m.group(1))
        out["n_draw"] = int(m.group(2))
        out["chains_kept"] = int(m.group(3))
    m = _UNCONV_RE.search(summary_text or "")
    out["unconverged"] = bool(m)
    if m:
        out["max_rhat"] = float(m.group(1))
        out["worst_rhat_var"] = m.group(2)
        out["min_ess"] = float(m.group(3))
    if log_text is not None:
        out["not_equilibrated"] = "NOT EQUILIBRATED" in log_text
    return out


def parse_wrapup(fit_dir, prefix=None):
    """The wrap-up verdicts of a fit: events/<NNN>/fitresults/ by default,
    or the files of an explicit output ``prefix`` (``out/006`` ->
    out/006_summary.txt and out/006.log, the SED-tied layout)."""
    if prefix is not None:
        p = Path(prefix)
        summ = [x for x in [Path(str(p) + "_summary.txt")] if x.exists()]
        logs = [x for x in [Path(str(p) + ".log")] if x.exists()]
        st = summ[0].read_text(errors="replace") if summ else ""
        lt = logs[0].read_text(errors="replace") if logs else None
        return parse_wrapup_text(st, lt)
    fr = Path(fit_dir) / "fitresults"
    summ = sorted(fr.glob("*_summary.txt"))
    logs = sorted(fr.glob("RMDC26_*[0-9].log"))
    st = summ[0].read_text(errors="replace") if summ else ""
    lt = logs[0].read_text(errors="replace") if logs else None
    return parse_wrapup_text(st, lt)


# ---------------------------------------------------------------------------
# Pure statistics
# ---------------------------------------------------------------------------


def split_rhat(m):
    """Split-Rhat of an (n_chain, n_draw) array."""
    m = np.asarray(m, dtype=float)
    if m.ndim != 2 or m.shape[1] < 4:
        return float("nan")
    h = m.shape[1] // 2
    m = np.concatenate([m[:, :h], m[:, h : 2 * h]], axis=0)
    n = m.shape[1]
    w = m.var(axis=1, ddof=1).mean()
    b = n * m.mean(axis=1).var(ddof=1)
    if not np.isfinite(w) or w <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def stuck_fraction(lp, dlp=STUCK_DLP):
    """Fraction of chains whose second-half mean lp is > dlp below the best
    chain's (diag30.py's definition, on the FULL stored lp)."""
    lp = np.asarray(lp, dtype=float)
    cm = lp[:, lp.shape[1] // 2 :].mean(axis=1)
    return float((cm < cm.max() - dlp).mean())


def percentiles(v, q=(15.865, 50.0, 84.135)):
    """(median, minus, plus): the 68% interval as positive offsets."""
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    lo, med, hi = np.percentile(v, q)
    return float(med), float(med - lo), float(hi - med)


def wrap_pi(x):
    return (np.asarray(x, dtype=float) + math.pi) % (2 * math.pi) - math.pi


def circ_percentiles(rad):
    """percentiles() of an angle in radians, unwrapped around its circular
    mean, with the median reported in [0, 2 pi)."""
    a = np.asarray(rad, dtype=float).ravel()
    a0 = math.atan2(np.sin(a).mean(), np.cos(a).mean())
    med, lo, hi = percentiles(a0 + wrap_pi(a - a0))
    return float(med % (2 * math.pi)), lo, hi


def total_error(bridge_err, floor=SYSTEMATIC_FLOOR):
    """Bridge error and the systematic floor, in quadrature."""
    e = np.asarray(bridge_err, dtype=float)
    e = np.where(np.isfinite(e), e, 0.0)
    return np.sqrt(e**2 + floor**2)


def indistinguishable_groups(lnz, sigma):
    """Partition modes into groups the evidence cannot order.

    Greedy from the top: the best remaining mode leads a group, and every
    remaining mode whose lnZ is within the leader's and its own total errors
    in quadrature, ``lnZ_lead - lnZ_k <= sqrt(sigma_lead**2 + sigma_k**2)``,
    joins it.  Repeat on what is left.  No chaining: membership is always
    measured against the leader, so a long run of modes 2 nats apart does
    not collapse into one group.  Non-finite lnZ never joins a group.

    Returns a list of index lists, best group first.
    """
    lnz = np.asarray(lnz, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    left = [i for i in np.argsort(-lnz) if np.isfinite(lnz[i])]
    groups = []
    while left:
        lead = left[0]
        g = [
            i
            for i in left
            if lnz[lead] - lnz[i] <= math.hypot(sig[lead], sig[i])
        ]
        groups.append([int(i) for i in g])
        left = [i for i in left if i not in g]
    return groups


def softmax(lnz):
    """exp(lnZ - max) normalized; non-finite lnZ gets 0."""
    lnz = np.asarray(lnz, dtype=float)
    p = np.zeros_like(lnz)
    ok = np.isfinite(lnz)
    if ok.any():
        z = np.exp(lnz[ok] - lnz[ok].max())
        p[ok] = z / z.sum()
    return p


def robust_probabilities(lnz, bridge_err, floor=SYSTEMATIC_FLOOR):
    """The relative probabilities a submission reports.

    1. ``p_raw`` = softmax(lnZ) over every mode of every architecture of
       the event (lnZ is NORMALIZED, so the architectures share one scale).
    2. Modes are grouped by ``indistinguishable_groups`` with the total
       error ``sqrt(err_bridge**2 + floor**2)``.
    3. Each group's TOTAL raw probability is shared EQUALLY among its
       members (``p_robust``).  The group totals, and so the sum, are kept;
       only the split inside a group -- which the evidence cannot make -- is
       replaced by the uninformative one.

    Returns dict(p_raw, p_robust, groups, sigma, group_of).
    """
    lnz = np.asarray(lnz, dtype=float)
    sig = total_error(bridge_err, floor)
    p_raw = softmax(lnz)
    groups = indistinguishable_groups(lnz, sig)
    p = p_raw.copy()
    group_of = [-1] * lnz.size
    for gi, g in enumerate(groups):
        tot = float(p_raw[g].sum())
        for i in g:
            p[i] = tot / len(g)
            group_of[i] = gi
    return {
        "p_raw": p_raw,
        "p_robust": p,
        "groups": groups,
        "sigma": sig,
        "group_of": group_of,
    }


def refused_competitors(lnz, accepted, n_sigma=3.0, floor=SYSTEMATIC_FLOOR):
    """Refused modes whose INDICATIVE lnZ is not far below the best accepted.

    A refused bridge contributes no probability, so a refused mode that the
    estimate puts above, or within ``n_sigma`` floors of, the best ACCEPTED
    mode means the accepted probabilities may be missing real mass.  Returns
    [(index, lnZ_refused - lnZ_best_accepted)], largest first.  Empty when
    nothing is accepted (the event is then INDICATIVE as a whole).
    """
    lnz = np.asarray(lnz, dtype=float)
    acc = np.asarray(accepted, dtype=bool)
    if not acc.any():
        return []
    best = float(np.max(lnz[acc]))
    out = [
        (int(i), float(lnz[i] - best))
        for i in np.flatnonzero(~acc & np.isfinite(lnz))
        if lnz[i] >= best - n_sigma * floor
    ]
    return sorted(out, key=lambda t: -t[1])


def active_and_renormalized(p, eligible=None, min_prob=ACTIVE_MIN_PROB):
    """(is_active, relative_probability) per solution.

    Active: eligible and p >= min_prob.  The active probabilities are
    renormalized to sum to 1 (the validator's rule for active solutions);
    an inactive solution keeps its unrenormalized p, which is then an
    upper-tail statement and never enters the export.  If nothing passes
    (no accepted lnZ at all), nothing is active and p is returned as is.
    """
    p = np.asarray(p, dtype=float)
    el = np.ones(p.size, bool) if eligible is None else np.asarray(eligible)
    act = el & (p >= min_prob)
    out = p.copy()
    if act.any() and p[act].sum() > 0:
        out[act] = p[act] / p[act].sum()
    return act, out


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def base_param(dim):
    """'star.pm_ra_raw[1]' -> 'star.pm_ra_raw'."""
    return dim.split("[", 1)[0]


def unlisted_mode_params(beyond, dim_names, n_total,
                         frac_min=INVALID_FRAC_MAX,
                         coverage=UNLISTED_COVERAGE,
                         max_params=UNLISTED_MAX_PARAMS,
                         geometric=GEOMETRIC_RAW):  # fmt: skip
    """The geometric parameter(s) a raw-z rejection concentrates in, or [].

    ``beyond`` is a boolean (n_rejected, n_dims) matrix: which dimensions of
    each raw-z-REJECTED draw lie beyond the robust-z cut; ``n_total`` the
    draws the mode pass saw.  Fires when rejected / total > frac_min and a
    set of at most ``max_params`` geometric parameters (elements of one
    vector count as one parameter) contains EVERY out-of-range dimension of
    at least ``coverage`` of the rejected draws.  The smallest such set is
    returned, most-implicated first.
    """
    B = np.asarray(beyond, dtype=bool)
    if B.ndim != 2 or B.shape[0] == 0 or n_total <= 0:
        return []
    if B.shape[0] / n_total <= frac_min:
        return []
    params = sorted({base_param(d) for d in dim_names})
    col = {
        p: np.array([base_param(d) == p for d in dim_names]) for p in params
    }
    counts = {p: int(B[:, col[p]].any(axis=1).sum()) for p in params}
    cands = [p for p in sorted(params, key=lambda p: -counts[p])
             if p in geometric and counts[p] > 0][: max(4, max_params)]  # fmt: skip
    from itertools import combinations

    for k in range(1, max_params + 1):
        best = None
        for S in combinations(cands, k):
            inside = np.zeros(len(dim_names), bool)
            for p in S:
                inside |= col[p]
            cov = float((~B[:, ~inside].any(axis=1)).mean())
            if cov >= coverage and (best is None or cov > best[0]):
                best = (cov, S)
        if best is not None:
            return sorted(best[1], key=lambda p: -counts[p])
    return []


def outlier_kinds(rows):
    """{file: sorted kinds} from robust-audit rows (outputs/outliers.py)."""
    out = {}
    for r in rows or []:
        kinds = {k for k, _ in r.get("flags", [])}
        out.setdefault(r["name"], set()).update(kinds)
    return {k: sorted(v) for k, v in out.items()}


def reliability_flags(sol):
    """(unreliable, warnings): two lists of str for one solution record.

    UNRELIABLE -- its lnZ, and so its probability, should not be trusted:
      the event is in FORCED_UNRELIABLE; the report's tree did not
      reproduce the sampler's lp; the bridge was refused (there is no
      accepted lnZ); an err_scale outside ERR_SCALE_RANGE (the noise model
      is fitting the data, e.g. 016 F213 21.0, 173 F213 8.1, 163 F087 3.2);
      or, for a 1S1L solution, an ANOMALY outlier flag (the mixture absorbs
      points AT the event: the single lens is missing a signal and its
      lnZ is subsidized).
    WARNING -- worth reading before relying on it: NOT EQUILIBRATED (lp
      still trending; first, because it undercuts Rhat/ESS), unconverged,
      a mode held by < MIN_MODE_CHAINS chains, stuck chains, lp Rhat,
      POSSIBLE_UNLISTED_MODE or the invalid-draw fraction, re2, ANOMALY on
      a 1S2L solution (the binary leaves residual structure at the event;
      on 2026-10-21 it fired on most 1S2L modes, so it does not condemn
      one), VARIABILITY / FORGIVEN / AT_CAP with the out_scale Rhat, and
      out_scale Rhat on its own.
    """
    bad, warn = [], []
    forced = FORCED_UNRELIABLE.get(sol.get("event"))
    if forced:
        bad.append(forced)
    lc = sol.get("lpcheck") or {}
    if lc and not lc.get("dlp_max_abs", 0.0) < LPCHECK_TOL:
        bad.append(
            "lp NOT reproduced by the report's tree (max |dlp| %.3g)"
            % lc.get("dlp_max_abs", float("nan"))
        )
    st = sol.get("bridge_state", "ok")
    if sol.get("bridge_refused"):
        bad.append("bridge refused (%s)" % st)
    lo, hi = ERR_SCALE_RANGE
    for band, v in sorted((sol.get("err_scale") or {}).items()):
        if np.isfinite(v) and not lo <= v <= hi:
            bad.append(
                "err_scale %s = %.3g outside [%g, %g]" % (band, v, lo, hi)
            )
    kinds = sol.get("outlier_kinds") or {}
    for f, ks in sorted(kinds.items()):
        if "ANOMALY" in ks:
            (bad if sol.get("arch") == ARCH_PSPL else warn).append(
                "ANOMALY outliers in %s" % f
            )
    if sol.get("equilibrated") is False:
        warn.insert(0, "NOT EQUILIBRATED (lp trending)")
    if sol.get("converged") is False:
        warn.append(
            "unconverged (max Rhat %.3g on %s, min ESS %.0f)"
            % (sol.get("max_rhat", np.nan), sol.get("worst_rhat_var"),
               sol.get("min_ess", np.nan))
        )  # fmt: skip
    nc = sol.get("n_chains")
    if nc is not None and nc < MIN_MODE_CHAINS:
        warn.append("mode held by only %d chain(s)" % nc)
    if sol.get("stuck_frac", 0.0) > STUCK_FRAC_MAX:
        warn.append(
            "stuck chains %.2f > %.1f" % (sol["stuck_frac"], STUCK_FRAC_MAX)
        )
    if sol.get("lp_rhat", 1.0) > LP_RHAT_MAX:
        warn.append("lp Rhat %.3g" % sol["lp_rhat"])
    unl = sol.get("unlisted_mode_params") or []
    if unl:
        warn.append(
            "POSSIBLE_UNLISTED_MODE (%s): %.1f%% of draws raw-z rejected, "
            "concentrated there -- the mode list may be incomplete"
            % (", ".join(unl), 100 * sol.get("rawz_frac", float("nan")))
        )
    elif sol.get("invalid_frac", 0.0) > INVALID_FRAC_MAX:
        warn.append("invalid draws %.1f%%" % (100 * sol["invalid_frac"]))
    re2 = sol.get("re2", 0.0)
    if not sol.get("bridge_refused") and np.isfinite(re2) and re2 > RE2_FLAG:
        warn.append("bridge re2 %.2g" % re2)
    osr = sol.get("out_scale_rhat") or {}
    for f, ks in sorted(kinds.items()):
        for k in ks:
            if k == "ANOMALY":
                continue
            if k == "AT_CAP":
                band = f.split("_")[-1]
                warn.append(
                    "AT_CAP in %s (out_scale Rhat %.3g)"
                    % (f, osr.get(band, float("nan")))
                )
            else:
                warn.append("%s in %s" % (k, f))
    for band, r in sorted(osr.items()):
        if np.isfinite(r) and r > OUT_SCALE_RHAT_MAX:
            if not any(("AT_CAP" in w and band in w) for w in warn):
                warn.append("out_scale %s Rhat %.3g" % (band, r))
    return bad, warn


# ---------------------------------------------------------------------------
# microlens-submit
# ---------------------------------------------------------------------------


def solution_id(ev, arch, mode):
    """Deterministic, unique within the submission: RMDC26_000094_1S2L_m1."""
    return "%s_%s_m%d" % (event_id(ev), arch, int(mode) + 1)


def submit_parameters(sol):
    """(parameters, parameter_uncertainties) in the challenge's names.

    ``sol["summary"]`` holds (median, minus, plus) per OUR name in user
    units, alpha already in radians.

    Each uncertainty is ONE number, max(minus, plus): the larger 68%
    half-width.  microlens-submit 0.17.9 documents an asymmetric
    ``[lower, upper]`` form, but its validator rejects any pair with
    lower > upper (``validate_parameter_uncertainties``), which a skewed
    posterior produces about half the time.  The conservative symmetric
    value passes everywhere, and the exact asymmetric interval is in the
    solution's notes (``notes_markdown``).
    """
    names = {"t0": "t_0", "u0": "u_0", "tE": "t_E", "rho": "rho",
             "piEN": "pi_E_N", "piEE": "pi_E_E"}  # fmt: skip
    if sol["arch"] == ARCH_BINARY:
        names.update({"s": "s", "q": "q", "alpha": "alpha_rad"})
    summ = sol["summary"]
    par, unc = {}, {}
    for out, ours in names.items():
        if ours not in summ:
            continue
        med, lo, hi = summ[ours]
        par[out] = float(med)
        unc[out] = float(max(lo, hi))
    for band, k in sorted(BAND_INDEX.items(), key=lambda kv: kv[1]):
        for kind, ours in (("S", "f_source"), ("B", "f_blend")):
            key = "%s_%s" % (ours, band)
            if key in summ:
                med, lo, hi = summ[key]
                par["F%d_%s" % (k, kind)] = float(med)
                unc["F%d_%s" % (k, kind)] = float(max(lo, hi))
    return par, unc


def higher_order_effects(sol):
    eff = []
    s = sol["summary"]
    if "rho" in s and sol.get("finite_source", True):
        eff.append("finite-source")
    if "pi_E_N" in s and "pi_E_E" in s:
        eff.append("parallax")
    return eff


def solution_json(sol, notes_path):
    """One microlens-submit Solution as a JSON-able dict (field names from
    microlens_submit 0.17.9 models/solution.py)."""
    par, unc = submit_parameters(sol)
    check_submitted_values(par, sol["arch"])
    bands = sorted(
        {str(BAND_INDEX[b]) for b in sol.get("bands", []) if b in BAND_INDEX},
        key=int,
    )
    eff = higher_order_effects(sol)
    if "parallax" in eff:
        # Also inside `parameters`: the validator looks for the parallax
        # effect's optional `t_ref` there, and counts it as metadata.
        par["t_ref"] = float(sol["t0_par"])
    out = {
        "solution_id": sol["solution_id"],
        "model_type": sol["arch"],
        "bands": bands,
        "higher_order_effects": eff,
        "t_ref": float(sol["t0_par"]) if "parallax" in eff else None,
        "parameters": par,
        "is_active": bool(sol["is_active"]),
        "alias": "%s mode %d" % (sol["arch"], int(sol["mode"]) + 1),
        "compute_info": sol.get("compute_info", {}),
        "notes_path": notes_path,
        "used_astrometry": False,
        "used_postage_stamps": False,
        "limb_darkening_model": "none (u1 = 0 in every band)",
        # keyed by band, a list of floats per band (models/solution.py's
        # own check; a "u_0": 0.0 form warns "missing bands")
        "limb_darkening_coeffs": {b: [0.0] for b in bands},
        "parameter_uncertainties": unc,
        "uncertainty_method": "mcmc_posterior",
        "confidence_level": 0.68,
        "physical_parameters": None,
        "log_likelihood": float(sol["log_likelihood"]),
        "relative_probability": float(sol["relative_probability"]),
        "n_data_points": int(sol["n_data_points"]),
    }
    # FULL tier only: the caller sets `physical_submit` (submit_physical);
    # a SCORED or legacy solution never carries the block.
    phys = sol.get("physical_submit")
    if phys:
        out["physical_parameters"] = dict(phys[0])
        out["physical_parameter_uncertainties"] = dict(phys[1])
    return out


def event_json(ev):
    return {"event_id": event_id(ev)}


def notes_markdown(sol, event):
    """The solution's notes (microlens-submit keeps notes in a Markdown file
    named by notes_path; there is no inline notes field)."""
    L = ["# %s" % sol["solution_id"], ""]
    if sol["unreliable"]:
        L.append("**UNRELIABLE:** " + "; ".join(sol["unreliable"]))
        L.append("")
    L.append(
        "Warnings: "
        + ("; ".join(sol["warnings"]) if sol["warnings"] else "none")
    )
    L.append("")
    L.append("## Parameters (posterior median, 68% interval)")
    L.append("")
    L.append("| parameter | median | minus | plus |")
    L.append("|---|---|---|---|")
    for k, v in sorted(sol.get("summary", {}).items()):
        L.append("| %s | %.10g | %.4g | %.4g |" % (k, v[0], v[1], v[2]))
    L.append("")
    L.append(
        "parameter_uncertainties in the solution file are max(minus, plus); "
        "the table above is the exact interval."
    )
    L.append("")
    L.append("## Evidence")
    L.append("")
    L.append(
        "lnZ = %.2f +/- %.2f (bridge) +/- %.1f (systematic floor), "
        "normalized (System.evidence_log_norm), bridge re2 %.3g, "
        "%d posterior draws in this mode."
        % (sol["lnZ"], sol["lnZ_err"], SYSTEMATIC_FLOOR, sol["re2"],
           sol["n_draws"])
    )  # fmt: skip
    L.append(
        "relative_probability %.4g (raw softmax %.4g; indistinguishable "
        "group %s: %s)."
        % (sol["relative_probability"], sol["p_raw"], sol["group"],
           ", ".join(sol["group_members"]))
    )  # fmt: skip
    L.append("")
    L.append("Architecture probabilities for this event (sum over solutions):")
    for arch, p in sorted(event["arch_prob"].items()):
        L.append("- %s: %.4g" % (arch, p))
    for m in event.get("missing", []):
        L.append("- %s" % m)
    L.append("")
    L.append(
        "log_likelihood is the hogg-mixture data log-likelihood at this "
        "mode's highest-posterior draw (Gaussian logL at the same point: "
        "%.2f)." % sol.get("gauss_loglike", float("nan"))
    )
    L.append("")
    L.append("## Conventions")
    L.append("")
    L.append(CONVENTIONS)
    return "\n".join(L) + "\n"


def dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return sorted(o)
        raise TypeError(type(o))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=conv) + "\n")


# ---------------------------------------------------------------------------
# The 2026-10 submission: SED-tied fits named by the campaign's MANIFEST
# ---------------------------------------------------------------------------
#
# The campaign agent's sed/make_manifest.py writes SUBMISSION_MANIFEST.txt
# (one block per event: tier, arm, out/ or out_long/, absolute paths), and
# sed/verdict.py writes the two tier lists.  Everything below is a plain
# function of that text and of numbers, so a regenerated manifest needs no
# code change, and tests/test_rmdc26_report.py drives it on synthetic input.

MANIFEST = CAMPAIGN / "SUBMISSION_MANIFEST.txt"
TIER_FULL_FILE = CAMPAIGN / "sed" / "tier_full.txt"
TIER_SCORED_FILE = CAMPAIGN / "sed" / "tier_scored.txt"
VERDICT_FILE = CAMPAIGN / "sed" / "verdict.out"
GRID_DCHI2_FILE = CAMPAIGN / "grid" / "dchi2_noplanet.json"
#: Direct single-lens-vs-binary comparisons the campaign ran where the grid
#: was not decisive (arch105/arch_direct.json: both arms at their own MAP,
#: chi2 on the data file's errors).  A direct comparison SUPERSEDES the
#: grid: it is the test the grid only approximates.
DIRECT_ARCH_GLOB = "arch*/arch_direct.json"
#: sed/make_notes.py's generated per-event notes.  Each event's notes_path
#: file is that note VERBATIM plus one appended section of ours
#: (event_notes_markdown); their file is never patched.
CAMPAIGN_NOTES_DIR = CAMPAIGN / "sed" / "notes"

#: FULL = required fields + the optional physical_parameters block;
#: SCORED = required fields only.  Anything else is not submitted.
TIER_FULL = "FULL"
TIER_SCORED = "SCORED"
#: microlens-submit's challenge tier (tier_validation.TIER_DEFINITIONS);
#: the RMDC26 data are the beginner tier.
CHALLENGE_TIER = "beginner"
#: The PI supplies the team name; until then this is the clearly marked
#: placeholder (it is NOT a valid name and must be replaced).
TEAM_NAME_PLACEHOLDER = "TEAM_NAME_TBD"

#: Architecture rule (PI-approved; decided by Cannon Dev as owner).  The
#: fits are one architecture per event, so for a 1S2L event the blind (s, q)
#: grid's dchi2_np = chi2(no-planet row) - chi2(best node) stands in for the
#: missing single-lens arm: above this, P(1S1L) ~ 0.  The campaign's grid
#: dchi2 is bimodal (nothing between 10 and 100 on 2026-10-24), so the exact
#: value of the cut does not move any event.
ARCH_DCHI2_MIN = 50.0
#: A posterior "pins" q when its larger 68% half-width is below this
#: fraction of the median (105: 2.6e-6 / 2.57e-5 = 10%).
Q_PINNED_REL = 0.25
#: The extinction-law caveat is written only above this max Rhat: the
#: campaign agent measured that every tied fit with max Rhat < 1.3 (29/29)
#: recovers the stated A_F146 within 3x (sed/make_long2.py), and withdrew
#: the caveat for those.
EXTINCTION_RHAT = 1.3

ARCH_UNCERTAIN = "ARCHITECTURE_UNCERTAIN"
ARCH_ANOMALY = "ANOMALY_POSSIBLE_COMPANION"

#: The alpha statement, verbatim as ruled (JDE: keep the VBM convention;
#: the organizers score alpha in whatever convention is stated).  The
#: numbers: degrees -> radians, no shift, wrapped to [0, 2 pi), never
#: negated (alpha_submit).
ALPHA_SENTENCE = (
    "alpha is the angle from the primary-to-secondary axis to the direction "
    "of LENS motion relative to the source (Skowron et al. 2011 A.4/A.6; "
    "identical to MulensModel and VBMicrolensing), counter-clockwise N->E, "
    "converted from degrees to radians with no other transformation. If "
    "your alpha is measured to the SOURCE trajectory instead, add pi."
)

#: Short form of the PI's conventions, one bullet each, in every note.
CONVENTION_BULLETS = (
    ALPHA_SENTENCE,
    "piEN/piEE are in the VBM / conventions.md frame (geocentric, C8-C10), "
    "anchored at t_ref = t0_par.",
    "Fluxes F0/F1/F2 _S/_B are F = 10**(-0.4 m_AB) on the challenge's AB "
    "magnitudes (zero point 0); 0 = F146, 1 = F087, 2 = F213.",
    "Zero limb darkening (u1 pinned at 0 in every band).",
    "t0 is the FULL BJD of the data's `bjd` column (e.g. 2463102.43884), "
    "with no offset subtracted.",
    "s, rho and u0 are in units of theta_E.",
)

#: Every submitted t0 is a FULL BJD; anything below this means an offset
#: was subtracted somewhere, and the writer refuses it.
T0_MIN = 2.4e6


def alpha_submit(alpha_deg):
    """The submitted alpha: degrees -> radians and NOTHING else (JDE ruled
    the VBM convention: no C21 +180, no sign flip, no C19 q-inversion
    shift), wrapped to [0, 2 pi)."""
    return np.mod(np.deg2rad(np.asarray(alpha_deg, dtype=float)), 2 * math.pi)


def check_submitted_values(par, arch):
    """Loud guards on one solution's `parameters` before it is written:
    t0 is a full BJD (> T0_MIN) and a 1S2L alpha lies in [0, 2 pi)."""
    if not par.get("t0", 0.0) > T0_MIN:
        raise ValueError("t0 %r is not a full BJD (> %g): an offset was "
                         "subtracted" % (par.get("t0"), T0_MIN))  # fmt: skip
    if arch == ARCH_BINARY and not 0.0 <= par["alpha"] < 2 * math.pi:
        raise ValueError("alpha %r rad outside [0, 2 pi)" % par["alpha"])


_MAN_HEAD = re.compile(
    r"^(\d+)\s+(FULL|SCORED)\s+"
    r"(?:arm=(\S+)\s+dir=(\S+)|\*\*\*\s*NO RESULTS\s*\*\*\*)"
)
_MAN_PATH = re.compile(
    r"^\s+(trace|results|modes|summary|config|params)\s+(\S+)"
    r"(\s+\[MISSING\])?\s*$"
)
_MAN_VERDICT = re.compile(r"^\s+verdict\s+(.*?)\s*$")


def parse_manifest(text):
    """The blocks of SUBMISSION_MANIFEST.txt, in file order.

    Each is {event, tier, arm, dir, paths: {trace, results, modes, summary,
    config, params}, missing: [keys marked MISSING], verdict: str|None},
    with ``no_results: True`` (and arm/dir None) for a "*** NO RESULTS ***"
    block.  An event listed twice is an error: the manifest is the single
    list of what is submitted.
    """
    out, cur, seen = [], None, set()
    for line in text.splitlines():
        m = _MAN_HEAD.match(line)
        if m:
            ev = int(m.group(1))
            if ev in seen:
                raise ValueError("manifest lists event %03d twice" % ev)
            seen.add(ev)
            cur = {"event": ev, "tier": m.group(2), "arm": m.group(3),
                   "dir": m.group(4), "paths": {}, "missing": [],
                   "verdict": None}  # fmt: skip
            if m.group(3) is None:
                cur["no_results"] = True
            out.append(cur)
            continue
        if cur is None:
            continue
        m = _MAN_PATH.match(line)
        if m:
            cur["paths"][m.group(1)] = m.group(2)
            if m.group(3):
                cur["missing"].append(m.group(1))
            continue
        m = _MAN_VERDICT.match(line)
        if m:
            cur["verdict"] = m.group(1)
    return out


def read_event_list(text):
    """Event numbers from a one-per-line list (tier_full.txt; # comments)."""
    evs = []
    for line in text.splitlines():
        tok = line.split("#")[0].split()
        if tok:
            evs.append(int(tok[0]))
    return evs


def select_submitted(entries, full, scored):
    """(submitted entries, problems): the TIER FILTER.

    The tier lists are authoritative: an event is submitted only when it is
    in tier_full.txt (FULL) or tier_scored.txt (SCORED) AND the manifest has
    results for it.  Everything else is "not ok" and is not submitted.  A
    manifest tier that disagrees with the lists, an event in both lists, a
    listed event missing from the manifest, and a NO RESULTS block are
    reported as problems (never silently dropped)."""
    full, scored = set(map(int, full)), set(map(int, scored))
    problems = []
    for ev in sorted(full & scored):
        problems.append("%03d is in BOTH tier lists: not submitted" % ev)
    sub = []
    in_man = set()
    for e in entries:
        ev = e["event"]
        in_man.add(ev)
        if ev in full and ev in scored:
            continue
        tier = (TIER_FULL if ev in full else
                TIER_SCORED if ev in scored else None)  # fmt: skip
        if tier is None:
            problems.append(
                "%03d is in the manifest (%s) but in neither tier list: "
                "not submitted" % (ev, e["tier"])
            )
            continue
        if e.get("no_results"):
            problems.append("%03d (%s): manifest has NO RESULTS" % (ev, tier))
            continue
        if e["tier"] != tier:
            problems.append(
                "%03d: manifest says %s, tier lists say %s (lists win)"
                % (ev, e["tier"], tier)
            )
        sub.append(dict(e, tier=tier))
    for ev in sorted((full | scored) - in_man):
        problems.append("%03d is in a tier list but not in the manifest" % ev)
    return sub, problems


def prefer_out_long(manifest_dir, probe):
    """(run dir, note): ALWAYS out_long/ when it holds a usable run.

    ``probe(name)`` describes ``<arm>/<ev>/<name>/``: {"trace": bool,
    "results": bool, "stale": str} (stale is staleness_from_mtimes' reason,
    "" when the trace belongs to its files).  out_long/ is used whenever its
    trace and results exist and it is not stale -- whatever the manifest
    says, since the long rerun is better converged and is what the verdict
    was computed from.  A long rerun still in flight (its whitening written,
    no trace yet; or a rerun overwriting an older out_long trace) is not
    usable, and out/ is used with a note.  (None, why) if neither is."""
    notes = []
    ol = probe("out_long")
    if ol["trace"] and ol["results"] and not ol["stale"]:
        if manifest_dir != "out_long":
            notes.append(
                "the manifest named %s/, but out_long/ holds a finished run "
                "and is used" % manifest_dir
            )
        return "out_long", "; ".join(notes)
    if ol["trace"] or ol["results"] or ol["stale"]:
        notes.append(
            "out_long/ exists but is not usable (%s)"
            % (
                ol["stale"]
                or ("no trace" if not ol["trace"] else "no results")
            )
        )
    o = probe("out")
    if o["trace"] and o["results"] and not o["stale"]:
        return "out", "; ".join(notes)
    notes.append(
        "out/ not usable (%s)"
        % (o["stale"] or ("no trace" if not o["trace"] else "no results"))
    )
    return None, "; ".join(notes)


def pick_config(cands, run_dir, ev, trace_mtime, manifest_config=None):
    """The config that wrote ``<run_dir>/<ev>_trace.nc``, or None.

    ``cands`` is [(path, prefix, mtime)] for the arm directory's run*.yaml.
    A config belongs to the trace when its ``prefix:`` is ``<run_dir>/<ev>``
    and it is not newer than the trace (a newer one is a later rerun's).
    The manifest's own choice wins when it qualifies; otherwise the newest
    qualifying one (run_long.yaml vs run_long2.yaml both write out_long/:
    make_manifest.py names only the first)."""
    want = "%s/%s" % (run_dir, ev3(ev))
    ok = [c for c in cands
          if str(c[1]).rstrip("/") == want and c[2] <= trace_mtime + 1]  # fmt: skip
    for c in ok:
        if manifest_config and str(c[0]) == str(manifest_config):
            return c[0]
    return max(ok, key=lambda c: c[2])[0] if ok else None


#: verdict.out's header before the campaign added the modes/stuck columns
#: (2026-10-24); used when the live file cannot be read.
VERDICT_HEADER_LEGACY = (
    "ev    LC verdict                         PHY verdict                    "
    "  Rhat    chi2    ratio  outfrc"
)
#: verdict.out column name -> parse_verdict_line key.
VERDICT_NUM_KEYS = {"Rhat": "rhat", "chi2": "chi2", "ratio": "ratio",
                    "outfrc": "outfrac", "modes": "modes", "stuck": "stuck"}  # fmt: skip


def verdict_header(path=None):
    """verdict.out's header line (the manifest copies rows without it)."""
    try:
        with open(path or VERDICT_FILE) as f:
            return f.readline().rstrip("\n")
    except OSError:
        return VERDICT_HEADER_LEGACY


def parse_verdict_line(line, header=None):
    """{lc, phy, rhat, chi2, ratio, outfrac, ...} from one sed/verdict.out
    row, by the HEADER's columns (``header`` defaults to the live file's).

    The campaign adds columns (modes, stuck on 2026-10-24), and the LC
    verdict can contain spaces ("modes stuck 2/98%"), so neither "the last
    four tokens" nor "token 1" is safe: the LC verdict is the row's text
    under the header's LC..PHY span, the numbers are the last N tokens for
    the N header names after "PHY verdict", and the PHY verdict is what is
    left between (free text, it may overflow its column)."""
    if not line:
        return {}
    header = header if header is not None else verdict_header()
    tok = line.split()
    if not tok or not tok[0].isdigit():
        return {}
    i_lc, i_phy = header.find("LC"), header.find("PHY")
    names = header[i_phy:].split()[2:] if i_phy >= 0 else []
    n = len(names)
    if i_lc < 0 or i_phy < 0 or n == 0 or len(tok) < 2 + n:
        return {}

    def num(x):
        try:
            return float(x)
        except ValueError:
            return float("nan")

    if line[i_phy - 1 : i_phy] not in (" ", ""):
        return {}  # the LC verdict overflowed its column: refuse to guess
    rest = line[i_phy:].split()
    out = {"lc": line[i_lc:i_phy].strip(), "phy": " ".join(rest[:-n])}
    for name, x in zip(names, rest[-n:]):
        out[VERDICT_NUM_KEYS.get(name, name)] = num(x)
    return out


def tier_reason(tier, verdict):
    """Why the event has its tier, from its verdict row."""
    v = verdict or {}
    lc, phy = v.get("lc", "?"), v.get("phy", "?")
    if tier == TIER_FULL:
        return (
            "FULL: the light-curve verdict (%s) and the physical-parameter "
            "verdict (%s) both passed (sed/verdict.py), so the optional "
            "physical_parameters block is submitted" % (lc, phy)
        )
    return (
        "SCORED: the light-curve verdict passed (%s) but the physical-"
        "parameter verdict did not (%s), so only the required fields are "
        "submitted and physical_parameters is withheld" % (lc, phy)
    )


def load_direct_arch(campaign=None):
    """{'105': {dchi2, n, k_extra, ...}} merged from every DIRECT_ARCH_GLOB
    file under the campaign directory ({} when there are none)."""
    out = {}
    for f in sorted(Path(campaign or CAMPAIGN).glob(DIRECT_ARCH_GLOB)):
        out.update(json.loads(f.read_text()))
    return out


def architecture_verdict(arch, dchi2_np=None, q=None, anomaly_files=(),
                         threshold=ARCH_DCHI2_MIN, q_pinned=Q_PINNED_REL,
                         direct=None):  # fmt: skip
    """The event's architecture statement (PI-approved rule).

    ``direct`` (a load_direct_arch record: dchi2 = chi2(1L1S) - chi2(2L1S)
    at each arm's MAP, n epochs, k_extra parameters) supersedes the grid for
    a 1S2L event: the same threshold applies to it, and the grid value is
    quoted only for the record.

    1S2L: from the blind (s, q) grid's dchi2_np = chi2(no-planet row) -
      chi2(best node).  dchi2_np > threshold: "binary lens favoured", P(1S1L)
      ~ 0, no flag.  Otherwise, or with no grid, ARCHITECTURE_UNCERTAIN; if
      the posterior nevertheless pins q (``q`` = (median, minus, plus) with
      max half-width / median < q_pinned) the tension is spelled out.
    1S1L: the dchi2 rule does not apply (single lenses were not gridded);
      the verdict is the outlier audit's.  ANOMALY in any band means the
      mixture absorbs points AT the event: a possible missed companion.

    Returns {flag (str|None), statement, dchi2_np, rule}.
    """
    if (
        arch == ARCH_BINARY
        and direct
        and np.isfinite(float(direct.get("dchi2", np.nan)))
    ):
        dd = float(direct["dchi2"])
        grid = (
            "" if dchi2_np is None
            else " (the blind grid gave dchi2_np = %.1f; the direct fit "
            "supersedes it)" % float(dchi2_np)
        )  # fmt: skip
        how = (
            "a direct single-lens fit to the same %s epochs, each arm at its "
            "own MAP with chi2 on the data file's errors, is worse than the "
            "binary by dchi2 = %.1f for %s extra parameters%s"
            % (direct.get("n", "?"), dd, direct.get("k_extra", "?"), grid)
        )
        if dd > threshold:
            return {"flag": None, "dchi2_np": dchi2_np, "rule": "direct",
                    "dchi2_direct": dd,
                    "statement": "binary lens favoured: %s (> %g), so "
                    "P(1S1L) ~ 0; relative_probability is normalized over "
                    "this event's 1S2L solutions." % (how, threshold)}  # fmt: skip
        return {"flag": ARCH_UNCERTAIN, "dchi2_np": dchi2_np,
                "rule": "direct", "dchi2_direct": dd,
                "statement": "ARCHITECTURE_UNCERTAIN: %s (<= %g), so a "
                "single lens is not excluded; relative_probability is "
                "conditional on 1S2L." % (how, threshold)}  # fmt: skip
    if arch == ARCH_BINARY:
        d = None if dchi2_np is None else float(dchi2_np)
        if d is None or not np.isfinite(d):
            return {
                "flag": ARCH_UNCERTAIN, "dchi2_np": None, "rule": "no-grid",
                "statement": "ARCHITECTURE_UNCERTAIN: no blind (s, q) grid "
                "result for this event, so the binary lens is not tested "
                "against a single lens; relative_probability is conditional "
                "on 1S2L.",
            }  # fmt: skip
        if d > threshold:
            return {
                "flag": None, "dchi2_np": d, "rule": "grid",
                "statement": "binary lens favoured by dchi2_np = %.1f on the "
                "blind grid (> %g), so P(1S1L) ~ 0; relative_probability is "
                "normalized over this event's 1S2L solutions."
                % (d, threshold),
            }  # fmt: skip
        st = (
            "ARCHITECTURE_UNCERTAIN: the blind (s, q) grid prefers a "
            "companion by only dchi2_np = %.1f (<= %g), so a single lens is "
            "not excluded; relative_probability is conditional on 1S2L."
            % (d, threshold)
        )
        if q is not None:
            med, lo, hi = (float(x) for x in q)
            rel = max(lo, hi) / abs(med) if med else float("inf")
            if np.isfinite(rel) and rel < q_pinned:
                st += (
                    " The blind grid prefers a companion by only dchi2 = "
                    "%.0f while the posterior pins q to %.0f%%; the coarse "
                    "grid can under-resolve a narrow solution, or the fit "
                    "may be overconfident; unresolved." % (d, 100 * rel)
                )
        return {"flag": ARCH_UNCERTAIN, "dchi2_np": d, "rule": "grid",
                "statement": st}  # fmt: skip
    anomaly_files = sorted(set(anomaly_files))
    base = (
        "Single lens (1S1L). Single-lens events were not gridded, so the "
        "blind-grid dchi2 rule does not apply; the architecture rests on "
        "the fit and the outlier audit."
    )
    if anomaly_files:
        return {
            "flag": ARCH_ANOMALY, "dchi2_np": None, "rule": "audit",
            "statement": base + " The outlier audit flags ANOMALY in %s: "
            "the mixture absorbs points at the event, so a companion may "
            "have been missed." % ", ".join(anomaly_files),
        }  # fmt: skip
    return {"flag": None, "dchi2_np": None, "rule": "audit",
            "statement": base + " The audit finds no ANOMALY at the event."}  # fmt: skip


def extinction_caveat(max_rhat, limit=EXTINCTION_RHAT):
    """The extinction-law systematic caveat, only for an unconverged fit
    (max Rhat > limit), else None.  Converged tied fits recover the stated
    extinction (29/29 below 1.3), so it was withdrawn for them."""
    if max_rhat is None or not np.isfinite(max_rhat) or max_rhat <= limit:
        return None
    return (
        "Extinction caveat: this fit has max Rhat %.3f > %.1f. Unconverged "
        "SED-tied fits often miss the challenge's stated extinction (above "
        "Rhat 1.3 only 41%% recover A_F146 within 3x, against 29/29 below "
        "it), and an extinction-law mismatch is the suspected systematic; "
        "the source and lens physical parameters may carry it."
        % (max_rhat, limit)
    )


def body_indices(config):
    """(lens primary index, source index) into the star vector, from the
    config's lens/source `body:` references (star.Lens -> 'Lens')."""
    names = [s["name"] for s in config.get("star") or []]

    def idx(block):
        for b in config.get(block) or []:
            body = str(b.get("body", ""))
            if body.startswith("star."):
                return names.index(body.split(".", 1)[1])
        raise ValueError("no star body in %s: %s" % (block, config.get(block)))

    return idx("lens"), idx("source")


#: The trace variables physical_draws reads (stored in user units: solMass,
#: pc, mas, mas/yr).
PHYSICAL_VARS = (
    "mulensevent.theta_E", "star.mass", "mulensevent.mlens_total",
    "star.distance", "mulensevent.pi_E_N", "mulensevent.pi_E_E",
    "mulensevent.mu_rel_mag", "mulensevent.mu_rel_geo_mag",
)  # fmt: skip


def physical_draws(d, arch, lens_idx, src_idx):
    """microlens-submit physical parameters, PER DRAW.

    ``d`` maps a trace variable to its draws for ONE mode, shape (n,) or
    (n, k).  Every output is formed draw by draw (never a product of
    medians), so its median and interval are the posterior's:

      thetaE [mas]   mulensevent.theta_E
      M1 [Msun]      star.mass[lens primary]
      Mtot [Msun]    mulensevent.mlens_total (1S2L) or M1 (1S1L)
      M2 [Msun]      Mtot - M1 (1S2L only; = q M1, with no planet-mass unit)
      D_L, D_S [kpc] star.distance[lens], [source] / 1000 (pc in the trace)
      piE            hypot(pi_E_N, pi_E_E)
      mu_rel [mas/yr] mulensevent.mu_rel_mag (heliocentric)

    Keys are restricted to validate_parameters.PHYSICAL_PARAMETERS.  The
    geocentric |mu_rel| (= thetaE / tE) is returned as ``mu_rel_geo`` for
    the notes only (not a schema key)."""

    def col(name, k=0):
        a = np.asarray(d[name], dtype=float)
        return a if a.ndim == 1 else a[:, k]

    m1 = col("star.mass", lens_idx)
    out = {"thetaE": col("mulensevent.theta_E")}
    if arch == ARCH_BINARY:
        mtot = col("mulensevent.mlens_total")
        out.update(Mtot=mtot, M1=m1, M2=mtot - m1)
    else:
        out["Mtot"] = m1
    out["D_L"] = col("star.distance", lens_idx) / 1000.0
    out["D_S"] = col("star.distance", src_idx) / 1000.0
    if "mulensevent.pi_E_N" in d and "mulensevent.pi_E_E" in d:
        out["piE"] = np.hypot(col("mulensevent.pi_E_N"),
                              col("mulensevent.pi_E_E"))  # fmt: skip
    if "mulensevent.mu_rel_mag" in d:
        out["mu_rel"] = col("mulensevent.mu_rel_mag")
    if "mulensevent.mu_rel_geo_mag" in d:
        out["mu_rel_geo"] = col("mulensevent.mu_rel_geo_mag")
    return out


#: Notes-only keys of physical_draws (not microlens-submit fields).
PHYSICAL_NOTES_ONLY = ("mu_rel_geo",)


def physical_summary(draws):
    """{key: (median, minus, plus)} of physical_draws' output."""
    return {k: percentiles(v) for k, v in draws.items()}


def submit_physical(summary):
    """(physical_parameters, physical_parameter_uncertainties) for the JSON:
    medians and max(minus, plus), schema keys only."""
    par, unc = {}, {}
    for k, (med, lo, hi) in sorted(summary.items()):
        if k in PHYSICAL_NOTES_ONLY or not np.isfinite(med):
            continue
        par[k] = float(med)
        unc[k] = float(max(lo, hi))
    return par, unc


#: Event-specific findings appended to the architecture statement.  (105's
#: grid caveat was retired 2026-10-24: its direct PSPL arm, job 55068758,
#: settles it through DIRECT_ARCH_GLOB.)
ARCH_EVENT_NOTES = {}

#: The heading of the one section this tool appends to the campaign's note.
NOTES_SECTION = "## Evidence, flags and conventions (exozippy report stage)"


def solution_flags(sol):
    """Short machine flags of one solution: UNRELIABLE / INDICATIVE /
    WARN(n)."""
    f = []
    if sol.get("unreliable"):
        f.append("UNRELIABLE")
    if sol.get("basis_indicative"):
        f.append("INDICATIVE")
    if sol.get("warnings"):
        f.append("WARN(%d)" % len(sol["warnings"]))
    return f


def event_notes_markdown(campaign_note, event):
    """The event's notes_path file: the campaign's generated note VERBATIM,
    then ONE appended machine-generated section with only what that note
    lacks -- per-mode lnZ (bridge error, the systematic floor, bridge
    status), relative_probability, the reliability flags, the architecture
    statement and the conventions block.  No other prose."""
    L = []
    if campaign_note:
        L.append(campaign_note.rstrip("\n"))
        L.append("")
    L.append(NOTES_SECTION)
    L.append("")
    L.append("Event flags: " + (", ".join(event.get("flags") or []) or "none"))
    L.append("")
    L.append("### Per-mode evidence and relative probability")
    L.append("")
    L.append(
        "lnZ is normalized (System.evidence_log_norm), from valid draws "
        "only with %d proposal draws; sigma_tot = sqrt(err_bridge**2 + "
        "%.0f**2) (systematic floor); modes within sigma_tot of a group "
        "leader share its probability equally. Probability basis: %s."
        % (MAX_POSTERIOR_DRAWS, SYSTEMATIC_FLOOR, event.get("basis", "?"))
    )
    L.append("")
    L.append("| solution | lnZ | err_bridge | sigma_tot | bridge | group "
             "| relative_probability | active | flags |")  # fmt: skip
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in event["solutions"]:
        st = s.get("bridge_state", "?")
        if s.get("bridge_refused"):
            st = "REFUSED (%s): lnZ indicative only" % st
        L.append(
            "| %s | %.2f | %.2f | %.2f | %s | %s | %.4g | %s | %s |"
            % (s["solution_id"], s["lnZ"], s["lnZ_err"],
               s.get("sigma_total", float("nan")), st, s.get("group", "-"),
               s["relative_probability"], "yes" if s["is_active"] else "no",
               ", ".join(solution_flags(s)) or "-")
        )  # fmt: skip
    L.append("")
    L.append("### Reliability flags")
    L.append("")
    for s in event["solutions"]:
        L.append("- %s: UNRELIABLE: %s. Warnings: %s." % (
            s["solution_id"], "; ".join(s.get("unreliable") or []) or "no",
            "; ".join(s.get("warnings") or []) or "none"))  # fmt: skip
    cav = event.get("extinction_caveat")
    if cav:
        L.append("- " + cav)
    L.append("")
    L.append("### Architecture")
    L.append("")
    L.append(event["architecture"]["statement"])
    L.append("")
    L.append("### Conventions")
    L.append("")
    for b in CONVENTION_BULLETS:
        L.append("- " + b)
    return "\n".join(L) + "\n"
