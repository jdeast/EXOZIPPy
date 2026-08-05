"""Data-driven whitening: measure each raw parameter's true local scale and
rescale the model's whitening in place.

The model is first built with PRELIMINARY whitening scales (defaults.yaml
init_scale, or a fraction of the bound span when absent -- see
Parameter.build_pymc).  probe_scales() then measures, per raw element, the
step in the current raw coordinate at which logp falls 0.5 nats below the
start (the EXOFASTv2 exofast_getmcmcscale convention: dlogp=0.5 <-> dchi2=1,
the 1-sigma step for a locally Gaussian posterior).  Because one raw unit is
one preliminary scale by construction, the measured step IS the multiplicative
correction to the preliminary scale.  apply_measured_whitening() feeds it to
Parameter.set_whitening(), which updates the pytensor.shared whitening scales
in place -- no model rebuild, and the posterior is provably unchanged (the
logit-uniform correction potential cancels the raw N(0,1) prior symbolically
for any scale; the scale affects conditioning only).

After the rescale, every raw direction costs ~0.5 nats per unit step: the
"curvature = -1" conditioning the retired curvature check used to ask users
to approximate by hand-tuning init_scale, reached here without iteration.

The probe originated in the PTDE sampler's chain-initialization
(samplers/ptde.py imports it from here); the whitening pass uses a wider
dynamic range, since a preliminary scale can be off by many orders of
magnitude while PTDE re-probes an already-whitened model where every scale
is ~1.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# target = 0.5 nats (dlogp=0.5 <-> dchi2=1, the EXOFASTv2 convention); for a
# locally Gaussian posterior that step IS the 1-sigma width along the element.
PROBE_TARGET_DELTA = 0.5
_PROBE_MAX_STEP = 1.0e3  # raw units; beyond this a direction is "flat"
_PROBE_MIN_STEP = 1.0e-8  # raw units; floor on a returned step
_PROBE_RTOL = 0.05  # accept a step whose drop is within 5% of target
_PROBE_MAX_BISECT = 40
_PROBE_FLAT_SCALE = 1.0  # fallback when neither direction reaches target;
# raw variables are N(0,1) by construction, so 1.0
# is the natural unit for a direction logp ignores.

# The whitening pass widens the probe's dynamic range: a preliminary scale
# (defaults.yaml, or the span-fraction fallback) can be off by 10+ orders of
# magnitude in either direction, where PTDE's defaults -- tuned for a model
# that is already whitened -- would hit the 1e-8 step floor or declare a
# merely-distant contour "flat".  Bracketing is geometric and bisection
# halves, so the wider range costs only tens of extra logp calls per element,
# and only for elements whose preliminary scale was badly off.
_WHITEN_MAX_STEP = 1.0e9
_WHITEN_MIN_STEP = 1.0e-14
_WHITEN_MAX_BISECT = 90


def probe_step_1d(
    eval_delta,
    sign,
    target=PROBE_TARGET_DELTA,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    rtol=_PROBE_RTOL,
    max_bisect=_PROBE_MAX_BISECT,
):
    """Step magnitude in one direction whose logp drop reaches the target.

    `eval_delta(step)` returns logp(start) - logp(start + step), so it is 0 at
    step=0 and grows as the step walks logp downhill.  Returns None when the
    direction is flat out to max_step.

    Brackets by geometric growth, then bisects.  delta may start NEGATIVE --
    raw_start is initval, not the MAP, so one direction is typically uphill --
    and growing through the turnover still lands on the step where logp has
    fallen `target` below the START, which is the quantity being asked for.
    """
    # BRACKET: lo always has delta < target (delta(0) == 0); hi is the first
    # step at/above target, or non-finite (a hard prior wall counts as "past").
    lo, hi = 0.0, None
    step = 1.0
    while step <= max_step:
        d = eval_delta(sign * step)
        if (not np.isfinite(d)) or d >= target:
            hi = step
            break
        lo = step
        step *= 2.0
    if hi is None:
        return None

    # BISECT.
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        d = eval_delta(sign * mid)
        if np.isfinite(d) and abs(d - target) <= rtol * target:
            return mid
        if (not np.isfinite(d)) or d > target:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < min_step:
            break
    return max(0.5 * (lo + hi), min_step)


def probe_scales(
    raw_start,
    logp_fn,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    max_bisect=_PROBE_MAX_BISECT,
    flat_scale=_PROBE_FLAT_SCALE,
):
    """Adaptive probe: per-element scale whose raw step costs ~0.5 nats.

    Follows EXOFASTv2's exofast_getmcmcscale in searching each sign
    independently for the step where logp falls PROBE_TARGET_DELTA below the
    start, but takes the NEARER of the two contours rather than their average
    (a single direction is used when the other is flat; _PROBE_FLAT_SCALE when
    both are).  EXOFASTv2 averages because it probes from an AMOEBA best fit,
    where the two directions are near-symmetric.  Our seeds are not at the
    mode -- an mkprior seed for examples/ogle0383 probes from ~41 nats below
    the MAP -- and off the mode the two directions are wildly asymmetric: the
    uphill one only turns over on the far side of the mode, so averaging it in
    inflates the scale (u1_raw: 1.2 nearer contour, ~5.7 far side, 3.4
    averaged).  Chains then jitter past the logit transform's saturation walls
    and start pinned at the bounds, which is the pathology this scale exists to
    avoid.  The nearer contour is a well-defined local width, is exactly sigma
    at a mode (where both contours coincide), and only ever errs tight -- which
    downstream consumers absorb (chain-scatter multipliers, DE's own ensemble
    expansion, and NUTS mass-matrix adaptation).

    The step is found by bracket + bisect.  An earlier version instead walked a
    fixed probe ladder [0.003 ... 2.0], took the FIRST magnitude registering any
    drop at all, and extrapolated with a quadratic assumption,
    scale = |dp| * sqrt(target/delta).  That assumption only holds AT a maximum.
    raw_start is initval, not the MAP, so logp is generally LINEAR in a probe
    around it -- doubling the probe doubles the drop (measured ratio 2.00, not
    the 4.0 a quadratic implies) -- and for a linear response the formula
    collapses to sqrt(0.5*|dp|/g), which never converges on the true step; it
    just tracks the smallest rung of the ladder.  On examples/ogle0383 that
    returned 0.048 where the true 0.5-nat step is 0.566, under-dispersing every
    chain by 12-18x and leaving the ensemble still equilibrating well into the
    recorded draws.

    Returns (map_lp, scales) where scales maps each raw-start key to an array
    shaped like its entry.
    """
    map_lp = float(logp_fn(raw_start))

    def _delta_at(key, i, step):
        probe = {k: v.copy() for k, v in raw_start.items()}
        probe[key].flat[i] += step
        return map_lp - float(logp_fn(probe))

    scales = {}
    tight = []
    flat = []
    for key, val in raw_start.items():
        n = val.size
        s = np.full(n, flat_scale)
        for i in range(n):

            def eval_delta(step, key=key, i=i):
                return _delta_at(key, i, step)

            steps = [
                probe_step_1d(
                    eval_delta,
                    sign,
                    max_step=max_step,
                    min_step=min_step,
                    max_bisect=max_bisect,
                )
                for sign in (1.0, -1.0)
            ]
            steps = [x for x in steps if x is not None]
            if steps:
                s.flat[i] = min(float(np.min(steps)), max_step)
            else:
                flat.append(f"{key}[{i}]")
            if s.flat[i] < 0.5:
                tight.append(f"{key}[{i}]: scale={s.flat[i]:.3g}")
        scales[key] = s.reshape(val.shape)

    if tight:
        logger.debug(
            "whitening probe: tightly-constrained params: " + "; ".join(tight)
        )
    if flat:
        logger.debug(
            f"whitening probe: logp flat to within {PROBE_TARGET_DELTA} nats "
            f"out to +/-{max_step:g} for: "
            + "; ".join(flat)
            + f" -- using scale={flat_scale}"
        )
    return map_lp, scales


# A first-round multiplier at/beyond these marks was CLIPPED by the probe's
# dynamic range -- the true scale was not resolved.  Escalation applies the
# clipped value (already a huge improvement) and re-probes just those
# elements in the new raw coordinates, where the residual error is within
# range again.  Two rounds cover preliminary scales off by ~28 orders of
# magnitude, far beyond any physical case.
_CLIP_LO = 2.0 * _WHITEN_MIN_STEP
_CLIP_HI = 0.5 * _WHITEN_MAX_STEP
_ESCALATION_ROUNDS = 2


def _param_for_raw(lookup, key):
    name = key[: -len("_raw")] if key.endswith("_raw") else key
    return lookup.get(name)


def _probe_selected(raw_start, logp_fn, elems):
    """Re-probe only the given (raw_name, flat_index) elements."""
    map_lp = float(logp_fn(raw_start))
    out = {}
    for key, i in elems:

        def eval_delta(step, key=key, i=i):
            probe = {k: v.copy() for k, v in raw_start.items()}
            probe[key].flat[i] += step
            return map_lp - float(logp_fn(probe))

        steps = [
            probe_step_1d(
                eval_delta,
                sign,
                max_step=_WHITEN_MAX_STEP,
                min_step=_WHITEN_MIN_STEP,
                max_bisect=_WHITEN_MAX_BISECT,
            )
            for sign in (1.0, -1.0)
        ]
        steps = [x for x in steps if x is not None]
        out[(key, i)] = (
            min(float(np.min(steps)), _WHITEN_MAX_STEP) if steps else np.nan
        )
    return out


def apply_measured_whitening(system, model, raw_start=None, logp_fn=None):
    """Measure every sampled element's true raw-space scale and rescale the
    model's whitening in place.

    Probes from the relaxation-engine start (raw_start), then routes each raw
    variable's measured 0.5-nat step to its Parameter.set_whitening(), which
    updates the shared whitening scales so the same compiled model is now
    whitened by the data-driven scales.  Elements whose probe failed (flat
    both ways) keep their preliminary scale; elements whose raw N(0,1) is the
    prior (unbounded Gaussians) are never rescaled.

    Elements whose first-round multiplier hit the probe's dynamic-range
    limits (a preliminary scale off by more than ~9-14 orders of magnitude,
    e.g. a period constrained to nanoseconds against day-scale bounds) are
    escalated: the clipped correction is applied, then just those elements
    are re-probed in the new raw coordinates, where the residual error is
    resolvable.  Anything still clipped after the escalation rounds gets a
    warning naming the element (fix its defaults.yaml init_scale).

    Returns a report dict:
      map_lp       -- logp at the start
      multipliers  -- {raw_name: array} CUMULATIVE measured step per element
                      in preliminary-scale units (~1 means the preliminary
                      scale was already right; NaN = flat direction)
      raw_scales   -- {raw_name: array} per-element scale in the FINAL raw
                      units (1.0 for rescaled elements by construction; the
                      measured value for deliberately-untouched Gaussian-
                      prior elements) -- what PTDE's chain dispersion uses
                      instead of re-probing.
    """
    if raw_start is None:
        raw_start = system.get_raw_start(model)
    if logp_fn is None:
        logp_fn = model.compile_logp()

    n_elements = sum(v.size for v in raw_start.values())
    logger.debug(
        f"Whitening: probing {n_elements} raw element(s) for their "
        f"data-driven scales..."
    )
    # flat_scale=NaN: a flat direction keeps its preliminary scale
    # (set_whitening skips non-finite entries) and stays flagged in the
    # report so the diagnostics table can warn about it.
    map_lp, scales = probe_scales(
        raw_start,
        logp_fn,
        max_step=_WHITEN_MAX_STEP,
        min_step=_WHITEN_MIN_STEP,
        max_bisect=_WHITEN_MAX_BISECT,
        flat_scale=np.nan,
    )

    lookup = {p.label: p for p in system.get_all_parameters()}
    multipliers = {
        k: np.asarray(v, dtype=float).copy() for k, v in scales.items()
    }
    raw_scales = {}
    applied = {}  # raw_name -> bool array: element was actually rescaled
    for key, mult in scales.items():
        flat_mult = np.asarray(mult, dtype=float).reshape(-1)
        par = _param_for_raw(lookup, key)
        if par is None or getattr(par, "_whiten_state", None) is None:
            logger.warning(
                f"Whitening: no parameter found for raw variable '{key}'; "
                f"its scale stays preliminary."
            )
            raw_scales[key] = np.ones_like(flat_mult).reshape(mult.shape)
            applied[key] = np.zeros(flat_mult.size, dtype=bool)
            continue
        post = par.set_whitening(flat_mult)
        raw_scales[key] = np.asarray(post, dtype=float).reshape(mult.shape)
        # applied: finite multiplier that landed as 1.0 in the new raw units
        applied[key] = np.isfinite(flat_mult) & (
            np.asarray(post, dtype=float) == 1.0
        )

    # Escalation: re-probe elements whose multiplier was clipped by the
    # probe's dynamic range.
    for round_i in range(_ESCALATION_ROUNDS):
        clipped = [
            (key, i)
            for key, mult in multipliers.items()
            for i in range(mult.size)
            if applied[key][i]
            and np.isfinite(mult.flat[i])
            and not (_CLIP_LO < abs(mult.flat[i]) < _CLIP_HI)
        ]
        if not clipped:
            break
        logger.warning(
            f"Whitening: {len(clipped)} element(s) hit the probe's dynamic "
            f"range (preliminary scale off by >9 orders of magnitude); "
            f"escalation round {round_i + 1}: "
            + "; ".join(f"{k}[{i}]" for k, i in clipped)
        )
        res = _probe_selected(raw_start, logp_fn, clipped)
        by_param = {}
        for (key, i), m in res.items():
            by_param.setdefault(key, {})[i] = m
        for key, elems in by_param.items():
            par = _param_for_raw(lookup, key)
            n = multipliers[key].size
            mult2 = np.ones(n)
            for i, m in elems.items():
                if np.isfinite(m) and m > 0:
                    mult2[i] = m
                    multipliers[key].flat[i] *= m
            par.set_whitening(mult2)

    still = [
        f"{key}[{i}]: cumulative={multipliers[key].flat[i]:.3g}"
        for key, mult in multipliers.items()
        for i in range(mult.size)
        if applied[key][i]
        and np.isfinite(mult.flat[i])
        and not (_CLIP_LO < abs(mult.flat[i]) < _CLIP_HI)
    ]
    if still:
        logger.warning(
            "Whitening: scale still unresolved after escalation for: "
            + "; ".join(still)
            + " -- set a closer init_scale in the component's defaults.yaml."
        )

    logger.debug(
        f"Whitening: rescaled in place (start lp={map_lp:.1f}); every raw "
        f"unit step now costs ~{PROBE_TARGET_DELTA} nats."
    )
    return {
        "map_lp": map_lp,
        "multipliers": multipliers,
        "raw_scales": raw_scales,
    }


def measure_barrier_scales(system, model, raw_start):
    """Measure and set the soft-bound barrier steepness scales in place.

    For every parameter element carrying a soft-bound barrier (derived, or
    the rare half-bounded sampled element), the natural steepness scale is
    its posterior 1-sigma width.  With the model freshly whitened, one raw
    unit IS one sigma along every sampled direction, so the measured width
    of element k is the quadrature sum of its responses to unit steps:
    sqrt(sum_j (D_k(e_j) - D_k(0))^2).  This replaces the old sympy forward
    Jacobian pass: n_sampled+1 forward evaluations of the (cheap) parameter
    transform graphs, no symbolic solving, no timeouts, and it captures
    nonlinearity over the actual 1-sigma step.

    User-pinned elements (bound_scale) are respected by set_barrier_scales.
    Returns {param_label: measured_scale_vector}.
    """
    import pytensor.tensor as pt

    params = [
        p
        for p in system.get_all_parameters()
        if getattr(p, "_barrier_state", None) is not None
    ]
    if not params:
        return {}

    outs = []
    for p in params:
        v = p.value
        outs.append(pt.flatten(v) if v.ndim > 0 else pt.reshape(v, (1,)))
    fn = model.compile_fn(
        model.replace_rvs_by_values(outs),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    base = [np.asarray(b, dtype=float) for b in fn(raw_start)]
    var = [np.zeros_like(b) for b in base]
    n_steps = 0
    for key, val in raw_start.items():
        for i in range(np.asarray(val).size):
            step_pt = {k: v.copy() for k, v in raw_start.items()}
            step_pt[key].flat[i] += 1.0
            resp = fn(step_pt)
            for m, (b, r) in enumerate(zip(base, resp)):
                var[m] += (np.asarray(r, dtype=float) - b) ** 2
            n_steps += 1

    measured = {}
    for p, v in zip(params, var):
        s = np.sqrt(v)
        p.set_barrier_scales(s)
        measured[p.label] = s
    logger.debug(
        f"Whitening: measured soft-bound barrier scales for "
        f"{len(params)} parameter(s) from {n_steps} unit-step responses."
    )
    return measured


# If updating the barrier steepness moved the START's logp by more than this
# many nats, the barriers were active where the whitening probe measured its
# contours -- the measured scales are stale by up to the barrier-scale
# change.  One correction round (re-probe at ~unit scales, cheap) removes
# the first-order coupling; the residual is second-order.
_BARRIER_FEEDBACK_NATS = 0.1


def measure_and_whiten(system, model, raw_start=None, logp_fn=None):
    """Full startup whitening: probe + rescale, then measured barrier
    steepness, with one correction round if the two interact.

    The whitening probe measures logp contours that INCLUDE the preliminary
    soft-bound barriers; updating the barriers afterwards therefore changes
    the surface the scales were measured on.  In the healthy case the start
    sits more than one transition width (0.01 * scale) inside every bound,
    the barrier penalty at the start is ~0 before AND after the update, and
    the feedback is exactly zero.  When the update does move the start's
    logp, the whitening is re-measured once against the final barriers
    (cheap: the model is already ~whitened, so every bracket lands within a
    few evaluations) and the barriers are re-measured with the corrected
    unit steps.
    """
    if raw_start is None:
        raw_start = system.get_raw_start(model)
    if logp_fn is None:
        logp_fn = model.compile_logp()

    report = apply_measured_whitening(system, model, raw_start, logp_fn)
    lp_before = float(logp_fn(raw_start))
    measure_barrier_scales(system, model, raw_start)
    lp_after = float(logp_fn(raw_start))

    if abs(lp_after - lp_before) > _BARRIER_FEEDBACK_NATS:
        logger.debug(
            f"Whitening: barrier update moved the start logp by "
            f"{lp_after - lp_before:+.2f} nats (a soft bound is active at "
            f"the start); re-measuring scales against the final barriers."
        )
        report2 = apply_measured_whitening(system, model, raw_start, logp_fn)
        # Fold the correction into the cumulative multipliers for reporting.
        for key, m2 in report2["multipliers"].items():
            m1 = report["multipliers"].get(key)
            if m1 is not None:
                keep = np.isfinite(m2)
                m1[keep] = m1[keep] * m2[keep]
        report["raw_scales"] = report2["raw_scales"]
        report["map_lp"] = report2["map_lp"]
        measure_barrier_scales(system, model, raw_start)

    return report


def save_whitening(system, path, map_lp=None):
    """Persist the ABSOLUTE whitening + barrier state next to the trace.

    The absolute logit-space scales (not multipliers) are stored so a reload
    reproduces the sampled trace's raw coordinates exactly, independent of
    the rebuilt model's preliminary scales.
    """
    data = {"version": 1, "map_lp": map_lp, "params": {}}
    for p in system.get_all_parameters():
        exporter = getattr(p, "export_whitening", None)
        state = exporter() if exporter is not None else None
        if state:
            data["params"][p.label] = state
    with open(path, "w") as f:
        json.dump(data, f)
    logger.debug(f"Whitening: state saved to {path}")


def load_whitening(system, path):
    """Apply a persisted whitening state.  Returns True on success.

    Validates EVERYTHING up front (every persisted parameter must exist with
    matching shapes, and every currently-whitened parameter must be covered)
    before touching any shared variable, so a mismatch -- a changed model --
    leaves the build untouched and the caller falls back to a fresh probe.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"Whitening: could not read {path} ({e}).")
        return False
    if data.get("version") != 1:
        logger.warning(f"Whitening: {path} has an unknown version.")
        return False
    saved = data.get("params", {})
    lookup = {p.label: p for p in system.get_all_parameters()}

    # Validate coverage and shapes before applying anything.
    whitened_now = {
        p.label
        for p in system.get_all_parameters()
        if getattr(p, "_whiten_state", None) is not None
    }
    if not whitened_now.issubset(saved.keys()):
        missing = sorted(whitened_now - set(saved))
        logger.warning(
            f"Whitening: persisted state does not cover {missing}; "
            f"re-measuring."
        )
        return False
    for label, state in saved.items():
        par = lookup.get(label)
        if par is None:
            logger.warning(
                f"Whitening: persisted parameter '{label}' no longer exists; "
                f"re-measuring."
            )
            return False
        ws = getattr(par, "_whiten_state", None)
        bs = getattr(par, "_barrier_state", None)
        if "scale_logits" in state:
            if (
                ws is None
                or len(state["scale_logits"])
                != np.asarray(ws["sv_scale_logits"].get_value()).size
            ):
                logger.warning(
                    f"Whitening: persisted state for '{label}' does not "
                    f"match the model; re-measuring."
                )
                return False
        if "barrier_scales" in state:
            if (
                bs is None
                or len(state["barrier_scales"])
                != np.asarray(bs["sv"].get_value()).size
            ):
                logger.warning(
                    f"Whitening: persisted barrier state for '{label}' does "
                    f"not match the model; re-measuring."
                )
                return False

    for label, state in saved.items():
        lookup[label].load_whitening(state)
    logger.debug(f"Whitening: state restored from {path} (no probe needed).")
    return True
