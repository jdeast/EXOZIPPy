"""Data-driven whitening: measure each raw parameter's true local scale and
rescale the model's whitening in place.

The model is first built with PRELIMINARY whitening scales (defaults.yaml
init_scale, or a fraction of the bound span when absent -- see
Parameter.build_pymc).  probe_scales() then measures, per raw element, the
step in the current raw coordinate whose symmetric second difference of logp
reaches 1 nat -- 0.5 nats per side, the EXOFASTv2 exofast_getmcmcscale
convention (dlogp=0.5 <-> dchi2=1, the 1-sigma step for a locally Gaussian
posterior), made gradient-immune: the second difference cancels the linear
Taylor term identically, so the measurement returns the local curvature width
1/sqrt(h) even when the start is far below the mode (where the one-sided drop
it replaces measured 0.5/|gradient| instead, arbitrarily tight).  Directions
with no measurable positive curvature fall back to the one-sided nearer
contour; see probe_scales.  Because one raw unit is one preliminary scale by
construction, the measured step IS the multiplicative correction to the
preliminary scale.  apply_measured_whitening() feeds it to
Parameter.set_whitening(), which updates the pytensor.shared whitening scales
in place -- no model rebuild, and the posterior is provably unchanged.

That proof covers the LOGIT branch only -- an element with two finite bounds,
where section C of Parameter.build_pymc cancels the raw N(0,1) prior
symbolically for any scale, so the scale affects conditioning alone.  It is
exactly the set set_whitening rescales.  Every NON-logit element (anything
without two finite bounds) is left untouched, because there the raw N(0,1) IS
the prior: N(mu, sigma) when a sigma was given, and N(initval, init_scale)
when the bounds are infinite and no sigma was -- the one place init_scale is
a posterior term rather than conditioning, and one a data-measured multiplier
would turn into a data-dependent prior width.  build_pymc warns when it builds
such an element.  So the invariant holds for the pass as a whole, but only
because the branch it does not hold for is never rescaled.

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
import os

import numpy as np

from exozippy.components.parameter import raised_raw_cancellation_clip

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

# An element whose gradient contributes more than this many nats over one
# measured scale unit is "gradient-dominated": for a locally quadratic logp,
# |g|*sigma is the start's displacement from its conditional optimum in
# sigmas, so 3.0 means "the start is >~3 sigma off along this direction".
# The curvature probe still measures the width correctly there (that is its
# point); the warning exists because everything else that uses the START --
# chain scatter, tempered burn-in, the reported start table -- degrades with
# displacement, and a seed polish would fix it at the source.
_GRAD_DOMINATED_NATS = 3.0

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

_BANNER = "!" * 70


class StaleWhiteningError(RuntimeError):
    """A saved trace's whitening file does not apply to this build.

    Raised only on the REUSE path (``recompute_trace: false`` with an
    existing trace).  See :func:`restore_whitening_for_trace`.
    """


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


def _probe_element(
    eval_delta,
    target=PROBE_TARGET_DELTA,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    rtol=_PROBE_RTOL,
    max_bisect=_PROBE_MAX_BISECT,
):
    """Measure one element's scale: symmetric curvature first, then the
    one-sided nearer contour, then None (flat).

    The symmetric second difference C(s) = delta(+s) + delta(-s)
    (= 2*lp(x0) - lp(x0+s) - lp(x0-s)) cancels the gradient term of the
    Taylor expansion identically, so bracket+bisect on C(s) = 2*target
    returns 1/sqrt(h) -- the local Gaussian sigma -- whether or not the
    start is at the mode.  A non-finite side (hard prior wall inside +/-s)
    counts as past-target, so a wall bounds the scale at its own distance,
    matching the one-sided behavior.

    When C never reaches the target within max_step (h <= 0: a locally
    linear direction, a ridge, or a saddle) the curvature is meaningless
    and the measurement falls back to the one-sided probe of each sign,
    taking the NEARER contour -- which errs tight, the direction every
    downstream consumer absorbs.  When both signs are flat too, returns
    (None, "flat", nan) and the caller applies its flat fallback.

    Returns (scale_or_None, method, grad_nats) where method is one of
    "curvature" / "linear" / "flat" and grad_nats = |delta(+s) - delta(-s)|/2
    at the accepted curvature scale (nan otherwise): the gradient's nats over
    one scale unit, i.e. the start's displacement from its conditional
    optimum in sigmas for a locally quadratic logp.
    """
    cache = {}

    def d(step):
        if step not in cache:
            cache[step] = eval_delta(step)
        return cache[step]

    def curv(s):
        dp, dm = d(s), d(-s)
        if not (np.isfinite(dp) and np.isfinite(dm)):
            return np.inf  # a wall inside +/-s: treat as past-target (tight)
        return dp + dm

    target_c = 2.0 * target
    # BRACKET on the symmetric drop; same geometry as probe_step_1d.
    lo, hi = 0.0, None
    step = 1.0
    while step <= max_step:
        if curv(step) >= target_c:
            hi = step
            break
        lo = step
        step *= 2.0

    if hi is not None:
        s_h = None
        for _ in range(max_bisect):
            mid = 0.5 * (lo + hi)
            c = curv(mid)
            if np.isfinite(c) and abs(c - target_c) <= rtol * target_c:
                s_h = mid
                break
            if c > target_c:
                hi = mid
            else:
                lo = mid
            if (hi - lo) < min_step:
                break
        if s_h is None:
            s_h = max(0.5 * (lo + hi), min_step)
        dp, dm = d(s_h), d(-s_h)
        grad_nats = (
            0.5 * abs(dp - dm)
            if np.isfinite(dp) and np.isfinite(dm)
            else np.nan
        )
        return min(s_h, max_step), "curvature", grad_nats

    # FALLBACK: no measurable positive curvature out to max_step.  The
    # one-sided searches reuse the cached evaluations from the bracket above.
    steps = [
        probe_step_1d(
            d,
            sign,
            target=target,
            max_step=max_step,
            min_step=min_step,
            rtol=rtol,
            max_bisect=max_bisect,
        )
        for sign in (1.0, -1.0)
    ]
    steps = [x for x in steps if x is not None]
    if steps:
        return min(float(np.min(steps)), max_step), "linear", np.nan
    return None, "flat", np.nan


def probe_scales(
    raw_start,
    logp_fn,
    max_step=_PROBE_MAX_STEP,
    min_step=_PROBE_MIN_STEP,
    max_bisect=_PROBE_MAX_BISECT,
    flat_scale=_PROBE_FLAT_SCALE,
    diagnostics=None,
):
    """Adaptive probe: per-element scale whose raw step costs ~0.5 nats.

    Per element the measurement is _probe_element's three-rung ladder:

    1. Symmetric curvature: bracket+bisect on the second difference
       C(s) = 2*lp(x0) - lp(x0+s) - lp(x0-s) = h*s^2 + O(s^4), whose target
       crossing is 1/sqrt(h) -- the local Gaussian sigma -- for ANY gradient.
       The one-sided drop this replaces contains the gradient term g*s, so
       from a start that is not the mode it measured 0.5/|g| instead of
       sigma: on examples/ob140939 a start ~5900 nats below the posterior
       returned scales ~1000x too small, the mass matrix re-widened the raw
       posterior to sd ~5e3, and the sampler diverged on 86% of draws
       against parameter.py's _RAW_CANCELLATION_CLIP wall at |raw|=1e4.
    2. When C never reaches the target (h <= 0: locally linear directions,
       ridges, saddles), fall back to EXOFASTv2's exofast_getmcmcscale
       convention of searching each sign for the one-sided 0.5-nat drop,
       taking the NEARER contour (not the average: EXOFASTv2 probes from an
       AMOEBA best fit where the directions are near-symmetric; off the mode
       the uphill contour only turns over on the far side, and averaging it
       in inflates the scale -- chains then jitter past the logit
       saturation walls and start pinned at the bounds).  The nearer contour
       only ever errs tight, which downstream consumers absorb
       (chain-scatter multipliers, DE ensemble expansion, NUTS adaptation).
    3. When both signs are flat too, `flat_scale`.

    The step search is bracket + bisect.  An earlier version instead walked a
    fixed probe ladder [0.003 ... 2.0], took the FIRST magnitude registering
    any drop at all, and extrapolated with a quadratic assumption,
    scale = |dp| * sqrt(target/delta) -- which only holds AT a maximum, and
    for the generic locally-linear response collapses to sqrt(0.5*|dp|/g),
    tracking the smallest rung of the ladder (examples/ogle0383: 0.048
    returned where the true 0.5-nat step is 0.566, under-dispersing every
    chain 12-18x).

    `diagnostics`, when given, is a dict filled with:
      "linear_fallback" -- ["key[i]", ...] elements measured by rung 2
      "flat"            -- ["key[i]", ...] elements that fell to rung 3
      "gradient_nats"   -- {"key[i]": g} rung-1 elements' |gradient| in nats
                           per measured scale unit (the start's displacement
                           from its conditional optimum in sigmas, locally)

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
    linear_fallback = []
    gradient_nats = {}
    for key, val in raw_start.items():
        n = val.size
        s = np.full(n, flat_scale)
        for i in range(n):

            def eval_delta(step, key=key, i=i):
                return _delta_at(key, i, step)

            scale, method, g_nats = _probe_element(
                eval_delta,
                max_step=max_step,
                min_step=min_step,
                max_bisect=max_bisect,
            )
            if scale is not None:
                s.flat[i] = scale
            if method == "flat":
                flat.append(f"{key}[{i}]")
            elif method == "linear":
                linear_fallback.append(f"{key}[{i}]")
            elif np.isfinite(g_nats):
                gradient_nats[f"{key}[{i}]"] = float(g_nats)
            if s.flat[i] < 0.5:
                tight.append(f"{key}[{i}]: scale={s.flat[i]:.3g}")
        scales[key] = s.reshape(val.shape)

    if diagnostics is not None:
        diagnostics["linear_fallback"] = linear_fallback
        diagnostics["flat"] = flat
        diagnostics["gradient_nats"] = gradient_nats

    if tight:
        logger.debug(
            "whitening probe: tightly-constrained params: " + "; ".join(tight)
        )
    if linear_fallback:
        logger.debug(
            "whitening probe: no measurable positive curvature (h <= 0) "
            "along: "
            + "; ".join(linear_fallback)
            + " -- using the nearer one-sided contour (errs tight)."
        )
    if flat:
        logger.debug(
            f"whitening probe: logp flat to within {PROBE_TARGET_DELTA} nats "
            f"out to +/-{max_step:g} for: "
            + "; ".join(flat)
            + f" -- using scale={flat_scale}"
        )
    return map_lp, scales


# The probe's reach upward is NOT _WHITEN_MAX_STEP.  parameter.py's
# _RAW_CANCELLATION_CLIP puts a near-vertical wall in logp at |raw| = the
# clip (past it the +0.5*raw**2 correction stops tracking pm.Normal's
# -0.5*raw**2, so the drop reaches 0.5 nats within 0.5/clip of the wall).
# The probe cannot see any real contour beyond it, so at the sampler's clip
# of 1e4 an element whose preliminary scale was too tight by more than ~4
# orders measured exactly 1e4 -- inside the old escalation window, hence no
# re-probe and no warning, and the model stayed silently under-whitened
# (review 1.2.1: the ob140939 divergence family).  The wall is a sampler
# safety device, not a property of the posterior, so the probe RAISES it for
# its own duration.
#
# How far may it honestly be raised?  The clip exists because the two
# -0.5*raw**2 terms cancel only to float64 precision of the larger one, a
# residual of ~0.5*raw**2 * 2**-52 nats.  That reaches the probe's own 0.5-nat
# target near |raw| = sqrt(2**53) ~ 7e7, which is the hard ceiling.  At 1e6
# the residual is 0.5*1e12*2.2e-16 ~ 1.1e-4 nats against a 0.5-nat target
# (2e-4 relative, well inside _PROBE_RTOL = 5%), and it buys 6 orders of
# magnitude of reach per round.  Do not raise it toward 7e7 to buy the last
# order: the noise there is the measurement.
_PROBE_RAW_CLIP = 1.0e6

# A measured multiplier at/beyond these marks was CLIPPED by one of the
# probe's three limits -- the step floor, the max step, or the wall above --
# so the true scale was only bounded, not resolved.  Each mark is half the
# corresponding limit, i.e. one bracketing step inside it.  Escalation applies
# the clipped value (already a huge improvement) and re-probes just those
# elements in the new raw coordinates, where the residual error is within
# range again; it repeats until nothing is clipped, so the reach is not a
# hardcoded number of orders of magnitude on either side.  Per round that is
# ~14 orders down (_WHITEN_MIN_STEP) and ~6 up (_PROBE_RAW_CLIP, the binding
# limit -- _WHITEN_MAX_STEP is far beyond it); the cap below only stops a
# pathological non-converging case, and reaching it warns.
_CLIP_LO = 2.0 * _WHITEN_MIN_STEP
_CLIP_HI = 0.5 * min(_WHITEN_MAX_STEP, _PROBE_RAW_CLIP)
_MAX_ESCALATION_ROUNDS = 8


def _param_for_raw(lookup, key):
    name = key[: -len("_raw")] if key.endswith("_raw") else key
    return lookup.get(name)


def _refetch_raw_start(system, model, fallback):
    """The current canonical raw start, re-read after a rescale.

    set_whitening anchors raw = 0 and re-expresses a nonzero raw_initval (a
    pre-whitening seed polish moved the start off the anchor) in the new
    coordinates, so any raw-start dict captured BEFORE a rescale is stale
    afterwards unless it was all zeros -- which it was, historically,
    making this a no-op for unpolished runs.  Systems without get_raw_start
    (test stubs) keep the fallback."""
    getter = getattr(system, "get_raw_start", None)
    if getter is None:
        return fallback
    return getter(model)


def _clipped_elements(latest, applied):
    """The (raw_name, flat_index) pairs whose latest multiplier was clipped.

    Only elements that were actually rescaled (`applied`) can be re-probed;
    a non-finite multiplier is a flat direction, which is a different report
    and not something another round can resolve.
    """
    return [
        (key, i)
        for key, mult in latest.items()
        for i in range(mult.size)
        if applied[key][i]
        and np.isfinite(mult.flat[i])
        and not (_CLIP_LO < abs(mult.flat[i]) < _CLIP_HI)
    ]


def _probe_selected(raw_start, logp_fn, elems):
    """Re-probe only the given (raw_name, flat_index) elements."""
    map_lp = float(logp_fn(raw_start))
    out = {}
    for key, i in elems:

        def eval_delta(step, key=key, i=i):
            probe = {k: v.copy() for k, v in raw_start.items()}
            probe[key].flat[i] += step
            return map_lp - float(logp_fn(probe))

        scale, _method, _g = _probe_element(
            eval_delta,
            max_step=_WHITEN_MAX_STEP,
            min_step=_WHITEN_MIN_STEP,
            max_bisect=_WHITEN_MAX_BISECT,
        )
        out[(key, i)] = scale if scale is not None else np.nan
    return out


def apply_measured_whitening(system, model, raw_start=None, logp_fn=None):
    """Measure every sampled element's true raw-space scale and rescale the
    model's whitening in place.

    Probes from the relaxation-engine start (raw_start), then routes each raw
    variable's measured 0.5-nat step to its Parameter.set_whitening(), which
    updates the shared whitening scales so the same compiled model is now
    whitened by the data-driven scales.  Elements whose probe failed (flat
    both ways) keep their preliminary scale; elements whose raw N(0,1) is the
    prior -- every NON-LOGIT element, i.e. anything without two finite
    bounds, whether its width came from a sigma or from init_scale -- are
    never rescaled.

    The whole measurement runs with parameter.py's raw-cancellation clip
    raised to _PROBE_RAW_CLIP: at its sampling value that wall, not the
    posterior, is what a badly-too-tight element's 0.5-nat contour would be
    (see _PROBE_RAW_CLIP).  It is restored before this returns.

    Elements whose measured multiplier hit the probe's dynamic-range limits
    (a preliminary scale off by more than ~6 orders of magnitude too tight or
    ~14 too loose, e.g. a period constrained to nanoseconds against day-scale
    bounds) are escalated: the clipped correction is applied, then just those
    elements are re-probed in the new raw coordinates, where the residual
    error is resolvable.  That repeats until nothing is clipped, so the reach
    is a product of rounds rather than a fixed number of orders of magnitude;
    an element still clipped at _MAX_ESCALATION_ROUNDS gets a warning naming
    it (fix its defaults.yaml init_scale).

    Returns a report dict:
      map_lp       -- logp at the start
      multipliers  -- {raw_name: array} CUMULATIVE measured step per element
                      in preliminary-scale units (~1 means the preliminary
                      scale was already right; NaN = flat direction)
      raw_scales   -- {raw_name: array} per-element scale in the FINAL raw
                      units (1.0 for rescaled elements by construction; the
                      measured value for the deliberately-untouched
                      non-logit elements, whose raw N(0,1) IS their prior)
                      -- what PTDE's chain dispersion uses instead of
                      re-probing.
    """
    if raw_start is None:
        raw_start = system.get_raw_start(model)
    if logp_fn is None:
        logp_fn = model.compile_logp()

    # The clip is raised for the WHOLE pass, not just around each probe call:
    # the escalation loop below alternates probing with set_whitening, and a
    # measurement taken under a different wall than the one it is compared
    # against is exactly the confusion this fixes.  Restored on exit, by the
    # context manager, on the error path too -- sampling must never see the
    # raised value.
    with raised_raw_cancellation_clip(_PROBE_RAW_CLIP):
        return _measure_and_apply(system, model, raw_start, logp_fn)


def _measure_and_apply(system, model, raw_start, logp_fn):
    """apply_measured_whitening's body, under the raised probe-time clip."""
    n_elements = sum(v.size for v in raw_start.values())
    logger.debug(
        f"Whitening: probing {n_elements} raw element(s) for their "
        f"data-driven scales..."
    )
    # flat_scale=NaN: a flat direction keeps its preliminary scale
    # (set_whitening skips non-finite entries) and stays flagged in the
    # report so the diagnostics table can warn about it.
    probe_diag = {}
    map_lp, scales = probe_scales(
        raw_start,
        logp_fn,
        max_step=_WHITEN_MAX_STEP,
        min_step=_WHITEN_MIN_STEP,
        max_bisect=_WHITEN_MAX_BISECT,
        flat_scale=np.nan,
        diagnostics=probe_diag,
    )

    # Loud diagnostics: neither condition invalidates the measured scales
    # (curvature is gradient-immune, and the linear fallback errs tight),
    # but both are signatures of a start far from its conditional optimum,
    # which degrades everything that uses the START itself -- chain scatter,
    # tempering, the reported start table.  ob140939's 86%-divergence run
    # would have printed ~16 gradient-dominated elements here.
    grad_dom = {
        k: g
        for k, g in probe_diag.get("gradient_nats", {}).items()
        if g > _GRAD_DOMINATED_NATS
    }
    if grad_dom:
        worst = sorted(grad_dom.items(), key=lambda kv: -kv[1])
        logger.warning(
            "Whitening: gradient-dominated start -- displaced ~N sigma from "
            "its conditional optimum along: "
            + "; ".join(f"{k} (~{g:.0f} sigma)" for k, g in worst)
            + " -- scales are curvature-measured and stay valid, but "
            "consider a polished start (seed_polish) or closer initvals."
        )
    if probe_diag.get("linear_fallback"):
        logger.warning(
            "Whitening: no measurable positive curvature (ridge/saddle/"
            "linear logp) along: "
            + "; ".join(probe_diag["linear_fallback"])
            + " -- using the nearer one-sided contour, which errs tight."
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

    # The rescale above re-expressed a polished (nonzero) start in the new
    # raw coordinates; re-read it so the escalation re-probe measures
    # around the same physical point.
    raw_start = _refetch_raw_start(system, model, raw_start)

    # Escalation: re-probe elements whose LATEST measured multiplier was
    # clipped by one of the probe's limits, until none is -- or the round cap
    # is reached.  Keyed on the latest measurement, never on the cumulative
    # product: after a successful escalation the CUMULATIVE is outside the
    # window by construction (that is what "the preliminary scale was off by
    # 14 orders" means), so testing it re-probed resolved elements forever
    # and then declared them "still unresolved" (review 1.2.2).
    latest = {
        key: np.asarray(m, dtype=float).copy()
        for key, m in multipliers.items()
    }
    round_i = 0
    while True:
        clipped = _clipped_elements(latest, applied)
        if not clipped or round_i >= _MAX_ESCALATION_ROUNDS:
            break
        round_i += 1
        logger.warning(
            f"Whitening: {len(clipped)} element(s) hit the probe's dynamic "
            f"range (preliminary scale off by >6 orders of magnitude too "
            f"tight, or >14 too loose); escalation round {round_i}: "
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
                # A flat (NaN) re-probe leaves the cumulative alone and drops
                # the element from the loop: nothing more can be measured.
                latest[key].flat[i] = m
                if np.isfinite(m) and m > 0:
                    mult2[i] = m
                    multipliers[key].flat[i] *= m
            par.set_whitening(mult2)
        raw_start = _refetch_raw_start(system, model, raw_start)

    still = [
        f"{key}[{i}]: last measured={latest[key].flat[i]:.3g}, "
        f"cumulative={multipliers[key].flat[i]:.3g}"
        for key, i in _clipped_elements(latest, applied)
    ]
    if still:
        logger.warning(
            f"Whitening: scale still unresolved after {round_i} escalation "
            f"round(s) for: "
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
        "probe_diagnostics": probe_diag,
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
    # Each rescale re-expresses a polished (nonzero) start in the new raw
    # coordinates -- re-read it before evaluating or probing there again
    # (a no-op for the historical all-zeros start).
    raw_start = _refetch_raw_start(system, model, raw_start)
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
        raw_start = _refetch_raw_start(system, model, raw_start)
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


def _validate_whitening_state(system, saved, lookup):
    """Check a persisted params mapping against the current build.

    Returns None when it applies cleanly, or a human-readable reason string.
    Every vector the apply step will touch is checked here -- both whitening
    vectors AND the barrier vector -- so the apply step below cannot fail
    part way through and leave the model in a half-restored state (the two
    are different measures: rescaling the whitening is posterior-preserving,
    but the barrier IS a posterior term, so a model carrying one file's
    barriers and another's whitening has a logp that was never sampled).
    """
    # Coverage: every parameter that carries restorable state must appear in
    # the file.  Barrier-ONLY parameters (derived ones: no _whiten_state, a
    # soft bound) count -- their barrier steepness is a posterior term, so a
    # file that predates them describes a different logp than the one that
    # produced the trace being reused.
    stateful_now = {
        p.label
        for p in system.get_all_parameters()
        if getattr(p, "_whiten_state", None) is not None
        or getattr(p, "_barrier_state", None) is not None
    }
    missing = sorted(stateful_now - set(saved))
    if missing:
        return f"persisted state does not cover {missing}"

    for label, state in saved.items():
        par = lookup.get(label)
        if par is None:
            return f"persisted parameter '{label}' no longer exists"
        ws = getattr(par, "_whiten_state", None)
        bs = getattr(par, "_barrier_state", None)

        has_whiten_keys = "scale_logits" in state or "gaussian_scales" in state
        if has_whiten_keys:
            if ws is None:
                return f"'{label}' is no longer whitened"
            # BOTH vectors are validated: Parameter.load_whitening applies
            # them together, and checking only scale_logits let a bad
            # gaussian_scales abort mid-loop after earlier parameters had
            # already been written.
            for key, sv in (
                ("scale_logits", ws["sv_scale_logits"]),
                ("gaussian_scales", ws["sv_gaussian_scales"]),
            ):
                if key not in state:
                    return f"'{label}' is missing '{key}'"
                if len(state[key]) != np.asarray(sv.get_value()).size:
                    return f"'{label}' {key} does not match the model"
        elif ws is not None:
            return f"'{label}' is whitened now but was not when saved"

        if "barrier_scales" in state:
            if bs is None:
                return f"'{label}' no longer has a soft bound"
            if (
                len(state["barrier_scales"])
                != np.asarray(bs["sv"].get_value()).size
            ):
                return f"'{label}' barrier state does not match the model"
        elif bs is not None:
            return f"'{label}' has a soft bound that the persisted state omits"

    return None


def _apply_whitening_file(system, path):
    """Read, validate and apply a persisted whitening state.

    Returns ``None`` on success, or a human-readable reason the file does
    not apply.  Nothing is written to any shared variable until the whole
    file has been validated, so a rejected file leaves the build untouched
    (never half-restored).

    This is the shared body of :func:`load_whitening` (fresh path: warn and
    re-measure) and :func:`restore_whitening_for_trace` (reuse path: raise).
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return f"{path} could not be read ({e})"
    if data.get("version") != 1:
        return f"{path} has an unknown version ({data.get('version')!r})"
    saved = data.get("params", {})
    lookup = {p.label: p for p in system.get_all_parameters()}

    reason = _validate_whitening_state(system, saved, lookup)
    if reason is not None:
        return reason

    for label, state in saved.items():
        if not lookup[label].load_whitening(state):
            # Unreachable: the validation above covers every shape
            # load_whitening checks.  Loud rather than silent if it ever is.
            return (
                f"applying persisted state for '{label}' failed AFTER "
                f"validation; the build may be partially restored"
            )
    logger.debug(f"Whitening: state restored from {path} (no probe needed).")
    return None


def load_whitening(system, path):
    """Apply a persisted whitening state.  Returns True on success.

    Validates EVERYTHING up front -- every persisted parameter must exist,
    every vector it carries must match the built shape, and every parameter
    that carries whitening OR barrier state now must appear in the file --
    before touching any shared variable, so a mismatch (a changed model, a
    truncated file) leaves the build untouched and the caller falls back to
    a fresh probe.

    A mismatch WARNS and re-measures rather than raising: when nothing has
    been sampled yet the whitening is a free choice of coordinates, and
    unlike foreign posterior draws it can be honestly recomputed from the
    model at load time, so there is nothing to salvage by stopping the run.

    That reasoning holds ONLY while no existing draws depend on the answer.
    A caller that is about to REUSE a saved trace must call
    :func:`restore_whitening_for_trace` instead.
    """
    reason = _apply_whitening_file(system, path)
    if reason is not None:
        logger.warning(f"Whitening: {reason}; re-measuring.")
        return False
    return True


def restore_whitening_for_trace(system, path, trace_path):
    """Restore the whitening a REUSED trace was sampled under.

    Returns ``"restored"`` when the file applied, or ``"unverifiable"`` when
    no whitening file exists (a warning is logged and the build keeps its
    PRELIMINARY scales).  Raises :class:`StaleWhiteningError` when a file
    exists but does not apply to this build.

    Why this is not ``load_whitening``'s warn-and-re-measure: a trace's raw
    draws are coordinates in the whitened space, so they only decode to the
    physical values they were sampled at under the whitening that produced
    them.  Re-measuring is a probe of the CURRENT start, which the draws
    never saw; every posterior mode label, Laplace ledger width, recomputed
    lp and model curve drawn at a stored draw would then be computed at a
    different physical point than the sampler visited, with nothing
    downstream able to notice.  This is exactly
    :class:`~exozippy.trace_meta.StaleTraceError`'s situation, and the
    asymmetry that module documents ("whitening can honestly be
    re-measured") is an argument about a run that is about to sample, not
    about one decoding draws that already exist.

    Nothing is written to ``path`` on this path, ever: overwriting the
    whitening a trace was sampled under destroys the only record of how to
    read that trace, and a re-measured file would be indistinguishable from
    a genuine one on the NEXT reload -- laundering a detected mismatch into
    a silent one.
    """
    if not os.path.exists(path):
        # Same policy as trace_meta's absent-fingerprint case: this is NOT a
        # detected mismatch.  Traces sampled with 'measure_scales: false',
        # or before whitening was persisted at all, legitimately have no
        # file -- and for the former the preliminary scales in the fresh
        # build ARE the ones it was sampled with.  Keep them: a probe of
        # the current start is not more correct than they are, merely less
        # reproducible, and it would move the soft-bound barriers (a real
        # posterior term) away from the sampled model as well.
        logger.warning(
            f"{_BANNER}\n"
            f"UNVERIFIABLE WHITENING: reusing the saved trace {trace_path}, "
            f"but no whitening state was found at {path}, so the raw draws "
            f"cannot be checked against -- or decoded under -- the "
            f"coordinates they were sampled in. This is NOT a detected "
            f"mismatch; the model keeps its PRELIMINARY scales and nothing "
            f"is re-measured or written. If the trace was sampled with "
            f"measured scales, its physical Deterministics are still "
            f"correct, but anything recomputed from the raw draws (mode "
            f"labels, the seed ledger, a recomputed lp) may not be. To be "
            f"certain, re-sample with 'sampler: {{recompute_trace: true}}'."
            f"\n{_BANNER}"
        )
        return "unverifiable"

    reason = _apply_whitening_file(system, path)
    if reason is None:
        return "restored"

    raise StaleWhiteningError(
        f"{_BANNER}\n"
        f"STALE WHITENING: {path} does not describe the model this config "
        f"builds, and the saved trace {trace_path} is being REUSED "
        f"('sampler: {{recompute_trace: false}}').\n"
        f"  why it does not apply: {reason}\n"
        f"A trace's raw draws are coordinates in the whitened space: they "
        f"only decode to the physical values the sampler visited under the "
        f"whitening they were sampled with. Re-measuring it here would "
        f"probe the CURRENT start instead -- silently changing what every "
        f"stored draw means -- and would overwrite the only record of how "
        f"to read this trace. The draws cannot be repaired at load time.\n"
        f"REMEDY: re-sample with 'sampler: {{recompute_trace: true}}' in the "
        f"config, or restore the {path} that was written alongside "
        f"{trace_path} (and revert the model edit that invalidated it).\n"
        f"{_BANNER}"
    )


def prepare_whitening(
    system, model, raw_start, path, trace_path, *, reusing_trace
):
    """Put the model into the whitened coordinates this run must use.

    The two situations this serves look alike and are not:

    * ``reusing_trace=False`` -- about to SAMPLE.  The whitening is a free
      choice of coordinates, so it is measured from the data and persisted
      to ``path``.  Whatever was there before is superseded, because the
      draws it described are about to be superseded too.  Returns the probe
      report (the caller must re-read its raw start: the rescale re-expressed
      it in the new coordinates).
    * ``reusing_trace=True`` -- about to DECODE draws that already exist.
      The whitening is then a property of those draws, not a choice, so it
      is restored and never re-measured or rewritten (see
      :func:`restore_whitening_for_trace`; a mismatch raises).  Returns
      ``None`` -- there is no probe and the raw start did not move.
    """
    if reusing_trace:
        restore_whitening_for_trace(system, path, trace_path)
        return None
    report = measure_and_whiten(system, model, raw_start)
    save_whitening(system, path, map_lp=report["map_lp"])
    return report
