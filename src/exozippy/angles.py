"""Sampling a precisely MEASURED angle directly instead of through its latents.

An angle with no seam is sampled as a 2-D direction ``(x, y)`` with structural
N(0, 1) priors, ``angle = arctan2(y, x)`` (orbit ``xbigomega``/``ybigomega``
and ``xomega``/``yomega``, lens ``xalpha``/``yalpha``;
``components/parameter.md``, "Structural priors").  That is the right
coordinate for an angle the data barely constrain: the posterior is a convex
blob around the origin, which NUTS and differential evolution both handle.
It is the WRONG one for an angle the data pin: the posterior in ``(x, y)`` is
then a narrow RAY from the origin -- ``sigma * r`` across, a Rayleigh width
along -- tilted at the angle itself, which a diagonal metric cannot
straighten, with a funnel at ``r -> 0`` where the ray's width vanishes.  Gaia
BH3's node is measured to 0.13 deg: a 230:1 ridge, step size ~0.001, every
draw at max tree depth, and chains trapped in the funnel.

So, JDE 2026-11-04: "Do the angle reparameterization automatically with a
warning and user override."  For each angle element whose implied sigma at
the (polished) start is below :data:`DIRECT_ANGLE_MAX_SIGMA`, the angle itself
is sampled, logit-uniform on a window ``[c - h, c + h]`` centered on the
start ``c``: a full turn (``h = pi``) by default, so the only seam sits
opposite the start where a measured posterior has no mass; half a turn
(``h = pi/2``) where the component says the angle is degenerate with its own
antipode (an ascending node with no radial data, ``Orbit.direct_angle_
halfwidth``), so the window holds exactly ONE of the two labels and the
post-sampling fold has nothing to collapse.  The prior is unchanged: uniform
on a full turn is exactly the latents' angle marginal, and uniform on the
half-turn window is the folded one.

THE DECISION NEEDS THE START AND CHANGES THE GRAPH, so it is taken after the
seed polish and the whitening probe (``run.py``): the latent model is built,
polished and whitened, the angle's CONDITIONAL sigma is probed along the
circle at the polished point (:func:`choose_direct_angles`; two logp
evaluations per element when the angle is NOT tight, which is the common case
and leaves the run bit-identical), a tight one must also pass MARGINALLY
(the Laplace covariance from the Hessian, :class:`_MarginalProbe`), and only
when something switches is the model rebuilt (:func:`adopt_direct_angles`),
the polished point carried over physically, and the probe run again on the
new model.  A user override is read at stage 3 and
applied at the end of ``System.prepare`` (:func:`apply_forced`), so the GUI and
every other caller of ``build_model`` honors it too.  Which elements were
direct, and their windows, travel with the trace (``trace_meta.ANGLES_ATTR``),
because a raw draw decodes only through the window it was sampled in: the
reuse path restores them before it builds (:func:`restore_from_trace`).

The DECLARATION is the component's, in defaults.yaml on the angle parameter::

    bigomega:
      direction_latents: {pair: [xbigomega, ybigomega], expression: default}

``expression`` names the block that derives the angle from the pair, so an
angle with two parameterizations (``omega``: sqrt(e)cos/sin, or its own
direction vector under ``fitvcve``) is eligible only where it uses the pair.
Nothing here knows which component an angle belongs to.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .manifest import interpret_manifest_entry, normalize_selector

logger = logging.getLogger("exozippy")

# THE THRESHOLD, in radians (5.7 deg).  Below it an angle is sampled directly.
#
# Why this number.  With an isotropic N(0, I) prior on (x, y) and a likelihood
# that depends on the angle alone, the radius is Rayleigh-distributed
# INDEPENDENTLY of the data: mean sqrt(pi/2) = 1.253, standard deviation
# sqrt(2 - pi/2) = 0.655.  An angle measured to sigma therefore puts the
# posterior on a ray 1.253 sigma wide at its typical radius and 0.655 long,
# an aspect ratio of 0.523 / sigma that a diagonal metric cannot remove because
# the ray is tilted at the angle itself -- and narrower still toward the
# funnel at r -> 0 (0.5% of the mass is inside r = 0.1, where it is 12x
# narrower again).  At sigma = 0.1 the ratio is 5.2, about where the leapfrog
# count NUTS needs to cross the ridge (~ the ratio) starts to cost more than
# a tree depth or two; above it the latents are a mildly elongated blob and
# fine.  On the other side, the direct coordinate's only hazard is mass at its
# seam, which at sigma = 0.1 is 31 sigma away on a full-turn window.  The
# probe's sigma is CONDITIONAL on every other coordinate and the marginal can
# be wider -- on examples/GaiaBH1 the node's conditional 0.415 deg against a
# posterior sd of 2.2 deg, 5.3x -- so the seam has to absorb that factor:
# even at 5x the full-turn seam is 6 marginal sigma clear.  A HALF-turn
# window (a node-degenerate orbit) has its seam half as far away, so its
# threshold is scaled by the same factor (`threshold`, below): 0.05 rad, and
# the seam is again 31 conditional sigma away.  Both hazards are far from
# this value, which is what makes it a safe default rather than a tuned one.
DIRECT_ANGLE_MAX_SIGMA = 0.1


def threshold(halfwidth):
    """The switch threshold (rad) for a window of half-width ``halfwidth``:
    DIRECT_ANGLE_MAX_SIGMA on a full turn, scaled with the seam's distance."""
    return DIRECT_ANGLE_MAX_SIGMA * float(halfwidth) / np.pi


# The probe's contour, the same 0.5 nats the whitening probe uses: the
# distance along the circle at which logp has fallen by 0.5 nats is one
# Gaussian sigma.
PROBE_DROP_NATS = 0.5

# A multi-seed start must sit inside this fraction of the window's half-width
# around the center, or the switch is refused for that element: a seed near
# the seam would start a chain against a wall, and two seeds a half-turn apart
# (a lens alpha and its reflection) are two modes a window centered on one of
# them could cut.
SEED_SPREAD_FRACTION = 0.5

# Halvings of the probe step below the threshold before the sigma is called
# unmeasurably small (0.1 * 2**-40 ~ 1e-13 rad), then geometric bisections.
_MAX_HALVINGS = 40
_BISECTIONS = 8


@dataclass
class AngleSlot:
    """One element of a declared angle that is derived from its latents."""

    comp: Any
    angle: str
    x: str
    y: str
    element: int
    n_elements: int
    request: Optional[bool]
    halfwidth: float
    flag: str
    user_bounded: bool

    @property
    def key(self):
        return f"{self.comp.prefix}.{self.angle}"

    @property
    def where(self):
        names = getattr(self.comp, "names", None) or []
        name = (
            names[self.element]
            if self.element < len(names)
            else str(self.element)
        )
        return f"{self.comp.prefix}.{name}"


def direction_latents(base_defaults, comp_type, param):
    """The ``direction_latents:`` declaration of ``comp_type.param``, or None.

    Validated here, so a malformed declaration fails the first time anything
    asks rather than wherever it is first used.
    """
    spec = ((base_defaults.get(comp_type) or {}).get(param) or {}).get(
        "direction_latents"
    )
    if spec is None:
        return None
    pair = spec.get("pair") if isinstance(spec, dict) else None
    expr = spec.get("expression") if isinstance(spec, dict) else None
    if (
        not isinstance(pair, (list, tuple))
        or len(pair) != 2
        or not all(isinstance(p, str) for p in pair)
        or not isinstance(expr, str)
    ):
        raise ValueError(
            f"[{comp_type}.{param}] defaults.yaml 'direction_latents' must be "
            f"{{pair: [<x>, <y>], expression: <block>}}; got {spec!r}."
        )
    return {"pair": list(pair), "expression": expr}


def _n_elements(comp, entry):
    shape = entry.shape
    if shape:
        return int(np.prod(tuple(shape)))
    return int(comp.n_elements)


def _latent_mask(entry, expression, n, where):
    """Elements of ``entry`` derived by the ``expression`` block (and active)."""
    if entry.expr_selectors:
        sel = entry.expr_selectors.get(expression)
        mask = (
            np.zeros(n, dtype=bool)
            if sel is None
            else normalize_selector(sel, n, where)
        )
    else:
        mask = np.full(n, entry.expr_key == expression, dtype=bool)
    return mask & entry.activity_mask(n, where)


def _base_manifest(system):
    """The pristine manifest entries every direct choice is applied FROM.

    Stored the first time an angle is switched, so re-applying a different set
    of choices (the reuse path restoring a trace's) starts from what stage 3
    declared rather than from an earlier transform.
    """
    base = getattr(system, "_direct_angle_base", None)
    if base is None:
        base = {}
        system._direct_angle_base = base
    return base


def find_slots(system):
    """Every element of every declared angle that its latents derive.

    Also validates the user overrides: a ``fit<angle>`` flag on an element
    that has no latents to switch from is a request nothing can honor, and it
    raises naming the element.
    """
    base_defaults = system.config_manager.base_defaults
    base = _base_manifest(system)
    slots = []
    for comp in system.active_components.values():
        manifest = getattr(comp, "manifest", None) or {}
        declared = [
            name
            for name in (base_defaults.get(comp.prefix) or {})
            if direction_latents(base_defaults, comp.prefix, name)
        ]
        for angle in declared:
            decl = direction_latents(base_defaults, comp.prefix, angle)
            if angle not in manifest:
                # Not a parameter of this system at all (an orbit with no
                # astrometry has no node): a flag asking for it is a request
                # nothing can honor.
                flag, requests = comp.direct_angle_requests(angle)
                asked = [
                    comp.names[i]
                    for i, r in enumerate(requests)
                    if r is not None
                ]
                if asked:
                    raise ValueError(
                        f"[{comp.prefix}] '{flag}' is set on {asked}, but "
                        f"this system has no {comp.prefix}.{angle} to "
                        f"sample. Remove the key."
                    )
                continue
            raw = base.get((comp.prefix, angle), manifest[angle])
            entry = interpret_manifest_entry(raw)
            n = _n_elements(comp, entry)
            where = f"{comp.prefix}.{angle}"
            mask = _latent_mask(entry, decl["expression"], n, where)
            flag, requests = comp.direct_angle_requests(angle)
            if len(requests) != n:
                raise ValueError(
                    f"[{where}] {comp.prefix}.direct_angle_requests returned "
                    f"{len(requests)} entries for {n} element(s)."
                )
            stray = [
                i
                for i, r in enumerate(requests)
                if r is not None and not mask[i]
            ]
            if stray:
                names = [
                    comp.names[i] if i < len(comp.names) else str(i)
                    for i in stray
                ]
                raise ValueError(
                    f"[{comp.prefix}] '{flag}' is set on {names}, but "
                    f"{angle} is not sampled through the direction latents "
                    f"{decl['pair'][0]}/{decl['pair'][1]} there, so there is "
                    f"no coordinate choice to make. Remove the key."
                )
            if not mask.any():
                continue
            halfwidth = np.asarray(
                comp.direct_angle_halfwidth(angle), dtype=float
            ).reshape(-1)
            if halfwidth.size != n:
                raise ValueError(
                    f"[{where}] {comp.prefix}.direct_angle_halfwidth returned "
                    f"{halfwidth.size} entries for {n} element(s)."
                )
            bounded = comp.user_wrote_field(angle, "lower") | (
                comp.user_wrote_field(angle, "upper")
            )
            for i in np.flatnonzero(mask):
                slots.append(
                    AngleSlot(
                        comp=comp,
                        angle=angle,
                        x=decl["pair"][0],
                        y=decl["pair"][1],
                        element=int(i),
                        n_elements=n,
                        request=requests[i],
                        halfwidth=float(halfwidth[i]),
                        flag=flag,
                        user_bounded=bool(bounded[i])
                        if bounded.size == n
                        else bool(bounded.any()),
                    )
                )
    return slots


# ---------------------------------------------------------------------------
# Applying a set of choices to the manifests
# ---------------------------------------------------------------------------
def _entry_dict(raw):
    """A manifest entry as a mutable dict (expr selectors spelled out)."""
    from .components.parameterization import merge_options

    return merge_options(copy.deepcopy(raw))


def apply_choices(system, choices):
    """Rewrite the manifests so every element in ``choices`` is sampled directly.

    ``choices`` is ``{"<prefix>.<angle>": {"elements", "center", "halfwidth",
    "scale", ...}}`` -- the shape :func:`choices_to_json` writes -- with
    centers, half-widths and scales in INTERNAL units.  Applied from the
    pristine stage-3 manifest every time, so the call is idempotent and a
    later call REPLACES an earlier one.  The angle loses those elements from
    its latent expression (so they are sampled, on the window
    ``Parameter.direct_window`` writes); its two latents become INACTIVE
    there -- not part of that instance's parameterization at all, the same
    role a linear-law band's u2 has.  Records the choices on
    ``system.direct_angles``.
    """
    base_defaults = system.config_manager.base_defaults
    base = _base_manifest(system)
    comps = {c.prefix: c for c in system.active_components.values()}
    # Restore every entry a previous application touched.
    for (prefix, name), raw in base.items():
        comps[prefix].manifest[name] = copy.deepcopy(raw)
    for key, choice in choices.items():
        prefix, angle = key.split(".", 1)
        if prefix not in comps:
            raise KeyError(
                f"direct-angle choice for '{key}': no active component "
                f"'{prefix}' in this system."
            )
        comp = comps[prefix]
        decl = direction_latents(base_defaults, prefix, angle)
        if decl is None or angle not in comp.manifest:
            raise KeyError(
                f"direct-angle choice for '{key}': '{angle}' is not a "
                f"declared direction-latent angle of {prefix} in this system."
            )
        for name in [angle] + decl["pair"]:
            base.setdefault((prefix, name), copy.deepcopy(comp.manifest[name]))
        entry = interpret_manifest_entry(comp.manifest[angle])
        n = _n_elements(comp, entry)
        direct = np.zeros(n, dtype=bool)
        center = np.full(n, np.nan)
        half = np.full(n, np.nan)
        scale = np.full(n, np.nan)
        for j, i in enumerate(choice["elements"]):
            direct[int(i)] = True
            center[int(i)] = float(choice["center"][j])
            half[int(i)] = float(choice["halfwidth"][j])
            scale[int(i)] = float(choice["scale"][j])
        latent = _latent_mask(entry, decl["expression"], n, key)
        if np.any(direct & ~latent):
            raise ValueError(
                f"direct-angle choice for '{key}' names element(s) "
                f"{np.flatnonzero(direct & ~latent).tolist()}, which are not "
                f"derived from {decl['pair'][0]}/{decl['pair'][1]} in this "
                f"system (a trace from a different topology?)."
            )

        out = _entry_dict(comp.manifest[angle])
        selectors = {}
        expr = out.pop("expr_key", None)
        if isinstance(expr, dict):
            selectors = {
                k: normalize_selector(v, n, key) for k, v in expr.items()
            }
        elif expr is not None:
            selectors = {expr: entry.activity_mask(n, key)}
        sel = selectors.get(decl["expression"])
        if sel is not None:
            sel = sel & ~direct
            if sel.any():
                selectors[decl["expression"]] = sel
            else:
                del selectors[decl["expression"]]
        if selectors:
            out["expr_key"] = selectors
        out["direct_window"] = {
            "center": center.tolist(),
            "halfwidth": half.tolist(),
            "scale": scale.tolist(),
        }
        comp.manifest[angle] = out

        for name in decl["pair"]:
            latent_entry = _entry_dict(comp.manifest[name])
            pair_n = _n_elements(comp, interpret_manifest_entry(latent_entry))
            if pair_n != n:
                raise ValueError(
                    f"[{key}] the latent '{prefix}.{name}' has {pair_n} "
                    f"element(s) and the angle {n}; a direction pair is one "
                    f"latent element per angle element."
                )
            active = normalize_selector(latent_entry.get("mask"), n, name)
            latent_entry["mask"] = active & ~direct
            comp.manifest[name] = latent_entry
    system.direct_angles = copy.deepcopy(choices)


def choices_to_json(choices):
    """``choices`` as the JSON text stamped on the trace (exact floats)."""
    return json.dumps(choices, sort_keys=True)


def choices_from_json(text, where):
    try:
        choices = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{where}: the direct-angle record is not JSON ({exc})."
        ) from exc
    if not isinstance(choices, dict):
        raise ValueError(
            f"{where}: the direct-angle record is not a JSON object."
        )
    for key, choice in choices.items():
        fields = ("elements", "center", "halfwidth", "scale")
        if not isinstance(choice, dict) or not all(
            f in choice for f in fields
        ):
            raise ValueError(
                f"{where}: direct-angle record for '{key}' lacks one of "
                f"{fields}."
            )
        lengths = {len(choice[f]) for f in fields}
        if len(lengths) != 1:
            raise ValueError(
                f"{where}: direct-angle record for '{key}' has fields of "
                f"different lengths."
            )
    return choices


def _add_choice(choices, slot, center, scale, sigma, forced, marginal=None):
    rec = choices.setdefault(
        slot.key,
        {
            "elements": [],
            "center": [],
            "halfwidth": [],
            "scale": [],
            "sigma": [],
            "marginal_sigma": [],
            "forced": [],
        },
    )
    rec["elements"].append(slot.element)
    rec["center"].append(float(center))
    rec["halfwidth"].append(float(slot.halfwidth))
    rec["scale"].append(float(scale))
    rec["sigma"].append(None if sigma is None else float(sigma))
    rec["marginal_sigma"].append(None if marginal is None else float(marginal))
    rec["forced"].append(bool(forced))


# ---------------------------------------------------------------------------
# Stage 3/4: the user override
# ---------------------------------------------------------------------------
def apply_forced(system):
    """Honor every ``fit<angle>: true`` (end of ``System.prepare``).

    The window is centered on the element's RESOLVED start -- the latents'
    direction, which the relaxation engine seeds from a user angle -- and the
    angle's preliminary whitening scale is the switch threshold itself (the
    probe measures the real one).  A user bound on the angle is refused: the
    direct coordinate's support IS its window, and a second, user-chosen
    support would put a seam wherever it ends.  Returns the choices applied.
    """
    system.direct_angles = {}
    choices = {}
    for slot in find_slots(system):
        if slot.request is not True:
            continue
        if slot.user_bounded:
            raise ValueError(
                f"[{slot.where}] '{slot.flag}: true' samples {slot.angle} "
                f"directly on a window centered on its start, and that window "
                f"is its support -- a user lower/upper on "
                f"{slot.comp.prefix}.{slot.angle} would be a second, "
                f"different one. Remove the bound (a mu/sigma prior is fine) "
                f"or the flag."
            )
        center = _resolved_start_angle(system, slot)
        _add_choice(choices, slot, center, DIRECT_ANGLE_MAX_SIGMA, None, True)
    if choices:
        apply_choices(system, choices)
    return choices


def _resolved_start_angle(system, slot):
    """The angle the latents resolve to at the build start (internal rad)."""
    comp = slot.comp
    vals = []
    for name in (slot.x, slot.y):
        cfg = system.config_manager.resolve(
            comp.prefix,
            name,
            shape=(slot.n_elements,),
            names=getattr(comp, "names", None),
        )
        iv = np.atleast_1d(np.asarray(cfg.get("initval"), dtype=float))
        v = iv[slot.element] if iv.size == slot.n_elements else iv[0]
        if not np.isfinite(v):
            raise ValueError(
                f"[{slot.where}] '{slot.flag}: true' centers the window on "
                f"the start, but {comp.prefix}.{name} resolved to no start; "
                f"give {comp.prefix}.{slot.angle} an initval."
            )
        vals.append(float(v))
    return float(np.arctan2(vals[1], vals[0]))


# ---------------------------------------------------------------------------
# After the polish: the automatic decision
# ---------------------------------------------------------------------------
def _raw_position(par, element):
    tf = par._raw_transform
    if tf is None or element not in list(tf["sampled_idx"]):
        raise ValueError(
            f"[{par.label}] element {element} is not sampled, so it has no "
            f"raw coordinate to rotate."
        )
    return f"{par.label}_raw", list(tf["sampled_idx"]).index(element)


def _wrap(angle, center):
    """``angle`` moved by whole turns into ``[center - pi, center + pi)``."""
    return center + np.mod(angle - center + np.pi, 2.0 * np.pi) - np.pi


class _AngleProbe:
    """logp along the circle through one latent pair, all else fixed.

    Evaluated on the UNIT circle, not at the start's own radius: the prior is
    isotropic, so the logp change along any circle is the likelihood's alone,
    and the unit circle stays well clear of the funnel the polish may have
    left the latents in (a MAP of the latents sits at r -> 0).
    """

    def __init__(self, slot, raw_start, logp_fn):
        self.slot = slot
        self.logp_fn = logp_fn
        self.xp = getattr(slot.comp, slot.x)
        self.yp = getattr(slot.comp, slot.y)
        self.xkey, self.xi = _raw_position(self.xp, slot.element)
        self.ykey, self.yi = _raw_position(self.yp, slot.element)
        self.raw = {k: np.array(v, copy=True) for k, v in raw_start.items()}
        x0 = float(
            self.xp.element_phys_from_raw(
                slot.element, raw_start[self.xkey].reshape(-1)[self.xi]
            )
        )
        y0 = float(
            self.yp.element_phys_from_raw(
                slot.element, raw_start[self.ykey].reshape(-1)[self.yi]
            )
        )
        self.theta0 = float(np.arctan2(y0, x0))
        self.n_evals = 0
        self.lp0 = self.logp(0.0)

    def logp(self, delta):
        theta = self.theta0 + delta
        point = {k: np.array(v, copy=True) for k, v in self.raw.items()}
        xs = point[self.xkey].reshape(-1)
        ys = point[self.ykey].reshape(-1)
        xs[self.xi] = self.xp.element_raw_from_phys(
            self.slot.element, np.cos(theta)
        )
        ys[self.yi] = self.yp.element_raw_from_phys(
            self.slot.element, np.sin(theta)
        )
        point[self.xkey] = xs.reshape(np.shape(self.raw[self.xkey]))
        point[self.ykey] = ys.reshape(np.shape(self.raw[self.ykey]))
        self.n_evals += 1
        return float(self.logp_fn(point))

    def drop(self, delta):
        lp = self.logp(delta)
        if np.isnan(lp):
            raise FloatingPointError(
                f"logp is NaN at {self.slot.where}.{self.slot.angle} "
                f"rotated by {delta:+.3g} rad"
            )
        return self.lp0 - lp  # +inf for a -inf logp: excluded

    def side_sigma(self, sign, threshold):
        """The 0.5-nat distance on one side, or None when it exceeds
        ``threshold`` (the angle is not tight on this side)."""
        if self.drop(sign * threshold) < PROBE_DROP_NATS:
            return None
        hi = threshold
        lo = None
        for _ in range(_MAX_HALVINGS):
            trial = hi / 2.0
            if self.drop(sign * trial) < PROBE_DROP_NATS:
                lo = trial
                break
            hi = trial
        if lo is None:
            return hi  # tighter than the probe resolves; report its floor
        for _ in range(_BISECTIONS):
            mid = np.sqrt(lo * hi)
            if self.drop(sign * mid) < PROBE_DROP_NATS:
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))


def measure_angle_sigma(slot, raw_start, logp_fn, threshold=None):
    """``(theta0, sigma)`` for one slot at ``raw_start``.

    ``sigma`` is the LARGER of the two one-sided 0.5-nat distances along the
    circle (so an angle must be tight on both sides to switch), or None when
    either side is wider than ``threshold`` -- two evaluations decide that,
    which is all a loose angle costs.  It is CONDITIONAL on every other
    coordinate, the same kind of number the whitening probe measures.
    """
    threshold = DIRECT_ANGLE_MAX_SIGMA if threshold is None else threshold
    probe = _AngleProbe(slot, raw_start, logp_fn)
    if not np.isfinite(probe.lp0):
        raise FloatingPointError(
            f"logp is {probe.lp0} at the start of {slot.where}"
        )
    sides = []
    for sign in (+1.0, -1.0):
        s = probe.side_sigma(sign, threshold)
        if s is None:
            return probe.theta0, None
        sides.append(s)
    return probe.theta0, max(sides)


def _seed_angles(system, slot, theta0):
    """Seed 0's angle plus every other polished seed's, internal rad."""
    out = [theta0]
    seeds = getattr(system.config_manager, "seed_resolved", None) or []
    prefix = slot.comp.prefix
    for k in range(1, len(seeds)):
        resolved = seeds[k] or {}
        x = resolved.get(f"{prefix}.{slot.element}.{slot.x}")
        y = resolved.get(f"{prefix}.{slot.element}.{slot.y}")
        if x is None or y is None:
            out.append(theta0)  # not seeded: seed 0's raw start is kept
        else:
            out.append(float(np.arctan2(float(y), float(x))))
    return out


# The marginal check (JDE 2026-11-05, PR #561 review): the probe's sigma
# above is CONDITIONAL on every other coordinate, and an angle degenerate
# with another one is tight conditionally while its marginal is wide -- a
# face-on orbit's node, of which only Omega + omega is measured, would pass
# the probe and then pile against its window's walls.  So an angle also has
# to pass the threshold MARGINALLY, from the Laplace covariance at the start:
# Sigma = (-H)^-1 over every raw coordinate, sigma_theta^2 = J Sigma J^T
# with J = d theta / d raw (two nonzero entries, the latents').  The
# Hessian is taken in the WHITENED raw coordinates (the decision runs after
# the probe, so a unit step is ~0.5 nats on every axis): central
# differences of the gradient (2n gradient calls) when the model has one,
# else central second differences of logp (~2n^2 calls) when that fits
# _FD_HESSIAN_BUDGET_S, else the switch is refused.
_GRAD_STEP = 1e-3
_LOGP_STEP = 0.1
_FD_HESSIAN_BUDGET_S = 300.0
# A non-positive curvature direction is ignored when the angle's Jacobian
# has less than this fraction of its norm along it (the direction is a
# nuisance the angle does not couple to); above it the marginal is refused.
_NONMAX_OVERLAP = 1e-3


class _Flat:
    """Flatten/unflatten a raw point dict in a fixed key order."""

    def __init__(self, point, order):
        self.keys = list(order)
        self.shapes = [np.shape(point[k]) for k in self.keys]
        self.sizes = [int(np.size(point[k])) for k in self.keys]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])

    def flat(self, point):
        return np.concatenate(
            [np.asarray(point[k], dtype=float).reshape(-1) for k in self.keys]
        )

    def point(self, x):
        return {
            k: x[self.offsets[j] : self.offsets[j + 1]].reshape(self.shapes[j])
            for j, k in enumerate(self.keys)
        }

    def index(self, key, pos):
        return int(self.offsets[self.keys.index(key)] + pos)


def _hessian_from_grad(grad_fn, flat, x0):
    n = x0.size
    h = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = _GRAD_STEP
        h[:, i] = (
            grad_fn(flat.point(x0 + e)) - grad_fn(flat.point(x0 - e))
        ) / (2.0 * _GRAD_STEP)
    return 0.5 * (h + h.T)


def _hessian_from_logp(logp_fn, flat, x0, lp0):
    n = x0.size
    s = _LOGP_STEP
    h = np.empty((n, n))

    def lp(dx):
        return float(logp_fn(flat.point(x0 + dx)))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = s
        h[i, i] = (lp(ei) - 2.0 * lp0 + lp(-ei)) / s**2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = s
            h[i, j] = h[j, i] = (
                lp(ei + ej) - lp(ei - ej) - lp(-ei + ej) + lp(-ei - ej)
            ) / (4.0 * s * s)
    return h


class _MarginalProbe:
    """The Laplace covariance at one start, shared by every angle slot."""

    def __init__(self, model, raw_start, logp_fn, lp_seconds):
        from .polish import _compile_logp_grad

        self.flat = _Flat(raw_start, [v.name for v in model.value_vars])
        self.x0 = self.flat.flat(raw_start)
        self.cov = None
        self.reason = None
        fn = _compile_logp_grad(model)
        n = self.x0.size
        if fn is not None:

            def grad_fn(point):
                return np.concatenate(
                    [
                        np.asarray(g, dtype=float).reshape(-1)
                        for g in fn(point)[1:]
                    ]
                )

            hess = _hessian_from_grad(grad_fn, self.flat, self.x0)
            self.method = f"Hessian from {2 * n} gradient calls"
        else:
            cost = 2.0 * n * n * lp_seconds
            if cost > _FD_HESSIAN_BUDGET_S:
                self.reason = (
                    f"the model has no gradient and a finite-difference "
                    f"Hessian over its {n} coordinates would take ~{cost:.0f} s "
                    f"(budget {_FD_HESSIAN_BUDGET_S:.0f} s)"
                )
                return
            lp0 = float(logp_fn(raw_start))
            hess = _hessian_from_logp(logp_fn, self.flat, self.x0, lp0)
            self.method = f"Hessian from {2 * n * n} logp calls"
        if not np.all(np.isfinite(hess)):
            self.reason = "the Hessian at the start is not finite"
            return
        # Eigen-decomposed rather than Cholesky-inverted: a start that is not
        # a maximum along some NUISANCE direction (a polish that stopped
        # early, a multimodal coordinate) has no full Laplace covariance, but
        # the angle's marginal only needs the directions it couples to.  So
        # a non-positive curvature is fatal only if the angle's Jacobian
        # overlaps it (angle_sigma); elsewhere it is irrelevant.
        self.curv, self.vecs = np.linalg.eigh(-hess)
        self.cov = True

    def angle_sigma(self, slot):
        """The angle's marginal sigma (rad), or None with ``self.reason``."""
        if self.cov is None:
            return None
        jac = self._jacobian(slot)
        proj = self.vecs.T @ jac
        bad = self.curv <= 0.0
        overlap = np.abs(proj[bad]) / max(np.linalg.norm(jac), 1e-300)
        if np.any(overlap > _NONMAX_OVERLAP):
            self.reason = (
                "the start is not a maximum along a direction the angle "
                f"couples to (overlap {float(overlap.max()):.2g}), so no "
                "Laplace marginal exists there"
            )
            return None
        good = ~bad
        return float(np.sqrt(np.sum(proj[good] ** 2 / self.curv[good])))

    def _jacobian(self, slot):
        """d angle / d raw, over the flattened raw point."""
        xp = getattr(slot.comp, slot.x)
        yp = getattr(slot.comp, slot.y)
        xk, xi = _raw_position(xp, slot.element)
        yk, yi = _raw_position(yp, slot.element)
        jx, jy = self.flat.index(xk, xi), self.flat.index(yk, yi)
        rx, ry = self.x0[jx], self.x0[jy]
        d = 1e-6
        x = float(xp.element_phys_from_raw(slot.element, rx))
        y = float(yp.element_phys_from_raw(slot.element, ry))
        dx = float(
            xp.element_phys_from_raw(slot.element, rx + d)
            - xp.element_phys_from_raw(slot.element, rx - d)
        ) / (2 * d)
        dy = float(
            yp.element_phys_from_raw(slot.element, ry + d)
            - yp.element_phys_from_raw(slot.element, ry - d)
        ) / (2 * d)
        r2 = x * x + y * y
        jac = np.zeros(self.x0.size)
        jac[jx] = -y / r2 * dx
        jac[jy] = x / r2 * dy
        return jac


def choose_direct_angles(system, model, raw_start, logp_fn=None):
    """Decide which angle elements to sample directly, at ``raw_start``.

    Returns ``(choices, logp_fn)``: ``choices`` covers every direct element,
    forced ones included, and is EMPTY of new entries -- equal to
    ``system.direct_angles`` -- when nothing switched, in which case the
    caller keeps the model it has (and may reuse ``logp_fn``, compiled only
    if an automatic element exists).  An element switches only when BOTH its
    conditional sigma (the circle probe) and its MARGINAL sigma (the Laplace
    covariance, ``_MarginalProbe``) are below the threshold; ``raw_start``
    must be in WHITENED coordinates (run.py calls this after the probe), so
    the Hessian's finite-difference steps are well scaled.  Logs one WARNING
    per switched element, naming both sigmas, the threshold, the window and
    the override.
    """
    current = copy.deepcopy(getattr(system, "direct_angles", None) or {})
    auto = [
        s
        for s in find_slots(system)
        if s.request is None and not _is_direct(current, s)
    ]
    _log_forced(system, current)
    if not auto:
        return current, logp_fn
    if logp_fn is None:
        logp_fn = model.compile_logp()
    choices = copy.deepcopy(current)
    switched = False
    marginal = None
    for slot in auto:
        if slot.user_bounded:
            logger.info(
                "[%s] %s keeps its direction latents: the params file bounds "
                "it, and the direct coordinate's support would replace that "
                "bound with its own window.",
                slot.where,
                slot.angle,
            )
            continue
        try:
            theta0, sigma = measure_angle_sigma(
                slot, raw_start, logp_fn, threshold(slot.halfwidth)
            )
        except FloatingPointError as exc:
            logger.warning(
                "[%s] %s keeps its direction latents: the automatic "
                "coordinate probe could not evaluate the model (%s).",
                slot.where,
                slot.angle,
                exc,
            )
            continue
        if sigma is None:
            logger.debug(
                "[%s] %s: wider than %.3g rad at the start; latents kept.",
                slot.where,
                slot.angle,
                threshold(slot.halfwidth),
            )
            continue
        seeds = _seed_angles(system, slot, theta0)
        spread = max(abs(_wrap(a, theta0) - theta0) for a in seeds)
        if spread > SEED_SPREAD_FRACTION * slot.halfwidth:
            logger.warning(
                "[%s] %s is measured to %s at the start, below the %s "
                "threshold for sampling it directly, but the starting seeds "
                "disagree about it by up to %s -- a window centered on seed 0 "
                "could cut another seed's mode -- so it keeps its direction "
                "latents. Set '%s: true' on its block to force the direct "
                "coordinate anyway.",
                slot.where,
                slot.angle,
                _fmt(slot, sigma),
                _fmt(slot, threshold(slot.halfwidth)),
                _fmt(slot, spread),
                slot.flag,
            )
            continue
        if marginal is None:
            t0 = time.perf_counter()
            logp_fn(raw_start)
            marginal = _MarginalProbe(
                model, raw_start, logp_fn, time.perf_counter() - t0
            )
        sigma_m = marginal.angle_sigma(slot)
        if sigma_m is None:
            logger.warning(
                "[%s] %s is measured to %s at the start conditionally, but "
                "its MARGINAL sigma cannot be measured (%s), so it keeps its "
                "direction latents. Set '%s: true' on its block to force the "
                "direct coordinate.",
                slot.where,
                slot.angle,
                _fmt(slot, sigma),
                marginal.reason,
                slot.flag,
            )
            continue
        if sigma_m >= threshold(slot.halfwidth):
            logger.info(
                "[%s] %s is measured to %s conditionally but %s marginally "
                "(%s), above the %s threshold -- it is degenerate with "
                "another coordinate, so it keeps its direction latents.",
                slot.where,
                slot.angle,
                _fmt(slot, sigma),
                _fmt(slot, sigma_m),
                marginal.method,
                _fmt(slot, threshold(slot.halfwidth)),
            )
            continue
        _add_choice(choices, slot, theta0, sigma, sigma, False, sigma_m)
        switched = True
        _warn_switch(slot, theta0, sigma, sigma_m, marginal.method)
    if not switched:
        return current, logp_fn
    return choices, logp_fn


def has_auto_slots(system):
    """Is there any angle element the automatic switch would probe?"""
    current = getattr(system, "direct_angles", None) or {}
    return any(
        s.request is None and not _is_direct(current, s)
        for s in find_slots(system)
    )


def _is_direct(choices, slot):
    rec = choices.get(slot.key)
    return rec is not None and slot.element in rec["elements"]


def _fmt(slot, rad):
    """``rad`` (internal) in the angle Parameter's user unit, e.g. '0.131 deg'."""
    par = getattr(slot.comp, slot.angle)
    val = float(np.atleast_1d(par.from_internal(rad, index=slot.element))[0])
    return f"{val:.3g} {par.get_unit_str(slot.element)}".rstrip()


def _fmt_window(slot, center, half):
    par = getattr(slot.comp, slot.angle)
    lo = float(
        np.atleast_1d(par.from_internal(center - half, index=slot.element))[0]
    )
    hi = float(
        np.atleast_1d(par.from_internal(center + half, index=slot.element))[0]
    )
    unit = par.get_unit_str(slot.element)
    return f"[{lo:.6g}, {hi:.6g}] {unit}".rstrip()


def _warn_switch(slot, theta0, sigma, sigma_m, method):
    aspect = 0.523 / sigma
    logger.warning(
        "[%s] %s is measured to %s at the start conditionally and %s "
        "marginally (%s; threshold %s), so it is "
        "sampled DIRECTLY, uniform on the window %s centered on the start, "
        "instead of through its direction latents %s/%s, whose posterior "
        "would be a narrow ray from the origin (~%.0f:1) that a diagonal "
        "metric cannot straighten. The prior is unchanged. Set '%s: false' "
        "on the '%s' block to keep the latents ('%s: true' forces the direct "
        "coordinate).",
        slot.where,
        slot.angle,
        _fmt(slot, sigma),
        _fmt(slot, sigma_m),
        method,
        _fmt(slot, threshold(slot.halfwidth)),
        _fmt_window(slot, theta0, slot.halfwidth),
        slot.x,
        slot.y,
        aspect,
        slot.flag,
        slot.where,
        slot.flag,
    )


def _log_forced(system, choices):
    if not choices:
        return
    slots = {(s.key, s.element): s for s in find_slots(system)}
    for key, rec in choices.items():
        for j, i in enumerate(rec["elements"]):
            if not rec.get("forced", [False] * len(rec["elements"]))[j]:
                continue
            slot = slots.get((key, i))
            if slot is None:
                continue
            logger.info(
                "[%s] %s is sampled directly ('%s: true'), uniform on %s "
                "centered on its start.",
                slot.where,
                slot.angle,
                slot.flag,
                _fmt_window(slot, rec["center"][j], rec["halfwidth"][j]),
            )


def adopt_direct_angles(system, old_model, raw_start, choices):
    """Rebuild with ``choices`` applied and carry the start across.

    The caller must have LEFT ``old_model``'s context: a ``pm.Model`` built
    inside another one becomes its sub-model.  Seed 0 is carried over as a
    PHYSICAL point -- every sampled element's value in the old model, the
    switched angles as ``arctan2`` of their latents -- through the new
    model's own frozen transforms (``System.apply_polished_starts``), so the
    likelihood at the new start is the old one exactly and only the prior's
    coordinate term differs; seeds k > 0 get the angle written into
    ``config_manager.seed_resolved`` beside the latents they already carry,
    wrapped into the window.  Returns the new model.
    """
    old = {}
    for par in system.get_all_parameters():
        key = f"{par.label}_raw"
        if key in raw_start:
            old[par.label] = (par, np.asarray(raw_start[key], dtype=float))
    new_angles = {}
    for key, rec in choices.items():
        prefix, angle = key.split(".", 1)
        comp = next(
            c for c in system.active_components.values() if c.prefix == prefix
        )
        decl = direction_latents(
            system.config_manager.base_defaults, prefix, angle
        )
        xp = getattr(comp, decl["pair"][0])
        yp = getattr(comp, decl["pair"][1])
        for j, i in enumerate(rec["elements"]):
            ap = getattr(comp, angle)
            if ap.element_is_sampled(i):
                continue  # already direct in the old model (forced)
            xk, xi = _raw_position(xp, i)
            yk, yi = _raw_position(yp, i)
            x = float(
                xp.element_phys_from_raw(i, raw_start[xk].reshape(-1)[xi])
            )
            y = float(
                yp.element_phys_from_raw(i, raw_start[yk].reshape(-1)[yi])
            )
            new_angles[(key, i)] = (
                float(_wrap(np.arctan2(y, x), rec["center"][j])),
                xp.label,
                yp.label,
                decl["pair"],
                rec["center"][j],
            )

    apply_choices(system, choices)
    model = system.build_model()

    new_raw = {}
    for par in system.get_all_parameters():
        tf = getattr(par, "_raw_transform", None)
        if tf is None:
            continue
        n = par._n_elements()
        phys = np.full(n, np.nan)
        before = old.get(par.label)
        if before is not None:
            old_par, old_raw = before
            vals = old_par.phys_from_raw(old_raw)
            for i in old_par._raw_transform["sampled_idx"]:
                phys[i] = vals[i]
        for i in tf["sampled_idx"]:
            hit = new_angles.get((par.label, i))
            if hit is not None:
                phys[i] = hit[0]
        missing = [i for i in tf["sampled_idx"] if not np.isfinite(phys[i])]
        if missing:
            raise ValueError(
                f"[{par.label}] element(s) {missing} are sampled in the "
                f"rebuilt model but had no sampled value in the model it "
                f"replaces, and are not a switched angle -- the direct-angle "
                f"rebuild changed more than the angles."
            )
        new_raw[f"{par.label}_raw"] = par.raw_from_initval(phys)
    system.apply_polished_starts([new_raw], [0])

    seeds = getattr(system.config_manager, "seed_resolved", None) or []
    for k in range(1, len(seeds)):
        resolved = seeds[k]
        if resolved is None:
            continue
        for (key, i), (_theta0, _xl, _yl, pair, center) in new_angles.items():
            prefix, angle = key.split(".", 1)
            x = resolved.get(f"{prefix}.{i}.{pair[0]}")
            y = resolved.get(f"{prefix}.{i}.{pair[1]}")
            if x is None or y is None:
                continue
            resolved[f"{prefix}.{i}.{angle}"] = float(
                _wrap(np.arctan2(float(y), float(x)), center)
            )
    return model


# ---------------------------------------------------------------------------
# The reuse path
# ---------------------------------------------------------------------------
def restore_from_trace(system, trace_path):
    """Apply the direct-angle choices a saved trace was sampled under.

    Called BEFORE ``build_model`` on every path that decodes an existing
    trace's raw draws (run.py's ``recompute_trace: false``, the summary
    plot): those draws exist only in the coordinates they were sampled in.
    A trace with no record predates this module and was sampled entirely in
    latents (or with only the user-forced elements, which ``apply_forced``
    already reproduced from the same config); its absence is therefore
    exact, not a guess.  Returns the choices applied.
    """
    from .trace_meta import ANGLES_ATTR, read_trace_root_attr

    text = read_trace_root_attr(trace_path, ANGLES_ATTR)
    if text is None:
        return getattr(system, "direct_angles", None) or {}
    choices = choices_from_json(text, str(trace_path))
    apply_choices(system, choices)
    return choices
