"""
- Provide a single entrypoint to "materialize" the parameter inside a pm.Model() context: Parameter.build_pymc().
- Make user overrides explicit and predictable: users can only tighten bounds; sigma==0 means fixed/deterministic at mu/initval.
- Clean up posterior summarization using quantiles (median, +/- 1σ by default).
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pymc as pm
import pytensor
import pytensor.graph.traversal
import pytensor.tensor as pt
from astropy import units as u

from exozippy.constants import SIGMA_1_HIGH, SIGMA_1_LOW
from exozippy.manifest import normalize_selector
from exozippy.outputs.texutils import (
    DIGIT_WORDS,
    idx_to_words,
    latex_escape,
    mode_suffix,
)
from exozippy.potentials import soft_lower_bound, soft_upper_bound

logger = logging.getLogger(__name__)


class SeedBoundViolation(Exception):
    """Raised by Parameter.raw_from_initval when a seed's solved start falls
    outside a parameter's hard bounds. Multi-seed sampling skips such seeds
    rather than clipping them (a clipped start is in no posterior basin)."""


Number = Union[int, float, np.floating]

# Section C of build_pymc adds +0.5*raw**2 to exactly cancel the -0.5*raw**2
# the built-in pm.Normal(0,1) prior contributes for every logit-transformed
# raw element.  Both are real (not symbolically fused) floating-point terms,
# so the cancellation is only exact to ~machine epsilon of the LARGER term:
# once a runaway proposal pushes |raw| past this bound, squaring it loses so
# many bits that the residual grows like raw**2 * 2**-52 -- and because
# PTDE only accepts logp increases, that residual (positive by chance about
# half the time) gets selected and reinforced, driving |raw| to 1e17+ and
# the stored lp to 1e15..1e39 (observed in examples/DC2018_128 runaway PTDE
# chains). Clipping keeps the cancellation exact within float64 precision
# for any legitimate raw excursion and lets pm.Normal's own, unclipped
# -0.5*raw**2 dominate beyond it -- an ordinary restoring force instead of a
# numerical time bomb.
_RAW_CANCELLATION_CLIP = 1.0e4

# ...and it lives in a pytensor.shared, not in the graph as a literal,
# because the clip is a SAMPLER safety device that also lands on a
# MEASUREMENT path.  Past |raw| = clip the correction stops tracking
# pm.Normal's -0.5*raw**2, so logp there acquires a genuine -- and
# near-vertical: the drop reaches 0.5 nats within 0.5/clip of it -- quadratic
# wall.  The startup whitening probe walks outward from the start looking for
# the 0.5-nat contour, so for an element whose preliminary scale is too tight
# by more than ~4 orders it finds THIS wall instead of the posterior's own
# contour, reports a multiplier of exactly the clip, and leaves the model
# under-whitened with nothing anomalous to report (review 1.2.1).
# whitening.py therefore RAISES the clip for the duration of the probe (see
# _PROBE_RAW_CLIP there for the float64 argument bounding how far it may
# honestly be raised) and restores it before anything samples.  Do not fold
# it back into a literal: the sampler-time value and the probe-time value
# answer two different questions -- how far may a runaway chain wander, vs
# how far may a measurement look -- and only a shared variable can hold both
# without rebuilding the model.
_raw_cancellation_clip_sv = pytensor.shared(
    np.asarray(_RAW_CANCELLATION_CLIP, dtype="float64"),
    name="raw_cancellation_clip",
)


def get_raw_cancellation_clip():
    """The clip on |raw| currently in force in every built model's section C."""
    return float(_raw_cancellation_clip_sv.get_value())


def set_raw_cancellation_clip(value):
    """Set that clip in place (shared variable: no rebuild, no recompile).

    Returns the previous value so a caller can restore it.  The whitening
    probe is the only caller; sampling always runs at
    ``_RAW_CANCELLATION_CLIP``.
    """
    previous = get_raw_cancellation_clip()
    _raw_cancellation_clip_sv.set_value(
        np.asarray(float(value), dtype="float64")
    )
    return previous


@contextmanager
def raised_raw_cancellation_clip(value):
    """Temporarily raise the raw-cancellation clip; restore it on exit."""
    previous = set_raw_cancellation_clip(value)
    try:
        yield previous
    finally:
        set_raw_cancellation_clip(previous)


# phys_logit clips the sigmoid's argument to +/-30: sigmoid(30) = 1 - 9.4e-14,
# closer to 1.0 than float64 can distinguish for any practical downstream use.
# Every raw value that pushes |lq| past this is therefore physically identical
# to the boundary value already -- no additional posterior mass lives out
# there that isn't already accounted for at the boundary itself. Section C
# adds a penalty beyond this same threshold (_LOGIT_SATURATION_PENALTY_K
# below) so a data-unconstrained direction's chain can't wander arbitrarily
# far into that degenerate, numerically-unsafe plateau -- without touching
# the exact-uniform correction anywhere inside it.
_LOGIT_SATURATION_LQ = 30.0

# Quadratic-in-excess coefficient for the above penalty: -k*(|lq|-30)**2.
# Picked to bite gently (a few nats) just past the threshold -- consistent
# with ordinary sampling noise -- and overwhelmingly (thousands of nats) by
# |lq| ~ 100, so it stops a runaway without shifting probability mass on
# the representable [-30, 30] interior it leaves untouched.
#
# Tempering note: PTDE tempers the FULL logp (ptde.py accepts on
# dlogp / T), so a rung at temperature T sees this wall softened to a
# Gaussian of width sigma_lq = sqrt(T / (2k)) past the clip. The quadratic
# (not linear) growth is what makes the guard survive tempering at all --
# a bounded-slope penalty just rescales under 1/T. With k = 0.5 the
# hottest default rung (T_max = 200) stays within |lq| ~ 70, comparable to
# the +/-30 interior; if T_max is ever pushed to O(10^4), scale k with it
# (k ~ T_max / 400 keeps the 3-sigma excursion at the interior width).
# Raising k costs nothing statistically -- the posterior-invariance
# argument is k-independent (the wall lives entirely where phys is the
# same clipped value) -- so when in doubt, larger is safe.
_LOGIT_SATURATION_PENALTY_K = 0.5

# Preliminary whitening scale for a sampled bounded element whose
# defaults.yaml provides no init_scale: this fraction of (upper - lower).
# It only needs to land within the whitening probe's dynamic range -- the
# measured rescale (Parameter.set_whitening) replaces it before sampling.
_PRELIM_SCALE_SPAN_FRACTION = 0.1

# The labels ConfigManager.initval_source may return.  Both start-value
# errors key an advice table on one of these, so an unrecognized label is
# normalized to "default" rather than KeyError-ing while rendering the very
# message it is decorating.
_INITVAL_SOURCES = frozenset({"user", "data", "solved", "default"})


# ----------------------------
# Helper functions
# ----------------------------


def _log_normal_mass(alpha, beta):
    """log(Phi(beta) - Phi(alpha)) for standardized bounds, symbolically.

    Built from erf/erfc with a branch selected per side, never as a plain
    difference of Phi CDFs: when both bounds sit on the same tail, Phi is
    1 - eps on both and the subtraction throws away nearly every significant
    digit (the same trap galacticmodel's truncated-lognormal bracket
    documents).  Phi(x) = 0.5*erfc(-x/sqrt(2)), so on the upper tail the mass
    is a difference of two SMALL erfc values and on the lower tail the
    mirror; only a straddling interval is safe with erf.

    Both branches are evaluated (erf/erfc are finite everywhere, so the
    unselected branch cannot poison the gradient with a NaN) and the mass is
    floored at the smallest normal double before the log: an interval that
    far out is already a ~700-nat penalty, and the floor keeps the potential
    finite rather than inserting a -inf with no gradient to follow.
    """
    root2 = math.sqrt(2.0)
    a, b = alpha / root2, beta / root2
    upper_tail = 0.5 * (pt.erfc(a) - pt.erfc(b))  # 0 <= alpha
    lower_tail = 0.5 * (pt.erfc(-b) - pt.erfc(-a))  # beta <= 0
    straddling = 0.5 * (pt.erf(b) - pt.erf(a))
    mass = pt.switch(
        pt.ge(alpha, 0.0),
        upper_tail,
        pt.switch(pt.le(beta, 0.0), lower_tail, straddling),
    )
    return pt.log(pt.maximum(mass, np.finfo(float).tiny))


def _tighten_bounds(
    lower: Optional[Number],
    upper: Optional[Number],
    user_lower: Optional[Number],
    user_upper: Optional[Number],
) -> Tuple[Optional[Number], Optional[Number]]:
    """Users may only tighten bounds, never expand them."""
    if user_lower is not None:
        lower = user_lower if lower is None else max(lower, user_lower)
    if user_upper is not None:
        upper = user_upper if upper is None else min(upper, user_upper)
    return lower, upper


def _latex_varname(label: str, prefix: str = "ez") -> str:
    """
    Create a LaTeX-safe macro name from a label:
    - remove underscores and periods
    - replace digits with words (outputs.texutils.DIGIT_WORDS -- the same
      table idx_to_words uses, because this builds the <varname> half of
      the very name idx_to_words builds the <idx> half of)
    - prefix to avoid global collisions

    The digit substitutions are safe in any order: no replacement word
    contains a digit, so none of them can be re-processed by a later rule.
    """
    var = label.replace(".", "").replace("_", "")
    for digit, word in DIGIT_WORDS.items():
        var = var.replace(digit, word)
    return prefix + var


def _as_flat_array(x: Any) -> np.ndarray:
    """Flatten posterior-like input to a 1D numpy array."""
    if x is None:
        raise ValueError("posterior is None")
    # Supports xarray / arviz objects via `.values`, and raw arrays/lists.
    arr = getattr(x, "values", x)
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("posterior has zero size")
    return arr


def to_vec(val, n_elements, fill=np.nan):
    if val is None:
        return np.full(n_elements, fill, dtype=float)

    # 1. Unpack Astropy Quantities first
    # If it's a Quantity, we want the internal value (which might be a Tensor)
    raw_val = getattr(val, "value", val)

    # 2. Check if the underlying value is a Tensor
    if hasattr(raw_val, "owner") or "TensorVariable" in str(type(raw_val)):
        return raw_val

    # 3. Handle evaluate-able tensors (for initvals)
    if hasattr(raw_val, "eval"):
        try:
            raw_val = raw_val.eval()
        except:
            return np.full(n_elements, fill, dtype=float)

    arr = np.atleast_1d(raw_val)

    # 4. Handle arrays of tensors (rare, but happens in stacking)
    if arr.size > 0 and hasattr(arr[0], "eval"):
        try:
            arr = np.array(
                [
                    float(x.eval()) if hasattr(x, "eval") else float(x)
                    for x in arr
                ]
            )
        except:
            return np.full(n_elements, fill, dtype=float)

    # 5. Scalar conversion (This is where the crash was!)
    if arr.size == 1:
        # Bypass float() if it's STILL a tensor (e.g. a 1-element tensor)
        if hasattr(arr[0], "owner"):
            return arr[0]
        return np.full(n_elements, float(arr[0]), dtype=float)

    res = np.full(n_elements, fill, dtype=float)
    n_to_copy = min(n_elements, arr.size)
    res[:n_to_copy] = arr.astype(float)[:n_to_copy]
    return res


def sampled_bounds(param):
    """(lower, upper) of a Parameter's hard support as float arrays.

    Returns ``None`` when the bounds are missing, non-finite, or symbolic.
    Every caller so far is a prior NORMALIZER (the galacticmodel IMF
    branches, ``star``'s volume prior), and all of them treat ``None`` as
    "leave this prior unnormalized" -- a constant offset never changes the
    sampling, so a bound the component cannot read is not worth failing a
    fit over.

    NOTE this reads the STATIC bounds.  An element carrying a dynamic
    (linked) ``lower``/``upper`` -- ``element_links`` -- has its real
    support re-mapped inside ``build_pymc``; callers that care must check
    ``param.element_links`` themselves.
    """
    try:
        # atleast_1d: a scalar bound must still broadcast against the
        # (n_elements,) sampled vector, and np.select wants real arrays.
        lower = np.atleast_1d(np.asarray(param.lower, dtype=float))
        upper = np.atleast_1d(np.asarray(param.upper, dtype=float))
    except (AttributeError, TypeError, ValueError):
        return None

    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None
    return lower, upper


class UnitTranslator:
    # Essential "Pretty" Mapping
    SOLAR_DENSITY_UNIT = u.def_unit(
        "rho_sun", 3.0 * u.M_sun / (4.0 * np.pi * u.R_sun**3)
    )

    PRETTY_MAP = {
        u.solMass: r"M_\odot",
        u.solRad: r"R_\odot",
        u.solLum: r"L_\odot",
        u.jupiterMass: r"M_{\rm J}",
        u.jupiterRad: r"R_{\rm J}",
        u.earthMass: r"M_\oplus",
        u.earthRad: r"R_\oplus",
        u.day: r"\rm days",
        u.Gyr: r"\rm Gyr",
        u.dimensionless_unscaled: "",
        u.Unit(""): "",
        u.dex: "",
        # combined units
        u.m / u.s: r"\rm m~s^{-1}",
        u.dex(u.cm / u.s**2): r"\rm cgs",
        u.g / u.cm**3: r"\rm g~cm$^{-3}$",
        SOLAR_DENSITY_UNIT: r"\rho_\odot",
        u.erg / u.second / u.cm**2: r"\rm erg~s$^{-1}$~cm$^{-2}$",
    }

    @classmethod
    def get_latex(cls, unit, label=None):
        """Strict translator: returns pretty string or raises ValueError.

        `label` is only used to name the offending parameter in the error
        message; it is optional so the translator stays usable standalone.
        """
        # Check direct hits (handles aliases like u.R_sun vs u.solRad)
        if unit in cls.PRETTY_MAP:
            return cls.PRETTY_MAP[unit]

        # 2. Check if it's a valid Astropy unit
        try:
            # Cast to Unit object to ensure it's valid
            valid_unit = u.Unit(unit)

            # If valid, return the standard inline LaTeX string
            # We strip the $ symbols so it can be wrapped in \ensuremath or
            # placed inside existing math environments.
            return valid_unit.to_string("latex_inline").replace("$", "")

        except (TypeError, ValueError, AttributeError):
            # 3. If it's not a unit object or a string astropy understands
            where = f" for {label}" if label is not None else ""
            raise ValueError(
                f"Unit '{unit}' is not a recognized Astropy unit. "
                f"Specify valid units or set 'user_unit_latex'{where} "
                f"manually in your parameter files."
            )


# ----------------------------
# Data containers
# ----------------------------


@dataclass(slots=True)
class PosteriorSummary:
    """Numeric + formatted summary for tables."""

    median: float
    err_minus: float
    err_plus: float

    def format(self, sigfigs: int = 2) -> Tuple[str, str, str]:
        """
        Return (median_str, err_minus_str, err_plus_str) with sensible rounding:
        - errors rounded to `sigfigs` significant figures
        - median rounded to match the more precise error
        """
        if (
            math.isnan(self.median)
            or math.isnan(self.err_minus)
            or math.isnan(self.err_plus)
        ):
            return ("NaN", "NaN", "NaN")

        em = abs(self.err_minus)
        ep = abs(self.err_plus)
        if em == 0 and ep == 0:
            # A pinned element reaching this path (e.g. a Deterministic of
            # fixed inputs): full-precision repr here put
            # '0.31622776601683794 +/- 0' in the table.
            return (f"{self.median:.6g}", "0", "0")

        # Determine decimal places from error sig figs
        def decimals_from_sigfigs(val: float) -> int:
            if val == 0:
                return 0
            return -int(math.floor(math.log10(abs(val)))) + (sigfigs - 1)

        n_minus = decimals_from_sigfigs(em)
        n_plus = decimals_from_sigfigs(ep)

        # Choose a rounding for the median that matches the tighter error
        n_med = max(n_minus, n_plus)

        med_s = str(round(self.median, n_med))
        em_s = str(round(em, n_minus))
        ep_s = str(round(ep, n_plus))
        return (med_s, em_s, ep_s)

    def latex_value(self, sigfigs: int = 2) -> str:
        med_s, em_s, ep_s = self.format(sigfigs=sigfigs)
        if em_s == ep_s == "0":
            # Zero spread = effectively fixed; render like the fixed path
            # rather than 'x \pm 0'.
            return f"\\equiv {med_s}"
        if em_s == ep_s:
            return f"{med_s}\\pm{ep_s}"
        return f"{med_s}^{{+{ep_s}}}_{{-{em_s}}}"


def _broadcast_to_shape(val, shape, label, name):
    """
    Standardizes input to match 'shape'.
    - If shape is (), returns a float.
    - If shape is (N,), returns an array of length N.
    """
    if val is None:
        return None

    # Ensure it's a numpy array to handle both scalars and lists
    arr = np.atleast_1d(val)

    # 1. Scalar Case
    if shape == ():
        return float(arr[0])

    # 2. Vector Case (shape is (N,))
    n_target = shape[0]

    if arr.size == 1:
        # User gave one value (e.g. 1.0) -> Broadcast to [1.0, 1.0, ...]
        return np.full(shape, float(arr[0]))

    if arr.size == n_target:
        # User gave a vector of the correct length
        return arr.astype(float)

    raise ValueError(
        f"Dimension mismatch for {label} ({name}): "
        f"Expected scalar or length {n_target}, got {arr.size}"
    )


def _fmt_prior_value(val, is_latex=True):
    """Format one number for the Prior column (shared by every branch).

    Out-of-range magnitudes render as powers of ten -- '-1.00e+05' set the
    Prior column's width on every galactic-kinematic row, and in math mode
    the raw 'e+05' typesets as an italic e plus a spaced binary '+'.
    """
    if val is None or np.isnan(val):
        return "nan"
    if np.isinf(val):
        return (
            (r"\infty" if val > 0 else r"-\infty")
            if is_latex
            else ("inf" if val > 0 else "-inf")
        )
    if val == 0 or 0.001 <= abs(val) < 10000:
        return f"{val:.4f}".rstrip("0").rstrip(".")
    mant, exp = f"{val:.2e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp = int(exp)
    if not is_latex:
        return f"{mant}e{exp}"
    if mant == "1":
        return rf"10^{{{exp}}}"
    if mant == "-1":
        return rf"-10^{{{exp}}}"
    return rf"{mant}\times10^{{{exp}}}"


def _normalize_prior_elements(elements):
    """Element selector for a PriorContribution -> frozenset of ints, or None.

    Accepts None (every element), a boolean mask, or any iterable of indices.
    """
    if elements is None:
        return None
    arr = np.atleast_1d(np.asarray(elements))
    if arr.dtype == bool:
        return frozenset(int(i) for i in np.nonzero(arr)[0])
    return frozenset(int(i) for i in arr.ravel())


@dataclass(frozen=True, slots=True)
class PriorContribution:
    """A prior term a COMPONENT adds, declared against a Parameter.

    ``Parameter.get_prior_str`` describes a prior from the Parameter's own
    fields -- ``sigma``, ``mu``, ``lower``/``upper``.  A ``pm.Potential`` a
    component adds in stage 7 is invisible to that, so a parameter carrying
    one was reported as whatever its own fields implied, which for a bounded
    no-sigma element is "Uniform".  Three shipped priors were misreported
    that way: ``star.distance``'s d^2 volume prior, ``star.logmass``'s IMF,
    and the free-floating-planet mass function that replaces the IMF per
    star.

    A component declares one of these next to the potential it is adding
    (``Parameter.add_prior_contribution``) and both report paths -- run.py's
    startup audit table and the LaTeX ``\\...prior`` macros -- compose it
    with the parameter's own fields.  Reporting therefore stays completely
    component-agnostic: ``parameter.py`` never learns a component's name,
    and a component that adds a prior and forgets to declare it is the only
    way to get a wrong table.

    Fields:

    ``latex`` / ``text``
        The two renderings of the term.

    ``elements``
        Which elements of a vector parameter the term covers, as a frozenset
        of indices, or None for all of them.  Per-element because the choice
        genuinely is per element: ``mass_function: ffp`` swaps ONE star off
        the stellar IMF.

    ``supersedes_bounds``
        True when the term replaces the uniform-over-bounds prior the logit
        transform would otherwise imply, rather than multiplying a prior the
        Parameter states itself.  The volume prior and the IMFs are of this
        kind -- they are densities over exactly the parameter's own support,
        which they normalize over -- so the rendered text is the term plus
        that support, and the word "Uniform" never appears.  An explicit
        Gaussian ``sigma`` is NOT dropped: a parallax measurement times the
        volume prior is two statements and the table must make both.

    ``support_phrase``
        How the table NOTE joins the term to the interval, when
        ``supersedes_bounds`` is set: "<term>, <support_phrase> [lo, hi]".
        The default says "normalized on", which is exactly right for the
        volume prior and the IMFs -- they ARE normalized densities over
        that interval.  It is NOT right for every superseding term: the
        SED's soft bound on ``star.loggsed`` supersedes the same false
        "Uniform" but is a barrier at the interval's edges, not a density
        over it, and a note claiming normalization would be a wrong
        statement in the place this mechanism exists to make right ones.
    """

    latex: str
    text: str
    elements: Optional[frozenset] = None
    supersedes_bounds: bool = False
    support_phrase: str = "normalized on"


@dataclass(frozen=True)
class ElementExpression:
    """One expression and the ELEMENTS of a parameter vector it supplies.

    The per-element generalization of ``Parameter.expression``: a component
    hands ``build_pymc`` a list of these when different instances take their
    value from different physics (``ecc`` from sqrt(e)cos/sin(omega) on one
    orbit and from V_c/V_e on the next; ``mass`` derived from ``log_q`` for
    some planets and sampled linearly for others).  Elements no entry claims
    keep their own sampled coordinate.

    ``mask`` is a boolean array over the parameter's elements.  ``expr`` is a
    callable (or node) exactly as ``Parameter.expression`` is, evaluated over
    the elements the mask selects.  ``output_only`` marks a REPORTED element
    (manifest role 3): derived, consumed by nothing, and therefore given NO
    potential -- a prior or barrier on a quantity nothing reads would be a
    logp term with no data behind it.
    """

    mask: Any
    expr: Any
    output_only: bool = False
    # True when the component already SLICED the expression's dependencies to
    # this mask, so the result has one entry per selected element rather than
    # one per element of the parameter.  Slicing is how an unused instance's
    # inputs are kept out of the expression entirely (no 0*NaN from a domain
    # the other parameterization never promised); see
    # Component._element_expression, which proves the alignment before it
    # slices and verifies the sliced result numerically.
    sliced: bool = False


# ----------------------------
# Parameter
# ----------------------------


@dataclass(slots=True)
class Parameter:
    """
    A single model parameter with:
    - metadata for documentation/tables
    - optional bounds/prior specification
    - a method to create a PyMC random variable or Deterministic in-model

    IMPORTANT:
    - __init__ has no PyMC side effects.
    - Call build_pymc(...) inside a `with pm.Model():` context.
    """

    label: str
    unit: Any = None  # astropy Unit or None (kept as metadata)
    unit_latex: Optional[str] = (
        ""  # I'll keep a look up table for units, but this can be specified by the user ->
    )

    internal_unit: Any = (
        None  # this is the internally used unit that simplifies the math
    )
    initval: Optional[Number] = None
    # Preliminary whitening scale (physical units). Optional: None falls back
    # to a fraction of the bound span in build_pymc; either way the probe-based
    # rescale (set_whitening) supersedes it before sampling.
    init_scale: Optional[Number] = None
    # Optional user override for soft-bound barrier steepness (physical
    # units): the barrier's transition width is 0.01 * bound_scale.  Only
    # meaningful on elements that get a soft barrier (derived or half-bounded
    # sampled, with a finite bound); pins the element against the measured
    # update (set_barrier_scales).
    bound_scale: Optional[Number] = None
    force_node: bool = False
    names: Optional[Sequence[str]] = None
    # ACTIVITY selector (manifest `mask`): which elements are parameters of
    # their instance's parameterization at all.  Elements outside it are
    # INACTIVE (manifest role 4) -- a non-MIST star's EEP, a linear-law band's
    # u2: held at `inactive_value` (or their resolved initval) purely so the
    # vector has a number, never sampled, given no potential, and suppressed
    # from every report, because a value nothing reads is at best meaningless
    # and at worst read as a result.  None (the default, and every parameter
    # that predates the vocabulary) means every element is active.
    mask: Any = None
    # The value inactive elements are held at.  None = whatever the element
    # resolved to; set it where the other parameterization DEFINES the value
    # (u2 == 0 exactly under a linear limb-darkening law).
    inactive_value: Optional[Number] = None

    # If expression is provided, parameter becomes deterministic (pm.Deterministic).
    # You can pass expression at build time too.
    expression: Any = None
    # Per-ELEMENT expressions: a list of ElementExpression, for a vector whose
    # instances take their values from different physics.  Mutually exclusive
    # with `expression` (which is the whole-vector case and keeps its own,
    # byte-for-byte unchanged, build path).
    element_expressions: Optional[Sequence[ElementExpression]] = None
    shape: tuple = ()

    # User-defined per-element links (see linking.py), wired up by
    # Component.add_parameter: {"hard"|"mu"|"lower"|"upper": {elem_idx:
    # {"fn": callable(phys_internal_vector) -> scalar tensor (internal units),
    #  "intra_deps": set of same-parameter element indices referenced}}}.
    element_links: Any = None

    # "Physical" bounds (can be tightened by user_params, not expanded).
    lower: Optional[Number] = None
    upper: Optional[Number] = None

    # Optional Gaussian prior
    mu: Optional[Number] = None
    sigma: Optional[Number] = None

    print_to_table: bool = True
    debug_print: Optional[bool] = None
    user_modified: bool = False
    user_prior_modified: bool = False
    # Per-element role masks, written by build_pymc (see element_is_sampled,
    # element_is_derived, element_is_active).  They start as scalar False so a
    # Parameter that was never built answers conservatively.
    is_derived: Any = False
    is_sampled: Any = False
    is_reported: Any = False
    is_active: Any = True
    # Raw-space starting values for the sampled elements (set in build_pymc):
    # 0 for logit elements, (initval - mu)/sigma for Gaussian-path elements.
    raw_initval: Optional[np.ndarray] = None
    # Frozen per-element forward transform (set in build_pymc); lets
    # raw_from_initval map an alternate physical initval (a different
    # multi-seed start) to raw space using this build's bounds/scale. See
    # raw_from_initval for the dict schema.
    _raw_transform: Optional[dict] = field(default=None, init=False)
    # Whitening state (set in build_pymc when any element is sampled): the
    # pytensor.shared handles carrying the whitening scales plus the numpy
    # context set_whitening needs to update them in place. See set_whitening.
    _whiten_state: Optional[dict] = field(default=None, init=False)
    # Soft-bound barrier state (set in build_pymc when any element gets a
    # barrier): the shared barrier-scale vector plus the pinned/needs masks
    # set_barrier_scales consults. See set_barrier_scales.
    _barrier_state: Optional[dict] = field(default=None, init=False)
    # REPORTED elements awaiting their second build phase (set in build_pymc,
    # consumed by finalize_deferred): the expressions to patch in and the shape
    # to broadcast to.  None for every parameter without role-3 elements, which
    # is every parameter that does not flip a parameterization.
    _deferred_reported: Optional[dict] = field(default=None, init=False)

    user_params: Optional[Mapping[str, Mapping[str, Any]]] = None
    auto_estimated: bool = False
    # Params file these values came from, if any (Component.add_parameter
    # forwards ConfigManager.param_file).  Metadata only -- it is quoted in
    # the "pinned with no value" error so the user is told which file to edit.
    source_file: Optional[str] = None
    # Optional callable (component, param, element=i) -> "user" | "data" |
    # "solved" | "default": ConfigManager.initval_source, forwarded by
    # Component.add_parameter.  Metadata only -- it is quoted in the
    # "start value outside its hard bounds" error so the message can say
    # whether the offending number was written by the user or derived.
    initval_source: Any = None

    # LaTeX/table metadata
    latex: Optional[str] = ""
    description: Optional[str] = ""
    latex_prefix: str = "ez"

    # Runtime fields
    value: Any = field(
        default=None, init=False
    )  # pm RV or pm.Deterministic after build_pymc()
    latex_varname: str = field(default="", init=False)
    posterior: Any = (
        None  # user stores idata posterior samples here if desired
    )
    summary: Optional[PosteriorSummary] = field(default=None, init=False)
    # one entry per posterior mode (same structure as summary), filled by
    # compute_mode_summaries when a mode report exists
    mode_summaries: Optional[list] = field(default=None, init=False)
    table_note: Optional[str] = None
    # Prior terms added from OUTSIDE this Parameter -- a component's
    # pm.Potential -- declared via add_prior_contribution so the reported
    # tables can describe them. See PriorContribution.
    prior_contributions: List["PriorContribution"] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """
        Minimalist Identity Setup.
        Parses string units to Astropy objects strictly and enforces list structure.
        Applies universal unit conversion to move everything to internal math space.
        """

        def parse_u(val):
            if not isinstance(val, str):
                return val
            if val == "":
                return u.dimensionless_unscaled
            # Strict mode: Let Astropy raise an error if the string is invalid
            return u.Unit(val)

        # 1. INTERNAL UNIT: Always a single scalar
        if isinstance(self.internal_unit, str):
            self.internal_unit = parse_u(self.internal_unit)
        elif self.internal_unit is None:
            self.internal_unit = u.dimensionless_unscaled

        # 2. USER UNIT: Parse strings AND enforce list structure
        if isinstance(self.unit, str):
            self.unit = [parse_u(self.unit)]
        elif isinstance(self.unit, (list, np.ndarray)):
            self.unit = [parse_u(x) for x in self.unit]
        else:
            self.unit = [
                u.dimensionless_unscaled if self.unit is None else self.unit
            ]

        # 3. GET LATEX DISPLAY NAME (Use the first unit in the list)
        try:
            self.unit_latex = UnitTranslator.get_latex(
                self.unit[0], label=self.label
            )
        except (TypeError, ValueError, AttributeError):
            self.unit_latex = ""

        # 4. STRUCTURAL NAMING
        self.latex_varname = _latex_varname(
            self.label, prefix=self.latex_prefix
        )

        # --- 5. THE GATEKEEPER CONVERSION ---
        # Convert ALL numeric fields from User Units to Internal Units ONCE upon creation.
        # This catches YAML defaults AND dynamic data-driven kwargs.
        factors = self._get_conversion_factors()

        def convert(val):
            if val is None:
                return None

            # Unpack Astropy Quantities first
            raw_val = getattr(val, "value", val)

            # Symbolic nodes (has 'owner') cannot be numerically scaled — preserve as-is.
            # Unit conversion is a concrete operation; bounds/scales must be numeric.
            if hasattr(raw_val, "owner"):
                return raw_val

            # Evaluate constant tensor nodes (e.g. pt.constant(5.0))
            if hasattr(raw_val, "eval"):
                try:
                    raw_val = raw_val.eval()
                except Exception:
                    # Free-variable tensor that can't eval: not valid as a bound/scale
                    return np.full(np.atleast_1d(factors).shape, np.nan)

            arr = np.atleast_1d(raw_val)

            # Final check: Ensure we aren't storing an object-array of Tensors
            if arr.dtype == object:
                arr = np.array(
                    [
                        float(x.eval()) if hasattr(x, "eval") else float(x)
                        for x in arr
                    ]
                )

            return arr.astype(float) / factors

        # --- APPLY THE CONVERSION ---
        self.initval = convert(self.initval)
        self.init_scale = convert(self.init_scale)
        self.bound_scale = convert(self.bound_scale)
        self.lower = convert(self.lower)
        self.upper = convert(self.upper)
        self.mu = convert(self.mu)
        self.sigma = convert(self.sigma)

    def element_is_sampled(self, index=0):
        """True if element ``index`` is a FREE sampled element of this vector.

        ``build_pymc`` writes the per-element ``is_sampled`` mask (neither
        pinned by ``sigma: 0`` nor derived from an expression), so this is the
        authoritative answer to "will the sampler move this?" -- as opposed to
        guessing from the topology, which cannot see a user's ``sigma: 0`` or
        a component's per-element ``"overrides"`` pin.

        Callable only after the model has been built (stage 6 onwards); before
        that the mask does not exist and this conservatively returns False.
        """
        return self._element_role("is_sampled", index, default=False)

    def element_is_derived(self, index=0):
        """True if element ``index``'s value comes from an expression.

        The per-element form of ``expression is not None``, which is a
        WHOLE-VECTOR question and the wrong one for a vector whose instances
        chose different parameterizations.  REPORTED elements (role 3) count as
        derived here -- their value is an expression -- and are told apart by
        ``element_is_reported`` where the difference matters (they carry no
        potential).

        Callable only after the model has been built; before that the mask does
        not exist and this falls back to the whole-vector answer.
        """
        if not self._built_roles():
            return self.expression is not None or bool(
                self.element_expressions
            )
        return self._element_role("is_derived", index, default=False)

    def element_is_reported(self, index=0):
        """True if element ``index`` is derived but consumed by nothing."""
        return self._element_role("is_reported", index, default=False)

    def element_is_active(self, index=0):
        """False if element ``index`` is not a parameter of its instance.

        INACTIVE elements (manifest role 4) are held at a bookkeeping value and
        must be suppressed from every report; see the ``mask`` field.  Answered
        from the ``mask`` field before the build and from the build's own array
        after it, so the reporting layer gets the same answer either way.
        """
        if not self._built_roles():
            if self.mask is None:
                return True
            n = self._n_elements()
            return bool(normalize_selector(self.mask, n, self.label)[index])
        return self._element_role("is_active", index, default=True)

    def _n_elements(self):
        """Element count from ``shape`` (1 for a scalar), as build_pymc reads it."""
        actual_shape = (
            self.shape if isinstance(self.shape, tuple) else (self.shape,)
        )
        return int(np.prod(actual_shape)) if actual_shape != () else 1

    def _built_roles(self):
        """Has build_pymc written the per-element role masks yet?

        Keyed on the TYPE, not on the size: the dataclass defaults are scalar
        bools (so an unbuilt Parameter answers conservatively) and build_pymc
        replaces them with arrays.
        """
        return isinstance(getattr(self, "is_sampled", None), np.ndarray)

    def _element_role(self, attr, index, default):
        """One element's entry in a role mask, with the pre-build fallback."""
        mask = getattr(self, attr, None)
        if mask is None:
            return default
        mask = np.atleast_1d(mask)
        if mask.size == 0:
            return default
        return bool(mask[index] if mask.size > index else mask[0])

    def element_start(self, index=0):
        """Element ``index``'s start value, in INTERNAL units.

        Prefer this over ``float(param.value[index].eval())`` whenever a
        build-time constant is wanted: ``value`` of a *sampled* element is a
        random variable, so ``.eval()`` DRAWS FROM ITS PRIOR rather than
        returning the start value.  Falls back to evaluating the node only
        when ``initval`` is not a plain number (a linked/symbolic initval).
        """
        init = getattr(self, "initval", None)
        if init is not None:
            try:
                arr = np.atleast_1d(np.asarray(init, dtype=float))
            except (TypeError, ValueError):
                arr = None
            if arr is not None and arr.size:
                return float(arr[index] if arr.size > index else arr[0])
        return float(np.atleast_1d(self.value[index].eval())[0])

    def _initval_present(self, n_elements):
        """Per-element mask: does this parameter actually carry a start value?

        ``build_pymc`` reads ``initval`` through ``to_vec(..., fill=0.0)``, so
        by the time it has a vector, "no value" and "the value 0.0" look
        identical.  This answers the question *before* that fill, mirroring
        ``to_vec``'s own broadcasting rules so the mask lines up element for
        element with the vector ``to_vec`` returns:

        - ``None``          -> nothing anywhere; every element absent.
        - a symbolic node   -> a value (a linked/derived start); every element
                               present.  It is deliberately not evaluated.
        - a length-1 array  -> broadcast, so every element takes its value.
        - a longer array    -> element-wise; elements past its end are absent
                               (``to_vec`` fills them), and ``NaN`` means
                               absent (``ConfigManager.resolve`` writes NaN
                               into an array for "this element was never set").
        """
        init = self.initval
        if init is None:
            return np.zeros(n_elements, dtype=bool)
        raw = getattr(init, "value", init)
        if hasattr(raw, "owner") or "TensorVariable" in str(type(raw)):
            return np.ones(n_elements, dtype=bool)
        try:
            arr = np.atleast_1d(np.asarray(raw, dtype=float))
        except (TypeError, ValueError):
            # Not numeric and not obviously symbolic: assume it is a value.
            return np.ones(n_elements, dtype=bool)
        if arr.size == 0:
            return np.zeros(n_elements, dtype=bool)
        if arr.size == 1:
            return np.full(n_elements, not bool(np.isnan(arr[0])))
        present = np.zeros(n_elements, dtype=bool)
        n_copy = min(n_elements, arr.size)
        present[:n_copy] = ~np.isnan(arr[:n_copy])
        return present

    def get_display_label(self, index=0):
        parts = self.label.split(".")
        # If it's something like 'star.radius' (len 2) -> 'star.0.radius'
        # If it's already 'inst.gamma' -> 'inst.EXPERT.gamma'
        prefix = parts[0]
        attr = parts[-1]

        if self.names and index < len(self.names):
            return f"{prefix}.{self.names[index]}.{attr}"

        # If no names, use the index: star.0.radius
        n_elements = np.prod(self.shape).astype(int) if self.shape != () else 1
        if n_elements > 1:
            return f"{prefix}.{index}.{attr}"

        return self.label

    # What to do about an out-of-bounds start, keyed on where the number came
    # from (ConfigManager.initval_source).  The user wrote one of these
    # numbers; they did not write the other three, and telling them to "fix
    # the initval in your params file" for a value the relaxation engine
    # derived sends them looking for a line that is not there.
    _OUT_OF_BOUNDS_ADVICE = {
        "user": (
            "This start value is the 'initval' in your params file. Change it "
            "to a value inside the bounds, or -- if the value is right -- "
            "widen the bound. Note bounds may only be TIGHTENED by a params "
            "file, so a value outside the range in the component's "
            "defaults.yaml cannot be reached by editing bounds alone."
        ),
        "solved": (
            "This start value was DERIVED by the relaxation engine from your "
            "other inputs -- it is not written anywhere, so there is no line "
            "to edit. Landing outside the bound means the inputs it was "
            "solved from are inconsistent with that bound. Either seed this "
            "parameter directly with an in-bounds 'initval' (that outranks "
            "the derivation), or revisit the values it was derived from."
        ),
        "data": (
            "This start value was estimated from your DATA by the component "
            "that loaded it, not written in your params file. Either seed "
            "this parameter directly with an in-bounds 'initval' (a user "
            "value outranks a data-derived hint), or widen the bound if the "
            "estimate is right."
        ),
        "default": (
            "This start value is the default from the component's "
            "defaults.yaml, and the bound excluding it was tightened "
            "elsewhere (most likely a 'lower'/'upper' in your params file). "
            "Add an explicit 'initval' inside the new bound, or relax the "
            "bound."
        ),
    }

    def _user_constraint_fields(self, i):
        """Constraint fields the USER wrote for element ``i``, as a sorted list.

        Only ``mu``/``sigma``/``lower``/``upper`` -- the fields that state a
        posterior term or a support, as opposed to ``initval``, which is a
        start value and cannot move a posterior.  Read from the params file
        entries the ConfigManager forwarded (``user_params``), never from the
        resolved vectors: every parameter has bounds and many have a sigma from
        defaults.yaml, so a resolved value says nothing about who asked for it.

        All three spellings ConfigManager.resolve accepts are checked (index,
        instance name, and the 2-part broadcast), because a user may write any
        of them and the specific ones win.  Metadata for a warning only: any
        lookup fault degrades to "the user wrote nothing".
        """
        params = self.user_params or {}
        if not params:
            return []
        try:
            comp, pname = self.label.split(".", 1)
        except ValueError:
            return []
        keys = [f"{comp}.{int(i)}.{pname}", f"{comp}.{pname}"]
        names = self.names
        if names is not None and len(np.atleast_1d(names)) > i:
            keys.insert(1, f"{comp}.{np.atleast_1d(names)[i]}.{pname}")
        found = set()
        for key in keys:
            entry = params.get(key)
            if isinstance(entry, Mapping):
                found |= {
                    f for f in ("mu", "sigma", "lower", "upper") if f in entry
                }
        return sorted(found)

    def _element_initval_source(self, i):
        """Classify where element ``i``'s start came from.

        Returns "user", "data", "solved" or "default" (see
        ConfigManager.initval_source).  Pure metadata for an error message,
        so a fault in the lookup must never replace the diagnosis it is
        decorating: anything that goes wrong degrades to "default".  That
        includes an unrecognized label -- both callers key an advice table on
        the result, and a KeyError raised while rendering an error message
        would replace the diagnosis with a traceback about the decoration.
        """
        if not callable(self.initval_source):
            return "default"
        comp, _, pname = self.label.rpartition(".")
        name = (
            self.names[i]
            if self.names is not None and i < len(self.names)
            else None
        )
        try:
            src = self.initval_source(comp, pname, element=int(i), name=name)
        except Exception:  # noqa: BLE001 -- metadata only, never fatal
            return "default"
        return src if src in _INITVAL_SOURCES else "default"

    def _unit_suffix(self):
        """The user unit as a message suffix (e.g. ' solMass'), or ''."""
        try:
            unit_str = str(self.unit[0]) if self.unit else ""
        except (TypeError, IndexError):
            return ""
        if not unit_str or unit_str == "dimensionless":
            return ""
        return f" {unit_str}"

    def _out_of_bounds_message(self, offenders, inits, lowers, uppers):
        """Render the fatal 'start value outside its hard bounds' error.

        ``offenders`` is the list of element indices; the three arrays are the
        build's internal-unit vectors.  Values are reported in the USER unit
        so the numbers match what the user typed, and every offending element
        is listed.
        """

        def user_units(val, i):
            return self.from_internal(val, index=i)

        unit = self._unit_suffix()
        sources = set()
        lines = []
        for i in offenders:
            src = self._element_initval_source(i)
            sources.add(src)
            lines.append(
                f"  {self.get_display_label(int(i))}: start "
                f"{user_units(inits[i], i):.10g}{unit} is outside its bounds "
                f"[{user_units(lowers[i], i):.10g}, "
                f"{user_units(uppers[i], i):.10g}]{unit} (start value from: "
                f"{src})"
            )

        where = (
            f" (params file: {self.source_file})" if self.source_file else ""
        )
        advice = "\n".join(
            self._OUT_OF_BOUNDS_ADVICE[s] for s in sorted(sources)
        )
        return (
            f"Start value outside its hard bounds{where}:\n"
            + "\n".join(lines)
            + "\n"
            + f"These bounds are the parameter's SUPPORT, not a preference: "
            f"'{self.label}' is sampled through a logit transform onto "
            f"[lower, upper], so a start outside it has no raw coordinate at "
            f"all. EXOZIPPy used to move such a start onto the bound and "
            f"carry on, which produced a plausible-looking fit from a point "
            f"nobody chose; it now refuses.\n" + advice
        )

    # What to do about a SAMPLED element with no start value, keyed on the
    # provenance ConfigManager recorded for it.  For the overwhelmingly common
    # case nothing recorded anything and the label is "default"; the other
    # three mean some channel is on record as the source while no number
    # actually landed, which is worth saying out loud because it points at the
    # channel rather than at the user.
    _NO_START_ADVICE = {
        "default": (
            "Nothing supplied a start value: not your params file, not the "
            "component's defaults.yaml, not a component hint or manifest "
            "'overrides' entry, and the relaxation engine derived none. "
            "Fix: add an 'initval' for this element to your params file -- or, "
            "if every fit of this topology needs one, add an 'initval' to the "
            "parameter's entry in the component's defaults.yaml."
        ),
        "user": (
            "Your params file names this parameter, but the entry supplies no "
            "usable 'initval' -- a 'lower'/'upper', a 'sigma', or a link on "
            "some other field is not a start value. Fix: add 'initval:' to "
            "that same entry."
        ),
        "solved": (
            "The relaxation engine is on record as this element's source, but "
            "it left the element itself unsolved, so no number reached the "
            "model. Fix: seed it directly with an 'initval' (a user value "
            "outranks a derivation), or supply the inputs the derivation "
            "needs."
        ),
        "data": (
            "A data-derived hint from the component that loaded your data is "
            "on record as this element's source, but no number reached the "
            "model -- most likely the hint covered other elements of the same "
            "vector and not this one. Fix: seed this element directly with an "
            "'initval'."
        ),
    }

    def _no_start_value_message(self, offenders, lowers, uppers):
        """Render the fatal 'sampled element with no start value' error.

        Same family as the out-of-bounds error, but a genuinely different
        mistake, so it says so.  A sampled element's start is ``initval`` and
        there is no second channel for it: ``to_vec``'s ``fill=0.0`` used to
        turn "nobody said" into the number 0.0 in whatever internal unit the
        parameter happens to carry -- a start nobody chose, indistinguishable
        downstream from a deliberate one -- and where the missing value was
        spelled ``NaN`` instead it went on to build ``log(NaN/(1-NaN))`` into
        the transform, so the fit died later inside PyMC's initial-point check
        naming a raw variable rather than the parameter the user has to fix.

        Bounds are quoted (in the user unit) only where they are finite; an
        unbounded sampled element reaches this too.
        """

        def user_units(val, i):
            return self.from_internal(val, index=i)

        unit = self._unit_suffix()
        sources = set()
        lines = []
        for i in offenders:
            src = self._element_initval_source(i)
            sources.add(src)
            if np.isfinite(lowers[i]) and np.isfinite(uppers[i]):
                bounds = (
                    f" (bounds [{user_units(lowers[i], i):.10g}, "
                    f"{user_units(uppers[i], i):.10g}]{unit};"
                )
            else:
                bounds = " (unbounded;"
            lines.append(
                f"  {self.get_display_label(int(i))}: no start value"
                f"{bounds} provenance: {src})"
            )

        where = (
            f" (params file: {self.source_file})" if self.source_file else ""
        )
        advice = "\n".join(self._NO_START_ADVICE[s] for s in sorted(sources))
        return (
            f"Sampled parameter with no start value{where}:\n"
            + "\n".join(lines)
            + "\n"
            + "A sampled element has to start SOMEWHERE, and 'initval' is the "
            "only thing that says where. EXOZIPPy used to fill a missing one "
            "with 0.0 in internal units and carry on, which produced a "
            "plausible-looking fit from a point nobody chose; it now "
            "refuses.\n" + advice
        )

    def _element_expression_specs(self, expr_raw, n_elements):
        """``(per-element specs, whole-vector expression)`` for this build.

        Exactly one of the two is populated.  A single expression covering
        EVERY element -- whether it arrived as ``expression`` or as one
        all-True ``ElementExpression`` -- is returned as the whole-vector case,
        so it keeps ``build_pymc``'s original code path and produces a
        bit-identical graph; anything genuinely mixed comes back as specs.
        """
        specs = list(self.element_expressions or ())
        if expr_raw is not None and specs:
            raise ValueError(
                f"Parameter '{self.label}': both a whole-vector 'expression' "
                f"and per-element 'element_expressions' were supplied. An "
                f"element takes its value from exactly one of them; declare "
                f"the whole-vector case as a single ElementExpression if the "
                f"parameter needs both spellings."
            )
        if not specs:
            return [], expr_raw

        out = []
        for spec in specs:
            mask = normalize_selector(spec.mask, n_elements, self.label)
            if not mask.any():
                continue  # a mode nothing selected: nothing to build
            out.append(
                (mask, spec.expr, bool(spec.output_only), bool(spec.sliced))
            )
        if (
            len(out) == 1
            and not out[0][2]
            and not out[0][3]
            and bool(out[0][0].all())
            and not self._inactive_mask(n_elements).any()
        ):
            return [], out[0][1]
        return out, None

    # (verify_element_slices lives on System; see it for why both graphs are
    # kept.  Nothing about the patching below depends on that check passing --
    # it is a claim about the physics, not about the assembly.)

    def _patch_elements(self, phys_val, mask, expr, sliced):
        """Overwrite ``mask``'s elements of ``phys_val`` with ``expr``'s value.

        ``sliced`` says the expression was evaluated on dependencies already
        cut down to these elements, so its result has one entry per selected
        element; otherwise it spans the whole vector and is indexed here.  A
        scalar result broadcasts (a one-element mask, or physics that returns a
        scalar for a whole group).
        """
        val = expr() if callable(expr) else expr
        if hasattr(val, "value") and hasattr(val, "unit"):
            val = (
                val.value
            )  # strip astropy units, as the whole-vector path does
        if isinstance(val, (list, tuple)):
            val = pt.stack(list(val))
        elif isinstance(val, np.ndarray) and val.dtype == object:
            val = pt.stack(val.tolist())
        val = pt.as_tensor_variable(val)

        idx = np.nonzero(mask)[0]
        if val.ndim == 0:
            piece = (
                pt.tile(val, idx.size) if idx.size > 1 else val.reshape((1,))
            )
        elif sliced:
            piece = val
        else:
            piece = val[idx]
        return pt.set_subtensor(phys_val[idx], piece)

    def _inactive_mask(self, n_elements):
        """Boolean mask of the INACTIVE elements (the ``mask`` complement)."""
        if self.mask is None:
            return np.zeros(int(n_elements), dtype=bool)
        return ~normalize_selector(self.mask, n_elements, self.label)

    def build_pymc(self, ndx=0, expression=None):
        """
        Materializes the Parameter in the PyMC graph.

        Sampling cases:
          - Bounded (lower+upper finite): logit transform — hard bounds.
            raw ~ N(0,1), val = lower + (upper-lower)*sigmoid(logit_init + scale_logit*raw).
            raw=0 maps exactly to initval; no soft barriers. The N(0,1) raw
            density is cancelled by a correction potential, so the implied
            prior is exactly U(lower, upper); with sigma > 0 a Gaussian
            potential on the physical value makes it a truncated normal, and
            sigma sets the whitening scale.
          - Unbounded with sigma > 0: raw ~ N(0,1), val = mu + sigma * raw.
            The raw prior IS the Gaussian; no separate potential needed.
          - Unbounded with NO sigma: linear, val = initval + init_scale * raw.
            Nothing cancels the raw N(0,1) here either, so this element's
            prior IS N(initval, init_scale).  build_pymc warns about it (see
            implicit_prior_idx below), because it is the one case where
            init_scale is a modeling statement rather than conditioning.

        All raw variables are N(0,1). init_scale is always in physical units;
        for logit params it is converted to logit-space internally via the
        Jacobian and affects only tuning/conditioning, never the posterior --
        that is section C cancelling the raw N(0,1) for ANY scale, and it is
        why the startup whitening rescale is provably posterior-preserving
        THERE.  The claim is scoped to that branch and does not extend to the
        two linear ones above, where the raw N(0,1) is the prior itself; so
        on a logit element init_scale is only PRELIMINARY (the whitening
        scales live in pytensor.shared variables, and set_whitening()
        replaces them in place with the probe-measured posterior scales
        before sampling, no rebuild needed), while on a linear element
        set_whitening deliberately leaves the scale alone.

        ROLES ARE PER ELEMENT.  Every case above is chosen element by element,
        and so is whether an element is sampled at all: an expression may
        supply SOME elements of the vector (``element_expressions``) and the
        ``mask`` may declare others to be no parameter of their instance at
        all.  The whole-vector paths are preserved exactly -- all elements
        derived by one expression, or none derived -- so a build that does not
        use the per-element vocabulary produces a bit-identical graph.
        """
        import pymc as pm
        import pytensor.tensor as pt

        expr_raw = self.expression if expression is None else expression
        expr_specs, expr_raw = self._element_expression_specs(
            expr_raw, self._n_elements()
        )

        # 1. SETUP SHAPES
        actual_shape = (
            self.shape if isinstance(self.shape, tuple) else (self.shape,)
        )
        n_elements = int(np.prod(actual_shape)) if actual_shape != () else 1

        inits = to_vec(self.initval, n_elements, fill=0.0)
        scales = to_vec(self.init_scale, n_elements, fill=np.nan)
        mus = to_vec(self.mu, n_elements, fill=np.nan)
        sigmas = to_vec(self.sigma, n_elements, fill=np.nan)
        lowers = to_vec(self.lower, n_elements, fill=-np.inf)
        uppers = to_vec(self.upper, n_elements, fill=np.inf)

        # 2. IDENTIFY ROLES, PER ELEMENT
        #
        # `is_derived` covers every element whose value comes from an
        # expression, whether the whole vector shares one (the historical case)
        # or each instance names its own; `is_reported` is the subset of those
        # that nothing consumes (manifest role 3), which differ only in taking
        # no potential.  `is_inactive` is the `mask` complement: not a
        # parameter of that instance's parameterization at all (role 4), held
        # at a bookkeeping value and reported nowhere.
        is_derived = np.full(n_elements, expr_raw is not None, dtype=bool)
        is_reported = np.zeros(n_elements, dtype=bool)
        for mask, _expr, output_only, _sliced in expr_specs:
            is_derived |= mask
            if output_only:
                is_reported |= mask
        is_inactive = self._inactive_mask(n_elements)
        if np.any(is_inactive & is_derived):
            clash = np.nonzero(is_inactive & is_derived)[0].tolist()
            raise ValueError(
                f"Parameter '{self.label}': element(s) {clash} are masked out "
                f"as inactive AND claimed by an expression. An element is "
                f"either not a parameter of its instance or it has a value; "
                f"fix the component's mask or its expression selector."
            )
        # An inactive element is pinned at a value nothing reads.  Where the
        # other parameterization DEFINES that value (u2 == 0 under a linear
        # limb-darkening law) the component says so and it lands here, ahead of
        # every check below -- including the pin-must-say-what-it-pins-to one,
        # which such a pin now satisfies by construction.
        if np.any(is_inactive) and self.inactive_value is not None:
            fill = to_vec(self.inactive_value, n_elements, fill=np.nan)
            take = is_inactive & np.isfinite(fill)
            inits = np.where(take, fill, inits)
        # sigma == 0 is the ONE way for a USER to pin an element.  A tiny
        # init_scale used to pin one too (`scales <= 1e-12`), which contradicted
        # the premise that init_scale never affects the posterior -- it is a
        # preliminary whitening scale the startup probe supersedes, not a
        # modeling statement -- and gave pinning a second, undocumented
        # spelling.  An inactive element is fixed regardless of its sigma: the
        # component has said it is not a parameter here, and honoring a
        # leftover sigma would sample a dimension no likelihood term reads.
        is_fixed = ((sigmas == 0) | is_inactive) & ~is_derived
        is_sampled = ~(is_fixed | is_derived)

        # A PIN MUST SAY WHAT IT PINS TO.  `sigma: 0` is the one way to fix an
        # element, and there is no second channel for the value it is fixed
        # AT: the physical value of a fixed element is exactly inits[i], and
        # to_vec fills a missing initval with 0.0.  So an element pinned with
        # no value from ANY source -- the params file, defaults.yaml, a
        # component "overrides" entry or hint, a link, or the relaxation
        # engine's solution -- is silently held at zero in whatever internal
        # unit it happens to carry, and nothing downstream can tell that apart
        # from a deliberate pin at zero.  It also cannot be reported: for a
        # fixed element with no initval to_latex_def emits no macro at all
        # while latex.py's _value_cells still references one, so the generated
        # table is an undefined control sequence by construction.
        #
        # Refuse, for the same reason validate_sigma_has_center refuses a
        # sigma with no center: it does not describe a model.  If the user is
        # fixing a parameter they should know what they are fixing it to.
        #
        # This runs at stage 6, which is deliberate: it is the earliest point
        # that sees EVERY channel a value can arrive through.  The manifest
        # "overrides" channel that pins whole vectors (GP, robust likelihood,
        # band LD) and the plain manifest options are both applied inside
        # this stage; a check at ConfigManager construction or at stage 4
        # would have to guess about them and would fire falsely.
        #
        # Three exemptions, all because the value comes from somewhere other
        # than initval, or because nothing reads it:
        #   - DERIVED elements: their value is the expression.  `sigma: 0`
        #     there is a no-op, already warned about below -- a different
        #     mistake with a different fix, so it keeps its own message.
        #   - HARD-LINKED elements: the link expression IS the value.
        #   - INACTIVE elements: the pin is bookkeeping for a parameter that
        #     does not exist on that instance.  The error's whole argument is
        #     that a pinned value is a modeling statement nobody made -- but
        #     here nothing reads the value, nothing reports it, and the user
        #     never asked for the pin, so there is no statement to get wrong
        #     and no fix to advise.  (Where the value IS defined, the component
        #     supplies `inactive_value` and the exemption never applies.)
        has_value = self._initval_present(n_elements)
        hard_linked = set((self.element_links or {}).get("hard", {}))
        pinned_no_value = [
            i
            for i in np.where(is_fixed & ~has_value & ~is_inactive)[0]
            if i not in hard_linked
        ]
        if pinned_no_value:
            where = (
                f" (params file: {self.source_file})"
                if self.source_file
                else ""
            )
            offenders = ", ".join(
                self.get_display_label(int(i)) for i in pinned_no_value
            )
            raise ValueError(
                f"Pinned parameter with no value{where}: {offenders}. "
                f"'sigma: 0' fixes a parameter, but no start value was "
                f"supplied for it anywhere -- not in the params file, not in "
                f"{self.label.split('.')[0]}/defaults.yaml, and nothing "
                f"derived one -- so it would be held at 0.0 in internal "
                f"units, a value nobody chose. "
                f"Fix: add an explicit 'initval' (the value you mean to fix "
                f"it at) to the same params-file entry as the 'sigma: 0', or "
                f"drop the 'sigma: 0' and let it be fitted."
            )

        # A SAMPLED ELEMENT MUST SAY WHERE IT STARTS.  The sibling of the pin
        # check above, and for the same reason: `initval` is the ONLY channel
        # for a start value, and `to_vec`'s `fill=0.0` turns "nobody said" into
        # the number 0.0 in whatever internal unit the parameter carries.  For
        # a pinned element that is the whole answer; for a sampled one it is
        # the chain's starting point, the point the whitening probe measures
        # around and the point every multi-seed start is derived from -- and
        # 0.0 is a perfectly ordinary-looking value, so nothing downstream can
        # tell it apart from a start the user chose.  Where the missing value
        # arrives spelled NaN instead (ConfigManager.resolve writes NaN into a
        # vector for "this element was never set"), the logit branch built
        # log(NaN/(1-NaN)) and the fit died much later inside PyMC's
        # initial-point check, naming a raw variable instead of the parameter.
        #
        # Stage 6 for the same reason the pin check is: it is the earliest
        # point that sees EVERY channel a value can arrive through --
        # defaults.yaml, the params file, a component hint, the manifest
        # "overrides" and "options" channels, and the relaxation engine's
        # solution have all landed in `initval` by now, so a check here cannot
        # fire falsely on a value that was simply going to arrive later.
        #
        # No exemption list is needed here, unlike the pin check.  A DERIVED
        # element is excluded already (`is_sampled` is false for it), and so
        # is a HARD-LINKED one: Component._wire_user_links only classifies an
        # initval link as "hard" when that element's sigma is 0, so a hard
        # link implies a pin and the pin check above owns it.  A SOFT link
        # (an initval link with sigma > 0, or a `mu` link) does reach here,
        # and should: it adds a Gaussian potential tying the element to an
        # expression, but the element is still sampled and still has to start
        # somewhere.  A symbolic initval counts as present and is deliberately
        # not evaluated.
        sampled_no_value = [
            int(i) for i in np.where(is_sampled & ~has_value)[0]
        ]
        if sampled_no_value:
            raise ValueError(
                self._no_start_value_message(sampled_no_value, lowers, uppers)
            )

        # init_scale is a PRELIMINARY whitening scale only (the probe-based
        # rescale in set_whitening supersedes it), so it is optional: a
        # missing entry falls back to a fraction of the bound span, or sigma
        # when unbounded.  Non-sampled elements just need a finite
        # placeholder (a NaN scale would poison phys_linear via NaN * raw=0).
        # A non-POSITIVE scale takes the same fallback: a whitening scale of
        # zero is not a scale, it is a degenerate raw direction the sampler
        # cannot move (and it used to be silently reinterpreted as a pin --
        # the `scales <= 1e-12` clause deleted above).  The user's sigma is
        # synced into init_scale, so `sigma: 0` lands here; that element is
        # already is_fixed and the placeholder never reaches the posterior.
        for i in np.where(~(np.isfinite(scales) & (scales > 0)))[0]:
            if np.isfinite(lowers[i]) and np.isfinite(uppers[i]):
                scales[i] = _PRELIM_SCALE_SPAN_FRACTION * (
                    uppers[i] - lowers[i]
                )
            elif not np.isnan(sigmas[i]) and sigmas[i] > 0:
                scales[i] = sigmas[i]
            else:
                scales[i] = 1.0

        # Warn if user tried to fix a derived parameter — sigma=0 has no effect on derived params.
        if np.any(is_derived & (sigmas == 0)):
            logger.warning(
                f"Parameter '{self.label}': sigma=0 has no effect on a derived parameter "
                f"To hold it constant, you must fix the corresponding sampled parameter(s)."
            )
        # A CONSTRAINT ON AN INACTIVE ELEMENT IS DROPPED, so say so.  This is
        # the one genuinely lossy case in a parameterization switch: a prior or
        # a bound on an element that flipped to DERIVED still applies (section
        # A's Gaussian, section B's barrier), and a start value still feeds the
        # relaxation engine -- but an element that is no longer a parameter at
        # all has nothing to carry the constraint, so the user has to know.
        # Deliberately not an error: the point of per-element roles is that one
        # params file can be carried across a parameterization toggle.
        for i in np.nonzero(is_inactive)[0]:
            fields = self._user_constraint_fields(int(i))
            if not fields:
                continue
            where = f" ({self.source_file})" if self.source_file else ""
            logger.warning(
                f"Parameter '{self.get_display_label(int(i))}': your "
                f"{'/'.join(fields)}{where} is DROPPED -- this element is not "
                f"a parameter of its instance's parameterization, so it is "
                f"held at a bookkeeping value, given no prior, and reported "
                f"nowhere. Put the constraint on the quantity this instance "
                f"actually samples, or change the instance's parameterization "
                f"if you meant to fit it."
            )
        self.is_sampled = is_sampled
        self.is_derived = is_derived
        self.is_reported = is_reported
        self.is_active = ~is_inactive

        if np.any(is_sampled):
            if self.lower is None or self.upper is None:
                raise ValueError(
                    f"Developer Error: Sampled parameter '{self.label}' MUST have explicit "
                    f"'lower' and 'upper' bounds defined in its defaults.yaml."
                )

        # A START OUTSIDE THE HARD BOUNDS IS FATAL.  Two finite bounds mean the
        # element is logit-transformed below and [lower, upper] IS its support
        # -- there is no representable raw coordinate for a value outside it,
        # and the transform's own inverse diverges at the wall.  The old code
        # clipped such a start onto the wall (np.clip on q) behind a warning
        # that described a different, benign situation, so a fit that began
        # somewhere the user never asked for looked exactly like a fit that
        # began where they did.  Refuse instead: a start value nobody chose is
        # not a model, and no amount of sampling recovers the fact that the
        # question asked was not the question answered.
        #
        # Deliberately NOT covered here:
        #   - SOFT barriers (a single finite bound, or a derived element):
        #     those are penalties, not support.  A start on the wrong side of
        #     one is legal and merely improbable, and the barrier's gradient
        #     is what pulls it back.
        #   - EXACTLY ON a bound: representable as a physical value, just
        #     infinitely far away in logit space, so such a start HAS to move.
        #     Section 3 nudges those inward to the q_floor and logs the exact
        #     displacement; see the q_floor comment there.
        #   - FIXED elements (`sigma: 0`): not sampled, so no transform and no
        #     barrier ever reads their bounds.  Nothing is clipped there.
        # Every offending element of a vector is reported, not just the first.
        #
        # A NON-FINITE start never reaches here: NaN satisfies no bound either,
        # but "you asked to start outside the bounds" is the wrong diagnosis
        # for "nothing gave this element a start at all", and the fixes differ.
        # The no-start-value check above catches every spelling of that
        # (missing, NaN, or short of the vector's length) before this one runs.
        two_finite = np.isfinite(lowers) & np.isfinite(uppers)
        checkable = is_sampled & two_finite & (uppers > lowers)
        bound_violations = [
            int(i)
            for i in np.where(checkable)[0]
            if not (lowers[i] <= inits[i] <= uppers[i])
        ]
        if bound_violations:
            raise ValueError(
                self._out_of_bounds_message(
                    bound_violations, inits, lowers, uppers
                )
            )

        # 3. PER-ELEMENT PARAMETERIZATION
        # use_logit[i]: finite bounds → logit transform (hard bounds). A sigma
        #   prior on a bounded element is applied as a Gaussian potential on
        #   the physical value (section A), giving truncated-normal semantics.
        # has_sigma_prior[i]: explicit Gaussian prior (sigma > 0)
        use_logit = np.zeros(n_elements, dtype=bool)
        has_sigma_prior = np.zeros(n_elements, dtype=bool)

        # Logit transform: logit_q_init + init_scale_logit * raw → sigmoid → physical
        logit_q_inits = np.zeros(n_elements)
        init_scale_logits = np.zeros(n_elements)
        # Per-element clip floor on q = (val-lower)/span; stored so
        # raw_from_initval can re-derive a raw start for an alternate initval
        # (multi-seed sampling) using the SAME transform as this build.
        q_floors = np.zeros(n_elements)

        # Gaussian: val = gaussian_mus + gaussian_scales * raw
        gaussian_mus = np.copy(inits)
        gaussian_scales = np.copy(scales)

        # Sampled elements with neither two finite bounds nor a sigma: their
        # prior is the uncancelled raw N(0,1), i.e. N(initval, init_scale).
        implicit_prior_idx = []

        for i in range(n_elements):
            if not is_sampled[i]:
                continue

            has_sigma = not np.isnan(sigmas[i]) and sigmas[i] > 0
            has_bounds = not np.isinf(lowers[i]) and not np.isinf(uppers[i])
            has_sigma_prior[i] = has_sigma

            if has_bounds:
                use_logit[i] = True
                span = uppers[i] - lowers[i]
                if span <= 0:
                    raise ValueError(
                        f"Parameter '{self.label}'[{i}]: lower bound equals or exceeds "
                        f"upper bound ({lowers[i]} >= {uppers[i]}). To hold a parameter "
                        f"at a fixed value, set 'sigma: 0' instead of collapsing the bounds."
                    )
                q_raw = (inits[i] - lowers[i]) / span
                # Use the tighter of sigma and init_scale as the whitening scale.
                # Section C cancels the raw N(0,1) prior (leaving a flat prior in
                # physical space), so the prior shape is determined solely by the
                # Gaussian potential in section A — always N(mu, sigma) regardless
                # of whiten.  Using min(sigma, init_scale) makes chain initialization
                # spread by init_scale in physical space when init_scale < sigma
                # (e.g. xalpha/yalpha where sigma=1 encodes a uniform-angle
                # prior but init_scale reflects the actual alpha uncertainty).
                whiten = min(sigmas[i], scales[i]) if has_sigma else scales[i]
                # Keep the start off the exact bound. The floor is in units of
                # the whitening scale (1e-6*scale inside the bound is
                # "essentially at the bound" in problem units); a span-based
                # floor would be arbitrarily large for wide bounds. The 1e-12
                # absolute floor keeps logit(q) within the ±30 sigmoid clip.
                #
                # q_raw is guaranteed to be in [0, 1] here: the pre-pass above
                # made anything outside the bounds fatal.  So this clip is
                # exactly and only the ON-THE-BOUND case -- a value that IS in
                # the support but sits at (or unrepresentably close to) a
                # wall, where logit(q) diverges and there is no raw coordinate
                # to start from.  Such a start HAS to move; a start from
                # outside does not have to be moved, it has to be refused,
                # which is why one warns and the other raises.  A default that
                # sits on its own bound (an angle defaulting to 0 on a
                # [0, 2pi) range) is common and legitimate; raising on it
                # would be noise.  The warning reports the actual displacement
                # rather than a rule of thumb: the floor is the LARGER of
                # 1e-6 * whitening scale and 1e-12 * span, and on a parameter
                # whose span dwarfs its scale (transit jitter_variance, span
                # 1e5, scale ~1e-8) it is the span term that binds, so the
                # move can be a sizeable fraction of the start value itself.
                q_floor = min(max(1e-6 * whiten / span, 1e-12), 0.25)
                q_floors[i] = q_floor
                q_init = np.clip(q_raw, q_floor, 1.0 - q_floor)
                if q_init != q_raw:
                    nudged_to = lowers[i] + q_init * span
                    logger.warning(
                        f"Parameter '{self.label}'[{i}]: start value "
                        f"{inits[i]} sits on (or unrepresentably close to) "
                        f"its bounds [{lowers[i]}, {uppers[i]}], where the "
                        f"logit transform diverges and no raw start exists; "
                        f"nudged inward to {nudged_to}, a move of "
                        f"{abs(nudged_to - inits[i]):.3g} (internal units). "
                        f"This is the ON-THE-BOUND case only -- a start "
                        f"OUTSIDE the bounds is refused outright, never "
                        f"clipped. Set an 'initval' further inside the bound "
                        f"to remove this nudge."
                    )
                logit_q_inits[i] = np.log(q_init / (1.0 - q_init))
                jac = (
                    q_init * (1.0 - q_init) * span
                )  # dval/d(logit_q) at initval
                # Near a wall jac → 0 and whiten/jac would explode, saturating
                # the sigmoid within one tiny raw step (parameter frozen at the
                # wall). Flooring jac at min(whiten, span/4) caps the logit
                # step at ~1, so a pinned start escapes multiplicatively —
                # one e-fold in (val - bound) per unit raw step — while
                # interior starts are unaffected.
                init_scale_logits[i] = whiten / max(
                    jac, min(whiten, span / 4.0)
                )
            elif has_sigma:
                # Unbounded with sigma: non-centered Gaussian; the raw N(0,1)
                # IS the prior.
                has_mu = not np.isnan(mus[i])
                gaussian_mus[i] = mus[i] if has_mu else inits[i]
                gaussian_scales[i] = sigmas[i]
            else:
                # No two finite bounds and no sigma: fall back to linear with
                # N(0,1).  Nothing cancels that raw prior, so this element's
                # prior IS N(initval, init_scale) -- the one place where
                # init_scale is a posterior term rather than pure
                # conditioning.  set_whitening therefore refuses to rescale
                # it (a data-measured multiplier would make the prior width
                # data-dependent), so say so once per parameter.
                gaussian_mus[i] = inits[i]
                gaussian_scales[i] = scales[i]
                implicit_prior_idx.append(i)

        if implicit_prior_idx:
            logger.warning(
                f"Parameter '{self.label}': element(s) {implicit_prior_idx} "
                f"are sampled with no finite lower/upper pair and no sigma, "
                f"so their prior is N(initval, init_scale) from defaults -- "
                f"init_scale is a real prior width here, and the whitening "
                f"probe deliberately leaves it alone. Give the element two "
                f"finite bounds (uniform prior) or a sigma (explicit "
                f"Gaussian prior) to state the prior yourself."
            )

        # 4. BUILD RAW VARIABLES
        raw_elements = [None] * n_elements

        # Fixed / derived: constant 0 in raw space
        for i in np.where(is_fixed | is_derived)[0]:
            raw_elements[i] = pt.constant(0.0)

        if np.any(is_sampled):
            idx = np.where(is_sampled)[0]
            # Start each raw element so the physical value equals initval.
            # Logit elements: raw=0 maps to initval by construction.
            # Gaussian elements: val = mu + sigma*raw, so raw must start at
            # (initval - mu)/sigma (0 when mu is absent, since mu falls back
            # to initval).  The prior stays exactly N(mu, sigma); only the
            # starting point moves.
            raw_initvals = np.zeros(len(idx))
            for j, i in enumerate(idx):
                if not use_logit[i]:
                    raw_initvals[j] = (inits[i] - gaussian_mus[i]) / max(
                        gaussian_scales[i], 1e-30
                    )
            # Saved so run.py can override model.initial_point() with the correct raw start.
            self.raw_initval = raw_initvals
            # Freeze the per-element forward transform so raw_from_initval can
            # map an ALTERNATE physical initval (a different seed) to raw space
            # using exactly the bounds/scale this build used. Only the start
            # moves between seeds; the transform (and hence bounds) is fixed.
            self._raw_transform = {
                "sampled_idx": idx,
                "use_logit": use_logit.copy(),
                "lowers": lowers.copy(),
                "uppers": uppers.copy(),
                "logit_q_inits": logit_q_inits.copy(),
                "init_scale_logits": init_scale_logits.copy(),
                "q_floors": q_floors.copy(),
                "gaussian_mus": gaussian_mus.copy(),
                "gaussian_scales": gaussian_scales.copy(),
            }
            par_raw = pm.Normal(
                f"{self.label}_raw",
                mu=0,
                sigma=1.0,
                shape=len(idx),
                initval=raw_initvals,
            )
            for j, actual_idx in enumerate(idx):
                raw_elements[actual_idx] = par_raw[j]

        # 5. RECONSTRUCT PHYSICAL VALUE
        raw_vector = pt.stack(raw_elements)

        if expr_raw is not None:
            phys_val = expr_raw() if callable(expr_raw) else expr_raw
        else:
            # The whitening constants enter the graph as pytensor.shared
            # variables (not baked constants) when anything is sampled, so
            # set_whitening can replace the preliminary scales with the
            # measured ones in place -- every function compiled from this
            # model picks up the new values without a rebuild.  The posterior
            # is invariant to these values by construction: section C cancels
            # the raw N(0,1) prior symbolically for ANY scale, so they affect
            # conditioning only.
            if np.any(is_sampled):
                sv_logit_q_inits = pytensor.shared(
                    logit_q_inits.astype(float),
                    name=f"{self.label}_logit_q_init",
                    shape=logit_q_inits.shape,
                )
                sv_scale_logits = pytensor.shared(
                    init_scale_logits.astype(float),
                    name=f"{self.label}_scale_logit",
                    shape=init_scale_logits.shape,
                )
                sv_gaussian_scales = pytensor.shared(
                    gaussian_scales.astype(float),
                    name=f"{self.label}_gaussian_scale",
                    shape=gaussian_scales.shape,
                )
                self._whiten_state = {
                    "sv_logit_q_inits": sv_logit_q_inits,
                    "sv_scale_logits": sv_scale_logits,
                    "sv_gaussian_scales": sv_gaussian_scales,
                    "has_sigma_prior": has_sigma_prior.copy(),
                }
            else:
                sv_logit_q_inits = pt.as_tensor_variable(logit_q_inits)
                sv_scale_logits = pt.as_tensor_variable(init_scale_logits)
                sv_gaussian_scales = pt.as_tensor_variable(gaussian_scales)

            # Logit branch: lower + (upper-lower)*sigmoid(logit_init + scale_logit*raw)
            #
            # The bound constants are SANITIZED on the non-logit elements
            # (dummy [0, 1]) before they enter the graph.  Their real bounds
            # are infinite there, and -inf + inf*sigmoid = NaN (or +inf for a
            # half-bounded element): a NaN/inf sitting in the UNSELECTED
            # branch of the pt.where below, which is the where-trap -- the
            # switch VJP multiplies it by a zero and 0*inf = NaN poisons the
            # gradient of the whole vector on every backend.  (A
            # canonicalization rewrite currently sinks that zero into the
            # switch and hides it, but a rewriter is not a correctness
            # guarantee; with rewrites off the NaN is right there.)  Section
            # A already sanitizes its sigmas the same way.  keep_bounds is a
            # superset of use_logit, so every logit element is untouched and
            # the selected branch is bit-for-bit what it always was.
            keep_bounds = use_logit | (
                np.isfinite(lowers) & np.isfinite(uppers)
            )
            safe_lowers = np.where(keep_bounds, lowers, 0.0)
            safe_uppers = np.where(keep_bounds, uppers, 1.0)
            lq = sv_logit_q_inits + sv_scale_logits * raw_vector
            phys_logit = pt.as_tensor_variable(
                safe_lowers
            ) + pt.as_tensor_variable(safe_uppers - safe_lowers) * pt.sigmoid(
                pt.clip(lq, -_LOGIT_SATURATION_LQ, _LOGIT_SATURATION_LQ)
            )

            # Gaussian / linear branch: mu + sigma * raw  (or initval + scale * raw)
            phys_linear = (
                pt.as_tensor_variable(gaussian_mus)
                + sv_gaussian_scales * raw_vector
            )

            if np.all(use_logit):
                phys_val = phys_logit
            elif not np.any(use_logit):
                phys_val = phys_linear
            else:
                phys_val = pt.where(
                    pt.as_tensor_variable(use_logit), phys_logit, phys_linear
                )

            # 5a. PER-ELEMENT EXPRESSIONS.  The transform above already
            # supplied every element (a derived element's raw is the constant
            # 0, so its slot holds a harmless finite number); each expression
            # now overwrites the elements it supplies.
            #
            # pt.set_subtensor, never pt.where over the two VALUE vectors: an
            # expression evaluated at an unused element's bookkeeping pin may
            # legitimately be NaN (sqrt of a negative eccentricity the other
            # parameterization never promised), and where's VJP multiplies the
            # unselected branch by zero -- 0*NaN poisons the gradient of the
            # whole vector on every backend.  set_subtensor keeps the unused
            # entries out of the output entirely, and the component's own
            # dependency slicing (Component._element_expression) keeps them out
            # of the expression in the first place wherever it can prove the
            # alignment.
            # REPORTED elements (role 3) are deliberately NOT patched here --
            # see finalize_deferred.  Their expressions read quantities that,
            # on other elements, are derived from THIS parameter, so they can
            # only be built once every parameter exists.
            for mask, expr, output_only, sliced in expr_specs:
                if output_only:
                    continue
                phys_val = self._patch_elements(phys_val, mask, expr, sliced)

        # Strip Astropy units
        if hasattr(phys_val, "value") and hasattr(phys_val, "unit"):
            phys_val = phys_val.value

        if isinstance(phys_val, (list, tuple)):
            phys_val = pt.stack(list(phys_val))
        elif isinstance(phys_val, np.ndarray) and phys_val.dtype == object:
            phys_val = pt.stack(phys_val.tolist())

        # 5b. USER-DEFINED ELEMENT LINKS (dynamic bounds + hard links)
        links = self.element_links or {}
        dyn_bounds = {}  # element index -> (lo_t, up_t, span_t) tensors
        if links:
            if expr_raw is not None and any(
                k in links for k in ("hard", "lower", "upper")
            ):
                raise ValueError(
                    f"Parameter '{self.label}': hard/bound links are not supported "
                    f"on derived (expression) parameters; only 'mu' links are."
                )

            # Dynamic bounds: re-map the element's sigmoid coordinate q into
            # the tensor-valued interval.  q comes from the same logit raw
            # coordinate, so the bound is a hard constraint by construction.
            dyn_idx = sorted(
                set(links.get("lower", {})) | set(links.get("upper", {}))
            )
            for i in dyn_idx:
                if not use_logit[i] and is_sampled[i]:
                    raise ValueError(
                        f"Parameter '{self.label}'[{i}]: a dynamic bound link "
                        f"requires finite static lower/upper bounds (used to "
                        f"set up the logit transform)."
                    )
                lo_t = (
                    links["lower"][i]["fn"](phys_val)
                    if i in links.get("lower", {})
                    else pt.constant(lowers[i])
                )
                up_t = (
                    links["upper"][i]["fn"](phys_val)
                    if i in links.get("upper", {})
                    else pt.constant(uppers[i])
                )
                span_t = pt.maximum(up_t - lo_t, 1e-12)
                q_i = pt.sigmoid(
                    pt.clip(lq[i], -_LOGIT_SATURATION_LQ, _LOGIT_SATURATION_LQ)
                )
                phys_val = pt.set_subtensor(phys_val[i], lo_t + span_t * q_i)
                # Kept for section A3: with a sigma the conditional prior is a
                # TRUNCATED normal, whose mass depends on these tensors.
                dyn_bounds[i] = (lo_t, up_t, span_t)
                # NO -log(span) normalization term here, deliberately.  The
                # reparameterization already supplies it: with lq = c + s*raw
                # and section C cancelling the raw N(0,1), the raw-space
                # density is q(1-q), and dval/draw = span*q*(1-q)*s, so
                # p(val) = 1/(sqrt(2pi)*s*span) -- exactly U(lo, up), whose
                # integral over the interval is independent of span for ANY
                # span, dynamic or not.  Adding -log(span) would multiply the
                # joint by another 1/span and reward the bound-source
                # parameter for shrinking the interval (an ordering link
                # lower: star.B.av over av in [0, 100] would give av_B a
                # spurious 1/(100 - av_B) factor pushing it to the wall).

            # Hard links (initval link with sigma=0): the element deterministically
            # tracks its expression.  Same-parameter references are applied in
            # dependency order so chains (A := f(B), B := g(C)) resolve correctly.
            hard = links.get("hard", {})
            if hard:
                import graphlib

                intra_graph = {
                    i: (set(spec.get("intra_deps", ())) & set(hard))
                    for i, spec in hard.items()
                }
                try:
                    order = list(
                        graphlib.TopologicalSorter(intra_graph).static_order()
                    )
                except graphlib.CycleError as e:
                    raise ValueError(
                        f"Parameter '{self.label}': circular hard links between "
                        f"elements: {e}"
                    )
                for i in order:
                    phys_val = pt.set_subtensor(
                        phys_val[i], hard[i]["fn"](phys_val)
                    )

        # 6. ASSIGN TO SELF.VALUE
        track_node = bool(np.any(is_sampled)) or self.force_node or bool(links)

        if actual_shape == ():
            val_to_save = phys_val if expr_raw is not None else phys_val[0]
        else:
            val_to_save = pt.broadcast_to(
                pt.as_tensor_variable(phys_val), actual_shape
            )

        # REPORTED elements defer BOTH their patch and the Deterministic (see
        # finalize_deferred).  Consumers built between here and there read this
        # phase's tensor, which is correct by construction: every element they
        # could read is identical in both phases, because a reported element is
        # consumed by nothing.  A Deterministic created now would record the
        # unpatched vector, so it waits.
        deferred = [
            (mask, expr, sliced)
            for mask, expr, output_only, sliced in expr_specs
            if output_only
        ]
        if deferred:
            # `expr` is None for the ordinary path: Component.add_parameter
            # hands over the mask now and the wiring later (see
            # finalize_reported), because resolving a reported expression's
            # deps mid-build would recurse.  A spec that DOES carry its
            # expression (a Parameter built directly, as the tests do) is
            # applied by finalize_deferred with no argument.
            self._deferred_reported = {
                "specs": deferred,
                "shape": actual_shape,
            }
            self.value = val_to_save
        elif track_node:
            self.value = pm.Deterministic(self.label, val_to_save)
        else:
            self.value = val_to_save

        # 7. PRIORS AND SOFT BOUNDS
        val_flat = pt.flatten(self.value)

        # A. Gaussian potential on the physical value for:
        #    - derived parameters with sigma, and
        #    - bounded (logit-transformed) sampled parameters with sigma, whose
        #      raw N(0,1) is cancelled by section C → uniform × this Gaussian
        #      = truncated normal.
        #    Unbounded sampled Gaussian params encode their prior in raw ~
        #    N(0,1); no double-count.
        #    REPORTED elements (role 3) are excluded even though they are
        #    derived: nothing consumes them, so a prior there would be a logp
        #    term on a quantity the model never uses -- and the same statement
        #    is already being made on the coordinate that instance samples.
        gaussian_prior_mask = (
            (
                (is_derived & ~is_reported)
                | (is_sampled & use_logit & has_sigma_prior)
            )
            & ~np.isnan(sigmas)
            & (sigmas > 0)
        )
        # Elements with a dynamic (linked) prior center get their Gaussian
        # potential below with a tensor-valued mu — exclude them here so the
        # penalty is not double-counted against the static center.
        mu_links = links.get("mu", {}) if links else {}
        for i in mu_links:
            gaussian_prior_mask[i] = False
        prior_mus = np.where(~np.isnan(mus), mus, inits)
        if np.any(gaussian_prior_mask):
            mask = pt.as_tensor_variable(gaussian_prior_mask)
            penalty = (
                -0.5
                * (
                    (val_flat - pt.as_tensor_variable(prior_mus))
                    / pt.as_tensor_variable(np.where(sigmas > 0, sigmas, 1.0))
                )
                ** 2
            )
            pm.Potential(
                f"gaussian_prior.{self.label}",
                pm.math.sum(pt.where(mask, penalty, 0.0)),
            )

        # A2. Gaussian potentials with LINKED (tensor-valued) centers: soft
        #     links tie this element to an expression of other parameters,
        #     penalizing the difference at every step of the chain.
        for i, spec in mu_links.items():
            sig_i = sigmas[i]
            if np.isnan(sig_i) or sig_i <= 0:
                raise ValueError(
                    f"Parameter '{self.label}'[{i}]: a soft link (Gaussian "
                    f"penalty on a linked center) requires sigma > 0; got "
                    f"sigma={sig_i}. Use sigma: 0 for a hard link."
                )
            mu_t = spec["fn"](val_flat)
            pm.Potential(
                f"link_mu.{self.label}.{i}",
                -0.5 * ((val_flat[i] - mu_t) / sig_i) ** 2,
            )

        # A3. TRUNCATION NORMALIZATION for a DYNAMIC (linked) bound combined
        #     with a Gaussian prior.  Sections A/A2 add an UNNORMALIZED
        #     Gaussian on top of the reparameterization's exact U(lo, up), so
        #     the conditional prior on this element is a truncated normal
        #     whose mass depends on the bound-source parameter b -- and an
        #     unaccounted conditional mass reweights b's own posterior.
        #
        #     The correction is +log(span) - log(Phi(beta) - Phi(alpha)).
        #     Derivation: with lq = c + s*raw, p(val | b) as built is
        #     exp(-0.5 z^2) / (span * s * sqrt(2 pi)) (see the dynamic-bound
        #     block above for the U(lo, up) half), which integrates over
        #     [lo, up] to sigma*Z/(span*s), Z = Phi(beta) - Phi(alpha).
        #     Dividing by that leaves the normalized truncated normal, so the
        #     added term is log(span) - log(Z) up to a constant.
        #
        #     The +log(span) is NOT the -log(span) double-count review 1.5
        #     removed -- it has the opposite sign and it belongs to the
        #     Gaussian, not to the uniform.  The sigma -> infinity limit is
        #     the check: there Z -> span/(sigma*sqrt(2 pi)), the whole
        #     correction collapses to a CONSTANT, and the pure-uniform case
        #     is recovered exactly.  Adding -log(Z) alone would leave
        #     -log(span) behind in that limit, i.e. reintroduce precisely the
        #     bias review 1.5 removed (it rewards the bound source for
        #     shrinking the interval).
        #
        #     Static bounds need none of this: Z is then a constant.
        for i, (lo_t, up_t, span_t) in dyn_bounds.items():
            if i in mu_links:
                mu_i = mu_links[i]["fn"](val_flat)
            elif gaussian_prior_mask[i]:
                mu_i = pt.as_tensor_variable(float(prior_mus[i]))
            else:
                continue  # no Gaussian on this element: U(lo, up) already
            sig_i = float(sigmas[i])
            alpha = (lo_t - mu_i) / sig_i
            beta = (up_t - mu_i) / sig_i
            pm.Potential(
                f"trunc_norm.{self.label}.{i}",
                pt.log(span_t) - _log_normal_mass(alpha, beta),
            )

        # B. Soft bounds for derived params (and the rare half-bounded sampled
        #    param, where only one bound is finite so the logit transform does
        #    not apply). Fully-bounded sampled params: sigmoid is a hard
        #    constraint — no barrier needed.
        #    Fixed params: constant, so barrier adds only a harmless constant — skip.
        #    REPORTED elements get none, for the reason section A gives.
        needs_barrier = (
            (is_derived & ~is_reported) | (is_sampled & ~use_logit)
        ) & ~is_fixed
        has_lower = ~np.isinf(lowers) & needs_barrier
        has_upper = ~np.isinf(uppers) & needs_barrier
        if np.any(has_lower | has_upper):
            # PRELIMINARY barrier steepness from init_scale (falls back to
            # gaussian_scales for Gaussian params, where gaussian_scales =
            # sigma).  These are replaced after the whitening rescale by the
            # measured 1-sigma response of this parameter to unit raw steps
            # (whitening.measure_barrier_scales -> set_barrier_scales), via
            # the shared variable below.  A user bound_scale pins an element
            # (a modeling choice: barrier transition width = 0.01 * scale).
            barrier_scales = np.where(use_logit, scales, gaussian_scales)
            # A missing scale (e.g. a derived vector element the relaxation
            # engine never resolved) must soften the barrier, not poison the
            # whole logp with NaN.
            barrier_scales = np.where(
                np.isfinite(barrier_scales) & (barrier_scales > 0),
                barrier_scales,
                1.0,
            )
            user_bound = to_vec(self.bound_scale, n_elements, fill=np.nan)
            pinned = np.isfinite(user_bound) & (user_bound > 0)
            barrier_scales = np.where(pinned, user_bound, barrier_scales)

            sv_barrier = pytensor.shared(
                barrier_scales.astype(float),
                name=f"{self.label}_barrier_scale",
                shape=barrier_scales.shape,
            )
            self._barrier_state = {
                "sv": sv_barrier,
                "pinned": pinned,
                "needs_barrier": (has_lower | has_upper).copy(),
            }

            if np.any(has_lower):
                mask = pt.as_tensor_variable(has_lower)
                penalty = soft_lower_bound(
                    val_flat, pt.as_tensor_variable(lowers), sv_barrier
                )
                pm.Potential(
                    f"low_bound.{self.label}",
                    pm.math.sum(pt.where(mask, penalty, 0.0)),
                )

            if np.any(has_upper):
                mask = pt.as_tensor_variable(has_upper)
                penalty = soft_upper_bound(
                    val_flat, pt.as_tensor_variable(uppers), sv_barrier
                )
                pm.Potential(
                    f"up_bound.{self.label}",
                    pm.math.sum(pt.where(mask, penalty, 0.0)),
                )

        # C. Flat-prior correction for logit-transformed sampled parameters.
        #    raw ~ N(0,1) through the sigmoid gives a logit-normal prior in
        #    physical space. Adding log(q*(1-q)) + raw²/2 cancels both the
        #    sigmoid distortion AND the N(0,1) raw density, leaving an exactly
        #    uniform prior on [lower, upper] — the same logp PyMC's Interval
        #    transform gives pm.Uniform, but in our initval-centered,
        #    init_scale-whitened raw coordinate. init_scale then only affects
        #    tuning/conditioning, not the posterior.
        if np.any(use_logit) and expr_raw is None:
            logit_mask = pt.as_tensor_variable(use_logit)
            # log(q*(1-q)) from the *unclipped* logit: smooth, and decays ~ -|lq|
            # at the walls so the sampler always feels a restoring gradient
            # (computing it through the clipped sigmoid would plateau, leaving
            # a flat region where a chain could drift unboundedly).
            # |lq| is capped at 700: pytensor's JAX softplus NaNs in the
            # gradient once exp(|lq|) overflows (an unselected jnp.where
            # branch; see potentials.py) -- beyond 700 the exact value is
            # linear in lq anyway, so the restoring slope is unchanged.
            lq_safe = pt.clip(lq, -700.0, 700.0)
            log_jac = (
                -pt.softplus(lq_safe)
                - pt.softplus(-lq_safe)
                - pt.maximum(pt.abs(lq) - 700.0, 0.0)
            )
            # raw_vector is clipped before squaring: see _RAW_CANCELLATION_CLIP
            # (a shared variable -- the whitening probe raises it in place).
            raw_cancel_safe = pt.clip(
                raw_vector,
                -_raw_cancellation_clip_sv,
                _raw_cancellation_clip_sv,
            )
            # Saturation guard: log_jac's restoring slope approaches a
            # constant (not a growing one) as |lq| -> infinity, so a
            # data-unconstrained direction (whitening sets a large scale_logit
            # for it) can push |lq| far past _LOGIT_SATURATION_LQ before
            # feeling much resistance -- even though phys_logit has already
            # clipped there, so no distinguishable physical state, and no
            # posterior mass, lives beyond it. This adds a quadratic-in-excess
            # penalty only past that threshold: exactly zero (and the
            # correction above stays an exact uniform prior) on the
            # representable interior, growing sharply beyond it so the raw
            # coordinate can't wander into that degenerate, numerically
            # unsafe plateau. Independent of scale_logit and of any
            # component's physics -- keyed on lq, the same coordinate
            # phys_logit's clip already uses.
            saturation_excess = pt.maximum(
                pt.abs(lq) - _LOGIT_SATURATION_LQ, 0.0
            )
            saturation_penalty = -_LOGIT_SATURATION_PENALTY_K * pt.sqr(
                saturation_excess
            )
            correction = pt.where(
                logit_mask,
                log_jac + 0.5 * pt.sqr(raw_cancel_safe) + saturation_penalty,
                pt.zeros_like(raw_vector),
            )
            pm.Potential(
                f"logit_uniform_prior.{self.label}", pt.sum(correction)
            )

        return self.value

    def finalize_deferred(self, specs=None):
        """Patch REPORTED elements and create this parameter's Deterministic.

        The second half of a two-phase build, called by ``System.build_model``
        once every parameter exists (inside the model context).  A REPORTED
        element (manifest role 3) is derived from a quantity that, on OTHER
        elements of some parameter, is derived from this one -- a V_c/V_e orbit
        reports ``secosw`` computed from its ``ecc``/``omega``, while a
        sqrt(e)cos/sin orbit derives its ``ecc`` from ``secosw``.  Per element
        that is perfectly acyclic; per PARAMETER it is a cycle, which is why
        the build order cannot place these expressions and why they wait here.
        ``graph.py`` contributes no edge for them, for the same reason.

        Safe by construction, in both directions:

        * Nothing can have consumed the patched values.  A reported element is
          consumed by nothing (the vocabulary's definition, and what makes the
          cycle dissolve), so every consumer that read ``self.value`` during
          stage 6 or 6 read an element this patch does not touch.
        * The patch adds no logp term.  It creates one ``pm.Deterministic`` and
          no potential: ``build_pymc`` already excludes reported elements from
          the Gaussian prior (section A) and the soft barriers (section B), and
          they carry no raw coordinate to correct (section C).  A prior on a
          quantity the model does not consume would be a logp term with no data
          behind it.

        ``specs`` are the wired ``ElementExpression``s, supplied by
        ``Component.finalize_reported`` (which could only build them now).  With
        no argument, the specs recorded at build time are used, which is the
        path a Parameter built directly takes.

        Idempotent: the deferred state is cleared, so a second call is a no-op
        (the GUI builds a model more than once per System).
        """
        import pymc as pm
        import pytensor.tensor as pt

        state = self._deferred_reported
        if not state:
            return self.value

        n_elements = self._n_elements()
        if specs is not None:
            patches = [
                (
                    normalize_selector(s.mask, n_elements, self.label),
                    s.expr,
                    bool(s.sliced),
                )
                for s in specs
            ]
        else:
            patches = state["specs"]

        missing = [
            i for i, (_m, expr, _s) in enumerate(patches) if expr is None
        ]
        if missing:
            raise ValueError(
                f"Parameter '{self.label}': reported element group(s) {missing} "
                f"have no expression to apply. Component.finalize_reported "
                f"supplies them; a Parameter built directly must pass its own "
                f"ElementExpression list to finalize_deferred."
            )

        phys_val = pt.flatten(pt.as_tensor_variable(self.value))
        for mask, expr, sliced in patches:
            phys_val = self._patch_elements(phys_val, mask, expr, sliced)

        shape = state["shape"]
        if shape == ():
            val_to_save = phys_val[0]
        else:
            val_to_save = pt.broadcast_to(phys_val, shape)

        self._deferred_reported = None
        self.value = pm.Deterministic(self.label, val_to_save)
        return self.value

    def generate_posterior(self, posterior_bundle, param_lookup=None):
        """Evaluate this parameter's expression over the posterior.

        Parameters
        ----------
        posterior_bundle : mapping of name → array
            Posterior draws.  When ``param_lookup`` is provided the values are
            assumed to be in *user* units (as stored in the trace); they are
            converted to internal units before evaluating the PyTensor expression
            and the result is converted back to user units before returning.
            When ``param_lookup`` is ``None`` the bundle is assumed to already
            be in internal units (e.g. single-point evaluation during
            ``inspect_start``).
        param_lookup : dict[str, Parameter], optional
            Map from parameter label to Parameter object, used to look up
            conversion factors for the input variables.
        """
        if self.label in posterior_bundle:
            return posterior_bundle[self.label]
        if self.expression is None:
            return None

        expr = (
            self.expression() if callable(self.expression) else self.expression
        )

        # --- Strip Astropy Units before graph walking ---
        if hasattr(expr, "value") and hasattr(expr, "unit"):
            expr = expr.value

        all_nodes = pytensor.graph.traversal.ancestors([expr])

        inputs_in_posterior = [
            n
            for n in all_nodes
            if hasattr(n, "name") and n.name in posterior_bundle
        ]

        # fixed parameter, just return the scalar (convert output to user units)
        if not inputs_in_posterior:
            val = np.asarray(expr.eval(), dtype=float)
            if param_lookup is not None:
                val = val * np.squeeze(
                    np.asarray(self._get_conversion_factors(), dtype=float)
                )
            if val.size > 1:
                # Keep the posterior convention: elements first, SAMPLES
                # last.  A bare (n_el,) array is indistinguishable from
                # n_el samples of a scalar, and _summarize_array then pools
                # the ELEMENTS into one summary -- star.luminosity with
                # pinned teff/radius reported one pooled '3.0 +/- 1.4' for
                # two constant per-star values, and the single unsuffixed
                # macro it emitted left the table's per-element references
                # undefined.
                return val.reshape(-1, 1)
            return val.item()

        # 1. Compile the function for a single evaluation
        calc_func = pytensor.function(
            inputs_in_posterior, expr, on_unused_input="ignore"
        )

        # 2. Extract the data arrays and align dimensions
        input_data = []
        n_samples = None

        for n in inputs_in_posterior:
            data = posterior_bundle[n.name]
            val = getattr(data, "values", data)

            # az.extract puts the 'sample' dimension LAST.
            # Move it to the FIRST dimension so we can loop over it safely: (n_samples, *shape)
            val = np.moveaxis(val, -1, 0)

            # When the posterior is in user units, convert each input to internal
            # units so the PyTensor expression (compiled with internal-unit nodes)
            # receives the values it expects.
            if param_lookup is not None and n.name in param_lookup:
                in_factor = np.squeeze(
                    np.asarray(
                        param_lookup[n.name]._get_conversion_factors(),
                        dtype=float,
                    )
                )
                val = val / in_factor

            if n_samples is None:
                n_samples = val.shape[0]

            input_data.append(val)

        # 3. Evaluate the first sample to dynamically determine the output dimension
        # Reshape each sample slice to match the PyTensor node's expected ndim.
        # A scalar variable lands as 0-D after arr[0], but build_pymc may have
        # compiled calc_func with a 1-D (n=1) input; atleast_nd fixes that.
        def _match_ndim(val, node):
            target = node.ndim if hasattr(node, "ndim") else 0
            while np.ndim(val) < target:
                val = np.atleast_1d(val)
            return val

        first_args = [
            _match_ndim(arr[0], n)
            for arr, n in zip(input_data, inputs_in_posterior)
        ]
        first_result = np.asarray(calc_func(*first_args))

        # 4. Loop through the remaining samples
        # Create an array of shape (n_samples, *shape)
        result = np.zeros((n_samples,) + first_result.shape)
        result[0] = first_result

        for i in range(1, n_samples):
            args = [
                _match_ndim(arr[i], n)
                for arr, n in zip(input_data, inputs_in_posterior)
            ]
            result[i] = calc_func(*args)

        # Convert internal-unit result to user units when the inputs came from a
        # user-unit posterior (param_lookup provided).
        if param_lookup is not None:
            out_factor = np.squeeze(
                np.asarray(self._get_conversion_factors(), dtype=float)
            )
            result = result * out_factor

        # Return the proper shape with 'sample' at the end again to match ArviZ's format
        return np.moveaxis(result, 0, -1)

    def get_scale(self):
        return {self.name: self.init_scale}

    # ---------
    # Units (metadata convenience)
    # ---------

    def get_physical_value(self, model, point):
        """
        Translates a PyMC 'point' (which uses interval-space)
        back to this parameter's physical value.
        """
        # Compile a quick function that takes the point and returns the RV value
        fn = model.compile_fn(self.value, on_unused_input="ignore")
        return fn(point)

    def _get_conversion_factors(self):
        """
        Calculates the numerical conversion factor from internal -> user units.
        Safely handles self.unit as a single Unit, a scalar Quantity, or a list/array.
        Halts immediately on invalid linear unit conversions.
        """
        is_sequence = isinstance(self.unit, (list, tuple)) or (
            isinstance(self.unit, np.ndarray)
            and getattr(self.unit, "ndim", 0) > 0
        )

        def _process_single(u_user):
            target_u = getattr(u_user, "unit", u_user)
            i_str = str(self.internal_unit)
            u_str = str(target_u)

            # 1. Protection: Ignore Dex math completely.
            # If both are log-space (dex), treat the multiplier as 1.0.
            if "dex" in u_str and "dex" in i_str:
                return 1.0

            # 2. Protection: Strict Linear conversion
            try:
                return float(self.internal_unit.to(target_u))
            except Exception as e:
                # Halt immediately if units are incompatible (e.g., mass to time)
                raise ValueError(
                    f"[{self.label}] Conversion failure from '{u_str}' to '{i_str}'. "
                    f"Ensure units are valid astropy strings. Original error: {e}"
                )

        if is_sequence:
            return np.array(
                [_process_single(u) for u in self.unit], dtype=np.float64
            )

        return _process_single(self.unit)

    def set_whitening(self, raw_scale):
        """Rescale the whitening in place from a measured raw-space scale.

        ``raw_scale`` has one entry per SAMPLED element (shaped like
        ``self.raw_initval``): the distance, in the CURRENT raw coordinate,
        of the 0.5-nat logp contour along that element (as measured by the
        whitening probe).  Multiplying the whitening scale by it puts that
        contour at exactly one raw unit, which is the "curvature = -1"
        conditioning the old init_scale tuning loop approximated by hand.

        Deliberately does NOT recompute logit_q_inits / q_floors: the
        anchor (raw = 0) stays exactly where build_pymc placed it, so the
        update is a pure scale change in logit space.  A NONZERO
        ``raw_initval`` (a pre-whitening seed polish moved the start off the
        anchor) is rescaled by 1/multiplier in the same pass -- lq = lq0 +
        scale*raw is invariant under (scale, raw) -> (scale*m, raw/m) -- so
        the start stays the same PHYSICAL point the probe measured around.
        Elements whose raw N(0,1) IS the prior -- every NON-LOGIT element,
        i.e. anything without two finite bounds -- are never touched.  Their
        scale is the prior width (sigma when one was given, init_scale when
        the bounds are infinite and none was), not a whitening choice; only
        the logit branch's correction potential cancels the raw N(0,1), and
        only there is the rescale provably posterior-preserving.  Non-finite
        or non-positive entries (a failed probe) leave that element's scale
        unchanged.

        Because the scales live in pytensor.shared variables, every function
        already compiled from this model sees the new values immediately;
        anything compiled afterwards (dlogp, JAX/nutpie traces, PTDE worker
        pools) does too.  ``_raw_transform`` and ``init_scale`` are synced so
        multi-seed raw starts and diagnostics stay consistent.

        Returns the post-rescale scale of each sampled element in the NEW
        raw units (1.0 where the multiplier was applied; the measured value
        where the element is deliberately untouched) -- the per-element
        dispersion PTDE's chain initialization can use directly instead of
        re-probing.  Returns None when nothing is sampled.
        """
        ws = self._whiten_state
        tf = self._raw_transform
        if ws is None or tf is None:
            return None
        idx = tf["sampled_idx"]
        raw_scale = np.asarray(raw_scale, dtype=float).reshape(-1)
        if raw_scale.size != len(idx):
            raise ValueError(
                f"Parameter '{self.label}': set_whitening got {raw_scale.size} "
                f"scales for {len(idx)} sampled elements."
            )

        scale_logits = ws["sv_scale_logits"].get_value().copy()
        gauss_scales = ws["sv_gaussian_scales"].get_value().copy()
        # Post-rescale scale of each sampled element in the NEW raw units:
        # 1.0 where the multiplier was applied (the contour now sits at one
        # raw unit); the measured value where the element is deliberately not
        # rescaled (Gaussian-prior elements); 1.0 where the probe failed.
        post = np.ones(len(idx))
        rescaled = np.zeros(len(idx), dtype=bool)
        for j, i in enumerate(idx):
            m = raw_scale[j]
            if not np.isfinite(m) or m <= 0:
                continue
            if tf["use_logit"][i]:
                scale_logits[i] *= m
                rescaled[j] = True
            else:
                # NON-LOGIT elements are never rescaled.  Nothing cancels
                # their raw N(0,1) (section C fires only for use_logit), so
                # the raw prior IS this element's prior: N(mu, sigma) when a
                # sigma was given, N(initval, init_scale) when the bounds are
                # infinite and no sigma was.  The multiplier is measured from
                # the data, so rescaling would make the prior WIDTH
                # data-dependent -- circular, and a violation of the
                # "posterior provably unchanged" invariant that only holds
                # for the logit branch.  Report the measured scale instead
                # (PTDE uses it to disperse chains).
                post[j] = m

        # Keep a polished (nonzero) raw start pinned to the same physical
        # point through the rescale.  Historically raw_initval was always 0
        # for rescaled elements, making this a silent no-op.
        if self.raw_initval is not None:
            ri = np.asarray(self.raw_initval, dtype=float).reshape(-1).copy()
            if ri.size == len(idx):
                for j in np.nonzero(rescaled)[0]:
                    ri[j] /= raw_scale[j]
                self.raw_initval = ri

        self._apply_whitening_state(scale_logits, gauss_scales)
        return post

    def _apply_whitening_state(self, scale_logits, gauss_scales):
        """Push new whitening scale vectors into the shared variables and
        keep every mirror consistent: the frozen forward transform
        (raw_from_initval / phys_from_raw for multi-seed starts) and the
        physical-units init_scale used for reporting (diagnostics table,
        get_mcmc_init) -- dphys/draw at the start, i.e. scale_logit *
        q_init*(1-q_init)*span for logit elements, the scale itself for
        linear elements."""
        ws = self._whiten_state
        tf = self._raw_transform
        scale_logits = np.asarray(scale_logits, dtype=float)
        gauss_scales = np.asarray(gauss_scales, dtype=float)
        ws["sv_scale_logits"].set_value(scale_logits)
        ws["sv_gaussian_scales"].set_value(gauss_scales)
        tf["init_scale_logits"] = scale_logits.copy()
        tf["gaussian_scales"] = gauss_scales.copy()

        n_elements = (
            int(np.prod(self.shape)) if self.shape not in ((), None) else 1
        )
        phys_scales = to_vec(self.init_scale, n_elements, fill=1.0)
        raw_init = (
            np.asarray(self.raw_initval, dtype=float).reshape(-1)
            if self.raw_initval is not None
            else None
        )
        for j, i in enumerate(tf["sampled_idx"]):
            if tf["use_logit"][i]:
                # Evaluate dphys/draw at the START (anchor + scale*raw_init):
                # identical to the anchor historically (raw_init 0), but a
                # polished start sits off the anchor.
                lq = tf["logit_q_inits"][i]
                if raw_init is not None and raw_init.size > j:
                    lq = lq + scale_logits[i] * raw_init[j]
                q_init = 1.0 / (1.0 + np.exp(-np.clip(lq, -100.0, 100.0)))
                span = tf["uppers"][i] - tf["lowers"][i]
                phys_scales[i] = (
                    scale_logits[i] * q_init * (1.0 - q_init) * span
                )
            else:
                phys_scales[i] = gauss_scales[i]
        self.init_scale = (
            float(phys_scales[0]) if self.shape == () else phys_scales
        )

    def set_barrier_scales(self, phys_scales):
        """Replace the soft-bound barrier steepness scales in place.

        ``phys_scales`` is a FULL-length per-element vector in internal
        units: the measured 1-sigma response of this parameter to unit raw
        steps (whitening.measure_barrier_scales).  Only elements that have a
        barrier and are not pinned by a user bound_scale update; non-finite
        or non-positive entries are skipped.  Unlike the whitening scales,
        the barrier IS a posterior term -- this replaces the preliminary
        steepness with the data-driven one before sampling starts.
        """
        bs = self._barrier_state
        if bs is None:
            return
        phys_scales = np.asarray(phys_scales, dtype=float).reshape(-1)
        cur = bs["sv"].get_value().copy()
        if phys_scales.size != cur.size:
            raise ValueError(
                f"Parameter '{self.label}': set_barrier_scales got "
                f"{phys_scales.size} scales for {cur.size} elements."
            )
        ok = (
            bs["needs_barrier"]
            & ~bs["pinned"]
            & np.isfinite(phys_scales)
            & (phys_scales > 0)
        )
        cur[ok] = phys_scales[ok]
        bs["sv"].set_value(cur)

    def export_whitening(self):
        """Snapshot the ABSOLUTE whitening/barrier state for persistence.

        Returns None when nothing is sampled and no barrier exists.  The
        absolute logit-space scales (not the multipliers) are stored so a
        reload reproduces the sampled trace's raw coordinates exactly, even
        if the preliminary scales of a rebuilt model were to differ.
        """
        out = {}
        if self._whiten_state is not None:
            out["scale_logits"] = (
                self._whiten_state["sv_scale_logits"].get_value().tolist()
            )
            out["gaussian_scales"] = (
                self._whiten_state["sv_gaussian_scales"].get_value().tolist()
            )
        if self._barrier_state is not None:
            out["barrier_scales"] = (
                self._barrier_state["sv"].get_value().tolist()
            )
        return out or None

    def load_whitening(self, state):
        """Apply a persisted export_whitening snapshot to this build.

        Returns False (leaving the build untouched) on any shape mismatch --
        the caller should fall back to a fresh probe.
        """
        ws = self._whiten_state
        if "scale_logits" in state:
            if ws is None:
                return False
            sl = np.asarray(state["scale_logits"], dtype=float)
            gs = np.asarray(state["gaussian_scales"], dtype=float)
            if (
                sl.shape != ws["sv_scale_logits"].get_value().shape
                or gs.shape != ws["sv_gaussian_scales"].get_value().shape
            ):
                return False
            self._apply_whitening_state(sl, gs)
        if "barrier_scales" in state:
            bs = self._barrier_state
            if bs is None:
                return False
            b = np.asarray(state["barrier_scales"], dtype=float)
            if b.shape != bs["sv"].get_value().shape:
                return False
            bs["sv"].set_value(b)
        return True

    def raw_from_initval(self, initval_internal):
        """Map an alternate physical initval (internal units) to the raw N(0,1)
        start for this parameter's sampled elements, using the frozen forward
        transform from build_pymc.

        Used by multi-seed sampling to build one raw start dict per seed while
        keeping the bounds/scale fixed at seed 0.  Returns an array shaped like
        self.raw_initval (one entry per sampled element).

        Raises SeedBoundViolation if a sampled element's value is non-finite,
        or if a logit element's value falls outside its [lower, upper] bound
        -- a clipped start would sit in no basin, so the caller must skip that
        seed loudly rather than silently move it.
        """
        tf = getattr(self, "_raw_transform", None)
        if tf is None:
            # No sampled elements (fully fixed/derived) -> empty raw start.
            return np.zeros(0)
        idx = tf["sampled_idx"]
        n_elements = (
            int(np.prod(self.shape)) if self.shape not in ((), None) else 1
        )
        v = to_vec(initval_internal, n_elements)
        v = np.asarray(v, dtype=float).reshape(-1)
        raw = np.zeros(len(idx))
        for j, i in enumerate(idx):
            # A non-finite seed value fails BOTH bound comparisons below (every
            # comparison with NaN is False), so it used to sail through the
            # logit branch and land a NaN raw coordinate in a chain start dict
            # with nothing naming the element -- exactly the failure class the
            # stage-6 start checks removed for seed 0 (review 2.2.1).  It is
            # the same situation as an out-of-bounds seed and gets the same
            # treatment: the multi-seed caller skips the whole seed loudly.
            if not np.isfinite(v[i]):
                raise SeedBoundViolation(
                    f"{self.label}[{i}] seed initval is {v[i]} -- a seed must "
                    f"say where it starts"
                )
            if tf["use_logit"][i]:
                lower, upper = tf["lowers"][i], tf["uppers"][i]
                span = upper - lower
                q = (v[i] - lower) / span
                # Strictly outside [lower, upper] is a real violation. Exactly
                # AT a bound (q==0 or q==1, e.g. an angle default of 0 sitting
                # on its own lower bound) is not -- build_pymc nudges those
                # inward via q_floor for every seed, including seed 0, so
                # rejecting them here would inconsistently single out
                # non-seed-0 starts for a non-problem.
                if q < 0.0 or q > 1.0:
                    raise SeedBoundViolation(
                        f"{self.label}[{i}] seed initval {v[i]:.6g} is outside "
                        f"bounds [{lower:.6g}, {upper:.6g}]"
                    )
                qf = tf["q_floors"][i]
                q = np.clip(q, qf, 1.0 - qf)
                lq = np.log(q / (1.0 - q))
                scale_logit = tf["init_scale_logits"][i]
                raw[j] = (lq - tf["logit_q_inits"][i]) / max(
                    scale_logit, 1e-30
                )
            else:
                raw[j] = (v[i] - tf["gaussian_mus"][i]) / max(
                    tf["gaussian_scales"][i], 1e-30
                )
        return raw

    def phys_from_raw(self, raw_vec):
        """Map raw N(0,1) coordinates to physical values (internal units).

        The inverse of raw_from_initval, using the same frozen forward
        transform from build_pymc (including the +/-30 sigmoid clip), so a
        value produced here maps back through raw_from_initval consistently.

        Takes one entry per SAMPLED element (shaped like self.raw_initval) and
        returns a FULL-length element vector, which is what raw_from_initval
        expects back.  Non-sampled entries are placeholders: raw_from_initval
        only reads the sampled ones.
        """
        tf = getattr(self, "_raw_transform", None)
        if tf is None:
            return np.zeros(0)
        idx = tf["sampled_idx"]
        n_elements = (
            int(np.prod(self.shape)) if self.shape not in ((), None) else 1
        )
        raw = np.asarray(raw_vec, dtype=float).reshape(-1)
        out = np.zeros(n_elements)
        for j, i in enumerate(idx):
            if tf["use_logit"][i]:
                lq = (
                    tf["logit_q_inits"][i]
                    + tf["init_scale_logits"][i] * raw[j]
                )
                q = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            lq,
                            -_LOGIT_SATURATION_LQ,
                            _LOGIT_SATURATION_LQ,
                        )
                    )
                )
                out[i] = (
                    tf["lowers"][i] + (tf["uppers"][i] - tf["lowers"][i]) * q
                )
            else:
                out[i] = (
                    tf["gaussian_mus"][i] + tf["gaussian_scales"][i] * raw[j]
                )
        return out

    # ------------------------------------------------------------------
    # Unit conversion.  These three are the ONLY way anything outside this
    # class converts a number between the user unit and the internal one.
    #
    # DIRECTION, because there are two reciprocal factor functions in the
    # codebase and confusing them is silent: `_get_conversion_factors`
    # (here) is the INTERNAL -> USER multiplier, while
    # `ConfigManager.get_conversion_factor` is the USER -> INTERNAL one.
    # So `* factor` means opposite things in parameter.py and config.py --
    # which is exactly how `outputs/ledger.py` came to divide where it had
    # to multiply and report every converted parameter wrong by factor**2
    # (planet.mass in examples/hd80606: 1.45e-06 Mjup for a start of 1.596).
    # Call these methods and the direction is in the name.
    # ------------------------------------------------------------------

    def element_factor(self, index=0):
        """Internal -> user multiplier for element ``index``, as a float.

        The one owner of the "element ``index``, or element 0 when the
        factor vector is shorter" rule.  A scalar ``unit:`` normalizes to a
        one-element list in ``__post_init__``, so the fallback is the
        ordinary case for every vector parameter; a genuinely per-element
        ``unit:`` (config.py's ``elem_units``) is what makes the index
        meaningful.
        """
        f = np.atleast_1d(
            np.asarray(self._get_conversion_factors(), dtype=float)
        )
        return float(f[index] if index < f.size else f[0])

    def _directional_factor(self, val, index):
        """Factor for :meth:`to_internal` / :meth:`from_internal` to apply."""
        if index is not None:
            return self.element_factor(index)
        f = np.asarray(self._get_conversion_factors(), dtype=float)
        n_val = np.size(val)
        # Both sides genuinely vectors and disagreeing is a bug, not a
        # broadcast: it silently mixed elements before ledger.py grew a
        # local guard against it.  Size 1 on either side broadcasts.
        if f.size > 1 and n_val > 1 and f.size != n_val:
            raise ValueError(
                f"[{self.label}] {f.size} unit conversion factors for "
                f"{n_val} values -- cannot convert."
            )
        return f

    def to_internal(self, val=None, index=None):
        """USER units -> INTERNAL units.

        ``index`` converts a single element with that element's own factor;
        ``None`` converts a whole vector (or a scalar) at once.
        """
        target = val if val is not None else self.value
        return target / self._directional_factor(target, index)

    def from_internal(self, val=None, index=None):
        """INTERNAL units -> USER units.

        ``index`` converts a single element with that element's own factor;
        ``None`` converts a whole vector (or a scalar) at once.
        """
        target = val if val is not None else self.value
        # Safety check for unitless parameters
        if self.unit is None or self.internal_unit is None:
            return target
        return target * self._directional_factor(target, index)

    # ---------
    # LaTeX helpers
    # ---------

    def to_latex_def(self, sigfigs: int = 2) -> str:

        def _fixed(val):
            """'\\equiv <value>' at sensible precision; \\nodata if not finite.

            The raw float repr used to reach the table verbatim
            ('\\equiv 2.249201909620218'), and those cells set the whole
            deluxetable's column width.
            """
            try:
                v = float(val)
            except (TypeError, ValueError):
                return r"\nodata"
            if not np.isfinite(v):
                return r"\nodata"
            return rf"\equiv {v:.6g}"

        # FIXED PARAMETER PATH
        if self.posterior is None:
            if self.initval is not None:
                physical_inits = self.from_internal(self.initval)
                # Sized from the SHAPE, not from the initval: a vector whose
                # initval is a broadcast scalar (a manifest-options value on a
                # fully pinned vector) used to emit ONE unsuffixed macro while
                # the table body cites a suffixed one per element -- an
                # "Undefined control sequence" at compile, by construction.
                # The \nodata branch below already sized itself this way.
                n = self._n_elements()
                inits = np.broadcast_to(
                    np.atleast_1d(physical_inits), (n,)
                ).copy()

                if len(inits) > 1:
                    lines = []
                    for i, val in enumerate(inits):
                        idx_str = idx_to_words(i)
                        lines.append(
                            rf"\providecommand{{\{self.latex_varname}{idx_str}}}{{\ensuremath{{{_fixed(val)}}}}}"
                            + "\n"
                        )
                    return "".join(lines)
                else:
                    return (
                        rf"\providecommand{{\{self.latex_varname}}}{{\ensuremath{{{_fixed(inits[0])}}}}}"
                        + "\n"
                    )
            # NO VALUE AT ALL (no posterior, no initval -- e.g. a derived
            # parameter decoded from a trace that predates it).  The table
            # body still cites the macro for every element, so the emitter
            # MUST define it or the document dies with 'Undefined control
            # sequence' at the end of the fit (the DC2018_128
            # star.luminosity case).  \nodata is aastex's blank-cell mark.
            n = int(np.prod(self.shape)) if self.shape != () else 1
            if n > 1:
                return "".join(
                    rf"\providecommand{{\{self.latex_varname}{idx_to_words(i)}}}{{\nodata}}"
                    + "\n"
                    for i in range(n)
                )
            return (
                rf"\providecommand{{\{self.latex_varname}}}{{\nodata}}" + "\n"
            )

        # SAMPLED PARAMETER PATH
        if self.summary is None:
            self.compute_summary()

        if isinstance(self.summary, list):
            lines = []
            for i, summ in enumerate(self.summary):
                val = summ.latex_value(sigfigs=sigfigs)
                idx_str = idx_to_words(i)
                lines.append(
                    rf"\providecommand{{\{self.latex_varname}{idx_str}}}{{\ensuremath{{{val}}}}}"
                    + "\n"
                )
            return "".join(lines)

        val = self.summary.latex_value(sigfigs=sigfigs)
        return (
            rf"\providecommand{{\{self.latex_varname}}}{{\ensuremath{{{val}}}}}"
            + "\n"
        )

    def to_latex_mode_defs(self, sigfigs: int = 2) -> str:
        """Per-mode \\providecommand defs, suffixed modeone, modetwo, ...

        Macro names extend the unsuffixed ones (``\\<varname><idx><suffix>``)
        so every mode's value can be cited in the same document.  Fixed
        parameters have no per-mode defs (their unsuffixed def applies to
        every mode).
        """
        if self.posterior is None or self.mode_summaries is None:
            return ""
        lines = []
        for k, summary in enumerate(self.mode_summaries):
            suffix = mode_suffix(k)
            if isinstance(summary, list):
                for i, summ in enumerate(summary):
                    val = summ.latex_value(sigfigs=sigfigs)
                    lines.append(
                        rf"\providecommand{{\{self.latex_varname}{idx_to_words(i)}{suffix}}}"
                        rf"{{\ensuremath{{{val}}}}}" + "\n"
                    )
            else:
                val = summary.latex_value(sigfigs=sigfigs)
                lines.append(
                    rf"\providecommand{{\{self.latex_varname}{suffix}}}"
                    rf"{{\ensuremath{{{val}}}}}" + "\n"
                )
        return "".join(lines)

    def get_unit_str(self, index=0):
        u_list = np.atleast_1d(self.unit)
        u_obj = u_list[index] if index < len(u_list) else u_list[0]
        return (
            u_obj.to_string()
            if u_obj and u_obj.to_string() != "dimensionless"
            else ""
        )

    # ------------------------------------------------------------------
    # Component-declared prior contributions (see PriorContribution).
    # ------------------------------------------------------------------

    def add_prior_contribution(
        self,
        latex,
        text=None,
        elements=None,
        supersedes_bounds=False,
        support_phrase="normalized on",
    ):
        """Declare that something outside this Parameter adds a prior term.

        Call this next to the ``pm.Potential`` it describes, in the
        component's ``build_likelihood``.  ``latex``/``text`` are the two
        renderings of the term ("$\\propto d^{2}$" / "propto d^2"); ``text``
        defaults to ``latex`` with its ``$`` stripped.  ``elements`` selects
        which elements of a vector the term covers -- ``None`` for all, or an
        index iterable / boolean mask (the FFP mass function applies per
        star).  ``supersedes_bounds`` says the term REPLACES the implicit
        uniform prior over the element's bounds rather than multiplying an
        explicit prior of this Parameter's own; ``support_phrase`` says how
        the table note joins the term to that interval (the default claims
        normalization, which a barrier must not).  See PriorContribution.

        Idempotent: re-declaring an identical contribution (a second
        ``build_model()`` on the same System, as the GUI does) is a no-op,
        so priors cannot accumulate copies of themselves.
        """
        if text is None:
            text = str(latex).replace("$", "")
        contribution = PriorContribution(
            latex=str(latex),
            text=str(text),
            elements=_normalize_prior_elements(elements),
            supersedes_bounds=bool(supersedes_bounds),
            support_phrase=str(support_phrase),
        )
        if contribution in self.prior_contributions:
            return contribution
        self.prior_contributions.append(contribution)
        return contribution

    def prior_contributions_at(self, index=0):
        """The declared contributions that cover element ``index``."""
        return [
            c
            for c in self.prior_contributions
            if c.elements is None or index in c.elements
        ]

    def _prior_scalar(self, val, index):
        """One element of a numeric prior field, in USER units (or None).

        The one reader of a numeric field for the Prior column: both
        ``_support_str`` and ``_own_prior_str`` go through it, where each
        used to carry its own byte-identical copy.
        """
        if val is None:
            return None
        arr = np.atleast_1d(val)
        raw = arr[index] if index < len(arr) else arr[0]
        if hasattr(raw, "eval"):
            try:
                raw = raw.eval()
            except Exception:
                return None
        try:
            f_val = float(raw)
        except (TypeError, ValueError):
            return None
        if np.isnan(f_val):
            return None
        return float(self.from_internal(f_val, index=index))

    def _interval_str(self, index, latex):
        """``[lower, upper]`` for this element, or '' when not both finite.

        Used to keep the support in the rendered prior when a component
        contribution supersedes the bounds-derived uniform: the term says
        something about exactly that interval (the volume prior and the IMF
        branches normalize over it; the SED's grid barrier turns on at its
        edges), and dropping the numbers would make the new text less
        informative than the "Uniform" it replaces.
        """
        lo = self._prior_scalar(self.lower, index)
        hi = self._prior_scalar(self.upper, index)
        if lo is None or hi is None or np.isinf(lo) or np.isinf(hi):
            return ""
        l_s = _fmt_prior_value(lo, latex)
        h_s = _fmt_prior_value(hi, latex)
        return rf"$[{l_s}, {h_s}]$" if latex else f"[{l_s}, {h_s}]"

    def _support_str(self, index, latex):
        """``_interval_str`` prefixed with " on ", or '' when unavailable."""
        interval = self._interval_str(index, latex)
        return f" on {interval}" if interval else ""

    def get_prior_str(self, index=0, latex=True):
        """The Prior column text for one element.

        Composes what this Parameter knows about its own prior (a Gaussian
        ``sigma``, a pin, or the uniform implied by its bounds) with any
        prior contributions a component declared against it
        (``add_prior_contribution``).  Both report paths go through here --
        run.py's startup audit table asks for ``latex=False`` and
        ``to_latex_prior_def`` for ``latex=True`` -- so a component that
        declares its potential is described in both.
        """
        own, kind = self._own_prior_str(index=index, latex=latex)

        contributions = self.prior_contributions_at(index)
        if not contributions:
            return own

        # A pinned element is a delta function; a potential evaluated on it
        # is a constant that cannot move the posterior, so "Fixed" stays the
        # honest and complete statement.
        if kind == "fixed":
            return own

        parts = []
        supersede = any(c.supersedes_bounds for c in contributions)
        if own and not (supersede and kind == "bounds"):
            parts.append(own)
        parts.extend(c.latex if latex else c.text for c in contributions)
        rendered = (r" $\times$ " if latex else " * ").join(parts)
        if supersede and kind == "bounds":
            rendered += self._support_str(index, latex)
        return rendered

    def _own_prior_str(self, index=0, latex=True):
        """This Parameter's own prior text, plus which branch produced it.

        Returns ``(text, kind)`` with ``kind`` in
        ``{"fixed", "gaussian", "bounds", "none"}``.  The body is the
        historical ``get_prior_str``, unchanged apart from reporting the
        branch it took; ``get_prior_str`` returns it verbatim whenever no
        component has declared a contribution.
        """

        # One reader for every Prior-column number, shared with
        # _support_str: this was a byte-identical private copy of
        # _prior_scalar apart from letting a non-float raise instead of
        # reporting no number.
        def _scalar(val):
            return self._prior_scalar(val, index)

        # One formatter for every Prior-column number (shared with
        # _support_str) -- a second private copy is how '-1.00e+05' would
        # creep back into half the branches.
        _fmt = _fmt_prior_value

        # Per element, deliberately: a vector whose instances chose different
        # parameterizations is derived on some elements and sampled on others,
        # so a whole-vector `self.expression is not None` would report one
        # instance's parameterization for all of them.
        derived_here = self.element_is_derived(index)

        sig = _scalar(self.sigma)
        if sig == 0:
            # Unchanged, INCLUDING for a derived element: `sigma: 0` there is a
            # no-op the build already warns about, and examples/ob08092 ships
            # one (star.mass), so "Fixed" is what its table has always said.
            # Correcting that is a reporting change of its own, not a side
            # effect of making the column per element.
            return "Fixed", "fixed"

        lo = _scalar(self.lower)
        hi = _scalar(self.upper)

        # Determine if there are actual constraints to print
        has_prior = sig is not None and sig > 0
        has_bounds = lo is not None or hi is not None

        # Derived parameters with no custom constraint have no prior to display.
        if derived_here and not (has_prior or has_bounds):
            return "", "none"

        mu = _scalar(self.mu)
        if mu is None:
            mu = _scalar(self.initval)

        if not latex:
            strs = []
            if has_prior:
                strs.append(f"N({_fmt(mu, False)}, {_fmt(sig, False)})")
            if strs:
                return " * ".join(strs), "gaussian"

            if has_bounds:
                l_s = (
                    _fmt(lo, False)
                    if (lo is not None and not np.isinf(lo))
                    else ""
                )
                h_s = (
                    _fmt(hi, False)
                    if (hi is not None and not np.isinf(hi))
                    else ""
                )
                if l_s and h_s:
                    return f"U({l_s}, {h_s})", "bounds"
                if l_s:
                    return f"> {l_s}", "bounds"
                if h_s:
                    return f"< {h_s}", "bounds"

            if derived_here:
                return "", "none"

        # --- LaTeX Formatting Block ---
        strs = []
        if has_prior:
            strs.append(rf"$\mathcal{{N}}({_fmt(mu)}, {_fmt(sig)})$")

        if strs:
            return r" $\times$ ".join(strs), "gaussian"

        if has_bounds:
            l_s, h_s = _fmt(lo), _fmt(hi)

            # Safe infinity checks to avoid TypeErrors if lo/hi are None
            lo_is_inf = (lo is None) or np.isinf(lo)
            hi_is_inf = (hi is None) or np.isinf(hi)

            if not lo_is_inf and not hi_is_inf:
                return rf"$\mathcal{{U}}({l_s}, {h_s})$", "bounds"
            if not lo_is_inf:
                return rf"$> {l_s}$", "bounds"
            if not hi_is_inf:
                return rf"$< {h_s}$", "bounds"

        return "", "none"

    def prior_cell_and_notes(self, index=0):
        """``(cell_latex, [note_latex, ...])`` for one Prior-column element.

        The cell carries only what fits a column: the Parameter's OWN prior
        ("Fixed", a Gaussian, the uniform-over-bounds).  Component-declared
        contributions (``add_prior_contribution``) go to TABLE NOTES -- the
        long Galactic-kinematic and IMF texts used to be appended to the
        cell and set the whole table's Prior-column width.  When a
        contribution supersedes the bounds-derived uniform, the note gains
        the support it is normalized over and the cell keeps nothing but
        the note mark (the caller appends it).
        """
        own, kind = self._own_prior_str(index=index, latex=True)
        contributions = self.prior_contributions_at(index)
        # A pinned element is a delta function; a potential evaluated on it
        # is a constant that cannot move the posterior, so "Fixed" stays the
        # honest and complete statement.
        if not contributions or kind == "fixed":
            return own, []
        supersede = any(c.supersedes_bounds for c in contributions)
        interval = self._interval_str(index, latex=True)
        notes = []
        for c in contributions:
            note = c.latex
            if c.supersedes_bounds and interval:
                # The phrase, not a hard-coded "normalized on", says how
                # this contribution relates to the interval: a normalized
                # density over it, or a barrier at its edges.
                note += f", {c.support_phrase} {interval}"
            notes.append(note)
        if supersede and kind == "bounds":
            return "", notes
        return own, notes

    def to_latex_prior_def(self, mark_for=None) -> str:
        """Generate the \\providecommand(s) for the prior column value.

        The command name is ``\\<latex_varname><idx>prior`` -- the same
        ``<idx>`` word suffix ``to_latex_def`` puts on the value macros -- so
        the table body can reference the prior symbolically rather than
        inlining the string.  A scalar parameter keeps the unsuffixed
        ``\\<latex_varname>prior``.

        ONE COMMAND PER ELEMENT, not per parameter: elements of a vector
        stopped sharing a prior when the manifest's per-element "overrides"
        channel arrived (GP and robust-likelihood hyperparameters are
        full-length vectors with the files that did not opt in pinned via
        ``sigma: 0``).  Reading element 0 and reusing it made such a vector
        report "Fixed" for its genuinely sampled elements -- a false
        statement of the prior in a published table.

        ``mark_for`` is the caller's note registry (text -> tablenote
        letter, outputs/latex.py's ``note_marks``): component-declared
        prior contributions render as shared table notes rather than
        inline cell text.  Without it (a caller with no note machinery)
        the historical inline composition is used.
        """
        n_elements = (
            int(np.prod(self.shape)) if self.shape not in ((), None) else 1
        )
        lines = []
        for i in range(n_elements):
            idx_str = idx_to_words(i) if n_elements > 1 else ""
            if mark_for is None:
                prior_str = self.get_prior_str(index=i, latex=True)
            else:
                prior_str, notes = self.prior_cell_and_notes(index=i)
                for note in notes:
                    prior_str += rf"\tablenotemark{{{mark_for(note)}}}"
            lines.append(
                rf"\providecommand{{\{self.latex_varname}{idx_str}prior}}"
                rf"{{{prior_str}}}" + "\n"
            )
        return "".join(lines)

    def _value_cells(self, idx_str: str, mode_suffixes: Optional[list]) -> str:
        """The Value column(s) of a table row.

        With ``mode_suffixes`` (multimodal table), sampled parameters get one
        cell per mode referencing the suffixed macros; fixed parameters span
        all mode columns with their single unsuffixed macro.
        """
        base = "\\" + self.latex_varname + idx_str
        if not mode_suffixes:
            return base + r"\dotfill"
        if self.posterior is None:
            return rf"\multicolumn{{{len(mode_suffixes)}}}{{c}}{{{base}}}"
        return " & ".join(base + sfx + r"\dotfill" for sfx in mode_suffixes)

    def to_table_line(
        self,
        sigfigs: int = 2,
        note_mark: Optional[str] = None,
        mode_suffixes: Optional[list] = None,
    ) -> str:
        if self.latex is None:
            raise ValueError(f"{self.label}: latex symbol not set.")
        if self.description is None:
            raise ValueError(f"{self.label}: description not set.")

        safe_unit = self.unit_latex.replace("$", "") if self.unit_latex else ""
        unit_text = "" if not safe_unit else rf" (\ensuremath{{{safe_unit}}})"
        mark_text = rf"\tablenotemark{{{note_mark}}}" if note_mark else ""
        # The Description column is trusted LaTeX (same contract as
        # table_note): descriptions may carry math ($\alpha$, subscripts).
        # Authors escape their own literal underscores/percent signs; the
        # audit in tests/test_prose.py::test_descriptions_are_valid_latex
        # catches raw specials in shipped defaults.yaml files.
        desc = self.description

        n_elements = np.prod(self.shape).astype(int) if self.shape != () else 1

        lines = []
        for i in range(n_elements):
            idx_str = idx_to_words(i) if n_elements > 1 else ""

            if n_elements > 1:
                if self.names and i < len(self.names):
                    clean_name = latex_escape(str(self.names[i]))
                    symbol = self.latex + r"_{\rm " + clean_name + r"}"
                else:
                    symbol = f"{self.latex}_{{{i}}}"
            else:
                symbol = self.latex

            if self.print_to_table:
                val_txt = self._value_cells(idx_str, mode_suffixes)
            else:
                if self.summary is None:
                    self.compute_summary()

                summ = (
                    self.summary[i]
                    if isinstance(self.summary, list)
                    else self.summary
                )
                val_txt = (
                    r"\ensuremath{"
                    + summ.latex_value(sigfigs=sigfigs)
                    + "}"
                    + r"\dotfill"
                )

            # Per-element prior macro (see to_latex_prior_def): a vector's
            # elements may carry different priors.
            prior_text = "\\" + self.latex_varname + idx_str + "prior"

            lines.append(
                rf"~~~~${symbol}$" + mark_text + rf"\dotfill & "
                rf"{desc}{unit_text}\dotfill & "
                rf"{val_txt} & "
                rf"{prior_text} \\" + "\n"
            )

        return "".join(lines)

    def to_table_line_at(
        self,
        index: int,
        sigfigs: int = 2,
        note_mark: Optional[str] = None,
        mode_suffixes: Optional[list] = None,
    ) -> str:
        """Single table row for element ``index``, without an instance subscript.

        Used when the enclosing section header already identifies the instance.
        """
        if self.latex is None:
            raise ValueError(f"{self.label}: latex symbol not set.")
        if self.description is None:
            raise ValueError(f"{self.label}: description not set.")

        n_elements = np.prod(self.shape).astype(int) if self.shape != () else 1
        idx_str = idx_to_words(index) if n_elements > 1 else ""

        safe_unit = self.unit_latex.replace("$", "") if self.unit_latex else ""
        unit_text = "" if not safe_unit else rf" (\ensuremath{{{safe_unit}}})"
        mark_text = rf"\tablenotemark{{{note_mark}}}" if note_mark else ""
        desc = self.description  # trusted LaTeX, same contract as table_note

        if self.print_to_table:
            val_txt = self._value_cells(idx_str, mode_suffixes)
        else:
            if self.summary is None:
                self.compute_summary()
            summ = (
                self.summary[index]
                if isinstance(self.summary, list)
                else self.summary
            )
            val_txt = (
                r"\ensuremath{"
                + summ.latex_value(sigfigs=sigfigs)
                + "}"
                + r"\dotfill"
            )

        prior_text = "\\" + self.latex_varname + idx_str + "prior"

        return (
            rf"~~~~${self.latex}$" + mark_text + rf"\dotfill & "
            rf"{desc}{unit_text}\dotfill & "
            rf"{val_txt} & "
            rf"{prior_text} \\" + "\n"
        )

    # ---------
    # Posterior summary
    # ---------
    @staticmethod
    def _summarize_array(arr: np.ndarray) -> Any:
        """Median + 68% interval over the LAST axis (the samples).

        Returns a PosteriorSummary, or a list of them for vector parameters.
        """

        def get_stat(data):
            if data.size == 0 or not np.isfinite(data).any():
                return PosteriorSummary(
                    median=float("nan"),
                    err_minus=float("nan"),
                    err_plus=float("nan"),
                )
            med = float(np.nanquantile(data, 0.5))
            lo = float(np.nanquantile(data, SIGMA_1_LOW))
            hi = float(np.nanquantile(data, SIGMA_1_HIGH))
            return PosteriorSummary(
                median=med, err_minus=med - lo, err_plus=hi - med
            )

        if arr.ndim > 1:
            # Flatten any extra vector dimensions, iterate over the first axis,
            # and compute statistics over the LAST axis (the samples)
            n_elem = int(np.prod(arr.shape[:-1]))
            arr_2d = arr.reshape(n_elem, arr.shape[-1])
            summary = [get_stat(arr_2d[i, :]) for i in range(n_elem)]

            # If the "vector" only has 1 element, unwrap it so it formats as a clean scalar
            if len(summary) == 1:
                summary = summary[0]
            return summary
        return get_stat(arr)

    def compute_summary(self) -> Any:
        """Median and 68% interval over the trace, in user units.

        The interval width is not a knob: ``_summarize_array`` reports the
        1-sigma quantiles every consumer (LaTeX tables, CSV, mode report)
        assumes.  This used to take an ``nsigma`` argument that nothing read,
        so ``compute_summary(nsigma=2)`` silently returned 1 sigma.
        """
        # arr from az.extract places the 'sample' dimension LAST.
        # Posterior is stored in user units (from the user-unit trace Deterministic).
        arr = np.asarray(
            getattr(self.posterior, "values", self.posterior), dtype=float
        )
        self.summary = self._summarize_array(arr)
        return self.summary

    def compute_mode_summaries(self, mode_labels, n_modes: int) -> Any:
        """Per-mode posterior summaries.

        ``mode_labels`` is an integer array aligned with the sample (last)
        axis of ``self.posterior``; -1 marks invalid/unassigned draws.  The
        result mirrors ``summary`` (PosteriorSummary or list of them), one
        entry per mode.
        """
        arr = np.asarray(
            getattr(self.posterior, "values", self.posterior), dtype=float
        )
        labels = np.asarray(mode_labels)
        if arr.ndim == 0 or arr.shape[-1] != labels.size:
            # Constant over the trace (e.g. generate_posterior's fixed branch
            # returns the bare value with no sample axis): identical in every
            # mode.
            self.mode_summaries = [self._summarize_array(arr)] * n_modes
        else:
            self.mode_summaries = [
                self._summarize_array(arr[..., labels == k])
                for k in range(n_modes)
            ]
        return self.mode_summaries
