"""
- Provide a single entrypoint to "materialize" the parameter inside a pm.Model() context: Parameter.build_pymc().
- Make user overrides explicit and predictable: users can only tighten bounds; sigma==0 means fixed/deterministic at mu/initval.
- Clean up posterior summarization using quantiles (median, +/- 1σ by default).
"""

from __future__ import annotations

import pytensor
import pytensor.tensor as pt
import pytensor.graph.basic  # Add this to be safe for the .ancestors call

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import math
from astropy import units as u
import re

import pymc as pm
import pytensor.tensor as pt

# local imports
import logging
from exozippy.constants import SIGMA_1_LOW, SIGMA_1_HIGH
from exozippy.potentials import soft_lower_bound, soft_upper_bound

logger = logging.getLogger(__name__)

Number = Union[int, float, np.floating]


# ----------------------------
# Helper functions
# ----------------------------

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
    - replace digits with words
    - prefix to avoid global collisions
    """
    old = [".","_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    new = ["","", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    var = label
    for o, n in zip(old, new):
        var = var.replace(o, n)
    return prefix + var

def _idx_to_words(n):
    words = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
             '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
             '8': 'eight', '9': 'nine'}
    return "".join(words[char] for char in str(n))

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
    if hasattr(raw_val, 'owner') or "TensorVariable" in str(type(raw_val)):
        return raw_val

    # 3. Handle evaluate-able tensors (for initvals)
    if hasattr(raw_val, 'eval'):
        try:
            raw_val = raw_val.eval()
        except:
            return np.full(n_elements, fill, dtype=float)

    arr = np.atleast_1d(raw_val)

    # 4. Handle arrays of tensors (rare, but happens in stacking)
    if arr.size > 0 and hasattr(arr[0], 'eval'):
        try:
            arr = np.array([float(x.eval()) if hasattr(x, 'eval') else float(x) for x in arr])
        except:
            return np.full(n_elements, fill, dtype=float)

    # 5. Scalar conversion (This is where the crash was!)
    if arr.size == 1:
        # Bypass float() if it's STILL a tensor (e.g. a 1-element tensor)
        if hasattr(arr[0], 'owner'): return arr[0]
        return np.full(n_elements, float(arr[0]), dtype=float)

    res = np.full(n_elements, fill, dtype=float)
    n_to_copy = min(n_elements, arr.size)
    res[:n_to_copy] = arr.astype(float)[:n_to_copy]
    return res

class UnitTranslator:
    # Essential "Pretty" Mapping
    SOLAR_DENSITY_UNIT = u.def_unit('rho_sun', 3.0 * u.M_sun / (4.0 * np.pi * u.R_sun ** 3))

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
        u.dex(u.cm / u.s ** 2): r"\rm cgs",
        u.g / u.cm ** 3: r"\rm g~cm$^{-3}$",
        SOLAR_DENSITY_UNIT : r"\rho_\odot",
        u.erg / u.second / u.cm ** 2: r"\rm erg~s$^{-1}$~cm$^{-2}$"
    }

    @classmethod
    def get_latex(cls, unit):
        """Strict translator: returns pretty string or raises ValueError."""
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
            return valid_unit.to_string('latex_inline').replace('$', '')

        except (TypeError, ValueError, AttributeError):
            # 3. If it's not a unit object or a string astropy understands
            raise ValueError(
                f"Unit '{unit}' is not a recognized Astropy unit"
                f"Specify valid units or set 'user_unit_latex' for" 
                f"{self.label} manually in your parameter files."
            )

# Example Usage:
# unit = u.solMass
# print(UnitTranslator.to_latex(unit)) -> "M_\odot"


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
        if math.isnan(self.median) or math.isnan(self.err_minus) or math.isnan(self.err_plus):
            return ("NaN", "NaN", "NaN")

        em = abs(self.err_minus)
        ep = abs(self.err_plus)
        if em == 0 and ep == 0:
            return (str(self.median), "0", "0")

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
    unit_latex: Optional[str] = "" # I'll keep a look up table for units, but this can be specified by the user ->

    internal_unit: Any = None # this is the internally used unit that simplifies the math
    initval: Optional[Number] = None
    init_scale: Optional[Number] = 1.0
    force_node: bool = False
    names: Optional[Sequence[str]] = None
    mask: Any = None

    # If expression is provided, parameter becomes deterministic (pm.Deterministic).
    # You can pass expression at build time too.
    expression: Any = None
    shape: tuple = ()

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
    is_derived: bool = False
    is_sampled: bool = False
    # Raw-space starting values for the sampled elements (set in build_pymc):
    # 0 for logit elements, (initval - mu)/sigma for Gaussian-path elements.
    raw_initval: Optional[np.ndarray] = None

    user_params: Optional[Mapping[str, Mapping[str, Any]]] = None
    auto_estimated: bool = False

    # LaTeX/table metadata
    latex: Optional[str] = ""
    description: Optional[str] = ""
    latex_prefix: str = "ez"

    # Runtime fields
    value: Any = field(default=None, init=False)  # pm RV or pm.Deterministic after build_pymc()
    latex_varname: str = field(default="", init=False)
    posterior: Any = None  # user stores idata posterior samples here if desired
    summary: Optional[PosteriorSummary] = field(default=None, init=False)
    table_note: Optional[str] = None

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
            self.unit = [u.dimensionless_unscaled if self.unit is None else self.unit]

        # 3. GET LATEX DISPLAY NAME (Use the first unit in the list)
        try:
            self.unit_latex = UnitTranslator.get_latex(self.unit[0])
        except:
            self.unit_latex = ""

        # 4. STRUCTURAL NAMING
        self.latex_varname = _latex_varname(self.label, prefix=self.latex_prefix)

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
            if hasattr(raw_val, 'owner'):
                return raw_val

            # Evaluate constant tensor nodes (e.g. pt.constant(5.0))
            if hasattr(raw_val, 'eval'):
                try:
                    raw_val = raw_val.eval()
                except Exception:
                    # Free-variable tensor that can't eval: not valid as a bound/scale
                    return np.full(np.atleast_1d(factors).shape, np.nan)

            arr = np.atleast_1d(raw_val)

            # Final check: Ensure we aren't storing an object-array of Tensors
            if arr.dtype == object:
                arr = np.array([float(x.eval()) if hasattr(x, 'eval') else float(x) for x in arr])

            return arr.astype(float) / factors
        # --- APPLY THE CONVERSION ---
        self.initval = convert(self.initval)
        self.init_scale = convert(self.init_scale)
        self.lower = convert(self.lower)
        self.upper = convert(self.upper)
        self.mu = convert(self.mu)
        self.sigma = convert(self.sigma)

    def get_value(self, point):
        """
        Smart lookup: returns the value from the 'point' dict if it exists,
        otherwise 'wakes up' the expression and calculates it.
        """
        # 1. Direct hit (it was a named node)
        if self.label in point:
            return float(point[self.label])

        # 2. It's an anonymous parameter
        if self.expression is not None:
            # We use the existing generate_posterior logic,
            # but treat the 'point' as a bundle with 1 draw.
            # Convert point values to 0-d arrays for pytensor.function
            bundle = {k: np.array(v) for k, v in point.items()}
            return float(self.generate_posterior(bundle))

        # 3. Fallback to initval if everything else fails
        if self.initval is not None:
            return float(self.initval)

        raise KeyError(f"Parameter {self.label} not found in point and has no expression.")

    def get_display_label(self, index=0):
        parts = self.label.split('.')
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

        All raw variables are N(0,1). init_scale is always in physical units;
        for logit params it is converted to logit-space internally via the
        Jacobian and affects only tuning/conditioning, never the posterior.
        """
        import pytensor.tensor as pt
        import pymc as pm

        expr_raw = self.expression if expression is None else expression

        # 1. SETUP SHAPES
        actual_shape = self.shape if isinstance(self.shape, tuple) else (self.shape,)
        n_elements = int(np.prod(actual_shape)) if actual_shape != () else 1

        inits = to_vec(self.initval, n_elements, fill=0.0)
        scales = to_vec(self.init_scale, n_elements, fill=1.0)
        mus = to_vec(self.mu, n_elements, fill=np.nan)
        sigmas = to_vec(self.sigma, n_elements, fill=np.nan)
        lowers = to_vec(self.lower, n_elements, fill=-np.inf)
        uppers = to_vec(self.upper, n_elements, fill=np.inf)

        # 2. IDENTIFY ROLES
        is_derived = np.full(n_elements, expr_raw is not None, dtype=bool)
        is_fixed = ((sigmas == 0) | (scales <= 1e-12)) & ~is_derived
        is_sampled = ~(is_fixed | is_derived)

        # Warn if user tried to fix a derived parameter — sigma=0 has no effect on derived params.
        if np.any(is_derived & (sigmas == 0)):
            logger.warning(
                f"Parameter '{self.label}': sigma=0 has no effect on a derived parameter "
                f"To hold it constant, you must fix the corresponding sampled parameter(s)."
            )
        self.is_sampled = is_sampled

        if np.any(is_sampled):
            if self.lower is None or self.upper is None:
                raise ValueError(
                    f"Developer Error: Sampled parameter '{self.label}' MUST have explicit "
                    f"'lower' and 'upper' bounds defined in its defaults.yaml."
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

        # Gaussian: val = gaussian_mus + gaussian_scales * raw
        gaussian_mus = np.copy(inits)
        gaussian_scales = np.copy(scales)

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
                # (e.g. cosalpha/sinalpha where sigma=1 encodes a uniform-angle
                # prior but init_scale reflects the actual alpha uncertainty).
                whiten = (min(sigmas[i], scales[i]) if has_sigma else scales[i])
                # Keep the start off the exact bound. The floor is in units of
                # the whitening scale (1e-6*scale inside the bound is
                # "essentially at the bound" in problem units); a span-based
                # floor would be arbitrarily large for wide bounds. The 1e-12
                # absolute floor keeps logit(q) within the ±30 sigmoid clip.
                q_floor = min(max(1e-6 * whiten / span, 1e-12), 0.25)
                q_init = np.clip(q_raw, q_floor, 1.0 - q_floor)
                if q_init != q_raw:
                    logger.warning(
                        f"Parameter '{self.label}'[{i}]: initval {inits[i]} is at or "
                        f"within 1e-6*init_scale of bounds [{lowers[i]}, {uppers[i]}]; "
                        f"starting value nudged to {lowers[i] + q_init * span}."
                    )
                logit_q_inits[i] = np.log(q_init / (1.0 - q_init))
                jac = q_init * (1.0 - q_init) * span  # dval/d(logit_q) at initval
                # Near a wall jac → 0 and whiten/jac would explode, saturating
                # the sigmoid within one tiny raw step (parameter frozen at the
                # wall). Flooring jac at min(whiten, span/4) caps the logit
                # step at ~1, so a pinned start escapes multiplicatively —
                # one e-fold in (val - bound) per unit raw step — while
                # interior starts are unaffected.
                init_scale_logits[i] = whiten / max(jac, min(whiten, span / 4.0))
            elif has_sigma:
                # Unbounded with sigma: non-centered Gaussian; the raw N(0,1)
                # IS the prior.
                has_mu = not np.isnan(mus[i])
                gaussian_mus[i] = mus[i] if has_mu else inits[i]
                gaussian_scales[i] = sigmas[i]
            else:
                # Unbounded, no sigma: fall back to linear with N(0,1)
                gaussian_mus[i] = inits[i]
                gaussian_scales[i] = scales[i]

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
                    raw_initvals[j] = ((inits[i] - gaussian_mus[i])
                                       / max(gaussian_scales[i], 1e-30))
            # Saved so run.py can build the true raw starting point explicitly
            # (model.initial_point() is not trusted to honor these).
            self.raw_initval = raw_initvals
            par_raw = pm.Normal(f"{self.label}_raw",
                                mu=0,
                                sigma=1.0,
                                shape=len(idx),
                                initval=raw_initvals)
            for j, actual_idx in enumerate(idx):
                raw_elements[actual_idx] = par_raw[j]

        # 5. RECONSTRUCT PHYSICAL VALUE
        raw_vector = pt.stack(raw_elements)

        if expr_raw is not None:
            phys_val = expr_raw() if callable(expr_raw) else expr_raw
        else:
            # Logit branch: lower + (upper-lower)*sigmoid(logit_init + scale_logit*raw)
            lq = pt.as_tensor_variable(logit_q_inits) + pt.as_tensor_variable(init_scale_logits) * raw_vector
            phys_logit = (pt.as_tensor_variable(lowers)
                          + pt.as_tensor_variable(uppers - lowers) * pt.sigmoid(pt.clip(lq, -30.0, 30.0)))

            # Gaussian / linear branch: mu + sigma * raw  (or initval + scale * raw)
            phys_linear = pt.as_tensor_variable(gaussian_mus) + pt.as_tensor_variable(gaussian_scales) * raw_vector

            if np.all(use_logit):
                phys_val = phys_logit
            elif not np.any(use_logit):
                phys_val = phys_linear
            else:
                phys_val = pt.where(pt.as_tensor_variable(use_logit), phys_logit, phys_linear)

        # Strip Astropy units
        if hasattr(phys_val, 'value') and hasattr(phys_val, 'unit'):
            phys_val = phys_val.value

        if isinstance(phys_val, (list, tuple)):
            phys_val = pt.stack(list(phys_val))
        elif isinstance(phys_val, np.ndarray) and phys_val.dtype == object:
            phys_val = pt.stack(phys_val.tolist())

        # 6. ASSIGN TO SELF.VALUE
        track_node = bool(np.any(is_sampled)) or self.force_node

        if actual_shape == ():
            val_to_save = phys_val if expr_raw is not None else phys_val[0]
        else:
            val_to_save = pt.broadcast_to(pt.as_tensor_variable(phys_val), actual_shape)

        if track_node:
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
        gaussian_prior_mask = ((is_derived | (is_sampled & use_logit & has_sigma_prior))
                               & ~np.isnan(sigmas) & (sigmas > 0))
        if np.any(gaussian_prior_mask):
            prior_mus = np.where(~np.isnan(mus), mus, inits)
            mask = pt.as_tensor_variable(gaussian_prior_mask)
            penalty = -0.5 * ((val_flat - pt.as_tensor_variable(prior_mus))
                              / pt.as_tensor_variable(np.where(sigmas > 0, sigmas, 1.0))) ** 2
            pm.Potential(f"gaussian_prior.{self.label}",
                         pm.math.sum(pt.where(mask, penalty, 0.0)))

        # B. Soft bounds for derived params (and the rare half-bounded sampled
        #    param, where only one bound is finite so the logit transform does
        #    not apply). Fully-bounded sampled params: sigmoid is a hard
        #    constraint — no barrier needed.
        #    Fixed params: constant, so barrier adds only a harmless constant — skip.
        needs_barrier = (is_derived | (is_sampled & ~use_logit)) & ~is_fixed
        if np.any(needs_barrier):
            # Use init_scale for barrier steepness (falls back to gaussian_scales
            # for Gaussian params, where gaussian_scales = sigma).
            barrier_scales = np.where(use_logit, scales, gaussian_scales)

            has_lower = ~np.isinf(lowers) & needs_barrier
            if np.any(has_lower):
                mask = pt.as_tensor_variable(has_lower)
                penalty = soft_lower_bound(
                    val_flat, pt.as_tensor_variable(lowers), barrier_scales)
                pm.Potential(f"low_bound.{self.label}",
                             pm.math.sum(pt.where(mask, penalty, 0.0)))

            has_upper = ~np.isinf(uppers) & needs_barrier
            if np.any(has_upper):
                mask = pt.as_tensor_variable(has_upper)
                penalty = soft_upper_bound(
                    val_flat, pt.as_tensor_variable(uppers), barrier_scales)
                pm.Potential(f"up_bound.{self.label}",
                             pm.math.sum(pt.where(mask, penalty, 0.0)))

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
            log_jac = -pt.softplus(lq) - pt.softplus(-lq)
            correction = pt.where(logit_mask,
                                  log_jac + 0.5 * pt.sqr(raw_vector),
                                  pt.zeros_like(raw_vector))
            pm.Potential(f"logit_uniform_prior.{self.label}", pt.sum(correction))

        return self.value

    def generate_posterior(self, posterior_bundle):
        if self.label in posterior_bundle:
            return posterior_bundle[self.label]
        if self.expression is None:
            return None

        expr = self.expression() if callable(self.expression) else self.expression

        # --- Strip Astropy Units before graph walking ---
        if hasattr(expr, 'value') and hasattr(expr, 'unit'):
            expr = expr.value

        all_nodes = pytensor.graph.basic.ancestors([expr])

        inputs_in_posterior = [
            n for n in all_nodes
            if hasattr(n, 'name') and n.name in posterior_bundle
        ]

        # fixed parameter, just return the scalar
        if not inputs_in_posterior:
            val = np.asarray(expr.eval(), dtype=float)
            if val.size > 1:
                return val
            return val.item()

        # 1. Compile the function for a single evaluation
        calc_func = pytensor.function(
            inputs_in_posterior,
            expr,
            on_unused_input='ignore'
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

            if n_samples is None:
                n_samples = val.shape[0]

            input_data.append(val)

        # 3. Evaluate the first sample to dynamically determine the output dimension
        # Reshape each sample slice to match the PyTensor node's expected ndim.
        # A scalar variable lands as 0-D after arr[0], but build_pymc may have
        # compiled calc_func with a 1-D (n=1) input; atleast_nd fixes that.
        def _match_ndim(val, node):
            target = node.ndim if hasattr(node, 'ndim') else 0
            while np.ndim(val) < target:
                val = np.atleast_1d(val)
            return val

        first_args = [_match_ndim(arr[0], n)
                      for arr, n in zip(input_data, inputs_in_posterior)]
        first_result = np.asarray(calc_func(*first_args))

        # 4. Loop through the remaining samples
        # Create an array of shape (n_samples, *shape)
        result = np.zeros((n_samples,) + first_result.shape)
        result[0] = first_result

        for i in range(1, n_samples):
            args = [_match_ndim(arr[i], n)
                    for arr, n in zip(input_data, inputs_in_posterior)]
            result[i] = calc_func(*args)

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
        fn = model.compile_fn(self.value, on_unused_input='ignore')
        return fn(point)

    def _get_conversion_factors(self):
        """
        Calculates the numerical conversion factor from internal -> user units.
        Safely handles self.unit as a single Unit, a scalar Quantity, or a list/array.
        Halts immediately on invalid linear unit conversions.
        """
        is_sequence = isinstance(self.unit, (list, tuple)) or \
                      (isinstance(self.unit, np.ndarray) and getattr(self.unit, 'ndim', 0) > 0)

        def _process_single(u_user):
            target_u = getattr(u_user, 'unit', u_user)
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
            return np.array([_process_single(u) for u in self.unit], dtype=np.float64)

        return _process_single(self.unit)
    def _get_conversion_factors_old(self):
        """
        Calculates the numerical conversion factor from internal -> user units.
        Safely handles self.unit as a single Unit, a scalar Quantity, or a list/array.
        """
        # A list/tuple is safe. An ndarray/Quantity is only safe if it has dimensions.
        is_sequence = isinstance(self.unit, (list, tuple)) or \
                      (isinstance(self.unit, np.ndarray) and getattr(self.unit, 'ndim', 0) > 0)

        if is_sequence:
            factors = []
            for u_user in self.unit:
                # getattr extracts the base Unit if u_user is accidentally a Quantity
                target = getattr(u_user, 'unit', u_user)
                factors.append(self.internal_unit.to(target))
            return np.array(factors, dtype=np.float64)

        # Scalar fallback
        target = getattr(self.unit, 'unit', self.unit)
        return float(self.internal_unit.to(target))

    # converts user units to internal units
    def to_internal(self, val=None):
        target = val if val is not None else self.value
        return target/self._get_conversion_factors()

    # converts internal units to user units
    def from_internal(self, val=None):
        target = val if val is not None else self.value
        # Safety check for unitless parameters
        if self.unit is None or self.internal_unit is None:
            return target
        return target * self._get_conversion_factors()

    # converts internal units to arbitrary units
    def to_unit(self, target_unit: Any) -> Any:
        return self.value * self.internal_unit.to(target_unit).value


    # ---------
    # LaTeX helpers
    # ---------

    def to_latex_def(self, sigfigs: int = 2) -> str:

        # FIXED PARAMETER PATH
        if self.posterior is None:
            if self.initval is not None:
                physical_inits = self.from_internal(self.initval)
                inits = np.atleast_1d(physical_inits)

                if len(inits) > 1:
                    lines = []
                    for i, val in enumerate(inits):
                        idx_str = _idx_to_words(i)
                        lines.append(
                            rf"\providecommand{{\{self.latex_varname}{idx_str}}}{{\ensuremath{{\equiv {val}}}}}" + "\n")
                    return "".join(lines)
                else:
                    return rf"\providecommand{{\{self.latex_varname}}}{{\ensuremath{{\equiv {inits[0]}}}}}" + "\n"
            return ""

        # SAMPLED PARAMETER PATH
        if self.summary is None:
            self.compute_summary()

        if isinstance(self.summary, list):
            lines = []
            for i, summ in enumerate(self.summary):
                val = summ.latex_value(sigfigs=sigfigs)
                idx_str = _idx_to_words(i)
                lines.append(rf"\providecommand{{\{self.latex_varname}{idx_str}}}{{\ensuremath{{{val}}}}}" + "\n")
            return "".join(lines)

        val = self.summary.latex_value(sigfigs=sigfigs)
        return rf"\providecommand{{\{self.latex_varname}}}{{\ensuremath{{{val}}}}}" + "\n"

    def get_unit_str(self, index=0):
        u_list = np.atleast_1d(self.unit)
        u_obj = u_list[index] if index < len(u_list) else u_list[0]
        return u_obj.to_string() if u_obj and u_obj.to_string() != 'dimensionless' else ""

    def get_prior_str(self, index=0, latex=True):
        def _scalar(val):
            if val is None: return None
            arr = np.atleast_1d(val)
            raw = arr[index] if index < len(arr) else arr[0]
            if hasattr(raw, 'eval'):
                try:
                    raw = raw.eval()
                except:
                    return None
            f_val = float(raw)
            if np.isnan(f_val): return None
            if self.unit is None or self.internal_unit is None: return f_val
            f = np.atleast_1d(self._get_conversion_factors())
            return f_val * float(f[index] if index < len(f) else f[0])

        def _fmt(val, is_latex=True):
            if val is None or np.isnan(val): return "nan"
            if np.isinf(val): return (r"\infty" if val > 0 else r"-\infty") if is_latex else (
                "inf" if val > 0 else "-inf")
            if 0.001 <= abs(val) < 10000: return f"{val:.4f}".rstrip("0").rstrip(".")
            return f"{val:.2e}"

        sig = _scalar(self.sigma)
        if sig == 0: return "Fixed"

        lo = _scalar(self.lower)
        hi = _scalar(self.upper)

        # Determine if there are actual constraints to print
        has_prior = (sig is not None and sig > 0)
        has_bounds = (lo is not None or hi is not None)

        # Derived parameters with no custom constraint have no prior to display.
        if self.expression is not None and not (has_prior or has_bounds):
            return ""

        mu = _scalar(self.mu)
        if mu is None:
            mu = _scalar(self.initval)

        if not latex:
            strs = []
            if has_prior:
                strs.append(f"N({_fmt(mu, False)}, {_fmt(sig, False)})")
            if strs: return " * ".join(strs)

            if has_bounds:
                l_s = _fmt(lo, False) if (lo is not None and not np.isinf(lo)) else ""
                h_s = _fmt(hi, False) if (hi is not None and not np.isinf(hi)) else ""
                if l_s and h_s: return f"U({l_s}, {h_s})"
                if l_s: return f"> {l_s}"
                if h_s: return f"< {h_s}"

            if self.expression is not None:
                return ""

        # --- LaTeX Formatting Block ---
        strs = []
        if has_prior:
            strs.append(rf"$\mathcal{{N}}({_fmt(mu)}, {_fmt(sig)})$")

        if strs: return r" $\times$ ".join(strs)

        if has_bounds:
            l_s, h_s = _fmt(lo), _fmt(hi)

            # Safe infinity checks to avoid TypeErrors if lo/hi are None
            lo_is_inf = (lo is None) or np.isinf(lo)
            hi_is_inf = (hi is None) or np.isinf(hi)

            if not lo_is_inf and not hi_is_inf: return rf"$\mathcal{{U}}({l_s}, {h_s})$"
            if not lo_is_inf: return rf"$> {l_s}$"
            if not hi_is_inf: return rf"$< {h_s}$"

        return ""

    def to_latex_prior_def(self) -> str:
        """Generate a \\providecommand for the prior column value.

        The command name is ``\\<latex_varname>prior`` so the table body can
        reference it symbolically rather than inlining the prior string.  A
        single command is generated per parameter (not per element) because
        all elements of a vector parameter share the same prior.
        """
        prior_str = self.get_prior_str(index=0, latex=True)
        if not prior_str:
            return rf"\providecommand{{\{self.latex_varname}prior}}{{}}" + "\n"
        return rf"\providecommand{{\{self.latex_varname}prior}}{{{prior_str}}}" + "\n"

    def to_table_line(self, sigfigs: int = 2) -> str:
        if self.latex is None:
            raise ValueError(f"{self.label}: latex symbol not set.")
        if self.description is None:
            raise ValueError(f"{self.label}: description not set.")

        safe_unit = self.unit_latex.replace('$', '') if self.unit_latex else ""
        unit_text = "" if not safe_unit else rf" (\ensuremath{{{safe_unit}}})"

        n_elements = np.prod(self.shape).astype(int) if self.shape != () else 1

        lines = []
        for i in range(n_elements):
            idx_str = _idx_to_words(i) if n_elements > 1 else ""

            if n_elements > 1:
                if self.names and i < len(self.names):
                    clean_name = str(self.names[i]).replace("_", r"\_")
                    symbol = self.latex + r"_{\rm " + clean_name + r"}"
                else:
                    symbol = f"{self.latex}_{{{i}}}"
            else:
                symbol = self.latex

            if self.print_to_table:
                val_txt = "\\" + self.latex_varname + idx_str
            else:
                if self.summary is None:
                    self.compute_summary()

                summ = self.summary[i] if isinstance(self.summary, list) else self.summary
                val_txt = r"\ensuremath{" + summ.latex_value(sigfigs=sigfigs) + "}"

            prior_text = "\\" + self.latex_varname + "prior"

            lines.append(
                rf"~~~~${symbol}$\dotfill & "
                rf"{self.description}{unit_text}\dotfill & "
                rf"{val_txt}\dotfill & "
                rf"{prior_text} \\" + "\n"
            )

        return "".join(lines)

    def to_table_line_at(self, index: int, sigfigs: int = 2) -> str:
        """Single table row for element ``index``, without an instance subscript.

        Used when the enclosing section header already identifies the instance.
        """
        if self.latex is None:
            raise ValueError(f"{self.label}: latex symbol not set.")
        if self.description is None:
            raise ValueError(f"{self.label}: description not set.")

        n_elements = np.prod(self.shape).astype(int) if self.shape != () else 1
        idx_str = _idx_to_words(index) if n_elements > 1 else ""

        safe_unit = self.unit_latex.replace('$', '') if self.unit_latex else ""
        unit_text = "" if not safe_unit else rf" (\ensuremath{{{safe_unit}}})"

        if self.print_to_table:
            val_txt = "\\" + self.latex_varname + idx_str
        else:
            if self.summary is None:
                self.compute_summary()
            summ = self.summary[index] if isinstance(self.summary, list) else self.summary
            val_txt = r"\ensuremath{" + summ.latex_value(sigfigs=sigfigs) + "}"

        prior_text = "\\" + self.latex_varname + "prior"

        return (
            rf"~~~~${self.latex}$\dotfill & "
            rf"{self.description}{unit_text}\dotfill & "
            rf"{val_txt}\dotfill & "
            rf"{prior_text} \\" + "\n"
        )

    # ---------
    # Posterior summary
    # ---------
    def compute_summary(self, nsigma: float = 1.0) -> Any:
        # arr from az.extract places the 'sample' dimension LAST
        arr = getattr(self.posterior, "values", self.posterior)
        arr = self.from_internal(arr)

        def get_stat(data):
            med = float(np.nanquantile(data, 0.5))
            lo = float(np.nanquantile(data, SIGMA_1_LOW))
            hi = float(np.nanquantile(data, SIGMA_1_HIGH))
            return PosteriorSummary(median=med, err_minus=med - lo, err_plus=hi - med)

        if arr.ndim > 1:
            # Flatten any extra vector dimensions, iterate over the first axis,
            # and compute statistics over the LAST axis (the samples)
            arr_2d = arr.reshape(-1, arr.shape[-1])
            self.summary = [get_stat(arr_2d[i, :]) for i in range(arr_2d.shape[0])]

            # If the "vector" only has 1 element, unwrap it so it formats as a clean scalar
            if len(self.summary) == 1:
                self.summary = self.summary[0]
        else:
            self.summary = get_stat(arr)

        return self.summary