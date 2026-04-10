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

import pymc as pm
import pytensor.tensor as pt

# local imports
from exozippy.constants import SIGMA_1_LOW, SIGMA_1_HIGH

import ipdb


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
    gaussian_width: Optional[Number] = None

    print_to_table: bool = True
    user_modified: bool = False
    user_prior_modified: bool = False

    user_params: Optional[Mapping[str, Mapping[str, Any]]] = None

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

            # 1. If it's a symbolic PyTensor node, execute it to get the raw numbers
            if hasattr(val, 'eval'):
                try:
                    val = val.eval()
                except Exception:
                    # If it cannot be evaluated (e.g., missing inputs), safely fallback
                    return np.full(np.atleast_1d(factors).shape, np.nan)

            arr = np.atleast_1d(val)

            # 2. In case it's an array containing individual PyTensor objects
            if arr.size > 0 and hasattr(arr[0], 'eval'):
                try:
                    arr = np.array([float(x.eval()) if hasattr(x, 'eval') else float(x) for x in arr])
                except Exception:
                    return np.full(np.atleast_1d(factors).shape, np.nan)

            # 3. Safe to cast and convert (User -> Internal)
            return arr.astype(float) / factors

        self.initval = convert(self.initval)
        self.init_scale = convert(self.init_scale)
        self.lower = convert(self.lower)
        self.upper = convert(self.upper)
        self.mu = convert(self.mu)
        self.sigma = convert(self.sigma)
        self.gaussian_width = convert(self.gaussian_width)

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
        Materializes the Parameter in the PyMC graph using Non-Centered Scaling.
        Assumes self.initval, lower, upper, mu, sigma, and init_scale
        are already in INTERNAL UNITS (broadcasted to self.shape).
        """
        import pytensor.tensor as pt
        import pymc as pm

        expr_raw = self.expression if expression is None else expression

        # 1. SETUP SHAPES
        actual_shape = self.shape if isinstance(self.shape, tuple) else (self.shape,)
        n_elements = int(np.prod(actual_shape)) if actual_shape != () else 1

        # 2. PREP WORK VECTORS
        def to_vec(val, fill=np.nan):
            if val is None: return np.full(n_elements, fill, dtype=float)
            if hasattr(val, 'eval'):
                try:
                    val = val.eval()
                except:
                    return np.full(n_elements, fill, dtype=float)
            arr = np.atleast_1d(val)
            if arr.size > 0 and hasattr(arr[0], 'eval'):
                try:
                    arr = np.array([float(x.eval()) if hasattr(x, 'eval') else float(x) for x in arr])
                except:
                    return np.full(n_elements, fill, dtype=float)
            if arr.size == 1: return np.full(n_elements, float(arr[0]), dtype=float)
            res = np.full(n_elements, fill, dtype=float)
            n_to_copy = min(n_elements, arr.size)
            res[:n_to_copy] = arr.astype(float)[:n_to_copy]
            return res

        inits = to_vec(self.initval, fill=0.0)
        scales = to_vec(self.init_scale, fill=1.0)
        mus = to_vec(self.mu, fill=np.nan)
        sigmas = to_vec(self.sigma, fill=np.nan)
        g_widths = to_vec(self.gaussian_width, fill=np.nan)
        lowers = to_vec(self.lower, fill=-np.inf)
        uppers = to_vec(self.upper, fill=np.inf)

        # 3. IDENTIFY ROLES & PARAMETERIZATION SCENARIOS
        is_derived = np.full(n_elements, expr_raw is not None, dtype=bool)
        is_fixed = ((sigmas == 0) | (scales <= 1e-12)) & ~is_derived
        is_sampling = ~(is_fixed | is_derived)

        # --- DYNAMIC PARAMETERIZATION LOGIC ---
        transform_mus = np.copy(inits)
        transform_scales = np.copy(scales)
        raw_sigmas = np.full(n_elements, 1000.0)
        apply_gwidth_potential = np.zeros(n_elements, dtype=bool)

        for i in range(n_elements):
            if not is_sampling[i]:
                continue

            has_sigma = not np.isnan(sigmas[i]) and sigmas[i] > 0
            has_gwidth = not np.isnan(g_widths[i]) and g_widths[i] > 0
            has_mu = not np.isnan(mus[i])
            actual_mu = mus[i] if has_mu else inits[i]

            if has_sigma and has_gwidth:
                # Case 4: pm.Normal with mu +/- sigma, add extra potential for gaussian_width
                transform_mus[i] = actual_mu
                transform_scales[i] = sigmas[i]
                raw_sigmas[i] = 1.0
                apply_gwidth_potential[i] = True

            elif has_sigma or has_gwidth:
                # Case 3: pm.Normal with mu +/- sigma (or g_width), no extra potential needed
                transform_mus[i] = actual_mu
                transform_scales[i] = sigmas[i] if has_sigma else g_widths[i]
                raw_sigmas[i] = 1.0

            else:
                # Case 1 & 2: Nothing or init_scale -> pm.Normal with 1000*init_scale
                transform_mus[i] = inits[i]
                transform_scales[i] = scales[i]
                raw_sigmas[i] = 1000.0

        raw_elements = [None] * n_elements

        # A. FIXED OR DERIVED: Constant 0.0 in the "raw" unit-normal space
        if np.any(is_fixed | is_derived):
            for i in np.where(is_fixed | is_derived)[0]:
                raw_elements[i] = pt.constant(0.0)

        # B. SAMPLING
        if np.any(is_sampling):
            idx = np.where(is_sampling)[0]
            # PyMC seamlessly supports vectorized mixed scales!
            par_raw = pm.Normal(f"{self.label}_raw",
                                mu=0,
                                sigma=raw_sigmas[idx],
                                shape=len(idx),
                                initval=np.zeros(len(idx)))

            for j, actual_idx in enumerate(idx):
                raw_elements[actual_idx] = par_raw[j]

        # 4. RECONSTRUCT PHYSICAL VALUE
        raw_vector = pt.stack(raw_elements)

        if expr_raw is not None:
            phys_val = expr_raw() if callable(expr_raw) else expr_raw
        else:
            phys_val = (raw_vector * pt.as_tensor_variable(transform_scales)) + pt.as_tensor_variable(transform_mus)

        # 5. ASSIGN TO SELF.VALUE
        track_node = bool(np.any(is_sampling)) or self.force_node

        if actual_shape == ():
            val_to_save = phys_val if expr_raw is not None else phys_val[0]
        else:
            val_to_save = phys_val.reshape(actual_shape)

        if track_node:
            self.value = pm.Deterministic(self.label, val_to_save)
        else:
            self.value = val_to_save

        # 6. APPLY SCIENTIFIC PRIORS & SOFT BOUNDARIES
        softness = 0.01
        # Use transform_scales so bounds adapt cleanly to tight sigmas
        steepness_val = 4.4 / (np.maximum(transform_scales, 1e-12) * softness)
        steepness = pt.as_tensor_variable(steepness_val)

        val_flat = pt.flatten(self.value)

        # A. Additional Gaussian Width Potentials (Only triggers for Case 4)
        if np.any(apply_gwidth_potential):
            mask = pt.as_tensor_variable(apply_gwidth_potential)
            penalty = -0.5 * ((val_flat - pt.as_tensor_variable(transform_mus)) / pt.as_tensor_variable(g_widths)) ** 2
            pm.Potential(f"gwidth_prior.{self.label}", pm.math.sum(pt.where(mask, penalty, 0.0)))

        # B. Soft Lower Bounds
        has_lower = ~np.isinf(lowers)
        if np.any(has_lower):
            mask = pt.as_tensor_variable(has_lower)
            penalty = pm.math.log(pt.sigmoid((val_flat - pt.as_tensor_variable(lowers)) * steepness))
            pm.Potential(f"low_bound.{self.label}", pm.math.sum(pt.where(mask, penalty, 0.0)))

        # C. Soft Upper Bounds
        has_upper = ~np.isinf(uppers)
        if np.any(has_upper):
            mask = pt.as_tensor_variable(has_upper)
            penalty = pm.math.log(pt.sigmoid((pt.as_tensor_variable(uppers) - val_flat) * steepness))
            pm.Potential(f"up_bound.{self.label}", pm.math.sum(pt.where(mask, penalty, 0.0)))

        return self.value

    def generate_posterior(self, posterior_bundle):
        if self.label in posterior_bundle:
            return posterior_bundle[self.label]
        if self.expression is None:
            return None

        expr = self.expression() if callable(self.expression) else self.expression
        all_nodes = pytensor.graph.basic.ancestors([expr])

        inputs_in_posterior = [
            n for n in all_nodes
            if hasattr(n, 'name') and n.name in posterior_bundle
        ]

        # fixed parameter, just return the scalar
        if not inputs_in_posterior:
            return float(expr.eval())

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
        first_args = [arr[0] for arr in input_data]
        first_result = np.asarray(calc_func(*first_args))

        # 4. Loop through the remaining samples
        # Create an array of shape (n_samples, *shape)
        result = np.zeros((n_samples,) + first_result.shape)
        result[0] = first_result

        for i in range(1, n_samples):
            args = [arr[i] for arr in input_data]
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

        # FIXED PARAMETER PATH (Restored)
        if self.posterior is None:
            if self.initval is not None:
                inits = np.atleast_1d(self.initval)
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

        if self.expression is not None: return "Derived"

        sig = _scalar(self.sigma)
        if sig == 0: return "Fixed"

        mu = _scalar(self.mu)
        if mu is None:
            mu = _scalar(self.initval)

        g_w = _scalar(self.gaussian_width)

        if not latex:
            strs = []
            if sig is not None and sig > 0:
                strs.append(f"N({_fmt(mu, False)}, {_fmt(sig, False)})")
            if g_w is not None and g_w > 0:
                strs.append(f"N({_fmt(mu, False)}, {_fmt(g_w, False)})")
            if strs: return " * ".join(strs)

            lo, hi = _scalar(self.lower), _scalar(self.upper)
            if lo is not None or hi is not None:
                l_s, h_s = _fmt(lo, False) if not np.isinf(lo) else "", _fmt(hi, False) if not np.isinf(hi) else ""
                if l_s and h_s: return f"U({l_s}, {h_s})"
                if l_s: return f"> {l_s}"
                if h_s: return f"< {h_s}"
            return ""

        strs = []
        if sig is not None and sig > 0:
            strs.append(rf"$\mathcal{{N}}({_fmt(mu)}, {_fmt(sig)})$")
        if g_w is not None and g_w > 0:
            strs.append(rf"$\mathcal{{N}}({_fmt(mu)}, {_fmt(g_w)})$")

        if strs: return r" $\times$ ".join(strs)

        lo, hi = _scalar(self.lower), _scalar(self.upper)
        if lo is not None or hi is not None:
            l_s, h_s = _fmt(lo), _fmt(hi)
            if not np.isinf(lo) and not np.isinf(hi): return rf"$\mathcal{{U}}({l_s}, {h_s})$"
            if not np.isinf(lo): return rf"$> {l_s}$"
            if not np.isinf(hi): return rf"$< {h_s}$"
        return ""

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

            # --- THE NEW SUBSCRIPT LOGIC ---
            if n_elements > 1:
                if self.names and i < len(self.names):
                    # Wrap in \rm so it looks clean, and escape underscores if they exist
                    clean_name = str(self.names[i]).replace("_", r"\_")
                    # Use raw string concatenation to avoid f-string backslash errors
                    symbol = self.latex + r"_{\rm " + clean_name + r"}"
                else:
                    symbol = f"{self.latex}_{{{i}}}"
            else:
                symbol = self.latex
            # -------------------------------

            if self.print_to_table:
                val_txt = "\\" + self.latex_varname + idx_str
            else:
                if self.summary is None:
                    self.compute_summary()

                summ = self.summary[i] if isinstance(self.summary, list) else self.summary
                val_txt = r"\ensuremath{" + summ.latex_value(sigfigs=sigfigs) + "}"

            prior_text = self.get_prior_str(index=i)

            lines.append(
                rf"~~~~${symbol}$\dotfill & "
                rf"{self.description}{unit_text}\dotfill & "
                rf"{val_txt}\dotfill & "
                rf"{prior_text} \\" + "\n"
            )

        return "".join(lines)

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