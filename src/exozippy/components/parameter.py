"""
exozippy.params_obsolete.parameter

Refactor notes (high-level):
- Separate *spec/metadata* (label, units, latex, description, bounds, priors) from the *PyMC variable*.
- Avoid side effects in __init__ (no pm.* calls, no prints).
- Provide a single entrypoint to "materialize" the parameter inside a pm.Model() context: Parameter.build_pymc().
- Make user overrides explicit and predictable: users can only tighten bounds; sigma==0 means fixed/deterministic at mu/initval.
- Clean up posterior summarization using quantiles (median, +/- 1σ by default).

This module is PyMC-facing; if you later want a backend-agnostic parameter spec, split this into:
  - spec.py (pure metadata)
  - pymc.py (adapter that creates pm RVs)
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


def _erf_sigma_to_interval_mass(nsigma: float) -> float:
    """Mass inside +/- nsigma for a normal distribution."""
    return math.erf(nsigma / math.sqrt(2.0))

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

    # If expression is provided, parameter becomes deterministic (pm.Deterministic).
    # You can pass expression at build time too.
    expression: Any = None

    # "Physical" bounds (can be tightened by user_params, not expanded).
    lower: Optional[Number] = None
    upper: Optional[Number] = None

    # Optional Gaussian prior
    mu: Optional[Number] = None
    sigma: Optional[Number] = None

    print_to_table: bool = True

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
    shape: Tuple[int, ...] = ()  # Default to scalar

    def __post_init__(self) -> None:
        """
        Minimalist Identity Setup.
        No math, no scaling, no indexing.
        """
        # 1. Ensure unit is a list for consistency
        if not isinstance(self.unit, (list, np.ndarray)):
            self.unit = [self.unit]

        # 2. Get the 'Pretty Name' for plots/tables
        # (Just use the first unit as the label for now)
        try:
            self.unit_latex = UnitTranslator.get_latex(self.unit[0])
        except:
            self.unit_latex = ""

        # 3. Structural Naming
        self.latex_varname = _latex_varname(self.label, prefix=self.latex_prefix)

    def __post_init__old(self) -> None:

        # 1. Unit Handling & Conversion Factors
        # Default internal units to Solar/Days/CGS
        for unit in self.unit:
            try:
                # Calculate factor to get from User -> Internal
                # Example: u.jupiterMass.to(u.solMass) -> 0.000954
                trial = unit.to(unit.internal_unit)
            except u.UnitConversionError:
                raise TypeError(f"Cannot convert {unit.unit} to {unit.internal_unit} for {self.label}, only units supported by astropy.Unit (https://docs.astropy.org/en/stable/units/standard_units.html) are allowed")

        # 2. Set the 'unit_latex' property strictly
        self.unit_latex = None

        # Check user_params for an explicit override
        if self.user_params and self.label in self.user_params:
            self.unit_latex = self.user_params[self.label].get("unit_latex")

        # If no override, try the translator
        if self.unit_latex is None:
            if self.unit:
                # This will raise the ValueError if not understood
                self.unit_latex = UnitTranslator.get_latex(self.unit)
            else:
                self.unit_latex = ""

        if self.user_params and self.label in self.user_params:
            overrides = self.user_params[self.label]

            # 1. Update values
            if "mu" in overrides: self.mu = overrides["mu"]
            if "sigma" in overrides: self.sigma = overrides["sigma"]

            # 2. Tighten bounds (never expand)
            if "lower" in overrides:
                self.lower = max(self.lower, overrides["lower"]) if self.lower is not None else overrides["lower"]

            if "upper" in overrides:
                self.upper = min(self.upper, overrides["upper"]) if self.upper is not None else overrides["upper"]

        self.latex_varname = _latex_varname(self.label, prefix=self.latex_prefix)

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

    def build_pymc(self, ndx=0, expression=None):
        """
        Materializes the Parameter in the PyMC graph.
        - Handles User -> Internal unit scaling.
        - Supports vectorization (shape > 1).
        - Supports indexed YAML overrides (planet.0.mass).
        - Uses the 'Potential Hammer' for mixed priors in a single vector.
        """
        expr_raw = self.expression if expression is None else expression

        # 1. SETUP SHAPES
        actual_shape = self.shape if isinstance(self.shape, tuple) else (self.shape,)
        n_elements = np.prod(actual_shape).astype(int) if actual_shape != () else 1

        # 2. PREP WORK VECTORS (User Units)
        def prep_vec(val, fill=np.nan):
            if val is None:
                return np.full(n_elements, fill, dtype=float)

            v = np.atleast_1d(val)
            if v.size == 1:
                return np.full(n_elements, v[0], dtype=float)
            if v.size == n_elements:
                return v.astype(float).copy()
            return np.full(n_elements, fill, dtype=float)

        inits = prep_vec(self.initval)
        mus = prep_vec(self.mu)
        sigmas = prep_vec(self.sigma)
        lowers = prep_vec(self.lower, fill=-np.inf)
        uppers = prep_vec(self.upper, fill=np.inf)
        scales = prep_vec(self.init_scale, fill=1.0)

        # Track extra Gaussian priors (gaussian_width in YAML)
        add_extra_gaussian = np.zeros(n_elements, dtype=bool)
        user_mus = np.zeros(n_elements)
        user_sigmas = np.zeros(n_elements)

        # --- 3. THE SIEVE: Initialize Units and Apply YAML Overrides ---

        # Initialize units list FIRST so it can be indexed in the sieve
        # Ensure we store the Unit part, not a Quantity
        base_u = self.unit.unit if hasattr(self.unit, 'unit') else self.unit
        units = [base_u] * n_elements

        if self.user_params is not None:
            # A. Global Overrides (e.g., "planet.mass")
            if self.label in self.user_params:
                up = self.user_params[self.label]
                if "unit" in up:
                    u_ov = up["unit"]

                    new_u = u_ov.unit if hasattr(u_ov, 'unit') else u_ov
                    units = [new_u] * n_elements
                if "initval" in up: inits[:] = float(up["initval"])
                if "mu" in up: mus[:] = float(up["mu"])
                if "sigma" in up: sigmas[:] = float(up["sigma"])
                if "lower" in up: lowers[:] = float(up["lower"])
                if "upper" in up: uppers[:] = float(up["upper"])
                if "init_scale" in up: scales[:] = float(up["init_scale"])
                if "gaussian_width" in up:
                    add_extra_gaussian[:] = True
                    user_sigmas[:] = float(up["gaussian_width"])
                    user_mus[:] = mus.copy()

            # B. Indexed Overrides (e.g., "planet.0.mass")
            label_parts = self.label.split('.')
            if len(label_parts) >= 2:
                prefix, attr = label_parts[0], label_parts[-1]
                for i in range(n_elements):
                    idx_key = f"{prefix}.{i}.{attr}"
                    if idx_key in self.user_params:
                        ov = self.user_params[idx_key]

                        # Update Unit for this specific index
                        if "unit" in ov:
                            new_unit = ov["unit"]

                            # If it's a string from YAML, cast it to an Astropy Unit
                            if isinstance(new_unit, str):
                                try:
                                    units[i] = u.Unit(new_unit)
                                except ValueError:
                                    raise ValueError(
                                        f"Unit {new_unit} for {indexed_key} not understood. Must be supported by astropy.units.")
                            else:
                                # Handle cases where it's already an Astropy object
                                units[i] = new_unit.unit if hasattr(new_unit, 'unit') else new_unit

                        if "initval" in ov: inits[i] = float(ov["initval"])
                        if "mu" in ov: mus[i] = float(ov["mu"])
                        if "sigma" in ov: sigmas[i] = float(ov["sigma"])
                        if "lower" in ov: lowers[i] = float(ov["lower"])
                        if "upper" in ov: uppers[i] = float(ov["upper"])
                        if "init_scale" in ov: scales[i] = float(ov["init_scale"])

            # --- 4. VECTORIZED UNIT CONVERSION ---
            factors = np.zeros(n_elements)

            # Ensure units is definitely a list we can index
            if not isinstance(units, list):
                units = [units] * n_elements

            for i in range(n_elements):
                # Grab the specific unit object for this index
                u_item = units[i]

                # If u_item is somehow a list (the cause of your error),
                # we grab the first element to save the run.
                if isinstance(u_item, list):
                    u_item = u_item[0]

                try:
                    # We use float() because u.to() returns a Quantity or a float
                    # This is faster and avoids the .value AttributeError
                    factors[i] = float(u_item.to(self.internal_unit))
                except (AttributeError, TypeError):
                    # Fallback for complex astropy objects
                    factors[i] = u_item.to(self.internal_unit).value

        # Apply scaling to all vectors
        inits *= factors
        mus *= factors
        sigmas *= factors
        lowers *= factors
        uppers *= factors
        scales *= factors
        user_mus *= factors
        user_sigmas  *= factors

        # --- 5. INITIALIZATION & CLIPPING LOGIC ---
        for i in range(n_elements):
            if scales[i] < 0:
                scales[i] = abs(inits[i] * scales[i])
            if np.isnan(inits[i]) and not np.isnan(mus[i]):
                inits[i] = mus[i]
            if not np.isinf(lowers[i]) and not np.isinf(uppers[i]):
                eps = scales[i] * 1e-6
                inits[i] = np.clip(inits[i], lowers[i] + eps, uppers[i] - eps)

        # Sync class attributes for external use (tables/plots)
        self.initval = inits[0] if n_elements == 1 else inits
        self.init_scale = scales[0] if n_elements == 1 else scales

        # --- NEW: EXACT SHAPE ENFORCEMENT ---
        # This safely collapses the 1D arrays back to 0D floats if actual_shape is (),
        # or properly shapes them to (N,) if actual_shape is a vector.
        def enforce_shape(arr):
            if actual_shape == ():
                return float(arr[0])
            return arr.reshape(actual_shape)

        self.initval = enforce_shape(inits)
        self.init_scale = enforce_shape(scales)
        self.mu = enforce_shape(mus)
        self.sigma = enforce_shape(sigmas)
        self.lower = enforce_shape(lowers)
        self.upper = enforce_shape(uppers)
        user_mus = enforce_shape(user_mus)
        user_sigmas = enforce_shape(user_sigmas)

        # --- 6. THE LEAN GATEKEEPER ---
        has_constraining_logic = any([
            not np.all(np.isnan(self.mu)),
            not np.all(np.isnan(self.sigma)),
            not np.all(np.isinf(self.lower)),
            not np.all(np.isinf(self.upper)),
            self.force_node,
            np.any(add_extra_gaussian)
        ])

        if expr_raw is not None and not has_constraining_logic:
            self.value = expr_raw() if callable(expr_raw) else expr_raw
            return None

        # --- 7. DETERMINISTIC PATH ---
        if expr_raw is not None:

            actual_expr = expr_raw() if callable(expr_raw) else expr_raw
            self.value = pm.Deterministic(self.label, actual_expr)

            char_scale = self.init_scale
            if char_scale is None:
                # If bound is 100, scale is 1.0. If bound is 0, scale is 0.1
                char_scale = np.maximum(np.abs(u_val) * 0.01 if u_val is not None else 1.0, 0.1)
            steepness = 10.0 / char_scale

            if not np.all(np.isinf(lowers)):
                # this is a differentiable "wall" boundary at the lower bound
                pm.Potential(f"{self.label}_lower_bound",-pt.softplus(steepness * (self.lower - self.value)))
                #switches are really hard to sample and poorly behaved (cause divergences)
                # pm.Potential(f"{self.label}_lower_bound", pt.switch(pt.all(self.value >= lowers), 0.0, -np.inf))
            if not np.all(np.isinf(uppers)):
                #pm.Potential(f"{self.label}_upper_bound", pt.switch(pt.all(self.value <= uppers), 0.0, -np.inf))
                pm.Potential(f"{self.label}_upper_bound",-pt.softplus(steepness * (self.value-self.upper)))
            return self.value

        # --- 8. STOCHASTIC / FIXED PATH ---
        if not np.any(np.isnan(mus)) and np.all(sigmas == 0):
            self.value = pm.Deterministic(self.label, pt.as_tensor_variable(self.mu))
        else:
            if np.any(np.isinf(self.lower)) or np.any(np.isinf(self.upper)):
                raise ValueError(f"{self.label}: Vectorized parameters require finite bounds.")

            self.value = pm.Uniform(
                self.label, lower=self.lower, upper=self.upper, initval=self.initval, shape=actual_shape
            )

        # --- 9. THE POTENTIAL HAMMER ---
        for i in range(n_elements):
            if not np.isnan(self.mu[i]) and not np.isnan(self.sigma[i]) and self.sigma[i] > 0:
                logp = -0.5 * ((self.value[i] - self.mu[i]) / self.sigma[i]) ** 2
                pm.Potential(f"prior.{self.label}.{i}", logp)

            if add_extra_gaussian[i]:
                extra_logp = -0.5 * ((self.value[i] - user_mus[i]) / user_sigmas[i]) ** 2
                pm.Potential(f"user_prior.{self.label}.{i}", extra_logp)

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

    def generate_posterior_scalar(self, posterior_bundle):
        """
        Calculates the posterior for this parameter.

        1. If it was a named node (Deterministic/RV), extract from posterior_bundle.
        2. If it was a 'Floating' parameter (value is None), build a mini-graph
           from the expression/recipe and calculate it now via NumPy.
        """
        # Case 1: The sampler already calculated it (it was a named node)
        if self.label in posterior_bundle:
            return posterior_bundle[self.label]

        # Case 2: We have no way to calculate this (no expression and not in idata)
        if self.expression is None:
            return None

        # Case 3: Late-Binding calculation for 'Floating' parameters
        # If expression is a lambda/recipe, call it now to build the symbolic nodes
        expr = self.expression() if callable(self.expression) else self.expression

        # A. Find the 'Roots' (the sampled variables) this math depends on.
        # This scans the symbolic graph to find every parent node.
        all_nodes = pytensor.graph.basic.ancestors([expr])

        # B. Filter for parent nodes that actually exist in our MCMC results.
        # We look for nodes that have a 'name' attribute matching a key in the bundle.
        inputs_in_posterior = [
            n for n in all_nodes
            if hasattr(n, 'name') and n.name in posterior_bundle
        ]

        # C. Compile a high-speed NumPy function from the Symbolic Expression.
        # This turns the PyTensor math into a vectorized execution block.
        calc_func = pytensor.function(
            inputs_in_posterior,
            expr,
            on_unused_input='ignore'
        )

        # D. Feed the MCMC chains into the function.
        # We need to flatten the (chain, draw) dimensions into a single vector
        # so PyTensor can process it as a 'batch'.
        input_data = []
        for n in inputs_in_posterior:
            data = posterior_bundle[n.name]
            # If it's xarray/dataset, flatten to 1D
            if hasattr(data, "values"):
                input_data.append(data.values.flatten())
            else:
                input_data.append(np.asarray(data).flatten())

        # E. Return the result reshaped to match the original posterior structure
        # or just keep it flat for the compute_summary logic.
        return calc_func(*input_data)

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
        def _idx_to_words(n):
            words = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
                     '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                     '8': 'eight', '9': 'nine'}
            return "".join(words[char] for char in str(n))

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

    def to_latex_def_scalar(self, sigfigs: int = 2) -> str:
        """
        Returns:
          \\providecommand{\\<varname>}{\\ensuremath{<value^{+..}_{-..}>}}
        """

        # fixed parameters:
        if self.posterior is None:
            if self.initval is not None:
                # Use \equiv for fixed values
                val = f"\\equiv {self.initval}"
                return rf"\providecommand{{\{self.latex_varname}}}{{\ensuremath{{{val}}}}}" + "\n"
            return ""  # Skip if it has no value and no posterior

        if self.summary is None:
            self.compute_summary()

        assert self.summary is not None
        val = self.summary.latex_value(sigfigs=sigfigs)
        return rf"\providecommand{{\{self.latex_varname}}}{{\ensuremath{{{val}}}}}" + "\n"

    def get_prior_str(self) -> str:
        r"""Returns a LaTeX string representing the prior (e.g., $\mathcal{N}(0,1)$)."""

        # 1. Fixed Parameter
        if self.posterior is None and self.initval is not None:
            return r"Fixed"

        # Helper to extract scalar from 1D arrays AND convert to user units
        def _scalar(val):
            if val is None: return None
            # Extract the raw float
            raw_val = float(np.atleast_1d(val)[0])
            # Convert back to user units for the table
            return float(self.from_internal(raw_val))

        # Helper for clean numerical formatting
        def _fmt(val):
            if val is None:
                return ""
            if np.isinf(val):
                return r"\infty" if val > 0 else r"-\infty"

            # Use standard formatting for zero and numbers between 0.001 and 10000
            if val == 0 or (0.001 <= abs(val) < 10000):
                # Format to 4 decimal places, then safely strip trailing zeros and the decimal point
                return f"{val:.4f}".rstrip("0").rstrip(".")

            # Fallback to scientific notation for very large or very small numbers
            return f"{val:.3g}"

        # 2. Gaussian/Normal Prior
        mu, sig = _scalar(self.mu), _scalar(self.sigma)
        if mu is not None and sig is not None and sig > 0:
            return rf"$\mathcal{{N}}({_fmt(mu)}, {_fmt(sig)})$"

        # 3. Uniform Prior
        lo, hi = _scalar(self.lower), _scalar(self.upper)
        if lo is not None and hi is not None:
            # Ignore bounds if they are effectively (-inf, inf)
            if np.isinf(lo) and np.isinf(hi):
                return ""
            return rf"$\mathcal{{U}}({_fmt(lo)}, {_fmt(hi)})$"

        return ""

    def get_prior_str_scalar(self) -> str:
        r"""Returns a LaTeX string representing the prior (e.g., $\mathcal{N}(0,1)$)."""

        # 1. Fixed Parameter
        if self.posterior is None and self.initval is not None:
            return r"Fixed"

        # 2. Gaussian/Normal Prior
        if self.mu is not None and self.sigma is not None and self.sigma > 0:
            return rf"$\mathcal{{N}}({self.mu}, {self.sigma})$"

        # 3. Uniform Prior
        if self.lower is not None and self.upper is not None:
            return rf"$\mathcal{{U}}({self.lower}, {self.upper})$"

        return ""

    def to_table_line(self, sigfigs: int = 2) -> str:
        if self.latex is None:
            raise ValueError(f"{self.label}: latex symbol not set.")
        if self.description is None:
            raise ValueError(f"{self.label}: description not set.")

        # Strip any existing $ symbols and wrap in \ensuremath for safety
        safe_unit = self.unit_latex.replace('$', '') if self.unit_latex else ""
        unit_text = "" if not safe_unit else rf" (\ensuremath{{{safe_unit}}})"

        if self.print_to_table:
            val_txt = "\\" + self.latex_varname
        else:
            if self.summary is None:
                self.compute_summary()
            assert self.summary is not None
            val_txt = r"\ensuremath{" + self.summary.latex_value(sigfigs=sigfigs) + "}"

        prior_text = self.get_prior_str()

        return (
                rf"~~~~${self.latex}$\dotfill & "
                rf"{self.description}{unit_text}\dotfill & "
                rf"{val_txt}\dotfill & "
                rf"{prior_text} \\" + "\n"
        )

    def to_table_line_scalar(self, sigfigs: int = 2) -> str:
        """
        Format a line for a LaTeX table:
          $symbol$ ... description (units) ... \\varname \\\\
        """
        if self.latex is None:
            raise ValueError(f"{self.label}: latex symbol not set.")
        if self.description is None:
            raise ValueError(f"{self.label}: description not set.")

        unit_text = "" if not self.unit_latex else f" ({self.unit_latex})"

        if self.print_to_table:
            val_txt = "\\" + self.latex_varname
        else:
            if self.summary is None:
                self.compute_summary()
            assert self.summary is not None
            val_txt = r"\ensuremath{" + self.summary.latex_value(sigfigs=sigfigs) + "}"

        prior_text = self.get_prior_str()

        #return f"~~~~${self.latex}$ \\dotfill & {self.description}{unit_text} \\dotfill & {val_txt} \\\\\n"

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

        mass = _erf_sigma_to_interval_mass(nsigma)
        lo_q, hi_q = 0.5 - mass / 2.0, 0.5 + mass / 2.0

        def get_stat(data):
            med = float(np.quantile(data, 0.5))
            lo = float(np.quantile(data, lo_q))
            hi = float(np.quantile(data, hi_q))
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

    def compute_summary_scalar(self, nsigma: float = 1.0) -> PosteriorSummary:
        """
        Compute median and +/- interval corresponding to a normal-equivalent nsigma mass.
        For nsigma=1: uses the central 68.27% interval (16th/84th percentiles).
        """
        arr = _as_flat_array(self.posterior)

        mass = _erf_sigma_to_interval_mass(nsigma)
        lo_q = 0.5 - mass / 2.0
        hi_q = 0.5 + mass / 2.0

        med = float(np.quantile(arr, 0.5))
        lo = float(np.quantile(arr, lo_q))
        hi = float(np.quantile(arr, hi_q))

        self.summary = PosteriorSummary(median=med, err_minus=med - lo, err_plus=hi - med)
        return self.summary
