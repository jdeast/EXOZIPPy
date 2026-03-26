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

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import math

try:
    # Keep astropy optional-ish: it's metadata only here.
    from astropy import units as u  # type: ignore
except Exception:  # pragma: no cover
    u = None  # type: ignore

import pymc as pm
import pytensor.tensor as pt


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


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    return float(x)


def _latex_varname(label: str, prefix: str = "ez") -> str:
    """
    Create a LaTeX-safe macro name from a label:
    - remove underscores
    - replace digits with words
    - prefix to avoid global collisions
    """
    old = ["_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    new = ["", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
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
            return f"${med_s}\\pm{ep_s}$"
        return f"${med_s}^{{+{ep_s}}}_{{-{em_s}}}$"


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
    initval: Optional[Number] = None

    # If expression is provided, parameter becomes deterministic (pm.Deterministic).
    # You can pass expression at build time too.
    expression: Any = None

    # "Physical" bounds (can be tightened by user_params, not expanded).
    lower: Optional[Number] = None
    upper: Optional[Number] = None

    # Optional Gaussian prior
    mu: Optional[Number] = None
    sigma: Optional[Number] = None

    # LaTeX/table metadata
    latex: Optional[str] = None
    latex_unit: Optional[str] = None
    description: Optional[str] = None
    latex_prefix: str = "ez"

    # Runtime fields
    value: Any = field(default=None, init=False)  # pm RV or pm.Deterministic after build_pymc()
    latex_varname: str = field(default="", init=False)
    posterior: Any = None  # user stores idata posterior samples here if desired
    summary: Optional[PosteriorSummary] = field(default=None, init=False)
    table_note: Optional[str] = None

    def __post_init__(self) -> None:
        self.latex_varname = _latex_varname(self.label, prefix=self.latex_prefix)

    # ---------
    # PyMC construction
    # ---------

    def build_pymc(
        self,
        ndx: int = 0,
        user_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
        expression: Any = None,
    ) -> Any:
        """
        Create and store `self.value` as a PyMC RV or Deterministic.

        Parameters
        ----------
        ndx : int
            Used to create unique Potential names.
        user_params : dict-like
            Optional overrides; keys are labels (exact) or base label for labels like "foo_0".
            Supported keys per parameter:
              - initval, mu, sigma, lower, upper
            Rules:
              - sigma==0 => fixed parameter at mu (or initval if mu missing)
              - bounds can only be tightened
              - if a sigma prior is provided and you already have sigma, the additional prior
                becomes an extra Potential (so both penalties apply)
        expression : pytensor expression, optional
            Overrides self.expression if provided. If expression is not None, parameter is deterministic.

        Returns
        -------
        PyMC variable (RV or Deterministic)
        """
        expr = self.expression if expression is None else expression

        # Apply user overrides (with conservative rules)
        add_extra_gaussian_potential = False
        user_mu = None
        user_sigma = None

        lower = _safe_float(self.lower)
        upper = _safe_float(self.upper)
        mu = _safe_float(self.mu)
        sigma = _safe_float(self.sigma)
        initval = _safe_float(self.initval)

        lkey = None
        if user_params is not None:
            if self.label in user_params:
                lkey = self.label
            elif "_0" in self.label:
                base = self.label.split("_")[0]
                if base in user_params:
                    lkey = base

        if lkey is not None:
            up = user_params[lkey]

            # initval/mu synchronization: if only one is provided, mirror it
            if "initval" in up:
                initval = _safe_float(up["initval"])
                if "mu" not in up:
                    mu = initval
            if "mu" in up:
                mu = _safe_float(up["mu"])
                if "initval" not in up:
                    initval = mu

            # sigma override logic
            if "sigma" in up:
                user_sigma = _safe_float(up["sigma"])
                user_mu = mu
                if user_sigma == 0:
                    # fixed parameter at mu (fall back to initval if mu missing)
                    if mu is None:
                        if initval is None:
                            raise ValueError(f"{self.label}: sigma==0 but neither mu nor initval provided.")
                        mu = initval
                elif user_sigma is not None and user_sigma > 0:
                    if sigma is None:
                        sigma = user_sigma
                    else:
                        # You already have a Gaussian; user wants an additional penalty.
                        add_extra_gaussian_potential = True

            # bounds: tighten only
            lower, upper = _tighten_bounds(lower, upper, _safe_float(up.get("lower")), _safe_float(up.get("upper")))

        # Deterministic path
        if expr is not None:
            self.value = pm.Deterministic(self.label, expr)

            # Constraints on deterministic parameters are always Potentials.
            if lower is not None:
                pm.Potential(f"{self.label}__lowerbound_{ndx}", pt.switch(self.value >= lower, 0.0, -np.inf))
            if upper is not None:
                pm.Potential(f"{self.label}__upperbound_{ndx}", pt.switch(self.value <= upper, 0.0, -np.inf))
            if mu is not None and sigma is not None and sigma > 0:
                pm.Potential(f"{self.label}__gaussianprior_{ndx}", -0.5 * ((self.value - mu) / sigma) ** 2)

            # Optional second Gaussian penalty from user override
            if add_extra_gaussian_potential and user_mu is not None and user_sigma is not None and user_sigma > 0:
                pm.Potential(
                    f"{self.label}__user_gaussianprior_{ndx}",
                    -0.5 * ((self.value - user_mu) / user_sigma) ** 2,
                )
            return self.value

        # Stochastic/fixed path
        # Interpret (mu provided, sigma missing/0) as "fixed at mu" rather than an invalid Uniform.
        if mu is not None and (sigma is None or sigma == 0):
            self.value = pm.Deterministic(self.label, mu)

            # Enforce bounds via potentials if present (since Deterministic)
            if lower is not None:
                pm.Potential(f"{self.label}__lowerbound_{ndx}", pt.switch(self.value >= lower, 0.0, -np.inf))
            if upper is not None:
                pm.Potential(f"{self.label}__upperbound_{ndx}", pt.switch(self.value <= upper, 0.0, -np.inf))
            return self.value

        # Now sigma is either None or >0, and mu may be None.
        if mu is not None and sigma is not None and sigma > 0:
            if lower is not None or upper is not None:
                # bounded normal via truncation
                self.value = pm.Truncated(
                    self.label,
                    dist=pm.Normal.dist(mu=mu, sigma=sigma),
                    lower=lower,
                    upper=upper,
                    initval=initval,
                )
            else:
                self.value = pm.Normal(self.label, mu=mu, sigma=sigma, initval=initval)

        else:
            # Uniform requires bounds. If none are provided, fail loudly (better than silent nonsense).
            if lower is None or upper is None:
                raise ValueError(
                    f"{self.label}: Uniform prior requires both lower and upper bounds "
                    f"(got lower={lower}, upper={upper})."
                )
            self.value = pm.Uniform(self.label, lower=lower, upper=upper, initval=initval)

        # Optional extra gaussian penalty (second prior) even for stochastic RVs
        if add_extra_gaussian_potential and user_mu is not None and user_sigma is not None and user_sigma > 0:
            pm.Potential(
                f"{self.label}__user_gaussianprior_{ndx}",
                -0.5 * ((self.value - user_mu) / user_sigma) ** 2,
            )

        return self.value

    # ---------
    # Units (metadata convenience)
    # ---------

    def to_unit(self, target_unit: Any) -> Any:
        """
        Convert numeric value between astropy units (metadata convenience).
        NOTE: self.value may be a PyMC/pytensor variable; this is only meaningful
        if self.value is numeric (e.g., fixed/deterministic scalar).
        """
        if u is None:
            raise RuntimeError("astropy is not available; cannot convert units.")
        if self.unit is None:
            raise ValueError(f"{self.label}: no unit set.")
        q = (float(self.value) * self.unit)  # type: ignore[arg-type]
        return q.to(target_unit)

    # ---------
    # LaTeX helpers
    # ---------

    def to_latex_macro(self, sigfigs: int = 2) -> str:
        """
        Returns:
          \\providecommand{\\<varname>}{\\ensuremath{<value^{+..}_{-..}>}}
        """
        if self.summary is None:
            self.compute_summary()
        assert self.summary is not None
        val = self.summary.latex_value(sigfigs=sigfigs)
        return r"\providecommand{\\" + self.latex_varname + r"}{\ensuremath{" + val + r"}}\n"

    def to_table_line(self, use_variable: bool = True, sigfigs: int = 2) -> str:
        """
        Format a line for a LaTeX table:
          $symbol$ ... description (units) ... \\varname \\\\
        """
        if self.latex is None:
            raise ValueError(f"{self.label}: latex symbol not set.")
        if self.description is None:
            raise ValueError(f"{self.label}: description not set.")
        unit_text = "" if self.latex_unit is None else f" ({self.latex_unit})"

        if use_variable:
            val_txt = "\\" + self.latex_varname
        else:
            if self.summary is None:
                self.compute_summary()
            assert self.summary is not None
            val_txt = r"\ensuremath{" + self.summary.latex_value(sigfigs=sigfigs) + "}"

        return f"${self.latex}$ \\dotfill & {self.description}{unit_text} \\dotfill & {val_txt} \\\\\n"

    # ---------
    # Posterior summary
    # ---------

    def compute_summary(self, nsigma: float = 1.0) -> PosteriorSummary:
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
