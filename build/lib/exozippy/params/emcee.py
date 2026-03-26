# src/exozippy/params_obsolete/emcee.py
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .pack import ParameterSet
from .spec import PriorSpec


def _log_uniform(x: float, lower: float, upper: float) -> float:
    return 0.0 if (lower <= x <= upper) else -np.inf


def _log_normal(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def _log_lognormal(x: float, mu: float, sigma: float) -> float:
    if x <= 0:
        return -np.inf
    lx = np.log(x)
    return _log_normal(lx, mu, sigma) - np.log(x)


def _log_halfnormal(x: float, sigma: float) -> float:
    if x < 0:
        return -np.inf
    # HalfNormal is Normal(0,sigma) restricted to x>=0, with normalization factor 2
    return _log_normal(x, 0.0, sigma) + np.log(2.0)


def logprior_theta(
    pset: ParameterSet,
    theta: Sequence[float],
    user_params: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> float:
    """
    Log prior for emcee on the free-parameter vector theta.
    Uses the same specs as PyMC.

    Notes:
    - Bounds from spec (tightened by user_params) are treated as hard bounds.
    - For priors that already imply support (e.g. LogNormal), bounds further restrict support.
    """
    theta = np.asarray(theta, dtype=float)
    free = pset.free_specs()
    if theta.size != len(free):
        return -np.inf

    lp = 0.0
    for i, spec in enumerate(free):
        x = float(theta[i])

        lo, hi = spec.tightened_bounds(user=user_params)
        if lo is not None and x < lo:
            return -np.inf
        if hi is not None and x > hi:
            return -np.inf

        prior = spec.prior
        if prior is None:
            return -np.inf

        d = prior.dist.lower()
        kw = prior.kwargs

        if d == "uniform":
            lower = float(kw.get("lower", lo))
            upper = float(kw.get("upper", hi))
            lp += _log_uniform(x, lower, upper)
        elif d == "normal":
            lp += _log_normal(x, float(kw["mu"]), float(kw["sigma"]))
        elif d == "truncnormal":
            # Treat as Normal + hard bounds (already applied above)
            lp += _log_normal(x, float(kw["mu"]), float(kw["sigma"]))
        elif d == "lognormal":
            lp += _log_lognormal(x, float(kw["mu"]), float(kw["sigma"]))
        elif d == "halfnormal":
            lp += _log_halfnormal(x, float(kw["sigma"]))
        else:
            raise ValueError(f"{spec.name}: unsupported prior dist '{prior.dist}'")

        if not np.isfinite(lp):
            return -np.inf

    return float(lp)
