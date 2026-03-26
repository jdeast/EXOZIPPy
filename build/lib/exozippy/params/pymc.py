from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pymc as pm

from .spec import ParameterSpec, PriorSpec
from .pack import ParameterSet


def _make_rv(name: str, prior: PriorSpec, init: Optional[float], lo: Optional[float], hi: Optional[float]) -> Any:
    d = prior.dist.lower()
    kw = dict(prior.kwargs)
    if init is not None:
        kw.setdefault("initval", init)

    if d == "normal":
        mu = kw["mu"]
        sigma = kw["sigma"]
        if lo is not None or hi is not None:
            return pm.Truncated(name, dist=pm.Normal.dist(mu=mu, sigma=sigma), lower=lo, upper=hi, initval=kw.get("initval"))
        return pm.Normal(name, mu=mu, sigma=sigma, initval=kw.get("initval"))

    if d == "uniform":
        lower = kw.get("lower", lo)
        upper = kw.get("upper", hi)
        if lower is None or upper is None:
            raise ValueError(f"{name}: Uniform requires lower and upper")
        return pm.Uniform(name, lower=lower, upper=upper, initval=kw.get("initval"))

    if d == "lognormal":
        return pm.LogNormal(name, mu=kw["mu"], sigma=kw["sigma"], initval=kw.get("initval"))

    if d == "halfnormal":
        return pm.HalfNormal(name, sigma=kw["sigma"], initval=kw.get("initval"))

    if d == "truncnormal":
        mu = kw["mu"]
        sigma = kw["sigma"]
        lower = kw.get("lower", lo)
        upper = kw.get("upper", hi)
        return pm.Truncated(name, dist=pm.Normal.dist(mu=mu, sigma=sigma), lower=lower, upper=upper, initval=kw.get("initval"))

    raise ValueError(f"{name}: unsupported prior '{prior.dist}'")


def build_pymc_rvs_from_specs(
    specs: Mapping[str, ParameterSpec],
    aliases: Mapping[str, str],
    user_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build one RV per canonical (aliased) name, then map all names to canonical RV.

    Returns:
      values dict with keys for every original spec name mapping to a PyMC RV/tensor.
    """
    built: Dict[str, Any] = {}
    values: Dict[str, Any] = {}

    # build canonical RVs
    for name, spec in specs.items():
        canon = aliases.get(name, name)
        if canon in built:
            continue
        if spec.init is None:
            raise ValueError(f"{name}: init required")
        init = float(spec.init)
        lo, hi = spec.tightened_bounds(user=user_params)
        if spec.fixed:
            built[canon] = pm.Deterministic(canon, init)
        else:
            if spec.prior is None:
                raise ValueError(f"{name}: free parameter requires prior")
            built[canon] = _make_rv(canon, spec.prior, init, lo, hi)

    # map all names -> canonical rv
    for name in specs.keys():
        canon = aliases.get(name, name)
        values[name] = built[canon]

    # also include canonical keys themselves for convenience
    for canon, rv in built.items():
        values[canon] = rv

    return values
