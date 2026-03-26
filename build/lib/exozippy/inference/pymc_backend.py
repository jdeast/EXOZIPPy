from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from exozippy.constraints.safe_eval import compile_expr
from exozippy.params.pymc import build_pymc_rvs_from_specs
from exozippy.params.transforms.eccentricity import make_ecc_transform
from exozippy.rv.keplerian import rv_keplerian
from exozippy.data.rv import RVInstrumentData


def _apply_hard_constraint(name: str, cond) -> None:
    pm.Potential(name, pt.switch(cond, 0.0, -np.inf))


def build_rv_only_model(
    loaded,  # LoadedModel
    rv_data: Mapping[str, RVInstrumentData],
    user_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Builds:
    - sampled RVs (including linked aliases)
    - derived e,omega per planet from chosen parameterization
    - hard constraints (ecc transforms + params_obsolete.yaml constraints)
    - RV likelihood per instrument
    """
    # 1) sampled/base RVs
    values = build_pymc_rvs_from_specs(loaded.specs, loaded.aliases, user_params=user_params)

    # 2) derive internal e,omega per planet
    # we inject derived values into the same mapping so constraints + RV model can refer to them by name.
    for pl in loaded.planets:
        tr = make_ecc_transform(loaded.ecc_kind_by_planet[pl])
        derived = tr.to_internal(values, planet=pl)
        # register deterministic nodes with consistent names
        e_name = f"planet.{pl}.e"
        w_name = f"planet.{pl}.omega"
        values[e_name] = pm.Deterministic(e_name, derived[e_name])
        values[w_name] = pm.Deterministic(w_name, derived[w_name])

        # transform-specific constraints
        for cname, cond in tr.constraints(values, planet=pl):
            _apply_hard_constraint(f"constraint_{cname}", cond)

    # 3) compile and apply user/global constraints from params_obsolete.yaml
    # constraints can reference any known key in `values`
    for c in loaded.constraints:
        compiled = compile_expr(c.expr, values)
        cond = compiled(values)
        _apply_hard_constraint(f"constraint_{c.name}", cond)

    # 4) RV likelihood
    # model: sum planets + gamma_inst, add jitter in quadrature
    # planet required keys from templates: P_days, tp, K_mps
    # internal e/omega always present as deterministics
    for inst, d in rv_data.items():
        mu = values[f"inst.{inst}.gamma"] + pt.zeros_like(pt.as_tensor_variable(d.t))
        for pl in loaded.planets:
            mu = mu + rv_keplerian(
                t=d.t,
                period=values[f"planet.{pl}.P_days"],
                t_peri=values[f"planet.{pl}.tp"],
                ecc=values[f"planet.{pl}.e"],
                omega=values[f"planet.{pl}.omega"],
                K=values[f"planet.{pl}.K_mps"],
                gamma=0.0,
            )

        # jitter
        jitter = values[f"inst.{inst}.jitter"]
        sig = np.asarray(d.yerr, dtype=float)
        sigma_eff = pm.math.sqrt(sig * sig + jitter * jitter)

        pm.Normal(f"rv_obs_{inst}", mu=mu, sigma=sigma_eff, observed=d.y)

        # optional: store deterministic model
        pm.Deterministic(f"rv_mu_{inst}", mu)

    return values
