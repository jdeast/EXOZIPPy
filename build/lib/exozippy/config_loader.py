from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from exozippy.params.spec import ParameterSpec, PriorSpec
from exozippy.data.rv import load_rv_data

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    name: str
    expr: str
    kind: str = "hard"


@dataclass(frozen=True, slots=True)
class LoadedModel:
    fit: Dict[str, Any]
    overrides: Dict[str, Any]
    planets: List[str]
    instruments: List[str]
    specs: Dict[str, ParameterSpec]          # base/sampled variables only
    aliases: Dict[str, str]                  # name->canonical (kept trivial here)
    constraints: List[ConstraintSpec]        # from future override file if desired
    ecc_kind_by_planet: Dict[str, str]       # default hk for now


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML required. Install: pip install pyyaml (or exozippy[yaml])")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def _name_planet(pl: str, param: str) -> str:
    return f"planet.{pl}.{param}"


def _name_inst(inst: str, param: str) -> str:
    return f"inst.{inst}.{param}"


def _prior(dist: str, **kwargs) -> PriorSpec:
    return PriorSpec(dist=dist, kwargs=kwargs)


def _make_default_specs(
    planets: List[str],
    instruments: List[str],
    rv_data,
    ecc_default: str = "hk",
) -> Tuple[Dict[str, ParameterSpec], Dict[str, str]]:
    """
    Build all required parameters with data-informed defaults.
    Override file will patch these.
    """
    specs: Dict[str, ParameterSpec] = {}

    # Data summaries for defaults
    all_y = np.concatenate([d.y for d in rv_data.values()]) if rv_data else np.array([0.0])
    all_t = np.concatenate([d.t for d in rv_data.values()]) if rv_data else np.array([0.0])
    all_err = np.concatenate([d.yerr for d in rv_data.values()]) if rv_data else np.array([1.0])

    y_scale = float(np.nanstd(all_y)) if np.isfinite(all_y).all() else 100.0
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = 100.0
    err_med = float(np.nanmedian(all_err)) if np.isfinite(all_err).all() else 1.0
    if not np.isfinite(err_med) or err_med <= 0:
        err_med = 1.0

    t_min, t_max = float(np.nanmin(all_t)), float(np.nanmax(all_t))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        t_min, t_max = 0.0, 1.0
    t_ref = 0.5 * (t_min + t_max)

    # Instruments: gamma & jitter
    for inst in instruments:
        gname = _name_inst(inst, "gamma")
        jname = _name_inst(inst, "jitter")

        specs[gname] = ParameterSpec(
            name=gname,
            prior=_prior("normal", mu=float(np.nanmedian(all_y)), sigma=10.0 * y_scale),
            init=float(np.nanmedian(all_y)),
            fixed=False,
            bounds=(None, None),
        )

        specs[jname] = ParameterSpec(
            name=jname,
            prior=_prior("halfnormal", sigma=5.0 * err_med),
            init=err_med,
            fixed=False,
            bounds=(0.0, None),
        )

    # Planets: P_days, tc, K_mps, plus ecc sampling vars (default hk)
    for pl in planets:
        P = _name_planet(pl, "P_days")
        tc = _name_planet(pl, "tc")
        K = _name_planet(pl, "K_mps")

        # Broad but proper defaults; init may be overridden
        specs[P] = ParameterSpec(
            name=P,
            prior=_prior("lognormal", mu=np.log(10.0), sigma=2.0),  # broad: ~ days to many 1000s days
            init=10.0,
            fixed=False,
            bounds=(1e-6, None),
        )

        # tc: prefer uniform across span if no init; but PyMC Uniform needs bounds.
        specs[tc] = ParameterSpec(
            name=tc,
            prior=_prior("uniform", lower=t_min, upper=t_max),
            init=t_ref,
            fixed=False,
            bounds=(t_min, t_max),
        )

        specs[K] = ParameterSpec(
            name=K,
            prior=_prior("halfnormal", sigma=5.0 * y_scale),
            init=max(1.0, 0.5 * y_scale),
            fixed=False,
            bounds=(0.0, None),
        )

        if ecc_default.lower() == "hk":
            h = _name_planet(pl, "h")
            k = _name_planet(pl, "k")
            specs[h] = ParameterSpec(
                name=h,
                prior=_prior("normal", mu=0.0, sigma=0.3),
                init=0.0,
                fixed=False,
                bounds=(None, None),
            )
            specs[k] = ParameterSpec(
                name=k,
                prior=_prior("normal", mu=0.0, sigma=0.3),
                init=0.0,
                fixed=False,
                bounds=(None, None),
            )
        else:
            raise ValueError(f"Only hk default implemented in defaults builder for now: got {ecc_default}")

    # aliases trivial for now (no linking yet)
    aliases = {name: name for name in specs.keys()}
    return specs, aliases


def _apply_overrides(specs: Dict[str, ParameterSpec], overrides: Mapping[str, Any], planets: List[str]) -> None:
    """
    Override file is structured like:
      planets:
        b:
          P_days: {init: ...}
          tc:     {init: ...}
          K_mps:  {init: ...}

    Only apply keys present. Keep defaults for everything else.
    """
    planets_node = overrides.get("planets", {}) or {}
    if not isinstance(planets_node, dict):
        raise ValueError("params_obsolete override: planets must be a mapping")

    for pl in planets:
        pnode = planets_node.get(pl, {}) or {}
        if not isinstance(pnode, dict):
            continue
        for short, onode in pnode.items():
            if not isinstance(onode, dict):
                continue
            full = _name_planet(pl, str(short))
            if full not in specs:
                raise ValueError(f"Override refers to unknown parameter: {full}")
            old = specs[full]

            init = onode.get("init", old.init)
            fixed = onode.get("fixed", old.fixed)
            bounds = onode.get("bounds", old.bounds)
            prior_node = onode.get("prior", None)

            prior = old.prior
            if prior_node is not None:
                if not isinstance(prior_node, dict) or "dist" not in prior_node:
                    raise ValueError(f"{full}: prior override must be mapping with 'dist'")
                dist = str(prior_node["dist"])
                kw = {k: v for k, v in prior_node.items() if k != "dist"}
                prior = PriorSpec(dist=dist, kwargs=kw)

            specs[full] = ParameterSpec(
                name=old.name,
                prior=prior,
                init=init,
                fixed=bool(fixed),
                bounds=tuple(bounds) if isinstance(bounds, (list, tuple)) else old.bounds,
                shape=old.shape,
            )


def load_model_from_fit_yaml(fit_yaml_path: Path) -> LoadedModel:
    fit = _load_yaml(fit_yaml_path)

    planets = [p["name"] for p in (fit.get("model", {}) or {}).get("planets", [])]
    instruments = [i["name"] for i in ((fit.get("data", {}) or {}).get("rv", {}) or {}).get("instruments", [])]

    # Load RV data early so defaults can be data-informed
    rv_data = load_rv_data(fit, base_dir=fit_yaml_path.parent)

    # Default ecc parameterization for now: hk
    ecc_kind_by_planet = {pl: "hk" for pl in planets}

    # Create full default parameter set
    specs, aliases = _make_default_specs(planets, instruments, rv_data, ecc_default="hk")

    # Apply override file
    params_file = ((fit.get("parameters", {}) or {}).get("file"))
    if not params_file:
        raise ValueError("fit.yaml missing parameters.file")
    overrides_path = (fit_yaml_path.parent / params_file).resolve()
    overrides = _load_yaml(overrides_path)
    _apply_overrides(specs, overrides, planets)

    # constraints/links can be added later; keep empty now
    constraints: List[ConstraintSpec] = []

    return LoadedModel(
        fit=fit,
        overrides=overrides,
        planets=planets,
        instruments=instruments,
        specs=specs,
        aliases=aliases,
        constraints=constraints,
        ecc_kind_by_planet=ecc_kind_by_planet,
    )
