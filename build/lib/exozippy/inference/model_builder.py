# src/exozippy/inference_old/model_builder.py
"""
A practical pattern for building complex, nested, dynamically-chosen PyMC models
*within a single* pm.Model() context, without writing one giant block.

Key idea:
- Only the top-level caller opens `with pm.Model() as model:`
- Everything else is pure Python that *assumes* a model is active.
- Sub-builders return references (dicts of tensors / RVs), not models.

This file is intentionally "boring": it wires components together.
Your physics lives elsewhere (rv/transit/dynamics/...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

import numpy as np
import pymc as pm

from exozippy.params.pack import ParameterSet
from exozippy.params.pymc import build_pymc_rvs


# ----------------------------
# Protocols for plugins
# ----------------------------

class Component(Protocol):
    """A model component that can attach RVs/Deterministics/Potentials in the active pm.Model."""
    name: str

    def build(self, ctx: "BuildContext") -> Dict[str, Any]:
        """
        Build this component into the active pm.Model context.

        Returns a dict of produced objects (RVs, deterministics, tensors),
        namespaced by the component.
        """
        ...


# ----------------------------
# Shared build context
# ----------------------------

@dataclass(slots=True)
class BuildContext:
    """
    Passed to every component so they can:
    - access config & data
    - register/get already-built components
    - share parameter registry
    """
    config: Mapping[str, Any]
    data: Mapping[str, Any]

    # Global parameter registry (namespaced free/fixed etc. live here)
    # You can store ParameterSet factories or built RV dictionaries.
    params: Dict[str, Any] = field(default_factory=dict)

    # Cache for built components
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def put_artifact(self, name: str, artifact: Dict[str, Any]) -> Dict[str, Any]:
        self.artifacts[name] = artifact
        return artifact

    def require(self, name: str) -> Dict[str, Any]:
        if name not in self.artifacts:
            raise KeyError(f"Component '{name}' not built yet")
        return self.artifacts[name]


# ----------------------------
# Concrete components (examples)
# ----------------------------

@dataclass(slots=True)
class PlanetKeplerianComponent:
    """
    Builds planet orbital parameters (priors) and exposes a parameter dict.
    """
    name: str
    pset: ParameterSet
    user_params: Optional[Mapping[str, Mapping[str, Any]]] = None

    def build(self, ctx: BuildContext) -> Dict[str, Any]:
        # Build RVs in the current model context.
        rvs = build_pymc_rvs(self.pset, user_params=self.user_params)

        # Optionally stash for other components
        ctx.params[self.name] = rvs

        # Return namespaced artifact
        return {"rvs": rvs}


@dataclass(slots=True)
class InstrumentOffsetsComponent:
    """
    Example: per-instrument gamma offsets + jitters.

    Expects ctx.data["instruments"] = list[str]
    Produces:
      - gamma[inst], jitter[inst]
    """
    name: str = "instruments"

    def build(self, ctx: BuildContext) -> Dict[str, Any]:
        instruments = list(ctx.data["instruments"])

        # Priors configurable
        gamma_sigma = float(ctx.get("gamma_sigma_mps", 100.0))
        jitter_sigma = float(ctx.get("jitter_sigma_mps", 10.0))

        gamma = {}
        jitter = {}
        for inst in instruments:
            gamma[inst] = pm.Normal(f"gamma_{inst}", mu=0.0, sigma=gamma_sigma)
            # jitter >= 0
            jitter[inst] = pm.HalfNormal(f"jitter_{inst}", sigma=jitter_sigma)

        return {"gamma": gamma, "jitter": jitter}


@dataclass(slots=True)
class RVLikelihoodComponent:
    """
    Adds the RV likelihood terms.

    Expects:
      - planet RV parameters from ctx.require(planet_component_name)["rvs"]
      - instrument offsets/jitters from ctx.require("instruments")
      - data arrays:
          ctx.data["rv"]: dict[inst -> (t, y, yerr)]
    """
    name: str
    planet_component_name: str

    def build(self, ctx: BuildContext) -> Dict[str, Any]:
        from exozippy.rv.keplerian import rv_keplerian  # your pt-based model

        planet = ctx.require(self.planet_component_name)["rvs"]
        inst_art = ctx.require("instruments")
        gamma = inst_art["gamma"]
        jitter = inst_art["jitter"]

        rv_data = ctx.data["rv"]  # dict inst -> (t, y, yerr)

        # You can add deterministic model traces if you want.
        model_mu = {}

        for inst, (t, y, yerr) in rv_data.items():
            mu = rv_keplerian(
                t=t,
                period=planet["b.period_days"],
                t_peri=planet["b.t_peri"],
                ecc=planet["b.ecc"],
                omega=planet["b.omega"],
                K=planet["b.K_mps"],
                gamma=gamma[inst],
            )
            model_mu[inst] = pm.Deterministic(f"rv_mu_{inst}", mu)

            # combine measurement error and jitter in quadrature
            sigma = np.asarray(yerr, dtype=float)
            sigma_eff = pm.math.sqrt(sigma * sigma + jitter[inst] * jitter[inst])

            pm.Normal(f"rv_obs_{inst}", mu=mu, sigma=sigma_eff, observed=y)

        return {"mu": model_mu}


# ----------------------------
# Orchestrator: builds everything in order
# ----------------------------

@dataclass(slots=True)
class ModelBuilder:
    """
    Orchestrates a complex model build without nested pm.Model contexts.

    Usage:
      builder = ModelBuilder(config, data)
      with pm.Model() as model:
          ctx = builder.build()
          idata = pm.sample(...)
    """
    config: Mapping[str, Any]
    data: Mapping[str, Any]

    # registry of named components
    components: Dict[str, Component] = field(default_factory=dict)

    def add(self, component: Component) -> "ModelBuilder":
        if component.name in self.components:
            raise KeyError(f"Component name already exists: {component.name}")
        self.components[component.name] = component
        return self

    def build(self) -> BuildContext:
        """
        Build all components in a dependency-safe order.
        For now: explicit order from config["build_order"].
        (You can replace with a DAG/toposort later.)
        """
        ctx = BuildContext(config=self.config, data=self.data)

        order = list(self.config.get("build_order", []))
        if not order:
            # Sensible default if user doesn't specify.
            order = list(self.components.keys())

        for name in order:
            if name not in self.components:
                raise KeyError(f"build_order references unknown component '{name}'")
            comp = self.components[name]
            artifact = comp.build(ctx)
            ctx.put_artifact(name, artifact)

        return ctx


# ----------------------------
# Example of use (put in your script, not library)
# ----------------------------

def build_rv_model_example(config: Mapping[str, Any], data: Mapping[str, Any], planet_pset: ParameterSet):
    builder = ModelBuilder(config=config, data=data)

    builder.add(PlanetKeplerianComponent(name="planet_b", pset=planet_pset, user_params=config.get("user_params")))
    builder.add(InstrumentOffsetsComponent())  # name="instruments"
    builder.add(RVLikelihoodComponent(name="rv_like", planet_component_name="planet_b"))
# src/exozippy/params_obsolete/transforms/eccentricity.py
"""
Eccentricity parameterization transforms.

Goal:
- Keep the *physics* layer (RV/orbit model) consuming internal parameters:
    e in [0,1), omega in (-pi, pi]
- Allow the user (via config) to choose sampling parameterization:
    - direct: (e, omega)
    - hk: (h, k) where h=sqrt(e)cos(omega), k=sqrt(e)sin(omega)
    - ecosesin: (ec, es) where ec=e*cos(omega), es=e*sin(omega)
    - (extend with custom parameterizations later)

This file provides:
- Transform classes with a common interface:
    sampled_names(prefix) -> tuple[str,...]
    internal_names(prefix) -> ("e", "omega") with planet prefix
    to_internal(values, prefix) -> dict with keys e_name, omega_name
    constraints(values, prefix) -> list of constraint expressions

The "values" mapping can hold either:
- PyTensor/PyMC variables (inside a pm.Model context), OR
- floats / numpy arrays (emcee land).

We provide small helper functions to "evaluate constraints" for:
- PyMC: build hard-constraint Potentials
- emcee: quickly return -inf if violated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Protocol

import numpy as np


# ----------------------------
# Backend-agnostic math helpers
# ----------------------------

def _is_tensor(x: Any) -> bool:
    # Heuristic: pytensor TensorVariable has `.type` and `.owner` usually.
    return hasattr(x, "type") and (hasattr(x, "owner") or hasattr(x, "tag"))


def _atan2(y: Any, x: Any) -> Any:
    if _is_tensor(x) or _is_tensor(y):
        import pytensor.tensor as pt
        return pt.arctan2(y, x)
    return np.arctan2(y, x)


def _sqrt(x: Any) -> Any:
    if _is_tensor(x):
        import pytensor.tensor as pt
        return pt.sqrt(x)
    return np.sqrt(x)


def _cos(x: Any) -> Any:
    if _is_tensor(x):
        import pytensor.tensor as pt
        return pt.cos(x)
    return np.cos(x)


def _sin(x: Any) -> Any:
    if _is_tensor(x):
        import pytensor.tensor as pt
        return pt.sin(x)
    return np.sin(x)


def _clip01_open(x: Any, eps: float = 1e-12) -> Any:
    """
    Clip into [0, 1-eps] for numerical stability when transforming.
    For PyMC, clipping changes the model; so we *do not* clip tensors by default.
    Use constraints instead.
    """
    if _is_tensor(x):
        return x
    return np.minimum(np.maximum(x, 0.0), 1.0 - eps)


def _wrap_pi(x: Any) -> Any:
    """
    Wrap angle to (-pi, pi].
    For tensors, use pt.mod; for floats use numpy.
    """
    twopi = 2.0 * np.pi
    if _is_tensor(x):
        import pytensor.tensor as pt
        return pt.mod(x + np.pi, twopi) - np.pi
    return (x + np.pi) % twopi - np.pi


# ----------------------------
# Interface
# ----------------------------

class EccTransform(Protocol):
    """
    A transform that defines which variables are sampled and how (e, omega) are derived.
    """

    kind: str

    def sampled_names(self, planet: str) -> Tuple[str, ...]:
        ...

    def internal_names(self, planet: str) -> Tuple[str, str]:
        ...

    def to_internal(self, values: Mapping[str, Any], planet: str) -> Dict[str, Any]:
        ...

    def constraints(self, values: Mapping[str, Any], planet: str) -> List[Tuple[str, Any]]:
        """
        Returns a list of (name, expr) constraints.
        For hard constraints: expr should be a boolean-like condition (>=0 or True/False).
        We'll interpret these with backend helpers.
        """
        ...


def _pname(planet: str, short: str) -> str:
    # Naming convention; you can change to "pl.{planet}.{short}" if you prefer.
    return f"{planet}.{short}"


# ----------------------------
# Transforms
# ----------------------------

@dataclass(frozen=True, slots=True)
class DirectEccTransform:
    """
    Sample (e, omega) directly.
    """
    kind: str = "direct"

    def sampled_names(self, planet: str) -> Tuple[str, ...]:
        return (_pname(planet, "e"), _pname(planet, "omega"))

    def internal_names(self, planet: str) -> Tuple[str, str]:
        return (_pname(planet, "e"), _pname(planet, "omega"))

    def to_internal(self, values: Mapping[str, Any], planet: str) -> Dict[str, Any]:
        e_name, w_name = self.internal_names(planet)
        e = values[e_name]
        w = values[w_name]
        # Optionally wrap omega numerically; in tensor land, don't alter.
        w = _wrap_pi(w)
        return {e_name: e, w_name: w}

    def constraints(self, values: Mapping[str, Any], planet: str) -> List[Tuple[str, Any]]:
        e_name, w_name = self.internal_names(planet)
        e = values[e_name]
        w = values[w_name]
        return [
            (f"{e_name}_ge0", e >= 0.0),
            (f"{e_name}_lt1", e < 1.0),
            # omega bounds are optional because it's periodic; enforce only if you want:
            (f"{w_name}_gtmPi", w > -np.pi),
            (f"{w_name}_lePi", w <= np.pi),
        ]


@dataclass(frozen=True, slots=True)
class HKTransform:
    """
    Sample (h, k) where:
      h = sqrt(e) cos(omega)
      k = sqrt(e) sin(omega)

    Internal outputs:
      e = h^2 + k^2
      omega = atan2(k, h)
    """
    kind: str = "hk"

    def sampled_names(self, planet: str) -> Tuple[str, ...]:
        return (_pname(planet, "h"), _pname(planet, "k"))

    def internal_names(self, planet: str) -> Tuple[str, str]:
        return (_pname(planet, "e"), _pname(planet, "omega"))

    def to_internal(self, values: Mapping[str, Any], planet: str) -> Dict[str, Any]:
        h = values[_pname(planet, "h")]
        k = values[_pname(planet, "k")]
        e_name, w_name = self.internal_names(planet)

        e = h * h + k * k
        w = _atan2(k, h)
        w = _wrap_pi(w)
        return {e_name: e, w_name: w}

    def constraints(self, values: Mapping[str, Any], planet: str) -> List[Tuple[str, Any]]:
        h = values[_pname(planet, "h")]
        k = values[_pname(planet, "k")]
        # hard constraint: h^2+k^2 < 1
        e = h * h + k * k
        return [
            (f"{planet}.hk_disk", e < 1.0),
        ]


@dataclass(frozen=True, slots=True)
class EccSinCosTransform:
    """
    Sample (ec, es) where:
      ec = e cos(omega)
      es = e sin(omega)

    Internal outputs:
      e = sqrt(ec^2 + es^2)
      omega = atan2(es, ec)

    NOTE: ec/es allow e near 0 nicely, but the implied prior differs from hk.
    """
    kind: str = "ecosesin"

    def sampled_names(self, planet: str) -> Tuple[str, ...]:
        return (_pname(planet, "ec"), _pname(planet, "es"))

    def internal_names(self, planet: str) -> Tuple[str, str]:
        return (_pname(planet, "e"), _pname(planet, "omega"))

    def to_internal(self, values: Mapping[str, Any], planet: str) -> Dict[str, Any]:
        ec = values[_pname(planet, "ec")]
        es = values[_pname(planet, "es")]
        e_name, w_name = self.internal_names(planet)

        # e = sqrt(ec^2 + es^2)
        e = _sqrt(ec * ec + es * es)
        w = _atan2(es, ec)
        w = _wrap_pi(w)
        return {e_name: e, w_name: w}

    def constraints(self, values: Mapping[str, Any], planet: str) -> List[Tuple[str, Any]]:
        ec = values[_pname(planet, "ec")]
        es = values[_pname(planet, "es")]
        e = _sqrt(ec * ec + es * es)
        return [
            (f"{planet}.e_ge0", e >= 0.0),
            (f"{planet}.e_lt1", e < 1.0),
        ]


# ----------------------------
# Constraint application helpers
# ----------------------------

def apply_hard_constraints_pymc(constraints: Sequence[Tuple[str, Any]], prefix: str = "c") -> None:
    """
    In PyMC, add hard constraints via Potentials returning 0 or -inf.
    """
    import pymc as pm
    import pytensor.tensor as pt

    for i, (name, cond) in enumerate(constraints):
        # cond should be a boolean tensor
        pm.Potential(f"{prefix}_{name}_{i}", pt.switch(cond, 0.0, -np.inf))


def check_constraints_numpy(constraints: Sequence[Tuple[str, Any]]) -> bool:
    """
    In numpy/emcee land, constraints are booleans.
    Return True if all satisfied.
    """
    for _, cond in constraints:
        if isinstance(cond, (bool, np.bool_)):
            ok = bool(cond)
        else:
            ok = bool(np.all(cond))
        if not ok:
            return False
    return True


# ----------------------------
# Factory
# ----------------------------

def make_ecc_transform(kind: str) -> EccTransform:
    k = kind.lower().strip()
    if k in ("direct", "eomega", "e_omega"):
        return DirectEccTransform()
    if k in ("hk", "sqrt(e)cos", "sqrt(e)sin"):
        return HKTransform()
    if k in ("ecosesin", "ecos_esin", "ec_es", "eces"):
        return EccSinCosTransform()
    raise ValueError(f"Unknown eccentricity parameterization: '{kind}'")
    with pm.Model() as model:
        ctx = builder.build()
        # sample settings live in config
        idata = pm.sample(
            draws=int(config.get("draws", 2000)),
            tune=int(config.get("tune", 2000)),
            chains=int(config.get("chains", 4)),
            target_accept=float(config.get("target_accept", 0.9)),
        )
    return model, idata
