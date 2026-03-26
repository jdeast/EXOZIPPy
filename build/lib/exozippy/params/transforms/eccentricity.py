from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Protocol, Tuple

import numpy as np


def _is_tensor(x: Any) -> bool:
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


def _wrap_pi(x: Any) -> Any:
    twopi = 2.0 * np.pi
    if _is_tensor(x):
        import pytensor.tensor as pt
        return pt.mod(x + np.pi, twopi) - np.pi
    return (x + np.pi) % twopi - np.pi


class EccTransform(Protocol):
    kind: str

    def sampled_names(self, planet: str) -> Tuple[str, ...]:
        ...

    def internal_names(self, planet: str) -> Tuple[str, str]:
        ...

    def to_internal(self, values: Mapping[str, Any], planet: str) -> Dict[str, Any]:
        ...

    def constraints(self, values: Mapping[str, Any], planet: str) -> List[Tuple[str, Any]]:
        ...


def _pname(planet: str, short: str) -> str:
    return f"planet.{planet}.{short}"


@dataclass(frozen=True, slots=True)
class DirectEccTransform:
    kind: str = "direct"

    def sampled_names(self, planet: str) -> Tuple[str, ...]:
        return (_pname(planet, "e"), _pname(planet, "omega"))

    def internal_names(self, planet: str) -> Tuple[str, str]:
        return (_pname(planet, "e"), _pname(planet, "omega"))

    def to_internal(self, values: Mapping[str, Any], planet: str) -> Dict[str, Any]:
        e_name, w_name = self.internal_names(planet)
        e = values[e_name]
        w = _wrap_pi(values[w_name])
        return {e_name: e, w_name: w}

    def constraints(self, values: Mapping[str, Any], planet: str) -> List[Tuple[str, Any]]:
        e = values[_pname(planet, "e")]
        w = values[_pname(planet, "omega")]
        return [
            (f"planet.{planet}.e_ge0", e >= 0.0),
            (f"planet.{planet}.e_lt1", e < 1.0),
            (f"planet.{planet}.omega_gtmPi", w > -np.pi),
            (f"planet.{planet}.omega_lePi", w <= np.pi),
        ]


@dataclass(frozen=True, slots=True)
class HKTransform:
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
        w = _wrap_pi(_atan2(k, h))
        return {e_name: e, w_name: w}

    def constraints(self, values: Mapping[str, Any], planet: str) -> List[Tuple[str, Any]]:
        h = values[_pname(planet, "h")]
        k = values[_pname(planet, "k")]
        e = h * h + k * k
        return [(f"planet.{planet}.hk_disk", e < 1.0)]


@dataclass(frozen=True, slots=True)
class EccSinCosTransform:
    kind: str = "ecosesin"

    def sampled_names(self, planet: str) -> Tuple[str, ...]:
        return (_pname(planet, "ec"), _pname(planet, "es"))

    def internal_names(self, planet: str) -> Tuple[str, str]:
        return (_pname(planet, "e"), _pname(planet, "omega"))

    def to_internal(self, values: Mapping[str, Any], planet: str) -> Dict[str, Any]:
        ec = values[_pname(planet, "ec")]
        es = values[_pname(planet, "es")]
        e_name, w_name = self.internal_names(planet)
        e = _sqrt(ec * ec + es * es)
        w = _wrap_pi(_atan2(es, ec))
        return {e_name: e, w_name: w}

    def constraints(self, values: Mapping[str, Any], planet: str) -> List[Tuple[str, Any]]:
        ec = values[_pname(planet, "ec")]
        es = values[_pname(planet, "es")]
        e = _sqrt(ec * ec + es * es)
        return [
            (f"planet.{planet}.e_ge0", e >= 0.0),
            (f"planet.{planet}.e_lt1", e < 1.0),
        ]


def make_ecc_transform(kind: str):
    k = kind.lower().strip()
    if k in ("direct", "eomega", "e_omega"):
        return DirectEccTransform()
    if k in ("hk", "sqrt(e)cos", "sqrt(e)sin"):
        return HKTransform()
    if k in ("ecosesin", "ecos_esin", "ec_es"):
        return EccSinCosTransform()
    raise ValueError(f"Unknown eccentricity parameterization: '{kind}'")
