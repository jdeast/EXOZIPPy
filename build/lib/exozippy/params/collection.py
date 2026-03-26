# src/exozippy/params_obsolete/collection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .parameter import Parameter


@dataclass(slots=True)
class ParameterSet:
    params: Tuple[Parameter, ...]

    def labels(self) -> List[str]:
        return [p.label for p in self.params]

    def defaults(self) -> Dict[str, float]:
        """
        Numeric defaults for each parameter.
        Preference: initval > mu. Fails if neither is set for a non-expression param.
        """
        out: Dict[str, float] = {}
        for p in self.params:
            if p.expression is not None:
                # deterministics don't have independent numeric defaults
                continue
            if p.initval is not None:
                out[p.label] = float(p.initval)
            elif p.mu is not None:
                out[p.label] = float(p.mu)
            else:
                raise ValueError(f"{p.label}: no initval or mu for numeric default.")
        return out

    def build_pymc(self, user_params: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
        """
        Build all PyMC variables inside a pm.Model() context.
        Returns dict[label -> pm RV/Deterministic].
        """
        out: Dict[str, Any] = {}
        for i, p in enumerate(self.params):
            out[p.label] = p.build_pymc(ndx=i, user_params=user_params)
        return out

    def pack(self, values: Mapping[str, float], labels: Optional[Sequence[str]] = None) -> np.ndarray:
        """
        Dict -> theta vector (ordered).
        Only packs labels you request (default: all non-deterministic params_obsolete in this set).
        """
        if labels is None:
            labels = [p.label for p in self.params if p.expression is None]
        theta = np.empty(len(labels), dtype=float)
        for i, lab in enumerate(labels):
            if lab not in values:
                raise KeyError(f"Missing value for {lab}")
            theta[i] = float(values[lab])
        return theta

    def unpack(self, theta: Sequence[float], labels: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """
        theta vector -> dict (ordered).
        """
        if labels is None:
            labels = [p.label for p in self.params if p.expression is None]
        theta = np.asarray(theta, dtype=float)
        if theta.size != len(labels):
            raise ValueError(f"theta has size {theta.size} but expected {len(labels)}")
        return {lab: float(theta[i]) for i, lab in enumerate(labels)}

    def to_floats(self, values: Mapping[str, Any]) -> Dict[str, float]:
        """
        Ensure all values are concrete floats. This will FAIL for symbolic tensors.
        Use this for non-PyMC code paths.
        """
        out: Dict[str, float] = {}
        for k, v in values.items():
            # numpy scalars are fine; pytensor variables are not.
            if hasattr(v, "owner") or hasattr(v, "type"):  # heuristic for pytensor/pm vars
                raise TypeError(f"{k} is symbolic; cannot convert to float without a draw/value.")
            out[k] = float(np.asarray(v))
        return out

    def values_from_idata(self, idata: Any, chain: int = 0, draw: int = 0) -> Dict[str, float]:
        """
        Pull one draw from an ArviZ InferenceData produced by pm.sample().
        Returns dict[label -> float] for the parameters that appear in posterior.
        """
        post = idata.posterior
        out: Dict[str, float] = {}
        for p in self.params:
            lab = p.label
            if lab in post:
                out[lab] = float(post[lab].isel(chain=chain, draw=draw).values)
        return out
