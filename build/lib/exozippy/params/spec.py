from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np

Number = Union[int, float, np.floating]


@dataclass(frozen=True, slots=True)
class PriorSpec:
    dist: str
    kwargs: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    prior: Optional[PriorSpec] = None
    init: Optional[Number] = None
    fixed: bool = False
    bounds: Tuple[Optional[Number], Optional[Number]] = (None, None)
    shape: Tuple[int, ...] = ()

    def tightened_bounds(self, user: Optional[Mapping[str, Mapping[str, Any]]] = None):
        lo, hi = self.bounds
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if user is None:
            return lo, hi
        u = user.get(self.name)
        if u is None and "_0" in self.name:
            u = user.get(self.name.split("_")[0])
        if u is None:
            return lo, hi
        ulo = u.get("lower", None)
        uhi = u.get("upper", None)
        if ulo is not None:
            lo = float(ulo) if lo is None else max(lo, float(ulo))
        if uhi is not None:
            hi = float(uhi) if hi is None else min(hi, float(uhi))
        return lo, hi
