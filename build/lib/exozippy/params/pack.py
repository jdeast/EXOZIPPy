from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .spec import ParameterSpec


@dataclass(slots=True)
class ParameterSet:
    specs: Tuple[ParameterSpec, ...]

    def free_specs(self) -> List[ParameterSpec]:
        return [s for s in self.specs if not s.fixed]

    def names_free(self) -> List[str]:
        return [s.name for s in self.free_specs()]

    def theta0(self) -> np.ndarray:
        free = self.free_specs()
        theta = np.empty(len(free), dtype=float)
        for i, s in enumerate(free):
            if s.init is None:
                raise ValueError(f"{s.name}: init is required for free parameters")
            theta[i] = float(s.init)
        return theta
