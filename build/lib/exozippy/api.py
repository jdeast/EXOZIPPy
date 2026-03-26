# src/exozippy/api.py
"""
exozippy.api

Stable "front door" API for exozippy.

Design goals:
- Keep the public surface small and stable while internals churn.
- Avoid importing heavy optional dependencies at import time.
- Provide typed, testable interfaces for transit/RV modeling + fitting.

This file should mostly orchestrate calls into implementation modules living under
exozippy.transit/, exozippy.rv/, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


# -----------------------------
# Core domain objects
# -----------------------------

@dataclass(frozen=True, slots=True)
class Star:
    """Host star parameters (keep minimal; expand as you learn what you need)."""
    m_sun: float  # stellar mass in solar masses
    r_sun: float  # stellar radius in solar radii
    teff_k: Optional[float] = None
    feh: Optional[float] = None
    rho_cgs: Optional[float] = None


@dataclass(frozen=True, slots=True)
class Planet:
    """Planet parameters (minimal working set for RV/transit)."""
    name: str = "b"

    # Orbital elements
    period_days: float = 1.0
    t0_bjd: float = 0.0  # reference epoch (mid-transit or periastron; your convention)
    ecc: float = 0.0
    omega_rad: float = 0.0  # argument of periastron [rad]
    inc_rad: Optional[float] = None  # inclination [rad], optional for RV-only

    # Size/mass
    rp_rs: Optional[float] = None  # Rp/R*
    mp_mjup: Optional[float] = None  # planet mass in M_Jup (optional if using K)
    k_mps: Optional[float] = None  # RV semi-amplitude [m/s] (optional if using Mp)

    # Transit geometry
    a_rs: Optional[float] = None  # a/R*
    b: Optional[float]
