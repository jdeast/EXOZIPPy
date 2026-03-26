# src/exozippy/rv/keplerian.py
"""
Keplerian radial velocity model (RV canon):

Parameters:
- period [days]
- t_peri [days]
- ecc in [0, 1)
- omega [rad]  (argument of periastron)
- K [m/s]      (semi-amplitude)
- gamma [m/s]  (systemic velocity offset)

Model (Lovis & Fischer form):
  RV(t) = gamma + K * [cos(ω + f(t)) + e cos ω]
where f(t) is true anomaly.

Sign convention:
- Positive RV = redshift (standard in RV literature).
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt

from exozippy.dynamics.kepler import as_tensor, mean_anomaly, true_anomaly_sin_cos


def rv_keplerian(
    t,
    period,
    t_peri,
    ecc,
    omega,
    K,
    gamma=0.0,
    n_iter: int = 8,
):
    """
    Compute RV(t) for one planet.

    Works with floats/ndarrays or PyTensor variables.

    Notes:
    - For ecc == 0, this reduces to gamma + K*cos(omega + M),
      and omega is formally degenerate; we still compute the general form.
    """
    t = as_tensor(t)
    period = as_tensor(period)
    t_peri = as_tensor(t_peri)
    ecc = as_tensor(ecc)
    omega = as_tensor(omega)
    K = as_tensor(K)
    gamma = as_tensor(gamma)

    M = mean_anomaly(t, period, t_peri)

    # True anomaly
    sinf, cosf = true_anomaly_sin_cos(M, ecc, n_iter=n_iter)

    # cos(ω+f) = cosω cosf - sinω sinf
    cosw = pt.cos(omega)
    sinw = pt.sin(omega)
    cos_w_plus_f = cosw * cosf - sinw * sinf

    return gamma + K * (cos_w_plus_f + ecc * cosw)


def rv_multi(
    t,
    planets,
    gamma=0.0,
    n_iter: int = 8,
):
    """
    Sum RV contributions from multiple planets.

    planets: iterable of dict-like objects with keys:
      period, t_peri, ecc, omega, K
    """
    t = as_tensor(t)
    rv = as_tensor(gamma) + pt.zeros_like(t, dtype="float64")

    for p in planets:
        rv = rv + rv_keplerian(
            t=t,
            period=p["period"],
            t_peri=p["t_peri"],
            ecc=p["ecc"],
            omega=p["omega"],
            K=p["K"],
            gamma=0.0,
            n_iter=n_iter,
        )
    return rv
