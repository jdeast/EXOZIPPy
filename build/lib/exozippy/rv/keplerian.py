from __future__ import annotations

import numpy as np
import pytensor.tensor as pt


def as_tensor(x):
    return x if isinstance(x, pt.TensorVariable) else pt.as_tensor_variable(x)


def mean_anomaly(t, period, t_peri):
    t = as_tensor(t)
    period = as_tensor(period)
    t_peri = as_tensor(t_peri)
    n = 2.0 * np.pi / period
    return n * (t - t_peri)


def solve_kepler_E(M, e, n_iter: int = 8):
    M = as_tensor(M)
    e = as_tensor(e)
    twopi = 2.0 * np.pi
    Mp = pt.mod(M + np.pi, twopi) - np.pi
    E = Mp + e * pt.sin(Mp)
    for _ in range(int(n_iter)):
        sinE = pt.sin(E)
        cosE = pt.cos(E)
        f = E - e * sinE - Mp
        fp = 1.0 - e * cosE
        E = E - f / fp
    return E


def true_anomaly_sin_cos(M, e, n_iter: int = 8):
    E = solve_kepler_E(M, e, n_iter=n_iter)
    sinE = pt.sin(E)
    cosE = pt.cos(E)
    one_minus_e_cosE = 1.0 - e * cosE
    sqrt_1me2 = pt.sqrt(pt.maximum(1.0 - e * e, 0.0))
    cosf = (cosE - e) / one_minus_e_cosE
    sinf = (sqrt_1me2 * sinE) / one_minus_e_cosE
    return sinf, cosf


def rv_keplerian(t, period, t_peri, ecc, omega, K, gamma=0.0, n_iter: int = 8):
    t = as_tensor(t)
    period = as_tensor(period)
    t_peri = as_tensor(t_peri)
    ecc = as_tensor(ecc)
    omega = as_tensor(omega)
    K = as_tensor(K)
    gamma = as_tensor(gamma)

    M = mean_anomaly(t, period, t_peri)
    sinf, cosf = true_anomaly_sin_cos(M, ecc, n_iter=n_iter)

    cosw = pt.cos(omega)
    sinw = pt.sin(omega)
    cos_w_plus_f = cosw * cosf - sinw * sinf

    return gamma + K * (cos_w_plus_f + ecc * cosw)
