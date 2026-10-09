"""The timing information that chooses the sampled epoch (orbit/timing.py).

JDE 2026-10-08: "The true optimal epoch is the information weighted epoch,
which will heavily favor transits, but a long rv baseline can beat it out
... Carter+2008 estimates the tc precision from transits.  There's probably
something similar for rvs, and that could be used to estimate the
information weighted epoch."  These pin each estimator to its closed form,
and the claim that the information-weighted mean epoch IS the epoch at which
the conjunction and the period decorrelate.
"""

import numpy as np
import pytest

from exozippy.components.orbit import timing

P = 3.0
TC = 2455000.0


def _rv(t, tc, period, K=50.0):
    return K * np.cos(2.0 * np.pi * (t - tc) / period + np.pi / 2.0)


def test_circular_rv_information_is_k_two_pi_over_p_sine_squared():
    """
    Given a circular orbit,
    When the per-RV information on tc is computed,
    Then it is (K (2 pi / P) sin(phase) / sigma)^2 -- zero at quadrature,
      largest at conjunction.
    """
    t = TC + np.linspace(0.0, P, 13)
    K, sigma = 50.0, 5.0
    got = timing.rv_information(t, sigma, TC, P, 0.0, np.pi / 2.0, K)
    phase = 2.0 * np.pi * (t - TC) / P
    want = (K * 2.0 * np.pi / P * np.cos(phase) / sigma) ** 2
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9 * want.max())


@pytest.mark.parametrize("ecc,omega", [(0.0, 1.0), (0.4, 2.0), (0.8, -0.7)])
def test_rv_information_is_the_squared_derivative(ecc, omega):
    """
    Given an eccentric orbit,
    When the per-RV information is computed,
    Then it equals the finite-difference (dv/dtc / sigma)^2 of the
      Keplerian at fixed e and omega.
    """
    K, sigma = 30.0, 2.0
    t = TC + np.linspace(-2.0 * P, 3.0 * P, 41)

    def v(tc):
        f = timing.true_anomaly(t, tc, P, ecc, omega)
        return K * (np.cos(f + omega) + ecc * np.cos(omega))

    # h large enough that tc + h is not rounded at a 2.5e6-day epoch
    h = 1e-4
    deriv = (v(TC + h) - v(TC - h)) / (2.0 * h)
    got = timing.rv_information(t, sigma, TC, P, ecc, omega, K)
    want = (deriv / sigma) ** 2
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-8 * want.max())


def test_a_dense_transit_carries_carters_information():
    """
    Given one uniformly sampled transit,
    When the per-point information is summed,
    Then it is Carter et al. (2008) eq. (23)'s sigma_tc^-2 = 2 Gamma
      delta^2 / (tau sigma^2).
    """
    depth, T, tau, sigma = 0.01, 0.12, 0.015, 1e-3
    cadence = 2.0 / 1440.0
    t = TC + np.arange(-0.3, 0.3, cadence)
    w = timing.transit_information(t, sigma, TC, P, depth, T, tau)
    carter = 2.0 / cadence * depth**2 / (tau * sigma**2)
    assert w.sum() == pytest.approx(carter, rel=0.02)
    # and nothing out of transit
    assert (w[np.abs(t - TC) > 0.5 * (T + tau)] == 0).all()


def test_carters_durations():
    """
    Given a circular orbit,
    When the transit shape is computed,
    Then T = T0 sqrt(1 - b^2) and tau = T0 p / sqrt(1 - b^2) with T0 = P /
      (pi a/R*) -- Carter et al. (2008) eqs. (7), (9), (10).
    """
    ar, p, b = 10.0, 0.1, 0.5
    T, tau = timing.transit_shape(P, ar, p, b / ar, 0.0, 0.0)
    T0 = P / (np.pi * ar)
    assert T == pytest.approx(T0 * np.sqrt(1 - b**2))
    assert tau == pytest.approx(T0 * p / np.sqrt(1 - b**2))


def test_an_exposure_longer_than_the_ingress_smears_it():
    depth, T, tau, sigma = 0.01, 0.12, 0.005, 1e-3
    t = TC + np.arange(-0.3, 0.3, 30.0 / 1440.0)
    sharp = timing.transit_information(t, sigma, TC, P, depth, T, tau)
    smeared = timing.transit_information(
        t, sigma, TC, P, depth, T, tau, exptime=30.0 / 1440.0
    )
    assert smeared.sum() < sharp.sum()


def test_face_on_circular_sky_speed():
    t = TC + np.linspace(0, P, 7)
    v2 = timing.sky_speed_sq(t, TC, P, 0.0, 0.3, 1.0, 2.0)
    np.testing.assert_allclose(v2, (2.0 * np.pi * 2.0 / P) ** 2)


def _fisher_optimal_epoch(times, model, sigmas, tc, period):
    """The epoch E at which cov(tc + E P, P) = 0, from the 2x2 Fisher matrix
    of (tc, P) by finite differences of ``model(t, tc, P)``."""
    h_tc, h_p = 1e-4, 1e-7
    g_tc = (
        model(times, tc + h_tc, period) - model(times, tc - h_tc, period)
    ) / (2 * h_tc)
    g_p = (model(times, tc, period + h_p) - model(times, tc, period - h_p)) / (
        2 * h_p
    )
    G = np.column_stack([g_tc, g_p]) / sigmas[:, None]
    cov = np.linalg.inv(G.T @ G)
    return -cov[0, 1] / cov[1, 1]


def test_the_information_weighted_epoch_is_where_tc_and_p_decorrelate():
    """
    Given RVs spread unevenly over many periods with uneven errors,
    When the information-weighted mean epoch is computed,
    Then it equals the zero-covariance epoch of the (tc, P) Fisher matrix
      -- EXOFASTv2's optimal epoch, before sampling.
    """
    rng = np.random.default_rng(1)
    t = np.sort(
        np.concatenate(
            [
                TC + rng.uniform(-200, -150, 30),
                TC + rng.uniform(300, 320, 10),
            ]
        )
    )
    sig = rng.uniform(2.0, 10.0, t.size)
    w = timing.rv_information(t, sig, TC, P, 0.0, np.pi / 2.0, 50.0)
    center = (np.sum(w * t) / w.sum() - TC) / P
    exact = _fisher_optimal_epoch(t, _rv, sig, TC, P)
    assert center == pytest.approx(exact, abs=1e-3)


def test_a_long_rv_baseline_can_outweigh_a_short_transit_cluster():
    """
    Given three noisy transits at one end and many precise RVs over years
      at the other (JDE: "a long rv baseline can beat it out"),
    When each is weighed by its information,
    Then the RVs carry more, and the weighted epoch sits with them -- the
      case an eclipses-first rank could not represent.
    """
    cadence = 10.0 / 1440.0
    transits = np.concatenate(
        [TC + k * P + np.arange(-0.1, 0.1, cadence) for k in (0, 1, 2)]
    )
    w_tr = timing.transit_information(
        transits, 0.02, TC, P, 0.002, 0.08, 0.008
    )
    rv = TC + 1000.0 + np.linspace(0.0, 1500.0, 300)
    w_rv = timing.rv_information(rv, 1.0, TC, P, 0.0, np.pi / 2.0, 100.0)
    assert w_rv.sum() > w_tr.sum()
    t = np.concatenate([transits, rv])
    w = np.concatenate([w_tr, w_rv])
    assert np.sum(w * t) / w.sum() > TC + 1000.0


def test_the_rv_optimum_marginalizes_the_amplitude_and_offsets():
    """
    Given circular RVs from two instruments, unevenly spread in time,
    When the RVs' own optimal epoch is computed,
    Then it is the zero-covariance epoch of the (tc, P) block of the FULL
      Fisher matrix -- tc, P, K and one offset per file -- coded here
      independently.
    """
    rng = np.random.default_rng(4)
    t1 = TC + np.sort(rng.uniform(-400, -380, 12))
    t2 = TC + np.sort(rng.uniform(100, 400, 20))
    s1, s2 = np.full(t1.size, 3.0), np.full(t2.size, 15.0)
    K = 40.0
    ((E, info),) = timing.rv_epoch_information(
        [t1, t2], [s1, s2], [(TC, P, 0.0, np.pi / 2.0, K)], [False]
    )
    t = np.concatenate([t1, t2])
    sig = np.concatenate([s1, s2])
    ph = 2.0 * np.pi * (t - TC) / P
    # v = g_file + K cos(ph + pi/2) = g_file - K sin(ph)
    d_tc = K * np.cos(ph) * 2.0 * np.pi / P
    d_p = K * np.cos(ph) * 2.0 * np.pi * (t - TC) / P**2
    d_k = -np.sin(ph)
    g1 = np.r_[np.ones(t1.size), np.zeros(t2.size)]
    g2 = 1.0 - g1
    G = np.column_stack([d_tc, d_p, d_k, g1, g2]) / sig[:, None]
    C = np.linalg.inv(G.T @ G)
    assert E == pytest.approx(-C[0, 1] / C[1, 1], rel=1e-4, abs=1e-3)
    assert info == pytest.approx(
        1.0 / (C[0, 0] - C[0, 1] ** 2 / C[1, 1]), rel=1e-4
    )


def test_amplitude_and_jitter_come_from_the_data():
    """
    Given RVs with a known amplitude, offsets, and extra white noise beyond
      their quoted errors in one file,
    When the amplitudes and jitters are estimated,
    Then both are recovered -- the noise the fit's jitter term will find.
    """
    rng = np.random.default_rng(7)
    t1 = TC + np.sort(rng.uniform(0, 300, 200))
    t2 = TC + np.sort(rng.uniform(0, 300, 200))
    shape1 = timing.rv_shape(t1, TC, P, 0.0, np.pi / 2.0)
    shape2 = timing.rv_shape(t2, TC, P, 0.0, np.pi / 2.0)
    e1, e2 = np.full(t1.size, 2.0), np.full(t2.size, 2.0)
    v1 = 5.0 + 30.0 * shape1 + rng.normal(0, 2.0, t1.size)
    v2 = -40.0 + 30.0 * shape2 + rng.normal(0, np.hypot(2.0, 10.0), t2.size)
    K, jit = timing.rv_amplitudes_and_jitter(
        [v1, v2], [e1, e2], [[shape1], [shape2]]
    )
    assert K[0] == pytest.approx(30.0, rel=0.05)
    assert jit[0] < 1.5
    assert jit[1] == pytest.approx(10.0, rel=0.15)
