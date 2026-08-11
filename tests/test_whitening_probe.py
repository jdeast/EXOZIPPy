"""Symmetric-curvature whitening probe: the three-rung measurement ladder.

Rung 1: symmetric second difference -> curvature width 1/sqrt(h),
        gradient-immune (the ob140939 failure mode).
Rung 2: h <= 0 (linear/ridge/saddle) -> nearer one-sided contour (errs tight).
Rung 3: flat both ways -> flat_scale.

Plus the probe diagnostics that make a bad start loud instead of silent.
"""

import numpy as np
import pytest

from exozippy.whitening import (
    _probe_element,
    probe_scales,
)


def _delta_fn(logp_1d, x0=0.0):
    """Wrap a 1-d logp into the eval_delta(step) the probe consumes."""
    lp0 = logp_1d(x0)

    def eval_delta(step):
        return lp0 - logp_1d(x0 + step)

    return eval_delta


# ---------------------------------------------------------------------------
# rung 1: curvature
# ---------------------------------------------------------------------------


def test_probe_element_returns_sigma_at_the_mode():
    """
    Given a Gaussian logp probed from its mode,
    When _probe_element measures it,
    Then the scale is sigma via the curvature rung, with ~zero gradient.
    """
    # ARRANGE
    sigma = 0.37
    eval_delta = _delta_fn(lambda x: -0.5 * (x / sigma) ** 2)

    # ACT
    scale, method, g_nats = _probe_element(eval_delta)

    # ASSERT
    assert method == "curvature"
    assert scale == pytest.approx(sigma, rel=0.05)
    assert g_nats == pytest.approx(0.0, abs=1e-6)


def test_probe_element_is_gradient_immune_far_from_the_mode():
    """
    Given a Gaussian logp probed from 200 sigma below its mode (a start
      ~20000 nats off, the ob140939 regime),
    When _probe_element measures it,
    Then the scale is still sigma -- the second difference cancels the
      gradient term identically -- and grad_nats reports the ~200-sigma
      displacement.

    The one-sided drop this rung replaces returns ~0.5/|g| = sigma/400
    here: the 1000x-too-tight scales that re-widened ob140939's raw
    posterior into the _RAW_CANCELLATION_CLIP wall (86% divergences).
    """
    # ARRANGE: mode at 200, sigma 1, start at 0
    delta_sigmas = 200.0
    eval_delta = _delta_fn(lambda x: -0.5 * (x - delta_sigmas) ** 2)

    # ACT
    scale, method, g_nats = _probe_element(eval_delta)

    # ASSERT
    assert method == "curvature"
    assert scale == pytest.approx(1.0, rel=0.05)
    assert g_nats == pytest.approx(delta_sigmas, rel=0.1)
    # the old one-sided measurement for reference: ~400x too tight
    assert 0.5 / delta_sigmas < 0.01 * scale


def test_probe_element_anisotropic_scales_far_from_mode():
    """
    Given a 2-element Gaussian with very different sigmas, probed off-mode,
    When probe_scales measures it,
    Then each element recovers its own sigma (no cross-contamination).
    """
    # ARRANGE
    sigmas = np.array([0.02, 30.0])
    modes = np.array([1.0, -900.0])  # 50 and 30 sigma displaced
    start = {"x": np.zeros(2)}

    def logp_fn(p):
        return float(-0.5 * np.sum(((p["x"] - modes) / sigmas) ** 2))

    # ACT
    _, scales = probe_scales(start, logp_fn)

    # ASSERT
    np.testing.assert_allclose(scales["x"], sigmas, rtol=0.05)


# ---------------------------------------------------------------------------
# rung 2: h <= 0 fallback to the nearer one-sided contour
# ---------------------------------------------------------------------------


def test_probe_element_linear_logp_falls_back_to_one_sided():
    """
    Given a logp exactly linear in the element (h = 0),
    When _probe_element measures it,
    Then curvature is unmeasurable and the one-sided 0.5-nat step 0.5/g is
      returned via the linear rung.
    """
    # ARRANGE
    g = 0.32
    eval_delta = _delta_fn(lambda x: -g * x)

    # ACT
    scale, method, g_nats = _probe_element(eval_delta)

    # ASSERT
    assert method == "linear"
    assert scale == pytest.approx(0.5 / g, rel=0.05)
    assert np.isnan(g_nats)


def test_probe_element_negative_curvature_falls_back():
    """
    Given a start where logp is CONVEX along the element (h < 0 for the
      concavity the probe needs: a shoulder between basins) but still
      downhill in one direction,
    When _probe_element measures it,
    Then the curvature rung declines (C(s) < 0 at every scale) and the
      nearer one-sided contour is used -- which errs tight, never wide.
    """
    # ARRANGE: logp = -g*x + k*x^2 (convex, k > 0).  C(s) = -2k*s^2 < 0
    # everywhere; the +x direction still drops 0.5 nats at the near root of
    # k*s^2 - g*s + 0.5 = 0, and -x rises monotonically (no contour).
    g, k = 0.5, 0.01
    eval_delta = _delta_fn(lambda x: -g * x + k * x**2)
    expected = (g - np.sqrt(g**2 - 4 * k * 0.5)) / (2 * k)

    # ACT
    scale, method, g_nats = _probe_element(eval_delta)

    # ASSERT
    assert method == "linear"
    assert scale == pytest.approx(expected, rel=0.05)
    assert np.isnan(g_nats)


# ---------------------------------------------------------------------------
# rung 3: flat
# ---------------------------------------------------------------------------


def test_probe_element_flat_direction_returns_none():
    """
    Given an element logp ignores entirely,
    When _probe_element measures it,
    Then it reports (None, "flat", nan) so the caller applies flat_scale.
    """
    # ARRANGE
    eval_delta = _delta_fn(lambda x: 0.0)

    # ACT
    scale, method, g_nats = _probe_element(eval_delta)

    # ASSERT
    assert scale is None
    assert method == "flat"
    assert np.isnan(g_nats)


def test_probe_element_wall_bounds_the_scale():
    """
    Given a hard prior wall (logp = -inf) nearer than the curvature width,
    When _probe_element measures it,
    Then the scale is bounded at the wall distance (errs tight) rather than
      proposing across the wall.
    """
    # ARRANGE: sigma 3 Gaussian at the start, wall at x = -0.5
    wall = -0.5

    def logp(x):
        if x <= wall:
            return -np.inf
        return -0.5 * (x / 3.0) ** 2

    eval_delta = _delta_fn(logp)

    # ACT
    scale, method, _ = _probe_element(eval_delta)

    # ASSERT
    assert method == "curvature"
    assert 0.0 < scale <= abs(wall) * 1.05


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def test_probe_scales_fills_diagnostics():
    """
    Given elements exercising all three rungs plus a displaced Gaussian,
    When probe_scales runs with a diagnostics dict,
    Then gradient_nats flags the displaced element, linear_fallback the
      linear one, and flat the ignored one.
    """
    # ARRANGE: x[0] displaced Gaussian, x[1] linear, x[2] flat
    start = {"x": np.zeros(3)}

    def logp_fn(p):
        x = p["x"]
        return float(-0.5 * (x[0] - 50.0) ** 2 - 0.7 * x[1])

    diag = {}

    # ACT
    _, scales = probe_scales(start, logp_fn, diagnostics=diag)

    # ASSERT
    assert diag["gradient_nats"]["x[0]"] == pytest.approx(50.0, rel=0.1)
    assert diag["linear_fallback"] == ["x[1]"]
    assert diag["flat"] == ["x[2]"]
    assert scales["x"][0] == pytest.approx(1.0, rel=0.05)
    assert scales["x"][1] == pytest.approx(0.5 / 0.7, rel=0.05)
    assert scales["x"][2] == pytest.approx(1.0)  # flat_scale


def test_gradient_dominated_start_warns_by_name(caplog):
    """
    Given a model whose start is far below its conditional optimum,
    When apply_measured_whitening probes it,
    Then a WARNING names the gradient-dominated element and its displacement
      -- the loud version of the ob140939 failure signature.
    """
    import logging

    from exozippy.whitening import apply_measured_whitening

    # ARRANGE: a minimal stand-in system exposing one whitened parameter is
    # heavyweight to build here; drive the warning through probe_scales'
    # diagnostics contract instead, monkeypatching a no-op system.
    class _NoParams:
        def get_all_parameters(self):
            return []

        def get_raw_start(self, model):
            return {"x": np.zeros(1)}

    def logp_fn(raw):
        return float(-0.5 * (raw["x"][0] - 40.0) ** 2)

    class _Model:
        def compile_logp(self):
            return logp_fn

    with caplog.at_level(logging.WARNING, logger="exozippy.whitening"):
        # ACT
        report = apply_measured_whitening(_NoParams(), _Model())

    # ASSERT
    assert report["probe_diagnostics"]["gradient_nats"]["x[0]"] == (
        pytest.approx(40.0, rel=0.1)
    )
    assert any(
        "gradient-dominated start" in r.message and "x[0]" in r.message
        for r in caplog.records
    )
