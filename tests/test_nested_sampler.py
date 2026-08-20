"""`method: nested` -- the unit-cube bridge and the dynesty backend.

The bridge recovers each sampled element's logit transform NUMERICALLY from
the model's own compiled raw -> physical graph (probes at raw = 0, +/-1,
+/-40, one element at a time) and verifies it by round-trip before sampling.
These tests build models with exozippy's raw-variable structure by hand --
a `<name>_raw` standard Normal plus a sigmoid Deterministic `<name>` --
which is exactly the contract the bridge codes against, with likelihoods
whose evidence and mode masses are known in closed form.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.samplers.nested import (
    NestedBridgeError,
    UnitCubeBridge,
    nested_sample,
)


def _make_model(lower, upper, logl_fn):
    """Exozippy-style build: x_raw ~ N(0,1), x = logit-bounded Deterministic,
    the logit correction that makes the prior uniform on [lower, upper], and
    an arbitrary log-likelihood Potential on x."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    span = upper - lower
    with pm.Model() as model:
        raw = pm.Normal("x_raw", 0.0, 1.0, shape=lower.size)
        q = pm.math.sigmoid(raw)
        x = pm.Deterministic("x", lower + span * q)
        # cancel the raw N(0,1) so the prior on x is exactly uniform
        pm.Potential(
            "logit_correction",
            pt.sum(pt.log(q) + pt.log(1 - q))
            + 0.5 * pt.sum(raw**2)
            + 0.5 * lower.size * np.log(2 * np.pi),
        )
        pm.Potential("loglike", logl_fn(x))
    return model


def test_bridge_recovers_the_transform_and_round_trips():
    """
    Given a two-element logit-bounded model,
    When the bridge probes it numerically,
    Then bounds, centers and scales reproduce the graph to round-off and the
      round-trip verification passes.
    """
    # Arrange
    model = _make_model([2.0, -5.0], [4.0, 10.0], lambda x: 0.0 * x[0])

    # Act
    b = UnitCubeBridge(model)
    b.verify()

    # Assert
    np.testing.assert_allclose(b.lower, [2.0, -5.0], atol=1e-8)
    np.testing.assert_allclose(b.lower + b.span, [4.0, 10.0], atol=1e-8)
    np.testing.assert_allclose(b.c, 0.0, atol=1e-9)
    np.testing.assert_allclose(b.s, 1.0, rtol=1e-9)


def test_bridge_refuses_a_coupled_support():
    """
    Given an element whose interval depends on ANOTHER element (a dynamic
      linked bound),
    When the bridge verifies,
    Then it raises rather than sampling a wrong prior.

    The per-element probes cannot see the coupling (each is measured with
    the others at the origin); the round-trip check exists for exactly this.
    """
    with pm.Model() as model:
        a_raw = pm.Normal("a_raw", 0.0, 1.0)
        b_raw = pm.Normal("b_raw", 0.0, 1.0)
        a = pm.Deterministic("a", 1.0 + 2.0 * pm.math.sigmoid(a_raw))
        # b's upper bound IS a -- a dynamic interval
        pm.Deterministic("b", 0.0 + a * pm.math.sigmoid(b_raw))

    bridge = UnitCubeBridge(model)
    with pytest.raises(NestedBridgeError, match="round-trip"):
        bridge.verify()


def test_nested_recovers_evidence_and_posterior_of_a_gaussian():
    """
    Given a uniform prior on [0, 10] and a Gaussian likelihood N(3, 0.5),
    When nested sampling runs,
    Then logZ matches the analytic value log(1/10) and the posterior mean
      and sd match the Gaussian.
    """
    # Arrange: 1-D; evidence = integral of N(x;3,0.5)/10 dx = 1/10.
    model = _make_model(
        [0.0],
        [10.0],
        lambda x: (
            -0.5 * ((x[0] - 3.0) / 0.5) ** 2 - np.log(0.5 * np.sqrt(2 * np.pi))
        ),
    )

    # Act
    idata = nested_sample(model, None, nlive=200, dlogz=0.1, cores=1, seed=3)

    # Assert
    logz = idata.posterior.attrs["nested_logz"]
    assert logz == pytest.approx(np.log(0.1), abs=0.15)
    x = np.asarray(idata.posterior["x"]).ravel()
    assert x.mean() == pytest.approx(3.0, abs=0.05)
    assert x.std() == pytest.approx(0.5, abs=0.08)


def test_nested_weighs_two_modes_by_mass_not_peak():
    """
    Given two well-separated Gaussian modes where the SECOND is taller but
      narrower, arranged so the wider mode carries 4x the mass,
    When nested sampling runs,
    Then the posterior mass lands ~4:1 for the WIDE mode -- the opposite of
      a peak-height comparison.

    This is the property the whole feature exists for: mode weights with
    the volume in them.
    """

    # Arrange: p(x) propto 0.8*N(2, 0.4) + 0.2*N(8, 0.02).
    # Peak heights: 0.8/0.4 = 2.0 vs 0.2/0.02 = 10.0 -- the narrow mode is
    # 5x TALLER, but carries 1/4 of the mass.
    def logl(x):
        a = pt.log(0.8) - 0.5 * ((x[0] - 2.0) / 0.4) ** 2 - pt.log(0.4)
        b = pt.log(0.2) - 0.5 * ((x[0] - 8.0) / 0.02) ** 2 - pt.log(0.02)
        return pt.logaddexp(a, b)

    model = _make_model([0.0], [10.0], logl)

    # Act
    idata = nested_sample(model, None, nlive=400, dlogz=0.1, cores=1, seed=7)

    # Assert
    x = np.asarray(idata.posterior["x"]).ravel()
    frac_wide = float(np.mean(x < 5.0))
    assert frac_wide == pytest.approx(0.8, abs=0.06)
    # and the weighted `nested` group is attached for the ledger consumer
    assert hasattr(idata, "nested")
    assert idata.nested.sizes["point"] > 0


def test_run_py_exposes_the_nested_keys():
    """The sampler-config vocabulary knows the nested knobs."""
    from exozippy.run import KNOWN_SAMPLER_KEYS

    for key in ("nested_backend", "nlive", "dlogz", "walks"):
        assert key in KNOWN_SAMPLER_KEYS, key


def test_bridge_maps_raw_elements_past_pinned_physical_elements():
    """
    Given a parameter whose physical vector has a PINNED middle element (so
      raw index 1 corresponds to physical index 2),
    When the bridge probes the model,
    Then the element map lands on the moving elements and the transform
      round-trips.

    This is the bug the d=27 pilot exposed: raw vectors hold only the
    SAMPLED elements, physical vectors hold ALL of them, and assuming flat
    raw index k = flat physical index k read back a pinned constant --
    "non-positive span on star.pm_ra" -- for the element actually sampled.
    """
    with pm.Model() as model:
        raw = pm.Normal("x_raw", 0.0, 1.0, shape=2)
        q = pm.math.sigmoid(raw)
        pm.Deterministic(
            "x",
            pt.stack(
                [
                    1.0 + 2.0 * q[0],  # sampled -> physical [0]
                    pt.constant(7.5),  # PINNED  -> physical [1]
                    -4.0 + 8.0 * q[1],  # sampled -> physical [2]
                ]
            ),
        )
        pm.Potential(
            "logit_correction",
            pt.sum(pt.log(q) + pt.log(1 - q))
            + 0.5 * pt.sum(raw**2)
            + np.log(2 * np.pi),
        )

    b = UnitCubeBridge(model)
    b.verify()

    assert b._col_map == [("x", 0), ("x", 2)]
    np.testing.assert_allclose(b.lower, [1.0, -4.0], atol=1e-8)
    np.testing.assert_allclose(b.lower + b.span, [3.0, 4.0], atol=1e-8)


def test_bridge_classifies_and_samples_a_normal_prior_element():
    """
    Given a model mixing a logit-bounded element with a raw-Normal-prior one
      (an unbounded linear transform, no correction potential -- the
      Gaussian-kinematics pattern),
    When the bridge classifies and nested sampling runs,
    Then the element is recognized as non-logit, and its posterior under a
      Gaussian likelihood matches the analytic product of prior and
      likelihood.

    This is the d=27 pilot's failure as a regression: pm_ra/pm_dec/rv and
    the log-normal error scales are raw-Normal-prior elements, and fitting a
    logit to them produced c ~ 0, s ~ 0.05, raw(u) ~ +/-50 -- a wrong prior.
    """
    with pm.Model() as model:
        b_raw = pm.Normal("b_raw", 0.0, 1.0)
        q = pm.math.sigmoid(b_raw)
        pm.Deterministic("b", 1.0 + 2.0 * q)
        pm.Potential(
            "logit_correction",
            pt.log(q)
            + pt.log(1 - q)
            + 0.5 * b_raw**2
            + 0.5 * np.log(2 * np.pi),
        )
        g_raw = pm.Normal("g_raw", 0.0, 1.0)
        g = pm.Deterministic("g", 5.0 + 2.0 * g_raw)  # prior N(5, 2)
        # likelihood N(9, 1) on g -> posterior N(8.2, sqrt(0.8))
        pm.Potential("g_like", -0.5 * ((g - 9.0) / 1.0) ** 2)

    b = UnitCubeBridge(model)
    b.verify()
    kinds = dict(zip(b.flat_names, b.is_logit))
    assert kinds["b_raw[0]"] if "b_raw[0]" in kinds else kinds["b_raw"]
    g_key = "g_raw[0]" if "g_raw[0]" in kinds else "g_raw"
    assert not kinds[g_key], "Gaussian element misclassified as logit"

    idata = nested_sample(model, None, nlive=300, dlogz=0.1, cores=1, seed=5)
    g_post = np.asarray(idata.posterior["g"]).ravel()
    assert g_post.mean() == pytest.approx(8.2, abs=0.1)
    assert g_post.std() == pytest.approx(np.sqrt(0.8), abs=0.1)
