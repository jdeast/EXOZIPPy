"""
Shared quadratic limb-darkening occultation flux (components/limbdark.py).

exoplanet_core's quad_solution_vector returns s in starry's Green's basis, so
the quadratic law has to go through a change-of-basis before the dot product.
Getting it wrong is exact at u2 = 0 and silently biased otherwise -- which is
how components/rm.py carried the naive [1-u1-u2, u1, u2] basis undetected.
These tests pin the helper against brute-force disk integration at u2 != 0.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.limbdark import quad_limb_darkened_flux


def _brute(b, r, u1, u2, n=2000):
    """Reference normalized flux by direct integration of the occulted region.

    Polar grid centered on the occulter; I(mu) = 1 - u1(1-mu) - u2(1-mu)^2
    integrated over the part of the occulter disk that overlaps the star,
    divided by the full-disk flux pi*(1 - u1/3 - u2/6).
    """
    rr = (np.arange(n) + 0.5) / n * r
    th = (np.arange(n) + 0.5) / n * 2.0 * np.pi
    R, T = np.meshgrid(rr, th, indexing="ij")
    d2 = (b + R * np.cos(T)) ** 2 + (R * np.sin(T)) ** 2
    mu = np.sqrt(np.maximum(0.0, 1.0 - d2))
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    d_area = (r / n) * (2.0 * np.pi / n) * R
    blocked = np.sum(np.where(d2 < 1.0, intensity * d_area, 0.0))
    return 1.0 - blocked / (np.pi * (1.0 - u1 / 3.0 - u2 / 6.0))


def _flux(bv, r, u1, u2):
    b = pt.dvector("b")
    fn = pytensor.function([b], quad_limb_darkened_flux(b, r, u1, u2))
    return fn(bv)


_B = np.array([0.0, 0.2, 0.5, 0.8, 0.95, 1.0, 1.05, 2.0])


@pytest.mark.parametrize(
    "u1, u2",
    [(0.4, 0.3), (0.3, 0.2), (0.6, -0.1), (0.0, 0.5), (0.4, 0.0)],
)
def test_matches_brute_force_disk_integration(u1, u2):
    """Given a quadratic limb-darkening law with u2 != 0, When we evaluate the
    helper across ingress/egress, Then it matches direct numerical integration
    of the occulted intensity to << 1% of the transit depth."""
    r = 0.1
    ref = np.array([_brute(b, r, u1, u2) for b in _B])
    got = _flux(_B, r, u1, u2)
    depth = np.max(1.0 - ref)
    np.testing.assert_allclose(got, ref, atol=1e-4 * depth)


def test_naive_mu_basis_would_fail():
    """The naive [1-u1-u2, u1, u2] coefficients (rm.py's pre-fix basis) are
    wrong by ~87% of the transit depth at u2 = 0.3 -- and even give flux > 1.
    Pinned so the regression can't come back looking harmless."""
    from exoplanet_core.pymc import ops

    r, u1, u2 = 0.1, 0.4, 0.3
    b = pt.dvector("b")
    s_vec = ops.quad_solution_vector(b, r + pt.zeros_like(b))
    c = pt.stack([1.0 - u1 - u2, u1, u2])
    s_off = pt.as_tensor_variable(np.array([np.pi, 2.0 * np.pi / 3.0, 0.0]))
    naive = pytensor.function([b], pt.dot(s_vec, c) / pt.dot(s_off, c))(_B)

    ref = np.array([_brute(b_, r, u1, u2) for b_ in _B])
    depth = np.max(1.0 - ref)
    assert np.max(np.abs(naive - ref)) / depth > 0.5
    assert np.max(naive) > 1.0  # unphysical: "occultation" brightens the star


def test_agrees_with_naive_basis_at_u2_zero():
    """At u2 = 0 the Green's-basis and mu-power coefficients coincide -- the
    reason the wrong basis survived every u2 = 0 test."""
    from exoplanet_core.pymc import ops

    r, u1 = 0.1, 0.4
    b = pt.dvector("b")
    s_vec = ops.quad_solution_vector(b, r + pt.zeros_like(b))
    c = pt.stack([1.0 - u1, u1, 0.0])
    s_off = pt.as_tensor_variable(np.array([np.pi, 2.0 * np.pi / 3.0, 0.0]))
    naive = pytensor.function([b], pt.dot(s_vec, c) / pt.dot(s_off, c))(_B)
    np.testing.assert_allclose(_flux(_B, r, u1, 0.0), naive, rtol=1e-12)


def test_unocculted_flux_is_unity():
    """Well outside the stellar disk the normalization must return exactly 1."""
    got = _flux(np.array([1.5, 3.0, 10.0]), 0.1, 0.4, 0.3)
    np.testing.assert_allclose(got, 1.0, atol=1e-12)


def test_broadcasts_vector_coefficients():
    """Per-row u1/u2 (transit.py's per-band mapped vectors) broadcast against a
    2-D separation array, matching the scalar result row by row."""
    r = 0.1
    b2 = np.tile(_B, (3, 1))  # (3, n_b)
    u1v = np.array([0.4, 0.3, 0.0])
    u2v = np.array([0.3, 0.2, 0.5])
    b = pt.dmatrix("b")
    u1 = pt.dvector("u1")
    u2 = pt.dvector("u2")
    fn = pytensor.function(
        [b, u1, u2],
        quad_limb_darkened_flux(b, r, u1[:, None], u2[:, None]),
    )
    got = fn(b2, u1v, u2v)
    for i in range(3):
        np.testing.assert_allclose(
            got[i], _flux(_B, r, u1v[i], u2v[i]), rtol=1e-12
        )


def test_is_differentiable():
    """Finite gradients wrt u1/u2 (the coefficients are sampled)."""
    b = pt.dvector("b")
    u1 = pt.dscalar("u1")
    u2 = pt.dscalar("u2")
    flux = quad_limb_darkened_flux(b, 0.1, u1, u2)
    g = pytensor.function([b, u1, u2], pt.grad(pt.sum(flux), [u1, u2]))
    for grad in g(np.array([0.0, 0.5, 0.95]), 0.4, 0.3):
        assert np.isfinite(grad)
