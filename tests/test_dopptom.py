"""
Doppler Tomography component -- unit tests.

The differentiable shadow kernel (components/dopptom) is validated against a
brute-force numpy reference (fine-grid discrete convolution of the rotation
half-ellipse with the Gaussian broadening), mirroring EXOFASTv2's
dopptom_chi2.pro construction; plus normalization, differentiability, and an
end-to-end finite-logp build on the KELT-17 TRES DT example (skipped if the
example data are not present).
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.dopptom.dopptom import (
    cheb_gauss2,
    choose_quadrature_order,
    dt_shadow,
)


def _numpy_shadow(v, center, halfwidth, sigma):
    """Brute-force reference: half-ellipse (unit area) x Gaussian by discrete
    convolution on a fine grid (EXOFASTv2 dopptom_chi2 construction)."""
    dv = 0.005
    grid = np.arange(v.min() - 8 * sigma, v.max() + 8 * sigma, dv)
    c2 = ((grid - center) / halfwidth) ** 2
    prof = np.where(
        c2 < 1,
        2.0 / (np.pi * halfwidth) * np.sqrt(np.clip(1 - c2, 0, None)),
        0.0,
    )
    kx = np.arange(-int(6 * sigma / dv), int(6 * sigma / dv) + 1) * dv
    ker = np.exp(-0.5 * (kx / sigma) ** 2)
    ker /= ker.sum()
    conv = np.convolve(prof, ker, mode="same")
    return np.interp(v, grid, conv)


def test_shadow_matches_bruteforce_convolution():
    """The Chebyshev-Gauss quadrature shadow equals the discrete ellipse (x)
    Gaussian convolution to <1% of the peak, across the profile."""
    v = np.linspace(-60.0, 60.0, 400)
    vsini, p, sigma = 44.0, 0.096, 3.5
    subx = np.array([-0.6, 0.0, 0.55])
    sh = pytensor.function(
        [], dt_shadow(v, pt.as_tensor_variable(subx), vsini, p, sigma)
    )()
    for i, ux in enumerate(subx):
        ref = _numpy_shadow(v, vsini * ux, vsini * p, sigma)
        assert np.max(np.abs(sh[i] - ref)) < 0.01 * ref.max()


def test_shadow_high_width_ratio_accuracy():
    """A narrow-line fast rotator: ellipse half-width 20x the Gaussian
    sigma (PR #323 review -- a fixed 16-node rule errs by ~half the peak
    here).  With the order chosen by choose_quadrature_order the profile
    matches brute force to <1% of the peak, at every strip position."""
    v = np.linspace(-120.0, 120.0, 1200)
    vsini, p, sigma = 100.0, 0.1, 0.5  # halfwidth/sigma = 20
    ratio = vsini * p / sigma
    n_gl = choose_quadrature_order(ratio)
    assert n_gl > 16
    subx = np.array([-0.6, 0.0, 0.55])
    sh = pytensor.function(
        [],
        dt_shadow(v, pt.as_tensor_variable(subx), vsini, p, sigma, n_gl=n_gl),
    )()
    for i, ux in enumerate(subx):
        ref = _numpy_shadow(v, vsini * ux, vsini * p, sigma)
        assert np.max(np.abs(sh[i] - ref)) < 0.01 * ref.max()


def test_choose_quadrature_order_bounds():
    """Floor for the wide-line regime, growth with the ratio, and a cap
    for pathological inputs."""
    assert choose_quadrature_order(0.5) == 16
    assert choose_quadrature_order(1.4) == 16  # the KELT-17 regime
    n20 = choose_quadrature_order(20.0)
    assert 60 <= n20 <= 192
    assert choose_quadrature_order(1e6) == 192


def test_shadow_unit_area():
    """Each shadow profile integrates to 1 (analytic normalization)."""
    v = np.linspace(-120.0, 120.0, 2401)
    sh = pytensor.function(
        [],
        dt_shadow(v, pt.as_tensor_variable(np.array([0.3])), 44.0, 0.1, 3.0),
    )()
    assert np.trapezoid(sh[0], v) == pytest.approx(1.0, abs=1e-6)


def test_shadow_differentiable():
    """Finite gradients wrt vsini and the subplanet coordinate (NUTS needs
    them through the quadrature)."""
    v = np.linspace(-60.0, 60.0, 200)
    vs = pt.dscalar("vs")
    ux = pt.dscalar("ux")
    sh = dt_shadow(v, pt.stack([ux]), vs, 0.096, 3.5)
    g = pytensor.function([vs, ux], pt.grad(pt.sum(pt.sqr(sh)), [vs, ux]))
    for vals in [(44.0, -0.5), (20.0, 0.0), (60.0, 0.9)]:
        assert all(np.isfinite(x) for x in g(*vals))


def test_cheb_weights_sum_to_one():
    for n in (8, 32, 64):
        _, w = cheb_gauss2(n)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# End-to-end: the fast KELT-17 DT example builds and yields a finite logp
# (skipped if the DT FITS data are not shipped).  kelt17_fast.yaml, not
# kelt17_dt.yaml, deliberately: the full config carries an SED whose
# TYCHO/SLOAN BC tables are generated on first use by downloading the
# NextGen spectra -- hundreds of MB from Zenodo inside a unit test on a
# fresh clone, or a hard failure with no network (review on PR #323).
# The fast config has no sed: block and exercises the same DT component.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def kelt17_fast_system():
    import os

    import yaml

    from exozippy.system import System

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exdir = os.path.join(root, "examples", "kelt17")
    cfgfile = os.path.join(exdir, "kelt17_fast.yaml")
    if not os.path.exists(cfgfile):
        pytest.skip("kelt17 fast DT example not present")
    with open(cfgfile) as fh:
        cfg = yaml.safe_load(fh)
    cwd = os.getcwd()
    try:
        os.chdir(exdir)
        s = System(cfg)
        s.prepare()
        model = s.build_model()
    finally:
        os.chdir(cwd)
    return s, model


def test_dt_system_logp_finite(kelt17_fast_system):
    s, model = kelt17_fast_system
    lp = float(model.compile_logp()(model.initial_point()))
    assert np.isfinite(lp), f"DT example logp not finite: {lp}"


def test_shadow_window_check_fires_from_sampled_pair(
    kelt17_fast_system, caplog
):
    """The post-fit window check must work from the SAMPLED
    sqrt(vsini)cos/sin(lambda) pair alone: in a single-orbit fit the
    derived orbit.vsini is not a trace variable (force_node is off on
    the every-orbit-targeted path), and a guard on it left the check
    dead in exactly the fits the window is sized for (review on
    d961ec7).  A point whose sv pair implies ~3x the start vsini must
    trigger the clipped-window warning WITHOUT any 'orbit.vsini' key;
    one near the start values must not."""
    import logging

    s, model = kelt17_fast_system
    dt = s.dopptom
    half = dt._window_half_kms[0]
    assert half is not None and np.isfinite(half)

    # sv pair for ~3x the KELT-17 start vsini (44.2 -> ~133 km/s):
    # sc^2 + ss^2 = vsini [m/s]
    hot = {
        "orbit.svcoslam": np.array([np.sqrt(133e3)]),
        "orbit.svsinlam": np.array([0.0]),
        "star.vline": np.array([5490.0]),
    }
    cool = {
        "orbit.svcoslam": np.array([-90.87]),
        "orbit.svsinlam": np.array([-189.59]),
        "star.vline": np.array([5490.0]),
    }
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        dt._check_shadow_window([cool])
        assert not any("shadow window" in r.message for r in caplog.records)
        assert not any("skipped" in r.message for r in caplog.records)
        caplog.clear()
        dt._check_shadow_window([hot])
        assert any("shadow window" in r.message for r in caplog.records), (
            "clipped-window warning did not fire"
        )
        # and the check must not have been skipped for a missing key
        assert not any("skipped" in r.message for r in caplog.records)
