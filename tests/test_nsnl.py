"""Tests for the NSNL (N-source, N-lens) pathway.

The 2S2L configuration mirrors examples/ob161003 (OGLE-2016-BLG-1003,
Jung et al. 2017): two source stars sharing a binary star lens.
"""
import numpy as np
import pytest

from exozippy.system import System
from exozippy.config import ConfigManager


def _config_2s2l():
    return {
        "star": [
            {"name": "Lens"},
            {"name": "LensB"},
            {"name": "SourceA"},
            {"name": "SourceB"},
        ],
        "lens": [{
            "name": "Lens",
            "lenses": ["star.0", "star.1"],
            "sources": ["star.2", "star.3"],
            "finite_source": True,
        }],
    }


def _params_2s2l():
    coords = {"initval": 264.10513, "sigma": 0}
    coords_dec = {"initval": -27.188861, "sigma": 0}
    p = {
        "lens.SourceA.t_0": {"initval": 2457551.038, "init_scale": 0.2},
        "lens.SourceB.t_0": {"initval": 2457552.517, "init_scale": 0.13},
        "lens.SourceA.u_0": {"initval": 0.059, "init_scale": 0.013},
        "lens.SourceB.u_0": {"initval": 0.135, "init_scale": 0.006},
        "lens.SourceA.t_E": {"initval": 28.931, "init_scale": 0.7},
        "lens.SourceB.t_E": {"initval": 28.931, "init_scale": 0.7},
        "lens.SourceA.rho": {"initval": 0.000451, "init_scale": 0.0002},
        "lens.SourceB.rho": {"initval": 0.001293, "init_scale": 0.00016},
        "lens.Lens.s": {"initval": 1.033, "init_scale": 0.011},
        "lens.Lens.alpha": {"initval": 131.757, "init_scale": 0.9},
        "lens.Lens.q": {"initval": 1.188, "init_scale": 0.04},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for s in ("Lens", "LensB", "SourceA", "SourceB"):
        p[f"star.{s}.ra"] = dict(coords)
        p[f"star.{s}.dec"] = dict(coords_dec)
    return p


@pytest.fixture(scope="module")
def system_2s2l():
    """Given a 2S2L config seeded with the Jung+2017 standard solution,
    when the system is prepared and built, provide (system, model)."""
    system = System(_config_2s2l(), user_params=_params_2s2l())
    system.prepare()
    model = system.build_model()
    return system, model


def test_source_name_keys_are_rewritten(system_2s2l):
    """Given per-source params addressed by source star name (lens.SourceB.t_0),
    when the Lens component initializes, then the keys are rewritten to the
    canonical slot-index form (lens.1.t_0)."""
    system, _ = system_2s2l
    up = system.config_manager.user_params
    assert "lens.1.t_0" in up
    assert "lens.SourceB.t_0" not in up
    assert float(up["lens.1.t_0"]["initval"]) == pytest.approx(2457552.517)


def test_per_source_shapes_and_initvals(system_2s2l):
    """Given two sources, when parameters are materialized, then the per-source
    vectors have shape (2,) with each source's own initval."""
    system, _ = system_2s2l
    lens = system.lens
    assert lens.n_sources == 2
    assert lens.t_0.shape == (2,)
    assert lens.u_0.shape == (2,)
    assert lens.rho.shape == (2,)
    np.testing.assert_allclose(
        lens.t_0.initval, [2457551.038, 2457552.517], rtol=1e-9)
    np.testing.assert_allclose(lens.u_0.initval, [0.059, 0.135], rtol=1e-9)
    np.testing.assert_allclose(lens.rho.initval, [0.000451, 0.001293], rtol=1e-6)


def test_source_map_covers_all_sources(system_2s2l):
    """Given sources star.2 and star.3, when maps are built, then source_map
    has one entry per source body."""
    system, _ = system_2s2l
    np.testing.assert_array_equal(system.lens.source_map, [2, 3])


def test_total_mass_convention(system_2s2l):
    """Given a binary lens with q=1.188, when the derived chain is resolved,
    then theta_E**2 = KAPPA * (M1 + M2) * pi_rel (total-mass convention) and
    t_E = theta_E / (mu_rel / 365.25) reproduces the user's t_E for both
    sources."""
    import pytensor
    from exozippy.constants import KAPPA
    system, model = system_2s2l
    with model:
        f = pytensor.function(
            model.free_RVs,
            [system.star.mass.value, system.lens.mlens_total.value,
             system.lens.theta_E.value, system.lens.pi_rel.value,
             system.lens.t_E.value],
            on_unused_input="ignore")
        ip = model.initial_point()
        zeros = [np.zeros_like(ip[v.name]).astype(float) for v in model.free_RVs]
        mass, m_tot, theta_E, pi_rel, t_E = [np.atleast_1d(x) for x in f(*zeros)]

    m1, m2 = mass[0], mass[1]
    np.testing.assert_allclose(m_tot[0], m1 + m2, rtol=1e-6)
    np.testing.assert_allclose(m2 / m1, 1.188, rtol=0.01)
    np.testing.assert_allclose(theta_E**2, KAPPA * m_tot[0] * pi_rel, rtol=1e-5)
    np.testing.assert_allclose(t_E, [28.931, 28.931], rtol=0.02)


def test_finite_logp_at_start(system_2s2l):
    """Given the seeded 2S2L system, when logp is evaluated at the starting
    point, then it is finite."""
    system, model = system_2s2l
    lp = model.compile_logp()(system.get_raw_start(model))
    assert np.isfinite(lp)


def test_magnification_per_source_differs(system_2s2l):
    """Given two sources with different trajectories, when the magnification
    is evaluated at SourceA's peak time, then the two sources' magnifications
    differ (each source follows its own trajectory)."""
    import pytensor
    system, model = system_2s2l
    t = np.array([2457551.038])
    obs = np.zeros((1, 3))
    with model:
        A0 = system.lens.get_magnification_op(t, obs, system, index=0)
        A1 = system.lens.get_magnification_op(t, obs, system, index=1)
        f = pytensor.function(model.free_RVs, [A0, A1], on_unused_input="ignore")
        ip = model.initial_point()
        zeros = [np.zeros_like(ip[v.name]).astype(float) for v in model.free_RVs]
        a0, a1 = f(*zeros)
    assert np.isfinite(a0).all() and np.isfinite(a1).all()
    assert abs(float(a0[0]) - float(a1[0])) > 1e-3


def test_single_source_backward_compat():
    """Given a legacy 1S1L config (lens_ndx/source_ndx shorthand), when the
    system is prepared and built, then scalar-per-event shapes and a finite
    logp are preserved."""
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}],
    }
    user_params = {
        "lens.Lens.t_0": {"initval": 2460025.0, "init_scale": 0.1},
        "lens.Lens.u_0": {"initval": 0.3, "init_scale": 0.01},
        "lens.Lens.t_E": {"initval": 30.0, "init_scale": 1.0},
        "star.Lens.ra": {"initval": 264.0, "sigma": 0},
        "star.Lens.dec": {"initval": -27.0, "sigma": 0},
        "star.Source.ra": {"initval": 264.0, "sigma": 0},
        "star.Source.dec": {"initval": -27.0, "sigma": 0},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()

    assert system.lens.n_sources == 1
    assert system.lens.t_0.shape == (1,)
    np.testing.assert_array_equal(system.lens.source_map, [1])
    lp = model.compile_logp()(system.get_raw_start(model))
    assert np.isfinite(lp)
