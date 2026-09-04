"""Tests for the NSNL (N-source, N-lens) pathway.

The 2S2L configuration mirrors examples/ob161003 (OGLE-2016-BLG-1003,
Jung et al. 2017): two source stars sharing a binary star lens.
"""

import numpy as np
import pytest

from exozippy.config import ConfigManager
from exozippy.system import System


def _config_2s2l():
    return {
        "star": [
            {"name": "Lens"},
            {"name": "LensB"},
            {"name": "SourceA"},
            {"name": "SourceB"},
        ],
        "lens": [
            {
                "name": "Lens",
                "lenses": ["star.0", "star.1"],
                "sources": ["star.2", "star.3"],
                "finite_source": True,
            }
        ],
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
        lens.t_0.initval, [2457551.038, 2457552.517], rtol=1e-9
    )
    np.testing.assert_allclose(lens.u_0.initval, [0.059, 0.135], rtol=1e-9)
    np.testing.assert_allclose(
        lens.rho.initval, [0.000451, 0.001293], rtol=1e-6
    )


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
            [
                system.star.mass.value,
                system.lens.mlens_total.value,
                system.lens.theta_E.value,
                system.lens.pi_rel.value,
                system.lens.t_E.value,
            ],
            on_unused_input="ignore",
        )
        ip = model.initial_point()
        zeros = [
            np.zeros_like(ip[v.name]).astype(float) for v in model.free_RVs
        ]
        mass, m_tot, theta_E, pi_rel, t_E = [
            np.atleast_1d(x) for x in f(*zeros)
        ]

    m1, m2 = mass[0], mass[1]
    np.testing.assert_allclose(m_tot[0], m1 + m2, rtol=1e-6)
    np.testing.assert_allclose(m2 / m1, 1.188, rtol=0.01)
    np.testing.assert_allclose(
        theta_E**2, KAPPA * m_tot[0] * pi_rel, rtol=1e-5
    )
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
        f = pytensor.function(
            model.free_RVs, [A0, A1], on_unused_input="ignore"
        )
        ip = model.initial_point()
        zeros = [
            np.zeros_like(ip[v.name]).astype(float) for v in model.free_RVs
        ]
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


# ---------------------------------------------------------------------------
# The relaxation engine's answer must reach source slot 0.
#
# The engine solves the per-source chain at index-form paths (lens.0.theta_E,
# lens.1.theta_E).  finalize_user_params used to file a NEW entry under the
# CONFIG INSTANCE name instead -- and index 0 is the one index a lens instance
# name collides with, so lens.0.theta_E became lens.Lens.theta_E.  resolve()
# addresses element j of a per-source vector as lens.j.<param> or
# lens.<SOURCE STAR NAME>.<param> ("SourceA"), never lens.Lens.<param>, so the
# solved value for slot 0 was dropped: with no initval in
# mulensing/defaults.yaml (theta_E is derived) apply_value allocated a
# NaN-filled vector and wrote only element 1.
# ---------------------------------------------------------------------------


def test_derived_solutions_are_injected_under_the_index_key(system_2s2l):
    """Given a lens event named 'Lens' whose per-source vectors are named for
    the SOURCE stars, when the relaxation engine's solution is injected back,
    then slot 0's entry uses the index form that resolve() can read, not the
    lens instance's own name."""
    system, _ = system_2s2l
    up = system.config_manager.user_params

    for param in ("theta_E", "pi_rel", "mu_rel_mag"):
        assert f"lens.0.{param}" in up, f"lens.0.{param} was not injected"
        assert f"lens.Lens.{param}" not in up, (
            f"lens.Lens.{param} is unreadable: resolve() addresses source "
            f"slot 0 as lens.0.{param} or lens.SourceA.{param}"
        )


def test_per_source_derived_initvals_are_finite(system_2s2l):
    """Given a 2-source event, when the derived per-source chain is resolved,
    then EVERY element carries the engine's value -- no NaN hole where a slot
    got no readable entry."""
    system, _ = system_2s2l
    lens = system.lens

    for param in ("theta_E", "pi_rel", "mu_rel_mag", "mu_ra_rel"):
        initval = np.atleast_1d(getattr(lens, param).initval)
        assert np.all(np.isfinite(initval)), (
            f"lens.{param}.initval = {initval} has a non-finite element"
        )

    # Both sources share one lens, so the shared chain is the same for both.
    np.testing.assert_allclose(
        lens.theta_E.initval, [0.8392544170490658] * 2, rtol=1e-9
    )
    np.testing.assert_allclose(lens.pi_rel.initval, [0.125] * 2, rtol=1e-9)


def test_no_parameter_has_a_non_finite_initval(system_2s2l):
    """Given the built 2S2L system, when every Parameter is inspected, then
    none carries a non-finite initval -- an initval is either a value or
    absent (None), never NaN on some elements and a number on others."""
    system, _ = system_2s2l

    offenders = []
    for par in system.get_all_parameters():
        if par.initval is None:
            continue
        try:
            arr = np.asarray(par.initval, dtype=float)
        except (TypeError, ValueError):
            continue  # symbolic (linked) start; not a numeric vector
        if arr.size and not np.all(np.isfinite(arr)):
            offenders.append(f"{par.label} = {arr}")

    assert offenders == []


def test_the_event_rate_prior_is_counted_once_per_encounter(system_2s2l):
    """Given a 2S event, when build_likelihood adds the event-rate prior,
    then the term equals ONE copy of log(mu_rel) + log(theta_E), not the sum
    over sources.

    Review 8.6.18.  Gamma propto mu_rel * theta_E (Batista+2011) is the
    sky-sweep rate of ONE lens past ONE source system, and both operands are
    stored per source -- so the old `pt.sum` carried the selection correction
    SQUARED on a 2S event.  Live on examples/ob161003, and not a small
    correction: it is the whole event-rate term again, and that term is
    exactly what tilts the lens mass and distance against the galactic prior.

    ASSERTING THE SHAPE INSTEAD WOULD BE VACUOUS, and the first draft of this
    test was: `pt.sum(...)` is already ndim 0, so "the potential is a scalar"
    passes before AND after the fix.  The VALUE is the distinguishing
    statement, and the teeth below make the difference observable -- the two
    sources' terms must differ, or summing and taking element 0 would agree
    and this test could not fail.

    Evaluated at `system.get_raw_start(model)`, NOT at `model.initial_point()`
    and NOT through `Parameter.value` on its own: the initial point is keyed
    by raw variables that are not in a sub-graph, and reading `.value`
    outside a compiled function draws from the prior rather than giving the
    start.  Both are documented traps and both produced wrong numbers while
    this was being written.
    """
    import pytensor

    system, model = system_2s2l
    terms = [
        p
        for p in model.potentials
        if "event_rate_prior" in str(getattr(p, "name", "") or "")
    ]
    assert len(terms) == 1, [str(p.name) for p in model.potentials]

    vv = list(model.value_vars)
    fn = pytensor.function(
        vv,
        [
            terms[0],
            system.lens.mu_rel_geo_mag.value,
            system.lens.theta_E.value,
        ],
        on_unused_input="ignore",
    )
    start = system.get_raw_start(model)
    got, mu, th = fn(*[start[v.name] for v in vv])

    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    th = np.atleast_1d(np.asarray(th, dtype=float))
    assert mu.size == 2 and th.size == 2
    per_source = np.log(mu) + np.log(th)

    # TEETH: if the two sources' terms were equal, one copy and the sum
    # would be indistinguishable and this test could not fail.
    assert abs(per_source[1] - per_source[0]) > 1e-3, per_source

    assert float(got) == pytest.approx(per_source[0], rel=1e-9)
    assert float(got) != pytest.approx(per_source.sum(), rel=1e-6)


def test_the_beta_bound_is_counted_once_per_lens_orbit(system_2s2l):
    """Given a 2S event, when a beta bound exists, then it too is a scalar.

    The same shape as the event-rate term (8.6.18): beta =
    E_kin,perp/E_pot,perp is a property of the LENS's orbit and is declared
    shape=(n_sources,) only because it is derived through theta_E, which is
    stored per source.  Bounding it per source bounded one physical quantity
    N times.

    Skipped rather than asserted-absent when the model has no such term: the
    2S2L fixture has no orbital motion, so this pins the SHAPE for whenever
    the two features are combined -- which no shipped example does today,
    which is precisely why the defect was latent.
    """
    _, model = system_2s2l
    terms = [
        p
        for p in model.potentials
        if "beta" in str(getattr(p, "name", "") or "")
    ]
    if not terms:
        pytest.skip("no orbital motion in the 2S2L fixture, so no beta bound")
    for p in terms:
        assert p.ndim == 0, str(p.name)
