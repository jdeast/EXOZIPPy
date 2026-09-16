"""Tests for the NSNL (N-source, N-lens) pathway.

The 2S2L configuration mirrors examples/ob161003 (OGLE-2016-BLG-1003,
Jung et al. 2017): two source stars sharing a binary star lens.

Post-split (8.6.17) that topology is spelled across four components: one
`mulensevent:` block for the event-level quantities, one `lens:` entry per
lens BODY (element 0 = the primary), one `source:` entry per source body,
and the star blocks for the bodies themselves.  Two consequences run through
every test below:

  * the per-source trajectory (t_0, u_0, rho) is on `source`, while t_E,
    theta_E, pi_rel and mu_rel are EVENT-LEVEL SCALARS of shape (1,) --
    ruling R1 (co-moving systems only) says one encounter has exactly one
    relative proper motion, so those are no longer stored per source;
  * the companion geometry (s, alpha, q) stays on `lens` but shifts by one
    element: companion j is lens element j+1, so this binary's geometry is
    lens.1.<param> (equivalently lens.LensB.<param>).
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
        # The event instance is NAMED so that the inject-back test below has
        # a name form to distinguish from the index form (see
        # test_derived_solutions_are_injected_under_the_index_key).
        "mulensevent": [{"name": "Event", "finite_source": True}],
        # One entry per BODY; the primary lens is first.
        "lens": [{"body": "star.Lens"}, {"body": "star.LensB"}],
        "source": [{"body": "star.SourceA"}, {"body": "star.SourceB"}],
    }


def _params_2s2l():
    coords = {"initval": 264.10513, "sigma": 0}
    coords_dec = {"initval": -27.188861, "sigma": 0}
    p = {
        # Per source body ...
        "source.SourceA.t_0": {"initval": 2457551.038},
        "source.SourceB.t_0": {"initval": 2457552.517},
        "source.SourceA.u_0": {"initval": 0.059},
        "source.SourceB.u_0": {"initval": 0.135},
        "source.SourceA.rho": {"initval": 0.000451},
        "source.SourceB.rho": {"initval": 0.001293},
        # ... one t_E for the one encounter (pre-split this file seeded the
        # SAME 28.931 d on both sources' per-source copies) ...
        "mulensevent.t_E": {"initval": 28.931},
        # ... and the geometry on the COMPANION's lens entry, element 1.
        "lens.LensB.s": {"initval": 1.033},
        "lens.LensB.alpha": {"initval": 131.757},
        "lens.LensB.q": {"initval": 1.188},
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
    """Given per-source params addressed by source star name
    (source.SourceB.t_0), when the Source component initializes, then the keys
    are rewritten to the canonical slot-index form (source.1.t_0).

    Source instances are NAMED AFTER THEIR BODY STAR now (the Mann/Torres
    idiom, derive_body_names), so the name the user writes is the source
    body's own name and it folds onto that body's slot."""
    system, _ = system_2s2l
    up = system.config_manager.user_params
    assert "source.1.t_0" in up
    assert "source.SourceB.t_0" not in up
    assert float(up["source.1.t_0"]["initval"]) == pytest.approx(2457552.517)


def test_per_source_shapes_and_initvals(system_2s2l):
    """Given two sources, when parameters are materialized, then the per-source
    vectors have shape (2,) with each source's own initval -- and the
    event-level t_E is a single scalar rather than a second copy per source
    (ruling R1)."""
    system, _ = system_2s2l
    source = system.source
    event = system.mulensevent
    assert event.n_sources == 2
    assert source.n_elements == 2
    assert source.t_0.shape == (2,)
    assert source.u_0.shape == (2,)
    assert source.rho.shape == (2,)
    np.testing.assert_allclose(
        source.t_0.initval, [2457551.038, 2457552.517], rtol=1e-9
    )
    np.testing.assert_allclose(source.u_0.initval, [0.059, 0.135], rtol=1e-9)
    np.testing.assert_allclose(
        source.rho.initval, [0.000451, 0.001293], rtol=1e-6
    )
    assert event.t_E.shape == (1,)


def test_source_map_covers_all_sources(system_2s2l):
    """Given sources star.2 and star.3, when maps are built, then the SOURCE
    component's star_map has one entry per source body.

    The event's own `source_map` is a different, deliberately length-1 map:
    under R1 the source SYSTEM's barycentric kinematics are represented by
    body 0 (design 11.3), so mulensevent.source_map names the PRIMARY source
    star only.  Both are asserted here so the two are not confused again."""
    system, _ = system_2s2l
    np.testing.assert_array_equal(system.source.star_map, [2, 3])
    np.testing.assert_array_equal(system.mulensevent.source_map, [2])


def test_total_mass_convention(system_2s2l):
    """Given a binary lens with q=1.188, when the derived chain is resolved,
    then theta_E**2 = KAPPA * (M1 + M2) * pi_rel (total-mass convention) and
    t_E = theta_E / (mu_rel / 365.25) reproduces the user's t_E.

    mlens_total is mulensevent.mlens_total: the old lens.0.mlens_total path
    names the MASKED PRIMARY body now, not the event."""
    import pytensor

    from exozippy.constants import KAPPA

    system, model = system_2s2l
    event = system.mulensevent
    with model:
        f = pytensor.function(
            model.free_RVs,
            [
                system.star.mass.value,
                event.mlens_total.value,
                event.theta_E.value,
                event.pi_rel.value,
                event.t_E.value,
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
    # ONE t_E for the one encounter, on the seeded value.
    np.testing.assert_allclose(t_E, [28.931], rtol=0.02)


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
    event = system.mulensevent
    with model:
        A0 = event.get_magnification_op(t, obs, system, index=0)
        A1 = event.get_magnification_op(t, obs, system, index=1)
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


def test_single_source_point_lens():
    """Given the simplest 1S1L config -- one lens body, one source body -- when
    the system is prepared and built, then the one-source shapes and a finite
    logp are preserved.

    This was `test_single_source_backward_compat`, and the config it built
    used the `lens_ndx`/`source_ndx` shorthand of the single pre-split lens
    block.  That shorthand is a HARD BREAK now (ruling R3): body_entries
    refuses it by name and prints the new three-block shape, which
    tests/test_mulens_body_config.py pins.  So there is no backward
    compatibility left to assert -- what survives is the substance, that the
    minimal topology still builds and that a one-source event's per-source
    vectors are length 1."""
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [{"finite_source": False}],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
    }
    user_params = {
        "source.Source.t_0": {"initval": 2460025.0},
        "source.Source.u_0": {"initval": 0.3},
        "mulensevent.t_E": {"initval": 30.0},
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

    assert system.mulensevent.n_sources == 1
    assert system.source.t_0.shape == (1,)
    np.testing.assert_array_equal(system.source.star_map, [1])
    np.testing.assert_array_equal(system.mulensevent.source_map, [1])
    lp = model.compile_logp()(system.get_raw_start(model))
    assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# The relaxation engine's answer must reach the element it solved.
#
# The engine solves at index-form paths (mulensevent.0.theta_E).
# finalize_user_params used to file a NEW entry under the CONFIG INSTANCE
# NAME instead, and index 0 is the one index an instance name collides with,
# so element 0's answer became <comp>.<name>.theta_E.  Pre-split that was
# fatal on exactly this configuration: the single lens instance was named
# "Lens" while its per-source vectors borrowed the SOURCE stars' names, so
# lens.Lens.theta_E matched none of element 0's three resolve() keys and the
# solved value was silently dropped -- with no initval in
# mulensing/defaults.yaml (theta_E is derived), apply_value allocated a
# NaN-filled vector and wrote only element 1.
#
# The borrowed names are gone (every post-split component names its elements
# after its own entries), so that particular collision cannot recur.  The
# WRITER'S RULE is what these tests still pin, because everything downstream
# of the engine reads the index form: get_conversion_factor,
# propagated_scales, scale_hints and the engine's own symbol paths.
# ---------------------------------------------------------------------------


def test_derived_solutions_are_injected_under_the_index_key(system_2s2l):
    """Given an event instance named 'Event', when the relaxation engine's
    solution is injected back, then the new entries use the INDEX form
    (mulensevent.0.<param>), not the config instance's own name."""
    system, _ = system_2s2l
    up = system.config_manager.user_params

    for param in ("theta_E", "pi_rel", "mu_rel_mag"):
        assert f"mulensevent.0.{param}" in up, (
            f"mulensevent.0.{param} was not injected"
        )
        assert f"mulensevent.Event.{param}" not in up, (
            f"mulensevent.Event.{param} is the config-instance-name form; "
            f"the index form is the internal spelling every downstream "
            f"reader uses"
        )


def test_event_level_derived_initvals_are_finite(system_2s2l):
    """Given a 2-source event, when the derived event-level chain is resolved,
    then it carries the engine's value -- no NaN hole where the solution got
    no readable entry.

    Was `test_per_source_derived_initvals_are_finite`: these four are
    event-level scalars now (shape (1,)), so "every element" is one element.
    The VALUES are unchanged from the per-source era -- 0.8392544170490658
    mas and pi_rel = 0.125 -- which is the statement that collapsing the
    duplicate per-source copies onto one did not move the start."""
    system, _ = system_2s2l
    event = system.mulensevent

    for param in ("theta_E", "pi_rel", "mu_rel_mag", "mu_ra_rel"):
        initval = np.atleast_1d(getattr(event, param).initval)
        assert initval.size == 1
        assert np.all(np.isfinite(initval)), (
            f"mulensevent.{param}.initval = {initval} has a non-finite element"
        )

    np.testing.assert_allclose(
        event.theta_E.initval, [0.8392544170490658], rtol=1e-9
    )
    np.testing.assert_allclose(event.pi_rel.initval, [0.125], rtol=1e-9)


def test_no_active_element_has_a_non_finite_initval(system_2s2l):
    """Given the built 2S2L system, when every Parameter is inspected, then no
    element that is a parameter of its instance carries a non-finite initval
    -- an initval is either a value or absent (None), never NaN on some
    elements and a number on others.

    INACTIVE elements are exempt, and the exemption is the post-split shape
    rather than a loosening: `lens` has one element per lens BODY, and the
    geometry (s, alpha, q) is not a parameter of ELEMENT 0, the primary --
    manifest role 4, held at the manifest's inactive_value, given no prior
    and reported nowhere.  The relaxation engine leaves those slots NaN
    because nothing ever solves them, and the build overwrites them with the
    bookkeeping pin, so a NaN there is not a start value at all (the same
    distinction review 1.6.5 turned on for the derived q).  What must still
    hold, and does, is that every element the sampler or a report can see has
    a number."""
    system, _ = system_2s2l

    offenders = []
    n_active_checked = 0
    exempted = []
    for par in system.get_all_parameters():
        if par.initval is None:
            continue
        try:
            arr = np.asarray(par.initval, dtype=float)
        except (TypeError, ValueError):
            continue  # symbolic (linked) start; not a numeric vector
        if not arr.size:
            continue
        flat = arr.ravel()
        active = [i for i in range(flat.size) if par.element_is_active(int(i))]
        n_active_checked += len(active)
        if np.all(np.isfinite(flat)):
            continue
        bad, skipped = [], []
        for i in np.flatnonzero(~np.isfinite(flat)):
            (bad if par.element_is_active(int(i)) else skipped).append(int(i))
        if bad:
            offenders.append(f"{par.label} = {arr} (active elements {bad})")
        if skipped:
            exempted.append(f"{par.label}{skipped}")

    assert offenders == []

    # The exemption above is only safe while `element_is_active` still
    # discriminates.  If it ever answered False everywhere -- a plausible
    # breakage, since the masked primary is what it exists to describe --
    # every offender would be exempted and this test would pass while
    # inspecting nothing.  So pin both halves: active elements WERE examined,
    # and the exemption really did fire on this 2S2L fixture (the masked
    # primary's s/alpha/q), which is the case it was written for.
    assert n_active_checked > 0, (
        "no active element was examined: element_is_active answered False "
        "for every element, so the exemption swallowed the whole test"
    )
    assert exempted, (
        "nothing was exempted, so the masked lens primary no longer carries "
        "a non-finite initval -- welcome, but then this test's exemption is "
        "dead code and the stricter 'every element is finite' form should "
        "come back"
    )


def test_the_event_rate_prior_is_counted_once_per_encounter(system_2s2l):
    """Given a 2S event, when build_likelihood adds the event-rate prior,
    then there is ONE such term and it is log(mu_rel_geo) + log(theta_E) on
    the event's own scalars.

    Review 8.6.18.  Gamma propto mu_rel * theta_E (Batista+2011) is the
    sky-sweep rate of ONE lens past ONE source system, and both operands used
    to be stored per source -- so the old `pt.sum` carried the selection
    correction SQUARED on a 2S event.  Live on examples/ob161003, and not a
    small correction: it is the whole event-rate term again, and that term is
    exactly what tilts the lens mass and distance against the galactic prior.

    THE 8.6.17 SPLIT REMOVED THE TEETH THIS TEST USED TO HAVE, and saying so
    is more useful than pretending otherwise.  The original asserted that the
    term equalled source 0's copy and NOT the sum, which could only fail while
    mu_rel and theta_E were per-source vectors whose elements differed.  Under
    ruling R1 they are event-level scalars of shape (1,), so a sum and an
    element-0 read are the same number by construction -- the shape finally
    matches the physics, and the defect is unrepresentable rather than merely
    absent.  What is left to pin is therefore the shape claim itself (size 1,
    asserted below, which is what makes the sum safe) together with the
    term's value; if anyone ever restores a per-source mu_rel or theta_E, the
    size assertions here fail and this term has to be reconsidered.

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

    event = system.mulensevent
    vv = list(model.value_vars)
    fn = pytensor.function(
        vv,
        [
            terms[0],
            event.mu_rel_geo_mag.value,
            event.theta_E.value,
        ],
        on_unused_input="ignore",
    )
    start = system.get_raw_start(model)
    got, mu, th = fn(*[start[v.name] for v in vv])

    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    th = np.atleast_1d(np.asarray(th, dtype=float))
    # ONE encounter, one weight: both operands are event-level scalars even
    # though the event has TWO sources.
    assert mu.size == 1 and th.size == 1, (mu, th)

    assert float(got) == pytest.approx(
        float(np.log(mu[0]) + np.log(th[0])), rel=1e-9
    )


def test_the_beta_bound_is_counted_once_per_lens_orbit(system_2s2l):
    """Given a 2S event, when a beta bound exists, then it too is a scalar.

    The same shape as the event-rate term (8.6.18): beta =
    E_kin,perp/E_pot,perp is a property of the LENS's orbit, and it used to be
    declared shape=(n_sources,) only because it is derived through theta_E,
    which was stored per source.  Bounding it per source bounded one physical
    quantity N times.  Post-split beta is a PER-COMPANION vector on the lens
    (with the primary masked), and the bound reads the companion's element,
    so the term is a scalar for structural reasons; this pins that.

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
