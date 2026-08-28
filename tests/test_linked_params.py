"""
Tests for user-defined parameter links (linking.py).

A params-file entry may set any numeric field (initval, mu, sigma, lower,
upper, init_scale) to a string expression referencing other parameters:

  - hard link:   star.A.age: {initval: star.B.age, sigma: 0}
  - soft link:   star.A.age: {initval: star.B.age, sigma: 1}
  - bound link:  star.A.av:  {lower: star.B.av}
  - algebra:     orbit.b.omega: {initval: "orbit.c.omega + math.pi", sigma: 0}

Unit convention: referenced parameters contribute their values in their own
user units; the expression result is interpreted in the target's user unit.
"""

import copy

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.components.star.star import Star
from exozippy.config import ConfigManager
from exozippy.linking import _SUPPORTED_FUNCS, parse_link_expression

DEG2RAD = np.pi / 180.0


def _build_star_param(user_params, param_names, config=None):
    """Build star parameters through the full config -> model pipeline.

    Returns (config_manager, star, model).
    """
    config = config or {"star": [{"name": "A"}, {"name": "B"}]}
    cm = ConfigManager(user_params, system_config=config)
    cm.finalize_user_params()

    star = Star(config["star"], cm)
    star.manifest = {p: None for p in param_names}

    with pm.Model() as model:
        for p in param_names:
            star.add_parameter(model, p, system=None)

    return cm, star, model


def _eval(model, tensor, point):
    """Evaluate a graph tensor at a raw point (same pattern as
    System.get_internal_point: free RVs are fed directly as inputs)."""
    import pytensor

    fn = pytensor.function(model.free_RVs, tensor, on_unused_input="ignore")
    return np.asarray(fn(*[point[rv.name] for rv in model.free_RVs]))


# ----------------------------------------------------------------------
# Hard link: sigma = 0
# ----------------------------------------------------------------------


def test_hard_link_fixes_parameter_to_other_parameter():
    """
    Given star.A.age = {initval: star.B.age, sigma: 0},
    When the model is built and star.B.age moves,
    Then star.A.age is never sampled and always equals star.B.age exactly.
    """
    # ARRANGE / ACT
    user_params = {"star.A.age": {"initval": "star.B.age", "sigma": 0}}
    cm, star, model = _build_star_param(user_params, ["age"])

    # ASSERT: A (index 0) is not sampled; only B's raw element exists
    assert list(star.age.is_sampled) == [False, True]
    point = model.initial_point()
    assert point["star.age_raw"].shape == (1,)

    # ASSERT: A tracks B exactly for arbitrary raw values
    for raw in (-3.0, 0.0, 1.7):
        point["star.age_raw"] = np.array([raw])
        ages = _eval(model, star.age.value, point)
        assert np.isclose(ages[0], ages[1], rtol=1e-12)


def test_hard_link_relaxation_engine_solves_snapshot_initval():
    """
    Given star.A.logmass hard-linked to star.B.logmass = 0.5,
    When finalize_user_params runs,
    Then the engine seeds A's initval from B AND derives dependent
    parameters (mass) through the physics relations using the linked value.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {
        "star.B.logmass": {"initval": 0.5},
        "star.A.logmass": {"initval": "star.B.logmass", "sigma": 0},
    }
    cm = ConfigManager(user_params, system_config=config)

    # ACT
    cm.finalize_user_params()

    # ASSERT: A's initval snapshot equals B's value
    assert np.isclose(
        cm.user_params["star.0.logmass"]["initval"], 0.5, rtol=1e-3
    )
    # ASSERT: physics relations saw the linked value (mass = 10**logmass)
    mass_entry = cm.user_params.get("star.A.mass") or cm.user_params.get(
        "star.0.mass"
    )
    assert mass_entry is not None
    assert np.isclose(mass_entry["initval"], 10**0.5, rtol=1e-3)


# ----------------------------------------------------------------------
# Soft link: sigma > 0
# ----------------------------------------------------------------------


def test_soft_link_samples_both_and_penalizes_difference():
    """
    Given star.A.age = {initval: star.B.age, sigma: 1},
    When the model is built,
    Then both ages are sampled and a Gaussian potential penalizes
    (A - B) / sigma dynamically.
    """
    # ARRANGE / ACT
    user_params = {"star.A.age": {"initval": "star.B.age", "sigma": 1.0}}
    cm, star, model = _build_star_param(user_params, ["age"])

    # ASSERT: both elements sampled
    assert list(star.age.is_sampled) == [True, True]
    point = model.initial_point()
    assert point["star.age_raw"].shape == (2,)

    pot = next(p for p in model.potentials if p.name == "link_mu.star.age.0")

    # ASSERT: potential equals -0.5*((A - B)/sigma)^2 at arbitrary points
    for raw in ([0.0, 0.0], [1.0, -1.0], [-2.0, 0.5]):
        point["star.age_raw"] = np.array(raw)
        ages = _eval(model, star.age.value, point)
        pot_val = _eval(model, pot, point)
        expected = -0.5 * (ages[0] - ages[1]) ** 2
        assert np.isclose(pot_val, expected, rtol=1e-10)


def test_soft_link_does_not_double_count_static_gaussian_prior():
    """
    Given a soft link on element A only,
    When the model is built,
    Then no static gaussian_prior potential is applied to element A
    (the linked potential replaces it, not stacks with it).
    """
    # ARRANGE / ACT
    user_params = {"star.A.age": {"initval": "star.B.age", "sigma": 1.0}}
    cm, star, model = _build_star_param(user_params, ["age"])

    # ASSERT: the only Gaussian-type potential on star.age is the link
    static_names = [
        p.name for p in model.potentials if p.name == "gaussian_prior.star.age"
    ]
    assert static_names == []


# ----------------------------------------------------------------------
# Dynamic bound link
# ----------------------------------------------------------------------


def test_lower_bound_link_enforces_ordering_constraint():
    """
    Given star.A.av = {lower: star.B.av},
    When the model is built,
    Then A's extinction can never fall below B's, for any raw values,
    and no span-normalization potential is added (the logit
    reparameterization is already span-normalized -- see review 1.5).
    """
    # ARRANGE / ACT
    user_params = {
        "star.A.av": {"initval": 0.7, "lower": "star.B.av"},
        "star.B.av": {"initval": 0.5},
    }
    cm, star, model = _build_star_param(user_params, ["av"])

    # ASSERT: both sampled, no -log(span) potential
    assert list(star.av.is_sampled) == [True, True]
    assert not any(p.name.startswith("link_span") for p in model.potentials)

    # ASSERT: constraint holds across the raw space, including extremes
    point = model.initial_point()
    for raw_a in (-8.0, -2.0, 0.0, 2.0, 8.0):
        for raw_b in (-5.0, 0.0, 5.0):
            point["star.av_raw"] = np.array([raw_a, raw_b])
            av = _eval(model, star.av.value, point)
            assert av[0] >= av[1] - 1e-12
            assert av[0] <= 100.0 + 1e-9


def _raw_from_q(param, i, q):
    """Raw coordinate at which element i's sigmoid coordinate equals q."""
    tf = param._raw_transform
    lq = np.log(q / (1.0 - q))
    return (lq - tf["logit_q_inits"][i]) / tf["init_scale_logits"][i]


def test_dynamic_bound_link_leaves_the_bound_source_prior_unbiased():
    """
    Given star.A.av = {lower: star.B.av} and no likelihood,
    When the bounded element A is marginalized out by quadrature,
    Then the implied marginal density of the bound SOURCE av_B is flat, i.e.
    exactly its own U(0, 100) prior -- carrying a dynamic bound costs av_B
    nothing.  A -log(span) potential would instead make it proportional to
    1/(100 - av_B), a factor ~50 preference for the wall across the spans
    probed here (review 1.5).
    """
    # ARRANGE
    user_params = {
        "star.A.av": {"initval": 0.7, "lower": "star.B.av"},
        "star.B.av": {"initval": 0.5},
    }
    cm, star, model = _build_star_param(user_params, ["av"])
    logp_c = model.compile_logp()
    point = model.initial_point()
    tf = star.av._raw_transform
    s_a = tf["init_scale_logits"][0]

    # Integrate A out in its own sigmoid coordinate q rather than in raw:
    # the measure becomes dq / (s_a * q * (1-q)) and the integrand is smooth
    # and bounded over the whole interval.
    q_a = np.linspace(1e-6, 1.0 - 1e-6, 201)
    raw_a = _raw_from_q(star.av, 0, q_a)

    # ACT: p(av_B) at spans from 99.5% down to 50% of the static interval.
    # Dividing the raw-space marginal by B's own sigmoid Jacobian factor
    # q_b*(1-q_b) (the rest of dav_B/draw_b is the constant span*s_b)
    # converts it to a density in av_B.
    q_bs = (0.005, 0.05, 0.2, 0.5)
    p_avb = []
    for q_b in q_bs:
        raw_b = _raw_from_q(star.av, 1, q_b)

        def _logp(ra):
            point["star.av_raw"] = np.array([ra, raw_b])
            return float(logp_c(point))

        dens = np.exp(np.array([_logp(ra) for ra in raw_a]))
        marginal = np.trapezoid(dens / (s_a * q_a * (1.0 - q_a)), q_a)
        p_avb.append(marginal / (q_b * (1.0 - q_b)))

    # ASSERT: flat in av_B (av_B = 100*q_b here, so 0.5 -> 50 mag)
    p_avb = np.array(p_avb)
    assert np.allclose(p_avb, p_avb[0], rtol=1e-6), (
        f"marginal p(av_B) varies with the span: "
        f"{dict(zip([100.0 * q for q in q_bs], p_avb))}"
    )


def test_log_normal_mass_matches_scipy_on_both_tails():
    """
    Given standardized truncation bounds on the upper tail, the lower tail
    and straddling zero,
    When _log_normal_mass evaluates log(Phi(beta) - Phi(alpha)),
    Then it matches scipy to full double precision -- including the tails,
    where a plain difference of Phi CDFs cancels away every digit.
    """
    # ARRANGE
    import pytensor
    from scipy.stats import norm

    from exozippy.components.parameter import _log_normal_mass

    a_t, b_t = pt.dscalar("a"), pt.dscalar("b")
    fn = pytensor.function([a_t, b_t], _log_normal_mass(a_t, b_t))

    cases = [
        (-1.0, 2.0),  # straddling
        (0.5, 3.0),  # upper tail
        (-3.0, -0.5),  # lower tail
        (6.0, 9.0),  # deep upper tail: Phi difference is 1e-9 of ~1
        (-9.0, -6.0),  # deep lower tail
    ]

    # ACT / ASSERT.  The reference itself has to pick its tail: sf on the
    # upper side, cdf on the lower.  (norm.sf(-9) - norm.sf(-6) is the naive
    # spelling and is wrong in its 9th digit, which is the whole point of the
    # branch selection under test.)
    for a, b in cases:
        expected = (
            np.log(norm.sf(a) - norm.sf(b))
            if a >= 0.0
            else np.log(norm.cdf(b) - norm.cdf(a))
        )
        assert fn(a, b) == pytest.approx(expected, rel=1e-12), (a, b)


def test_dynamic_bound_plus_sigma_leaves_the_bound_source_unbiased():
    """
    Given star.A.av = {lower: star.B.av} AND a Gaussian prior on av_A, so the
    conditional prior on av_A is a TRUNCATED normal whose mass depends on
    av_B,
    When av_A is marginalized out by quadrature,
    Then the implied marginal density of av_B is still flat.

    Review 1.2.4: sections A/A2 add an unnormalized Gaussian on top of the
    reparameterization's exact U(lo, up), so without the A3 correction the
    marginal picks up the truncated mass Z(av_B)/span(av_B) -- a factor of
    ~2.7 across the av_B values probed here, all of it spurious.
    """
    # ARRANGE
    from scipy.stats import norm

    mu, sigma = 1.0, 1.0
    user_params = {
        "star.A.av": {
            "initval": 1.0,
            "lower": "star.B.av",
            "mu": mu,
            "sigma": sigma,
        },
        "star.B.av": {"initval": 0.05},
    }
    cm, star, model = _build_star_param(user_params, ["av"])
    assert any(
        p.name.startswith("trunc_norm.star.av") for p in model.potentials
    )
    logp_c = model.compile_logp()
    point = model.initial_point()
    tf = star.av._raw_transform
    c_a, s_a = tf["logit_q_inits"][0], tf["init_scale_logits"][0]

    # Integrate av_A out on a grid in av_A itself: the Gaussian is 1 mag wide
    # inside a ~100 mag span, so a uniform grid in q would barely resolve it.
    # The measure is draw_a = dav_a / (span * s_a * q * (1 - q)).
    av_bs = (0.05, 0.5, 1.0, 1.5)
    p_avb = []
    zs_over_span = []
    for av_b in av_bs:
        q_b = av_b / 100.0
        raw_b = _raw_from_q(star.av, 1, q_b)
        span = 100.0 - av_b
        av_a = np.linspace(av_b + 1e-9, mu + 10.0 * sigma, 4001)
        q_a = (av_a - av_b) / span
        raw_a = (np.log(q_a / (1.0 - q_a)) - c_a) / s_a

        def _logp(ra):
            point["star.av_raw"] = np.array([ra, raw_b])
            return float(logp_c(point))

        dens = np.exp(np.array([_logp(ra) for ra in raw_a]))
        marginal = np.trapezoid(dens / (span * s_a * q_a * (1.0 - q_a)), av_a)
        p_avb.append(marginal / (q_b * (1.0 - q_b)))
        zs_over_span.append(
            (norm.sf((av_b - mu) / sigma) - norm.sf((100.0 - mu) / sigma))
            / span
        )

    # ASSERT: flat in av_B ...
    p_avb = np.array(p_avb)
    assert np.allclose(p_avb, p_avb[0], rtol=1e-4), (
        f"marginal p(av_B) varies with the truncated mass: "
        f"{dict(zip(av_bs, p_avb))}"
    )
    # ... and the bias it would have carried is far larger than that
    # tolerance, so this test really does discriminate.
    zs_over_span = np.array(zs_over_span)
    assert zs_over_span.max() / zs_over_span.min() > 2.0


# ----------------------------------------------------------------------
# Algebraic expressions
# ----------------------------------------------------------------------


def test_algebraic_link_orbit_omega_snapshot():
    """
    Given orbit.b.omega = {initval: "orbit.c.omega + math.pi", sigma: 0},
    When the relaxation engine runs,
    Then b's initval snapshot equals c's omega plus pi, in omega's USER
    units (degrees) -- referenced params contribute their user-unit values.
    """
    # ARRANGE
    config = {"orbit": [{"name": "b"}, {"name": "c"}]}
    user_params = {
        "orbit.c.omega": {"initval": 90.0},
        "orbit.b.omega": {"initval": "orbit.c.omega + math.pi", "sigma": 0},
    }
    cm = ConfigManager(user_params, system_config=config)

    # ACT
    cm.finalize_user_params()

    # ASSERT: snapshot in user units (deg): 90 + pi
    assert np.isclose(
        cm.user_params["orbit.0.omega"]["initval"], 90.0 + np.pi, rtol=1e-6
    )


def test_algebraic_hard_link_with_unit_conversion_in_graph():
    """
    Given star.A.ra = {initval: "star.B.ra + 10", sigma: 0} where ra has
    user unit deg and internal unit rad,
    When the model is built,
    Then A's internal (radian) value always equals B's plus 10 degrees.
    """
    # ARRANGE / ACT
    user_params = {
        "star.B.ra": {"initval": 200.0},
        "star.A.ra": {"initval": "star.B.ra + 10", "sigma": 0},
    }
    cm, star, model = _build_star_param(user_params, ["ra"])

    # ASSERT
    point = model.initial_point()
    for raw in (-1.0, 0.0, 2.5):
        point["star.ra_raw"] = np.array([raw])
        ra = _eval(model, star.ra.value, point)  # internal units (rad)
        assert np.isclose(ra[0], ra[1] + 10.0 * DEG2RAD, rtol=1e-12)


def test_cross_parameter_hard_link_within_component():
    """
    Given star.A.av hard-linked to "0.01 * star.A.age" (a different
    parameter of the same component),
    When av is built,
    Then age is materialized automatically and av tracks it dynamically.
    """
    # ARRANGE / ACT
    user_params = {"star.A.av": {"initval": "0.01 * star.A.age", "sigma": 0}}
    cm, star, model = _build_star_param(user_params, ["age", "av"])

    # ASSERT
    point = model.initial_point()
    for raw in (-1.0, 0.0, 1.0):
        point["star.age_raw"] = np.array([raw, 0.0])
        age = _eval(model, star.age.value, point)
        av = _eval(model, star.av.value, point)
        assert np.isclose(av[0], 0.01 * age[0], rtol=1e-12)


# ----------------------------------------------------------------------
# Initialization-only link (no sigma)
# ----------------------------------------------------------------------


def test_initval_link_without_sigma_seeds_start_only():
    """
    Given star.A.age = {initval: star.B.age} with star.B.age = 2.0 and no sigma,
    When the model is built,
    Then A starts at B's value but remains independently sampled with no
    runtime tie (no link potential).
    """
    # ARRANGE / ACT
    user_params = {
        "star.B.age": {"initval": 2.0},
        "star.A.age": {"initval": "star.B.age"},
    }
    cm, star, model = _build_star_param(user_params, ["age"])

    # ASSERT: snapshot applied, both sampled, no dynamic potentials
    assert np.isclose(cm.user_params["star.0.age"]["initval"], 2.0, rtol=1e-3)
    assert list(star.age.is_sampled) == [True, True]
    assert not any(p.name.startswith("link_") for p in model.potentials)
    # Starting point equals the linked value
    point = model.initial_point()
    ages = _eval(model, star.age.value, point)
    assert np.isclose(ages[0], 2.0, rtol=1e-6)


# ----------------------------------------------------------------------
# Static-field links: sigma / init_scale snapshots
# ----------------------------------------------------------------------


def test_sigma_link_snapshots_numerically():
    """
    Given star.A.age with sigma = "0.5 * star.B.age" and star.B.age = 4.0,
    When finalize_user_params runs,
    Then sigma resolves to the static numeric snapshot 2.0.

    The explicit mu is required: a linked sigma is still a Gaussian prior, and
    validate_sigma_has_center refuses a prior with no stated center (it would
    otherwise be centered on a possibly data-derived start value).
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {
        "star.B.age": {"initval": 4.0},
        "star.A.age": {"mu": 3.0, "sigma": "0.5 * star.B.age"},
    }
    cm = ConfigManager(user_params, system_config=config)

    # ACT
    cm.finalize_user_params()

    # ASSERT
    assert np.isclose(cm.user_params["star.0.age"]["sigma"], 2.0, rtol=1e-3)


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------


def test_unknown_instance_in_link_raises():
    """
    Given a link referencing a non-existent instance star.C.age,
    When the ConfigManager is constructed,
    Then a ValueError names the bad reference.
    """
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {"star.A.age": {"initval": "star.C.age", "sigma": 0}}
    with pytest.raises(ValueError, match="no instance named"):
        ConfigManager(user_params, system_config=config)


def test_non_numeric_non_link_string_raises():
    """
    Given a garbage string in a numeric field,
    When the ConfigManager is constructed,
    Then a ValueError is raised rather than a deep numpy crash.
    """
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {"star.A.age": {"initval": "not_a_number"}}
    with pytest.raises(ValueError, match="neither a number"):
        ConfigManager(user_params, system_config=config)


def test_self_referencing_link_raises():
    """
    Given a link whose expression references its own target,
    When the ConfigManager is constructed,
    Then a ValueError is raised.
    """
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {"star.A.age": {"initval": "2 * star.A.age", "sigma": 0}}
    with pytest.raises(ValueError, match="references its own"):
        ConfigManager(user_params, system_config=config)


def test_circular_hard_links_raise():
    """
    Given two elements hard-linked to each other (A := B, B := A),
    When the parameter is built,
    Then a circular-link ValueError is raised.
    """
    user_params = {
        "star.A.age": {"initval": "star.B.age", "sigma": 0},
        "star.B.age": {"initval": "star.A.age", "sigma": 0},
    }
    with pytest.raises(ValueError, match="[Cc]ircular"):
        _build_star_param(user_params, ["age"])


# ----------------------------------------------------------------------
# The caller's params dict is never mutated (review item 1.10)
# ----------------------------------------------------------------------


def test_construction_does_not_mutate_callers_params_dict():
    """
    Given a params dict holding a hard link,
    When a ConfigManager is constructed from it,
    Then the caller's dict still holds the link expression.

    extract_links strips the link strings out of the entries it scans.  Those
    entries must be copies: standardize_param_names deepcopies every one, so
    the strip lands on ConfigManager's own dict, not the caller's.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {
        "star.B.age": {"initval": 4.0},
        "star.A.age": {"initval": "star.B.age", "sigma": 0},
    }
    before = copy.deepcopy(user_params)

    # ACT
    ConfigManager(user_params, system_config=config)

    # ASSERT
    assert user_params == before


def test_links_survive_a_second_configmanager_from_the_same_dict():
    """
    Given a params dict already used to build one ConfigManager,
    When a second ConfigManager is built from the same dict,
    Then it finds the same links.

    A lost link is silent and changes the posterior: the target stops tracking
    its expression and is fixed at its default/solved value instead.  Bites any
    in-process reuse -- run_fit twice, solve_api.solve() twice (documented as
    safe to repeat), a shared test fixture.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {
        "star.B.age": {"initval": 4.0},
        "star.A.age": {"initval": "star.B.age", "sigma": 0},
    }
    first = ConfigManager(user_params, system_config=config)

    # ACT
    second = ConfigManager(user_params, system_config=config)

    # ASSERT
    assert set(second.links) == set(first.links) == {"star.0.age"}
    assert set(second.links["star.0.age"]) == {"initval"}
    assert (
        second.links["star.0.age"]["initval"].expr_str
        == first.links["star.0.age"]["initval"].expr_str
    )


def test_solve_does_not_inject_results_into_callers_params_dict():
    """
    Given a params dict driving a full finalize_user_params solve,
    When the relaxation engine injects its solution back,
    Then the caller's dict is unchanged.

    finalize_user_params writes initval/derived into the entries it resolved.
    Were those shared, a reused dict would come back carrying the previous
    solve's answer at PRECEDENCE_USER.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    user_params = {
        "star.A.teff": {"initval": 5800.0},
        "star.A.radius": {"initval": 1.0},
    }
    before = copy.deepcopy(user_params)
    cm = ConfigManager(user_params, system_config=config)

    # ACT
    cm.finalize_user_params()

    # ASSERT
    assert user_params == before


# ---------------------------------------------------------------------------
# An unknown function is a PARSE error  (review 1.1.2 / 7.1.1c)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("call", ["sqr(star.B.age)", "log10(star.B.age)"])
def test_an_unknown_function_raises_at_parse_time(call):
    """
    Given a link expression calling a function nothing can build -- a typo
      (`sqr`) or a name sympy does not define (`log10`),
    When the ConfigManager parses the params entry,
    Then it raises, naming the function and the supported set.

    sympify turns any unknown name into an AppliedUndef, and the parser used
    to validate only free SYMBOLS, so these sailed through.  Hard and soft
    links then failed loudly in sympy_to_pytensor -- but a seed-only initval
    link and a mu link never build a graph: they are consumed by
    _apply_directed_links, whose float(expr.evalf()) can never evaluate an
    undefined function and whose failure is logged as "not evaluable yet" at
    DEBUG.  The seed silently never applied.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}

    # ACT / ASSERT
    with pytest.raises(ValueError, match="not a function link expressions"):
        ConfigManager({"star.A.age": {"initval": call}}, system_config=config)


def test_a_seed_only_link_is_covered_too():
    """
    Given the seed-only spelling (an initval link with NO sigma) carrying an
      unknown function,
    When the ConfigManager parses it,
    Then it raises.

    This is the spelling the old behavior failed silently on -- it is the
    reason the check moved to parse time rather than being left to the
    graph builder, so it is pinned separately from the hard/soft links.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}

    # ACT / ASSERT
    with pytest.raises(ValueError, match="not a function link expressions"):
        ConfigManager(
            {"star.A.age": {"initval": "sqr(star.B.age)"}},
            system_config=config,
        )


@pytest.mark.parametrize(
    "expr",
    [
        "sqrt(star.B.age)",
        "log(star.B.age)/log(10)",
        "exp(star.B.age) + max(star.A.age, star.B.age)",
    ],
)
def test_supported_functions_still_parse(expr):
    """
    Given expressions using the supported vocabulary (including sqrt, which
      is a Pow rather than a Function, and the log(x)/log(10) spelling the
      error message recommends),
    When parse_link_expression runs,
    Then it returns an expression naming the referenced parameters.

    The new check rejects by absence from one table, so the assertion that
    matters is that the table is not too small.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}

    # ACT
    parsed, deps = parse_link_expression(expr, config)

    # ASSERT
    assert parsed is not None
    assert deps and all(d.startswith("star.") for d in deps)


def test_every_supported_function_can_be_built():
    """
    Given the table the parser validates against,
    When sympy_to_pytensor builds each head,
    Then every one of them resolves to a pytensor op.

    The two used to be independent literals, and the drift they permitted
    only ever failed silently (a function the parser accepted and the
    builder could not build).  The builder now derives its map from the
    parser's table; this pins that every name in it is real.
    """
    # ARRANGE
    import pytensor.tensor as pt

    # ACT / ASSERT
    for head, attr in _SUPPORTED_FUNCS.items():
        assert hasattr(pt, attr), f"{head} -> pt.{attr}"
