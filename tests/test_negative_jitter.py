"""Negative jitter is a result, not a pathology.

``Instrument._jitter_floor`` deliberately lets ``jitter_variance`` go negative
(down to ``-0.95 * min(err)**2``, the validity limit at which
``sqrt(err**2 + jitter_variance)`` is still real), because a negative jitter
variance says the quoted error bars are too large -- real, useful information.
The reported ``jitter`` used to be clamped to 0 over that whole half-axis,
which cost a Lucy-Sweeney-style upward bias on a marginally detected jitter
and left a zero-gradient plateau for any prior or link on ``jitter`` to push
against.

These pin the fix: the report is the SIGNED square root, one shared
implementation for all three additive-noise components, negative values
survive the relaxation engine and the output tables, and ``total_sigma``
stays real everywhere the sampler can reach.
"""

import csv

import astropy.units as u
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.factory import discover_components
from exozippy.components.instrument import Instrument, calc_jitter
from exozippy.components.parameter import PosteriorSummary
from exozippy.outputs.latex import build_csv_output
from exozippy.physics_registry import PHYSICS_REGISTRY
from exozippy.system import System

JITTER_TO_INTERNAL = (u.m / u.s).to(u.solRad / u.d)
# The fixture's errors are a flat 3 m/s, so the floor is -0.95 * 3**2.
EXPECTED_FLOOR = -0.95 * 9.0


@pytest.fixture(scope="module")
def jitter_fn():
    """The reported-jitter relation and its gradient, as compiled functions."""
    v = pt.dscalar("jitter_variance")
    j = calc_jitter(v)
    return pytensor.function([v], [j, pt.grad(j, v)])


# ---------------------------------------------------------------------------
# The relation itself
# ---------------------------------------------------------------------------
def test_reported_jitter_is_the_signed_root_of_a_negative_variance(jitter_fn):
    """
    Given a negative jitter_variance (legal: it is above _jitter_floor),
    When the reported jitter is evaluated,
    Then it is -sqrt(|v|) with the true sqrt gradient -- not the old clamp to
    zero, which reported 0 with zero gradient across the whole half-axis.
    """
    for v in (-8.0, -4.0, -1.0, -0.25, -1e-6):
        value, grad = jitter_fn(v)

        assert float(value) == pytest.approx(-np.sqrt(-v), rel=1e-12)
        assert float(value) < 0.0
        assert float(grad) == pytest.approx(0.5 / np.sqrt(-v), rel=1e-9)
        assert float(grad) > 0.0


def test_reported_jitter_is_monotonic_across_zero(jitter_fn):
    """
    Given a sweep of jitter_variance spanning the negative and positive sides,
    When the reported jitter is evaluated,
    Then it increases monotonically through zero (the sign carries through, so
    the report is invertible rather than folded onto the positive axis).
    """
    grid = np.linspace(-25.0, 25.0, 201)
    values = np.array([float(jitter_fn(v)[0]) for v in grid])

    assert np.all(np.diff(values) > 0.0)
    # Odd about zero: equal and opposite variances report equal and opposite
    # jitters, which is what "folded onto the positive axis" would destroy.
    assert values == pytest.approx(-values[::-1], abs=1e-12)


def test_gradient_is_finite_at_the_default_zero_variance_start(jitter_fn):
    """
    Given jitter_variance = 0 exactly -- every component's defaults.yaml
    initval, so every fit starts there,
    When the reported jitter and its gradient are evaluated,
    Then both are finite (the pre-fix pt.switch took the sqrt branch here and
    returned d jitter / d jitter_variance = inf).

    The radicand floor is what buys this, following calc_theta_E: sqrt'(0) is
    infinite and clamping the result afterwards multiplies that infinity by
    pt.maximum's zero gradient, giving NaN.
    """
    value, grad = jitter_fn(0.0)

    assert float(value) == 0.0
    assert np.isfinite(float(grad))


def test_the_radicand_floor_is_far_below_any_real_error_bar(jitter_fn):
    """
    Given the floor quantizes the report near zero,
    When the smallest resolvable jitter is computed,
    Then it is <= 1e-15 in internal units -- below float64's grip on any real
    error bar, so the floor cannot round a physical jitter away.
    """
    tiny = float(jitter_fn(1e-25)[0])

    assert 0.0 < tiny <= 1e-12
    assert float(jitter_fn(-1e-25)[0]) == pytest.approx(-tiny, rel=1e-12)
    assert float(jitter_fn(1.0)[0]) == pytest.approx(1.0, rel=1e-12)
    assert float(jitter_fn(-1.0)[0]) == pytest.approx(-1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# One implementation, not three
# ---------------------------------------------------------------------------
def test_one_shared_jitter_relation_serves_all_three_components():
    """
    Given rvinstrument, transit and astrometryinstrument all report a jitter
    from the same additive noise model,
    When the physics registry is populated,
    Then a single 'calc_jitter' owns the relation and the three byte-identical
    per-component copies (calc_transit_jitter / calc_astrom_jitter) are gone.
    """
    discover_components()

    assert PHYSICS_REGISTRY["calc_jitter"] is calc_jitter
    assert "calc_transit_jitter" not in PHYSICS_REGISTRY
    assert "calc_astrom_jitter" not in PHYSICS_REGISTRY


def test_every_additive_noise_component_wires_the_shared_relation():
    """
    Given each component's defaults.yaml names the func for its 'jitter',
    When those blocks are read,
    Then all three name 'calc_jitter' (so a change to the convention cannot
    reach one component and miss another).
    """
    from exozippy.config import ConfigManager

    cm = ConfigManager({}, {})
    for comp in ("rvinstrument", "transit", "astrometryinstrument"):
        block = cm.base_defaults[comp]["jitter"]["expressions"]["default"]
        assert block["func_name"] == "calc_jitter"
        assert block["deps"] == ["jitter_variance"]


# ---------------------------------------------------------------------------
# End to end through a model
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rv_file(tmp_path_factory):
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, 40))
    rv = 30.0 * np.sin(2 * np.pi * t / 17.0) + rng.normal(0, 3.0, 40)
    err = np.full(40, 3.0)
    path = tmp_path_factory.mktemp("negjitter") / "a.rv"
    np.savetxt(path, np.column_stack([t, rv, err]))
    return str(path)


def _build(rv_file, extra_params=None):
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": [{"name": "A_inst", "file": rv_file}],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(17.0)},
        "orbit.b.tc": {"initval": 2455010.0},
    }
    if extra_params:
        params.update(extra_params)
    system = System(config, params)
    system.prepare()
    model = system.build_model()
    return system, model


@pytest.fixture(scope="module")
def rv_system(rv_file):
    return _build(rv_file)


_EVALUATORS = {}


def _evaluator(system, model, node):
    """Compile (once) a node as a function of jitter_variance's raw value."""
    if id(node) not in _EVALUATORS:
        jv = system.rvinstrument.jitter_variance
        value_var = model.rvs_to_values[model.named_vars[f"{jv.label}_raw"]]
        _EVALUATORS[id(node)] = pytensor.function(
            [value_var], model.replace_rvs_by_values([node])[0]
        )
    return _EVALUATORS[id(node)]


def _eval_at(system, model, node, jittervar_internal):
    """Evaluate a model node with jitter_variance set to a chosen value."""
    jv = system.rvinstrument.jitter_variance
    raw = jv.raw_from_initval(np.array([jittervar_internal]))
    return np.asarray(_evaluator(system, model, node)(raw))


def test_the_model_reports_a_negative_jitter_with_a_live_gradient(rv_system):
    """
    Given a model whose jitter_variance is set inside its negative floor,
    When the reported jitter Deterministic and the model's dlogp are
    evaluated there,
    Then the jitter is negative (in the data's own m/s) and every dlogp entry
    is finite -- so a prior on jitter has something to push against.
    """
    system, model = rv_system
    jv = system.rvinstrument.jitter_variance
    jitter = system.rvinstrument.jitter
    lower = float(np.atleast_1d(jv.lower)[0])
    assert lower < 0.0  # the region under test is the negative half-axis

    point = model.initial_point()
    dlogp_fn = model.compile_dlogp()
    raw_name = f"{jv.label}_raw"

    previous = -np.inf
    for frac in (0.999, 0.75, 0.5, 0.25):
        internal = frac * lower
        value = float(
            np.atleast_1d(_eval_at(system, model, jitter.value, internal))[0]
        )
        user = float(np.atleast_1d(jitter.from_internal(value))[0])

        assert user < 0.0
        assert user == pytest.approx(
            -np.sqrt(-internal) / JITTER_TO_INTERNAL, rel=1e-6
        )
        assert user > previous  # monotone in jitter_variance
        previous = user

        probe = dict(point)
        probe[raw_name] = jv.raw_from_initval(np.array([internal]))
        assert np.all(np.isfinite(dlogp_fn(probe)))


def test_total_sigma_stays_real_down_to_the_floor(rv_system):
    """
    Given the jitter-variance floor is what keeps err**2 + jitter_variance
    positive,
    When total_sigma is evaluated across the negative region and at the floor,
    Then every per-observation sigma is finite and strictly positive, and the
    floor is the -0.95 * min(err)**2 validity limit it claims to be.

    This is the agreement the reporting convention relies on: the signed root
    is applied to a quantity the floor has already kept above -min(err)**2.
    """
    system, model = rv_system
    jv = system.rvinstrument.jitter_variance
    lower = float(np.atleast_1d(jv.lower)[0])

    assert jv.from_internal(lower) == pytest.approx(EXPECTED_FLOOR)
    assert system.rvinstrument.jittervar_lower[0] == pytest.approx(
        EXPECTED_FLOOR
    )

    err_node = pt.as_tensor_variable(system.rvinstrument.err)
    sigma_node = system.rvinstrument.total_sigma(err_node)

    for frac in (1.0, 0.999, 0.5, 0.0, -1.0):
        sigma = _eval_at(system, model, sigma_node, frac * lower)
        assert np.all(np.isfinite(sigma)), f"non-finite sigma at {frac}*floor"
        assert np.all(sigma > 0.0), f"non-positive sigma at {frac}*floor"


def test_a_negative_jitter_seed_survives_as_a_negative_variance(rv_file):
    """
    Given a user who seeds the instrument's jitter negative,
    When the relaxation engine derives the sampled jitter_variance,
    Then the variance is negative too.

    The symbolic bridge is the SIGNED square (jitter * |jitter|).  As
    jitter**2 it folded the seed onto +4 m2/s2 -- a silent sign flip on the
    only direction of that relation that matters.
    """
    system, _ = _build(
        rv_file, {"rvinstrument.A_inst.jitter": {"initval": -2.0}}
    )

    jv = system.rvinstrument.jitter_variance
    seeded = float(np.atleast_1d(jv.from_internal(jv.initval))[0])

    assert seeded == pytest.approx(-4.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Reporting: a negative jitter has to render
# ---------------------------------------------------------------------------
def test_posterior_summary_formats_a_negative_median():
    """
    Given a posterior whose median jitter is negative,
    When the table formatter runs,
    Then it renders the sign rather than a NaN (the sigfig logic takes log10
    of the ERRORS' magnitudes, never of the median).
    """
    summary = PosteriorSummary(median=-2.7183, err_minus=0.42, err_plus=0.38)

    med, em, ep = summary.format(sigfigs=2)

    assert float(med) < 0.0
    assert "nan" not in summary.latex_value().lower()
    assert summary.latex_value().startswith("-2.7")


def test_csv_output_renders_a_negative_jitter_row(rv_system, tmp_path):
    """
    Given a posterior that spent its time at negative jitter_variance -- the
    reported jitters here are the MODEL's, evaluated at those draws, not
    numbers made up for the test,
    When the machine-readable CSV is written,
    Then the jitter row carries the negative value and no NaN.
    """
    system, model = rv_system
    jitter = system.rvinstrument.jitter
    jv = system.rvinstrument.jitter_variance
    lower = float(np.atleast_1d(jv.lower)[0])

    rng = np.random.default_rng(3)
    draws = rng.uniform(0.9 * lower, 0.3 * lower, 60)
    reported = np.array(
        [
            float(
                np.atleast_1d(
                    jitter.from_internal(
                        _eval_at(system, model, jitter.value, d)
                    )
                )[0]
            )
            for d in draws
        ]
    )
    assert np.all(reported < 0.0)  # pre-fix these were all exactly 0

    saved_posterior, saved_summary = jitter.posterior, jitter.summary
    path = str(tmp_path / "results.csv")
    try:
        jitter.posterior = reported.reshape(1, -1)
        jitter.summary = None
        build_csv_output(system, path)

        with open(path) as fh:
            rows = [
                r
                for r in csv.reader(fh)
                if r and not r[0].startswith("#") and "jitter" in r[0]
            ]
    finally:
        jitter.posterior, jitter.summary = saved_posterior, saved_summary

    jitter_rows = [r for r in rows if not r[0].endswith("jitter_variance")]
    assert jitter_rows, f"no jitter row in {rows}"
    for row in jitter_rows:
        assert float(row[1]) < 0.0
        assert not any("nan" in cell.lower() for cell in row)


def test_the_jitter_floor_is_still_the_thing_that_bounds_the_sqrt():
    """
    Given the reported jitter no longer guards its own sign,
    When the floor formula is checked,
    Then it is still -0.95 * min(err)**2 -- the validity limit below which
    total_sigma's sqrt (not the report's) goes imaginary.
    """
    err = np.array([0.3, 0.1, 0.25])

    assert Instrument._jitter_floor(err) == pytest.approx(-0.95 * 0.1**2)
    assert Instrument._jitter_floor(err) > -(np.min(err) ** 2)
