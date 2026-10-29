"""A per-element bound on a DERIVED vector must not NaN the whole logp.

`ConfigManager.resolve` writes NaN into a vector for "this element was never
given one", so `orbit.BC.period: {lower: 3}` on a three-orbit system resolves
to `lowers = [nan, 3, nan]`.  `build_pymc`'s soft-barrier gate was
`~np.isinf(lowers)`, and `~np.isinf(nan)` is True -- so those elements took
the barrier, `soft_lower_bound(v, nan)` returned NaN, and the WHOLE model logp
was NaN with nothing naming the parameter.

Found while doing review 8.8.8(c), which asks for exactly that entry on
`examples/kelt4`'s hierarchical triple: the start logp went from 82862.6 to
`nan`.  A bound nobody stated is no bound.
"""

import numpy as np
import pytensor.tensor as pt
import pytest

from exozippy.system import System

_CONFIG = {
    "star": [{"name": "A", "mist": False}],
    "planet": [{"name": "b"}, {"name": "c"}],
    "orbit": [
        {"name": "b", "primary": ["A"], "companion": ["b"]},
        {"name": "c", "primary": ["A"], "companion": ["c"]},
    ],
}

_BASE = {
    "star.A.mass": {"initval": 1.0, "sigma": 0.05},
    "star.A.radius": {"initval": 1.0, "sigma": 0.05},
    "orbit.b.tc": {"initval": 2455010.0},
    "orbit.c.tc": {"initval": 2455010.0},
    "orbit.b.period": {"initval": 5.0},
    "orbit.c.period": {"initval": 50.0},
    "planet.b.radius": {"initval": 1.0},
    "planet.c.radius": {"initval": 1.0},
}


def _logp(**extra):
    params = dict(_BASE)
    params.update(extra)
    system = System(dict(_CONFIG), params)
    system.prepare()
    model = system.build_model()
    point = model.initial_point()
    return (
        float(model.compile_logp()(point)),
        np.atleast_1d(model.compile_dlogp()(point)),
        system,
    )


@pytest.mark.slow
def test_a_bound_on_one_element_of_a_derived_vector_keeps_logp_finite():
    """
    Given a soft `lower` on ONE orbit's derived period and none on the
      other -- the spelling review 8.8.8(c) asks for on examples/kelt4,
    When the model is built and scored at the start,
    Then the logp and its gradient are finite, and the element that has no
      bound is treated as unbounded rather than as bounded by NaN.
    """
    # Arrange / Act
    logp, grads, system = _logp(
        **{"orbit.c.period": {"initval": 50.0, "lower": 3.0}}
    )

    # Assert
    lowers = np.atleast_1d(system.orbit.period.lower)
    assert np.isnan(lowers[0]) and lowers[1] == 3.0
    assert np.isfinite(logp)
    assert np.isfinite(grads).all()


@pytest.mark.slow
def test_it_agrees_with_the_all_elements_bounded_case():
    """
    Given the same system with the bound on BOTH elements, satisfied by a
      wide margin in each,
    When both are scored,
    Then the logp is the same as with no bound at all -- the barrier costs
      nothing where it is not active, so the one-element case is not merely
      finite but numerically the same answer.
    """
    none_logp, _, _ = _logp()
    both_logp, _, _ = _logp(
        **{
            "orbit.b.period": {"initval": 5.0, "lower": 3.0},
            "orbit.c.period": {"initval": 50.0, "lower": 3.0},
        }
    )
    one_logp, _, _ = _logp(
        **{"orbit.c.period": {"initval": 50.0, "lower": 3.0}}
    )
    assert both_logp == pytest.approx(none_logp, rel=1e-12)
    assert one_logp == pytest.approx(none_logp, rel=1e-12)


@pytest.mark.slow
def test_an_active_per_element_bound_still_bites():
    """
    Given a `lower` that the element's start VIOLATES,
    When the model is scored,
    Then the logp is lower than without it.  The fix must not have turned
      the barrier off; it only stops a bound nobody stated from being read
      as a NaN one.
    """
    free_logp, _, _ = _logp()
    bound_logp, _, _ = _logp(
        **{"orbit.b.period": {"initval": 5.0, "lower": 500.0}}
    )
    assert bound_logp < free_logp - 1.0


def _closed_form_lower(val, threshold, scale):
    """``potentials.soft_lower_bound`` (softness 0.01) written out:
    log(sigmoid((val - threshold) * 4.4 / (0.01 * scale))), in the stable
    spelling -logaddexp(0, -z).  Past the 700-nat clip the shipped form
    adds the remainder back linearly, which -logaddexp(0, -z) equals to
    float precision there (log sigmoid(z) = z + log1p(exp(z)))."""
    z = (val - threshold) * 4.4 / (0.01 * scale)
    return -np.logaddexp(0.0, -z)


@pytest.mark.slow
def test_an_active_per_element_bound_penalizes_exactly_its_own_element():
    """
    Given `lower: 500` on orbit b's period only, which BOTH orbits' starts
      (5 and 50 days) would violate,
    When the built low_bound.orbit.period potential is evaluated at the
      start,
    Then it equals the closed-form soft lower bound of element b's period
      alone, with the barrier scale the build holds, to rel 1e-9 -- and
      differs from both the element-c-only and the both-elements sums.

    test_an_active_per_element_bound_still_bites only asks for a logp drop
    > 1 nat, which a bound applied to the wrong element (or to both) also
    produces, because c violates 500 too.
    """
    # ARRANGE
    params = dict(_BASE)
    params["orbit.b.period"] = {"initval": 5.0, "lower": 500.0}
    system = System(dict(_CONFIG), params)
    system.prepare()
    model = system.build_model()
    point = model.initial_point()
    period = system.orbit.period
    scale = np.asarray(
        period.export_whitening()["barrier_scales"], dtype=float
    )
    names = list(system.orbit.names)
    ib, ic = names.index("b"), names.index("c")
    pot = [p for p in model.potentials if p.name == "low_bound.orbit.period"]
    assert len(pot) == 1, [p.name for p in model.potentials]

    # ACT
    fn = model.compile_fn(
        model.replace_rvs_by_values([pot[0], pt.flatten(period.value)]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    built, periods = fn(point)
    built = float(built)
    periods = np.atleast_1d(np.asarray(periods, dtype=float))

    # ASSERT
    assert periods[ib] == pytest.approx(5.0, rel=1e-9)
    assert periods[ic] == pytest.approx(50.0, rel=1e-9)
    own = _closed_form_lower(periods[ib], 500.0, scale[ib])
    wrong = _closed_form_lower(periods[ic], 500.0, scale[ic])
    assert own < -1.0 and wrong < -1.0  # both would bite: c is a real decoy
    assert built == pytest.approx(own, rel=1e-9)
    assert built != pytest.approx(wrong, rel=1e-3)
    assert built != pytest.approx(own + wrong, rel=1e-3)
