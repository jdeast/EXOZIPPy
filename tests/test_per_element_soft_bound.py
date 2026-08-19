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
import pytest

from exozippy.system import System

_CONFIG = {
    "star": [{"name": "A", "mist": False}],
    "planet": [{"name": "b", "orbit_ndx": 0}, {"name": "c", "orbit_ndx": 1}],
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
