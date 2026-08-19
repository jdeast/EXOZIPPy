"""The circular-orbit Kepler fast path (review 6.8.2).

A circular fit -- `secosw` and `sesinw` pinned at 0, as `examples/kelt17`
writes it -- was paying an exoplanet-core Newton solve per epoch, per orbit,
for an answer that is one sine and one cosine: at `e = 0` the mean, eccentric
and true anomalies all coincide.

Two things have to hold.  The fast path must fire only on a STRUCTURALLY
circular orbit -- a pin, never a value that merely starts at zero, since an
unpinned eccentricity can move and the whole run would then carry the wrong
RV phase.  And it must be selected at GRAPH-BUILD time, not with a
`pt.switch`, which would build (and evaluate) the Newton solve anyway.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.orbit import physics
from exozippy.system import System

_CONFIG = {
    "star": [{"name": "A", "mist": False}],
    "planet": [{"name": "b"}],
    "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
}


def _system(**extra):
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.period": {"initval": 5.0},
        "orbit.b.tc": {"initval": 2455010.0},
        "planet.b.radius": {"initval": 1.0},
    }
    params.update(extra)
    system = System(dict(_CONFIG), params)
    system.prepare()
    return system


_PINNED = {
    "orbit.b.secosw": {"initval": 0.0, "sigma": 0},
    "orbit.b.sesinw": {"initval": 0.0, "sigma": 0},
}


# ---------------------------------------------------------------------------
# 1. The predicate
# ---------------------------------------------------------------------------


def test_a_pinned_sqrt_e_pair_is_recognized_as_circular():
    """
    Given the sqrt(e) pair pinned at zero -- how a circular fit is written,
    When the orbit is asked,
    Then it reports the orbit as structurally circular.
    """
    system = _system(**_PINNED)
    assert list(system.orbit.circular_orbits) == [True]


def test_an_unpinned_pair_at_zero_is_NOT_circular():
    """
    Given an eccentricity that merely STARTS at zero, with no pin,
    When the orbit is asked,
    Then it is not circular.  This is the distinction the whole fast path
      rests on: an unpinned e can move, and solving as if it were circular
      would silently give the run the wrong RV phase for every draw after
      the first.
    """
    system = _system(
        **{
            "orbit.b.secosw": {"initval": 0.0},
            "orbit.b.sesinw": {"initval": 0.0},
        }
    )
    assert list(system.orbit.circular_orbits) == [False]


def test_a_half_pinned_pair_is_not_circular():
    """
    Given only ONE of the pair pinned,
    When the orbit is asked,
    Then it is not circular -- the other coordinate is still free, so e is.
    """
    system = _system(
        **{
            "orbit.b.secosw": {"initval": 0.0, "sigma": 0},
            "orbit.b.sesinw": {"initval": 0.01},
        }
    )
    assert list(system.orbit.circular_orbits) == [False]


def test_a_pin_at_a_nonzero_value_is_not_circular():
    """
    Given the pair pinned at a NON-zero value -- a fixed eccentric orbit,
    When the orbit is asked,
    Then it is not circular; the pin says the eccentricity is known, not
      that it is zero.
    """
    system = _system(
        **{
            "orbit.b.secosw": {"initval": 0.3, "sigma": 0},
            "orbit.b.sesinw": {"initval": 0.0, "sigma": 0},
        }
    )
    assert list(system.orbit.circular_orbits) == [False]


# ---------------------------------------------------------------------------
# 2. The solve itself
# ---------------------------------------------------------------------------


def test_the_fast_path_is_the_exact_answer_at_zero_eccentricity():
    """
    Given a mean anomaly and e = 0,
    When the fast path and the Newton solve are both evaluated,
    Then they agree to the solver's tolerance.  The fast path is the EXACT
      answer -- f = E = M at e = 0 -- so where they differ it is the
      iterate that is approximate.
    """
    # Arrange
    m = pt.dvector("M")
    fast = physics.solve_kepler(m, 0.0, circular=True)
    slow = physics.solve_kepler(m, pt.zeros_like(m), circular=False)
    fn = pytensor.function([m], list(fast) + list(slow))

    # Act
    grid = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 97)
    sinf_fast, cosf_fast, sinf_slow, cosf_slow = fn(grid)

    # Assert
    np.testing.assert_allclose(sinf_fast, np.sin(grid), atol=0, rtol=1e-15)
    np.testing.assert_allclose(cosf_fast, np.cos(grid), atol=0, rtol=1e-15)
    np.testing.assert_allclose(sinf_fast, sinf_slow, atol=1e-9)
    np.testing.assert_allclose(cosf_fast, cosf_slow, atol=1e-9)


def test_the_fast_path_removes_the_kepler_op_from_the_graph():
    """
    Given a circular solve,
    When the graph is inspected,
    Then it contains no exoplanet-core Kepler op at all.  That is the
      saving -- and the reason the choice is structural rather than a
      `pt.switch`, which would build the solve and then discard it.
    """
    m = pt.dvector("M")

    def op_names(outs):
        return {
            type(v.owner.op).__name__
            for v in pytensor.graph.traversal.ancestors(list(outs))
            if v.owner is not None
        }

    fast = op_names(physics.solve_kepler(m, 0.0, circular=True))
    slow = op_names(physics.solve_kepler(m, pt.zeros_like(m), circular=False))

    assert not any("epler" in name for name in fast), fast
    assert any("epler" in name for name in slow), slow


# ---------------------------------------------------------------------------
# 3. End to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pinned", [True, False])
def test_the_model_builds_and_scores_either_way(pinned):
    """
    Given a circular (pinned) or an eccentric (free) orbit,
    When the model is built,
    Then the start logp and gradient are finite.  Measured on
      `examples/kelt17`, the shipped circular fit, the fast path moves the
      start logp by 4.9e-9 nats (3.3e-12 relative) -- the gap between
      exoplanet-core's converged Newton iterate and the exact answer, in
      the exact direction.
    """
    system = _system(**(_PINNED if pinned else {}))
    assert bool(system.orbit.circular_orbits[0]) is pinned
    model = system.build_model()
    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.isfinite(np.atleast_1d(model.compile_dlogp()(point))).all()
