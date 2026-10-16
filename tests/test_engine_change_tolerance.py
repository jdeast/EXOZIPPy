"""When does the relaxation engine count a value as having moved?
(reviews 1.1.9, 3.14.22)

The change test is `config.value_moved` (relaxation.value_moved): a move
counts when it is at least CHANGE_TOL_IN_SCALES of the symbol's init_scale.
Since review 3.14.22 every engine symbol has a scale -- DECLARED
(defaults.yaml, a component scale hint, a user sigma) or PROPAGATED from the
sampled parameters through the relations -- so the relative-to-magnitude
fallback the 1.1.9 version kept for scale-less symbols is gone: a symbol
with no scale RAISES, naming it.  The same tolerance decides whether a
relation is violated (relaxation.Engine.violation) and so whether a user
value on a derived element holds.

The other half of 1.1.9 -- "a strictly higher-ranked answer always
replaces the value" -- no longer needs a function: under the KS rule a
higher precedence is never traded against a lower one at all (it fixes its
seeds first, and the lower tier absorbs).  End to end: tests/test_tp_seed.py
(a tp-seeded orbit whose tc lies near the defaults.yaml backstop, which the
old relative test discarded).
"""

import pytest

from exozippy.config import CHANGE_TOL_IN_SCALES, value_moved

BACKSTOP = 2460000.0  # orbit.tc's defaults.yaml initval, days


@pytest.mark.parametrize("magnitude", [0.0, 1.0, 2.46e6])
def test_the_tolerance_is_in_init_scale_units_not_relative(magnitude):
    """
    Given a symbol with init_scale `scale`, at any magnitude (0, 1, a BJD),
    When a value moves by 0.5x and by 2x the tolerance in scale units,
    Then the first is not a change and the second is -- the same answer at
      every magnitude.  Relative to the magnitude, the 2x move on a BJD
      (2e-8 d) would have been 1e-14 and "unchanged".
    """
    scale = 0.001
    step = CHANGE_TOL_IN_SCALES * scale

    assert not value_moved(magnitude + 0.5 * step, magnitude, scale, "p")
    assert value_moved(magnitude + 2.0 * step, magnitude, scale, "p")


def test_a_bjd_move_of_ten_days_counts_even_though_it_is_tiny_relatively():
    """
    Given tc (init_scale 0.001 d) moved by 10 d at 2.46e6 -- 4e-6 relative,
      far inside the old 1e-3 relative band (review 1.1.9),
    When the move is tested,
    Then it is a change.
    """
    assert value_moved(BACKSTOP - 10.0, BACKSTOP, 0.001, "orbit.0.tc")


def test_a_periodic_move_is_measured_the_short_way_round():
    """
    Given an angle with period 2 pi and init_scale 0.01 rad,
    When it moves from just below 2 pi to just above 0 (a 1e-7 rad step
      across the cut),
    Then it has not moved -- the difference is wrapped first.
    """
    two_pi = 6.283185307179586
    assert not value_moved(5e-8, two_pi - 5e-8, 0.01, "orbit.0.omega", two_pi)
    assert value_moved(0.5, two_pi - 0.5, 0.01, "orbit.0.omega", two_pi)


@pytest.mark.parametrize("scale", [None, 0.0, float("nan")])
def test_a_symbol_without_a_scale_raises_naming_it(scale):
    """
    Given a symbol with no usable init_scale,
    When its change is tested,
    Then it RAISES, naming the symbol and where to declare one -- the
      relative fallback this used to apply (and log at DEBUG) is gone
      (review 3.14.22: no constant backstop, no relative substitute).
    """
    with pytest.raises(ValueError, match="star.0.foo") as err:
        value_moved(1.0 + 1e-2, 1.0, scale, "star.0.foo")
    assert "star/defaults.yaml" in str(err.value)
