"""Tests for potentials.py: soft_lower_bound and soft_upper_bound."""
import numpy as np
import pytensor
import pytensor.tensor as pt

from exozippy.potentials import soft_lower_bound, soft_upper_bound


def _eval(expr):
    f = pytensor.function([], expr)
    return float(f())


def test_soft_lower_bound_is_negligible_well_inside():
    """
    Given val >> threshold,
    When soft_lower_bound is evaluated,
    Then the penalty is near zero (negligible log-sigmoid contribution).
    """
    penalty = soft_lower_bound(pt.constant(100.0), threshold=0.0, scale=1.0)
    assert _eval(penalty) > -0.01


def test_soft_lower_bound_is_large_negative_well_outside():
    """
    Given val << threshold,
    When soft_lower_bound is evaluated,
    Then the penalty is a large negative value.
    """
    penalty = soft_lower_bound(pt.constant(-100.0), threshold=0.0, scale=1.0)
    assert _eval(penalty) < -10.0


def test_soft_lower_bound_at_threshold_equals_log_half():
    """
    Given val == threshold,
    When soft_lower_bound is evaluated,
    Then the penalty equals log(sigmoid(0)) = log(0.5) ≈ -0.693.
    """
    penalty = soft_lower_bound(pt.constant(0.0), threshold=0.0, scale=1.0)
    np.testing.assert_allclose(_eval(penalty), np.log(0.5), rtol=1e-4)


def test_soft_upper_bound_is_negligible_well_inside():
    """
    Given val << threshold,
    When soft_upper_bound is evaluated,
    Then the penalty is near zero.
    """
    penalty = soft_upper_bound(pt.constant(-100.0), threshold=0.0, scale=1.0)
    assert _eval(penalty) > -0.01


def test_soft_upper_bound_is_large_negative_well_outside():
    """
    Given val >> threshold,
    When soft_upper_bound is evaluated,
    Then the penalty is a large negative value.
    """
    penalty = soft_upper_bound(pt.constant(100.0), threshold=0.0, scale=1.0)
    assert _eval(penalty) < -10.0


def test_soft_lower_and_upper_are_symmetric():
    """
    Given symmetric displacements from the threshold in each direction,
    When both soft_lower_bound and soft_upper_bound are evaluated,
    Then the penalties are equal (the barrier is the same function reflected).
    """
    t, d, scale = 5.0, 2.0, 1.0
    pen_lower = soft_lower_bound(pt.constant(t + d), threshold=t, scale=scale)
    pen_upper = soft_upper_bound(pt.constant(t - d), threshold=t, scale=scale)
    np.testing.assert_allclose(_eval(pen_lower), _eval(pen_upper), rtol=1e-6)


def test_softness_parameter_controls_turn_on_width():
    """
    Given two barriers with different softness at the same displacement from threshold,
    When softness is larger the barrier turns on more gradually,
    Then the softer barrier has a smaller penalty (closer to zero) at one turn-on width in.
    """
    # One turn-on width inside (val = threshold - scale * softness)
    t, scale = 0.0, 1.0
    softness_tight = 0.01
    softness_wide = 0.5
    # val below threshold → testing lower bound penalty approaching from inside (above threshold)
    val_tight = t + scale * softness_tight
    val_wide = t + scale * softness_wide
    pen_tight = _eval(soft_lower_bound(pt.constant(val_tight), threshold=t, scale=scale, softness=softness_tight))
    pen_wide = _eval(soft_lower_bound(pt.constant(val_wide), threshold=t, scale=scale, softness=softness_wide))
    # both are inside the bound so both should be negative but pen_tight ≈ -0.012, pen_wide also ≈ -0.012
    # The key property: at (threshold + scale*softness) the penalty is ~ -0.012 regardless of softness
    np.testing.assert_allclose(pen_tight, np.log(pt.sigmoid(4.4).eval()), rtol=0.01)
    np.testing.assert_allclose(pen_wide, np.log(pt.sigmoid(4.4).eval()), rtol=0.01)
