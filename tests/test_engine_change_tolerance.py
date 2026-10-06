"""When does the relaxation engine count a re-solved value as a change?
(review 1.1.9)

Two rules, both pinned here at the level of `config._meaningful_change` /
`config._value_moved`, with the end-to-end case (a tp-seeded orbit whose tc
lies near the defaults.yaml backstop) in tests/test_tp_seed.py:

  * a STRICTLY HIGHER-RANKED answer always replaces the value.  The old code
    kept the old value and promoted ITS rank instead, so a stale backstop
    was reported "solved";
  * otherwise the move is measured in units of the symbol's init_scale, not
    relative to its magnitude.  The relative test was ~2460 d wide on a BJD.
"""

import logging

import pytest

from exozippy.config import (
    CHANGE_TOL_IN_SCALES,
    PRECEDENCE_DEFAULT,
    PRECEDENCE_DERIVED_MIXED,
    _meaningful_change,
    _value_moved,
)

BACKSTOP = 2460000.0  # orbit.tc's defaults.yaml initval, days


def test_a_higher_ranked_value_1e4_relative_away_replaces_the_old_one():
    """
    Given a value held at PRECEDENCE_DEFAULT and a solver answer 1e-4
      RELATIVE away at PRECEDENCE_DERIVED_MIXED (246 d on a BJD, inside the
      old 1e-3 relative band),
    When the engine asks whether to apply it,
    Then it applies it -- and leaves the provenance to the caller, which
      writes the new value WITH the new rank.  (The old function wrote the
      new rank onto the OLD value and returned False.)
    """
    new = BACKSTOP * (1 + 1e-4)

    assert _meaningful_change(
        new,
        BACKSTOP,
        PRECEDENCE_DERIVED_MIXED,
        PRECEDENCE_DEFAULT,
        1e-3,
        None,
        "orbit.0.tc",
    )


def test_a_higher_rank_replaces_even_an_identical_value():
    """
    Given the same number arriving at a strictly higher rank,
    When the engine asks,
    Then it is applied: the caller then records the rank against a value
      the solver actually produced, instead of the rank being promoted onto
      a value the solver never touched.
    """
    assert _meaningful_change(
        1.0, 1.0, PRECEDENCE_DERIVED_MIXED, PRECEDENCE_DEFAULT, 1e-3, 0.1, "x"
    )


def test_an_equal_rank_sub_tolerance_move_is_not_a_change():
    """
    Given an equal-rank re-solve that moved by less than the tolerance,
    When the engine asks,
    Then it is NOT a change -- the guard that stops two agreeing derivation
      paths from running the loop to max_iter.
    """
    scale = 0.001
    old = BACKSTOP
    new = old + 0.5 * CHANGE_TOL_IN_SCALES * scale

    assert not _meaningful_change(
        new,
        old,
        PRECEDENCE_DERIVED_MIXED,
        PRECEDENCE_DERIVED_MIXED,
        1e-3,
        scale,
        "orbit.0.tc",
    )


@pytest.mark.parametrize("magnitude", [0.0, 1.0, 2.46e6])
def test_the_tolerance_is_in_init_scale_units_not_relative(magnitude):
    """
    Given a symbol with init_scale `scale`, at any magnitude (0, 1, a BJD),
    When a value moves by 0.5x and by 2x the tolerance in scale units,
    Then the first is not a change and the second is -- the same answer at
      every magnitude.  Relative to the magnitude, the 2x move on a BJD
      (2e-6 d) would have been 1e-12 and "unchanged".
    """
    scale = 0.001
    step = CHANGE_TOL_IN_SCALES * scale

    assert not _value_moved(
        magnitude + 0.5 * step, magnitude, scale, 1e-3, "p"
    )
    assert _value_moved(magnitude + 2.0 * step, magnitude, scale, 1e-3, "p")


def test_a_bjd_move_of_a_period_counts_even_though_it_is_tiny_relatively():
    """
    Given tc (init_scale 0.001 d) moved by 10 d at 2.46e6 -- 4e-6 relative,
      far inside the old 1e-3 band,
    When an equal-rank re-solve asks,
    Then it is a change.
    """
    assert _value_moved(BACKSTOP - 10.0, BACKSTOP, 0.001, 1e-3, "orbit.0.tc")


def test_a_symbol_without_init_scale_falls_back_to_relative_and_says_so(
    caplog,
):
    """
    Given a symbol the engine holds no init_scale for,
    When its change is tested,
    Then the old relative test applies AND a DEBUG line names the symbol --
      the fallback is visible, not silent.
    """
    with caplog.at_level(logging.DEBUG, logger="exozippy.config"):
        assert not _value_moved(1.0 + 1e-4, 1.0, None, 1e-3, "star.0.foo")
        assert _value_moved(1.0 + 1e-2, 1.0, None, 1e-3, "star.0.foo")

    assert "star.0.foo: no init_scale" in caplog.text
