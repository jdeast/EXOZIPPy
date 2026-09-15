"""Per-temperature start dispersion for the PTDE ladder (review 8.4.7).

JDE's ruling: make it a PER-TEMPERATURE knob, and ABSOLUTE rather than a
multiplier -- "defaulting to 3 for the t=1 rung (so absolute, not multiplier,
but recreating the current behavior) and then scaling with temperature."

What it replaces: both PTDE variants resolved ONE T=1 population and
replicated it verbatim to every rung, under the comment "hotter chains spread
quickly during tune" -- an assumption the code stated and never verified.  So
there was no per-rung dispersion to set, and nothing per-rung to check.

One test per clause of the ruling, plus the two properties the item itself
asked for (per-rung, and visible from outside):

  1. 'auto' is 3*sqrt(T), and T=1 lands on exactly 3.
  2. ABSOLUTE: a scalar is that value at EVERY rung -- which is also what the
     sampler did before 8.4.7, so it is the control arm.
  3. Callers that pass nothing are bit-identical to the old rule.
  4. Each rung is really dispersed at its own value.
  5. Rung 0 IS the returned T=1 population.
  6. A wrong-length list raises instead of leaving hot rungs unset.
  7. A non-positive dispersion raises (a zero defeats Rhat -- review 2.4.5).
  8. `initvals` is one start per chain and cannot be dispersed per rung.
  9. The one model class whose T=1 value CHANGES says so out loud.
"""

import logging

import numpy as np
import pytest

from exozippy.samplers._common import (
    DEFAULT_T1_DISPERSION,
    build_rung_populations,
    legacy_dispersion,
    resolve_start_dispersion,
)
from exozippy.samplers.ptde import _make_starts

LADDER = np.array([1.0, 4.0, 25.0, 200.0])


def _seed(n_elem=4):
    return {"a": np.zeros(n_elem)}


def _quad_logp(p):
    return -0.5 * float(np.sum(np.asarray(p["a"], dtype=float) ** 2))


# ---------------------------------------------------------------------------
# Clause 1: 'auto' is 3*sqrt(T), and T=1 is exactly 3.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [None, "auto", "AUTO"])
def test_auto_is_three_root_t_and_t1_is_exactly_three(spec):
    """
    Given no explicit dispersion,
    When it is resolved against a ladder,
    Then every rung gets 3*sqrt(T), and the T=1 rung gets exactly 3.

    The sqrt law is the point: a rung at temperature T targets a
    distribution ~sqrt(T) wider, so this keeps each rung's population
    matched to the width it is dispersed against.
    """
    # ACT
    disp, desc = resolve_start_dispersion(spec, LADDER, n_params=40)

    # ASSERT
    np.testing.assert_allclose(disp, DEFAULT_T1_DISPERSION * np.sqrt(LADDER))
    assert disp[0] == pytest.approx(3.0)
    assert "sqrt(T)" in desc


# ---------------------------------------------------------------------------
# Clause 2: absolute, not a multiplier -- and the scalar IS the control arm.
# ---------------------------------------------------------------------------


def test_a_scalar_is_that_value_at_every_rung():
    """
    Given a single number,
    When it is resolved,
    Then every rung gets exactly that number -- not that number times
      anything.

    This spelling reproduces the pre-8.4.7 sampler, where the hot rungs were
    verbatim copies of the T=1 population and therefore carried T=1's
    dispersion, so it is the control arm for measuring the change.
    """
    # ACT
    disp, desc = resolve_start_dispersion(2.5, LADDER, n_params=40)

    # ASSERT
    np.testing.assert_allclose(disp, [2.5, 2.5, 2.5, 2.5])
    assert "flat" in desc


def test_an_explicit_list_is_taken_per_rung():
    # ARRANGE / ACT
    disp, _ = resolve_start_dispersion([1.0, 2.0, 3.0, 4.0], LADDER, 40)
    # ASSERT
    np.testing.assert_allclose(disp, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Clause 3: nothing changes for callers that do not ask.
# ---------------------------------------------------------------------------


def test_dispersion_none_reproduces_the_legacy_rule_bit_for_bit():
    """
    Given a caller that passes no dispersion at all (de_metropolis has no
      ladder, so it never will),
    When starts are made,
    Then they are bit-identical to passing the old min(sqrt(500/D), 3).

    The old rule is kept as `legacy_dispersion` for exactly this comparison.
    """
    # ARRANGE
    seeds, scales = [_seed()], {"a": np.ones(4)}

    # ACT
    a, _ = _make_starts(
        6, seeds, _quad_logp, np.random.default_rng(3), raw_scales=scales
    )
    b, _ = _make_starts(
        6,
        seeds,
        _quad_logp,
        np.random.default_rng(3),
        raw_scales=scales,
        dispersion=legacy_dispersion(4),
    )

    # ASSERT -- exact equality, not approx
    for sa, sb in zip(a, b):
        np.testing.assert_array_equal(sa["a"], sb["a"])


def test_legacy_rule_is_the_documented_formula():
    assert legacy_dispersion(4) == pytest.approx(3.0)  # cap binds
    assert legacy_dispersion(55) == pytest.approx(3.0)  # still binds
    assert legacy_dispersion(2000) == pytest.approx(0.5)  # cap releases


# ---------------------------------------------------------------------------
# Clauses 4 and 5: the rungs really differ, and rung 0 is the T=1 population.
# ---------------------------------------------------------------------------


def test_each_rung_is_dispersed_at_its_own_value():
    """
    Given a ladder and 'auto' dispersion,
    When the per-rung populations are built,
    Then the measured between-chain spread RISES with temperature.

    This is the property that did not exist before: every rung was a copy of
    T=1, so all four spreads would have been identical.
    """
    # ARRANGE
    rng = np.random.default_rng(11)

    # ACT
    t1, idx, pops, disp, _ = build_rung_populations(
        None,
        None,
        40,
        _quad_logp,
        rng,
        _seed(),
        temperatures=LADDER,
        dispersion_spec=None,
        raw_starts=[_seed()],
        seed_indices=[0],
        raw_scales={"a": np.ones(4)},
    )

    # ASSERT
    assert len(pops) == len(LADDER)
    spreads = [float(np.std([s["a"][0] for s in pop], ddof=1)) for pop in pops]
    assert spreads == sorted(spreads), spreads
    # and the hottest rung is far wider than T=1, not marginally
    assert spreads[-1] > 3.0 * spreads[0]


def test_rung_zero_is_the_returned_t1_population():
    """The ensemble start plots and chain_seed_index describe rung 0, so it
    must BE rung 0 rather than a separate draw."""
    # ARRANGE / ACT
    t1, idx, pops, _, _ = build_rung_populations(
        None,
        None,
        8,
        _quad_logp,
        np.random.default_rng(5),
        _seed(),
        temperatures=LADDER,
        dispersion_spec=None,
        raw_starts=[_seed()],
        seed_indices=[0],
        raw_scales={"a": np.ones(4)},
    )

    # ASSERT
    assert len(idx) == 8
    for a, b in zip(t1, pops[0]):
        np.testing.assert_array_equal(a["a"], b["a"])


# ---------------------------------------------------------------------------
# Clauses 6 and 7: the two ways a bad spec must fail loudly.
# ---------------------------------------------------------------------------


def test_a_wrong_length_list_raises():
    """A short list is consumed positionally, so it would leave the hot rungs
    at a value nobody chose."""
    with pytest.raises(ValueError, match="entries but the ladder"):
        resolve_start_dispersion([1.0, 2.0], LADDER, 40)


@pytest.mark.parametrize("bad", [0.0, -1.0, [1.0, 0.0, 2.0, 3.0], np.nan])
def test_a_non_positive_or_non_finite_dispersion_raises(bad):
    """Zero would start every chain of that rung at one point, which is the
    one thing Rhat cannot diagnose for itself (review 2.4.5)."""
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_start_dispersion(bad, LADDER, 40)


def test_an_uninterpretable_spec_raises():
    with pytest.raises(ValueError, match="must be a number"):
        resolve_start_dispersion("wide-ish", LADDER, 40)


# ---------------------------------------------------------------------------
# Clause 8: initvals cannot be dispersed.
# ---------------------------------------------------------------------------


def test_initvals_are_replicated_to_every_rung():
    """
    Given an explicit initvals list (one start per chain, never scattered),
    When per-rung populations are built,
    Then every rung gets that same list.
    """
    # ARRANGE
    initvals = [{"a": np.full(4, float(i))} for i in range(4)]

    # ACT
    t1, idx, pops, _, desc = build_rung_populations(
        None,
        None,
        4,
        _quad_logp,
        np.random.default_rng(1),
        _seed(),
        temperatures=LADDER,
        dispersion_spec=None,
        initvals=initvals,
    )

    # ASSERT
    assert len(pops) == len(LADDER)
    for pop in pops:
        assert pop is t1
    assert "initvals" in desc


def test_initvals_with_per_rung_dispersion_raises():
    """Asking for different dispersion per rung while supplying one explicit
    start per chain is a contradiction, and is refused rather than ignored."""
    initvals = [{"a": np.zeros(4)} for _ in range(4)]
    with pytest.raises(ValueError, match="nothing to disperse"):
        build_rung_populations(
            None,
            None,
            4,
            _quad_logp,
            np.random.default_rng(1),
            _seed(),
            temperatures=LADDER,
            dispersion_spec=[1.0, 2.0, 3.0, 4.0],
            initvals=initvals,
        )


# ---------------------------------------------------------------------------
# Clause 9: the one model class whose T=1 value changes must say so.
# ---------------------------------------------------------------------------


def test_a_high_dimensional_model_is_warned_that_t1_widened(caplog):
    """
    Given a model with D > 500, where the OLD rule gave sqrt(500/D) < 1,
    When 'auto' resolves to an absolute 3.0 at T=1,
    Then it warns, with the old value and the key that restores it.

    This is the only place the absolute spelling changes the T=1 start, and
    the item's own instruction was "never a blind default change".
    """
    # ARRANGE / ACT
    with caplog.at_level(logging.WARNING):
        disp, _ = resolve_start_dispersion(
            None, LADDER, n_params=2000, log=logging.getLogger("exozippy")
        )

    # ASSERT
    assert disp[0] == pytest.approx(3.0)
    msg = caplog.text
    assert "0.500" in msg
    assert "start_dispersion" in msg
