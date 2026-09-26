"""The DC2018 sweep generator dropped the +/-90 d t_0 pad, and DC2018-107 paid.

JDE ruled the pad in on 2026-09-04, asked whether to implement it generally:
"the span-based rule should ship" -- span-based over a peak-based
`t_0 +/- few t_E` rule because it needs only the loaded time column, which
exists for every microlensing config at lifecycle stage 1.  It is written up
in configs/DC2018_128_tightpriors.params.yaml item 3.

dc18_sweep_config.py bounded t_0 to the RAW span instead, under a comment
asserting "t_0 CANNOT LIE OUTSIDE THE OBSERVATIONS".  Event 107's answer-key
t_0 is 7.67 d BEFORE its first epoch, so the truth sat outside the
parameter's own support; source.t_0 is sampled through a logit transform onto
[lower, upper], so there was no raw coordinate for it and the fit refused to
start at all.
"""

import importlib
import sys
from pathlib import Path

import pytest

DC18 = Path(__file__).resolve().parents[1] / "examples" / "DC2018"

# DC2018-107, from the answer key and its own light curves.
SPAN_LO = 2458346.505461
SPAN_HI = 2460059.241444
TRUTH_T0_107 = 2458338.831308834


@pytest.fixture(scope="module")
def gen():
    sys.path.insert(0, str(DC18))
    try:
        yield importlib.import_module("dc18_sweep_config")
    finally:
        sys.path.remove(str(DC18))


def test_the_pad_is_the_ruled_ninety_days(gen):
    assert gen.T0_PAD_DAYS == 90.0


def test_t0_bounds_pad_both_sides(gen):
    b = gen.t0_bounds(SPAN_LO, SPAN_HI)
    assert b["lower"] == pytest.approx(SPAN_LO - 90.0)
    assert b["upper"] == pytest.approx(SPAN_HI + 90.0)


def test_event_107s_truth_is_inside_the_padded_support(gen):
    """The regression itself.  Non-vacuity: the first assertion pins that 107
    is genuinely OUTSIDE the raw span, so the second is not trivially true."""
    assert TRUTH_T0_107 < SPAN_LO, (
        "107's truth is no longer outside the raw span, so this test cannot "
        "detect the bug it exists for -- re-derive it from the answer key"
    )
    b = gen.t0_bounds(SPAN_LO, SPAN_HI)
    assert b["lower"] < TRUTH_T0_107 < b["upper"]


def test_the_pad_is_wide_enough_to_be_worth_having(gen):
    """7.67 d of the 90 d pad is used on the worst shipped case, so the margin
    is real rather than a coincidence of this one event."""
    b = gen.t0_bounds(SPAN_LO, SPAN_HI)
    assert TRUTH_T0_107 - b["lower"] > 80.0
