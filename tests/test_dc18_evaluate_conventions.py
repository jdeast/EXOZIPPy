"""The DC2018 evaluator's two convention bugs, both found 2026-09-24 on event 004.

Neither is a crash: each produced a confident WRONG score.  The u_0 one put the
truth in an 8% mode at 16.4 sigma when it was in the 90.6% favourite at 1.04,
on 11 of the 30 static events (every one whose truth u_0 is negative).
"""

import importlib
import sys
from pathlib import Path

import pytest

DC18 = Path(__file__).resolve().parents[1] / "examples" / "DC2018"


@pytest.fixture(scope="module")
def dc18():
    sys.path.insert(0, str(DC18))
    try:
        yield (
            importlib.import_module("dc18_evaluate"),
            importlib.import_module("dc18_common"),
        )
    finally:
        sys.path.remove(str(DC18))


def test_lnz_parser_keeps_the_value_off_the_error_bar(dc18, tmp_path):
    """`lnZ=92499.73+/-0.41` must parse as 92499.73, not raise on '92499.73+'.

    A bare [-+0-9.eE]+ swallows the leading '+' of the '+/-'.  The evaluator
    crashed on every multimodal event whose bridge sampling SUCCEEDED, so the
    failure was invisible on the unimodal runs that made up most of the sweep.
    """
    ev, _ = dc18
    modes = tmp_path / "modes.txt"
    modes.write_text(
        "weight provenance: occupancy\n"
        "  - bridge-sampling evidence per mode -- mode 1: lnZ=92499.73+/-0.41 "
        "(re2=0.168); mode 2: REFUSED [re2-too-large]\n"
    )
    out = ev._parse_modes_txt(str(modes))
    assert out["lnZ"][0] == pytest.approx(92499.73)


def test_u_0_is_compared_in_absolute_value(dc18):
    """(u_0, alpha) -> -(u_0, alpha) is EXACT for a static binary with no
    parallax (conventions.md C23), so the mode-aware evaluator merges the
    mirror pair by folding u_0.  It did not until 2026-09-24, and scored the
    sign as a pull.
    """
    ev, common = dc18
    assert ev.ABS_COMPARED == {"u_0"}


@pytest.mark.parametrize("u0_truth", [0.1418, -0.1418])
@pytest.mark.parametrize("u0_fit", [0.1401, -0.1401])
def test_the_two_dc2018_scorers_agree_on_the_u_0_mirror(
    dc18, u0_truth, u0_fit
):
    """The divergence that actually went wrong on 2026-09-24: two scorers for
    the same truth table disagreed about the mirror for months.

    dc18_common now compares u_0 SIGNED in the fit's branch
    (mirror_branch_truth; the key's sign maps by the identity, C22) while
    dc18_evaluate folds |u_0|.  For u_0 the two must give the same residual
    on every sign combination, and dc18_common must REPORT a mirrored
    comparison, not hide it.
    """
    ev, common = dc18
    u0_t, alpha_t, mirrored = common.mirror_branch_truth(
        u0_truth, 300.0, u0_fit
    )
    assert abs(u0_fit - u0_t) == pytest.approx(
        abs(abs(u0_fit) - abs(u0_truth))
    )
    assert u0_t * u0_fit > 0
    assert mirrored == (u0_fit * u0_truth < 0)
    assert alpha_t == pytest.approx(60.0 if mirrored else 300.0)
