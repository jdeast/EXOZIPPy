"""The DC2018 evaluator's two convention bugs, both found 2026-09-24 on event 004.

Neither is a crash: each produced a confident WRONG score.  The u_0 one put the
truth in an 8% mode at 16.4 sigma when it was in the 90.6% favourite at 1.04,
on 11 of the 30 static events (every one whose truth u_0 is negative).
"""
import importlib
import re
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
    parallax (conventions.md C23), so the truth table's trajectory-side sign
    is not one the fits carry.  dc18_common has folded it since the comparison
    table existed; dc18_evaluate did not, and scored the sign as a pull.
    """
    ev, common = dc18
    assert "u_0" in ev.ABS_COMPARED


def test_the_two_dc2018_scorers_fold_the_same_keys(dc18):
    """The divergence itself, which is what actually went wrong: two scorers
    for the same truth table disagreed about a convention for months.

    Per docs/testing.md rule 3, this asserts the scan FIRED -- a guard that
    reads another module's source goes vacuous the moment that module spells
    the rule differently, and would then pass while watching nothing.
    """
    ev, common = dc18
    src = Path(common.__file__).read_text()
    folded = set(re.findall(r'truth\["(\w+)"\]\s*=\s*abs\(', src))
    assert folded, (
        "found no `truth[...] = abs(...)` in dc18_common; the rule moved or "
        "was respelled, so this guard is watching nothing -- re-derive it"
    )
    assert folded == ev.ABS_COMPARED
