"""The NUMBERS in <prefix>_results.csv's two error columns (review 1.11.4).

``up_err`` must carry err_PLUS and ``low_err`` err_MINUS, in BOTH layouts.
The defect this file pins shipped because every existing assertion about
this file checked the column NAMES or that the cells were non-empty -- which
is docs/testing.md's "covering a code path is not testing its numbers"
exactly.  ``PosteriorSummary.format`` returns (median, err_minus, err_plus)
and ``outputs/latex.py`` unpacked it as ``med, ep, em``, so a
``10.0 +5.0 -1.0`` posterior was published as ``+1.0 -5.0``.

So: deliberately ASYMMETRIC summaries, with the two errors differing by
enough that a swap cannot round into agreement, asserted against the parsed
CSV BY COLUMN NAME taken from the file's own header (never by position, so a
new column cannot quietly re-point these assertions).  The plain and
mode-keyed layouts are separate call sites in ``build_csv_output`` and are
asserted separately rather than one being assumed to follow the other.

The cross-check that carries no constant of its own: ``latex_value`` reads
the summary's fields BY NAME and was never affected, so the CSV cells are
also checked against ``^{+...}_{-...}`` as that renderer spells them.
"""

import csv

import numpy as np
import pytest

from exozippy.components.parameter import Parameter, PosteriorSummary
from exozippy.outputs.latex import build_csv_output
from exozippy.outputs.modes import ModeInfo, ModeReport

# Distinct by a factor of 5 in one direction and 35 in the other: a swap
# cannot pass as rounding, and nor can a symmetric stand-in.
PLAIN = PosteriorSummary(median=10.0, err_minus=1.0, err_plus=5.0)
MODE_A = PosteriorSummary(median=2.0, err_minus=0.02, err_plus=0.7)
MODE_B = PosteriorSummary(median=7.0, err_minus=0.4, err_plus=0.011)


class _FakeComp:
    label = "toy"


def _fake_system(comp):
    class _Sys:
        name = "test"

        def get_all_components(self):
            return [comp]

    return _Sys()


def _toy_param():
    return Parameter(
        label="toy.x",
        latex="x",
        description="toy parameter",
        initval=10.0,
        lower=0.0,
        upper=100.0,
    )


def _read_rows(path):
    """Parse the CSV using the fieldnames from ITS OWN header comment.

    The header is written as '# parname, value, up_err, low_err'; keying the
    assertions off it means this test reads the columns the way a consumer
    does (by name) and survives a column being added anywhere in the row.
    """
    lines = path.read_text().splitlines()
    assert lines[0].startswith("# ")
    fieldnames = [c.strip() for c in lines[0][2:].split(",")]
    data = [ln for ln in lines if not ln.lstrip().startswith("#")]
    return fieldnames, list(csv.DictReader(data, fieldnames=fieldnames))


def _two_mode_report():
    """A minimal 2-mode report: enough for the mode-keyed layout to fire."""
    modes = [
        ModeInfo(
            index=k,
            weight=w,
            n_draws=100,
            lp_med=0.0,
            lp_max=0.0,
            delta_lp_max=0.0,
            per_chain_weight=np.ones(2),
            weight_err=0.01,
        )
        for k, w in enumerate((0.7, 0.3))
    ]
    return ModeReport(
        labels=np.zeros((2, 100), dtype=int),
        modes=modes,
        n_valid=200,
        n_invalid=0,
        n_unassigned=0,
        provenance="occupancy",
        weights_reliable=True,
        n_transitions=10,
        feature_vars=["toy.x_raw"],
    )


def _assert_cells(row, summary):
    """The one assertion this file exists for, plus its no-constant twin."""
    assert row["value"] == str(summary.median)
    assert float(row["up_err"]) == pytest.approx(summary.err_plus)
    assert float(row["low_err"]) == pytest.approx(summary.err_minus)
    # Not the same statement twice: latex_value is the path that reads the
    # summary's fields by NAME, so agreeing with it pins the CSV against a
    # renderer that could not have made the positional mistake.
    assert (
        summary.latex_value(sigfigs=2)
        == f"{row['value']}^{{+{row['up_err']}}}_{{-{row['low_err']}}}"
    )


def test_plain_layout_up_err_is_err_plus(tmp_path):
    """
    Given an asymmetric posterior summary (10.0 +5.0 -1.0),
    When build_csv_output writes the PLAIN four-column results CSV,
    Then the 'up_err' cell holds 5.0 and 'low_err' holds 1.0 -- the columns
      mean what their names say.
    """
    # ARRANGE
    comp = _FakeComp()
    comp.x = _toy_param()
    comp.x.summary = PLAIN
    csv_path = tmp_path / "results.csv"

    # ACT
    build_csv_output(_fake_system(comp), str(csv_path))

    # ASSERT
    fieldnames, rows = _read_rows(csv_path)
    assert fieldnames == ["parname", "value", "up_err", "low_err"]
    (row,) = [r for r in rows if r["parname"] == "toy.x"]
    _assert_cells(row, PLAIN)


def test_mode_keyed_layout_up_err_is_err_plus(tmp_path):
    """
    Given a two-mode report and per-mode summaries that are asymmetric in
      OPPOSITE directions (mode 1 skewed up, mode 2 skewed down),
    When build_csv_output writes the mode-keyed seven-column results CSV,
    Then the 'all' row and BOTH per-mode rows put err_plus in 'up_err' --
      the per-mode branch is a second call site, so it is asserted here and
      not assumed to follow the plain one.

    The opposite skews matter: a swap that happened to be tested only
    against same-signed skews could hide in the direction both rows share.
    """
    # ARRANGE
    comp = _FakeComp()
    comp.x = _toy_param()
    comp.x.summary = PLAIN
    comp.x.mode_summaries = [MODE_A, MODE_B]
    report = _two_mode_report()
    csv_path = tmp_path / "results.csv"

    # ACT
    build_csv_output(_fake_system(comp), str(csv_path), mode_report=report)

    # ASSERT
    fieldnames, rows = _read_rows(csv_path)
    assert fieldnames == [
        "parname",
        "mode",
        "weight",
        "weight_err",
        "value",
        "up_err",
        "low_err",
    ]
    by_mode = {r["mode"]: r for r in rows if r["parname"] == "toy.x"}
    assert set(by_mode) == {"all", "1", "2"}
    _assert_cells(by_mode["all"], PLAIN)
    _assert_cells(by_mode["1"], MODE_A)
    _assert_cells(by_mode["2"], MODE_B)


def test_format_returns_named_fields_so_it_cannot_be_mis_unpacked():
    """
    Given PosteriorSummary.format's three-element return,
    When a caller asks for a field by NAME,
    Then the names bind to the documented order (median, err_minus,
      err_plus) -- and the value is still a plain tuple positionally, so the
      NamedTuple hardening is backward compatible with every existing
      unpack.
    """
    formatted = PLAIN.format(sigfigs=2)

    assert (formatted.median, formatted.err_minus, formatted.err_plus) == (
        "10.0",
        "1.0",
        "5.0",
    )
    assert tuple(formatted) == ("10.0", "1.0", "5.0")
    med, em, ep = formatted
    assert (med, em, ep) == ("10.0", "1.0", "5.0")
