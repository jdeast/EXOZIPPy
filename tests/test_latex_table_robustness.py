"""
Reachable-but-untested failure paths in the LaTeX table builder.

Each test here reaches a shape the builders do not currently survive, and
each shape is one a real fit can produce even though no shipped component
produces it today.  They are grouped in one file because they share the
stub system/component scaffolding, not because they share a cause.
"""

import re

import numpy as np

from exozippy.components.parameter import Parameter
from exozippy.outputs.latex import build_latex_output
from exozippy.outputs.modes import ModeInfo, ModeReport


class _FakeComp:
    label = "star"


def _fake_system(comp):
    class _Sys:
        name = "test"

        def get_all_components(self):
            return [comp]

    return _Sys()


def _vector_param(label, latex, n_elements, names=None, seed=0):
    """A printable vector Parameter with a posterior, as wrap-up sees it."""
    p = Parameter(
        label=label,
        latex=latex,
        description="desc",
        initval=np.full(n_elements, 1.0),
        lower=0.0,
        upper=10.0,
        shape=(n_elements,),
        names=names,
    )
    rng = np.random.default_rng(seed)
    p.posterior = rng.normal(size=(n_elements, 200)) + 5.0
    return p


def _macro_xref(var_text, table_text):
    """(cited, defined) \\ez... macro names, the compile-time cross-reference."""
    defined = set(re.findall(r"\\providecommand\{\\(ez[A-Za-z]+)\}", var_text))
    cited = set(re.findall(r"\\(ez[A-Za-z]+)", table_text))
    return cited, defined


# ---------------------------------------------------------------------------
# 2.11.2 -- mixed-length printable vectors in one component
# ---------------------------------------------------------------------------


def test_mixed_length_vectors_cite_no_undefined_macro(tmp_path, caplog):
    """
    Given one component holding printable vectors of DIFFERENT lengths
      (three teffs, two masses),
    When build_latex_output writes the table,
    Then the short vector contributes no row for the instances it has no
      element for -- the instance loop runs to the longest vector, so it used
      to emit \\ezstarmasstwo, which the variable file never defines and
      pdflatex rejects as an undefined control sequence at the end of the fit.
    """
    # ARRANGE
    comp = _FakeComp()
    comp.teff = _vector_param("star.teff", "T", 3, names=["A", "B", "C"])
    comp.mass = _vector_param("star.mass", "M", 2, names=["A", "B"], seed=1)
    var_path, table_path = tmp_path / "v.tex", tmp_path / "t.tex"

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.outputs.latex"):
        build_latex_output(
            _fake_system(comp),
            var_filename=str(var_path),
            table_filename=str(table_path),
        )

    # ASSERT
    cited, defined = _macro_xref(var_path.read_text(), table_path.read_text())
    assert not (cited - defined)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "star.mass" in msgs and "star.teff" in msgs


def test_mixed_length_vectors_still_print_every_element_they_have(tmp_path):
    """
    Given the same mismatched pair,
    When the table is written,
    Then nothing the short vector DOES have is lost -- both of its elements
      still get a row, under the right instance sub-heads.
    """
    # ARRANGE
    comp = _FakeComp()
    comp.teff = _vector_param("star.teff", "T", 3, names=["A", "B", "C"])
    comp.mass = _vector_param("star.mass", "M", 2, names=["A", "B"], seed=1)
    var_path, table_path = tmp_path / "v.tex", tmp_path / "t.tex"

    # ACT
    build_latex_output(
        _fake_system(comp),
        var_filename=str(var_path),
        table_filename=str(table_path),
    )

    # ASSERT
    table = table_path.read_text()
    for macro in ("ezstarmasszero", "ezstarmassone"):
        assert f"\\{macro}\\dotfill" in table
    assert "ezstarmasstwo" not in table
    for macro in ("ezstarteffzero", "ezstarteffone", "ezstartefftwo"):
        assert f"\\{macro}\\dotfill" in table


def test_equal_length_vectors_warn_about_nothing(tmp_path, caplog):
    """
    Given the ordinary case -- every printable vector in the component has
      the same length,
    When the table is written,
    Then no mismatch warning is emitted; the guard must be silent on the
      shape every shipped component actually has.
    """
    # ARRANGE
    comp = _FakeComp()
    comp.teff = _vector_param("star.teff", "T", 3, names=["A", "B", "C"])
    comp.mass = _vector_param(
        "star.mass", "M", 3, names=["A", "B", "C"], seed=1
    )
    var_path, table_path = tmp_path / "v.tex", tmp_path / "t.tex"

    # ACT
    with caplog.at_level("WARNING", logger="exozippy.outputs.latex"):
        build_latex_output(
            _fake_system(comp),
            var_filename=str(var_path),
            table_filename=str(table_path),
        )

    # ASSERT
    assert not [r for r in caplog.records if "elements" in r.getMessage()]
    cited, defined = _macro_xref(var_path.read_text(), table_path.read_text())
    assert not (cited - defined)


# ---------------------------------------------------------------------------
# 2.11.3 -- stale mode_summaries surviving a re-run with a different mode count
# ---------------------------------------------------------------------------


def _mode_report(n_modes):
    modes = [
        ModeInfo(
            index=k,
            weight=1.0 / n_modes,
            n_draws=200 // n_modes,
            lp_med=0.0,
            lp_max=0.0,
            delta_lp_max=0.0,
            per_chain_weight=np.array([1.0 / n_modes]),
        )
        for k in range(n_modes)
    ]
    return ModeReport(
        labels=np.zeros((1, 200), dtype=int),
        modes=modes,
        n_valid=200,
        n_invalid=0,
        n_unassigned=0,
        provenance="occupancy",
        weights_reliable=True,
        n_transitions=20,
        feature_vars=["a_raw"],
    )


def _scalar_param(values, label="star.teff", latex="T"):
    p = Parameter(
        label=label,
        latex=latex,
        description="desc",
        initval=5000.0,
        lower=3000.0,
        upper=7000.0,
    )
    p.posterior = np.asarray(values, dtype=float)
    return p


def _write_table(system, tmp_path, tag, mode_report):
    var_path = tmp_path / f"v{tag}.tex"
    table_path = tmp_path / f"t{tag}.tex"
    build_latex_output(
        system,
        var_filename=str(var_path),
        table_filename=str(table_path),
        mode_report=mode_report,
    )
    return var_path.read_text(), table_path.read_text()


def test_growing_mode_count_defines_every_cited_mode_macro(tmp_path):
    """
    Given a System whose parameters already carry two-mode summaries from an
      earlier report, re-reported with a THREE-mode report (the GUI and
      exozippy-modes both re-report a live System),
    When the table is written,
    Then the third mode's macro is defined -- _ensure_mode_summaries used to
      early-return on a non-None mode_summaries without checking its length,
      so the table cited \\ez...modethree against a two-entry list.
    """
    # ARRANGE
    comp = _FakeComp()
    comp.teff = _scalar_param(np.linspace(1.0, 3.0, 200))
    system = _fake_system(comp)
    system.mode_labels = np.repeat([0, 1], 100)
    _write_table(system, tmp_path, "2", _mode_report(2))

    # ACT
    system.mode_labels = np.repeat([0, 1, 2], [67, 67, 66])
    var_text, table_text = _write_table(system, tmp_path, "3", _mode_report(3))

    # ASSERT
    cited, defined = _macro_xref(var_text, table_text)
    assert not (cited - defined)
    assert "modethree" in var_text


def test_shrinking_mode_count_reports_the_new_split(tmp_path):
    """
    Given a System re-reported with FEWER modes than last time,
    When the table is written,
    Then the per-mode values are recomputed against the new labels rather
      than being the previous run's splits relabelled -- the silent half of
      the same early return, and the dangerous one, since it reports numbers
      that were never in this report at all.
    """
    # ARRANGE
    comp = _FakeComp()
    comp.teff = _scalar_param(np.repeat([0.0, 4.0, 8.0], [67, 67, 66]))
    system = _fake_system(comp)
    system.mode_labels = np.repeat([0, 1, 2], [67, 67, 66])
    _write_table(system, tmp_path, "a", _mode_report(3))
    stale = [s.median for s in comp.teff.mode_summaries]
    assert stale == [0.0, 4.0, 8.0]

    # ACT: the second report merges the first two peaks into one mode
    system.mode_labels = np.repeat([0, 1], [134, 66])
    var_text, _ = _write_table(system, tmp_path, "b", _mode_report(2))

    # ASSERT
    assert [s.median for s in comp.teff.mode_summaries] == [2.0, 8.0]
    assert "modethree" not in var_text
