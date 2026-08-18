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
