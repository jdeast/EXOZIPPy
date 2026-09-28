"""build_bc_grid refuses a filter the shipped tables do not cover -- it no
longer synthesizes one from the plotting spectra.

The on-the-fly hook (make_bc.generate_missing_facility) built a column
from the R = 150 NextGen spectra, ~2 percent wrong.  With the shipped
tables now from the full-resolution spectra (PR #335) that column would
sit under the SED's error scale unreported, so a missing filter is an
error naming the request path (JDE, 2026-09-28).
"""

import pytest

from exozippy.components.sed import bc_grid, make_bc
from exozippy.components.sed.bc_grid import build_bc_grid

_MODEL_ROOT = bc_grid.DEFAULT_MODEL_ROOT


def test_a_missing_facility_is_refused_and_nothing_is_synthesized(
    monkeypatch, tmp_path
):
    """
    Given a real SVO filter whose facility has no shipped table,
    When the BC grid is built,
    Then it raises naming the filter and the request path, and make_bc's
      table writer is never called.
    """
    called = []
    monkeypatch.setattr(
        make_bc, "make_bc_tables", lambda *a, **k: called.append(a)
    )
    with pytest.raises(NotImplementedError) as exc:
        build_bc_grid(
            ["Paranal/VISTA.Ks"], model="NextGen", model_root=_MODEL_ROOT
        )
    msg = str(exc.value)
    assert "Paranal/VISTA.Ks" in msg
    assert "NOT generated on the fly" in msg
    assert "generate_NextGen_BC_Tables.py" in msg
    assert called == []
    assert (tmp_path / "nothing").exists() is False


def test_a_missing_column_in_an_existing_facility_is_refused_too(monkeypatch):
    """
    Given a facility that ships (2MASS) and a column it lacks,
    When the BC grid is built,
    Then it raises the same refusal rather than merging a synthesized column
      into the shipped table.
    """
    monkeypatch.setattr(
        bc_grid,
        "bc_table_filter_columns",
        lambda path: ["2MASS_H", "2MASS_Ks"],
    )
    with pytest.raises(NotImplementedError, match="NOT generated on the fly"):
        build_bc_grid(
            ["2MASS/2MASS.J"], model="NextGen", model_root=_MODEL_ROOT
        )


def test_the_hook_is_gone():
    assert not hasattr(make_bc, "generate_missing_facility")
