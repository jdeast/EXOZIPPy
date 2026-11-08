"""build_bc_grid never synthesizes a column itself -- it refuses one that
neither the published tables nor the per-user synthesized-column cache
holds.

The on-the-fly hook (make_bc.generate_missing_facility) built a column
from the R = 150 NextGen spectra, ~2 percent wrong, and was removed
(JDE, 2026-09-28).  A band filter whose SED flux a component reads is now
synthesized BEFORE the grid build, from the full-resolution spectra
(models/NextGen/bc_synthesis.py; tests/test_bc_synthesis.py); what reaches
the grid build without a column -- a .sed row's filter, or one where
synthesis does not apply -- is an error naming the request path.
"""

import pytest

from exozippy.components.sed import bc_grid, make_bc
from exozippy.components.sed.bc_grid import build_bc_grid

_MODEL_ROOT = bc_grid.DEFAULT_MODEL_ROOT


def test_a_missing_facility_is_refused_and_nothing_is_synthesized(
    monkeypatch, tmp_path
):
    """
    Given a real SVO filter whose facility has no published table,
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
            ["CFHT/MegaCam.g"], model="NextGen", model_root=_MODEL_ROOT
        )
    msg = str(exc.value)
    assert "CFHT/MegaCam.g" in msg
    assert "not in the published tables" in msg
    assert "does not generate them" in msg
    assert "generate_NextGen_BC_Tables.py" in msg
    assert "bc_synthesis.py" in msg
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
    with pytest.raises(NotImplementedError, match="does not generate them"):
        build_bc_grid(
            ["2MASS/2MASS.J"], model="NextGen", model_root=_MODEL_ROOT
        )


def test_the_hook_is_gone():
    assert not hasattr(make_bc, "generate_missing_facility")
