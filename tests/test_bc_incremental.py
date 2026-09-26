"""
Incremental BC tables and selective reads (components/sed/bc_grid.py):
  - write_bc_table merges cell by cell and may add grid nodes
  - complete_axes / peek_grid_axes report what a set of filters COVERS
  - bc_nodes_to_compute lists only the (node, filter) cells still missing
  - build_bc_grid reads only the requested columns and the bracketed rows
"""

import numpy as np
import pandas as pd
import pytest

from exozippy.components.sed import bc_grid
from exozippy.components.sed.bc_grid import (
    BC_PARAM_COLS,
    bc_nodes_to_compute,
    build_bc_grid,
    complete_axes,
    peek_grid_axes,
    read_bc_meta,
    read_bc_table,
    write_bc_table,
)

_TEFF = (5000.0, 6000.0)
_LOGG = (4.0, 4.5)
_FEH = (-0.5, 0.0)


def _table(cols, av, value=None):
    """A complete toy table on (_TEFF, _LOGG, _FEH, av). BC = the node's
    own function (teff/1000 + logg + 10*feh + 100*Av) plus 1000*i per
    column, so a value in the wrong cell cannot pass for the right one."""
    rows = [
        (t, g, f, 0.0, a, 3.1)
        for t in _TEFF
        for g in _LOGG
        for f in _FEH
        for a in av
    ]
    df = pd.DataFrame(rows, columns=BC_PARAM_COLS)
    base = df["teff"] / 1000 + df["logg"] + 10 * df["feh"] + 100 * df["Av"]
    for i, col in enumerate(cols):
        df[col] = base + 1000 * i if value is None else value
    return df


def _meta(cols, **extra):
    return {c: {"svo_id": f"Fa/Fa.{c}", **extra} for c in cols}


@pytest.fixture
def ragged_root(tmp_path):
    """Toy model 'Toy', facility 'Fa': Fa_X computed on Av = 0 .. 0.2,
    Fa_Y only on Av = 0 .. 0.1 (the table was extended for X alone)."""
    path = tmp_path / "Toy" / "BCs" / "Fa.bc.parquet"
    write_bc_table(
        _table(["Fa_X", "Fa_Y"], av=(0.0, 0.05, 0.1)),
        path,
        _meta(["Fa_X", "Fa_Y"]),
    )
    write_bc_table(
        _table(["Fa_X"], av=(0.2,)),
        path,
        _meta(["Fa_X"], note="extension"),
        allow_new_nodes=True,
    )
    return tmp_path, path


# ---------------------------------------------------------------------------
# Section 1 -- writing
# ---------------------------------------------------------------------------


def test_extending_a_column_keeps_old_cells_and_leaves_others_nan(
    ragged_root,
):
    """
    Given a table extended to a new Av value for Fa_X only,
    When it is read back,
    Then every original cell is unchanged, Fa_X has the new Av values,
    Fa_Y is NaN there (not computed), and Fa_X's metadata was UPDATED by
    the extension rather than replaced by it.
    """
    # ARRANGE
    _, path = ragged_root
    expected = _table(["Fa_X", "Fa_Y"], av=(0.0, 0.05, 0.1, 0.2))

    # ACT
    df = read_bc_table(path).merge(
        expected, on=["teff", "logg", "feh", "Av"], suffixes=("", "_exp")
    )

    # ASSERT
    assert len(df) == len(expected)
    np.testing.assert_array_equal(df["Fa_X"], df["Fa_X_exp"])
    old = df["Av"] < 0.15
    np.testing.assert_array_equal(df.loc[old, "Fa_Y"], df.loc[old, "Fa_Y_exp"])
    assert df.loc[~old, "Fa_Y"].isna().all()
    meta = read_bc_meta(path)["filters"]
    assert meta["Fa_X"] == {"svo_id": "Fa/Fa.Fa_X", "note": "extension"}
    assert meta["Fa_Y"] == {"svo_id": "Fa/Fa.Fa_Y"}


def test_nan_cells_in_new_data_never_overwrite_existing_values(ragged_root):
    """
    Given a table holding Fa_Y on Av = 0 .. 0.1,
    When a frame covering Av = 0 .. 0.2 is merged in whose Fa_Y is NaN
    except at Av = 0.2,
    Then the Av <= 0.1 cells keep their values and Av = 0.2 is filled.
    """
    # ARRANGE
    _, path = ragged_root
    new = _table(["Fa_Y"], av=(0.0, 0.05, 0.1, 0.2), value=-1.0)
    new.loc[new["Av"] < 0.15, "Fa_Y"] = np.nan

    # ACT
    write_bc_table(new, path, _meta(["Fa_Y"]))
    df = read_bc_table(path)

    # ASSERT
    assert (df.loc[df["Av"] == 0.2, "Fa_Y"] == -1.0).all()
    assert (df.loc[df["Av"] < 0.15, "Fa_Y"] != -1.0).all()
    assert df["Fa_Y"].notna().all()


def test_a_replaced_column_resumes_without_keeping_old_pipeline_cells(
    tmp_path,
):
    """
    Given a column written by another pipeline (legacy metadata),
    When a new pipeline checkpoints only the first half of its rebuild
    (the Av = 0 slab) with replace_columns, and is then interrupted,
    Then the unfinished half is NaN rather than legacy values, the
    metadata is the new pipeline's alone, and the work list for a resumed
    run is exactly the unfinished half.
    """
    # ARRANGE
    path = tmp_path / "Fa.bc.parquet"
    write_bc_table(
        _table(["A"], av=(0.0, 1.0), value=-9.0),
        path,
        {"A": {"svo_id": "Fa/Fa.A", "generator": "legacy", "src": "txt"}},
    )
    first_half = _table(["A"], av=(0.0,))
    new_meta = {"A": {"svo_id": "Fa/Fa.A", "generator": "new"}}

    # ACT
    write_bc_table(first_half, path, new_meta, replace_columns=["A"])
    df = read_bc_table(path)
    todo = bc_nodes_to_compute(
        path,
        {"teff": _TEFF, "logg": _LOGG, "feh": _FEH, "av": (0.0, 1.0)},
        ["A"],
        reusable=lambda meta: meta.get("generator") == "new",
    )

    # ASSERT
    assert (df.loc[df["Av"] == 0.0, "A"] != -9.0).all()
    assert df.loc[df["Av"] == 1.0, "A"].isna().all()
    assert read_bc_meta(path)["filters"]["A"] == new_meta["A"]
    assert set(todo["Av"]) == {1.0} and todo["A"].all()


# ---------------------------------------------------------------------------
# Section 2 -- coverage
# ---------------------------------------------------------------------------


def test_peek_grid_axes_reports_the_coverage_of_the_requested_filters(
    ragged_root,
):
    """
    Given Fa_X covering Av to 0.2 and Fa_Y only to 0.1,
    When peek_grid_axes is asked about X alone, about X and Y, and about
    no filter in particular,
    Then X alone reaches 0.2 and the other two stop at 0.1 (the part
    every filter involved has actually been computed on).
    """
    # ARRANGE
    root, _ = ragged_root

    # ACT
    x_only = peek_grid_axes("Toy", root, filters=["Fa/Fa.X"])
    both = peek_grid_axes("Toy", root, filters=["Fa/Fa.X", "Fa/Fa.Y"])
    anyf = peek_grid_axes("Toy", root)

    # ASSERT
    np.testing.assert_array_equal(x_only["av_pts"], [0.0, 0.05, 0.1, 0.2])
    np.testing.assert_array_equal(both["av_pts"], [0.0, 0.05, 0.1])
    np.testing.assert_array_equal(anyf["av_pts"], [0.0, 0.05, 0.1])
    np.testing.assert_array_equal(x_only["teff_pts"], _TEFF)


def test_complete_axes_trims_a_partly_computed_edge_slab():
    """
    Given a column whose last Av slab was only half computed (an
    interrupted run),
    When complete_axes is asked for its coverage,
    Then that slab is dropped and the other axes are untouched.
    """
    # ARRANGE
    df = _table(["A"], av=(0.0, 1.0, 2.0))
    df.loc[(df["Av"] == 2.0) & (df["teff"] == 6000.0), "A"] = np.nan

    # ACT
    axes = complete_axes(df, ["A"])

    # ASSERT
    np.testing.assert_array_equal(axes["av_pts"], [0.0, 1.0])
    np.testing.assert_array_equal(axes["teff_pts"], _TEFF)


def test_complete_axes_raises_on_an_interior_hole():
    """
    Given a column on a 3x3x3x3 grid missing only its centre node,
    When complete_axes is asked for its coverage,
    Then it raises rather than returning a box with a hole in it (every
    edge slab is complete, so no trimming can remove the hole).
    """
    # ARRANGE
    ax = {
        "teff": (5000.0, 6000.0, 7000.0),
        "logg": (4.0, 4.5, 5.0),
        "feh": (-0.5, 0.0, 0.3),
        "Av": (0.0, 1.0, 2.0),
    }
    mesh = np.meshgrid(*ax.values(), indexing="ij")
    df = pd.DataFrame({k: m.ravel() for k, m in zip(ax, mesh)})
    df["A"] = 1.0
    centre = (
        (df["teff"] == 6000.0)
        & (df["logg"] == 4.5)
        & (df["feh"] == 0.0)
        & (df["Av"] == 1.0)
    )
    df.loc[centre, "A"] = np.nan

    # ACT & ASSERT
    with pytest.raises(ValueError, match="interior"):
        complete_axes(df, ["A"])


def test_complete_axes_never_collapses_a_two_point_axis():
    """
    Given a column on a two-point logg axis missing one node at logg=4.0,
    When complete_axes is asked for its coverage,
    Then it raises rather than trimming logg to a single point (which the
    interpolator could not use) -- the Av slab with the hole is the one
    that may go, and here Av has two points too.
    """
    # ARRANGE
    df = _table(["A"], av=(0.0, 1.0))
    hole = (
        (df["teff"] == 5000.0)
        & (df["logg"] == 4.0)
        & (df["feh"] == -0.5)
        & (df["Av"] == 1.0)
    )
    df.loc[hole, "A"] = np.nan

    # ACT & ASSERT
    with pytest.raises(ValueError, match="interior"):
        complete_axes(df, ["A"])


# ---------------------------------------------------------------------------
# Section 3 -- the work list for a generator
# ---------------------------------------------------------------------------


def test_nodes_to_compute_lists_only_missing_cells(ragged_root):
    """
    Given Fa_X on Av to 0.2, Fa_Y to 0.1, and a target grid extended to
    Av = 0.4 with a new filter Fa_Z,
    When bc_nodes_to_compute builds the work list,
    Then X is due at Av = 0.4 only, Y at 0.2 and 0.4, Z everywhere, and
    no node with nothing due is listed.
    """
    # ARRANGE
    _, path = ragged_root
    av = (0.0, 0.05, 0.1, 0.2, 0.4)
    axes = {"teff": _TEFF, "logg": _LOGG, "feh": _FEH, "av": av}
    n_per_av = len(_TEFF) * len(_LOGG) * len(_FEH)

    # ACT
    todo = bc_nodes_to_compute(path, axes, ["Fa_X", "Fa_Y", "Fa_Z"])

    # ASSERT
    assert len(todo) == n_per_av * len(av)  # Z is due at every node
    assert set(todo.loc[todo["Fa_X"], "Av"]) == {0.4}
    assert set(todo.loc[todo["Fa_Y"], "Av"]) == {0.2, 0.4}
    assert todo["Fa_Z"].all()
    assert todo["Fa_X"].sum() == n_per_av


def test_nodes_to_compute_is_empty_once_everything_is_computed(ragged_root):
    """
    Given a table that holds Fa_X on the whole target grid,
    When bc_nodes_to_compute is asked for Fa_X,
    Then the work list is empty -- and becomes the whole grid when the
    column is declared not reusable (a different pipeline wrote it).
    """
    # ARRANGE
    _, path = ragged_root
    axes = {"teff": _TEFF, "logg": _LOGG, "feh": _FEH, "av": (0.0, 0.2)}

    # ACT
    reused = bc_nodes_to_compute(path, axes, ["Fa_X"])
    rejected = bc_nodes_to_compute(
        path, axes, ["Fa_X"], reusable=lambda meta: "note" not in meta
    )

    # ASSERT
    assert reused.empty
    assert len(rejected) == 2 * len(_TEFF) * len(_LOGG) * len(_FEH)


# ---------------------------------------------------------------------------
# Section 4 -- selective reads
# ---------------------------------------------------------------------------


def _spy_reads(monkeypatch):
    calls = []
    real = pd.read_parquet

    def spy(path, *args, **kwargs):
        calls.append(kwargs)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(bc_grid.pd, "read_parquet", spy)
    return calls


def test_build_bc_grid_reads_only_requested_columns_and_bracketed_rows(
    tmp_path, monkeypatch
):
    """
    Given a facility table with three filter columns on Av = 0 .. 0.2,
    When build_bc_grid is asked for ONE filter with an Av upper limit of
    0.09,
    Then the grid holds Av = 0, 0.05, 0.1 (the limit plus the point that
    brackets it), the values are the right node's, no read ever asked for
    the other two columns, and the data read pushed the Av range down.
    """
    # ARRANGE
    path = tmp_path / "Toy" / "BCs" / "Fa.bc.parquet"
    cols = ["Fa_X", "Fa_Y", "Fa_Z"]
    write_bc_table(_table(cols, av=(0.0, 0.05, 0.1, 0.2)), path, _meta(cols))
    calls = _spy_reads(monkeypatch)

    # ACT
    grid = build_bc_grid(
        ["Fa/Fa.Y"],
        model="Toy",
        model_root=tmp_path,
        bounds={"av": (0.0, 0.09)},
    )

    # ASSERT
    np.testing.assert_array_equal(grid["av_pts"], [0.0, 0.05, 0.1])
    t, g, f, a = np.meshgrid(
        grid["teff_pts"],
        grid["logg_pts"],
        grid["feh_pts"],
        grid["av_pts"],
        indexing="ij",
    )
    np.testing.assert_allclose(
        grid["bc_values"][..., 0], t / 1000 + g + 10 * f + 100 * a + 1000
    )
    assert calls, "the spy saw no reads -- the test would pass vacuously"
    for kw in calls:
        assert kw["columns"] is not None
        assert "Fa_X" not in kw["columns"] and "Fa_Z" not in kw["columns"]
    data_reads = [kw for kw in calls if "Fa_Y" in kw["columns"]]
    assert len(data_reads) == 1
    assert ("Av", "<=", pytest.approx(0.1, abs=1e-5)) in data_reads[0][
        "filters"
    ]


def test_build_bc_grid_uses_the_extended_part_only_where_every_filter_has_it(
    ragged_root,
):
    """
    Given Fa_X on Av to 0.2 and Fa_Y only to 0.1,
    When build_bc_grid loads X alone and then X with Y, unbounded,
    Then X alone spans Av to 0.2 and X with Y stops at 0.1.
    """
    # ARRANGE
    root, _ = ragged_root

    # ACT
    x_only = build_bc_grid(["Fa/Fa.X"], model="Toy", model_root=root)
    both = build_bc_grid(["Fa/Fa.X", "Fa/Fa.Y"], model="Toy", model_root=root)

    # ASSERT
    assert x_only["av_pts"][-1] == 0.2
    assert both["av_pts"][-1] == 0.1
    assert not np.isnan(both["bc_values"]).any()


def test_build_bc_grid_raises_when_bounds_exceed_a_filters_coverage(
    ragged_root,
):
    """
    Given Fa_Y computed only to Av = 0.1,
    When a fit bounded to Av <= 0.2 asks for it,
    Then build_bc_grid raises naming the axis, instead of extrapolating
    the missing Av range.
    """
    # ARRANGE
    root, _ = ragged_root

    # ACT & ASSERT
    with pytest.raises(ValueError, match="cover av"):
        build_bc_grid(
            ["Fa/Fa.Y"],
            model="Toy",
            model_root=root,
            bounds={"av": (0.0, 0.2)},
        )
