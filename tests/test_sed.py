"""
Unit tests for the SED component:
  - bc_grid.py  : BC table I/O, grid assembly, slicing, filter name resolution
  - physics.py  : registered physics functions (absbolmag, absmag, appmag, bc)
  - sed.py      : __init__ grid-bound injection, load_data, register_parameters
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest
import yaml

import exozippy.components.sed.physics  # registers calc_absbolmag etc.
from exozippy.components.sed.bc_grid import (
    BC_PARAM_COLS,
    DEFAULT_MODEL_ROOT,
    RegularGridInterpolator,
    _range_indices,
    bc_filter_columns,
    build_bc_grid,
    find_bc_table,
    peek_grid_axes,
    read_bc_table,
    resolve_filter_name,
    slice_bc,
    write_bc_table,
)
from exozippy.physics_registry import PHYSICS_REGISTRY

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MODEL_ROOT = DEFAULT_MODEL_ROOT
_2MASS_NEXTGEN = _MODEL_ROOT / "NextGen" / "BCs" / "2MASS.bc.parquet"


# A minimal grid_dict mirroring what build_bc_grid / slice_bc expect.
def _make_grid_dict(axes):
    return {
        "model": "NextGen",
        "grid": {k: list(v) for k, v in axes.items()},
    }


# ---------------------------------------------------------------------------
# Section 1 — BC table I/O (parquet)
# ---------------------------------------------------------------------------


def _tiny_bc_table(filter_cols, value=1.0, teffs=(5000.0, 6000.0)):
    """A complete 2x2x2x2 (teff, logg, feh, Av) BC table."""
    rows = [
        (t, g, f, 0.0, a, 3.1)
        for t in teffs
        for g in (4.0, 4.5)
        for f in (-0.5, 0.0)
        for a in (0.0, 1.0)
    ]
    df = pd.DataFrame(rows, columns=BC_PARAM_COLS)
    for i, col in enumerate(filter_cols):
        df[col] = value + i
    return df


def test_read_bc_table_returns_dataframe_with_correct_columns():
    """
    Given the shipped 2MASS NextGen BC table,
    When read_bc_table is called,
    Then the returned DataFrame should carry the stellar-parameter columns
    and exactly the three 2MASS filter columns.
    """
    # ARRANGE / ACT
    df = read_bc_table(_2MASS_NEXTGEN)

    # ASSERT
    for col in ("teff", "logg", "feh", "Av"):
        assert col in df.columns
    assert set(bc_filter_columns(df)) == {"2MASS_J", "2MASS_H", "2MASS_Ks"}


def test_read_bc_table_teff_column_is_linear_not_log():
    """
    Given the shipped 2MASS NextGen BC table,
    When read_bc_table is called,
    Then the 'teff' column should contain linear temperature values
    (not log10 values), so all entries should be greater than 100.
    """
    # ARRANGE / ACT
    df = read_bc_table(_2MASS_NEXTGEN)

    # ASSERT
    assert (df["teff"] > 100).all(), (
        "teff column contains values <= 100; looks like log10(teff) was stored"
    )


def test_read_bc_table_carries_per_filter_metadata():
    """
    Given the shipped 2MASS NextGen BC table,
    When read_bc_table is called,
    Then df.attrs['meta']['filters'] should name the SVO id of every
    filter column.
    """
    # ARRANGE / ACT
    meta = read_bc_table(_2MASS_NEXTGEN).attrs["meta"]

    # ASSERT
    assert meta["filters"]["2MASS_J"]["svo_id"] == "2MASS/2MASS.J"
    assert set(meta["filters"]) == {"2MASS_J", "2MASS_H", "2MASS_Ks"}


def test_find_bc_table_raises_not_implemented_for_missing_facility():
    """
    Given the NextGen model and a facility with no BC table,
    When find_bc_table is called,
    Then NotImplementedError should be raised (the error the SED and the
    auto-generator key on).
    """
    # ACT & ASSERT
    with pytest.raises(NotImplementedError):
        find_bc_table(_MODEL_ROOT, "NextGen", "NoSuchFacility")


def test_find_bc_table_raises_file_not_found_for_missing_model():
    """
    Given a model name with no BC tree,
    When find_bc_table is called,
    Then FileNotFoundError should be raised.
    """
    # ACT & ASSERT
    with pytest.raises(FileNotFoundError):
        find_bc_table(_MODEL_ROOT, "FakeModel_XYZ", "2MASS")


def test_write_bc_table_merge_keeps_other_columns_and_their_metadata(tmp_path):
    """
    Given a table with columns A and B,
    When only B is rewritten with new values and a new column C is added,
    Then A and its metadata are unchanged, B and C take the new values,
    and the metadata of all three is present.
    """
    # ARRANGE
    path = tmp_path / "Fac.bc.parquet"
    write_bc_table(
        _tiny_bc_table(["A", "B"], value=1.0),
        path,
        {"A": {"svo_id": "Fac/X.A"}, "B": {"svo_id": "Fac/X.B"}},
    )
    new = _tiny_bc_table(["B", "C"], value=10.0)

    # ACT
    write_bc_table(
        new, path, {"B": {"svo_id": "Fac/X.B2"}, "C": {"svo_id": "Fac/X.C"}}
    )
    df = read_bc_table(path)

    # ASSERT
    assert (df["A"] == 1.0).all()
    assert (df["B"] == 10.0).all()
    assert (df["C"] == 11.0).all()
    meta = df.attrs["meta"]["filters"]
    assert meta["A"]["svo_id"] == "Fac/X.A"
    assert meta["B"]["svo_id"] == "Fac/X.B2"
    assert meta["C"]["svo_id"] == "Fac/X.C"


def test_write_bc_table_merge_raises_on_grid_mismatch(tmp_path):
    """
    Given a table on one teff grid,
    When columns computed on a different teff grid are merged into it,
    Then a ValueError should be raised instead of writing NaNs.
    """
    # ARRANGE
    path = tmp_path / "Fac.bc.parquet"
    write_bc_table(_tiny_bc_table(["A"]), path, {"A": {"svo_id": "Fac/X.A"}})
    shifted = _tiny_bc_table(["B"], teffs=(5000.0, 7000.0))

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Grid-axis mismatch"):
        write_bc_table(shifted, path, {"B": {"svo_id": "Fac/X.B"}})


def test_build_bc_grid_raises_when_facilities_are_on_different_grids(tmp_path):
    """
    Given two facility tables whose teff axes differ but have the same
    number of points (so the NaN check alone could not see it),
    When build_bc_grid assembles a filter from each,
    Then a ValueError naming the teff grid should be raised rather than
    silently mis-binning the second facility.
    """
    # ARRANGE
    bc_dir = tmp_path / "Toy" / "BCs"
    write_bc_table(
        _tiny_bc_table(["Fa_X"]),
        bc_dir / "Fa.bc.parquet",
        {"Fa_X": {"svo_id": "Fa/Fa.X"}},
    )
    write_bc_table(
        _tiny_bc_table(["Fb_Y"], teffs=(5000.0, 7000.0)),
        bc_dir / "Fb.bc.parquet",
        {"Fb_Y": {"svo_id": "Fb/Fb.Y"}},
    )

    # ACT & ASSERT
    with pytest.raises(ValueError, match="different teff grid"):
        build_bc_grid(["Fa/Fa.X", "Fb/Fb.Y"], model="Toy", model_root=tmp_path)


def test_build_bc_grid_places_values_on_the_right_nodes(tmp_path):
    """
    Given a toy table whose BC equals teff/1000 + logg + 10*feh + 100*Av,
    When build_bc_grid assembles it,
    Then every node of bc_values holds that function of its own axes
    (a wrong axis order or a mis-binned row would break the identity).
    """
    # ARRANGE
    df = _tiny_bc_table(["Fa_X"])
    df["Fa_X"] = (
        df["teff"] / 1000 + df["logg"] + 10 * df["feh"] + 100 * df["Av"]
    )
    write_bc_table(
        df,
        tmp_path / "Toy" / "BCs" / "Fa.bc.parquet",
        {"Fa_X": {"svo_id": "Fa/Fa.X"}},
    )

    # ACT
    grid = build_bc_grid(["Fa/Fa.X"], model="Toy", model_root=tmp_path)

    # ASSERT
    t, g, f, a = np.meshgrid(
        grid["teff_pts"], grid["logg_pts"], grid["feh_pts"], grid["av_pts"],
        indexing="ij",
    )
    expected = t / 1000 + g + 10 * f + 100 * a
    np.testing.assert_allclose(grid["bc_values"][..., 0], expected)


# ---------------------------------------------------------------------------
# Section 2 — Filter name resolution
# ---------------------------------------------------------------------------


def test_resolve_filter_name_translates_void_to_mist():
    """
    Given a VOID-style filter label '2MASS.J' and the on-disk alias table,
    When resolve_filter_name is called requesting the MIST alias,
    Then the returned string should be '2MASS_J'.
    """
    # ARRANGE
    from exozippy.components.sed.bc_grid import _load_alias_table

    alias_df = _load_alias_table()

    # ACT
    mist_name = resolve_filter_name("2MASS.J", alias_df, alias="MIST")

    # ASSERT
    assert mist_name == "2MASS_J"


def test_resolve_filter_name_translates_void_to_svo():
    """
    Given a VOID-style filter label '2MASS.J' and the on-disk alias table,
    When resolve_filter_name is called requesting the SVO alias,
    Then the returned string should be '2MASS/2MASS.J'.
    """
    # ARRANGE
    from exozippy.components.sed.bc_grid import _load_alias_table

    alias_df = _load_alias_table()

    # ACT
    svo_name = resolve_filter_name("2MASS.J", alias_df, alias="SVO")

    # ASSERT
    assert svo_name == "2MASS/2MASS.J"


def test_resolve_filter_name_passthrough_when_alias_table_is_none():
    """
    Given no alias table (None) and any filter label,
    When resolve_filter_name is called,
    Then the user's original label should be returned unchanged.
    """
    # ARRANGE
    user_label = "Custom.Filter"

    # ACT
    result = resolve_filter_name(user_label, alias_df=None, alias="MIST")

    # ASSERT
    assert result == user_label


def test_alias_table_cells_are_stripped_of_alignment_whitespace():
    """
    Given filternames.txt, whose columns are hand-aligned with literal
    spaces (so cells such as 'TESS/TESS.Red     ' carry trailing blanks),
    When _load_alias_table loads it,
    Then no cell retains leading or trailing whitespace.

    Without the strip the padding is invisible in every printout and only
    shows up as a lookup that quietly misses -- see the companion test
    below for what that costs.
    """
    # ARRANGE
    from exozippy.components.sed.bc_grid import _load_alias_table

    alias_df = _load_alias_table()
    assert alias_df is not None, "shipped alias table should be findable"

    # ACT
    padded = [
        value
        for col in alias_df.select_dtypes(include="object").columns
        for value in alias_df[col].dropna()
        if value != value.strip()
    ]

    # ASSERT
    assert padded == []


def test_resolve_filter_name_matches_a_whitespace_padded_alias_cell():
    """
    Given 'TESS/TESS.Red', whose SVO cell in filternames.txt is padded
    with trailing spaces for column alignment,
    When resolve_filter_name looks it up for the MIST alias,
    Then the table's own answer 'TESS' comes back.

    Regression: unstripped, the .eq() row match fails, resolve falls
    through to synthesize_mist_name and returns 'TESS_Red' -- a BC column
    that does not exist in models/NextGen/BCs/TESS (whose one column is
    'TESS'). No exception is raised at resolve time; the fit dies later,
    or worse, regenerates a duplicate column under the wrong name.
    """
    # ARRANGE
    from exozippy.components.sed.bc_grid import _load_alias_table

    alias_df = _load_alias_table()

    # ACT
    mist_name = resolve_filter_name("TESS/TESS.Red", alias_df, alias="MIST")
    svo_name = resolve_filter_name("TESS.Red", alias_df, alias="SVO")

    # ASSERT
    assert mist_name == "TESS"
    assert svo_name == "TESS/TESS.Red"


# ---------------------------------------------------------------------------
# Section 3 — Grid axis slicer (_range_indices)
# ---------------------------------------------------------------------------


def test_range_indices_returns_bracketing_points_when_bounds_fall_between_grid():
    """
    Given a sorted grid [0, 1, 2, 3, 4] and a range (1.2, 3.7),
    When _range_indices is called,
    Then the returned indices should include the bracketing points
    (index 1 for lower bound, index 4 for upper bound).
    """
    # ARRANGE
    pts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # ACT
    idx = _range_indices(pts, 1.2, 3.7)

    # ASSERT
    assert idx[0] == 1  # brackets 1.2 from below
    assert idx[-1] == 4  # brackets 3.7 from above


def test_range_indices_with_none_lower_bound_starts_at_zero():
    """
    Given a sorted grid and a range (None, 2.5),
    When _range_indices is called,
    Then the first returned index should be 0.
    """
    # ARRANGE
    pts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # ACT
    idx = _range_indices(pts, None, 2.5)

    # ASSERT
    assert idx[0] == 0


def test_range_indices_with_none_upper_bound_ends_at_last():
    """
    Given a sorted grid and a range (1.5, None),
    When _range_indices is called,
    Then the last returned index should be len(pts) - 1.
    """
    # ARRANGE
    pts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # ACT
    idx = _range_indices(pts, 1.5, None)

    # ASSERT
    assert idx[-1] == len(pts) - 1


def test_range_indices_exact_grid_point_bounds_include_both_endpoints():
    """
    Given a sorted grid and a range that falls exactly on two grid points,
    When _range_indices is called,
    Then exactly those two indices should be returned.
    """
    # ARRANGE
    pts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # ACT
    idx = _range_indices(pts, 1.0, 3.0)

    # ASSERT
    np.testing.assert_array_equal(idx, [1, 2, 3])


# ---------------------------------------------------------------------------
# Section 4 — peek_grid_axes
# ---------------------------------------------------------------------------


def test_peek_grid_axes_returns_all_four_axis_keys():
    """
    Given the NextGen BC tree on disk,
    When peek_grid_axes is called,
    Then the returned dict should contain the keys
    'teff_pts', 'logg_pts', 'feh_pts', and 'av_pts'.
    """
    # ACT
    axes = peek_grid_axes(model="NextGen", model_root=_MODEL_ROOT)

    # ASSERT
    for key in ("teff_pts", "logg_pts", "feh_pts", "av_pts"):
        assert key in axes, f"Missing key: {key}"


def test_shipped_grid_yaml_axes_match_the_tables_on_disk():
    """
    Given the NextGen BC tree and its shipped <model>.grid.yaml,
    When the yaml's four axes are compared to peek_grid_axes,
    Then they should agree exactly.

    THIS FAILS SILENTLY WITHOUT THE TEST, which is why it exists.  The two
    are read by different code on different paths: the FIT path calls
    peek_grid_axes, which derives the axes from the tables themselves
    (df["Av"].unique()), while slice_bc and sed/plot.py call _create_AXES
    on the yaml.  So regenerating the tables on a new axis -- as extending
    Av to 20 mag for the galactic bulge did on 2026-09-17 -- leaves the
    yaml describing a grid that no longer exists, and nothing complains
    until someone slices or plots against the old extent and gets a wrong
    answer rather than an error.
    """
    # ARRANGE
    yaml_path = (
        pathlib.Path(_MODEL_ROOT) / "NextGen" / "BCs" / "NextGen.grid.yaml"
    )
    with open(yaml_path) as fh:
        declared = yaml.safe_load(fh)["grid"]

    # ACT
    actual = peek_grid_axes(model="NextGen", model_root=_MODEL_ROOT)

    # ASSERT
    for axis in ("teff", "logg", "feh", "av"):
        np.testing.assert_allclose(
            declared[axis],
            actual[f"{axis}_pts"],
            err_msg=(
                f"{yaml_path.name} declares a {axis} axis of "
                f"{declared[axis]} but the tables hold "
                f"{list(actual[f'{axis}_pts'])}. Regenerate the yaml to "
                "match the tables (the tables are authoritative; the fit "
                "path never reads the yaml)."
            ),
        )


def test_peek_grid_axes_teff_range_is_physically_plausible():
    """
    Given the NextGen BC tree on disk,
    When peek_grid_axes is called,
    Then the teff axis should span a range consistent with stellar
    atmospheres: minimum > 2000 K and maximum < 100000 K.
    """
    # ACT
    axes = peek_grid_axes(model="NextGen", model_root=_MODEL_ROOT)

    # ASSERT
    assert axes["teff_pts"].min() > 2000
    assert axes["teff_pts"].max() < 100_000


def test_peek_grid_axes_raises_for_nonexistent_model():
    """
    Given a model name that does not exist on disk,
    When peek_grid_axes is called,
    Then a FileNotFoundError should be raised.
    """
    # ACT & ASSERT
    with pytest.raises(FileNotFoundError):
        peek_grid_axes(model="FakeModel_XYZ", model_root=_MODEL_ROOT)


# ---------------------------------------------------------------------------
# Section 5 — build_bc_grid
# ---------------------------------------------------------------------------


def test_build_bc_grid_returns_dict_with_required_keys():
    """
    Given the NextGen model and a valid 2MASS filter list,
    When build_bc_grid is called,
    Then the returned dict should contain all required keys.
    """
    # ARRANGE
    filters = ["2MASS.J", "2MASS.H", "2MASS.Ks"]

    # ACT
    grid = build_bc_grid(
        user_filter_names=filters, model="NextGen", model_root=_MODEL_ROOT
    )

    # ASSERT
    for key in (
        "teff_pts",
        "logg_pts",
        "feh_pts",
        "av_pts",
        "bc_values",
        "filter_order",
    ):
        assert key in grid, f"Missing key: {key}"


def test_build_bc_grid_bc_values_shape_matches_axes_and_filters():
    """
    Given the NextGen model and three 2MASS filters,
    When build_bc_grid is called,
    Then bc_values.shape should equal
    (n_teff, n_logg, n_feh, n_av, n_filters).
    """
    # ARRANGE
    filters = ["2MASS.J", "2MASS.H", "2MASS.Ks"]

    # ACT
    grid = build_bc_grid(
        user_filter_names=filters, model="NextGen", model_root=_MODEL_ROOT
    )

    # ASSERT
    expected_shape = (
        len(grid["teff_pts"]),
        len(grid["logg_pts"]),
        len(grid["feh_pts"]),
        len(grid["av_pts"]),
        len(filters),
    )
    assert grid["bc_values"].shape == expected_shape


def test_build_bc_grid_contains_no_nan_values():
    """
    Given the NextGen model and a valid 2MASS filter list,
    When build_bc_grid is called,
    Then the bc_values array should contain no NaN entries,
    indicating the grid is fully populated.
    """
    # ARRANGE
    filters = ["2MASS.J", "2MASS.H", "2MASS.Ks"]

    # ACT
    grid = build_bc_grid(
        user_filter_names=filters, model="NextGen", model_root=_MODEL_ROOT
    )

    # ASSERT
    assert not np.any(np.isnan(grid["bc_values"])), (
        "bc_values contains NaN entries; grid may not be fully populated"
    )


def test_build_bc_grid_filter_order_matches_mist_names():
    """
    Given VOID-style filter names ['2MASS.J', '2MASS.H', '2MASS.Ks'],
    When build_bc_grid is called,
    Then filter_order should contain the corresponding MIST column names.
    """
    # ARRANGE
    filters = ["2MASS.J", "2MASS.H", "2MASS.Ks"]

    # ACT
    grid = build_bc_grid(
        user_filter_names=filters, model="NextGen", model_root=_MODEL_ROOT
    )

    # ASSERT
    assert grid["filter_order"] == ["2MASS_J", "2MASS_H", "2MASS_Ks"]


def test_build_bc_grid_raises_for_nonexistent_facility():
    """
    Given a filter name that maps to a facility not present in the BC tree,
    When build_bc_grid is called,
    Then a FileNotFoundError or NotImplementedError should be raised.
    """
    # ARRANGE — WISE is a real facility; ULTRAVIOLET is not
    filters = ["UV.FakeFilter"]

    # ACT & ASSERT
    with pytest.raises((FileNotFoundError, NotImplementedError, KeyError)):
        build_bc_grid(
            user_filter_names=filters, model="NextGen", model_root=_MODEL_ROOT
        )


# ---------------------------------------------------------------------------
# Section 6 — slice_bc
# ---------------------------------------------------------------------------


def test_slice_bc_reduces_axis_length_when_range_is_tighter_than_full_grid():
    """
    Given a fully assembled bc_values array and a teff range that covers
    fewer than all grid points,
    When slice_bc is called with that teff range,
    Then the returned array's teff axis should be shorter than the original.
    """
    # ARRANGE
    filters = ["2MASS.J", "2MASS.H", "2MASS.Ks"]
    grid = build_bc_grid(
        user_filter_names=filters, model="NextGen", model_root=_MODEL_ROOT
    )
    axes = (
        {k: grid[k] for k in ("teff", "logg", "feh", "av")}
        if "teff" in grid
        else {
            "teff": grid["teff_pts"],
            "logg": grid["logg_pts"],
            "feh": grid["feh_pts"],
            "av": grid["av_pts"],
        }
    )
    grid_dict = _make_grid_dict(axes)

    teff_lo = float(grid["teff_pts"][2])  # a few steps in from the edge
    teff_hi = float(grid["teff_pts"][-3])

    # ACT
    sliced, selected = slice_bc(
        grid_dict, grid["bc_values"], teff=(teff_lo, teff_hi)
    )

    # ASSERT
    assert sliced.shape[0] < grid["bc_values"].shape[0]
    assert "teff" in selected


def test_slice_bc_preserves_filter_axis_length():
    """
    Given a fully assembled bc_values array and a teff constraint,
    When slice_bc is called,
    Then the last (filter) axis of the returned array should equal
    the number of requested filters.
    """
    # ARRANGE
    filters = ["2MASS.J", "2MASS.H", "2MASS.Ks"]
    grid = build_bc_grid(
        user_filter_names=filters, model="NextGen", model_root=_MODEL_ROOT
    )
    axes = {
        "teff": grid["teff_pts"],
        "logg": grid["logg_pts"],
        "feh": grid["feh_pts"],
        "av": grid["av_pts"],
    }
    grid_dict = _make_grid_dict(axes)

    # ACT
    sliced, _ = slice_bc(
        grid_dict,
        grid["bc_values"],
        teff=(float(grid["teff_pts"][5]), float(grid["teff_pts"][15])),
    )

    # ASSERT
    assert sliced.shape[-1] == len(filters)


def test_slice_bc_raises_for_unknown_parameter_name():
    """
    Given a valid bc_values array and a grid_dict,
    When slice_bc is called with an unrecognized parameter name,
    Then a ValueError should be raised.
    """
    # ARRANGE
    filters = ["2MASS.J"]
    grid = build_bc_grid(
        user_filter_names=filters, model="NextGen", model_root=_MODEL_ROOT
    )
    axes = {
        "teff": grid["teff_pts"],
        "logg": grid["logg_pts"],
        "feh": grid["feh_pts"],
        "av": grid["av_pts"],
    }
    grid_dict = _make_grid_dict(axes)

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Unknown parameter"):
        slice_bc(grid_dict, grid["bc_values"], metallicity=(-0.5, 0.5))


# ---------------------------------------------------------------------------
# Section 7 — RegularGridInterpolator
# ---------------------------------------------------------------------------


def test_regular_grid_interpolator_recovers_exact_grid_node_value():
    """
    Given a 1-D grid and values equal to the grid points themselves (f(x)=x),
    When evaluate is called at an exact grid node,
    Then the interpolated value should match the tabulated value exactly.
    """
    # ARRANGE
    pts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values = pts.copy()
    interp = RegularGridInterpolator(points=[pts], values=values)

    coord = pt.as_tensor_variable([[3.0]])

    # ACT
    result = interp.evaluate(coord).eval()

    # ASSERT
    np.testing.assert_allclose(result.flatten(), [3.0], atol=1e-6)


def test_regular_grid_interpolator_midpoint_is_average_of_neighbors():
    """
    Given a 1-D linear grid (f(x) = x),
    When evaluate is called at the midpoint between two grid nodes,
    Then the interpolated value should equal the arithmetic mean of the
    two neighboring tabulated values.
    """
    # ARRANGE
    pts = np.array([0.0, 2.0, 4.0])
    values = pts.copy()
    interp = RegularGridInterpolator(points=[pts], values=values)

    coord = pt.as_tensor_variable([[1.0]])  # midpoint between 0 and 2

    # ACT
    result = interp.evaluate(coord).eval().flat[0]

    # ASSERT
    np.testing.assert_allclose(result, 1.0, atol=1e-6)


def test_regular_grid_interpolator_2d_recovers_bilinear_surface():
    """
    Given a 2-D grid where values equal (teff + logg),
    When evaluate is called at multiple interior points,
    Then the interpolated values should match the analytical surface
    f(t, g) = t + g to within floating-point tolerance.
    """
    # ARRANGE
    teff_pts = np.array([3000.0, 4000.0, 5000.0, 6000.0])
    logg_pts = np.array([3.5, 4.0, 4.5, 5.0])
    T, G = np.meshgrid(teff_pts, logg_pts, indexing="ij")
    values = (T + G).astype(float)

    interp = RegularGridInterpolator(
        points=[teff_pts, logg_pts], values=values
    )

    test_coords = np.array(
        [
            [3500.0, 3.75],
            [5000.0, 4.5],
            [4500.0, 4.25],
        ]
    )
    coord_tensor = pt.as_tensor_variable(test_coords)

    # ACT
    result = interp.evaluate(coord_tensor).eval().flatten()

    # ASSERT
    expected = test_coords[:, 0] + test_coords[:, 1]
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_regular_grid_interpolator_with_trailing_filter_axis():
    """
    Given a 2-D grid (teff × logg) with N_filter trailing BC values
    per node (mimicking real BC usage),
    When evaluate is called at an exact grid node,
    Then the output shape should be (1, n_filters) and values should
    match the tabulated BCs at that node.
    """
    # ARRANGE
    teff_pts = np.array([5000.0, 6000.0, 7000.0])
    logg_pts = np.array([4.0, 4.5, 5.0])
    n_filters = 3
    # bc[i, j, :] = [i, j, i+j] for easy verification
    values = np.zeros((3, 3, n_filters))
    for i in range(3):
        for j in range(3):
            values[i, j, :] = [float(i), float(j), float(i + j)]

    interp = RegularGridInterpolator(
        points=[teff_pts, logg_pts], values=values
    )

    # Query at node (1, 1) -> teff=6000, logg=4.5 -> expected [1, 1, 2]
    coord = pt.as_tensor_variable([[6000.0, 4.5]])

    # ACT
    result = interp.evaluate(coord).eval()

    # ASSERT
    np.testing.assert_allclose(result, [[1.0, 1.0, 2.0]], atol=1e-5)


# ---------------------------------------------------------------------------
# Section 8 — SED physics (registered functions)
# ---------------------------------------------------------------------------


def test_calc_absbolmag_solar_luminosity_matches_known_value():
    """
    Given a luminosity of 1.0 L_sun (which is the IAU definition anchor),
    When calc_absbolmag is evaluated,
    Then the result should equal the IAU 2015 nominal absolute solar
    bolometric magnitude of 4.74 mag.
    """
    # ARRANGE
    calc_absbolmag = PHYSICS_REGISTRY["calc_absbolmag"]
    L_sun = pt.as_tensor_variable(1.0)

    # ACT
    result = calc_absbolmag(L_sun).eval()

    # ASSERT — IAU 2015 Mbol,sun = 4.74
    np.testing.assert_allclose(result, 4.74, atol=0.01)


def test_calc_appmag_at_10pc_equals_absolute_magnitude():
    """
    Given an absolute magnitude M and a distance of exactly 10 pc,
    When calc_appmag is evaluated,
    Then the apparent magnitude should equal M (distance modulus = 0).
    """
    # ARRANGE
    calc_appmag = PHYSICS_REGISTRY["calc_appmag"]
    M_abs = pt.as_tensor_variable(5.0)
    distance_10pc = pt.as_tensor_variable(10.0)

    # ACT
    m_app = calc_appmag(M_abs, distance_10pc).eval()

    # ASSERT
    np.testing.assert_allclose(m_app, 5.0, atol=1e-6)


def test_calc_appmag_distance_modulus_is_correct_at_100pc():
    """
    Given an absolute magnitude of 0.0 and a distance of 100 pc,
    When calc_appmag is evaluated,
    Then the result should be 5.0 mag (distance modulus for 100 pc = 5).
    """
    # ARRANGE
    calc_appmag = PHYSICS_REGISTRY["calc_appmag"]
    M_abs = pt.as_tensor_variable(0.0)
    distance_100pc = pt.as_tensor_variable(100.0)

    # ACT
    m_app = calc_appmag(M_abs, distance_100pc).eval()

    # ASSERT
    np.testing.assert_allclose(m_app, 5.0, atol=1e-6)


def test_calc_absmag_from_bc_subtracts_bc_from_mbol():
    """
    Given an absolute bolometric magnitude and a bolometric correction,
    When calc_absmag_from_bc is evaluated,
    Then the result should equal Mbol - BC.
    """
    # ARRANGE
    calc_absmag_from_bc = PHYSICS_REGISTRY["calc_absmag_from_bc"]
    Mbol = pt.as_tensor_variable(4.74)
    BC_J = pt.as_tensor_variable(1.5)

    # ACT
    M_J = calc_absmag_from_bc(Mbol, BC_J).eval()

    # ASSERT
    np.testing.assert_allclose(M_J, 4.74 - 1.5, atol=1e-6)


def test_calc_bc_is_inverse_of_calc_absmag_from_bc():
    """
    Given an absolute bolometric magnitude and an absolute filter magnitude,
    When calc_bc is evaluated,
    Then the result should equal Mbol - M_filter (consistent with the
    definition BC = Mbol - M_filter so that calc_absmag_from_bc(Mbol, BC)
    recovers M_filter).
    """
    # ARRANGE
    calc_bc = PHYSICS_REGISTRY["calc_bc"]
    calc_absmag_from_bc = PHYSICS_REGISTRY["calc_absmag_from_bc"]
    Mbol = pt.as_tensor_variable(4.74)
    M_filter = pt.as_tensor_variable(3.24)

    # ACT
    BC = calc_bc(Mbol, M_filter).eval()
    M_recovered = calc_absmag_from_bc(
        pt.as_tensor_variable(Mbol.eval()), pt.as_tensor_variable(BC)
    ).eval()

    # ASSERT
    np.testing.assert_allclose(BC, 4.74 - 3.24, atol=1e-6)
    np.testing.assert_allclose(M_recovered, 3.24, atol=1e-6)


def test_calc_appmag_brighter_star_at_same_distance_has_lower_mag():
    """
    Given two stars with different absolute magnitudes at the same distance,
    When calc_appmag is evaluated for both,
    Then the star with the lower (brighter) absolute magnitude should have
    the lower apparent magnitude.
    """
    # ARRANGE
    calc_appmag = PHYSICS_REGISTRY["calc_appmag"]
    M_bright = pt.as_tensor_variable(2.0)
    M_faint = pt.as_tensor_variable(7.0)
    dist = pt.as_tensor_variable(50.0)

    # ACT
    m_bright = calc_appmag(M_bright, dist).eval()
    m_faint = calc_appmag(M_faint, dist).eval()

    # ASSERT
    assert m_bright < m_faint


# ---------------------------------------------------------------------------
# Section 9 — SED.load_data and register_parameters (integration-level)
# ---------------------------------------------------------------------------

# These tests require a minimal .sed YAML file. We create one in a tmp_path
# rather than depending on the HAT-P-3 test data being present, so the
# tests are self-contained and can run in CI.

_MINIMAL_SED_YAML = """\
model: NextGen
nstars: 1
filters:
  - name: 2MASS.J
    mag: 8.000
    err: 0.020
    magsys: Vega
  - name: 2MASS.H
    mag: 7.800
    err: 0.026
    magsys: Vega
  - name: 2MASS.Ks
    mag: 7.750
    err: 0.018
    magsys: Vega
"""


@pytest.fixture()
def minimal_sed_file(tmp_path):
    """Write a minimal .sed YAML file and return its path."""
    p = tmp_path / "test_star.sed"
    p.write_text(_MINIMAL_SED_YAML)
    return str(p)


def _make_sed(minimal_sed_file):
    """Instantiate a SED component around a minimal config."""
    from exozippy.components.sed.sed import SED
    from exozippy.config import ConfigManager

    user_params = {
        "star.teffsed": {"initval": 5778.0},
        "star.feh": {"initval": 0.0},
        "star.av": {"initval": 0.0},
    }
    cm = ConfigManager(user_params)
    config = {"file": minimal_sed_file, "model_root": str(_MODEL_ROOT)}
    return SED(config, cm), cm


def test_sed_init_registers_grid_bounds_on_the_override_channel(
    minimal_sed_file,
):
    """
    Given a SED component initialised with a valid .sed file pointing at
    the NextGen BC tree,
    When __init__ runs,
    Then config_manager.param_overrides should contain 'star.teffsed',
    'star.feh', and 'star.av' entries whose 'lower' and 'upper' keys
    bracket the BC grid's physical range -- and the user's own params should
    be untouched.

    The channel changed in 2026-08: these bounds used to be written straight
    into config_manager.user_params, where nothing downstream could tell them
    apart from something the user wrote.  The substance asserted here is
    unchanged (the grid's validity limits are registered, and they are
    physically sane); tests/test_component_override_channel.py covers the
    provenance consequences.
    """
    # ARRANGE / ACT
    sed, cm = _make_sed(minimal_sed_file)

    # ASSERT — bounds registered on the component override channel
    for key in ("star.teffsed", "star.feh", "star.av"):
        assert key in cm.param_overrides, f"config_manager missing key: {key}"
        entry = cm.param_overrides[key]
        assert "lower" in entry and "upper" in entry, (
            f"{key} entry is missing 'lower' / 'upper': {entry}"
        )
        # ...and NOT written into the user's params.
        user_entry = cm.user_params.get(key) or {}
        assert "lower" not in user_entry and "upper" not in user_entry

    # ASSERT — teff bounds are physically sane
    teff_entry = cm.param_overrides["star.teffsed"]
    assert teff_entry["lower"] > 100
    assert teff_entry["upper"] < 200_000


def _stub_system(star_names=("A",)):
    """Minimal stand-in for System: just the star roster load_data needs."""
    from types import SimpleNamespace

    star = SimpleNamespace(names=list(star_names), n_elements=len(star_names))
    return SimpleNamespace(star=star)


def test_sed_load_data_populates_filters_and_bc_grid(minimal_sed_file):
    """
    Given a SED component initialised with a three-filter .sed YAML,
    When load_data is called,
    Then self.filters should contain the three filter names,
    self.bc_grid_data should be a dict with 'bc_values', and
    self.mist_filters should list the corresponding MIST column names.
    """
    # ARRANGE
    sed, _ = _make_sed(minimal_sed_file)

    # ACT
    sed.load_data(system=_stub_system())

    # ASSERT
    assert len(sed.filters) == 3
    assert sed.bc_grid_data is not None
    assert "bc_values" in sed.bc_grid_data
    assert len(sed.mist_filters) == 3


def test_sed_load_data_raises_when_sed_file_is_none(minimal_sed_file):
    """
    Given a SED component whose 'file' config key is absent (None),
    When load_data is called,
    Then a ValueError should be raised indicating the missing key.
    """
    # ARRANGE
    from exozippy.components.sed.sed import SED
    from exozippy.config import ConfigManager

    cm = ConfigManager({})
    # Patch the sedfile away after construction so __init__ doesn't
    # blow up on peek_grid_axes before we can test load_data.
    sed, _ = _make_sed(minimal_sed_file)
    sed.sedfile = None

    # ACT & ASSERT
    with pytest.raises(ValueError, match="missing the required 'file' key"):
        sed.load_data(system=_stub_system())


def test_sed_register_parameters_creates_errscale_parameter(minimal_sed_file):
    """
    Given a SED component,
    When register_parameters is called and errscale is materialized via add_parameter,
    Then the component should expose a self.errscale Parameter.
    """
    import pymc as pm

    sed, _ = _make_sed(minimal_sed_file)

    # ACT
    sed.register_parameters(system=None)
    with pm.Model() as model:
        sed.add_parameter(model, "errscale", system=None)

    # ASSERT
    assert hasattr(sed, "errscale"), (
        "SED missing self.errscale after add_parameter"
    )
    assert sed.errscale is not None


# ---------------------------------------------------------------------------
# Section 10 — Multi-star photType parsing and blended/differential mags
# ---------------------------------------------------------------------------

_MULTISTAR_SED_YAML = """\
model: NextGen
filters:
  - name: 2MASS.J
    mag: 8.000
    err: 0.020
    photType:
      pos: [Lens, Source]
  - name: 2MASS.H
    mag: 2.100
    err: 0.050
    photType:
      pos: [Source]
      neg: [0]
  - name: 2MASS.Ks
    mag: 7.750
    err: 0.018
"""


@pytest.fixture()
def multistar_sed_file(tmp_path):
    """Write a two-star .sed YAML with blend, diff, and default rows."""
    p = tmp_path / "test_multistar.sed"
    p.write_text(_MULTISTAR_SED_YAML)
    return str(p)


def test_load_data_builds_blend_matrix_from_photType(multistar_sed_file):
    """
    Given a two-star system and a .sed file with a blended row (pos by
    names), a differential row (pos by name, neg by index), and a row
    with no photType,
    When load_data runs,
    Then blend_matrix holds +1/-1/0 coefficients per the EXOFASTv2
    convention, defaulting to an all-star blend when photType is absent,
    and combo_labels describe each combination.
    """
    # ARRANGE
    sed, _ = _make_sed(multistar_sed_file)
    system = _stub_system(star_names=("Lens", "Source"))

    # ACT
    sed.load_data(system)

    # ASSERT
    assert sed.blend_matrix.shape == (3, 2)
    assert sed.blend_matrix[0].tolist() == [1, 1]  # Lens+Source blend
    assert sed.blend_matrix[1].tolist() == [-1, 1]  # Source - Lens
    assert sed.blend_matrix[2].tolist() == [1, 1]  # default: all stars
    assert sed.combo_labels == ["Lens+Source", "Source-Lens", "Lens+Source"]
    # one observation per row, not per star
    assert sed.mag.shape == (3,)
    assert sed.err.shape == (3,)


def test_load_data_accepts_blend_alias_for_pos(tmp_path):
    """
    Given a .sed row using the original `blend:` keyword,
    When load_data runs,
    Then it is treated as an alias for `pos`.
    """
    # ARRANGE
    p = tmp_path / "alias.sed"
    p.write_text(
        "model: NextGen\n"
        "filters:\n"
        "  - name: 2MASS.J\n"
        "    mag: 8.0\n"
        "    err: 0.02\n"
        "    photType:\n"
        "      blend: [1]\n"
    )
    sed, _ = _make_sed(str(p))

    # ACT
    sed.load_data(_stub_system(star_names=("A", "B")))

    # ASSERT
    assert sed.blend_matrix[0].tolist() == [0, 1]
    assert sed.combo_labels == ["B"]


@pytest.mark.parametrize(
    "phot_yaml, match",
    [
        ("photType:\n      pos: [A]\n      neg: [A]", "both pos and neg"),
        ("photType:\n      flux: [A]", "Unknown photType key"),
        ("photType:\n      pos: []", "non-empty"),
        # Wording from the shared component.resolve_star_ref.
        ("photType:\n      pos: [Nobody]", "unknown star"),
        ("photType:\n      pos: [5]", "out of range"),
        ("photType:\n      blend: [A]\n      pos: [B]", "alias"),
    ],
)
def test_load_data_rejects_invalid_photType(tmp_path, phot_yaml, match):
    """
    Given malformed photType entries (pos/neg overlap, unknown keys,
    empty pos, unknown star names, out-of-range indices, blend+pos),
    When load_data runs,
    Then a ValueError naming the problem is raised.
    """
    # ARRANGE
    p = tmp_path / "bad.sed"
    p.write_text(
        "model: NextGen\n"
        "filters:\n"
        "  - name: 2MASS.J\n"
        "    mag: 8.0\n"
        "    err: 0.02\n"
        f"    {phot_yaml}\n"
    )
    sed, _ = _make_sed(str(p))

    # ACT & ASSERT
    with pytest.raises(ValueError, match=match):
        sed.load_data(_stub_system(star_names=("A", "B")))


def test_combined_appmag_matches_hand_computed_blend_and_diff(
    multistar_sed_file,
):
    """
    Given per-star predicted magnitudes m = [[10, 12, 14], [11, 12, 15]]
    and the multistar blend matrix (blend, diff, default-blend rows),
    When _combined_appmag_node is evaluated,
    Then blended rows equal -2.5*log10(sum of star fluxes) and the
    differential row equals m_pos - m_neg.
    """
    # ARRANGE
    sed, _ = _make_sed(multistar_sed_file)
    sed.load_data(_stub_system(star_names=("Lens", "Source")))

    m_star = np.array([[10.0, 12.0, 14.0], [11.0, 12.0, 15.0]])
    sed._m_pred_matrix = pt.as_tensor_variable(m_star)

    # ACT
    combined = sed._combined_appmag_node(system=None).eval()

    # ASSERT
    F = 10 ** (-0.4 * m_star)
    expected_blend_J = -2.5 * np.log10(F[0, 0] + F[1, 0])
    expected_diff_H = m_star[1, 1] - m_star[0, 1]  # -2.5*log10(F_S/F_L)
    expected_blend_K = -2.5 * np.log10(F[0, 2] + F[1, 2])
    np.testing.assert_allclose(
        combined,
        [expected_blend_J, expected_diff_H, expected_blend_K],
        rtol=1e-10,
    )


def test_predict_star_and_blend_appmag_use_filter_columns(multistar_sed_file):
    """
    Given a loaded two-star SED and a stubbed per-star magnitude matrix,
    When predict_star_appmag / predict_blend_appmag are called with a
    filter in any naming convention (user, MIST, SVO),
    Then they return that star's magnitude / the blended magnitude of
    the requested stars in the right grid column.
    """
    # ARRANGE
    sed, _ = _make_sed(multistar_sed_file)
    sed.load_data(_stub_system(star_names=("Lens", "Source")))

    m_star = np.array([[10.0, 12.0, 14.0], [11.0, 12.0, 15.0]])
    sed._m_pred_matrix = pt.as_tensor_variable(m_star)

    # ACT
    m_source_H = sed.predict_star_appmag(1, "2MASS.H", system=None).eval()
    m_blend_J = sed.predict_blend_appmag([0, 1], "2MASS_J", system=None).eval()

    # ASSERT
    assert m_source_H == pytest.approx(12.0)
    F = 10 ** (-0.4 * m_star[:, 0])
    assert m_blend_J == pytest.approx(-2.5 * np.log10(F.sum()))


def test_filter_column_raises_for_unknown_filter(multistar_sed_file):
    """
    Given a loaded SED,
    When filter_column is asked for a filter not in the BC grid,
    Then a KeyError naming the filter and the available ones is raised.
    """
    # ARRANGE
    sed, _ = _make_sed(multistar_sed_file)
    sed.load_data(_stub_system(star_names=("Lens", "Source")))

    # ACT & ASSERT
    with pytest.raises(KeyError, match="NotAFilter"):
        sed.filter_column("NotAFilter")
