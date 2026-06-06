"""
Unit tests for the SED component:
  - bc_grid.py  : file parsing, grid assembly, slicing, filter name resolution
  - physics.py  : registered physics functions (absbolmag, absmag, appmag, bc)
  - sed.py      : __init__ grid-bound injection, load_data, register_parameters
"""

import warnings
import numpy as np
import pytest
import pytensor.tensor as pt
import pytensor

from exozippy.components.sed.bc_grid import (
    _parse_feh_from_filename,
    _read_single_bc_file,
    _range_indices,
    peek_grid_axes,
    build_bc_grid,
    slice_bc,
    resolve_filter_name,
    RegularGridInterpolator,
    DEFAULT_BC_ROOT,
)
from exozippy.physics_registry import PHYSICS_REGISTRY
import exozippy.components.sed.physics  # registers calc_absbolmag etc.


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_BC_ROOT = DEFAULT_BC_ROOT
_SOLAR_FEH_2MASS_NEXTGEN = (
    _BC_ROOT / "NextGen" / "BCs" / "2MASS" / "feh+0.0_afe+0.0.2MASS"
)

# A minimal grid_dict mirroring what build_bc_grid / slice_bc expect.
def _make_grid_dict(axes):
    return {
        "model": "NextGen",
        "grid": {k: list(v) for k, v in axes.items()},
    }


# ---------------------------------------------------------------------------
# Section 1 — File parsing utilities
# ---------------------------------------------------------------------------


def test_parse_feh_from_solar_filename_returns_zero():
    """
    Given the filename 'feh+0.0_afe+0.0.2MASS',
    When _parse_feh_from_filename is called,
    Then the returned float should equal exactly 0.0.
    """
    # ARRANGE
    filename = "feh+0.0_afe+0.0.2MASS"

    # ACT
    result = _parse_feh_from_filename(filename)

    # ASSERT
    assert result == 0.0


def test_parse_feh_from_negative_filename_returns_correct_value():
    """
    Given the filename 'feh-2.5_afe+0.0.2MASS',
    When _parse_feh_from_filename is called,
    Then the returned float should equal -2.5.
    """
    # ARRANGE
    filename = "feh-2.5_afe+0.0.2MASS"

    # ACT
    result = _parse_feh_from_filename(filename)

    # ASSERT
    assert result == -2.5


def test_parse_feh_from_malformed_filename_raises_value_error():
    """
    Given a filename that does not match the expected pattern,
    When _parse_feh_from_filename is called,
    Then a ValueError should be raised.
    """
    # ARRANGE
    bad_filename = "not_a_valid_bc_filename.2MASS"

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Cannot parse feh"):
        _parse_feh_from_filename(bad_filename)


def test_read_single_bc_file_returns_dataframe_with_correct_columns():
    """
    Given the solar-metallicity 2MASS BC NextGen file on disk,
    When _read_single_bc_file is called,
    Then the returned DataFrame should contain 'teff', 'logg', 'feh', 'Av'
    and the three 2MASS filter columns.
    """
    # ARRANGE / ACT
    df, filter_cols = _read_single_bc_file(_SOLAR_FEH_2MASS_NEXTGEN)

    # ASSERT
    assert "teff" in df.columns
    assert "logg" in df.columns
    assert "feh" in df.columns
    assert "Av" in df.columns
    assert set(filter_cols) == {"2MASS_J", "2MASS_H", "2MASS_Ks"}


def test_read_single_bc_file_teff_column_is_linear_not_log():
    """
    Given the solar-metallicity 2MASS BC NextGen file on disk,
    When _read_single_bc_file is called,
    Then the 'teff' column should contain linear temperature values
    (not log10 values), so all entries should be greater than 100.
    """
    # ARRANGE / ACT
    df, _ = _read_single_bc_file(_SOLAR_FEH_2MASS_NEXTGEN)

    # ASSERT
    assert (df["teff"] > 100).all(), (
        "teff column contains values <= 100; looks like lgTef was not exponentiated"
    )


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
    assert idx[0] == 1   # brackets 1.2 from below
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
    axes = peek_grid_axes(model="NextGen", bc_root=_BC_ROOT)

    # ASSERT
    for key in ("teff_pts", "logg_pts", "feh_pts", "av_pts"):
        assert key in axes, f"Missing key: {key}"


def test_peek_grid_axes_teff_range_is_physically_plausible():
    """
    Given the NextGen BC tree on disk,
    When peek_grid_axes is called,
    Then the teff axis should span a range consistent with stellar
    atmospheres: minimum > 2000 K and maximum < 100000 K.
    """
    # ACT
    axes = peek_grid_axes(model="NextGen", bc_root=_BC_ROOT)

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
        peek_grid_axes(model="FakeModel_XYZ", bc_root=_BC_ROOT)


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
    grid = build_bc_grid(user_filter_names=filters, model="NextGen", bc_root=_BC_ROOT)

    # ASSERT
    for key in ("teff_pts", "logg_pts", "feh_pts", "av_pts", "bc_values", "filter_order"):
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
    grid = build_bc_grid(user_filter_names=filters, model="NextGen", bc_root=_BC_ROOT)

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
    grid = build_bc_grid(user_filter_names=filters, model="NextGen", bc_root=_BC_ROOT)

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
    grid = build_bc_grid(user_filter_names=filters, model="NextGen", bc_root=_BC_ROOT)

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
        build_bc_grid(user_filter_names=filters, model="NextGen", bc_root=_BC_ROOT)


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
    grid = build_bc_grid(user_filter_names=filters, model="NextGen", bc_root=_BC_ROOT)
    axes = {k: grid[k] for k in ("teff", "logg", "feh", "av")} if "teff" in grid else {
        "teff": grid["teff_pts"],
        "logg": grid["logg_pts"],
        "feh":  grid["feh_pts"],
        "av":   grid["av_pts"],
    }
    grid_dict = _make_grid_dict(axes)

    teff_lo = float(grid["teff_pts"][2])   # a few steps in from the edge
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
    grid = build_bc_grid(user_filter_names=filters, model="NextGen", bc_root=_BC_ROOT)
    axes = {
        "teff": grid["teff_pts"],
        "logg": grid["logg_pts"],
        "feh":  grid["feh_pts"],
        "av":   grid["av_pts"],
    }
    grid_dict = _make_grid_dict(axes)

    # ACT
    sliced, _ = slice_bc(
        grid_dict, grid["bc_values"],
        teff=(float(grid["teff_pts"][5]), float(grid["teff_pts"][15]))
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
    grid = build_bc_grid(user_filter_names=filters, model="NextGen", bc_root=_BC_ROOT)
    axes = {
        "teff": grid["teff_pts"],
        "logg": grid["logg_pts"],
        "feh":  grid["feh_pts"],
        "av":   grid["av_pts"],
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

    interp = RegularGridInterpolator(points=[teff_pts, logg_pts], values=values)

    test_coords = np.array([
        [3500.0, 3.75],
        [5000.0, 4.5],
        [4500.0, 4.25],
    ])
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

    interp = RegularGridInterpolator(points=[teff_pts, logg_pts], values=values)

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
    M_recovered = calc_absmag_from_bc(pt.as_tensor_variable(Mbol.eval()), pt.as_tensor_variable(BC)).eval()

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
    from exozippy.config import ConfigManager
    from exozippy.components.sed.sed import SED

    user_params = {
        "star.teffsed":  {"initval": 5778.0},
        "star.feh":      {"initval":  0.0},
        "star.av":       {"initval":  0.0},
    }
    cm = ConfigManager(user_params)
    config = {"file": minimal_sed_file, "bc_root": str(_BC_ROOT)}
    return SED(config, cm), cm


def test_sed_init_injects_grid_bounds_into_config_manager(minimal_sed_file):
    """
    Given a SED component initialised with a valid .sed file pointing at
    the NextGen BC tree,
    When __init__ runs,
    Then config_manager.user_params should contain 'star.teffsed',
    'star.feh', and 'star.av' entries whose 'lower' and 'upper' keys
    bracket the BC grid's physical range.
    """
    # ARRANGE / ACT
    sed, cm = _make_sed(minimal_sed_file)

    # ASSERT — keys injected
    for key in ("star.teffsed", "star.feh", "star.av"):
        assert key in cm.user_params, f"config_manager missing key: {key}"
        entry = cm.user_params[key]
        assert "lower" in entry and "upper" in entry, (
            f"{key} entry is missing 'lower' / 'upper': {entry}"
        )

    # ASSERT — teff bounds are physically sane
    teff_entry = cm.user_params["star.teffsed"]
    assert teff_entry["lower"] > 100
    assert teff_entry["upper"] < 200_000


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
    sed.load_data(system=None)

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
    from exozippy.config import ConfigManager
    from exozippy.components.sed.sed import SED

    cm = ConfigManager({})
    # Patch the sedfile away after construction so __init__ doesn't
    # blow up on peek_grid_axes before we can test load_data.
    sed, _ = _make_sed(minimal_sed_file)
    sed.sedfile = None

    # ACT & ASSERT
    with pytest.raises(ValueError, match="missing the required 'file' key"):
        sed.load_data(system=None)


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
    assert hasattr(sed, "errscale"), "SED missing self.errscale after add_parameter"
    assert sed.errscale is not None
