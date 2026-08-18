"""Tests for the generic Instrument time-system and column-layout features.

``time_offset:``, ``time_scale:``/``time_frame:`` (conversion to BJD_TDB)
and ``columns:`` all live on the shared Instrument base and are applied by
``_read_data``, so RVInstrument stands in for all four data components,
exactly as in test_instrument_mask.py.
"""

# RVInstrument stores rv/err internally in solRad/d; data files are m/s.
import astropy.units as u_
import numpy as np
import pytest

from exozippy.components.rvinstrument.rvinstrument import RVInstrument
from exozippy.config import ConfigManager

RV_FACTOR = (u_.m / u_.s).to(u_.solRad / u_.d)

# Vega, so the light-travel geometry is a real one.
RA_DEG, DEC_DEG = 279.23473479, 38.78368896
COORD_PARAMS = {"star.0.ra": RA_DEG, "star.0.dec": DEC_DEG}
# Absolute JDs spanning half a year, so the Romer delay actually varies.
JDS = (2457000.0, 2457030.0, 2457090.0, 2457180.0)


def _write_file(path, rows):
    with open(path, "w") as f:
        f.write("# test data\n")
        for row in rows:
            f.write(" ".join(f"{v:.9f}" for v in row) + "\n")


def _write_rv_file(path, times):
    _write_file(path, [(t, float(i), 1.0) for i, t in enumerate(times)])


def _load(tmp_path, entry_extra, times=JDS, user_params=None, rows=None):
    data_file = tmp_path / "inst.rv"
    if rows is not None:
        _write_file(data_file, rows)
    else:
        _write_rv_file(data_file, times)
    config = [{"name": "Inst", "file": str(data_file), **entry_extra}]
    inst = RVInstrument(config, ConfigManager(user_params or {}))
    inst.load_data(system=None)
    return inst


# ----------------------------------------------------------------------
# time_offset
# ----------------------------------------------------------------------
def test_no_time_keys_is_passthrough(tmp_path):
    """
    Given a file without any time keys,
    When the data are loaded,
    Then the times are exactly the file's own values (the default is
    BJD_TDB in, BJD_TDB out, byte for byte).
    """
    inst = _load(tmp_path, {}, times=(3.0, 1.0, 2.0))
    np.testing.assert_array_equal(inst.time, [1.0, 2.0, 3.0])


def test_time_offset_added_to_all_times(tmp_path):
    """
    Given a time_offset,
    When the data are loaded,
    Then every input time is shifted by exactly that amount.
    """
    inst = _load(tmp_path, {"time_offset": 2450000.0}, times=(1.5, 2.5))
    np.testing.assert_allclose(inst.time, [2450001.5, 2450002.5])


def test_non_numeric_time_offset_raises(tmp_path):
    """
    Given a non-numeric time_offset,
    When the instrument is constructed,
    Then a ValueError is raised at construction, before any load.
    """
    data_file = tmp_path / "inst.rv"
    _write_rv_file(data_file, JDS)
    config = [
        {"name": "Inst", "file": str(data_file), "time_offset": "tomorrow"}
    ]
    with pytest.raises(ValueError, match="time_offset must be a number"):
        RVInstrument(config, ConfigManager({}))


# ----------------------------------------------------------------------
# time_scale / time_frame conversion
# ----------------------------------------------------------------------
def test_jd_utc_to_bjd_tdb_matches_astropy(tmp_path):
    """
    Given times labeled JD_UTC,
    When the data are loaded,
    Then the stored times equal the independent astropy chain
    (scale conversion + barycentric light travel time) to < 1 us.
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    inst = _load(
        tmp_path,
        {"time_scale": "utc", "time_frame": "jd"},
        user_params=COORD_PARAMS,
    )

    coord = SkyCoord(ra=RA_DEG * u.deg, dec=DEC_DEG * u.deg)
    loc = EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)
    t = Time(np.array(JDS), format="jd", scale="utc", location=loc).tdb
    # ephemeris is pinned: other test files in the same xdist worker import
    # exozippy.ephemeris / mulensing.op, which set the astropy global to
    # 'jpl'; the converter always passes its own ephemeris explicitly.
    expected = (
        t.jd
        + t.light_travel_time(
            coord, kind="barycentric", ephemeris="builtin"
        ).jd
    )

    assert np.all(np.abs(inst.time - expected) * 86400.0 < 1e-6)


def test_scale_only_conversion_is_tdb_minus_utc(tmp_path):
    """
    Given BJD_UTC times (frame already barycentric, scale wrong),
    When the data are loaded WITHOUT any star coordinates,
    Then the shift is TDB-UTC (~69 s in 2015): the barycentric term
    cancels exactly, so no coordinates are needed for scale-only work.
    """
    from astropy.time import Time

    inst = _load(tmp_path, {"time_scale": "utc"})
    expected = Time(np.array(JDS), format="jd", scale="utc").tdb.jd
    shift = (inst.time - np.array(JDS)) * 86400.0
    np.testing.assert_allclose(inst.time, expected, rtol=0, atol=1e-9)
    assert np.all((shift > 67.0) & (shift < 70.0))


def test_hjd_conversion_matches_independent_bjd(tmp_path):
    """
    Given an HJD_TDB file built from known geocentric JD_TDB instants,
    When the data are loaded with time_frame hjd,
    Then the result lands on the independently computed BJD_TDB of the
    same instants (the fixed-point inversion of the heliocentric
    correction is exact to < 1 us).
    """
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    coord = SkyCoord(ra=RA_DEG * u.deg, dec=DEC_DEG * u.deg)
    loc = EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)

    # Build an HJD file from known geocentric JD_TDB times.  The ephemeris
    # is pinned for the same reason as in the JD_UTC test above.
    t_geo = Time(np.array(JDS), format="jd", scale="tdb", location=loc)
    hjd = (
        t_geo.jd
        + t_geo.light_travel_time(
            coord, kind="heliocentric", ephemeris="builtin"
        ).jd
    )
    bjd = (
        t_geo.jd
        + t_geo.light_travel_time(
            coord, kind="barycentric", ephemeris="builtin"
        ).jd
    )

    inst = _load(
        tmp_path,
        {"time_frame": "hjd"},
        times=tuple(hjd),
        user_params=COORD_PARAMS,
    )
    # HJD -> BJD_TDB must land on the independently computed BJD; the
    # tolerance covers the test file's 9-decimal write quantization
    # (1e-9 d ~ 86 us) plus a few float64 ulps of JD (~40 us each).
    assert np.all(np.abs(inst.time - bjd) * 86400.0 < 5e-4)
    # And the heliocentric/barycentric corrections genuinely differ (the
    # conversion did something frame-specific: 0.13-0.33 s toward Vega at
    # these epochs, set by where Jupiter and Saturn are).
    assert np.max(np.abs(hjd - bjd)) * 86400.0 > 0.1


def test_conversion_requires_user_coordinates(tmp_path):
    """
    Given a frame conversion but no user-set star ra/dec,
    When the data are loaded,
    Then a ValueError demands the coordinates instead of silently using
    the defaults.yaml placeholders.
    """
    with pytest.raises(ValueError, match="star.0.ra"):
        _load(tmp_path, {"time_frame": "jd", "time_scale": "utc"})


def test_conversion_requires_absolute_jds(tmp_path):
    """
    Given truncated times (BJD-2450000 style) and a conversion request,
    When the data are loaded,
    Then a ValueError points at time_offset,
    And supplying the offset makes the same conversion succeed.
    """
    with pytest.raises(ValueError, match="time_offset"):
        _load(
            tmp_path,
            {"time_scale": "utc"},
            times=(7000.0, 7010.0),
            user_params=COORD_PARAMS,
        )
    inst = _load(
        tmp_path,
        {"time_scale": "utc", "time_offset": 2450000.0},
        times=(7000.0, 7010.0),
        user_params=COORD_PARAMS,
    )
    assert np.all(inst.time > 2457000.0)


def test_observatory_location_shifts_by_less_than_earth_radius(tmp_path):
    """
    Given the same JD_UTC file converted with and without a geodetic
    time_location,
    When both are loaded,
    Then the results differ, by no more than the 21.3 ms geocenter-to-
    surface light travel time.
    """
    spec = {"time_scale": "utc", "time_frame": "jd"}
    inst_geo = _load(tmp_path, spec, user_params=COORD_PARAMS)
    inst_obs = _load(
        tmp_path,
        {**spec, "time_location": [-155.5, 19.8, 4200.0]},  # Mauna Kea
        user_params=COORD_PARAMS,
    )
    diff = np.abs(inst_obs.time - inst_geo.time) * 86400.0
    assert np.all(diff < 0.0214)
    assert np.any(diff > 1e-5)


def test_ut_alias_and_bad_scale_validation(tmp_path):
    """
    Given time_scale 'ut' and an unknown scale,
    When the instrument is constructed,
    Then 'ut' parses as ut1 and the unknown scale raises at construction.
    """
    data_file = tmp_path / "inst.rv"
    _write_rv_file(data_file, JDS)
    inst = RVInstrument(
        [{"name": "A", "file": str(data_file), "time_scale": "ut"}],
        ConfigManager({}),
    )
    assert inst.time_specs[0]["scale"] == "ut1"
    with pytest.raises(ValueError, match="time_scale"):
        RVInstrument(
            [{"name": "A", "file": str(data_file), "time_scale": "gps"}],
            ConfigManager({}),
        )
    with pytest.raises(ValueError, match="time_frame"):
        RVInstrument(
            [{"name": "A", "file": str(data_file), "time_frame": "mjd"}],
            ConfigManager({}),
        )


def test_converted_times_are_sorted_and_aligned(tmp_path):
    """
    Given an unsorted JD_UTC file whose rv column tags each row,
    When the data are loaded with a conversion,
    Then the output is time-sorted and the rv values still ride with
    their own rows.
    """
    rows = [
        (2457090.0, 2.0, 1.0),
        (2457000.0, 0.0, 1.0),
        (2457030.0, 1.0, 1.0),
    ]
    inst = _load(
        tmp_path,
        {"time_scale": "utc", "time_frame": "jd"},
        user_params=COORD_PARAMS,
        rows=rows,
    )
    assert np.all(np.diff(inst.time) > 0)
    np.testing.assert_allclose(inst.rv / RV_FACTOR, [0.0, 1.0, 2.0], atol=1e-9)


def _WHITEN(col):
    """The whitening _build_block_detrend applies (review 6.5.2)."""
    col = np.asarray(col, dtype=float)
    return (col - np.mean(col)) / np.std(col)


# ----------------------------------------------------------------------
# columns
# ----------------------------------------------------------------------
def test_columns_remap_matches_canonical_file(tmp_path):
    """
    Given a shuffled file (err, rv, x, time, y) and a columns spec,
    When the data are loaded,
    Then times/rv/err and the chosen detrend column equal a canonical
    (time, rv, err, detrend) load of the same values.
    """
    rows = [
        (0.5, 10.0, 99.0, 2.0, 7.0),
        (0.25, 20.0, 98.0, 1.0, 8.0),
    ]
    inst = _load(
        tmp_path,
        {"columns": {"time": 3, "rv": 1, "err": 0, "detrend": [4]}},
        rows=rows,
    )
    np.testing.assert_allclose(inst.time, [1.0, 2.0])
    np.testing.assert_allclose(inst.rv / RV_FACTOR, [20.0, 10.0])
    np.testing.assert_allclose(inst.err / RV_FACTOR, [0.25, 0.5])
    assert inst.total_detrend_cols == 1
    np.testing.assert_allclose(inst.detrend_matrix[:, 0], _WHITEN([8.0, 7.0]))


def test_partial_columns_defaults_unnamed_roles(tmp_path):
    """
    Given a columns spec naming only the detrend list,
    When the data are loaded,
    Then time/rv/err keep their canonical positions and only the listed
    detrend column is used (column 3 is deliberately skipped).
    """
    rows = [(1.0, 5.0, 0.1, 111.0, 42.0), (2.0, 6.0, 0.2, 222.0, 43.0)]
    inst = _load(tmp_path, {"columns": {"detrend": [4]}}, rows=rows)
    np.testing.assert_allclose(inst.time, [1.0, 2.0])
    np.testing.assert_allclose(inst.rv / RV_FACTOR, [5.0, 6.0])
    assert inst.total_detrend_cols == 1
    np.testing.assert_allclose(
        inst.detrend_matrix[:, 0], _WHITEN([42.0, 43.0])
    )


def test_columns_spec_without_detrend_disables_detrending(tmp_path):
    """
    Given a columns spec that lists no detrend columns on a file with
    extra columns,
    When the data are loaded,
    Then no detrend columns are picked up (an explicit layout leaves no
    'rest of the columns' to guess at).
    """
    rows = [(1.0, 5.0, 0.1, 111.0), (2.0, 6.0, 0.2, 222.0)]
    inst = _load(tmp_path, {"columns": {"time": 0}}, rows=rows)
    assert inst.total_detrend_cols == 0


def test_unknown_role_raises(tmp_path):
    """
    Given a columns spec naming a role this component does not have,
    When the data are loaded,
    Then a ValueError lists the valid roles.
    """
    with pytest.raises(ValueError, match="unknown role.*'flux'"):
        _load(tmp_path, {"columns": {"flux": 1}})


def test_column_index_out_of_range_raises(tmp_path):
    """
    Given a columns spec pointing past the file's last column,
    When the data are loaded,
    Then a ValueError reports the file's column count.
    """
    with pytest.raises(ValueError, match="0-based"):
        _load(tmp_path, {"columns": {"err": 12}})


def test_malformed_columns_specs_raise_at_construction(tmp_path):
    """
    Given structurally invalid columns specs,
    When the instrument is constructed,
    Then ValueErrors are raised before any file is read.
    """
    data_file = tmp_path / "inst.rv"
    _write_rv_file(data_file, JDS)

    def _construct(columns):
        RVInstrument(
            [{"name": "A", "file": str(data_file), "columns": columns}],
            ConfigManager({}),
        )

    with pytest.raises(ValueError, match="must be a mapping"):
        _construct([0, 1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        _construct({"time": -1})
    with pytest.raises(ValueError, match="detrend must be a list"):
        _construct({"detrend": 3})
    with pytest.raises(ValueError, match="non-negative"):
        _construct({"time": True})


# ----------------------------------------------------------------------
# columns: duplicate-index detection (review 3.5)
# ----------------------------------------------------------------------
def test_partial_columns_spec_colliding_with_a_default_raises(tmp_path):
    """
    Given a partial columns spec that puts time on column 1, which is
      also rv's unnamed canonical position,
    When the data are loaded,
    Then a ValueError names both roles and the shared column -- pre-fix
      the rv array was silently a second copy of the times.
    """
    # Arrange: file is (rv, time, err); only `time` is named.
    rows = [(10.0, 1.0, 0.1), (20.0, 2.0, 0.2)]

    # Act / Assert
    with pytest.raises(ValueError, match=r"same file column 1"):
        _load(tmp_path, {"columns": {"time": 1}}, rows=rows)


def test_two_explicitly_named_roles_on_one_column_raises(tmp_path):
    """
    Given a columns spec that names two roles on the same column,
    When the data are loaded,
    Then a ValueError is raised -- an explicit collision is a typo, not
      an opt-in.
    """
    rows = [(1.0, 10.0, 0.1), (2.0, 20.0, 0.2)]
    with pytest.raises(ValueError, match=r"same file column 2"):
        _load(tmp_path, {"columns": {"rv": 2, "err": 2}}, rows=rows)


def test_time_may_also_be_a_detrend_column(tmp_path):
    """
    Given a columns spec that lists the time column in `detrend` as well,
    When the data are loaded,
    Then the load succeeds and the detrend column IS the time vector --
      detrending against a linear time trend is the one legitimate reuse.
    """
    # Arrange
    rows = [(1.0, 5.0, 0.1), (2.0, 6.0, 0.2)]

    # Act
    inst = _load(
        tmp_path,
        {"columns": {"time": 0, "rv": 1, "err": 2, "detrend": [0]}},
        rows=rows,
    )

    # Assert
    np.testing.assert_allclose(inst.time, [1.0, 2.0])
    assert inst.total_detrend_cols == 1
    np.testing.assert_allclose(inst.detrend_matrix[:, 0], _WHITEN([1.0, 2.0]))


def test_repeated_detrend_column_raises(tmp_path):
    """
    Given a detrend list that names the same column twice,
    When the data are loaded,
    Then a ValueError is raised -- two identical basis vectors are an
      exactly degenerate pair of coefficients, so the time exemption
      covers time-as-detrend, not any repeat.
    """
    rows = [(1.0, 5.0, 0.1, 7.0), (2.0, 6.0, 0.2, 8.0)]
    with pytest.raises(ValueError, match=r"same file column 3"):
        _load(tmp_path, {"columns": {"detrend": [3, 3]}}, rows=rows)


def test_non_time_role_reused_as_detrend_column_raises(tmp_path):
    """
    Given a detrend list that names the rv (observable) column,
    When the data are loaded,
    Then a ValueError is raised -- only `time` is exempt, and detrending
      the observable against itself is degenerate.
    """
    rows = [(1.0, 5.0, 0.1), (2.0, 6.0, 0.2)]
    with pytest.raises(ValueError, match=r"same file column 1"):
        _load(
            tmp_path,
            {"columns": {"time": 0, "rv": 1, "err": 2, "detrend": [1]}},
            rows=rows,
        )


def test_full_columns_spec_without_collisions_still_loads(tmp_path):
    """
    Given a fully specified, collision-free columns spec,
    When the data are loaded,
    Then it loads unchanged -- the new check must not reject valid specs.
    """
    rows = [(0.5, 10.0, 2.0), (0.25, 20.0, 1.0)]
    inst = _load(
        tmp_path, {"columns": {"time": 2, "rv": 1, "err": 0}}, rows=rows
    )
    np.testing.assert_allclose(inst.time, [1.0, 2.0])
    np.testing.assert_allclose(inst.rv / RV_FACTOR, [20.0, 10.0])


def test_component_declared_shareable_roles_are_exempt():
    """
    Given an abs-mode astrometry layout whose single uncertainty column
      serves both sky axes (`err_e` and `err_n` on column 3),
    When the duplicate check runs,
    Then it is accepted -- the component declares that pair shareable --
      while rel mode, which declares none, rejects the same layout.
    """
    # Arrange
    from exozippy.components.instrument import Instrument

    check = Instrument._check_no_duplicate_columns

    # Act / Assert: abs mode, err_e and err_n both on column 3.
    check(
        "astrometryinstrument[A]",
        ("time", "ra", "dec", "err_e", "err_n"),
        [0, 1, 2, 3, 3],
        [],
        (("err_e", "err_n"),),
    )

    # rel mode declares no shareable pair (mas vs deg) -> raises.
    with pytest.raises(ValueError, match=r"same file column 3"):
        check(
            "astrometryinstrument[A]",
            ("time", "sep", "err_sep", "pa", "err_pa"),
            [0, 1, 3, 2, 3],
            [],
            (),
        )


def test_columns_compose_with_mask_and_offset(tmp_path):
    """
    Given a shuffled file with a row mask, a columns spec and a
    time_offset together,
    When the data are loaded,
    Then the mask applies to on-disk row order, the remapped time column
    gets the offset, and the survivors come out sorted.
    """
    rows = [
        (10.0, 3.0, 1.0),  # rv, time, err ; on-disk row 0 (masked out)
        (20.0, 1.0, 1.0),
        (30.0, 2.0, 1.0),
    ]
    inst = _load(
        tmp_path,
        {
            "columns": {"time": 1, "rv": 0, "err": 2},
            "mask": [0],
            "time_offset": 100.0,
        },
        rows=rows,
    )
    np.testing.assert_allclose(inst.time, [101.0, 102.0])
    np.testing.assert_allclose(inst.rv / RV_FACTOR, [20.0, 30.0])
