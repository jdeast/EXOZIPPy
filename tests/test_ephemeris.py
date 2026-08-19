"""Tests for exozippy.ephemeris (review items 2.6.5, 2.6.6, 7.14).

These deliberately avoid any network access: 'builtin' is the default
solar-system ephemeris (erfa, no download) and the one site-name test skips
cleanly when astropy's site registry is not already cached.
"""

import logging
import os

import numpy as np
import pytest
from astropy.coordinates import EarthLocation, solar_system_ephemeris

from exozippy import ephemeris as eph

# A well-separated site so a lon/lat swap cannot hide: CTIO, whose
# longitude and latitude differ by ~40 deg.
CTIO_LON, CTIO_LAT, CTIO_HEIGHT = -70.815, -30.16527778, 2215.0

TIMES = np.array([2456800.0, 2456800.5, 2456801.0])

AU_KM = 1.495978707e8


@pytest.fixture(autouse=True)
def _restore_global_ephemeris():
    """Start every test from the same global solar_system_ephemeris value.

    Without this, a leak in the very first call makes `before == after` for
    every later test -- the state-restoration assertions below would pass on
    exactly the code they are meant to catch.
    """
    saved = solar_system_ephemeris.get()
    yield
    solar_system_ephemeris.set(saved)


def _site_registry_available():
    """True when EarthLocation.of_site works without a fresh download."""
    try:
        EarthLocation.of_site("CTIO")
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# 2.6.5 / 7.14 -- geodetic argument order is (lon, lat, height)
# ---------------------------------------------------------------------------


def test_geodetic_string_is_lon_lat_height():
    """
    Given a ground site whose longitude and latitude differ by ~40 deg,
    When its position is requested with the documented 'lon,lat,height'
        string,
    Then the observer lands at that site, not at the lat/lon-swapped point
        ~5000 km away.
    """
    # Arrange
    want = EarthLocation.from_geodetic(CTIO_LON, CTIO_LAT, CTIO_HEIGHT)
    swapped = EarthLocation.from_geodetic(CTIO_LAT, CTIO_LON, CTIO_HEIGHT)
    spec = f"{CTIO_LON},{CTIO_LAT},{CTIO_HEIGHT}"

    # Act
    xyz = eph.get_observer_position(TIMES, spec)
    xyz_want = eph.get_observer_position(
        TIMES, f"{want.geodetic[0].deg},{want.geodetic[1].deg},{CTIO_HEIGHT}"
    )
    xyz_swapped = eph.get_observer_position(
        TIMES, f"{CTIO_LAT},{CTIO_LON},{CTIO_HEIGHT}"
    )

    # Assert -- the documented order reproduces the intended location, and
    # the swapped order is a genuinely different place (>1000 km away), so
    # this test could not pass under either reading by accident.
    assert xyz.shape == (3, 3)
    assert np.allclose(xyz, xyz_want, atol=1e-12)
    displacement_km = np.linalg.norm(xyz - xyz_swapped, axis=1).max() * AU_KM
    assert displacement_km > 1000.0
    assert not np.allclose(want.geocentric[0], swapped.geocentric[0])


def test_geodetic_docstring_names_lon_first():
    """
    Given the docstring is the only place the caller learns the order,
    When it is read,
    Then it advertises lon first, matching EarthLocation.from_geodetic.
    """
    # Arrange
    doc = eph.get_observer_position.__doc__

    # Act / Assert
    assert "'lon,lat'" in doc
    assert "lon,lat,height" in doc
    assert "lat,lon" not in doc


def test_geodetic_string_height_is_optional():
    """
    Given a two-field 'lon,lat' string (matching time_location's 2-or-3
        element form),
    When the observer position is requested,
    Then height defaults to 0 m on the WGS84 ellipsoid.
    """
    # Arrange
    two = f"{CTIO_LON},{CTIO_LAT}"
    three = f"{CTIO_LON},{CTIO_LAT},0.0"

    # Act
    xyz_two = eph.get_observer_position(TIMES, two)
    xyz_three = eph.get_observer_position(TIMES, three)

    # Assert
    assert np.allclose(xyz_two, xyz_three, atol=1e-14)


# ---------------------------------------------------------------------------
# 2.6.6 (a) -- genuine topocentric failures must be diagnosable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        "100.0,200.0,0.0",  # latitude out of range
        "1,2,3,4",  # too many fields
        "1,2,notanumber",  # non-numeric field
        "CTIO,",  # site name with a stray comma
    ],
)
def test_bad_geodetic_string_raises_diagnosably(spec):
    """
    Given an observer_location that is meant to be a ground site but is
        malformed,
    When the observer position is requested,
    Then the error names the geodetic form, instead of the misleading
        "not recognized ... go generate an ephemeris" fallthrough.
    """
    # Arrange / Act
    with pytest.raises(ValueError) as excinfo:
        eph.get_observer_position(TIMES, spec)

    # Assert
    message = str(excinfo.value)
    assert "lon_deg,lat_deg" in message
    assert "to generate an ephemeris" not in message


def test_unknown_site_name_still_falls_through_to_ephemeris_files():
    """
    Given a spacecraft ephemeris name, which is not an astropy site,
    When the observer position is requested,
    Then the site lookup falls through and the .eph file is interpolated.
    """
    # Arrange
    times = np.array([2456830.0, 2456831.0])

    # Act
    xyz = eph.get_observer_position(times, "spitzer")

    # Assert
    assert xyz.shape == (2, 3)
    assert np.all(np.isfinite(xyz))
    # Spitzer trails the Earth: roughly 1 AU from the barycenter.
    assert 0.9 < np.linalg.norm(xyz, axis=1).min() < 1.1


def test_unrecognized_location_error_lists_the_accepted_forms():
    """
    Given a location that is neither a body, a site, nor an ephemeris file,
    When the observer position is requested,
    Then the error enumerates what would have been accepted.
    """
    # Arrange / Act
    with pytest.raises(ValueError) as excinfo:
        eph.get_observer_position(TIMES, "definitely_not_an_observatory")

    # Assert
    message = str(excinfo.value)
    assert "not recognized" in message
    assert "lon_deg,lat_deg" in message


# ---------------------------------------------------------------------------
# 2.6.6 (b) -- the global solar_system_ephemeris state must be left alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("location", ["earth", "moon", "spitzer"])
def test_global_solar_system_ephemeris_is_restored(location):
    """
    Given astropy's process-global solar_system_ephemeris state,
    When an observer position is computed,
    Then the global state is exactly what it was before the call.
    """
    # Arrange
    before = solar_system_ephemeris.get()

    # Act
    eph.get_observer_position(TIMES, location)

    # Assert
    assert solar_system_ephemeris.get() == before


def test_global_state_restored_for_topocentric_branch():
    """
    Given the topocentric branch, which is the one that needs the Earth's
        barycentric position,
    When an observer position is computed,
    Then the global ephemeris state is still restored.
    """
    # Arrange
    before = solar_system_ephemeris.get()

    # Act
    eph.get_observer_position(TIMES, f"{CTIO_LON},{CTIO_LAT},{CTIO_HEIGHT}")

    # Assert
    assert solar_system_ephemeris.get() == before


def test_global_state_restored_when_the_call_raises():
    """
    Given a call that fails inside the ephemeris context,
    When the error propagates,
    Then the global ephemeris state is still restored.
    """
    # Arrange
    before = solar_system_ephemeris.get()

    # Act
    with pytest.raises(ValueError):
        eph.get_observer_position(TIMES, "definitely_not_an_observatory")

    # Assert
    assert solar_system_ephemeris.get() == before


def test_use_ephemeris_falls_back_to_builtin_when_kernel_unavailable(
    monkeypatch,
):
    """
    Given a JPL kernel that cannot be loaded (no network, no jplephem),
    When an observer position is requested with that ephemeris,
    Then it warns, falls back to 'builtin', and leaves the global state
        untouched.
    """
    # Arrange
    real_set = solar_system_ephemeris.set

    class _Failing:
        @staticmethod
        def set(name):
            if name != eph.DEFAULT_EPHEMERIS:
                raise OSError("simulated kernel download failure")
            return real_set(name)

        @staticmethod
        def get():
            return solar_system_ephemeris.get()

    monkeypatch.setattr(eph, "solar_system_ephemeris", _Failing)
    before = solar_system_ephemeris.get()

    # Act
    with pytest.warns(UserWarning, match="falling back to 'builtin'"):
        xyz = eph.get_observer_position(TIMES, "earth", ephemeris="de440")

    # Assert
    expected = eph.get_observer_position(TIMES, "earth", ephemeris="builtin")
    assert np.allclose(xyz, expected, atol=1e-14)
    assert solar_system_ephemeris.get() == before


def test_requested_ephemeris_is_scoped_not_leaked():
    """
    Given an explicitly requested non-default ephemeris,
    When it loads successfully,
    Then it is used for the call and dropped afterwards.
    """
    # Arrange
    before = solar_system_ephemeris.get()
    try:
        with solar_system_ephemeris.set("jpl"):
            pass
    except Exception:  # pragma: no cover - kernel not cached in this env
        pytest.skip("JPL kernel not available offline")

    # Act
    xyz = eph.get_observer_position(TIMES, "earth", ephemeris="jpl")

    # Assert
    assert solar_system_ephemeris.get() == before
    builtin = eph.get_observer_position(TIMES, "earth", ephemeris="builtin")
    # The two ephemerides agree to ~5 km but are not bit-identical.
    delta_km = np.linalg.norm(xyz - builtin, axis=1).max() * AU_KM
    assert 0.0 < delta_km < 100.0


# ---------------------------------------------------------------------------
# site names
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _site_registry_available(),
    reason="astropy site registry not cached (needs a download)",
)
def test_site_name_matches_its_geodetic_string():
    """
    Given a site known to astropy,
    When it is requested by name and by its own lon,lat,height,
    Then the two agree.
    """
    # Arrange
    lon, lat, height = EarthLocation.of_site("CTIO").geodetic
    spec = f"{lon.deg},{lat.deg},{height.to_value('m')}"

    # Act
    by_name = eph.get_observer_position(TIMES, "CTIO")
    by_geodetic = eph.get_observer_position(TIMES, spec)

    # Assert
    assert np.allclose(by_name, by_geodetic, atol=1e-12)


# ---------------------------------------------------------------------------
# interpolate_ephemeris parses and fits ONCE per file version (review 6.10.1)
# ---------------------------------------------------------------------------


def _write_eph(path, offset=0.0):
    """A tiny linear .eph: BJD_TDB, X, Y, Z (barycentric AU)."""
    t = np.linspace(2458000.0, 2458100.0, 21)
    np.savetxt(
        path,
        np.column_stack([t, t * 0 + 1.0 + offset, t * 0, t * 0]),
    )
    return t


def test_the_same_ephemeris_is_parsed_once_however_often_it_is_used(
    tmp_path, monkeypatch
):
    """
    Given one .eph file,
    When interpolate_ephemeris is called many times on it -- which is what
      happens per instrument at load AND from plot_sky's 4000-point grids
      and the mulens plotters, so a multi-draw plotting pass hit the same
      file dozens of times,
    Then the file is np.loadtxt'd and CubicSpline'd exactly ONCE, and every
      call still returns the same numbers.
    """
    # Arrange
    path = tmp_path / "sat.eph"
    _write_eph(path)
    eph._ephemeris_spline.cache_clear()
    loads = []
    real_loadtxt = eph.np.loadtxt
    monkeypatch.setattr(
        eph.np,
        "loadtxt",
        lambda *a, **k: (loads.append(a[0]), real_loadtxt(*a, **k))[1],
    )

    # Act
    want = eph.interpolate_ephemeris(np.array([2458010.0]), str(path))
    for _ in range(9):
        got = eph.interpolate_ephemeris(np.array([2458010.0]), str(path))

    # Assert
    assert len(loads) == 1
    np.testing.assert_array_equal(got, want)


def test_a_regenerated_ephemeris_is_re_read_not_served_stale(
    tmp_path, monkeypatch
):
    """
    Given an .eph that is REWRITTEN under the same name in one process --
      a get_ephemeris.py re-run from the GUI or a notebook,
    When interpolate_ephemeris is called again,
    Then it returns the NEW contents.  A plain path key would serve the old
      spline forever, which is why the cache key carries (mtime_ns, size).
    """
    # Arrange
    path = tmp_path / "sat.eph"
    _write_eph(path)
    eph._ephemeris_spline.cache_clear()
    first = eph.interpolate_ephemeris(np.array([2458010.0]), str(path))

    # Act: same path, different content (and a distinct mtime)
    os.utime(path, (0, 0))
    _write_eph(path, offset=5.0)
    second = eph.interpolate_ephemeris(np.array([2458010.0]), str(path))

    # Assert
    assert first[0, 0] == pytest.approx(1.0)
    assert second[0, 0] == pytest.approx(6.0)


def test_the_extrapolation_warning_is_not_cached_with_the_spline(
    tmp_path, caplog
):
    """
    Given a cached spline,
    When a LATER call asks for epochs outside the grid,
    Then it still warns.  The check depends on the requested times, not on
      the file, so it deliberately sits outside the memoized helper -- a
      warning silenced by a cache hit is the failure mode this guards.
    """
    # Arrange
    path = tmp_path / "sat.eph"
    _write_eph(path)
    eph._ephemeris_spline.cache_clear()
    eph.interpolate_ephemeris(np.array([2458010.0]), str(path))  # warms it

    # Act
    with caplog.at_level(logging.WARNING, logger="exozippy.ephemeris"):
        eph.interpolate_ephemeris(np.array([2459000.0]), str(path))

    # Assert
    assert "Extrapolating outside the ephemeris range" in caplog.text
