"""Time-system conversion (``TimeSystem._to_bjd_tdb``): a published known
answer and a round trip.

Review 7.16.2.  ``tests/test_instrument_time_columns.py`` already pins the
forward conversion against astropy's own chain (JD_UTC -> BJD_TDB, and
HJD_TDB -> BJD_TDB from known instants).  Two things it does not do:

* compare with a number that does not come from astropy at all.  The
  scale-only BJD_UTC -> BJD_TDB shift is TT - UTC = 32.184 s + the
  accumulated leap seconds (IERS Bulletin C; Eastman, Siverd & Gaudi 2010,
  Section 2), plus TDB - TT, whose amplitude is 1.66 ms.  That is a fact
  about the calendar, so a wrong scale name, a TAI/TT mix-up or a dropped
  32.184 s fails it whatever astropy does.
* close a ROUND TRIP through everything a real file uses at once: a
  truncated time (MJD via ``time_offset``), a UTC scale, a geocentric JD
  frame and a topocentric ``time_location``.  The inverse is built here by
  fixed-point iteration on astropy's barycentric correction, so the loader
  has to land back on the BJD_TDB it started from.

RVInstrument stands in for every data component, as in
test_instrument_time_columns.py.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from exozippy.components.rvinstrument.rvinstrument import RVInstrument
from exozippy.config import ConfigManager

# Vega, so the light-travel geometry is a real one.
RA_DEG, DEC_DEG = 279.23473479, 38.78368896
COORD_PARAMS = {"star.0.ra": RA_DEG, "star.0.dec": DEC_DEG}
MAUNA_KEA = (-155.5, 19.8, 4200.0)  # lon, lat (deg), height (m)


def _load(tmp_path, entry_extra, times, user_params=None, fmt="{:.12f}"):
    data_file = tmp_path / "inst.rv"
    with open(data_file, "w") as f:
        f.write("# test data\n")
        for i, t in enumerate(times):
            f.write(f"{fmt.format(t)} {float(i):.3f} 1.0\n")
    config = [{"name": "Inst", "file": str(data_file), **entry_extra}]
    inst = RVInstrument(config, ConfigManager(user_params or {}))
    inst.load_data(system=None)
    return inst


def test_bjd_utc_to_bjd_tdb_is_32_184_s_plus_leap_seconds(tmp_path):
    """
    Given BJD_UTC times on either side of the 2012-07-01 and 2015-07-01
      leap seconds (TAI - UTC = 34, 35, 36 s),
    When the file is loaded with time_scale utc (frame already bjd),
    Then each shift is 32.184 s + (TAI - UTC) to within TDB - TT's
      1.66 ms amplitude -- the published leap-second table, not astropy.
    """
    # (JD_UTC, TAI - UTC in s) from IERS Bulletin C.
    cases = [
        (2456000.0, 34.0),  # 2012-03-14
        (2456200.0, 35.0),  # 2012-09-30
        (2457100.0, 35.0),  # 2015-03-17
        (2457300.0, 36.0),  # 2015-10-03
        (2458000.0, 37.0),  # 2017-09-04 (after 2017-01-01)
    ]
    jds = [jd for jd, _ in cases]

    inst = _load(tmp_path, {"time_scale": "utc"}, times=jds)

    shift = (inst.time - np.array(jds)) * 86400.0
    expected = 32.184 + np.array([leap for _, leap in cases])
    # 1.66 ms TDB - TT amplitude, plus the ~40 us float64 JD quantization.
    np.testing.assert_allclose(shift, expected, rtol=0, atol=2e-3)


def test_mjd_utc_topocentric_round_trips_to_bjd_tdb(tmp_path):
    """
    Given known BJD_TDB instants, inverted HERE to the topocentric JD_UTC
      at which a Mauna Kea observer received the light (fixed-point
      iteration on astropy's barycentric correction), written as MJD,
    When the file is loaded with time_offset 2400000.5, time_scale utc,
      time_frame jd and that time_location,
    Then the loader returns the starting BJD_TDB to < 50 us (the float64
      JD quantization; one ulp of a JD is 40 us).
    """
    coord = SkyCoord(ra=RA_DEG * u.deg, dec=DEC_DEG * u.deg)
    loc = EarthLocation.from_geodetic(*MAUNA_KEA)
    bjd_tdb = np.array([2457000.1, 2457061.3, 2457123.7, 2457201.9])

    # Inverse: find t_utc with tdb(t_utc) + ltt_bary(t_utc) == bjd_tdb.
    t_tdb = bjd_tdb.copy()
    for _ in range(5):
        ltt = (
            Time(t_tdb, format="jd", scale="tdb", location=loc)
            .light_travel_time(coord, kind="barycentric", ephemeris="builtin")
            .jd
        )
        t_tdb = bjd_tdb - ltt
    jd_utc = Time(t_tdb, format="jd", scale="tdb", location=loc).utc.jd
    mjd_utc = jd_utc - 2400000.5
    # The barycentric correction toward Vega is minutes, so the round trip
    # crosses a real frame change, not a no-op.
    assert np.min(np.abs(bjd_tdb - jd_utc)) * 86400.0 > 60.0

    inst = _load(
        tmp_path,
        {
            "time_offset": 2400000.5,
            "time_scale": "utc",
            "time_frame": "jd",
            "time_location": list(MAUNA_KEA),
        },
        times=mjd_utc,
        user_params=COORD_PARAMS,
        fmt="{:.12f}",
    )

    np.testing.assert_allclose(
        inst.time, bjd_tdb, rtol=0, atol=50e-6 / 86400.0
    )


def test_tai_and_tt_scales_are_the_defined_offsets(tmp_path):
    """
    Given the same BJD instants labeled TAI and labeled TT,
    When each file is loaded (frame already bjd),
    Then TT -> TDB moves by TDB - TT only (|.| < 1.7 ms) and TAI -> TDB by
      exactly 32.184 s more -- TT = TAI + 32.184 s is a definition (IAU
      1991), not a model, so this pins the scale names to the right clocks.
    """
    jds = (2456000.0, 2457300.0, 2458000.0)
    tt = _load(tmp_path, {"time_scale": "tt"}, times=jds)
    tai = _load(tmp_path, {"time_scale": "tai"}, times=jds)

    shift_tt = (tt.time - np.array(jds)) * 86400.0
    shift_tai = (tai.time - np.array(jds)) * 86400.0
    assert np.all(np.abs(shift_tt) < 1.7e-3)
    np.testing.assert_allclose(shift_tai - shift_tt, 32.184, rtol=0, atol=1e-4)


def test_mission_time_offsets_are_the_archive_definitions():
    """
    Given getdata's per-mission offsets from the archive time column to
      BJD_TDB,
    When compared with the missions' own definitions,
    Then Kepler/K2 are BKJD = BJD - 2454833.0 (Kepler Archive Manual) and
      TESS is BTJD = BJD - 2457000.0 (TESS Science Data Products), and
      adding the offset and subtracting it again is the identity (round
      trip).
    """
    from exozippy.utilities.getdata import MISSIONS

    assert MISSIONS["Kepler"].bjd_offset == 2454833.0
    assert MISSIONS["K2"].bjd_offset == 2454833.0
    assert MISSIONS["TESS"].bjd_offset == 2457000.0
    btjd = np.array([1325.293, 2000.5])
    bjd = btjd + MISSIONS["TESS"].bjd_offset
    # One ulp of a full JD is 4.7e-10 d.
    np.testing.assert_allclose(
        bjd - MISSIONS["TESS"].bjd_offset, btjd, rtol=0, atol=1e-9
    )
