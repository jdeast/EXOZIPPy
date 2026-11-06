"""The observer is placed where it was when it RECEIVED the light (review 1.10.13).

A BJD_TDB tag is the arrival time at the solar-system barycenter; the
observer received that light earlier, by the Roemer delay (r . n_hat)/c, up
to 8.3 min, during which the Earth moves 1e-4 AU.  Every observer-position
lookup now solves for that instant first (conventions.md C4).  Four pins:

1. ``reception_instant`` against astropy's own barycentric light-travel time.
2. A spacecraft's delay is computed with the SPACECRAFT's position, not the
   Earth's (Spitzer trails the Earth by ~1 AU, so the two differ by minutes).
3. The delay we remove from a BJD_TDB tag is the one VBMicrolensing removes
   from an HJD tag -- the one code of the three compared in review 8.14.13
   that keeps the two time arguments apart.  Run with ``t_in_HJD`` off, VBM
   adds its light-travel term to the trajectory time, so its returned tau
   exposes the term epoch by epoch; it matches astropy's heliocentric delay
   to 0.2 s and ours to the Sun-barycenter difference.
4. ``MulensEvent.geocentric_frame`` anchors at the reception instant of
   ``t0_par`` with the physical velocity (C5); a data-free event gets its
   line of sight from the source star at stage 3, and before that it
   refuses to guess one.
"""

import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import (
    EarthLocation,
    SkyCoord,
    get_body_barycentric,
    get_body_barycentric_posvel,
)
from astropy.time import Time

from exozippy.constants import C_AU_PER_DAY
from exozippy.ephemeris import (
    get_observer_position,
    interpolate_ephemeris,
    observer_position_at_reception,
    reception_instant,
)
from exozippy.skyframe import sky_basis

# Toward the bulge: the line of sight lies close to the ecliptic, so the
# Roemer delay swings through nearly its full +/-8.3 min over a year.
RA_DEG, DEC_DEG = 269.271, -30.383
RA, DEC = np.radians(RA_DEG), np.radians(DEC_DEG)
COORD = SkyCoord(ra=RA_DEG * u.deg, dec=DEC_DEG * u.deg)
GEOCENTER = EarthLocation.from_geocentric(0, 0, 0, unit="m")


def _astropy_reception_jd(bjd):
    """bjd - ltt_bary(jd), iterated, with astropy doing the light travel."""
    jd = np.array(bjd, dtype=float)
    for _ in range(3):
        ltt = Time(
            jd, format="jd", scale="tdb", location=GEOCENTER
        ).light_travel_time(COORD, kind="barycentric")
        jd = bjd - ltt.jd
    return jd


def _earth(jd, helio=False):
    tt = Time(jd, format="jd", scale="tdb")
    e = get_body_barycentric("earth", tt)
    if helio:
        e = e - get_body_barycentric("sun", tt)
    return e.xyz.T.to(u.au).value


# ----------------------------------------------------------------------
# 1. the instant
# ----------------------------------------------------------------------
def test_reception_instant_is_the_barycentric_light_travel_fixed_point():
    """
    Given BJD_TDB tags spread over a year toward the bulge,
    When reception_instant solves jd = bjd - (r(jd) . n_hat)/c,
    Then it matches astropy's iterated barycentric light-travel time to
      0.1 ms, the delays reach several minutes of either sign, and the
      Earth placed at the instant rather than at the tag moves by ~1e-4 AU.
    """
    bjd = 2455775.5 + np.linspace(0.0, 365.0, 13)

    jd = reception_instant(bjd, RA, DEC)
    jd_ref = _astropy_reception_jd(bjd)

    np.testing.assert_allclose(jd, jd_ref, rtol=0, atol=1e-9)
    delay_min = (bjd - jd) * 1440.0
    assert delay_min.max() > 5.0 and delay_min.min() < -5.0
    shift = np.linalg.norm(
        observer_position_at_reception(bjd, RA, DEC)
        - get_observer_position(bjd),
        axis=1,
    )
    assert shift.max() > 8e-5 and shift.max() < 1.2e-4


# ----------------------------------------------------------------------
# 2. a spacecraft uses its own position
# ----------------------------------------------------------------------
def test_spacecraft_delay_uses_the_spacecrafts_own_position():
    """
    Given Spitzer, ~1 AU behind the Earth in 2015,
    When its reception instant is solved for a BJD_TDB tag,
    Then it is bjd minus Spitzer's own (r . n_hat)/c from its ephemeris file
      (to 1e-9 d), and differs from the Earth's instant by minutes -- the
      Earth's correction is NOT a stand-in for a distant observer's.
    """
    bjd = np.array([2457200.0, 2457290.0])
    n_hat = np.array(
        [np.cos(DEC) * np.cos(RA), np.cos(DEC) * np.sin(RA), np.sin(DEC)]
    )

    jd_sc = reception_instant(bjd, RA, DEC, observer_location="spitzer")
    jd_earth = reception_instant(bjd, RA, DEC)

    xyz_sc = get_observer_position(jd_sc, observer_location="spitzer")
    np.testing.assert_allclose(
        jd_sc, bjd - (xyz_sc @ n_hat) / C_AU_PER_DAY, rtol=0, atol=1e-9
    )
    assert np.max(np.abs(jd_sc - jd_earth)) * 1440.0 > 1.0
    # and the position returned is the file's position at that instant
    np.testing.assert_allclose(
        observer_position_at_reception(
            bjd, RA, DEC, observer_location="spitzer"
        ),
        interpolate_ephemeris(jd_sc, _spitzer_path()),
        rtol=0,
        atol=1e-12,
    )


def _spitzer_path():
    import os

    from exozippy.ephemeris import EPHEMERIDES_DIR

    return os.path.join(EPHEMERIDES_DIR, "spitzer.eph")


# ----------------------------------------------------------------------
# 3. against VBMicrolensing's own HJD <-> JD conversion
# ----------------------------------------------------------------------
def _vbm_coord_string():
    h = RA_DEG / 15.0
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = ((h - hh) * 60 - mm) * 60
    sgn = "-" if DEC_DEG < 0 else "+"
    d = abs(DEC_DEG)
    dd = int(d)
    dm = int((d - dd) * 60)
    ds = ((d - dd) * 60 - dm) * 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f} {sgn}{dd:02d}:{dm:02d}:{ds:05.2f}"


def test_reception_delay_matches_vbmicrolensings_light_travel_term():
    """
    Given a year of geocentric epochs toward the bulge,
    When VBMicrolensing is run with t_in_HJD off and pi_E = 0 -- it then
      ADDS its own light-travel term to the trajectory time, so its
      returned tau exposes that term epoch by epoch -- and our
      reception_instant is run on the same epochs' BJD_TDB tags,
    Then the two delays agree to the Sun-barycenter difference (< 6 s,
      against a term that swings by more than 10 min over the year): the
      Roemer delay we now remove from BJD_TDB tags before placing the
      Earth is the one VBM removes from HJD tags (ComputeParallax,
      t_in_HJD) -- the handling of the three codes in review 8.14.13 that
      keeps the two time arguments apart.  Placing the Earth at the tag
      is a delay of zero and fails this by minutes.
    VBM's internal Earth VECTOR is not compared: it differs from a
    geometric heliocentric Earth by a fixed ~8e-5 AU map that is VBM's own
    (review 8.14.13), too large to see a 1e-4 AU effect through.
    """
    import VBMicrolensing

    # Arrange: geocentric epochs, and their BJD_TDB tags
    jd = 2455600.0 + np.arange(0.0, 366.0, 7.0)
    tt = Time(jd, format="jd", scale="tdb", location=GEOCENTER)
    ltt_helio = tt.light_travel_time(COORD, kind="heliocentric").jd
    ltt_bary = tt.light_travel_time(COORD, kind="barycentric").jd
    bjd = jd + ltt_bary

    u0, tE, t0 = 0.3, 60.0, jd[len(jd) // 2] - 2450000.0
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.SetObjectCoordinates(_vbm_coord_string())
    vbm.parallaxsystem = 1
    vbm.t_in_HJD = False  # tags are JD: VBM adds ltt to the trajectory time
    par = [u0, math.log(tE), t0, 0.0, 0.0]

    # Act: VBM's tau = (JD + ltt(JD) - t0 - ltt(t0)) / tE, so
    _, y1, _ = [
        np.array(r)
        for r in vbm.PSPLLightCurveParallax(par, list(jd - 2450000.0))
    ]
    delay_vbm = -y1 * tE - (jd - 2450000.0 - t0)  # ltt(JD) - ltt(t0), days
    # ours: BJD_TDB tag minus the reception instant
    delay_ours = bjd - reception_instant(bjd, RA, DEC)
    delay_ours = delay_ours - np.interp(t0 + 2450000.0, jd, delay_ours)

    # Assert
    seconds = 86400.0
    assert (
        np.ptp(delay_vbm) * seconds > 600.0
    )  # the term is real: > 10 min range
    np.testing.assert_allclose(
        delay_vbm * seconds,
        (ltt_helio - np.interp(t0 + 2450000.0, jd, ltt_helio)) * seconds,
        rtol=0,
        atol=0.2,
    )  # VBM's term IS the heliocentric Roemer delay, to 0.2 s
    assert (
        np.max(np.abs(delay_ours - delay_vbm)) * seconds < 6.0
    )  # Sun vs barycenter
    assert (
        np.max(np.abs(0.0 - delay_vbm)) * seconds > 300.0
    )  # the Earth-at-tag delay


# ----------------------------------------------------------------------
# 4. the frame
# ----------------------------------------------------------------------
def _write_lc(path, t):
    """A flat 19th-mag light curve: load_data only needs epochs."""
    np.savetxt(
        path,
        np.column_stack([t, np.full_like(t, 19.0), np.full_like(t, 0.01)]),
    )


def _build(tmp_path, with_data=True):
    from exozippy.system import System

    cfg = {
        "name": "reception",
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [{"t0_par": 2455775.5}],
        "lens": [{"body": "star.Lens", "name": "Lens"}],
        "source": [{"body": "star.Source", "name": "Source"}],
    }
    if with_data:
        t = np.linspace(2455700.0, 2455850.0, 60)
        _write_lc(tmp_path / "lc.dat", t)
        cfg["mulensinstrument"] = [
            {"name": "OGLE", "file": str(tmp_path / "lc.dat")}
        ]
    params = {
        "source.Source.t_0": {"initval": 2455775.5},
        "source.Source.u_0": {"initval": 0.3},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.pm_ra": {"initval": 2.0},
        "star.Lens.pm_dec": {"initval": -5.0},
        "star.Source.pm_ra": {"initval": 0.0},
        "star.Source.pm_dec": {"initval": 0.0},
        "star.Source.ra": {"initval": RA_DEG},
        "star.Source.dec": {"initval": DEC_DEG},
        "star.Lens.ra": {"initval": RA_DEG},
        "star.Lens.dec": {"initval": DEC_DEG},
    }
    system = System(cfg, user_params=params)
    system.prepare()
    return system


def test_geocentric_frame_anchors_at_the_reception_instant_of_t0_par(tmp_path):
    """
    Given a microlensing system whose instrument has handed the event its
      line of sight,
    When geocentric_frame builds the Skowron anchors,
    Then the position is the Earth at the reception instant of t0_par
      (not at the tag), and the velocity is the physical one to 3e-7
      relative (astropy's own posvel at that instant), i.e. the +/-0.05 d
      central difference of C5.
    """
    system = _build(tmp_path)
    event = system.mulensevent
    t0_par = float(event.t0_par[0])

    t0, pos, vel = event.geocentric_frame()

    assert t0 == t0_par
    jd_rec = reception_instant(np.array([t0_par]), RA, DEC)
    np.testing.assert_allclose(
        pos, get_observer_position(jd_rec)[0], rtol=0, atol=1e-12
    )
    assert (
        np.linalg.norm(pos - get_observer_position(np.array([t0_par]))[0])
        > 3e-5
    )
    _, v_exact = get_body_barycentric_posvel(
        "earth", Time(jd_rec, format="jd", scale="tdb")
    )
    v_exact = v_exact.xyz.T.to(u.au / u.day).value[0]
    assert np.linalg.norm(vel - v_exact) / np.linalg.norm(v_exact) < 3e-7


def test_data_free_event_anchors_the_frame_at_the_source_star(tmp_path):
    """
    Given an event with no microlensing data (no instrument handed it a
      line of sight at stage 1),
    When the system is prepared and the frame is asked for,
    Then register_frame_constants resolved the line of sight from the
      source star's own coordinates at stage 3, and the frame is anchored
      at the reception instant toward it -- the same anchors a light-curve
      system gets.  Before stage 3 the event refuses to guess a direction.
    """
    from exozippy.system import System

    system = _build(tmp_path, with_data=False)

    t0, pos, vel = system.mulensevent.geocentric_frame()

    assert system.mulensevent._line_of_sight_rad == pytest.approx((RA, DEC))
    jd_rec = reception_instant(np.array([t0]), RA, DEC)
    np.testing.assert_allclose(
        pos, get_observer_position(jd_rec)[0], rtol=0, atol=1e-12
    )
    # an event that has not reached stage 3 has no direction to use
    bare = System(
        {
            "name": "bare",
            "star": [{"name": "Lens"}, {"name": "Source"}],
            "mulensevent": [{"t0_par": 2455775.5}],
            "lens": [{"body": "star.Lens", "name": "Lens"}],
            "source": [{"body": "star.Source", "name": "Source"}],
        },
        user_params={},
    )
    with pytest.raises(RuntimeError, match="set_line_of_sight"):
        bare.mulensevent.geocentric_frame()
