"""Observer positions and the five-parameter track: known answers in the
ICRS/J2000 equatorial frame, and round trips.

Review 7.16.2.  Every observer position the likelihood consumes -- annual
and satellite microlensing parallax, astrometric parallax factors, the
Roemer delay -- comes out of ``ephemeris.get_observer_position`` in
BARYCENTRIC ICRS/J2000 EQUATORIAL AU.  Three ways in, three checks:

* Spacecraft ``.eph`` files.  astroquery's Horizons default is the
  ECLIPTIC frame, rotated 23.4 deg from what every consumer assumes, and
  the files carry no frame the reader can check.  The orbit's own angular
  momentum is a frame fingerprint that needs no reference ephemeris: every
  shipped spacecraft orbits in (or within ~1 deg of) the ecliptic, so in
  the equatorial frame its orbit pole must be the ecliptic pole,
  (0, -sin eps, cos eps) with the IAU 2006 obliquity, and an ecliptic-frame
  file would put it at +Z, 23.4 deg away.  Each file's distance from the
  Earth (astropy) is then checked against what the spacecraft physically
  is: Gaia and the simulated Roman orbit at Sun-Earth L2, ~0.01 AU out;
  Spitzer and Kepler/K2 in Earth-trailing heliocentric orbits, a large
  fraction of an AU behind.
* Ground sites: barycentric Earth + the site's geocentric vector, checked
  against astropy's own GCRS position of that site.
* The gaia/abs five-parameter track (``AstrometryInstrument._star_track``):
  a linear tangent-plane model in mas, Julian years and mu_alpha* -- the
  pm-unit boundary.  Checked against astropy's rigorous space motion
  (``SkyCoord.apply_space_motion``) and propagated there and back.
"""

import glob
import os
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from exozippy.ephemeris import (
    EPHEMERIDES_DIR,
    get_observer_position,
    interpolate_ephemeris,
)

# IAU 2006 obliquity of the ecliptic at J2000, 84381.406 arcsec.
OBLIQUITY = np.radians(84381.406 / 3600.0)
ECLIPTIC_POLE_EQ = np.array([0.0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)])

# What each shipped spacecraft physically is: its distance from the Earth
# in AU, (min, max) over the file.  L2 is 1.5e6 km = 0.010 AU, with
# Lissajous excursions; Spitzer and Kepler drift away in Earth-trailing
# heliocentric orbits (Kepler was ~0.5 AU behind by K2's Campaign 9).
SPACECRAFT_EARTH_DISTANCE_AU = {
    "gaia.eph": (0.005, 0.012),
    "roman_simulated_2018dc.eph": (0.0005, 0.025),
    "spitzer.eph": (0.5, 2.0),
    "k2.eph": (0.1, 2.0),
}
# Orbit pole tolerance: Spitzer's heliocentric orbit is inclined ~1.1 deg
# to the ecliptic; the L2 orbits follow the Earth's.
POLE_TOLERANCE_DEG = {"spitzer.eph": 2.0}


def _eph_files():
    names = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(EPHEMERIDES_DIR, "*.eph"))
    )
    # Every shipped file is checked: a new one has to be added above.
    assert set(names) == set(SPACECRAFT_EARTH_DISTANCE_AU), names
    return names


def _xyz(name):
    data = np.loadtxt(os.path.join(EPHEMERIDES_DIR, name))
    # roman_simulated_2018dc.eph carries equatorial XYZ then ecliptic XYZ;
    # columns 1-3 are what interpolate_ephemeris reads for every file.
    return data[:, 0], data[:, 1:4]


@pytest.mark.parametrize("name", _eph_files())
def test_spacecraft_orbit_pole_is_the_ecliptic_pole_in_equatorial_axes(name):
    """
    Given a shipped spacecraft ephemeris,
    When its orbital angular momentum direction is computed from the
      tabulated barycentric positions,
    Then it is the ecliptic pole expressed in ICRS equatorial axes (IAU
      2006 obliquity), not +Z -- i.e. the file is equatorial, not ecliptic.
    """
    _, xyz = _xyz(name)
    pole = np.cross(xyz[:-1], xyz[1:]).sum(axis=0)
    pole /= np.linalg.norm(pole)

    angle = np.degrees(np.arccos(np.clip(pole @ ECLIPTIC_POLE_EQ, -1, 1)))
    assert angle < POLE_TOLERANCE_DEG.get(name, 0.1), angle
    # The ecliptic-frame mistake would sit at the equatorial pole.
    assert np.degrees(np.arccos(pole[2])) > 20.0


@pytest.mark.parametrize("name", _eph_files())
def test_spacecraft_distance_from_earth_is_what_the_spacecraft_is(name):
    """
    Given a shipped spacecraft ephemeris and astropy's barycentric Earth,
    When the spacecraft-Earth distance is computed over the file,
    Then it is the spacecraft's known distance: ~0.01 AU for an L2 orbit,
      a large fraction of an AU for an Earth-trailing one.
    """
    t, xyz = _xyz(name)
    step = max(1, len(t) // 200)
    earth = get_observer_position(t[::step], observer_location="earth")
    dist = np.linalg.norm(xyz[::step] - earth, axis=1)

    lo, hi = SPACECRAFT_EARTH_DISTANCE_AU[name]
    assert dist.min() > lo and dist.max() < hi, (dist.min(), dist.max())


# JPL Horizons geocentric state of Kepler at JD_TDB 2457523.5
# (2016-05-15 00:00 TDB, mid K2 Campaign 9), ICRF equatorial AU, captured
# 2026-10-29 with
#   Horizons(id="-227", location="500@399",
#            epochs=[2457523.5]).vectors(refplane="earth")
# which returns targetname "Kepler (spacecraft) (-227)".  The same query
# with id="-163" returns "WISE (spacecraft) (-163)", the id k2.eph was
# once generated from (review 1.10.14).
K2_EPOCH_JD_TDB = 2457523.5
K2_GEOCENTRIC_AU = np.array(
    [-0.4195137392825806, 0.622120342532367, 0.2745212053444134]
)


def test_k2_ephemeris_is_kepler_at_a_horizons_epoch():
    """
    Given k2.eph and Kepler's geocentric position from JPL Horizons at a
      mid-Campaign-9 epoch (captured above, query recorded),
    When the file is interpolated there and astropy's barycentric Earth is
      subtracted,
    Then the geocentric vector is Horizons' to < 1e-6 AU (150 km) -- the
      file is Kepler/K2 in the equatorial frame, ~0.8 AU from the Earth,
      not WISE (~4.5e-5 AU) and not an ecliptic-frame Kepler.
    """
    sc = get_observer_position(
        np.array([K2_EPOCH_JD_TDB]), observer_location="k2"
    )[0]
    earth = get_observer_position(
        np.array([K2_EPOCH_JD_TDB]), observer_location="earth"
    )[0]
    np.testing.assert_allclose(sc - earth, K2_GEOCENTRIC_AU, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", _eph_files())
def test_ephemeris_interpolation_returns_the_tabulated_states(name):
    """
    Given a shipped ephemeris file,
    When it is interpolated back at its own tabulated times,
    Then the tabulated positions are returned to machine precision (the
      file -> spline -> position round trip moves nothing).
    """
    t, xyz = _xyz(name)
    idx = np.linspace(0, len(t) - 1, 25).astype(int)
    np.testing.assert_allclose(
        interpolate_ephemeris(t[idx], os.path.join(EPHEMERIDES_DIR, name)),
        xyz[idx],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "site", ["CTIO", "-70.8,-30.2,2200", "149.07,-31.27,1165"]
)
def test_ground_site_is_earth_plus_the_astropy_gcrs_site_vector(site):
    """
    Given a ground site (astropy name or a lon,lat,height string),
    When get_observer_position places it,
    Then (site - geocenter) is astropy's GCRS position of that site to
      1 km (the two transformation paths differ by ~0.6 km of diurnal
      aberration-sized rotation, 4e-9 AU -- negligible against any
      parallax), and its length is the site's geocentric radius to 1 m.
    """
    t = np.array([2457000.0, 2457100.3, 2458000.7])
    geo = get_observer_position(t, observer_location="earth")
    topo = get_observer_position(t, observer_location=site)
    vec_km = (topo - geo) * u.au.to(u.km)

    if "," in site:
        lon, lat, h = (float(x) for x in site.split(","))
        loc = EarthLocation.from_geodetic(lon, lat, h)
    else:
        loc = EarthLocation.of_site(site)
    gcrs, _ = loc.get_gcrs_posvel(Time(t, format="jd", scale="tdb"))
    ref_km = gcrs.xyz.to_value(u.km).T

    assert np.max(np.linalg.norm(vec_km - ref_km, axis=1)) < 1.0
    radius_km = np.linalg.norm(
        u.Quantity([loc.x, loc.y, loc.z]).to_value(u.km)
    )
    np.testing.assert_allclose(
        np.linalg.norm(vec_km, axis=1), radius_km, rtol=0, atol=1e-3
    )


# --- the gaia/abs five-parameter track -------------------------------------


def _track(ra, dec, pm_ra, pm_dec, epoch, t, ra_ref, dec_ref):
    """_star_track at parallax 0, on numpy stand-ins for the parameters."""
    from exozippy.components.astrometryinstrument.astrometryinstrument import (
        AstrometryInstrument,
    )

    star = SimpleNamespace(
        ra=SimpleNamespace(value=np.array([ra])),
        dec=SimpleNamespace(value=np.array([dec])),
        pm_ra=SimpleNamespace(value=np.array([pm_ra])),
        pm_dec=SimpleNamespace(value=np.array([pm_dec])),
        parallax=SimpleNamespace(value=np.array([0.0])),
    )
    self = SimpleNamespace(epoch=epoch)
    d = {"ra_ref": ra_ref, "dec_ref": dec_ref}
    zeros = np.zeros_like(t)
    dE, dN = AstrometryInstrument._star_track(
        self, SimpleNamespace(star=star), d, 0, t, zeros, zeros
    )
    ev = lambda x: np.asarray(x.eval() if hasattr(x, "eval") else x)  # noqa: E731
    return ev(dE), ev(dN)


@pytest.mark.parametrize(
    "ra_deg,dec_deg", [(266.80104, -21.38297), (17.5, 62.0), (120.0, 5.0)]
)
def test_five_parameter_track_matches_astropy_space_motion(ra_deg, dec_deg):
    """
    Given a star at the Gaia DR3 reference epoch J2016.0 (JD 2457389.0
      TDB) with a proper motion of tens of mas/yr,
    When _star_track propagates it (parallax off) to epochs 10 years
      either side,
    Then the (East, North) offset from the reference position matches
      astropy's rigorous apply_space_motion to 0.05 mas (the linear model's
      own truncation, (mu t)^2 tan(dec) / 2 ~ 0.02 mas here) -- pinning
      mu_alpha* (not mu_alpha), mas, and the Julian year -- and propagating
      from J2016.0 to an epoch and back from there returns the J2016.0
      position to machine precision (round trip).
    """
    epoch = 2457389.0  # J2016.0, Gaia DR3's reference epoch, in TDB
    assert Time(epoch, format="jd", scale="tdb").jyear == pytest.approx(
        2016.0, abs=1e-12
    )
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    pm_ra, pm_dec = 42.0, -27.0  # mas/yr; pm_ra = mu_alpha cos(dec)
    t = epoch + np.array([-3652.5, -365.25, 0.0, 1000.0, 3652.5])

    dE, dN = _track(ra, dec, pm_ra, pm_dec, epoch, t, ra, dec)

    star = SkyCoord(
        ra=ra * u.rad,
        dec=dec * u.rad,
        pm_ra_cosdec=pm_ra * u.mas / u.yr,
        pm_dec=pm_dec * u.mas / u.yr,
        distance=1.0 * u.kpc,
        radial_velocity=0.0 * u.km / u.s,
        obstime=Time(epoch, format="jd", scale="tdb"),
    )
    moved = star.apply_space_motion(
        new_obstime=Time(t, format="jd", scale="tdb")
    )
    ref = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    exp_e, exp_n = ref.spherical_offsets_to(
        SkyCoord(ra=moved.ra, dec=moved.dec)
    )
    np.testing.assert_allclose(dE, exp_e.to_value(u.mas), rtol=0, atol=0.05)
    np.testing.assert_allclose(dN, exp_n.to_value(u.mas), rtol=0, atol=0.05)

    # Round trip in the code's own terms: re-anchor at an epoch t1 from the
    # track's position there, propagate back to J2016.0.
    t1 = epoch + 2000.0
    e1, n1 = _track(ra, dec, pm_ra, pm_dec, epoch, np.array([t1]), ra, dec)
    ra1 = ra + e1[0] / (np.cos(dec) * 180.0 / np.pi * 3.6e6)
    dec1 = dec + n1[0] / (180.0 / np.pi * 3.6e6)
    e0, n0 = _track(ra1, dec1, pm_ra, pm_dec, t1, np.array([epoch]), ra, dec)
    # 1e-6 mas: one ulp of a ~5 rad angle is 2e-7 mas.
    assert abs(e0[0]) < 1e-6 and abs(n0[0]) < 1e-6
