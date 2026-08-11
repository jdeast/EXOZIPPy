"""
Shared observer-ephemeris utilities.

Computes barycentric (ICRS/J2000 equatorial) observer positions in AU for
any component that needs them (microlensing parallax, astrometry parallax
factors, ...).  Supports major solar-system bodies, topocentric ground
sites, and interpolated spacecraft ephemeris files (see ephemerides/ and
ephemerides/get_ephemeris.py to generate new ones from JPL Horizons).
"""

import contextlib
import logging
import os
import warnings

import numpy as np
from astropy.coordinates import (
    ICRS,
    EarthLocation,
    get_body_barycentric,
    solar_system_ephemeris,
)
from astropy.coordinates.errors import UnknownSiteException
from astropy.time import Time
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

# Package-internal directory of spacecraft ephemeris files (*.eph)
EPHEMERIDES_DIR = os.path.join(os.path.dirname(__file__), "ephemerides")

# Same vocabulary and same default as the instrument time_scale/time_frame
# machinery's `time_ephemeris:` key: 'builtin' (erfa, no download) by
# default, 'jpl'/'de440'/... opt-in.  'builtin' sits 3.5e-8 AU (5 km) from
# 'jpl' on the Earth's barycentric position, i.e. 0.08% of an Earth radius --
# far below anything a parallax model can resolve.
DEFAULT_EPHEMERIS = "builtin"


@contextlib.contextmanager
def use_ephemeris(name=DEFAULT_EPHEMERIS):
    """
    Select astropy's solar-system ephemeris for the duration of a block.

    ``solar_system_ephemeris`` is *global* process state: setting it without
    restoring it changes every later astropy calculation in the interpreter,
    including ones made by unrelated components.  Astropy's ``set()`` returns
    a context manager for exactly this reason; this wrapper adds the fallback.

    Anything other than 'builtin' needs a JPL kernel (a download, plus the
    jplephem package).  If it cannot be loaded we warn and fall back to
    'builtin' rather than killing the fit over a network hiccup.
    """
    try:
        ctx = solar_system_ephemeris.set(name)
    except Exception as exc:
        if name == DEFAULT_EPHEMERIS:
            raise
        warnings.warn(
            f"could not load solar-system ephemeris {name!r} ({exc}); "
            f"falling back to {DEFAULT_EPHEMERIS!r} "
            f"(~3.5e-8 AU on the Earth's barycentric position)"
        )
        ctx = solar_system_ephemeris.set(DEFAULT_EPHEMERIS)
    with ctx:
        yield


def _parse_geodetic(spec):
    """
    ``EarthLocation`` from a 'lon,lat[,height]' string.

    The order is astropy's ``EarthLocation.from_geodetic(lon, lat, height)``
    order -- East longitude in degrees, latitude in degrees, height above the
    WGS84 ellipsoid in metres -- and it matches the instrument components'
    ``time_location: [lon_deg, lat_deg(, height_m)]`` key.  Bad input raises
    here, naming the order, instead of falling through to the "location not
    recognized, go generate an ephemeris" error, which sends the user looking
    in entirely the wrong place.
    """
    fields = [f.strip() for f in spec.split(",")]
    order = "'lon_deg,lat_deg' or 'lon_deg,lat_deg,height_m' (lon FIRST)"
    if len(fields) not in (2, 3):
        raise ValueError(
            f"observer_location {spec!r} contains a comma, so it is read as "
            f"a geodetic ground site, but it has {len(fields)} fields; "
            f"expected {order}."
        )
    try:
        values = [float(f) for f in fields]
    except ValueError as exc:
        raise ValueError(
            f"observer_location {spec!r} contains a comma, so it is read as "
            f"a geodetic ground site, but its fields are not all numbers; "
            f"expected {order}."
        ) from exc

    lon, lat = values[0], values[1]
    height = values[2] if len(values) == 3 else 0.0
    try:
        return EarthLocation.from_geodetic(lon, lat, height)
    except Exception as exc:
        raise ValueError(
            f"observer_location {spec!r} is not a valid geodetic ground "
            f"site ({exc}); expected {order}."
        ) from exc


def get_observer_position(
    time, observer_location="earth", ephemeris=DEFAULT_EPHEMERIS
):
    """
    High-precision observer position dispatcher.

    Parameters:
    -----------
    time : float or ndarray
        Time(s) at which to calculate coordinates (BJD_TDB).
    observer_location : str
        One of:
          - Major bodies ('earth', 'moon')
          - Topocentric ground sites: an astropy site name (e.g. 'CTIO',
            'Siding Spring'), or a geodetic 'lon,lat' / 'lon,lat,height'
            string.  The order is **lon first**, matching astropy's
            ``EarthLocation.from_geodetic(lon, lat, height)`` and the
            instruments' ``time_location: [lon_deg, lat_deg(, height_m)]``:
            East longitude in degrees, latitude in degrees, height above the
            WGS84 ellipsoid in metres (default 0).
          - Spacecraft ephemeris files ('gaia', 'spitzer', or a path to a
            file generated by ephemerides/get_ephemeris.py)
    ephemeris : str
        astropy solar-system ephemeris used for the Earth's barycentric
        position ('builtin', 'jpl', 'de440', ...).  Same vocabulary and same
        'builtin' default as the instruments' ``time_ephemeris:`` key.  It is
        applied in a scoped context manager, so the process-global
        ``solar_system_ephemeris`` state is left exactly as it was found.

    Returns:
    --------
    xyz_au : ndarray
        (N, 3) array of barycentric X, Y, Z coordinates in AU (ICRS/J2000
        equatorial frame).
    """
    with use_ephemeris(ephemeris):
        t_obj = Time(time, format="jd", scale="tdb")

        # 1. Major bodies.  Checked first: unambiguous, and it does not
        #    depend on 'earth'/'moon' happening to be absent from astropy's
        #    site registry.
        if observer_location in ["earth", "moon"]:
            return (
                get_body_barycentric(observer_location, t_obj)
                .xyz.to("au")
                .value.T
            )

        # 2. Terrestrial / topocentric: 'lon,lat[,height]' or a site name.
        #    Only an unknown *site name* falls through to the ephemeris-file
        #    search below; every other failure is raised, with context.  A
        #    blanket `except Exception: pass` here used to hide genuine
        #    failures (bad geodetic values, a broken site registry, an
        #    ITRS->ICRS/IERS error) behind the unrelated "location not
        #    recognized" message at the bottom of this function.
        loc = None
        if "," in observer_location:
            loc = _parse_geodetic(observer_location)
        else:
            try:
                loc = EarthLocation.of_site(observer_location)
            except UnknownSiteException:
                # Not a ground site; try the ephemeris files.
                loc = None

        if loc is not None:
            # Topocentric position: barycentric Earth + geocentric offset,
            # i.e. Earth's orbit plus the observer's specific spot on the
            # globe.
            return (
                loc.get_itrs(t_obj)
                .transform_to(ICRS())
                .cartesian.xyz.to("au")
                .value.T
            )

        # 3. Handle Satellite Ephemeris Files with Search Paths
        search_paths = [
            observer_location,  # absolute/relative path
            os.path.join(
                EPHEMERIDES_DIR, observer_location
            ),  # Package internal
            os.path.join(
                EPHEMERIDES_DIR, observer_location + ".eph"
            ),  # Package internal
        ]

        for path in search_paths:
            if os.path.exists(path):
                return interpolate_ephemeris(time, path)

        raise ValueError(
            f"observer location not recognized: {observer_location}; "
            f"expected 'earth'/'moon', an astropy site name, a geodetic "
            f"'lon_deg,lat_deg[,height_m]' string, or an ephemeris file -- "
            f"see $EXOZIPPY_PATH/src/exozippy/ephemerides/get_ephemeris.py "
            f"to generate one"
        )


def interpolate_ephemeris(time, ephemeris_file):
    """
    Interpolates a Barycentric ephemeris file.

    Parameters:
    -----------
    time : float or ndarray
        The time(s) at which to calculate coordinates (BJD_TDB).
    ephemeris_file : str
        Path to the file generated by ephemerides/get_ephemeris.py
        (Format: BJD_TDB, X, Y, Z).  The XYZ columns must be barycentric
        ICRS/J2000 **equatorial** -- that is what get_ephemeris.py's
        refplane='earth' produces, and astroquery's default
        (refplane='ecliptic') is rotated 23.4 deg from what every consumer
        of this function assumes.

    Returns:
    --------
    xyz_au : ndarray
        (N, 3) array of barycentric X, Y, Z coordinates in AU (ICRS/J2000
        equatorial frame).
    """
    # Load data, skipping the header lines
    # data[:, 0] = BJD_TDB, [:, 1:4] = X, Y, Z
    data = np.loadtxt(ephemeris_file)

    t_grid = data[:, 0]
    xyz_grid = data[:, 1:4]

    # Check if we are extrapolating (which is dangerous)
    t_min, t_max = np.min(t_grid), np.max(t_grid)
    if np.any(time < t_min) or np.any(time > t_max):
        warnings.warn(
            f"Extrapolating outside ephemeris range! "
            f"Grid: {t_min:.2f}-{t_max:.2f}, Requested: {np.min(time):.2f}"
        )

    # Create the spline object
    # bc_type='not-a-knot' is standard for smooth orbital curves
    cs = CubicSpline(t_grid, xyz_grid, axis=0, bc_type="not-a-knot")

    return cs(time)
