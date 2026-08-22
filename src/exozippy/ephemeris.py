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
import zlib
from functools import lru_cache

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


@lru_cache(maxsize=32)
def _ephemeris_spline(ephemeris_file, _stamp):
    """Parse one .eph file and fit its CubicSpline, once (review 6.10.1).

    Returns `(spline, t_min, t_max)`.  Keyed on the path AND a `_stamp` of
    `(st_mtime_ns, st_size, crc32)` taken by the caller, so an ephemeris
    regenerated under the same name is re-read rather than served stale --
    a plain path key would outlive a `get_ephemeris.py` re-run inside one
    process (the GUI, a notebook, a test that rewrites its tmp file).

    The crc32 is in the stamp because (mtime_ns, size) alone is not a
    content fingerprint: a same-size rewrite within one filesystem
    timestamp tick collides, and that is a REAL case, not a pathology --
    a GUI-triggered get_ephemeris.py re-run finishes well inside a
    second, and coarse-mtime filesystems (the cluster scratch the test
    suite runs on) truncate the tick to that scale.  The suite caught it
    as a load-dependent flake in test_a_regenerated_ephemeris_is_re_read
    (ezsuite 15363602: stale spline served exactly when two writes landed
    in one tick).  The cost is reading the file bytes on every call; the
    call sites are per-instrument load and plotting grids, not the
    likelihood, and the parse+fit (the expensive part) stays cached.

    What is NOT cached is the extrapolation check: it depends on the
    requested epochs, not on the file, and it has to warn every time.
    """
    data = np.loadtxt(ephemeris_file)
    t_grid = data[:, 0]
    xyz_grid = data[:, 1:4]
    # bc_type='not-a-knot' is standard for smooth orbital curves
    spline = CubicSpline(t_grid, xyz_grid, axis=0, bc_type="not-a-knot")
    return spline, float(np.min(t_grid)), float(np.max(t_grid))


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
    # Parse + fit once per (file, mtime, size); see _ephemeris_spline.  This
    # is called per instrument at load AND from plot_sky's 4000-point grids
    # and the mulens plotters, so a multi-draw plotting pass used to re-parse
    # the same .eph dozens of times.
    st = os.stat(ephemeris_file)
    with open(ephemeris_file, "rb") as fh:
        crc = zlib.crc32(fh.read())
    cs, t_min, t_max = _ephemeris_spline(
        ephemeris_file, (st.st_mtime_ns, st.st_size, crc)
    )

    # Check if we are extrapolating (which is dangerous)
    below = np.asarray(time) < t_min
    above = np.asarray(time) > t_max
    if np.any(below) or np.any(above):
        # The old message printed `Requested: {np.min(time)}` whichever end
        # was out of range, so an over-range epoch was reported with an
        # in-range number.  It also went through warnings.warn, which is
        # deduplicated and (nothing calls logging.captureWarnings) never
        # reaches <prefix>.log -- the file users actually keep.  A cubic
        # not-a-knot spline diverges fast outside its grid and this feeds
        # microlensing pi_E and astrometric parallax factors, so the epochs
        # and the overshoot both have to be nameable.
        n_bad = int(np.count_nonzero(below) + np.count_nonzero(above))
        logger.warning(
            f"Extrapolating outside the ephemeris range in "
            f"{ephemeris_file}: grid covers {t_min:.4f}-{t_max:.4f}, but "
            f"{n_bad} epoch(s) fall outside it "
            f"(min requested {np.min(time):.4f}, max {np.max(time):.4f}). "
            f"The cubic spline is being extrapolated; regenerate the "
            f"ephemeris to cover these epochs (see "
            f"ephemerides/get_ephemeris.py)."
        )

    return cs(time)
