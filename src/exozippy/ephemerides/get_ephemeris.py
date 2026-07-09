import logging
import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import argparse

logger = logging.getLogger(__name__)


def get_ephemeris(target_id, start_time, stop_time, step='1h', outfile=None):
    """
    Fetches Barycentric XYZ coordinates from JPL Horizons and saves to a file.

    Parameters:
    -----------
    target_id : str
        JPL ID (e.g., '-79' for Spitzer, '-163' for K2, 'Swift')
    start_time : str
        Start date (e.g., '2014-05-01')
    stop_time : str
        Stop date (e.g., '2014-09-01')
    step : str
        Time step (e.g., '1h', '1d', '10m')
    outfile : str
        Filename to save the results. Defaults to target_ephemeris.txt
    """

    logger.info(f"Querying JPL Horizons for {target_id}...")

    # '500@0' specifies the Solar System Barycenter as the origin
    obj = Horizons(id=target_id,
                   location='500@0',
                   epochs={'start': start_time, 'stop': stop_time, 'step': step})

    # Retrieve vectors (units are AU and AU/day by default).
    # refplane='earth' returns ICRF/J2000 *equatorial* coordinates, matching
    # astropy's get_body_barycentric used for observer_location='earth'.
    # The astroquery default (refplane='ecliptic') silently returns ecliptic
    # coordinates, rotated 23.4 deg from what the projection math assumes.
    vecs = obj.vectors(refplane='earth')

    # Convert dates to MJD for easy interpolation later
    # vecs['datetime_jd'] is the Julian Date
    t_tdb = Time(vecs['datetime_jd'], format='jd', scale='tdb')
    bjd_tdb = t_tdb.value

    # Extract coordinates
    x, y, z = vecs['x'], vecs['y'], vecs['z']

    # Combine into a table
    data = np.column_stack((bjd_tdb, x, y, z))

    if outfile is None:
        outfile = f"ephemeris_{target_id.replace('-', 'm')}.txt"

    header = (f"Target: {target_id} (Center: 500@0, frame: ICRF/J2000 equatorial)\n"
              f"Source: JPL Horizons\nColumns: BJD_TDB, X [AU], Y [AU], Z [AU]")

    np.savetxt(outfile, data, header=header, fmt=['%.8f', '%.12f', '%.12f', '%.12f'])
    logger.info(f"Ephemeris saved to {outfile}")
    return outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Barycentric Ephemeris from JPL")
    parser.add_argument("--id", type=str, default="-79", help="Target ID (default: -79 for Spitzer)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--stop", type=str, required=True, help="Stop date (YYYY-MM-DD)")
    parser.add_argument("--step", type=str, default="1h", help="Step size (default: 1h)")
    parser.add_argument("--out", type=str, help="Output filename")

    args = parser.parse_args()
    get_ephemeris(args.id, args.start, args.stop, args.step, args.out)

