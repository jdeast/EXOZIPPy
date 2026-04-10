import numpy as np
import erfa
from astropy.time import Time
from astropy.coordinates import get_body_barycentric
from astropy.coordinates.builtin_frames.utils import get_jd12


def get_deltas(time, t_0_par, skycoord):
    """
    Calculate normalized shifts due to parallax in North and East directions.
    """
    velocity = _velocity_of_earth(t_0_par)
    (projected_N, projected_E) = _calculate_projected(skycoord)

    # Get Earth's position at all observation times and the reference time
    pos = get_body_barycentric(body='earth', time=Time(time, format='jd', scale='tdb'))
    pos_ref = get_body_barycentric(body='earth', time=Time(t_0_par, format='jd', scale='tdb'))

    # Delta-S is the change in Earth's barycentric position
    delta_s = (pos_ref.xyz.T - pos.xyz.T).to('au').value

    # Account for the constant velocity component relative to reference epoch
    delta_s += np.outer(time - t_0_par, velocity)

    return {
        'N': np.dot(delta_s, projected_N),
        'E': np.dot(delta_s, projected_E)
    }


def _velocity_of_earth(full_BJD):
    """Returns Earth's velocity vector in AU/day."""
    t = Time(full_BJD, format='jd', scale='tdb')
    jd1, jd2 = get_jd12(t, 'tdb')
    # erfa returns [position, velocity]
    _, earth_pv_bary = erfa.epv00(jd1, jd2)
    return np.asarray(earth_pv_bary[1])


def _calculate_projected(skycoord):
    """Calculate North and East unit vectors projected on the sky plane."""
    direction = np.array(skycoord.cartesian.xyz.value)
    north = np.array([0., 0., 1.])

    east_vec = np.cross(north, direction)
    projected_E = east_vec / np.linalg.norm(east_vec)
    projected_N = np.cross(direction, projected_E)

    return projected_N, projected_E