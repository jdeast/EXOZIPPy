"""
The one definition of the sky-plane frame and of projections onto it.

THE CONVENTION
--------------
EXOZIPPy uses the century-old LEFT-HANDED astrometric frame throughout::

    +X = North      (plotted up)
    +Y = East       (plotted left)
    +Z = distance   (growing away from the observer)

It is left-handed as a set of physical directions (``X cross Y = -Z``), and
that is not an accident to be tidied away.  It is what simultaneously
satisfies the textbook definitions of the Keplerian orbital elements while
preserving the standard application of the Euler angles
``Rz(bigomega) Rx(inc) Rz(omega)``:

* ``bigomega`` comes out as the position angle of the ascending node
  measured EAST OF NORTH (at ``omega + f = 0`` the body sits at
  ``PA = bigomega`` exactly),
* ``omega`` is the argument of periastron of the orbit it names (in this
  codebase, of the PRIMARY's orbit -- ``omega_*``), and
* ``dZ/dt`` is the radial velocity with the right sign (positive =
  receding = redshift), because ``+Z`` is distance.

Adopting a right-handed relabelling breaks one of those three; this
convention has been repeatedly mangled in the exoplanet literature for
exactly that reason.  ``Orbit.get_sky_position`` / ``Orbit.get_radial_velocity``
implement it directly, and ``tests/test_skyframe.py`` pins all three claims
against a first-principles Euler construction.

WHAT LIVES HERE
---------------
Everything that turns a 3-D barycentric observer position (ICRS/J2000
equatorial, AU -- what ``ephemeris.get_observer_position`` returns) into
sky-plane East/North components.  Two quantities are wanted, and they differ
by a sign because they are DIFFERENT QUANTITIES, not different conventions:

* :func:`observer_sky_offset` -- the OBSERVER's own offset projected on the
  sky, in AU.  This is what the microlensing trajectory consumes (Gould 2004
  writes it ``s(t)``; that name is deliberately not used in this codebase,
  where ``lens.s`` is the binary separation in Einstein radii).
* :func:`parallax_factors` -- the apparent displacement OF THE SOURCE per
  unit parallax, consumed as ``+ plx * (P_E, P_N)``.  An observer displaced
  by ``R`` sees a source at distance ``d`` shifted by ``-R/d``, so this is
  exactly ``-observer_sky_offset`` and is defined as such below.

Both project onto the same basis, :func:`sky_basis`, and both are
orientation-consistent with ``Orbit.get_sky_position``: rotating from North
toward East is the positive (position-angle-increasing) sense.  Before this
module existed the two dot products were written out by hand at seven call
sites in three spellings, which is the silent-drift class the Op-path
annual-parallax bug came from -- a sign fix applied to one copy and not the
others.  Add call sites; do not re-derive the algebra at them.

The functions are backend-agnostic: pass ``xp=pytensor.tensor`` to build a
symbolic graph instead of a numpy array.  Only ``sin``/``cos`` and basic
arithmetic are used, so any array module with those works.
"""

import numpy as np

__all__ = ["sky_basis", "observer_sky_offset", "parallax_factors"]


def sky_basis(ra, dec):
    """Unit vectors toward East and North at (ra, dec), in ICRS equatorial XYZ.

    ``ra``/``dec`` are in RADIANS (the internal unit; user-facing degrees are
    converted by ``Parameter.__post_init__``).  Returns ``(e_hat, n_hat)``,
    each a numpy 3-vector, defined as the standard spherical tangent basis::

        u_hat = (cos dec cos ra, cos dec sin ra, sin dec)
        e_hat = d(u_hat)/d(ra) / cos dec = (-sin ra, cos ra, 0)
        n_hat = d(u_hat)/d(dec)          = (-sin dec cos ra, -sin dec sin ra, cos dec)

    Equivalently, and this is how MulensModel's ``Coordinates`` builds them,
    ``e_hat = normalize(z_hat x u_hat)`` and ``n_hat = u_hat x e_hat``; the
    two constructions agree to machine precision (pinned in
    ``tests/test_skyframe.py``), so the Op path and the symbolic path share
    one line of sight by construction rather than by coincidence.

    Scalars only -- this returns the basis itself, for callers that need to
    dot something other than an observer position into it (see
    ``Lens._earth_velocity_perp``, which projects the Earth's VELOCITY).
    Callers projecting an observer position want
    :func:`observer_sky_offset` instead.
    """
    sin_ra, cos_ra = np.sin(ra), np.cos(ra)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    e_hat = np.array([-sin_ra, cos_ra, 0.0])
    n_hat = np.array([-cos_ra * sin_dec, -sin_ra * sin_dec, cos_dec])
    return e_hat, n_hat


def observer_sky_offset(xyz, ra, dec, xp=np):
    """Project an observer position onto the sky plane at (ra, dec).

    ``xyz`` is ``(N, 3)`` in ICRS/J2000 equatorial AU -- either absolute
    barycentric positions or, for microlensing, the Skowron+2011 geocentric
    deviations from the linear reference trajectory
    (``MulensInstrument._abs_to_delta``).  ``ra``/``dec`` are in radians.
    ``xp`` is the array module (``numpy``, or ``pytensor.tensor`` for a
    symbolic graph).

    Returns ``(delta_e, delta_n)``, each ``(N,)``, in the same units as
    ``xyz``: the components of the observer's offset along ``e_hat`` and
    ``n_hat``.  This is ``sky_basis``'s two dot products, written out so the
    graph stays elementwise.

    NOTE the sign, and note that it is not a convention choice: this is the
    OBSERVER's displacement, so a source's apparent shift runs the other way
    (see :func:`parallax_factors`).  The microlensing trajectory wants this
    one, because the lens-source separation picks up ``-pi_rel * offset``:

        u(t) = u_0 + (t - t_0)/t_E * pi_E_hat - pi_E * offset(t)

    which is what ``Lens.get_magnification`` implements as
    ``tau -= delta_n*pi_E_N + delta_e*pi_E_E`` and
    ``u += delta_n*pi_E_E - delta_e*pi_E_N``.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    sin_ra, cos_ra = xp.sin(ra), xp.cos(ra)
    sin_dec, cos_dec = xp.sin(dec), xp.cos(dec)
    delta_e = -x * sin_ra + y * cos_ra
    delta_n = -x * cos_ra * sin_dec - y * sin_ra * sin_dec + z * cos_dec
    return delta_e, delta_n


def parallax_factors(xyz, ra, dec, xp=np):
    """Apparent sky displacement of a source, per unit parallax.

    Same inputs as :func:`observer_sky_offset`.  Returns ``(P_E, P_N)``, to
    be consumed as::

        dE = ... + plx * P_E
        dN = ... + plx * P_N

    with ``plx`` and the offsets in matching angular units (mas throughout
    ``astrometryinstrument``).

    An observer at barycentric position ``R`` (AU) sees a source at distance
    ``d`` displaced by ``-(R - (R.u_hat) u_hat)/d``, whose East and North
    components are ``-plx * (R.e_hat)`` and ``-plx * (R.n_hat)``.  So this is
    exactly the negative of the observer's own projected offset, and is
    defined that way rather than as a second hand-written dot product -- the
    two used to be independent copies, which read as an unexplained sign
    disagreement between the astrometry and microlensing components when it
    is only the difference between "where the observer moved" and "where the
    source appears to move".
    """
    delta_e, delta_n = observer_sky_offset(xyz, ra, dec, xp=xp)
    return -delta_e, -delta_n
