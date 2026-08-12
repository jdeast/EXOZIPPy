"""Numpy side of the galactic kinematic model.

``GalacticModel.build_likelihood`` evaluates the kinematic prior symbolically,
but the *mean* of that prior is also wanted in plain numpy, early, as a start
value: the relaxation engine has to seed ``star.pm_ra``/``star.pm_dec`` before
any PyMC node exists (stage 2), and with nothing better to go on it used to
leave both at their defaults.yaml value and let one of them absorb whatever the
``t_E`` constraint implied -- which is how the direction of relative proper
motion ended up arbitrary (see issue #93).

The line-of-sight basis here is the *same* code the likelihood uses; the
likelihood calls ``line_of_sight_basis`` and wraps the result in tensors, so
there is one implementation rather than two that can drift.

Sign/unit conventions, all matching the likelihood:

- ``pm_ra`` means mu_alpha * cos(delta) (astropy's ``pm_ra_cosdec``), in mas/yr.
  The model treats ``star.pm_ra`` as that quantity throughout.
- Velocities are galactocentric, km/s. ``v_phi`` is positive for co-rotation at
  every position.
- Distances in this module are kpc unless the name says ``_pc``.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianDifferential, Galactocentric, SkyCoord

from ...constants import (
    BULGE_ROTATION_ANGULAR_VELOCITY,
    DISK_ROTATION_VELOCITY,
    K_VEL_CONVERSION,
    SUN_GALCEN_V,
    SUN_GC_DISTANCE,
    SUN_Z_OFFSET,
    THICK_DISK_ROTATION_VELOCITY,
)

# One consistent galactocentric frame for the velocity transform, matching
# the density grid's R0/z_sun (genulens: R0 = 8160 pc, zsun = 25 pc,
# vsun = (10, 243, 7) km/s toward-GC/rotation/up).  Astropy's default
# Galactocentric (R0 = 8.122 kpc) would put the velocities in a slightly
# different frame than the densities.
#
# Defined here rather than in galacticmodel.py so this module stays the numpy
# layer that the pytensor layer imports, not the other way round.
GALACTOCENTRIC_FRAME = Galactocentric(
    galcen_distance=SUN_GC_DISTANCE * u.kpc,
    z_sun=SUN_Z_OFFSET * u.kpc,
    galcen_v_sun=CartesianDifferential(list(SUN_GALCEN_V) * (u.km / u.s)),
)

#: Populations whose mean velocity this module can supply.
POPULATIONS = ("thin_disk", "thick_disk", "bulge")


def line_of_sight_basis(ra_rad, dec_rad):
    """Linearize the (pm_ra_cosdec, pm_dec, rv) -> galactocentric velocity map.

    The map is affine, so one offset plus three unit-response columns describe
    it exactly:

        v_gal = v0 + M_rot @ [v_alpha, v_delta, v_rad]

    with ``v_alpha = K_VEL_CONVERSION * pm_ra * d_kpc`` (and likewise for
    ``v_delta``), ``v_rad`` in km/s.  Evaluated at an arbitrary 1 kpc because
    only the *direction* of the response matters -- the distance scaling is
    carried explicitly by the caller.

    Returns ``(M_rot, v0, cosl_cosb, sinl_cosb, sinb)``: a (3, 3) array, a (3,)
    array, and three scalars fixing the Galactic direction.
    """
    sc = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad)
    d = 1.0  # kpc, arbitrary: only the response direction is used
    pm_1 = 1.0 / (K_VEL_CONVERSION * d)  # mas/yr giving 1 km/s at d

    def _basis(pm_ra_cosdec, pm_dec, rv):
        return SkyCoord(
            ra=sc.ra,
            dec=sc.dec,
            distance=d * u.kpc,
            pm_ra_cosdec=pm_ra_cosdec * u.mas / u.yr,
            pm_dec=pm_dec * u.mas / u.yr,
            radial_velocity=rv * u.km / u.s,
        ).transform_to(GALACTOCENTRIC_FRAME)

    def _vec(gal):
        return np.array([gal.v_x.value, gal.v_y.value, gal.v_z.value])

    v0 = _vec(_basis(0, 0, 0))
    m_rot = np.column_stack(
        [
            _vec(_basis(pm_1, 0, 0)) - v0,
            _vec(_basis(0, pm_1, 0)) - v0,
            _vec(_basis(0, 0, 1)) - v0,
        ]
    )

    l_rad = sc.galactic.l.rad
    b_rad = sc.galactic.b.rad
    return (
        m_rot,
        v0,
        np.cos(l_rad) * np.cos(b_rad),
        np.sin(l_rad) * np.cos(b_rad),
        np.sin(b_rad),
    )


def galactic_xyz(dist_kpc, cosl_cosb, sinl_cosb, sinb):
    """Galactocentric cartesian position, genulens Dlb2xyz convention.

    The Sun sits ``SUN_Z_OFFSET`` above the plane, handled as a small rotation
    of z by ``bsun = z_sun / R0`` so a star at d = 0 lands at z = +z_sun.  x and
    y keep the flat convention (Sun at x = +R0, GC at the origin).
    """
    x = SUN_GC_DISTANCE - dist_kpc * cosl_cosb
    y = dist_kpc * sinl_cosb
    bsun = SUN_Z_OFFSET / SUN_GC_DISTANCE
    z = dist_kpc * sinb * np.cos(bsun) + x * np.sin(bsun)
    return x, y, z


def mean_polar_velocity(r_kpc, population):
    """Mean ``(v_r, v_phi, v_z)`` km/s of one kinematic branch.

    Every branch in the likelihood is a Gaussian centred on zero radial and
    vertical motion plus a rotational term, so the mean is exactly the centre
    of that Gaussian -- read straight off ``log_vel_thin`` / ``log_vel_thick`` /
    ``log_vel_bulge``.
    """
    if population == "thin_disk":
        return 0.0, DISK_ROTATION_VELOCITY, 0.0
    if population == "thick_disk":
        return 0.0, THICK_DISK_ROTATION_VELOCITY, 0.0
    if population == "bulge":
        # Cylindrical rotation, Omega * r (Koshimoto & Bennett 2020).
        return 0.0, BULGE_ROTATION_ANGULAR_VELOCITY * r_kpc, 0.0
    raise ValueError(
        f"unknown population {population!r}; expected one of {POPULATIONS}"
    )


def expected_proper_motion(ra_rad, dec_rad, distance_pc, population):
    """Mean proper motion under the kinematic prior, in mas/yr.

    Inverts ``v_gal = v0 + M_rot @ [v_alpha, v_delta, v_rad]`` at the branch's
    mean velocity.  Returns ``(pm_ra_cosdec, pm_dec, rv_kms)``.

    This is the prior's *mean*, not a draw: it is a start value, and the
    dispersions (which are large -- tens of km/s) are what the prior itself
    contributes during sampling.
    """
    dist_kpc = float(distance_pc) / 1e3
    if not dist_kpc > 0:
        raise ValueError(f"distance_pc must be positive; got {distance_pc!r}")

    m_rot, v0, cosl_cosb, sinl_cosb, sinb = line_of_sight_basis(
        ra_rad, dec_rad
    )
    x, y, _z = galactic_xyz(dist_kpc, cosl_cosb, sinl_cosb, sinb)
    r = float(np.hypot(x, y))

    v_r, v_phi, v_z = mean_polar_velocity(r, population)
    # Inverse of get_polar_velocity: [v_r, v_phi] = R(-phi) [v_x, v_y].
    cos_phi, sin_phi = x / r, y / r
    v_x = v_r * cos_phi - v_phi * sin_phi
    v_y = v_r * sin_phi + v_phi * cos_phi

    v_alpha, v_delta, v_rad = np.linalg.solve(
        m_rot, np.array([v_x, v_y, v_z], dtype=float) - v0
    )
    scale = K_VEL_CONVERSION * dist_kpc
    return v_alpha / scale, v_delta / scale, v_rad
