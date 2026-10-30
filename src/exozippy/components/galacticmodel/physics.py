"""Numpy side of the galactic kinematic model.

``GalacticModel.build_likelihood`` evaluates the kinematic prior symbolically,
but the *mean* of that prior is also wanted in plain numpy, early, as a start
value: the relaxation engine has to seed ``star.pm_ra``/``star.pm_dec`` before
any PyMC node exists (stage 3), and with nothing better to go on it used to
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

The model frame (ONE frame for positions AND velocities; review 1.7.1):
astropy's ``Galactocentric`` frame as configured in ``GALACTOCENTRIC_FRAME``
(origin at Sgr A*, R0/z_sun/v_sun from constants.py), with the x axis
MIRRORED so that x points from the Galactic center toward the Sun.  That is
genulens's convention (Koshimoto et al. 2021, ``Dlb2xyz`` with
``CenSgrA = 1``): Sun at x ~ +R0, y toward l = 90 deg, z toward the NGP, and
the Sun's velocity (vx, vy, vz)sun = (-10, 243, 7) km/s.  The density model
(bar angle, disk profiles) is written in that convention, so the mirror
keeps it unchanged, and the Sun at +x moving +y makes the planar angular
momentum ``x v_y - y v_x`` POSITIVE for co-rotation -- the ``v_phi`` sign
``polar_velocity`` uses.  Positions and velocities are BOTH read off
astropy and mirrored by ``MODEL_FRAME_MIRROR``; until 2026-10 the positions
were built by hand in the mirrored convention while the velocities were
astropy's unmirrored ones, so ``v_x`` entered ``v_r``/``v_phi`` with the
wrong sign (an exactly circular orbit at 4 kpc toward the bulge got
v_r = +47 km/s).
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianDifferential, Galactocentric, SkyCoord

from ...constants import (
    BULGE_BAR_ANGLE,
    BULGE_DENSITY_C1,
    BULGE_DENSITY_C2,
    BULGE_DENSITY_X_0,
    BULGE_DENSITY_Y_0,
    BULGE_DENSITY_Z_0,
    BULGE_RC,
    BULGE_RC_WIDTH,
    BULGE_ROTATION_ANGULAR_VELOCITY,
    BULGE_X_B_ZX,
    BULGE_X_C1,
    BULGE_X_C2,
    BULGE_X_FRACTION,
    BULGE_X_RC,
    BULGE_X_X_0,
    BULGE_X_Y_0,
    BULGE_X_Z_0,
    DISK_ROTATION_VELOCITY,
    K_VEL_CONVERSION,
    SUN_GALCEN_V,
    SUN_GC_DISTANCE,
    SUN_Z_OFFSET,
    THICK_DISK_ROTATION_VELOCITY,
)

# The one galactocentric frame for positions AND velocities (mirrored in x
# by MODEL_FRAME_MIRROR below), with the density model's R0/z_sun
# (genulens: R0 = 8160 pc, zsun = 25 pc, vsun = (10, 243, 7) km/s
# toward-GC/rotation/up).  Astropy's default Galactocentric (R0 = 8.122 kpc)
# would be a different frame from the densities'.
#
# Defined here rather than in galacticmodel.py so this module stays the numpy
# layer that the pytensor layer imports, not the other way round.
GALACTOCENTRIC_FRAME = Galactocentric(
    galcen_distance=SUN_GC_DISTANCE * u.kpc,
    z_sun=SUN_Z_OFFSET * u.kpc,
    galcen_v_sun=CartesianDifferential(list(SUN_GALCEN_V) * (u.km / u.s)),
)

#: astropy Galactocentric -> model frame: mirror x (see the module
#: docstring).  Applied to positions and velocities alike; it is its own
#: inverse.
MODEL_FRAME_MIRROR = np.diag([-1.0, 1.0, 1.0])

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

    The position is affine in distance too, ``xyz = sun_xyz + d * los_hat``,
    so the same astropy transform supplies it exactly.  All four arrays are
    in the model frame (astropy's, x mirrored; see the module docstring), so
    positions and velocities can never again come from different frames.

    Returns ``(M_rot, v0, sun_xyz, los_hat)``: a (3, 3) array (orthogonal,
    det -1 because of the mirror), the Sun's galactocentric velocity (3,),
    the Sun's galactocentric position (3,) kpc and the unit sight-line
    vector (3,).  ``line_of_sight_basis(...)[2:]`` is the ``los`` tuple
    ``galactic_xyz`` takes.
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
        return MODEL_FRAME_MIRROR @ np.array(
            [gal.v_x.value, gal.v_y.value, gal.v_z.value]
        )

    def _pos(gal):
        return MODEL_FRAME_MIRROR @ np.array(
            [
                gal.x.to_value(u.kpc),
                gal.y.to_value(u.kpc),
                gal.z.to_value(u.kpc),
            ]
        )

    at_rest = _basis(0, 0, 0)
    v0 = _vec(at_rest)
    m_rot = np.column_stack(
        [
            _vec(_basis(pm_1, 0, 0)) - v0,
            _vec(_basis(0, pm_1, 0)) - v0,
            _vec(_basis(0, 0, 1)) - v0,
        ]
    )

    sun_xyz = _pos(
        SkyCoord(ra=sc.ra, dec=sc.dec, distance=0.0 * u.kpc).transform_to(
            GALACTOCENTRIC_FRAME
        )
    )
    los_hat = (_pos(at_rest) - sun_xyz) / d
    return m_rot, v0, sun_xyz, los_hat


def hinge(t, xp=np):
    """Smooth max(t, 0) (C-infinity), transition width ~50 pc in kpc units --
    keeps the density plateaus/cutoffs (and the thin-disk metallicity
    gradient's ends) differentiable.  Pure arithmetic: ``xp`` is np or pt."""
    return 0.5 * (t + xp.sqrt(t * t + 0.0025))


def _smooth_abs(u, xp):
    # |u| with a 1e-6 (in units of the scale length) rounding at 0, so the
    # super-ellipsoid powers below have a finite gradient on every axis.
    return xp.sqrt(u * u + 1e-12)


def _super_ellipsoid(xn, yn, zn, c1, c2, xp):
    """K21's boxy radius ((|x|^c1 + |y|^c1)^(c2/c1) + |z|^c2)^(1/c2) of
    normalized coordinates (genulens calc_rhoB)."""
    a = _smooth_abs(xn, xp) ** c1 + _smooth_abs(yn, xp) ** c1
    return (a ** (c2 / c1) + _smooth_abs(zn, xp) ** c2) ** (1.0 / c2)


def bar_log_profile(x, y, z, r, xp=np):
    """log of the bar's density relative to its central value: Koshimoto,
    Baba & Bennett 2021's E+E_X bar (genulens's default model 5 with the
    X-shape, ``calc_rhoB``), review 1.7.3.

    Main bar: exp(-r_s), r_s the boxy super-ellipsoid radius with axes
    (x0, y0, z0) and exponents (C1, C2), in the bar frame rotated by
    BULGE_BAR_ANGLE from the Sun-GC line; X-shape: two exp(-r_s) lobes
    sheared by +/- b_zX z in x_bar, amplitude f_X.  Each is multiplied by
    the Gaussian cylindrical cutoff beyond its own R_c (width 0.5 kpc,
    genulens's srob) -- written with the smooth ``hinge``, so it is
    C-infinity where genulens's ``if (R >= Rc)`` is C1.  The sum is formed
    with logaddexp (three finite terms, no selection).

    ``x, y, z`` model-frame kpc, ``r`` the smoothed cylindrical radius.
    Pure arithmetic: ``xp`` is np or pt.
    """
    cos_bar = np.cos(BULGE_BAR_ANGLE)
    sin_bar = np.sin(BULGE_BAR_ANGLE)
    xb = x * cos_bar + y * sin_bar
    yb = -x * sin_bar + y * cos_bar
    main = (
        -_super_ellipsoid(
            xb / BULGE_DENSITY_X_0,
            yb / BULGE_DENSITY_Y_0,
            z / BULGE_DENSITY_Z_0,
            BULGE_DENSITY_C1,
            BULGE_DENSITY_C2,
            xp,
        )
        - 0.5 * (hinge(r - BULGE_RC, xp) / BULGE_RC_WIDTH) ** 2
    )
    x_cut = (
        np.log(BULGE_X_FRACTION)
        - 0.5 * (hinge(r - BULGE_X_RC, xp) / BULGE_RC_WIDTH) ** 2
    )
    lobes = [
        x_cut
        - _super_ellipsoid(
            (xb + sign * BULGE_X_B_ZX * z) / BULGE_X_X_0,
            yb / BULGE_X_Y_0,
            z / BULGE_X_Z_0,
            BULGE_X_C1,
            BULGE_X_C2,
            xp,
        )
        for sign in (-1.0, 1.0)
    ]
    return xp.logaddexp(main, xp.logaddexp(lobes[0], lobes[1]))


def galactic_xyz(dist_kpc, sun_xyz, los_hat):
    """Galactocentric cartesian position (kpc) in the model frame.

    ``sun_xyz + dist_kpc * los_hat`` with both vectors from
    ``line_of_sight_basis``: astropy's Galactocentric position, x mirrored
    (Sun at x ~ +R0, z = +z_sun at d = 0, origin at Sgr A*) -- genulens's
    ``Dlb2xyz`` orientation with ``CenSgrA = 1``.  Pure arithmetic, so the
    same function serves numpy (the hints, the normalizer) and pytensor
    tensors (the likelihood).
    """
    x = sun_xyz[0] + dist_kpc * los_hat[0]
    y = sun_xyz[1] + dist_kpc * los_hat[1]
    z = sun_xyz[2] + dist_kpc * los_hat[2]
    return x, y, z


def polar_velocity(x, y, r, v_x, v_y):
    """Planar ``(v_r, v_phi)`` km/s of a model-frame velocity at ``(x, y)``.

    ``r`` is the cylindrical radius the caller uses (the likelihood passes
    its smoothed ``sqrt(x^2 + y^2 + 1e-6)``).  In the model frame (x
    mirrored; module docstring) ``v_phi`` is positive for co-rotation.  Pure
    arithmetic: numpy and pytensor alike.
    """
    cos_phi = x / r
    sin_phi = y / r
    v_r = v_y * sin_phi + v_x * cos_phi
    v_phi = v_y * cos_phi - v_x * sin_phi
    return v_r, v_phi


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

    m_rot, v0, sun_xyz, los_hat = line_of_sight_basis(ra_rad, dec_rad)
    x, y, _z = galactic_xyz(dist_kpc, sun_xyz, los_hat)
    r = float(np.hypot(x, y))

    v_r, v_phi, v_z = mean_polar_velocity(r, population)
    # Inverse of polar_velocity: [v_r, v_phi] = R(-phi) [v_x, v_y].
    cos_phi, sin_phi = x / r, y / r
    v_x = v_r * cos_phi - v_phi * sin_phi
    v_y = v_r * sin_phi + v_phi * cos_phi

    v_alpha, v_delta, v_rad = np.linalg.solve(
        m_rot, np.array([v_x, v_y, v_z], dtype=float) - v0
    )
    scale = K_VEL_CONVERSION * dist_kpc
    return v_alpha / scale, v_delta / scale, v_rad
