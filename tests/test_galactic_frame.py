"""The galactic model's frame, pinned to an external reference.

The kinematic prior evaluates a velocity Gaussian in galactocentric polar
coordinates, so it needs a position and a velocity IN THE SAME FRAME.  Until
2026-10 it had two (review 1.7.1): positions were built by hand with the Sun
at x = +R0 while the velocities came from astropy's Galactocentric frame,
which puts the Sun at x = -R0, so ``v_x`` entered ``v_r`` and ``v_phi`` with
the wrong sign -- an exactly circular orbit at 4 kpc toward the bulge got
v_r = +47 km/s.  The co-rotation tests in test_galactic_model.py compare
log-density MARGINS through the mixture, whose hotter branches absorbed the
error, so they passed throughout.

Every test here instead checks a computed QUANTITY against astropy (or
closed-form physics) to float precision.  The model frame is astropy's
``Galactocentric`` as configured in ``physics.GALACTOCENTRIC_FRAME``, with x
mirrored (``physics.MODEL_FRAME_MIRROR``) into genulens's orientation, for
positions and velocities alike -- see the physics.py module docstring.
"""

import astropy.units as u
import numpy as np
import pymc as pm
import pytest
from astropy.coordinates import ICRS, Galactic, SkyCoord
from test_galactic_model import _make_gm, _MockParam, _MockSystem

from exozippy.components.galacticmodel import galacticmodel as gm_mod
from exozippy.components.galacticmodel.physics import (
    GALACTOCENTRIC_FRAME,
    MODEL_FRAME_MIRROR,
    expected_proper_motion,
    galactic_xyz,
    line_of_sight_basis,
    mean_polar_velocity,
    polar_velocity,
)
from exozippy.constants import (
    BULGE_ROTATION_ANGULAR_VELOCITY,
    DISK_ROTATION_VELOCITY,
    K_VEL_CONVERSION,
    SUN_GALCEN_V,
    SUN_GC_DISTANCE,
    SUN_Z_OFFSET,
)

KMS = u.km / u.s
MASYR = u.mas / u.yr

# The sight-line grid of review 1.7.1: both sides of the Galactic center,
# the anticenter, in and out of the plane, from the solar neighborhood to
# beyond the Galactic center.
GRID_L = (0.0, 5.0, -5.0, 30.0, -30.0, 90.0, -90.0, 180.0)
GRID_B = (0.0, 3.0, -3.0)
GRID_D = (0.5, 2.0, 4.0, SUN_GC_DISTANCE, 12.0)  # kpc

TOL_KMS = 1e-6


def _astropy_galcen(l_deg, b_deg, d_kpc):
    """astropy's Galactocentric position (kpc, astropy's own axes)."""
    gc = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        distance=d_kpc * u.kpc,
        frame=Galactic(),
    ).transform_to(GALACTOCENTRIC_FRAME)
    return np.array(
        [gc.x.to_value(u.kpc), gc.y.to_value(u.kpc), gc.z.to_value(u.kpc)]
    )


def _observe(xyz_astropy, v_astropy):
    """ICRS (ra, dec [rad], d [kpc], pm_ra_cosdec, pm_dec [mas/yr],
    rv [km/s]) of a star given in astropy's Galactocentric frame."""
    star = SkyCoord(
        x=xyz_astropy[0] * u.kpc,
        y=xyz_astropy[1] * u.kpc,
        z=xyz_astropy[2] * u.kpc,
        v_x=v_astropy[0] * KMS,
        v_y=v_astropy[1] * KMS,
        v_z=v_astropy[2] * KMS,
        frame=GALACTOCENTRIC_FRAME,
    ).transform_to(ICRS())
    return (
        float(star.ra.rad),
        float(star.dec.rad),
        float(star.distance.to_value(u.kpc)),
        float(star.pm_ra_cosdec.to_value(MASYR)),
        float(star.pm_dec.to_value(MASYR)),
        float(star.radial_velocity.to_value(KMS)),
    )


def _model_velocity(basis, d_kpc, pm_ra, pm_dec, rv_kms):
    """The likelihood's (pm_ra, pm_dec, rv) -> galactocentric map, the
    expression get_galactocentric_velocity builds in pytensor."""
    m_rot, v0 = basis[0], basis[1]
    return v0 + m_rot @ np.array(
        [
            K_VEL_CONVERSION * pm_ra * d_kpc,
            K_VEL_CONVERSION * pm_dec * d_kpc,
            rv_kms,
        ]
    )


def _model_polar(ra, dec, d_kpc, pm_ra, pm_dec, rv_kms):
    """(v_r, v_phi, v_z, r) through the model's own path."""
    basis = line_of_sight_basis(ra, dec)
    x, y, _z = galactic_xyz(d_kpc, *basis[2:])
    r = np.hypot(x, y)
    v = _model_velocity(basis, d_kpc, pm_ra, pm_dec, rv_kms)
    v_r, v_phi = polar_velocity(x, y, r, v[0], v[1])
    return v_r, v_phi, v[2], r


def _polar_to_astropy(xyz_astropy, v_rad, v_co, v_z):
    """Velocity (astropy axes) with radial-outward ``v_rad``, ``v_co``
    along the Sun's sense of rotation and vertical ``v_z``."""
    x, y = xyz_astropy[0], xyz_astropy[1]
    big_r = np.hypot(x, y)
    e_r = np.array([x, y, 0.0]) / big_r
    e_c = np.array([y, -x, 0.0]) / big_r
    return v_rad * e_r + v_co * e_c + np.array([0.0, 0.0, v_z])


def _corotating(xyz_astropy, speed):
    """Velocity (astropy axes) of an exactly circular orbit through
    ``xyz_astropy`` at ``speed``, in the SUN's sense of rotation: the Sun
    sits at x < 0 moving +y, so co-rotation is v = speed * (y, -x, 0)/R."""
    x, y = xyz_astropy[0], xyz_astropy[1]
    big_r = np.hypot(x, y)
    return np.array([speed * y / big_r, -speed * x / big_r, 0.0])


# ---------------------------------------------------------------------------
# 1. Circular orbits are exactly circular
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l_deg", GRID_L)
def test_circular_orbit_has_zero_radial_and_vertical_velocity(l_deg):
    """
    Given stars on EXACTLY circular co-rotating orbits, built in astropy at
      every (b, d) of the grid along this l, at the thin disk's mean speed
      and (inside the bar) at the bulge's cylindrical Omega * r,
    When their observables (pm, rv) go through the model's own
      pm/rv -> velocity -> polar path,
    Then v_r = 0, v_z = 0 and v_phi = the orbit's speed, to 1e-6 km/s.

    On master (two frames) v_r was +47 km/s at 4 kpc toward ob140939.
    """
    bad = []
    for b_deg in GRID_B:
        for d_kpc in GRID_D:
            xyz = _astropy_galcen(l_deg, b_deg, d_kpc)
            big_r = np.hypot(xyz[0], xyz[1])
            speeds = [DISK_ROTATION_VELOCITY]
            if big_r < 3.0:
                speeds.append(BULGE_ROTATION_ANGULAR_VELOCITY * big_r)
            for speed in speeds:
                obs = _observe(xyz, _corotating(xyz, speed))
                v_r, v_phi, v_z, r = _model_polar(*obs)
                if not (
                    abs(v_r) < TOL_KMS
                    and abs(v_z) < TOL_KMS
                    and abs(v_phi - speed) < TOL_KMS
                    and abs(r - big_r) < 1e-9
                ):
                    bad.append(
                        f"(l, b, d)=({l_deg}, {b_deg}, {d_kpc}) "
                        f"speed={speed:.3f}: v_r={v_r:.3g} "
                        f"v_phi-speed={v_phi - speed:.3g} v_z={v_z:.3g} "
                        f"r-R={r - big_r:.3g}"
                    )
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("population", ("thin_disk", "thick_disk", "bulge"))
def test_prior_mean_proper_motion_is_a_circular_orbit_in_astropy(population):
    """
    Given the prior's mean proper motion (the start-value hint,
      physics.expected_proper_motion) toward ob140939 and Baade's window,
    When astropy maps that (pm, rv) back to its Galactocentric frame,
    Then the star is on an exactly circular orbit in the Sun's sense at
      the branch's mean speed: R . v = 0, v_z = 0, and the azimuthal speed
      equals mean_polar_velocity's v_phi, to 1e-6 km/s.
    """
    for l_deg, b_deg in ((1.0, -3.9), (6.65, 3.71)):  # Baade, ob140939-ish
        sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame=Galactic())
        ra, dec = sc.icrs.ra.rad, sc.icrs.dec.rad
        for d_kpc in (2.5, 4.0, 7.5):
            pm_ra, pm_dec, rv = expected_proper_motion(
                ra, dec, d_kpc * 1e3, population
            )
            gc = SkyCoord(
                ra=ra * u.rad,
                dec=dec * u.rad,
                distance=d_kpc * u.kpc,
                pm_ra_cosdec=pm_ra * MASYR,
                pm_dec=pm_dec * MASYR,
                radial_velocity=rv * KMS,
            ).transform_to(GALACTOCENTRIC_FRAME)
            x, y = gc.x.to_value(u.kpc), gc.y.to_value(u.kpc)
            vx, vy = gc.v_x.to_value(KMS), gc.v_y.to_value(KMS)
            big_r = np.hypot(x, y)
            v_rad = (x * vx + y * vy) / big_r
            # Sun's sense: the Sun's x * v_y - y * v_x is NEGATIVE in
            # astropy's axes, so the co-rotating azimuthal speed is minus it.
            v_az = -(x * vy - y * vx) / big_r
            _, want, _ = mean_polar_velocity(big_r, population)
            assert abs(v_rad) < TOL_KMS, (l_deg, d_kpc, v_rad)
            assert abs(gc.v_z.to_value(KMS)) < TOL_KMS
            assert v_az == pytest.approx(want, abs=TOL_KMS)


# ---------------------------------------------------------------------------
# 2. Positions are astropy's, mirrored in x
# ---------------------------------------------------------------------------


def test_positions_are_astropy_galactocentric_mirrored_in_x():
    """
    Given every (l, b, d) of the grid,
    When the model builds the galactocentric position,
    Then it is astropy's Galactocentric position with x mirrored, to 1e-12
      kpc -- the documented mapping onto genulens's orientation.
    """
    for l_deg in GRID_L:
        for b_deg in GRID_B:
            sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame=Galactic())
            basis = line_of_sight_basis(sc.icrs.ra.rad, sc.icrs.dec.rad)
            for d_kpc in GRID_D:
                got = np.array(galactic_xyz(d_kpc, *basis[2:]))
                want = MODEL_FRAME_MIRROR @ _astropy_galcen(
                    l_deg, b_deg, d_kpc
                )
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_galactic_center_and_sun_land_where_they_should():
    """
    Given the sight line to (l, b) = (0, 0),
    When a star sits at d = R0, and when it sits at d = 0,
    Then the first is within ~10 pc of the origin (Sgr A*, the origin, is
      0.06 deg off (0, 0)) and the second is the Sun, at
      (sqrt(R0^2 - z_sun^2), 0, z_sun) -- x POSITIVE, genulens's side.
    """
    sc = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame=Galactic())
    basis = line_of_sight_basis(sc.icrs.ra.rad, sc.icrs.dec.rad)
    x, y, z = galactic_xyz(SUN_GC_DISTANCE, *basis[2:])
    assert np.sqrt(x**2 + y**2 + z**2) < 0.012
    sun = np.array(galactic_xyz(0.0, *basis[2:]))
    np.testing.assert_allclose(
        sun,
        [np.sqrt(SUN_GC_DISTANCE**2 - SUN_Z_OFFSET**2), 0.0, SUN_Z_OFFSET],
        rtol=0,
        atol=1e-12,
    )


def test_positions_track_genulens_dlb2xyz():
    """
    Given genulens's own position transform (Koshimoto et al. 2021,
      CoordinateTransformer::distance_l_b_to_xyz with CenSgrA = 1: Sgr A* at
      (l, b) = (-0.056, -0.046) deg, z tilted by z_sun/R0, x NOT tilted),
    When compared with the model's position over the bulge grid,
    Then the two agree to < 15 pc -- the residual is genulens translating
      to Sgr A* where astropy rotates to it, far below every density scale.
    """

    def dlb2xyz(d, l_deg, b_deg):
        bsun = SUN_Z_OFFSET / SUN_GC_DISTANCE
        b, l_rad = np.radians(b_deg), np.radians(l_deg)
        xt = SUN_GC_DISTANCE - d * np.cos(b) * np.cos(l_rad)
        yt = d * np.cos(b) * np.sin(l_rad)
        zt = d * np.sin(b)
        return np.array([xt, yt, zt * np.cos(bsun) + xt * np.sin(bsun)])

    sgra = dlb2xyz(SUN_GC_DISTANCE, -0.056, -0.046)
    for l_deg in (0.0, 5.0, -5.0, 1.0):
        for b_deg in GRID_B + (-3.9,):
            sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame=Galactic())
            basis = line_of_sight_basis(sc.icrs.ra.rad, sc.icrs.dec.rad)
            for d_kpc in GRID_D:
                got = np.array(galactic_xyz(d_kpc, *basis[2:]))
                want = dlb2xyz(d_kpc, l_deg, b_deg) - sgra
                assert np.max(np.abs(got - want)) < 0.015, (l_deg, b_deg)


# ---------------------------------------------------------------------------
# 3. The Sun
# ---------------------------------------------------------------------------


def test_a_star_at_the_sun_at_rest_has_the_suns_velocity():
    """
    Given a star at d = 0 with heliocentric pm = rv = 0, on several sight
      lines,
    When its galactocentric velocity is computed,
    Then it is the Sun's, (vx, vy, vz)sun = (-10, 243, 7) km/s in the model
      frame (genulens's printed (vx, vy, vz)sun exactly), its v_phi is the
      Sun's azimuthal speed 243 km/s and its v_r the Sun's -10 km/s
      (moving toward the Galactic center).
    """
    want = MODEL_FRAME_MIRROR @ np.asarray(SUN_GALCEN_V, dtype=float)
    np.testing.assert_allclose(want, [-10.0, 243.0, 7.0], rtol=0, atol=0)
    for l_deg, b_deg in ((0, 0), (1.0, -3.9), (90, 3), (180, -3), (-30, 0)):
        sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame=Galactic())
        ra, dec = sc.icrs.ra.rad, sc.icrs.dec.rad
        v_r, v_phi, v_z, _ = _model_polar(ra, dec, 0.0, 0.0, 0.0, 0.0)
        assert v_r == pytest.approx(want[0], abs=1e-9)
        assert v_phi == pytest.approx(want[1], abs=1e-9)
        assert v_z == pytest.approx(want[2], abs=1e-9)


# ---------------------------------------------------------------------------
# 4. The velocity map is astropy's, and inverts exactly
# ---------------------------------------------------------------------------


def test_velocity_map_is_astropy_and_round_trips():
    """
    Given random (d, pm_ra, pm_dec, rv) on several sight lines,
    When the model maps them to a galactocentric velocity,
    Then the result is astropy's full transform with x mirrored (1e-9
      km/s), and the inverse the start-value hints use (np.linalg.solve on
      the same basis) returns the input to machine precision.
    """
    rng = np.random.default_rng(171)
    for l_deg, b_deg in ((1.0, -3.9), (6.65, 3.71), (-60, 10), (180, 0)):
        sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame=Galactic())
        ra, dec = sc.icrs.ra.rad, sc.icrs.dec.rad
        basis = line_of_sight_basis(ra, dec)
        for _ in range(5):
            d = rng.uniform(0.3, 15.0)
            pm_ra, pm_dec = rng.normal(0, 8, 2)
            rv = rng.normal(0, 100)
            v = _model_velocity(basis, d, pm_ra, pm_dec, rv)
            gc = SkyCoord(
                ra=ra * u.rad,
                dec=dec * u.rad,
                distance=d * u.kpc,
                pm_ra_cosdec=pm_ra * MASYR,
                pm_dec=pm_dec * MASYR,
                radial_velocity=rv * KMS,
            ).transform_to(GALACTOCENTRIC_FRAME)
            want = MODEL_FRAME_MIRROR @ np.array(
                [
                    gc.v_x.to_value(KMS),
                    gc.v_y.to_value(KMS),
                    gc.v_z.to_value(KMS),
                ]
            )
            np.testing.assert_allclose(v, want, rtol=0, atol=1e-9)

            back = np.linalg.solve(basis[0], v - basis[1])
            scale = K_VEL_CONVERSION * d
            np.testing.assert_allclose(
                [back[0] / scale, back[1] / scale, back[2]],
                [pm_ra, pm_dec, rv],
                rtol=1e-12,
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# 5. Mirror symmetry of the axisymmetric branches
# ---------------------------------------------------------------------------


def _kinematic_lp_at(ra, dec, d_kpc, pm_ra, pm_dec, rv_kms):
    gm = _make_gm()
    system = _MockSystem()
    system.star.ra = _MockParam(ra)
    system.star.dec = _MockParam(dec)
    system.star.distance = _MockParam(d_kpc * 1e3)
    system.star.pm_ra = _MockParam(pm_ra)
    system.star.pm_dec = _MockParam(pm_dec)
    system.star.rv = _MockParam(rv_kms * 1e3)
    with pm.Model() as model:
        gm.build_likelihood(model, system)
    return float(model["galacticmodel.kinematic_prior"].eval())


def test_disk_kinematic_prior_is_mirror_symmetric(monkeypatch):
    """
    Given the disk branches alone (the bar is not axisymmetric, so its
      density is switched off: log n_bulge -> -inf to double precision),
    When a star at (R, phi, z) with polar velocity (v_R, v_phi, v_z) is
      compared with its mirror image at (R, -phi, z) with the SAME polar
      velocity -- l -> -l about the frame's own center, each built in
      astropy and observed from the Sun,
    Then the kinematic logp is identical (1e-9 nats): the thin and thick
      disks are axisymmetric, so nothing in the frame may tell the two
      apart.
    """
    monkeypatch.setattr(gm_mod, "BULGE_CENTRAL_NUMBER_DENSITY", 1e-300)
    rng = np.random.default_rng(5)
    for l_deg, b_deg, d_kpc in (
        (6.65, 3.71, 2.5),
        (30, -3, 4.0),
        (5, 0, 12.0),
        (90, 3, 1.0),
    ):
        xyz = _astropy_galcen(l_deg, b_deg, d_kpc)
        polar = rng.normal(0, 40, 3) + [0.0, DISK_ROTATION_VELOCITY, 0.0]
        twin = xyz * np.array([1.0, -1.0, 1.0])
        lps = []
        for p in (xyz, twin):
            ra, dec, d, pm_ra, pm_dec, rv = _observe(
                p, _polar_to_astropy(p, *polar)
            )
            assert d == pytest.approx(d_kpc, rel=1e-12)
            lps.append(_kinematic_lp_at(ra, dec, d, pm_ra, pm_dec, rv))
        assert lps[0] == pytest.approx(lps[1], abs=1e-9, rel=0), (l_deg, lps)


# ---------------------------------------------------------------------------
# 6. Sense of rotation
# ---------------------------------------------------------------------------


def test_disk_and_bulge_rotate_with_the_sun():
    """
    Given a bulge star co-rotating with the Sun in astropy's sense (the
      sign of x v_y - y v_x matches the Sun's) and its counter-rotating
      twin,
    When the model decomposes their velocities,
    Then the co-rotating one has v_phi = +Omega r > 0 and the twin
      v_phi = -Omega r, and the Sun itself has v_phi > 0 -- the model's
      positive v_phi IS astropy's sense of Galactic rotation.
    """
    sun_lz = -SUN_GC_DISTANCE * SUN_GALCEN_V[1]  # astropy: x_sun < 0
    for l_deg, b_deg, d_kpc in ((1.0, -3.9, 7.5), (-3.0, 2.0, 8.5)):
        xyz = _astropy_galcen(l_deg, b_deg, d_kpc)
        big_r = np.hypot(xyz[0], xyz[1])
        speed = BULGE_ROTATION_ANGULAR_VELOCITY * big_r
        v_co = _corotating(xyz, speed)
        assert np.sign(xyz[0] * v_co[1] - xyz[1] * v_co[0]) == np.sign(sun_lz)
        _, v_phi, _, _ = _model_polar(*_observe(xyz, v_co))
        assert v_phi == pytest.approx(speed, abs=TOL_KMS)
        assert v_phi > 0
        _, v_phi, _, _ = _model_polar(*_observe(xyz, -v_co))
        assert v_phi == pytest.approx(-speed, abs=TOL_KMS)
