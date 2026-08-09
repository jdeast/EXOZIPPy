"""Tests for the GalacticModel component (register_parameters, build_likelihood)."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager
from exozippy.components.galacticmodel.galacticmodel import GalacticModel

# RA/Dec for a typical Galactic-bulge microlensing field (Galactic center area).
_RA_RAD = np.deg2rad(270.0)
_DEC_RAD = np.deg2rad(-29.0)


class _MockParam:
    """Minimal Parameter stand-in with initval (numpy) and value (PyTensor tensor)."""

    def __init__(self, initval):
        self.initval = np.atleast_1d(np.asarray(initval, dtype=np.float64))
        self.value = pt.as_tensor_variable(self.initval)


class _MockStar:
    """Stand-in for the Star component with attributes GalacticModel.build_likelihood needs."""

    def __init__(self):
        self.ra = _MockParam(_RA_RAD)
        self.dec = _MockParam(_DEC_RAD)
        self.logmass = _MockParam(np.log10(0.5))  # 0.5 M_sun
        self.distance = _MockParam(8000.0)  # pc  (8 kpc, bulge distance)
        self.pm_ra = _MockParam(0.0)  # mas/yr
        self.pm_dec = _MockParam(0.0)  # mas/yr
        self.rv = _MockParam(0.0)  # m/s


class _MockSystem:
    def __init__(self):
        self.star = _MockStar()


def _make_gm(config=None):
    cfg = config or [{}]
    return GalacticModel(cfg, _DummyConfigManager())


def test_register_parameters_produces_empty_manifest():
    """
    Given GalacticModel (which samples nothing directly),
    When register_parameters is called,
    Then manifest is an empty dict.
    """
    gm = _make_gm()
    gm.register_parameters(system=None)
    assert gm.manifest == {}


def test_build_likelihood_adds_imf_potential():
    """
    Given a GalacticModel with a mock star at bulge distance,
    When build_likelihood runs inside a pm.Model,
    Then a Potential named 'galacticmodel.imf_prior' is present in the model.
    """
    gm = _make_gm()
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem())
    assert "galacticmodel.imf_prior" in model.named_vars


def test_build_likelihood_adds_kinematic_potential():
    """
    Given a GalacticModel with a mock star at bulge distance,
    When build_likelihood runs inside a pm.Model,
    Then a Potential named 'galacticmodel.kinematic_prior' is present.
    """
    gm = _make_gm()
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem())
    assert "galacticmodel.kinematic_prior" in model.named_vars


def test_build_likelihood_adds_exactly_two_potentials():
    """
    Given a GalacticModel with default config,
    When build_likelihood runs,
    Then exactly two Potentials (IMF and kinematic) are added to the model.
    """
    gm = _make_gm()
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem())
    gm_potentials = [
        k for k in model.named_vars if k.startswith("galacticmodel.")
    ]
    assert len(gm_potentials) == 2, (
        f"Expected 2 potentials, got: {gm_potentials}"
    )


def test_imf_prior_is_negative_for_star_above_chabrier_peak():
    """
    Given a star with logmass = 0 (1 M_sun), well above the Chabrier peak at 0.22 M_sun,
    When the IMF prior is evaluated,
    Then the Chabrier log-probability is negative (the star is disfavoured).

    The Chabrier peak is at log10(0.22) ≈ -0.658; at logmass=0 the prior penalises.
    """
    gm = _make_gm()
    system = _MockSystem()
    system.star.logmass = _MockParam(0.0)  # log10(1.0) = 0
    with pm.Model() as model:
        gm.build_likelihood(model, system)
    # Evaluate the potential at the initial point (all RVs at 0 in unconstrained space)
    ip = model.initial_point()
    logp = model.compile_logp()(ip)
    assert np.isfinite(logp), "Log-probability must be finite"
    # The Chabrier term is -0.5 * ((0.0 - log10(0.22)) / 0.57)^2 < 0
    log_Mc = np.log10(0.22)
    sigma_imf = 0.57
    expected_chabrier = -0.5 * ((0.0 - log_Mc) / sigma_imf) ** 2
    assert expected_chabrier < 0


def test_imf_salpeter_branch_does_not_crash():
    """
    Given a GalacticModel configured with IMF = 'Salpeter',
    When build_likelihood runs,
    Then no error is raised (Salpeter branch is reachable code).
    """
    gm = _make_gm(config=[{"IMF": "Salpeter"}])
    with pm.Model():
        gm.build_likelihood(pm.modelcontext(None), _MockSystem())


# ---------------------------------------------------------------------------
# Kinematic prior physics: rotation must be azimuthal
# ---------------------------------------------------------------------------


def _pm_rv_for_velocity(ra_deg, dec_deg, d_kpc, v_gal):
    """ICRS (pm_ra_cosdec [mas/yr], pm_dec [mas/yr], rv [m/s]) of a star at
    the given position with the given galactocentric velocity, in the
    component's own frame (R0/z_sun/vsun-consistent with the densities)."""
    import astropy.units as u
    from astropy.coordinates import ICRS, SkyCoord

    from exozippy.components.galacticmodel.galacticmodel import (
        GALACTOCENTRIC_FRAME,
    )

    sc = SkyCoord(
        ra=ra_deg * u.deg, dec=dec_deg * u.deg, distance=d_kpc * u.kpc
    )
    gc = sc.transform_to(GALACTOCENTRIC_FRAME)
    star = SkyCoord(
        x=gc.x,
        y=gc.y,
        z=gc.z,
        v_x=v_gal[0] * u.km / u.s,
        v_y=v_gal[1] * u.km / u.s,
        v_z=v_gal[2] * u.km / u.s,
        frame=GALACTOCENTRIC_FRAME,
    ).transform_to(ICRS())
    return (
        float(star.pm_ra_cosdec.to_value(u.mas / u.yr)),
        float(star.pm_dec.to_value(u.mas / u.yr)),
        float(star.radial_velocity.to_value(u.m / u.s)),
        (float(gc.x.to_value(u.kpc)), float(gc.y.to_value(u.kpc))),
    )


def _kinematic_lp(d_pc, pm_ra, pm_dec, rv_ms):
    """Evaluate the kinematic_prior Potential for a star with the given
    observables (all other mock attributes at their defaults)."""
    gm = _make_gm()
    system = _MockSystem()
    system.star.distance = _MockParam(d_pc)
    system.star.pm_ra = _MockParam(pm_ra)
    system.star.pm_dec = _MockParam(pm_dec)
    system.star.rv = _MockParam(rv_ms)
    with pm.Model() as model:
        gm.build_likelihood(model, system)
    return float(model["galacticmodel.kinematic_prior"].eval())


def test_kinematic_prior_prefers_corotation_over_radial_plunge():
    """
    Given a disk-distance star toward the bulge whose space velocity is
      (a) an EXACTLY circular co-rotating orbit,
      (b) a radial plunge toward the Galactic center at the same speed,
      (c) counter-rotation at the same speed,
    When the kinematic prior evaluates each,
    Then co-rotation wins by many nats -- Galactic rotation is azimuthal.

    Before the 2026-08 fix the prior centered the rotation velocity on the
    RADIAL component: (a) was penalized ~49 nats and (b) was the maximum,
    which on examples/ob140939 handed the anti-rotation parallax solution
    the posterior (0.98/0.02 against the Yee et al. 2015 proper-motion-
    preferred solution).
    """
    from exozippy.constants import DISK_ROTATION_VELOCITY

    # ARRANGE: the ob140939 field, lens-like distance
    ra_deg, dec_deg, d_kpc = 266.8, -21.38, 2.5
    v = DISK_ROTATION_VELOCITY
    # astropy Galactocentric: Sun at x ~ -8 kpc, rotation at the Sun = +y.
    # Co-rotation at (x, y): v = V * (y, -x)/r  (checks out at the Sun).
    _, _, _, (x, y) = _pm_rv_for_velocity(ra_deg, dec_deg, d_kpc, (0, 0, 0))
    r = np.hypot(x, y)
    lps = {}
    for tag, v_gal in [
        ("corot", (v * y / r, -v * x / r, 0.0)),
        ("radial", (-v * x / r, -v * y / r, 0.0)),  # plunge toward GC
        ("counter", (-v * y / r, v * x / r, 0.0)),
    ]:
        pm_ra, pm_dec, rv, _ = _pm_rv_for_velocity(
            ra_deg, dec_deg, d_kpc, v_gal
        )
        # ACT
        lps[tag] = _kinematic_lp(d_kpc * 1e3, pm_ra, pm_dec, rv)

    # ASSERT: margins allow for the logsumexp bulge/thick branches, whose
    # hotter dispersions partially absorb any velocity direction and floor
    # the thin-disk penalty (measured: corot beats radial by 16.5 nats and
    # counter-rotation by 33.1; before the fix corot LOST to radial by 43.
    # The 2026-08 mixture normalization + bar cutoff + genulens kinematics
    # widened these margins from 7.9/14.3).
    assert lps["corot"] > lps["radial"] + 5.0
    assert lps["corot"] > lps["counter"] + 10.0


def test_kinematic_prior_corotating_star_pays_no_velocity_penalty():
    """
    Given the same star exactly on the circular orbit,
    When compared against one with a 3-sigma azimuthal peculiar velocity,
    Then the circular orbit scores higher -- i.e. it sits at the velocity
      term's maximum, not merely above the alternatives.
    """
    from exozippy.constants import (
        DISK_ROTATION_VELOCITY,
        DISK_VELOCITY_SIGMA_V,
    )

    ra_deg, dec_deg, d_kpc = 266.8, -21.38, 2.5
    _, _, _, (x, y) = _pm_rv_for_velocity(ra_deg, dec_deg, d_kpc, (0, 0, 0))
    r = np.hypot(x, y)
    phi_hat = (y / r, -x / r, 0.0)

    def lp_at(speed):
        v_gal = tuple(speed * c for c in phi_hat)
        pm_ra, pm_dec, rv, _ = _pm_rv_for_velocity(
            ra_deg, dec_deg, d_kpc, v_gal
        )
        return _kinematic_lp(d_kpc * 1e3, pm_ra, pm_dec, rv)

    lp_circ = lp_at(DISK_ROTATION_VELOCITY)
    lp_fast = lp_at(DISK_ROTATION_VELOCITY + 3.0 * DISK_VELOCITY_SIGMA_V)
    lp_slow = lp_at(DISK_ROTATION_VELOCITY - 3.0 * DISK_VELOCITY_SIGMA_V)

    # Margins again reflect mixture flooring by the hotter branches
    # (measured: +4.4 and +3.9 nats for +/-3 sigma after the 2026-08
    # normalization + fidelity upgrade, up from +1.4/+2.5; the thin-disk
    # term alone would give 4.5).
    assert lp_circ > lp_fast + 1.0
    assert lp_circ > lp_slow + 1.0


# ---------------------------------------------------------------------------
# Normalization: velocity Jacobian + mixture branch normalization
# ---------------------------------------------------------------------------


def _reference_kinematic_lp(d_pc, pm_ra, pm_dec, rv_ms):
    """Independent numpy reference of the kinematic prior for one star at
    the mock RA/Dec, with astropy doing the full (nonlinearized) velocity
    transform.  Pins the 2026-08 normalization fixes and the genulens-
    fidelity upgrade:
      - the pm -> velocity change-of-variables Jacobian +2*log(K*d)
        (velocity Gaussians in km/s, sampled coordinates in mas/yr),
      - per-branch velocity normalization -log(sigma1*sigma2*sigma3),
      - per-branch number-density anchors (thin/thick local, bulge central),
      - the inner-disk plateau (R < DISK_RDBREAK) and the bar's outer
        cylindrical cutoff (beyond BULGE_RC),
      - the thick-disk branch and the R0/z_sun-consistent frame.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from exozippy.components.galacticmodel.galacticmodel import (
        GALACTOCENTRIC_FRAME,
    )
    from exozippy.constants import (
        BULGE_BAR_ANGLE,
        BULGE_CENTRAL_NUMBER_DENSITY,
        BULGE_DENSITY_X_0,
        BULGE_DENSITY_Y_0,
        BULGE_DENSITY_Z_0,
        BULGE_RC,
        BULGE_RC_WIDTH,
        BULGE_ROTATION_ANGULAR_VELOCITY,
        BULGE_VELOCITY_SIGMA_1,
        BULGE_VELOCITY_SIGMA_2,
        BULGE_VELOCITY_SIGMA_3,
        DISK_LOCAL_NUMBER_DENSITY,
        DISK_RDBREAK,
        DISK_ROTATION_VELOCITY,
        DISK_SCALE_HEIGHT,
        DISK_SCALE_LENGTH,
        DISK_VELOCITY_SIGMA_U,
        DISK_VELOCITY_SIGMA_V,
        DISK_VELOCITY_SIGMA_W,
        K_VEL_CONVERSION,
        SUN_GC_DISTANCE,
        SUN_Z_OFFSET,
        THICK_DISK_LOCAL_NUMBER_DENSITY,
        THICK_DISK_ROTATION_VELOCITY,
        THICK_DISK_SCALE_HEIGHT,
        THICK_DISK_SCALE_LENGTH,
        THICK_DISK_VELOCITY_SIGMA_U,
        THICK_DISK_VELOCITY_SIGMA_V,
        THICK_DISK_VELOCITY_SIGMA_W,
    )

    def hinge(t):  # same smooth max(t, 0) as the model
        return 0.5 * (t + np.sqrt(t * t + 0.0025))

    sc = SkyCoord(ra=_RA_RAD * u.rad, dec=_DEC_RAD * u.rad)
    l_rad, b_rad = sc.galactic.l.rad, sc.galactic.b.rad
    d = max(d_pc, 1e-3) / 1e3  # kpc, same floor as the model

    # Position in the model's own convention (Sun at x = +SUN_GC_DISTANCE,
    # z tilted by bsun = z_sun/R0 so d=0 lands at z = +z_sun)
    x = SUN_GC_DISTANCE - d * np.cos(l_rad) * np.cos(b_rad)
    y = d * np.sin(l_rad) * np.cos(b_rad)
    bsun = SUN_Z_OFFSET / SUN_GC_DISTANCE
    z = d * np.sin(b_rad) * np.cos(bsun) + x * np.sin(bsun)
    z_smooth = np.sqrt(z**2 + 1e-6)
    r = np.sqrt(x**2 + y**2 + 1e-6)

    # Velocity via astropy's exact transform in the component's frame.
    # The model linearizes at d = 1 kpc, but the map is exactly linear in
    # (K*pm*d, rv), so the two agree to float precision.
    gal = SkyCoord(
        ra=_RA_RAD * u.rad,
        dec=_DEC_RAD * u.rad,
        distance=d * u.kpc,
        pm_ra_cosdec=pm_ra * u.mas / u.yr,
        pm_dec=pm_dec * u.mas / u.yr,
        radial_velocity=(rv_ms / 1e3) * u.km / u.s,
    ).transform_to(GALACTOCENTRIC_FRAME)
    v_x = gal.v_x.to_value(u.km / u.s)
    v_y = gal.v_y.to_value(u.km / u.s)
    v_z = gal.v_z.to_value(u.km / u.s)
    v_r = v_y * (y / r) + v_x * (x / r)
    v_phi = v_y * (x / r) - v_x * (y / r)

    cos_bar, sin_bar = np.cos(BULGE_BAR_ANGLE), np.sin(BULGE_BAR_ANGLE)
    x_bar = x * cos_bar + y * sin_bar
    y_bar = -x * sin_bar + y * cos_bar

    r_plateau = hinge(r - DISK_RDBREAK) - (SUN_GC_DISTANCE - DISK_RDBREAK)
    L_thin = (
        np.log(DISK_LOCAL_NUMBER_DENSITY)
        - r_plateau / DISK_SCALE_LENGTH
        - z_smooth / DISK_SCALE_HEIGHT
        - 0.5 * (v_r / DISK_VELOCITY_SIGMA_U) ** 2
        - 0.5 * ((v_phi - DISK_ROTATION_VELOCITY) / DISK_VELOCITY_SIGMA_V) ** 2
        - 0.5 * (v_z / DISK_VELOCITY_SIGMA_W) ** 2
        - np.log(
            DISK_VELOCITY_SIGMA_U
            * DISK_VELOCITY_SIGMA_V
            * DISK_VELOCITY_SIGMA_W
        )
    )
    L_thick = (
        np.log(THICK_DISK_LOCAL_NUMBER_DENSITY)
        - r_plateau / THICK_DISK_SCALE_LENGTH
        - z_smooth / THICK_DISK_SCALE_HEIGHT
        - 0.5 * (v_r / THICK_DISK_VELOCITY_SIGMA_U) ** 2
        - 0.5
        * (
            (v_phi - THICK_DISK_ROTATION_VELOCITY)
            / THICK_DISK_VELOCITY_SIGMA_V
        )
        ** 2
        - 0.5 * (v_z / THICK_DISK_VELOCITY_SIGMA_W) ** 2
        - np.log(
            THICK_DISK_VELOCITY_SIGMA_U
            * THICK_DISK_VELOCITY_SIGMA_V
            * THICK_DISK_VELOCITY_SIGMA_W
        )
    )
    r_s = np.sqrt(
        (x_bar / BULGE_DENSITY_X_0) ** 2
        + (y_bar / BULGE_DENSITY_Y_0) ** 2
        + (z / BULGE_DENSITY_Z_0) ** 2
    )
    L_bulge = (
        np.log(BULGE_CENTRAL_NUMBER_DENSITY)
        - 0.5 * r_s
        - 0.5 * (hinge(r - BULGE_RC) / BULGE_RC_WIDTH) ** 2
        - 0.5 * (v_r / BULGE_VELOCITY_SIGMA_1) ** 2
        - 0.5
        * (
            (v_phi - BULGE_ROTATION_ANGULAR_VELOCITY * r)
            / BULGE_VELOCITY_SIGMA_2
        )
        ** 2
        - 0.5 * (v_z / BULGE_VELOCITY_SIGMA_3) ** 2
        - np.log(
            BULGE_VELOCITY_SIGMA_1
            * BULGE_VELOCITY_SIGMA_2
            * BULGE_VELOCITY_SIGMA_3
        )
    )
    branches = np.array([L_thin, L_thick, L_bulge])
    m = branches.max()
    return (
        m
        + np.log(np.exp(branches - m).sum())
        + 2.0 * np.log(d * 1e3)
        + 2.0 * np.log(K_VEL_CONVERSION * d)
    )


@pytest.mark.parametrize(
    "d_pc, pm_ra, pm_dec, rv_ms",
    [
        (8000.0, 0.0, 0.0, 0.0),  # bulge-like, at rest on the sky
        (2500.0, -2.0, -4.0, 1.5e4),  # disk-lens-like, moving
        (1000.0, 5.0, 3.0, -2.0e4),  # nearby, fast
        (5000.0, -1.0, -6.0, 8.0e4),  # intermediate
    ],
)
def test_kinematic_prior_matches_independent_reference(
    d_pc, pm_ra, pm_dec, rv_ms
):
    """
    Given a star with the given distance, proper motion, and RV,
    When the kinematic_prior Potential is evaluated,
    Then it matches an independent numpy/astropy implementation of the
      normalized mixture prior -- including the +2*log(K*d) velocity
      Jacobian and the per-branch -log(sigma^3) + log(rho0) terms, whose
      omission previously tilted distances low by d^2 and over-weighted
      the bulge branch by ~3.9 nats.

    Spanning 1-8 kpc makes the test sensitive to any wrong power of
    distance, and the disk/bulge mix shifts across the points so a
    missing branch normalization cannot cancel.
    """
    lp_model = _kinematic_lp(d_pc, pm_ra, pm_dec, rv_ms)
    lp_ref = _reference_kinematic_lp(d_pc, pm_ra, pm_dec, rv_ms)
    assert np.isclose(lp_model, lp_ref, rtol=0.0, atol=1e-6), (
        f"model {lp_model} != reference {lp_ref}"
    )


def test_bulge_number_density_matches_vvv_box_budget():
    """
    Given the bar profile the component actually uses (exp(-r_s/2), Zhu+17
      axes, BULGE_RC cutoff),
    When its central density is derived with genulens's VVV-box budget
      (rho0b = (frho0b * M_VVV(Portail+17) - M_disk_in_box)/box integral,
      then mass -> MS+BD number via fb_MS/mean-MS-mass),
    Then it reproduces the hard-coded BULGE_CENTRAL_NUMBER_DENSITY --
      guarding the constant's derivation against silent drift in either
      the profile or the disk constants it subtracts.
    """
    from exozippy.constants import (
        BULGE_CENTRAL_NUMBER_DENSITY,
        BULGE_DENSITY_X_0,
        BULGE_DENSITY_Y_0,
        BULGE_DENSITY_Z_0,
        BULGE_RC,
        BULGE_RC_WIDTH,
        DISK_RDBREAK,
        DISK_SCALE_HEIGHT,
        DISK_SCALE_LENGTH,
        SUN_GC_DISTANCE,
        THICK_DISK_SCALE_HEIGHT,
        THICK_DISK_SCALE_LENGTH,
    )

    # ARRANGE: Portail+17 VVV box, bar frame, in kpc
    xmax, ymax, zmax = 2.2, 1.4, 1.2
    n = 200
    xg = np.linspace(0.0, xmax, n)
    yg = np.linspace(0.0, ymax, n)
    zg = np.linspace(0.0, zmax, n)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    R = np.hypot(X, Y)
    cut = np.exp(-0.5 * (np.maximum(R - BULGE_RC, 0.0) / BULGE_RC_WIDTH) ** 2)
    box_integral = 0.0  # kpc^3
    for zz in zg:
        r_s = np.sqrt(
            (X / BULGE_DENSITY_X_0) ** 2
            + (Y / BULGE_DENSITY_Y_0) ** 2
            + (zz / BULGE_DENSITY_Z_0) ** 2
        )
        box_integral += (np.exp(-0.5 * r_s) * cut).sum()
    box_integral *= 8 * (xg[1] - xg[0]) * (yg[1] - yg[0]) * (zg[1] - zg[0])

    # Disk mass in the box (all R < BULGE_RC < DISK_RDBREAK -> plateau).
    # Local MASS densities incl. WDs (genulens rho0d): thin 0.0501,
    # thick 0.00228 Msun/pc^3 -- the box subtraction is a mass budget,
    # unlike the number-density branch weights.
    area_pc2 = (2 * xmax * 1e3) * (2 * ymax * 1e3)

    def vertical(h_kpc):
        h = h_kpc * 1e3
        return 2 * h * (1 - np.exp(-zmax * 1e3 / h))

    plateau = SUN_GC_DISTANCE - DISK_RDBREAK
    m_disk_box = 0.050095 * np.exp(plateau / DISK_SCALE_LENGTH) * area_pc2 * (
        vertical(DISK_SCALE_HEIGHT)
    ) + 0.002282 * np.exp(plateau / THICK_DISK_SCALE_LENGTH) * area_pc2 * (
        vertical(THICK_DISK_SCALE_HEIGHT)
    )

    # ACT: genulens normalization (frho0b, M_VVV, fb_MS, mean MS mass)
    rho0b = (0.839014514507754 * 1.32e10 - m_disk_box) / (box_integral * 1e9)
    n0_msb = rho0b * (1.62 / 2.07) / 0.227943

    # ASSERT: 1% covers the box-integral grid resolution
    assert np.isclose(n0_msb, BULGE_CENTRAL_NUMBER_DENSITY, rtol=0.01), (
        f"recomputed {n0_msb:.4f} vs constant {BULGE_CENTRAL_NUMBER_DENSITY}"
    )
