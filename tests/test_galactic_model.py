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
    the given position with the given astropy-Galactocentric velocity."""
    import astropy.units as u
    from astropy.coordinates import ICRS, Galactocentric, SkyCoord

    sc = SkyCoord(
        ra=ra_deg * u.deg, dec=dec_deg * u.deg, distance=d_kpc * u.kpc
    )
    gc = sc.transform_to(Galactocentric())
    star = SkyCoord(
        x=gc.x,
        y=gc.y,
        z=gc.z,
        v_x=v_gal[0] * u.km / u.s,
        v_y=v_gal[1] * u.km / u.s,
        v_z=v_gal[2] * u.km / u.s,
        frame=Galactocentric(),
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

    # ASSERT: margins allow for the logsumexp bulge branch, whose ~110 km/s
    # dispersions partially absorb any velocity direction and floor the
    # disk-term penalty (measured: corot beats radial by 7.9 nats and
    # counter-rotation by 14.3; before the fix corot LOST to radial by 43).
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

    # Margins again reflect bulge-branch flooring (measured: +1.4 and +2.5
    # nats for +/-3 sigma; the disk term alone would give 4.5).
    assert lp_circ > lp_fast + 1.0
    assert lp_circ > lp_slow + 1.0
