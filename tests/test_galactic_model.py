"""Tests for the GalacticModel component (register_parameters, build_likelihood)."""

import logging
import math

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager, _DummySystem, _MockParam
from exozippy.components.galacticmodel.galacticmodel import (
    GalacticModel,
    chabrier_logmass_logp,
)
from exozippy.components.galacticmodel.galacticmodel import (
    _chabrier_log_norm as _component_chabrier_log_norm,
)

# RA/Dec for a typical Galactic-bulge microlensing field (Galactic center area).
_RA_RAD = np.deg2rad(270.0)
_DEC_RAD = np.deg2rad(-29.0)


# star/defaults.yaml hard bounds on the sampled log10 mass (dex solMass);
# they are the support the power-law IMF prior normalizes over.
_LOGMASS_LOWER = -9.0
_LOGMASS_UPPER = 2.5


class _MockStar:
    """Stand-in for the Star component with attributes GalacticModel.build_likelihood needs."""

    def __init__(self):
        self.ra = _MockParam(_RA_RAD)
        self.dec = _MockParam(_DEC_RAD)
        self.logmass = _MockParam(
            np.log10(0.5), lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER
        )  # 0.5 M_sun
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


@pytest.mark.parametrize("imf", ["Kroupa", "kroupa", "chabier", "salpetre"])
def test_unimplemented_imf_raises_instead_of_being_ignored(imf):
    """
    Given a GalacticModel configured with an IMF that is not implemented,
    When the component is constructed,
    Then a ValueError names the supported options.

    The power-law options used to be accepted and then silently ignored:
    imf_slope was computed and consumed only by commented-out code, so the
    prior was always the Chabrier lognormal no matter what the user asked
    for.  Kroupa is still unsupported -- it is a broken power law, not a
    single slope.
    """
    # Act / Assert
    with pytest.raises(ValueError, match="not implemented"):
        _make_gm(config=[{"IMF": imf}])


@pytest.mark.parametrize(
    "imf", ["Chabrier", "chabrier", "Salpeter", "SALPETER"]
)
def test_supported_imfs_are_accepted_case_insensitively(imf):
    """
    Given a GalacticModel configured with an implemented IMF, in any case,
    When build_likelihood runs,
    Then the IMF potential is added (the option is live, not decorative).
    """
    # Arrange
    gm = _make_gm(config=[{"IMF": imf}])

    # Act
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem())

    # Assert
    assert "galacticmodel.imf_prior" in model.named_vars


# ---------------------------------------------------------------------------
# IMF priors: the change of variables into the sampled log10-mass coordinate,
# and the normalization over that coordinate's bounded support
# ---------------------------------------------------------------------------


def _imf_lp(masses, config=None):
    """The imf_prior Potential for stars of the given masses (M_sun)."""
    gm = _make_gm(config=config or [{}])
    system = _MockSystem()
    system.star.logmass = _MockParam(
        np.log10(masses), lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER
    )
    with pm.Model() as model:
        gm.build_likelihood(model, system)
    return float(model["galacticmodel.imf_prior"].eval())


def _salpeter_analytic(logmass):
    """log p(log10 M) for dN/dM ~ M^-2.35, normalized over the logmass
    bounds -- derived here independently of the implementation.

    p(x) dx with x = log10(M): dN/dx = (dN/dM)(dM/dx) = M^-a * M ln10, so
    p(x) ~ 10^((1-a)x) with (1-a) = -1.35, and
    Z = int 10^(kx) dx = (10^(k*up) - 10^(k*lo)) / (k ln10).
    """
    k = 1.0 - 2.35
    ln10 = np.log(10.0)
    log_z = np.log(
        (10.0 ** (k * _LOGMASS_UPPER) - 10.0 ** (k * _LOGMASS_LOWER))
        / (k * ln10)
    )
    return k * ln10 * np.asarray(logmass) - log_z


# Chabrier 2003 SYSTEM IMF (PASP 115, 763, Table 1), written out here from
# the paper rather than imported, so a silent edit to the component's
# constants is a test failure.  It is PIECEWISE: lognormal below 1 Msun,
# dN/dlog m ~ m^-1.3 above it.
_LOG_MC = np.log10(0.22)
_SIGMA_IMF = 0.57
_HIGH_MASS_X = 1.3
_MATCH = 0.0  # 1 Msun
_BLEND_10_90_DEX = 0.2


def _chabrier_lognormal_log_norm():
    """log Z for the lognormal segment ALONE over the logmass bounds.

    Not the model's normalizer any more -- the blend has no closed form --
    but kept because it is the one piece of this prior that still HAS an
    analytic answer, so it is what pins the low-mass segment (where the
    lognormal is the whole density to 1e-3) without going through quadrature
    at all.  stdlib math.erf, not the component's scipy/np.select branches.

    p(x) ~ exp(-0.5 ((x - mu)/sigma)^2), so with u = (x - mu)/sigma
        Z = int_lo^hi exp(-u^2/2) sigma du
          = sigma sqrt(2 pi) [Phi(u_hi) - Phi(u_lo)]
    """
    u_lo = (_LOGMASS_LOWER - _LOG_MC) / _SIGMA_IMF
    u_hi = (_LOGMASS_UPPER - _LOG_MC) / _SIGMA_IMF
    phi_diff = 0.5 * (
        math.erf(u_hi / math.sqrt(2.0)) - math.erf(u_lo / math.sqrt(2.0))
    )
    return (
        math.log(_SIGMA_IMF)
        + 0.5 * math.log(2.0 * math.pi)
        + math.log(phi_diff)
    )


def _chabrier_unnormalized(logmass):
    """The blended Chabrier density, unnormalized, written independently.

    Same three statements the component makes -- lognormal, a tail whose
    amplitude is fixed by continuity at the match, a logistic ramp between
    them -- but composed as a plain WEIGHTED SUM of densities rather than
    through logaddexp/softplus, which is exactly the algebra the component's
    stable formulation claims to be equivalent to.  Fine here because these
    test masses are near the peak, where nothing underflows.
    """
    x = np.asarray(logmass, dtype=float)
    lognormal = np.exp(-0.5 * ((x - _LOG_MC) / _SIGMA_IMF) ** 2)
    at_match = np.exp(-0.5 * ((_MATCH - _LOG_MC) / _SIGMA_IMF) ** 2)
    tail = at_match * 10.0 ** (-_HIGH_MASS_X * (x - _MATCH))
    s = _BLEND_10_90_DEX / (2.0 * np.log(9.0))
    w = 1.0 / (1.0 + np.exp((x - _MATCH) / s))
    return np.log(w * lognormal + (1.0 - w) * tail)


def _chabrier_log_norm():
    """log Z of the blended density, by ADAPTIVE quadrature.

    scipy's Gauss-Kronrod, deliberately, against the component's uniform
    trapezoid: agreeing to 1e-10 is then a statement about the integral and
    not about two copies of one algorithm.  Split at the match point because
    the integrand's curvature changes there.
    """
    from scipy.integrate import quad

    def f(x):
        return np.exp(_chabrier_unnormalized(x))

    lo, _ = quad(f, _LOGMASS_LOWER, _MATCH, limit=400)
    hi, _ = quad(f, _MATCH, _LOGMASS_UPPER, limit=400)
    return math.log(lo + hi)


def _chabrier_analytic(logmass):
    """log p(log10 M) for the Chabrier IMF, normalized over the logmass
    bounds."""
    return _chabrier_unnormalized(logmass) - _chabrier_log_norm()


_ANALYTIC = {"salpeter": _salpeter_analytic, "chabrier": _chabrier_analytic}


def test_salpeter_imf_logp_matches_the_analytic_power_law():
    """
    Given the Salpeter IMF selected on the galacticmodel block,
    When the IMF prior is evaluated for stars of 0.3 and 1.0 M_sun,
    Then it equals the analytic normalized log density in the SAMPLED
      coordinate, (1 - alpha) * ln10 * log10(M) - log(Z).

    The change of variables is the whole point: dN/dM ~ M^-2.35 is a density
    in M, but the potential is applied to star.logmass, so the Jacobian
    dM/dlog10(M) = M ln10 turns the exponent -2.35 into a slope of -1.35 in
    the sampled coordinate.  Dropping it would tilt the prior by a full dex
    per dex.
    """
    # Arrange
    masses = [0.3, 1.0]
    expected = float(np.sum(_salpeter_analytic(np.log10(masses))))

    # Act
    got = _imf_lp(masses, config=[{"IMF": "Salpeter"}])

    # Assert
    assert got == pytest.approx(expected, rel=1e-12)


def test_salpeter_imf_slope_is_the_mass_space_exponent_plus_one():
    """
    Given two stars one dex apart in mass,
    When the Salpeter IMF prior is evaluated for each,
    Then the logp difference is exactly (SALPETER_IMF_SLOPE + 1) * ln10 per
      dex -- i.e. -1.35 * ln10, not -2.35 * ln10.

    This is normalization-independent, so it isolates the constant that is
    easy to get wrong: SALPETER_IMF_SLOPE is the signed MASS-space exponent
    (-alpha), not an already-converted log-space slope.
    """
    # Arrange
    from exozippy.constants import SALPETER_IMF_SLOPE

    cfg = [{"IMF": "salpeter"}]

    # Act
    lp_lo = _imf_lp([0.1], config=cfg)
    lp_hi = _imf_lp([1.0], config=cfg)

    # Assert
    assert SALPETER_IMF_SLOPE == -2.35
    assert lp_hi - lp_lo == pytest.approx(
        (SALPETER_IMF_SLOPE + 1.0) * np.log(10.0), rel=1e-12
    )


@pytest.mark.parametrize("imf", ["chabrier", "salpeter"])
def test_both_imf_priors_are_normalized_densities(imf):
    """
    Given either supported IMF,
    When exp(logp) is integrated over the sampled logmass support,
    Then the integral is 1.

    BOTH branches must be proper normalized densities over the SAME
    support, otherwise switching IMF moves logp by an arbitrary offset
    instead of by a meaningful amount.  The chabrier branch dropped its
    truncated-lognormal constant until 2026-08, which is exactly that
    asymmetry.

    The Potential sums over stars, so the per-element density comes from
    the analytic form; the second assertion is what ties that form to the
    implementation (evaluated on the same grid, as 200001 "stars").
    """
    # Arrange
    grid = np.linspace(_LOGMASS_LOWER, _LOGMASS_UPPER, 200001)
    analytic = _ANALYTIC[imf]
    gm = _make_gm(config=[{"IMF": imf}])
    system = _MockSystem()
    system.star.logmass = _MockParam(
        grid, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER
    )

    # Act
    with pm.Model() as model:
        gm.build_likelihood(model, system)
    integral = np.trapezoid(np.exp(analytic(grid)), grid)
    total = float(model["galacticmodel.imf_prior"].eval())

    # Assert
    assert integral == pytest.approx(1.0, rel=1e-6)
    assert total == pytest.approx(float(np.sum(analytic(grid))), rel=1e-12)


def test_salpeter_imf_gradient_is_finite():
    """
    Given the Salpeter IMF prior,
    When its gradient w.r.t. the sampled log10 mass is evaluated,
    Then it is finite and equals the constant slope (1 - alpha) * ln10.
    """
    # Arrange
    gm = _make_gm(config=[{"IMF": "salpeter"}])
    system = _MockSystem()
    logmass = pt.dvector("logmass")
    system.star.logmass = _MockParam(
        [np.log10(0.3), np.log10(1.0)],
        lower=_LOGMASS_LOWER,
        upper=_LOGMASS_UPPER,
    )
    system.star.logmass.value = logmass

    # Act
    with pm.Model() as model:
        gm.build_likelihood(model, system)
    node = model["galacticmodel.imf_prior"]
    grad = pt.grad(pt.sum(node), logmass).eval(
        {logmass: np.array([np.log10(0.3), np.log10(1.0)])}
    )

    # Assert
    assert np.all(np.isfinite(grad))
    assert np.allclose(grad, (1.0 - 2.35) * np.log(10.0))


def test_chabrier_is_the_default_and_carries_the_truncation_normalizer():
    """
    Given a galacticmodel block with no IMF key,
    When both potentials are evaluated for the standard 0.5 Msun mock star,
    Then the IMF prior is the blended density minus its normalizer, and the
      kinematic prior is untouched.

    Two deliberate shifts have landed on this number and both are recorded
    here so a third one has to be argued for rather than re-pinned:

      -0.19563865   unnormalized lognormal, the original
      -0.55245825   ... minus the truncated-lognormal constant, 0.35681960
                    (normalizing chabrier, so its logp is comparable with
                    salpeter's)
      -0.54883196   ... the matched m^-1.3 tail (review 3.7.1), which moves
                    the normalizer to 0.35391522 and this star by +0.0037

    The tail barely reaches down here, which is the point: 0.5 Msun is 6.6
    blend scales below the match, so the tail contributes 1.3e-3 of the
    mixture and moves the density itself by 7.2e-4 nats.  The rest of the
    +0.0037 is the normalizer.  A fix meant to change massive stars had
    better not move a half-solar-mass one by more than that.
    """
    # Arrange
    gm = _make_gm()
    lognormal_only = -0.19563864866861083  # unnormalized, the original
    log_z = _chabrier_log_norm()

    # Act
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem())
    imf = float(model["galacticmodel.imf_prior"].eval())
    kinematic = float(model["galacticmodel.kinematic_prior"].eval())

    # Assert
    assert gm.imf == "chabrier"
    # The blend barely reaches the density itself this far below the match...
    assert imf + log_z == pytest.approx(lognormal_only, abs=1e-3)
    # ... so most of the delta from the original is the normalizer.
    assert log_z == pytest.approx(0.3539152164266602, rel=1e-9)
    assert imf == pytest.approx(-0.5488319618, rel=1e-9)
    # untouched by the IMF change (baseline captured from master)
    assert kinematic == 10.09330069291524


def test_multiple_galacticmodel_blocks_raise():
    """
    Given two galacticmodel config blocks,
    When the component is constructed,
    Then a ValueError explains that one sight line takes one block.

    Only config[0] was ever read for IMF/anchor_idx, but the extra blocks
    leaked into the likelihood through the pre-computed (n_blocks, 3, 3)
    rotation stack: with 2 blocks and 1 star the whole kinematic prior was
    broadcast to shape (2,) and therefore counted TWICE.
    """
    # Act / Assert
    with pytest.raises(ValueError, match="exactly one config block"):
        _make_gm(config=[{"name": "a"}, {"name": "b"}])


def test_kinematic_prior_scales_with_star_count_not_block_count():
    """
    Given one galacticmodel block and one star, then the same block with two
    identical stars,
    When the kinematic prior is evaluated,
    Then the two-star value is exactly twice the one-star value.

    This pins the anchor geometry as scalars broadcasting over stars: the
    old (n_blocks, 3, 3) stack made the sum's length depend on the number of
    config blocks rather than the number of stars.
    """
    # Arrange
    one = _MockSystem()
    two = _MockSystem()
    for attr in ("ra", "dec", "logmass", "distance", "pm_ra", "pm_dec", "rv"):
        val = float(np.atleast_1d(getattr(one.star, attr).initval)[0])
        setattr(two.star, attr, _MockParam([val, val]))

    # Act
    lps = []
    for system in (one, two):
        gm = _make_gm()
        with pm.Model() as model:
            gm.build_likelihood(model, system)
        lps.append(float(model["galacticmodel.kinematic_prior"].eval()))

    # Assert
    assert np.isclose(lps[1], 2.0 * lps[0], rtol=1e-12)


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


# ---------------------------------------------------------------------------
# A power-law IMF raises star.logmass's unphysical floor
# ---------------------------------------------------------------------------


_HBL_DEX = np.log10(0.075)  # hydrogen-burning limit, dex(solMass)


def _logmass_lower(imf=None, user_params=None, names=("A",)):
    """Resolved star.logmass lower bound(s) under the given IMF."""
    from exozippy.components.star.star import Star
    from exozippy.config import ConfigManager

    cm = ConfigManager(dict(user_params or {}))
    star = Star([{"name": n} for n in names], cm)
    system = _DummySystem()
    system.config_manager = cm
    if imf is not None:
        system.galacticmodel = GalacticModel([{"IMF": imf}], cm)

    star.register_parameters(system)
    with pm.Model() as model:
        star.add_parameter(model=model, param_name="logmass", system=system)
    return np.atleast_1d(star.logmass.lower)


def test_salpeter_raises_the_logmass_floor_and_warns(caplog):
    """
    Given a galacticmodel selecting the Salpeter power law,
    When the star parameters are registered,
    Then star.logmass's lower bound is raised from the unphysical -9 dex to
      the hydrogen-burning limit, with a warning naming the parameter and
      both the old and new bound.

    Under chabrier the -9 floor is inert (density ~exp(-107) there), but a
    power law rises toward low mass at 3.11 nats/dex without limit, so the
    floor would become the answer instead of a safety rail.
    """
    # Act
    with caplog.at_level(logging.WARNING):
        lower = _logmass_lower("salpeter")

    # Assert
    assert lower[0] == pytest.approx(_HBL_DEX, rel=1e-12)
    text = caplog.text
    assert "star.A.logmass" in text
    assert "-9" in text and "0.075 solMass" in text
    assert "chabrier" in text


def test_chabrier_leaves_the_logmass_floor_untouched_and_silent(caplog):
    """
    Given the default Chabrier IMF (and no galacticmodel at all),
    When the star parameters are registered,
    Then star.logmass keeps its defaults.yaml floor of -9 dex and nothing is
      warned -- a planetary-mass lens must still be expressible as a star.
    """
    # Act
    with caplog.at_level(logging.WARNING):
        with_gm = _logmass_lower("chabrier")
        without_gm = _logmass_lower(None)

    # Assert
    assert with_gm[0] == -9.0
    assert without_gm[0] == -9.0
    assert "raising the lower bound" not in caplog.text


def test_user_logmass_bound_above_the_floor_is_preserved():
    """
    Given a user bound TIGHTER than the Salpeter floor,
    When the floor is applied,
    Then the user's bound survives -- the floor may only raise a bound,
      never lower one (the combination is max(user_lower, floor)).
    """
    # Act
    lower = _logmass_lower(
        "salpeter", user_params={"star.A.logmass": {"lower": -0.5}}
    )

    # Assert
    assert lower[0] == pytest.approx(-0.5, rel=1e-12)
    assert lower[0] > _HBL_DEX


def test_user_logmass_bound_below_the_floor_is_raised(caplog):
    """
    Given a user bound BELOW the Salpeter floor,
    When the floor is applied,
    Then it is raised to the floor and warned about: asking for a stellar
      power-law IMF and for support below the hydrogen-burning limit at the
      same time is incoherent, and bounds may only ever be tightened.
    """
    # Act
    with caplog.at_level(logging.WARNING):
        lower = _logmass_lower(
            "salpeter", user_params={"star.A.logmass": {"lower": -3.0}}
        )

    # Assert
    assert lower[0] == pytest.approx(_HBL_DEX, rel=1e-12)
    assert "-3 dex" in caplog.text


def test_floor_applies_to_every_star_the_imf_prior_sums_over():
    """
    Given several modeled stars,
    When the Salpeter floor is applied,
    Then EVERY star gets it -- the IMF potential is a plain sum over the
      whole stars.logmass vector (lens and source alike), so the support
      must be raised over exactly that set.
    """
    # Act
    lower = _logmass_lower("salpeter", names=("Lens", "Source", "C"))

    # Assert
    assert len(lower) == 3
    assert np.allclose(lower, _HBL_DEX)


def test_salpeter_model_builds_with_finite_logp_and_gradient():
    """
    Given a full System whose galacticmodel selects Salpeter,
    When the model is built and evaluated at its initial point,
    Then the raised bound is in force and both logp and its gradient are
      finite (the logit transform must still have a healthy span).

    This also exercises the production lookup path (System.active_components)
    rather than the attribute fallback the unit tests above use.
    """
    # Arrange
    from exozippy.system import System

    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "galacticmodel": [{"name": "gm", "IMF": "Salpeter"}],
    }
    user_params = {
        "star.Lens.ra": {"initval": 270.0, "sigma": 0},
        "star.Lens.dec": {"initval": -29.0, "sigma": 0},
        "star.Source.ra": {"initval": 270.0, "sigma": 0},
        "star.Source.dec": {"initval": -29.0, "sigma": 0},
    }

    # Act
    system = System(config, user_params)
    system.prepare()
    model = system.build_model()
    ip = model.initial_point()
    logp = float(np.asarray(model.compile_logp()(ip)))
    dlogp = model.compile_dlogp()(ip)

    # Assert
    assert np.allclose(np.atleast_1d(system.star.logmass.lower), _HBL_DEX)
    assert np.isfinite(logp)
    assert np.all(np.isfinite(dlogp))


# ---------------------------------------------------------------------------
# The Chabrier high-mass tail (review 3.7.1)
#
# The prior labeled "Chabrier (2003) IMF" was the lognormal SEGMENT alone,
# applied over the whole [-9, 2.5] support, under a comment saying one "would
# smoothly match it to a Salpeter tail" -- in the conditional, describing what
# one could do.  Nobody had.  The real IMF is piecewise, and the lognormal
# steepens without limit above its peak, so massive stars were over-penalized
# by an error that GREW with mass.  Not only a massive-lens concern: the
# imf_prior is one pt.sum over the whole star vector, sources included, and
# bulge source stars sit near 1 Msun.
# ---------------------------------------------------------------------------


def _chabrier_slope(x, h=1e-5):
    """d logp / dx in nats per dex, by central difference."""
    lo = _imf_lp([10.0 ** (x - h)])
    hi = _imf_lp([10.0 ** (x + h)])
    return (hi - lo) / (2.0 * h)


@pytest.mark.parametrize("mass", [3.0, 10.0, 100.0])
def test_high_mass_slope_is_the_power_law_not_the_lognormal(mass):
    """
    Given a star well above the 1 Msun match point,
    When the Chabrier prior's slope in log mass is measured,
    Then it is the tail's constant -x*ln10 = -2.9934 nats/dex, NOT the
      lognormal's -(x - log Mc)/sigma^2, which keeps steepening.

    Slope rather than value, deliberately: it is normalization-independent,
    so it isolates the functional form from the quadrature constant.  It is
    also the quantity the defect was ABOUT -- the lognormal gives -2.02
    nats/dex at 1 Msun and -5.10 at 10, so the two forms disagree more and
    more the further up you go.
    """
    # Arrange
    x = np.log10(mass)
    expected = -_HIGH_MASS_X * np.log(10.0)
    lognormal_slope = -(x - _LOG_MC) / _SIGMA_IMF**2

    # Act
    got = _chabrier_slope(x)

    # Assert
    assert got == pytest.approx(expected, rel=1e-4)
    # ... and the lognormal would have been steeper, by more and more:
    # 0.50 nats/dex at 3 Msun, 2.11 at 10, 4.34 at 100.
    assert lognormal_slope < expected
    assert (expected - lognormal_slope) > 0.4 * np.log10(mass)


@pytest.mark.parametrize("mass", [0.05, 0.22, 0.5])
def test_low_mass_slope_is_still_the_lognormal(mass):
    """
    Given a star well below the match point,
    When the slope is measured,
    Then it is the lognormal's -(x - log Mc)/sigma^2.

    The tail must not leak downward: the lognormal segment IS Chabrier's
    measurement over the mass range that dominates real microlensing lenses,
    and it is the half that was already right.
    """
    # Arrange
    x = np.log10(mass)
    expected = -(x - _LOG_MC) / _SIGMA_IMF**2

    # Act / Assert -- abs=0.02, not tighter: at 0.5 Msun (the closest of
    # these to the match) the ramp has already admitted 1.3e-3 of tail, which
    # tilts the slope by 0.012.  That is the blend doing its job, not a leak.
    assert _chabrier_slope(x) == pytest.approx(expected, abs=0.02)


def test_the_two_segments_meet_at_one_solar_mass():
    """
    Given the match point,
    When the prior's slope is measured exactly there,
    Then it is the MIDPOINT of the two segments' slopes.

    The exact piecewise IMF is only C0 at 1 Msun -- its slope jumps from
    -2.02 to -2.99 nats/dex, a logp KINK of the same class as the SHO
    kernel's Q = 1/2 switch.  A symmetric blend puts the join's slope halfway
    between, which is the signature that the ramp is centred on the match and
    that its two halves carry equal weight there.
    """
    # Arrange
    lognormal_slope = -(0.0 - _LOG_MC) / _SIGMA_IMF**2  # -2.0239
    tail_slope = -_HIGH_MASS_X * np.log(10.0)  # -2.9934

    # Act
    got = _chabrier_slope(0.0)

    # Assert
    assert got == pytest.approx(0.5 * (lognormal_slope + tail_slope), abs=1e-3)


def test_the_smoothing_costs_less_than_the_imfs_own_uncertainty():
    """
    Given the smoothed prior and the EXACT piecewise form,
    When they are compared across the whole transition,
    Then they differ by less than 0.02 nats anywhere.

    This is the trade the smoothing was accepted on.  Blending across a
    width D deviates by roughly D * 0.97 / 8, so 0.2 dex costs ~0.02 nats;
    against that, Chabrier's high-mass exponent is itself uncertain at
    x = 1.3 +/- 0.3, i.e. +/-0.69 nats/dex of slope, ~0.7 nats across a
    decade.  The smoothing is ~30x smaller than the IMF's own uncertainty
    and so cannot be the limiting error.  If someone widens the ramp, this
    test is where they find out what it cost.
    """
    # Arrange -- the exact piecewise form, unnormalized.
    x = np.linspace(-2.0, 2.0, 200001)
    lognormal = -0.5 * ((x - _LOG_MC) / _SIGMA_IMF) ** 2
    at_match = -0.5 * ((_MATCH - _LOG_MC) / _SIGMA_IMF) ** 2
    tail = at_match - _HIGH_MASS_X * np.log(10.0) * (x - _MATCH)
    piecewise = np.where(x <= _MATCH, lognormal, tail)

    # Act
    smoothed = _chabrier_unnormalized(x)

    # Assert
    deviation = np.abs(smoothed - piecewise)
    assert deviation.max() < 0.02
    # ... and it is LOCAL: a third of a dex away it is already negligible.
    assert deviation[np.abs(x - _MATCH) > 0.3].max() < 1e-3


def test_the_prior_and_its_gradient_are_finite_across_the_transition():
    """
    Given the Chabrier prior built symbolically,
    When logp and dlogp are evaluated on a dense grid straddling 1 Msun,
    Then both are finite everywhere and the gradient has no jump.

    The whole reason for smoothing rather than implementing the exact
    piecewise form is that the sampler has to cross this point.  A `pt.where`
    over the two branch log-densities would also be finite here -- both
    branches are -- which is exactly why the implementation uses a weighted
    SUM instead: no branch selection at all, so the documented where-trap is
    unreachable by construction rather than by argument.
    """
    # Arrange -- pt.dvector, never a bare python float array folded into the
    # graph: the gradient is what is on trial, so the input has to be a real
    # symbolic input.
    grid = np.linspace(-0.6, 0.6, 4001)
    param = _MockParam(0.0, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER)
    x = pt.dvector("x")

    # Act
    out = pt.sum(chabrier_logmass_logp(x, param))
    fn = pytensor.function([x], [out, pytensor.grad(out, x)])
    value, grad = fn(grid)

    # Assert
    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))
    # No jump: the largest step between neighbouring gradients is what a
    # kink would blow up.  The exact piecewise form jumps by 0.97 nats/dex
    # in one step; the blend's largest step is ~4e-3 on this grid.
    assert np.abs(np.diff(grad)).max() < 0.05


def test_the_quadrature_normalizer_matches_the_analytic_one_below_the_match():
    """
    Given a logmass support lying entirely below the match point,
    When the component's quadrature normalizer is compared with the ANALYTIC
      truncated-lognormal constant,
    Then they agree to 1e-8.

    The blend has no closed-form integral, so the analytic constant had to
    become quadrature -- and a numerical normalizer with nothing to check it
    against is exactly the kind of thing that is quietly wrong by a factor.
    On a support the tail cannot reach, the closed form is still exact, so it
    validates the quadrature end to end.  The analytic form is written here
    from erf, per-side, and NEVER as a difference of Phi CDFs: this support
    sits far into one tail, where 1-eps minus 1-eps' discards nearly every
    significant digit.
    """
    # Arrange -- 5.2 sigma below the match, so the tail's weight is ~1e-25.
    lower, upper = -9.0, -1.0
    param = _MockParam(0.0, lower=lower, upper=upper)
    u_lo = (lower - _LOG_MC) / _SIGMA_IMF
    u_hi = (upper - _LOG_MC) / _SIGMA_IMF
    # Both bounds BELOW the mean, so use the mirrored erfc form.
    mass = 0.5 * (
        math.erfc(-u_hi / math.sqrt(2.0)) - math.erfc(-u_lo / math.sqrt(2.0))
    )
    analytic = (
        math.log(_SIGMA_IMF) + 0.5 * math.log(2.0 * math.pi) + math.log(mass)
    )

    # Act
    got = float(np.atleast_1d(_component_chabrier_log_norm(param))[0])

    # Assert -- rel=1e-8 is the trapezoid's own dx^2 error on this support,
    # not slack: the two answers differ by 1.7e-9 relative.
    assert got == pytest.approx(analytic, rel=1e-8)


def test_a_ten_solar_mass_star_is_no_longer_over_penalized():
    """
    Given a 10 Msun star and a 0.5 Msun star,
    When the Chabrier prior is evaluated for each,
    Then the massive star is charged 3.46 nats relative to the low-mass one,
      where the lognormal-only form charged 4.03.

    The measured statement of the defect, in the coordinate that matters (a
    difference, so the normalizer cancels).  0.57 nats at 10 Msun, and the
    gap keeps widening -- the two forms' SLOPES differ by 2.1 nats/dex there
    and 4.3 at 100 Msun, so this is the mild end of it.
    """
    # Arrange / Act
    heavy = _imf_lp([10.0])
    light = _imf_lp([0.5])

    # The lognormal-only form, unnormalized -- the constant cancels in the
    # difference, so no normalizer is needed.
    def lognormal_only(x):
        return -0.5 * ((x - _LOG_MC) / _SIGMA_IMF) ** 2

    old_gap = lognormal_only(1.0) - lognormal_only(np.log10(0.5))

    # Assert
    new_gap = heavy - light
    assert new_gap == pytest.approx(-3.46, abs=0.05)
    assert old_gap == pytest.approx(-4.03, abs=0.05)
    assert new_gap > old_gap  # the tail is KINDER to massive stars
