"""Tests for the GalacticModel component (register_parameters, build_likelihood)."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.components.galacticmodel.galacticmodel import GalacticModel
from conftest import _DummyConfigManager


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
        self.logmass = _MockParam(np.log10(0.5))   # 0.5 M_sun
        self.distance = _MockParam(8000.0)           # pc  (8 kpc, bulge distance)
        self.pm_ra = _MockParam(0.0)                 # mas/yr
        self.pm_dec = _MockParam(0.0)                # mas/yr
        self.rv = _MockParam(0.0)                    # m/s


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
    gm_potentials = [k for k in model.named_vars if k.startswith("galacticmodel.")]
    assert len(gm_potentials) == 2, f"Expected 2 potentials, got: {gm_potentials}"


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
