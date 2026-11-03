"""Tests for the GalacticModel component (register_parameters, build_likelihood).

The component is the genulens port (``model: koshimoto21``) since review
1.7.6 removed the 3-population ``model: simplified``; its physics is pinned
against genulens in tests/test_koshimoto21.py and its component wiring in
tests/test_koshimoto21_model.py.  This file keeps the component-level
contracts that do not depend on which Galactic model is inside: the empty
manifest, the two named potentials, one block per sight line, and the
broadcast over stars.
"""

import numpy as np
import pymc as pm
import pytest

from conftest import _DummyConfigManager, _MockParam
from exozippy.components.galacticmodel.galacticmodel import GalacticModel

# RA/Dec for a typical Galactic-bulge microlensing field (Galactic center area).
_RA_RAD = np.deg2rad(270.0)
_DEC_RAD = np.deg2rad(-29.0)


# star/defaults.yaml hard bounds on the sampled log10 mass (dex solMass);
# they are the support the mass function normalizes over.
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

    def position_root(self, i):
        # No derived lens positions here (Star.position_root).
        return i


class _MockSystem:
    def __init__(self):
        self.star = _MockStar()
        # No component models a star's mass loss (Component.initial_logmass)
        # or selects a source (Component.selected_sources).
        self.active_components = {}


def _make_gm(config=None):
    return GalacticModel(list(config or [{}]), _DummyConfigManager())


def _potential_names(model):
    return sorted(
        p.name for p in model.potentials if p.name.startswith("galacticmodel.")
    )


def test_register_parameters_produces_empty_manifest():
    """
    Given GalacticModel (which samples nothing directly),
    When register_parameters is called,
    Then manifest is an empty dict.
    """
    gm = _make_gm()
    gm.register_parameters(system=None)
    assert gm.manifest == {}


def test_build_likelihood_adds_exactly_the_imf_and_kinematic_potentials():
    """
    Given a GalacticModel with a mock star at bulge distance,
    When build_likelihood runs inside a pm.Model,
    Then exactly two Potentials are added, 'galacticmodel.imf_prior' and
      'galacticmodel.kinematic_prior', and both are finite.

    ``galacticmodel.p_remnant`` is a Deterministic (reporting only), so it
    is not among the potentials.
    """
    # Arrange
    gm = _make_gm()

    # Act
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem())

    # Assert
    assert _potential_names(model) == [
        "galacticmodel.imf_prior",
        "galacticmodel.kinematic_prior",
    ]
    assert "galacticmodel.p_remnant" in model.named_vars
    for name in _potential_names(model):
        assert np.isfinite(float(model[name].eval())), name


def test_imf_prior_is_zero_without_a_free_floating_planet():
    """
    Given no star on the FFP mass function,
    When build_likelihood runs,
    Then the imf_prior potential is exactly 0: every star's mass prior is
      genulens's present-day mass function inside the kinematic mixture.
    """
    gm = _make_gm()
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem())
    assert float(model["galacticmodel.imf_prior"].eval()) == 0.0


def test_multiple_galacticmodel_blocks_raise():
    """
    Given two galacticmodel config blocks,
    When the component is constructed,
    Then a ValueError explains that one sight line takes one block.

    Only config[0] was ever read for anchor_idx and the options, but the
    extra blocks leaked into the likelihood through a pre-computed
    (n_blocks, 3, 3) rotation stack: with 2 blocks and 1 star the whole
    kinematic prior was broadcast to shape (2,) and therefore counted TWICE.
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
        param = getattr(one.star, attr)
        val = float(np.atleast_1d(param.initval)[0])
        setattr(
            two.star,
            attr,
            _MockParam([val, val], lower=param.lower, upper=param.upper),
        )

    # Act
    lps = []
    for system in (one, two):
        gm = _make_gm()
        with pm.Model() as model:
            gm.build_likelihood(model, system)
        lps.append(float(model["galacticmodel.kinematic_prior"].eval()))

    # Assert
    assert np.isclose(lps[1], 2.0 * lps[0], rtol=1e-12)
