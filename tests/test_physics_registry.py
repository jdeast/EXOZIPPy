"""PHYSICS_REGISTRY is a flat namespace -- names must not collide.

Regression cover for a silent shadowing bug: star and planet both defined
`calc_logg`, with different signatures (logmass vs linear mass). The registry
is keyed by bare function name, so whichever imported last won -- planet's --
and star.logg was computed as LOGG_CONST + log10(logmass) - 2*log10(radius).
That is wrong for every star and NaN for any star below 1 solMass (logmass <
0). Nothing consumed star.logg, so it survived unnoticed until
components/torres needed it.
"""

import numpy as np
import pytensor.tensor as pt
import pytest

from exozippy.constants import LOGG_CONST
from exozippy.physics_registry import PHYSICS_REGISTRY, register_physics
from exozippy.components.factory import discover_components


def test_no_duplicate_physics_names_across_components():
    """
    Given every component's physics module is imported,
    When the registry is populated,
    Then no name was registered twice (the flat namespace has one owner each).
    """
    # Arrange / Act -- importing every component module is what would collide
    discover_components()

    # Assert -- register_physics raises on a real duplicate, so reaching here
    # means the namespace is clean; keep an explicit check on the two that did.
    assert "calc_logg" not in PHYSICS_REGISTRY, (
        "'calc_logg' is ambiguous: star's takes logmass, planet's takes linear "
        "mass. Use calc_logg_from_logmass / calc_logg_from_mass."
    )
    assert "calc_logg_from_logmass" in PHYSICS_REGISTRY
    assert "calc_logg_from_mass" in PHYSICS_REGISTRY
    assert "calc_density" in PHYSICS_REGISTRY


def test_register_physics_rejects_a_shadowing_duplicate():
    """
    Given a name already registered by another module,
    When a different function registers under it,
    Then it raises rather than silently shadowing.
    """
    # Arrange
    def _make():
        @register_physics
        def calc_density(mass, radius):   # noqa: F811 - deliberate collision
            return mass

    # Act / Assert
    with pytest.raises(ValueError, match="Duplicate physics function"):
        _make()


def test_star_logg_uses_logmass_not_linear_mass():
    """
    Given a 0.7 solMass star (logmass < 0, as any M dwarf has),
    When star's logg relation is evaluated through the registry,
    Then it returns the correct logg -- not NaN from log10 of a negative.
    """
    # Arrange -- 0.7 solMass, 0.7 solRad
    logmass, radius = np.log10(0.7), 0.7
    fn = PHYSICS_REGISTRY["calc_logg_from_logmass"]
    lm, r = pt.dscalar("logmass"), pt.dscalar("radius")

    # Act
    got = float(fn(lm, r).eval({lm: logmass, r: radius}))

    # Assert
    expected = LOGG_CONST + logmass - 2.0 * np.log10(radius)
    assert np.isfinite(got)
    assert got == pytest.approx(expected, rel=1e-12)


def test_planet_logg_uses_linear_mass():
    """
    Given a planet mass in solar masses,
    When planet's logg relation is evaluated through the registry,
    Then it logs the mass itself.
    """
    # Arrange
    mass, radius = 0.001, 0.1
    fn = PHYSICS_REGISTRY["calc_logg_from_mass"]
    m, r = pt.dscalar("mass"), pt.dscalar("radius")

    # Act
    got = float(fn(m, r).eval({m: mass, r: radius}))

    # Assert
    expected = LOGG_CONST + np.log10(mass) - 2.0 * np.log10(radius)
    assert got == pytest.approx(expected, rel=1e-12)


def test_the_two_logg_relations_are_not_interchangeable():
    """
    Given the two relations take different inputs,
    When both are handed the same number,
    Then they disagree -- which is why sharing one registry name corrupted
    star.logg.
    """
    # Arrange
    x, r = pt.dscalar("x"), pt.dscalar("r")
    point = {x: 0.5, r: 1.2}

    # Act
    a = float(PHYSICS_REGISTRY["calc_logg_from_logmass"](x, r).eval(point))
    b = float(PHYSICS_REGISTRY["calc_logg_from_mass"](x, r).eval(point))

    # Assert
    assert a != pytest.approx(b, rel=1e-6)
