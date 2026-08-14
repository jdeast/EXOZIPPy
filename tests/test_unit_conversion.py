"""Unit conversion has ONE owner.

``Parameter._get_conversion_factors`` is the INTERNAL -> USER multiplier;
``ConfigManager.get_conversion_factor`` is the USER -> INTERNAL one.  They are
reciprocals with near-identical names, which is how ``outputs/ledger.py`` came
to divide where it had to multiply.  These tests pin the direction, the
round trip, and the per-element rule that ``Parameter.element_factor`` owns.
"""

import numpy as np
import pymc as pm
import pytest

from exozippy.components.parameter import Parameter
from exozippy.config import unit_conversion
from exozippy.outputs.ledger import build_seed_ledger

# jupiterMass -> solMass is the conversion every shipped RV example carries.
MJUP_PER_MSUN = float(unit_conversion("solMass", "jupiterMass", "test"))


class _StubSystem:
    """Minimal system surface build_seed_ledger reads."""

    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params


# ----------------------------------------------------------------------
# The owner: element_factor / to_internal / from_internal
# ----------------------------------------------------------------------


def test_user_to_internal_to_user_round_trips():
    """Given a parameter whose user and internal units differ,
    When a user-unit value is converted to internal units and back,
    Then the original value is recovered exactly."""
    # ARRANGE
    p = Parameter(
        label="planet.0.mass",
        unit="jupiterMass",
        internal_unit="solMass",
        initval=1.0,
    )
    user_value = 3.75

    # ACT
    internal = p.to_internal(user_value)
    back = p.from_internal(internal)

    # ASSERT
    assert float(np.ravel(internal)[0]) == pytest.approx(
        user_value / MJUP_PER_MSUN, rel=1e-12
    )
    assert float(np.ravel(back)[0]) == pytest.approx(user_value, rel=1e-12)


def test_post_init_stores_internal_units_and_from_internal_undoes_it():
    """Given a defaults-style initval written in the user unit,
    When __post_init__ converts it,
    Then the stored field is internal and from_internal returns the input."""
    # ARRANGE / ACT
    p = Parameter(
        label="planet.0.mass",
        unit="jupiterMass",
        internal_unit="solMass",
        initval=1.0,
    )

    # ASSERT
    assert float(np.ravel(p.initval)[0]) == pytest.approx(
        1.0 / MJUP_PER_MSUN, rel=1e-12
    )
    assert float(np.ravel(p.from_internal(p.initval))[0]) == pytest.approx(
        1.0, rel=1e-12
    )


def test_element_factor_is_per_element_for_per_element_units():
    """Given a vector parameter whose elements carry DIFFERENT user units,
    When element_factor is asked for each element,
    Then each gets its own factor, not element 0's."""
    # ARRANGE
    p = Parameter(
        label="planet.mass",
        unit=["jupiterMass", "earthMass"],
        internal_unit="solMass",
        initval=[1.0, 1.0],
    )

    # ACT
    f0 = p.element_factor(0)
    f1 = p.element_factor(1)

    # ASSERT
    assert f0 == pytest.approx(MJUP_PER_MSUN, rel=1e-12)
    assert f1 == pytest.approx(
        float(unit_conversion("solMass", "earthMass", "test")), rel=1e-12
    )
    assert f1 > f0


def test_element_factor_falls_back_to_element_zero():
    """Given the ordinary scalar `unit:` (one factor for a whole vector),
    When element_factor is asked for a high index,
    Then it returns element 0's factor rather than raising."""
    # ARRANGE
    p = Parameter(
        label="planet.mass",
        unit="jupiterMass",
        internal_unit="solMass",
        initval=[1.0, 2.0, 3.0],
    )

    # ACT / ASSERT
    assert p.element_factor(2) == pytest.approx(p.element_factor(0))


def test_indexed_conversion_of_a_scalar_stays_a_scalar():
    """Given a parameter with per-element units,
    When a single value is converted with index=,
    Then the result is that element's scalar -- not a whole vector.

    The whole-vector call returns an n-element array for a scalar input,
    which is what made run.py's startup table raise 'can only convert an
    array of size 1' the moment a config used a per-instance `unit:`.
    """
    # ARRANGE
    p = Parameter(
        label="planet.mass",
        unit=["jupiterMass", "earthMass"],
        internal_unit="solMass",
        initval=[1.0, 1.0],
    )
    internal = float(np.ravel(p.initval)[1])

    # ACT
    out = p.from_internal(internal, index=1)

    # ASSERT
    assert np.size(out) == 1
    assert float(out) == pytest.approx(1.0, rel=1e-12)
    assert np.size(p.from_internal(internal)) == 2  # the un-indexed form


def test_to_internal_and_from_internal_are_inverses_per_element():
    """Given per-element units,
    When a value round-trips through to_internal/from_internal at an index,
    Then it is recovered, for every element."""
    # ARRANGE
    p = Parameter(
        label="planet.mass",
        unit=["jupiterMass", "earthMass"],
        internal_unit="solMass",
        initval=[1.0, 1.0],
    )

    # ACT / ASSERT
    for i, value in enumerate([2.5, 17.0]):
        internal = p.to_internal(value, index=i)
        assert float(p.from_internal(internal, index=i)) == pytest.approx(
            value, rel=1e-12
        )


def test_mismatched_vector_lengths_raise_rather_than_broadcast():
    """Given per-element factors and a value vector of a different length,
    When they are converted together,
    Then it raises instead of silently mixing elements."""
    # ARRANGE
    p = Parameter(
        label="planet.mass",
        unit=["jupiterMass", "earthMass"],
        internal_unit="solMass",
        initval=[1.0, 1.0],
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match="unit conversion factors"):
        p.from_internal(np.array([1.0, 2.0, 3.0]))


# ----------------------------------------------------------------------
# The direction contract the ledger bug violated
# ----------------------------------------------------------------------


def test_parameter_factor_is_the_reciprocal_of_the_config_factor():
    """Given the same (user, internal) unit pair,
    When both conversion helpers are asked for a factor,
    Then Parameter's is internal->user and config's is user->internal.

    This is the whole reason the two must never be swapped: `* factor`
    means opposite things in parameter.py and config.py.
    """
    # ARRANGE
    p = Parameter(
        label="orbit.0.omega", unit="deg", internal_unit="rad", initval=90.0
    )

    # ACT
    param_factor = float(
        np.ravel(np.asarray(p._get_conversion_factors(), dtype=float))[0]
    )
    config_factor = float(unit_conversion("deg", "rad", "test"))

    # ASSERT
    assert param_factor == pytest.approx(1.0 / config_factor, rel=1e-12)
    # and the stored initval used the config direction
    assert float(np.ravel(p.initval)[0]) == pytest.approx(
        np.deg2rad(90.0), rel=1e-12
    )


def test_seed_ledger_reports_user_units_not_reciprocal_ones():
    """Given a seeded solution for a parameter with a real unit conversion,
    When the ledger records its physical center,
    Then the number is in USER units -- the same one from_internal gives.

    Regression: the ledger DIVIDED by the internal->user factor, so every
    converted parameter in <prefix>_results.csv, the rejected-seeds text and
    the rejected-modes LaTeX table was wrong by factor**2 (planet.mass in
    examples/hd80606: 1.45e-06 jupiterMass reported for a start of 1.596).
    """
    # ARRANGE: an angle in the units every orbit uses -- deg out, rad in.
    p = Parameter(
        label="orbit.omega",
        unit="deg",
        internal_unit="rad",
        initval=90.0,
        lower=0.0,
        upper=360.0,
    )
    with pm.Model() as model:
        node = p.build_pymc()
        pm.Potential("like", -0.5 * ((node - np.deg2rad(90.0)) / 0.01) ** 2)

    raw0 = {"orbit.omega_raw": np.zeros_like(np.atleast_1d(p.raw_initval))}

    # ACT
    ledger = build_seed_ledger(_StubSystem([p]), model, [raw0], [0])
    reported = float(np.ravel(ledger[0].phys["orbit.omega"])[0])

    # ASSERT
    assert reported == pytest.approx(90.0, rel=1e-3)
    # and specifically NOT the reciprocal-direction answer
    assert reported != pytest.approx(np.deg2rad(90.0) / (180.0 / np.pi))
