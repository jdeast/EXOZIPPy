"""Unit-conversion regressions in ConfigManager.finalize_user_params.

Both bugs here only bite when a user sets a non-default `unit:` on a
parameter, and both silently corrupt the starting point rather than raising:

(a) the inject-back loop looked up the user's `unit:` override by the NAME
    form of the path (planet.b.mass) while standardize_param_names stores
    every entry under the INDEX form (planet.0.mass), so the override was
    never found and the default unit's factor was used;

(b) the relaxation engine's default-armor step converted resolve()'s initval
    -- which is already in USER units -- with no full_path at all, so it too
    always used the default unit's factor.
"""

import astropy.units as u
import numpy as np
import pytest

from exozippy.config import ConfigManager

PLANET_SYSTEM = {
    "star": [{"name": "A"}],
    "planet": [{"name": "b", "star_ndx": 0, "orbit_ndx": 0}],
    "orbit": [{"name": "b"}],
}


def test_inject_back_honors_user_unit_on_a_named_instance():
    """
    Given a user initval given in a non-default unit on a NAMED instance
    (planet.b.mass = 1.0 earthMass, defaults.yaml unit jupiterMass),
    When finalize_user_params injects the solved values back into user_params,
    Then the stored initval is still 1.0 in the user's own unit.

    Regression: the inject-back divided the internal (solMass) value by the
    jupiterMass factor because get_conversion_factor was handed the name-form
    path, which is absent from the standardized user_params.  The user's
    explicit 1.0 earthMass came back as 0.00315 earthMass -- a 318x error in
    the start that then cascades into every derived start (q, log_q, K).
    """
    # ARRANGE
    user_params = {
        "planet.b.mass": {"initval": 1.0, "unit": "earthMass"},
        "star.A.mass": {"initval": 1.0},
    }

    # ACT
    cm = ConfigManager(user_params, system_config=PLANET_SYSTEM)
    cm.finalize_user_params()

    # ASSERT -- the entry lives under the index form; unit and value must agree
    entry = cm.user_params["planet.0.mass"]
    assert entry["unit"] == "earthMass"
    assert np.isclose(entry["initval"], 1.0, rtol=1e-6), (
        f"expected 1.0 earthMass, got {entry['initval']:.6g}; "
        f"{1.0 / float(u.jupiterMass.to(u.earthMass)):.6g} means the "
        f"jupiterMass factor was used for an earthMass parameter."
    )
    # ... and the internal value the model will see is the earthMass one.
    assert np.isclose(
        cm._last_resolved["planet.0.mass"],
        float(u.earthMass.to(u.solMass)),
        rtol=1e-6,
    )


def test_inject_back_unchanged_without_a_unit_override():
    """
    Given the same setup but no `unit:` key (so the default applies),
    When finalize_user_params runs,
    Then the injected initval is the user's value in the default unit.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {
            "planet.b.mass": {"initval": 1.0},
            "star.A.mass": {"initval": 1.0},
        },
        system_config=PLANET_SYSTEM,
    )
    cm.finalize_user_params()

    # ASSERT
    assert np.isclose(cm.user_params["planet.0.mass"]["initval"], 1.0)
    assert np.isclose(
        cm._last_resolved["planet.0.mass"],
        float(u.jupiterMass.to(u.solMass)),
        rtol=1e-6,
    )


def test_default_armor_honors_user_unit():
    """
    Given a parameter with a `unit:` override and a Gaussian prior but no
    explicit initval (star.A.ra: mu 4.73 rad; defaults.yaml unit is deg),
    When the relaxation engine seeds it in its default-armor step,
    Then the internal (radian) value is 4.73, not 4.73 deg-converted.

    Regression: the armor step called get_conversion_factor with no
    full_path, so resolve()'s user-unit initval was multiplied by the
    default deg->rad factor and the start landed at 0.0826 rad.  star.ra is
    used because no physics relation rewrites it, so the armor value is
    observable in isolation.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {"star.A.ra": {"mu": 4.73, "sigma": 1e-5, "unit": "rad"}},
        system_config={"star": [{"name": "A"}]},
    )
    cm.finalize_user_params()

    # ASSERT
    internal = cm._last_resolved["star.0.ra"]
    assert np.isclose(internal, 4.73, rtol=1e-9), (
        f"expected 4.73 rad, got {internal:.6g}; "
        f"{4.73 * float(u.deg.to(u.rad)):.6g} means the deg factor was "
        f"applied to a parameter the user declared in rad."
    )


def test_default_armor_unchanged_without_a_unit_override():
    """
    Given the same prior expressed in the default unit (deg),
    When the engine seeds it,
    Then the internal value is the radian conversion of 271.0 deg.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {"star.A.ra": {"mu": 271.0, "sigma": 1e-3}},
        system_config={"star": [{"name": "A"}]},
    )
    cm.finalize_user_params()

    # ASSERT
    assert np.isclose(
        cm._last_resolved["star.0.ra"], np.radians(271.0), rtol=1e-9
    )


# ---------------------------------------------------------------------------
# Review item 3.12: an invalid `unit:` string must RAISE, not be swallowed.
#
# Before the fix both conversion sites (resolve's base->user scaling and
# get_conversion_factor's user->internal factor) wrapped the astropy call in
# `except Exception: return 1.0`.  A factor of 1.0 does not mean "no
# conversion was needed" -- it means the user's number is reinterpreted in
# whatever the internal unit happens to be.  `unit: earthMasses` (one typo)
# turned 1.0 Earth mass into 1.0 SOLAR mass, a factor of 333000, and nothing
# anywhere said so.
# ---------------------------------------------------------------------------


def test_unparseable_unit_raises_naming_the_string_and_parameter():
    """
    Given a user `unit:` string astropy cannot parse ("earthMasses"),
    When ConfigManager resolves the parameter,
    Then it raises, naming both the offending string and the parameter.

    Pre-fix: no error at all, and the internal (solMass) start value came
    out as 1.0 -- the user's 1 Earth mass read as 1 solar mass.
    """
    # ARRANGE
    cm = ConfigManager(
        {
            "planet.b.mass": {"initval": 1.0, "unit": "earthMasses"},
            "star.A.mass": {"initval": 1.0},
        },
        system_config=PLANET_SYSTEM,
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"earthMasses"):
        cm.finalize_user_params()

    with pytest.raises(ValueError, match=r"planet\.0\.mass"):
        cm.finalize_user_params()


def test_incompatible_unit_raises_naming_both_units():
    """
    Given a user `unit:` that parses but is dimensionally wrong for the
    parameter (a time unit on a mass),
    When ConfigManager resolves the parameter,
    Then it raises, naming both units.

    Pre-fix: the UnitConversionError was swallowed to factor 1.0, so
    `planet.b.mass: {initval: 1.0, unit: day}` started the chain at 1.0
    solMass.
    """
    # ARRANGE
    cm = ConfigManager(
        {
            "planet.b.mass": {"initval": 1.0, "unit": "day"},
            "star.A.mass": {"initval": 1.0},
        },
        system_config=PLANET_SYSTEM,
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"solMass"):
        cm.finalize_user_params()


def test_unit_on_a_dimensionless_parameter_raises():
    """
    Given a `unit:` on a parameter that declares no internal_unit
    (planet.e is dimensionless),
    When ConfigManager resolves it,
    Then it raises rather than silently ignoring the key.

    A unit that cannot be honored is not a default -- the user asked for a
    conversion that never happened.
    """
    # ARRANGE
    cm = ConfigManager(
        {"planet.b.e": {"initval": 0.1, "unit": "deg"}},
        system_config=PLANET_SYSTEM,
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"planet\.0\.e"):
        cm.finalize_user_params()


def test_unit_override_applies_only_to_the_element_that_declared_it():
    """
    Given a two-planet system where only planet b declares
      `unit: earthMass` (planet c means the default, jupiterMass),
    When resolve() builds the vector,
    Then planet c keeps the default unit AND the defaults.yaml bounds in
    that unit.

    Pre-fix, resolve() scanned every element's keys and stopped at the first
    `unit:` it found, then applied that unit -- and its scaling -- to the
    WHOLE vector.  planet c was relabeled earthMass while
    get_conversion_factor (which is per element) still told the relaxation
    engine jupiterMass, and planet c's hard bounds, which ARE its uniform
    prior range, came out 318x too wide.
    """
    # ARRANGE
    two_planets = {
        "star": [{"name": "A"}],
        "planet": [
            {"name": "b", "star_ndx": 0, "orbit_ndx": 0},
            {"name": "c", "star_ndx": 0, "orbit_ndx": 1},
        ],
        "orbit": [{"name": "b"}, {"name": "c"}],
    }
    cm = ConfigManager(
        {
            "planet.b.mass": {"initval": 5.0, "unit": "earthMass"},
            "planet.c.mass": {"initval": 2.0},
        },
        system_config=two_planets,
    )

    # ACT
    res = cm.resolve("planet", "mass", shape=(2,), names=["b", "c"])

    # ASSERT
    assert list(res["unit"]) == ["earthMass", "jupiterMass"]
    # planet c's bounds must still be the defaults.yaml numbers, unscaled
    assert np.isclose(res["lower"][1], -1000.0)
    assert np.isclose(res["upper"][1], 260000.0)
    # and planet b's are the same numbers converted into ITS unit
    factor = float(u.jupiterMass.to(u.earthMass))
    assert np.isclose(res["lower"][0], -1000.0 * factor)
    # the per-element factor the engine uses must agree with the label
    assert np.isclose(
        cm.get_conversion_factor("planet", "mass", "planet.1.mass"),
        float(u.jupiterMass.to(u.solMass)),
    )


def test_get_conversion_factor_raises_on_an_unparseable_user_unit():
    """
    Given a bad `unit:` string,
    When get_conversion_factor is asked for the multiplier directly,
    Then it raises instead of returning 1.0.

    get_conversion_factor is called from mkparam, the ledger and the GUI's
    solve path as well as from the engine, so the silent 1.0 leaked wrong
    numbers into places that never build a Parameter (whose own unit
    parsing is already strict).
    """
    # ARRANGE
    cm = ConfigManager(
        {"planet.b.mass": {"initval": 1.0, "unit": "earthMasses"}},
        system_config=PLANET_SYSTEM,
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match=r"earthMasses"):
        cm.get_conversion_factor("planet", "mass", "planet.0.mass")
