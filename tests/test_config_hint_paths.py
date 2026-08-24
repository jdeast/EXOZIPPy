"""The component-hint channels must all translate paths and units identically.

``add_hint``, ``add_scale_hint``, ``add_seed_hints`` and ``seed_start_value``
each turn a component-supplied path into the index form ``resolve()`` looks up
and each scale their number into internal units.  ``_translate_and_scale``
exists so there is exactly one implementation of that; ``add_scale_hint``
nevertheless carried a line-for-line COPY of the body until 2026-08, which is
the kind of duplication that stays correct right up until one copy is fixed.

A divergence would be silent in both directions: a scale hint filed under the
untranslated ``star.Lens.distance`` is simply never found by ``resolve()``
(the component's carefully chosen bulge scale vanishes and the generic 0.1 pc
default stands), and a scale hint left in degrees on a parameter whose
internal unit is radians is wrong by 57x with nothing to flag it.
"""

import numpy as np

from exozippy.config import ConfigManager

# planet.mass is jupiterMass -> solMass and star.ra is deg -> rad, so both a
# named-instance translation and a real unit conversion are exercised.
SYSTEM = {
    "star": [{"name": "A"}, {"name": "B"}],
    "planet": [{"name": "b", "star_ndx": 0, "orbit_ndx": 0}],
    "orbit": [{"name": "b"}],
}


def _cm(user_params=None):
    return ConfigManager(user_params or {}, system_config=SYSTEM)


def test_hint_and_scale_hint_agree_on_a_named_instance():
    """
    Given a component hinting the same named-instance path through the value
      channel and the scale channel,
    When both are pushed,
    Then both land under the same index-form key with the same internal value.
    """
    # ARRANGE
    cm = _cm()

    # ACT
    cm.add_hint("planet.b.mass", 2.0)
    cm.add_scale_hint("planet.b.mass", 2.0)

    # ASSERT
    assert set(cm.hints) == {"planet.0.mass"}
    assert set(cm.scale_hints) == set(cm.hints)
    assert cm.scale_hints["planet.0.mass"] == cm.hints["planet.0.mass"]
    # ... and the shared value really is in internal (solMass) units.
    assert cm.hints["planet.0.mass"] < 0.01


def test_hint_and_scale_hint_agree_on_an_angle_conversion():
    """
    Given a path whose defaults.yaml unit (deg) differs from its internal
      unit (rad),
    When a value hint and a scale hint are pushed for the second star,
    Then both are converted to radians under the index-form key.

    A copy of the translation that forgot the conversion would leave a scale
    of 90 rad on a parameter bounded in radians -- large enough to swamp the
    whitening probe, small enough to look plausible in a log.
    """
    # ARRANGE
    cm = _cm()

    # ACT
    cm.add_hint("star.B.ra", 90.0)
    cm.add_scale_hint("star.B.ra", 90.0)

    # ASSERT
    assert np.isclose(cm.hints["star.1.ra"], np.pi / 2)
    assert cm.scale_hints["star.1.ra"] == cm.hints["star.1.ra"]


def test_scale_hint_honors_a_user_unit_override():
    """
    Given the user overriding the unit of an index-form path,
    When a component pushes a scale hint on that same path,
    Then the hint is converted with the USER's unit, exactly as a value hint is.

    This is the case a divergence would have mattered most in: the two
    channels read the same parameter with two different factors, so the start
    value and the scale that seeds the whitening probe around it would
    disagree by 318x with no error anywhere.
    """
    # ARRANGE
    cm = _cm({"planet.0.mass": {"initval": 1.0, "unit": "earthMass"}})

    # ACT
    cm.add_hint("planet.0.mass", 1.0)
    cm.add_scale_hint("planet.0.mass", 1.0)

    # ASSERT -- one earthMass in solMass, through both channels
    assert np.isclose(cm.hints["planet.0.mass"], 3.0035e-06, rtol=1e-3)
    assert cm.scale_hints["planet.0.mass"] == cm.hints["planet.0.mass"]


def test_untranslatable_paths_are_stored_verbatim():
    """
    Given a 2-part broadcast path and a 3-part path naming no known instance,
    When they are pushed through both hint channels,
    Then both channels store them under the path exactly as given.

    The broadcast form is how a component says "every element"; resolve()
    checks it as its first candidate key, so rewriting it would silently
    demote a system-wide hint to element 0 only.
    """
    # ARRANGE
    cm = _cm()

    # ACT
    cm.add_hint("star.distance", 8000.0)
    cm.add_scale_hint("star.distance", 500.0)
    cm.add_hint("star.Nobody.distance", 8000.0)
    cm.add_scale_hint("star.Nobody.distance", 500.0)

    # ASSERT
    assert set(cm.hints) == {"star.distance", "star.Nobody.distance"}
    assert set(cm.scale_hints) == set(cm.hints)


def test_seed_hints_and_scale_hints_share_the_translation():
    """
    Given per-seed observables and a scale hint on the same named path,
    When both are registered,
    Then the seed set is keyed by the same index-form path in the same units,
      and seed_start_value round-trips back to the user's number.
    """
    # ARRANGE
    cm = _cm()

    # ACT
    cm.add_seed_hints([{"star.B.ra": 90.0}])
    cm.add_scale_hint("star.B.ra", 90.0)

    # ASSERT
    assert set(cm.seed_hint_sets[0]) == {"star.1.ra"}
    assert cm.seed_hint_sets[0]["star.1.ra"] == cm.scale_hints["star.1.ra"]
    assert np.isclose(cm.seed_start_value("star.B.ra"), 90.0)


# ---------------------------------------------------------------------------
# The OTHER half of the round trip: resolve() reading a stored scale back out
# (review 1.14.1).
#
# add_scale_hint stores with the ELEMENT's user -> internal factor (the tests
# above pin that).  resolve() has to divide by the SAME factor to hand
# Parameter a number in the element's user unit, because
# Parameter._get_conversion_factors re-applies it.  It used to divide by
# get_conversion_factor(component_type, param_name) with no full_path -- the
# DEFAULTS-unit factor -- so the two halves disagreed by exactly the unit
# override whenever there was one: 318x for a mass relabelled earthMass, 57x
# for a deg -> rad angle.  Silent, and not cosmetic: init_scale seeds the
# whitening probe, and for an unbounded element with no sigma it IS the prior
# width.
# ---------------------------------------------------------------------------


def test_a_scale_hint_round_trips_through_a_user_unit_override():
    """
    Given a user `unit: earthMass` on planet.0.mass (defaults jupiterMass),
    When a component pushes a 0.5-earthMass scale hint and resolve() reads it
      back,
    Then init_scale comes back as 0.5 in the element's own unit.

    Regression (1.14.1): resolve() divided by the defaults-unit factor alone,
    returning 0.00157 -- the same number 318x too small, in a field nothing
    ever cross-checks.
    """
    # ARRANGE
    cm = _cm({"planet.0.mass": {"initval": 1.0, "unit": "earthMass"}})

    # ACT
    cm.add_scale_hint("planet.0.mass", 0.5)
    resolved = cm.resolve("planet", "mass")

    # ASSERT
    assert resolved["unit"] == "earthMass"
    assert np.isclose(resolved["init_scale"][0], 0.5)


def test_a_scale_hint_round_trips_through_a_user_angle_unit():
    """
    Given a user `unit: arcsec` on star.1.ra (defaults deg, internal rad),
    When a component pushes a 3600-arcsec scale hint and resolve() reads it
      back,
    Then init_scale is 3600 arcsec, not 1 degree's worth of some other unit.

    The angle case is the second reachable spelling of the same defect, and it
    exercises a THREE-unit chain (arcsec -> deg -> rad) rather than the mass
    case's two.
    """
    # ARRANGE
    cm = _cm({"star.1.ra": {"initval": 10.0, "unit": "arcsec"}})

    # ACT
    cm.add_scale_hint("star.1.ra", 3600.0)
    resolved = cm.resolve("star", "ra", shape=(2,))

    # ASSERT
    assert np.isclose(resolved["init_scale"][1], 3600.0)


def test_a_propagated_scale_round_trips_through_a_user_unit_override():
    """
    Given a previous solve's propagated scale, stored in INTERNAL units,
    When the element carries a user `unit:` override and resolve() reads it
      back,
    Then it comes back in that user unit.

    propagated_scales is the second of the two loops 1.14.1 names.  It is
    written by the relaxation engine rather than by add_scale_hint, but it is
    read back through the identical arithmetic and had the identical bug.
    """
    # ARRANGE: one earthMass, expressed in internal solMass.
    cm = _cm({"planet.0.mass": {"initval": 1.0, "unit": "earthMass"}})
    internal = cm.get_conversion_factor(
        "planet", "mass", full_path="planet.0.mass"
    )
    cm.propagated_scales = {"planet.0.mass": 2.0 * internal}

    # ACT
    resolved = cm.resolve("planet", "mass")

    # ASSERT
    assert np.isclose(resolved["init_scale"][0], 2.0)


def test_no_unit_override_leaves_both_scale_channels_untouched():
    """
    Given no user `unit:` anywhere -- which is every shipped example,
    When a scale hint and a propagated scale are read back,
    Then both are unchanged by the 1.14.1 fix: elem_scaling is exactly 1.0, so
      the new divisor is the old one.

    This is the "nothing should move" half of the fix, asserted rather than
    assumed.
    """
    # ARRANGE
    cm = _cm()
    jup_to_sol = cm.get_conversion_factor("planet", "mass")
    cm.add_scale_hint("planet.0.mass", 0.25)
    cm.propagated_scales = {
        "star.0.ra": 4.0 * cm.get_conversion_factor("star", "ra")
    }

    # ACT
    mass = cm.resolve("planet", "mass")
    ra = cm.resolve("star", "ra", shape=(2,))

    # ASSERT -- the defaults units, and the defaults-unit factor
    assert mass["unit"] == "jupiterMass"
    assert np.isclose(mass["init_scale"][0], 0.25)
    assert np.isclose(cm.scale_hints["planet.0.mass"], 0.25 * jup_to_sol)
    assert np.isclose(ra["init_scale"][0], 4.0)
