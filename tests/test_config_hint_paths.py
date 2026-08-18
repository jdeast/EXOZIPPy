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
