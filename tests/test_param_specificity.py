"""The MOST SPECIFIC spelling of a parameter wins -- for every field.

``ConfigManager.resolve()`` accepts three spellings of one element: the 2-part
broadcast ``star.teff``, the index form ``star.0.teff`` and the name form
``star.A.teff``.  ``_element_keys`` is the one place their order lives, and
five call sites consume it -- but they used to consume it in two opposite
ways.  The loops that APPLY EVERY MATCH (component overrides, user params)
let the most specific entry land last and win; the lookups that stop at the
FIRST hit (the ``unit:`` scan, the propagated-scales scan, the scale-hints
scan) took the list as written and so picked the BROADCAST.  One config could
therefore have ``star.teff: {unit: K}`` beat ``star.0.teff: {unit: ...}`` for
the unit while losing to it for every numeric field.

The ruling: most specific wins everywhere.  ``_lookup_keys`` rotates the
broadcast key to the end for the first-hit lookups, so both traversals agree.
That is not cosmetic for the two scale channels: ``init_scale`` seeds the
whitening probe, and for an unbounded element with no sigma it IS the prior
width, i.e. a posterior term.

Explicitly out of scope, and pinned below so it stays that way: the index and
name forms name exactly ONE element each and are equally specific, so which of
those two wins is left exactly as it was (the index form in the first-hit
lookups, the name form in the apply-every-match loops).
"""

import numpy as np
import pytest

from exozippy.config import ConfigManager

SYSTEM = {
    "star": [{"name": "A"}, {"name": "B"}],
    "planet": [{"name": "b", "star_ndx": 0, "orbit_ndx": 0}],
    "orbit": [{"name": "b"}],
}


def _canonical_cm(user_params):
    """A manager whose user params are taken verbatim, both spellings intact.

    ``standardize_param_names`` resolves broadcast-vs-specific itself for a
    LIST component (it expands a 2-part key only into the indices no 3-part
    key claimed), so this constructor mode -- params already in canonical
    form -- is how both spellings are presented to one element in a unit
    test.  The collision is reachable in the wild through the channels the
    standardizer does not touch: a flat-dict component's params (``sed.x``
    and ``sed.0.x`` both survive pass 1/2 verbatim), ``add_override`` and
    ``add_scale_hint`` paths (stored as the component spelled them), and the
    relaxation engine's own ledger, which registers unmapped 2-part keys as
    leaf symbols and can hand them back through ``propagated_scales``.
    """
    return ConfigManager(user_params, system_config=None)


# ---------------------------------------------------------------------------
# 1. the `unit:` scan  (break-on-first-hit; CHANGED)
# ---------------------------------------------------------------------------


def test_specific_unit_beats_broadcast_unit():
    """
    Given a broadcast `planet.mass: {unit: earthMass}` and a specific
      `planet.0.mass: {unit: jupiterMass}` for the same element,
    When resolve() picks the element's unit,
    Then the specific spelling wins and its scaling is what the numbers carry.

    Pre-fix the scan broke on the first candidate, which is the broadcast, so
    every defaults.yaml number for this element was rescaled by 318x while
    the user's own initval -- applied by the loop that lets the most specific
    entry land last -- was read in jupiterMass.  The two halves of one entry
    disagreed about what unit the parameter was in.
    """
    # ARRANGE
    cm = _canonical_cm(
        {
            "planet.mass": {"unit": "earthMass"},
            "planet.0.mass": {"unit": "jupiterMass"},
        }
    )

    # ACT
    cfg = cm.resolve("planet", "mass", shape=(1,))

    # ASSERT
    assert cfg["unit"] == "jupiterMass"
    # defaults.yaml lower is -1000 in the parameter's own unit; an earthMass
    # reading would have scaled it by ~318.
    assert np.isclose(cfg["lower"][0], -1000.0)


def test_broadcast_unit_still_covers_elements_with_no_specific_entry():
    """
    Given a broadcast `unit: km` and a specific `unit: pc` on element 0 only,
    When resolve() scales the defaults.yaml numbers per element,
    Then element 0 is scaled in pc and element 1 in km.

    Demoting the broadcast key is a precedence change, not a removal: it must
    still cover every element no more specific entry claims.  The assertion is
    on the SCALING rather than the reported `unit` string because the string
    is rewritten by the user-params loop further down (which always preferred
    the specific entry); the scan is what sets the numbers.
    """
    # ARRANGE
    km_per_pc = 3.0856775814913673e13
    cm = _canonical_cm(
        {
            "star.distance": {"unit": "km"},
            "star.0.distance": {"unit": "pc"},
        }
    )

    # ACT
    cfg = cm.resolve("star", "distance", shape=(2,))

    # ASSERT -- defaults.yaml lower is 0.001 pc.
    assert np.isclose(cfg["lower"][0], 0.001)
    assert np.isclose(cfg["lower"][1], 0.001 * km_per_pc, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. the scale-hints scan  (break-on-first-hit; CHANGED)
# ---------------------------------------------------------------------------


def test_specific_scale_hint_beats_broadcast_scale_hint():
    """
    Given a component pushing a broadcast scale hint and a per-element one on
      the same parameter,
    When resolve() reads init_scale,
    Then element 0 gets the specific hint and element 1 the broadcast hint.

    init_scale is not cosmetic: it seeds the whitening probe, and for an
    unbounded element with no sigma it is the prior width itself.
    """
    # ARRANGE -- the real channel, no dict poking: add_scale_hint keeps a
    # 2-part path 2-part and translates a 3-part one to the index form.
    cm = ConfigManager({}, system_config=SYSTEM)
    cm.add_scale_hint("star.distance", 100.0)
    cm.add_scale_hint("star.A.distance", 5.0)

    # ACT
    cfg = cm.resolve("star", "distance", shape=(2,))

    # ASSERT
    assert np.isclose(cfg["init_scale"][0], 5.0)
    assert np.isclose(cfg["init_scale"][1], 100.0)


# ---------------------------------------------------------------------------
# 3. the propagated-scales scan  (break-on-first-hit; CHANGED)
# ---------------------------------------------------------------------------


def test_specific_propagated_scale_beats_broadcast_propagated_scale():
    """
    Given the last relaxation solve leaving both a broadcast and an
      index-form init_scale for one parameter,
    When resolve() reads init_scale,
    Then element 0 takes the index-form scale and element 1 the broadcast one.

    propagated_scales is keyed by whatever path the engine's ledger carries,
    and it carries 2-part rows for every unmapped user_params key it turned
    into a leaf symbol -- so the two tiers really do coexist there.
    """
    # ARRANGE -- stored in INTERNAL units; distance is pc -> pc, factor 1.
    cm = ConfigManager({}, system_config=SYSTEM)
    factor = cm.get_conversion_factor("star", "distance")
    cm.propagated_scales = {
        "star.distance": 100.0 * factor,
        "star.0.distance": 5.0 * factor,
    }

    # ACT
    cfg = cm.resolve("star", "distance", shape=(2,))

    # ASSERT
    assert np.isclose(cfg["init_scale"][0], 5.0)
    assert np.isclose(cfg["init_scale"][1], 100.0)


# ---------------------------------------------------------------------------
# 4 and 5. the apply-every-match loops  (UNCHANGED -- pinned)
# ---------------------------------------------------------------------------


def test_specific_user_param_beats_broadcast_user_param():
    """
    Given a broadcast and a specific user initval for one element,
    When resolve() applies the user's params,
    Then the specific value wins.

    This half of the rule always held (the loop applies every match in order
    and the most specific lands last); it is pinned here so the two halves
    are stated in one place.
    """
    # ARRANGE
    cm = _canonical_cm(
        {
            "star.distance": {"initval": 100.0},
            "star.0.distance": {"initval": 5.0},
        }
    )

    # ACT
    cfg = cm.resolve("star", "distance", shape=(2,))

    # ASSERT
    assert np.isclose(cfg["initval"][0], 5.0)
    assert np.isclose(cfg["initval"][1], 100.0)


def test_specific_component_override_beats_broadcast_component_override():
    """
    Given a component registering a broadcast override and a per-element one
      through ConfigManager.add_override,
    When resolve() layers them in,
    Then the specific one wins for element 0 and the broadcast covers the rest.

    initval rather than a bound, deliberately: apply_value combines competing
    bounds as max/min order-independently, so only a plain field can see the
    ordering at all.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=SYSTEM)
    cm.add_override("star.av", initval=1.0)
    cm.add_override("star.0.av", initval=2.0)

    # ACT
    cfg = cm.resolve("star", "av", shape=(2,))

    # ASSERT
    assert np.isclose(cfg["initval"][0], 2.0)
    assert np.isclose(cfg["initval"][1], 1.0)


def test_user_params_still_beat_a_component_override():
    """
    Given a component override and a user entry on the SAME element,
    When resolve() runs,
    Then the user's value wins.

    Demoting the broadcast key must not disturb the tier order between the
    channels themselves: overrides are layered under the params file.
    """
    # ARRANGE
    cm = ConfigManager({"star.A.av": {"initval": 7.0}}, system_config=SYSTEM)
    cm.add_override("star.av", initval=1.0)

    # ACT
    cfg = cm.resolve("star", "av", shape=(2,))

    # ASSERT
    assert np.isclose(cfg["initval"][0], 7.0)
    assert np.isclose(cfg["initval"][1], 1.0)


# ---------------------------------------------------------------------------
# Scope guard: the index/name tier is deliberately untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,expected_winner",
    [("unit", "index"), ("initval", "name")],
)
def test_index_vs_name_order_is_unchanged(field, expected_winner):
    """
    Given BOTH equally specific spellings of element 0 (index and name),
    When resolve() reads a first-hit field (unit) and an apply-every-match
      field (initval),
    Then the index form wins the first, the name form wins the second.

    Both spellings name one element, so neither is more specific and the
    winner is arbitrary either way; the two channels happen to disagree.
    Changing that would be a second, unasked-for semantic change, so the fix
    reorders only the broadcast tier and this pins the rest.
    """
    # ARRANGE
    cm = _canonical_cm(
        {
            "star.0.distance": {"unit": "pc", "initval": 5.0},
            "star.A.distance": {"unit": "km", "initval": 9.0},
        }
    )

    # ACT
    cfg = cm.resolve("star", "distance", shape=(1,), names=["A"])

    # ASSERT
    if field == "unit":
        # The SCAN (which scales the defaults) took the index form: 0.001 pc
        # unscaled.  The reported `unit` string is written by the params loop
        # instead and so still says km -- the one place the two halves of an
        # entry disagree, and deliberately left alone.
        assert np.isclose(cfg["lower"][0], 0.001)
        assert cfg["unit"] == "km"
    else:
        # 9.0 is the number the name-form entry supplied; the point is which
        # entry supplied it, not its unit.
        assert np.isclose(cfg["initval"][0], 9.0)
