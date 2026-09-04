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

BOUNDS WERE THE ONE SILENT EXCEPTION, closed by review 1.1.5.  ``apply_value``
resolves ``lower`` with max() and ``upper`` with min() rather than assigning,
which is order-independent and so defeated ``_element_keys`` entirely: a
broadcast ``sed.av: {lower: 1.0}`` beat a specific ``sed.0.av: {lower: 0.5}``
and nothing was logged.  The ruling is that a specific entry is the EXCEPTION
to a broadcast, not an addition to it, so the user-vs-user contest is now
settled by specificity BEFORE the strictest-wins clip runs.  The clip itself
is untouched and still governs user-vs-defaults, which is the case it was
written for.

AND THE BROADCAST IS EXPANDED PER FIELD, NOT PER KEY (review 1.1.6).
``standardize_param_names`` Pass 2 used to skip any index a Pass-1 key had
already claimed, which made "explicit beats broadcast" a claim about whole
ENTRIES rather than about the fields they set: a specific entry mentioning
some OTHER field silently cancelled the broadcast for that element and it
fell back to defaults.  Setting a BOUND on one star discarded that star's
broadcast START VALUE.  The broadcast now supplies the base entry and the
specific one overrides it field by field.

AND THE BROADCAST FORM IS USER-FACING ONLY (review 2.14.8, JDE's ruling
2026-09-04).  It is a convenience for writing a params file: standardization
expands it into one key per element at construction, and nothing downstream
sees it again.  So the INTERNAL channels -- ``add_hint``, ``add_scale_hint``,
``add_seed_hints``, ``seed_start_value`` -- refuse it outright rather than
expanding it, because one scalar cannot answer for every element once the
elements may carry different ``unit:`` overrides (which review 1.1.5 made
possible).  A component knows which element it means; it says so.

The index and name forms name exactly ONE element each and are equally
specific, so "most specific wins" cannot adjudicate between them -- and the
two traversals disagreed about it (index in the first-hit lookups, name in
the apply-every-match loops).  The ruling is that such a config is
ill-formed and RAISES: one element, one spelling.  The tests that used to
pin which of the two won now pin the raise instead; they were pinning an
arbitrary winner, and the winner no longer exists.
"""

import numpy as np
import pytest

from exozippy.config import ConfigManager, _reject_duplicate_spellings

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
    Given a broadcast scale hint and a per-element one on the same parameter,
    When resolve() reads init_scale,
    Then element 0 gets the specific hint and element 1 the broadcast hint.

    init_scale is not cosmetic: it seeds the whitening probe, and for an
    unbounded element with no sigma it is the prior width itself.

    BUILT WITHOUT A system_config, since review 2.14.8.  This test used to
    push the broadcast through `add_scale_hint` on a full SYSTEM, which is
    now refused outright -- a component must say which element it means.
    The precedence being pinned here is still live, because a 2-part scale
    hint still ARRIVES by the routes the guard cannot see: this constructor
    mode, and the relaxation engine's own ledger, which registers unmapped
    2-part keys as leaf symbols and hands them back through
    `propagated_scales`.  So the rule is unchanged and only the door the
    hint comes through has narrowed.
    """
    # ARRANGE
    cm = _canonical_cm({})
    cm.add_scale_hint("star.distance", 100.0)
    cm.add_scale_hint("star.0.distance", 5.0)

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
    ordering at all.  That is still true of THIS channel after review 1.1.5 --
    an override bound is a validity limit and strictest-wins is the point of
    it (see add_override's docstring).  It is the USER channel that changed;
    the pair of tests below pin the difference.
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
# One element, one spelling: index + name RAISES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,entries",
    [
        ("initval", {"initval": 5.0}),
        ("unit", {"unit": "pc"}),
    ],
)
def test_index_and_name_spellings_of_one_element_raise(field, entries):
    """
    Given BOTH equally specific spellings of star 0 (index and name),
    When a ConfigManager is constructed on them,
    Then it raises, naming both keys.

    This replaces two assertions that pinned WHICH spelling won -- the index
    form for the first-hit fields, the name form for the apply-every-match
    ones.  Neither is more specific, so both answers were arbitrary and the
    config could carry a number scaled under one entry and labelled under the
    other.  Telling the user beats picking for them.  Parameterized over a
    numeric field and `unit:` because they travel through the two different
    traversals.
    """
    # ARRANGE
    user_params = {
        "star.0.distance": dict(entries),
        "star.A.distance": dict(entries),
    }

    # ACT / ASSERT
    with pytest.raises(ValueError) as exc:
        ConfigManager(user_params, system_config=SYSTEM)

    msg = str(exc.value)
    assert "star.0.distance" in msg and "star.A.distance" in msg
    assert "same" in msg.lower()


def test_identical_content_still_raises():
    """
    Given two spellings of one element carrying byte-identical entries,
    When a ConfigManager is constructed on them,
    Then it still raises.

    "They happen to agree" would be a second rule to maintain, and the config
    is no less confusing to read for it: one element is still addressed
    twice, and standardize_param_names would still keep only one of the two.
    No shipped example or params file does this (all 19 checked).
    """
    # ARRANGE
    entry = {"initval": 5.0, "sigma": 0.5, "mu": 5.0}

    # ACT / ASSERT
    with pytest.raises(ValueError, match="DUPLICATE PARAMETER SPELLING"):
        ConfigManager(
            {"star.0.distance": dict(entry), "star.A.distance": dict(entry)},
            system_config=SYSTEM,
        )


@pytest.mark.parametrize("specific", ["star.0.distance", "star.A.distance"])
def test_broadcast_plus_specific_never_raises(specific):
    """
    Given the broadcast spelling AND one specific spelling of one element,
    When a ConfigManager is constructed and resolve() runs,
    Then nothing raises and the specific entry wins that element while the
      broadcast covers the other.

    The whole point of the boundary: broadcast + specific is well defined
    ("most specific wins") and is a legitimate, shipped idiom -- SED's
    add_override channel and seven example params files rely on it.  Breaking
    it would be a serious regression, so it is pinned for BOTH specific
    spellings.
    """
    # ARRANGE
    cm = ConfigManager(
        {"star.distance": {"initval": 100.0}, specific: {"initval": 5.0}},
        system_config=SYSTEM,
    )

    # ACT
    cfg = cm.resolve("star", "distance", shape=(2,), names=["A", "B"])

    # ASSERT
    assert np.isclose(cfg["initval"][0], 5.0)
    assert np.isclose(cfg["initval"][1], 100.0)


def test_a_name_that_is_the_index_string_is_not_a_duplicate():
    """
    Given a component instance literally named "0" at index 0,
    When the duplicate-spelling check reads a `star.0.teff` entry,
    Then it does not raise: the two spellings are the SAME string, so there
      is only one entry and nothing to adjudicate.

    The function is called directly because validate_instance_names refuses
    an all-digit name at ConfigManager construction, one line earlier -- the
    degenerate case is unreachable from a real config, and this pins that the
    check does not depend on that other guard to stay correct.
    """
    # ARRANGE
    config = {"star": [{"name": "0"}]}

    # ACT / ASSERT -- no exception
    _reject_duplicate_spellings({"star.0.teff": {"initval": 5000.0}}, config)


def test_component_supplied_names_also_raise_at_resolve():
    """
    Given a component whose per-element `names` are NOT its own config
      instances' names (the lens's per-source vectors), and a user naming one
      element by both index and borrowed name,
    When resolve() runs for that parameter,
    Then it raises, naming both keys.

    standardize_param_names cannot fold this pair: it only knows config
    instance names, so `lens.SourceA.t_0` survives verbatim and BOTH keys
    reach resolve() as live entries.  That is the only path on which the
    unit-label-vs-scaling desync was ever reachable, so the raise has to
    cover it or the desync survives.
    """
    # ARRANGE
    cm = _canonical_cm(
        {
            "star.0.distance": {"unit": "pc", "initval": 5.0},
            "star.A.distance": {"unit": "km", "initval": 9.0},
        }
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match="DUPLICATE PARAMETER SPELLING"):
        cm.resolve("star", "distance", shape=(1,), names=["A"])


def test_engine_injected_index_entry_is_not_a_second_spelling():
    """
    Given a user entry in the borrowed-name form and the index-form entry
      finalize_user_params injects its solved start value under,
    When resolve() runs,
    Then it does not raise and the user's entry still wins.

    The check reads the keys the USER wrote, not the live user_params dict.
    examples/ob161003 ends every prepare() in exactly this state, so reading
    the dict would fail the fit at stage 6.
    """
    # ARRANGE
    cm = _canonical_cm({"star.A.distance": {"initval": 9.0}})
    cm.user_params["star.0.distance"] = {"initval": 7.0, "derived": True}

    # ACT
    cfg = cm.resolve("star", "distance", shape=(1,), names=["A"])

    # ASSERT
    assert np.isclose(cfg["initval"][0], 9.0)


# ---------------------------------------------------------------------------
# A broadcast key must cover the WHOLE vector  (review 1.1.1 / 7.1.1a)
# ---------------------------------------------------------------------------

# One config entry, per-source parameter vectors -- the examples/ob161003
# shape, reduced to what ConfigManager needs (it never builds components).
TWO_SOURCE_SYSTEM = {
    "star": [{"name": "Lens"}, {"name": "SourceA"}, {"name": "SourceB"}],
    "lens": [
        {
            "name": "Lens",
            "lenses": ["star.0"],
            "sources": ["star.1", "star.2"],
        }
    ],
}


def test_broadcast_shorter_than_the_vector_raises():
    """
    Given a 2-part broadcast `lens.t_0` on a system whose lens has ONE config
      entry but TWO sources, so the parameter has two elements,
    When resolve() is asked for the full vector,
    Then it raises, naming the per-element spellings that do work.

    standardize_param_names expands a broadcast key by the CONFIG LIST length
    -- all it can see, running before any manifest exists -- so `lens.t_0`
    became `lens.0.t_0` alone and element 1 fell back to the defaults.yaml
    backstop with no message.  Not a start-value-only defect: the same
    expansion carries `sigma`/`mu`/`lower`, i.e. the PRIOR reached one source
    and not the other, which is a silent posterior change on a 2S2L fit.
    """
    # ARRANGE
    cm = ConfigManager(
        {"lens.t_0": 2450000.0}, system_config=TWO_SOURCE_SYSTEM
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match="BROADCAST KEY DOES NOT COVER"):
        cm.resolve("lens", "t_0", shape=(2,), names=["SourceA", "SourceB"])


def test_the_error_names_the_per_element_spellings():
    """
    Given the same under-covering broadcast, resolved with per-element names,
    When the error is raised,
    Then it quotes the name-form spelling of every element.

    The fix is a raise, so the message IS the feature: a user who wrote one
    line has to be told which lines to write instead, in the spelling their
    own config supports (ob161003 addresses these by the source star's name).
    """
    # ARRANGE
    cm = ConfigManager(
        {"lens.t_0": 2450000.0}, system_config=TWO_SOURCE_SYSTEM
    )

    # ACT
    with pytest.raises(ValueError) as excinfo:
        cm.resolve("lens", "t_0", shape=(2,), names=["SourceA", "SourceB"])

    # ASSERT
    assert "lens.SourceA.t_0" in str(excinfo.value)
    assert "lens.SourceB.t_0" in str(excinfo.value)


def test_broadcast_longer_than_the_vector_raises():
    """
    Given two RV instruments sharing ONE detrend column between them, so
      `detrend_coeffs` has one element while the config list has two,
    When resolve() is asked for the vector,
    Then it raises.

    The defect is bidirectional and this is the other side: Pass 2 writes
    indexed keys for elements that do not exist.  `detrend_coeffs` is the
    live surface (its shape is a column count, not an instrument count), and
    it is a second reason the fix cannot be "fill the missing elements" --
    there is nothing to fill.
    """
    # ARRANGE
    cm = ConfigManager(
        {"rvinstrument.detrend_coeffs": 0.1},
        system_config={"rvinstrument": [{"name": "A"}, {"name": "B"}]},
    )

    # ACT / ASSERT
    with pytest.raises(ValueError, match="BROADCAST KEY DOES NOT COVER"):
        cm.resolve("rvinstrument", "detrend_coeffs", shape=(1,))


def test_a_matching_broadcast_is_untouched():
    """
    Given the ordinary case -- one config entry per element,
    When resolve() runs,
    Then the broadcast covers every element and nothing raises.

    The check is a length comparison, so this is the assertion that it costs
    the shipped configs nothing: every example that broadcasts (`star.teff`,
    `rvinstrument.gamma`, ...) is this case.
    """
    # ARRANGE
    cm = ConfigManager(
        {"star.distance": {"initval": 100.0}}, system_config=SYSTEM
    )

    # ACT
    cfg = cm.resolve("star", "distance", shape=(2,), names=["A", "B"])

    # ASSERT
    assert np.allclose(cfg["initval"], 100.0)


def test_single_element_resolution_is_not_checked():
    """
    Given a broadcast key and a resolve() for ONE element of a longer vector,
    When resolve() runs with shape=() and element=1,
    Then it does not raise.

    `shape=()` with `element=` is how the relaxation engine and
    Instrument._time_coord read one element at a time; n_elements is 1 there
    by construction, so comparing it against the config-list length would
    reject every ordinary broadcast on a multi-instance component.  Only an
    explicit vector shape states the parameter's real length.
    """
    # ARRANGE
    cm = ConfigManager(
        {"star.distance": {"initval": 100.0}}, system_config=SYSTEM
    )

    # ACT
    cfg = cm.resolve("star", "distance", shape=(), element=1)

    # ASSERT
    assert np.isclose(cfg["initval"][0], 100.0)


# ---------------------------------------------------------------------------
# 1b. user BOUNDS  (review 1.1.5: the one field the rule did not reach)
# ---------------------------------------------------------------------------

# A flat-dict component, because that is where the collision is REACHABLE.
# For a LIST component `standardize_param_names` expands a 2-part key only
# into the indices no 3-part key claimed, so the element never sees the
# broadcast at all -- pinned by test_a_list_component_resolves_the_collision_
# before_resolve below, which is why 1.1.5's own `star.av` example does not
# reproduce and `sed.av` does.
_FLAT = {"sed": {"file": "x.yaml"}}


@pytest.mark.parametrize(
    "field, broadcast, specific",
    [("lower", 1.0, 0.5), ("upper", 1.0, 5.0)],
)
def test_specific_user_bound_beats_broadcast_user_bound(
    field, broadcast, specific
):
    """
    Given a broadcast user bound and a specific one on the same element,
    When resolve() applies the user's params,
    Then the SPECIFIC bound wins outright, even though it is the looser one.

    This is review 1.1.5.  The specific spelling is the exception to the
    broadcast, not a second opinion to be combined with it, so the looser
    specific value is the answer -- which is exactly what max()/min() could
    never produce and why the ordering was invisible.

    Both fields are exercised because "strictest" points in OPPOSITE
    directions for the two: a fix that special-cased `lower` alone would
    still lose `upper`, and vice versa.
    """
    # ARRANGE
    cm = ConfigManager(
        {"sed.av": {field: broadcast}, "sed.0.av": {field: specific}},
        system_config=_FLAT,
    )

    # ACT
    cfg = cm.resolve("sed", "av", shape=(1,))

    # ASSERT
    assert np.isclose(cfg[field][0], specific)


def test_a_broadcast_user_bound_still_covers_an_element_that_states_none():
    """
    Given a broadcast user bound and a specific entry that sets a DIFFERENT
      field,
    When resolve() applies the user's params,
    Then the broadcast bound still lands.

    The fix suppresses a broadcast bound only where a more specific spelling
    STATES THAT BOUND.  A specific entry about something else must not
    silently cancel the broadcast -- that would trade one silent override for
    another.
    """
    # ARRANGE
    cm = ConfigManager(
        {"sed.av": {"lower": 1.0}, "sed.0.av": {"initval": 3.0}},
        system_config=_FLAT,
    )

    # ACT
    cfg = cm.resolve("sed", "av", shape=(1,))

    # ASSERT
    assert np.isclose(cfg["lower"][0], 1.0)
    assert np.isclose(cfg["initval"][0], 3.0)


@pytest.mark.parametrize(
    "field, user, validity, expected",
    [("lower", 0.5, 2.0, 2.0), ("upper", 5.0, 3.0, 3.0)],
)
def test_a_user_bound_still_loses_to_a_component_validity_bound(
    field, user, validity, expected
):
    """
    Given a component-computed validity bound and a user bound outside it,
    When resolve() layers them,
    Then the STRICTEST still wins -- the half 1.1.5 deliberately did not touch.

    Two rules, kept apart: user-vs-user is decided by specificity, but
    user-vs-defaults is decided by strictness, because a validity bound marks
    where the likelihood stops being meaningful (a grid edge, a variance
    floor) and a user preference cannot widen it.  A fix that made the user's
    bound simply assign would have broken this, silently, in the direction
    that produces NaNs.
    """
    # ARRANGE
    cm = ConfigManager({"sed.0.av": {field: user}}, system_config=_FLAT)
    cm.add_override("sed.av", **{field: validity})

    # ACT
    cfg = cm.resolve("sed", "av", shape=(1,))

    # ASSERT
    assert np.isclose(cfg[field][0], expected)


def test_a_list_component_resolves_the_collision_before_resolve():
    """
    Given the broadcast and specific spellings on a LIST component,
    When the ConfigManager standardizes its params,
    Then only the specific key survives for the element that claimed it.

    Pinned because it is why review 1.1.5's own worked example (`star.av`
    vs `star.0.av`) does NOT reproduce, and a later change to
    standardize_param_names that started keeping both would quietly hand the
    user-params loop a case it now has to decide.  The rule is the same
    either way -- the specific entry wins -- so this test states WHERE the
    decision is made, not what it is.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {"star.av": {"lower": 1.0}, "star.0.av": {"lower": 0.5}},
        system_config={"star": [{"name": "A"}, {"name": "B"}]},
    )

    # ASSERT
    kept = {k: v for k, v in cm.user_params.items() if k.endswith(".av")}
    assert kept == {"star.0.av": {"lower": 0.5}, "star.1.av": {"lower": 1.0}}
    cfg = cm.resolve("star", "av", shape=(2,), names=["A", "B"])
    assert np.isclose(cfg["lower"][0], 0.5)
    assert np.isclose(cfg["lower"][1], 1.0)


# ---------------------------------------------------------------------------
# 4. single-element mode: the index form and the name form must name the
#    SAME element  (review 2.1.8)
# ---------------------------------------------------------------------------


def test_name_form_follows_element_not_the_loop_variable():
    """
    Given resolve() in single-element mode -- shape=(), element=1 -- with
      per-element names,
    When _element_keys builds the spellings of the element being resolved,
    Then the name form names element 1, not element 0.

    In single-element mode the loop variable is always 0 while the element
    being resolved is `element`.  `_eff_idx` exists to carry exactly that
    distinction, and the index form used it -- but the name form was built
    from `names[i]`, so the two spellings named DIFFERENT elements
    (`star.1.distance` alongside `star.A.distance`) and a name-form entry for
    the element actually being resolved was never found.

    A flat-dict component is the vehicle because it is where a name-form key
    SURVIVES: for a list component standardize_param_names folds the name
    form into the index form at construction, so `user_params` never holds
    one and the name-form spelling is dead for that channel.
    """
    # ARRANGE
    cm = ConfigManager(
        {"sed.B.av": {"initval": 7.0}},
        system_config={"sed": {"file": "x.yaml"}},
    )

    # ACT
    cfg = cm.resolve("sed", "av", shape=(), element=1, names=["A", "B"])

    # ASSERT
    assert np.isclose(cfg["initval"][0], 7.0)


def test_a_component_override_on_the_name_form_reaches_single_element_mode():
    """
    Given a component override registered under the NAME form,
    When resolve() runs in single-element mode for that element,
    Then the override is applied.

    The second reachable channel for 2.1.8: `add_override` stores the key as
    the COMPONENT spelled it, with no standardization pass to fold it, so a
    cross-component override written by name was invisible in single-element
    mode.  Measured before the fix: the default 10.0 was returned instead.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=SYSTEM)
    cm.add_override("star.B.distance", initval=42.0)

    # ACT
    cfg = cm.resolve("star", "distance", shape=(), element=1, names=["A", "B"])

    # ASSERT
    assert np.isclose(cfg["initval"][0], 42.0)


# ---------------------------------------------------------------------------
# 5. the broadcast form is USER-FACING ONLY  (review 2.14.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", ["add_hint", "add_scale_hint"])
def test_an_internal_channel_refuses_a_broadcast_path(channel):
    """
    Given a LIST component,
    When a component pushes a hint under the 2-part broadcast spelling,
    Then it raises.

    Broadcasting is a user-facing convenience, translated once at
    construction and never seen again.  A hint is pushed BY A COMPONENT,
    which knows which element it means.  Refused rather than expanded
    because one scalar cannot answer for every element: since review 1.1.5
    the elements may carry different `unit:` overrides, so there is no
    single unit the value could be read in.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=SYSTEM)

    # ACT / ASSERT
    with pytest.raises(ValueError, match="broadcast form"):
        getattr(cm, channel)("star.distance", 100.0)


@pytest.mark.parametrize("spelling", ["star.0.distance", "star.A.distance"])
def test_an_internal_channel_still_takes_a_per_element_path(spelling):
    """
    Given the same component,
    When it pushes the index form or the name form,
    Then the hint is accepted and filed under the index form.

    The guard must reject ONLY the broadcast.  The name form is not a
    broadcast -- it names exactly one element -- and it is translated, not
    refused, which is the distinction the ruling turns on.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=SYSTEM)

    # ACT
    cm.add_hint(spelling, 100.0)

    # ASSERT
    assert "star.0.distance" in cm.hints


def test_a_flat_dict_components_two_part_key_is_not_a_broadcast():
    """
    Given a FLAT-DICT component, whose canonical spelling has two parts,
    When a hint is pushed under it,
    Then it is accepted.

    The regression this guard could most easily cause.  `sed.av` has two
    parts but is not a broadcast -- there is no list to broadcast over, and
    it is that component's ONLY spelling.  The check therefore keys on the
    component's shape in system_config, not on the number of dots.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config={"sed": {"file": "x.yaml"}})

    # ACT
    cm.add_hint("sed.av", 1.0)

    # ASSERT
    assert "sed.av" in cm.hints


def test_no_system_config_stays_permissive():
    """
    Given a ConfigManager built with no system_config,
    When a 2-part path is pushed,
    Then it is accepted.

    That mode cannot tell a broadcast from a flat-dict component's canonical
    key -- there is no system_config to ask -- and it is the tests-and-
    direct-drivers path, not production.  Guessing there would break working
    callers to guard a case production cannot reach.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=None)

    # ACT
    cm.add_hint("star.distance", 100.0)

    # ASSERT
    assert "star.distance" in cm.hints


# ---------------------------------------------------------------------------
# 6. the broadcast is expanded PER FIELD  (review 1.1.6)
# ---------------------------------------------------------------------------


def test_a_specific_entry_about_another_field_keeps_the_broadcast_value():
    """
    Given a broadcast start value and a specific entry that sets only a BOUND,
    When the params are standardized and resolved,
    Then that element keeps the broadcast start AND gets its own bound.

    This is review 1.1.6, and it was a silently wrong START VALUE rather than
    a merely surprising one: before the fix element 1 fell all the way back to
    the defaults.yaml radius (1.0) because its `lower` entry had claimed the
    index, so putting a bound on one star discarded that star's start value.
    """
    # ARRANGE
    cm = ConfigManager(
        {"star.radius": 3.0, "star.B.radius": {"lower": 0.1}},
        system_config=SYSTEM,
    )

    # ACT
    cfg = cm.resolve("star", "radius", shape=(2,), names=["A", "B"])

    # ASSERT
    assert np.isclose(cfg["initval"][0], 3.0)
    assert np.isclose(cfg["initval"][1], 3.0)
    assert np.isclose(cfg["lower"][1], 0.1)


def test_a_specific_entry_about_the_SAME_field_still_wins():
    """
    Given a broadcast start value and a specific one for the same element,
    When both are standardized,
    Then the specific value wins for that element.

    The half that must NOT change.  Merging per field is only correct if the
    specific entry still overrides the broadcast wherever the two actually
    collide -- a fix that merged in the other order, or that concatenated
    rather than overrode, would invert exactly this.
    """
    # ARRANGE
    cm = ConfigManager(
        {"star.radius": 3.0, "star.B.radius": {"initval": 9.0}},
        system_config=SYSTEM,
    )

    # ACT
    cfg = cm.resolve("star", "radius", shape=(2,), names=["A", "B"])

    # ASSERT
    assert np.isclose(cfg["initval"][0], 3.0)
    assert np.isclose(cfg["initval"][1], 9.0)


def test_two_bare_scalars_leave_the_stored_shape_alone():
    """
    Given a broadcast scalar and a specific scalar for one element,
    When the params are standardized,
    Then the specific entry is still stored as a bare scalar.

    Both spellings mean `initval`, so there is nothing to inherit and the
    entry does not need promoting to a dict.  Pinned because the merge could
    easily have rewritten every such entry into `{"initval": x}` -- harmless
    to `resolve()`, which accepts both, but a gratuitous change to the shape
    of `user_params` that other readers and tests describe.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {"star.radius": 3.0, "star.B.radius": 9.0}, system_config=SYSTEM
    )

    # ASSERT
    assert cm.user_params["star.0.radius"] == 3.0
    assert cm.user_params["star.1.radius"] == 9.0


def test_a_per_element_unit_over_an_inherited_broadcast_value_raises():
    """
    Given a broadcast VALUE and a specific entry that only overrides `unit`,
    When the params are standardized,
    Then it raises, naming the inherited field and both keys.

    The one case the merge must NOT resolve by guessing.  A broadcast
    `{initval: 3, unit: solRad}` under a specific `{unit: jupiterRad}` leaves
    "3" with no determinate meaning -- 3 solRad restated in jupiterRad, or 3
    jupiterRad?  Broadcasting is a convenience for the unambiguous case, so
    the ambiguous one is refused and the user is told to spell the value out
    per element.

    Before 1.1.6 this silently produced the DEFAULT radius for that element,
    which is neither reading of the user's intent.
    """
    # ARRANGE / ACT / ASSERT
    with pytest.raises(ValueError, match="BROADCAST VALUE WITH A PER-ELEMENT"):
        ConfigManager(
            {
                "star.radius": {"initval": 3.0, "unit": "solRad"},
                "star.B.radius": {"unit": "jupiterRad"},
            },
            system_config=SYSTEM,
        )


def test_a_per_element_unit_with_its_own_value_is_fine():
    """
    Given a broadcast value and a specific entry carrying BOTH its own unit
      and its own value,
    When the params are standardized,
    Then nothing raises: no number is inherited across a unit change.

    The guard must fire on the ambiguity, not on the mere presence of a
    per-element unit -- which is a legitimate and shipped idiom.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {
            "star.radius": {"initval": 3.0, "unit": "solRad"},
            "star.B.radius": {"initval": 9.0, "unit": "jupiterRad"},
        },
        system_config=SYSTEM,
    )

    # ASSERT
    assert cm.user_params["star.1.radius"]["unit"] == "jupiterRad"
    assert np.isclose(cm.user_params["star.1.radius"]["initval"], 9.0)


# ---------------------------------------------------------------------------
# 7. the engine only ever sees CANONICAL paths  (review 3.14.15c)
# ---------------------------------------------------------------------------

_FLATDICT = {"sed": {"file": "x.yaml"}}


@pytest.mark.parametrize(
    "channel, path",
    [
        ("hints", "star.B.distance"),
        ("hints", "star.distance"),
        ("scale_hints", "star.B.distance"),
        ("scale_hints", "star.distance"),
    ],
)
def test_a_non_canonical_path_is_refused_at_the_engine(channel, path):
    """
    Given a non-canonical path written straight into a hint channel's store,
      bypassing the translation its public method performs,
    When the relaxation engine's hint-layering step checks its inputs,
    Then it raises, naming the path and the canonical spelling.

    This is the assertion that the boundary translation was TOTAL, and it is
    the guard for a whole family of defects rather than for one: reviews
    1.1.5, 1.1.6, 2.1.8, 2.14.6 and 2.14.8 were all the same thing -- a
    user-facing spelling surviving past the boundary, with something
    downstream left to guess which spelling it had. Each was found
    separately, by hand, months apart.

    The stores are written directly here on purpose. Going through
    `add_hint` would be translated (or refused) at the door, which is
    exactly right and exactly why it cannot exercise THIS check. What is
    pinned is the second line of defence.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=SYSTEM)
    getattr(cm, channel)[path] = 1.0

    # ACT / ASSERT
    with pytest.raises(ValueError, match="NON-CANONICAL PATH AT THE ENGINE"):
        cm._assert_canonical_engine_paths({})


def test_a_non_canonical_seed_hint_is_refused_too():
    """
    Given a non-canonical path in the per-seed hint set,
    When the engine checks its inputs,
    Then it raises.

    Three channels reach this step and all three are checked. A guard that
    covered only `hints` would leave the seed path -- the one carrying
    MMEXOFAST solutions, i.e. the least hand-written of the three -- open.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=SYSTEM)

    # ACT / ASSERT
    with pytest.raises(ValueError, match="NON-CANONICAL PATH AT THE ENGINE"):
        cm._assert_canonical_engine_paths({"star.B.distance": 1.0})


def test_the_index_form_passes_the_engine_check():
    """
    Given index-form paths in every channel,
    When the engine checks its inputs,
    Then nothing raises.

    Measured across all twenty shipped example config/params pairs: every one
    of the 72 hint paths and 20 scale-hint paths is already index form, so
    this check is a no-op on everything that ships.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=SYSTEM)
    cm.add_hint("star.0.distance", 3000.0)
    cm.add_scale_hint("star.1.distance", 5.0)

    # ACT
    cm._assert_canonical_engine_paths({"star.0.teff": 5000.0})

    # ASSERT -- reaching here without raising IS the assertion


def test_a_flat_dict_components_two_part_key_passes_the_engine_check():
    """
    Given a flat-dict component's 2-part key,
    When the engine checks its inputs,
    Then it passes, because there the 2-part form IS canonical.

    The asymmetry is real and is review 3.14.17: a single-instance component
    has no list to broadcast over, so `sed.av` is its only spelling. The
    check therefore keys on the component's SHAPE in system_config rather
    than on the number of dots -- the same distinction 2.14.8's push-time
    guard makes, deliberately, so the two cannot disagree about what
    canonical means.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=_FLATDICT)
    cm.hints["sed.av"] = 1.0

    # ACT
    cm._assert_canonical_engine_paths({})

    # ASSERT -- no raise


def test_the_engine_check_does_not_gate_on_the_symbol_map():
    """
    Given a canonical hint path that is NOT a relaxation-engine symbol,
    When the engine checks its inputs,
    Then it passes.

    Pinned because "is it in master_symbol_map" is the obvious formulation
    of this check and it is WRONG. Measured across the twenty shipped
    examples, 25 of 72 hint paths are legitimately absent from the map --
    every `mulensinstrument.<i>.f_source`/`f_blend`/`q_flux` -- because
    `resolve()` reads hints too and a hinted parameter need not be an engine
    symbol. Gating on the map would have refused every microlensing example.
    Shape is the invariant; symbol membership is not.
    """
    # ARRANGE -- star.0.av is a real parameter (the SED's extinction) and is
    # deliberately NOT in the symbol map: `star` declares no symbolic
    # relation for it.  Checked in the assert below rather than assumed,
    # because the first draft of this test used star.0.distance, which IS a
    # symbol, and so demonstrated nothing.
    cm = ConfigManager({}, system_config=SYSTEM)
    cm.add_hint("star.0.av", 0.5)
    assert "star.0.av" not in getattr(cm, "master_symbol_map", {})

    # ACT
    cm._assert_canonical_engine_paths({})

    # ASSERT -- no raise


def test_no_system_config_skips_the_engine_check():
    """
    Given a ConfigManager built with no system_config,
    When the engine checks its inputs,
    Then a 2-part path is accepted.

    That mode cannot tell a broadcast from a flat-dict component's canonical
    key, and it is the tests-and-direct-drivers path. Same reason 2.14.8's
    push-time guard is permissive there; the two must agree, or a path
    refused at one end and accepted at the other would be worse than either
    rule alone.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config=None)
    cm.hints["star.distance"] = 1.0

    # ACT
    cm._assert_canonical_engine_paths({})

    # ASSERT -- no raise
