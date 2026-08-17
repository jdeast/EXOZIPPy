"""One table declares a per-instance parameterization; one helper pins opt-ins.

`components/parameterization.py` exists because turning "this band is
quadratic, that one is linear" into per-element masks is mechanical, and four
components (band, planet, star, orbit) need exactly the same transformation.
Writing it per component is how four implementations of one idea drift apart --
the way the three byte-identical opt-in pin loops (`Instrument._register_gp`,
`Instrument._register_robust`, `Band._pinned_manifest_entry`) already had, which
`pin_unselected` now replaces.

The distinction these tests pin hardest is between the two pins:

  * a parameter that is not part of an instance's parameterization at all is
    INACTIVE -- pinned through the structural `mask` channel, reported nowhere,
    and NOT user-overridable, because freeing it would add a dimension no
    likelihood term reads;
  * a parameter that exists everywhere but is only wanted on some instances is
    pinned through `"overrides"`, which layers UNDER the params file, so a user
    who wants it back still wins.
"""

import numpy as np
import pytest

from exozippy.components.parameterization import (
    merge_overrides,
    mode_manifest,
    pin_unselected,
)
from exozippy.manifest import interpret_manifest_entry

# The limb-darkening table, which is PR 3's real case: a quadratic band samples
# Kipping q1/q2 and derives u1/u2 from them; a linear band samples u1 directly
# and has no q1/q2 and no u2 at all.
_LD_TABLE = {
    "quadratic": {
        "q1": None,
        "q2": None,
        "u1": "kipping",
        "u2": "kipping",
    },
    "linear": {"u1": None},
}


def test_a_single_mode_system_produces_a_plain_manifest():
    """
    Given every element in the same mode,
    When the table is expanded,
    Then no entry carries a mask and the derived ones name their block as a
      bare string.

    The bar for this work is that a system not using the vocabulary builds a
    bit-identical graph, and `build_pymc` keeps its whole-vector path only for
    a bare-string expr_key with nothing masked out -- so the single-mode case
    must produce exactly the manifest a component would have hand-written.
    """
    manifest = mode_manifest(
        ["quadratic", "quadratic"],
        _LD_TABLE,
        options={"u2": {"inactive_value": 0.0}},
    )

    assert manifest["q1"] == {}
    # A bare string, exactly as a hand-written manifest spells it -- and the
    # inactive_value is dropped, because with nothing masked out it could not
    # do anything.
    assert manifest["u1"] == "kipping"
    assert manifest["u2"] == "kipping"


def test_a_mixed_system_masks_and_selects_per_element():
    """
    Given a quadratic band and a linear band,
    When the table is expanded,
    Then q1/q2 are active only on the quadratic band, u1 is derived there and
      sampled on the linear one, and u2 exists only on the quadratic band.

    This is the case that used to RAISE ("all bands must share one ld_law"),
    whose only workaround was quadratic everywhere with q2 pinned at 0.5 -- a
    prior uniform in q1 rather than in u1.
    """
    manifest = mode_manifest(["quadratic", "linear"], _LD_TABLE)

    assert manifest["q1"]["mask"].tolist() == [True, False]
    assert manifest["q2"]["mask"].tolist() == [True, False]
    assert manifest["u2"]["mask"].tolist() == [True, False]
    # u1 is active on both, derived on the quadratic band only.
    assert "mask" not in manifest["u1"]
    assert list(manifest["u1"]["expr_key"]) == ["kipping"]
    assert manifest["u1"]["expr_key"]["kipping"].tolist() == [True, False]


def test_a_parameter_no_instance_uses_is_left_out_entirely():
    """
    Given every element in the linear mode,
    When the table is expanded,
    Then q1, q2 and u2 do not appear at all.

    Not "appear, wholly inactive": a component hand-writing this manifest
    omitted them, and its consumers key on `"u2" in band.manifest` to
    substitute zeros -- so declaring a fully inactive vector would change the
    graph of a system that made ONE choice, which is exactly what this work
    must not do.
    """
    manifest = mode_manifest(["linear", "linear"], _LD_TABLE)

    assert set(manifest) == {"u1"}
    assert manifest["u1"] == {}


def test_the_expanded_entries_parse_as_manifest_entries():
    """
    Given the mixed expansion,
    When each entry is read by the manifest interpreter,
    Then the roles come back as declared.

    The helper writes the manifest vocabulary; it must not invent a dialect,
    so the assertion is that `interpret_manifest_entry` -- the ONE reader --
    agrees with what the table said.
    """
    manifest = mode_manifest(["quadratic", "linear"], _LD_TABLE)

    u1 = interpret_manifest_entry(manifest["u1"])
    q1 = interpret_manifest_entry(manifest["q1"])

    assert u1.names_expression and u1.is_per_element
    assert u1.expression_configs(
        {"kipping": {"func_name": "f"}}, n_elements=2, where="band.u1"
    )[0].mask.tolist() == [True, False]
    assert q1.names_expression is False
    assert q1.activity_mask(2).tolist() == [True, False]


def test_a_value_the_other_mode_defines_is_stated_not_inherited():
    """
    Given per-parameter options alongside the table,
    When the entries are expanded,
    Then they carry through.

    A linear-law band's u2 is exactly 0, not "whatever the quadratic default
    was": an inactive element's pin should say what the physics says, so the
    number cannot drift with an unrelated defaults.yaml edit.
    """
    manifest = mode_manifest(
        ["quadratic", "linear"],
        _LD_TABLE,
        options={"u2": {"inactive_value": 0.0}},
    )

    assert manifest["u2"]["inactive_value"] == 0.0
    assert interpret_manifest_entry(manifest["u2"]).inactive_value == 0.0


def test_an_unknown_mode_raises_naming_the_table():
    """
    Given an element whose mode is not in the table,
    When the table is expanded,
    Then it raises, naming the unknown mode and the known ones.

    A mode with no table entry would silently give that instance NO parameters
    -- every one of them inactive -- which is never what a caller means.
    """
    with pytest.raises(ValueError, match="unknown parameterization mode"):
        mode_manifest(["quadratic", "kipling"], _LD_TABLE)


def test_a_mode_list_of_the_wrong_length_raises():
    """
    Given fewer modes than elements,
    When the table is expanded,
    Then it raises.

    The mode list is per element; a short one would leave real instances
    unaccounted for, which is the same sizing hazard review 1.1.1 reports for
    broadcast config keys.
    """
    with pytest.raises(ValueError, match="mode\\(s\\) for 3 element"):
        mode_manifest(["quadratic", "linear"], _LD_TABLE, n_elements=3)


def test_conflicting_options_between_modes_raise():
    """
    Given two modes that give one parameter different values for the same
      manifest option,
    When the table is expanded,
    Then it raises rather than letting one mode win.

    A manifest option is per parameter, not per element, so there is no
    honest way to honor both -- and picking one silently is exactly the class
    of bug the single manifest interpreter exists to prevent.
    """
    table = {
        "a": {"x": {"table_note": "from a"}},
        "b": {"x": {"table_note": "from b"}},
    }

    with pytest.raises(ValueError, match="conflicting 'table_note'"):
        mode_manifest(["a", "b"], table)


def test_a_per_element_expr_key_inside_a_table_cell_raises():
    """
    Given a table cell that itself carries a per-element expr_key,
    When the table is expanded,
    Then it raises.

    The table IS the per-element statement; a second one nested inside it would
    have two sources for one element's role.
    """
    table = {"a": {"x": {"expr_key": {"blk": [True, False]}}}}

    with pytest.raises(ValueError, match="per-element expr_key"):
        mode_manifest(["a", "a"], table)


# ---------------------------------------------------------------------------
# The opt-in pin (the three retired copies)
# ---------------------------------------------------------------------------


def test_pin_unselected_pins_through_the_user_overridable_channel():
    """
    Given three elements of which one opted in,
    When the opt-in pin is built,
    Then the other two are pinned with sigma: 0 through "overrides", and the
      opted-in element is left alone (NaN = "leave this element alone").

    "overrides" layers UNDER the params file on purpose: these parameters do
    exist for every instance (a GP amplitude, a limb-darkening coefficient
    nothing currently reads), so a user who explicitly wants one back should
    win.  Contrast an INACTIVE element, whose pin is structural.
    """
    entry = pin_unselected(3, [1])

    sigma = entry["overrides"]["sigma"]
    assert sigma[1] != sigma[1]  # NaN: untouched
    assert sigma[0] == 0.0 and sigma[2] == 0.0
    assert "mask" not in entry


@pytest.mark.parametrize(
    "entry,expected_expr",
    [
        ("kipping", "kipping"),
        ({"expr_key": "kipping"}, "kipping"),
        ({"expr_key": {"kipping": [True, False]}}, {"kipping": [True, False]}),
        (None, None),
        ({}, None),
    ],
)
def test_merging_an_override_never_drops_a_derivation(entry, expected_expr):
    """
    Given a manifest entry in each shape the vocabulary allows,
    When an override is merged into it,
    Then the override lands and the entry's expression survives.

    Review 4.5.3: a caller that ADDS an option to an existing entry has to read
    the vocabulary as a writer, and the obvious
    `dict(entry) if isinstance(entry, dict) else {}` silently drops a
    bare-string expr_key -- turning a derived parameter into a sampled one, with
    no message.  Latent when it was reported (nothing pinned such an entry);
    per-band limb-darkening laws make bare-string entries reachable by the
    autopin, so the shape is read through the interpreter now.
    """
    merged = merge_overrides(entry, {"sigma": [0.0, np.nan]})

    assert merged["overrides"]["sigma"] == [0.0, np.nan] or np.isnan(
        merged["overrides"]["sigma"][1]
    )
    if expected_expr is None:
        assert "expr_key" not in merged
    else:
        assert merged["expr_key"] == expected_expr


def test_merging_an_override_keeps_the_ones_already_there():
    """
    Given an entry that already carries overrides,
    When another override is merged,
    Then both are present and the new one wins on a collision.
    """
    merged = merge_overrides(
        {"overrides": {"lower": [1.0], "sigma": [1.0]}}, {"sigma": [0.0]}
    )

    assert merged["overrides"] == {"lower": [1.0], "sigma": [0.0]}


def test_pin_unselected_is_a_free_parameter_when_everything_opted_in():
    """
    Given every element opted in,
    When the pin is built,
    Then the entry is empty -- a free parameter carrying no options.

    Byte-for-byte what the three hand-written loops returned, so migrating
    them cannot move a start point.
    """
    assert pin_unselected(2, [0, 1]) == {}
    assert pin_unselected(2, np.array([True, True])) == {}
