"""Component start-value hints must be visible to ConfigManager.resolve().

``add_hint`` is the one correct channel for a component-supplied start value:
it is ranked, the relaxation engine solves from it, and the provenance
ledger, ``initval_source`` and ``export_solution`` all read that rank.  But
the engine does not run until stage 4, and several components ask
``resolve()`` for a start value at stage 2/3 -- ``Orbit`` builds ``tc``'s
HARD window as ``tc_init +/- P/2`` there, so a seed it cannot see does not
merely start the chain in the wrong place, it makes the right place
unreachable and ``Parameter.build_pymc`` correctly calls that fatal.

``resolve()`` did not layer ``self.hints`` until review 3.14.3, so those
readers could not see a hint at all and ``globalsearch.seed_start`` wrote
every searched period and epoch a SECOND time through ``add_override``
purely to be visible to them -- one number, two channels, two chances to
drift.  These tests pin the layering and, above all, its PRECEDENCE: under
the user's params, over defaults.yaml and the override channel.  Reversing
that order would let a component's guess silently overwrite a number the
user typed, in the one place the ledger cannot see it happen.

Companion files: ``tests/test_config_hint_paths.py`` (the hint channels agree
on path translation and units) and ``tests/test_global_search.py`` (the
end-to-end stage-2 reader, ``Orbit``'s ``tc`` window).
"""

import numpy as np
import pytest

from exozippy.config import ConfigManager

# planet.mass is jupiterMass -> solMass and star.ra is deg -> rad, so a real
# unit conversion is exercised on both the value and the unit-override path.
SYSTEM = {
    "star": [{"name": "A"}, {"name": "B"}],
    "planet": [{"name": "b", "star_ndx": 0, "orbit_ndx": 0}],
    "orbit": [{"name": "b"}],
}


def _cm(user_params=None):
    return ConfigManager(user_params or {}, system_config=SYSTEM)


def _initval(cm, comp, param, shape=(1,), element=None):
    cfg = cm.resolve(comp, param, shape=shape, element=element)
    val = cfg["initval"]
    return None if val is None else np.atleast_1d(val).astype(float)


def test_a_hint_is_visible_to_a_stage_3_resolve_reader():
    """
    Given a component that pushes a ranked start value with add_hint,
    When a stage-3 reader asks resolve() for that parameter -- before the
      relaxation engine has run, which is all such a reader ever sees,
    Then resolve() returns the hinted value rather than the defaults.yaml one.

    This is the whole of review 3.14.3.  Orbit's tc window is built from
    exactly this call, so a hint invisible here is a hard bound centred on
    the wrong epoch.
    """
    # Arrange
    cm = _cm()
    before = _initval(cm, "planet", "mass")

    # Act
    cm.add_hint("planet.b.mass", 2.5)
    after = _initval(cm, "planet", "mass")

    # Assert
    assert after[0] == pytest.approx(2.5)
    assert before is None or before[0] != pytest.approx(2.5)


def test_a_user_param_beats_a_hint():
    """
    Given the user naming a start value in params.yaml,
    When a component hints a different one for the same element,
    Then resolve() returns the USER's value.

    Every hint rank is below PRECEDENCE_USER and the relaxation engine enforces
    that with an explicit guard.  resolve() has no ledger to consult, so the
    ORDER of the two loops is the only thing enforcing it here.
    """
    # Arrange
    cm = _cm({"planet.0.mass": {"initval": 7.0}})

    # Act
    cm.add_hint("planet.b.mass", 2.5)
    got = _initval(cm, "planet", "mass")

    # Assert
    assert got[0] == pytest.approx(7.0)


def test_a_user_mu_with_no_initval_still_beats_a_hint():
    """
    Given the user giving only a prior centre (mu) for a parameter,
    When a component hints a start value for it,
    Then resolve() starts at the user's mu, not at the hint.

    resolve() promotes a lone mu to the start value precisely because the
    user's prior centre beats an arbitrary default; a hint must not slip in
    between the two and undo it.
    """
    # Arrange
    cm = _cm({"planet.0.mass": {"mu": 3.0, "sigma": 0.5}})

    # Act
    cm.add_hint("planet.b.mass", 2.5)
    got = _initval(cm, "planet", "mass")

    # Assert
    assert got[0] == pytest.approx(3.0)


def test_a_hint_beats_the_component_override_channel():
    """
    Given a component-computed override carrying an initval,
    When another component hints a measured start value for the same element,
    Then the hint wins.

    An "overrides" dict carries component-computed DEFAULTS and validity
    bounds and has no rank; a hint is a ranked measurement.  No shipped
    component writes an initval through both channels, so nothing moves
    today -- this pins the answer for the day one does.
    """
    # Arrange
    cm = _cm()
    cm.add_override("planet.0.mass", initval=3.0)

    # Act
    cm.add_hint("planet.0.mass", 2.5)
    got = _initval(cm, "planet", "mass")

    # Assert
    assert got[0] == pytest.approx(2.5)


def test_a_hint_does_not_touch_bounds_or_the_prior():
    """
    Given a parameter with defaults.yaml bounds,
    When a component hints a start value for it,
    Then only initval moves: lower, upper, mu and sigma are untouched.

    A hint is one scalar feeding initval.  Widening a bound or inventing a
    prior from the same number is what the override channel is for, and it
    is ranked and reported differently.
    """
    # Arrange
    cm = _cm()
    before = cm.resolve("planet", "mass", shape=(1,))

    # Act
    cm.add_hint("planet.b.mass", 2.5)
    after = cm.resolve("planet", "mass", shape=(1,))

    # Assert
    for key in ("lower", "upper", "mu", "sigma", "init_scale"):
        np.testing.assert_array_equal(
            np.asarray(before[key], dtype=float),
            np.asarray(after[key], dtype=float),
            err_msg=f"a hint moved {key}",
        )
    assert after["user_modified"] is False


def test_a_hint_comes_back_in_the_parameters_user_unit():
    """
    Given a parameter whose defaults.yaml unit (deg) is not its internal
      unit (rad),
    When a component hints 90 degrees for the second star,
    Then add_hint stores pi/2 and resolve() hands back 90.

    resolve() returns user units and Parameter re-applies the conversion, so
    a hint layered in without dividing the factor out again would be wrong
    by 57x -- silently, since both numbers are plausible.
    """
    # Arrange
    cm = _cm()

    # Act
    cm.add_hint("star.B.ra", 90.0)
    got = _initval(cm, "star", "ra", shape=(2,))

    # Assert
    assert cm.hints["star.1.ra"] == pytest.approx(np.pi / 2)
    assert got[1] == pytest.approx(90.0)


def test_a_hint_honors_a_user_unit_override():
    """
    Given the user relabelling a planet mass in earthMass,
    When a component hints 5 (earthMass, the unit add_hint reads),
    Then resolve() hands back 5 in that same unit.

    The element's own `unit:` is what Parameter will convert with, so the
    hint has to be divided by the USER -> internal factor, not the
    defaults -> internal one.  Getting that wrong is a factor of 318 here.
    """
    # Arrange
    cm = _cm({"planet.0.mass": {"unit": "earthMass"}})

    # Act
    cm.add_hint("planet.0.mass", 5.0)
    cfg = cm.resolve("planet", "mass", shape=(1,))

    # Assert
    assert cfg["unit"] == "earthMass"
    assert np.atleast_1d(cfg["initval"])[0] == pytest.approx(5.0)


def test_the_most_specific_hint_spelling_wins():
    """
    Given a broadcast hint on every star and a per-element hint on one,
    When resolve() builds the vector,
    Then the broadcast value covers the unclaimed element and the specific
      one wins where it is written.

    Same rule as every other field resolve() layers, and it comes from the
    same shared candidate list (_lookup_keys), so the three spellings cannot
    drift apart per channel.

    BUILT WITHOUT A system_config, since review 2.14.8.  A component may no
    longer PUSH a broadcast hint -- that is refused now, because one scalar
    cannot answer for every element once the elements may carry different
    `unit:` overrides (review 1.1.5) -- but `_lookup_keys` still consults the
    broadcast spelling, because a 2-part key still ARRIVES by the routes the
    guard cannot see: this constructor mode, and the relaxation engine's own
    ledger, which registers unmapped 2-part keys as leaf symbols and hands
    them back through `propagated_scales`.  The layering rule under test is
    unchanged; only the door has narrowed.
    """
    # Arrange
    cm = ConfigManager({}, system_config=None)

    # Act
    cm.add_hint("star.teff", 5000.0)
    cm.add_hint("star.1.teff", 4000.0)
    got = _initval(cm, "star", "teff", shape=(2,))

    # Assert
    assert got[0] == pytest.approx(5000.0)
    assert got[1] == pytest.approx(4000.0)


def test_a_single_element_resolve_reads_its_own_hint():
    """
    Given per-element hints on two stars,
    When each element is resolved one at a time with `element=`,
    Then each reads its own hint and neither bleeds into the other.

    The relaxation engine's default-armor loop resolves one element at a
    time this way, so a hint keyed off the local loop index instead of the
    requested element would seed every star from star 0.
    """
    # Arrange
    cm = _cm()
    cm.add_hint("star.0.teff", 5000.0)
    cm.add_hint("star.1.teff", 4000.0)

    # Act
    first = _initval(cm, "star", "teff", shape=(), element=0)
    second = _initval(cm, "star", "teff", shape=(), element=1)

    # Assert
    assert first[0] == pytest.approx(5000.0)
    assert second[0] == pytest.approx(4000.0)
