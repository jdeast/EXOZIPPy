"""ConfigManager.probe_start: the start a stage 1-3 reader must use.

`resolve()` and raw `user_params` show only what was WRITTEN for a path.  A
start the user's entries IMPLY through a relation -- `star.mass: 0.92` ->
`star.logmass` -- exists only once the relaxation engine runs at stage 4, so
every earlier reader used to see the defaults.yaml value instead (review
1.8.6, the EEP seed; review 2.6.30(b), the microlensing flux bootstrap).
`probe_start` runs the engine on a snapshot and returns what it solves.  See
config.md, "Reading a start before stage 4".
"""

import copy
import math

import pytest

from exozippy.config import (
    PRECEDENCE_DERIVED_DATA,
    PRECEDENCE_DERIVED_USER,
    PRECEDENCE_USER,
    ConfigManager,
)

_STAR = {"star": [{"name": "A"}]}


def _cm(params, config=_STAR):
    return ConfigManager(params, system_config=config)


def _state(cm):
    """Everything a probe could conceivably leave behind."""
    snap = {
        attr: copy.deepcopy(getattr(cm, attr))
        for attr in cm._PROBE_SNAPSHOT_ATTRS
    }
    snap["hints"] = dict(cm.hints)
    snap["hint_ranks"] = dict(cm.hint_ranks)
    snap["seed_hint_sets"] = copy.deepcopy(cm.seed_hint_sets)
    snap["master_symbol_map"] = dict(cm.master_symbol_map)
    return snap


def test_a_user_mass_implies_the_logmass_start():
    """
    Given a params file that seeds star.mass (as every shipped one does),
    When the logmass start is probed before stage 4,
    Then it is log10 of that mass, provenanced as solved from ONLY user
      input -- not the defaults.yaml 0.0 resolve() reports at that point.
    """
    # Arrange
    cm = _cm({"star.A.mass": {"initval": 0.92}})
    assert cm.resolve("star", "logmass", element=0)["initval"] == 0.0

    # Act
    got = cm.probe_start(["star.0.logmass", "star.0.mass"])

    # Assert
    logmass = got["star.0.logmass"]
    assert logmass.value == pytest.approx(math.log10(0.92), rel=1e-12)
    assert logmass.user_value == pytest.approx(math.log10(0.92), rel=1e-12)
    assert logmass.rank == PRECEDENCE_DERIVED_USER
    assert logmass.informed
    assert got["star.0.mass"].rank == PRECEDENCE_USER
    assert got["star.0.mass"].source == "user"


def test_the_probe_rolls_every_mutation_back():
    """
    Given a ConfigManager with a user entry and a component hint,
    When starts are probed -- including a path no relation knows, which the
      probe registers as a temporary leaf --
    Then user_params, the hints, the export snapshots, the diagnostics, the
      symbol map and the rest of the engine's state are exactly as before.
    """
    # Arrange
    cm = _cm({"star.A.mass": {"initval": 0.92}})
    cm.add_hint("star.A.teff", 5100.0)
    before = _state(cm)
    assert "star.0.initfeh" not in cm.master_symbol_map

    # Act
    cm.probe_start(["star.0.logmass", "star.0.teff", "star.0.initfeh"])

    # Assert
    after = _state(cm)
    for key, value in before.items():
        assert after[key] == value, f"probe_start leaked {key}"


def test_a_hint_and_a_seed_are_seen_and_a_user_entry_outranks_a_seed():
    """
    Given a component hint, a seed-0 hint set, and a user entry on a path the
      seed also names,
    When starts are probed,
    Then the hint and the seed are layered at their data rank and the user
      entry wins its path -- the layering stage 4's seed-0 solve uses.
    """
    # Arrange
    cm = _cm({"star.A.radius": {"initval": 0.85}})
    cm.add_hint("star.A.teff", 5100.0)
    cm.add_seed_hints(
        [{"star.A.feh": 0.2, "star.A.radius": 3.0}], source="test"
    )

    # Act
    got = cm.probe_start(["star.0.teff", "star.0.feh", "star.0.radius"])

    # Assert
    assert got["star.0.teff"].value == pytest.approx(5100.0)
    assert got["star.0.teff"].rank == PRECEDENCE_DERIVED_DATA
    assert got["star.0.feh"].value == pytest.approx(0.2)
    assert got["star.0.feh"].informed
    assert got["star.0.radius"].value == pytest.approx(0.85)
    assert got["star.0.radius"].rank == PRECEDENCE_USER


def test_value_is_internal_and_user_value_honors_a_unit_override():
    """
    Given a distance written in kpc (the internal unit is pc),
    When it is probed,
    Then `value` is internal (pc) and `user_value` is the user's own number
      -- the conversion goes through from_internal, never by hand.
    """
    # Arrange
    cm = _cm({"star.A.distance": {"initval": 1.5, "unit": "kpc"}})

    # Act
    got = cm.probe_start(["star.0.distance"])["star.0.distance"]

    # Assert
    assert got.value == pytest.approx(1500.0)
    assert got.user_value == pytest.approx(1.5)


def test_a_default_is_derivable_but_not_informed():
    """
    Given no entry on a parameter that carries a defaults.yaml initval,
    When it is probed,
    Then the default comes back as a derivable start that is NOT informed --
      the distinction probe_derivable draws.
    """
    got = _cm({}).probe_start(["star.0.teff"])["star.0.teff"]

    assert got.derivable
    assert got.value == pytest.approx(5778.0)
    assert not got.informed
    assert got.source == "default"


def test_a_path_the_engine_cannot_reach_is_reported_not_derivable():
    """
    Given a parameter with no defaults.yaml initval, no entry, no hint and no
      relation that solves it from those,
    When it is probed,
    Then it comes back typed as "not derivable" -- an answer, not an error.
    """
    # mulensevent.t_E has no defaults.yaml initval; with no event geometry
    # at all nothing can solve it.
    cm = _cm({}, {"mulensevent": [{"name": "event"}]})

    got = cm.probe_start(["mulensevent.0.t_E"])["mulensevent.0.t_E"]

    assert not got.derivable
    assert got.value is None and got.user_value is None and got.rank is None
    assert got.source == "not derivable"


def test_an_engine_failure_raises_naming_the_path(monkeypatch):
    """
    Given a relaxation engine that fails,
    When starts are probed,
    Then the failure RAISES naming the probed path, and the state is still
      rolled back -- never swallowed as "not derivable" (review 2.1.17; the
      old probe_derivable did exactly that, flipping the MMEXOFAST trigger).
    """
    # Arrange
    cm = _cm({"star.A.mass": {"initval": 0.92}})
    before = _state(cm)

    def _boom(*args, **kwargs):
        cm.user_params["star.0.teff"] = {"initval": -1.0}  # a mutation
        raise ZeroDivisionError("engine bug")

    monkeypatch.setattr(cm, "resolve_and_validate_parameters", _boom)

    # Act / Assert
    with pytest.raises(RuntimeError, match=r"star\.0\.logmass") as info:
        cm.probe_start(["star.0.logmass"])
    assert isinstance(info.value.__cause__, ZeroDivisionError)
    with pytest.raises(RuntimeError, match=r"star\.0\.logmass"):
        cm.probe_derivable(["star.0.logmass"])
    assert _state(cm) == before


@pytest.mark.parametrize(
    "path,error",
    [
        ("star.A.logmass", ValueError),  # name form: a caller bug
        ("star.logmass", ValueError),  # 2-part broadcast spelling
        ("star.0.not_a_parameter", KeyError),
    ],
)
def test_a_non_canonical_or_unknown_path_raises(path, error):
    """
    Given a path in anything but the canonical index form, or one naming no
      parameter,
    When it is probed,
    Then it raises rather than guessing (one internal spelling).
    """
    with pytest.raises(error):
        _cm({}).probe_start([path])
