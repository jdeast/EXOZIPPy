"""Half-built features must refuse loudly, not accept-and-ignore.

The `orbit:` block had a config surface that was parsed and then silently
dropped on the floor (code review 2026-08-08, item 5.11): `fitvcve: true`
selected `defaults.yaml` expressions naming undefined physics functions, and
the `vcve`/`b` parameter blocks were in no manifest at all (`b`'s deps even
name a nonexistent `orbit.ar`).

BOTH halves are implemented now (review 8.8.3): `fitvcve` samples V_c/V_e with
the likelihood marginalized over both roots of its quadratic inversion, and
`fitchord` samples the transit chord and derives cos i from it.  So what these
tests pin is the opposite of what they used to -- that neither key raises, and
that a params entry naming `vcve` or `chord` is honored as a start value.

What the guard still covers, and the reason this file did not simply go away,
is the one job it always did independently of the parameterizations: an
`orbit.<name>.b` in a params file is otherwise SILENTLY IGNORED.  The impact
parameter lives on the planet, so the entry raises and says so; `tests/
test_chord.py` covers what the chord half itself does.

(The `star:` half of that review item -- `sedfile:`, `mist:`, `parsec:` --
was reverted: `sedfile` is deleted outright and the evolutionary-model
switches are ungated, because the star component is now ready for an
`evolutionarymodel` component to land against unchanged.  See
tests/test_star_evolutionary_model.py.)
"""

import numpy as np
import pytest

from exozippy.components.orbit import Orbit
from exozippy.config import ConfigManager


def _orbit(blocks, user_params=None):
    """Construct an Orbit from a list of orbit config blocks."""
    return Orbit(blocks, ConfigManager(dict(user_params or {})))


# ----------------------------------------------------------------------
# 5.11 -- orbit fitvcve / vcve / b
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def test_orbit_fitvcve_is_implemented_and_no_longer_guarded():
    """
    Given an orbit block asking for the V_c/V_e parameterization,
    When the Orbit component is constructed,
    Then it does NOT raise, and the orbit records the V_c/V_e mode.

    History: this guard raised for two reasons, and both are gone.  The
    structural one -- `fitvcve` is per orbit but `Parameter.build_pymc` derived
    a whole vector -- was fixed by the per-element roles; the physics one
    (calc_ecc_from_vcve / calc_omega_from_vcve undefined) by implementing them,
    with the discrete root choice the paper uses replaced by a marginalization
    over both roots.  What is left guarded is the CHORD half; see below.
    """
    orbit = _orbit([{"name": "b", "fitvcve": True}])

    assert orbit.fitvcve == [True]
    assert orbit.ecc_modes == ["vcve"]


def test_orbit_fitchord_is_implemented_and_no_longer_guarded():
    """
    Given an orbit block asking for the transit-chord parameterization,
    When the Orbit component is constructed,
    Then it does not raise, and the switch is recorded.

    The half that used to be the guard's whole subject.  Its own behavior --
    the inversion, the shields, the Jacobian's direction, the roles and the
    transit-only default -- is tests/test_chord.py; what this pins is only
    that asking for it is no longer an error.
    """
    orbit = _orbit([{"name": "b", "fitchord": True}])

    assert orbit.fitchord == [True]


def test_fitchord_follows_fitvcve():
    """
    Given `fitvcve: true` and no explicit `fitchord`,
    When the component is constructed,
    Then fitchord follows it.

    JDE's coupling rule: `fitvcve: false` forces `fitchord: false` unless
    fitchord is set separately, which falls out of fitchord defaulting to
    whatever fitvcve resolved to.  It mattered more when the chord half raised
    -- a followed value must not have tripped that guard -- and it still
    matters, because it is what makes `fitvcve: true` turn on the PAIR the
    paper validated rather than half of it.
    """
    orbit = _orbit([{"name": "b", "fitvcve": True}])

    assert orbit.fitchord == [True]

    off = _orbit([{"name": "b", "fitvcve": False}])
    assert off.fitchord == [False]


def test_a_vcve_params_entry_is_honored_not_rejected():
    """
    Given a params-file entry naming the orbit's V_c/V_e, in each accepted
      spelling,
    When the component is constructed and the entry resolved,
    Then it is used -- the parameter exists now, so the entry is a start value
      rather than a silently ignored key.
    """
    blocks = [{"name": "b", "fitvcve": True}]
    for path in ("orbit.b.vcve", "orbit.0.vcve", "orbit.vcve"):
        # A system_config, unlike the bare _orbit helper: the NAME spelling is
        # standardized to an index by ConfigManager, which needs to know the
        # instance names to do it.
        cm = ConfigManager(
            {path: {"initval": 0.8}}, system_config={"orbit": blocks}
        )
        orbit = Orbit(blocks, cm)
        resolved = orbit.config_manager.resolve("orbit", "vcve", shape=(1,))
        assert float(np.atleast_1d(resolved["initval"])[0]) == pytest.approx(
            0.8
        ), path


def test_orbit_b_user_param_raises_and_points_at_planet_b():
    """
    Given a params-file entry on an orbit-level impact parameter,
    When the Orbit component is constructed,
    Then it raises, explaining where the impact parameter really lives.

    The orbit has no `b`: it is defined by a planet's radius ratio, so it is
    `planet.<n>.b`, and the orbit's own scaled semi-major axis is
    `planet.<n>.ar`.  The entry would otherwise be silently ignored -- the
    failure mode `config._reject_renamed_arsun` exists to prevent -- and the
    message now also points at `fitchord`, which is what someone reaching for
    this from the orbit side usually wants.
    """
    # Arrange
    blocks = [{"name": "b"}]

    # Act
    with pytest.raises(NotImplementedError) as exc:
        _orbit(blocks, {"orbit.b.b": {"initval": 0.5}})

    # Assert
    msg = str(exc.value)
    assert "planet.<name>.ar" in msg
    assert "planet.<name>.b" in msg
    # ...and at the chord, which is what an orbit-side impact-parameter
    # constraint usually means now that fitchord exists.
    assert "fitchord" in msg


@pytest.mark.parametrize(
    "user_params",
    [
        {},
        {"orbit.0.tc": {"initval": 2460000.0}},
        {"orbit.0.period": {"initval": 3.0}},
        {"orbit.b.secosw": {"initval": 0.01}},
    ],
    ids=["empty", "tc", "period", "secosw"],
)
def test_ordinary_orbit_params_do_not_raise(user_params):
    """
    Given ordinary orbit entries -- including ones whose instance is named
      "b", the spelling closest to the dead `b` parameter,
    When the Orbit component is constructed,
    Then nothing raises: only the last dotted segment selects a parameter,
      so an orbit CALLED "b" is untouched by the guard.
    """
    # Arrange / Act
    orbit = _orbit([{"name": "b"}], user_params)

    # Assert
    assert orbit.fitvcve == [False]
