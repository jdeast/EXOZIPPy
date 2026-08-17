"""Half-built features must refuse loudly, not accept-and-ignore.

The `orbit:` block had a config surface that was parsed and then silently
dropped on the floor (code review 2026-08-08, item 5.11): `fitvcve: true`
selected `defaults.yaml` expressions naming undefined physics functions, and
the `vcve`/`b` parameter blocks were in no manifest at all (`b`'s deps even
name a nonexistent `orbit.ar`).

Half of it is implemented now.  `fitvcve` works (review 8.8.3: the V_c/V_e
parameterization, with the likelihood marginalized over both roots of its
quadratic inversion), so what these tests pin about it is the opposite of what
they used to: that it does NOT raise and that a params entry naming `vcve` is
honored.  The CHORD half -- `fitchord`, `cosi`'s `from_b`, the orbit-level `b`
-- is still unbuilt, and what the guard now covers is that a user who asks for
THAT finds out immediately, from a message naming the key they set, while the
ordinary configs which merely carry its defaults still build.

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


def test_orbit_fitchord_raises_and_names_the_missing_physics():
    """
    Given an orbit block asking for the transit-chord parameterization,
    When the Orbit component is constructed,
    Then it raises NotImplementedError naming the undefined physics function,
      and points at the V_c/V_e half that does work.

    The paper pairs V_c/V_e with fitting the chord; only the eccentricity half
    is implemented, and the guard is the feature until the other lands.
    """
    with pytest.raises(NotImplementedError) as exc:
        _orbit([{"name": "b", "fitchord": True}])

    msg = str(exc.value)
    assert "fitchord" in msg
    assert "orbit.b" in msg
    assert "calc_cosi_from_b" in msg
    assert "fitvcve" in msg


def test_fitchord_following_fitvcve_does_not_raise():
    """
    Given `fitvcve: true` and no explicit `fitchord`,
    When the component is constructed,
    Then it does not raise, and fitchord merely FOLLOWS fitvcve.

    JDE's coupling rule: `fitvcve: false` forces `fitchord: false` unless
    fitchord is set separately.  A followed value was not asked for, so it must
    not trip the guard on the unimplemented half -- otherwise turning on the
    eccentricity parameterization would be impossible until the chord lands.
    """
    orbit = _orbit([{"name": "b", "fitvcve": True}])

    assert orbit.fitchord == [True]


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
    Given a params-file entry on the dead orbit-level impact parameter,
    When the Orbit component is constructed,
    Then it raises NotImplementedError explaining that the deps name an
      `orbit.ar` that does not exist, and points at the live `planet.<n>.b`.
    """
    # Arrange
    blocks = [{"name": "b"}]

    # Act
    with pytest.raises(NotImplementedError) as exc:
        _orbit(blocks, {"orbit.b.b": {"initval": 0.5}})

    # Assert
    msg = str(exc.value)
    assert "orbit.ar" in msg
    assert "planet.<name>.b" in msg
    assert "calc_cosi_from_b" in msg


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
