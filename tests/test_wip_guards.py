"""Half-built features must refuse loudly, not accept-and-ignore.

The `orbit:` block has a config surface that was parsed and then silently
dropped on the floor (code review 2026-08-08, item 5.11): `fitvcve: true`
selects `defaults.yaml` expressions naming undefined physics functions, and
the `vcve`/`b` parameter blocks are in no manifest at all (`b`'s deps even
name a nonexistent `orbit.ar`).

It is not being implemented or deleted here.  What these tests pin is that
a user who asks for it finds out immediately, from a message that names the
key they set -- and, just as importantly, that the ordinary configs which
merely carry its defaults still build.

(The `star:` half of that review item -- `sedfile:`, `mist:`, `parsec:` --
was reverted: `sedfile` is deleted outright and the evolutionary-model
switches are ungated, because the star component is now ready for an
`evolutionarymodel` component to land against unchanged.  See
tests/test_star_evolutionary_model.py.)
"""

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
def test_orbit_fitvcve_raises_and_names_the_missing_physics():
    """
    Given an orbit block asking for the V_c/V_e parameterization,
    When the Orbit component is constructed,
    Then it raises NotImplementedError naming the key, the two undefined
      physics functions the from_vcve expressions call, and the manifest
      `mask` field that is the real blocker.
    """
    # Arrange
    blocks = [{"name": "b", "fitvcve": True}]

    # Act
    with pytest.raises(NotImplementedError) as exc:
        _orbit(blocks)

    # Assert
    msg = str(exc.value)
    assert "'fitvcve: true'" in msg
    assert "orbit.b" in msg
    assert "calc_ecc_from_vcve" in msg
    assert "calc_omega_from_vcve" in msg
    assert "mask" in msg


def test_orbit_fitvcve_still_raises_if_set_after_construction():
    """
    Given an orbit whose fitvcve flag is flipped after __init__,
    When it registers its parameters,
    Then stage 2 raises too -- otherwise secosw/sesinw/cosi would be
      silently masked out of the manifest with nothing put in their place.
    """
    # Arrange
    orbit = _orbit([{"name": "b"}], {"orbit.0.tc": {"initval": 2460000.0}})
    orbit.fitvcve = [True]

    # Act / Assert
    with pytest.raises(NotImplementedError, match="fitvcve"):
        orbit.register_parameters(system=None)


@pytest.mark.parametrize(
    "path",
    ["orbit.b.vcve", "orbit.0.vcve", "orbit.vcve"],
    ids=["by-name", "by-index", "broadcast"],
)
def test_orbit_vcve_user_param_raises(path):
    """
    Given a params-file entry naming the orbit's vcve, in any of the three
      accepted spellings,
    When the Orbit component is constructed,
    Then it raises NotImplementedError naming the entry -- an unknown
      parameter path is otherwise silently ignored.
    """
    # Arrange
    blocks = [{"name": "b"}]

    # Act
    with pytest.raises(NotImplementedError) as exc:
        _orbit(blocks, {path: {"initval": 0.5}})

    # Assert
    msg = str(exc.value)
    assert "vcve" in msg
    assert "fitvcve" in msg
    assert "calc_ecc_from_vcve" in msg
    assert "calc_omega_from_vcve" in msg


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
