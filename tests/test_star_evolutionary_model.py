"""The star component is ready for an `evolutionarymodel` component.

No such component ships.  The point of these tests is that landing one
should require **no edit to the star component**: the two coordinates an
evolutionary track is indexed by (`initfeh`, `eep`) have entries in
star/defaults.yaml, the per-star `mist:`/`parsec:` switches are parsed, and
the `in_system("evolutionarymodel")` branch of `Star.register_parameters`
declares all three of age/initfeh/eep.  Nothing else in the suite can
exercise that branch, so it would otherwise rot unobserved -- which is how
it acquired an `initfeh` that had no defaults.yaml entry in the first place.

Topology is faked the way the branch itself detects it: `in_system` asks
`hasattr(system, name) or name in system.config`, so a plain
`evolutionarymodel:` key in the config is enough.  `System` warns that the
key matches no registered component and carries on, which is exactly the
state of the world until the component exists.

The complementary claim -- that the branch is inert, so adding the two
defaults.yaml entries moved no shipped fit -- is pinned here too, and by
tests/test_examples_prepare.py over every shipped config.
"""

import pytest

from exozippy.components.star import Star
from exozippy.config import ConfigManager
from exozippy.system import System

# The values these two entries carry in star/defaults.yaml, restated here
# so a silent edit to either side is a test failure rather than a shrug.
TRACK_DEFAULTS = {
    "initfeh": {
        "initval": 0.0,
        "init_scale": 0.1,
        "lower": -4.0,
        "upper": 0.5,
        "unit": "dex",
    },
    "eep": {
        "initval": 354.1661,
        "init_scale": 10.0,
        "lower": 0.0,
        "upper": 1710.0,
        "unit": "",
    },
}


def _config(stars, evolutionary):
    """A minimal star-only system config, with or without the fake block."""
    config = {"sampler": {"draws": 10}, "star": stars}
    if evolutionary:
        config["evolutionarymodel"] = {}
    return config


def _prepared(stars, evolutionary=True):
    """A prepared System over `stars`, optionally with the faked topology."""
    system = System(_config(stars, evolutionary), {})
    system.prepare()
    return system


# ----------------------------------------------------------------------
# The branch is ready
# ----------------------------------------------------------------------
def test_manifest_declares_age_initfeh_and_eep():
    """
    Given a system whose topology names an evolutionarymodel component,
    When the star component registers its parameters,
    Then its manifest declares all three track coordinates -- the present-day
      age plus the (initial metallicity, EEP) pair a track is indexed by.
    """
    # Arrange / Act
    star = _prepared([{"name": "A"}]).star

    # Assert
    assert {"age", "initfeh", "eep"} <= set(star.manifest)


def test_track_parameters_get_the_same_mask_as_age():
    """
    Given two stars, only one of which opted into an evolutionary model,
    When the star component registers its parameters,
    Then initfeh and eep carry the same per-star mask age does -- the three
      are one coordinate system and cannot be masked apart.
    """
    # Arrange
    stars = [{"name": "A"}, {"name": "B", "mist": False}]

    # Act
    manifest = _prepared(stars).star.manifest

    # Assert
    assert manifest["age"]["mask"] == [True, False]
    assert manifest["initfeh"]["mask"] == manifest["age"]["mask"]
    assert manifest["eep"]["mask"] == manifest["age"]["mask"]


@pytest.mark.parametrize("name", sorted(TRACK_DEFAULTS))
def test_track_parameters_resolve_to_their_defaults(name):
    """
    Given the faked evolutionarymodel topology,
    When the model is built and the new Parameters are materialized,
    Then each resolves to its star/defaults.yaml start, scale and bounds --
      the manifest entry and the defaults.yaml entry agree, so the component
      that lands will find real numbers rather than a KeyError.
    """
    # Arrange
    system = _prepared([{"name": "A"}])
    expected = TRACK_DEFAULTS[name]

    # Act
    system.build_model()
    param = getattr(system.star, name)

    # Assert
    assert param.initval[0] == pytest.approx(expected["initval"])
    assert param.init_scale[0] == pytest.approx(expected["init_scale"])
    assert param.lower[0] == pytest.approx(expected["lower"])
    assert param.upper[0] == pytest.approx(expected["upper"])
    assert str(param.unit[0]) == expected["unit"]
    # Sampled, not derived: an evolutionary model reads a track AT (initfeh,
    # eep), so these are the free coordinates and `age` is what it returns.
    assert param.is_sampled
    assert not param.is_derived


def test_track_parameters_become_free_random_variables():
    """
    Given the faked topology,
    When the PyMC model is built,
    Then initfeh and eep are sampled -- the branch produces a working model,
      not merely a manifest entry.
    """
    # Arrange
    system = _prepared([{"name": "A"}])

    # Act
    model = system.build_model()

    # Assert
    raw = {v.name for v in model.free_RVs}
    assert "star.initfeh_raw" in raw
    assert "star.eep_raw" in raw


# ----------------------------------------------------------------------
# ...and inert until it fires
# ----------------------------------------------------------------------
def test_branch_is_inert_without_an_evolutionarymodel_block():
    """
    Given the same stars with no evolutionarymodel in the topology,
    When the model is built,
    Then neither initfeh nor eep is declared or materialized -- which is why
      adding them to star/defaults.yaml moves no shipped fit.
    """
    # Arrange
    system = _prepared([{"name": "A"}], evolutionary=False)

    # Act
    model = system.build_model()

    # Assert
    assert "initfeh" not in system.star.manifest
    assert "eep" not in system.star.manifest
    raw = {v.name for v in model.free_RVs}
    assert not {"star.initfeh_raw", "star.eep_raw"} & raw


# ----------------------------------------------------------------------
# mist: / parsec: semantics
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "block,mist,parsec",
    [
        ({"name": "A"}, True, False),
        ({"name": "A", "mist": True}, True, False),
        ({"name": "A", "mist": False}, False, False),
        ({"name": "A", "parsec": True}, True, True),
        ({"name": "A", "mist": False, "parsec": True}, False, True),
    ],
    ids=["absent", "mist-true", "mist-false", "parsec-true", "parsec-only"],
)
def test_evolutionary_model_switches_parse_and_never_raise(
    block, mist, parsec
):
    """
    Given any spelling of the per-star evolutionary-model switches,
    When the Star component is constructed,
    Then it parses them and nothing raises: `mist`'s historical default is
      True, so an absent key means opted in, and a config written today
      against the evolutionary model that has not landed yet must keep
      working unchanged when it does.
    """
    # Arrange / Act
    star = Star([block], ConfigManager({}))

    # Assert
    assert star.mist == [mist]
    assert star.parsec == [parsec]
