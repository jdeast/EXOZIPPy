"""The star side of the `evolutionarymodel` component.

The claim this file was written to pin is that landing an evolutionary model
requires **no edit to the star component**: the two coordinates a track is
indexed by (`initfeh`, `eep`) have entries in star/defaults.yaml, the
per-star `mist:`/`parsec:` switches are parsed, and the
`in_system("evolutionarymodel")` branch of `Star.register_parameters`
declares all three of age/initfeh/eep.  The component has now landed and
that held -- star.py is untouched -- so these tests exercise the branch
through the real component rather than through a faked topology.

They still run against a SYNTHETIC grid (`conftest.write_synthetic_mist_grid`)
rather than the shipped one: the packaged MIST parquet is ~130 MB and
gitignored, so no test may depend on it.

Two claims are pinned about the interaction rather than about either side
alone.  First, the bounds star/defaults.yaml carries for initfeh and eep are
PLACEHOLDERS: the component narrows them to whatever its grid actually spans,
through `ConfigManager.add_override`.  Second, the branch is inert without an
`evolutionarymodel:` block, which is why adding the two defaults.yaml entries
moved no shipped fit -- also pinned by tests/test_examples_prepare.py over
every shipped config.
"""

import pytest

from conftest import write_synthetic_mist_grid
from exozippy.components.star import Star
from exozippy.config import ConfigManager
from exozippy.system import System

# The values these two entries carry in star/defaults.yaml, restated here
# so a silent edit to either side is a test failure rather than a shrug.
# `lower`/`upper` are deliberately absent: they are placeholders the
# component overrides to its grid's real extent -- see
# test_the_component_narrows_the_placeholder_bounds_to_its_grid.
TRACK_DEFAULTS = {
    "initfeh": {"initval": 0.0, "init_scale": 0.1, "unit": "dex"},
    "eep": {"initval": 354.1661, "init_scale": 10.0, "unit": ""},
}

# The synthetic grid's axis extents (conftest.write_synthetic_mist_grid).
GRID_INITFEH = (-0.5, 0.5)
GRID_EEP = (1.0, 807.0)


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    """A tiny MIST grid, shared by every test in this module."""
    return write_synthetic_mist_grid(
        tmp_path_factory.mktemp("mist_models")
    )


def _config(stars, evolutionary, model_root=None, blocks=None):
    """A minimal star-only system config, with or without the component."""
    config = {"sampler": {"draws": 10}, "star": stars}
    if evolutionary:
        if blocks is None:
            blocks = [{"star": stars[0].get("name", 0)}]
        config["evolutionarymodel"] = [
            {**b, "model_root": model_root} for b in blocks
        ]
    return config


def _prepared(stars, evolutionary=True, model_root=None, blocks=None):
    """A prepared System over `stars`, optionally with the component."""
    system = System(
        _config(stars, evolutionary, model_root, blocks), {}
    )
    system.prepare()
    return system


# ----------------------------------------------------------------------
# The branch is ready
# ----------------------------------------------------------------------
def test_manifest_declares_age_initfeh_and_eep(model_root):
    """
    Given a system whose topology names an evolutionarymodel component,
    When the star component registers its parameters,
    Then its manifest declares all three track coordinates -- the present-day
      age plus the (initial metallicity, EEP) pair a track is indexed by.
    """
    # Arrange / Act
    star = _prepared([{"name": "A"}], model_root=model_root).star

    # Assert
    assert {"age", "initfeh", "eep"} <= set(star.manifest)


def test_track_parameters_get_the_same_mask_as_age(model_root):
    """
    Given two stars, only one of which opted into an evolutionary model,
    When the star component registers its parameters,
    Then initfeh and eep carry the same per-star mask age does -- the three
      are one coordinate system and cannot be masked apart.
    """
    # Arrange
    stars = [{"name": "A"}, {"name": "B", "mist": False}]

    # Act
    manifest = _prepared(stars, model_root=model_root).star.manifest

    # Assert
    assert manifest["age"]["mask"] == [True, False]
    assert manifest["initfeh"]["mask"] == manifest["age"]["mask"]
    assert manifest["eep"]["mask"] == manifest["age"]["mask"]


def test_a_star_with_no_track_has_no_track_coordinates():
    """
    Given two stars, only one of which opted into an evolutionary model,
    When the model is built,
    Then the opted-in star's initfeh/eep are sampled and the other star's are
      not parameters at all: inactive, so nothing samples them and nothing
      reports them.

    The mask was declared and never read until per-element roles landed, so the
    star that opted OUT used to get two free, likelihood-free dimensions -- and
    a table row for each, reporting a quantity that describes a track it has
    none of.
    """
    # Arrange
    system = _prepared([{"name": "A"}, {"name": "B", "mist": False}])

    # Act
    model = system.build_model()

    # Assert
    for name in ("initfeh", "eep", "age"):
        param = getattr(system.star, name)
        assert param.is_active.tolist() == [True, False], name
        assert param.is_sampled.tolist() == [True, False], name
        assert param.element_is_active(0) and not param.element_is_active(1)
    # One raw element each (star A only), not two.
    raw = {v.name: v for v in model.free_RVs}
    assert raw["star.initfeh_raw"].type.shape == (1,)
    assert raw["star.eep_raw"].type.shape == (1,)


def test_no_star_opting_in_declares_no_track_coordinates():
    """
    Given an evolutionarymodel topology in which NO star opted in,
    When the star component registers its parameters,
    Then the three track coordinates are not declared at all.

    A wholly inactive vector would be a Parameter nothing samples, nothing
    reads and nothing reports -- so it should not exist, exactly as it does not
    when the block is absent.
    """
    stars = [{"name": "A", "mist": False}, {"name": "B", "mist": False}]

    manifest = _prepared(stars).star.manifest

    assert not {"age", "initfeh", "eep"} & set(manifest)


def test_a_premature_block_warns_that_nothing_reads_the_track(caplog):
    """
    Given an evolutionarymodel block that no component backs,
    When the star component registers its parameters,
    Then it warns, naming the opted-in stars.

    Review 3.8.2: the branch is driven by the config KEY, so it fires for a
    premature block, and its coordinates are then sampled with nothing reading
    them.  The unrecognized-key warning System already emits does not say that.
    """
    with caplog.at_level("WARNING", logger="exozippy"):
        _prepared([{"name": "A"}, {"name": "B", "mist": False}])

    hits = [
        r.getMessage()
        for r in caplog.records
        if "no such component is registered" in r.getMessage()
    ]
    assert len(hits) == 1
    assert "A" in hits[0] and "eep" in hits[0]


@pytest.mark.parametrize("name", sorted(TRACK_DEFAULTS))
def test_track_parameters_resolve_to_their_defaults(name, model_root):
    """
    Given an evolutionarymodel block,
    When the model is built and the new Parameters are materialized,
    Then each resolves to its star/defaults.yaml start, scale and unit -- the
      manifest entry and the defaults.yaml entry agree, so the component
      finds real numbers rather than a KeyError.

    `eep`'s start is the defaults.yaml one only because the component's own
    data-driven seed is a HINT, which loses to nothing here: the star's
    teff/radius/feh/age are all at their defaults too, so any EEP the seed
    search picks is as good as another.  Bounds are checked separately, in
    the narrowing test below.
    """
    # Arrange
    system = _prepared([{"name": "A"}], model_root=model_root)
    expected = TRACK_DEFAULTS[name]

    # Act
    system.build_model()
    param = getattr(system.star, name)

    # Assert
    assert param.init_scale[0] == pytest.approx(expected["init_scale"])
    assert str(param.unit[0]) == expected["unit"]
    # Sampled, not derived: an evolutionary model reads a track AT (initfeh,
    # eep), so these are the free coordinates and `age` is what it returns.
    assert param.is_sampled
    assert not param.is_derived


def test_the_component_narrows_the_placeholder_bounds_to_its_grid(model_root):
    """
    Given star/defaults.yaml's deliberately wide initfeh/eep bounds,
    When an evolutionarymodel block loads its grid,
    Then both are narrowed to the grid's own axis extent -- the interpolator
      extrapolates meaninglessly past its edges, so this is a validity limit,
      and it arrives through add_override rather than as a user param.
    """
    # Arrange
    system = _prepared([{"name": "A"}], model_root=model_root)

    # Act
    system.build_model()

    # Assert
    assert system.star.initfeh.lower[0] == pytest.approx(GRID_INITFEH[0])
    assert system.star.initfeh.upper[0] == pytest.approx(GRID_INITFEH[1])
    assert system.star.eep.lower[0] == pytest.approx(GRID_EEP[0])
    assert system.star.eep.upper[0] == pytest.approx(GRID_EEP[1])
    # add_override, not user_params: the ledger must not report these bounds
    # as the user's own (see EvolutionaryModel._inject_grid_bounds).  Note
    # user_params is NOT empty of these paths -- finalize_user_params injects
    # the engine's own solution back under the index form -- so the claim is
    # specifically that no BOUND arrived that way.
    overrides = system.config_manager.param_overrides
    assert overrides["star.A.initfeh"] == {
        "lower": GRID_INITFEH[0],
        "upper": GRID_INITFEH[1],
    }
    assert overrides["star.A.eep"] == {
        "lower": GRID_EEP[0],
        "upper": GRID_EEP[1],
    }
    assert not any(
        {"lower", "upper"} & set(entry)
        for key, entry in system.config_manager.user_params.items()
        if isinstance(entry, dict) and key.split(".")[-1] in ("initfeh", "eep")
    )


def test_track_parameters_become_free_random_variables(model_root):
    """
    Given a star an evolutionarymodel block names,
    When the PyMC model is built,
    Then initfeh and eep are sampled -- the branch produces a working model,
      not merely a manifest entry.
    """
    # Arrange
    system = _prepared([{"name": "A"}], model_root=model_root)

    # Act
    model = system.build_model()

    # Assert
    raw = {v.name for v in model.free_RVs}
    assert "star.initfeh_raw" in raw
    assert "star.eep_raw" in raw


def test_a_star_no_block_names_gets_its_track_parameters_pinned(model_root):
    """
    Given two stars and one evolutionarymodel block,
    When the PyMC model is built,
    Then the unmodeled star's initfeh/eep/age are pinned rather than left as
      free dimensions no likelihood term reads.

    Star materializes all three for EVERY star as soon as the block exists
    -- its per-star `{"mask": ...}` is the declared-but-unconsumed manifest
    field -- so without the pin the second star contributes three dead
    parameters.  See EvolutionaryModel._pin_unmodeled_stars.
    """
    # Arrange
    stars = [{"name": "A"}, {"name": "B"}]

    # Act
    system = _prepared(
        stars, model_root=model_root, blocks=[{"star": "A"}]
    )
    system.build_model()

    # Assert
    for name in ("initfeh", "eep", "age"):
        param = getattr(system.star, name)
        assert param.is_sampled[0], f"star.A.{name} should be sampled"
        assert not param.is_sampled[1], f"star.B.{name} should be pinned"
        assert param.sigma[1] == 0.0


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
