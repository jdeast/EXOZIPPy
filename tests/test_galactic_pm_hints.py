"""Proper-motion start values come from the galactic model, not from a guess.

Before this, ``star.pm_ra``/``star.pm_dec`` sat at their defaults.yaml value and
the relaxation engine let ONE of them absorb whatever the seeded ``t_E``
implied, leaving the other at the default.  That made the direction of relative
proper motion arbitrary: the pair
``mu_rel_mag**2 = mu_ra_rel**2 + mu_dec_rel**2`` is one equation in two
unknowns, and the MMEXOFAST seed carries no ``pi_E`` to break it (issue #93).

Seeding both components at the kinematic prior's mean closes that hole.  The
interesting part is what the provenance ranking then does on its own: the
proper motions and the MMEXOFAST ``t_E`` are both PRECEDENCE_DERIVED_DATA, so
``t_E = theta_E / |mu_rel_geo|`` is over-determined, and Condition B rewrites
its lowest-rank symbol -- ``theta_E``, through the lens mass (defaults.yaml,
rank 20) and distance (rank 25).  So the measured ``t_E`` survives and the lens
mass absorbs the difference, which is the standard microlensing chain rather
than a special case anyone had to code.
"""

import copy
import os
import pathlib
import shutil
import tempfile

import numpy as np
import pytest
import yaml

from exozippy.components.galacticmodel.physics import (
    POPULATIONS,
    expected_proper_motion,
)
from exozippy.system import System

EXAMPLE_DIR = pathlib.Path(__file__).parent / ".." / "examples" / "DC2018_128"

pytestmark = pytest.mark.slow


def _inputs():
    with open(EXAMPLE_DIR / "DC2018_128.yaml") as f:
        config = yaml.safe_load(f)
    with open(EXAMPLE_DIR / "DC2018_128.params.yaml") as f:
        user_params = yaml.safe_load(f)
    for entry in config.get("planet", []):
        entry.setdefault("mass_parameterization", "linear")
    return config, user_params


def _prepared(config, user_params):
    work = pathlib.Path(tempfile.mkdtemp()) / "DC2018_128"
    shutil.copytree(
        EXAMPLE_DIR,
        work,
        ignore=shutil.ignore_patterns("fitresults", ".#*", "#*#"),
    )
    cwd = os.getcwd()
    os.chdir(work)
    try:
        system = System(copy.deepcopy(config), copy.deepcopy(user_params))
        system.prepare()
        system.build_model()
        return {p.label: p for p in system.get_all_parameters()}
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def params():
    return _prepared(*_inputs())


def test_proper_motions_are_seeded_at_the_galactic_prior_mean(params):
    """
    Given a microlensing config with a galacticmodel block and known RA/Dec,
    When the relaxation engine resolves start values,
    Then both proper-motion components equal the prior's mean for that
    line of sight, distance and population.
    """
    # Arrange: the same expectation the seeding claims to use.  DC2018_128 is
    # at (267.595, -28.982) deg; lens seeded as thin disk at 4 kpc, source as
    # bulge at 8 kpc, matching the distance hints in Lens.register_parameters.
    ra, dec = np.radians(267.595), np.radians(-28.982)
    lens_pm = expected_proper_motion(ra, dec, 4000.0, "thin_disk")
    source_pm = expected_proper_motion(ra, dec, 8000.0, "bulge")

    # Act
    pm_ra = np.atleast_1d(np.asarray(params["star.pm_ra"].initval, float))
    pm_dec = np.atleast_1d(np.asarray(params["star.pm_dec"].initval, float))

    # Assert
    assert pm_ra[0] == pytest.approx(lens_pm[0], rel=1e-6)
    assert pm_dec[0] == pytest.approx(lens_pm[1], rel=1e-6)
    assert pm_ra[1] == pytest.approx(source_pm[0], rel=1e-6)
    assert pm_dec[1] == pytest.approx(source_pm[1], rel=1e-6)


def test_neither_pm_component_is_left_at_zero(params):
    """
    Given the seeded start values,
    When mu_rel and pi_E are formed from them,
    Then no component is (numerically) zero.

    The old seeding put the whole magnitude in one component and left the other
    at 0 exactly -- so pi_E_N or pi_E_E started at a cusp of its normalization.
    """
    # Act / Assert
    for label in (
        "lens.mu_ra_rel",
        "lens.mu_dec_rel",
        "lens.pi_E_N",
        "lens.pi_E_E",
    ):
        value = float(
            np.atleast_1d(np.asarray(params[label].initval, float))[0]
        )
        assert abs(value) > 1e-6, f"{label} was seeded at ~0 ({value!r})"


def test_measured_t_E_survives_and_theta_E_yields(params):
    """
    Given proper motions at PRECEDENCE_DERIVED_DATA that disagree with the seeded t_E,
    When Condition B rewrites the over-determined relation,
    Then t_E keeps the MMEXOFAST value and theta_E is what moves.

    This is the load-bearing assertion: it is the difference between "the
    provenance ranking sorts this out" and "the measured timescale silently got
    overwritten by a prior".
    """
    # Arrange: t_E as MMEXOFAST measured it, in days.
    import json

    with open(EXAMPLE_DIR / "mmexofast.json") as f:
        mmx = json.load(f)
    seeded_t_E = float(mmx["fits"][0]["parameters"]["t_E"])

    # Act
    t_E = float(
        np.atleast_1d(np.asarray(params["lens.t_E"].initval, float))[0]
    )
    theta_E = float(
        np.atleast_1d(np.asarray(params["lens.theta_E"].initval, float))[0]
    )
    mu_rel = float(
        np.atleast_1d(np.asarray(params["lens.mu_rel_mag"].initval, float))[0]
    )

    # Assert
    assert t_E == pytest.approx(seeded_t_E, rel=1e-6), (
        "t_E moved off the MMEXOFAST seed; the proper-motion hints are "
        "outranking a measured quantity."
    )
    # theta_E is the symbol that absorbed the difference, so it must be
    # consistent with t_E and the prior's mu_rel rather than with the old
    # mass/distance guess (which gave 0.5677 mas).
    assert theta_E == pytest.approx(t_E / 365.25 * mu_rel, rel=0.05)


def test_unknown_population_is_rejected():
    """
    Given a population name the kinematic prior does not define,
    When an expected proper motion is requested,
    Then it raises rather than silently returning a disk velocity.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="unknown population"):
        expected_proper_motion(0.0, 0.0, 4000.0, "halo")
    assert "thin_disk" in POPULATIONS


def test_seeds_are_skipped_when_parallax_is_already_measured():
    """
    Given a config that supplies pi_E_N/pi_E_E (a measured parallax vector),
    When start values are resolved,
    Then the proper motions are NOT seeded from the prior.

    pi_E is parallel to mu_rel, so a measured pi_E already fixes the direction.
    Seeding the prior's direction on top of it contradicts the measurement:
    on examples/ob140939 (Yee+2015 pi_E) doing so took chi2/N at the seed from
    3.04 to 179.1.
    """
    # Arrange
    config, user_params = _inputs()
    user_params["lens.Lens.pi_E_N"] = {"initval": -0.2}
    user_params["lens.Lens.pi_E_E"] = {"initval": 0.1}

    ra, dec = np.radians(267.595), np.radians(-28.982)
    prior_pm_dec = expected_proper_motion(ra, dec, 4000.0, "thin_disk")[1]

    # Act
    params = _prepared(config, user_params)

    # Assert: whatever the engine derives from the measured pi_E, it is NOT the
    # prior's mean.  (It need not be the bare default either -- deriving the
    # proper motion FROM the measurement is the desired outcome; the point is
    # that the prior did not overwrite it.)
    pm_dec = np.atleast_1d(np.asarray(params["star.pm_dec"].initval, float))
    assert pm_dec[0] != pytest.approx(prior_pm_dec, rel=1e-3), (
        "the galactic-model mean overrode a measured pi_E direction"
    )


def test_seeds_are_skipped_for_a_multi_source_event():
    """
    Given more than one source body,
    When start values are resolved,
    Then the proper motions are NOT seeded from the prior.

    Every source would take the same bulge mean, tying their mu_rel together;
    a resolved binary source distinguishes them.  On examples/ob161003 (two
    sources, t_E and rho pinned for each) seeding took chi2/N from 1.72 to 3.9.
    """
    # Arrange: add a second source body to the lens block.
    config, user_params = _inputs()
    lens_block = (
        config["lens"][0]
        if isinstance(config["lens"], list)
        else config["lens"]
    )
    sources = lens_block.get("sources")
    if not sources:
        pytest.skip("example does not use the explicit sources: list")
    star_entries = config["star"]
    star_entries.append(
        copy.deepcopy(star_entries[int(sources[0].split(".")[1])])
    )
    star_entries[-1]["name"] = "SourceB"
    lens_block["sources"] = list(sources) + [f"star.{len(star_entries) - 1}"]

    # Act
    params = _prepared(config, user_params)

    # Assert
    pm_dec = np.atleast_1d(np.asarray(params["star.pm_dec"].initval, float))
    assert pm_dec[0] == pytest.approx(-3.0)


def test_seeds_are_skipped_without_user_coordinates():
    """
    Given a config whose RA/Dec are left at the defaults.yaml placeholder,
    When start values are resolved,
    Then the proper motions fall back to defaults rather than being seeded off
    a placeholder direction.

    The prior's mean is a function of direction, so seeding from a placeholder
    would be worse than not seeding.
    """
    # Arrange
    config, user_params = _inputs()
    for key in list(user_params):
        if key.endswith(".ra") or key.endswith(".dec"):
            del user_params[key]

    # Act
    params = _prepared(config, user_params)

    # Assert: back to the defaults.yaml value, not a galactic-model number.
    pm_ra = np.atleast_1d(np.asarray(params["star.pm_ra"].initval, float))
    assert pm_ra[1] == pytest.approx(-3.0)
