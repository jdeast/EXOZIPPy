# tests/test_multiseed_config.py
"""Tests for list-valued initvals feeding the relaxation engine (P4 layer a).

A `params.yaml` entry of the form `initval: [v0, v1, ...]` asks the relaxation
engine to solve K complete, mutually-consistent start points inside ONE
prepare() call (sharing one symbol environment -- see the module docstring
in config.py's finalize_user_params and the project's known relaxation-engine
cross-build nondeterminism note). Bounds/scales resolve once, from seed 0.
"""
import numpy as np
import pytest

from exozippy.config import ConfigManager


def test_list_initval_produces_k_solved_seeds_with_shared_bounds():
    """
    Given a star.mass initval list of length 2,
    When ConfigManager.finalize_user_params runs,
    Then it solves 2 mutually-consistent seed points (mass ~ list values via
    the mass = 10**logmass relation) and the logmass BOUNDS are identical
    across seeds (only the start position varies).
    """
    system_config = {"star": [{"name": "Lens"}]}
    user_params = {
        "star.Lens.mass": {"initval": [0.3, 0.7], "init_scale": 0.01},
    }

    cm = ConfigManager(user_params, system_config=system_config)
    cm.finalize_user_params()

    assert cm.seed_resolved is not None
    assert len(cm.seed_resolved) == 2

    # Each seed's solved logmass must back out to the corresponding mass value.
    logmass_0 = cm.seed_resolved[0]["star.0.logmass"]
    logmass_1 = cm.seed_resolved[1]["star.0.logmass"]
    assert np.isclose(10 ** logmass_0, 0.3, rtol=1e-3)
    assert np.isclose(10 ** logmass_1, 0.7, rtol=1e-3)
    assert not np.isclose(logmass_0, logmass_1)

    # Seed 0 remains the canonical single start injected into user_params.
    assert np.isclose(cm.user_params["star.0.mass"]["initval"], 0.3, rtol=1e-3)

    # Bounds resolve once, from seed 0, regardless of which seed is queried:
    # resolve() only ever sees seed 0's initval (list[0]), so repeated calls
    # return the same lower/upper for logmass.
    cfg = cm.resolve("star", "logmass", element=0)
    assert cfg["lower"] is not None and cfg["upper"] is not None
    lower0, upper0 = cfg["lower"][0], cfg["upper"][0]
    cfg_again = cm.resolve("star", "logmass", element=0)
    assert cfg_again["lower"][0] == lower0
    assert cfg_again["upper"][0] == upper0


def test_length_one_list_broadcasts_and_is_equivalent_to_scalar():
    """
    Given an initval list of length 1,
    When finalize_user_params runs,
    Then it behaves like an ordinary scalar initval (K==1, no seed_resolved).
    """
    system_config = {"star": [{"name": "Lens"}]}
    user_params = {"star.Lens.mass": {"initval": [0.42], "init_scale": 0.01}}

    cm = ConfigManager(user_params, system_config=system_config)
    cm.finalize_user_params()

    assert cm.seed_resolved is None
    assert np.isclose(cm.user_params["star.0.mass"]["initval"], 0.42, rtol=1e-3)


def test_no_list_initvals_leaves_seed_resolved_none():
    """
    Given an ordinary scalar-only params file (the common case),
    When finalize_user_params runs,
    Then seed_resolved stays None -- multi-seed sampling is fully opt-in and
    must not alter the legacy single-start code path.
    """
    system_config = {"star": [{"name": "Lens"}]}
    user_params = {"star.Lens.mass": {"initval": 0.5}}

    cm = ConfigManager(user_params, system_config=system_config)
    cm.finalize_user_params()

    assert cm.seed_resolved is None


def test_mismatched_seed_list_lengths_raises():
    """
    Given two initval lists of different lengths (neither length 1),
    When finalize_user_params runs,
    Then it raises ValueError rather than silently truncating/broadcasting.
    """
    system_config = {"star": [{"name": "Lens"}, {"name": "Source"}]}
    user_params = {
        "star.Lens.mass": {"initval": [0.3, 0.5, 0.7]},
        "star.Source.mass": {"initval": [0.1, 0.2]},
    }

    cm = ConfigManager(user_params, system_config=system_config)
    with pytest.raises(ValueError, match="[Ii]nconsistent seed count"):
        cm.finalize_user_params()


def test_seed_hints_and_user_list_both_present_prefer_user_list():
    """
    Given both a component seed_hint_sets entry AND a user initval list for the
    same parameter,
    When finalize_user_params runs,
    Then the user's explicit list wins (matches the RANK_USER > seed-hint-rank
    provenance rule: an explicit user list always overrides a component's
    data-driven seed hint).
    """
    system_config = {"star": [{"name": "Lens"}]}
    user_params = {"star.Lens.mass": {"initval": [0.3, 0.7]}}

    cm = ConfigManager(user_params, system_config=system_config)
    # Component-style seed hint disagreeing with the user's list.
    cm.add_seed_hints([{"star.Lens.mass": 0.9}, {"star.Lens.mass": 0.95}])
    cm.finalize_user_params()

    assert cm.seed_resolved is not None
    logmass_0 = cm.seed_resolved[0]["star.0.logmass"]
    logmass_1 = cm.seed_resolved[1]["star.0.logmass"]
    assert np.isclose(10 ** logmass_0, 0.3, rtol=1e-3)
    assert np.isclose(10 ** logmass_1, 0.7, rtol=1e-3)
