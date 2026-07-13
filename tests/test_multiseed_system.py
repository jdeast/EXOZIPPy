# tests/test_multiseed_system.py
"""End-to-end tests for System.get_raw_starts (P4 layers a+c glue).

Verifies that a multi-seed params file produces one raw start per seed, that
a seed whose solved start violates a hard bound is skipped (not clipped -- a
clipped start sits in no posterior basin), and that seed provenance is
available for the sampler to round-robin against.
"""
import logging

import numpy as np
import pytest

from exozippy.system import System


def _base_user_params():
    """Same minimal PSPL setup as test_microlensing_physics.py's
    test_pspl_magnification_accuracy, factored out so both seed-list tests
    share one baseline."""
    return {
        "lens.Lens.t_0":        {"initval": 2460025.0},
        "lens.Lens.pi_E_N":     {"initval": 0.0, "sigma": 0.0},
        "lens.Lens.pi_E_E":     {"initval": 0.0, "sigma": 0.0},
        "star.Lens.distance":   {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.mass":       {"initval": 0.5},
        "star.Lens.pm_ra":      {"initval": 0.0},
        "star.Lens.pm_dec":     {"initval": 0.0},
        "star.Source.pm_ra":    {"initval": 0.0},
        "star.Source.pm_dec":   {"initval": 0.0},
        "star.Source.ra":       {"initval": 0.0},
        "star.Source.dec":      {"initval": 0.0},
        "star.Lens.ra":         {"initval": 0.0},
        "star.Lens.dec":        {"initval": 0.0},
    }


def _config():
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"name": "Lens", "lens_ndx": 0, "source_ndx": 1}],
    }


def test_get_raw_starts_two_in_bounds_seeds_yields_two_distinct_starts():
    """
    Given a params file with a u_0 initval list of 2 in-bounds values,
    When System.prepare()/build_model() run and get_raw_starts is called,
    Then it returns 2 raw starts (seed indices [0, 1]) whose raw u_0 element
    differs between seeds (only the start moved; nothing was skipped).
    """
    user_params = _base_user_params()
    user_params["lens.Lens.u_0"] = {"initval": [0.3, 0.6], "init_scale": 0.01}

    system = System(_config(), user_params=user_params)
    system.prepare()
    model = system.build_model()

    starts, seed_indices = system.get_raw_starts(model)

    assert seed_indices == [0, 1]
    assert len(starts) == 2
    key = "lens.u_0_raw"
    assert key in starts[0] and key in starts[1]
    assert not np.allclose(starts[0][key], starts[1][key])


def test_get_raw_starts_skips_seed_whose_start_violates_a_bound(caplog):
    """
    Given a params file with a u_0 initval list where the second value (50.0)
    lies outside the hard bound [-18, 18] (mulensing defaults.yaml),
    When get_raw_starts is called,
    Then the offending seed is logged and skipped (NOT clipped into range --
    a clipped start would sit in no posterior basin) and only the in-bounds
    seed 0 start is returned.
    """
    user_params = _base_user_params()
    user_params["lens.Lens.u_0"] = {"initval": [0.3, 50.0], "init_scale": 0.01}

    system = System(_config(), user_params=user_params)
    system.prepare()
    model = system.build_model()

    with caplog.at_level(logging.WARNING, logger="exozippy.system"):
        starts, seed_indices = system.get_raw_starts(model)

    assert seed_indices == [0]
    assert len(starts) == 1
    assert any("bound" in rec.message.lower() for rec in caplog.records)


def test_get_raw_starts_single_start_when_no_seed_list():
    """
    Given an ordinary (non-list) params file,
    When get_raw_starts is called,
    Then it returns exactly [get_raw_start(model)] with seed_indices == [0]
    -- the legacy single-start behavior is exactly reproduced.
    """
    user_params = _base_user_params()
    user_params["lens.Lens.u_0"] = {"initval": 0.3}

    system = System(_config(), user_params=user_params)
    system.prepare()
    model = system.build_model()

    starts, seed_indices = system.get_raw_starts(model)
    base = system.get_raw_start(model)

    assert seed_indices == [0]
    assert len(starts) == 1
    for k in base:
        assert np.allclose(starts[0][k], base[k])
