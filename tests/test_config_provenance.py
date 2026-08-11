# tests/test_config_provenance.py
"""Provenance-ranking regressions from section 2.1 of the 2026-08-08 review.

The ConfigManager's whole job is to reconcile conflicting constraints under a
declared hierarchy -- user > data-derived > default (see config.py's class
docstring).  Each test here pins one place where that hierarchy was violated
or where a user's entry vanished without a diagnostic:

  2.1.1  the standalone orbit.m_total solver overwrote an explicit user value
         every iteration, DOWNGRADING its provenance to RANK_DERIVED_MIXED.
  2.1.2  MMEXOFAST seed hints entered at RANK_USER, so a seed clobbered a
         user's scalar initval and a seed disagreeing with a genuine user
         entry tripped the "over-constrained" contradiction clause.
  2.1.3  a 3-part key with a typo'd INSTANCE name was kept as an inert leaf
         symbol -- value, bounds and prior silently discarded.
"""

import pytest

from exozippy.config import (
    RANK_DERIVED_MIXED,
    RANK_USER,
    ConfigManager,
)

MJUP_PER_MSUN = 1047.348644  # solMass -> jupiterMass

_PSPL_CONFIG = {
    "star": [
        {"name": "Lens", "mist": False},
        {"name": "Source", "mist": False},
    ],
    "lens": [{"name": "Lens", "lenses": ["star.0"], "sources": ["star.1"]}],
}

_BINARY_CONFIG = {
    "star": [
        {"name": "Lens", "mist": False},
        {"name": "Source", "mist": False},
    ],
    "planet": [{"name": "Comp"}],
    "lens": [
        {
            "name": "Lens",
            "lenses": ["star.0", "planet.0"],
            "sources": ["star.1"],
        }
    ],
}


def _orbit_config():
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
    }


def _orbit_params():
    return {
        "star.A.mass": {"initval": 1.0},
        "planet.b.mass": {"initval": 0.001 * MJUP_PER_MSUN},
        "orbit.b.logP": {"initval": 1.0},
        "orbit.b.tc": {"initval": 2456000.0},
    }


# ---------------------------------------------------------------------------
# 2.1.1  standalone solvers must respect provenance
# ---------------------------------------------------------------------------


def test_standalone_solver_does_not_overwrite_a_user_value():
    """
    Given an explicit user orbit.b.m_total that disagrees with the sum of the
    orbit's member-body masses,
    When the relaxation engine runs (the standalone m_total solver fires once
    per iteration),
    Then the user's value survives at RANK_USER -- the solver is skipped, not
    allowed to overwrite it and downgrade its provenance to
    RANK_DERIVED_MIXED.
    """
    params = _orbit_params()
    params["orbit.b.m_total"] = {"initval": 5.0}  # body sum is 1.001

    cm = ConfigManager(params, system_config=_orbit_config())
    cm.finalize_user_params()

    assert cm._last_resolved["orbit.0.m_total"] == pytest.approx(5.0)
    assert cm._last_provenance["orbit.0.m_total"] == RANK_USER
    assert "orbit.0.m_total" not in cm._last_solved_by


def test_standalone_solver_still_fires_without_a_user_value():
    """
    Given the same orbit with no user m_total,
    When the engine runs,
    Then the standalone solver still supplies the member-mass sum at
    RANK_DERIVED_MIXED -- the guard blocks only ranks it cannot beat.
    """
    cm = ConfigManager(_orbit_params(), system_config=_orbit_config())
    cm.finalize_user_params()

    assert cm._last_resolved["orbit.0.m_total"] == pytest.approx(
        1.001, rel=1e-3
    )
    assert cm._last_provenance["orbit.0.m_total"] == RANK_DERIVED_MIXED
    assert "standalone solver" in cm._last_solved_by["orbit.0.m_total"]
