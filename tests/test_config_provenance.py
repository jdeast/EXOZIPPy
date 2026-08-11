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

import numpy as np
import pytest

from exozippy.config import (
    RANK_DERIVED_DATA,
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


# ---------------------------------------------------------------------------
# 2.1.2  seed hints are data-derived, not user statements
# ---------------------------------------------------------------------------


def test_seed_hint_does_not_override_a_user_scalar_initval():
    """
    Given a user scalar initval for lens.Lens.t_0 and an MMEXOFAST seed hint
    naming a very different t_0,
    When the engine solves,
    Then the USER's value is the start and keeps RANK_USER: a seed is a
    (fancy) derivation from the data and every user entry outranks it.
    """
    cm = ConfigManager(
        {"lens.Lens.t_0": {"initval": 2455000.0}},
        system_config=_PSPL_CONFIG,
    )
    cm.add_seed_hints([{"lens.0.t_0": 2459999.0}])
    cm.finalize_user_params()

    assert cm._last_resolved["lens.0.t_0"] == pytest.approx(2455000.0)
    assert cm._last_provenance["lens.0.t_0"] == RANK_USER


def test_seed_hint_beats_a_default_and_lands_at_derived_data_rank():
    """
    Given no user entry for lens.Lens.t_0,
    When a seed hint supplies one,
    Then it is used, at RANK_DERIVED_DATA -- above defaults.yaml, below any
    user entry.
    """
    cm = ConfigManager({}, system_config=_PSPL_CONFIG)
    cm.add_seed_hints([{"lens.0.t_0": 2459999.0}])
    cm.finalize_user_params()

    assert cm._last_resolved["lens.0.t_0"] == pytest.approx(2459999.0)
    assert cm._last_provenance["lens.0.t_0"] == RANK_DERIVED_DATA


def test_seed_hint_conflicting_with_a_user_entry_is_not_over_constrained():
    """
    Given a user lens.Lens.s and an MMEXOFAST seed for lens.0.log_s that
    disagrees (they are two coordinates for one fact, tied by s = 10**log_s),
    When the engine runs,
    Then no "over-constrained" contradiction is raised: the seed is not a
    user statement, so the relation is resolved in the user's favor -- s
    keeps its value and log_s is back-solved from it.
    """
    cm = ConfigManager(
        {"lens.Lens.s": {"initval": 1.5}}, system_config=_BINARY_CONFIG
    )
    cm.add_seed_hints([{"lens.0.log_s": float(np.log10(2.5))}])
    cm.finalize_user_params()

    assert not [
        d for d in cm.diagnostics if "Over-constrained" in d["message"]
    ]
    assert cm._last_resolved["lens.0.s"] == pytest.approx(1.5)
    assert cm._last_resolved["lens.0.log_s"] == pytest.approx(
        np.log10(1.5), rel=1e-3
    )


def test_user_initval_list_still_outranks_a_seed_hint_per_seed():
    """
    Given both a per-seed user initval LIST and a seed hint set for the same
    path,
    When the K seeds are solved,
    Then each seed starts at the user's element for that seed -- splitting the
    two sources into separate rank channels must not lose the list's priority.
    """
    cm = ConfigManager(
        {"lens.Lens.t_0": {"initval": [2455000.0, 2455010.0]}},
        system_config=_PSPL_CONFIG,
    )
    cm.add_seed_hints([{"lens.0.t_0": 2459999.0}, {"lens.0.t_0": 2459998.0}])
    cm.finalize_user_params()

    assert cm.seed_resolved is not None and len(cm.seed_resolved) == 2
    assert cm.seed_resolved[0]["lens.0.t_0"] == pytest.approx(2455000.0)
    assert cm.seed_resolved[1]["lens.0.t_0"] == pytest.approx(2455010.0)


def test_seed_hints_still_vary_the_start_across_seeds():
    """
    Given seed hints only (the ordinary MMEXOFAST case: no user entry),
    When the K seeds are solved,
    Then each seed lands on its own hint value -- the demotion to
    RANK_DERIVED_DATA must not collapse the seeds onto one start.
    """
    cm = ConfigManager({}, system_config=_PSPL_CONFIG)
    cm.add_seed_hints([{"lens.0.t_0": 2459999.0}, {"lens.0.t_0": 2459888.0}])
    cm.finalize_user_params()

    assert cm.seed_resolved is not None and len(cm.seed_resolved) == 2
    assert cm.seed_resolved[0]["lens.0.t_0"] == pytest.approx(2459999.0)
    assert cm.seed_resolved[1]["lens.0.t_0"] == pytest.approx(2459888.0)


def test_seed_hints_do_not_change_the_mmexofast_auto_trigger():
    """
    Given a params file whose entries make the microlensing observables
    derivable,
    When probe_derivable is asked (the MMEXOFAST auto-trigger's question),
    Then the answer is unchanged by the presence of seed hints: the probe
    reads user_params only, and its test is provenance > RANK_DEFAULT, which
    RANK_DERIVED_DATA (60) satisfies anyway.  Getting this wrong re-runs the
    fitter on every restart.
    """
    params = {
        "lens.Lens.t_0": {"initval": 2460000.0},
        "lens.Lens.u_0": {"initval": 0.5},
        "lens.Lens.t_E": {"initval": 25.0},
    }
    paths = ["lens.0.t_0", "lens.0.u_0", "lens.0.t_E"]

    cm = ConfigManager(dict(params), system_config=_PSPL_CONFIG)
    before = cm.probe_derivable(paths)

    cm2 = ConfigManager(dict(params), system_config=_PSPL_CONFIG)
    cm2.add_seed_hints([{"lens.0.t_0": 2459999.0}])
    after = cm2.probe_derivable(paths)

    assert before == set(paths)
    assert after == before
