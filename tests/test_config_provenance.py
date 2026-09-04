# tests/test_config_provenance.py
"""Provenance-ranking regressions from section 2.1 of the 2026-08-08 review.

The ConfigManager's whole job is to reconcile conflicting constraints under a
declared hierarchy -- user > data-derived > default (see config.py's class
docstring).  Each test here pins one place where that hierarchy was violated
or where a user's entry vanished without a diagnostic:

  2.1.1  the standalone orbit.m_total solver overwrote an explicit user value
         every iteration, DOWNGRADING its provenance to PRECEDENCE_DERIVED_MIXED.
  2.1.2  MMEXOFAST seed hints entered at PRECEDENCE_USER, so a seed clobbered a
         user's scalar initval and a seed disagreeing with a genuine user
         entry tripped the "over-constrained" contradiction clause.
  2.1.3  a 3-part key with a typo'd INSTANCE name was kept as an inert leaf
         symbol -- value, bounds and prior silently discarded.
  1.1.3  a user's numeric `mu` with no `initval` never reached the relaxation
         engine, so the engine reasoned from a hint or a default while
         `resolve()` made the `mu` the model's actual start.
"""

import numpy as np
import pytest

from exozippy.config import (
    PRECEDENCE_DERIVED_DATA,
    PRECEDENCE_DERIVED_MIXED,
    PRECEDENCE_USER,
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
    Then the user's value survives at PRECEDENCE_USER -- the solver is skipped, not
    allowed to overwrite it and downgrade its provenance to
    PRECEDENCE_DERIVED_MIXED.
    """
    params = _orbit_params()
    params["orbit.b.m_total"] = {"initval": 5.0}  # body sum is 1.001

    cm = ConfigManager(params, system_config=_orbit_config())
    cm.finalize_user_params()

    assert cm._last_resolved["orbit.0.m_total"] == pytest.approx(5.0)
    assert cm._last_provenance["orbit.0.m_total"] == PRECEDENCE_USER
    assert "orbit.0.m_total" not in cm._last_solved_by


def test_standalone_solver_still_fires_without_a_user_value():
    """
    Given the same orbit with no user m_total,
    When the engine runs,
    Then the standalone solver still supplies the member-mass sum at
    PRECEDENCE_DERIVED_MIXED -- the guard blocks only ranks it cannot beat.
    """
    cm = ConfigManager(_orbit_params(), system_config=_orbit_config())
    cm.finalize_user_params()

    assert cm._last_resolved["orbit.0.m_total"] == pytest.approx(
        1.001, rel=1e-3
    )
    assert cm._last_provenance["orbit.0.m_total"] == PRECEDENCE_DERIVED_MIXED
    assert "standalone solver" in cm._last_solved_by["orbit.0.m_total"]


# ---------------------------------------------------------------------------
# 2.1.2  seed hints are data-derived, not user statements
# ---------------------------------------------------------------------------


def test_seed_hint_does_not_override_a_user_scalar_initval():
    """
    Given a user scalar initval for lens.Lens.t_0 and an MMEXOFAST seed hint
    naming a very different t_0,
    When the engine solves,
    Then the USER's value is the start and keeps PRECEDENCE_USER: a seed is a
    (fancy) derivation from the data and every user entry outranks it.
    """
    cm = ConfigManager(
        {"lens.Lens.t_0": {"initval": 2455000.0}},
        system_config=_PSPL_CONFIG,
    )
    cm.add_seed_hints([{"lens.0.t_0": 2459999.0}])
    cm.finalize_user_params()

    assert cm._last_resolved["lens.0.t_0"] == pytest.approx(2455000.0)
    assert cm._last_provenance["lens.0.t_0"] == PRECEDENCE_USER


def test_seed_hint_beats_a_default_and_lands_at_derived_data_rank():
    """
    Given no user entry for lens.Lens.t_0,
    When a seed hint supplies one,
    Then it is used, at PRECEDENCE_DERIVED_DATA -- above defaults.yaml, below any
    user entry.
    """
    cm = ConfigManager({}, system_config=_PSPL_CONFIG)
    cm.add_seed_hints([{"lens.0.t_0": 2459999.0}])
    cm.finalize_user_params()

    assert cm._last_resolved["lens.0.t_0"] == pytest.approx(2459999.0)
    assert cm._last_provenance["lens.0.t_0"] == PRECEDENCE_DERIVED_DATA


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
    PRECEDENCE_DERIVED_DATA must not collapse the seeds onto one start.
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
    reads user_params only, and its test is provenance > PRECEDENCE_DEFAULT, which
    PRECEDENCE_DERIVED_DATA (60) satisfies anyway.  Getting this wrong re-runs the
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


# ---------------------------------------------------------------------------
# 2.1.3  a typo'd instance name must be loud
# ---------------------------------------------------------------------------


def test_typoed_instance_name_raises_and_names_the_valid_instances():
    """
    Given a params key whose INSTANCE name is a typo ('star.Aa.teff' for
    'star.A.teff'),
    When the ConfigManager standardizes the names,
    Then it raises a STRICT NAMING ERROR listing the valid instance names --
    rather than keeping the key as an inert leaf and silently discarding the
    user's value, prior and sigma.
    """
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    params = {
        "star.Aa.teff": {"initval": 4000.0, "mu": 4000.0, "sigma": 100.0}
    }

    with pytest.raises(ValueError, match="STRICT NAMING ERROR") as exc:
        ConfigManager(params, system_config=config)

    msg = str(exc.value)
    assert "star.Aa.teff" in msg
    assert "'A'" in msg and "'B'" in msg


def test_numeric_index_and_named_instances_still_pass():
    """
    Given the legitimate 3-part spellings -- a numeric index and a real
    instance name --
    When the names are standardized,
    Then both resolve to index form and neither raises.
    """
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    cm = ConfigManager(
        {
            "star.0.teff": {"initval": 5000.0},
            "star.B.teff": {"initval": 4000.0},
        },
        system_config=config,
    )

    assert set(cm.user_params) == {"star.0.teff", "star.1.teff"}


def test_flat_dict_component_three_part_key_does_not_raise():
    """
    Given a 3-part key naming a flat-dict (non-list) component,
    When the names are standardized,
    Then the key is kept as-is: there are no instances to check it against.
    """
    config = {"sed": {"file": "x.sed"}}
    cm = ConfigManager(
        {"sed.whatever.errscale": {"initval": 1.0}}, system_config=config
    )

    assert "sed.whatever.errscale" in cm.user_params


def test_name_borrowed_from_another_component_is_accepted():
    """
    Given a mann block that has not derived its `name:` yet -- ConfigManager
    is built BEFORE the component loop in System.__init__, and Mann.__init__
    is what copies `star: "A"` into `name: "A"` --
    When 'mann.A.ks_offset' is standardized,
    Then it is kept rather than rejected: 'A' is declared somewhere in the
    config (as a star), and the legal-name universe is every declared name,
    not just the addressed component's own list.
    """
    config = {
        "star": [{"name": "A"}],
        "mann": [{"star": "A", "constrain": ["mass"], "ks": 8.782}],
    }
    cm = ConfigManager(
        {"mann.A.ks_offset": {"initval": 0.1}}, system_config=config
    )

    assert "mann.A.ks_offset" in cm.user_params


def test_lens_per_source_element_name_is_accepted():
    """
    Given the NSNL spelling examples/ob161003 ships -- 'lens.SourceA.t_0'
    addresses element j of the lens's PER-SOURCE vectors by the source star's
    name, while the lens block itself has a single entry named 'Lens' --
    When the names are standardized,
    Then the key survives.  A per-parameter `names` list is a manifest option
    resolved at stage 3, so a check restricted to the lens block's own entry
    names would reject a shipped, working example.
    """
    config = {
        "star": [
            {"name": "Lens"},
            {"name": "LensB"},
            {"name": "SourceA"},
            {"name": "SourceB"},
        ],
        "lens": [
            {
                "name": "Lens",
                "lenses": ["star.0", "star.1"],
                "sources": ["star.2", "star.3"],
            }
        ],
    }
    cm = ConfigManager(
        {"lens.SourceA.t_0": {"initval": 2457551.04}}, system_config=config
    )

    assert "lens.SourceA.t_0" in cm.user_params


def test_name_declared_nowhere_raises_even_for_a_borrowing_component():
    """
    Given the same lens topology but a source name that appears nowhere in
    the config,
    When the names are standardized,
    Then it raises: the accepted set is wide (any declared name) but not
    unbounded, so a real typo is still caught.
    """
    config = {
        "star": [{"name": "Lens"}, {"name": "SourceA"}],
        "lens": [
            {"name": "Lens", "lenses": ["star.0"], "sources": ["star.1"]}
        ],
    }

    with pytest.raises(ValueError, match="STRICT NAMING ERROR"):
        ConfigManager(
            {"lens.SourceZ.t_0": {"initval": 2457551.04}},
            system_config=config,
        )


# ---------------------------------------------------------------------------
# 1.1.3: a lone `mu` IS the start, so the engine has to see it
# ---------------------------------------------------------------------------

_STAR_CONFIG = {"star": [{"name": "A", "mist": False}]}


def test_a_lone_mu_reaches_the_engine_at_user_precedence():
    """
    Given a user entry with `mu` and `sigma` but NO `initval`,
    When a component hint competes for the same parameter,
    Then the engine resolves the user's mu, at PRECEDENCE_USER.

    This is the divergence review 1.1.3 reproduced. `resolve()` promotes a
    lone `mu` to the start deliberately -- a user's prior centre beats an
    arbitrary default -- so the mu IS what the model starts at. The engine
    read only `initval`, so a numeric mu reached it solely through the
    default-armor pass at PRECEDENCE_DEFAULT: right value, precedence wrong
    by 80, which left any relation touching the symbol free to solve it away
    from the number the user wrote.
    """
    # Arrange
    cm = ConfigManager(
        {"star.0.distance": {"mu": 4000.0, "sigma": 100.0}},
        system_config=_STAR_CONFIG,
    )
    cm.add_hint("star.0.distance", 6500.0)

    # Act
    cm.finalize_user_params()

    # Assert
    assert cm._last_resolved["star.0.distance"] == pytest.approx(4000.0)
    assert cm._last_provenance["star.0.distance"] == PRECEDENCE_USER


def test_the_engine_and_resolve_agree_on_a_lone_mu():
    """
    Given the same entry,
    When the engine's resolved value is compared to resolve()'s initval,
    Then they are the same number.

    The agreement is the property worth pinning, not either value alone.
    Before the fix these two answered differently for one input -- the model
    started at 4000 while the ledger, export_solution and initval_source all
    reported 6500 and labelled it "data" -- so a build_pymc error blamed a
    number nobody had written.
    """
    # Arrange
    cm = ConfigManager(
        {"star.0.distance": {"mu": 4000.0, "sigma": 100.0}},
        system_config=_STAR_CONFIG,
    )
    cm.add_hint("star.0.distance", 6500.0)
    cm.finalize_user_params()

    # Act
    resolved = cm.resolve("star", "distance", shape=(1,))

    # Assert
    assert float(np.atleast_1d(resolved["initval"])[0]) == pytest.approx(
        cm._last_resolved["star.0.distance"]
    )


def test_an_explicit_initval_still_wins_over_mu():
    """
    Given an entry carrying BOTH `initval` and `mu`,
    When the engine reads it,
    Then the `initval` is used.

    The fallback must not become an override: `mu` fills in only where no
    `initval` was written. Pins the precedence between the two fields of one
    user entry, which is the thing a careless fix would invert.
    """
    # Arrange
    cm = ConfigManager(
        {"star.0.distance": {"initval": 1500.0, "mu": 4000.0, "sigma": 100.0}},
        system_config=_STAR_CONFIG,
    )

    # Act
    cm.finalize_user_params()

    # Assert
    assert cm._last_resolved["star.0.distance"] == pytest.approx(1500.0)
    assert cm._last_provenance["star.0.distance"] == PRECEDENCE_USER


def test_a_lone_mu_on_an_unmapped_leaf_also_reaches_the_engine():
    """
    Given a lone `mu` on a path with no symbol in the master map,
    When finalize registers it as a leaf,
    Then the mu is pushed into the solver's initial state.

    The leaf fallback is a SECOND copy of the same read, and fixing only the
    mapped loop would have left this half broken -- which is why the fix is
    one shared helper rather than two edits.

    ASSERT THE PRECEDENCE, NOT THE VALUE. The value was already 7.5 before
    the fix, because the default-armor pass supplied it; what was wrong was
    that it arrived at PRECEDENCE_DEFAULT (20) rather than PRECEDENCE_USER
    (100). An earlier draft of this test checked only the value and therefore
    passed against the bug -- which is the whole shape of review 1.1.3: the
    right number at the wrong precedence.
    """
    # Arrange
    cm = ConfigManager(
        {"star.0.some_unmapped_thing": {"mu": 7.5, "sigma": 0.5}},
        system_config=_STAR_CONFIG,
    )

    # Act
    cm.finalize_user_params()

    # Assert
    assert "star.0.some_unmapped_thing" in cm.master_symbol_map
    assert cm._last_resolved.get(
        "star.0.some_unmapped_thing"
    ) == pytest.approx(7.5)
    assert cm._last_provenance["star.0.some_unmapped_thing"] == PRECEDENCE_USER
