"""ConfigManager's stage-1 probes and seed-hint read-back.

``probe_derivable`` runs the relaxation engine on a snapshot and must roll
every mutation back (its live caller is ``globalsearch``); ``seed_start_value``
reads a registered seed set back in USER units.  These tests used to live in
the external-fitter loader's test file, which was their first caller; the loader is
gone (JDE 2026-10-01) and the contracts are the ConfigManager's own.
"""

import copy

import numpy as np

# Index-form paths throughout: a bare ConfigManager never runs the
# components' normalize_config_block hooks, so the body-derived instance
# names ("source.Source.t_0") do not exist here.
_BINARY_CONFIG = {
    "star": [{"name": "Lens"}, {"name": "Source"}],
    "planet": [{"name": "b"}],
    "mulensevent": [{}],
    "lens": [{"body": "star.Lens"}, {"body": "planet.b"}],
    "source": [{"body": "star.Source"}],
}


def _cm(params, config=None):
    from exozippy.config import ConfigManager

    return ConfigManager(params, system_config=config or _BINARY_CONFIG)


def _full_pspl_params():
    return {
        "source.0.t_0": {"initval": 2458554.9},
        "source.0.u_0": {"initval": 0.14},
        "mulensevent.0.t_E": {"initval": 18.2},
    }


def test_probe_derivable_leaves_no_trace():
    """
    Given a ConfigManager,
    When the derivability probe runs,
    Then user_params, diagnostics and the export snapshots are unchanged --
    the probe must not pre-empt the real solve at stage 4.
    """
    cm = _cm(_full_pspl_params())
    before = (
        copy.deepcopy(cm.user_params),
        list(cm.diagnostics),
        dict(cm._last_resolved),
    )
    # Every attribute the engine is declared to write, not just the three
    # spelled out above: "rolls every mutation back" is the contract, so the
    # sweep is what actually pins it.
    all_before = {
        attr: copy.deepcopy(getattr(cm, attr))
        for attr in cm._PROBE_SNAPSHOT_ATTRS
    }

    cm.probe_derivable(["mulensevent.0.t_E"])

    assert cm.user_params == before[0]
    assert cm.diagnostics == before[1]
    assert cm._last_resolved == before[2]
    for attr, value in all_before.items():
        assert getattr(cm, attr) == value, f"probe leaked {attr}"


def test_probe_derivable_rolls_back_a_solver_timeout_blacklist(monkeypatch):
    """
    Given every sympy inversion times out while the derivability probe runs,
    When probe_derivable returns,
    Then symbolic_blacklist is unchanged.

    The blacklist is consulted for the rest of the process, so a 2 s timeout
    inside this throwaway stage-1a probe would otherwise permanently disable
    that inversion for the real stage-3 solve -- which runs against different
    inputs and might well have solved it in time.
    """
    import exozippy.config as cfgmod

    def _always_times_out(*args, **kwargs):
        # The engine's own timeout exception: _execute_solve guards the solve
        # with the shared _sympy_time_limit context manager (which restores
        # the previous SIGALRM handler -- review 2.1.3), and that raises
        # SymbolicTimeout, not the bare TimeoutError the hand-armed alarm it
        # replaced used to raise.
        raise cfgmod.SymbolicTimeout("Symbolic solver timed out!")

    cm = _cm(_full_pspl_params())
    monkeypatch.setattr(cfgmod.sp, "solve", _always_times_out)

    cm.probe_derivable(["mulensevent.0.t_E"])

    assert cm.symbolic_blacklist == set()

    # The blacklisting itself still works; it is only the probe that must not
    # keep it.  This also proves the timeout path was reached at all, so the
    # assertion above cannot pass vacuously.
    cm.resolve_and_validate_parameters({})
    assert cm.symbolic_blacklist


def test_seed_start_value_returns_user_units():
    """
    Given seed hints pushed through the real ConfigManager (which stores
    internal units -- alpha in radians),
    When seed_start_value reads them back,
    Then the values come back in USER units (alpha in degrees), matching what
    a raw user_params entry would hold, so stage-1 consumers (the mulens flux
    bootstrap) can use either source interchangeably.
    """
    cm = _cm({})
    cm.add_seed_hints(
        [
            {
                "source.0.t_0": 2458554.82,
                "source.0.u_0": 0.131,
                "mulensevent.0.t_E": 19.16,
                "lens.1.log_s": -0.066,
                "lens.1.alpha": -50.37,
                "lens.1.q": 9.26e-4,
            },
            {"lens.1.alpha": -50.47},
        ],
        source="test",
    )

    stored = cm.seed_hint_sets[0]["lens.1.alpha"]
    assert np.isclose(stored, np.deg2rad(-50.37))  # internal storage: rad

    assert np.isclose(cm.seed_start_value("lens.1.alpha"), -50.37)
    assert np.isclose(cm.seed_start_value("source.0.t_0"), 2458554.82)
    assert np.isclose(cm.seed_start_value("lens.1.log_s"), -0.066)
    assert np.isclose(cm.seed_start_value("lens.1.alpha", seed=1), -50.47)
    assert cm.seed_start_value("source.0.rho") is None  # never pushed
    assert cm.seed_start_value("source.0.t_0", seed=5) is None  # no such seed
