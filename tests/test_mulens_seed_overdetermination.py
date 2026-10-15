"""Seeding t_E / pi_E on top of the physical chain -- review 8.6.15.

In a physical-parameter microlensing fit t_E and pi_E are DERIVED from the
lens and source masses, distances and proper motions.  Seeding them
alongside the leaf seeds over-determines the chain, and the relaxation
engine reconciles it by moving whichever leaves were left unseeded.  Under
the precedence rule (review 3.14.22) with GEOCENTRIC t_E / pi_E relations
(review 8.6.23) the unseeded leaves absorb seeds that can hold, and a seed
that cannot (|pi_E| is fixed by the masses and distances) is dropped and
reported by name; the user-start echo check,
``ModelAuditor.check_user_starts`` (PR #252), reports it against the
COMPILED graph with the component's own recipe.

These tests pin both on the shipped examples/ob170114 (Mroz et al. 2026),
and that the recipe (``Parameter.seed_remedy`` on mulensevent.t_E /
pi_E_*) reaches the CLI warning once.  They are slow (ob170114 builds,
~20 s each).
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.diagnostics import ModelAuditor
from exozippy.system import System

pytestmark = pytest.mark.slow

EXAMPLE = Path(__file__).parent.parent / "examples" / "ob170114"

# The published (Mroz+26 Table B.1, Std 2L1S) values the example's leaf
# seeds reproduce as DERIVED quantities.
T_E_PUB = 173.0
PI_E_N_PUB = 0.167
PI_E_E_PUB = 0.127
PHI_PI_PUB = np.degrees(np.arctan2(PI_E_E_PUB, PI_E_N_PUB))  # 37.25 deg

SEED_KEYS = ("mulensevent.t_E", "mulensevent.pi_E_N", "mulensevent.pi_E_E")


def _load():
    config = yaml.safe_load((EXAMPLE / "ob170114.yaml").read_text())
    params = yaml.safe_load((EXAMPLE / config["parameter_file"]).read_text())
    return config, params


def _build_and_audit(monkeypatch, config, params):
    monkeypatch.chdir(EXAMPLE)
    system = System(config, copy.deepcopy(params))
    system.prepare()
    model = system.build_model()
    with model:
        auditor = ModelAuditor(model, system, system.get_mcmc_init(model))
        findings = auditor.check_user_starts()
        built = auditor.user_start_values()
    return system, model, auditor, findings, built


def _seed_over_the_chain(pi_e_e):
    """ob170114 with the source system's pm seeds REMOVED and t_E / pi_E
    re-seeded -- the state the example was in before its params file
    learned the recipe (measured 2026-08-27: phi_pi rotated 35 deg,
    t_E 173 -> 205)."""
    config, params = _load()
    for leaf in ("pm_ra", "pm_dec"):
        del params[f"star.Source.{leaf}"]
        del params[f"star.SComp.{leaf}"]
    params["mulensevent.t_E"] = {"initval": T_E_PUB}
    params["mulensevent.pi_E_N"] = {"initval": PI_E_N_PUB}
    params["mulensevent.pi_E_E"] = {"initval": pi_e_e}
    return config, params


def test_published_t_E_pi_E_seeds_over_the_chain_are_delivered(monkeypatch):
    """
    Given the published t_E / pi_E seeded over the chain, the source pm
      seeds removed,
    When the model is built and the user-start check runs,
    Then the model is built at the published t_E and pi_E direction and
      nothing is reported for them.  Under the old engine (alphabetical,
      HELIOCENTRIC relations) this built t_E = 202.65 d and phi_pi = 1.1
      deg; under the precedence rule (review 3.14.22) the unseeded source
      pm absorbs the seeds, and with the geocentric relations (8.6.23) the
      model's t_E is the engine's.  |pi_E| is fixed by the masses and
      distances, so pi_E_E -- last in keep-order -- is not held exactly,
      but the re-derived value is within the user-start tolerance of 0.127.
    """
    config, params = _seed_over_the_chain(PI_E_E_PUB)

    system, _model, _auditor, findings, built = _build_and_audit(
        monkeypatch, config, params
    )

    # pi_E_E is the one the engine could not hold exactly, but it lands
    # within the user-start tolerance, so no drop is reported either.
    assert system.config_manager.diagnostics == []
    assert not set(SEED_KEYS) & {f["key"] for f in findings}
    assert built["mulensevent.t_E"] == pytest.approx(T_E_PUB, rel=1e-4)
    phi_pi = np.degrees(
        np.arctan2(built["mulensevent.pi_E_E"], built["mulensevent.pi_E_N"])
    )
    assert phi_pi == pytest.approx(PHI_PI_PUB, abs=2.0)


def test_a_pi_E_seed_the_chain_cannot_hold_is_reported_with_the_recipe(
    monkeypatch, caplog
):
    """
    Given the same chain with pi_E_E seeded at 4x the published value --
      a |pi_E| the pinned masses and distances cannot give,
    When the model is built and the user-start check runs,
    Then the engine drops pi_E_E and says so, the echo check reports it
      with the value the model is built at, carrying the mulensevent
      seed_remedy, and the CLI warning (run.inspect_start) prints the
      recipe ONCE.
    """
    config, params = _seed_over_the_chain(4 * PI_E_E_PUB)

    system, model, auditor, findings, built = _build_and_audit(
        monkeypatch, config, params
    )

    assert "mulensevent.0.pi_E_E" in [
        d["param_paths"][0] for d in system.config_manager.diagnostics
    ]
    by_key = {f["key"]: f for f in findings}
    assert "mulensevent.pi_E_E" in by_key, sorted(by_key)
    finding = by_key["mulensevent.pi_E_E"]
    assert finding["requested"] == pytest.approx(4 * PI_E_E_PUB)
    assert finding["produced"] == pytest.approx(built["mulensevent.pi_E_E"])
    assert "DROP the t_E and pi_E seeds" in finding["remedy"]
    assert built["mulensevent.t_E"] == pytest.approx(T_E_PUB, rel=1e-4)

    from exozippy.run import inspect_start

    caplog.set_level(logging.WARNING)
    with model:
        inspect_start(
            model,
            system,
            auditor.transformed_inits,
            build_start_misses=findings,
        )
    (block,) = [
        r.getMessage()
        for r in caplog.records
        if "not built at every value you set" in r.getMessage()
    ]
    assert "mulensevent.pi_E_E: you set 0.508, the model is built at" in block
    assert block.count("DROP the t_E and pi_E seeds") == 1


def test_a_consistent_t_E_seed_over_the_shipped_leaves_is_silent(
    monkeypatch,
):
    """
    Given the SHIPPED ob170114 seeds (all four pm leaves, masses and
      distances pinned) plus t_E re-added at the published 173 d -- the
      item's literal probe, re-run on 2026-10-05,
    When the model is built and the user-start check runs,
    Then nothing is reported for t_E and the model is built at the
      published value: with every leaf seeded the derived t_E already IS
      173 (172.9987), so the seed costs nothing and there is nothing to
      reconcile.  The warning above is about a conflict, not about the
      mere presence of a seed on a derived parameter.
    """
    config, params = _load()
    params["mulensevent.t_E"] = {"initval": T_E_PUB}

    system, _model, _auditor, findings, built = _build_and_audit(
        monkeypatch, config, params
    )

    # The engine's t_E relation is GEOCENTRIC, as the model's is (review
    # 8.6.23), so the consistent seed holds in the ledger too: nothing is
    # dropped.
    assert system.config_manager.diagnostics == []
    assert "mulensevent.t_E" not in {f["key"] for f in findings}
    assert built["mulensevent.t_E"] == pytest.approx(T_E_PUB, rel=1e-4)
