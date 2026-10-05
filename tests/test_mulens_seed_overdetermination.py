"""Seeding t_E / pi_E on top of the physical chain -- review 8.6.15.

In a physical-parameter microlensing fit t_E and pi_E are DERIVED from the
lens and source masses, distances and proper motions.  Seeding them
alongside the leaf seeds over-determines the chain, and the relaxation
engine reconciles it by moving whichever leaves were left unseeded.
Neither of the engine's own detectors sees this (the Contradiction Clause
needs every symbol of ONE violated relation at user rank, and the chain's
intermediates sit at RANK_DERIVED_USER), so ``cm.diagnostics`` stays empty
and the only thing that can report it is the user-start echo check,
``ModelAuditor.check_user_starts`` (PR #252), which compares every user
initval against the COMPILED graph at the build start.

These tests pin that the check fires on exactly this case, on the shipped
examples/ob170114 (Mroz et al. 2026), and that the component's own recipe
(``Parameter.seed_remedy`` on mulensevent.t_E / pi_E_*) reaches the CLI
warning once.  They are slow (two ob170114 builds, ~20 s each).
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


def test_t_E_pi_E_seeds_over_the_chain_are_reported_with_the_recipe(
    monkeypatch, caplog
):
    """
    Given the shipped ob170114 seeds with the source system's proper-motion
      seeds REMOVED and t_E / pi_E re-seeded at the published values --
      the state the example was in before its params file learned the
      recipe (measured 2026-08-27: phi_pi rotated 35 deg, t_E 173 -> 205),
    When the model is built and the user-start check runs,
    Then the engine records nothing (this is the blind spot), but all
      three seeds are reported as not delivered, with the values the model
      is built at -- t_E ~ 202.65 d and phi_pi ~ 1.1 deg against 173 d and
      37.25 deg -- each carrying the mulensevent seed_remedy; and the CLI
      warning (run.inspect_start) prints both values per key and the
      recipe ONCE, naming the three keys.
    """
    # Arrange
    config, params = _load()
    for leaf in ("pm_ra", "pm_dec"):
        del params[f"star.Source.{leaf}"]
        del params[f"star.SComp.{leaf}"]
    params["mulensevent.t_E"] = {"initval": T_E_PUB}
    params["mulensevent.pi_E_N"] = {"initval": PI_E_N_PUB}
    params["mulensevent.pi_E_E"] = {"initval": PI_E_E_PUB}

    # Act
    system, model, auditor, findings, built = _build_and_audit(
        monkeypatch, config, params
    )

    # Assert -- the engine's own detectors are blind to it ...
    assert system.config_manager.diagnostics == []
    # ... so the echo check is what reports it, on exactly these keys.
    by_key = {f["key"]: f for f in findings}
    assert set(SEED_KEYS) <= set(by_key), sorted(by_key)
    t_e = by_key["mulensevent.t_E"]
    assert t_e["requested"] == pytest.approx(T_E_PUB)
    # Measured 202.651 d (2026-10-05); a 5% band leaves room for the
    # platform-dependent build while staying far from 173.
    assert t_e["produced"] == pytest.approx(202.65, rel=0.05)
    phi_pi = np.degrees(
        np.arctan2(
            by_key["mulensevent.pi_E_E"]["produced"],
            by_key["mulensevent.pi_E_N"]["produced"],
        )
    )
    assert abs(phi_pi - PHI_PI_PUB) > 20.0, phi_pi  # measured 1.08 deg
    remedies = {by_key[k]["remedy"] for k in SEED_KEYS}
    assert len(remedies) == 1
    (remedy,) = remedies
    assert "DROP the t_E and pi_E seeds" in remedy

    # The CLI path: the same findings through run.inspect_start.
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
    assert "mulensevent.t_E: you set 173, the model is built at" in block
    assert block.count("DROP the t_E and pi_E seeds") == 1
    assert (
        "For mulensevent.pi_E_E, mulensevent.pi_E_N, mulensevent.t_E:" in block
    )


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

    assert system.config_manager.diagnostics == []
    assert "mulensevent.t_E" not in {f["key"] for f in findings}
    assert built["mulensevent.t_E"] == pytest.approx(T_E_PUB, rel=1e-4)
