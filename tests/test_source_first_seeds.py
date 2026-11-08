"""Source before lens: where the engine places a microlensing event whose
observables (t_E, pi_E, rho, q) are seeded in place of the physical leaves.

JDE ruling 2026-11-08: "In general, we should know the source properties
better than the lens properties."  So every source-side guess (distance at
MULENS_SOURCE, a finite source's radius at MULENS_SOURCE_RADIUS, galactic-
model proper motion and mass at DERIVED_DATA) outranks every lens-side one
(MULENS_LENS_MASS > MULENS_LENS_PM > MULENS_LENS_DISTANCE), and the lens
absorbs what the observables imply.  config.md, "Source before lens".

Each test runs System.prepare() only (no model build) and reads the
engine's solved starts and keep report.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.config import (
    PRECEDENCE_DEFAULT,
    PRECEDENCE_MULENS_LENS_DISTANCE,
    PRECEDENCE_MULENS_LENS_MASS,
    PRECEDENCE_MULENS_LENS_PM,
    PRECEDENCE_MULENS_SOURCE,
    PRECEDENCE_MULENS_SOURCE_RADIUS,
    ConfigManager,
)
from exozippy.system import System

EXAMPLES = Path(__file__).parent.parent / "examples"
KAPPA = 8.14385327673037  # mas / Msun, the engine's theta_E relation


def _prepare(monkeypatch, example, config_name, edit_params=None, edit=None):
    d = EXAMPLES / example
    config = yaml.safe_load((d / config_name).read_text())
    params = yaml.safe_load((d / config["parameter_file"]).read_text()) or {}
    if edit_params:
        params = edit_params(copy.deepcopy(params))
    if edit:
        config = edit(copy.deepcopy(config))
    config["parameter_file"] = None
    monkeypatch.chdir(d)
    system = System(config, params)
    system.prepare()
    return system


def _source_moves(rep, source_idx):
    return [m for m, *_ in rep["moved"] if m.startswith(f"star.{source_idx}.")]


def test_the_ladder_puts_every_source_guess_above_every_lens_guess():
    """
    Given the precedence ladder,
    When the microlensing levels are compared,
    Then the source guesses (distance, then a finite source's radius)
      outrank the lens guesses, and among those the
      mass outranks the proper motion, which outranks the distance -- all
      above the defaults.yaml backstop (JDE 2026-11-08).
    """
    assert (
        PRECEDENCE_MULENS_SOURCE
        > PRECEDENCE_MULENS_SOURCE_RADIUS
        > PRECEDENCE_MULENS_LENS_MASS
        > PRECEDENCE_MULENS_LENS_PM
        > PRECEDENCE_MULENS_LENS_DISTANCE
        > PRECEDENCE_DEFAULT
    )


def test_a_decision_level_declared_twice_at_different_levels_raises(
    monkeypatch,
):
    """
    Given a ConfigManager with star.1.radius declared at
      MULENS_SOURCE_RADIUS,
    When the same path is declared again at the same level, and then at
      another,
    Then the repeat is accepted and the conflict raises naming the path:
      two components disagreeing about one start must not be settled by
      whichever declared last.
    """
    system = _prepare(monkeypatch, "ob08092", "ob08092.yaml")
    cm = system.config_manager
    cm.declare_decision_level("star.1.radius", "MULENS_SOURCE_RADIUS")
    cm.declare_decision_level("star.1.radius", "MULENS_SOURCE_RADIUS")
    with pytest.raises(ValueError, match="star.1.radius"):
        cm.declare_decision_level("star.1.radius", "MULENS_LENS_MASS")


def _seed_rho(params):
    params["source.Source.rho"] = {"initval": 0.002}
    return params


def test_a_seeded_rho_and_pi_e_are_absorbed_by_the_lens_not_the_source(
    monkeypatch,
):
    """
    Given examples/KMT-2019-BLG-1806 (t_E, pi_E and q seeded, no physical
      leaf; the source star seeded from its own flux through the user's
      zeropoint tie) with a finite-source rho of 0.002 seeded as well --
      the shape of OGLE-2011-BLG-0975 seeded from a published solution,
    When the starts are solved,
    Then every seed is kept, the SOURCE keeps its starts (the flux-seeded
      radius, reported "data", and the 8 kpc distance) and the LENS
      absorbs the observables the way a microlenser reads them:
      theta_E = theta_star / rho, M_total = theta_E / (kappa |pi_E|).
      Before (the lens-mass guess at DERIVED_DATA, tied with the source
      seed) the mass and pi_E fixed theta_E and the source radius had to
      absorb rho -- OGLE-2011-BLG-0975: a 34.6 Rsun source, a 54 pc lens.
    """
    system = _prepare(
        monkeypatch,
        "KMT-2019-BLG-1806",
        "KMT-2019-BLG-1806.yaml",
        _seed_rho,
    )
    cm = system.config_manager
    v = cm._last_resolved
    rep = cm._last_keep_report
    for p in (
        "mulensevent.0.t_E",
        "mulensevent.0.pi_E_N",
        "mulensevent.0.pi_E_E",
        "source.0.rho",
        "lens.1.q",
    ):
        assert p in rep["kept"], (p, rep["dropped"])
    assert not rep["dropped"], rep["dropped"]
    assert not rep["escalated"], rep["escalated"]
    assert _source_moves(rep, 1) == []
    r_s = v["star.1.radius"]
    assert r_s == pytest.approx(cm.hints["star.1.radius"])
    assert v["star.1.distance"] == pytest.approx(8000.0)
    assert cm.initval_source("star", "radius", element=1) == "data"

    # theta_star = R_S / d_S in mas (1 Rsun = 0.00465047 au)
    theta_e = 0.00465047 * r_s / 8000.0 * 1e3 / 0.002
    assert v["mulensevent.0.theta_E"] == pytest.approx(theta_e, rel=1e-4)
    pie = np.hypot(-0.060, -0.057)
    assert v["mulensevent.0.mlens_total"] == pytest.approx(
        theta_e / (KAPPA * pie), rel=1e-4
    )
    assert v["lens.1.q"] == pytest.approx(1.9187e-05, rel=1e-9)


def test_a_stated_planet_q_is_never_replaced_and_the_host_keeps_its_mass(
    monkeypatch,
):
    """
    Given examples/KMT-2019-BLG-1806 as shipped (user t_E, pi_E and a planet
      companion's q = 1.9187e-5),
    When the starts are solved,
    Then the user's q is the start, the companion's log_q is NOT declared
      likelihood-backed (a stated q already says what the caustic
      measures), and the host keeps a stellar mass.  With the lens mass
      below the backed level, a backed log_q = -3 default was fixed before
      the user's q could determine it, and the host mass -- the one seed
      left to reconcile two proportional statements -- went to zero
      (prototype of the source-first order: q replaced by 1e-3, M = 0).
    """
    system = _prepare(
        monkeypatch, "KMT-2019-BLG-1806", "KMT-2019-BLG-1806.yaml"
    )
    cm = system.config_manager
    v = cm._last_resolved
    assert "planet.0.log_q" not in cm.likelihood_backed
    assert v["lens.1.q"] == pytest.approx(1.9187e-05, rel=1e-9)
    assert v["planet.0.log_q"] == pytest.approx(np.log10(1.9187e-05))
    assert v["star.0.mass"] > 0.05
    assert not cm._last_keep_report["dropped"]


def _drop_q(params):
    params.pop("lens.Companion.q")
    return params


def test_an_unstated_companion_q_is_still_likelihood_backed(monkeypatch):
    """
    Given examples/KMT-2019-BLG-1806 with the companion's q removed,
    When its mass coordinate is registered,
    Then log_q is declared likelihood-backed as before (review 8.6.42):
      only a STATED q releases it.
    """
    system = _prepare(
        monkeypatch,
        "KMT-2019-BLG-1806",
        "KMT-2019-BLG-1806.yaml",
        _drop_q,
    )
    assert "planet.0.log_q" in system.config_manager.likelihood_backed


def test_a_seeded_t_e_without_theta_e_is_absorbed_by_the_lens_distance(
    monkeypatch,
):
    """
    Given examples/DC2018_128 (t_E and q seeded, a point source -- so no
      theta_E from rho -- and the galactic model on),
    When the starts are solved,
    Then both proper motions keep the galactic model's seeds, the lens
      keeps its mass guess, and t_E sets theta_E through the LENS distance,
      the lowest lens guess; the user's q is the start and nothing
      escalates.  A first trial of the source-first order (lens mass below
      the companion's backed log_q) put this lens at 0.5 pc with a
      ~1e-6 Msun host: the backed log_q = -3 default and the user's q
      disagreed, the host mass was the only seed left to reconcile two
      proportional statements, and the proper-motion settle that followed
      then had to deliver t_E at |mu_rel| ~ 0, where it is a kink.
    """
    system = _prepare(monkeypatch, "DC2018_128", "DC2018_128.yaml")
    cm = system.config_manager
    v = cm._last_resolved
    rep = cm._last_keep_report
    assert "mulensevent.0.t_E" in rep["kept"]
    assert not rep["escalated"], rep["escalated"]
    assert not rep["unsettled"], rep["unsettled"]
    assert "planet.0.log_q" not in cm.likelihood_backed
    for i in (0, 1):
        for axis in ("pm_ra", "pm_dec"):
            assert cm.initval_source("star", axis, element=i) == "data"
    assert _source_moves(rep, 1) == []
    assert v["star.0.mass"] == pytest.approx(10**-0.5)
    assert 1000.0 < v["star.0.distance"] < 8000.0
    assert v["lens.1.q"] == pytest.approx(1.10576828e-03, rel=1e-9)


def test_a_theta_e_the_user_fixed_is_absorbed_by_the_source_radii(
    monkeypatch,
):
    """
    Given examples/ob161003 (2S2L): user proper motions for the lens and
      source A and a user t_E -- so theta_E = mu_rel t_E is the user's --
      and a user rho for each of the two sources,
    When the starts are solved,
    Then the lens cannot absorb rho (theta_E is not the lens's to move),
      so the residual escalates to the source tier and the source RADII
      take it (0.65 and 1.87 Rsun, the radii those rhos imply at the
      user's theta_E), with the lens at its 0.32 Msun guess and the source
      distances untouched at 8 kpc -- the radii decide on their own tier
      below the distances (MULENS_SOURCE_RADIUS).  The first trial of
      the source-first order put the host at 112 Msun and both sources at
      5.15 kpc: the failed mass settle's least-violated point was held for
      the escalation, and the escalated tier could move source B's
      distance but not its radius.
    """
    system = _prepare(monkeypatch, "ob161003", "ob161003.yaml")
    cm = system.config_manager
    v = cm._last_resolved
    rep = cm._last_keep_report
    assert not rep["dropped"], rep["dropped"]
    assert not rep["unsettled"], rep["unsettled"]
    assert v["star.0.mass"] == pytest.approx(10**-0.5)
    assert v["mulensevent.0.mlens_total"] < 2.0
    for s_idx in (2, 3):
        assert v[f"star.{s_idx}.distance"] == pytest.approx(8000.0)
    assert v["star.2.radius"] == pytest.approx(0.651, rel=0.01)
    assert v["star.3.radius"] == pytest.approx(1.866, rel=0.01)


def test_a_finite_source_radius_decides_at_the_source_level(monkeypatch):
    """
    Given examples/KMT-2019-BLG-1806 (a finite source),
    When the event's kinematic hints are registered,
    Then the source radius is declared at MULENS_SOURCE_RADIUS and the
      lens's is not.
    """
    system = _prepare(
        monkeypatch, "KMT-2019-BLG-1806", "KMT-2019-BLG-1806.yaml"
    )
    cm = system.config_manager
    assert (
        cm.decision_levels.get("star.1.radius")
        == PRECEDENCE_MULENS_SOURCE_RADIUS
    )
    assert "star.0.radius" not in cm.decision_levels


def test_a_point_source_radius_is_not_ranked_against_rho(monkeypatch):
    """
    Given examples/kb180087_obj3, a POINT source whose params file still
      carries an unmeasured rho of 3.3e-6,
    When the starts are solved,
    Then the source radius is not declared at MULENS_SOURCE_RADIUS (the model
      reads no rho), so that rho sets no theta_E: the lens keeps its mass
      guess and a stellar distance.  Ranked like a finite source's, the
      1 Rsun radius turned that rho into theta_E = 177 mas, no lens could
      absorb it, and prepare() failed in the flux bootstrap.
    """
    system = _prepare(monkeypatch, "kb180087_obj3", "kb180087_obj3.yaml")
    cm = system.config_manager
    v = cm._last_resolved
    assert "star.1.radius" not in cm.decision_levels
    assert v["star.0.mass"] == pytest.approx(10**-0.5)
    assert 100.0 < v["star.0.distance"] < 8000.0


def test_a_decision_level_never_moves_a_reported_rank():
    """
    Given the ladder,
    When a decision level is declared on a path,
    Then the declaration is a NAME of the ladder -- a bare number raises,
      as a defaults.yaml `rank:` does.
    """
    cm = ConfigManager.__new__(ConfigManager)
    with pytest.raises(ValueError, match="not a level of the precedence"):
        cm.declare_decision_level("star.0.radius", 30)
