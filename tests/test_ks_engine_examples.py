"""The KS relaxation rule on shipped examples (review 3.14.22).

The toy-relation tests are in tests/test_ks_engine.py; these are the cases
the review item and the engine proposal named on real configs, run through
System.prepare() only (no model build), so each costs one prepare.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from exozippy.config import ConfigManager
from exozippy.system import System

EXAMPLES = Path(__file__).parent.parent / "examples"
MASS_SUFFIXES = (".mass", ".mlens_total", ".m_total")


def _prepare(monkeypatch, example, config_name, edit_params=None):
    d = EXAMPLES / example
    config = yaml.safe_load((d / config_name).read_text())
    params = yaml.safe_load((d / config["parameter_file"]).read_text()) or {}
    if edit_params:
        params = edit_params(copy.deepcopy(params))
    config["parameter_file"] = None
    monkeypatch.chdir(d)
    system = System(config, params)
    system.prepare()
    return system


def test_a_lens_mass_hint_below_the_distance_seed_moves_no_mass_negative(
    monkeypatch,
):
    """
    Given examples/DC2018_128 with the lens-mass hint dropped to precedence
      22, below the lens-distance seed (review 3.14.22(2)),
    When the starts are solved,
    Then no mass comes out negative and the HOST's logmass is what absorbs
      the pinned mass ratio.  The old engine rewrote `planet.0.mass` (it
      sorts before `star.0.mass`) to -0.2548 solMass and started at a logp
      of -912820.
    """
    orig = ConfigManager.finalize_user_params

    def finalize(self):
        for p in list(self.hints):
            if p.endswith(".distance") and self.hints[p] == 4000.0:
                lm = p.replace(".distance", ".logmass")
                if lm in self.hints:
                    self.hint_ranks[lm] = 22
        return orig(self)

    monkeypatch.setattr(ConfigManager, "finalize_user_params", finalize)
    system = _prepare(monkeypatch, "DC2018_128", "DC2018_128.yaml")
    cm = system.config_manager
    masses = {
        k: v for k, v in cm._last_resolved.items() if k.endswith(MASS_SUFFIXES)
    }
    assert masses and all(v > 0 for v in masses.values()), masses
    moved = {m for m, *_ in cm._last_keep_report["moved"]}
    assert "star.0.logmass" in moved


def test_a_pinned_density_is_absorbed_by_the_radius_not_the_mist_mass(
    monkeypatch,
):
    """
    Given examples/hat3 hat3_mist (an evolutionary model on star A) with the
      user's mass and radius pins replaced by a pinned density,
    When the starts are solved,
    Then the RADIUS absorbs the density and the mass keeps its seed: the
      evolutionary model declares the mass likelihood-backed
      (PRECEDENCE_LIKELIHOOD_BACKED), so the ordinary radius default gives
      first.  And the density is delivered -- the start's density is the
      pinned 2.6 g/cm3 (the relation now carries the 3/(4 pi) of a mean
      density, as the model's calc_density does).  The old engine moved the
      mass (alphabetically first) to 0.44 solMass.
    """

    def edit(params):
        params.pop("star.mass", None)
        params.pop("star.radius", None)
        params["star.A.density"] = {"initval": 2.6}
        return params

    system = _prepare(monkeypatch, "hat3", "hat3_mist.yaml", edit)
    cm = system.config_manager
    mass = cm._last_resolved["star.0.mass"]
    radius = cm._last_resolved["star.0.radius"]
    # the mass is its defaults.yaml 1 solMass, untouched by the density
    assert mass == pytest.approx(1.0, rel=1e-12)
    rho_internal = 3.0 / (4.0 * np.pi) * mass / radius**3
    rho_user = cm.from_internal("star.0.density", rho_internal)
    assert rho_user == pytest.approx(2.6, rel=1e-6)
    rep = cm._last_keep_report
    assert "star.0.density" in rep["kept"]
    assert "star.0.logmass" not in {m for m, *_ in rep["moved"]}
    assert "star.0.radius" in {a[0] for a in rep["absorbed"]}


def test_a_pinned_t_E_is_absorbed_by_the_lens_distance(monkeypatch):
    """
    Given examples/ob08092, whose params pin t_E on top of the lens and
      source seeds,
    When the starts are solved,
    Then t_E is held exactly and the LENS distance -- the lowest seed on
      the chain, below the source distance -- is what absorbs it (review
      3.14.22(1): theta_E, a derived symbol, carries no rank of its own to
      lose a tie with, so the old alphabetical trap cannot arise).
    """
    system = _prepare(monkeypatch, "ob08092", "ob08092.yaml")
    cm = system.config_manager
    rep = cm._last_keep_report
    assert "mulensevent.0.t_E" in rep["kept"]
    lens = next(
        a for a in rep["absorbed"] if a[0].endswith(".distance") and a[3] == 25
    )
    assert lens[2] != pytest.approx(lens[1])  # moved off the 4 kpc guess
