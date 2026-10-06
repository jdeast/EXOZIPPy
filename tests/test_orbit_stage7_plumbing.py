"""The orbit's fitvcve/fitchord stage-7 terms never skip (review 2.8.6).

`Orbit._add_vcve_terms`, `_add_chord_terms` and `_unclipped_ecc` used to
RETURN when a Parameter or stashed node they read was missing -- silently
dropping the V_c/V_e Jacobian, the root mixture, the chord Jacobian or the
collision bound.  JDE 2026-10-06: every node they read must exist by stage 7,
and a configuration that could leave one absent must be refused at stage 3.

Two halves, tested separately:

* the one CONFIGURATION that reached stage 7 without what these terms read --
  an implicit orbit paired with a planet that does not exist, made a chord
  orbit -- now resolves (or raises) at prepare(), naming the orbit;
* with that in place, a missing node at stage 7 is an internal bookkeeping
  bug, and each one raises RuntimeError naming the orbit.
"""

import numpy as np
import pytest

from exozippy.components.orbit.orbit import Orbit
from exozippy.system import System

_PARAMS = {
    "star.0.radius": {"initval": 1.61, "sigma": 0.05},
    "star.0.mass": {"initval": 1.204, "sigma": 0.05},
    "star.0.teff": {"initval": 6207, "sigma": 100},
    "star.0.feh": {"initval": -0.116, "sigma": 0.08},
    "orbit.0.period": {"initval": 2.99},
    "orbit.0.tc": {"initval": 2459634.3},
    "planet.0.radius": {"initval": 1.7},
}


def _params(cfg):
    return {k: v for k, v in _PARAMS.items() if k.split(".")[0] in cfg}


@pytest.fixture(scope="module")
def transit_lc(tmp_path_factory):
    path = tmp_path_factory.mktemp("stage7") / "lc.dat"
    rng = np.random.default_rng(11)
    t = np.linspace(2459634.1, 2459634.5, 150)
    np.savetxt(
        path,
        np.column_stack(
            [t, 1.0 + rng.normal(0.0, 1e-3, t.size), np.full(t.size, 1e-3)]
        ),
    )
    return str(path)


def _transit_config(lc, orbits=None):
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": orbits or [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "inst0", "file": lc, "band": "TESS"}],
    }


# ---------------------------------------------------------------------------
# 1. Stage 3: the configurations
# ---------------------------------------------------------------------------


def test_fitchord_on_an_orbit_with_no_planet_raises_at_prepare():
    """
    Given an implicit orbit that no planet block points at, in a system with
      no planets at all, and an explicit `fitchord: true`,
    When the system is prepared,
    Then it raises naming the orbit and saying it holds no planet.

    Before review 2.8.6 the implicit pairing's phantom `planet.0` made this a
    chord orbit, built from all-zero p and a/R* placeholders: a finite logp, a
    Jacobian and a geometry bound evaluated on a planet that does not exist.
    """
    cfg = {
        "star": [{"name": "A", "mist": False}],
        "orbit": [{"name": "x", "fitchord": True}],
    }
    with pytest.raises(ValueError, match=r"\[orbit\.x\].*fitchord.*holds 0"):
        System(cfg, user_params=_params(cfg)).prepare()


def test_an_orbit_with_a_phantom_planet_has_no_chord(transit_lc):
    """
    Given a one-planet transit fit with a second implicit orbit nothing points
      at (so both default to the transit-only parameterization),
    When the system is prepared,
    Then the planet's orbit samples the chord and the planet-less one is
      `nochord`.

    Before review 2.8.6 the second orbit was a chord orbit reading planet
    index 1 of a one-planet vector: the graph compiled (with rewrite failures)
    and returned a finite, meaningless logp.
    """
    cfg = _transit_config(transit_lc, [{"name": "b"}, {"name": "c"}])
    system = System(cfg, user_params=_params(cfg))
    system.prepare()
    assert system.orbit._chord_planet == [0, -1]
    assert system.orbit.inc_modes == ["chord", "nochord"]


# ---------------------------------------------------------------------------
# 2. Stage 7: internal invariants
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(transit_lc):
    """A transit-only fit: both halves on by default, one model built."""
    cfg = _transit_config(transit_lc)
    system = System(cfg, user_params=_params(cfg))
    system.prepare()
    model = system.build_model()
    assert system.orbit.ecc_modes == ["vcve"]
    assert system.orbit.inc_modes == ["chord"]
    return system, model


@pytest.mark.slow
@pytest.mark.parametrize("missing", ["vcve", "ecc", "omega"])
def test_vcve_terms_raise_on_a_missing_parameter(built, monkeypatch, missing):
    """A fitvcve orbit with no built vcve/ecc/omega raises, naming orbit 'b'."""
    system, model = built
    monkeypatch.setattr(system.orbit, missing, None)
    with model, pytest.raises(RuntimeError, match=rf"fitvcve.*'b'.*{missing}"):
        system.orbit._add_vcve_terms(system)


@pytest.mark.slow
@pytest.mark.parametrize("missing", ["chord", "ecc", "esinw"])
def test_chord_terms_raise_on_a_missing_parameter(built, monkeypatch, missing):
    """A fitchord orbit with no built chord/ecc/esinw raises, naming 'b'."""
    system, model = built
    monkeypatch.setattr(system.orbit, missing, None)
    with (
        model,
        pytest.raises(RuntimeError, match=rf"fitchord.*'b'.*{missing}"),
    ):
        system.orbit._add_chord_terms(system)


@pytest.mark.slow
def test_chord_terms_raise_without_this_builds_geometry(built, monkeypatch):
    """No stashed planet geometry at stage 7 raises rather than skipping."""
    system, model = built
    monkeypatch.setattr(system.orbit, "_chord_geometry", None)
    with model, pytest.raises(RuntimeError, match=r"'b'.*planet geometry"):
        system.orbit._add_chord_terms(system)


@pytest.mark.slow
def test_vcve_terms_raise_without_the_unclipped_root(built, monkeypatch):
    """No unclipped root node for a fitvcve orbit raises, naming 'b'."""
    system, model = built
    monkeypatch.setattr(system.orbit, "_vcve_unclipped_nodes", None)
    with model, pytest.raises(RuntimeError, match=r"unclipped.*'b'"):
        system.orbit._add_vcve_terms(system)


@pytest.mark.slow
@pytest.mark.parametrize("missing", ["vcve", "omega"])
def test_collision_bound_raises_without_the_vcve_root(
    built, monkeypatch, missing
):
    """The collision bound on a fitvcve orbit never silently disappears."""
    system, model = built
    monkeypatch.setattr(system.orbit, missing, None)
    with model, pytest.raises(RuntimeError, match=rf"'b'.*{missing}"):
        system.orbit._add_eccentricity_bound(system)


@pytest.mark.slow
def test_collision_bound_raises_on_a_mode_list_of_the_wrong_length(
    built, monkeypatch
):
    """An eccentricity mode list not one-per-orbit raises (was: all-hk)."""
    system, model = built
    monkeypatch.setattr(system.orbit, "ecc_modes", ["vcve", "hk"])
    with model, pytest.raises(RuntimeError, match=r"2 eccentricity mode"):
        system.orbit._unclipped_ecc()


@pytest.mark.slow
def test_collision_bound_raises_without_planet_max_ecc(built, monkeypatch):
    """A present planet component always carries max_ecc; a miss raises."""
    system, model = built
    monkeypatch.setattr(system.planet, "max_ecc", None)
    with model, pytest.raises(RuntimeError, match=r"planet\.max_ecc"):
        system.orbit._add_eccentricity_bound(system)


@pytest.mark.slow
def test_a_lost_chord_geometry_fails_the_build(transit_lc, monkeypatch):
    """
    Given a chord fit whose stage-6 geometry stash is lost,
    When the model is built end to end,
    Then build_model raises at stage 7 instead of dropping the Jacobian.
    """
    real = Orbit._chord_context

    def lossy(self, model, system):
        ctx = real(self, model, system)
        self._chord_geometry = None
        return ctx

    monkeypatch.setattr(Orbit, "_chord_context", lossy)
    cfg = _transit_config(transit_lc)
    system = System(cfg, user_params=_params(cfg))
    system.prepare()
    with pytest.raises(RuntimeError, match=r"'b'.*planet geometry"):
        system.build_model()


def test_the_stashed_nodes_are_per_build_caches():
    """A rebuild clears the chord geometry and the unclipped root first."""
    assert {"_chord_geometry", "_vcve_unclipped_nodes"} <= set(
        Orbit.per_build_caches
    )
