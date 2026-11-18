"""planet.radius exists only where something reads it (review 2.8.17, #PR).

A microlensing planet is read through its MASS (the lens's q); nothing in a
2L1S likelihood reads its radius.  It used to be sampled anyway -- one free
dimension refilling its prior, plus R_P, rho_P and log g_P table rows that
were pure prior.  `Planet.radius_consumers` is the readership predicate
(transit, rm, dopptom, the Chen & Kipping relation, a chord-sampling orbit, a
user prior), and `Planet._apply_radius_activity` makes the radius INACTIVE on
every planet it finds no reader for -- the treatment
`Star._apply_structure_activity` gives a star radius nothing reads -- with
density and logg inheriting the role (`Component._inherit_inactivity`).
"""

import numpy as np
import pytest

from exozippy.components.planet.planet import Planet
from exozippy.manifest import interpret_manifest_entry
from exozippy.system import System


def _binary_config():
    """Minimal single-source binary lens: star.Lens hosts planet b (lens
    element 1).  No data, no orbit: the static pairing a 2L1S fit uses."""
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "b"}],
        "mulensevent": [{"finite_source": False}],
        "lens": [{"body": "star.Lens"}, {"body": "planet.b"}],
        "source": [{"body": "star.Source"}],
    }


def _binary_params(extra=None):
    p = {
        "source.Source.t_0": {"initval": 2460025.0, "init_scale": 0.1},
        "source.Source.u_0": {"initval": 0.1, "init_scale": 0.01},
        "mulensevent.t_E": {"initval": 30.0, "init_scale": 1.0},
        "lens.b.s": {"initval": 0.98},
        "lens.b.alpha": {"initval": 60.0, "init_scale": 0.9},
        "lens.b.q": {"initval": 1e-3, "init_scale": 1e-4},
    }
    for nm in ("Lens", "Source"):
        p[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        p[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    # Stated proper motions, so the relaxation engine has no default-tier
    # start to relax (see test_the_free_radius_leaves_with_only_its_own_terms:
    # otherwise the comparison measures the solver's tolerance).
    for nm, pm in (("Lens", -5.4), ("Source", -0.6)):
        p[f"star.{nm}.pm_ra"] = {"initval": pm}
        p[f"star.{nm}.pm_dec"] = {"initval": pm}
    p.update(extra or {})
    return p


def _prepared(extra_params=None):
    system = System(_binary_config(), user_params=_binary_params(extra_params))
    system.prepare()
    return system


def _radius_active(planet):
    """The per-planet activity the manifest declares for `radius`."""
    options = interpret_manifest_entry(planet.manifest["radius"]).options
    mask = options.get("mask")
    return (
        [True] * planet.n_elements if mask is None else [bool(m) for m in mask]
    )


def test_a_lens_planet_radius_is_inactive():
    """
    Given a 2L1S microlensing system whose companion is a planet,
    When it is prepared,
    Then nothing reads the planet's radius and the manifest declares it
      inactive (still declared: density and logg name it in their deps).
    """
    # Act
    system = _prepared()

    # Assert
    planet = system.planet
    assert planet.radius_consumers(system) == [[]]
    assert "radius" in planet.manifest
    assert _radius_active(planet) == [False]


@pytest.mark.parametrize(
    "entry",
    [
        {"mu": 0.1, "sigma": 0.01},
        {"lower": 0.01},
    ],
    ids=["prior", "bound"],
)
def test_a_user_constraint_on_the_radius_reads_it(entry):
    """
    Given the same system with a params-file prior or bound on planet.radius,
    When it is prepared,
    Then the user's constraint is the reader and the radius stays active.
    """
    # Act
    system = _prepared({"planet.b.radius": dict(entry)})

    # Assert
    assert system.planet.radius_consumers(system) == [["user prior on radius"]]
    assert _radius_active(system.planet) == [True]


def test_a_user_prior_on_a_quantity_derived_from_the_radius_reads_it():
    """
    Given a params-file prior on planet.density, which derives from radius,
    When the system is prepared,
    Then the radius stays active (the prior is a likelihood term in R_P).
    """
    # Act
    system = _prepared({"planet.density": {"mu": 1.3, "sigma": 0.2}})

    # Assert
    assert _radius_active(system.planet) == [True]


def test_a_user_pin_is_subsumed_by_the_inactive_role():
    """
    Given a params-file `sigma: 0` on planet.radius,
    When the system is prepared,
    Then it is a pin, not a reader, and the radius is inactive -- exactly
      Star's rule for its structure parameters.
    """
    # Act
    system = _prepared({"planet.b.radius": {"initval": 0.1, "sigma": 0}})

    # Assert
    assert _radius_active(system.planet) == [False]


def test_chen_reads_the_radius():
    """
    Given the 2L1S system with `chen: true` forced on the planet,
    When it is prepared,
    Then the Chen & Kipping potential is the reader and the radius is active.
    """
    # Arrange
    config = _binary_config()
    config["planet"][0]["chen"] = True

    # Act
    system = System(config, user_params=_binary_params())
    system.prepare()

    # Assert
    assert system.planet.radius_consumers(system) == [["chen"]]
    assert _radius_active(system.planet) == [True]


def _term_logps(model, point):
    """{term name: float} for every free RV, observed RV and potential."""
    names = [
        v.name for v in model.free_RVs + model.observed_RVs + model.potentials
    ]
    fn = model.compile_fn(
        model.logp(sum=False, jacobian=True),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return {n: float(np.sum(v)) for n, v in zip(names, fn(point), strict=True)}


def _start(system, model):
    point = dict(model.initial_point())
    point.update(system.get_raw_start(model))
    return point


def _built(extra_params=None):
    """(system, per-term start logps, raw start, free RV names)."""
    system = _prepared(extra_params)
    model = system.build_model()
    terms = _term_logps(model, _start(system, model))
    return (
        system,
        terms,
        system.get_raw_start(model),
        {v.name for v in model.free_RVs},
    )


def _rule_off(monkeypatch):
    """The behavior before review 2.8.17: every planet radius active."""
    monkeypatch.setattr(
        Planet, "_apply_radius_activity", lambda self, system: None
    )


@pytest.mark.slow
def test_inactive_is_bit_identical_to_the_old_hand_pin(monkeypatch):
    """
    Given the 2L1S system built now, and built with the rule switched off
      but the radius pinned by hand (`sigma: 0`), which is how every shipped
      microlensing example with a planet held it,
    When each model's per-term logp is evaluated at its own start,
    Then the two models sample the same leaves from the same start and every
      term -- prior, potential and likelihood -- is BIT-IDENTICAL, so the hand
      pins are redundant; and the rows derived from the bookkeeping radius
      are not reported.
    """
    # Arrange / Act
    system, new, new_start, new_free = _built()
    _rule_off(monkeypatch)
    _, old, old_start, old_free = _built({"planet.b.radius": {"sigma": 0}})

    # Assert
    assert new_free == old_free
    assert not any(n.startswith("planet.radius") for n in new_free)
    assert new == old
    assert set(new_start) == set(old_start)
    for name, value in new_start.items():
        np.testing.assert_array_equal(value, old_start[name], err_msg=name)
    assert np.isfinite(sum(new.values()))
    planet = system.planet
    for name in ("radius", "density", "logg"):
        assert not getattr(planet, name).element_is_active(0), name


@pytest.mark.slow
def test_the_free_radius_leaves_with_only_its_own_terms(monkeypatch):
    """
    Given the 2L1S system built now, and built with the rule switched off
      and the radius left free (the old default: a sampled, prior-only
      dimension),
    When each model's per-term logp is evaluated at its own start,
    Then exactly the planet.radius coordinate and its prior terms are gone,
      and every other start coordinate and every shared term is
      BIT-IDENTICAL.

    The fixture states the proper-motion starts on purpose.  Left at their
    defaults, the relaxation engine relaxes them jointly with every other
    default-tier start, to a solver tolerance; the old radius was one more
    seed in that solve (its defaults.yaml value seeds a SAMPLED element and
    only fills an inactive one), so removing it moved the solved proper
    motions by ~4e-9 mas/yr and the event-rate prior by 1.5e-8 nats.  That
    is a different start, not a different model, and stating the starts
    removes it.
    """
    # Arrange / Act
    _, new, new_start, new_free = _built()
    _rule_off(monkeypatch)
    _, old, old_start, old_free = _built()

    # Assert: exactly the radius left the sampled set...
    removed = old_free - new_free
    assert removed and all(n.startswith("planet.radius") for n in removed), (
        removed
    )
    assert new_free <= old_free
    # ...with the terms it carried, and nothing else.
    dropped = set(old) - set(new)
    assert dropped and all("planet.radius" in n for n in dropped), dropped
    assert set(new) <= set(old)
    for name, value in new_start.items():
        np.testing.assert_array_equal(value, old_start[name], err_msg=name)
    for name, value in new.items():
        assert value == old[name], (name, value, old[name])
