"""The mass-ratio start on a lens with three or more bodies.

Every companion's q is solved by the relaxation engine from the body masses:
symbolic_physics maps one relation set per companion and the total mass is a
sum over every body (review 8.6.13; mulensing.md, "Three or more lens
bodies").  Until 2026-10 only the first companion (lens element 1) was solved;
elements 2 and up carried a NaN bookkeeping initval, `_validate_q_start` had
to exempt them (review 1.6.5), and MulensEvent warned that a 3+ body lens
REQUIRED explicit body masses and that a user q on an extra companion could
not reach its mass (review 2.6.6).  Those exemptions and warnings are gone
with the cause; these tests pin what replaced them.
"""

import numpy as np
import pytest
from test_band_autopin_ld import T0, TE, U0, _write_pspl_lc

from exozippy.system import System


def _triple_lens_config(lc):
    """One stellar primary plus two planetary companions -- three lens BODIES,
    so n_companions == 2 (companion slot 1 is LENS ELEMENT 2: element 0 is
    the masked primary)."""
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "b"}, {"name": "c"}],
        # Neither companion moves, so each is paired with its host by a
        # STATIC orbit: with two planets in the lens the host is not left
        # implicit (orbit.bodies.planet_membership).
        "orbit": [
            {
                "name": nm,
                "type": "static",
                "primary": ["Lens"],
                "companion": [nm],
            }
            for nm in ("b", "c")
        ],
        "mulensevent": [
            {
                "finite_source": False,
                "t0_par": T0,
            }
        ],
        "lens": [
            {"body": "star.Lens"},
            {"body": "planet.b"},
            {"body": "planet.c"},
        ],
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": lc}],
    }


def _triple_lens_params(**extra):
    params = {
        "source.Source.t_0": {"initval": T0},
        "source.Source.u_0": {"initval": U0},
        "mulensevent.t_E": {"initval": TE},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("Lens", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    params.update(extra)
    return params


@pytest.fixture(scope="module")
def triple_lens_lc(tmp_path_factory):
    return _write_pspl_lc(tmp_path_factory.mktemp("triple_lens") / "lc.dat")


@pytest.fixture(scope="module")
def triple_lens_system(triple_lens_lc):
    """A three-body lens whose params file seeds only companion slot 0's q,
    prepared and BUILT -- the build is the assertion of the first test and the
    fixture of the second."""
    system = System(
        _triple_lens_config(triple_lens_lc),
        # Companion slot 0 is lens ELEMENT 1.
        user_params=_triple_lens_params(**{"lens.1.q": {"initval": 1e-3}}),
    )
    system.prepare()
    system.build_model()
    return system


def test_every_companion_q_is_solved(triple_lens_system):
    """
    Given a three-body lens whose params file seeds only the first
      companion's q (lens element 1),
    When the model is built,
    Then the second companion's q (lens element 2) has a finite start too:
      the engine solves it from the body masses.  It used to be a NaN
      bookkeeping initval that `_validate_q_start` had to exempt.
    """
    q = triple_lens_system.lens.q
    q0 = np.atleast_1d(np.asarray(q.initval, dtype=float))
    assert q0.size == 3
    assert np.all(np.isfinite(q0[1:]))
    # Element 0 is the masked primary; every other element is derived.
    assert not q.element_is_active(0)
    assert all(q.element_is_derived(i) for i in range(1, q0.size))


def test_a_nan_q_start_raises_on_any_companion(
    triple_lens_system, monkeypatch
):
    """
    Given the same lens with a NaN start on the SECOND companion,
    When the start values are validated,
    Then it raises: every companion is solved now, so a NaN means a failed
      solve, not an unsolved element's bookkeeping.
    """
    lens = triple_lens_system.lens
    q0 = np.array(lens.q.initval, dtype=float)
    q0[2] = np.nan
    monkeypatch.setattr(lens.q, "initval", q0)
    with pytest.raises(ValueError, match="not a number"):
        lens._validate_q_start()


def test_no_body_masses_are_required(triple_lens_lc, caplog):
    """
    Given a three-body lens with no body-mass entries,
    When the system is prepared,
    Then it does not warn that masses are required, and every companion's q
      has a solved start.
    """
    system = System(
        _triple_lens_config(triple_lens_lc), user_params=_triple_lens_params()
    )
    with caplog.at_level("WARNING"):
        system.prepare()
    assert "3+ bodies" not in caplog.text
    led = system.config_manager._last_resolved
    assert np.isfinite(led["lens.1.q"]) and np.isfinite(led["lens.2.q"])


def test_a_user_q_on_an_extra_companion_is_delivered(triple_lens_lc):
    """
    Given a three-body lens whose params file sets lens.2.q -- the SECOND
      companion -- and no body masses,
    When the system is prepared,
    Then the engine moves that companion's mass to deliver it: M_2 = q M_1,
      in the internal solMass of both (planet.mass is declared in
      jupiterMass).  It used to warn that the q could not reach the mass.
    """
    q2 = 1e-3
    system = System(
        _triple_lens_config(triple_lens_lc),
        user_params=_triple_lens_params(**{"lens.2.q": {"initval": q2}}),
    )
    system.prepare()
    led = system.config_manager._last_resolved
    m1 = 10.0 ** led["star.0.logmass"]
    assert led["lens.2.q"] == pytest.approx(q2)
    assert led["planet.1.mass"] / m1 == pytest.approx(q2, rel=1e-9)


def test_the_seeded_companion_mass_is_in_internal_units(triple_lens_lc):
    """
    Given a 3+ body lens whose companions are PLANETS (planet.mass is
      declared in jupiterMass, internal solMass),
    When the engine solves mlens_total and the per-element q from the
      user-supplied body masses,
    Then the seed agrees with what the built graph computes.

    History: the seed used to come from `MulensEvent._mass_initval` (gone
    with review 8.6.13's per-companion relations), which promised solMass
    and returned the raw user_params value.  For `star.mass` those coincide; `planet.mass` is
    jupiterMass, so on any lens with 2+ companions -- the only case that
    reaches that seeding branch -- mlens_total and every per-element q were
    seeded 1047x too large, and theta_E and t_E followed.  Measured before
    the fix: q seeded 1.995e-3 where the graph computes 1.9047e-6.

    THE ASSERTION IS SEED-vs-BUILT, deliberately, rather than a
    hand-computed number: it needs no jupiter/solar constant of its own, so
    it cannot repeat the mistake it exists to catch (CLAUDE.md -- never
    hand-write a conversion), and it fails on a slip in EITHER direction.

    Why the rest of this file missed it: every other test here asserts SHAPE
    and ROLES -- q0.size, isnan on the unseeded slot, active/derived -- and
    never a seeded VALUE, so a purely numeric error in the path they all
    exercise was invisible to them.
    """
    # Arrange -- a mass for every body, the planet masses in THEIR user unit.
    system = System(
        _triple_lens_config(triple_lens_lc),
        _triple_lens_params(
            **{
                "star.Lens.logmass": {"initval": -0.3},
                "planet.b.mass": {"initval": 1.0e-3},
                "planet.c.mass": {"initval": 2.0e-3},
            }
        ),
    )
    system.prepare()
    model = system.build_model()

    # Act -- the seed the engine resolved, and what the graph computes.
    q = system.lens.q
    seeded = np.atleast_1d(np.asarray(q.initval, dtype=float))

    import pytensor

    start = system.get_raw_start(model)
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([q.value]),
        on_unused_input="ignore",
    )
    (built,) = fn(*[start[v.name] for v in model.value_vars])
    built = np.atleast_1d(np.asarray(built, dtype=float))

    # Assert -- on the ACTIVE elements only: element 0 is the masked
    # primary, whose seed is NaN by design and whose built value is the
    # bookkeeping pin.
    active = [i for i in range(seeded.size) if q.element_is_active(i)]
    assert active, "no active companion elements; the fixture stopped biting"
    for i in active:
        assert seeded[i] == pytest.approx(built[i], rel=1e-6), (
            f"element {i}: seeded {seeded[i]:.6e} but the graph computes "
            f"{built[i]:.6e} (ratio {seeded[i] / built[i]:.1f}); a body mass "
            f"is being seeded in its USER unit instead of solMass"
        )


# ---------------------------------------------------------------------------
# Mixed companion types (review 8.6.21)
# ---------------------------------------------------------------------------


def _mixed_config(lc, companions):
    """A lens of [star primary] + `companions`, each ("star"|"planet", name).

    Built per test rather than as a module fixture because the POINT is to
    vary the body order.
    """
    stars = [{"name": "Lens"}, {"name": "Source"}]
    planets = []
    lens_entries = [{"body": "star.Lens"}]
    for kind, name in companions:
        if kind == "star":
            stars.append({"name": name})
            lens_entries.append({"body": f"star.{name}"})
        else:
            planets.append({"name": name})
            lens_entries.append({"body": f"planet.{name}"})

    cfg = {
        "star": stars,
        "mulensevent": [{"finite_source": False, "t0_par": T0}],
        "lens": lens_entries,
        "source": [{"body": "star.Source"}],
        "mulensinstrument": [{"name": "OGLE", "file": lc}],
    }
    if planets:
        cfg["planet"] = planets
        # Each planet companion is paired with the lens primary by a static
        # orbit (it does not move); with a second lens star beside it, its
        # host is not left implicit (orbit.bodies.planet_membership).
        cfg["orbit"] = [
            {
                "name": p["name"],
                "type": "static",
                "primary": ["Lens"],
                "companion": [p["name"]],
            }
            for p in planets
        ]
    return cfg


def _mixed_params(companions):
    params = _triple_lens_params(**{"star.Lens.logmass": {"initval": -0.3}})
    for kind, name in companions:
        if kind == "star":
            params[f"star.{name}.logmass"] = {"initval": -1.5}
        else:
            params[f"planet.{name}.mass"] = {"initval": 1.0e-3}
    return params


def _built_q(system, model):
    import pytensor

    q = system.lens.q
    start = system.get_raw_start(model)
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([q.value]),
        on_unused_input="ignore",
    )
    (built,) = fn(*[start[v.name] for v in model.value_vars])
    return q, np.atleast_1d(np.asarray(built, dtype=float))


def _q_by_kind(lc, companions):
    """{companion kind: built q} for one body order."""
    system = System(_mixed_config(lc, companions), _mixed_params(companions))
    system.prepare()
    model = system.build_model()
    q, built = _built_q(system, model)
    seeded = np.atleast_1d(np.asarray(q.initval, dtype=float))
    out = {}
    for j, (kind, _name) in enumerate(companions):
        elem = j + 1  # element 0 is the masked primary
        assert q.element_is_active(elem), f"element {elem} is not active"
        out[kind] = (built[elem], seeded[elem])
    return out


@pytest.mark.slow
def test_a_mixed_type_lens_builds_and_pairs_each_body_correctly(
    triple_lens_lc,
):
    """
    Given a lens whose companions are one STAR and one PLANET,
    When q is built,
    Then each element's q is its OWN body's mass ratio, in either body
      order.

    ORDER INDEPENDENCE IS THE POINT.  `companion_body_map` holds each body's
    index within its own component, so read through a single typed dep the
    planet's element pointed at a STAR -- for [star, star, planet] at
    star.0, the primary, giving q = M1/M1 = 1.0 with the planet's mass never
    entering.  Measured before the fix: [1.0, 0.0631, 1.0] and, in the other
    order, [1.0, 1.9e-06, 1.9e-06] -- both companions taking one value.

    The fix scatters per-type masses into the elements each type owns
    (`calc_q_mixed`, `pt.set_subtensor`), which is order-free.  A
    `pt.concatenate` of per-type vectors -- the reserved branch in `calc_q`
    -- would pass a grouped lens and silently mis-pair an interleaved one,
    because concatenation yields companions in TYPE order and that equals
    ELEMENT order only when the config happens to group them.  Hence
    comparing the two orders against EACH OTHER.
    """
    # Arrange / Act -- the same two bodies, both ways round.
    star_first = _q_by_kind(
        triple_lens_lc, [("star", "LensB"), ("planet", "b")]
    )
    planet_first = _q_by_kind(
        triple_lens_lc, [("planet", "b"), ("star", "LensB")]
    )

    # Assert -- a body's q does not depend on where it sits.
    for kind in ("star", "planet"):
        assert star_first[kind][0] == pytest.approx(
            planet_first[kind][0], rel=1e-9
        ), (
            f"the {kind} companion's q changed with body order: "
            f"{star_first[kind][0]:.6e} vs {planet_first[kind][0]:.6e} -- the "
            f"per-type masses are being paired by type order, not element"
        )

    # And the two companions are genuinely different, so the check above is
    # not comparing a value to itself.
    assert star_first["star"][0] != pytest.approx(
        star_first["planet"][0], rel=1e-3
    ), (
        "the star and planet companions have indistinguishable q; fixture is weak"
    )


@pytest.mark.slow
def test_a_mixed_type_lens_seeds_what_it_builds(triple_lens_lc):
    """
    Given a mixed-type lens,
    When the engine seeds q and the graph computes it,
    Then they agree per element.

    The same seed-vs-built cross-check that caught the jupiterMass seeding
    bug (8.6.24), applied to the mixed path: it needs no expected constant
    of its own, so it cannot encode the error it guards against.
    """
    for companions in (
        [("star", "LensB"), ("planet", "b")],
        [("planet", "b"), ("star", "LensB")],
    ):
        got = _q_by_kind(triple_lens_lc, companions)
        for kind, (built, seeded) in got.items():
            assert seeded == pytest.approx(built, rel=1e-6), (
                f"{kind} companion in order {[c[0] for c in companions]}: "
                f"seeded {seeded:.6e} but the graph computes {built:.6e}"
            )
