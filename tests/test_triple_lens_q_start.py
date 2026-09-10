"""A NaN bookkeeping initval on the DERIVED mass ratio must not kill the fit.

The relaxation engine's mass-sum and q relations are binary-only
(`mulensing/symbolic_physics.get_symbol_map` maps one companion slot), so for a
lens with three or more bodies `MulensEvent.register_parameters` seeds the
per-slot q initvals from USER mass entries only.  With no such entries the hint
is skipped and `resolve()` leaves the unseeded elements NaN, because `q` has no
defaults.yaml initval.

`_validate_q_start` then raised at stage 7 -- over a parameter that is
DERIVED, whose runtime value comes from the mass nodes (finite defaults) and
never from that initval.  Exactly the false-positive class
`_validate_pspl_start`'s docstring warns about (the ob161003 theta_E lesson).
Review 1.6.5.

The raise is kept for companion slot 0 -- LENS ELEMENT 1 post-split, since
element 0 is the masked primary -- which the engine really does solve, so a NaN
there really does mean a non-finite lens body mass.
"""

import numpy as np
import pytest
from test_band_autopin_ld import T0, TE, U0, _write_pspl_lc

from exozippy.system import System


def _triple_lens_config(lc):
    """One stellar primary plus two planetary companions -- three lens BODIES,
    so n_companions == 2 and the binary-only relations cannot cover companion
    slot 1 (which is LENS ELEMENT 2: element 0 is the masked primary)."""
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "planet": [{"name": "b", "star_ndx": 0}, {"name": "c", "star_ndx": 0}],
        "mulensevent": [
            {
                "finite_source": False,
                "t0_par": T0,
                # Never shell out to MMEXOFAST from a unit test.
                "mmexofast": False,
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


def test_partially_seeded_derived_q_builds(triple_lens_system):
    """
    Given a three-body lens whose params file seeds only companion slot 0's q
      (lens element 1),
    When the model is built,
    Then it builds: slot 1's -- lens element 2's -- NaN initval is the
      engine's bookkeeping for a DERIVED parameter, not a start value, and the
      graph recomputes q from the mass nodes.  This used to raise at stage 7
      (review 1.6.5).

    The fixture doing the building is the point -- the assertions below only
    confirm that the state which used to be fatal is still exactly what it
    was, and that only the verdict on it changed.
    """
    # Arrange / Act -- see the fixture

    # Assert
    q = triple_lens_system.lens.q
    q0 = np.atleast_1d(np.asarray(q.initval, dtype=float))
    # Three bodies -> three elements; element 2 is the unseeded companion.
    assert q0.size == 3
    assert np.isnan(q0[2])
    # Element 0 is the masked primary: q is not a parameter of it at all
    # (inactive, so neither sampled nor derived).  Every element that IS a
    # parameter is derived from the body masses.
    assert not q.element_is_active(0)
    assert all(q.element_is_derived(i) for i in range(1, q0.size))


def test_sampled_q_with_a_nan_start_still_raises(
    triple_lens_system, monkeypatch
):
    """
    Given the same lens but with q's elements reported as SAMPLED,
    When the start values are validated,
    Then the NaN still raises: for a sampled element the initval IS the start,
      so the exemption above must not widen into "a NaN in q is fine".
    """
    # Arrange -- only the ROLE is falsified; the initval vector is untouched,
    # which is what isolates the derived-vs-sampled distinction the fix turns
    # on.
    lens = triple_lens_system.lens
    monkeypatch.setattr(
        type(lens.q), "element_is_derived", lambda self, index=0: False
    )

    # Act / Assert
    with pytest.raises(ValueError, match="not a number"):
        lens._validate_q_start()


# ---------------------------------------------------------------------------
# The requirement itself, said out loud at config time (review 2.6.6)
# ---------------------------------------------------------------------------


def test_missing_body_masses_warn_at_config_time(triple_lens_lc, caplog):
    """
    Given a three-body lens with no body-mass entries,
    When the lens registers its parameters,
    Then it WARNS, naming the bodies and the requirement: the engine's
      mass-sum and q relations are binary-only, so nothing else can supply
      mlens_total or the per-companion q starts.  This used to be an INFO,
      which for a user who has only ever fitted 2-body lenses (where the
      engine derives all of it) is no signal at all.
    """
    # Arrange
    system = System(
        _triple_lens_config(triple_lens_lc), user_params=_triple_lens_params()
    )

    # Act
    with caplog.at_level("WARNING"):
        system.prepare()

    # Assert
    assert "3+ bodies" in caplog.text
    assert "planet.0" in caplog.text and "planet.1" in caplog.text


def test_a_user_q_on_an_extra_companion_warns(triple_lens_lc, caplog):
    """
    Given a three-body lens whose params file sets lens.2.q -- the SECOND
      companion, since companion slot j is lens element j+1,
    When the lens registers its parameters,
    Then it warns that the entry sets a derived parameter's START but cannot
      set the companion mass the runtime value is computed from -- so the fit
      runs at the masses, not at the q that was typed (review 2.6.6).
    """
    # Arrange
    system = System(
        _triple_lens_config(triple_lens_lc),
        user_params=_triple_lens_params(**{"lens.2.q": {"initval": 1e-3}}),
    )

    # Act
    with caplog.at_level("WARNING"):
        system.prepare()

    # Assert
    assert "lens.2.q" in caplog.text
    assert "CANNOT set the companion mass" in caplog.text


def test_the_seeded_companion_mass_is_in_internal_units(triple_lens_lc):
    """
    Given a 3+ body lens whose companions are PLANETS (planet.mass is
      declared in jupiterMass, internal solMass),
    When MulensEvent seeds mlens_total and the per-element q from the
      user-supplied body masses,
    Then the seed agrees with what the built graph computes.

    `MulensEvent._mass_initval` promises solMass and used to return the raw
    user_params value.  For `star.mass` those coincide; `planet.mass` is
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
    # Arrange -- a mass for every body, which is what makes the 3+ body
    # branch seed rather than warn, with the planet masses in THEIR user unit.
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
