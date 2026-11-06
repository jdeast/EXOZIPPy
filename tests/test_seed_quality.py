"""Guard the goodness of fit AT THE SEED for every microlensing example.

The seed (``System.get_raw_start``) is where a production fit starts, so its
chi2 is the number
that says whether initialization is doing its job -- reproducing the solution
the observables came from.  Nothing checked it before, and that is exactly how a
change to the proper-motion start values passed 1272 tests while making two
examples' starts 60x and 2x worse:

    ob140939  chi2/N  3.04 -> 179.1
    ob161003  chi2/N  1.72 ->   3.9

A whole-suite pass means the model still *builds* and the physics still agrees
with its references.  It says nothing about starting somewhere sensible.

The bounds below are deliberately loose (a generous ceiling per example, not the
measured value to 4 decimals): they are here to catch a regression of that
magnitude, not to freeze the seeding.  Tighten only with a reason.

Note chi2/N > 1 at the seed is expected and is not a defect: a PSPL seed fits
with pi_E = 0, so its seeded t_E/u_0/t_0 describe a no-parallax model while this one
applies a derived pi_E, and the published configs leave the lens mass, distance
and proper motion open for the engine to fill.
"""

import glob
import os
import pathlib
import shutil
import tempfile

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.components.galacticmodel.physics import expected_proper_motion
from exozippy.diagnostics import ModelAuditor
from exozippy.system import System

EXAMPLES_DIR = pathlib.Path(__file__).parent / ".." / "examples"

pytestmark = pytest.mark.slow

# (example, ceiling on chi2/N at the seed).  Re-measured 2026-09-07 after the
# mulensevent split and the multi-source galactic-pm gate's retirement:
#   DC2018_128 1.213, ob08092 1.406, ob140939 3.113, ob161003 1.702
# (2026-08-11, pre-split: 1.213, 1.415, 3.039, 1.717 -- the split moved every
# one of these by less than the ceilings' slack, which is the point.)
CEILINGS = [
    ("DC2018_128", 2.0),
    ("ob08092", 2.5),
    ("ob140939", 5.0),
    ("ob161003", 3.0),
]


def _seed_chi2(name, strip_prefixes=(), extra_params=None):
    """chi2 and chi2/N of the mulens observation at the raw start point.

    `strip_prefixes` drops matching keys from the params file before
    building, which is how the multi-source proper-motion seeding is reached
    for every star at once: ob161003 pins the pm of two of its four stars,
    and a star whose pm the user set is not seeded (#93).  `extra_params`
    is merged in after the strip (already-standardized entries).

    Returns ``(chi2, chi2/N, N, pms, misses, system)``: ``misses`` is
    ``ModelAuditor.check_user_starts`` at the build start, the metric the
    seeding is judged by (requested vs produced).
    """
    src = EXAMPLES_DIR / name
    # PREFER THE CANONICAL CONFIG, <name>.yaml, AND SORT THE FALLBACK.
    # `glob.glob(...)[0]` is FILESYSTEM ORDER, so which config this picked
    # depended on directory layout: in a long-lived checkout it returned
    # `ob140939_ab_deo.yaml` and `ob161003_long.yaml` -- pre-split A/B
    # variants with no `source:` block -- while the params files those
    # configs name carry post-split `source.*` keys, so the build died with
    # a STRICT NAMING ERROR.  In a fresh worktree the same glob returned the
    # shipped `ob140939.yaml` / `ob161003.yaml` and every test passed.  So
    # this file passed or failed according to where it was checked out,
    # which cost a full afternoon of hunting a regression that did not
    # exist.  The canonical config is also the one whose seed quality anyone
    # reading these numbers would assume was measured.
    candidates = sorted(
        p
        for p in glob.glob(str(src / "*.yaml"))
        if "params" not in os.path.basename(p)
        and "hpc" not in os.path.basename(p)
    )
    canonical = str(src / f"{name}.yaml")
    cfg_path = canonical if canonical in candidates else candidates[0]
    # A TEMP ROOT WE OWN MUST BE REMOVED BY US.  pytest prunes its
    # own tmp_path trees to the last three runs; a bare mkdtemp() is
    # nobody's to collect, and this helper runs once per event per
    # test.  74 such copies had accumulated on a shared login node
    # (143 GB, because the ignore pattern above was also missing the
    # suffixed fitresults dirs) before an admin asked for the disk
    # back.  Cleaned in the finally below, next to the chdir.
    tmproot = pathlib.Path(tempfile.mkdtemp())
    work = tmproot / name
    shutil.copytree(
        src, work, ignore=shutil.ignore_patterns("fitresults*", ".#*", "#*#")
    )
    cwd = os.getcwd()
    os.chdir(work)
    try:
        with open(os.path.basename(cfg_path)) as f:
            config = yaml.safe_load(f)
        param_file = config.get("parameter_file")
        user_params = {}
        if param_file and os.path.exists(param_file):
            with open(param_file) as f:
                user_params = yaml.safe_load(f) or {}

        if strip_prefixes:
            for key in list(user_params):
                if key.startswith(tuple(strip_prefixes)):
                    user_params.pop(key)
        user_params.update(extra_params or {})

        system = System(config, user_params)
        system.prepare()
        model = system.build_model()

        # The REAL start, not an assumed all-zeros vector: get_raw_start is
        # 0 for logit-transformed elements but (initval - mu)/sigma for
        # Gaussian-path ones, which is nonzero whenever a prior mean differs
        # from the start value.
        point = system.get_raw_start(model)

        obs = [v for v in model.observed_RVs if "mulens" in v.name]
        assert obs, f"{name} has no mulens observation"
        node = obs[0]
        ins = node.owner.inputs
        fn = pytensor.function(
            model.value_vars,
            model.replace_rvs_by_values([ins[-2], ins[-1]]),
            on_unused_input="ignore",
        )
        mu, sigma = [
            np.asarray(a, dtype=float).ravel()
            for a in fn(*[point[v.name] for v in model.value_vars])
        ]
        data = np.asarray(node.tag.observations.eval(), dtype=float).ravel()
        chi2 = float(np.sum(((data - mu) / sigma) ** 2))
        # The resolved proper motions come back too: chi2 alone cannot say
        # whether the galactic seeding fired or was refused upstream, and
        # "no change" from a gate that was never reached looks identical to
        # "no change" from a gate that did nothing.
        pms = {
            p.label: np.atleast_1d(np.asarray(p.initval, dtype=float))
            for p in system.get_all_parameters()
            if p.label.endswith((".pm_ra", ".pm_dec"))
        }
        misses = ModelAuditor(
            model, system, system.get_mcmc_init(model)
        ).check_user_starts()
        return chi2, chi2 / data.size, data.size, pms, misses, system
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmproot, ignore_errors=True)


@pytest.mark.parametrize("name,ceiling", CEILINGS)
def test_seed_is_a_sensible_starting_point(name, ceiling):
    """
    Given a microlensing example,
    When its model is evaluated at its raw start point,
    Then the light curve already fits to better than `ceiling` chi2 per point.

    This is the check that would have caught the proper-motion seeding
    regression; see the module docstring.
    """
    # Arrange / Act
    chi2, reduced, n, _pms, _misses, _system = _seed_chi2(name)

    # Assert
    assert np.isfinite(chi2), f"{name}: chi2 at the seed is not finite"
    assert reduced < ceiling, (
        f"{name}: chi2/N at the seed is {reduced:.3f} (chi2={chi2:.1f}, "
        f"N={n}), above the {ceiling} ceiling.  The start values no longer "
        f"reproduce the solution the observables came from -- check what the "
        f"relaxation engine resolved for the lens mass, distance and proper "
        f"motion."
    )


def test_a_multi_source_event_is_seeded_from_the_galactic_model():
    """The retired multi-source gate: seeding two sources is an improvement.

    The gate refused to seed proper motions for any event with more than one
    source, because pre-split each source carried its own mu_rel vector and
    one prior mean forced them together.  Under R1 there is one mu_rel per
    event, resolved through source body 0, so the same mean is information:
    source body 1's pm feeds only its own distance/rho chain.

    ob161003 cannot show this as shipped -- it pins the pm of the Lens and
    SourceA.  Since #93 the shipped config seeds SourceB, but the engine
    derives SourceB's pm from the user Lens pm and the event's one mu_rel
    (PRECEDENCE_DERIVED_USER, which outranks the seed), so the start is
    byte-identical.
    Stripping those pins leaves the galactic model as the only source of a
    proper motion for every star.  Measured 2026-09-07 (old engine):

        gate present (no pm seeded)   chi2/N 10.91
        gate retired (pm seeded)      chi2/N  6.94

    Both halves are asserted, because chi2 alone cannot distinguish "seeded
    and better" from "refused upstream and unchanged".
    """
    # Arrange / Act -- remove the pm entries and NOTHING else, so the
    # galactic model is the only thing that can supply a proper motion while
    # every other published value stays.  (Stripping `star.Source`/
    # `star.Lens` wholesale would also drop the distances and radii, and the
    # numbers above would not be the numbers being measured.)
    chi2, reduced, n, pms, _misses, _system = _seed_chi2(
        "ob161003",
        strip_prefixes=(
            "star.SourceA.pm_",
            "star.SourceB.pm_",
            "star.Lens.pm_",
            "star.LensB.pm_",
        ),
    )

    # Assert: the seeding fired, and both sources took the SAME bulge mean.
    # Element order is star config order: 0=Lens, 1=LensB, 2=SourceA,
    # 3=SourceB.
    for axis in ("pm_ra", "pm_dec"):
        vec = pms["star." + axis]
        assert vec.size == 4, f"expected 4 stars, got {vec.size}"
        assert vec[2] == pytest.approx(vec[3]), (
            f"the two sources got different {axis} seeds ({vec[2]} vs "
            f"{vec[3]}); under R1 one mu_rel per event means one bulge mean "
            f"for both source bodies"
        )
        assert vec[0] != pytest.approx(vec[2]), (
            f"the lens and the sources got the same {axis} seed; the lens is "
            f"seeded from the thin disk at 4 kpc and the sources from the "
            f"bulge at 8 kpc, so equal values mean neither was seeded"
        )

    # Assert: and it is better than refusing to seed at all (10.91).
    assert np.isfinite(chi2)
    assert reduced < 8.0, (
        f"chi2/N at the seed is {reduced:.3f} (chi2={chi2:.1f}, N={n}) with "
        f"no proper motion supplied.  Refusing to seed measured 10.91 and "
        f"seeding measured 6.94, so above 8.0 means the multi-source "
        f"seeding has stopped helping and its retirement needs revisiting."
    )


def _bulge_mean_at(system, star_idx):
    """The galactic model's bulge pm mean at star ``star_idx``'s position.

    Computed from the RESOLVED ra/dec with the same helper and the same
    8 kpc the seeding uses, so the comparison carries no constant of its own.
    """
    cm = system.config_manager
    n = system.star.n_elements
    ra = np.atleast_1d(cm.resolve("star", "ra", shape=(n,))["initval"])
    dec = np.atleast_1d(cm.resolve("star", "dec", shape=(n,))["initval"])
    pm_ra, pm_dec, _rv = expected_proper_motion(
        float(np.radians(ra[star_idx])),
        float(np.radians(dec[star_idx])),
        8000.0,
        "bulge",
    )
    return pm_ra, pm_dec


def test_a_user_lens_pm_seeds_the_source_and_keeps_the_lens():
    """A user pm on one star no longer turns off the other star's seed (#93).

    Given ob08092 with a user lens proper motion and no pi_E,
    When the model is built,
    Then the source is seeded at the galactic model's bulge mean, the lens
    starts at exactly the user's pm, and check_user_starts reports no miss.

    Before #93 any user pm skipped ALL galactic seeding, so the engine
    derived the source pm itself (measured: mu_rel_geo PA 172.8 deg, against
    32.4 deg for the shipped config).  Under the precedence ladder (#455)
    the user pm (PRECEDENCE_USER) outranks the seed
    (PRECEDENCE_DERIVED_DATA), so a seed could not bend it and that star
    needs none; the source still gets its seed, and t_E is absorbed through
    the lens distance.
    """
    # Arrange / Act
    lens_pm = (-2.0, -5.0)
    _chi2, reduced, _n, pms, misses, system = _seed_chi2(
        "ob08092",
        extra_params={
            "star.Lens.pm_ra": {"initval": lens_pm[0]},
            "star.Lens.pm_dec": {"initval": lens_pm[1]},
        },
    )

    # Assert: nothing the user set is missed (the metric the engine is
    # judged by), and the lens pm is the user's.  Element order is star
    # config order: 0=Lens, 1=Source.
    assert misses == [], f"user starts not delivered: {misses}"
    assert pms["star.pm_ra"][0] == pytest.approx(lens_pm[0], abs=1e-9)
    assert pms["star.pm_dec"][0] == pytest.approx(lens_pm[1], abs=1e-9)

    # Assert: the source is seeded at the bulge mean, as a data-tier start.
    want_ra, want_dec = _bulge_mean_at(system, 1)
    assert pms["star.pm_ra"][1] == pytest.approx(want_ra, abs=1e-6)
    assert pms["star.pm_dec"][1] == pytest.approx(want_dec, abs=1e-6)
    for axis in ("pm_ra", "pm_dec"):
        src = system.config_manager.initval_source(
            "star", axis, element=1, name="Source"
        )
        assert src == "data", (
            f"star.Source.{axis} start came from {src!r}, not from the "
            f"galactic-model seed"
        )

    # Assert: still a sensible start (the shipped config's ceiling).
    assert reduced < 2.5, f"chi2/N at the seed is {reduced:.3f}"


def test_a_user_pi_e_still_turns_off_the_galactic_seeds():
    """The one remaining gate: a user pi_E fixes the direction of mu_rel.

    Given ob08092 with a user pi_E and no user proper motion,
    When the model is built,
    Then neither star's pm start comes from the galactic model, and the
    user pi_E is delivered.

    Two seeded proper motions would fight a user pi_E over the DIRECTION of
    mu_rel, and no distance can absorb a direction (#93).
    """
    # Arrange / Act
    _chi2, _reduced, _n, _pms, misses, system = _seed_chi2(
        "ob08092",
        extra_params={
            "mulensevent.pi_E_N": {"initval": 0.3},
            "mulensevent.pi_E_E": {"initval": -0.1},
        },
    )

    # Assert
    assert misses == [], f"user starts not delivered: {misses}"
    for i, name in enumerate(system.star.names):
        for axis in ("pm_ra", "pm_dec"):
            src = system.config_manager.initval_source(
                "star", axis, element=i, name=name
            )
            assert src != "data", (
                f"star.{name}.{axis} start came from a galactic-model seed "
                f"despite a user pi_E"
            )


def test_a_bounds_only_pm_entry_does_not_turn_off_the_seed():
    """Only a VALUE counts as the user setting a pm (JDE 2026-10-20, #93).

    Given ob08092 with a bounds-only star.Source.pm_ra entry (lower/upper,
    no initval and no mu) that contains the bulge mean,
    When the model is built,
    Then the source is still seeded at the bulge mean, from the "data"
    tier, and nothing the user set is missed.

    A bounds-only entry states no proper motion, so it must not turn the
    seed off the way an initval or a mu does.
    """
    # Arrange / Act
    _chi2, _reduced, _n, pms, misses, system = _seed_chi2(
        "ob08092",
        extra_params={"star.Source.pm_ra": {"lower": -20.0, "upper": 20.0}},
    )

    # Assert
    assert misses == [], f"user starts not delivered: {misses}"
    want_ra, want_dec = _bulge_mean_at(system, 1)
    assert -20.0 < want_ra < 20.0, "the test's bounds must contain the seed"
    assert pms["star.pm_ra"][1] == pytest.approx(want_ra, abs=1e-6)
    assert pms["star.pm_dec"][1] == pytest.approx(want_dec, abs=1e-6)
    for axis in ("pm_ra", "pm_dec"):
        src = system.config_manager.initval_source(
            "star", axis, element=1, name="Source"
        )
        assert src == "data", (
            f"star.Source.{axis} start came from {src!r}: a bounds-only "
            f"entry turned the galactic-model seed off"
        )


def test_a_seed_outside_a_bounds_only_pm_entry_fails_by_the_usual_rule():
    """A seed that lands outside a bounds-only entry meets the usual rule.

    Given ob08092 with a bounds-only star.Source.pm_ra entry whose support
    [0, 20] excludes the bulge mean (about -3 mas/yr toward this event),
    When the model is built,
    Then the seed is still offered (the entry gives no value), and the
    build refuses the out-of-support start exactly as it refuses any
    data-derived start: naming the parameter and the "data" provenance,
    and telling the user to seed it or widen the bound.  The seeder
    neither clips nor silently stands down.
    """
    # Act / Assert
    with pytest.raises(ValueError, match="outside its hard bounds") as exc:
        _seed_chi2(
            "ob08092",
            extra_params={"star.Source.pm_ra": {"lower": 0.0, "upper": 20.0}},
        )
    msg = str(exc.value)
    assert "star.Source.pm_ra" in msg
    assert "start value from: data" in msg
