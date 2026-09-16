"""The `mulensevent` acceptance gate, file B of two (review 8.6.17).

Replays GROUP_B of the recorded fixtures -- ob07224, ob08092, ob140939,
ob161003, ob170114 and OGLE_0383LD -- against their recorded per-term logp
decompositions.  Because the two fast, deterministic PSPL examples (ob08092,
ob140939) are in this group, the INSTRUMENT'S OWN acceptance tests (the
decomposition reconciles, the term names pair up, the fixture stores its
start) run unmarked from here on every suite run; and because ob161003 is
here, so is the analytic reconciliation of the one deliberate model change.
Its twin, test_mulens_acceptance_a.py, replays GROUP_A; the tolerances, the
helpers and the whole account of WHY the gate exists are in
mulens_acceptance_replay.py.

WHY TWO FILES.  `--dist loadfile` pins a file to one xdist worker, so a CI
shard can never finish faster than its slowest file's SERIAL time.  As one
file this gate cost 873 worker-seconds on CI (2026-09-14), the heaviest file
in the suite and the floor under the shard that carried it.  Two files of
~435 s each dissolve the floor.  A NEW FIXTURE goes into whichever of
GROUP_A / GROUP_B is lighter in tests/durations.json; file A's
`test_the_two_files_partition_the_fixture_set` is what fails if it goes in
neither or both.
"""

import os
import unittest.mock

import numpy as np
import pytest
from mulens_acceptance import (
    compare,
    decompose,
    is_reference_platform,
    platform_fingerprint,
    term_names,
)
from mulens_acceptance_replay import (
    FIXTURES,
    GROUP_B,
    UNMARKED,
    build,
    check_recorded_decomposition,
    fixture_files,
    load,
    load_fixture,
    replay_cases,
)

# The collapse reconciliation below compares this machine against the
# PRE-SPLIT recording, and it used byte equality: at matched parameters the
# trajectory is unchanged by construction, so motion means a wiring bug.
# That is true of the mathematics and false of the arithmetic -- macOS builds
# on Accelerate, Linux on scipy-openblas, `scipy.optimize.nnls` disagrees
# between them, and it sets seeds, scales and bounds at build time (review
# 3.14.20).  CI duly failed at 6.8e-14 relative.
#
# So the edge stays sharp but moves off zero, sized from both measured
# magnitudes: ob161003's worst cross-platform term is 2.0e-13 relative, while
# the stage-2 controls for this comparison fired at 5.7e-04 and -8.7e-02 and
# a collapsed per-source pair moves by 50%.  1e-9 is three orders above the
# noise and five below the smallest signal.
COLLAPSE_RTOL = 1e-9
# NO absolute floor here, deliberately, and it cost a red run to learn why:
# most of the collapsed terms are bound potentials evaluated at an interior
# point, i.e. near zero.  An atol of 1e-12 made them compare EQUAL to the
# recording, so they never entered `moved`, and the "every collapsed term
# must actually have halved" check then fired -- the tolerance swallowed the
# very signal the test exists to see.  A relative test is scale-free and
# right for them: halving is exact in binary, so on another platform the only
# difference is in the last bits of `after` itself.


@pytest.mark.parametrize("name", sorted(UNMARKED))
def test_the_decomposition_reconciles(name):
    """
    Given a shipped microlensing example,
    When its logp is decomposed term by term,
    Then the parts sum to `compile_logp`.

    THE INSTRUMENT'S OWN ACCEPTANCE, and it is checked before any fixture is
    trusted.  Two plausible decompositions were wrong while this was written:
    summing potentials plus `model.logp(rv)` per RV is 30.9 nats SHORT on
    ob161003 (missing the transform jacobians), and
    `model.logp(vars=[rv], jacobian=True)` per RV is 27.7 nats OVER on the
    same model because `logp(vars=...)` does not decompose additively.  Both
    looked right.  This assertion is the control that must fire.
    """
    # ARRANGE
    path = os.path.join(FIXTURES, name + ".json")
    if not os.path.exists(path):
        pytest.skip(f"no fixture for {name}")
    system, model = build(load(path))

    # ACT
    _, total, reconciles, summed = decompose(system, model)

    # ASSERT
    assert reconciles, f"parts sum to {summed!r}, compile_logp is {total!r}"


@pytest.mark.parametrize("name", sorted(UNMARKED))
def test_the_fixture_stores_the_point_it_was_measured_at(name):
    """
    Given a recorded fixture,
    When it is loaded,
    Then it carries the start point every term was evaluated at.

    Without the point, the cross-machine comparison below silently becomes
    "each machine solves its own start and we compare the answers", which
    cannot distinguish a solver converging differently from a likelihood
    function that differs by platform.  A fixture missing its start is not a
    weaker fixture, it is a different and much vaguer measurement.
    """
    # ARRANGE / ACT
    path = os.path.join(FIXTURES, name + ".json")
    if not os.path.exists(path):
        pytest.skip(f"no fixture for {name}")
    fixture = load(path)

    # ASSERT
    assert fixture.get("start"), "fixture has no stored start point"


@pytest.mark.parametrize("name", sorted(UNMARKED))
def test_the_term_names_match_the_logp_terms(name):
    """
    Given a built model,
    When the term names are paired with `model.logp(sum=False)`,
    Then there are exactly as many names as terms.

    `logp(sum=False)` returns `basic_RVs` then `potentials`, and the pairing
    is positional. A reordering upstream would keep the SUM correct while
    mislabelling every term -- a decomposition that reconciles and lies.
    """
    # ARRANGE
    path = os.path.join(FIXTURES, name + ".json")
    if not os.path.exists(path):
        pytest.skip(f"no fixture for {name}")
    _, model = build(load(path))

    # ACT / ASSERT
    assert len(term_names(model)) == len(model.logp(sum=False))


@pytest.mark.slow
@pytest.mark.parametrize("name", replay_cases(GROUP_B))
def test_the_model_still_matches_its_recorded_decomposition(name):
    """
    Given a stage-0 fixture recorded before the split,
    When the example is rebuilt and decomposed,
    Then every term matches.

    This is the refactor's regression gate. It is EXPECTED to fail during the
    split -- that is its purpose. A failure must be read term by term and
    every moved term explained; a term that vanished alongside one that
    appeared with the same value is a rename, which the diff reports
    separately so it is not mistaken for a match.

    Evaluated at the fixture's STORED start, not at this machine's own solved
    one, so a cross-machine failure means the likelihood function differs
    rather than the solver -- see TERM_RTOL in mulens_acceptance_replay.py.
    """
    check_recorded_decomposition(name)


# ---------------------------------------------------------------------------
# ob161003 (2S2L): the ONE deliberate model change (design section 4).
#
# Under R1 the per-source event-level vectors (t_E, theta_E, pi_rel, pi_E,
# the mu_rel family, mlens_total) collapse to scalars, so every logp term
# that used to SUM an identical per-source pair is now counted once.  The
# trajectory at matched t_0/u_0/t_E/s/q/alpha/rho is UNCHANGED, so the
# observed-data term must MATCH the stage-0 recording -- not merely be
# close.  Any motion there is a wiring bug, not a model change; that is the
# sharpest test in the whole refactor and it is held at ZERO tolerance.
# ---------------------------------------------------------------------------

# The stage-0 engine solved the proper-motion split itself (sp.nsolve); the
# shipped params file now seeds the same split in clean round numbers
# (star.Lens.pm_dec = -13.595474606, everything else -3.0), which differ
# from the nsolve output at the ~1e-11 level.  Seeding nsolve's own values
# here puts the post-split model at the stage-0 recording's EXACT start, so
# the data term can be compared to the byte.  Values dumped from the
# pre-split tree at full precision (2026-09, stage-3 acceptance).
_OB161003_PRESPLIT_PM = {
    "star.SourceA.pm_ra": {"initval": -2.999999999986347},
    "star.SourceA.pm_dec": {"initval": -2.999999999986347},
    "star.SourceB.pm_ra": {"initval": -2.999999999986347},
    "star.SourceB.pm_dec": {"initval": -2.999999999986347},
    "star.Lens.pm_ra": {"initval": -2.999999999986347},
    "star.Lens.pm_dec": {"initval": -13.595474606031832},
    "star.LensB.pm_ra": {"initval": -2.999999999986347},
    "star.LensB.pm_dec": {"initval": -2.999999999986347},
}

# Every term whose pre-split value was an identical per-source pair summed
# over TWO sources and is now a scalar: the post-split value must be
# EXACTLY half the recording (bitwise -- halving a double is exact).  All
# but mu_rel_geo_mag's soft lower bound are the ~1e-304 placeholder values
# of soft bounds evaluated far from their turn-on.
_OB161003_HALVED = frozenset(
    {
        "POT:mulensevent.mu_rel_singularity",
        "POT:mulensevent.theta_E_singularity",
        "POT:low_bound.mulensevent.mu_ra_rel",
        "POT:low_bound.mulensevent.mu_dec_rel",
        "POT:low_bound.mulensevent.mu_ra_rel_geo",
        "POT:low_bound.mulensevent.mu_dec_rel_geo",
        "POT:low_bound.mulensevent.mu_rel_mag",
        "POT:low_bound.mulensevent.mu_rel_geo_mag",
        "POT:low_bound.mulensevent.pi_rel",
        "POT:low_bound.mulensevent.t_E",
        "POT:low_bound.mulensevent.theta_E",
        "POT:up_bound.mulensevent.mu_ra_rel",
        "POT:up_bound.mulensevent.mu_dec_rel",
        "POT:up_bound.mulensevent.mu_ra_rel_geo",
        "POT:up_bound.mulensevent.mu_dec_rel_geo",
        "POT:up_bound.mulensevent.mu_rel_mag",
        "POT:up_bound.mulensevent.mu_rel_geo_mag",
        "POT:up_bound.mulensevent.pi_rel",
        "POT:up_bound.mulensevent.theta_E",
    }
)

# The raw->value transform jacobians of the reseeded pm leaves reassociate
# at the last ulp (initval now enters as a literal seed rather than as
# nsolve's output); everything else must match to the byte.
_U1_PRIOR_TERM = "POT:logit_uniform_prior.band.u1"


def _u1_span_narrowing_delta(system):
    """The logit-uniform potential's change when u1's span goes [0, 2] -> [0, 1].

    For a bounded parameter x in [lo, hi] the potential is
    log((x - lo) (hi - x) / (hi - lo)) - log(hi - lo): the transform's
    Jacobian plus the uniform density.  At a fixed x that is
    log(x (2 - x)) - 2 log 2 before the cap and log(x (1 - x)) after it,
    summed over the bands (one shared potential).  The u1 start is read from
    the built system, so a params-file change moves this with it."""
    u1 = np.atleast_1d(np.asarray(system.band.u1.initval, dtype=float))
    before = np.log(u1 * (2.0 - u1)) - 2.0 * np.log(2.0)
    after = np.log(u1 * (1.0 - u1))
    return float(np.sum(after - before))


_OB161003_ULP_OK = frozenset(
    {
        "POT:logit_uniform_prior.star.pm_ra",
        "POT:logit_uniform_prior.star.pm_dec",
    }
)


@pytest.mark.slow
def test_ob161003_event_potentials_are_scalar_and_single():
    """
    Given the ob161003 (2S2L) model,
    When its logp terms are enumerated,
    Then the event potentials exist exactly once, the event-rate and
    singularity potentials are SCALAR (no reduction over sources), and no
    pre-split lens.* twin survives.

    The design's section-4 potential inventory at its stage-3 vehicle: with
    TWO sources, a scalar event-rate potential is structurally incapable of
    the double count review 8.6.18 removed (DC2018_128's single-source
    inventory, in file A, could not distinguish the sum from the scalar).
    source_behind_lens deliberately stays a sum: it is one term PER SOURCE
    STAR (each source must sit behind the lens).
    """
    # ARRANGE
    _, model = build(load_fixture("ob161003"))

    # ACT
    names = term_names(model)
    counts = {n: names.count(n) for n in names}

    # ASSERT
    dupes = {n: c for n, c in counts.items() if c > 1}
    assert not dupes, f"duplicated logp terms: {dupes}"

    for pot in (
        "POT:mulensevent.event_rate_prior",
        "POT:mulensevent.source_behind_lens",
        "POT:mulensevent.mu_rel_singularity",
        "POT:mulensevent.theta_E_singularity",
    ):
        assert counts.get(pot) == 1, f"{pot} missing or duplicated"

    stale = [
        n
        for n in names
        if n.startswith("POT:lens.")
        and n.split(".", 1)[1]
        in (
            "event_rate_prior",
            "source_behind_lens",
            "mu_rel_singularity",
            "theta_E_singularity",
            "fitpirel_jacobian",
            "fitu0te_jacobian",
        )
    ]
    assert not stale, f"pre-split event potentials survive on lens: {stale}"

    # Scalar means scalar: log(mu_rel_geo) + log(theta_E) with no pt.sum
    # over sources.  A sum over the collapsed (1,) event vector would pass
    # every value check while silently re-growing with a second source.
    pots = {f"POT:{p.name}": p for p in model.potentials}
    for pot in (
        "POT:mulensevent.event_rate_prior",
        "POT:mulensevent.mu_rel_singularity",
        "POT:mulensevent.theta_E_singularity",
    ):
        assert pots[pot].ndim == 0, (
            f"{pot} is not scalar (ndim {pots[pot].ndim})"
        )


@pytest.mark.slow
def test_ob161003_collapse_reconciles_against_the_presplit_recording():
    """
    Given the post-split ob161003 model started at the stage-0 recording's
      exact point (the pre-split engine's own solved proper motions seeded
      verbatim),
    When its decomposition is compared to the pre-split recording,
    Then the OBSERVED-DATA term matches to COLLAPSE_RTOL, the only moved
      terms are the collapsed per-source pairs -- each half to within
      that same tolerance -- and nothing appears or vanishes.

    This is design section 4's analytic reconciliation, held as a permanent
    pin.  The data term is the sharp edge: the trajectory at matched
    parameters is unchanged by construction, so motion there is a wiring
    bug.  Measured at stage 3 on the recording machine: 48 of 69 terms
    bit-identical (data term delta exactly 0.0), 19 terms exactly halved,
    and the two reseeded-pm transform jacobians moved by one ulp.

    The comparison is at COLLAPSE_RTOL rather than at zero because it runs
    on more than one platform: see that constant.  It is NOT the replay
    tolerance -- this test solves its own start rather than replaying a
    stored one, so it carries the build difference as well as the
    evaluation difference, and it is held four orders tighter than the
    replay tests all the same.
    """
    # ARRANGE -- the pre-split recording (labels translated, values
    # untouched) and the post-split model at its exact start.
    presplit = load(os.path.join(FIXTURES, "presplit", "ob161003.json"))
    fixture = load_fixture("ob161003")
    system, model = build(fixture, extra_params=_OB161003_PRESPLIT_PM)

    # ACT -- decompose at this model's own solved start (which the seeds
    # above pin to the recording's start).
    parts, _, reconciles, _ = decompose(system, model)
    assert reconciles, "the instrument stopped reconciling; fix it first"
    moved, appeared, vanished = compare(
        presplit["terms"], parts, rtol=COLLAPSE_RTOL
    )

    # ASSERT -- the sharp edge first: the data term to the byte.
    data_now = parts["RV:mulensinstrument.model"]
    data_ref = presplit["terms"]["RV:mulensinstrument.model"]
    assert abs(data_now - data_ref) <= COLLAPSE_RTOL * abs(data_ref), (
        f"the observed-data logp moved at matched trajectory: a wiring "
        f"bug. {data_ref} -> {data_now} "
        f"(relative {(data_now - data_ref) / abs(data_ref):.3e}, and "
        f"cross-platform noise on this example is ~2e-13)"
    )

    assert not appeared, f"terms appeared: {sorted(appeared)}"
    assert not vanished, f"terms vanished: {sorted(vanished)}"

    unexplained = {}
    halved_seen = set()
    for name, (before, after, delta) in moved.items():
        if name in _OB161003_HALVED:
            # Exactly one of the two identical per-source copies remains.
            # Halving a finite double IS exact, but `after` is this
            # machine's own value rather than a halved copy of `before`,
            # so on another platform it lands within rounding of half
            # instead of on it.  The signal here is 50%; the tolerance is
            # 1e-9, so nothing is hidden.
            half = before / 2.0
            if abs(after - half) <= COLLAPSE_RTOL * abs(half):
                halved_seen.add(name)
                continue
        if name in _OB161003_ULP_OK and abs(delta) <= 1e-13 * abs(before):
            continue
        if name == _U1_PRIOR_TERM:
            # Not a collapse term: the linear-law u1 prior span narrowed
            # from [0, 2] to [0, 1] when the validity cap started applying
            # to consumed bands (review 1.5.4), AFTER this recording. At an
            # unchanged u1 the logit-uniform potential moves by a closed
            # form per band, so the recording still reconciles analytically
            # rather than being re-recorded (it is the stage-0 model by
            # definition).
            expected = _u1_span_narrowing_delta(system)
            if abs(delta - expected) <= COLLAPSE_RTOL * abs(before):
                continue
        unexplained[name] = (before, after, delta)
    assert not unexplained, f"unexplained logp motion: {unexplained}"

    # Every collapsed term must actually have moved: a halved term that
    # MATCHES the recording would mean the per-source pair is back.
    missing = _OB161003_HALVED - halved_seen
    assert not missing, (
        f"expected these collapsed terms to be half the recording, but "
        f"they matched it -- the per-source duplication is back: "
        f"{sorted(missing)}"
    )


def test_both_tolerance_tiers_are_reachable():
    """
    Given a fixture recorded on this machine and one recorded elsewhere,
    When the tier is selected,
    Then the first gets REFERENCE_RTOL and the second the physics tolerance.

    THE POINT IS THE SECOND HALF.  Every fixture in the tree carries this
    machine's fingerprint, so on the reference platform the strict tier is
    the only one the suite ever exercises -- and a bug making
    `is_reference_platform` answer True unconditionally would be invisible
    here while holding macOS to a tolerance eight orders too tight.  That is
    the shape of the vacuity failures in docs/testing.md: the guarded path
    passes and the unguarded one is never entered.

    So assert both directions, with a FABRICATED foreign fingerprint rather
    than a second real platform.
    """
    # Arrange -- a real fixture, the same one relabelled foreign, and one
    # with the key removed (an old recording).
    path = sorted(fixture_files())[0]
    name = os.path.splitext(os.path.basename(path))[0]
    fixture = load(path)
    assert fixture.get("platform"), (
        f"{name} carries no fingerprint, so it cannot select the strict "
        f"tier; the generator should have recorded one"
    )

    # STAMP THE OWN-MACHINE CASE, do not assume the fixture is one.  The
    # shipped fixtures carry the RECORDING box's fingerprint (Linux), so
    # asserting on them directly passes only there and fails on every other
    # runner -- which is exactly how this test went red on macOS.  What is
    # under test is the SELECTOR, not where the suite happens to run.
    own = dict(fixture)
    own["platform"] = platform_fingerprint()

    foreign = dict(fixture)
    foreign["platform"] = {
        "system": "NotAnOS",
        "machine": "notanarch",
        "blas": "notablas",
    }
    unstamped = {k: v for k, v in fixture.items() if k != "platform"}

    # Act / Assert.  The strict tier needs BOTH the opt-in and the
    # fingerprint, so drive the env var explicitly rather than depending on
    # how the suite happens to be invoked.
    with unittest.mock.patch.dict(
        os.environ, {"EXOZIPPY_ACCEPTANCE_STRICT": "1"}
    ):
        assert is_reference_platform(own), (
            "opted in and the fingerprint matches, yet the strict tier was "
            "not selected -- it is unreachable and every run silently takes "
            "the loose one"
        )
        assert not is_reference_platform(foreign), (
            "a fixture recorded on a fabricated foreign platform was "
            "accepted as this machine's own, which would hold a foreign "
            "platform to REFERENCE_RTOL"
        )
        assert not is_reference_platform(unstamped), (
            "a fixture with NO fingerprint claimed the strict tier; an old "
            "recording must get the physics bound rather than a bit-identity "
            "claim nobody measured"
        )

    # And WITHOUT the opt-in, nothing claims the strict tier -- the default
    # has to be the safe one.  PR #251 failed exactly here: CI's Linux
    # runner shares this box's coarse fingerprint (system/machine/BLAS) but
    # not its rounding, so an inferred claim held it to a tolerance eight
    # orders too tight.
    env = dict(os.environ)
    env.pop("EXOZIPPY_ACCEPTANCE_STRICT", None)
    with unittest.mock.patch.dict(os.environ, env, clear=True):
        assert not is_reference_platform(own), (
            "the strict tier was claimed WITHOUT the opt-in; any machine of "
            "this platform class would then be held to REFERENCE_RTOL"
        )
